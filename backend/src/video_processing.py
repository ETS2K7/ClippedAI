# pylint: disable=no-member,too-many-locals,too-many-statements,too-many-branches,too-many-arguments,too-many-positional-arguments

import os
import subprocess
import json
from typing import List, Dict, Any, Tuple

import cv2
try:
    import modal
except ImportError:
    modal = None
import numpy as np
from scenedetect import detect, ContentDetector

from config import get_logger

logger = get_logger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
# All split cells use the same 9:16 portrait crop (608px) as single-speaker mode.
# This is the critical guarantee that prevents any participant from appearing
# in more than one cell simultaneously: two independent 608px windows centred on
# speakers ≥608px apart are geometrically non-overlapping by definition.
CROP_W_1 = 608   # universal 9:16 crop width for every layout cell

OUT_W, OUT_H = 1080, 1920

# Crop widths for different speaker layouts (AR-safe from 16:9 source)
CROP_W_1          = 608  # 1-speaker: full-frame 9:16 crop
CROP_W_2          = 1080 # 2-speaker: vertical split (1080×960 per speaker)
CROP_W_3T         = 1080 # 3-speaker top featured (1080×960)
CROP_W_3S         = 540  # 3-speaker side-by-side bottom (540×960 each)
CROP_W_4          = 540  # 4-speaker grid (540×960 each)

# the source crop must be 608×541 to keep both axes at the same scale factor
# (1.776× horiz, 1.774× vert — 0.1% difference, imperceptible).
# Using 608×1080 → 1080×960 gives 1.776× vs 0.888× — 2× scale mismatch → distortion.
CROP_H_HALF = int(round(CROP_W_1 * (OUT_H // 2) / OUT_W))  # = 541

# Gaussian smoothing frames (0 = Hard cuts, no sliding)
SIGMA = 0  

# Stabilisation thresholds (entry = min frames before mode activates,
# gap = min gap frames before mode drops — prevents rapid re-entry)
MIN_SPLIT_2_ENTRY = 20   # ~0.8 s
MIN_SPLIT_2_GAP   = 30   # increased to prevent flickering out
MIN_SPLIT_3_ENTRY = 25   # ~1.0 s
MIN_SPLIT_3_GAP   = 25
MIN_SPLIT_4_ENTRY = 30   # ~1.2 s
MIN_SPLIT_4_GAP   = 30

# Minimum frames a new speaker/camera-angle position must be held before the
# crop switches to follow it.  A scene cut to a different angle must persist
# for this many consecutive frames or the framing stays on the previous face.
# 0 frames — reaction is instant. Stability is handled by Step 7d.
MIN_SPEAKER_SWITCH_FRAMES = 0

# Detection thresholds
MIN_FACE_W_RATIO  = 0.04  # Face must be ≥4% of frame width to count
SPLIT_MARGIN      = 0.08  # 2-speaker: each face must be 8% past centre
MIN_FACE_SEP      = 0.10  # 3/4-speaker: min separation between adjacent faces
# Minimum cx distance before 608px crops stop overlapping (|cx_r - cx_l| < CROP_W_1)
SPLIT_MIN_CX_SEP  = CROP_W_1  # 608px


# ─── FFmpeg Core Utilities ───────────────────────────────────────────────────

def _get_video_codec_args(use_gpu: bool, is_merge: bool = False) -> List[str]:
    """Returns standardized GPU/CPU video codec arguments to maintain pipeline sync."""
    if use_gpu:
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "constqp", "-qp", "28"]
    if is_merge:
        return ["-c:v", "copy"]
    return ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23"]

def _run_ffmpeg(cmd: List[str], error_ctx: str):
    """Executes FFmpeg with central error boundary mapping. Captures stderr for diagnostics."""
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        stderr_out = e.stderr.decode("utf-8", errors="replace")[-2000:] if e.stderr else "(no stderr)"
        logger.error(f"FFmpeg failed [{error_ctx}]:\n{stderr_out}")
        raise RuntimeError(f"{error_ctx} failed: {e}") from e

# ─── Phase 4 ──────────────────────────────────────────────────────────────────

def extract_segment(input_file: str, clip: Dict[str, Any], idx: int, work_dir: str = "", use_gpu: bool = False) -> str:
    """FFmpeg segment extraction for a given clip timestamp range.

    Re-encodes to H.264 + AAC (not stream copy). YouTube / Apify often delivers
    AV1-in-MP4 or WebM; stream-copied segments break OpenCV decoding and
    scenedetect, producing empty tracked outputs and Phase 7 merge failures
    (``[0:v]ass=...`` matches no streams).
    
    Optimized with NVENC GPU acceleration when available.
    """
    logger.info(
        f"==================== PHASE 4: SEGMENT EXTRACTION (Clip {idx}) ===================="
    )
    start = clip["start_time"]
    dur = clip["end_time"] - start
    out = f"{work_dir}/temp_extracted_clip_{idx}.mp4" if work_dir else f"temp_extracted_clip_{idx}.mp4"
    logger.info(f"Extracting {out} [{start}s to {clip['end_time']}s] (GPU: {use_gpu})...")
    
    # Base command
    cmd = [
        "ffmpeg", "-y",
        "-i", input_file,
        "-ss", str(start),
        "-t", str(dur),
    ]
    
    cmd.extend(_get_video_codec_args(use_gpu, is_merge=False))
    cmd.extend([
        "-c:a", "aac",
        "-b:a", "128k",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        out,
    ])
    
    _run_ffmpeg(cmd, f"segment extraction {idx}")
    return out


# ─── Phase 7 ──────────────────────────────────────────────────────────────────

def merge_and_cleanup(tracked_vid: str, extract_vid: str, sub_file: str | None, idx: int, work_dir: str = "", use_gpu: bool = False, fonts_dir: str = ""):
    """Merges subtitle-ass video with audio from original segment.

    Uses FFmpeg to burn in ASS subtitles and copy audio from the extracted segment.
    This ensures the output has both video (with subs) and audio in the correct format.
    
    Args:
        fonts_dir: Path to directory containing custom .ttf fonts (e.g. Komika Axis).
                   Passed to libass via fontsdir= so it can resolve font families.
    """
    logger.info(
        f"==================== PHASE 7: MERGE & CLEANUP (Clip {idx}) ===================="
    )
    out_file = f"{work_dir}/clip_{idx}.mp4" if work_dir else f"output/clip_{idx}.mp4"

    # Re-encode tracked .avi (MJPG) to H.264, optionally burn in ASS subtitles, and mux audio.
    # -c:v libx264 is always available; we intentionally do NOT use NVENC here because
    # MJPG pixel format (yuvj420p) requires colour-range conversion that NVENC rejects.
    # The ass= filter path must have colons escaped for FFmpeg's filter syntax on Linux.
    cmd = [
        "ffmpeg", "-y",
        "-i", tracked_vid,
        "-i", extract_vid,
    ]

    if sub_file:
        safe_sub = sub_file.replace("\\", "/").replace(":", "\\:")
        if fonts_dir:
            safe_fonts = fonts_dir.replace("\\", "/").replace(":", "\\:")
            ass_filter = f"ass={safe_sub}:fontsdir={safe_fonts}"
        else:
            ass_filter = f"ass={safe_sub}"
        cmd.extend(["-vf", ass_filter])

    cmd.extend([
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-movflags", "+faststart",
        "-shortest",
        out_file,
    ])

    _run_ffmpeg(cmd, f"merge for clip {idx}")

    # Clean up intermediate files
    try:
        os.remove(tracked_vid)
        os.remove(extract_vid)
        if sub_file:
            os.remove(sub_file)
    except OSError as e:
        logger.warning(f"Failed to clean up temporary files for clip {idx}: {e}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _face_cx(face: Dict) -> float:
    return (face["x1"] + face["x2"]) / 2.0

def _face_cy(face: Dict) -> float:
    return (face["y1"] + face["y2"]) / 2.0

def _face_score(face: Dict) -> float:
    return face.get("raw_score", 0.0)

def _face_area(face: Dict) -> float:
    """Bounding-box area — used to pick the most prominent face when nobody is
    actively speaking.  TalkNet raw_score is a speaking-confidence signal and
    is unreliable (negative) for silent frames; the largest face in frame is a
    far better proxy for 'the main subject' in that case."""
    return (face["x2"] - face["x1"]) * (face["y2"] - face["y1"])


def get_centered_crop(
    frame_img: np.ndarray,
    cx: float,
    crop_w: int,
    cy: float = -1.0,
    crop_h: int = -1,
) -> np.ndarray:
    """
    Returns a crop_w × crop_h strip of frame_img centred as close as possible
    to (cx, cy), clamped so the window never exits the frame boundaries.

    cy / crop_h default (-1) means "use full frame height".
    """
    h_img, w_img = frame_img.shape[:2]

    # ─ Horizontal ───────────────────────────────────────────────────────
    crop_w = min(crop_w, w_img)
    x1 = int(round(cx - crop_w / 2))
    x1 = max(0, min(w_img - crop_w, x1))

    # ─ Vertical ───────────────────────────────────────────────────────
    if crop_h <= 0 or crop_h >= h_img:
        # Full frame height — no vertical crop needed
        return frame_img[:, x1: x1 + crop_w]

    # Centre on the detected face y-position (cy)
    y1 = int(round(cy - crop_h / 2))
    y1 = max(0, min(h_img - crop_h, y1))

    return frame_img[y1: y1 + crop_h, x1: x1 + crop_w]


def _ar_safe_crop(
    frame: np.ndarray,
    cx: float,
    cy: float,
    cell_w: int,
    cell_h: int,
) -> np.ndarray:
    """
    Aspect-ratio-safe crop-to-fill.

    Computes the largest crop window that:
      1. Fits entirely within the source frame
      2. Has the EXACT same aspect ratio as the target output cell (cell_w × cell_h)
      3. Is centred on (cx, cy)
      4. Maximises coverage (up to CROP_W_1 pixels wide)

    This guarantees zero distortion when the crop is resized to (cell_w, cell_h),
    regardless of the input video resolution (720p, 1080p, 4K, portrait, etc.).

    Math proof:
      crop_w / crop_h == cell_w / cell_h
      ⇒ resize(crop, (cell_w, cell_h)) applies the SAME scale factor to both axes
      ⇒ no stretch or squash in any direction.
    """
    h_img, w_img = frame.shape[:2]
    target_ar = cell_w / cell_h  # target aspect ratio (width/height)

    # Start with the ideal crop width
    crop_w = min(CROP_W_1, w_img)
    crop_h = int(round(crop_w / target_ar))

    # If the ideal height exceeds the frame, become height-constrained instead
    if crop_h > h_img:
        crop_h = h_img
        crop_w = int(round(crop_h * target_ar))
        crop_w = min(crop_w, w_img)

    return get_centered_crop(frame, cx, crop_w, cy=cy, crop_h=crop_h)


def _smooth_segment(raw: np.ndarray, default: float, sigma: int) -> np.ndarray:
    """Thin wrapper — implementation lives in signal_helpers.py for testability."""
    from src.signal_helpers import smooth_segment
    return smooth_segment(raw, default, sigma)


# ─── Split-state stabilisation ────────────────────────────────────────────────

def _stabilize_segment(raw: np.ndarray, min_entry: int, min_gap: int) -> np.ndarray:
    """Thin wrapper — implementation lives in signal_helpers.py for testability."""
    from src.signal_helpers import stabilize_segment
    return stabilize_segment(raw, min_entry, min_gap)


def _stabilize_speaker_identity(
    raw_cx: np.ndarray,
    min_frames: int,
    px_threshold: float,
) -> np.ndarray:
    """Thin wrapper — implementation lives in signal_helpers.py for testability."""
    from src.signal_helpers import stabilize_speaker_identity
    return stabilize_speaker_identity(raw_cx, min_frames, px_threshold)


def _stabilize_bool_state(
    raw: np.ndarray,
    scene_boundaries: List[int],
    min_entry: int,
    min_gap: int,
) -> np.ndarray:
    """
    Stabilizes a boolean signal (like 'is split-screen active').
    
    Previous version stabilized per-scene, which created 'dead zones'
    at the end of scenes where a layout change would be suppressed if
    it happened <min_entry frames before a cut.
    
    New version stabilizes GLOBALLY to allow layouts to transition 
    cleanly across scene boundaries.
    """
    return _stabilize_segment(raw, min_entry, min_gap)


# ─── Per-layout frame renderers ───────────────────────────────────────────────
#
# Every renderer now uses _ar_safe_crop to compute the crop window, guaranteeing
# that the crop's aspect ratio exactly matches the output cell's aspect ratio.
# This eliminates stretching/squashing for ALL input resolutions (720p, 1080p,
# 4K, portrait, non-standard, etc.).

def _cell_full(frame: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """Single-speaker: AR-safe crop → 1080×1920."""
    return cv2.resize(_ar_safe_crop(frame, cx, cy, OUT_W, OUT_H), (OUT_W, OUT_H))


def _cell_half(frame: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """2-speaker top/bottom cell: AR-safe crop → 1080×960."""
    return cv2.resize(
        _ar_safe_crop(frame, cx, cy, OUT_W, OUT_H // 2),
        (OUT_W, OUT_H // 2),
    )


def _cell_3_top(frame: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """3-speaker featured top cell: AR-safe crop → 1080×960."""
    return cv2.resize(
        _ar_safe_crop(frame, cx, cy, OUT_W, OUT_H // 2),
        (OUT_W, OUT_H // 2),
    )


def _cell_3_side(frame: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """3-speaker bottom side cell: AR-safe crop → 540×960."""
    return cv2.resize(
        _ar_safe_crop(frame, cx, cy, OUT_W // 2, OUT_H // 2),
        (OUT_W // 2, OUT_H // 2),
    )


def _cell_quad(frame: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """4-speaker cell: AR-safe crop → 540×960."""
    return cv2.resize(
        _ar_safe_crop(frame, cx, cy, OUT_W // 2, OUT_H // 2),
        (OUT_W // 2, OUT_H // 2),
    )


def _render_split_2(
    frame: np.ndarray,
    cx_left: float, cx_right: float,
    cy_left: float, cy_right: float,
) -> np.ndarray:
    """Vertical stack: left speaker top, right speaker bottom."""
    top    = _cell_half(frame, cx_left,  cy_left)
    bottom = _cell_half(frame, cx_right, cy_right)
    return cv2.vconcat([top, bottom])


def _render_split_3(
    frame: np.ndarray,
    cx_top: float, cx_bl: float, cx_br: float,
    cy_top: float, cy_bl: float, cy_br: float,
) -> np.ndarray:
    """
    1 + 2 layout:
      Top  (full width 1080×960) — featured speaker, face-centred vertical crop.
      Bottom left  (540×960)     — left speaker, face-centred crop.
      Bottom right (540×960)     — right speaker, face-centred crop.
    """
    top       = _cell_3_top(frame, cx_top, cy_top)
    bot_left  = _cell_3_side(frame, cx_bl, cy_bl)
    bot_right = _cell_3_side(frame, cx_br, cy_br)
    bottom    = cv2.hconcat([bot_left, bot_right])
    return cv2.vconcat([top, bottom])


def _render_split_4(
    frame: np.ndarray,
    cx_tl: float, cx_tr: float,
    cx_bl: float, cx_br: float,
    cy_tl: float, cy_tr: float,
    cy_bl: float, cy_br: float,
) -> np.ndarray:
    """
    2 × 2 grid (each cell 540×960, 9:16):
      Top row:    leftmost speaker | second speaker
      Bottom row: third speaker   | rightmost speaker
    """
    tl = _cell_quad(frame, cx_tl, cy_tl)
    tr = _cell_quad(frame, cx_tr, cy_tr)
    bl = _cell_quad(frame, cx_bl, cy_bl)
    br = _cell_quad(frame, cx_br, cy_br)
    top = cv2.hconcat([tl, tr])
    bot = cv2.hconcat([bl, br])
    return cv2.vconcat([top, bot])


# ─── Prominent-face extraction ────────────────────────────────────────────────

def _prominent_distinct_faces(
    faces: List[Dict], w: int, max_n: int = 4
) -> List[Dict]:
    """
    Returns up to max_n faces that are:
      • Wide enough to be a foreground subject (≥ MIN_FACE_W_RATIO × frame width).
      • Sufficiently separated horizontally (≥ MIN_FACE_SEP × frame width apart).
    Result is sorted by cx (left → right).
    """
    min_w = w * MIN_FACE_W_RATIO
    min_sep = w * MIN_FACE_SEP
    prominent = sorted(
        [f for f in faces if abs(f["x2"] - f["x1"]) >= min_w],
        key=_face_cx,
    )
    distinct: List[Dict] = []
    for f in prominent:
        if not distinct or (_face_cx(f) - _face_cx(distinct[-1])) >= min_sep:
            distinct.append(f)
            if len(distinct) == max_n:
                break
    return distinct


# ─── Phase 5 ──────────────────────────────────────────────────────────────────

def track_speaker_and_frame(
    clip_file: str, idx: int, clip: Dict[str, Any], words: List[Dict[str, Any]], work_dir: str = "",
    tracker=None,
    remote_cache=None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Phase 5: Multi-speaker tracking and adaptive 9:16 reframing.

    Supported layouts:
      1 speaker  → full-frame 9:16 crop (CROP_W_1).
      2 speakers → vertical split, each 1080×960 (CROP_W_2).
      3 speakers → 1 featured top + 2 side-by-side bottom (CROP_W_3T / CROP_W_3S).
      4 speakers → 2×2 grid, each 540×960 (CROP_W_4).

    The 2-speaker path is pixel-identical to the previous implementation.
    3/4-speaker modes use higher stabilisation thresholds so they only activate
    in genuine panel/multi-host footage.

    Args:
        tracker: Optional pre-initialised LocalFastASDTracker. When None (default),
                 the production Modal remote is used. Pass a LocalFastASDTracker
                 instance for fully local processing without Modal.
    """
    logger.info(
        f"==================== PHASE 5: SPEAKER TRACKING & FRAMING (Clip {idx}) "
        f"===================="
    )

    # ── 1. Fast-ASD ──────────────────────────────────────────────────────────
    import hashlib
    with open(clip_file, "rb") as f:
        _video_hash = hashlib.sha256(f.read()).hexdigest()

    if remote_cache is not None and _video_hash in remote_cache:
        logger.info(f"[FastASD] 🟢 Remote cache hit — skipping TalkNet (key={_video_hash[:8]})")
        tracking_data = remote_cache[_video_hash]
    else:
        try:
            if tracker is not None:
                logger.info("Calling local Fast-ASD tracker...")
                _, s, DET = tracker
                
                import threading
                import shutil
                import sys
                import importlib.util
                
                tid = threading.get_ident()
                module_name = f"demoTalkNet_{tid}"
                new_path = f"/fast-asd/talknet/{module_name}.py"
                save_dir = f"/tmp/talknet_save_{tid}/"
                
                # 1. Dynamically copy and patch the TalkNet source code for this specific thread
                shutil.copy("/fast-asd/talknet/demoTalkNet.py", new_path)
                with open(new_path, "r") as f:
                    content = f.read()
                
                # Rewrite the hardcoded global directory to a thread-specific directory
                content = content.replace('save_path = "save/"', f'save_path = "{save_dir}"')
                
                with open(new_path, "w") as f:
                    f.write(content)
                    
                # 2. Import the isolated module
                spec = importlib.util.spec_from_file_location(module_name, new_path)
                local_demoTalkNet = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(local_demoTalkNet)
                sys.modules[module_name] = local_demoTalkNet
                
                try:
                    # 3. Execute TalkNet in the isolated thread context
                    tracking_data = local_demoTalkNet.main(
                        s=s,
                        DET=DET,
                        video_path=clip_file,
                        start_seconds=0,
                        end_seconds=-1,
                        return_visualization=False,
                        in_memory_threshold=0,
                    )
                finally:
                    # 4. Garbage Collection: Prevent resource leaks by deleting the clone and its temp files
                    logger.info(f"Cleaning up TalkNet clone for thread {tid}")
                    if module_name in sys.modules:
                        del sys.modules[module_name]
                    if os.path.exists(new_path):
                        os.remove(new_path)
                    if os.path.exists(save_dir):
                        shutil.rmtree(save_dir, ignore_errors=True)
            else:
                logger.error("Fast-ASD tracker not initialized")
                raise RuntimeError("Fast-ASD tracker not initialized")
                
            if remote_cache is not None:
                try:
                    remote_cache[_video_hash] = tracking_data
                    logger.info(f"[FastASD] Cached TalkNet data to remote Dict")
                except Exception as e:
                    logger.warning(f"[FastASD] Remote cache write failed: {e}")
                    
        except Exception as e:
            logger.error(f"Fast-ASD tracker failed: {e}")
            raise RuntimeError(f"ASD tracking failed: {e}") from e

    # ── 2. Video metadata ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(clip_file)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        logger.warning(f"Invalid fps ({fps}), defaulting to 25")
        fps = 25.0

    if w < CROP_W_1:
        logger.warning(
            f"Video width ({w}px) is narrower than crop width ({CROP_W_1}px). "
            f"Output will use full frame width."
        )

    logger.info(f"Video: {w}x{h} @ {fps}fps, {frames_count} frames")

    # Use MJPG codec into a .avi container — always available in OpenCV builds.
    # The merge step will re-encode this to H.264 via FFmpeg.
    fourcc   = cv2.VideoWriter_fourcc(*"MJPG")
    out_path = f"{work_dir}/temp_tracked_{idx}.avi" if work_dir else f"temp_tracked_{idx}.avi"
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (OUT_W, OUT_H))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open for clip {idx} (codec: MJPG)")

    frame_faces: Dict[int, List[Dict]] = {
        item["frame_number"]: item["faces"] for item in tracking_data
    }

    # ── 3. Scene detection ────────────────────────────────────────────────────
    scene_list   = detect(clip_file, ContentDetector())
    scene_cuts   = {s[0].get_frames() for s in scene_list}
    scene_boundaries = sorted([0] + list(scene_cuts) + [frames_count])
    logger.info(f"Scene cuts detected at frames: {sorted(scene_cuts)}")

    # ── 4. Diarization override — corrects ASD speaker identity ─────────────
    clip_start_ms = clip.get("start_time", 0) * 1000.0
    clip_end_ms   = clip.get("end_time",   0) * 1000.0
    clip_words = [
        wd for wd in words
        if wd.get("end",   0) >= clip_start_ms - 2000
        and wd.get("start", 0) <= clip_end_ms   + 2000
    ]
    speaker_array: List[Any] = [None] * frames_count
    w_ptr = 0
    for fi in range(frames_count):
        ms = clip_start_ms + (fi / fps) * 1000.0
        while w_ptr < len(clip_words) - 1 and ms > clip_words[w_ptr].get("end", 0):
            w_ptr += 1
        wd = clip_words[w_ptr] if w_ptr < len(clip_words) else None
        if wd and wd.get("start", 0) - 200 <= ms <= wd.get("end", 0) + 200:
            speaker_array[fi] = wd.get("speaker")

    # Clip-level speaker→side map for fallback attribution
    clip_spk_xs: Dict = {}
    for fi in range(frames_count):
        faces = frame_faces.get(fi, [])
        spk   = speaker_array[fi]
        if len(faces) == 1 and spk is not None:
            clip_spk_xs.setdefault(spk, []).append(_face_cx(faces[0]))
    clip_side_map: Dict = {
        spk: (1 if float(np.median(xs)) > w / 2 else 0)
        for spk, xs in clip_spk_xs.items() if len(xs) >= 10
    }

    # Per-scene: re-attribute the 'speaking' flag in ambiguous multi-face frames
    WINDOW = 50
    for seg_start, seg_end in zip(scene_boundaries[:-1], scene_boundaries[1:]):
        scene_spk_xs: Dict = {}
        for fi in range(seg_start, seg_end):
            faces = frame_faces.get(fi, [])
            spk   = speaker_array[fi]
            if len(faces) == 1 and spk is not None:
                scene_spk_xs.setdefault(spk, []).append(_face_cx(faces[0]))
        scene_x_map: Dict = {
            spk: float(np.median(xs))
            for spk, xs in scene_spk_xs.items() if len(xs) >= 5
        }
        if not scene_x_map:
            continue
        for fi in range(seg_start, seg_end):
            faces  = frame_faces.get(fi, [])
            active = [f for f in faces if f.get("speaking", False)]
            if len(active) < 2 or len(faces) < 2:
                continue
            spk = speaker_array[fi]
            if spk is None:
                continue
            window_spks = {
                s for s in speaker_array[max(0, fi - WINDOW): fi + WINDOW + 1]
                if s is not None
            }
            if len(window_spks) > 1:
                continue
            if spk in scene_x_map:
                target_x = scene_x_map[spk]
                best = min(faces, key=lambda f: abs(_face_cx(f) - target_x))
            elif spk in clip_side_map:
                best = sorted(faces, key=lambda f: f["x1"])[clip_side_map[spk]]
            else:
                continue
            for f in faces:
                f["speaking"] = (f["x1"] == best["x1"] and f["y1"] == best["y1"])

    # ── 5. Per-scene multi-speaker confirmation gate ──────────────────────────
    # Diarization gates visual detection: only trigger multi-speaker layouts
    # in scenes where AssemblyAI confirmed ≥2 distinct speakers are present.
    frame_to_scene_start = np.zeros(frames_count, dtype=int)
    scene_speaker_count: Dict[int, int] = {}
    for seg_start, seg_end in zip(scene_boundaries[:-1], scene_boundaries[1:]):
        frame_to_scene_start[seg_start:seg_end] = seg_start
        speakers_in_scene = {
            speaker_array[fi]
            for fi in range(seg_start, seg_end)
            if speaker_array[fi] is not None
        }
        scene_speaker_count[seg_start] = len(speakers_in_scene)

    # ── 5b. Precompute per-scene speaker → position maps ────────────────────
    # Used by the diarization-anchored fallback (step C.5 below) to pick the
    # correct face when TalkNet has no confident speaker but diarization does.
    # Only built from frames with exactly 1 face (unambiguous face→speaker mapping).
    scene_spk_x_map: Dict[int, Dict] = {}
    for seg_start, seg_end in zip(scene_boundaries[:-1], scene_boundaries[1:]):
        spk_xs: Dict = {}
        for fi in range(seg_start, seg_end):
            faces_in = frame_faces.get(fi, [])
            spk = speaker_array[fi]
            if len(faces_in) == 1 and spk is not None:
                spk_xs.setdefault(spk, []).append(_face_cx(faces_in[0]))
        scene_spk_x_map[seg_start] = {
            spk: float(np.median(xs))
            for spk, xs in spk_xs.items() if len(xs) >= 5
        }

    # ── 6. Build raw per-frame speaker arrays (N = 1..4) ─────────────────────
    #
    # raw_n_spk[fi]         = number of distinct speakers detected (1–4).
    # raw_spk_cx[fi, 0..3]  = their cx positions (NORMALIZED 0-1), -1 if absent.
    #
    # Detection hierarchy for frame fi:
    #   A)   TalkNet simultaneous speaking (both marked speaking=True)    → highest confidence
    #   B)   Visual: ≥2 prominent, well-separated faces in multi-spk scene → standard path
    #   C)   Single TalkNet speaking face                                  → single speaker
    #   C.5) Diarization-anchored: no TalkNet speaker, but diarization     → picks closest face
    #        knows who's talking — pick face nearest known position
    #   D)   Best-scoring face (no confirmed speaker)                      → single speaker fallback
    #
    raw_n_spk  = np.ones(frames_count, dtype=int)
    raw_spk_cx = np.full((frames_count, 4), -1.0)
    raw_spk_cy = np.full((frames_count, 4), -1.0) 

    def _norm_x(f): return ((f["x1"] + f["x2"]) / 2.0) / w
    def _norm_y(f): return ((f["y1"] + f["y2"]) / 2.0) / h

    # Per-path frame counters — logged per scene to diagnose framing issues
    path_counts = {"A": 0, "B": 0, "C": 0, "C5": 0, "D": 0, "NOFACE": 0}

    for fi in range(frames_count):
        faces    = frame_faces.get(fi, [])
        speaking = [f for f in faces if f.get("speaking", False)]
        scene_s  = int(frame_to_scene_start[fi])
        n_spk_scene = scene_speaker_count.get(scene_s, 1)

        # —— A) TalkNet simultaneous
        if len(speaking) >= 2:
            by_x = sorted(speaking[:4], key=_face_cx)
            n = len(by_x)
            raw_n_spk[fi] = n
            for i, f in enumerate(by_x):
                raw_spk_cx[fi, i] = _norm_x(f)
                raw_spk_cy[fi, i] = _norm_y(f)
            path_counts["A"] += 1
            continue

        # —— B) Visual presence
        if len(faces) >= 2 and n_spk_scene >= 2:
            distinct = _prominent_distinct_faces(faces, w)

            if len(distinct) >= 4 and n_spk_scene >= 4:
                cxs = [_norm_x(f) for f in distinct[:4]]
                cys = [_norm_y(f) for f in distinct[:4]]
                # Geometric separation check (normalized: 608 / 1280 = 0.475)
                sep_thresh = SPLIT_MIN_CX_SEP / w
                separable = all(cxs[i+1] - cxs[i] >= sep_thresh for i in range(3))
                if separable:
                    raw_n_spk[fi] = 4
                    for i in range(4):
                        raw_spk_cx[fi, i] = cxs[i]
                        raw_spk_cy[fi, i] = cys[i]
                    continue

            if len(distinct) >= 3 and n_spk_scene >= 3:
                cxs = [_norm_x(f) for f in distinct[:3]]
                cys = [_norm_y(f) for f in distinct[:3]]
                sep_thresh = SPLIT_MIN_CX_SEP / w
                separable = all(cxs[i+1] - cxs[i] >= sep_thresh for i in range(2))
                if separable:
                    raw_n_spk[fi] = 3
                    for i in range(3):
                        raw_spk_cx[fi, i] = cxs[i]
                        raw_spk_cy[fi, i] = cys[i]
                    continue

            if len(distinct) >= 2:
                # Diarization gate: each face clearly in its own half
                cxs = [_norm_x(f) for f in distinct]
                cys = [_norm_y(f) for f in distinct]
                clearly_left  = cxs[0]  < (0.5 - SPLIT_MARGIN)
                clearly_right = cxs[-1] > (0.5 + SPLIT_MARGIN)
                sep_thresh    = SPLIT_MIN_CX_SEP / w
                separable     = (cxs[-1] - cxs[0]) >= sep_thresh

                if clearly_left and clearly_right and separable:
                    raw_n_spk[fi] = 2
                    raw_spk_cx[fi, 0] = cxs[0]
                    raw_spk_cx[fi, 1] = cxs[-1]
                    raw_spk_cy[fi, 0] = cys[0]
                    raw_spk_cy[fi, 1] = cys[-1]
                    continue

        # —— C) Single TalkNet confirmed
        if len(speaking) == 1:
            raw_n_spk[fi] = 1
            raw_spk_cx[fi, 0] = _norm_x(speaking[0])
            raw_spk_cy[fi, 0] = _norm_y(speaking[0])
            path_counts["C"] += 1
            continue

        # —— C.5) Diarization-anchored fallback
        # TalkNet has no confident speaker, but diarization knows who's talking.
        # Pick the detected face closest to that speaker's known position.
        spk = speaker_array[fi]
        if faces and spk is not None:
            scene_s = int(frame_to_scene_start[fi])
            spk_positions = scene_spk_x_map.get(scene_s, {})
            if spk in spk_positions:
                target_x = spk_positions[spk]
                closest = min(faces, key=lambda f: abs(_face_cx(f) - target_x))
                largest = max(faces, key=_face_area)
                # If the speaker's face is much smaller than the dominant face,
                # the camera is framing a different subject (reaction/cutaway shot).
                # In that case prefer the most prominent face over positional anchor.
                if _face_area(closest) < 0.35 * _face_area(largest):
                    best = largest
                else:
                    best = closest
            elif spk in clip_side_map:
                faces_by_x = sorted(faces, key=lambda f: f["x1"])
                best = faces_by_x[min(clip_side_map[spk], len(faces_by_x) - 1)]
            else:
                best = max(faces, key=_face_area)
            
            raw_n_spk[fi] = 1
            raw_spk_cx[fi, 0] = _norm_x(best)
            raw_spk_cy[fi, 0] = _norm_y(best)
            path_counts["C5"] += 1
            continue

        if faces:
            best = max(faces, key=_face_area)
            raw_n_spk[fi] = 1
            raw_spk_cx[fi, 0] = _norm_x(best)
            raw_spk_cy[fi, 0] = _norm_y(best)
            path_counts["D"] += 1
        else:
            path_counts["NOFACE"] += 1

    # ── 7. Stabilise each speaker-count level independently ──────────────────
    #
    # For n ≥ k, we build a boolean mask and run the two-pass stabiliser.
    # Higher speaker counts require longer minimum durations (more evidence needed).
    # We cascade from 4 → 3 → 2 → 1 so that a stable 4-speaker detection
    # takes priority over a stable 2-speaker detection in the same frame.
    #
    stable_n = np.ones(frames_count, dtype=int)  # default: single speaker

    for n_level, min_entry, min_gap in [
        (2, MIN_SPLIT_2_ENTRY, MIN_SPLIT_2_GAP),
        (3, MIN_SPLIT_3_ENTRY, MIN_SPLIT_3_GAP),
        (4, MIN_SPLIT_4_ENTRY, MIN_SPLIT_4_GAP),
    ]:
        raw_mask   = raw_n_spk >= n_level
        stable_mask = _stabilize_bool_state(raw_mask, scene_boundaries, min_entry, min_gap)
        stable_n[stable_mask] = n_level   # higher counts overwrite lower

    # Logging
    for level in [2, 3, 4]:
        raw_c    = int((raw_n_spk  >= level).sum())
        stable_c = int((stable_n   >= level).sum())
        if raw_c > 0 or stable_c > 0:
            logger.info(
                f"  Split-{level}: {raw_c} raw → {stable_c} stable "
                f"(removed {raw_c - stable_c} flickering frames)"
            )

    # ── 7c. Per-slot speaker-identity debounce ────────────────────────────────
    #
    # After the split-layout decision is stable, independently debounce each
    # speaker slot's cx position.  If TalkNet suddenly assigns speaking=True to
    # a face at a very different x position (camera angle cut), require that
    # new position to persist for MIN_SPEAKER_SWITCH_FRAMES consecutive frames
    # before the framing commits.  Until then, the last known position is held.
    #
    # This runs across the FULL clip (not per-scene) so that a scene boundary
    # does not reset the debounce counter mid-cut — the whole-clip view ensures
    # a brief reaction-shot scene is correctly suppressed.
    #
    # ── 7c. Per-slot speaker-identity debounce ────────────────────────────────
    IDENTITY_PX_THRESHOLD = 0.238  # 304px / 1280 — normalised coords
    if MIN_SPEAKER_SWITCH_FRAMES > 0:
        for slot in range(4):
            col = raw_spk_cx[:, slot].copy()
            if np.any(col != -1):
                raw_spk_cx[:, slot] = _stabilize_speaker_identity(
                    col, MIN_SPEAKER_SWITCH_FRAMES, IDENTITY_PX_THRESHOLD
                )
    else:
        # Instant tracking mode: fill gaps by holding the last known position
        # (forward-fill then backward-fill).
        #
        # Why NOT np.interp (linear interpolation):
        # Linear interpolation creates an artificial sliding trajectory between
        # detections. For a stationary speaker at x=0.4 with a 20-frame ASD
        # gap, interp produces values from 0.4 → wherever they're next detected,
        # even if they never moved. smooth_segment then sees that spread, its
        # cluster std exceeds STATIONARY_STD_THRESHOLD, the lock fails, and the
        # speaker is misclassified as "moving" — causing visible drift in a
        # completely stationary shot.
        #
        # Hold-fill keeps gaps at the last known position, so a stationary
        # speaker's array stays constant → tight cluster → lock fires → zero drift.
        def _hold_fill(arr: np.ndarray, vidx: np.ndarray, aidx: np.ndarray) -> np.ndarray:
            """Forward-fill -1 gaps then backward-fill any leading -1 region."""
            vmask = arr != -1
            fwd = np.where(vmask, aidx, 0)
            np.maximum.accumulate(fwd, out=fwd)
            out = arr[fwd]
            first_v = vidx[0]
            if first_v > 0:
                out[:first_v] = arr[first_v]
            return out

        logger.info(
            f"  [Path usage] A={path_counts['A']} B={path_counts['B']} "
            f"C={path_counts['C']} C5={path_counts['C5']} "
            f"D={path_counts['D']} NOFACE={path_counts['NOFACE']} "
            f"(total={frames_count})"
        )

        for slot in range(4):
            for seg_start, seg_end in zip(scene_boundaries[:-1], scene_boundaries[1:]):
                seg = slice(seg_start, seg_end)
                
                # X coordinate hold-fill
                col_x = raw_spk_cx[seg, slot]
                valid_x = col_x != -1
                if np.any(valid_x):
                    valid_idx = np.where(valid_x)[0]
                    all_idx = np.arange(len(col_x))
                    raw_spk_cx[seg, slot] = _hold_fill(col_x, valid_idx, all_idx)

                # Y coordinate hold-fill
                col_y = raw_spk_cy[seg, slot]
                valid_y = col_y != -1
                if np.any(valid_y):
                    valid_y_idx = np.where(valid_y)[0]
                    all_y_idx = np.arange(len(col_y))
                    raw_spk_cy[seg, slot] = _hold_fill(col_y, valid_y_idx, all_y_idx)

    # ── 7d. Scene-boundary layout snapping ────────────────────────────────────
    # If a layout change (n=1 -> n=2) happens within 50 frames of a scene cut,
    # snap the layout change to the exact frame of the scene cut.
    for cut in scene_boundaries:
        # Look forward/backward 50 frames
        window = range(max(0, cut - 50), min(frames_count, cut + 50))
        for f in window:
            if f > 0 and stable_n[f] != stable_n[f-1]:
                # Snap!
                stable_n[min(f, cut):max(f, cut)] = stable_n[f]
                break

    # ── 8. Scene-segmented per-speaker smoothing ──────────────────────────────
    #
    # Both cx and cy are smoothed independently per scene per slot.
    smooth_spk_cx = np.full((frames_count, 4), -1.0)
    smooth_spk_cy = np.full((frames_count, 4), -1.0)

    cx_defaults = [0.5, 0.25, 0.75, 0.5]
    cy_defaults = [0.35] * 4

    for slot in range(4):
        raw_cx_col = raw_spk_cx[:, slot].copy()
        raw_cy_col = raw_spk_cy[:, slot].copy()
        for seg_start, seg_end in zip(scene_boundaries[:-1], scene_boundaries[1:]):
            seg = slice(seg_start, seg_end)
            smooth_spk_cx[seg_start:seg_end, slot] = _smooth_segment(
                raw_cx_col[seg].copy(), cx_defaults[slot], SIGMA
            )
            smooth_spk_cy[seg_start:seg_end, slot] = _smooth_segment(
                raw_cy_col[seg].copy(), cy_defaults[slot], SIGMA
            )

    logger.info("Rendering adaptive multi-speaker reframing...")

    # ── 8.5. Final post-smoothing shot enforcement ────────────────────────────
    # Enforces a strict 50-frame minimum by looking at the ACTUAL rendered
    # parameters (layout + smoothed cx).
    MIN_SHOT_FRAMES = 50
    changed = True
    while changed:
        changed = False
        i = 0
        while i < frames_count:
            val_n = stable_n[i]
            val_cx = smooth_spk_cx[i, 0]
            j = i
            while j < frames_count:
                if stable_n[j] != val_n: break
                if abs(smooth_spk_cx[j, 0] - val_cx) > 0.023: break  # ~30px / 1280
                j += 1
            
            duration = j - i
            if duration < MIN_SHOT_FRAMES and j < frames_count:
                # Swallow the flicker!
                stable_n[i:j] = stable_n[j]
                for s in range(4):
                    smooth_spk_cx[i:j, s] = smooth_spk_cx[j, s]
                    smooth_spk_cy[i:j, s] = smooth_spk_cy[j, s]
                changed = True
                break # Restart loop to re-evaluate
            i = j

    # ── 9. Build chunk_meta for subtitle positioning ──────────────────────────
    chunk_meta: List[Dict[str, Any]] = []
    if frames_count > 0:
        # Map stable_n == 1 → "SINGLE", ≥ 2 → "SPLIT"
        def _flag(n: int) -> str:
            return "SINGLE" if n == 1 else "SPLIT"

        cur_flag  = _flag(stable_n[0])
        cur_start = 0

        for fi in range(1, frames_count):
            flag = _flag(stable_n[fi])
            if flag != cur_flag:
                chunk_meta.append({
                    "start_frame": cur_start,
                    "end_frame":   fi - 1,
                    "start_ms":    (cur_start / fps) * 1000.0,
                    "end_ms":      ((fi - 1)  / fps) * 1000.0,
                    "flag":        cur_flag,
                    "motion":      "STATIONARY",
                    "med_x":       float(smooth_spk_cx[cur_start:fi, 0].mean()),
                })
                cur_flag  = flag
                cur_start = fi
        
        chunk_meta.append({
            "start_frame": cur_start,
            "end_frame":   frames_count - 1,
            "start_ms":    (cur_start          / fps) * 1000.0,
            "end_ms":      ((frames_count - 1) / fps) * 1000.0,
            "flag":        cur_flag,
            "motion":      "STATIONARY",
            "med_x":       float(smooth_spk_cx[cur_start:frames_count, 0].mean()),
        })

    # ── 10. Render ─────────────────────────────────────────────────────────────
    # Re-open capture to reliably restart from frame 0 (fixes OpenCV seek bug)
    cap.release()
    cap = cv2.VideoCapture(clip_file)
    render_cap = cap  # Keep reference for finally block
    render_writer = writer  # Keep reference for finally block
    
    try:
        for fidx in range(frames_count):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            n  = int(stable_n[fidx])
            cx = smooth_spk_cx[fidx]   # shape (4,) cx per speaker slot
            cy = smooth_spk_cy[fidx]   # shape (4,) cy per speaker slot

            if n == 4:
                out_frame = _render_split_4(
                    frame, cx[0] * w, cx[1] * w, cx[2] * w, cx[3] * w,
                    cy[0] * h, cy[1] * h, cy[2] * h, cy[3] * h,
                )
            elif n == 3:
                out_frame = _render_split_3(
                    frame, cx[1] * w, cx[0] * w, cx[2] * w,
                    cy[1] * h, cy[0] * h, cy[2] * h,
                )
            elif n == 2:
                out_frame = _render_split_2(frame, cx[0] * w, cx[1] * w, cy[0] * h, cy[1] * h)
            else:
                out_frame = _cell_full(frame, cx[0] * w, cy[0] * h)

            writer.write(out_frame)
    finally:
        # Ensure resources are released even if an error occurs
        if render_writer is not None:
            render_writer.release()
        if render_cap is not None:
            render_cap.release()

    logger.info(f"Tracking complete. Output: {out_path}")
    return out_path, chunk_meta
