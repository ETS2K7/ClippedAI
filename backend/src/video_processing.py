# pylint: disable=no-member,too-many-locals,too-many-statements,too-many-branches,too-many-arguments,too-many-positional-arguments

import os
import subprocess
import json
from typing import List, Dict, Any, Tuple

import cv2
import modal
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

# Gaussian smoothing frames (~1 s at 25 fps)
SIGMA = 25  

# Stabilisation thresholds (entry = min frames before mode activates,
# gap = min gap frames before mode drops — prevents rapid re-entry)
MIN_SPLIT_2_ENTRY = 20   # ~0.8 s
MIN_SPLIT_2_GAP   = 20
MIN_SPLIT_3_ENTRY = 25   # ~1.0 s
MIN_SPLIT_3_GAP   = 25
MIN_SPLIT_4_ENTRY = 30   # ~1.2 s
MIN_SPLIT_4_GAP   = 30

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
        "-ss", str(start),
        "-i", input_file,
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

def merge_and_cleanup(tracked_vid: str, extract_vid: str, sub_file: str, idx: int, work_dir: str = "", use_gpu: bool = False):
    """Merges subtitle-ass video with audio from original segment.

    Uses FFmpeg to burn in ASS subtitles and copy audio from the extracted segment.
    This ensures the output has both video (with subs) and audio in the correct format.
    
    Optimized with NVENC GPU acceleration when available.
    """
    logger.info(
        f"==================== PHASE 7: MERGE & CLEANUP (Clip {idx}) ===================="
    )
    out_file = f"{work_dir}/clip_{idx}.mp4" if work_dir else f"output/clip_{idx}.mp4"

    # Re-encode tracked .avi (MJPG) to H.264, burn in ASS subtitles, and mux audio.
    # -c:v libx264 is always available; we intentionally do NOT use NVENC here because
    # MJPG pixel format (yuvj420p) requires colour-range conversion that NVENC rejects.
    # The ass= filter path must have colons escaped for FFmpeg's filter syntax on Linux.
    safe_sub = sub_file.replace("\\", "/").replace(":", "\\:")
    cmd = [
        "ffmpeg", "-y",
        "-i", tracked_vid,
        "-i", extract_vid,
        "-vf", f"ass={safe_sub}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        out_file,
    ]

    _run_ffmpeg(cmd, f"merge for clip {idx}")

    # Clean up intermediate files
    try:
        os.remove(tracked_vid)
        os.remove(extract_vid)
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


def _stabilize_bool_state(
    raw: np.ndarray,
    scene_boundaries: List[int],
    min_entry: int,
    min_gap: int,
) -> np.ndarray:
    """Applies _stabilize_segment per scene independently."""
    result = np.zeros(len(raw), dtype=bool)
    for seg_start, seg_end in zip(scene_boundaries[:-1], scene_boundaries[1:]):
        result[seg_start:seg_end] = _stabilize_segment(
            raw[seg_start:seg_end].copy(), min_entry, min_gap
        )
    return result


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
    clip_file: str, idx: int, clip: Dict[str, Any], words: List[Dict[str, Any]], work_dir: str = ""
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
    """
    logger.info(
        f"==================== PHASE 5: SPEAKER TRACKING & FRAMING (Clip {idx}) "
        f"====================\n"
    )

    # ── 1. Fast-ASD ──────────────────────────────────────────────────────────
    logger.info("Calling Modal Fast-ASD tracker...")
    Tracker = modal.Cls.from_name("fast-asd-tracker", "FastASDTracker")
    tracker = Tracker()
    
    # Check file size before loading into memory to prevent memory exhaustion
    file_size_mb = os.path.getsize(clip_file) / (1024 * 1024)
    MAX_VIDEO_SIZE_MB = 500  # 500MB limit for safety
    
    if file_size_mb > MAX_VIDEO_SIZE_MB:
        logger.warning(
            f"Video file is large ({file_size_mb:.1f}MB). "
            f"Loading into memory may cause issues. Consider using shorter clips."
        )
    
    with open(clip_file, "rb") as vf:
        video_bytes = vf.read()
    try:
        result_json = tracker.process_video.remote(video_bytes)
        tracking_data = json.loads(result_json)
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

    # ── 6. Build raw per-frame speaker arrays (N = 1..4) ─────────────────────
    #
    # raw_n_spk[fi]         = number of distinct speakers detected (1–4).
    # raw_spk_cx[fi, 0..3]  = their cx positions (left→right), -1 if absent.
    #
    # Detection hierarchy for frame fi:
    #   A) TalkNet simultaneous speaking (both marked speaking=True)    → highest confidence
    #   B) Visual: ≥2 prominent, well-separated faces in multi-spk scene → standard path
    #   C) Single TalkNet speaking face                                  → single speaker
    #   D) Best-scoring face (no confirmed speaker)                      → single speaker fallback
    #
    raw_n_spk  = np.ones(frames_count, dtype=int)
    raw_spk_cx = np.full((frames_count, 4), -1.0)
    raw_spk_cy = np.full((frames_count, 4), -1.0)  # vertical face centres for crop positioning

    for fi in range(frames_count):
        faces    = frame_faces.get(fi, [])
        speaking = [f for f in faces if f.get("speaking", False)]
        scene_s  = int(frame_to_scene_start[fi])
        n_spk_scene = scene_speaker_count.get(scene_s, 1)

        # —— A) TalkNet simultaneous (keeps backward compat, rarely fires but high confidence)
        if len(speaking) >= 2:
            by_x = sorted(speaking[:4], key=_face_cx)
            n = len(by_x)
            raw_n_spk[fi] = n
            for i, f in enumerate(by_x):
                raw_spk_cx[fi, i] = _face_cx(f)
                raw_spk_cy[fi, i] = _face_cy(f)
            continue

        # —— B) Visual presence (active-speaker only, geometrically deduplicated)
        #
        # Rules enforced here:
        #   1. Only active-speaker scenes trigger split (diarization gate maintained).
        #      Non-speakers are never assigned their own cell.
        #   2. Each participant appears in exactly ONE cell. This is guaranteed by
        #      the cx-separation guard: two 608px crops centred ≥608px apart are
        #      non-overlapping by geometry, so neither speaker can appear in the
        #      other's cell.
        #   3. 3/4-speaker layouts require stronger diarization confirmation.
        if len(faces) >= 2 and n_spk_scene >= 2:
            distinct = _prominent_distinct_faces(faces, w)

            if len(distinct) >= 4 and n_spk_scene >= 4:
                # Four-speaker layout candidate — verify all adjacent pairs are
                # geometrically separable (non-overlapping 608px crops).
                cxs = [_face_cx(f) for f in distinct[:4]]
                cys = [_face_cy(f) for f in distinct[:4]]
                separable = all(
                    cxs[i + 1] - cxs[i] >= SPLIT_MIN_CX_SEP
                    for i in range(len(cxs) - 1)
                )
                if separable:
                    raw_n_spk[fi] = 4
                    for i, (cx_v, cy_v) in enumerate(zip(cxs, cys)):
                        raw_spk_cx[fi, i] = cx_v
                        raw_spk_cy[fi, i] = cy_v
                    continue

            if len(distinct) >= 3 and n_spk_scene >= 3:
                cxs = [_face_cx(f) for f in distinct[:3]]
                cys = [_face_cy(f) for f in distinct[:3]]
                separable = all(
                    cxs[i + 1] - cxs[i] >= SPLIT_MIN_CX_SEP
                    for i in range(len(cxs) - 1)
                )
                if separable:
                    raw_n_spk[fi] = 3
                    for i, (cx_v, cy_v) in enumerate(zip(cxs, cys)):
                        raw_spk_cx[fi, i] = cx_v
                        raw_spk_cy[fi, i] = cy_v
                    continue

            if len(distinct) >= 2:
                left_cx  = _face_cx(distinct[0])
                right_cx = _face_cx(distinct[-1])

                # Diarization gate: scene has confirmed ≥2 active speakers AND
                # each face is clearly anchored in its own half of the frame.
                clearly_left  = left_cx  < w * (0.5 - SPLIT_MARGIN)
                clearly_right = right_cx > w * (0.5 + SPLIT_MARGIN)
                diarization_ok = clearly_left and clearly_right

                # Geometric uniqueness guard: crops must be non-overlapping.
                # Two 608px crops centred on left_cx and right_cx overlap iff
                # right_cx - left_cx < 608.  Reject the split if too close.
                geometrically_separable = (right_cx - left_cx) >= SPLIT_MIN_CX_SEP

                if diarization_ok and geometrically_separable:
                    raw_n_spk[fi] = 2
                    raw_spk_cx[fi, 0] = left_cx
                    raw_spk_cx[fi, 1] = right_cx
                    raw_spk_cy[fi, 0] = _face_cy(distinct[0])
                    raw_spk_cy[fi, 1] = _face_cy(distinct[-1])
                    continue

        # —— C) Single TalkNet-confirmed speaker
        if len(speaking) == 1:
            raw_n_spk[fi] = 1
            raw_spk_cx[fi, 0] = _face_cx(speaking[0])
            raw_spk_cy[fi, 0] = _face_cy(speaking[0])
            continue

        # —— D) Best-scoring detected face (fallback)
        if faces:
            best = max(faces, key=_face_score)
            raw_n_spk[fi] = 1
            raw_spk_cx[fi, 0] = _face_cx(best)
            raw_spk_cy[fi, 0] = _face_cy(best)

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

    # ── 8. Scene-segmented per-speaker smoothing ──────────────────────────────
    #
    # Both cx and cy are smoothed independently per scene per slot.
    smooth_spk_cx = np.full((frames_count, 4), -1.0)
    smooth_spk_cy = np.full((frames_count, 4), -1.0)

    cx_defaults = [w / 2.0, w * 0.25, w * 0.75, w / 2.0]
    cy_defaults = [h * 0.35] * 4   # faces sit in the upper portion of talking-head footage

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
                # 2×2 grid: AR-safe crop per speaker → 540×960 cells
                out_frame = _render_split_4(
                    frame, cx[0], cx[1], cx[2], cx[3],
                    cy[0], cy[1], cy[2], cy[3],
                )

            elif n == 3:
                # 1 + 2: top cell face-centred, bottom cells face-centred
                # cx[0]=leftmost, cx[1]=middle (featured top), cx[2]=rightmost
                out_frame = _render_split_3(
                    frame, cx[1], cx[0], cx[2],
                    cy[1], cy[0], cy[2],
                )

            elif n == 2:
                # Vertical stack: AR-safe crop per speaker → 1080×960 cells
                out_frame = _render_split_2(frame, cx[0], cx[1], cy[0], cy[1])

            else:
                # Single speaker: AR-safe 9:16 crop → 1080×1920
                out_frame = _cell_full(frame, cx[0], cy[0])

            writer.write(out_frame)
    finally:
        # Ensure resources are released even if an error occurs
        if render_writer is not None:
            render_writer.release()
        if render_cap is not None:
            render_cap.release()

    logger.info(f"Tracking complete. Output: {out_path}")
    return out_path, chunk_meta
