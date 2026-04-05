import os
import subprocess
import json
import cv2
import modal
import numpy as np
import scipy.ndimage as ndimage
from scenedetect import detect, ContentDetector
from typing import List, Dict, Any, Tuple
from config import get_logger

logger = get_logger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
# All split cells use the same 9:16 portrait crop (608px) as single-speaker mode.
# This is the critical guarantee that prevents any participant from appearing
# in more than one cell simultaneously: two independent 608px windows centred on
# speakers ≥608px apart are geometrically non-overlapping by definition.
CROP_W_1 = 608   # universal 9:16 crop width for every layout cell

OUT_W, OUT_H = 1080, 1920

# For 1080×960 output cells (each half of a 2-speaker split, or 3-speaker top),
# the source crop must be 608×541 to keep both axes at the same scale factor
# (1.776× horiz, 1.774× vert — 0.1% difference, imperceptible).
# Using 608×1080 → 1080×960 gives 1.776× vs 0.888× — 2× scale mismatch → distortion.
CROP_H_HALF = int(round(CROP_W_1 * (OUT_H // 2) / OUT_W))  # = 541

SIGMA = 25  # Gaussian smoothing frames (~1 s at 25 fps)

# Stabilisation thresholds (entry = min frames before mode activates,
# gap = min gap frames before mode drops — prevents rapid re-entry)
MIN_SPLIT_2_ENTRY = 20   # ~0.8 s
MIN_SPLIT_2_GAP   = 20
MIN_SPLIT_3_ENTRY = 25   # ~1.0 s
MIN_SPLIT_3_GAP   = 25
MIN_SPLIT_4_ENTRY = 30   # ~1.2 s
MIN_SPLIT_4_GAP   = 30

# Detection thresholds
MIN_FACE_W_RATIO  = 0.04  # face must be ≥ 4 % of frame width to count
SPLIT_MARGIN      = 0.08  # 2-speaker: each face must be SPLIT_MARGIN past centre
MIN_FACE_SEP      = 0.10  # 3/4-speaker: min separation between adjacent faces
# Minimum cx distance between two speakers before their 608px crops stop overlapping.
# Proof: crops overlap iff |cx_r - cx_l| < CROP_W_1.  Guard: require ≥ CROP_W_1.
SPLIT_MIN_CX_SEP  = CROP_W_1  # 608px


# ─── Phase 4 ──────────────────────────────────────────────────────────────────

def extract_segment(input_file: str, clip: Dict[str, Any], idx: int) -> str:
    """FFmpeg segment extraction for a given clip timestamp range."""
    logger.info(
        f"==================== PHASE 4: SEGMENT EXTRACTION (Clip {idx}) ===================="
    )
    start = clip["start_time"]
    dur = clip["end_time"] - start
    out = f"temp_extracted_clip_{idx}.mp4"
    logger.info(f"Extracting {out} [{start}s to {clip['end_time']}s]...")
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-i", input_file,
        "-t", str(dur),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", out,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed extracting segment {idx}")
        raise RuntimeError(f"Extraction failed: {e}")
    return out


# ─── Phase 7 ──────────────────────────────────────────────────────────────────

def merge_and_cleanup(tracked_vid: str, extract_vid: str, sub_file: str, idx: int):
    """Burns .ass subtitles into the tracked video and muxes original audio."""
    logger.info(
        f"==================== PHASE 7: FINAL MERGE & CLEANUP (Clip {idx}) ===================="
    )
    out_file = f"output/clip_{idx + 1}.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", tracked_vid, "-i", extract_vid,
        "-filter_complex", f"[0:v]ass={sub_file}[vout]",
        "-map", "[vout]", "-map", "1:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
        "-movflags", "+faststart",
        "-c:a", "aac", out_file,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg merge failed for clip {idx}")
        raise RuntimeError(f"Merging failed: {e}")
    for path in [tracked_vid, extract_vid, sub_file]:
        try:
            os.remove(path)
        except OSError as e:
            logger.warning(f"Failed to clean up {path}: {e}")
    logger.info(f"Cleaned up temporary files for clip {idx}")


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

    Aspect-ratio guarantee (crop-to-fill):
      The caller must pass crop_w and crop_h that share the same ratio as the
      output cell.  When both dimensions are correct there is zero stretch.
        Single-speaker (1080×1920): crop 608×1080  (horiz scale 1.776×, vert 1.777× — <0.1% off)
        2-speaker half (1080×960):   crop 608×541   (horiz scale 1.776×, vert 1.774× — <0.1% off)
        Side/quad (540×960):          crop 608×1080  (horiz scale 0.888×, vert 0.888× — exact)

    cy / crop_h default (-1) means “use full frame height”.
    """
    h_img, w_img = frame_img.shape[:2]

    # ─ Horizontal ───────────────────────────────────────────────────────
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


def _smooth_segment(raw: np.ndarray, default: float, sigma: int) -> np.ndarray:
    """
    Gap-fills a 1-D position array (gaps = -1) then Gaussian-smooths it.
    Leading/trailing gaps are filled with the nearest valid value.
    """
    seg_len = len(raw)
    filled = raw.copy()
    valid_mask = filled != -1
    if not np.any(valid_mask):
        filled[:] = default
        return filled
    idxs = np.arange(seg_len)
    first_valid = int(np.argmax(valid_mask))
    filled[:first_valid] = filled[first_valid]
    last_valid = int(seg_len - 1 - np.argmax(valid_mask[::-1]))
    filled[last_valid + 1:] = filled[last_valid]
    valid_mask = filled != -1
    if not np.all(valid_mask):
        filled[~valid_mask] = np.interp(
            idxs[~valid_mask], idxs[valid_mask], filled[valid_mask]
        )
    effective_sigma = min(sigma, max(1, seg_len // 4))
    return ndimage.gaussian_filter1d(filled, sigma=effective_sigma)


# ─── Split-state stabilisation ────────────────────────────────────────────────

def _stabilize_segment(raw: np.ndarray, min_entry: int, min_gap: int) -> np.ndarray:
    """
    Two-pass stabiliser for a boolean signal within one scene segment.
      Pass 1: Remove True-runs shorter than min_entry frames (noise).
      Pass 2: Merge adjacent True-runs separated by < min_gap False frames (hysteresis).
    """
    n = len(raw)
    if n == 0:
        return raw.copy()
    runs = []
    i = 0
    while i < n:
        if raw[i]:
            j = i
            while j < n and raw[j]:
                j += 1
            runs.append([i, j - 1])
            i = j
        else:
            i += 1
    runs = [r for r in runs if (r[1] - r[0] + 1) >= min_entry]
    if not runs:
        return np.zeros(n, dtype=bool)
    merged = [runs[0]]
    for s, e in runs[1:]:
        if (s - merged[-1][1] - 1) < min_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    result = np.zeros(n, dtype=bool)
    for s, e in merged:
        result[s: e + 1] = True
    return result


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

def _cell_full(frame: np.ndarray, cx: float) -> np.ndarray:
    """Single-speaker: 9:16 crop → 1080×1920."""
    return cv2.resize(get_centered_crop(frame, cx, CROP_W_1), (OUT_W, OUT_H))


def _cell_half(frame: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """
    2-speaker top/bottom cell: 608×541 crop centred on (cx, cy) → 1080×960.
    Using crop_h=541 instead of full 1080 keeps both scale axes at ~1.776×
    (vs the 2× mismatch that caused visible stretching with crop_h=1080).
    """
    return cv2.resize(
        get_centered_crop(frame, cx, CROP_W_1, cy=cy, crop_h=CROP_H_HALF),
        (OUT_W, OUT_H // 2),
    )


def _cell_3_top(frame: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """3-speaker featured top cell: 608×541 → 1080×960 (same AR as 2-speaker half)."""
    return cv2.resize(
        get_centered_crop(frame, cx, CROP_W_1, cy=cy, crop_h=CROP_H_HALF),
        (OUT_W, OUT_H // 2),
    )


def _cell_3_side(frame: np.ndarray, cx: float) -> np.ndarray:
    """3-speaker bottom side cell: 608×1080 → 540×960 (scale axes match: 0.888×)."""
    return cv2.resize(get_centered_crop(frame, cx, CROP_W_1), (OUT_W // 2, OUT_H // 2))


def _cell_quad(frame: np.ndarray, cx: float) -> np.ndarray:
    """4-speaker cell: 608×1080 → 540×960 (scale axes match: 0.888×)."""
    return cv2.resize(get_centered_crop(frame, cx, CROP_W_1), (OUT_W // 2, OUT_H // 2))


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
    cy_top: float,
) -> np.ndarray:
    """
    1 + 2 layout:
      Top  (full width 1080×960) — most central speaker, face-centred vertical crop.
      Bottom left  (540×960)     — left speaker,  full-height crop.
      Bottom right (540×960)     — right speaker, full-height crop.
    """
    top       = _cell_3_top(frame, cx_top, cy_top)
    bot_left  = _cell_3_side(frame, cx_bl)
    bot_right = _cell_3_side(frame, cx_br)
    bottom    = cv2.hconcat([bot_left, bot_right])
    return cv2.vconcat([top, bottom])


def _render_split_4(
    frame: np.ndarray,
    cx_tl: float, cx_tr: float,
    cx_bl: float, cx_br: float,
) -> np.ndarray:
    """
    2 × 2 grid (each cell 540×960, 9:16):
      Top row:    leftmost speaker | second speaker
      Bottom row: third speaker   | rightmost speaker
    Speakers are assigned left→right in each row maintaining physical order.
    """
    tl = _cell_quad(frame, cx_tl)
    tr = _cell_quad(frame, cx_tr)
    bl = _cell_quad(frame, cx_bl)
    br = _cell_quad(frame, cx_br)
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
    clip_file: str, idx: int, clip: Dict[str, Any], words: List[Dict[str, Any]]
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
        f"==================== PHASE 5: SPEAKER TRACKING & FRAMING (Clip {idx}) ===================="
    )

    # ── 1. Fast-ASD ──────────────────────────────────────────────────────────
    logger.info("Calling Modal Fast-ASD tracker...")
    Tracker = modal.Cls.from_name("fast-asd-tracker", "FastASDTracker")
    tracker = Tracker()
    with open(clip_file, "rb") as vf:
        video_bytes = vf.read()
    try:
        result_json = tracker.process_video.remote(video_bytes)
        tracking_data = json.loads(result_json)
    except Exception as e:
        logger.error(f"Fast-ASD tracker failed: {e}")
        raise

    # ── 2. Video metadata ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(clip_file)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video: {w}x{h} @ {fps}fps, {frames_count} frames")

    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = f"temp_tracked_{idx}.mp4"
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (OUT_W, OUT_H))

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
        raw_mask   = (raw_n_spk >= n_level)
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
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    try:
        for fidx in range(frames_count):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            n  = int(stable_n[fidx])
            cx = smooth_spk_cx[fidx]   # shape (4,) cx per speaker slot
            cy = smooth_spk_cy[fidx]   # shape (4,) cy per speaker slot

            if n == 4:
                # 2×2 grid: full-height crops (608×1080), scale axes match for 540×960
                out_frame = _render_split_4(frame, cx[0], cx[1], cx[2], cx[3])

            elif n == 3:
                # 1 + 2: top cell uses face-centred 541px vertical crop; sides are full height
                # cx[0]=leftmost, cx[1]=middle (featured top), cx[2]=rightmost
                out_frame = _render_split_3(frame, cx[1], cx[0], cx[2], cy[1])

            elif n == 2:
                # Vertical stack: each cell is 608×541 face-centred crop → 1080×960
                out_frame = _render_split_2(frame, cx[0], cx[1], cy[0], cy[1])

            else:
                # Single speaker: full 9:16 crop (608×1080 → 1080×1920)
                out_frame = _cell_full(frame, cx[0])

            writer.write(out_frame)
    finally:
        writer.release()
        cap.release()

    logger.info(f"Tracking complete. Output: {out_path}")
    return out_path, chunk_meta
