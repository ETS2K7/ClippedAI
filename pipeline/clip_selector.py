"""
Pipeline Step 5 — Clip Selection
Candidate generation via sliding window + Cerebras LLM ranking.
This is the CORE IP — clip quality lives or dies here.
"""

import os
import sys
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    WINDOW_SIZES, WINDOW_STEP_RATIO, CANDIDATE_MULTIPLIER, OVERLAP_THRESHOLD,
    ENERGY_WEIGHT, SCENE_WEIGHT, KEYWORD_WEIGHT, VIRAL_KEYWORDS,
    CEREBRAS_MODEL, CEREBRAS_API_URL, CEREBRAS_TEMPERATURE, CEREBRAS_MAX_TOKENS,
    LLM_SYSTEM_PROMPT, RD_MODE,
)


def select_clips(
    transcript: dict,
    scenes: list,
    audio: dict,
    video_title: str,
    settings: dict,
) -> list:
    """
    Generate candidates from signals, then rank with LLM.

    Returns: List of selected clips with titles, scores, and metadata.
    """
    max_clips = settings.get("max_clips", 5)
    min_duration = settings.get("min_duration", 15)
    max_duration = settings.get("max_duration", 60)

    # Phase A: Generate candidates
    candidates = generate_candidates(
        transcript, scenes, audio, min_duration, max_duration, max_clips
    )

    if not candidates:
        print("  ⚠️ No candidates generated — video may be too short")
        return []

    print(f"  🧠 Generated {len(candidates)} candidates, sending to LLM...")

    # Phase B: Rank with LLM
    ranked = rank_with_llm(candidates, video_title, max_clips)

    return ranked


def generate_candidates(
    transcript: dict,
    scenes: list,
    audio: dict,
    min_duration: float,
    max_duration: float,
    max_clips: int,
) -> list:
    """
    Generate candidate clip regions using TWO strategies:
    1. Sliding windows (coverage-based)
    2. Event-centered windows (around peaks, scene changes)

    Scores each window with energy, scene density, and keyword signals.
    """
    segments = transcript["segments"]
    if not segments:
        return []

    all_windows = []
    target_count = max_clips * CANDIDATE_MULTIPLIER

    # Start from first speech, not t=0 (avoids awkward silence openings)
    first_speech_start = segments[0]["start"]
    video_duration = segments[-1]["end"]

    # ── Strategy 1: Sliding windows (coverage) ──
    for window_size in WINDOW_SIZES:
        step = window_size * WINDOW_STEP_RATIO
        t = first_speech_start
        while t + min_duration <= video_duration:
            window = _make_window(segments, audio, scenes, t, t + window_size,
                                  min_duration, max_duration)
            if window:
                all_windows.append(window)
            t += step

    # ── Strategy 2: Event-centered windows ──
    # Center windows around energy peaks
    for peak in audio.get("peaks", []):
        center = peak["time"]
        for radius in [10, 15]:  # Try 20s and 30s clips centered on peak
            win_start = max(first_speech_start, center - radius)
            win_end = min(video_duration, center + radius)
            window = _make_window(segments, audio, scenes, win_start, win_end,
                                  min_duration, max_duration)
            if window:
                all_windows.append(window)

    # Center windows around scene boundaries
    for scene in scenes:
        scene_time = scene["start"]
        # Clip starting at scene change (hook potential)
        for length in [15, 25]:
            win_start = scene_time
            win_end = min(video_duration, scene_time + length)
            window = _make_window(segments, audio, scenes, win_start, win_end,
                                  min_duration, max_duration)
            if window:
                all_windows.append(window)

    # Deduplicate overlapping windows
    all_windows.sort(key=lambda w: w["composite_score"], reverse=True)
    deduped = _deduplicate(all_windows)

    # Return top N
    result = deduped[:target_count]
    for i, c in enumerate(result):
        c["id"] = i

    print(f"    (sliding: {len(all_windows)} raw, {len(deduped)} deduped, {len(result)} kept)")
    return result


def _make_window(segments, audio, scenes, start, end, min_dur, max_dur):
    """Score and validate a single candidate window. Returns dict or None."""
    actual_duration = end - start
    if actual_duration < min_dur:
        return None
    if actual_duration > max_dur:
        end = start + max_dur

    # Snap to sentence boundaries
    start_snapped, end_snapped = _snap_to_sentences(segments, start, end)
    if end_snapped - start_snapped < min_dur:
        return None

    text = _get_text_in_range(segments, start_snapped, end_snapped)
    if len(text.split()) < 10:
        return None

    energy_score = _score_energy(audio, start_snapped, end_snapped)
    scene_score = _score_scenes(scenes, start_snapped, end_snapped)
    keyword_score = _score_keywords(text)

    composite = (
        energy_score * ENERGY_WEIGHT +
        scene_score * SCENE_WEIGHT +
        keyword_score * KEYWORD_WEIGHT
    )

    # Extract hook text (first ~3 seconds of transcript)
    hook_text = _get_text_in_range(segments, start_snapped, start_snapped + 3.0)

    return {
        "start": round(start_snapped, 2),
        "end": round(end_snapped, 2),
        "duration": round(end_snapped - start_snapped, 2),
        "transcript_text": text,
        "hook_text": hook_text.strip() if hook_text else "",
        "energy_score": round(energy_score, 4),
        "scene_score": round(scene_score, 4),
        "keyword_score": round(keyword_score, 4),
        "composite_score": round(composite, 4),
    }


def rank_with_llm(candidates: list, video_title: str, max_clips: int) -> list:
    """
    Send candidates to Cerebras LLM for viral ranking.
    In RD_MODE: raises exception on failure (no silent fallback).
    In production: falls back to composite scoring.
    """
    import httpx

    api_key = os.environ.get("CEREBRAS_API_KEY", "")
    if not api_key:
        msg = "No CEREBRAS_API_KEY — cannot rank clips"
        if RD_MODE:
            raise RuntimeError(f"🛑 LLM FAILED: {msg}")
        print(f"  ⚠️ {msg}")
        return _fallback_ranking(candidates, max_clips, msg)

    # Build user prompt — structurally isolate hook text from full transcript
    candidate_blocks = []
    for c in candidates:
        block = (
            f"---\n"
            f"CANDIDATE {c['id']}\n"
            f"Time: {c['start']:.1f}s – {c['end']:.1f}s ({c['duration']:.0f}s)\n"
            f"Signals: energy={c['energy_score']:.2f}, "
            f"scenes={c['scene_score']:.2f}, keywords={c['keyword_score']:.2f}\n"
            f"HOOK (first 3 seconds): \"{c.get('hook_text', '')}\"\n"
            f"FULL TRANSCRIPT:\n\"{c['transcript_text'][:1500]}\"\n"
            f"---"
        )
        candidate_blocks.append(block)

    user_prompt = (
        f'Video title: "{video_title}"\n\n'
        f"Here are {len(candidates)} candidate clips. Rank them by virality potential.\n\n"
        f"{''.join(candidate_blocks)}\n\n"
        f"CRITICAL: Evaluate the FIRST 3 SECONDS of each candidate's transcript. "
        f"If the opening is weak, generic, or lacks a hook, mark keep=false.\n\n"
        f"Return a JSON array with exactly {len(candidates)} objects, one per candidate:\n"
        f"[\n"
        f'  {{\n'
        f'    "candidate_id": <int>,\n'
        f'    "rank": <int 1..N>,\n'
        f'    "virality_score": <float 0-1>,\n'
        f'    "keep": <bool>,\n'
        f'    "title": "<string, 5-10 words>",\n'
        f'    "description": "<string, 1-2 sentences>",\n'
        f'    "hashtags": ["<string>", ...],\n'
        f'    "hook_strength": "<weak|medium|strong>",\n'
        f'    "reasoning": "<string, why this is/isn\'t engaging>"\n'
        f"  }}\n"
        f"]\n\n"
        f"Return ONLY the JSON array. No other text."
    )

    # Call Cerebras API
    print(f"  🔗 Calling Cerebras API (model={CEREBRAS_MODEL}, {len(candidates)} candidates)...")
    try:
        response = httpx.post(
            CEREBRAS_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": CEREBRAS_MODEL,
                "messages": [
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": CEREBRAS_TEMPERATURE,
                "max_tokens": CEREBRAS_MAX_TOKENS,
            },
            timeout=120.0,
        )
        if response.status_code != 200:
            err = f"HTTP {response.status_code}: {response.text[:300]}"
            print(f"  ❌ Cerebras: {err}")
            if RD_MODE:
                raise RuntimeError(f"🛑 LLM FAILED: {err}")
            return _fallback_ranking(candidates, max_clips, err)
    except httpx.TimeoutException as e:
        err = f"Timeout after 120s: {e}"
        print(f"  ❌ Cerebras: {err}")
        if RD_MODE:
            raise RuntimeError(f"🛑 LLM FAILED: {err}")
        return _fallback_ranking(candidates, max_clips, err)

    # Parse response
    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        print(f"  📝 LLM response ({len(content)} chars)")

        # Try parsing as direct JSON
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                rankings = parsed
            elif isinstance(parsed, dict):
                # LLM may wrap array in an object
                for key in ("clips", "rankings", "candidates", "results"):
                    if key in parsed and isinstance(parsed[key], list):
                        rankings = parsed[key]
                        break
                else:
                    for v in parsed.values():
                        if isinstance(v, list) and len(v) > 0:
                            rankings = v
                            break
                    else:
                        raise ValueError(f"No array in JSON object. Keys: {list(parsed.keys())}")
            else:
                raise ValueError(f"Unexpected JSON type: {type(parsed)}")
        except json.JSONDecodeError:
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if not json_match:
                raise ValueError(f"No JSON array in response")
            rankings = json.loads(json_match.group())

        print(f"  ✅ Parsed {len(rankings)} rankings from LLM")
    except Exception as e:
        err = f"{type(e).__name__}: {e} | content: {content[:200] if 'content' in dir() else 'N/A'}"
        print(f"  ❌ Parse failed: {err}")
        if RD_MODE:
            raise RuntimeError(f"🛑 LLM PARSE FAILED: {err}")
        return _fallback_ranking(candidates, max_clips, err)

    # Merge LLM rankings with candidate data
    ranked_map = {r["candidate_id"]: r for r in rankings}
    results = []
    for c in candidates:
        r = ranked_map.get(c["id"])
        if r and r.get("keep", False):
            results.append({
                "start": c["start"],
                "end": c["end"],
                "duration": c["duration"],
                "transcript_text": c["transcript_text"],
                "virality_score": r.get("virality_score", 0),
                "title": r.get("title", f"Clip {c['id']}"),
                "description": r.get("description", ""),
                "hashtags": r.get("hashtags", []),
                "hook_strength": r.get("hook_strength", "unknown"),
                "reasoning": r.get("reasoning", ""),
                "energy_score": c["energy_score"],
                "scene_score": c["scene_score"],
                "keyword_score": c["keyword_score"],
            })

    # Sort by virality score
    results.sort(key=lambda x: x["virality_score"], reverse=True)

    # Post-LLM dedup — remove clips that overlap with higher-scored ones
    deduped = []
    for clip in results:
        overlaps = False
        for kept in deduped:
            overlap_start = max(clip["start"], kept["start"])
            overlap_end = min(clip["end"], kept["end"])
            if overlap_end > overlap_start:
                overlap_dur = overlap_end - overlap_start
                min_dur = min(clip["duration"], kept["duration"])
                if min_dur > 0 and (overlap_dur / min_dur) > 0.15:
                    overlaps = True
                    break
        if not overlaps:
            deduped.append(clip)

    final = deduped[:max_clips]
    # Sort final by timestamp for natural order
    final.sort(key=lambda x: x["start"])

    if not final and RD_MODE:
        raise RuntimeError(
            f"🛑 LLM marked ALL candidates as keep=false or all overlap. "
            f"Rankings: {json.dumps(rankings, indent=2)[:1000]}"
        )

    print(f"  🧠 LLM selected {len(final)} clips (from {len(results)} kept, {len(results) - len(deduped)} removed for overlap)")
    return final


def _fallback_ranking(candidates: list, max_clips: int, reason: str = "unknown") -> list:
    """Use composite scores when LLM is unavailable. Only used when RD_MODE=False."""
    print(f"  ⚠️ Fallback ranking — reason: {reason}")
    sorted_c = sorted(candidates, key=lambda x: x["composite_score"], reverse=True)
    results = []
    for c in sorted_c[:max_clips]:
        results.append({
            "start": c["start"],
            "end": c["end"],
            "duration": c["duration"],
            "transcript_text": c["transcript_text"],
            "virality_score": c["composite_score"],
            "title": f"Highlight at {c['start']:.0f}s",
            "description": c["transcript_text"][:100],
            "hashtags": [],
            "hook_strength": "unknown",
            "reasoning": f"Fallback (composite score). LLM error: {reason}",
            "energy_score": c["energy_score"],
            "scene_score": c["scene_score"],
            "keyword_score": c["keyword_score"],
        })
    results.sort(key=lambda x: x["start"])
    return results


# ──────────────────────────────────────
# Scoring helpers
# ──────────────────────────────────────

def _snap_to_sentences(segments: list, start: float, end: float) -> tuple:
    """
    Snap start/end to natural sentence boundaries using punctuation markers.
    Looks within 5s window for sentence-ending punctuation (. ? !).
    Falls back to segment boundaries if no punctuation found.
    """
    best_start = start
    best_end = end

    # Snap START forward to a sentence beginning (after punctuation in previous segment)
    # Look up to 5s forward for a segment that starts after a sentence-ender
    for seg in segments:
        if seg["start"] < start:
            continue
        if seg["start"] > start + 5.0:
            break
        # Check if previous segment ended with sentence punctuation
        seg_idx = segments.index(seg)
        if seg_idx > 0:
            prev_text = segments[seg_idx - 1].get("text", "").strip()
            if prev_text and prev_text[-1] in '.?!"':
                best_start = seg["start"]
                break
        elif seg_idx == 0:
            # First segment — good start point
            best_start = seg["start"]
            break

    # Snap END backward to a sentence ending (segment that ends with punctuation)
    # Look up to 5s backward
    for seg in reversed(segments):
        if seg["end"] > end:
            continue
        if seg["end"] < end - 5.0:
            break
        seg_text = seg.get("text", "").strip()
        if seg_text and seg_text[-1] in '.?!"':
            best_end = seg["end"]
            break

    # Fallback: if no punctuation found, snap to nearest segment boundary within 3s
    if best_start == start:
        for seg in segments:
            if seg["start"] >= start and seg["start"] <= start + 3.0:
                best_start = seg["start"]
                break
    if best_end == end:
        for seg in reversed(segments):
            if seg["end"] <= end and seg["end"] >= end - 3.0:
                best_end = seg["end"]
                break

    return best_start, best_end


def _get_text_in_range(segments: list, start: float, end: float) -> str:
    """Get concatenated transcript text within a time range."""
    texts = []
    for seg in segments:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        texts.append(seg["text"])
    return " ".join(texts)


def _score_energy(audio: dict, start: float, end: float) -> float:
    """
    Score energy peaks per-window-duration (not global).
    Higher density of peaks within the window = higher score.
    """
    peaks_in_window = [p for p in audio["peaks"] if start <= p["time"] <= end]
    duration = end - start
    if duration <= 0:
        return 0.0

    # Peaks per 10 seconds — normalized so 2+ peaks/10s = 1.0
    peak_density = len(peaks_in_window) / (duration / 10.0)
    return min(peak_density / 2.0, 1.0)


def _score_scenes(scenes: list, start: float, end: float) -> float:
    """Score based on scene change density in the window."""
    changes = 0
    for scene in scenes:
        if start < scene["start"] < end:
            changes += 1

    duration_min = (end - start) / 60.0
    if duration_min == 0:
        return 0.0
    density = changes / duration_min
    # Normalize: 4+ changes per minute = 1.0
    return min(density / 4.0, 1.0)


def _score_keywords(text: str) -> float:
    """Score based on viral keyword density."""
    text_lower = text.lower()
    word_count = len(text_lower.split())
    if word_count == 0:
        return 0.0

    matches = sum(1 for kw in VIRAL_KEYWORDS if kw in text_lower)
    # Normalize: 5+ matches in a segment = 1.0
    return min(matches / 5.0, 1.0)


def _deduplicate(windows: list) -> list:
    """Remove windows that overlap >50% with a higher-scored window."""
    kept = []
    for w in windows:
        overlaps = False
        for k in kept:
            overlap_start = max(w["start"], k["start"])
            overlap_end = min(w["end"], k["end"])
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                min_duration = min(w["duration"], k["duration"])
                if min_duration > 0 and (overlap_duration / min_duration) > OVERLAP_THRESHOLD:
                    overlaps = True
                    break
        if not overlaps:
            kept.append(w)
    return kept
