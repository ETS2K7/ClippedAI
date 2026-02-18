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
    LLM_SYSTEM_PROMPT,
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
    max_duration = settings.get("max_duration", 90)

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
    Sliding window over transcript to find candidate clip regions.
    Scores each window with energy, scene density, and keyword signals.
    """
    segments = transcript["segments"]
    if not segments:
        return []

    # Build flat timeline of words and text
    all_windows = []
    target_count = max_clips * CANDIDATE_MULTIPLIER

    for window_size in WINDOW_SIZES:
        step = window_size * WINDOW_STEP_RATIO

        # Slide over the transcript
        video_duration = segments[-1]["end"]
        t = 0.0
        while t + min_duration <= video_duration:
            window_end = min(t + window_size, video_duration)
            actual_duration = window_end - t

            if actual_duration < min_duration:
                t += step
                continue
            if actual_duration > max_duration:
                window_end = t + max_duration
                actual_duration = max_duration

            # Snap to sentence boundaries
            start_snapped, end_snapped = _snap_to_sentences(segments, t, window_end)
            if end_snapped - start_snapped < min_duration:
                t += step
                continue

            # Gather transcript text in window
            text = _get_text_in_range(segments, start_snapped, end_snapped)
            if len(text.split()) < 10:  # Skip near-empty windows
                t += step
                continue

            # Score the window
            energy_score = _score_energy(audio, start_snapped, end_snapped)
            scene_score = _score_scenes(scenes, start_snapped, end_snapped)
            keyword_score = _score_keywords(text)

            composite = (
                energy_score * ENERGY_WEIGHT +
                scene_score * SCENE_WEIGHT +
                keyword_score * KEYWORD_WEIGHT
            )

            all_windows.append({
                "start": round(start_snapped, 2),
                "end": round(end_snapped, 2),
                "duration": round(end_snapped - start_snapped, 2),
                "transcript_text": text,
                "energy_score": round(energy_score, 4),
                "scene_score": round(scene_score, 4),
                "keyword_score": round(keyword_score, 4),
                "composite_score": round(composite, 4),
            })

            t += step

    # Deduplicate overlapping windows
    all_windows.sort(key=lambda w: w["composite_score"], reverse=True)
    deduped = _deduplicate(all_windows)

    # Return top N
    result = deduped[:target_count]
    # Assign IDs
    for i, c in enumerate(result):
        c["id"] = i

    return result


def rank_with_llm(candidates: list, video_title: str, max_clips: int) -> list:
    """
    Send candidates to Cerebras LLM for viral ranking.
    Falls back to composite_score if LLM fails.
    """
    import httpx

    api_key = os.environ.get("CEREBRAS_API_KEY", "")
    if not api_key:
        print("  ⚠️ No CEREBRAS_API_KEY — falling back to composite scoring")
        return _fallback_ranking(candidates, max_clips)

    # Build user prompt
    candidate_blocks = []
    for c in candidates:
        block = (
            f"---\n"
            f"CANDIDATE {c['id']}\n"
            f"Time: {c['start']:.1f}s – {c['end']:.1f}s ({c['duration']:.0f}s)\n"
            f"Signals: energy={c['energy_score']:.2f}, "
            f"scenes={c['scene_score']:.2f}, keywords={c['keyword_score']:.2f}\n"
            f"Transcript:\n\"{c['transcript_text'][:500]}\"\n"
            f"---"
        )
        candidate_blocks.append(block)

    user_prompt = (
        f'Video title: "{video_title}"\n\n'
        f"Here are {len(candidates)} candidate clips. Rank them by virality potential.\n\n"
        f"{''.join(candidate_blocks)}\n\n"
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
        f'    "reasoning": "<string, why this is/isn\'t engaging>"\n'
        f"  }}\n"
        f"]\n\n"
        f"Return ONLY the JSON array. No other text."
    )

    # Call Cerebras API
    print(f"  🔗 Calling Cerebras API (model={CEREBRAS_MODEL})...")
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
                "response_format": {"type": "json_object"},
            },
            timeout=90.0,
        )
        if response.status_code != 200:
            print(f"  ❌ Cerebras HTTP {response.status_code}: {response.text[:500]}")
            return _fallback_ranking(candidates, max_clips)
    except Exception as e:
        print(f"  ❌ Cerebras API error: {type(e).__name__}: {e}")
        return _fallback_ranking(candidates, max_clips)

    # Parse response
    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        print(f"  📝 LLM response received ({len(content)} chars)")

        # Extract JSON from response (may be wrapped in markdown or object)
        # Try parsing as direct JSON first
        try:
            parsed = json.loads(content)
            # If it's a dict with a "clips" or "rankings" key, extract the array
            if isinstance(parsed, dict):
                for key in ("clips", "rankings", "candidates", "results"):
                    if key in parsed and isinstance(parsed[key], list):
                        rankings = parsed[key]
                        break
                else:
                    raise ValueError("JSON object has no recognizable array key")
            elif isinstance(parsed, list):
                rankings = parsed
            else:
                raise ValueError(f"Unexpected JSON type: {type(parsed)}")
        except json.JSONDecodeError:
            # Fallback: extract JSON array from response text
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if not json_match:
                raise ValueError(f"No JSON array found in response: {content[:200]}")
            rankings = json.loads(json_match.group())

        print(f"  ✅ Parsed {len(rankings)} rankings from LLM")
    except Exception as e:
        print(f"  ❌ Failed to parse LLM response: {type(e).__name__}: {e}")
        print(f"     Raw content: {content[:300] if 'content' in dir() else 'N/A'}")
        return _fallback_ranking(candidates, max_clips)

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
                "reasoning": r.get("reasoning", ""),
                "energy_score": c["energy_score"],
                "scene_score": c["scene_score"],
                "keyword_score": c["keyword_score"],
            })

    # Sort by virality, take top max_clips
    results.sort(key=lambda x: x["virality_score"], reverse=True)
    final = results[:max_clips]
    # Sort final by timestamp for natural order
    final.sort(key=lambda x: x["start"])

    print(f"  🧠 LLM selected {len(final)} clips (from {len(results)} kept)")
    return final


def _fallback_ranking(candidates: list, max_clips: int) -> list:
    """Use composite scores when LLM is unavailable."""
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
            "reasoning": "Ranked by composite score (LLM unavailable)",
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
    """Snap start/end to nearest segment boundaries."""
    # Find closest segment start >= our start
    best_start = start
    best_end = end

    for seg in segments:
        if seg["start"] >= start - 1.0 and seg["start"] <= start + 3.0:
            best_start = seg["start"]
            break

    for seg in reversed(segments):
        if seg["end"] <= end + 1.0 and seg["end"] >= end - 3.0:
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
    """Count energy peaks within the window, normalized."""
    peaks_in_window = [p for p in audio["peaks"] if start <= p["time"] <= end]
    total_peaks = len(audio["peaks"])
    if total_peaks == 0:
        return 0.0
    return min(len(peaks_in_window) / max(total_peaks * 0.1, 1), 1.0)


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
