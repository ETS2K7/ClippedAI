"""
LLM-powered clip selection.

Stage 1: Algorithmic candidate generation (7 signals)
Stage 2: Groq Llama 4 Scout structured ranking (candidate_index only)
"""

import json
import logging
from typing import Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Stage 1: Algorithmic Candidate Generation
# ─────────────────────────────────────────────

def generate_candidates(
    transcript: dict,
    scenes: list[dict],
    audio_events: list[dict],
    settings: Optional[dict] = None,
) -> list[dict]:
    """
    Generate clip candidates using 7 signals:
      1. Audio energy peaks
      2. Keyword triggers (questions, reactions, emotional words)
      3. Scene density (rapid cuts = high energy)
      4. Sentiment shifts in transcript
      5. Speaker turn density
      6. Laughter / applause events
      7. Silence gaps (natural breakpoints)

    Returns list of candidate clips with scores.
    """
    settings = settings or {}
    min_dur = settings.get("min_duration", config.MIN_CLIP_DURATION)
    max_dur = settings.get("max_duration", config.MAX_CLIP_DURATION)
    ideal_dur = settings.get("ideal_duration", config.IDEAL_CLIP_DURATION)
    max_clips = settings.get("max_clips", config.MAX_CLIPS)

    segments = transcript.get("segments", [])
    if not segments:
        logger.warning("No transcript segments — cannot generate candidates")
        return []

    # Score every possible window
    duration = segments[-1]["end"] if segments else 0
    candidates = []

    # Sliding window at 5-second intervals
    step = 5.0
    for window_start in np.arange(0, max(0, duration - min_dur), step):
        for window_dur in range(ideal_dur[0], ideal_dur[1] + 1, 5):
            window_end = window_start + window_dur
            if window_end > duration:
                continue

            score = _score_window(
                window_start, window_end,
                segments, scenes, audio_events,
            )

            if score > 0.1:  # minimum viability threshold
                candidates.append({
                    "start": round(float(window_start), 3),
                    "end": round(float(window_end), 3),
                    "duration": round(float(window_dur), 1),
                    "algorithmic_score": round(float(score), 4),
                })

    # Sort by score, take top 3× max_clips
    candidates.sort(key=lambda c: c["algorithmic_score"], reverse=True)
    candidates = candidates[:max_clips * 3]

    # Remove overlapping candidates (enforce zero temporal overlap)
    candidates = _remove_overlapping(candidates)

    # Remove semantically similar candidates (enforce content diversity)
    candidates = _deduplicate_by_content(candidates, transcript)

    logger.info("Generated %d candidates from %.0fs video", len(candidates), duration)
    return candidates


def _score_window(
    start: float,
    end: float,
    segments: list[dict],
    scenes: list[dict],
    audio_events: list[dict],
) -> float:
    """Score a candidate window using 7 signals."""

    # 1. Word density (proxy for speech activity)
    words_in_window = []
    for seg in segments:
        for w in seg.get("words", []):
            if start <= w["start"] <= end:
                words_in_window.append(w)
    word_density = len(words_in_window) / max(1, end - start)
    signal_speech = min(1.0, word_density / 3.0)  # normalize: 3 words/sec = max

    # 2. Keyword triggers
    trigger_words = {
        "wow", "amazing", "incredible", "insane", "crazy", "unbelievable",
        "wait", "what", "oh", "no way", "seriously", "actually",
        "important", "secret", "truth", "real", "honest",
        "?",  # questions
    }
    text = " ".join(w["word"].lower() for w in words_in_window)
    trigger_count = sum(1 for tw in trigger_words if tw in text)
    signal_keywords = min(1.0, trigger_count / 5.0)

    # 3. Scene density
    scene_cuts = sum(1 for s in scenes if start <= s.get("boundary", 0) <= end)
    duration = end - start
    signal_scenes = min(1.0, scene_cuts / max(1, duration / 10))

    # 4. Speaker turn density
    speakers = set()
    speaker_changes = 0
    prev_speaker = None
    for seg in segments:
        if start <= seg["start"] <= end:
            spk = seg.get("speaker", "")
            speakers.add(spk)
            if prev_speaker and spk != prev_speaker:
                speaker_changes += 1
            prev_speaker = spk
    signal_turns = min(1.0, speaker_changes / 5.0)

    # 5. Audio events (laughter, applause)
    highlight_events = sum(
        1 for e in audio_events
        if start <= e["start"] <= end and
        e["event_type"] in ("laughter", "applause", "high_energy")
    )
    signal_audio = min(1.0, highlight_events / 3.0)

    # 6. Completeness — prefer starting/ending at sentence boundaries
    starts_at_sentence = any(
        abs(seg["start"] - start) < 1.0 for seg in segments
    )
    ends_at_sentence = any(
        abs(seg["end"] - end) < 1.0 for seg in segments
    )
    signal_completeness = (0.5 if starts_at_sentence else 0) + (0.5 if ends_at_sentence else 0)

    # 7. Ideal duration bonus
    ideal_min, ideal_max = config.IDEAL_CLIP_DURATION
    if ideal_min <= duration <= ideal_max:
        signal_duration = 1.0
    elif duration < ideal_min:
        signal_duration = duration / ideal_min
    else:
        signal_duration = max(0, 1.0 - (duration - ideal_max) / 30)

    # Weighted sum
    score = (
        0.25 * signal_speech +
        0.15 * signal_keywords +
        0.10 * signal_scenes +
        0.10 * signal_turns +
        0.15 * signal_audio +
        0.15 * signal_completeness +
        0.10 * signal_duration
    )

    return score


def _remove_overlapping(candidates: list[dict]) -> list[dict]:
    """Remove overlapping candidates, keeping highest scored."""
    selected = []
    for c in candidates:
        overlaps = False
        for s in selected:
            if c["start"] < s["end"] and c["end"] > s["start"]:
                overlaps = True
                break
        if not overlaps:
            selected.append(c)
    return selected


def _deduplicate_by_content(
    candidates: list[dict],
    transcript: dict,
) -> list[dict]:
    """
    Remove semantically similar candidates using TF-IDF cosine similarity.
    If two clips share > SEMANTIC_SIMILARITY_PENALTY similarity, discard  
    the lower-scoring one. Prevents duplicate-topic clips from podcasts.
    """
    if len(candidates) <= 1:
        return candidates

    segments = transcript.get("segments", [])

    # Extract text for each candidate
    texts = []
    for c in candidates:
        text = _extract_text_for_range(segments, c["start"], c["end"])
        texts.append(text if text.strip() else "empty")

    # Compute TF-IDF similarity
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)

        # Greedy dedup: keep highest scored, discard similar
        keep = [True] * len(candidates)
        for i in range(len(candidates)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(candidates)):
                if not keep[j]:
                    continue
                if sim_matrix[i][j] > config.SEMANTIC_SIMILARITY_PENALTY:
                    # Discard the lower-scored one
                    keep[j] = False
                    logger.info(
                        "Dedup: dropping candidate %.0f-%.0fs (sim=%.2f with %.0f-%.0fs)",
                        candidates[j]["start"], candidates[j]["end"],
                        sim_matrix[i][j],
                        candidates[i]["start"], candidates[i]["end"],
                    )

        return [c for c, k in zip(candidates, keep) if k]

    except ImportError:
        logger.warning("sklearn unavailable, skipping semantic dedup")
        return candidates


# ─────────────────────────────────────────────
# Stage 2: LLM Ranking
# ─────────────────────────────────────────────

def rank_with_llm(
    candidates: list[dict],
    transcript: dict,
    max_clips: int = config.MAX_CLIPS,
) -> list[dict]:
    """
    Rank candidates using Groq Llama 4 Scout.
    LLM returns candidate_index integers only — timestamps from Python.
    """
    if not candidates:
        return []

    # Build candidate summaries for the LLM
    candidate_summaries = []
    segments = transcript.get("segments", [])

    for i, c in enumerate(candidates):
        # Extract text within this candidate's time range
        text = _extract_text_for_range(segments, c["start"], c["end"])
        candidate_summaries.append({
            "index": i,
            "start": c["start"],
            "end": c["end"],
            "duration": c["duration"],
            "text_preview": text[:300],
            "algorithmic_score": c["algorithmic_score"],
        })

    # Call LLM
    ranked_indices = _call_llm(candidate_summaries, max_clips)

    # Map indices back to candidates with LLM ranking
    ranked_clips = []
    for rank, idx in enumerate(ranked_indices):
        if 0 <= idx < len(candidates):
            clip = candidates[idx].copy()
            clip["rank"] = rank + 1
            clip["llm_selected"] = True
            ranked_clips.append(clip)

    logger.info("LLM selected %d clips from %d candidates", len(ranked_clips), len(candidates))
    return ranked_clips


def _call_llm(
    candidate_summaries: list[dict],
    max_clips: int,
) -> list[int]:
    """
    Call Groq (primary) or Together.ai (fallback) to rank candidates.
    Returns list of candidate indices in ranked order.
    """
    prompt = _build_ranking_prompt(candidate_summaries, max_clips)

    for provider in config.LLM_PROVIDER_PRIORITY:
        try:
            if provider == "groq":
                return _call_groq(prompt, max_clips)
            elif provider == "together":
                return _call_together(prompt, max_clips)
        except Exception as e:
            logger.warning("LLM provider %s failed: %s", provider, e)
            continue

    # All providers failed — return top candidates by algorithmic score
    logger.error("All LLM providers failed, using algorithmic ranking")
    return list(range(min(max_clips, len(candidate_summaries))))


def _build_ranking_prompt(candidates: list[dict], max_clips: int) -> str:
    """Build the structured ranking prompt."""
    candidate_text = ""
    for c in candidates:
        candidate_text += (
            f"\n[Candidate {c['index']}] "
            f"({c['duration']:.0f}s, score={c['algorithmic_score']:.2f})\n"
            f"Text: {c['text_preview']}\n"
        )

    return f"""You are a viral short-form content curator. Your job is to select the {max_clips} best clips from a longer video for YouTube Shorts / TikTok / Instagram Reels.

CANDIDATES:
{candidate_text}

SELECTION CRITERIA:
1. Hook potential — would this grab attention in the first 3 seconds?
2. Completeness — is this a complete thought/story/joke?
3. Emotional impact — does this evoke strong reactions?
4. Shareability — would someone share this?
5. Diversity — pick varied content, not the same topic repeated.

RULES:
- Return ONLY a JSON array of candidate index integers, ranked best to worst.
- Return exactly {max_clips} indices (or fewer if not enough good candidates).
- Do NOT modify timestamps. Just return indices.

Example response: [3, 0, 7, 1, 5]

Your selection:"""


def _call_groq(prompt: str, max_clips: int) -> list[int]:
    """Call Groq API with structured output."""
    from groq import Groq

    client = Groq(api_key=config.GROQ_API_KEY)

    response = client.chat.completions.create(
        model=config.LLM_MODEL_RANKING,
        messages=[
            {"role": "system", "content": "You are a viral content curator. Respond only with a JSON array of integers."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=100,
        response_format={"type": "json_object"},
    )

    text = response.choices[0].message.content.strip()
    result = json.loads(text)

    # Handle both {"indices": [...]} and [...] formats
    if isinstance(result, list):
        return [int(x) for x in result[:max_clips]]
    elif isinstance(result, dict):
        indices = result.get("indices", result.get("selection", result.get("ranking", [])))
        return [int(x) for x in indices[:max_clips]]

    return list(range(max_clips))


def _call_together(prompt: str, max_clips: int) -> list[int]:
    """Call Together.ai API."""
    import openai

    client = openai.OpenAI(
        api_key=config.TOGETHER_API_KEY,
        base_url="https://api.together.xyz/v1",
    )

    response = client.chat.completions.create(
        model=config.LLM_MODEL_FALLBACK,
        messages=[
            {"role": "system", "content": "You are a viral content curator. Respond only with a JSON array of integers."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=100,
    )

    text = response.choices[0].message.content.strip()
    # Extract JSON array from response
    import re
    match = re.search(r'\[[\d\s,]+\]', text)
    if match:
        return [int(x) for x in json.loads(match.group())][:max_clips]

    return list(range(max_clips))


def _extract_text_for_range(
    segments: list[dict],
    start: float,
    end: float,
) -> str:
    """Extract transcript text within a time range."""
    words = []
    for seg in segments:
        for w in seg.get("words", []):
            if start <= w["start"] <= end:
                words.append(w["word"])
    return " ".join(words)


def save_selected_clips(clips: list[dict], output_path) -> None:
    """Save selected clips to JSON."""
    from pathlib import Path
    with open(Path(output_path), "w") as f:
        json.dump(clips, f, indent=2)
    logger.info("Saved %d selected clips: %s", len(clips), output_path)
