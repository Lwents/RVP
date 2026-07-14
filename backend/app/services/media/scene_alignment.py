from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict

from app.services.subtitles.timing import SubtitleEvent, parse_srt


class ReviewBeatLike(Protocol):
    time_hint: str
    start_seconds: float | None
    end_seconds: float | None
    purpose: str
    narration: str


class AlignedSceneHint(TypedDict):
    time_hint: str
    start_seconds: float
    end_seconds: float
    purpose: str
    narration: str
    alignment_score: float
    alignment_source: str


@dataclass(frozen=True)
class _CandidateWindow:
    start: float
    end: float
    text: str
    vector: dict[str, float]
    score: float = 0.0


_STOPWORDS = {
    "ai",
    "anh",
    "ba",
    "ban",
    "bang",
    "bi",
    "bo",
    "boi",
    "cac",
    "cai",
    "can",
    "cang",
    "cau",
    "cho",
    "chung",
    "co",
    "con",
    "cua",
    "cung",
    "da",
    "dang",
    "day",
    "de",
    "den",
    "di",
    "do",
    "duoc",
    "duoi",
    "gi",
    "hai",
    "hay",
    "hon",
    "khi",
    "khong",
    "la",
    "lai",
    "lam",
    "luc",
    "ma",
    "minh",
    "mot",
    "nay",
    "nen",
    "neu",
    "nhan",
    "nhieu",
    "nhung",
    "noi",
    "o",
    "phai",
    "qua",
    "ra",
    "rang",
    "rat",
    "sau",
    "se",
    "su",
    "tai",
    "tat",
    "the",
    "thi",
    "thay",
    "trong",
    "tro",
    "tu",
    "va",
    "vao",
    "ve",
    "vi",
    "voi",
}


def align_review_beats_to_transcript(
    beats: list[ReviewBeatLike],
    transcript_file: Path,
    source_duration: float,
    output_file: Path | None = None,
) -> list[AlignedSceneHint]:
    """Ground review beats to real transcript timestamps.

    This is a practical local version of text-to-clip retrieval: generate candidate
    subtitle windows, rank them against each review beat, then keep matches in
    movie order so narration and picture move together.
    """

    if not beats:
        return []

    events = [event for event in parse_srt(transcript_file) if event.text.strip()]
    candidates = _build_candidate_windows(events)
    aligned: list[AlignedSceneHint] = []
    last_start = -1.0

    for index, beat in enumerate(beats):
        query = " ".join(part for part in [beat.purpose, beat.narration] if part).strip()
        query_vector = _text_vector(query)
        preferred = _preferred_seconds(beat, index, len(beats), source_duration)
        candidate = _best_candidate(
            candidates,
            query_vector,
            preferred,
            last_start,
            index,
            len(beats),
            source_duration,
        )

        if candidate is None:
            start, end, source, score = _fallback_window(beat, index, len(beats), source_duration)
        else:
            start = candidate.start
            end = candidate.end
            source = "transcript_match"
            score = candidate.score

        if last_start >= 0 and start <= last_start + 6.0:
            start = min(max(0.0, source_duration - 3.0), last_start + 6.0)
            end = max(end, start + 3.0)
        start = max(0.0, min(start, max(0.0, source_duration - 1.0)))
        end = max(start + 3.0, min(end, source_duration))
        last_start = max(last_start, start)
        aligned.append(
            {
                "time_hint": _format_time_hint(start, end),
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "purpose": beat.purpose,
                "narration": beat.narration,
                "alignment_score": round(score, 4),
                "alignment_source": source,
            }
        )

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(aligned, ensure_ascii=False, indent=2), encoding="utf-8")
    return aligned


def _build_candidate_windows(events: list[SubtitleEvent]) -> list[_CandidateWindow]:
    if not events:
        return []

    candidates: list[_CandidateWindow] = []
    max_events_per_window = 8
    min_duration = 5.0
    max_duration = 42.0
    for start_index, start_event in enumerate(events):
        texts: list[str] = []
        end_time = start_event.end
        for end_index in range(start_index, min(len(events), start_index + max_events_per_window)):
            event = events[end_index]
            if event.start - start_event.start > max_duration:
                break
            texts.append(event.text)
            end_time = event.end
            duration = end_time - start_event.start
            if duration < min_duration and end_index + 1 < len(events):
                continue
            text = " ".join(texts)
            vector = _text_vector(text)
            if vector:
                candidates.append(_CandidateWindow(start_event.start, end_time, text, vector))
    return candidates


def _best_candidate(
    candidates: list[_CandidateWindow],
    query_vector: dict[str, float],
    preferred: float,
    last_start: float,
    beat_index: int,
    beat_count: int,
    source_duration: float,
) -> _CandidateWindow | None:
    if not candidates or not query_vector:
        return None

    min_start = 0.0 if last_start < 0 else min(source_duration, last_start + 6.0)
    expected = source_duration * (beat_index / max(beat_count - 1, 1))
    future_cap = source_duration * min(1.0, (beat_index + 4) / max(beat_count, 1)) + 30.0
    best: _CandidateWindow | None = None
    best_score = -1.0
    for candidate in candidates:
        if candidate.start < min_start:
            continue
        text_score = _cosine(query_vector, candidate.vector)
        if text_score <= 0:
            continue
        if candidate.start > future_cap and text_score < 0.55:
            continue
        preferred_score = _time_proximity(candidate.start, preferred, source_duration)
        order_score = _time_proximity(candidate.start, expected, source_duration)
        duration_score = 1.0 if 8.0 <= candidate.end - candidate.start <= 36.0 else 0.75
        score = text_score * 0.56 + preferred_score * 0.14 + order_score * 0.26 + duration_score * 0.04
        if score > best_score:
            best_score = score
            best = _CandidateWindow(candidate.start, candidate.end, candidate.text, candidate.vector, score)

    if best is not None and best.score >= 0.08:
        return best
    return None


def _fallback_window(
    beat: ReviewBeatLike,
    beat_index: int,
    beat_count: int,
    source_duration: float,
) -> tuple[float, float, str, float]:
    start = _coerce_seconds(beat.start_seconds)
    end = _coerce_seconds(beat.end_seconds)
    if start is not None:
        if end is None or end <= start:
            end = start + 18.0
        return start, end, "ai_hint", 0.0

    duration = min(24.0, max(8.0, source_duration / max(beat_count * 3, 1)))
    max_start = max(0.0, source_duration - duration)
    start = min(max_start, max_start * beat_index / max(beat_count - 1, 1))
    return start, start + duration, "chronological_fallback", 0.0


def _preferred_seconds(
    beat: ReviewBeatLike,
    beat_index: int,
    beat_count: int,
    source_duration: float,
) -> float:
    start = _coerce_seconds(beat.start_seconds)
    if start is not None:
        return start
    return source_duration * (beat_index / max(beat_count - 1, 1))


def _text_vector(text: str) -> dict[str, float]:
    tokens = _tokens(text)
    if not tokens:
        return {}

    vector: dict[str, float] = {}
    for token in tokens:
        vector[token] = vector.get(token, 0.0) + 1.0
    for left, right in zip(tokens, tokens[1:]):
        key = f"{left}_{right}"
        vector[key] = vector.get(key, 0.0) + 1.35
    return vector


def _tokens(text: str) -> list[str]:
    normalized = _strip_accents(text.lower())
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return [token for token in tokens if len(token) > 1 and token not in _STOPWORDS]


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(weight * right.get(key, 0.0) for key, weight in left.items())
    if dot <= 0:
        return 0.0
    left_norm = math.sqrt(sum(weight * weight for weight in left.values()))
    right_norm = math.sqrt(sum(weight * weight for weight in right.values()))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _time_proximity(value: float, target: float, source_duration: float) -> float:
    if source_duration <= 0:
        return 0.0
    distance = abs(value - target)
    # Gaussian-like decay with a wide window so text similarity stays dominant.
    sigma = max(90.0, source_duration * 0.18)
    return math.exp(-((distance / sigma) ** 2))


def _coerce_seconds(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


def _format_time_hint(start: float, end: float) -> str:
    return f"{_format_timestamp(start)}-{_format_timestamp(end)}"


def _format_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    whole = int(seconds)
    return f"{whole // 3600:02}:{(whole % 3600) // 60:02}:{whole % 60:02}"
