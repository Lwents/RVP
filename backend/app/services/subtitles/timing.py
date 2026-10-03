"""
Subtitle timing utilities: parse SRT, normalize, group events.
"""
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubtitleEvent:
    start: float
    end: float
    text: str


def parse_srt(path: Path) -> list[SubtitleEvent]:
    content = unicodedata.normalize("NFC", path.read_text(encoding="utf-8-sig"))
    blocks = re.split(r"\n\s*\n", content.strip())
    events: list[SubtitleEvent] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        time_index = next((i for i, line in enumerate(lines) if "-->" in line), -1)
        if time_index < 0:
            continue
        start_raw, end_raw = [part.strip() for part in lines[time_index].split("-->", 1)]
        text = " ".join(lines[time_index + 1 :])
        if text:
            events.append(SubtitleEvent(_timestamp_seconds(start_raw), _timestamp_seconds(end_raw), text))
    return events


def normalize_events(events: list[SubtitleEvent]) -> list[SubtitleEvent]:
    ordered = sorted(events, key=lambda item: item.start)
    normalized: list[SubtitleEvent] = []
    for index, event in enumerate(ordered):
        start = event.start
        end = event.end
        if end <= start:
            continue
        if index + 1 < len(ordered):
            next_start = ordered[index + 1].start
            if next_start < end:
                end = max(start + 0.6, next_start - 0.04)
        normalized.append(SubtitleEvent(start, end, event.text))
    return normalized


_INCOMPLETE_SENTENCE_RE = re.compile(
    r"\b(?:and|but|or|so|because|that|which|who|when|where|if|as|"
    r"và|nhưng|hoặc|vì|mà|thì|hay|khi|nếu|để|rằng|với|còn|dù|tuy|"
    r"although|though|unless|until|while|since|after|before|"
    r"however|therefore|moreover|furthermore|meanwhile|nevertheless)\s*$",
    re.IGNORECASE,
)


def _sentence_incomplete(text: str) -> bool:
    stripped = text.strip()
    return stripped.endswith(",") or bool(_INCOMPLETE_SENTENCE_RE.search(stripped))


def group_subtitle_events(
    events: list[SubtitleEvent],
    max_chars: int = 90,
    max_duration: float = 6.0,
    max_gap: float = 0.75,
) -> list[SubtitleEvent]:
    grouped: list[SubtitleEvent] = []
    current: list[SubtitleEvent] = []

    def current_text(extra: SubtitleEvent | None = None) -> str:
        items = current + ([extra] if extra else [])
        return " ".join(item.text.strip() for item in items if item.text.strip())

    def flush() -> None:
        if not current:
            return
        text = current_text()
        if text:
            grouped.append(SubtitleEvent(current[0].start, current[-1].end, text))
        current.clear()

    for event in normalize_events(events):
        if not event.text.strip():
            continue
        if not current:
            current.append(event)
            if _ends_sentence(event.text) and not _sentence_incomplete(event.text):
                flush()
            continue

        gap = event.start - current[-1].end
        duration = event.end - current[0].start
        merged_text = current_text(event)
        hard_limit = len(merged_text) > max_chars or duration > max_duration
        gap_cut = gap > max_gap and _ends_sentence(current_text()) and not _sentence_incomplete(current_text())

        if hard_limit or gap_cut:
            flush()

        current.append(event)
        if _ends_sentence(event.text) and not _sentence_incomplete(event.text):
            if len(current_text()) >= max_chars * 0.5 or gap > max_gap:
                flush()

    flush()
    return grouped


def split_long_subtitle_events(
    events: list[SubtitleEvent],
    max_chars: int = 42,
    max_duration: float = 3.2,
) -> list[SubtitleEvent]:
    split_events: list[SubtitleEvent] = []
    for event in normalize_events(events):
        text = " ".join(event.text.split())
        if not text:
            continue
        duration = max(0.01, event.end - event.start)
        if len(text) <= max_chars and duration <= max_duration:
            split_events.append(SubtitleEvent(event.start, event.end, text))
            continue

        chunks = _split_text_for_subtitles(text, max_chars)
        target_count = max(
            len(chunks),
            int((len(text) + max_chars - 1) // max_chars),
            int((duration + max_duration - 1e-9) // max_duration),
        )
        while len(chunks) < target_count:
            expanded = _split_longest_chunk(chunks, max_chars)
            if expanded == chunks:
                break
            chunks = expanded

        weights = [max(len(chunk.strip()), 1) for chunk in chunks]
        total_weight = sum(weights)
        # Reserve time for every cue before distributing the remaining time.
        # A short tail must not inherit a few milliseconds after a long cue.
        minimum = min(0.6, duration / len(chunks))
        remaining = max(0.0, duration - minimum * len(chunks))
        cursor = event.start
        for index, chunk in enumerate(chunks):
            if index == len(chunks) - 1:
                end = event.end
            else:
                end = min(event.end, cursor + minimum + remaining * (weights[index] / total_weight))
            split_events.append(SubtitleEvent(cursor, end, chunk))
            cursor = end

    return normalize_events(split_events)


def timestamp_seconds(value: str) -> float:
    return _timestamp_seconds(value)


def ass_time(value: float) -> str:
    value = max(0, value)
    hours = int(value // 3600)
    value -= hours * 3600
    minutes = int(value // 60)
    seconds = value - minutes * 60
    return f"{hours}:{minutes:02}:{seconds:05.2f}"


def srt_time(value: float) -> str:
    milliseconds = int(round(value * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def _timestamp_seconds(value: str) -> float:
    value = value.replace(",", ".")
    parts = value.split(":")
    if len(parts) != 3:
        return 0
    try:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        return 0


def _split_text_for_subtitles(text: str, max_chars: int) -> list[str]:
    clauses = _split_by_punctuation(text)
    if not clauses:
        return [text]

    chunks: list[str] = []
    current = ""
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        candidate = f"{current} {clause}".strip() if current else clause
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = clause
        else:
            current = candidate

    if current:
        chunks.append(current)

    flattened: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            flattened.append(chunk)
        else:
            flattened.extend(_split_by_words(chunk, max_chars))
    return flattened


def _split_by_punctuation(text: str) -> list[str]:
    parts = re.split(r"(?<=[,;:.!?…])\s+|(?<=-)\s+", text)
    cleaned = [part.strip() for part in parts if part.strip()]
    return cleaned or [text]


def _split_by_words(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_length + extra > max_chars:
            chunks.append(" ".join(current))
            current = [word]
            current_length = len(word)
        else:
            current.append(word)
            current_length += extra
    if current:
        chunks.append(" ".join(current))
    # Balance neighboring chunks so a final word does not become its own cue.
    for index in range(len(chunks) - 1, 0, -1):
        left, right = chunks[index - 1].split(), chunks[index].split()
        while len(left) > 1:
            candidate = " ".join([left[-1], *right])
            before = abs(len(" ".join(left)) - len(" ".join(right)))
            after = abs(len(" ".join(left[:-1])) - len(candidate))
            if len(candidate) > max_chars or after >= before:
                break
            right.insert(0, left.pop())
        chunks[index - 1], chunks[index] = " ".join(left), " ".join(right)
    return chunks


def _split_longest_chunk(chunks: list[str], max_chars: int) -> list[str]:
    if not chunks:
        return chunks
    index = max(range(len(chunks)), key=lambda idx: len(chunks[idx]))
    longest = chunks[index]
    pieces = _split_by_words(longest, max(12, min(max_chars, len(longest) // 2)))
    if len(pieces) <= 1:
        return chunks
    return chunks[:index] + pieces + chunks[index + 1 :]


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "?", "!", "…", "。", "？", "！"))
