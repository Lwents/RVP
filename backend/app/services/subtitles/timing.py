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


class SubtitleTimestampError(ValueError):
    """Raised when an SRT timecode cannot be parsed at all."""


# Khoảng hở tối thiểu giữa hai dòng phụ đề liền nhau.
_MIN_SUBTITLE_GAP = 0.04
# Dòng ngắn hơn mức này coi như rác, bỏ đi thay vì tạo chồng lấn.
_MIN_SUBTITLE_DURATION = 0.08


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
        if not text:
            continue
        try:
            start = timestamp_seconds(start_raw)
            end = timestamp_seconds(end_raw)
        except SubtitleTimestampError as exc:
            # Không im lặng quy về 0.0: dòng sai timecode sẽ nhảy về đầu video.
            print(f"Cảnh báo phụ đề: bỏ qua block sai timecode trong {path.name}. {exc}")
            continue
        events.append(SubtitleEvent(start, end, text))
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
                # Không được kéo dài quá dòng kế tiếp: đó chính là chồng lấn mà
                # hàm này sinh ra để loại bỏ. Dòng quá sát nhau thì bỏ hẳn.
                end = min(end, next_start - _MIN_SUBTITLE_GAP)
        if end - start < _MIN_SUBTITLE_DURATION:
            continue
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
        cursor = event.start
        for index, chunk in enumerate(chunks):
            if index == len(chunks) - 1:
                end = event.end
            else:
                end = min(event.end, cursor + max(0.35, duration * (weights[index] / total_weight)))
            split_events.append(SubtitleEvent(cursor, end, chunk))
            cursor = end

    return normalize_events(split_events)


def timestamp_seconds(value: str) -> float:
    """Parse ``HH:MM:SS,mmm``, ``MM:SS,mmm`` or ``SS.mmm`` into seconds.

    Raises :class:`SubtitleTimestampError` (a ``ValueError``) on genuinely
    malformed input instead of silently returning ``0.0`` and dropping the line
    to the start of the video.
    """

    text = value.strip().replace(",", ".")
    if not text:
        raise SubtitleTimestampError("Timecode phụ đề rỗng.")

    parts = text.split(":")
    if len(parts) > 3:
        raise SubtitleTimestampError(f"Timecode phụ đề không hợp lệ: {value!r}")

    try:
        seconds = float(parts[-1])
        minutes = int(parts[-2]) if len(parts) >= 2 else 0
        hours = int(parts[-3]) if len(parts) == 3 else 0
    except ValueError as exc:
        raise SubtitleTimestampError(f"Timecode phụ đề không hợp lệ: {value!r}") from exc

    if seconds < 0 or minutes < 0 or hours < 0:
        raise SubtitleTimestampError(f"Timecode phụ đề âm: {value!r}")
    return hours * 3600 + minutes * 60 + seconds


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
