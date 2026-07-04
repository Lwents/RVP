import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubtitleEvent:
    start: float
    end: float
    text: str


def parse_srt(path: Path) -> list[SubtitleEvent]:
    content = path.read_text(encoding="utf-8-sig", errors="ignore")
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
                end = max(start + 0.35, next_start - 0.04)
        normalized.append(SubtitleEvent(start, end, event.text))
    return normalized


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
            if _ends_sentence(event.text):
                flush()
            continue

        gap = event.start - current[-1].end
        duration = event.end - current[0].start
        merged_text = current_text(event)
        if gap > max_gap or duration > max_duration or len(merged_text) > max_chars:
            flush()

        current.append(event)
        if _ends_sentence(event.text) or len(current_text()) >= max_chars:
            flush()

    flush()
    return grouped


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


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "?", "!", "。", "？", "！"))
