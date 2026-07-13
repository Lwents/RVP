from pathlib import Path
import unicodedata

from app.models.job import DubbingRequest
from app.services.subtitles.timing import SubtitleEvent, ass_time, normalize_events, parse_srt


SUBTITLE_FONT_NAME = "Be Vietnam Pro ExtraBold"
SUBTITLE_FONT_FILE = "BeVietnamPro-ExtraBoldItalic.ttf"
SUBTITLE_FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"


def subtitle_font_dir() -> Path:
    font_file = SUBTITLE_FONT_DIR / SUBTITLE_FONT_FILE
    if not font_file.is_file():
        raise FileNotFoundError(f"Khong tim thay font phu de: {font_file}")
    return SUBTITLE_FONT_DIR


def srt_to_positioned_ass(srt_file: Path, ass_file: Path, width: int, height: int, request: DubbingRequest) -> Path:
    x = round(width * request.subtitle_x_percent / 100)
    y = round(height * request.subtitle_y_percent / 100)
    # ASS font metrics have a smaller cap-height than CSS pixels. The 1.25
    # correction yields a visible glyph height close to 5.5-6.5% of the frame.
    font_size = max(16, round(request.subtitle_font_size * height / 1080 * 1.25))
    outline = max(2, round(6 * height / 1080))
    shadow = max(1, round(2.5 * height / 1080))
    events = normalize_events(parse_srt(srt_file))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{SUBTITLE_FONT_NAME},{font_size},&H0000F2FF,&H0000F2FF,&H00000000,&H96000000,-1,-1,0,0,93,100,0,0,1,{outline},{shadow},2,{round(width * 0.08)},{round(width * 0.08)},{round(height * 0.08)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    # Be Vietnam Pro is condensed to 93% in ASS. This estimate keeps each line
    # within roughly 83% of the frame while allowing at most two balanced lines.
    max_chars = max(18, round((width * 0.83) / max(font_size * 0.54, 1)))
    lines = [header]
    for event in events:
        normalized_text = unicodedata.normalize("NFC", event.text).upper()
        safe_text = _ass_escape(_wrap_subtitle_text(normalized_text, max_chars))
        lines.append(
            f"Dialogue: 0,{ass_time(event.start)},{ass_time(event.end)},Default,,0,0,0,,{{\\an2\\pos({x},{y})}}{safe_text}\n"
        )

    ass_file.write_text("".join(lines), encoding="utf-8")
    return ass_file


def write_srt(events: list[SubtitleEvent], output_file: Path) -> Path:
    from app.services.subtitles.timing import srt_time

    with output_file.open("w", encoding="utf-8") as handle:
        for index, event in enumerate(events, start=1):
            handle.write(f"{index}\n")
            handle.write(f"{srt_time(event.start)} --> {srt_time(event.end)}\n")
            handle.write(f"{event.text.strip()}\n\n")
    return output_file


def _wrap_subtitle_text(text: str, max_chars: int) -> str:
    normalized = text.replace(r"\N", " ")
    normalized = " ".join(normalized.split())
    if len(normalized) <= max_chars:
        return normalized

    words = normalized.split()
    if len(words) < 2:
        return normalized

    best_split = 1
    best_score: tuple[int, int] | None = None
    for index in range(1, len(words)):
        first = " ".join(words[:index])
        second = " ".join(words[index:])
        overflow = max(0, len(first) - max_chars) + max(0, len(second) - max_chars)
        balance = abs(len(first) - len(second))
        score = (overflow, balance)
        if best_score is None or score < best_score:
            best_score = score
            best_split = index

    return f"{' '.join(words[:best_split])}\\N{' '.join(words[best_split:])}"


def _ass_escape(text: str) -> str:
    return text.replace("{", r"\{").replace("}", r"\}")
