from pathlib import Path

from app.models.job import DubbingRequest
from app.services.subtitles.timing import SubtitleEvent, ass_time, normalize_events, parse_srt


def srt_to_positioned_ass(srt_file: Path, ass_file: Path, width: int, height: int, request: DubbingRequest) -> Path:
    x = round(width * request.subtitle_x_percent / 100)
    y = round(height * request.subtitle_y_percent / 100)
    font_size = max(16, round(request.subtitle_font_size * height / 1080))
    outline = max(2, round(font_size * 0.09))
    shadow = max(1, round(font_size * 0.03))
    events = normalize_events(parse_srt(srt_file))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H99000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},5,20,20,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    max_chars = max(18, round(width / max(font_size, 1) * 1.45))
    lines = [header]
    for event in events:
        safe_text = _ass_escape(_wrap_subtitle_text(event.text, max_chars))
        lines.append(
            f"Dialogue: 0,{ass_time(event.start)},{ass_time(event.end)},Default,,0,0,0,,{{\\pos({x},{y})}}{safe_text}\n"
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
    return " ".join(normalized.split())


def _ass_escape(text: str) -> str:
    return text.replace("{", r"\{").replace("}", r"\}")
