from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Callable, TypedDict

from app.core import settings
from app.models.job import DubbingRequest
from app.services.media.ffmpeg import probe_video_duration, run_command, run_command_with_progress
from app.services.subtitles.ass import srt_to_positioned_ass, write_srt
from app.services.subtitles.timing import SubtitleEvent


class ReviewSceneHint(TypedDict, total=False):
    time_hint: str | None
    start_seconds: float | None
    end_seconds: float | None
    narration: str | None


def build_review_starts(source_duration: float, target_seconds: float, clip_count: int) -> list[float]:
    clip_count = max(1, clip_count)
    clip_length = max(1.0, target_seconds / clip_count)
    max_start = max(0.0, source_duration - clip_length - 0.5)
    if clip_count == 1 or max_start <= 0:
        return [0.0]
    return [min(max_start, (max_start * index) / (clip_count - 1)) for index in range(clip_count)]


def build_review_scenes(
    source_duration: float,
    target_seconds: float,
    scene_hints: list[ReviewSceneHint] | int | None,
) -> list[tuple[float, float]]:
    if isinstance(scene_hints, int):
        starts = build_review_starts(source_duration, target_seconds, scene_hints)
        clip_seconds = target_seconds / max(len(starts), 1)
        return [(start, clip_seconds) for start in starts]

    hints = scene_hints or []
    if not hints:
        clip_count = max(8, min(24, math.ceil(target_seconds / 30)))
        starts = build_review_starts(source_duration, target_seconds, clip_count)
        clip_seconds = target_seconds / max(len(starts), 1)
        return [(start, clip_seconds) for start in starts]

    durations = _allocate_weighted_durations(
        [_narration_weight(hint.get("narration")) for hint in hints],
        target_seconds,
    )
    fallback_starts = build_review_starts(source_duration, target_seconds, len(hints))

    scenes: list[tuple[float, float]] = []
    for index, hint in enumerate(hints):
        start = _coerce_seconds(hint.get("start_seconds"))
        end = _coerce_seconds(hint.get("end_seconds"))
        if start is None or end is None:
            parsed = _parse_time_hint(hint.get("time_hint") or "")
            if parsed:
                start, end = parsed

        fallback_start = fallback_starts[min(index, len(fallback_starts) - 1)]
        start = fallback_start if start is None else start
        end = start + durations[index] if end is None else end
        start = max(0.0, min(start, max(0.0, source_duration - 1.0)))
        end = max(start + 3.0, min(end, source_duration))
        duration = max(3.0, durations[index])
        if start + duration > source_duration:
            start = max(0.0, source_duration - duration)
        scenes.append((start, duration))
    return scenes


async def render_movie_review_video(
    ffmpeg: str,
    source_video: Path,
    narration_audio: Path,
    subtitle_file: Path,
    output_file: Path,
    work_dir: Path,
    target_minutes: int,
    scene_hints: list[ReviewSceneHint] | int | None,
    on_progress: Callable[[int], None],
    render_request: DubbingRequest | None = None,
) -> Path:
    target_seconds = max(60.0, float(target_minutes) * 60.0)
    source_duration = await probe_video_duration(ffmpeg, source_video)
    scenes = build_review_scenes(source_duration, target_seconds, scene_hints)

    segment_dir = work_dir / "review_segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    for old_file in segment_dir.glob("*"):
        old_file.unlink()

    segment_files: list[Path] = []
    for index, (start, clip_seconds) in enumerate(scenes, start=1):
        percent = 84 + int(((index - 1) / max(len(scenes), 1)) * 8)
        on_progress(percent)
        segment = segment_dir / f"segment_{index:04d}.mp4"
        fade_out_start = max(0.0, clip_seconds - 0.22)
        video_filter = (
            "scale=1280:720:force_original_aspect_ratio=decrease,"
            "pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,"
            f"fade=t=in:st=0:d=0.18,fade=t=out:st={fade_out_start:.3f}:d=0.18"
        )
        command = [
            ffmpeg,
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{clip_seconds:.3f}",
            "-i",
            str(source_video),
            "-an",
            "-vf",
            video_filter,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(segment),
        ]
        await run_command(command, "Khong cat duoc canh review.")
        segment_files.append(segment)

    concat_file = work_dir / "review_concat.txt"
    concat_file.write_text("\n".join(f"file '{_concat_path(path)}'" for path in segment_files), encoding="utf-8")
    silent_video = work_dir / "review_video_silent.mp4"
    await run_command(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(silent_video),
        ],
        "Khong ghep duoc cac canh review.",
    )

    on_progress(93)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    command = _build_review_output_command(
        ffmpeg,
        silent_video,
        narration_audio,
        subtitle_file,
        output_file,
        work_dir,
        target_seconds,
        render_request,
    )
    await run_command_with_progress(
        command,
        "Khong render duoc video review dau ra.",
        target_seconds,
        93,
        99,
        on_progress,
    )
    return output_file


def _build_review_output_command(
    ffmpeg: str,
    silent_video: Path,
    narration_audio: Path,
    subtitle_file: Path,
    output_file: Path,
    work_dir: Path,
    target_seconds: float,
    request: DubbingRequest | None,
) -> list[str]:
    command = [ffmpeg, "-y", "-i", str(silent_video), "-i", str(narration_audio)]
    filter_parts: list[str] = []
    video_label = "[0:v]"
    video_map = "0:v"
    next_input_index = 2
    width, height = 1280, 720

    if request:
        if request.blur_box_enabled:
            blur_y_percent, blur_height_percent = _render_blur_band(
                request.blur_box_y_percent,
                request.blur_box_height_percent,
            )
            blur_h = max(2, round(height * blur_height_percent / 100))
            blur_y = round(height * blur_y_percent / 100)
            blur_y = max(0, min(height - blur_h, blur_y))
            blur_radius = _boxblur_radius(width, blur_h)
            small_w, small_h = _mosaic_size(width, blur_h)
            filter_parts.append(f"{video_label}split=2[vblur_base][vblur_crop]")
            filter_parts.append(
                f"[vblur_crop]crop=iw:{blur_h}:0:{blur_y},"
                f"scale={small_w}:{small_h}:flags=bilinear,"
                f"scale={width}:{blur_h}:flags=neighbor,"
                f"boxblur={blur_radius}:10[blurred]"
            )
            filter_parts.append(f"[vblur_base][blurred]overlay=0:{blur_y}[vblur]")
            video_label = "[vblur]"

        for index, custom_blur in enumerate(request.custom_blur_boxes):
            cb_w = max(2, round(width * custom_blur.width_percent / 100))
            cb_h = max(2, round(height * custom_blur.height_percent / 100))
            cb_x = max(0, min(width - 2, round(width * custom_blur.x_percent / 100)))
            cb_y = max(0, min(height - 2, round(height * custom_blur.y_percent / 100)))
            cb_w = min(cb_w, width - cb_x)
            cb_h = min(cb_h, height - cb_y)
            if cb_w < 2 or cb_h < 2:
                continue

            blur_radius = _boxblur_radius(cb_w, cb_h)
            small_w, small_h = _mosaic_size(cb_w, cb_h)
            filter_parts.append(f"{video_label}split=2[vcb_base_{index}][vcb_crop_{index}]")
            filter_parts.append(
                f"[vcb_crop_{index}]crop={cb_w}:{cb_h}:{cb_x}:{cb_y},"
                f"scale={small_w}:{small_h}:flags=bilinear,"
                f"scale={cb_w}:{cb_h}:flags=neighbor,"
                f"boxblur={blur_radius}:10[cblur_{index}]"
            )
            filter_parts.append(f"[vcb_base_{index}][cblur_{index}]overlay={cb_x}:{cb_y}[vcb_{index}]")
            video_label = f"[vcb_{index}]"

        subtitle_filters: list[str] = []
        if request.cinematic_bars_enabled:
            bar_h = round(height * request.cinematic_bars_height_percent / 100)
            if bar_h > 0:
                subtitle_filters.append(f"drawbox=x=0:y=0:w=iw:h={bar_h}:color=black:t=fill")
                subtitle_filters.append(f"drawbox=x=0:y=ih-{bar_h}:w=iw:h={bar_h}:color=black:t=fill")

        if request.hard_subtitles:
            ass_file = srt_to_positioned_ass(subtitle_file, work_dir / "review_subtitles.positioned.ass", width, height, request)
            if request.subtitle_box_enabled and request.subtitle_box_opacity > 0:
                box_height = max(24, round(height * request.subtitle_box_height_percent / 100))
                box_y = round((height * request.subtitle_y_percent / 100) - (box_height / 2))
                box_y = max(0, min(height - box_height, box_y))
                alpha = round(request.subtitle_box_opacity / 100, 2)
                subtitle_filters.append(f"drawbox=x=0:y={box_y}:w=iw:h={box_height}:color=black@{alpha}:t=fill")
            subtitle_filters.append(f"subtitles='{_filter_path(ass_file)}'")

        if subtitle_filters:
            filter_parts.append(f"{video_label}{','.join(subtitle_filters)}[vsub]")
            video_label = "[vsub]"

        if request.watermark_file_name and request.logo_enabled:
            watermark = Path(settings.storage_dir) / request.watermark_file_name
            if watermark.exists():
                command.extend(["-i", str(watermark)])
                watermark_index = next_input_index
                next_input_index += 1
                filter_parts.append(f"[{watermark_index}:v]scale={request.logo_width}:-1[wm]")
                x_expr = f"(W-w)*{request.logo_x_percent}/100"
                y_expr = f"(H-h)*{request.logo_y_percent}/100"
                filter_parts.append(f"{video_label}[wm]overlay={x_expr}:{y_expr}[vwm]")
                video_label = "[vwm]"
    else:
        filter_parts.append(
            f"[0:v]subtitles='{_filter_path(subtitle_file)}':"
            "force_style='FontName=Arial,FontSize=24,"
            "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            "BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=34'[v]"
        )
        video_label = "[v]"

    if video_label != "[0:v]":
        video_map = video_label

    filter_parts.append(f"[1:a]apad,atrim=0:{target_seconds:.3f},asetpts=N/SR/TB[a]")

    command.extend(["-filter_complex", ";".join(filter_parts), "-map", video_map, "-map", "[a]"])
    command.extend(
        [
            "-t",
            f"{target_seconds:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(output_file),
        ]
    )
    return command


def write_review_subtitles(
    narration_script: str,
    output_file: Path,
    target_minutes: int,
    scene_hints: list[ReviewSceneHint] | None = None,
) -> Path:
    target_seconds = max(60.0, float(target_minutes) * 60.0)
    chunks = _split_review_subtitle_chunks(narration_script)
    if not chunks:
        chunks = ["Video review phim."]

    if scene_hints:
        durations = _allocate_weighted_durations(
            [_narration_weight(hint.get("narration")) for hint in scene_hints],
            target_seconds,
        )
        chunk_counts = _allocate_chunk_counts(len(chunks), durations)
    else:
        durations = [target_seconds]
        chunk_counts = [len(chunks)]

    events: list[SubtitleEvent] = []
    cursor = 0.0
    chunk_index = 0
    for beat_index, beat_duration in enumerate(durations):
        count = chunk_counts[beat_index] if beat_index < len(chunk_counts) else 0
        beat_chunks = chunks[chunk_index : chunk_index + count]
        chunk_index += count
        if not beat_chunks:
            cursor += beat_duration
            continue

        duration_per_chunk = beat_duration / len(beat_chunks)
        for index, chunk in enumerate(beat_chunks):
            start = cursor + index * duration_per_chunk
            end = min(target_seconds, cursor + (index + 1) * duration_per_chunk)
            events.append(SubtitleEvent(start=start, end=end, text=chunk))
        cursor += beat_duration

    if chunk_index < len(chunks):
        tail_start = events[-1].end if events else 0.0
        remaining = chunks[chunk_index:]
        duration_per_chunk = max(0.5, (target_seconds - tail_start) / len(remaining))
        for index, chunk in enumerate(remaining):
            start = min(target_seconds, tail_start + index * duration_per_chunk)
            end = min(target_seconds, tail_start + (index + 1) * duration_per_chunk)
            events.append(SubtitleEvent(start=start, end=end, text=chunk))

    return write_srt(events, output_file)


def _split_review_subtitle_chunks(text: str, max_chars: int = 38) -> list[str]:
    normalized = " ".join(text.replace("\n", " ").split())
    if not normalized:
        return []

    sentences = [part.strip() for part in re.split(r"(?<=[.!?…])\s+", normalized) if part.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = word
            else:
                current = candidate
        if current and current[-1:] in ".!?…" and len(current) >= max_chars * 0.5:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def _concat_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "'\\''")


def _filter_path(path: Path) -> str:
    return path.resolve().as_posix().replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def _boxblur_radius(width: int, height: int) -> int:
    return max(1, min(20, min(width, height) // 4 - 1))


def _mosaic_size(width: int, height: int) -> tuple[int, int]:
    return max(8, width // 80), max(4, height // 80)


def _render_blur_band(y_percent: int, height_percent: int) -> tuple[int, int]:
    if y_percent >= 65:
        y_percent = min(y_percent, 76)
        height_percent = max(height_percent, 100 - y_percent)
    return y_percent, min(40, max(2, height_percent))


def _narration_weight(text: str | None) -> float:
    normalized = " ".join((text or "").split())
    if not normalized:
        return 1.0
    return max(1.0, float(len(normalized)))


def _allocate_weighted_durations(weights: list[float], target_seconds: float) -> list[float]:
    if not weights:
        return [target_seconds]

    total_weight = sum(max(1.0, weight) for weight in weights)
    if total_weight <= 0:
        return [target_seconds / len(weights) for _ in weights]

    min_duration = min(7.0, max(2.5, target_seconds / len(weights) * 0.45))
    durations = [max(min_duration, target_seconds * max(1.0, weight) / total_weight) for weight in weights]
    total_duration = sum(durations)
    if total_duration <= 0:
        return [target_seconds / len(weights) for _ in weights]

    scale = target_seconds / total_duration
    fitted = [max(2.5, duration * scale) for duration in durations]
    drift = target_seconds - sum(fitted)
    fitted[-1] = max(2.5, fitted[-1] + drift)
    return fitted


def _allocate_chunk_counts(total_chunks: int, durations: list[float]) -> list[int]:
    if not durations:
        return [total_chunks]
    if total_chunks <= 0:
        return [0 for _ in durations]

    total_duration = sum(durations)
    if total_duration <= 0:
        counts = [total_chunks // len(durations) for _ in durations]
    else:
        counts = [int(round(total_chunks * duration / total_duration)) for duration in durations]

    if total_chunks >= len(durations):
        counts = [max(1, count) for count in counts]
    else:
        counts = [0 for _ in durations]
        for index in range(total_chunks):
            counts[index] = 1

    diff = total_chunks - sum(counts)
    index = 0
    while diff != 0 and counts:
        target = index % len(counts)
        if diff > 0:
            counts[target] += 1
            diff -= 1
        elif counts[target] > 0:
            counts[target] -= 1
            diff += 1
        index += 1
    return counts


def _coerce_seconds(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


def _parse_time_hint(value: str) -> tuple[float, float] | None:
    matches = re.findall(r"(\d{1,2}:\d{2}(?::\d{2})?(?:[,.]\d+)?)", value)
    if len(matches) < 2:
        return None
    start = _timestamp_to_seconds(matches[0])
    end = _timestamp_to_seconds(matches[1])
    if start is None or end is None or end <= start:
        return None
    return start, end


def _timestamp_to_seconds(value: str) -> float | None:
    parts = value.replace(",", ".").split(":")
    try:
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return None
    return None
