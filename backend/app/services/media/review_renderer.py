from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Callable, TypedDict

from app.services.media.ffmpeg import probe_video_duration, run_command, run_command_with_progress
from app.services.subtitles.ass import write_srt
from app.services.subtitles.timing import SubtitleEvent


class ReviewSceneHint(TypedDict, total=False):
    time_hint: str | None
    start_seconds: float | None
    end_seconds: float | None


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

    scenes: list[tuple[float, float]] = []
    for hint in scene_hints or []:
        start = _coerce_seconds(hint.get("start_seconds"))
        end = _coerce_seconds(hint.get("end_seconds"))
        if start is None or end is None:
            parsed = _parse_time_hint(hint.get("time_hint") or "")
            if parsed:
                start, end = parsed
        if start is None or end is None:
            continue
        start = max(0.0, min(start, max(0.0, source_duration - 1.0)))
        end = max(start + 3.0, min(end, source_duration))
        duration = max(3.0, min(45.0, end - start))
        if start + duration > source_duration:
            start = max(0.0, source_duration - duration)
        scenes.append((start, duration))

    if len(scenes) < 3:
        clip_count = max(8, min(24, math.ceil(target_seconds / 30)))
        starts = build_review_starts(source_duration, target_seconds, clip_count)
        clip_seconds = target_seconds / max(len(starts), 1)
        return [(start, clip_seconds) for start in starts]

    return _fit_scenes_to_duration(scenes, target_seconds)


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
            "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30",
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
    await run_command_with_progress(
        [
            ffmpeg,
            "-y",
            "-i",
            str(silent_video),
            "-i",
            str(narration_audio),
            "-filter_complex",
            (
                f"[0:v]subtitles='{_filter_path(subtitle_file)}':"
                "force_style='FontName=Arial,FontSize=28,"
                "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                "BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=42'[v];"
                f"[1:a]apad,atrim=0:{target_seconds:.3f},asetpts=N/SR/TB[a]"
            ),
            "-map",
            "[v]",
            "-map",
            "[a]",
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
        ],
        "Khong render duoc video review dau ra.",
        target_seconds,
        93,
        99,
        on_progress,
    )
    return output_file


def write_review_subtitles(narration_script: str, output_file: Path, target_minutes: int) -> Path:
    target_seconds = max(60.0, float(target_minutes) * 60.0)
    chunks = _split_review_subtitle_chunks(narration_script)
    if not chunks:
        chunks = ["Video review phim."]

    duration_per_chunk = target_seconds / len(chunks)
    events = [
        SubtitleEvent(
            start=index * duration_per_chunk,
            end=min(target_seconds, (index + 1) * duration_per_chunk),
            text=chunk,
        )
        for index, chunk in enumerate(chunks)
    ]
    return write_srt(events, output_file)


def _split_review_subtitle_chunks(text: str, max_chars: int = 52) -> list[str]:
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


def _fit_scenes_to_duration(scenes: list[tuple[float, float]], target_seconds: float) -> list[tuple[float, float]]:
    current_total = sum(duration for _, duration in scenes)
    if current_total <= 0:
        return scenes

    scale = target_seconds / current_total
    fitted = [(start, max(3.0, min(55.0, duration * scale))) for start, duration in scenes]
    fitted_total = sum(duration for _, duration in fitted)
    if fitted_total <= 0:
        return fitted

    correction = target_seconds / fitted_total
    return [(start, max(2.5, duration * correction)) for start, duration in fitted]


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
