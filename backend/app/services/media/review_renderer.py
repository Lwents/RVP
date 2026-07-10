from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

from app.services.media.ffmpeg import probe_video_duration, run_command, run_command_with_progress


def build_review_starts(source_duration: float, target_seconds: float, clip_count: int) -> list[float]:
    clip_count = max(1, clip_count)
    clip_length = max(1.0, target_seconds / clip_count)
    max_start = max(0.0, source_duration - clip_length - 0.5)
    if clip_count == 1 or max_start <= 0:
        return [0.0]
    return [min(max_start, (max_start * index) / (clip_count - 1)) for index in range(clip_count)]


async def render_movie_review_video(
    ffmpeg: str,
    source_video: Path,
    narration_audio: Path,
    output_file: Path,
    work_dir: Path,
    target_minutes: int,
    beat_count: int,
    on_progress: Callable[[int], None],
) -> Path:
    target_seconds = max(60.0, float(target_minutes) * 60.0)
    source_duration = await probe_video_duration(ffmpeg, source_video)
    clip_count = max(8, min(24, beat_count * 2 if beat_count else math.ceil(target_seconds / 30)))
    clip_seconds = target_seconds / clip_count

    segment_dir = work_dir / "review_segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    for old_file in segment_dir.glob("*"):
        old_file.unlink()

    starts = build_review_starts(source_duration, target_seconds, clip_count)
    segment_files: list[Path] = []
    for index, start in enumerate(starts, start=1):
        percent = 84 + int(((index - 1) / max(len(starts), 1)) * 8)
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
            f"[1:a]apad,atrim=0:{target_seconds:.3f},asetpts=N/SR/TB[a]",
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-t",
            f"{target_seconds:.3f}",
            "-c:v",
            "copy",
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


def _concat_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "'\\''")
