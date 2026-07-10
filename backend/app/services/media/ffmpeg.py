import asyncio
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

from app.core import settings


class MediaError(RuntimeError):
    pass


ProgressCallback = Callable[[int], None]


def find_ffmpeg() -> str | None:
    if settings.ffmpeg_path:
        configured = Path(settings.ffmpeg_path)
        if configured.exists():
            return str(configured)

    path_match = shutil.which("ffmpeg")
    if path_match:
        return path_match

    for root in [
        Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages",
        Path("C:/Program Files"),
        Path("C:/Program Files (x86)"),
    ]:
        if not root.exists():
            continue
        for match in root.rglob("ffmpeg.exe"):
            return str(match)
    return None


def find_ffprobe(ffmpeg: str) -> str | None:
    ffprobe = Path(ffmpeg).with_name("ffprobe.exe")
    if ffprobe.exists():
        return str(ffprobe)
    return shutil.which("ffprobe")


async def run_command(command: list[str], error_message: str) -> None:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await process.communicate()
        if process.returncode != 0:
            detail = stdout.decode(errors="ignore").strip() if stdout else ""
            raise MediaError(f"{error_message} {detail}")
    except asyncio.CancelledError:
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=3.0)
        except (asyncio.TimeoutError, ProcessLookupError):
            try:
                process.kill()
            except Exception:
                pass
        raise


async def run_command_with_progress(
    command: list[str],
    error_message: str,
    duration: float,
    progress_start: int,
    progress_end: int,
    on_progress: ProgressCallback,
) -> None:
    await run_ffmpeg_with_progress(command, error_message, duration, progress_start, progress_end, on_progress)


async def extract_audio(ffmpeg: str, source_video: Path, audio_file: Path) -> None:
    await run_command(
        [ffmpeg, "-y", "-i", str(source_video), "-vn", "-ac", "1", "-ar", "16000", str(audio_file)],
        "Không tách được audio từ video.",
    )


async def extract_demucs_audio(ffmpeg: str, source_video: Path, audio_file: Path) -> None:
    await run_command(
        [ffmpeg, "-y", "-i", str(source_video), "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(audio_file)],
        "Không tách được audio để xử lý Demucs.",
    )


async def copy_mp4(ffmpeg: str, source_video: Path, output_file: Path) -> None:
    await run_command(
        [ffmpeg, "-y", "-i", str(source_video), "-c", "copy", "-movflags", "+faststart", str(output_file)],
        "Không xuất được file MP4 đầu ra.",
    )


async def probe_video_size(ffmpeg: str, source_video: Path) -> tuple[int, int]:
    ffprobe = find_ffprobe(ffmpeg)
    if not ffprobe:
        return (1280, 720)

    def probe() -> tuple[int, int]:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=s=x:p=0",
                str(source_video),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return (1280, 720)
        match = re.search(r"(\d+)x(\d+)", completed.stdout)
        if not match:
            return (1280, 720)
        return (int(match.group(1)), int(match.group(2)))

    return await asyncio.to_thread(probe)


async def probe_video_duration(ffmpeg: str, source_video: Path) -> float:
    ffprobe = find_ffprobe(ffmpeg)
    if not ffprobe:
        return 0

    def probe() -> float:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(source_video),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return 0
        try:
            return float(completed.stdout.strip())
        except ValueError:
            return 0

    return await asyncio.to_thread(probe)


async def run_ffmpeg_with_progress(
    command: list[str],
    error_message: str,
    duration: float,
    progress_start: int,
    progress_end: int,
    on_progress: ProgressCallback,
) -> None:
    progress_command = command[:1] + ["-nostats", "-progress", "pipe:1"] + command[1:]

    process = await asyncio.create_subprocess_exec(
        *progress_command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    
    output_lines: list[str] = []
    last_update = 0.0

    try:
        assert process.stdout is not None
        while True:
            line_bytes = await process.stdout.readline()
            if not line_bytes:
                break
            line = line_bytes.decode(errors="ignore").strip()
            if line:
                output_lines.append(line)
            if not line.startswith("out_time_") or duration <= 0:
                continue
            seconds = _ffmpeg_progress_seconds(line)
            if seconds is None:
                continue
            now = time.monotonic()
            if now - last_update < 0.7:
                continue
            percent = progress_start + int(min(seconds / duration, 1.0) * (progress_end - progress_start))
            on_progress(max(progress_start, min(progress_end, percent)))
            last_update = now

        return_code = await process.wait()
        if return_code != 0:
            detail = "\n".join(output_lines[-40:]).strip()
            raise MediaError(f"{error_message} {detail}")
        on_progress(progress_end)
    except asyncio.CancelledError:
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=3.0)
        except (asyncio.TimeoutError, ProcessLookupError):
            try:
                process.kill()
            except Exception:
                pass
        raise


def _ffmpeg_progress_seconds(line: str) -> float | None:
    key, _, raw = line.partition("=")
    if not raw:
        return None
    try:
        if key in {"out_time_ms", "out_time_us"}:
            return int(raw) / 1_000_000
        if key == "out_time":
            return _timestamp_seconds(raw)
    except ValueError:
        return None
    return None


def _timestamp_seconds(value: str) -> float:
    value = value.replace(",", ".")
    parts = value.split(":")
    if len(parts) != 3:
        return 0
    try:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        return 0
