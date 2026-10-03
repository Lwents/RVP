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
    bundled_project = Path(__file__).resolve().parents[4] / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
    if bundled_project.exists():
        return str(bundled_project)

    if settings.ffmpeg_path:
        configured = Path(settings.ffmpeg_path)
        if configured.exists():
            return str(configured)

    path_match = shutil.which("ffmpeg")
    if path_match:
        return path_match

    # imageio-ffmpeg ships a portable binary on supported platforms. Using it
    # keeps local development working without a machine-wide FFmpeg install.
    try:
        import imageio_ffmpeg

        bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if bundled.exists():
            return str(bundled)
    except (ImportError, RuntimeError):
        pass

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
    def execute() -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    completed = await asyncio.to_thread(execute)
    if completed.returncode != 0:
        detail = completed.stdout.decode(errors="ignore").strip() if completed.stdout else ""
        raise MediaError(f"{error_message} {detail}".strip())


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


def audio_tempo_filter(speed: float) -> str:
    """Build a valid FFmpeg atempo chain for an arbitrary positive speed."""

    remaining = max(0.01, float(speed))
    factors: list[float] = []
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(max(0.5, min(2.0, remaining)))
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


async def adjust_audio_tempo(
    ffmpeg: str,
    source_audio: Path,
    output_audio: Path,
    speed: float,
) -> Path:
    """Change narration cadence without changing pitch."""

    output_audio.parent.mkdir(parents=True, exist_ok=True)
    await run_command(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source_audio),
            "-vn",
            "-af",
            audio_tempo_filter(speed),
            "-c:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(output_audio),
        ],
        "Không điều chỉnh được nhịp giọng đọc review.",
    )
    return output_audio


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
        def probe_with_ffmpeg() -> tuple[int, int]:
            completed = subprocess.run(
                [ffmpeg, "-hide_banner", "-i", str(source_video)],
                capture_output=True,
                check=False,
            )
            output = b"\n".join([completed.stdout, completed.stderr]).decode(errors="ignore")
            video_line = next((line for line in output.splitlines() if "Video:" in line), "")
            match = re.search(r"(?<![\d.])(\d{2,5})x(\d{2,5})(?![\d.])", video_line)
            if not match:
                return (1280, 720)
            return (int(match.group(1)), int(match.group(2)))

        return await asyncio.to_thread(probe_with_ffmpeg)

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
        def probe_with_ffmpeg() -> float:
            completed = subprocess.run(
                [ffmpeg, "-hide_banner", "-i", str(source_video)],
                capture_output=True,
                check=False,
            )
            output = b"\n".join([completed.stdout, completed.stderr]).decode(errors="ignore")
            match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", output)
            if not match:
                return 0
            return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))

        return await asyncio.to_thread(probe_with_ffmpeg)

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

    def execute() -> None:
        process = subprocess.Popen(
            progress_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output_lines: list[str] = []
        last_update = 0.0
        assert process.stdout is not None
        for line_bytes in iter(process.stdout.readline, b""):
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

        return_code = process.wait()
        if return_code != 0:
            detail = "\n".join(output_lines[-40:]).strip()
            raise MediaError(f"{error_message} {detail}".strip())
        on_progress(progress_end)

    await asyncio.to_thread(execute)


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
