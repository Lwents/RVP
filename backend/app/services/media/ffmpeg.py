import asyncio
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

from app.core import settings
from app.services.subtitles.timing import timestamp_seconds


class MediaError(RuntimeError):
    pass


ProgressCallback = Callable[[int], None]

# Một FFmpeg bị treo (ổ đĩa lỗi, filter kẹt, GPU không phản hồi) sẽ giữ job chạy
# vô hạn nếu không có timeout. Cấu hình bằng biến môi trường khi cần render dài.
DEFAULT_COMMAND_TIMEOUT_SECONDS = 7200.0
_PROCESS_KILL_GRACE_SECONDS = 5.0


def _default_timeout_seconds() -> float:
    raw = os.environ.get("AUTO_TRANSLATE_FFMPEG_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_COMMAND_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_COMMAND_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_COMMAND_TIMEOUT_SECONDS


def _resolve_timeout(timeout: float | None) -> float:
    if timeout is None:
        return _default_timeout_seconds()
    return timeout if timeout > 0 else _default_timeout_seconds()


def _kill_process(process: subprocess.Popen) -> None:
    """Terminate, then hard-kill, so no orphan FFmpeg survives the job."""

    if process.poll() is not None:
        return
    try:
        process.terminate()
    except OSError:
        return
    try:
        process.wait(timeout=_PROCESS_KILL_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        process.kill()
        process.wait(timeout=_PROCESS_KILL_GRACE_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _timeout_message(error_message: str, timeout_seconds: float) -> str:
    return (
        f"{error_message} FFmpeg chạy quá {int(timeout_seconds)} giây nên đã bị dừng. "
        "Kiểm tra file nguồn hoặc tăng AUTO_TRANSLATE_FFMPEG_TIMEOUT_SECONDS."
    ).strip()


def _binary_name(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def find_ffmpeg() -> str | None:
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

    if os.name != "nt":
        return None

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
    # FFprobe thường nằm cạnh FFmpeg (kể cả bản portable), sau đó mới tới PATH.
    sibling = Path(ffmpeg).with_name(_binary_name("ffprobe"))
    if sibling.exists():
        return str(sibling)

    path_match = shutil.which("ffprobe")
    if path_match:
        return path_match

    if os.name == "nt":
        return None

    windows_sibling = Path(ffmpeg).with_name("ffprobe.exe")
    if windows_sibling.exists():
        return str(windows_sibling)
    return None


async def run_command(command: list[str], error_message: str, timeout: float | None = None) -> None:
    timeout_seconds = _resolve_timeout(timeout)

    def execute() -> None:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            stdout, _ = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _kill_process(process)
            try:
                process.communicate(timeout=_PROCESS_KILL_GRACE_SECONDS)
            except (subprocess.TimeoutExpired, ValueError):
                pass
            raise MediaError(_timeout_message(error_message, timeout_seconds)) from None
        except BaseException:
            _kill_process(process)
            raise
        if process.returncode != 0:
            detail = stdout.decode(errors="ignore").strip() if stdout else ""
            raise MediaError(f"{error_message} {detail}".strip())

    await asyncio.to_thread(execute)


async def run_command_with_progress(
    command: list[str],
    error_message: str,
    duration: float,
    progress_start: int,
    progress_end: int,
    on_progress: ProgressCallback,
    timeout: float | None = None,
) -> None:
    await run_ffmpeg_with_progress(
        command,
        error_message,
        duration,
        progress_start,
        progress_end,
        on_progress,
        timeout=timeout,
    )


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


def _probe_size_error(source_video: Path, detail: str = "") -> MediaError:
    # Không được đoán (1280, 720): toạ độ khung blur tính theo kích thước bịa ra
    # sẽ rơi sai vị trí hoàn toàn trên video dọc.
    return MediaError(
        f"Không đọc được độ phân giải của video {source_video.name}. {detail}".strip()
    )


async def probe_video_size(ffmpeg: str, source_video: Path) -> tuple[int, int]:
    ffprobe = find_ffprobe(ffmpeg)
    if not ffprobe:
        def probe_with_ffmpeg() -> tuple[int, int]:
            completed = subprocess.run(
                [ffmpeg, "-hide_banner", "-i", str(source_video)],
                capture_output=True,
                check=False,
                timeout=_default_timeout_seconds(),
            )
            output = b"\n".join([completed.stdout, completed.stderr]).decode(errors="ignore")
            video_line = next((line for line in output.splitlines() if "Video:" in line), "")
            match = re.search(r"(?<![\d.])(\d{2,5})x(\d{2,5})(?![\d.])", video_line)
            if not match:
                raise _probe_size_error(source_video, output.strip()[-500:])
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
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=_default_timeout_seconds(),
        )
        if completed.returncode != 0:
            raise _probe_size_error(source_video, (completed.stderr or "").strip()[-500:])
        match = re.search(r"(\d+)x(\d+)", completed.stdout)
        if not match:
            raise _probe_size_error(source_video, (completed.stdout or "").strip()[-500:])
        return (int(match.group(1)), int(match.group(2)))

    return await asyncio.to_thread(probe)


async def probe_has_audio_stream(ffmpeg: str, source_video: Path) -> bool:
    """Report whether the file really carries an audio stream.

    Mapping ``[0:a]`` on a silent source aborts the whole render, so callers
    need to know before they build the audio graph.
    """

    ffprobe = find_ffprobe(ffmpeg)

    def probe() -> bool:
        if ffprobe:
            completed = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "a",
                    "-show_entries",
                    "stream=index",
                    "-of",
                    "csv=p=0",
                    str(source_video),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=_default_timeout_seconds(),
            )
            if completed.returncode == 0:
                return bool((completed.stdout or "").strip())

        completed_ffmpeg = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(source_video)],
            capture_output=True,
            check=False,
            timeout=_default_timeout_seconds(),
        )
        output = b"\n".join([completed_ffmpeg.stdout, completed_ffmpeg.stderr]).decode(errors="ignore")
        return any("Audio:" in line for line in output.splitlines())

    try:
        return await asyncio.to_thread(probe)
    except (OSError, subprocess.SubprocessError):
        # Khi không xác định được thì coi như có audio: giữ nguyên hành vi cũ.
        return True


async def probe_video_duration(ffmpeg: str, source_video: Path) -> float:
    """Return the media duration, or ``0.0`` meaning "không xác định được".

    ``0.0`` is the explicit unknown marker every caller already branches on
    (timeout ASR, target review, tiến độ render). Không bao giờ trả về một con
    số bịa ra khi probe thất bại.
    """

    ffprobe = find_ffprobe(ffmpeg)
    if not ffprobe:
        def probe_with_ffmpeg() -> float:
            completed = subprocess.run(
                [ffmpeg, "-hide_banner", "-i", str(source_video)],
                capture_output=True,
                check=False,
                timeout=_default_timeout_seconds(),
            )
            output = b"\n".join([completed.stdout, completed.stderr]).decode(errors="ignore")
            match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", output)
            if not match:
                print(f"Probe warning: không đọc được thời lượng của {source_video.name}.")
                return 0.0
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
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=_default_timeout_seconds(),
        )
        if completed.returncode != 0:
            print(
                f"Probe warning: ffprobe lỗi khi đọc thời lượng {source_video.name}. "
                f"{(completed.stderr or '').strip()[-300:]}"
            )
            return 0.0
        try:
            return float(completed.stdout.strip())
        except ValueError:
            print(f"Probe warning: không đọc được thời lượng của {source_video.name}.")
            return 0.0

    return await asyncio.to_thread(probe)


async def run_ffmpeg_with_progress(
    command: list[str],
    error_message: str,
    duration: float,
    progress_start: int,
    progress_end: int,
    on_progress: ProgressCallback,
    timeout: float | None = None,
) -> None:
    progress_command = command[:1] + ["-nostats", "-progress", "pipe:1"] + command[1:]
    timeout_seconds = _resolve_timeout(timeout)

    def execute() -> None:
        process = subprocess.Popen(
            progress_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # readline() có thể chờ mãi nếu FFmpeg treo và không in gì nữa, nên hạn
        # chót được canh bằng watchdog: giết tiến trình để vòng đọc gặp EOF.
        timed_out = threading.Event()

        def on_timeout() -> None:
            timed_out.set()
            _kill_process(process)

        watchdog = threading.Timer(timeout_seconds, on_timeout)
        watchdog.daemon = True
        watchdog.start()

        output_lines: list[str] = []
        last_update = 0.0
        assert process.stdout is not None
        try:
            with process.stdout:
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

            return_code = process.wait(timeout=_PROCESS_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            _kill_process(process)
            return_code = process.poll() or -1
        except BaseException:
            _kill_process(process)
            raise
        finally:
            watchdog.cancel()
            _kill_process(process)

        if timed_out.is_set():
            raise MediaError(_timeout_message(error_message, timeout_seconds))
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
            # Dùng chung parser với module phụ đề để không lệch cách xử lý.
            return timestamp_seconds(raw)
    except ValueError:
        return None
    return None
