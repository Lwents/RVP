import asyncio
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from app.core import settings
from app.models.job import ProcessingMode
from app.services.ai.context import analyze_video_context
from app.services.ai.asr import get_asr_engine
from app.services.ai.translation import get_translation_engine
from app.services.media.ffmpeg import find_ffmpeg
from app.services.subtitles.ass import write_srt
from app.services.subtitles.timing import group_subtitle_events, parse_srt, split_long_subtitle_events


class SubtitleSourceError(RuntimeError):
    pass


def translation_progress_percent(
    completed: int,
    total: int,
    start_percent: int,
    end_percent: int,
) -> int:
    ratio = completed / max(total, 1)
    return start_percent + int(
        max(0.0, min(1.0, ratio)) * max(0, end_percent - start_percent)
    )


def _platform_name(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if "douyin.com" in host:
        return "Douyin"
    if "tiktok.com" in host:
        return "TikTok"
    if "youtube.com" in host or "youtu.be" in host:
        return "YouTube"
    return "nền tảng"


def _find_js_runtime() -> str | None:
    if settings.ytdlp_js_runtime:
        return settings.ytdlp_js_runtime

    node = shutil.which("node")
    if node:
        return f"node:{Path(node).as_posix()}"

    deno = shutil.which("deno")
    if deno:
        return f"deno:{Path(deno).as_posix()}"

    return None


def _without_browser_cookies(command: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for item in command:
        if skip_next:
            skip_next = False
            continue
        if item == "--cookies-from-browser":
            skip_next = True
            continue
        cleaned.append(item)
    return cleaned


def _run_ytdlp(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout,
    )
    detail = completed.stderr.strip() or completed.stdout.strip()
    if completed.returncode != 0 and "--cookies-from-browser" in command and "could not copy chrome cookie database" in detail.lower():
        return subprocess.run(
            _without_browser_cookies(command),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout,
        )
    return completed


async def _cancel_and_collect(running_task: asyncio.Task) -> None:
    """Cancel the ASR task and actually wait for it to stop.

    Chỉ gọi ``cancel()`` là chưa đủ: task chưa được await thì lỗi của nó không
    bao giờ được lấy ra và Whisper vẫn chạy tiếp ở background.
    """

    running_task.cancel()
    try:
        await asyncio.wait({running_task})
    except asyncio.CancelledError:
        pass
    if running_task.done() and not running_task.cancelled():
        error = running_task.exception()
        if error:
            print(f"ASR task dừng kèm lỗi: {error}")


async def _run_with_heartbeat(
    task,
    progress: Callable[[str, int], None],
    stage: str,
    start_percent: int,
    end_percent: int,
    timeout_seconds: int,
):
    running_task = asyncio.create_task(task)
    elapsed = 0
    try:
        while not running_task.done():
            percent = min(end_percent, start_percent + elapsed)
            progress(stage, percent)
            await asyncio.sleep(5)
            elapsed += 1
            if elapsed * 5 >= timeout_seconds:
                await _cancel_and_collect(running_task)
                raise SubtitleSourceError(
                    f"{stage} quá {timeout_seconds} giây. Kiểm tra model Whisper hoặc giảm độ dài video."
                )
        return await running_task
    except asyncio.CancelledError:
        await _cancel_and_collect(running_task)
        raise


async def get_or_create_subtitles(
    source_url: str | None,
    audio_file: Path,
    work_dir: Path,
    source_language: str,
    progress: Callable[[str, int], None],
    source_video: Path | None = None,
    asr_timeout_seconds: int | None = None,
    processing_mode: ProcessingMode | str | None = None,
    translation_progress_start: int = 68,
    translation_progress_end: int = 69,
) -> Path:
    target_file = work_dir / f"subtitles.{settings.target_language}.srt"
    if target_file.exists() and target_file.stat().st_size > 0:
        progress("Dùng lại phụ đề đã dịch xong", translation_progress_end)
        return target_file

    if source_url and settings.prefer_youtube_subtitles:
        try:
            subtitle_file = await download_platform_subtitles(source_url, work_dir, progress)
            return await _prepare_target_subtitles(
                subtitle_file,
                work_dir,
                source_language,
                progress,
                source_video,
                processing_mode,
                translation_progress_start,
                translation_progress_end,
            )
        except SubtitleSourceError:
            progress("Nền tảng không có phụ đề phù hợp, tự tạo phụ đề bằng Whisper", 58)

    whisper_file = work_dir / "subtitles.whisper.srt"
    if whisper_file.exists() and whisper_file.stat().st_size > 0:
        progress("Dùng lại phụ đề Whisper đã nhận diện xong", 66)
        return await _prepare_target_subtitles(
            whisper_file,
            work_dir,
            source_language,
            progress,
            source_video,
            processing_mode,
            translation_progress_start,
            translation_progress_end,
        )

    asr_task = get_asr_engine().transcribe_to_srt(
        audio_file,
        whisper_file,
        source_language,
        processing_mode,
    )
    subtitle_file = await _run_with_heartbeat(
        asr_task,
        progress,
        "Đang nghe audio và tạo phụ đề tự động",
        62,
        70,
        asr_timeout_seconds or settings.asr_timeout_seconds,
    )
    return await _prepare_target_subtitles(
        subtitle_file,
        work_dir,
        source_language,
        progress,
        source_video,
        processing_mode,
        translation_progress_start,
        translation_progress_end,
    )


def asr_timeout_for_duration(video_duration_seconds: float) -> int:
    if video_duration_seconds <= 0:
        return max(settings.asr_timeout_seconds, 1800)
    duration_based_timeout = int(video_duration_seconds * 2.5)
    return min(max(settings.asr_timeout_seconds, duration_based_timeout, 1800), 6 * 60 * 60)


async def _prepare_target_subtitles(
    subtitle_file: Path,
    work_dir: Path,
    source_language: str,
    progress: Callable[[str, int], None],
    source_video: Path | None = None,
    processing_mode: ProcessingMode | str | None = None,
    translation_progress_start: int = 68,
    translation_progress_end: int = 69,
) -> Path:
    grouped_max_chars = max(settings.subtitle_group_max_chars, 42)
    grouped_max_duration = max(settings.subtitle_group_max_duration, 3.2)
    grouped_max_gap = max(settings.subtitle_group_max_gap, 0.45)

    events = group_subtitle_events(
        parse_srt(subtitle_file),
        max_chars=grouped_max_chars,
        max_duration=grouped_max_duration,
        max_gap=grouped_max_gap,
    )
    events = split_long_subtitle_events(
        events,
        max_chars=grouped_max_chars,
        max_duration=grouped_max_duration,
    )
    grouped_file = work_dir / "subtitles.grouped.srt"
    write_srt(events, grouped_file)

    target_language = settings.target_language
    if settings.translation_engine.lower() == "passthrough" or (
        source_language != "auto" and source_language.lower() == target_language.lower()
    ):
        return grouped_file

    progress("AI nhận diện tên phim, nhân vật và vai vế", max(0, translation_progress_start - 1))
    context = await analyze_video_context(source_video, events, work_dir)

    progress("Dịch phụ đề sang tiếng Việt", translation_progress_start)
    def translation_progress(completed: int, total: int) -> None:
        percent = translation_progress_percent(
            completed,
            total,
            translation_progress_start,
            translation_progress_end,
        )
        progress(
            f"Dịch phụ đề sang tiếng Việt ({completed}/{max(total, 1)} batch, đã lưu checkpoint)",
            percent,
        )

    translated_events = await get_translation_engine().translate_events(
        events,
        source_language,
        target_language,
        context=context,
        processing_mode=processing_mode,
        checkpoint_file=work_dir / "translation_checkpoint.json",
        on_batch_progress=translation_progress,
    )
    translated_events = split_long_subtitle_events(
        translated_events,
        max_chars=grouped_max_chars,
        max_duration=grouped_max_duration,
    )
    return write_srt(translated_events, work_dir / f"subtitles.{target_language}.srt")


async def download_platform_subtitles(url: str, work_dir: Path, progress: Callable[[str, int], None]) -> Path:
    progress(f"Tải phụ đề có sẵn từ {_platform_name(url)}", 55)
    output_template = str(work_dir / "subtitles.%(ext)s")
    ffmpeg = find_ffmpeg()
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--force-ipv4",
        "--socket-timeout",
        "25",
        "--retries",
        "1",
        "--skip-download",
        "--remote-components",
        "ejs:github",
        "--write-sub",
        "--write-auto-sub",
        "--sub-langs",
        "vi,vi-orig,en,en-orig,zh,zh-Hans,zh-Hant",
        "--sub-format",
        "srt",
        "--convert-subs",
        "srt",
        "--output",
        output_template,
    ]
    if ffmpeg:
        command.extend(["--ffmpeg-location", str(Path(ffmpeg).parent)])
    js_runtime = _find_js_runtime()
    if js_runtime:
        command.extend(["--js-runtimes", js_runtime])
    if settings.ytdlp_cookies_file:
        command.extend(["--cookies", settings.ytdlp_cookies_file])
    if settings.ytdlp_cookies_from_browser:
        command.extend(["--cookies-from-browser", settings.ytdlp_cookies_from_browser])
    command.append(url)

    def download() -> Path:
        completed = _run_ytdlp(command, timeout=90)
        for name in [
            "subtitles.vi.srt",
            "subtitles.vi-orig.srt",
            "subtitles.en.srt",
            "subtitles.en-orig.srt",
            "subtitles.zh.srt",
            "subtitles.zh-Hans.srt",
            "subtitles.zh-Hant.srt",
        ]:
            candidate = work_dir / name
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate

        matches = sorted(work_dir.glob("subtitles*.srt"), key=lambda item: item.stat().st_size, reverse=True)
        if matches:
            return matches[0]
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise SubtitleSourceError(f"yt-dlp không tải được phụ đề. {detail}")
        raise SubtitleSourceError("Video này không có phụ đề phù hợp để ghi vào video.")

    try:
        return await asyncio.to_thread(download)
    except subprocess.TimeoutExpired as exc:
        raise SubtitleSourceError("yt-dlp quá thời gian chờ khi tải phụ đề.") from exc
