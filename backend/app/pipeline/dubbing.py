from pathlib import Path
from typing import Callable

from app.core import settings
from app.models.job import JobStatus, VoiceGender
from app.services.ai.voice import VoiceError, get_voice_engine
from app.services.media.downloader import prepare_source_video
from app.services.media.ffmpeg import extract_audio, find_ffmpeg, probe_video_duration
from app.services.media.renderer import render_video
from app.services.store import job_store
from app.services.subtitles.source import get_or_create_subtitles


class PipelineError(RuntimeError):
    pass


async def process_dubbing_job(job_id: str) -> None:
    try:
        job = job_store.get(job_id)
        if not job:
            return

        work_dir = Path(settings.storage_dir) / "jobs" / job_id
        work_dir.mkdir(parents=True, exist_ok=True)

        def progress(stage: str, percent: int) -> None:
            job_store.update(job_id, stage=stage, progress=percent)

        voice_warning: str | None = None

        job_store.update(job_id, status=JobStatus.processing, stage="Kiểm tra môi trường xử lý", progress=5)
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise PipelineError("Chưa cài FFmpeg hoặc FFmpeg chưa nằm trong PATH.")

        if job.request.clone_voice:
            raise PipelineError("Clone giọng cần cấu hình voice engine riêng trước khi chạy.")
        if job.request.auto_publish:
            raise PipelineError("Auto publish cần cấu hình token YouTube/Facebook trước khi chạy.")

        source_video = await prepare_source_video(
            str(job.request.source_url) if job.request.source_url else None,
            job.request.local_file_path,
            work_dir,
            progress,
        )

        progress("Tách audio bằng FFmpeg", 25)
        audio_file = work_dir / "source_audio.wav"
        await extract_audio(ffmpeg, source_video, audio_file)
        video_duration = await probe_video_duration(ffmpeg, source_video)

        progress("Kiểm tra AI engine", 40)
        if settings.ai_engine != "passthrough":
            raise PipelineError(
                f"AI engine '{settings.ai_engine}' chưa được implement. "
                "Có thể nâng cấp riêng trong app/services/ai."
            )

        if job.request.hard_subtitles and job.request.source_has_hard_subtitles:
            subtitle_file = await get_or_create_subtitles(
                str(job.request.source_url) if job.request.source_url else None,
                audio_file,
                work_dir,
                job.request.source_language,
                progress,
            )
            progress("Tạo giọng đọc khớp phụ đề", 70)
            narration_audio, voice_warning = await _try_synthesize_voice(
                subtitle_file,
                work_dir / "narration_timed.wav",
                job.request.voice_gender,
                video_duration,
                progress,
            )
            output_file = work_dir / "output_keep_existing_subtitles.mp4"
            progress(
                "Render giọng đọc theo timecode" if narration_audio else "Edge TTS lỗi, render video với âm thanh gốc",
                78,
            )
            await render_video(
                ffmpeg,
                source_video,
                output_file,
                work_dir,
                job.request,
                lambda percent: job_store.update(job_id, progress=percent),
                subtitle_file=None,
                narration_audio=narration_audio,
                progress_start=78,
            )
        elif job.request.hard_subtitles:
            subtitle_file = await get_or_create_subtitles(
                str(job.request.source_url) if job.request.source_url else None,
                audio_file,
                work_dir,
                job.request.source_language,
                progress,
            )
            progress("Tạo giọng đọc khớp phụ đề", 70)
            narration_audio, voice_warning = await _try_synthesize_voice(
                subtitle_file,
                work_dir / "narration_timed.wav",
                job.request.voice_gender,
                video_duration,
                progress,
            )
            output_file = work_dir / "output_hardsub.mp4"
            progress(
                "Render phụ đề và giọng đọc theo live view" if narration_audio else "Edge TTS lỗi, render phụ đề với âm thanh gốc",
                78,
            )
            await render_video(
                ffmpeg,
                source_video,
                output_file,
                work_dir,
                job.request,
                lambda percent: job_store.update(job_id, progress=percent),
                subtitle_file=subtitle_file,
                narration_audio=narration_audio,
                progress_start=78,
            )
        else:
            output_file = work_dir / "output_passthrough.mp4"
            progress("Render video theo tuỳ chọn logo/âm thanh", 72)
            await render_video(
                ffmpeg,
                source_video,
                output_file,
                work_dir,
                job.request,
                lambda percent: job_store.update(job_id, progress=percent),
                subtitle_file=None,
            )

        job_store.update(
            job_id,
            status=JobStatus.completed,
            stage="Hoàn tất pipeline và phụ đề",
            progress=100,
            output_video_url=f"/api/jobs/{job_id}/download",
            output_file_path=str(output_file),
            seo_title="Video đã được xử lý",
            seo_description=_completion_description(voice_warning),
            error=voice_warning,
        )
    except Exception as exc:
        job_store.update(job_id, status=JobStatus.failed, stage="Xử lý thất bại", error=str(exc))


async def _try_synthesize_voice(
    subtitle_file: Path,
    output_file: Path,
    voice_gender: VoiceGender,
    video_duration: float,
    progress: Callable[[str, int], None],
) -> tuple[Path | None, str | None]:
    try:
        narration_audio = await get_voice_engine().synthesize_srt(
            subtitle_file,
            output_file,
            voice_gender,
            video_duration,
            progress,
        )
        return narration_audio, None
    except VoiceError as exc:
        if not settings.render_without_tts_on_error:
            raise
        progress("Edge TTS lỗi, tiếp tục render với âm thanh gốc", 76)
        return None, str(exc)


def _completion_description(voice_warning: str | None) -> str:
    base = (
        "Pipeline đã tải video, tạo phụ đề và render file MP4 đầu ra. "
        "Logo, phụ đề và xử lý âm thanh đã được áp dụng theo cấu hình."
    )
    if voice_warning:
        return f"{base} Edge TTS đang lỗi nên bản này giữ âm thanh gốc. Chi tiết: {voice_warning}"
    return f"{base} Giọng đọc đã được tạo theo timecode phụ đề."
