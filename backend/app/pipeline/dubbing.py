from pathlib import Path
import asyncio
import traceback
from datetime import UTC, datetime
from typing import Callable

from app.core import settings
from app.models.job import JobStatus, VoiceGender, BgmMode
from app.services.ai.voice import VoiceError, get_voice_engine
from app.services.media.downloader import prepare_source_video
from app.services.media.ffmpeg import extract_audio, extract_demucs_audio, find_ffmpeg, probe_video_duration
from app.services.media.renderer import render_video
from app.services.media.separation import separate_background_with_demucs
from app.services.presets import get_processing_profile
from app.services.store import job_store
from app.services.subtitles.source import asr_timeout_for_duration, get_or_create_subtitles
from app.models.job import PublishTarget

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
        narration_audio: Path | None = None
        profile = get_processing_profile(job.request.processing_mode)
        effective_request = job.request
        if (
            not profile.use_demucs
            and job.request.use_demucs
            and job.request.bgm_mode != BgmMode.none
        ):
            effective_request = job.request.model_copy(
                update={
                    "use_demucs": False,
                    "bgm_mode": (
                        BgmMode.ducking
                        if job.request.bgm_mode == BgmMode.demucs
                        else job.request.bgm_mode
                    ),
                }
            )

        job_store.update(job_id, status=JobStatus.processing, stage="Kiểm tra môi trường xử lý", progress=5)
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise PipelineError("Chưa cài FFmpeg hoặc FFmpeg chưa nằm trong PATH.")

        if job.request.clone_voice:
            raise PipelineError("Clone giọng cần cấu hình voice engine riêng trước khi chạy.")
        if job.request.auto_publish:
            publish_error = _auto_publish_blocker(job.request.auto_publish)
            if publish_error:
                raise PipelineError(publish_error)

        source_video = await prepare_source_video(
            str(job.request.source_url) if job.request.source_url else None,
            job.request.local_file_path,
            work_dir,
            progress,
        )

        progress("Tách audio bằng FFmpeg", 25)
        audio_file = work_dir / "source_audio.wav"
        if not audio_file.exists() or audio_file.stat().st_size == 0:
            await extract_audio(ffmpeg, source_video, audio_file)
        video_duration = await probe_video_duration(ffmpeg, source_video)
        asr_timeout_seconds = asr_timeout_for_duration(video_duration)

        progress("Kiểm tra AI engine", 40)
        if settings.ai_engine != "passthrough":
            raise PipelineError(
                f"AI engine '{settings.ai_engine}' chưa được implement. "
                "Có thể nâng cấp riêng trong app/services/ai."
            )

        bgm_audio: Path | None = None
        demucs_failed = False
        demucs_warning: str | None = None
        if effective_request.bgm_mode != BgmMode.none:
            if effective_request.use_demucs:
                progress("Tách nhạc nền và giọng nói gốc bằng Demucs", 45)
                try:
                    demucs_audio = work_dir / "demucs_audio.wav"
                    if not demucs_audio.exists() or demucs_audio.stat().st_size == 0:
                        await extract_demucs_audio(ffmpeg, source_video, demucs_audio)
                    separated_dir = work_dir / "separated" / "htdemucs" / demucs_audio.stem
                    existing_bgm = separated_dir / "no_vocals.wav"
                    bgm_audio = (
                        existing_bgm
                        if existing_bgm.exists() and existing_bgm.stat().st_size > 0
                        else await separate_background_with_demucs(demucs_audio, separated_dir)
                    )
                except Exception as exc:
                    # Không được im lặng trả về video gốc: người dùng yêu cầu bỏ
                    # giọng gốc thì phải thấy rõ là Demucs đã hỏng.
                    print(f"Demucs warning/error: {exc}")
                    bgm_audio = None
                    demucs_failed = True
                    demucs_warning = (
                        "Demucs tách nhạc nền/giọng gốc thất bại nên phần âm thanh chỉ được "
                        f"xử lý ở chế độ dự phòng. Chi tiết: {exc}"
                    )
            else:
                bgm_audio = None # Fallback to ducking or keeping original mix

        subtitle_file: Path | None = None
        if job.request.hard_subtitles and job.request.source_has_hard_subtitles:
            subtitle_file = await get_or_create_subtitles(
                str(job.request.source_url) if job.request.source_url else None,
                audio_file,
                work_dir,
                job.request.source_language,
                progress,
                source_video,
                asr_timeout_seconds=asr_timeout_seconds,
                processing_mode=job.request.processing_mode,
            )
            progress("Tạo giọng đọc khớp phụ đề", 70)
            narration_file = work_dir / "narration_timed.wav"
            if narration_file.exists() and narration_file.stat().st_size > 0:
                narration_audio, voice_warning = narration_file, None
            else:
                narration_audio, voice_warning = await _try_synthesize_voice(
                    subtitle_file,
                    narration_file,
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
                effective_request,
                lambda percent: job_store.update(job_id, progress=percent),
                subtitle_file=None,
                narration_audio=narration_audio,
                bgm_audio=bgm_audio,
                progress_start=78,
                demucs_failed=demucs_failed,
            )
        elif job.request.hard_subtitles:
            subtitle_file = await get_or_create_subtitles(
                str(job.request.source_url) if job.request.source_url else None,
                audio_file,
                work_dir,
                job.request.source_language,
                progress,
                source_video,
                asr_timeout_seconds=asr_timeout_seconds,
                processing_mode=job.request.processing_mode,
            )
            progress("Tạo giọng đọc khớp phụ đề", 70)
            narration_file = work_dir / "narration_timed.wav"
            if narration_file.exists() and narration_file.stat().st_size > 0:
                narration_audio, voice_warning = narration_file, None
            else:
                narration_audio, voice_warning = await _try_synthesize_voice(
                    subtitle_file,
                    narration_file,
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
                effective_request,
                lambda percent: job_store.update(job_id, progress=percent),
                subtitle_file=subtitle_file,
                narration_audio=narration_audio,
                bgm_audio=bgm_audio,
                progress_start=78,
                demucs_failed=demucs_failed,
            )
        else:
            output_file = work_dir / "output_passthrough.mp4"
            progress("Render video theo tuỳ chọn logo/âm thanh", 72)
            await render_video(
                ffmpeg,
                source_video,
                output_file,
                work_dir,
                effective_request,
                lambda percent: job_store.update(job_id, progress=percent),
                subtitle_file=None,
                bgm_audio=bgm_audio,
                demucs_failed=demucs_failed,
            )

        seo_title = "Video đã được xử lý"
        seo_description = _completion_description(voice_warning)
        seo_tags: list[str] = []

        if subtitle_file and subtitle_file.exists():
            progress("AI đang viết tiêu đề, mô tả và hashtag YouTube", 98)
            from app.services.ai.content import generate_video_details

            try:
                video_details = await generate_video_details(subtitle_file.read_text(encoding="utf-8"))
                seo_title = video_details.title
                seo_description = video_details.description
                seo_tags = video_details.tags
            except Exception as exc:
                print(f"Error generating YouTube metadata: {exc}")

        progress("AI đang đánh giá video đầu ra", 99)
        final_evaluation = None
        evaluation_stage = "Hoàn tất pipeline · chưa thể hậu kiểm"
        try:
            from app.services.ai.final_dubbing_evaluator import (
                discover_dubbing_evaluation_artifacts,
                dubbing_evaluation_stage,
                emergency_dubbing_evaluation,
                evaluate_final_dubbing_video,
            )

            try:
                artifacts = discover_dubbing_evaluation_artifacts(work_dir)
                final_evaluation = await evaluate_final_dubbing_video(
                    output_file,
                    work_dir,
                    subtitle_file=subtitle_file or artifacts.subtitle_file,
                    source_subtitle_file=artifacts.source_subtitle_file,
                    narration_audio=narration_audio or artifacts.narration_audio,
                    source_video=source_video or artifacts.source_video,
                    subtitles_expected=job.request.hard_subtitles,
                    narration_expected=job.request.hard_subtitles,
                    target_subtitle_burned=(
                        job.request.hard_subtitles
                        and not job.request.source_has_hard_subtitles
                    ),
                    translation_expected=(
                        settings.translation_engine.lower() != "passthrough"
                        and (
                            job.request.source_language == "auto"
                            or job.request.source_language.lower() != settings.target_language.lower()
                        )
                    ),
                    source_language=job.request.source_language,
                    target_language=settings.target_language,
                    video_speed=job.request.video_speed,
                )
            except Exception as exc:
                # Evaluation is advisory. Preserve the completed MP4 even if
                # discovery or the evaluator itself fails unexpectedly.
                print(f"Final dubbing evaluation error: {exc}")
                final_evaluation = emergency_dubbing_evaluation(str(exc))
            evaluation_stage = dubbing_evaluation_stage(final_evaluation)
        except Exception as exc:
            # Even an import/integration failure after render cannot turn the
            # already-created MP4 into a failed job.
            print(f"Final dubbing evaluator unavailable: {exc}")
        
        if job.request.auto_publish and PublishTarget.youtube in job.request.auto_publish:
            progress("Tự động đăng lên YouTube...", 92)
            from app.services.youtube.upload import upload_video_to_youtube
            try:
                # We can run upload in an executor since it's a blocking sync function
                loop = asyncio.get_running_loop()
                yt_response = await loop.run_in_executor(
                    None, 
                    upload_video_to_youtube,
                    str(output_file.absolute()),
                    seo_title,
                    seo_description,
                    seo_tags,
                    "public" # Hoặc private tuỳ ý
                )
                seo_description += f"\n\nĐã đăng lên YouTube thành công! Video ID: {yt_response['id']}"
            except Exception as e:
                print(f"Error uploading to YouTube: {e}")
                seo_description += f"\n\nLỗi đăng YouTube: {e}"

        job_store.update(
            job_id,
            status=JobStatus.completed,
            stage=evaluation_stage,
            progress=100,
            completed_at=datetime.now(UTC),
            output_video_url=f"/api/jobs/{job_id}/download",
            output_file_path=str(output_file),
            seo_title=seo_title,
            seo_description=seo_description,
            seo_tags=seo_tags,
            final_evaluation=final_evaluation,
            error=_job_warning(voice_warning, demucs_warning),
        )
    except asyncio.CancelledError:
        job_store.update(job_id, status=JobStatus.failed, stage="Đã huỷ", error="Người dùng huỷ tiến trình.")
        raise
    except Exception as exc:
        traceback.print_exc()
        error = str(exc).strip() or f"{type(exc).__name__}: {exc!r}"
        job_store.update(job_id, status=JobStatus.failed, stage="Xử lý thất bại", error=error)
    finally:
        from app.services import task_manager
        task_manager.unregister_task(job_id)


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


def _job_warning(*warnings: str | None) -> str | None:
    """Merge every non-fatal warning into the single field the UI shows."""

    messages = [warning.strip() for warning in warnings if warning and warning.strip()]
    return "\n".join(messages) if messages else None


def _missing_youtube_credentials() -> str | None:
    """Return a Vietnamese error when YouTube auto publish cannot possibly work."""

    credentials_file = Path(settings.youtube_credentials_file)
    if credentials_file.exists() and credentials_file.stat().st_size > 0:
        return None

    client_secrets_file = Path(settings.youtube_client_secrets_file)
    if not client_secrets_file.exists():
        return (
            f"Auto publish YouTube cần file {settings.youtube_client_secrets_file} "
            f"và {settings.youtube_credentials_file}. Hãy tạo OAuth client trên Google Cloud "
            "Console rồi kết nối tài khoản YouTube trước khi chạy."
        )
    return (
        f"Chưa xác thực YouTube ({settings.youtube_credentials_file} chưa có). "
        "Hãy kết nối tài khoản YouTube trước khi bật auto publish."
    )


def _auto_publish_blocker(targets: list[PublishTarget]) -> str | None:
    if PublishTarget.facebook in targets:
        return "Auto publish Facebook chưa được hỗ trợ, hãy bỏ chọn Facebook."
    if PublishTarget.youtube in targets:
        return _missing_youtube_credentials()
    return None


def _completion_description(voice_warning: str | None) -> str:
    base = (
        "Pipeline đã tải video, tạo phụ đề và render file MP4 đầu ra. "
        "Logo, phụ đề và xử lý âm thanh đã được áp dụng theo cấu hình."
    )
    if voice_warning:
        return f"{base} Edge TTS đang lỗi nên bản này giữ âm thanh gốc. Chi tiết: {voice_warning}"
    return f"{base} Giọng đọc đã được tạo theo timecode phụ đề."
