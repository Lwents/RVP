from pathlib import Path
from uuid import uuid4
from datetime import UTC, datetime

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.core import settings
from app.models.job import DubbingRequest, JobCreateResponse, JobProgress, UploadResponse
from app.services.processor import process_job
from app.services.store import job_store
from app.services.media.downloader import download_preview_video

router = APIRouter()

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


class ReviewDraftRequest(BaseModel):
    video_path: str
    target_minutes: int = Field(default=8, ge=1, le=30)
    style: str = "story"
    source_language: str = "auto"
    notes: str | None = None


class ReviewBeatResponse(BaseModel):
    time_hint: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    purpose: str
    narration: str


class ReviewDraftResult(BaseModel):
    title: str
    target_minutes: int
    hook: str
    summary: str
    narration_script: str
    beats: list[ReviewBeatResponse]
    thumbnail_text: str
    tags: list[str]
    subtitle_file_path: str | None = None
    output_video_url: str | None = None
    output_file_path: str | None = None


class ReviewDraftJob(BaseModel):
    job_id: str
    status: str
    progress: int = Field(ge=0, le=100)
    stage: str
    request: ReviewDraftRequest
    created_at: datetime
    updated_at: datetime
    result: ReviewDraftResult | None = None
    error: str | None = None


review_draft_jobs: dict[str, ReviewDraftJob] = {}


def _encode_round_logo_png(image_bytes: bytes) -> bytes:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Khong doc duoc file logo.")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    elif image.shape[2] != 4:
        raise ValueError("Dinh dang logo khong ho tro.")

    height, width = image.shape[:2]
    size = min(height, width)
    offset_y = max(0, (height - size) // 2)
    offset_x = max(0, (width - size) // 2)
    square = image[offset_y:offset_y + size, offset_x:offset_x + size].copy()

    alpha_mask = np.zeros((size, size), dtype=np.uint8)
    radius = max(1, size // 2 - max(2, size // 50))
    cv2.circle(alpha_mask, (size // 2, size // 2), radius, 255, -1, lineType=cv2.LINE_AA)
    square[:, :, 3] = np.minimum(square[:, :, 3], alpha_mask)

    ok, encoded = cv2.imencode(".png", square)
    if not ok:
        raise ValueError("Khong the ma hoa logo PNG.")
    return encoded.tobytes()


@router.post("/uploads/watermark", response_model=UploadResponse, tags=["uploads"])
async def upload_watermark(file: UploadFile = File(...)) -> UploadResponse:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Watermark must be an image.")

    data = await file.read()
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Watermark image must be smaller than 5MB.")

    storage_dir = Path(settings.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "watermark.png").suffix or ".png"
    safe_name = f"{uuid4()}{suffix}"
    destination = storage_dir / safe_name
    destination.write_bytes(data)

    return UploadResponse(
        file_name=safe_name,
        content_type=file.content_type,
        size=len(data),
        url=f"/api/uploads/watermark/{safe_name}",
    )


@router.post("/uploads/video", response_model=UploadResponse, tags=["uploads"])
async def upload_video(file: UploadFile = File(...)) -> UploadResponse:
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Video file must be mp4, mov, mkv, webm, avi, or m4v.")
    if file.content_type and not (file.content_type.startswith("video/") or file.content_type in {"application/octet-stream", "video/x-matroska"}):
        raise HTTPException(status_code=400, detail="Uploaded file does not look like a video.")

    storage_dir = Path(settings.storage_dir) / "uploads" / "videos"
    storage_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid4()}{suffix}"
    destination = storage_dir / safe_name

    size = 0
    with destination.open("wb") as handle:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > 5 * 1024 * 1024 * 1024:
                destination.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail="Video file must be smaller than 5GB.")
            handle.write(chunk)

    return UploadResponse(
        file_name=safe_name,
        content_type=file.content_type,
        size=size,
        url=f"/api/uploads/video/{safe_name}",
        local_file_path=str(destination),
    )


class UrlPreviewRequest(BaseModel):
    url: str


@router.post("/uploads/url_preview", response_model=UploadResponse, tags=["uploads"])
async def upload_url_preview(request: UrlPreviewRequest) -> UploadResponse:
    storage_dir = Path(settings.storage_dir) / "uploads" / "videos"
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    job_id = str(uuid4())
    work_dir = storage_dir / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        downloaded_path = await download_preview_video(request.url, work_dir)
        safe_name = f"{job_id}{downloaded_path.suffix}"
        final_destination = storage_dir / safe_name
        downloaded_path.rename(final_destination)
        
        return UploadResponse(
            file_name=safe_name,
            content_type="video/mp4",
            size=final_destination.stat().st_size,
            url=f"/api/uploads/video/{safe_name}",
            local_file_path=str(final_destination),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Cleanup work dir
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)


def _resolve_video_path(video_path: str) -> Path:
    path_obj = Path(video_path)
    if path_obj.exists():
        return path_obj.resolve()

    storage_path = Path(settings.storage_dir) / video_path
    if storage_path.exists():
        return storage_path.resolve()

    if video_path.startswith("storage/") or video_path.startswith("storage\\"):
        relative = video_path.replace("\\", "/").removeprefix("storage/")
        storage_relative = Path(settings.storage_dir) / relative
        if storage_relative.exists():
            return storage_relative.resolve()

    raise FileNotFoundError(f"Khong tim thay video: {video_path}")


def _update_review_job(job_id: str, **changes: object) -> ReviewDraftJob:
    current = review_draft_jobs[job_id]
    data = current.model_dump()
    data.update(changes)
    data["updated_at"] = datetime.now(UTC)
    updated = ReviewDraftJob.model_validate(data)
    review_draft_jobs[job_id] = updated
    return updated


async def _process_review_draft_job(job_id: str) -> None:
    from app.services.ai.content import generate_movie_review_plan
    from app.services.ai.voice import get_voice_engine
    from app.services.media.ffmpeg import extract_audio, find_ffmpeg, probe_video_duration
    from app.services.media.review_renderer import render_movie_review_video, write_review_subtitles
    from app.models.job import VoiceGender
    from app.services.subtitles.source import get_or_create_subtitles

    job = review_draft_jobs[job_id]
    try:
        _update_review_job(job_id, status="processing", progress=5, stage="Kiem tra video")
        source_video = _resolve_video_path(job.request.video_path)
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("Khong tim thay FFmpeg.")

        work_dir = Path(settings.storage_dir) / "review_jobs" / job_id
        work_dir.mkdir(parents=True, exist_ok=True)

        def progress(stage: str, percent: int) -> None:
            _update_review_job(job_id, stage=stage, progress=max(5, min(80, percent)))

        _update_review_job(job_id, progress=20, stage="Tach audio phim")
        audio_file = work_dir / "source_audio.wav"
        await extract_audio(ffmpeg, source_video, audio_file)
        video_duration = await probe_video_duration(ffmpeg, source_video)
        asr_timeout_seconds = _review_asr_timeout(video_duration)

        _update_review_job(job_id, progress=35, stage="Tao transcript va phu de tam")
        subtitle_file = await get_or_create_subtitles(
            None,
            audio_file,
            work_dir,
            job.request.source_language,
            progress,
            source_video,
            asr_timeout_seconds=asr_timeout_seconds,
        )

        _update_review_job(job_id, progress=82, stage="AI viet kich ban review phim")
        plan = await generate_movie_review_plan(
            subtitle_file.read_text(encoding="utf-8"),
            target_minutes=job.request.target_minutes,
            style=job.request.style,
            custom_prompt=job.request.notes,
        )
        _update_review_job(job_id, progress=84, stage="Tao giong doc review")
        narration_audio = await get_voice_engine().synthesize(
            plan.narration_script,
            work_dir / "review_narration.mp3",
            VoiceGender.female,
            lambda stage, percent: _update_review_job(job_id, stage=stage, progress=max(84, min(90, percent))),
        )

        _update_review_job(job_id, progress=90, stage="Cat ghep video review")
        output_file = work_dir / "review_output.mp4"
        review_subtitle_file = write_review_subtitles(
            plan.narration_script,
            work_dir / "review_subtitles.srt",
            job.request.target_minutes,
        )
        await render_movie_review_video(
            ffmpeg,
            source_video,
            narration_audio,
            review_subtitle_file,
            output_file,
            work_dir,
            job.request.target_minutes,
            [
                {
                    "time_hint": beat.time_hint,
                    "start_seconds": beat.start_seconds,
                    "end_seconds": beat.end_seconds,
                }
                for beat in plan.beats
            ],
            lambda percent: _update_review_job(job_id, progress=max(90, min(99, percent))),
        )

        result = ReviewDraftResult(
            title=plan.title,
            target_minutes=plan.target_minutes,
            hook=plan.hook,
            summary=plan.summary,
            narration_script=plan.narration_script,
            beats=[
                ReviewBeatResponse(
                    time_hint=beat.time_hint,
                    start_seconds=beat.start_seconds,
                    end_seconds=beat.end_seconds,
                    purpose=beat.purpose,
                    narration=beat.narration,
                )
                for beat in plan.beats
            ],
            thumbnail_text=plan.thumbnail_text,
            tags=plan.tags,
            subtitle_file_path=str(review_subtitle_file),
            output_video_url=f"/api/review/jobs/{job_id}/download",
            output_file_path=str(output_file),
        )
        _update_review_job(
            job_id,
            status="completed",
            progress=100,
            stage="Hoan tat ban review phim",
            result=result,
        )
    except Exception as exc:
        _update_review_job(
            job_id,
            status="failed",
            stage="Tao review phim that bai",
            error=str(exc),
        )


def _review_asr_timeout(video_duration_seconds: float) -> int:
    if video_duration_seconds <= 0:
        return max(settings.asr_timeout_seconds, 1800)

    # Full movie review jobs can legitimately need much longer than the default
    # short-video ASR timeout. Give Whisper time proportional to the movie length.
    duration_based_timeout = int(video_duration_seconds * 2.5)
    return min(max(settings.asr_timeout_seconds, duration_based_timeout, 1800), 6 * 60 * 60)


@router.post("/review/jobs", response_model=ReviewDraftJob, tags=["review"])
async def create_review_draft_job(request: ReviewDraftRequest) -> ReviewDraftJob:
    import asyncio

    now = datetime.now(UTC)
    job = ReviewDraftJob(
        job_id=str(uuid4()),
        status="queued",
        progress=0,
        stage="Da nhan phim",
        request=request,
        created_at=now,
        updated_at=now,
    )
    review_draft_jobs[job.job_id] = job
    asyncio.create_task(_process_review_draft_job(job.job_id))
    return job


@router.get("/review/jobs/{job_id}", response_model=ReviewDraftJob, tags=["review"])
async def get_review_draft_job(job_id: str) -> ReviewDraftJob:
    job = review_draft_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Review job not found.")
    return job


@router.get("/review/jobs/{job_id}/download", tags=["review"])
async def download_review_output(job_id: str) -> FileResponse:
    job = review_draft_jobs.get(job_id)
    if not job or not job.result or not job.result.output_file_path:
        raise HTTPException(status_code=404, detail="Review video not found.")
    output_file = Path(job.result.output_file_path)
    if not output_file.exists():
        raise HTTPException(status_code=404, detail="Review video file no longer exists.")
    return FileResponse(output_file, media_type="video/mp4", filename=f"review_{job_id}.mp4")


from app.services import task_manager

@router.post("/jobs", response_model=JobCreateResponse, tags=["jobs"])
async def create_job(request: DubbingRequest, background_tasks: BackgroundTasks) -> JobCreateResponse:
    job = job_store.create(request)
    
    import asyncio
    task = asyncio.create_task(process_job(job.job_id))
    task_manager.register_task(job.job_id, task)
    
    # We add a background task just to await it so it's not orphaned entirely if we wanted to
    # but create_task is already enough. We'll just rely on create_task.
    return JobCreateResponse(job_id=job.job_id, status=job.status)


@router.get("/jobs", response_model=list[JobProgress], tags=["jobs"])
async def list_jobs() -> list[JobProgress]:
    return job_store.list()


@router.delete("/jobs", tags=["jobs"])
async def clear_jobs() -> dict[str, str]:
    cancelled = task_manager.cancel_all_tasks()
    warnings = job_store.clear()

    message = "Jobs and generated files were moved to the recycle bin."
    if cancelled:
        message = f"{message} Cancelled {cancelled} running task(s)."
    if warnings:
        message = f"{message} Some files could not be moved because they are in use."

    response = {"message": message}
    if warnings:
        response["warnings"] = "\n".join(warnings)
    return response


@router.get("/jobs/{job_id}", response_model=JobProgress, tags=["jobs"])
async def get_job(job_id: str) -> JobProgress:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@router.post("/jobs/{job_id}/metadata", response_model=JobProgress, tags=["jobs"])
async def generate_job_metadata(job_id: str) -> JobProgress:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    work_dir = Path(settings.storage_dir) / "jobs" / job_id
    subtitle_file = next(
        (path for path in [
            work_dir / "subtitles.vi.srt",
            work_dir / "subtitles.grouped.srt",
            work_dir / "subtitles.whisper.srt",
        ] if path.exists()),
        None,
    )
    if not subtitle_file:
        raise HTTPException(status_code=404, detail="Job này chưa có phụ đề để AI viết nội dung YouTube.")

    from app.services.ai.content import generate_video_details

    details = await generate_video_details(subtitle_file.read_text(encoding="utf-8"))
    return job_store.update(
        job_id,
        seo_title=details.title,
        seo_description=details.description,
        seo_tags=details.tags,
    )


@router.get("/jobs/{job_id}/download", tags=["jobs"])
async def download_output(job_id: str) -> FileResponse:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if not job.output_file_path:
        raise HTTPException(status_code=404, detail="This job does not have an output video yet.")
    output_file = Path(job.output_file_path)
    if not output_file.exists():
        raise HTTPException(status_code=404, detail="Output file no longer exists.")
    return FileResponse(output_file, media_type="video/mp4", filename=output_file.name)

@router.post("/jobs/{job_id}/cancel", tags=["jobs"])
async def cancel_job(job_id: str) -> dict[str, str]:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if task_manager.cancel_task(job_id):
        return {"message": "Đã gửi yêu cầu hủy tiến trình."}
    return {"message": "Không thể hủy (job không chạy hoặc đã kết thúc)."}


from app.services.youtube.auth import get_youtube_auth_url, handle_oauth2_callback
from app.services.youtube.upload import get_channel_videos_stats
from pydantic import BaseModel

class YoutubeCallbackRequest(BaseModel):
    code: str
    redirect_uri: str = "http://localhost:5173/youtube/callback"

@router.post("/youtube/client-secret", tags=["youtube"])
async def upload_client_secret(file: UploadFile = File(...)):
    try:
        data = await file.read()
        import json
        secret_data = json.loads(data)
        if "web" not in secret_data and "installed" not in secret_data:
            raise HTTPException(status_code=400, detail="Định dạng file client_secret.json không hợp lệ. Phải chứa khoá 'web' hoặc 'installed'.")
        
        dest = Path(settings.youtube_client_secrets_file)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return {"status": "success", "message": "Đã lưu tệp client_secret.json thành công."}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Tệp tải lên không phải là định dạng JSON hợp lệ.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/youtube/auth-url", tags=["youtube"])
async def get_yt_auth_url(redirect_uri: str = "http://localhost:5173/youtube/callback"):
    try:
        url = get_youtube_auth_url(redirect_uri=redirect_uri)
        return {"url": url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/youtube/callback", tags=["youtube"])
async def yt_callback(req: YoutubeCallbackRequest):
    try:
        result = handle_oauth2_callback(req.code, redirect_uri=req.redirect_uri)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/youtube/stats", tags=["youtube"])
async def yt_stats():
    try:
        stats = get_channel_videos_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─── Auto-detect blur regions ─────────────────────────────────────────────────

class DetectRegionsRequest(BaseModel):
    video_path: str
    detect_sub: bool = True
    detect_logo: bool = True
    engine: str = "local"


@router.post("/analyze/detect-regions", tags=["analyze"])
async def detect_blur_regions(req: DetectRegionsRequest):
    """
    Phân tích video gốc để tự động nhận diện:
    - Vùng chứa sub chữ Trung/Hán
    - Vùng logo/watermark cố định của kênh gốc
    Hỗ trợ công nghệ Cục bộ (local) hoặc Trí tuệ nhân tạo (ai via 9router).
    """
    from app.services.ai.detect_regions import auto_detect_blur_regions_local, auto_detect_blur_regions_ai, review_and_build_blur_config
    import asyncio

    video_path = req.video_path
    path_obj = Path(video_path)
    
    # Try resolving relative to storage_dir if direct file doesn't exist
    if not path_obj.exists():
        path_obj = Path(settings.storage_dir) / video_path
        
    # Handle case where path already starts with storage folder name
    if not path_obj.exists() and video_path.startswith("storage/"):
        path_obj = Path(settings.storage_dir) / video_path[len("storage/"):]
        
    if not path_obj.exists():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy video: {video_path} (đã thử phân giải thành {path_obj})")

    video_path = str(path_obj.resolve())

    try:
        # Tự động chèn logo kênh sang bên trái từ thư mục người dùng
        logo_dir = Path(r"C:\Users\kirit\Pictures\logo")
        auto_logo_data = None
        if logo_dir.exists() and logo_dir.is_dir():
            # Quét tìm ảnh đầu tiên
            logo_src = None
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
                found = list(logo_dir.glob(ext))
                if found:
                    logo_src = found[0]
                    break
            
            if logo_src:
                try:
                    dest_dir = Path(settings.storage_dir)
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    safe_name = "logo_kenh.png"
                    encoded_logo = _encode_round_logo_png(logo_src.read_bytes())
                    (dest_dir / safe_name).write_bytes(encoded_logo)
                    auto_logo_data = {
                        "watermark_file_name": safe_name,
                        "logo_width": 88,
                        "logo_x_percent": 3,
                        "logo_y_percent": 4,
                        "logo_enabled": True,
                        "display_name": logo_src.name
                    }
                except Exception as e:
                    print(f"Lỗi sao chép logo từ {logo_src}: {e}")

        if req.engine == "ai":
            # Gọi trực tiếp vì hàm async
            try:
                regions = await auto_detect_blur_regions_ai(
                    video_path,
                    detect_sub=req.detect_sub,
                    detect_logo=req.detect_logo,
                )
            except Exception as e:
                print(f"AI detect failed, falling back to local detection: {e}")
                regions = []
            if not regions:
                loop = asyncio.get_event_loop()
                regions = await loop.run_in_executor(
                    None,
                    lambda: auto_detect_blur_regions_local(
                        video_path,
                        detect_sub=req.detect_sub,
                        detect_logo=req.detect_logo,
                    )
                )
        else:
            # Chạy local (đồng bộ) trong thread pool
            loop = asyncio.get_event_loop()
            regions = await loop.run_in_executor(
                None,
                lambda: auto_detect_blur_regions_local(
                    video_path,
                    detect_sub=req.detect_sub,
                    detect_logo=req.detect_logo,
                )
            )
        checked = review_and_build_blur_config(regions)
        return {
            "regions": checked["regions"],
            "count": len(checked["regions"]),
            "auto_logo": auto_logo_data,
            "config": checked["config"],
            "review": checked["review"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích video: {str(e)}")
