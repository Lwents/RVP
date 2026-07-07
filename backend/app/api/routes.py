from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.core import settings
from app.models.job import DubbingRequest, JobCreateResponse, JobProgress, UploadResponse
from app.services.processor import process_job
from app.services.store import job_store
from app.services.media.downloader import download_preview_video

router = APIRouter()

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


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
    job_store.clear()
    return {"message": "Jobs and generated files were moved to the recycle bin."}


@router.get("/jobs/{job_id}", response_model=JobProgress, tags=["jobs"])
async def get_job(job_id: str) -> JobProgress:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


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

@router.get("/youtube/auth-url", tags=["youtube"])
async def get_yt_auth_url():
    try:
        url = get_youtube_auth_url()
        return {"url": url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/youtube/callback", tags=["youtube"])
async def yt_callback(req: YoutubeCallbackRequest):
    try:
        result = handle_oauth2_callback(req.code)
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
