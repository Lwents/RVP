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
                    safe_name = f"logo_kenh{logo_src.suffix}"
                    import shutil
                    shutil.copy(logo_src, dest_dir / safe_name)
                    auto_logo_data = {
                        "watermark_file_name": safe_name,
                        "logo_x_percent": 2,
                        "logo_y_percent": 3,
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
