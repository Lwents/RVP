from pathlib import Path
from uuid import uuid4
from datetime import UTC, datetime
import asyncio
import shutil

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import json

from app.core import settings
from app.models.job import BgmMode, CustomBlurBox, DubbingRequest, JobCreateResponse, JobProgress, JobStatus, UploadResponse
from app.models.review import (
    AnalyzedScene,
    EditDecision,
    NarrationSegment,
    QualityIssue,
    ReviewQualityReport,
    SceneCandidate,
    StoryEvent,
    VerifiedReviewPackage,
)
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
    hard_subtitles: bool = True
    subtitle_x_percent: int = Field(default=50, ge=0, le=100)
    subtitle_y_percent: int = Field(default=92, ge=8, le=94)
    subtitle_font_size: int = Field(default=64, ge=16, le=120)
    subtitle_box_enabled: bool = False
    subtitle_box_opacity: int = Field(default=0, ge=0, le=100)
    subtitle_box_height_percent: int = Field(default=18, ge=8, le=45)
    logo_enabled: bool = True
    logo_width: int = Field(default=86, ge=32, le=800)
    logo_x_percent: int = Field(default=6, ge=0, le=100)
    logo_y_percent: int = Field(default=8, ge=0, le=100)
    cinematic_bars_enabled: bool = False
    cinematic_bars_height_percent: int = Field(default=10, ge=0, le=40)
    blur_box_enabled: bool = False
    blur_box_y_percent: int = Field(default=80, ge=0, le=100)
    blur_box_height_percent: int = Field(default=13, ge=5, le=100)
    custom_blur_boxes: list[CustomBlurBox] = Field(default_factory=list)
    watermark_file_name: str | None = None
    output_resolution: str = "original"


class ReviewBeatResponse(BaseModel):
    time_hint: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    purpose: str
    narration: str
    segment_id: str | None = None
    event_id: str | None = None
    scene_id: str | None = None
    voice_start: float | None = None
    voice_end: float | None = None
    match_score: float | None = None
    match_reason: str | None = None
    selected_candidate_id: str | None = None
    thumbnail_url: str | None = None
    candidates: list[SceneCandidate] = Field(default_factory=list)


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
    scenes: list[AnalyzedScene] = Field(default_factory=list)
    events: list[StoryEvent] = Field(default_factory=list)
    narration_segments: list[NarrationSegment] = Field(default_factory=list)
    edit_decision_list: list[EditDecision] = Field(default_factory=list)
    quality_report: ReviewQualityReport | None = None
    artifact_paths: dict[str, str] = Field(default_factory=dict)


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


class ReviewSegmentPatchRequest(BaseModel):
    narration: str | None = None
    candidate_id: str | None = None
    scene_id: str | None = None
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, ge=0)


REVIEW_JOBS_INDEX_FILE = Path(settings.storage_dir) / "review_jobs_index.json"
review_draft_jobs: dict[str, ReviewDraftJob] = {}


def _load_review_jobs() -> None:
    global review_draft_jobs
    if not REVIEW_JOBS_INDEX_FILE.exists():
        return
    try:
        data = json.loads(REVIEW_JOBS_INDEX_FILE.read_text(encoding="utf-8"))
        review_draft_jobs = {
            item["job_id"]: ReviewDraftJob.model_validate(item)
            for item in data
            if isinstance(item, dict) and item.get("job_id")
        }
        _mark_interrupted_review_jobs()
    except Exception:
        review_draft_jobs = {}


def _mark_interrupted_review_jobs() -> None:
    changed = False
    for job_id, job in list(review_draft_jobs.items()):
        if job.status not in {"queued", "processing"}:
            continue
        data = job.model_dump()
        data.update(
            {
                "status": "failed",
                "stage": "Review job bị gián đoạn khi backend khởi động lại",
                "error": "Backend đã restart trước khi review job hoàn tất. Hãy chạy lại job này.",
                "updated_at": datetime.now(UTC),
            }
        )
        review_draft_jobs[job_id] = ReviewDraftJob.model_validate(data)
        changed = True
    if changed:
        _save_review_jobs()


def _save_review_jobs() -> None:
    REVIEW_JOBS_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = [job.model_dump(mode="json") for job in review_draft_jobs.values()]
    REVIEW_JOBS_INDEX_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _trash_review_jobs() -> list[str]:
    try:
        from send2trash import send2trash
    except ImportError as exc:
        raise RuntimeError("Thiếu Send2Trash. Chạy pip install -r requirements.txt trong backend.") from exc

    warnings: list[str] = []
    review_dir = Path(settings.storage_dir) / "review_jobs"
    paths = [path for path in review_dir.iterdir() if path.is_dir()] if review_dir.exists() else []
    for path in paths:
        try:
            send2trash(str(path))
        except Exception as exc:
            warnings.append(f"{path}: {exc}")
    return warnings[:10]


_load_review_jobs()


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
    current = review_draft_jobs.get(job_id)
    if current is None:
        raise RuntimeError("Review job đã bị xóa khỏi lịch sử.")
    data = current.model_dump()
    data.update(changes)
    data["updated_at"] = datetime.now(UTC)
    updated = ReviewDraftJob.model_validate(data)
    review_draft_jobs[job_id] = updated
    _save_review_jobs()
    return updated


def _review_decisions_with_urls(job_id: str, decisions: list[EditDecision]) -> list[EditDecision]:
    def with_url(candidate: SceneCandidate) -> SceneCandidate:
        if not candidate.thumbnail_path:
            return candidate
        file_name = Path(candidate.thumbnail_path).name
        return candidate.model_copy(
            update={"thumbnail_url": f"/api/review/jobs/{job_id}/thumbnails/{file_name}"}
        )

    return [
        decision.model_copy(
            update={
                "source_clips": [with_url(candidate) for candidate in decision.source_clips],
                "alternatives": [with_url(candidate) for candidate in decision.alternatives],
            }
        )
        for decision in decisions
    ]


def _review_result_from_package(
    job_id: str,
    package: VerifiedReviewPackage,
    subtitle_file_path: str | None = None,
    output_file_path: str | None = None,
) -> ReviewDraftResult:
    segment_by_id = {segment.segment_id: segment for segment in package.narration_segments}
    beats: list[ReviewBeatResponse] = []
    for decision in package.edit_decision_list:
        segment = segment_by_id.get(decision.segment_id)
        selected = decision.source_clips[0] if decision.source_clips else None
        if selected is None:
            continue
        beats.append(
            ReviewBeatResponse(
                time_hint=_format_review_time_hint(selected.start_seconds, selected.end_seconds),
                start_seconds=selected.start_seconds,
                end_seconds=selected.end_seconds,
                purpose=segment.purpose if segment else "Minh họa sự kiện đã kiểm chứng",
                narration=decision.narration,
                segment_id=decision.segment_id,
                event_id=decision.event_id,
                scene_id=selected.scene_id,
                voice_start=decision.voice_start,
                voice_end=decision.voice_end,
                match_score=selected.match_score,
                match_reason=selected.match_reason,
                selected_candidate_id=decision.selected_candidate_id,
                thumbnail_url=selected.thumbnail_url,
                candidates=decision.alternatives,
            )
        )
    return ReviewDraftResult(
        title=package.title,
        target_minutes=package.target_minutes,
        hook=package.hook,
        summary=package.summary,
        narration_script=package.narration_script,
        beats=beats,
        thumbnail_text=package.thumbnail_text,
        tags=package.tags,
        subtitle_file_path=subtitle_file_path,
        output_video_url=f"/api/review/jobs/{job_id}/download" if output_file_path else None,
        output_file_path=output_file_path,
        scenes=package.scenes,
        events=package.events,
        narration_segments=package.narration_segments,
        edit_decision_list=package.edit_decision_list,
        quality_report=package.quality_report,
        artifact_paths=package.artifact_paths,
    )


def _review_scene_hints(decisions: list[EditDecision]) -> list[dict]:
    hints: list[dict] = []
    for decision in decisions:
        if not decision.source_clips:
            continue
        selected = decision.source_clips[0]
        hints.append(
            {
                "time_hint": _format_review_time_hint(selected.start_seconds, selected.end_seconds),
                "start_seconds": selected.start_seconds,
                "end_seconds": selected.end_seconds,
                "voice_start": decision.voice_start,
                "voice_end": decision.voice_end,
                "duration_seconds": max(0.8, decision.voice_end - decision.voice_start),
                "narration": decision.narration,
            }
        )
    return hints


def _format_review_time_hint(start: float, end: float) -> str:
    def stamp(value: float) -> str:
        seconds = max(0, int(value))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return f"{stamp(start)}-{stamp(end)}"


def _build_review_render_request(request: ReviewDraftRequest, source_video: Path) -> DubbingRequest:
    from app.models.job import VoiceGender

    return DubbingRequest(
        local_file_path=str(source_video),
        voice_gender=VoiceGender.female,
        bgm_mode=BgmMode.none,
        use_demucs=True,
        video_speed=1.0,
        auto_publish=[],
        clone_voice=False,
        hard_subtitles=request.hard_subtitles,
        source_has_hard_subtitles=False,
        subtitle_x_percent=request.subtitle_x_percent,
        subtitle_y_percent=request.subtitle_y_percent,
        subtitle_font_size=request.subtitle_font_size,
        subtitle_box_enabled=request.subtitle_box_enabled,
        subtitle_box_opacity=request.subtitle_box_opacity,
        subtitle_box_height_percent=request.subtitle_box_height_percent,
        source_language=request.source_language,
        ducking_volume_db=-12,
        output_resolution=request.output_resolution,
        logo_enabled=request.logo_enabled,
        logo_width=request.logo_width,
        logo_x_percent=request.logo_x_percent,
        logo_y_percent=request.logo_y_percent,
        cinematic_bars_enabled=request.cinematic_bars_enabled,
        cinematic_bars_height_percent=request.cinematic_bars_height_percent,
        blur_box_enabled=request.blur_box_enabled,
        blur_box_y_percent=request.blur_box_y_percent,
        blur_box_height_percent=request.blur_box_height_percent,
        custom_blur_boxes=request.custom_blur_boxes,
        watermark_file_name=request.watermark_file_name,
    )


async def _process_review_draft_job(job_id: str) -> None:
    from app.services.ai.review_analysis import (
        build_verified_review_package,
        replace_failed_decisions_with_alternatives,
        rescale_edl_voice_timeline,
        verify_rendered_review,
    )
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
        try:
            subtitle_file = await get_or_create_subtitles(
                None,
                audio_file,
                work_dir,
                job.request.source_language,
                progress,
                source_video,
                asr_timeout_seconds=asr_timeout_seconds,
            )
        except Exception as subtitle_error:
            # Review analysis is multilingual and can continue from the source
            # transcript if the contextual translation request temporarily fails.
            source_candidates = [
                work_dir / "subtitles.grouped.srt",
                work_dir / "subtitles.whisper.srt",
            ]
            subtitle_file = next((path for path in source_candidates if path.exists() and path.stat().st_size > 0), None)
            if subtitle_file is None:
                raise
            _update_review_job(
                job_id,
                progress=50,
                stage=f"Dịch tạm lỗi; Gemini sẽ phân tích trực tiếp phụ đề nguồn ({subtitle_error})",
            )

        package = await build_verified_review_package(
            source_video,
            subtitle_file,
            work_dir,
            video_duration,
            job.request.target_minutes,
            job.request.style,
            job.request.notes,
            lambda stage, percent: _update_review_job(
                job_id,
                stage=stage,
                progress=max(50, min(82, percent)),
            ),
        )

        package.edit_decision_list = _review_decisions_with_urls(job_id, package.edit_decision_list)
        preliminary_result = _review_result_from_package(job_id, package)
        if not package.quality_report.passed:
            _update_review_job(
                job_id,
                status="needs_review",
                progress=82,
                stage=(
                    f"QA {package.quality_report.overall_score:.1f}/100: "
                    "cần duyệt các câu màu đỏ trước khi render"
                ),
                result=preliminary_result,
            )
            return

        _update_review_job(job_id, progress=84, stage="Tạo giọng đọc từ kịch bản đã kiểm chứng")
        narration_audio = await get_voice_engine().synthesize(
            package.narration_script,
            work_dir / "review_narration.mp3",
            VoiceGender.female,
            lambda stage, percent: _update_review_job(job_id, stage=stage, progress=max(84, min(90, percent))),
        )
        narration_duration = await probe_video_duration(ffmpeg, narration_audio)
        if narration_duration <= 0:
            raise RuntimeError("Không đo được thời lượng giọng đọc review.")
        package.edit_decision_list = rescale_edl_voice_timeline(
            package.edit_decision_list,
            narration_duration,
        )
        (work_dir / "edit_decision_list.json").write_text(
            json.dumps(
                [item.model_dump(mode="json") for item in package.edit_decision_list],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        scene_hints = _review_scene_hints(package.edit_decision_list)

        _update_review_job(job_id, progress=90, stage="Cắt ghép theo EDL và đồng bộ giọng/cảnh")
        draft_file = work_dir / "review_draft.mp4"
        render_request = _build_review_render_request(job.request, source_video)
        review_subtitle_file = write_review_subtitles(
            package.narration_script,
            work_dir / "review_subtitles.srt",
            job.request.target_minutes,
            scene_hints,
            target_seconds=narration_duration,
        )
        await render_movie_review_video(
            ffmpeg,
            source_video,
            narration_audio,
            review_subtitle_file,
            draft_file,
            work_dir,
            job.request.target_minutes,
            scene_hints,
            lambda percent: _update_review_job(job_id, progress=max(90, min(99, percent))),
            render_request,
        )

        _update_review_job(job_id, progress=98, stage="Gemini kiểm tra lại hình và lời sau render")
        post_report = await verify_rendered_review(draft_file, package, work_dir)
        if not post_report.passed:
            revised_decisions, changed = replace_failed_decisions_with_alternatives(
                package.edit_decision_list,
                post_report,
            )
            if changed:
                package.edit_decision_list = revised_decisions
                scene_hints = _review_scene_hints(package.edit_decision_list)
                retry_file = work_dir / "review_draft_retry.mp4"
                _update_review_job(job_id, progress=98, stage="Tự thay cảnh lỗi và render lại lần cuối")
                await render_movie_review_video(
                    ffmpeg,
                    source_video,
                    narration_audio,
                    review_subtitle_file,
                    retry_file,
                    work_dir,
                    job.request.target_minutes,
                    scene_hints,
                    lambda percent: _update_review_job(job_id, progress=max(98, min(99, percent))),
                    render_request,
                )
                post_report = await verify_rendered_review(retry_file, package, work_dir)
                draft_file = retry_file

        package.quality_report = post_report
        package.artifact_paths["review_draft"] = str(draft_file)
        package.artifact_paths["post_render_quality"] = str(work_dir / "quality_report_post_render.json")
        if not post_report.passed:
            result = _review_result_from_package(
                job_id,
                package,
                subtitle_file_path=str(review_subtitle_file),
            )
            _update_review_job(
                job_id,
                status="needs_review",
                progress=99,
                stage=f"Post-render QA {post_report.overall_score:.1f}/100: chưa xuất bản final",
                result=result,
            )
            return

        output_file = work_dir / "review_final.mp4"
        await asyncio.to_thread(shutil.copy2, draft_file, output_file)
        package.artifact_paths["review_final"] = str(output_file)

        result = _review_result_from_package(
            job_id,
            package,
            subtitle_file_path=str(review_subtitle_file),
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
        if job_id in review_draft_jobs:
            _update_review_job(
                job_id,
                status="failed",
                stage="Tao review phim that bai",
                error=str(exc),
            )


async def _render_existing_review_job(job_id: str) -> None:
    from app.models.job import VoiceGender
    from app.services.ai.review_analysis import (
        rescale_edl_voice_timeline,
        verify_rendered_review,
    )
    from app.services.ai.voice import get_voice_engine
    from app.services.media.ffmpeg import find_ffmpeg, probe_video_duration
    from app.services.media.review_renderer import render_movie_review_video, write_review_subtitles

    job = review_draft_jobs.get(job_id)
    if not job or not job.result or not job.result.quality_report:
        return
    try:
        source_video = _resolve_video_path(job.request.video_path)
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("Không tìm thấy FFmpeg.")
        result = job.result
        package = VerifiedReviewPackage(
            title=result.title,
            target_minutes=result.target_minutes,
            hook=result.hook,
            summary=result.summary,
            narration_script=result.narration_script,
            thumbnail_text=result.thumbnail_text,
            tags=result.tags,
            scenes=result.scenes,
            events=result.events,
            narration_segments=result.narration_segments,
            edit_decision_list=result.edit_decision_list,
            quality_report=result.quality_report,
            artifact_paths=result.artifact_paths,
        )
        # Segment edits are verified and re-scored by patch_review_segment.
        # Re-running the multimodal judge for every segment here made an
        # already-approved job non-deterministic and doubled the AI latency.
        # Keep the saved pre-render gate, then run independent QA on the actual
        # rendered frames below.
        if not package.quality_report.passed:
            _update_review_job(
                job_id,
                status="needs_review",
                progress=82,
                stage=f"Pre-render QA {package.quality_report.overall_score:.1f}/100: cần duyệt lại",
                result=_review_result_from_package(job_id, package),
            )
            return
        work_dir = Path(settings.storage_dir) / "review_jobs" / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        _update_review_job(job_id, status="processing", progress=84, stage="Tạo lại giọng đọc sau khi duyệt")
        narration_audio = await get_voice_engine().synthesize(
            package.narration_script,
            work_dir / "review_narration.mp3",
            VoiceGender.female,
            lambda stage, percent: _update_review_job(job_id, stage=stage, progress=max(84, min(90, percent))),
        )
        duration = await probe_video_duration(ffmpeg, narration_audio)
        if duration <= 0:
            raise RuntimeError("Không đo được thời lượng voice review.")
        package.edit_decision_list = rescale_edl_voice_timeline(package.edit_decision_list, duration)
        (work_dir / "edit_decision_list.json").write_text(
            json.dumps(
                [item.model_dump(mode="json") for item in package.edit_decision_list],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        scene_hints = _review_scene_hints(package.edit_decision_list)
        subtitle_file = write_review_subtitles(
            package.narration_script,
            work_dir / "review_subtitles.srt",
            job.request.target_minutes,
            scene_hints,
            target_seconds=duration,
        )
        draft_file = work_dir / "review_draft_manual.mp4"
        _update_review_job(job_id, progress=90, stage="Render bản đã duyệt theo EDL")
        await render_movie_review_video(
            ffmpeg,
            source_video,
            narration_audio,
            subtitle_file,
            draft_file,
            work_dir,
            job.request.target_minutes,
            scene_hints,
            lambda percent: _update_review_job(job_id, progress=max(90, min(98, percent))),
            _build_review_render_request(job.request, source_video),
        )
        post_report = await verify_rendered_review(draft_file, package, work_dir)
        package.quality_report = post_report
        package.artifact_paths["review_draft"] = str(draft_file)
        package.artifact_paths["post_render_quality"] = str(work_dir / "quality_report_post_render.json")
        if not post_report.passed:
            _update_review_job(
                job_id,
                status="needs_review",
                progress=99,
                stage=f"Post-render QA {post_report.overall_score:.1f}/100: cần duyệt lại",
                result=_review_result_from_package(job_id, package, subtitle_file_path=str(subtitle_file)),
            )
            return
        final_file = work_dir / "review_final.mp4"
        await asyncio.to_thread(shutil.copy2, draft_file, final_file)
        package.artifact_paths["review_final"] = str(final_file)
        _update_review_job(
            job_id,
            status="completed",
            progress=100,
            stage="Hoàn tất video review đã qua QA",
            error=None,
            result=_review_result_from_package(
                job_id,
                package,
                subtitle_file_path=str(subtitle_file),
                output_file_path=str(final_file),
            ),
        )
    except Exception as exc:
        if job_id in review_draft_jobs:
            _update_review_job(job_id, status="failed", stage="Render bản review đã duyệt thất bại", error=str(exc))


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
    _save_review_jobs()
    asyncio.create_task(_process_review_draft_job(job.job_id))
    return job


@router.get("/review/jobs", response_model=list[ReviewDraftJob], tags=["review"])
async def list_review_draft_jobs() -> list[ReviewDraftJob]:
    return sorted(review_draft_jobs.values(), key=lambda item: item.created_at, reverse=True)


@router.delete("/review/jobs", tags=["review"])
async def clear_review_draft_jobs() -> dict[str, str]:
    warnings = _trash_review_jobs()
    review_draft_jobs.clear()
    _save_review_jobs()

    message = "Đã đưa toàn bộ file review job vào Thùng rác và xóa lịch sử review."
    if warnings:
        message = f"{message} Một số file chưa chuyển được vì đang được dùng."

    response = {"message": message}
    if warnings:
        response["warnings"] = "\n".join(warnings)
    return response


@router.get("/review/jobs/{job_id}", response_model=ReviewDraftJob, tags=["review"])
async def get_review_draft_job(job_id: str) -> ReviewDraftJob:
    job = review_draft_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Review job not found.")
    return job


@router.post("/review/jobs/{job_id}/render", response_model=ReviewDraftJob, tags=["review"])
async def render_review_draft_job(job_id: str) -> ReviewDraftJob:
    job = review_draft_jobs.get(job_id)
    if not job or not job.result:
        raise HTTPException(status_code=404, detail="Review job not found.")
    if job.status in {"queued", "processing"}:
        raise HTTPException(status_code=409, detail="Review job đang chạy.")
    if not job.result.quality_report or not job.result.quality_report.passed:
        raise HTTPException(status_code=409, detail="QA chưa đạt 90/100 hoặc vẫn còn câu dưới 75%.")
    updated = _update_review_job(job_id, status="processing", progress=83, stage="Đã nhận lệnh render bản duyệt", error=None)
    asyncio.create_task(_render_existing_review_job(job_id))
    return updated


@router.get("/review/jobs/{job_id}/thumbnails/{file_name}", tags=["review"])
async def get_review_scene_thumbnail(job_id: str, file_name: str) -> FileResponse:
    if job_id not in review_draft_jobs:
        raise HTTPException(status_code=404, detail="Review job not found.")
    safe_name = Path(file_name).name
    thumbnail = Path(settings.storage_dir) / "review_jobs" / job_id / "review_scenes" / safe_name
    if not thumbnail.exists() or thumbnail.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=404, detail="Scene thumbnail not found.")
    return FileResponse(thumbnail, media_type="image/jpeg")


@router.patch(
    "/review/jobs/{job_id}/segments/{segment_id}",
    response_model=ReviewDraftJob,
    tags=["review"],
)
async def patch_review_segment(
    job_id: str,
    segment_id: str,
    request: ReviewSegmentPatchRequest,
) -> ReviewDraftJob:
    job = review_draft_jobs.get(job_id)
    if not job or not job.result:
        raise HTTPException(status_code=404, detail="Review segment not found.")
    if job.status in {"queued", "processing"}:
        raise HTTPException(status_code=409, detail="Job đang xử lý; hãy đợi render xong rồi mới sửa.")
    if not any(
        value is not None
        for value in (
            request.narration,
            request.candidate_id,
            request.scene_id,
            request.start_seconds,
            request.end_seconds,
        )
    ):
        raise HTTPException(status_code=400, detail="Không có thay đổi nào để lưu.")

    result = job.result.model_copy(deep=True)
    beat = next((item for item in result.beats if item.segment_id == segment_id), None)
    decision = next((item for item in result.edit_decision_list if item.segment_id == segment_id), None)
    segment = next((item for item in result.narration_segments if item.segment_id == segment_id), None)
    if beat is None or decision is None:
        raise HTTPException(status_code=404, detail="Review segment not found.")

    narration = (request.narration or "").strip() if request.narration is not None else None
    if request.narration is not None and not narration:
        raise HTTPException(status_code=400, detail="Lời review không được để trống.")
    if narration is not None:
        from app.services.ai.review_analysis import verify_edited_narration

        linked_event = next((item for item in result.events if item.event_id == decision.event_id), None)
        if linked_event is None:
            raise HTTPException(status_code=409, detail="Segment không còn event bằng chứng để kiểm chứng.")
        valid, reason, required_visuals = await verify_edited_narration(
            narration,
            linked_event,
            result.scenes,
        )
        if not valid:
            raise HTTPException(status_code=422, detail=f"Câu sửa không khớp bằng chứng: {reason}")
        beat.narration = narration
        decision.narration = narration
        if segment:
            segment.narration = narration
            if required_visuals is not None:
                # A manual narration edit changes what the selected frame must
                # prove. Keeping the old requirements causes correct replacement
                # scenes to be rejected for details that are no longer narrated.
                segment.required_visuals = required_visuals

    chosen: SceneCandidate | None = None
    if request.candidate_id or request.scene_id:
        chosen = next(
            (
                item
                for item in decision.alternatives
                if (request.candidate_id and item.candidate_id == request.candidate_id)
                or (request.scene_id and item.scene_id == request.scene_id)
            ),
            None,
        )
        if chosen is None:
            raise HTTPException(status_code=400, detail="Cảnh thay thế không thuộc segment này.")
    elif request.start_seconds is not None or request.end_seconds is not None:
        chosen = decision.source_clips[0].model_copy(deep=True) if decision.source_clips else None
    elif narration is not None:
        # A narration edit changes the visual claim even when the user keeps
        # the same scene. Re-score that scene now so the saved QA result is the
        # one the render command can rely on deterministically.
        chosen = decision.source_clips[0].model_copy(deep=True) if decision.source_clips else None

    if chosen is not None:
        if request.start_seconds is not None:
            chosen.start_seconds = request.start_seconds
        if request.end_seconds is not None:
            chosen.end_seconds = request.end_seconds
        if chosen.end_seconds <= chosen.start_seconds:
            raise HTTPException(status_code=400, detail="Timestamp kết thúc phải lớn hơn bắt đầu.")
        decision.selected_candidate_id = chosen.candidate_id
        decision.source_clips = [chosen]
        beat.selected_candidate_id = chosen.candidate_id
        beat.scene_id = chosen.scene_id
        beat.start_seconds = chosen.start_seconds
        beat.end_seconds = chosen.end_seconds
        beat.time_hint = _format_review_time_hint(chosen.start_seconds, chosen.end_seconds)
        beat.match_score = chosen.match_score
        beat.match_reason = chosen.match_reason
        beat.thumbnail_url = chosen.thumbnail_url

        if segment is not None:
            from app.services.ai.review_analysis import _multimodal_rescore_selected_clips

            rescored = await _multimodal_rescore_selected_clips(
                [segment],
                result.scenes,
                [decision],
            )
            if rescored and rescored[0].source_clips:
                decision_index = next(
                    index
                    for index, item in enumerate(result.edit_decision_list)
                    if item.segment_id == segment_id
                )
                decision = rescored[0]
                result.edit_decision_list[decision_index] = decision
                chosen = decision.source_clips[0]
                beat.selected_candidate_id = chosen.candidate_id
                beat.scene_id = chosen.scene_id
                beat.start_seconds = chosen.start_seconds
                beat.end_seconds = chosen.end_seconds
                beat.time_hint = _format_review_time_hint(chosen.start_seconds, chosen.end_seconds)
                beat.match_score = chosen.match_score
                beat.match_reason = chosen.match_reason
                beat.thumbnail_url = chosen.thumbnail_url
                beat.candidates = decision.alternatives

    result.narration_script = " ".join(item.narration.strip() for item in result.beats if item.narration.strip())
    result.quality_report = _recalculate_review_quality(result)
    # Any accepted edit invalidates the previously rendered file. Keep the
    # file on disk for recovery, but never expose it as the output of the new
    # script/EDL; the user must render this revision again.
    result.output_file_path = None
    result.output_video_url = None
    for stale_artifact in ("review_final", "review_draft", "post_render_quality"):
        result.artifact_paths.pop(stale_artifact, None)
    work_dir = Path(settings.storage_dir) / "review_jobs" / job_id
    if work_dir.exists():
        (work_dir / "verified_review_script.json").write_text(
            json.dumps([item.model_dump(mode="json") for item in result.narration_segments], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "edit_decision_list.json").write_text(
            json.dumps([item.model_dump(mode="json") for item in result.edit_decision_list], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if result.quality_report:
            (work_dir / "quality_report.json").write_text(
                result.quality_report.model_dump_json(indent=2),
                encoding="utf-8",
            )

    next_status = "ready_to_render" if result.quality_report and result.quality_report.passed else "needs_review"
    next_stage = "Đã lưu chỉnh sửa; sẵn sàng render" if next_status == "ready_to_render" else "Đã lưu; còn câu cần duyệt"
    return _update_review_job(
        job_id,
        status=next_status,
        progress=82,
        stage=next_stage,
        result=result,
    )


def _recalculate_review_quality(result: ReviewDraftResult) -> ReviewQualityReport:
    previous = result.quality_report or ReviewQualityReport()
    scores = [item.match_score for item in result.beats if item.match_score is not None]
    direct = 100.0 * sum(score >= 0.75 for score in scores) / max(len(result.beats), 1)
    chronology_score = 100.0
    overall = (
        direct * 0.45
        + chronology_score * 0.25
        + previous.evidence_score * 0.20
        + previous.character_consistency_score * 0.10
    )
    stale_codes = {"LOW_VISUAL_MATCH", "POST_RENDER_VISUAL_MISMATCH", "VOICE_VIDEO_DRIFT"}
    retained = [issue for issue in previous.issues if issue.code not in stale_codes]
    retained.extend(
        QualityIssue(
            severity="error",
            code="LOW_VISUAL_MATCH",
            message=f"Điểm khớp cảnh {beat.match_score:.2f} dưới 0.75.",
            segment_id=beat.segment_id,
        )
        for beat in result.beats
        if beat.match_score is not None and beat.match_score < 0.75
    )
    passed = overall >= settings.review_quality_threshold and not any(issue.severity == "error" for issue in retained)
    return previous.model_copy(
        update={
            "phase": "pre_render",
            "overall_score": round(overall, 2),
            "direct_visual_match_percent": round(direct, 2),
            "chronology_score": chronology_score,
            "passed": passed,
            "issues": retained,
        }
    )


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


@router.post("/jobs/{job_id}/retry", response_model=JobCreateResponse, tags=["jobs"])
async def retry_job(
    job_id: str,
    auto_detect_blur: bool = False,
    blur_box_y_percent: int | None = None,
    blur_box_height_percent: int | None = None,
    subtitle_y_percent: int | None = None,
) -> JobCreateResponse:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status in {JobStatus.queued, JobStatus.processing}:
        raise HTTPException(status_code=409, detail="Job is already running.")

    request_updates = {
        key: value
        for key, value in {
            "blur_box_y_percent": blur_box_y_percent,
            "blur_box_height_percent": blur_box_height_percent,
            "subtitle_y_percent": subtitle_y_percent,
        }.items()
        if value is not None
    }
    if auto_detect_blur:
        from app.services.ai.detect_regions import auto_detect_blur_regions_ai, review_and_build_blur_config

        source_path = Path(job.request.local_file_path or "")
        if not source_path.exists():
            source_path = Path(settings.storage_dir) / "jobs" / job_id / "source.mp4"
        if not source_path.exists():
            raise HTTPException(status_code=404, detail="Không tìm thấy video nguồn để AI nhận diện lại vùng mờ.")
        regions = await auto_detect_blur_regions_ai(str(source_path), detect_sub=True, detect_logo=True)
        if not regions:
            raise HTTPException(status_code=502, detail="Gemini không trả về vùng mờ hợp lệ; chưa chạy lại job.")
        checked = review_and_build_blur_config(regions)
        blur_config = checked["config"]
        safe_custom_boxes = [
            box
            for box in blur_config["custom_blur_boxes"]
            if box.get("y_percent", 100) <= 15
            or box.get("y_percent", 0) + box.get("height_percent", 0) >= 85
        ]
        request_updates.update(
            {
                "blur_box_enabled": blur_config["blur_box_enabled"],
                "blur_box_y_percent": blur_config["blur_box_y_percent"],
                "blur_box_height_percent": blur_config["blur_box_height_percent"],
                "custom_blur_boxes": safe_custom_boxes,
                "subtitle_y_percent": blur_config["subtitle_y_percent"],
            }
        )
    request = DubbingRequest.model_validate({**job.request.model_dump(), **request_updates})
    job = job_store.update(
        job_id,
        request=request,
        status=JobStatus.queued,
        stage="Đang chạy lại từ dữ liệu đã xử lý",
        error=None,
        output_video_url=None,
        output_file_path=None,
        completed_at=None,
    )
    import asyncio
    task = asyncio.create_task(process_job(job_id))
    task_manager.register_task(job_id, task)
    return JobCreateResponse(job_id=job_id, status=job.status)


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
        engine_used = "local"
        ai_error: str | None = None
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
                ai_error = str(e).strip() or type(e).__name__
                regions = []
            if regions:
                engine_used = "ai"
            else:
                if ai_error is None:
                    ai_error = "Gemini did not return any valid detection regions."
                engine_used = "local_fallback"
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
            "engine_requested": req.engine,
            "engine_used": engine_used,
            "ai_model": settings.ai_model if req.engine == "ai" else None,
            "ai_error": ai_error,
            "auto_logo": auto_logo_data,
            "config": checked["config"],
            "review": checked["review"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích video: {str(e)}")
