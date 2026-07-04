from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator

LanguageOption = Literal["auto", "en", "zh", "vi"]


class VoiceGender(str, Enum):
    female = "female"
    male = "male"


class BgmMode(str, Enum):
    demucs = "demucs"
    ducking = "ducking"
    none = "none"


class LogoPosition(str, Enum):
    top_right = "top_right"
    top_left = "top_left"
    bottom_right = "bottom_right"
    bottom_left = "bottom_left"


class PublishTarget(str, Enum):
    youtube = "youtube"
    facebook = "facebook"


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class DubbingRequest(BaseModel):
    source_url: HttpUrl | None = None
    local_file_path: str | None = Field(default=None, max_length=500)
    voice_gender: VoiceGender = VoiceGender.female
    bgm_mode: BgmMode = BgmMode.demucs
    auto_publish: list[PublishTarget] = Field(default_factory=list)
    clone_voice: bool = False
    hard_subtitles: bool = True
    source_has_hard_subtitles: bool = False
    subtitle_x_percent: int = Field(default=50, ge=0, le=100)
    subtitle_y_percent: int = Field(default=78, ge=8, le=94)
    subtitle_font_size: int = Field(default=32, ge=16, le=72)
    subtitle_box_enabled: bool = True
    subtitle_box_opacity: int = Field(default=55, ge=0, le=100)
    subtitle_box_height_percent: int = Field(default=20, ge=8, le=45)
    source_language: LanguageOption = "auto"
    ducking_volume_db: int = Field(default=-12, ge=-36, le=0)
    logo_position: LogoPosition = LogoPosition.top_right
    logo_width: int = Field(default=150, ge=32, le=800)
    watermark_file_name: str | None = None

    @model_validator(mode="after")
    def validate_input_source(self) -> "DubbingRequest":
        if not self.source_url and not self.local_file_path:
            raise ValueError("Provide either a source URL or a local file path.")
        return self


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobProgress(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    stage: str
    request: DubbingRequest
    created_at: datetime
    updated_at: datetime
    output_video_url: str | None = None
    output_file_path: str | None = None
    seo_title: str | None = None
    seo_description: str | None = None
    error: str | None = None


class UploadResponse(BaseModel):
    file_name: str
    content_type: str | None
    size: int
    url: str
    local_file_path: str | None = None
