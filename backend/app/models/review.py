from __future__ import annotations

from pydantic import BaseModel, Field


class ReviewEvidence(BaseModel):
    visual: list[str] = Field(default_factory=list)
    dialogue: list[str] = Field(default_factory=list)
    subtitle: list[str] = Field(default_factory=list)


class ReviewKeyframe(BaseModel):
    time: float
    path: str


class AnalyzedScene(BaseModel):
    scene_id: str
    start_time: float
    end_time: float
    keyframes: list[ReviewKeyframe] = Field(default_factory=list)
    thumbnail_path: str | None = None
    characters: list[str] = Field(default_factory=list)
    location: str = "UNKNOWN"
    visible_actions: list[str] = Field(default_factory=list)
    dialogue_summary: str = ""
    event_summary: str = ""
    important_objects: list[str] = Field(default_factory=list)
    emotion: str = "UNKNOWN"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    temporal_mode: str = "present"
    black_frame: bool = False
    credits: bool = False
    evidence: ReviewEvidence = Field(default_factory=ReviewEvidence)


class StoryEvent(BaseModel):
    event_id: str
    order_index: int
    start_time: float
    end_time: float
    scene_ids: list[str] = Field(default_factory=list)
    characters: list[str] = Field(default_factory=list)
    summary: str
    cause: str = ""
    consequence: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_count: int = 0
    verification_status: str = "unverified"


class RequiredVisuals(BaseModel):
    characters: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)


class NarrationSegment(BaseModel):
    segment_id: str
    event_id: str
    narration: str
    required_visuals: RequiredVisuals = Field(default_factory=RequiredVisuals)
    forbidden_visuals: list[str] = Field(default_factory=list)
    candidate_scene_ids: list[str] = Field(default_factory=list)
    estimated_voice_duration: float = 3.0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    purpose: str = "Minh họa sự kiện đã kiểm chứng"


class SceneCandidate(BaseModel):
    candidate_id: str
    scene_id: str
    start_seconds: float
    end_seconds: float
    thumbnail_path: str | None = None
    thumbnail_url: str | None = None
    match_score: float = Field(default=0.0, ge=0.0, le=1.0)
    match_reason: str = ""


class EditDecision(BaseModel):
    segment_id: str
    event_id: str
    narration: str
    voice_start: float
    voice_end: float
    selected_candidate_id: str
    source_clips: list[SceneCandidate] = Field(default_factory=list)
    alternatives: list[SceneCandidate] = Field(default_factory=list)


class QualityIssue(BaseModel):
    severity: str
    code: str
    message: str
    segment_id: str | None = None


class ReviewQualityReport(BaseModel):
    phase: str = "pre_render"
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    direct_visual_match_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    chronology_score: float = Field(default=0.0, ge=0.0, le=100.0)
    evidence_score: float = Field(default=0.0, ge=0.0, le=100.0)
    character_consistency_score: float = Field(default=0.0, ge=0.0, le=100.0)
    passed: bool = False
    issues: list[QualityIssue] = Field(default_factory=list)


class VerifiedReviewPackage(BaseModel):
    title: str
    target_minutes: int
    hook: str
    summary: str
    narration_script: str
    thumbnail_text: str
    tags: list[str] = Field(default_factory=list)
    scenes: list[AnalyzedScene] = Field(default_factory=list)
    events: list[StoryEvent] = Field(default_factory=list)
    narration_segments: list[NarrationSegment] = Field(default_factory=list)
    edit_decision_list: list[EditDecision] = Field(default_factory=list)
    quality_report: ReviewQualityReport
    artifact_paths: dict[str, str] = Field(default_factory=dict)
