from __future__ import annotations

from dataclasses import dataclass

from app.models.job import ProcessingMode


@dataclass(frozen=True)
class ProcessingProfile:
    name: ProcessingMode
    whisper_beam_size: int
    whisper_word_timestamps: bool
    translation_batch_items: int
    translation_batch_chars: int
    translation_request_timeout_seconds: float
    translation_retries: int
    translation_batch_budget_seconds: float
    review_max_scenes: int
    review_keyframes_per_scene: int
    review_scene_batch_size: int
    review_ai_concurrency: int
    use_demucs: bool


_PROFILES: dict[ProcessingMode, ProcessingProfile] = {
    "fast": ProcessingProfile(
        name="fast",
        whisper_beam_size=1,
        whisper_word_timestamps=False,
        translation_batch_items=32,
        translation_batch_chars=6000,
        translation_request_timeout_seconds=75,
        translation_retries=2,
        translation_batch_budget_seconds=120,
        review_max_scenes=72,
        review_keyframes_per_scene=1,
        review_scene_batch_size=8,
        review_ai_concurrency=2,
        use_demucs=False,
    ),
    "balanced": ProcessingProfile(
        name="balanced",
        whisper_beam_size=3,
        whisper_word_timestamps=False,
        translation_batch_items=28,
        translation_batch_chars=5400,
        translation_request_timeout_seconds=120,
        translation_retries=3,
        translation_batch_budget_seconds=180,
        review_max_scenes=144,
        review_keyframes_per_scene=2,
        review_scene_batch_size=6,
        review_ai_concurrency=2,
        use_demucs=True,
    ),
    "quality": ProcessingProfile(
        name="quality",
        whisper_beam_size=5,
        whisper_word_timestamps=True,
        translation_batch_items=20,
        translation_batch_chars=4600,
        translation_request_timeout_seconds=180,
        translation_retries=3,
        translation_batch_budget_seconds=360,
        review_max_scenes=240,
        review_keyframes_per_scene=3,
        review_scene_batch_size=4,
        review_ai_concurrency=1,
        use_demucs=True,
    ),
}


def get_processing_profile(value: ProcessingMode | str | None) -> ProcessingProfile:
    normalized = str(value or "balanced").lower()
    # Accept the old internal label if a persisted development job contains it.
    if normalized == "best":
        normalized = "quality"
    if normalized not in _PROFILES:
        normalized = "balanced"
    return _PROFILES[normalized]  # type: ignore[index]
