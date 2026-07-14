from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from app.api import routes
from app.api.routes import ReviewDraftJob, ReviewDraftRequest, ReviewRetryRequest, _review_package_progress
from app.core import settings
from app.models.job import DubbingRequest
from app.services.ai.translation import GeminiTranslation, TranslationError
from app.services.presets import get_processing_profile
from app.services.subtitles.source import translation_progress_percent
from app.services.subtitles.timing import SubtitleEvent


class _CheckpointCompletions:
    def __init__(self, fail_from_call: int | None = None) -> None:
        self.fail_from_call = fail_from_call
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_from_call is not None and len(self.calls) >= self.fail_from_call:
            raise TimeoutError("upstream stalled")
        payload = json.loads(kwargs["messages"][1]["content"])
        items = [
            {"id": item["id"], "text": f"Câu dịch {item['id']}"}
            for item in payload["current_items"]
        ]
        content = json.dumps({"items": items, "memory_updates": {}}, ensure_ascii=False)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _CheckpointClient:
    def __init__(self, fail_from_call: int | None = None) -> None:
        self.chat = SimpleNamespace(
            completions=_CheckpointCompletions(fail_from_call=fail_from_call)
        )


class ProcessingModeTests(unittest.IsolatedAsyncioTestCase):
    def test_request_defaults_and_profiles(self) -> None:
        dubbing = DubbingRequest(local_file_path="source.mp4")
        review = ReviewDraftRequest(video_path="source.mp4")
        self.assertEqual(dubbing.processing_mode, "balanced")
        self.assertEqual(review.processing_mode, "balanced")

        fast = get_processing_profile("fast")
        balanced = get_processing_profile("balanced")
        quality = get_processing_profile("quality")
        self.assertLess(fast.whisper_beam_size, balanced.whisper_beam_size)
        self.assertLess(balanced.whisper_beam_size, quality.whisper_beam_size)
        self.assertLess(fast.review_max_scenes, balanced.review_max_scenes)
        self.assertLess(balanced.review_max_scenes, quality.review_max_scenes)
        self.assertFalse(fast.use_demucs)
        self.assertTrue(balanced.use_demucs)
        self.assertTrue(quality.use_demucs)

        with self.assertRaises(ValidationError):
            ReviewDraftRequest(video_path="source.mp4", processing_mode="best")

    def test_review_translation_progress_is_real_and_monotonic(self) -> None:
        values = [
            translation_progress_percent(done, 20, 70, 80)
            for done in (0, 5, 10, 15, 20)
        ]
        self.assertEqual(values, [70, 72, 75, 77, 80])
        self.assertEqual(_review_package_progress(53), 80)
        self.assertEqual(_review_package_progress(79), 83)

    async def test_review_retry_keeps_job_and_updates_requested_mode(self) -> None:
        job_id = "retry-checkpoint-job"
        now = datetime.now(UTC)
        job = ReviewDraftJob(
            job_id=job_id,
            status="failed",
            progress=68,
            stage="Dịch thất bại",
            request=ReviewDraftRequest(video_path="source.mp4"),
            created_at=now,
            updated_at=now,
            error="timeout",
        )
        routes.review_draft_jobs[job_id] = job
        captured: dict = {}

        def update(existing_job_id: str, **changes):
            self.assertEqual(existing_job_id, job_id)
            captured.update(changes)
            return job.model_copy(update=changes)

        def close_coroutine(existing_job_id: str, coroutine) -> None:
            self.assertEqual(existing_job_id, job_id)
            coroutine.close()

        try:
            with patch.object(routes, "_update_review_job", side_effect=update), patch.object(
                routes,
                "_start_review_task",
                side_effect=close_coroutine,
            ):
                retried = await routes.retry_review_draft_job(
                    job_id,
                    ReviewRetryRequest(processing_mode="fast"),
                )
        finally:
            routes.review_draft_jobs.pop(job_id, None)

        self.assertEqual(retried.job_id, job_id)
        self.assertEqual(captured["request"].processing_mode, "fast")
        self.assertEqual(captured["stage"], "Chạy lại từ checkpoint đã lưu")
        self.assertIsNone(captured["result"])

    async def test_translation_resumes_from_last_completed_batch(self) -> None:
        events = [
            SubtitleEvent(float(index), float(index + 1), f"source line {index + 1}")
            for index in range(35)
        ]
        original_key = settings.ninerouter_api_key
        first_client = _CheckpointClient(fail_from_call=2)
        second_client = _CheckpointClient()
        progress: list[tuple[int, int]] = []
        try:
            settings.ninerouter_api_key = "test-key"
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "translation_checkpoint.json"
                with patch("openai.AsyncOpenAI", return_value=first_client), patch(
                    "app.services.ai.translation.asyncio.sleep",
                    new=AsyncMock(),
                ):
                    with self.assertRaises(TranslationError):
                        await GeminiTranslation().translate_events(
                            events,
                            "en",
                            "vi",
                            processing_mode="fast",
                            checkpoint_file=checkpoint,
                            on_batch_progress=lambda done, total: progress.append((done, total)),
                        )

                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                self.assertEqual(saved["translated_items"], 32)
                self.assertFalse(saved["completed"])

                with patch("openai.AsyncOpenAI", return_value=second_client):
                    translated = await GeminiTranslation().translate_events(
                        events,
                        "en",
                        "vi",
                        processing_mode="fast",
                        checkpoint_file=checkpoint,
                        on_batch_progress=lambda done, total: progress.append((done, total)),
                    )

                self.assertEqual(len(translated), len(events))
                self.assertEqual(len(second_client.chat.completions.calls), 1)
                resumed_payload = json.loads(
                    second_client.chat.completions.calls[0]["messages"][1]["content"]
                )
                self.assertEqual(resumed_payload["current_items"][0]["id"], 33)
                self.assertEqual(progress[-1], (2, 2))
        finally:
            settings.ninerouter_api_key = original_key


if __name__ == "__main__":
    unittest.main()
