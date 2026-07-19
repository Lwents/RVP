from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.models.review import (
    AnalyzedScene,
    EditDecision,
    NarrationSegment,
    ReviewEvidence,
    ReviewQualityReport,
    SceneCandidate,
    StoryEvent,
    VerifiedReviewPackage,
)
from app.services.ai.final_review_evaluator import (
    _build_evaluation,
    _optional_score,
    _read_word_timings,
    _subtitle_sync_metrics,
    evaluate_final_review_video,
)
from app.services.ai.review_analysis import verify_rendered_review
from app.services.subtitles.timing import SubtitleEvent


def _package() -> VerifiedReviewPackage:
    scene = AnalyzedScene(
        scene_id="scene_0001",
        start_time=0,
        end_time=60,
        visible_actions=["Nhân vật mở cánh cửa"],
        event_summary="Nhân vật mở cánh cửa.",
        evidence=ReviewEvidence(
            visual=["Nhân vật mở cánh cửa"],
            dialogue=["Tôi sẽ mở cửa."],
        ),
    )
    event = StoryEvent(
        event_id="event_0001",
        order_index=1,
        start_time=0,
        end_time=60,
        scene_ids=[scene.scene_id],
        summary="Nhân vật mở cánh cửa.",
        evidence_count=2,
        confidence=0.95,
        verification_status="verified",
    )
    segment = NarrationSegment(
        segment_id="segment_0001",
        event_id=event.event_id,
        narration="Nhân vật mở cánh cửa.",
        candidate_scene_ids=[scene.scene_id],
    )
    candidate = SceneCandidate(
        candidate_id="candidate_0001",
        scene_id=scene.scene_id,
        start_seconds=0,
        end_seconds=2,
        match_score=0.95,
    )
    decision = EditDecision(
        segment_id=segment.segment_id,
        event_id=event.event_id,
        narration=segment.narration,
        voice_start=0,
        voice_end=60,
        selected_candidate_id=candidate.candidate_id,
        source_clips=[candidate],
        alternatives=[candidate],
    )
    report = ReviewQualityReport(
        phase="post_render",
        overall_score=96,
        direct_visual_match_percent=100,
        chronology_score=100,
        evidence_score=95,
        character_consistency_score=100,
        duration_adherence_score=100,
        story_coherence_score=94,
        source_coverage_score=95,
        style_adherence_score=92,
        passed=True,
    )
    return VerifiedReviewPackage(
        title="Review",
        target_minutes=1,
        hook="Hook",
        summary="Summary",
        narration_script=segment.narration,
        thumbnail_text="Review",
        scenes=[scene],
        events=[event],
        narration_segments=[segment],
        edit_decision_list=[decision],
        quality_report=report,
    )


def _healthy_signals() -> dict:
    return {
        "media_duration_seconds": 60.0,
        "expected_voice_duration_seconds": 60.0,
        "voice_video_drift_seconds": 0.0,
        "video_file_size_bytes": 1024,
        "post_render_passed": True,
        "post_render_issue_codes": [],
        "technical_duration_score": 100.0,
        "technical_health_score": 100.0,
        "subtitle_cue_count": 1,
        "subtitle_text_match_percent": 100.0,
        "subtitle_sync_score": 98.0,
        "subtitle_mean_offset_seconds": 0.03,
        "subtitle_max_offset_seconds": 0.08,
        "subtitle_out_of_bounds_cues": 0,
        "subtitle_timeline_coverage_percent": 70.0,
        "subtitle_sync_source": "word_timings",
    }


class FinalReviewEvaluatorTests(unittest.IsolatedAsyncioTestCase):
    def test_one_point_ai_score_stays_one_out_of_one_hundred(self) -> None:
        self.assertEqual(_optional_score(1), 1.0)

    def test_warning_preview_cannot_pass_final_evaluation(self) -> None:
        signals = _healthy_signals()
        signals["post_render_passed"] = False
        criteria = {
            key: {"score": 100, "feedback": "Tốt", "findings": []}
            for key in (
                "content_fidelity",
                "translation_accuracy",
                "av_subtitle_sync",
                "narrative_coherence",
                "technical_quality",
                "safety_compliance",
            )
        }

        evaluation = _build_evaluation(
            {"summary": "Video tốt.", "criteria": criteria},
            {key: 100.0 for key in criteria},
            signals,
            fallback_used=False,
        )

        self.assertFalse(evaluation.passed)
        self.assertEqual(evaluation.verdict, "needs_improvement")
        self.assertIn("chưa đạt post-render QA", evaluation.summary)
        self.assertTrue(any("post-render QA" in item for item in evaluation.recommendations))

    def test_word_timings_scale_after_audio_tempo_change(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "review_word_timings.json"
            path.write_text(
                json.dumps(
                    {
                        "audio_duration": 4.0,
                        "words": [
                            {"text": "Xin", "start": 0.4, "end": 1.0},
                            {"text": "chào", "start": 1.2, "end": 2.0},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            timings = _read_word_timings(path, target_duration=2.0)

        self.assertEqual(timings[0]["start"], 0.2)
        self.assertEqual(timings[1]["end"], 1.0)

    def test_subtitle_metric_detects_cues_that_lag_behind_voice(self) -> None:
        package = _package()
        package.narration_script = "Xin chào"
        package.edit_decision_list[0].narration = "Xin chào"
        words = [
            {"text": "Xin", "start": 0.10, "end": 0.35},
            {"text": "chào", "start": 0.40, "end": 0.75},
        ]
        aligned = [SubtitleEvent(0.02, 0.93, "Xin chào")]
        lagged = [SubtitleEvent(1.02, 1.93, "Xin chào")]

        aligned_metrics = _subtitle_sync_metrics(aligned, words, package, 2.0)
        lagged_metrics = _subtitle_sync_metrics(lagged, words, package, 2.0)

        self.assertGreater(aligned_metrics["subtitle_sync_score"], 95)
        self.assertLess(lagged_metrics["subtitle_sync_score"], 70)
        self.assertGreater(lagged_metrics["subtitle_mean_offset_seconds"], 0.8)

    async def test_ai_failure_returns_complete_fallback_and_writes_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            work_dir = Path(raw_dir)
            video = work_dir / "review_final.mp4"
            video.write_bytes(b"not-a-real-video")
            with (
                patch(
                    "app.services.ai.final_review_evaluator._collect_final_signals",
                    new=AsyncMock(return_value=_healthy_signals()),
                ),
                patch(
                    "app.services.ai.final_review_evaluator._ask_final_judge",
                    new=AsyncMock(side_effect=RuntimeError("router offline")),
                ),
            ):
                evaluation = await evaluate_final_review_video(
                    video,
                    _package(),
                    work_dir,
                )

            artifact = work_dir / "final_evaluation.json"
            self.assertTrue(artifact.is_file())
            self.assertTrue(evaluation.fallback_used)
            self.assertEqual(len(evaluation.criteria), 6)
            self.assertEqual(
                {item.key for item in evaluation.criteria},
                {
                    "content_fidelity",
                    "translation_accuracy",
                    "av_subtitle_sync",
                    "narrative_coherence",
                    "technical_quality",
                    "safety_compliance",
                },
            )
            self.assertEqual(sum(item.weight_percent for item in evaluation.criteria), 100)
            self.assertEqual(
                json.loads(artifact.read_text(encoding="utf-8"))["phase"],
                "final_evaluation",
            )

    async def test_empty_ai_criteria_is_marked_as_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            work_dir = Path(raw_dir)
            video = work_dir / "review_final.mp4"
            video.write_bytes(b"video")
            with (
                patch(
                    "app.services.ai.final_review_evaluator._collect_final_signals",
                    new=AsyncMock(return_value=_healthy_signals()),
                ),
                patch(
                    "app.services.ai.final_review_evaluator._ask_final_judge",
                    new=AsyncMock(return_value={"criteria": []}),
                ),
            ):
                evaluation = await evaluate_final_review_video(
                    video,
                    _package(),
                    work_dir,
                )

        self.assertTrue(evaluation.fallback_used)
        self.assertEqual(len(evaluation.criteria), 6)

    async def test_missing_direct_match_uses_returned_score(self) -> None:
        package = _package()
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "segments": [
                                    {
                                        "segment_id": "segment_0001",
                                        "score": 0.82,
                                        "reason": "Frame minh họa đúng hành động.",
                                    }
                                ]
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            ]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(return_value=response))
            )
        )
        with tempfile.TemporaryDirectory() as raw_dir:
            work_dir = Path(raw_dir)
            frame = work_dir / "frame.jpg"
            frame.write_bytes(b"frame")
            with (
                patch(
                    "app.services.ai.review_analysis._extract_render_qa_frames",
                    return_value=({"segment_0001": [frame]}, 60.0, set()),
                ),
                patch(
                    "app.services.ai.review_analysis.probe_video_duration",
                    new=AsyncMock(return_value=60.0),
                ),
                patch("app.services.ai.review_analysis._client", return_value=client),
            ):
                report = await verify_rendered_review(
                    work_dir / "review.mp4",
                    package,
                    work_dir,
                )

        self.assertEqual(report.direct_visual_match_percent, 100.0)
        self.assertFalse(
            any(issue.code == "POST_RENDER_VISUAL_MISMATCH" for issue in report.issues)
        )


if __name__ == "__main__":
    unittest.main()
