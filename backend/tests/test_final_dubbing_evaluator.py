from __future__ import annotations

import json
import tempfile
import unittest
import wave
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.job import BgmMode, DubbingRequest, JobProgress, JobStatus
from app.services.ai.final_dubbing_evaluator import (
    _aligned_semantic_pairs,
    _build_dubbing_evaluation,
    _collect_dubbing_signals,
    _source_target_timeline_score,
    _tts_segment_metrics,
    dubbing_evaluation_stage,
    emergency_dubbing_evaluation,
    evaluate_final_dubbing_video,
)
from app.services.subtitles.timing import SubtitleEvent


def _job(job_id: str, source: Path, output: Path) -> JobProgress:
    now = datetime.now(UTC)
    return JobProgress(
        job_id=job_id,
        status=JobStatus.completed,
        progress=100,
        stage="Hoàn tất",
        request=DubbingRequest(
            local_file_path=str(source),
            hard_subtitles=True,
            bgm_mode=BgmMode.none,
            use_demucs=False,
        ),
        created_at=now,
        updated_at=now,
        completed_at=now,
        output_file_path=str(output),
        output_video_url=f"/api/jobs/{job_id}/download",
    )


def _healthy_signals() -> dict:
    return {
        "output_file_size_bytes": 1024,
        "output_duration_seconds": 10.0,
        "source_duration_seconds": 10.0,
        "expected_output_duration_seconds": 10.0,
        "output_duration_drift_seconds": 0.0,
        "narration_duration_seconds": 10.0,
        "rendered_narration_drift_seconds": 0.0,
        "output_media_ok": True,
        "output_has_video_stream": True,
        "output_has_audio_stream": True,
        "subtitles_expected": True,
        "narration_expected": True,
        "target_subtitle_burned": True,
        "target_cue_count": 2,
        "source_cue_count": 1,
        "subtitle_out_of_bounds_cues": 0,
        "subtitle_overlap_count": 0,
        "subtitle_empty_cues": 0,
        "subtitle_timeline_coverage_percent": 50.0,
        "subtitle_mean_chars_per_second": 10.0,
        "subtitle_max_chars_per_second": 12.0,
        "subtitle_readability_score": 100.0,
        "source_target_timeline_score": 100.0,
        "source_target_mean_offset_seconds": 0.0,
        "subtitle_sync_score": 100.0,
        "subtitle_sync_source": "per_cue_tts_contract",
        "tts_segment_count": 2,
        "tts_expected_segment_count": 2,
        "tts_cue_coverage_percent": 100.0,
        "tts_manifest_matched": True,
        "tts_overrun_segment_count": 0,
        "voice_duration_score": 100.0,
        "technical_duration_score": 100.0,
        "translation_required": True,
        "source_target_text_identical": False,
        "semantic_comparison_available": True,
        "unavailable_reasons": [],
    }


def _write_wav(path: Path, duration: float, sample_rate: int = 8000) -> None:
    frames = max(1, round(duration * sample_rate))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)


class FinalDubbingEvaluatorTests(unittest.IsolatedAsyncioTestCase):
    def test_changed_cue_segmentation_aligns_by_time_not_index(self) -> None:
        source = [SubtitleEvent(0.0, 4.0, "Một câu nguồn dài")]
        target = [
            SubtitleEvent(0.0, 1.8, "Nửa đầu"),
            SubtitleEvent(1.8, 4.0, "nửa sau"),
        ]

        score, mean_offset = _source_target_timeline_score(source, target)
        pairs = _aligned_semantic_pairs(source, target)

        self.assertGreater(score, 99.0)
        self.assertEqual(mean_offset, 0.0)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["translation"], "Nửa đầu nửa sau")

    async def test_ai_failure_returns_six_criterion_fallback_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            work_dir = Path(raw_dir)
            video = work_dir / "output_hardsub.mp4"
            video.write_bytes(b"video")
            with (
                patch(
                    "app.services.ai.final_dubbing_evaluator._collect_dubbing_signals",
                    new=AsyncMock(return_value=_healthy_signals()),
                ),
                patch(
                    "app.services.ai.final_dubbing_evaluator._ask_dubbing_judge",
                    new=AsyncMock(side_effect=RuntimeError("router offline")),
                ),
            ):
                evaluation = await evaluate_final_dubbing_video(video, work_dir)

            self.assertTrue(evaluation.fallback_used)
            self.assertFalse(evaluation.passed)
            self.assertLess(evaluation.overall_score, 80.0)
            self.assertIn("Chấm dự phòng", dubbing_evaluation_stage(evaluation))
            self.assertEqual(len(evaluation.criteria), 6)
            self.assertLessEqual(
                next(item.score for item in evaluation.criteria if item.key == "translation_accuracy"),
                60.0,
            )
            artifact = work_dir / "final_evaluation.json"
            self.assertTrue(artifact.is_file())
            self.assertEqual(json.loads(artifact.read_text(encoding="utf-8"))["phase"], "final_evaluation")

    def test_legacy_or_overlong_tts_segments_are_not_verified_per_cue(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            work_dir = Path(raw_dir)
            aligned = work_dir / "tts_aligned"
            aligned.mkdir()
            events = [
                SubtitleEvent(0.0, 1.0, "Câu một."),
                SubtitleEvent(1.0, 2.0, "Câu hai."),
            ]
            # First cue has a 0.95s speech slot because it touches cue two.
            # A 1.20s legacy speech artifact proves it can overrun that slot.
            _write_wav(aligned / "0001_speech.wav", 1.20)
            _write_wav(aligned / "0002_speech.wav", 1.00)

            count, expected, coverage, matched, overruns = _tts_segment_metrics(
                work_dir,
                events,
                narration_duration=2.0,
            )

        self.assertEqual((count, expected), (2, 2))
        self.assertFalse(matched)
        self.assertEqual(overruns, 1)
        self.assertLessEqual(coverage, 50.0)

        signals = _healthy_signals()
        signals.update(
            {
                "subtitle_sync_score": 70.0,
                "subtitle_sync_source": "legacy_or_unverified_tts",
                "tts_manifest_matched": False,
                "tts_cue_coverage_percent": coverage,
                "tts_overrun_segment_count": overruns,
                "unavailable_reasons": ["Artifact TTS per-cue không khớp."],
            }
        )
        evaluation = _build_dubbing_evaluation(
            {},
            {key: 100.0 for key in (
                "content_fidelity",
                "translation_accuracy",
                "av_subtitle_sync",
                "narrative_coherence",
                "technical_quality",
                "safety_compliance",
            )},
            signals,
            fallback_used=True,
        )
        sync = next(item for item in evaluation.criteria if item.key == "av_subtitle_sync")
        self.assertLessEqual(sync.score, 70.0)
        self.assertIn("chưa xác minh", sync.feedback)
        self.assertIn(f"{sync.score:.1f}/100", sync.feedback)

    async def test_identical_auto_language_transcript_is_still_flagged_for_translation(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            work_dir = Path(raw_dir)
            video = work_dir / "output.mp4"
            source_video = work_dir / "source.mp4"
            narration = work_dir / "narration_timed.wav"
            subtitle = work_dir / "subtitles.vi.srt"
            for path in (video, source_video, narration):
                path.write_bytes(b"media")
            subtitle.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nOriginal text\n\n"
                "2\n00:00:01,000 --> 00:00:02,000\nStill original\n\n",
                encoding="utf-8",
            )
            aligned = work_dir / "tts_aligned"
            aligned.mkdir()
            _write_wav(aligned / "0001_speech.wav", 0.95)
            _write_wav(aligned / "0002_speech.wav", 1.00)
            with (
                patch(
                    "app.services.ai.final_dubbing_evaluator._probe_duration",
                    new=AsyncMock(return_value=2.0),
                ),
                patch(
                    "app.services.ai.final_dubbing_evaluator._probe_streams",
                    new=AsyncMock(return_value=(True, True)),
                ),
            ):
                signals = await _collect_dubbing_signals(
                    video,
                    work_dir,
                    subtitle,
                    subtitle,
                    narration,
                    source_video,
                    subtitles_expected=True,
                    narration_expected=True,
                    target_subtitle_burned=True,
                    translation_expected=True,
                    source_language="auto",
                    target_language="vi",
                    video_speed=1.0,
                )

        self.assertTrue(signals["translation_required"])
        self.assertTrue(signals["source_target_text_identical"])
        self.assertTrue(any("giống hệt" in item for item in signals["unavailable_reasons"]))

    async def test_missing_old_job_artifacts_have_explicit_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            work_dir = Path(raw_dir)
            video = work_dir / "output_hardsub.mp4"
            video.write_bytes(b"video")

            async def duration(_ffmpeg: str, path: Path | None) -> float:
                return 10.0 if path == video else 0.0

            with patch(
                "app.services.ai.final_dubbing_evaluator._probe_duration",
                new=AsyncMock(side_effect=duration),
            ):
                evaluation = await evaluate_final_dubbing_video(
                    video,
                    work_dir,
                    subtitles_expected=True,
                    narration_expected=True,
                )

        self.assertTrue(evaluation.fallback_used)
        self.assertFalse(evaluation.passed)
        joined = " ".join(evaluation.recommendations)
        self.assertIn("Thiếu phụ đề đích", joined)
        self.assertIn("narration_timed.wav", joined)
        self.assertEqual(len(evaluation.criteria), 6)

    def test_passthrough_job_marks_language_and_sync_criteria_not_applicable(self) -> None:
        signals = _healthy_signals()
        signals.update(
            {
                "subtitles_expected": False,
                "narration_expected": False,
                "target_cue_count": 0,
                "source_cue_count": 0,
                "translation_required": False,
                "subtitle_sync_source": "not_applicable",
                "unavailable_reasons": [],
            }
        )
        baselines = {
            "content_fidelity": 100.0,
            "translation_accuracy": 100.0,
            "av_subtitle_sync": 100.0,
            "narrative_coherence": 100.0,
            "technical_quality": 100.0,
            "safety_compliance": 100.0,
        }

        evaluation = _build_dubbing_evaluation({}, baselines, signals, fallback_used=True)

        sync = next(item for item in evaluation.criteria if item.key == "av_subtitle_sync")
        translation = next(item for item in evaluation.criteria if item.key == "translation_accuracy")
        self.assertIn("không áp dụng", sync.feedback)
        self.assertIn("không phát sinh phép dịch", translation.findings[0])


class FinalDubbingPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_pipeline_still_completes_when_evaluator_itself_crashes(self) -> None:
        from app.pipeline.dubbing import process_dubbing_job

        with tempfile.TemporaryDirectory() as raw_dir:
            storage_dir = Path(raw_dir)
            source = storage_dir / "input.mp4"
            source.write_bytes(b"source")
            now = datetime.now(UTC)
            job = JobProgress(
                job_id="pipeline-job",
                status=JobStatus.queued,
                progress=0,
                stage="Đã nhận cấu hình",
                request=DubbingRequest(
                    local_file_path=str(source),
                    hard_subtitles=False,
                    bgm_mode=BgmMode.none,
                    use_demucs=False,
                    logo_enabled=False,
                ),
                created_at=now,
                updated_at=now,
            )
            store = MagicMock()
            store.get.return_value = job

            async def render(*args, **_kwargs) -> None:
                Path(args[2]).write_bytes(b"rendered")

            with (
                patch("app.pipeline.dubbing.job_store", store),
                patch("app.pipeline.dubbing.settings.storage_dir", str(storage_dir)),
                patch("app.pipeline.dubbing.settings.ai_engine", "passthrough"),
                patch("app.pipeline.dubbing.find_ffmpeg", return_value="ffmpeg"),
                patch(
                    "app.pipeline.dubbing.prepare_source_video",
                    new=AsyncMock(return_value=source),
                ),
                patch("app.pipeline.dubbing.extract_audio", new=AsyncMock()),
                patch(
                    "app.pipeline.dubbing.probe_video_duration",
                    new=AsyncMock(return_value=10.0),
                ),
                patch(
                    "app.pipeline.dubbing.render_video",
                    new=AsyncMock(side_effect=render),
                ),
                patch(
                    "app.services.ai.final_dubbing_evaluator.evaluate_final_dubbing_video",
                    new=AsyncMock(side_effect=RuntimeError("unexpected evaluator bug")),
                ),
                patch("app.services.task_manager.unregister_task"),
            ):
                await process_dubbing_job(job.job_id)

        completed = [
            call.kwargs
            for call in store.update.call_args_list
            if call.kwargs.get("status") == JobStatus.completed
        ]
        failed = [
            call.kwargs
            for call in store.update.call_args_list
            if call.kwargs.get("status") == JobStatus.failed
        ]
        self.assertEqual(len(completed), 1)
        self.assertFalse(failed)
        self.assertTrue(completed[0]["final_evaluation"].fallback_used)

    async def test_manual_evaluate_endpoint_persists_scorecard_without_rerender(self) -> None:
        from app.api.routes import evaluate_job_output

        with tempfile.TemporaryDirectory() as raw_dir:
            storage_dir = Path(raw_dir)
            work_dir = storage_dir / "jobs" / "old-job"
            work_dir.mkdir(parents=True)
            source = work_dir / "source.mp4"
            output = work_dir / "output_hardsub.mp4"
            source.write_bytes(b"source")
            output.write_bytes(b"output")
            evaluation = emergency_dubbing_evaluation("AI offline")
            job = _job("old-job", source, output)
            updated = job.model_copy(
                update={
                    "stage": f"Hoàn tất pipeline · AI chấm {evaluation.overall_score:.1f}/100",
                    "final_evaluation": evaluation,
                }
            )
            store = MagicMock()
            store.get.return_value = job
            store.update.return_value = updated

            with (
                patch("app.api.routes.job_store", store),
                patch("app.api.routes.settings.storage_dir", str(storage_dir)),
                patch(
                    "app.services.ai.final_dubbing_evaluator.evaluate_final_dubbing_video",
                    new=AsyncMock(return_value=evaluation),
                ) as evaluator,
            ):
                result = await evaluate_job_output("old-job")

        evaluator.assert_awaited_once()
        self.assertIsNotNone(result.final_evaluation)
        self.assertEqual(result.status, JobStatus.completed)
        self.assertIn("final_evaluation", store.update.call_args.kwargs)

    def test_job_store_round_trip_keeps_final_evaluation(self) -> None:
        import app.services.store as store_module

        with tempfile.TemporaryDirectory() as raw_dir:
            source = Path(raw_dir) / "source.mp4"
            output = Path(raw_dir) / "output.mp4"
            source.write_bytes(b"source")
            output.write_bytes(b"output")
            evaluation = emergency_dubbing_evaluation("fallback")
            with patch.object(store_module.settings, "storage_dir", raw_dir):
                store = store_module.JobStore()
                created = store.create(DubbingRequest(local_file_path=str(source)))
                store.update(
                    created.job_id,
                    status=JobStatus.completed,
                    progress=100,
                    output_file_path=str(output),
                    final_evaluation=evaluation,
                    completed_at=datetime.now(UTC),
                )
                reloaded = store_module.JobStore().get(created.job_id)

        self.assertIsNotNone(reloaded)
        assert reloaded is not None
        self.assertIsNotNone(reloaded.final_evaluation)
        self.assertEqual(reloaded.final_evaluation.phase, "final_evaluation")


if __name__ == "__main__":
    unittest.main()
