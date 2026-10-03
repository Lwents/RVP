from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.api.routes import (
    ReviewBeatResponse,
    ReviewDraftResult,
    _apply_review_hard_gate,
    _recalculate_review_quality,
)
from app.models.job import DubbingRequest
from app.models.review import (
    AnalyzedScene,
    EditDecision,
    NarrationSegment,
    QualityIssue,
    ReviewEvidence,
    ReviewKeyframe,
    ReviewQualityReport,
    SceneCandidate,
    StoryEvent,
)
from app.services.ai.review_analysis import (
    NarrativeStyleAssessment,
    _STYLE_ENFORCEMENT,
    _STYLE_GUIDANCE,
    _analyze_scene_batch,
    _canonicalize_narration_names,
    _canonicalize_scene_character_names,
    _canonicalize_story_event_names,
    _event_segment_targets,
    _event_story_roles,
    _is_credits_only_scene,
    _judge_narrative_style,
    _match_segments_to_scenes,
    _multimodal_rescore_selected_clips,
    _narrative_transition_issues,
    _narration_meets_budget,
    _optional_bool,
    _optional_number,
    _parse_narrative_style_assessment,
    _post_render_segment_contract,
    _qa_frame_batches,
    _quality_report,
    _reconcile_character_identities,
    _rebalance_event_timeline,
    _reorder_decisions_to_segment_order,
    _reorder_intra_event_segments_by_verified_time,
    _replace_low_pre_render_matches,
    _repair_visual_atomic_segments,
    _repair_low_visual_narration,
    _review_narration_budget,
    _review_source_coverage_score,
    _resolved_multimodal_score,
    _style_contract,
    _validate_event_timeline,
    _visual_atomicity_issues,
    _visual_atomicity_invalid_indices,
    _write_verified_narration,
    review_duration_adherence,
    review_narration_tempo_factor,
    recenter_edl_on_verified_keyframes,
    replace_failed_decisions_with_alternatives,
    rescale_edl_voice_timeline,
)
from app.services.ai.character_names import build_character_name_registry
from app.services.media.review_renderer import (
    align_review_narration_ranges,
    align_review_subtitle_chunks,
    build_review_scene_plans,
    build_review_scenes,
    review_subtitle_single_line_max_chars,
    write_review_subtitles,
)
from app.services.subtitles.ass import srt_to_positioned_ass
from app.services.subtitles.timing import parse_srt


class ReviewTimelineTests(unittest.TestCase):
    def test_character_aliases_are_normalized_across_scene_event_and_narration(self) -> None:
        registry = build_character_name_registry(
            {
                "film_title": "Doraemon",
                "characters": [
                    {"name": "Gian"},
                    {"name": "Suneo", "aliases": ["Xê-kô"]},
                ],
            }
        )
        scenes = _canonicalize_scene_character_names(
            [
                AnalyzedScene(
                    scene_id="scene_0001",
                    start_time=0,
                    end_time=2,
                    characters=["Jaian", "Xê-kô"],
                    visible_actions=["Gian và Xê-kô chạy trên băng"],
                    event_summary="Gian đỡ Suneo đứng dậy.",
                )
            ],
            registry,
        )
        events = _canonicalize_story_event_names(
            [
                StoryEvent(
                    event_id="event_0001",
                    order_index=1,
                    start_time=0,
                    end_time=2,
                    scene_ids=["scene_0001"],
                    characters=["Gian", "Suneo"],
                    summary="Jaian gọi Xê-kô.",
                )
            ],
            registry,
        )
        segments = _canonicalize_narration_names(
            [
                NarrationSegment(
                    segment_id="segment_0001",
                    event_id="event_0001",
                    narration="Gian và Xê-kô cùng chạy trên băng.",
                )
            ],
            registry,
        )

        self.assertEqual(scenes[0].characters, ["Chaien", "Suneo"])
        self.assertEqual(scenes[0].visible_actions, ["Chaien và Suneo chạy trên băng"])
        self.assertEqual(events[0].characters, ["Chaien", "Suneo"])
        self.assertEqual(events[0].summary, "Chaien gọi Suneo.")
        self.assertEqual(segments[0].narration, "Chaien và Suneo cùng chạy trên băng.")

    def test_optional_bool_parses_string_false_without_python_truthiness(self) -> None:
        self.assertIs(_optional_bool("false"), False)
        self.assertIs(_optional_bool("true"), True)
        self.assertIsNone(_optional_bool("unknown"))

    def test_post_render_retry_keeps_verified_candidate_and_trims_failed_tail(self) -> None:
        scene = AnalyzedScene(
            scene_id="scene_0001",
            start_time=5.0,
            end_time=15.0,
            keyframes=[ReviewKeyframe(time=10.0, path="frame.jpg")],
        )
        segment = NarrationSegment(
            segment_id="segment_0001",
            event_id="event_0001",
            narration="Nobita pulls the lever.",
            candidate_scene_ids=[scene.scene_id],
        )
        selected = SceneCandidate(
            candidate_id="selected",
            scene_id=scene.scene_id,
            start_seconds=8.75,
            end_seconds=11.25,
            match_score=1.0,
            match_reason="Gemini: Keyframe 10.0s directly shows the action.",
        )
        decision = EditDecision(
            segment_id=segment.segment_id,
            event_id=segment.event_id,
            narration=segment.narration,
            voice_start=0.0,
            voice_end=4.0,
            selected_candidate_id=selected.candidate_id,
            source_clips=[selected],
            alternatives=[selected],
        )
        report = ReviewQualityReport(
            phase="post_render",
            issues=[
                QualityIssue(
                    severity="error",
                    code="POST_RENDER_VISUAL_MISMATCH",
                    message="Frame thứ ba chuyển sang hành động khác.",
                    segment_id=segment.segment_id,
                )
            ],
        )

        updated, changed = replace_failed_decisions_with_alternatives(
            [decision], report, [segment], [scene]
        )

        self.assertTrue(changed)
        self.assertEqual(updated[0].selected_candidate_id, selected.candidate_id)
        self.assertEqual(updated[0].source_clips[0].start_seconds, 9.0)
        self.assertEqual(updated[0].source_clips[0].end_seconds, 10.0)
    @staticmethod
    def _story_events(count: int = 5) -> list[StoryEvent]:
        return [
            StoryEvent(
                event_id=f"event_{index + 1:04d}",
                order_index=index + 1,
                start_time=float(index * 10),
                end_time=float(index * 10 + 8),
                scene_ids=[f"scene_{index + 1:04d}"],
                summary=f"Su kien {index + 1}",
                evidence_count=1,
                verification_status="verified",
            )
            for index in range(count)
        ]

    def test_eight_minute_budget_is_a_minimum_runtime_contract(self) -> None:
        budget = _review_narration_budget(8, "story")

        self.assertEqual(budget.target_words, 1640)
        self.assertEqual((budget.min_words, budget.max_words), (1476, 1886))
        self.assertGreaterEqual(budget.target_segments, 60)
        one_minute_script = [
            NarrationSegment(
                segment_id="segment_0001",
                event_id="event_0001",
                narration="Mot cau review rat ngan.",
                story_role="hook",
            )
        ]
        self.assertFalse(_narration_meets_budget(one_minute_script, self._story_events(), budget))

    def test_measured_duration_must_match_requested_minutes(self) -> None:
        exact_score, exact_ok = review_duration_adherence(480.0, 8)
        short_score, short_ok = review_duration_adherence(68.568, 8)

        self.assertEqual(exact_score, 100.0)
        self.assertTrue(exact_ok)
        self.assertLess(short_score, 10.0)
        self.assertFalse(short_ok)

    def test_only_near_target_narration_may_receive_a_safe_tempo_correction(self) -> None:
        self.assertAlmostEqual(review_narration_tempo_factor(550.968, 8) or 0, 1.14785, places=5)
        self.assertIsNone(review_narration_tempo_factor(68.568, 8))
        self.assertIsNone(review_narration_tempo_factor(720, 8))

    def test_post_render_qa_splits_large_ffmpeg_select_expression(self) -> None:
        frame_indices = list(range(213))

        batches = _qa_frame_batches(frame_indices)

        self.assertEqual(len(batches), 7)
        self.assertTrue(all(len(batch) <= 32 for batch in batches))
        self.assertEqual([value for batch in batches for value in batch], frame_indices)

    def test_story_arc_has_hook_context_conflict_climax_and_resolution(self) -> None:
        roles = _event_story_roles(self._story_events())

        self.assertEqual(
            list(roles.values()),
            ["hook", "context", "conflict", "climax", "resolution"],
        )

    def test_unverified_boundary_event_does_not_steal_story_arc_roles(self) -> None:
        events = self._story_events()
        events.insert(
            0,
            StoryEvent(
                event_id="unverified_intro",
                order_index=0,
                start_time=-5,
                end_time=-1,
                scene_ids=["scene_unknown"],
                summary="UNKNOWN",
                verification_status="unverified",
            ),
        )

        roles = _event_story_roles(events)

        self.assertNotIn("unverified_intro", roles)
        self.assertEqual(roles["event_0001"], "hook")
        self.assertEqual(roles["event_0005"], "resolution")

    def test_style_changes_atomic_segment_pacing_without_changing_runtime_words(self) -> None:
        styles = ("story", "fast", "emotional", "funny")
        budgets = {style: _review_narration_budget(8, style) for style in styles}

        self.assertEqual({budget.target_words for budget in budgets.values()}, {1640})
        self.assertGreater(budgets["fast"].target_segments, budgets["funny"].target_segments)
        self.assertGreater(budgets["funny"].target_segments, budgets["story"].target_segments)
        self.assertGreater(budgets["story"].target_segments, budgets["emotional"].target_segments)
        events = self._story_events(10)
        allocations = {
            style: _event_segment_targets(events, budgets[style], style)
            for style in styles
        }
        for style in styles:
            self.assertEqual(sum(allocations[style].values()), budgets[style].target_segments)
            self.assertTrue(all(value >= 1 for value in allocations[style].values()))
        self.assertEqual(len({tuple(values.values()) for values in allocations.values()}), len(styles))

    def test_every_supported_style_has_one_distinct_writer_and_qa_contract(self) -> None:
        styles = {"story", "fast", "emotional", "funny"}

        self.assertEqual(set(_STYLE_GUIDANCE), styles)
        self.assertEqual(set(_STYLE_ENFORCEMENT), styles)
        contracts = {style: _style_contract(style) for style in styles}
        self.assertEqual(len(set(contracts.values())), len(styles))
        for style, contract in contracts.items():
            self.assertIn(_STYLE_GUIDANCE[style], contract)
            self.assertIn(_STYLE_ENFORCEMENT[style], contract)

    def test_narrative_transition_gate_rejects_repeated_sau_do_openings(self) -> None:
        segments = [
            NarrationSegment(
                segment_id=f"segment_{index + 1:04d}",
                event_id=f"event_{index + 1:04d}",
                narration=(
                    f"Sau đó, nhân vật thực hiện hành động thứ {index + 1}."
                    if index < 6
                    else f"Nhân vật thực hiện hành động thứ {index + 1}."
                ),
                sequence_index=index + 1,
            )
            for index in range(8)
        ]

        issues = _narrative_transition_issues(segments)

        self.assertTrue(issues)
        self.assertTrue(any("sau đó" in issue.casefold() for issue in issues))

    def test_narrative_transition_gate_accepts_diverse_connectors_in_source_order(self) -> None:
        narrations = [
            "Nobita phát hiện chiếc nhẫn nằm giữa lớp băng.",
            "Từ dấu vết ấy, Doraemon xác định nơi cả nhóm cần đến.",
            "Thế nhưng, cánh cửa vừa mở thì bão tuyết lập tức ập tới.",
            "Cùng lúc ấy, Shizuka nhìn thấy một lối đi bên vách đá.",
            "Vì con đường chính đã bị chặn, cả nhóm chuyển sang lối nhỏ.",
            "Ở phía bên kia, Carla đang chờ cạnh cỗ máy cổ.",
            "Đến khi nguồn năng lượng sáng lên, bí mật của thành phố mới lộ diện.",
            "Cuối cùng, mọi người phối hợp để ngăn thảm họa.",
        ]
        segments = [
            NarrationSegment(
                segment_id=f"segment_{index + 1:04d}",
                event_id=f"event_{index + 1:04d}",
                narration=narration,
                sequence_index=index + 1,
            )
            for index, narration in enumerate(narrations)
        ]
        original_event_order = [segment.event_id for segment in segments]

        issues = _narrative_transition_issues(segments)

        self.assertEqual(issues, [])
        self.assertEqual([segment.event_id for segment in segments], original_event_order)

    def test_narrative_transition_gate_accepts_connector_free_storytelling(self) -> None:
        segments = [
            NarrationSegment(
                segment_id=f"segment_{index + 1:04d}",
                event_id=f"event_{index + 1:04d}",
                narration=narration,
                sequence_index=index + 1,
            )
            for index, narration in enumerate(
                [
                    "Chiếc nhẫn trong băng khiến Nobita chú ý.",
                    "Doraemon kiểm tra ký hiệu được khắc trên mặt nhẫn.",
                    "Một cánh cửa dẫn cả nhóm tới vùng đất phủ tuyết.",
                    "Cơn bão buộc mọi người trú vào hang đá.",
                    "Carla xuất hiện cùng manh mối về thành phố cổ.",
                    "Nguồn năng lượng dưới lòng đất bắt đầu mất kiểm soát.",
                    "Cả nhóm chia nhau khóa các cỗ máy đang rung chuyển.",
                    "Nỗ lực cuối cùng đã ngăn lớp băng sụp xuống.",
                ]
            )
        ]

        self.assertEqual(_narrative_transition_issues(segments), [])

    def test_narrative_transition_gate_ignores_sau_do_inside_a_sentence(self) -> None:
        segments = [
            NarrationSegment(
                segment_id=f"segment_{index + 1:04d}",
                event_id=f"event_{index + 1:04d}",
                narration=f"Nhân vật thứ {index + 1} kể lại chuyện xảy ra sau đó trong nhật ký.",
                sequence_index=index + 1,
            )
            for index in range(8)
        ]

        self.assertEqual(_narrative_transition_issues(segments), [])

    def test_false_positive_credits_flag_does_not_drop_plot_scene(self) -> None:
        plot_scene = AnalyzedScene(
            scene_id="scene_plot",
            start_time=10,
            end_time=15,
            credits=True,
            visible_actions=["Doraemon opens the gate"],
            event_summary="Doraemon opens the gate and the group escapes.",
            evidence=ReviewEvidence(visual=["open gate"]),
        )
        credits_only = AnalyzedScene(
            scene_id="scene_credits",
            start_time=15,
            end_time=20,
            credits=True,
            event_summary="UNKNOWN",
            evidence=ReviewEvidence(subtitle=["staff credits"]),
        )
        illustrated_credits = AnalyzedScene(
            scene_id="scene_illustrated_credits",
            start_time=20,
            end_time=25,
            credits=True,
            characters=["Doraemon"],
            location="ending card",
            visible_actions=["Characters are shown in static illustrations"],
            event_summary="Ending credits show character illustrations and staff text.",
            evidence=ReviewEvidence(visual=["illustrated staff roll"]),
        )
        plot_from_visual_evidence_only = AnalyzedScene(
            scene_id="scene_visual_only",
            start_time=25,
            end_time=30,
            credits=True,
            event_summary="UNKNOWN",
            evidence=ReviewEvidence(visual=["Doraemon opens the time gate and runs through it"]),
        )
        plot_from_subtitle_evidence_only = AnalyzedScene(
            scene_id="scene_subtitle_only",
            start_time=30,
            end_time=35,
            credits=True,
            event_summary="UNKNOWN",
            evidence=ReviewEvidence(subtitle=["Nobita warns everyone to escape immediately"]),
        )

        self.assertFalse(_is_credits_only_scene(plot_scene))
        self.assertTrue(_is_credits_only_scene(credits_only))
        self.assertTrue(_is_credits_only_scene(illustrated_credits))
        self.assertFalse(_is_credits_only_scene(plot_from_visual_evidence_only))
        self.assertFalse(_is_credits_only_scene(plot_from_subtitle_evidence_only))

    def test_sparse_ai_timeline_is_split_on_real_scene_boundaries(self) -> None:
        scenes = [
            AnalyzedScene(
                scene_id=f"scene_{index + 1:04d}",
                start_time=float(index * 5),
                end_time=float(index * 5 + 4),
                event_summary=f"Action {index + 1}",
                confidence=0.9,
                evidence=ReviewEvidence(visual=[f"action {index + 1}"]),
            )
            for index in range(8)
        ]
        timeline = [
            StoryEvent(
                event_id="event_0001",
                order_index=1,
                start_time=0,
                end_time=39,
                scene_ids=[scene.scene_id for scene in scenes],
                summary="All of act one",
                confidence=0.9,
                evidence_count=8,
                verification_status="verified",
            )
        ]

        balanced = _rebalance_event_timeline(timeline, scenes, target_count=6, min_count=5, max_count=7)

        self.assertEqual(len(balanced), 6)
        self.assertEqual([item.order_index for item in balanced], list(range(1, 7)))
        self.assertEqual(
            {scene_id for event in balanced for scene_id in event.scene_ids},
            {scene.scene_id for scene in scenes},
        )

    def test_evidence_backed_cautious_event_keeps_neutral_scene_facts(self) -> None:
        scene = AnalyzedScene(
            scene_id="scene_0001",
            start_time=10,
            end_time=18,
            event_summary="Nobita nhặt chiếc nhẫn lên khỏi băng.",
            evidence=ReviewEvidence(visual=["Nobita cầm chiếc nhẫn"]),
        )
        raw = [
            {
                "scene_ids": [scene.scene_id],
                "summary": "Một lời suy diễn không chắc chắn.",
                "cause": "Nguyên nhân do AI tự đoán.",
                "consequence": "Hệ quả do AI tự đoán.",
                "verification_status": "unverified",
            }
        ]

        timeline = _validate_event_timeline(raw, [scene])

        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].verification_status, "verified")
        self.assertEqual(timeline[0].summary, scene.event_summary)
        self.assertEqual(timeline[0].cause, "")
        self.assertEqual(timeline[0].consequence, "")

    def test_verified_label_cannot_bless_hallucinated_event_logic(self) -> None:
        scene = AnalyzedScene(
            scene_id="scene_0001",
            start_time=10,
            end_time=18,
            characters=["Nobita"],
            event_summary="Nobita opens a pink door in his bedroom.",
            visible_actions=["Nobita opens the pink door"],
            evidence=ReviewEvidence(visual=["Nobita and a pink door"]),
        )
        raw = [
            {
                "scene_ids": [scene.scene_id],
                "characters": ["Invented Villain"],
                "summary": "An invented villain destroys the city.",
                "cause": "A secret army attacks from space.",
                "consequence": "The entire city disappears.",
                "verification_status": "verified",
            }
        ]

        timeline = _validate_event_timeline(raw, [scene])

        self.assertEqual(timeline[0].summary, scene.event_summary)
        self.assertEqual(timeline[0].characters, ["Nobita"])
        self.assertEqual(timeline[0].cause, "")
        self.assertEqual(timeline[0].consequence, "")

    def test_event_timeline_adds_anchors_across_the_whole_source(self) -> None:
        scenes = [
            AnalyzedScene(
                scene_id=f"scene_{index + 1:04d}",
                start_time=float(index * 10),
                end_time=float(index * 10 + 8),
                event_summary=f"Action {index + 1}",
                confidence=0.9,
                evidence=ReviewEvidence(visual=[f"action {index + 1}"]),
            )
            for index in range(32)
        ]
        front_loaded = [
            StoryEvent(
                event_id=f"event_{index + 1:04d}",
                order_index=index + 1,
                start_time=scenes[index].start_time,
                end_time=scenes[index].end_time,
                scene_ids=[scenes[index].scene_id],
                summary=scenes[index].event_summary,
                evidence_count=1,
                verification_status="verified",
            )
            for index in range(8)
        ]

        balanced = _rebalance_event_timeline(front_loaded, scenes, target_count=8, min_count=7, max_count=9)
        claimed = {scene_id for event in balanced for scene_id in event.scene_ids}

        self.assertGreaterEqual(len(balanced), 8)
        self.assertLessEqual(len(balanced), 9)
        for bucket in (scenes[0:4], scenes[4:8], scenes[8:12], scenes[12:16], scenes[16:20], scenes[20:24], scenes[24:28], scenes[28:32]):
            self.assertTrue(any(scene.scene_id in claimed for scene in bucket))

    def test_target_sized_timeline_still_splits_one_overmerged_act(self) -> None:
        scenes = [
            AnalyzedScene(
                scene_id=f"scene_{index + 1:04d}",
                start_time=float(index * 30),
                end_time=float(index * 30 + 25),
                event_summary=f"Action {index + 1}",
                confidence=0.9,
                evidence=ReviewEvidence(visual=[f"action {index + 1}"]),
            )
            for index in range(12)
        ]
        timeline = [
            StoryEvent(
                event_id="event_0001",
                order_index=1,
                start_time=scenes[0].start_time,
                end_time=scenes[8].end_time,
                scene_ids=[scene.scene_id for scene in scenes[:9]],
                summary="Act one was overmerged",
                evidence_count=9,
                verification_status="verified",
            ),
            *[
                StoryEvent(
                    event_id=f"event_{index + 2:04d}",
                    order_index=index + 2,
                    start_time=scene.start_time,
                    end_time=scene.end_time,
                    scene_ids=[scene.scene_id],
                    summary=scene.event_summary,
                    evidence_count=1,
                    verification_status="verified",
                )
                for index, scene in enumerate(scenes[9:])
            ],
        ]

        balanced = _rebalance_event_timeline(timeline, scenes, target_count=4, min_count=4, max_count=6)

        self.assertGreater(len(balanced), 4)
        self.assertLessEqual(len(balanced), 6)
        self.assertLess(max(len(event.scene_ids) for event in balanced), 9)

    def test_source_coverage_uses_whole_movie_evidence_not_only_generated_events(self) -> None:
        scenes = [
            AnalyzedScene(
                scene_id=f"scene_{index + 1:04d}",
                start_time=float(index * 100),
                end_time=float(index * 100 + 20),
                event_summary=f"Action {index + 1}",
                evidence=ReviewEvidence(visual=[f"action {index + 1}"]),
            )
            for index in range(8)
        ]
        decisions = [
            EditDecision(
                segment_id=f"segment_{index + 1:04d}",
                event_id=f"event_{index + 1:04d}",
                narration="x",
                voice_start=float(index),
                voice_end=float(index + 1),
                selected_candidate_id=f"candidate_{index}",
                source_clips=[
                    SceneCandidate(
                        candidate_id=f"candidate_{index}",
                        scene_id=scenes[index].scene_id,
                        start_seconds=scenes[index].start_time,
                        end_seconds=scenes[index].end_time,
                        match_score=0.9,
                    )
                ],
            )
            for index in range(2)
        ]

        score = _review_source_coverage_score(scenes, decisions, target_minutes=2)

        self.assertLess(score, 50.0)

    def test_quality_gate_reports_segment_missing_from_edl(self) -> None:
        events = self._story_events(2)
        scenes = [
            AnalyzedScene(
                scene_id=event.scene_ids[0],
                start_time=event.start_time,
                end_time=event.end_time,
                event_summary=event.summary,
                visible_actions=[event.summary],
                evidence=ReviewEvidence(visual=[event.summary]),
            )
            for event in events
        ]
        roles = _event_story_roles(events)
        segments = [
            NarrationSegment(
                segment_id=f"segment_{index + 1:04d}",
                event_id=event.event_id,
                narration=" ".join(["tu"] * 98),
                story_role=roles[event.event_id],
            )
            for index, event in enumerate(events)
        ]
        decision = EditDecision(
            segment_id=segments[0].segment_id,
            event_id=events[0].event_id,
            narration=segments[0].narration,
            voice_start=0,
            voice_end=30,
            selected_candidate_id="candidate_1",
            source_clips=[
                SceneCandidate(
                    candidate_id="candidate_1",
                    scene_id=scenes[0].scene_id,
                    start_seconds=scenes[0].start_time,
                    end_seconds=scenes[0].end_time,
                    match_score=0.9,
                )
            ],
        )

        report = _quality_report(segments, events, scenes, [decision], target_minutes=1, style="story")

        self.assertEqual(
            [issue.segment_id for issue in report.issues if issue.code == "NO_CLIP"],
            [segments[1].segment_id],
        )

    def test_narrative_judge_parser_caps_logic_when_contradictions_exist(self) -> None:
        assessment = _parse_narrative_style_assessment(
            {
                "coherence_score": 96,
                "style_score": 91,
                "contradictions": ["Nobita is credited with Doraemon's action"],
                "early_spoilers": [],
                "feedback": "Fix actor attribution.",
            }
        )

        self.assertEqual(assessment.coherence_score, 84.0)
        self.assertEqual(assessment.style_score, 91.0)
        self.assertFalse(assessment.logic_passed)
        self.assertTrue(assessment.style_passed)

    def test_quality_gate_blocks_bad_narrative_logic_and_style(self) -> None:
        events = self._story_events()
        roles = _event_story_roles(events)
        scenes: list[AnalyzedScene] = []
        segments: list[NarrationSegment] = []
        decisions: list[EditDecision] = []
        for index, event in enumerate(events):
            scene = AnalyzedScene(
                scene_id=event.scene_ids[0],
                start_time=event.start_time,
                end_time=event.end_time,
                event_summary=event.summary,
                visible_actions=[event.summary],
                evidence=ReviewEvidence(visual=[event.summary]),
            )
            scenes.append(scene)
            segment = NarrationSegment(
                segment_id=f"segment_{index + 1:04d}",
                event_id=event.event_id,
                narration=" ".join(["tu"] * 39),
                story_role=roles[event.event_id],
            )
            segments.append(segment)
            candidate = SceneCandidate(
                candidate_id=f"candidate_{index}",
                scene_id=scene.scene_id,
                start_seconds=scene.start_time,
                end_seconds=scene.end_time,
                match_score=0.9,
            )
            decisions.append(
                EditDecision(
                    segment_id=segment.segment_id,
                    event_id=event.event_id,
                    narration=segment.narration,
                    voice_start=float(index),
                    voice_end=float(index + 1),
                    selected_candidate_id=candidate.candidate_id,
                    source_clips=[candidate],
                )
            )
        assessment = NarrativeStyleAssessment(
            coherence_score=72,
            style_score=70,
            contradictions=("Sai người thực hiện hành động",),
            feedback="Chưa đúng phong cách cảm xúc.",
        )

        report = _quality_report(
            segments,
            events,
            scenes,
            decisions,
            target_minutes=1,
            style="emotional",
            narrative_assessment=assessment,
        )

        self.assertTrue(any(issue.code == "NARRATIVE_LOGIC" for issue in report.issues))
        self.assertTrue(any(issue.code == "STYLE_MISMATCH" for issue in report.issues))
        self.assertFalse(report.passed)

    def test_narrative_score_88_is_rejected_when_feedback_still_reports_a_plot_jump(self) -> None:
        assessment = NarrativeStyleAssessment(
            coherence_score=88,
            style_score=95,
            feedback="A transition skips an important event.",
        )

        self.assertFalse(assessment.logic_passed)
        self.assertFalse(assessment.passed)

    def test_scene_matching_never_moves_backwards_in_source_time(self) -> None:
        scenes = [
            AnalyzedScene(
                scene_id=f"scene_{index + 1:04d}",
                start_time=float(index * 10),
                end_time=float(index * 10 + 8),
                visible_actions=[f"action {index + 1}"],
                event_summary=f"action {index + 1}",
                confidence=0.9,
                evidence=ReviewEvidence(visual=[f"action {index + 1}"]),
            )
            for index in range(3)
        ]
        event = StoryEvent(
            event_id="event_0001",
            order_index=1,
            start_time=0,
            end_time=28,
            scene_ids=[scene.scene_id for scene in scenes],
            summary="Three actions",
            evidence_count=3,
            verification_status="verified",
        )
        segments = [
            NarrationSegment(
                segment_id=f"segment_{index + 1:04d}",
                event_id=event.event_id,
                narration=f"action {index + 1}",
                candidate_scene_ids=[scene_id],
            )
            for index, scene_id in enumerate(("scene_0001", "scene_0003", "scene_0002"))
        ]

        decisions = _match_segments_to_scenes(segments, [event], scenes)
        starts = [item.source_clips[0].start_seconds for item in decisions]

        self.assertTrue(all(left <= right + 0.25 for left, right in zip(starts, starts[1:])))

    def test_scene_matching_spreads_one_event_across_its_source_window(self) -> None:
        scenes = [
            AnalyzedScene(
                scene_id=f"scene_{index + 1:04d}",
                start_time=float(index * 10),
                end_time=float(index * 10 + 8),
                event_summary="Cùng một diễn biến",
                confidence=0.9,
                evidence=ReviewEvidence(visual=["same event"]),
            )
            for index in range(4)
        ]
        event = StoryEvent(
            event_id="event_0001",
            order_index=1,
            start_time=0,
            end_time=38,
            scene_ids=[scene.scene_id for scene in scenes],
            summary="Cùng một diễn biến kéo dài",
            evidence_count=4,
            verification_status="verified",
        )
        segments = [
            NarrationSegment(
                segment_id=f"segment_{index + 1:04d}",
                event_id=event.event_id,
                narration=f"Câu {index + 1}",
                candidate_scene_ids=[scene.scene_id for scene in scenes],
            )
            for index in range(4)
        ]

        decisions = _match_segments_to_scenes(segments, [event], scenes)
        selected_ids = [item.source_clips[0].scene_id for item in decisions]

        self.assertEqual(selected_ids, [scene.scene_id for scene in scenes])

    def test_visual_atomicity_requires_one_exact_scene_per_sentence(self) -> None:
        scenes = [
            AnalyzedScene(
                scene_id="scene_0001",
                start_time=0,
                end_time=8,
                characters=["Nobita"],
                visible_actions=["Nobita opens the door"],
                important_objects=["Pink door"],
                location="Bedroom",
            ),
            AnalyzedScene(
                scene_id="scene_0002",
                start_time=10,
                end_time=18,
                characters=["Doraemon"],
                visible_actions=["Doraemon enters the ice field"],
                important_objects=["Ice"],
                location="Ice field",
            ),
        ]
        valid = NarrationSegment(
            segment_id="segment_0001",
            event_id="event_0001",
            narration="Nobita opens the pink door.",
            required_visuals={
                "characters": ["Nobita"],
                "actions": ["Nobita opens the door"],
                "objects": ["Pink door"],
                "locations": ["Bedroom"],
            },
            candidate_scene_ids=["scene_0001"],
            sequence_index=1,
        )
        invalid = valid.model_copy(
            update={
                "segment_id": "segment_0002",
                "candidate_scene_ids": ["scene_0001", "scene_0002"],
                "sequence_index": 2,
            }
        )

        self.assertEqual(_visual_atomicity_issues([valid], scenes), [])
        self.assertTrue(_visual_atomicity_issues([invalid], scenes))

    def test_dialogue_claim_requires_source_subtitle_from_the_same_scene(self) -> None:
        scene = AnalyzedScene(
            scene_id="scene_0001",
            start_time=0,
            end_time=8,
            characters=["Nobita"],
            visible_actions=["Nobita points at the pink door"],
            important_objects=["Pink door"],
            location="Bedroom",
        )
        segment = NarrationSegment(
            segment_id="segment_0001",
            event_id="event_0001",
            narration="Nobita giải thích cách mở cánh cửa màu hồng.",
            required_visuals={
                "characters": ["Nobita"],
                "actions": ["Nobita points at the pink door"],
                "objects": ["Pink door"],
                "locations": ["Bedroom"],
            },
            candidate_scene_ids=[scene.scene_id],
            sequence_index=1,
        )

        issues = _visual_atomicity_issues([segment], [scene])
        self.assertTrue(any("dialogue claim lacks source subtitle" in issue for issue in issues))

        supported = scene.model_copy(
            update={
                "evidence": ReviewEvidence(
                    subtitle=["Nobita: Đây là cách mở cánh cửa màu hồng."]
                )
            }
        )
        supported_issues = _visual_atomicity_issues([segment], [supported])
        self.assertFalse(
            any("dialogue claim lacks source subtitle" in issue for issue in supported_issues)
        )

        selected = SceneCandidate(
            candidate_id="candidate_0001",
            scene_id=scene.scene_id,
            start_seconds=0,
            end_seconds=2,
            match_score=0.9,
        )
        event = StoryEvent(
            event_id=segment.event_id,
            order_index=1,
            start_time=0,
            end_time=8,
            scene_ids=[scene.scene_id],
            summary="Nobita points at the door.",
            evidence_count=1,
            verification_status="verified",
        )
        decision = EditDecision(
            segment_id=segment.segment_id,
            event_id=segment.event_id,
            narration=segment.narration,
            voice_start=0,
            voice_end=2,
            selected_candidate_id=selected.candidate_id,
            source_clips=[selected],
        )
        report = _quality_report([segment], [event], [scene], [decision], target_minutes=1)
        self.assertTrue(any(issue.code == "UNSUPPORTED_NARRATION" for issue in report.issues))
        self.assertFalse(_apply_review_hard_gate(report).passed)

    def test_character_name_on_a_supported_object_is_not_a_missing_actor(self) -> None:
        scenes = [
            AnalyzedScene(
                scene_id="scene_statue",
                start_time=0,
                end_time=8,
                visible_actions=["The group uncovers a frozen Doraemon statue"],
                important_objects=["Frozen Doraemon statue"],
                event_summary="A frozen statue is uncovered.",
            ),
            AnalyzedScene(
                scene_id="scene_real_character",
                start_time=10,
                end_time=18,
                characters=["Doraemon"],
                visible_actions=["Doraemon runs"],
            ),
        ]
        segment = NarrationSegment(
            segment_id="segment_0001",
            event_id="event_0001",
            narration="Cả nhóm phát hiện một bức tượng Doraemon bị đóng băng.",
            required_visuals={
                "characters": [],
                "actions": ["The group uncovers a frozen Doraemon statue"],
                "objects": ["Frozen Doraemon statue"],
                "locations": [],
            },
            candidate_scene_ids=["scene_statue"],
            sequence_index=1,
        )

        self.assertEqual(_visual_atomicity_issues([segment], scenes), [])

    def test_global_atomic_repair_index_detects_a_backwards_candidate(self) -> None:
        scenes = [
            AnalyzedScene(
                scene_id="scene_early",
                start_time=0,
                end_time=8,
                visible_actions=["early action"],
            ),
            AnalyzedScene(
                scene_id="scene_late",
                start_time=20,
                end_time=28,
                visible_actions=["late action"],
            ),
        ]
        late = NarrationSegment(
            segment_id="segment_0001",
            event_id="event_0001",
            narration="Late action.",
            required_visuals={"actions": ["late action"]},
            candidate_scene_ids=["scene_late"],
            sequence_index=1,
        )
        early = NarrationSegment(
            segment_id="segment_0002",
            event_id="event_0001",
            narration="Early action.",
            required_visuals={"actions": ["early action"]},
            candidate_scene_ids=["scene_early"],
            sequence_index=2,
        )

        self.assertEqual(_visual_atomicity_invalid_indices([late, early], scenes), [1])

    def test_atomicity_rejects_adjacent_duplicate_narration_for_same_action(self) -> None:
        scene = AnalyzedScene(
            scene_id="scene_ice",
            start_time=20,
            end_time=28,
            visible_actions=["Nobita scratches his head near a pink door"],
        )
        first = NarrationSegment(
            segment_id="segment_0001",
            event_id="event_0001",
            narration="Nobita gãi đầu cạnh cánh cửa màu hồng trên cánh đồng băng.",
            required_visuals={"actions": ["Nobita scratches his head near a pink door"]},
            candidate_scene_ids=[scene.scene_id],
            sequence_index=1,
        )
        repeated = first.model_copy(
            update={
                "segment_id": "segment_0002",
                "narration": "Sau đó, Nobita vẫn đứng gãi đầu bên cánh cửa hồng trên đồng băng.",
                "sequence_index": 2,
            }
        )

        self.assertTrue(_visual_atomicity_issues([first, repeated], [scene]))
        self.assertEqual(_visual_atomicity_invalid_indices([first, repeated], [scene]), [1])

    def test_verified_keyframe_time_reorders_actions_only_inside_the_same_event(self) -> None:
        segments = [
            NarrationSegment(
                segment_id=f"segment_{index:04d}",
                event_id="event_0001" if index <= 3 else "event_0002",
                narration=text,
                sequence_index=index,
            )
            for index, text in enumerate(("A", "B", "C", "NEXT EVENT"), start=1)
        ]

        def decision(segment: NarrationSegment, source_start: float) -> EditDecision:
            candidate = SceneCandidate(
                candidate_id=f"candidate_{segment.segment_id}",
                scene_id="scene_shared",
                start_seconds=source_start,
                end_seconds=source_start + 2,
                match_score=0.9,
            )
            return EditDecision(
                segment_id=segment.segment_id,
                event_id=segment.event_id,
                narration=segment.narration,
                voice_start=0,
                voice_end=2,
                selected_candidate_id=candidate.candidate_id,
                source_clips=[candidate],
            )

        decisions = [
            decision(segments[0], 60),
            decision(segments[1], 96.58),
            decision(segments[2], 67.42),
            # Even if the next event has an earlier bad timestamp, it must not
            # be moved into event_0001; the quality gate will reject it.
            decision(segments[3], 5),
        ]

        reordered, changed = _reorder_intra_event_segments_by_verified_time(segments, decisions)

        self.assertTrue(changed)
        self.assertEqual([item.narration for item in reordered], ["A", "C", "B", "NEXT EVENT"])
        self.assertEqual([item.sequence_index for item in reordered], [1, 2, 3, 4])
        reordered_decisions = _reorder_decisions_to_segment_order(reordered, decisions)
        self.assertEqual([item.narration for item in reordered_decisions], ["A", "C", "B", "NEXT EVENT"])
        self.assertEqual(
            [item.source_clips[0].start_seconds for item in reordered_decisions],
            [60, 67.42, 96.58, 5],
        )
        self.assertEqual(
            [item.voice_start for item in reordered_decisions],
            sorted(item.voice_start for item in reordered_decisions),
        )

    def test_scene_matching_honors_atomic_primary_candidate(self) -> None:
        scenes = [
            AnalyzedScene(
                scene_id=f"scene_{index + 1:04d}",
                start_time=float(index * 10),
                end_time=float(index * 10 + 8),
                event_summary="Same verified event",
                confidence=0.9,
                evidence=ReviewEvidence(visual=["same event"]),
            )
            for index in range(3)
        ]
        event = StoryEvent(
            event_id="event_0001",
            order_index=1,
            start_time=0,
            end_time=28,
            scene_ids=[scene.scene_id for scene in scenes],
            summary="Same verified event",
            evidence_count=3,
            verification_status="verified",
        )
        segment = NarrationSegment(
            segment_id="segment_0001",
            event_id=event.event_id,
            narration="The middle scene is the direct evidence.",
            candidate_scene_ids=["scene_0002"],
        )

        decision = _match_segments_to_scenes([segment], [event], scenes)[0]

        self.assertEqual(decision.source_clips[0].scene_id, "scene_0002")
        self.assertEqual(decision.alternatives[0].scene_id, "scene_0002")

    def test_visual_recovery_never_selects_a_backward_alternative(self) -> None:
        def candidate(candidate_id: str, start: float, score: float) -> SceneCandidate:
            return SceneCandidate(
                candidate_id=candidate_id,
                scene_id=candidate_id,
                start_seconds=start,
                end_seconds=start + 2,
                match_score=score,
            )

        first = candidate("first", 10, 0.9)
        current = candidate("current", 20, 0.4)
        backward = candidate("backward", 5, 0.99)
        chronological = candidate("chronological", 25, 0.9)
        last = candidate("last", 30, 0.9)
        decisions = [
            EditDecision(
                segment_id="segment_0001",
                event_id="event_0001",
                narration="first",
                voice_start=0,
                voice_end=1,
                selected_candidate_id=first.candidate_id,
                source_clips=[first],
                alternatives=[first],
            ),
            EditDecision(
                segment_id="segment_0002",
                event_id="event_0002",
                narration="middle",
                voice_start=1,
                voice_end=2,
                selected_candidate_id=current.candidate_id,
                source_clips=[current],
                alternatives=[current, backward, chronological],
            ),
            EditDecision(
                segment_id="segment_0003",
                event_id="event_0003",
                narration="last",
                voice_start=2,
                voice_end=3,
                selected_candidate_id=last.candidate_id,
                source_clips=[last],
                alternatives=[last],
            ),
        ]

        updated, changed = _replace_low_pre_render_matches(decisions)

        self.assertTrue(changed)
        self.assertEqual(updated[1].source_clips[0].candidate_id, "chronological")

    def test_visual_recovery_does_not_swap_in_a_scene_that_cannot_show_the_sentence(self) -> None:
        scenes = [
            AnalyzedScene(
                scene_id="scene_battery",
                start_time=10,
                end_time=18,
                characters=["Nobita"],
                visible_actions=["Nobita inserts the battery"],
                important_objects=["Battery"],
            ),
            AnalyzedScene(
                scene_id="scene_belt",
                start_time=20,
                end_time=28,
                characters=["Nobita"],
                visible_actions=["Nobita holds the belt"],
                important_objects=["Belt"],
            ),
        ]
        segment = NarrationSegment(
            segment_id="segment_0001",
            event_id="event_0001",
            narration="Nobita lắp viên pin vào chiếc thắt lưng.",
            required_visuals={
                "characters": ["Nobita"],
                "actions": ["Nobita inserts the battery"],
                "objects": ["Battery"],
                "locations": [],
            },
            candidate_scene_ids=["scene_battery"],
        )
        current = SceneCandidate(
            candidate_id="current",
            scene_id="scene_battery",
            start_seconds=10,
            end_seconds=12,
            match_score=0.3,
        )
        unsupported = SceneCandidate(
            candidate_id="unsupported",
            scene_id="scene_belt",
            start_seconds=20,
            end_seconds=22,
            match_score=0.95,
        )
        decision = EditDecision(
            segment_id=segment.segment_id,
            event_id=segment.event_id,
            narration=segment.narration,
            voice_start=0,
            voice_end=2,
            selected_candidate_id=current.candidate_id,
            source_clips=[current],
            alternatives=[current, unsupported],
        )

        updated, changed = _replace_low_pre_render_matches([decision], [segment], scenes)

        self.assertFalse(changed)
        self.assertEqual(updated[0].source_clips[0].candidate_id, "current")

    def test_quality_gate_rejects_backward_source_timestamps(self) -> None:
        events = self._story_events(2)
        roles = _event_story_roles(events)
        segments = [
            NarrationSegment(
                segment_id=f"segment_{index + 1:04d}",
                event_id=event.event_id,
                narration=event.summary,
                story_role=roles[event.event_id],
            )
            for index, event in enumerate(events)
        ]
        decisions = [
            EditDecision(
                segment_id=segments[0].segment_id,
                event_id=events[0].event_id,
                narration=segments[0].narration,
                voice_start=0,
                voice_end=2,
                selected_candidate_id="candidate_1",
                source_clips=[
                    SceneCandidate(
                        candidate_id="candidate_1",
                        scene_id="scene_0001",
                        start_seconds=20,
                        end_seconds=22,
                        match_score=0.9,
                    )
                ],
            ),
            EditDecision(
                segment_id=segments[1].segment_id,
                event_id=events[1].event_id,
                narration=segments[1].narration,
                voice_start=2,
                voice_end=4,
                selected_candidate_id="candidate_2",
                source_clips=[
                    SceneCandidate(
                        candidate_id="candidate_2",
                        scene_id="scene_0002",
                        start_seconds=10,
                        end_seconds=12,
                        match_score=0.9,
                    )
                ],
            ),
        ]

        report = _quality_report(segments, events, [], decisions, target_minutes=1, style="story")

        self.assertEqual(report.chronology_score, 60.0)
        self.assertTrue(any(issue.code == "CHRONOLOGY" for issue in report.issues))

    def test_quality_gate_reports_requested_duration_mismatch(self) -> None:
        events = self._story_events()
        segments = [
            NarrationSegment(
                segment_id=f"segment_{index + 1:04d}",
                event_id=event.event_id,
                narration=event.summary,
                story_role=role,
                sequence_index=index + 1,
            )
            for index, (event, role) in enumerate(zip(events, _event_story_roles(events).values()))
        ]

        report = _quality_report(segments, events, [], [], target_minutes=8, style="story")

        self.assertTrue(any(issue.code == "NARRATION_DURATION_MISMATCH" for issue in report.issues))
        self.assertFalse(report.passed)

    def test_voice_timeline_uses_narration_weights_and_ends_at_actual_duration(self) -> None:
        decisions = [
            EditDecision(
                segment_id="segment_0001",
                event_id="event_0001",
                narration="Cau ngan.",
                voice_start=0.0,
                voice_end=5.0,
                selected_candidate_id="candidate_1",
            ),
            EditDecision(
                segment_id="segment_0002",
                event_id="event_0002",
                narration="Day la mot cau dan dai hon rat nhieu so voi cau dau tien.",
                voice_start=5.0,
                voice_end=10.0,
                selected_candidate_id="candidate_2",
            ),
        ]

        scaled = rescale_edl_voice_timeline(decisions, 12.345)

        first_duration = scaled[0].voice_end - scaled[0].voice_start
        second_duration = scaled[1].voice_end - scaled[1].voice_start
        self.assertLess(first_duration, second_duration)
        self.assertEqual(scaled[0].voice_start, 0.0)
        self.assertEqual(scaled[1].voice_start, scaled[0].voice_end)
        self.assertEqual(scaled[-1].voice_end, 12.345)
        self.assertAlmostEqual(
            sum(item.voice_end - item.voice_start for item in scaled),
            12.345,
            places=9,
        )

    def test_edit_clears_only_that_segments_post_render_mismatch(self) -> None:
        report = ReviewQualityReport(
            phase="post_render",
            overall_score=70.0,
            direct_visual_match_percent=50.0,
            chronology_score=100.0,
            evidence_score=100.0,
            character_consistency_score=100.0,
            duration_adherence_score=100.0,
            story_coherence_score=100.0,
            source_coverage_score=100.0,
            style_adherence_score=100.0,
            passed=False,
            issues=[
                QualityIssue(
                    severity="error",
                    code="POST_RENDER_VISUAL_MISMATCH",
                    message="segment one failed",
                    segment_id="segment_0001",
                ),
                QualityIssue(
                    severity="error",
                    code="POST_RENDER_VISUAL_MISMATCH",
                    message="segment two failed",
                    segment_id="segment_0002",
                ),
                QualityIssue(
                    severity="error",
                    code="LOW_VISUAL_MATCH",
                    message="stale pre-render score",
                    segment_id="segment_0001",
                ),
            ],
        )
        result = ReviewDraftResult(
            title="Review",
            target_minutes=1,
            hook="Hook",
            summary="Summary",
            narration_script=" ".join(["tu"] * 195),
            beats=[
                ReviewBeatResponse(
                    time_hint="00:00:00-00:00:02",
                    purpose="test",
                    narration="One.",
                    segment_id="segment_0001",
                    match_score=0.90,
                ),
                ReviewBeatResponse(
                    time_hint="00:00:02-00:00:04",
                    purpose="test",
                    narration="Two.",
                    segment_id="segment_0002",
                    match_score=0.90,
                ),
            ],
            thumbnail_text="Review",
            tags=[],
            quality_report=report,
        )

        after_first_edit = _recalculate_review_quality(
            result,
            edited_segment_id="segment_0001",
        )
        remaining_post_errors = [
            issue.segment_id
            for issue in after_first_edit.issues
            if issue.code == "POST_RENDER_VISUAL_MISMATCH"
        ]
        self.assertEqual(remaining_post_errors, ["segment_0002"])
        self.assertFalse(any(issue.code == "LOW_VISUAL_MATCH" for issue in after_first_edit.issues))
        self.assertFalse(after_first_edit.passed)
        self.assertEqual(after_first_edit.issues[0].severity, "error")

        result.quality_report = after_first_edit
        after_second_edit = _recalculate_review_quality(
            result,
            edited_segment_id="segment_0002",
        )
        self.assertFalse(
            any(issue.code == "POST_RENDER_VISUAL_MISMATCH" for issue in after_second_edit.issues)
        )
        self.assertTrue(after_second_edit.passed)

        result.quality_report = after_second_edit
        result.beats[0].match_score = 0.50
        with_recalculated_low_score = _recalculate_review_quality(
            result,
            edited_segment_id="segment_0001",
        )
        self.assertEqual(
            [
                issue.segment_id
                for issue in with_recalculated_low_score.issues
                if issue.code == "LOW_VISUAL_MATCH"
            ],
            ["segment_0001"],
        )
        self.assertFalse(with_recalculated_low_score.passed)
        self.assertEqual(
            next(issue for issue in with_recalculated_low_score.issues if issue.code == "LOW_VISUAL_MATCH").severity,
            "error",
        )

    def test_multimodal_score_does_not_drift_when_saved_again(self) -> None:
        self.assertEqual(_resolved_multimodal_score(0.20, 0.84), 0.84)
        self.assertEqual(_resolved_multimodal_score(0.73, 0.84), 0.84)

    def test_verified_window_is_preserved_and_slow_fitted_to_full_voice(self) -> None:
        hints = [
            {
                "start_seconds": 7.0,
                "end_seconds": 9.0,
                "voice_start": 0.0,
                "voice_end": 4.5,
                "duration_seconds": 4.5,
                "narration": "Câu dẫn dài hơn shot nguồn.",
            }
        ]
        clips = build_review_scenes(20.0, 4.5, hints)
        plans = build_review_scene_plans(20.0, 4.5, hints)

        self.assertAlmostEqual(sum(duration for _, duration in clips), 4.5, places=3)
        self.assertEqual(len(clips), 1)
        start, duration = clips[0]
        self.assertAlmostEqual(start, 7.0, places=3)
        self.assertAlmostEqual(duration, 4.5, places=3)
        self.assertAlmostEqual(plans[0].source_start, 7.0, places=3)
        self.assertAlmostEqual(plans[0].source_duration, 2.0, places=3)
        self.assertAlmostEqual(plans[0].output_duration, 4.5, places=3)

    def test_continuous_review_clip_stays_inside_source_at_boundaries(self) -> None:
        near_start = build_review_scenes(
            20.0,
            4.0,
            [{"start_seconds": 0.1, "end_seconds": 0.8, "duration_seconds": 4.0}],
        )
        near_end = build_review_scenes(
            20.0,
            4.0,
            [{"start_seconds": 19.3, "end_seconds": 19.9, "duration_seconds": 4.0}],
        )

        self.assertAlmostEqual(near_start[0][0], 0.05, places=3)
        self.assertAlmostEqual(near_start[0][1], 4.0, places=3)
        self.assertAlmostEqual(near_end[0][0], 19.2, places=3)
        self.assertAlmostEqual(near_end[0][1], 4.0, places=3)

        near_start_plan = build_review_scene_plans(
            20.0,
            4.0,
            [{"start_seconds": 0.1, "end_seconds": 0.8, "duration_seconds": 4.0}],
        )[0]
        near_end_plan = build_review_scene_plans(
            20.0,
            4.0,
            [{"start_seconds": 19.3, "end_seconds": 19.9, "duration_seconds": 4.0}],
        )[0]
        self.assertAlmostEqual(near_start_plan.source_duration, 0.8, places=3)
        self.assertAlmostEqual(near_end_plan.source_duration, 0.8, places=3)
        self.assertAlmostEqual(near_end_plan.source_start, 19.2, places=3)

    def test_keyframe_time_accepts_gateway_string_units(self) -> None:
        self.assertEqual(_optional_number("122.85s"), 122.85)
        self.assertEqual(_optional_number("122,85 giay"), 122.85)
        self.assertIsNone(_optional_number("unknown"))

    def test_saved_edl_recenters_from_verified_keyframe_reason(self) -> None:
        scene = AnalyzedScene(
            scene_id="scene_0001",
            start_time=100.0,
            end_time=130.0,
            keyframes=[
                {"time": 104.0, "path": "first.jpg"},
                {"time": 116.0, "path": "middle.jpg"},
                {"time": 122.85, "path": "verified.jpg"},
            ],
        )
        selected = SceneCandidate(
            candidate_id="segment_0001:scene_0001",
            scene_id=scene.scene_id,
            start_seconds=100.15,
            end_seconds=106.15,
            match_score=0.75,
            match_reason="Gemini visual-semantic: Keyframe 122.85s shows the coin.",
        )
        decision = EditDecision(
            segment_id="segment_0001",
            event_id="event_0001",
            narration="The coin spins.",
            voice_start=0.0,
            voice_end=4.0,
            selected_candidate_id=selected.candidate_id,
            source_clips=[selected],
            alternatives=[selected],
        )

        repaired = recenter_edl_on_verified_keyframes([decision], [scene])

        self.assertAlmostEqual(repaired[0].source_clips[0].start_seconds, 121.6, places=3)
        self.assertAlmostEqual(repaired[0].source_clips[0].end_seconds, 124.1, places=3)

    def test_adjacent_identical_verified_windows_are_partitioned_not_replayed(self) -> None:
        hints = [
            {
                "start_seconds": 7.0,
                "end_seconds": 9.0,
                "voice_start": float(index * 4),
                "voice_end": float((index + 1) * 4),
            }
            for index in range(2)
        ]

        plans = build_review_scene_plans(20.0, 8.0, hints)

        self.assertAlmostEqual(plans[0].source_start, 7.0, places=3)
        self.assertAlmostEqual(plans[0].source_duration, 1.0, places=3)
        self.assertAlmostEqual(plans[1].source_start, 8.0, places=3)
        self.assertAlmostEqual(plans[1].source_duration, 1.0, places=3)
        self.assertAlmostEqual(sum(item.output_duration for item in plans), 8.0, places=3)

    def test_render_plan_quantization_cannot_accumulate_frame_drift(self) -> None:
        hints = []
        cursor = 0.0
        for index in range(101):
            duration = 479.981583 / 101
            hints.append(
                {
                    "start_seconds": 10.0 + index * 3.0,
                    "end_seconds": 11.5 + index * 3.0,
                    "voice_start": cursor,
                    "voice_end": cursor + duration,
                }
            )
            cursor += duration

        plans = build_review_scene_plans(1000.0, 479.981583, hints)

        self.assertEqual(sum(round(item.output_duration * 30) for item in plans), 14400)
        self.assertAlmostEqual(sum(item.output_duration for item in plans), 480.0, places=6)

    def test_post_render_contract_excludes_unspoken_context_metadata(self) -> None:
        segment = NarrationSegment(
            segment_id="segment_0001",
            event_id="event_0001",
            narration="Doraemon gọt khối băng thành một chiếc ghế.",
            required_visuals={
                "characters": ["Doraemon", "Nobita"],
                "actions": ["Doraemon shapes ice into a chair"],
                "objects": ["Ice chair", "Yellow penguin gadget"],
                "locations": ["Snowy landscape"],
            },
        )
        decision = EditDecision(
            segment_id=segment.segment_id,
            event_id=segment.event_id,
            narration=segment.narration,
            voice_start=0,
            voice_end=3,
            selected_candidate_id="candidate",
        )

        contract = _post_render_segment_contract(decision, segment)

        self.assertIn("Doraemon shapes ice into a chair", contract)
        self.assertNotIn("Nobita", contract)
        self.assertNotIn("Yellow penguin gadget", contract)

    def test_voice_duration_is_not_stretched_to_requested_minutes(self) -> None:
        hints = []
        cursor = 0.0
        for index in range(20):
            duration = 7.825
            hints.append(
                {
                    "start_seconds": index * 20.0,
                    "end_seconds": index * 20.0 + 12.0,
                    "voice_start": cursor,
                    "voice_end": cursor + duration,
                    "duration_seconds": duration,
                }
            )
            cursor += duration
        clips = build_review_scenes(6052.0, 156.5, hints)

        self.assertAlmostEqual(sum(duration for _, duration in clips), 156.5, places=3)
        self.assertNotAlmostEqual(sum(duration for _, duration in clips), 480.0, places=1)

    def test_subtitles_follow_exact_voice_ranges(self) -> None:
        hints = [
            {
                "narration": "Câu đầu tiên.",
                "voice_start": 0.0,
                "voice_end": 2.4,
            },
            {
                "narration": "Câu thứ hai.",
                "voice_start": 2.4,
                "voice_end": 5.1,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = write_review_subtitles(
                "Câu đầu tiên. Câu thứ hai.",
                Path(directory) / "review.srt",
                target_minutes=8,
                scene_hints=hints,
                target_seconds=5.1,
            )
            events = parse_srt(output)

        self.assertEqual([item.text for item in events], ["Câu đầu tiên.", "Câu thứ hai."])
        self.assertAlmostEqual(events[0].end, 2.4, places=2)
        self.assertAlmostEqual(events[1].start, 2.4, places=2)
        self.assertAlmostEqual(events[-1].end, 5.1, places=2)

    def test_review_subtitles_are_split_before_render_and_never_wrap_to_two_lines(self) -> None:
        narration = (
            "Nobita và Doraemon bước vào căn phòng băng rồi nhìn thấy một cánh cửa màu hồng."
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            srt_file = write_review_subtitles(
                narration,
                root / "review.srt",
                target_minutes=1,
                scene_hints=[
                    {
                        "narration": narration,
                        "voice_start": 0.0,
                        "voice_end": 8.0,
                    }
                ],
                target_seconds=8.0,
                subtitle_font_size=120,
            )
            events = parse_srt(srt_file)
            limit = review_subtitle_single_line_max_chars(120)
            ass_file = srt_to_positioned_ass(
                srt_file,
                root / "review.ass",
                1280,
                720,
                DubbingRequest(local_file_path="source.mp4", subtitle_font_size=120),
                single_line=True,
            )
            ass_text = ass_file.read_text(encoding="utf-8")

        self.assertGreater(len(events), 1)
        self.assertTrue(all("\n" not in event.text and len(event.text) <= limit for event in events))
        self.assertIn("WrapStyle: 2", ass_text)
        dialogue_lines = [line for line in ass_text.splitlines() if line.startswith("Dialogue:")]
        self.assertTrue(dialogue_lines)
        self.assertTrue(all(r"\q2" in line and r"\N" not in line for line in dialogue_lines))

    def test_review_subtitle_chunks_follow_exact_tts_word_boundaries(self) -> None:
        chunks = ["Một câu ngắn.", "Đoạn kế tiếp dài hơn."]
        timings = [
            {"text": "Một", "start": 0.20, "end": 0.38},
            {"text": "câu", "start": 0.42, "end": 0.61},
            {"text": "ngắn.", "start": 0.66, "end": 0.98},
            {"text": "Đoạn", "start": 1.80, "end": 2.02},
            {"text": "kế", "start": 2.08, "end": 2.20},
            {"text": "tiếp", "start": 2.25, "end": 2.43},
            {"text": "dài", "start": 2.48, "end": 2.65},
            {"text": "hơn.", "start": 2.70, "end": 2.98},
        ]

        events = align_review_subtitle_chunks(chunks, timings, target_seconds=3.2)

        self.assertEqual([event.text for event in events], chunks)
        self.assertAlmostEqual(events[0].start, 0.12, places=2)
        self.assertAlmostEqual(events[0].end, 1.16, places=2)
        self.assertAlmostEqual(events[1].start, 1.72, places=2)
        self.assertAlmostEqual(events[1].end, 3.16, places=2)
        self.assertEqual(
            align_review_narration_ranges(chunks, timings, target_seconds=3.2),
            [(0.0, 1.72), (1.72, 3.2)],
        )

    def test_animation_unknown_alias_is_reconciled_by_hair_descriptor(self) -> None:
        scenes = [
            AnalyzedScene(
                scene_id="scene_0001",
                start_time=0,
                end_time=2,
                characters=["Leng Qingcheng (Red-haired woman)"],
                evidence=ReviewEvidence(visual=["red-haired woman"]),
            ),
            AnalyzedScene(
                scene_id="scene_0002",
                start_time=2,
                end_time=4,
                characters=["UNKNOWN_RED_HAIR"],
                visible_actions=["UNKNOWN_RED_HAIR crosses her arms"],
                evidence=ReviewEvidence(visual=["same red-haired woman"]),
            ),
        ]
        reconciled = _reconcile_character_identities(scenes)

        self.assertEqual(reconciled[1].characters, ["Leng Qingcheng (Red-haired woman)"])
        self.assertIn("Leng Qingcheng", reconciled[1].visible_actions[0])

    def test_manual_recalc_never_promotes_missing_new_qa_scores_to_100(self) -> None:
        result = ReviewDraftResult(
            title="Old review",
            target_minutes=1,
            hook="Hook",
            summary="Summary",
            narration_script=" ".join(["tu"] * 195),
            beats=[
                ReviewBeatResponse(
                    time_hint="00:00:00-00:00:02",
                    purpose="test",
                    narration=" ".join(["tu"] * 195),
                    segment_id="segment_0001",
                    match_score=0.9,
                    start_seconds=0,
                    end_seconds=2,
                )
            ],
            thumbnail_text="Review",
            tags=[],
            quality_report=ReviewQualityReport(
                evidence_score=100,
                character_consistency_score=100,
            ),
        )

        recalculated = _recalculate_review_quality(result)

        self.assertEqual(recalculated.story_coherence_score, 0.0)
        self.assertEqual(recalculated.source_coverage_score, 0.0)
        self.assertEqual(recalculated.style_adherence_score, 0.0)
        self.assertTrue(any(issue.code == "NARRATIVE_LOGIC" for issue in recalculated.issues))
        self.assertTrue(any(issue.code == "SOURCE_COVERAGE" for issue in recalculated.issues))
        self.assertTrue(any(issue.code == "STYLE_MISMATCH" for issue in recalculated.issues))
        self.assertFalse(recalculated.passed)

    def test_manual_recalc_updates_source_coverage_after_scene_choice(self) -> None:
        scenes = [
            AnalyzedScene(
                scene_id=f"scene_{index}",
                start_time=float(index * 100),
                end_time=float(index * 100 + 20),
                event_summary=f"event {index}",
                evidence=ReviewEvidence(visual=[f"event {index}"]),
            )
            for index in range(8)
        ]
        decisions = []
        beats = []
        for index in range(2):
            candidate = SceneCandidate(
                candidate_id=f"candidate_{index}",
                scene_id=scenes[index].scene_id,
                start_seconds=scenes[index].start_time,
                end_seconds=scenes[index].end_time,
                match_score=0.9,
            )
            decisions.append(
                EditDecision(
                    segment_id=f"segment_{index}",
                    event_id=f"event_{index}",
                    narration="x",
                    voice_start=float(index),
                    voice_end=float(index + 1),
                    selected_candidate_id=candidate.candidate_id,
                    source_clips=[candidate],
                )
            )
            beats.append(
                ReviewBeatResponse(
                    time_hint="00:00:00-00:00:02",
                    purpose="test",
                    narration="x",
                    segment_id=f"segment_{index}",
                    match_score=0.9,
                    start_seconds=candidate.start_seconds,
                    end_seconds=candidate.end_seconds,
                )
            )
        result = ReviewDraftResult(
            title="Review",
            target_minutes=1,
            hook="Hook",
            summary="Summary",
            narration_script=" ".join(["tu"] * 195),
            beats=beats,
            thumbnail_text="Review",
            tags=[],
            scenes=scenes,
            edit_decision_list=decisions,
            quality_report=ReviewQualityReport(
                evidence_score=100,
                character_consistency_score=100,
                story_coherence_score=100,
                source_coverage_score=100,
                style_adherence_score=100,
            ),
        )

        recalculated = _recalculate_review_quality(result)

        self.assertLess(recalculated.source_coverage_score, 50.0)
        self.assertTrue(any(issue.code == "SOURCE_COVERAGE" for issue in recalculated.issues))


class NarrativeJudgeRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_scene_analysis_preserves_exact_timestamped_source_dialogue(self) -> None:
        source_line = "Nobita: Tớ sẽ mở cánh cửa màu hồng ngay bây giờ."
        scene = AnalyzedScene(
            scene_id="scene_0001",
            start_time=0,
            end_time=8,
            dialogue_summary=source_line,
            evidence=ReviewEvidence(dialogue=[source_line]),
        )
        payload = {
            "scenes": [
                {
                    "scene_id": scene.scene_id,
                    "characters": ["Nobita"],
                    "location": "Bedroom",
                    "visible_actions": ["Nobita reaches for a pink door"],
                    "dialogue_summary": "Nobita announces his next action.",
                    "event_summary": "Nobita reaches for the door.",
                    "important_objects": ["Pink door"],
                    "emotion": "determined",
                    "confidence": 0.9,
                    "temporal_mode": "present",
                    "credits": False,
                    "evidence": {"visual": ["pink door"], "dialogue": [], "subtitle": []},
                }
            ]
        }
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response)))
        )

        with patch("app.services.ai.review_analysis._client", return_value=client):
            analyzed = await _analyze_scene_batch([scene], [])

        self.assertIn(source_line, analyzed[0].evidence.dialogue)

    async def test_style_judge_receives_the_exact_selected_contract_for_all_styles(self) -> None:
        event = ReviewTimelineTests._story_events(1)[0]
        segment = NarrationSegment(
            segment_id="segment_0001",
            event_id=event.event_id,
            narration="Nhan vat thuc hien hanh dong co trong canh.",
            story_role="hook",
            sequence_index=1,
        )
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "coherence_score": 95,
                                "style_score": 95,
                                "contradictions": [],
                                "early_spoilers": [],
                                "feedback": "Dat.",
                            }
                        )
                    )
                )
            ]
        )
        create = AsyncMock(return_value=response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        with patch("app.services.ai.review_analysis._client", return_value=client):
            for style in ("story", "fast", "emotional", "funny"):
                assessment = await _judge_narrative_style([segment], [event], style)
                self.assertTrue(assessment.passed)

        self.assertEqual(create.await_count, 4)
        for style, call in zip(("story", "fast", "emotional", "funny"), create.await_args_list):
            payload = json.loads(call.kwargs["messages"][1]["content"])
            self.assertEqual(payload["selected_style"], style)
            self.assertEqual(payload["style_contract"], _style_contract(style))

    async def test_low_keyframe_match_rewrites_only_that_segment(self) -> None:
        scenes = [
            AnalyzedScene(
                scene_id="scene_0001",
                start_time=0,
                end_time=8,
                characters=["Nobita"],
                visible_actions=["Nobita inserts the battery"],
                important_objects=["Battery"],
                location="Ice room",
            ),
            AnalyzedScene(
                scene_id="scene_0002",
                start_time=9,
                end_time=17,
                characters=["Nobita"],
                visible_actions=["Nobita holds the battery"],
                important_objects=["Battery"],
                location="Ice room",
            ),
        ]
        event = StoryEvent(
            event_id="event_0001",
            order_index=1,
            start_time=0,
            end_time=17,
            scene_ids=[scene.scene_id for scene in scenes],
            summary="Nobita examines the battery.",
            evidence_count=2,
            verification_status="verified",
        )
        segment = NarrationSegment(
            segment_id="segment_0001",
            event_id=event.event_id,
            narration="Nobita carefully inserts the battery into the device.",
            required_visuals={
                "characters": ["Nobita"],
                "actions": ["Nobita inserts the battery"],
                "objects": ["Battery"],
                "locations": ["Ice room"],
            },
            candidate_scene_ids=["scene_0001"],
            sequence_index=1,
        )
        failed = SceneCandidate(
            candidate_id="failed",
            scene_id="scene_0001",
            start_seconds=0,
            end_seconds=2,
            match_score=0.3,
        )
        decision = EditDecision(
            segment_id=segment.segment_id,
            event_id=event.event_id,
            narration=segment.narration,
            voice_start=0,
            voice_end=3,
            selected_candidate_id=failed.candidate_id,
            source_clips=[failed],
            alternatives=[failed],
        )
        payload = {
            "segments": [
                {
                    "segment_id": segment.segment_id,
                    "scene_id": "scene_0002",
                    "narration": "Nobita carefully holds the battery inside the icy room.",
                    "required_visuals": {
                        "characters": ["Nobita"],
                        "actions": ["Nobita holds the battery"],
                        "objects": ["Battery"],
                        "locations": ["Ice room"],
                    },
                    "confidence": 0.9,
                }
            ]
        }
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response)))
        )

        with patch("app.services.ai.review_analysis._client", return_value=client):
            repaired, changed = await _repair_low_visual_narration(
                [segment], [event], scenes, [decision], "story"
            )

        self.assertTrue(changed)
        self.assertEqual(repaired[0].candidate_scene_ids, ["scene_0002"])
        self.assertEqual(repaired[0].required_visuals.actions, ["Nobita holds the battery"])

    async def test_low_visual_repair_processes_more_than_twelve_failed_segments(self) -> None:
        scenes: list[AnalyzedScene] = []
        events: list[StoryEvent] = []
        segments: list[NarrationSegment] = []
        decisions: list[EditDecision] = []
        replacement_by_id: dict[str, dict] = {}
        for index in range(13):
            number = index + 1
            scene_id = f"scene_{number:04d}"
            event_id = f"event_{number:04d}"
            segment_id = f"segment_{number:04d}"
            action = f"Character waves beside marker {number}"
            scene = AnalyzedScene(
                scene_id=scene_id,
                start_time=float(index * 10),
                end_time=float(index * 10 + 8),
                visible_actions=[action],
                location="Open field",
                evidence=ReviewEvidence(visual=[action]),
            )
            event = StoryEvent(
                event_id=event_id,
                order_index=number,
                start_time=scene.start_time,
                end_time=scene.end_time,
                scene_ids=[scene_id],
                summary=action,
                evidence_count=1,
                verification_status="verified",
            )
            segment = NarrationSegment(
                segment_id=segment_id,
                event_id=event_id,
                narration=f"Nhân vật đứng cạnh cột mốc số {number} nhưng hành động chưa rõ ràng.",
                required_visuals={"actions": [action], "locations": ["Open field"]},
                candidate_scene_ids=[scene_id],
                sequence_index=number,
            )
            candidate = SceneCandidate(
                candidate_id=f"candidate_{number:04d}",
                scene_id=scene_id,
                start_seconds=scene.start_time,
                end_seconds=scene.start_time + 2,
                match_score=0.3,
            )
            decision = EditDecision(
                segment_id=segment_id,
                event_id=event_id,
                narration=segment.narration,
                voice_start=float(index * 2),
                voice_end=float(index * 2 + 2),
                selected_candidate_id=candidate.candidate_id,
                source_clips=[candidate],
                alternatives=[candidate],
            )
            replacement_by_id[segment_id] = {
                "segment_id": segment_id,
                "scene_id": scene_id,
                "narration": f"Nhân vật vẫy tay rõ ràng bên cột mốc số {number}.",
                "required_visuals": {
                    "characters": [],
                    "actions": [action],
                    "objects": [],
                    "locations": ["Open field"],
                },
                "confidence": 0.9,
            }
            scenes.append(scene)
            events.append(event)
            segments.append(segment)
            decisions.append(decision)

        async def create_response(**kwargs):
            text_parts = [
                part.get("text", "")
                for part in kwargs["messages"][0]["content"]
                if part.get("type") == "text"
            ]
            requested_ids = re.findall(
                r"SEGMENT (segment_\d+);",
                "\n".join(text_parts),
            )
            payload = {"segments": [replacement_by_id[item] for item in requested_ids]}
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
            )

        completions = SimpleNamespace(create=AsyncMock(side_effect=create_response))
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with patch("app.services.ai.review_analysis._client", return_value=client):
            repaired, changed = await _repair_low_visual_narration(
                segments,
                events,
                scenes,
                decisions,
                "story",
            )

        self.assertTrue(changed)
        self.assertEqual(
            sum(before.narration != after.narration for before, after in zip(segments, repaired)),
            13,
        )
        self.assertEqual(completions.create.await_count, 5)

    async def test_low_keyframe_repair_may_keep_the_scene_and_remove_unsupported_details(self) -> None:
        scene = AnalyzedScene(
            scene_id="scene_0001",
            start_time=0,
            end_time=8,
            characters=["Nobita"],
            visible_actions=["Nobita opens the pink door"],
            important_objects=["Pink door"],
            location="Bedroom",
        )
        event = StoryEvent(
            event_id="event_0001",
            order_index=1,
            start_time=0,
            end_time=8,
            scene_ids=[scene.scene_id],
            summary="Nobita opens the door.",
            evidence_count=1,
            verification_status="verified",
        )
        segment = NarrationSegment(
            segment_id="segment_0001",
            event_id=event.event_id,
            narration="Nobita mở cánh cửa rồi bay qua thành phố băng giá.",
            required_visuals={
                "characters": ["Nobita"],
                "actions": ["Nobita opens the pink door"],
                "objects": ["Pink door"],
                "locations": ["Bedroom"],
            },
            candidate_scene_ids=[scene.scene_id],
            sequence_index=1,
        )
        candidate = SceneCandidate(
            candidate_id="candidate_0001",
            scene_id=scene.scene_id,
            start_seconds=0,
            end_seconds=2,
            match_score=0.3,
        )
        decision = EditDecision(
            segment_id=segment.segment_id,
            event_id=segment.event_id,
            narration=segment.narration,
            voice_start=0,
            voice_end=2,
            selected_candidate_id=candidate.candidate_id,
            source_clips=[candidate],
            alternatives=[candidate],
        )
        payload = {
            "segments": [
                {
                    "segment_id": segment.segment_id,
                    "scene_id": scene.scene_id,
                    "narration": "Nobita chậm rãi mở cánh cửa màu hồng trong phòng ngủ.",
                    "required_visuals": {
                        "characters": ["Nobita"],
                        "actions": ["Nobita opens the pink door"],
                        "objects": ["Pink door"],
                        "locations": ["Bedroom"],
                    },
                    "confidence": 0.9,
                }
            ]
        }
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response)))
        )

        with patch("app.services.ai.review_analysis._client", return_value=client):
            repaired, changed = await _repair_low_visual_narration(
                [segment], [event], [scene], [decision], "story"
            )

        self.assertTrue(changed)
        self.assertEqual(repaired[0].candidate_scene_ids, [scene.scene_id])
        self.assertEqual(repaired[0].required_visuals.actions, segment.required_visuals.actions)
        self.assertNotEqual(repaired[0].narration, segment.narration)

    async def test_visual_atomic_repair_changes_only_invalid_sentence(self) -> None:
        scene = AnalyzedScene(
            scene_id="scene_0001",
            start_time=0,
            end_time=8,
            characters=["Nobita"],
            visible_actions=["Nobita opens the pink door", "Nobita walks outside"],
            important_objects=["Pink door"],
            location="Bedroom",
        )
        event = StoryEvent(
            event_id="event_0001",
            order_index=1,
            start_time=0,
            end_time=8,
            scene_ids=[scene.scene_id],
            summary="Nobita opens the door.",
            evidence_count=1,
            verification_status="verified",
        )
        invalid = NarrationSegment(
            segment_id="segment_0001",
            event_id=event.event_id,
            narration="Nobita opens the pink door and quickly walks outside.",
            required_visuals={
                "characters": ["Nobita"],
                "actions": scene.visible_actions,
                "objects": ["Pink door"],
                "locations": ["Bedroom"],
            },
            candidate_scene_ids=[scene.scene_id],
            sequence_index=1,
            story_role="hook",
        )
        payload = {
            "segments": [
                {
                    "sequence_index": 1,
                    "event_id": event.event_id,
                    "narration": "Nobita slowly opens the pink door with a worried expression.",
                    "required_visuals": {
                        "characters": ["Nobita"],
                        "actions": ["Nobita opens the pink door"],
                        "objects": ["Pink door"],
                        "locations": ["Bedroom"],
                    },
                    "candidate_scene_ids": [scene.scene_id],
                    "confidence": 0.9,
                }
            ]
        }
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response)))
        )

        with patch("app.services.ai.review_analysis._client", return_value=client):
            repaired = await _repair_visual_atomic_segments([invalid], [event], [scene], "story")

        self.assertEqual(repaired[0].segment_id, invalid.segment_id)
        self.assertEqual(repaired[0].required_visuals.actions, ["Nobita opens the pink door"])
        self.assertEqual(repaired[0].candidate_scene_ids, [scene.scene_id])

    async def test_direct_visual_match_boolean_accepts_conservative_raw_score(self) -> None:
        scene = AnalyzedScene(
            scene_id="scene_0001",
            start_time=0,
            end_time=8,
            event_summary="Nobita opens the door.",
            keyframes=[ReviewKeyframe(time=4.0, path="verified.jpg")],
        )
        segment = NarrationSegment(
            segment_id="segment_0001",
            event_id="event_0001",
            narration="Nobita opens the door.",
            candidate_scene_ids=[scene.scene_id],
        )
        selected = SceneCandidate(
            candidate_id="candidate_0001",
            scene_id=scene.scene_id,
            start_seconds=0,
            end_seconds=2,
            match_score=0.8,
        )
        decision = EditDecision(
            segment_id=segment.segment_id,
            event_id=segment.event_id,
            narration=segment.narration,
            voice_start=0,
            voice_end=2,
            selected_candidate_id=selected.candidate_id,
            source_clips=[selected],
            alternatives=[selected],
        )
        payload = {
            "segments": [
                {
                    "segment_id": segment.segment_id,
                    "direct_match": True,
                    "score": 0.72,
                    "reason": "The frame directly shows the described action.",
                    "best_keyframe_time": 4.0,
                }
            ]
        }
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response)))
        )

        with patch("app.services.ai.review_analysis._client", return_value=client):
            updated = await _multimodal_rescore_selected_clips([segment], [scene], [decision])

        self.assertEqual(updated[0].source_clips[0].match_score, 0.75)
        self.assertAlmostEqual(updated[0].source_clips[0].start_seconds, 2.8)
        self.assertAlmostEqual(updated[0].source_clips[0].end_seconds, 5.2)
        self.assertIn("Verified keyframe time: 4.000s", updated[0].source_clips[0].match_reason)

    async def test_budget_valid_narration_retries_once_with_judge_feedback(self) -> None:
        events = ReviewTimelineTests._story_events()
        scenes = [
            AnalyzedScene(
                scene_id=event.scene_ids[0],
                start_time=event.start_time,
                end_time=event.end_time,
                event_summary=event.summary,
                visible_actions=[event.summary],
                evidence=ReviewEvidence(visual=[event.summary]),
            )
            for event in events
        ]
        event_sequence = [0, 1, 1, 2, 2, 3, 3, 4, 4]
        generation = {
            "title": "Review",
            "hook": "Hook",
            "summary": "Summary",
            "thumbnail_text": "Review",
            "tags": ["review"],
            "segments": [
                {
                    "event_id": events[event_index].event_id,
                    "narration": " ".join([f"tu{index}"] * 22),
                    "required_visuals": {
                        "characters": [],
                        "actions": [events[event_index].summary],
                        "objects": [],
                        "locations": [],
                    },
                    "candidate_scene_ids": events[event_index].scene_ids,
                    "confidence": 0.9,
                }
                for index, event_index in enumerate(event_sequence)
            ],
        }
        failed_judge = {
            "coherence_score": 72,
            "style_score": 74,
            "contradictions": ["Sai quan hệ nguyên nhân-kết quả"],
            "early_spoilers": [],
            "feedback": "Nối lại hai sự kiện và tăng chất cảm xúc.",
        }
        passed_judge = {
            "coherence_score": 92,
            "style_score": 90,
            "contradictions": [],
            "early_spoilers": [],
            "feedback": "Đạt.",
        }

        def response(payload: dict) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))]
            )

        create = AsyncMock(
            side_effect=[
                response(generation),
                response(failed_judge),
                response(generation),
                response(passed_judge),
            ]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with patch("app.services.ai.review_analysis._client", return_value=client):
            _, segments, assessment = await _write_verified_narration(
                events,
                scenes,
                target_minutes=1,
                style="emotional",
                notes=None,
            )

        self.assertEqual(create.await_count, 4)
        self.assertEqual(len(segments), 9)
        self.assertTrue(assessment.passed)
        generation_prompt = create.await_args_list[0].kwargs["messages"][0]["content"]
        self.assertIn("Phong cach da chon (emotional)", generation_prompt)
        self.assertIn(_style_contract("emotional"), generation_prompt)
        retry_prompt = create.await_args_list[2].kwargs["messages"][0]["content"]
        self.assertIn("Sai quan hệ nguyên nhân-kết quả", retry_prompt)


if __name__ == "__main__":
    unittest.main()
