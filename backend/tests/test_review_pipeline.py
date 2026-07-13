from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.models.review import AnalyzedScene, ReviewEvidence
from app.services.ai.review_analysis import _reconcile_character_identities
from app.services.media.review_renderer import build_review_scenes, write_review_subtitles
from app.services.subtitles.timing import parse_srt


class ReviewTimelineTests(unittest.TestCase):
    def test_verified_window_is_reused_instead_of_crossing_into_next_scene(self) -> None:
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

        self.assertAlmostEqual(sum(duration for _, duration in clips), 4.5, places=3)
        self.assertGreater(len(clips), 1)
        self.assertTrue(all(start >= 7.0 for start, _ in clips))
        self.assertTrue(all(start + duration <= 9.001 for start, duration in clips))

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


if __name__ == "__main__":
    unittest.main()
