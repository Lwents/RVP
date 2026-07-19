import json
import tempfile
import unittest
from pathlib import Path

from app.services.ai.voice import (
    _clamp_events_to_duration,
    _read_edge_word_timings,
    _split_tts_chunks,
    _voice_timeline_events,
    _voice_timeline_specs,
    load_voice_word_timings,
)
from app.services.subtitles.timing import SubtitleEvent


class VoiceTimelineTests(unittest.TestCase):
    def test_long_single_line_review_is_split_into_safe_tts_requests(self) -> None:
        sentence = " ".join(f"từ{index}" for index in range(500)) + "."

        chunks = _split_tts_chunks(sentence, 500)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 500 for chunk in chunks))
        self.assertEqual(" ".join(chunks), sentence)

    def test_edge_word_boundaries_are_loaded_and_scaled_after_tempo_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "part.metadata.jsonl"
            metadata.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "WordBoundary",
                                "offset": 1_000_000,
                                "duration": 2_000_000,
                                "text": "Xin",
                            }
                        ),
                        json.dumps(
                            {
                                "type": "WordBoundary",
                                "offset": 4_000_000,
                                "duration": 2_000_000,
                                "text": "chào",
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            parsed = _read_edge_word_timings(metadata, part_start=1.5)
            timing_file = root / "timings.json"
            timing_file.write_text(
                json.dumps({"audio_duration": 3.0, "words": parsed}, ensure_ascii=False),
                encoding="utf-8",
            )
            scaled = load_voice_word_timings(timing_file, target_duration=6.0)

        self.assertAlmostEqual(parsed[0]["start"], 1.6, places=3)
        self.assertAlmostEqual(parsed[1]["end"], 2.1, places=3)
        self.assertAlmostEqual(scaled[0]["start"], 3.2, places=3)
        self.assertAlmostEqual(scaled[1]["end"], 4.2, places=3)

    def test_clamping_creates_new_frozen_events(self) -> None:
        original = SubtitleEvent(0.0, 6.0, "Một câu dài.")

        clamped = _clamp_events_to_duration([original], 3.2)

        self.assertEqual(clamped, [SubtitleEvent(0.0, 3.2, "Một câu dài.")])
        self.assertEqual(original.end, 6.0)

    def test_events_after_duration_are_removed(self) -> None:
        events = [
            SubtitleEvent(0.0, 1.0, "Câu đầu."),
            SubtitleEvent(3.15, 4.0, "Ngoài video."),
        ]

        self.assertEqual(
            _clamp_events_to_duration(events, 3.2),
            [SubtitleEvent(0.0, 1.0, "Câu đầu.")],
        )

    def test_tts_keeps_each_displayed_subtitle_in_its_own_time_slot(self) -> None:
        events = [
            SubtitleEvent(0.0, 2.0, "Câu đầu tiên."),
            SubtitleEvent(2.0, 4.0, "Câu thứ hai."),
        ]

        timeline = _voice_timeline_events(events, duration=4.0)

        self.assertEqual(timeline, events)
        self.assertEqual([item.text for item in timeline], [item.text for item in events])
        self.assertEqual(timeline[1].start, timeline[0].end)

    def test_short_final_cue_never_extends_voice_past_video_duration(self) -> None:
        events = [SubtitleEvent(0.85, 1.20, "Câu cuối.")]

        specs, cursor = _voice_timeline_specs(events, duration=1.0)

        self.assertEqual(len(specs), 1)
        _, _, available, gap = specs[0]
        self.assertAlmostEqual(gap + available, 1.0, places=6)
        self.assertAlmostEqual(cursor, 1.0, places=6)

    def test_close_cues_stay_in_their_slots_and_keep_total_duration(self) -> None:
        events = [
            SubtitleEvent(0.0, 0.20, "Một."),
            SubtitleEvent(0.20, 0.40, "Hai."),
        ]

        specs, cursor = _voice_timeline_specs(events, duration=0.40)

        self.assertEqual(len(specs), 2)
        timeline_duration = sum(gap + available for _, _, available, gap in specs)
        self.assertAlmostEqual(timeline_duration, 0.40, places=6)
        self.assertAlmostEqual(cursor, 0.40, places=6)
        self.assertLessEqual(specs[0][2], events[1].start - events[0].start)


if __name__ == "__main__":
    unittest.main()
