import unittest

from app.services.ai.voice import _clamp_events_to_duration
from app.services.subtitles.timing import SubtitleEvent


class VoiceTimelineTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
