from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.models.job import BgmMode, DubbingRequest
from app.services.media.renderer import _append_soft_box_blur
from app.services.media.review_renderer import _build_review_output_command


class StaticSubtitleBlurTests(unittest.TestCase):
    def test_subtitle_band_has_fixed_geometry_without_time_gate(self) -> None:
        filters: list[str] = []

        output = _append_soft_box_blur(
            filters,
            "[vbase]",
            1280,
            720,
            [(5.0, 80.0, 90.0, 13.0, None, None)],
            "substatic",
            strong_subtitle=True,
        )
        graph = ";".join(filters)

        self.assertEqual(output, "[substatic_out]")
        self.assertIn("between(X", graph)
        self.assertIn("between(Y", graph)
        self.assertIn("gblur=sigma=32:sigmaV=19:steps=3", graph)
        self.assertIn("alphamerge", graph)
        self.assertIn("overlay=0:0:format=auto", graph)
        self.assertNotIn("maskedmerge", graph)
        self.assertNotIn("T-", graph)

    def test_custom_box_keeps_lighter_default_blur(self) -> None:
        filters: list[str] = []

        _append_soft_box_blur(
            filters,
            "[vbase]",
            1280,
            720,
            [(2.0, 2.0, 15.0, 10.0, None, None)],
            "customsoft",
        )
        graph = ";".join(filters)

        self.assertIn("gblur=sigma=25:steps=2", graph)
        self.assertIn("maskedmerge", graph)
        self.assertNotIn("alphamerge", graph)

    def test_review_command_does_not_add_dynamic_mask_inputs(self) -> None:
        request = DubbingRequest(
            local_file_path="source.mp4",
            hard_subtitles=False,
            blur_box_enabled=True,
            logo_enabled=False,
            bgm_mode=BgmMode.none,
        )

        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            command = _build_review_output_command(
                "ffmpeg",
                work_dir / "silent.mp4",
                work_dir / "narration.mp3",
                work_dir / "subtitles.srt",
                work_dir / "output.mp4",
                work_dir,
                10.0,
                request,
            )

        graph = command[command.index("-filter_complex") + 1]
        self.assertEqual(command.count("-i"), 2)
        self.assertIn("reviewstatic", graph)
        self.assertNotIn("subtitle_blur_mask", " ".join(command))
        self.assertNotIn(".cleaned", " ".join(command))
        self.assertNotIn("T-", graph)


if __name__ == "__main__":
    unittest.main()
