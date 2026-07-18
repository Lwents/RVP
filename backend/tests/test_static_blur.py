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

    def test_review_render_disables_all_automatic_subtitle_wrapping(self) -> None:
        request = DubbingRequest(
            local_file_path="source.mp4",
            hard_subtitles=True,
            blur_box_enabled=True,
            logo_enabled=False,
            bgm_mode=BgmMode.none,
        )

        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            subtitle_file = work_dir / "subtitles.srt"
            subtitle_file.write_text(
                "1\n00:00:00,000 --> 00:00:02,000\nMột dòng phụ đề review.\n\n",
                encoding="utf-8",
            )
            command = _build_review_output_command(
                "ffmpeg",
                work_dir / "silent.mp4",
                work_dir / "narration.mp3",
                subtitle_file,
                work_dir / "output.mp4",
                work_dir,
                2.0,
                request,
            )
            ass_text = (work_dir / "review_subtitles.positioned.ass").read_text(encoding="utf-8")

        graph = command[command.index("-filter_complex") + 1]
        self.assertIn("wrap_unicode=0", graph)
        self.assertIn("WrapStyle: 2", ass_text)
        dialogue = next(line for line in ass_text.splitlines() if line.startswith("Dialogue:"))
        self.assertIn(r"\q2", dialogue)
        self.assertNotIn(r"\N", dialogue)


if __name__ == "__main__":
    unittest.main()
