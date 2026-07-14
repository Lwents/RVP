import unittest

import cv2
import numpy as np

from app.services.ai.detect_regions import _target_subtitle_height
from app.services.media.subtitle_mask import (
    TextBox,
    _box_coverage_mask,
    _text_glyph_mask,
    detect_subtitle_text_box,
)


def _frame_with_two_glyphs() -> np.ndarray:
    frame = np.full((720, 1280, 3), 24, dtype=np.uint8)
    cv2.rectangle(frame, (596, 620), (618, 648), (250, 250, 250), -1)
    cv2.rectangle(frame, (642, 620), (664, 648), (250, 250, 250), -1)
    return frame


class SubtitleMaskTests(unittest.TestCase):
    def test_detects_short_centered_two_glyph_subtitle(self) -> None:
        detected = detect_subtitle_text_box(_frame_with_two_glyphs(), 80, 13)
        self.assertIsNotNone(detected)
        assert detected is not None
        self.assertLess(detected.width, 100)
        self.assertGreater(detected.y, 580)

    def test_mask_follows_glyphs_instead_of_filling_bounding_box(self) -> None:
        frame = _frame_with_two_glyphs()
        box = TextBox(580, 605, 105, 65)
        mask = _text_glyph_mask(frame, box, padding=8, glyph_dilate=4)
        nonzero = cv2.countNonZero(mask)
        self.assertGreater(nonzero, 0)
        self.assertLess(nonzero, box.width * box.height * 0.65)
        points = cv2.findNonZero(mask)
        _, _, mask_width, mask_height = cv2.boundingRect(points)
        self.assertLess(mask_width, 100)
        self.assertLess(mask_height, 55)

    def test_fallback_band_keeps_only_text_like_components(self) -> None:
        frame = _frame_with_two_glyphs()
        fallback = TextBox(64, 576, 1152, 94)
        mask = _text_glyph_mask(
            frame,
            fallback,
            padding=0,
            glyph_dilate=4,
            strict_components=True,
        )
        self.assertGreater(cv2.countNonZero(mask), 0)
        points = cv2.findNonZero(mask)
        _, _, mask_width, mask_height = cv2.boundingRect(points)
        self.assertLess(mask_width, 140)
        self.assertLess(mask_height, 60)
        self.assertLess(cv2.countNonZero(mask), fallback.width * fallback.height * 0.08)

    def test_detector_unites_two_bottom_rows_and_ignores_bright_chair(self) -> None:
        frame = np.full((720, 1280, 3), 24, dtype=np.uint8)
        # Solid off-centre scene object crossing the top of the scan zone.
        cv2.rectangle(frame, (930, 574), (1060, 596), (245, 245, 245), -1)
        # Small attribution row immediately above the main subtitle row.
        for x in (520, 555, 590, 625, 660, 695):
            cv2.rectangle(frame, (x, 590), (x + 16, 609), (250, 250, 250), -1)
        for x in (500, 545, 590, 635, 680, 725):
            cv2.rectangle(frame, (x, 630), (x + 22, 657), (250, 250, 250), -1)

        detected = detect_subtitle_text_box(frame, 80, 13)

        self.assertIsNotNone(detected)
        assert detected is not None
        self.assertLessEqual(detected.y, 590)
        self.assertGreaterEqual(detected.bottom, 657)
        self.assertLess(detected.right, 900)

    def test_coverage_mask_is_a_compact_padded_region(self) -> None:
        box = TextBox(500, 600, 250, 42)
        mask = _box_coverage_mask(1280, 720, box, padding=20)
        points = cv2.findNonZero(mask)
        self.assertIsNotNone(points)
        x, y, width, height = cv2.boundingRect(points)
        self.assertEqual((x, y, width, height), (455, 580, 340, 82))
        self.assertLess(cv2.countNonZero(mask), 1280 * 720 * 0.035)

    def test_auto_region_height_is_not_forced_to_a_large_banner(self) -> None:
        self.assertEqual(_target_subtitle_height(3), 6)
        self.assertEqual(_target_subtitle_height(9), 9)
        self.assertEqual(_target_subtitle_height(20), 13)


if __name__ == "__main__":
    unittest.main()
