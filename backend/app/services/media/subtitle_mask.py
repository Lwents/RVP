from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class SubtitleMaskAssets:
    mask_file: Path
    cleaned_video_file: Path


async def build_subtitle_mask_video(
    source_video: Path,
    output_file: Path,
    preferred_y_percent: float = 80,
    preferred_height_percent: float = 13,
) -> SubtitleMaskAssets | None:
    """Create a feathered, frame-accurate mask around hard subtitles."""
    return await asyncio.to_thread(
        _build_subtitle_mask_video_sync,
        source_video,
        output_file,
        preferred_y_percent,
        preferred_height_percent,
    )


def detect_subtitle_text_box(
    frame: np.ndarray,
    preferred_y_percent: float = 80,
    preferred_height_percent: float = 13,
) -> TextBox | None:
    """Detect a horizontally aligned hard-subtitle row without OCR."""
    height, width = frame.shape[:2]
    preferred_top = preferred_y_percent / 100
    preferred_bottom = (preferred_y_percent + preferred_height_percent) / 100
    # Keep the scan close to the configured subtitle zone. Creator watermarks
    # often run around 70-75% height and must not be mistaken for subtitles.
    scan_top = int(height * max(0.78, min(0.90, preferred_top)))
    scan_bottom = int(height * min(0.96, max(preferred_bottom + 0.02, preferred_top + 0.08)))
    scan_left = int(width * 0.06)
    scan_right = int(width * 0.94)
    roi = frame[scan_top:scan_bottom, scan_left:scan_right]
    if roi.size == 0:
        return None

    blue, green, red = cv2.split(roi)
    maximum = np.maximum.reduce([blue, green, red])
    minimum = np.minimum.reduce([blue, green, red])
    spread = maximum - minimum

    # Most hard subtitles are white, yellow, or another bright saturated color.
    # Original captions may be rendered semi-transparent grey. A 155 floor
    # misses their faded edge characters, so use a lower neutral-colour floor
    # and rely on typographic row/position filtering to reject the scene.
    near_white = (minimum > 110) & (spread < 70)
    bright_color = (maximum > 215) & (spread > 75)
    seed = ((near_white | bright_color).astype(np.uint8)) * 255

    contours, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    components: list[TextBox] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        x += scan_left
        y += scan_top
        if y < height * 0.74:
            continue
        if not (2 <= box_width <= width * 0.09):
            continue
        if not (height * 0.018 <= box_height <= height * 0.09):
            continue
        if area < 4:
            continue
        # Wide, almost solid highlights (chair/table edges in the failing
        # animation) are not typographic strokes.
        fill_ratio = area / max(1, box_width * box_height)
        if fill_ratio > 0.82 and box_width > box_height * 2.5:
            continue
        components.append(TextBox(x, y, box_width, box_height))

    if not components:
        return None

    preferred_center = height * (preferred_top + preferred_bottom) / 2
    runs = _candidate_text_runs(components, width, height, preferred_center)
    if not runs:
        return None

    run = max(runs, key=lambda item: _text_run_score(item, width, height, preferred_center))

    # Source subtitles in the reported video consist of a small attribution
    # row immediately above the main Chinese row. Treat both as one compact
    # subtitle region; otherwise the upper row leaks through the blur.
    primary_box = _box_from_items(run)
    adjacent: list[tuple[float, list[TextBox]]] = []
    primary_center_x = (primary_box.x + primary_box.right) / 2
    for candidate in runs:
        if candidate is run:
            continue
        candidate_box = _box_from_items(candidate)
        if candidate_box == primary_box:
            continue
        candidate_center_x = (candidate_box.x + candidate_box.right) / 2
        vertical_gap = max(
            0,
            max(primary_box.y, candidate_box.y) - min(primary_box.bottom, candidate_box.bottom),
        )
        if vertical_gap > max(round(height * 0.045), max(primary_box.height, candidate_box.height)):
            continue
        if abs(candidate_center_x - primary_center_x) > width * 0.18:
            continue
        horizontal_overlap = min(primary_box.right, candidate_box.right) - max(primary_box.x, candidate_box.x)
        if horizontal_overlap <= 0 and abs(candidate_center_x - primary_center_x) > width * 0.10:
            continue
        score = _text_run_score(candidate, width, height, preferred_center) - vertical_gap
        adjacent.append((score, candidate))
    selected_rows = [run]
    if adjacent:
        selected_rows.append(max(adjacent, key=lambda item: item[0])[1])

    selected = [item for row in selected_rows for item in row]

    x0 = min(item.x for item in selected)
    x1 = max(item.right for item in selected)
    y0 = min(item.y for item in selected)
    y1 = max(item.bottom for item in selected)

    # Pull in detached dots, accents, digits, and punctuation belonging to the
    # same row, while rejecting scene objects separated by a large x gap.
    extension = max(8, round(np.median([item.height for item in selected]) * 1.5))
    y_tolerance = max(2, round(height * 0.006))
    row_tops = [float(np.median([item.y for item in row])) for row in selected_rows]
    row_bottoms = [float(np.median([item.bottom for item in row])) for row in selected_rows]
    median_height = float(np.median([item.height for item in selected]))
    related = [
        item
        for item in components
        if item.right >= x0 - extension
        and item.x <= x1 + extension
        and item.height <= median_height * 1.8
        and any(
            abs(item.y - row_top) <= y_tolerance * 2
            or abs(item.bottom - row_bottom) <= y_tolerance * 2
            for row_top, row_bottom in zip(row_tops, row_bottoms)
        )
    ]
    if related:
        x0 = min(item.x for item in related)
        x1 = max(item.right for item in related)
        y0 = min(item.y for item in related)
        y1 = max(item.bottom for item in related)

    return TextBox(x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _choose_text_run(
    components: list[TextBox],
    width: int,
    height: int,
    preferred_center_y: float,
) -> list[TextBox] | None:
    """Pick a typographic row while rejecting bright scene objects."""
    runs = _candidate_text_runs(components, width, height, preferred_center_y)
    if not runs:
        return None
    return max(runs, key=lambda item: _text_run_score(item, width, height, preferred_center_y))


def _candidate_text_runs(
    components: list[TextBox],
    width: int,
    height: int,
    preferred_center_y: float,
) -> list[list[TextBox]]:
    """Return unique, centered text-like rows in the configured lower zone."""
    del preferred_center_y  # Ranking uses it; filtering deliberately does not.
    unique: dict[tuple[tuple[int, int, int, int], ...], list[TextBox]] = {}
    y_tolerance = max(2, round(height * 0.006))
    for anchor in components:
        row = []
        for item in components:
            height_ratio = max(item.height, anchor.height) / max(1, min(item.height, anchor.height))
            if height_ratio > 1.9:
                continue
            if (
                abs(item.y - anchor.y) <= y_tolerance
                or abs(item.bottom - anchor.bottom) <= y_tolerance
            ):
                row.append(item)
        for run in _horizontal_runs(row, width, height):
            x0 = min(item.x for item in run)
            x1 = max(item.right for item in run)
            y0 = min(item.y for item in run)
            y1 = max(item.bottom for item in run)
            span = x1 - x0
            center_x = (x0 + x1) / 2
            single_centered = (
                len(run) == 1
                and span >= width * 0.015
                and width * 0.40 <= center_x <= width * 0.60
            )
            short_centered = (
                len(run) == 2
                and span >= width * 0.035
                and width * 0.30 <= center_x <= width * 0.70
            )
            regular = (
                len(run) >= 3
                and span >= width * 0.06
                and width * 0.30 <= center_x <= width * 0.70
            )
            if not (single_centered or short_centered or regular):
                continue
            key = tuple(sorted((item.x, item.y, item.width, item.height) for item in run))
            unique[key] = run
    return list(unique.values())


def _text_run_score(
    run: list[TextBox],
    width: int,
    height: int,
    preferred_center_y: float,
) -> float:
    box = _box_from_items(run)
    span = box.width
    distance_penalty = abs(((box.y + box.bottom) / 2) - preferred_center_y) / max(height, 1)
    height_variance = float(np.std([item.height for item in run])) / max(
        float(np.mean([item.height for item in run])), 1.0
    )
    return len(run) * 3 + (span / width) * 15 - distance_penalty * 24 - height_variance * 6


def _box_from_items(items: list[TextBox]) -> TextBox:
    x0 = min(item.x for item in items)
    x1 = max(item.right for item in items)
    y0 = min(item.y for item in items)
    y1 = max(item.bottom for item in items)
    return TextBox(x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _horizontal_runs(items: list[TextBox], width: int, height: int) -> list[list[TextBox]]:
    if not items:
        return []
    ordered = sorted(items, key=lambda item: item.x)
    median_height = float(np.median([item.height for item in ordered]))
    # Stylised Chinese captions can have unusually wide tracking. Keep them in
    # one row without widening the final mask beyond the actual outer glyphs.
    gap_limit = max(width * 0.045, median_height * 1.8, height * 0.045)
    runs: list[list[TextBox]] = [[ordered[0]]]
    current_right = ordered[0].right
    for item in ordered[1:]:
        if item.x - current_right > gap_limit:
            runs.append([item])
            current_right = item.right
            continue
        runs[-1].append(item)
        current_right = max(current_right, item.right)
    return runs


def _build_subtitle_mask_video_sync(
    source_video: Path,
    output_file: Path,
    preferred_y_percent: float,
    preferred_height_percent: float,
) -> SubtitleMaskAssets | None:
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        logger.warning("Khong mo duoc video de tao subtitle mask: %s", source_video)
        return None

    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_width <= 0 or source_height <= 0:
        capture.release()
        return None

    mask_width = min(640, source_width)
    mask_width -= mask_width % 2
    mask_height = max(2, round(source_height * mask_width / source_width))
    mask_height -= mask_height % 2
    scale_x = mask_width / source_width
    scale_y = mask_height / source_height
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.unlink(missing_ok=True)
    cleaned_output_file = output_file.with_name(f"{output_file.stem}.cleaned.mp4")
    cleaned_output_file.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(output_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (mask_width, mask_height),
        True,
    )
    if not writer.isOpened():
        capture.release()
        logger.warning("Khong tao duoc subtitle mask video: %s", output_file)
        return None
    cleaned_writer = cv2.VideoWriter(
        str(cleaned_output_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (mask_width, mask_height),
        True,
    )
    if not cleaned_writer.isOpened():
        capture.release()
        writer.release()
        output_file.unlink(missing_ok=True)
        logger.warning("Khong tao duoc video inpainting phu de: %s", cleaned_output_file)
        return None

    history: deque[TextBox] = deque(maxlen=7)
    last_box: TextBox | None = None
    pending_box: TextBox | None = None
    pending_frames = 0
    missed_frames = 0
    previous_mask: np.ndarray | None = None
    detected_frames = 0
    written_frames = 0
    padding = max(6, min(20, round(30 * mask_height / 1080)))
    feather_pixels = max(6, min(25, round(25 * mask_height / 1080)))
    feather_sigma = max(1.5, feather_pixels / 3)
    glyph_dilate = max(5, min(14, round(18 * mask_height / 1080)))
    inpaint_radius = max(3, min(7, round(5 * mask_height / 1080)))
    hold_frames = max(8, round(fps * 0.5))
    switch_frames = 3

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if (source_width, source_height) != (mask_width, mask_height):
                frame = cv2.resize(frame, (mask_width, mask_height), interpolation=cv2.INTER_AREA)

            detected = detect_subtitle_text_box(frame, preferred_y_percent, preferred_height_percent)
            fallback = False
            if detected is not None:
                # Do not let one bright object teleport the blur away from the
                # subtitle. A genuinely different region must persist for a
                # few frames before it replaces the tracked row.
                if last_box is not None and not _boxes_compatible(
                    last_box,
                    detected,
                    mask_width,
                    mask_height,
                ):
                    if pending_box is not None and _boxes_compatible(
                        pending_box,
                        detected,
                        mask_width,
                        mask_height,
                    ):
                        pending_frames += 1
                        pending_box = detected
                    else:
                        pending_box = detected
                        pending_frames = 1
                    if pending_frames < switch_frames:
                        detected = None
                    else:
                        history.clear()
                        pending_box = None
                        pending_frames = 0
                else:
                    pending_box = None
                    pending_frames = 0

            if detected is not None:
                detected_frames += 1
                history.append(detected)
                box = _smooth_box(detected, history, mask_width, mask_height)
                last_box = box
                missed_frames = 0
            elif last_box is not None and missed_frames < hold_frames:
                missed_frames += 1
                box = last_box
            else:
                missed_frames += 1
                last_box = None
                history.clear()
                fallback = True
                fallback_y = max(0.78, min(0.90, preferred_y_percent / 100))
                fallback_height = max(0.08, min(0.16, preferred_height_percent / 100))
                if fallback_y + fallback_height > 0.96:
                    fallback_y = 0.96 - fallback_height
                box = TextBox(
                    round(mask_width * 0.05),
                    round(mask_height * fallback_y),
                    round(mask_width * 0.90),
                    round(mask_height * fallback_height),
                )

            glyph_mask = _text_glyph_mask(
                frame,
                box,
                padding=0 if fallback else padding,
                glyph_dilate=glyph_dilate,
                strict_components=fallback,
            )

            if fallback:
                glyph_points = cv2.findNonZero(glyph_mask)
                if glyph_points is None:
                    coverage_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
                else:
                    gx, gy, gw, gh = cv2.boundingRect(glyph_points)
                    coverage_mask = _box_coverage_mask(
                        mask_width,
                        mask_height,
                        TextBox(gx, gy, gw, gh),
                        padding,
                    )
            else:
                coverage_mask = _box_coverage_mask(mask_width, mask_height, box, padding)

            soft_mask = cv2.GaussianBlur(coverage_mask, (0, 0), feather_sigma)
            if previous_mask is not None:
                # Fast attack, slow release prevents a one-frame detection
                # dropout from revealing the old Chinese glyphs.
                released = (previous_mask.astype(np.float32) * 0.82).astype(np.uint8)
                soft_mask = cv2.max(soft_mask, released)
            previous_mask = soft_mask
            writer.write(cv2.cvtColor(soft_mask, cv2.COLOR_GRAY2BGR))
            cleaned_frame = frame
            if cv2.countNonZero(glyph_mask) > 0:
                cleaned_frame = cv2.inpaint(
                    frame,
                    glyph_mask,
                    inpaint_radius,
                    cv2.INPAINT_TELEA,
                )
            cleaned_writer.write(cleaned_frame)
            written_frames += 1
    finally:
        capture.release()
        writer.release()
        cleaned_writer.release()

    if written_frames == 0 or not output_file.is_file():
        output_file.unlink(missing_ok=True)
        cleaned_output_file.unlink(missing_ok=True)
        return None
    logger.info(
        "Da tao subtitle mask %s: %d/%d frame co bbox text (tong %d frame).",
        output_file,
        detected_frames,
        frame_count,
        written_frames,
    )
    return SubtitleMaskAssets(output_file, cleaned_output_file)


def _bright_text_seed(frame: np.ndarray) -> np.ndarray:
    blue, green, red = cv2.split(frame)
    maximum = np.maximum.reduce([blue, green, red])
    minimum = np.minimum.reduce([blue, green, red])
    spread = maximum - minimum
    near_white = (minimum > 110) & (spread < 70)
    bright_color = (maximum > 215) & (spread > 75)
    return ((near_white | bright_color).astype(np.uint8)) * 255


def _text_glyph_mask(
    frame: np.ndarray,
    box: TextBox,
    padding: int,
    glyph_dilate: int,
    strict_components: bool = False,
) -> np.ndarray:
    """Return a glyph-shaped mask instead of filling the whole subtitle box."""
    height, width = frame.shape[:2]
    crop_pad = padding + glyph_dilate
    x0 = max(0, box.x - crop_pad)
    y0 = max(0, box.y - crop_pad)
    x1 = min(width, box.right + crop_pad)
    y1 = min(height, box.bottom + crop_pad)
    full_mask = np.zeros((height, width), dtype=np.uint8)
    if x1 <= x0 or y1 <= y0:
        return full_mask

    crop = frame[y0:y1, x0:x1]
    seed = _bright_text_seed(crop)
    if not strict_components:
        # Inside a detector-approved compact box, keep every bright stroke.
        # Component height filtering used to discard thin anti-aliased parts
        # of Chinese glyphs, leaving readable outlines after the blur.
        kernel_size = glyph_dilate * 2 + 1
        glyph_mask = cv2.dilate(
            seed,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
            iterations=1,
        )
        full_mask[y0:y1, x0:x1] = glyph_mask
        return full_mask

    count, labels, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
    indexed_components: list[tuple[int, TextBox]] = []
    for label in range(1, count):
        component_x, component_y, component_w, component_h, area = stats[label]
        if area < 4:
            continue
        if not (2 <= component_w <= width * 0.09):
            continue
        if not (height * 0.012 <= component_h <= height * 0.09):
            continue
        indexed_components.append(
            (
                label,
                TextBox(x0 + component_x, y0 + component_y, component_w, component_h),
            )
        )

    if strict_components:
        chosen = _choose_text_run(
            [item for _, item in indexed_components],
            width,
            height,
            (box.y + box.bottom) / 2,
        )
        if not chosen:
            return full_mask
        run_x0 = min(item.x for item in chosen)
        run_x1 = max(item.right for item in chosen)
        median_top = float(np.median([item.y for item in chosen]))
        median_bottom = float(np.median([item.bottom for item in chosen]))
        median_height = float(np.median([item.height for item in chosen]))
        extension = max(6, round(median_height * 1.2))
        y_tolerance = max(2, round(height * 0.006))
        selected_labels = {
            label
            for label, item in indexed_components
            if item.right >= run_x0 - extension
            and item.x <= run_x1 + extension
            and item.height <= median_height * 1.8
            and (
                abs(item.y - median_top) <= y_tolerance * 2
                or abs(item.bottom - median_bottom) <= y_tolerance * 2
            )
        }
    glyph_mask = np.zeros_like(seed)
    for label in selected_labels:
        glyph_mask[labels == label] = 255
    if strict_components and not selected_labels:
        return full_mask
    kernel_size = glyph_dilate * 2 + 1
    glyph_mask = cv2.dilate(
        glyph_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        iterations=1,
    )
    full_mask[y0:y1, x0:x1] = glyph_mask
    return full_mask


def _box_coverage_mask(
    width: int,
    height: int,
    box: TextBox,
    padding: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    # A few faded edge glyphs are darker than the detector threshold. Add a
    # small horizontal-only safety margin (about 10px at 1280 wide) so those
    # final characters cannot peek out while the vertical band stays tight.
    horizontal_padding = padding + max(
        2,
        round(width * 0.008),
        round(box.width * 0.10),
    )
    x0 = max(0, box.x - horizontal_padding)
    y0 = max(0, box.y - padding)
    x1 = min(width, box.right + horizontal_padding)
    y1 = min(height, box.bottom + padding)
    if x1 > x0 and y1 > y0:
        cv2.rectangle(mask, (x0, y0), (x1 - 1, y1 - 1), 255, -1)
    return mask


def _boxes_compatible(
    previous: TextBox,
    current: TextBox,
    width: int,
    height: int,
) -> bool:
    previous_center_x = (previous.x + previous.right) / 2
    current_center_x = (current.x + current.right) / 2
    previous_center_y = (previous.y + previous.bottom) / 2
    current_center_y = (current.y + current.bottom) / 2
    return (
        abs(previous_center_x - current_center_x) <= width * 0.18
        and abs(previous_center_y - current_center_y)
        <= max(height * 0.045, max(previous.height, current.height) * 1.5)
    )


def _smooth_box(current: TextBox, history: deque[TextBox], width: int, height: int) -> TextBox:
    median_x = int(np.median([item.x for item in history]))
    median_y = int(np.median([item.y for item in history]))
    median_right = int(np.median([item.right for item in history]))
    median_bottom = int(np.median([item.bottom for item in history]))
    safety = max(2, round(height * 0.004))
    x0 = max(0, min(current.x, median_x))
    y0 = max(0, min(current.y, median_y) - safety)
    x1 = min(width, max(current.right, median_right))
    y1 = min(height, max(current.bottom, median_bottom) + safety)
    return TextBox(x0, y0, max(1, x1 - x0), max(1, y1 - y0))
