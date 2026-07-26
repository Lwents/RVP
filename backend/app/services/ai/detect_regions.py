"""
Tự động nhận diện vùng sub chữ Trung và logo/watermark trong video gốc.
Hỗ trợ cả thuật toán xử lý ảnh Cục bộ (Local) và Trí tuệ nhân tạo (AI via 9router).
"""
from __future__ import annotations

import asyncio
import logging
import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from app.core import settings
from app.services.ai.openai_client import get_async_openai

logger = logging.getLogger(__name__)

# ─── Tuning knobs ────────────────────────────────────────────────────────────
SAMPLE_COUNT = 12          # số frame lấy mẫu để phân tích cục bộ
AI_SAMPLE_COUNT = 32       # mẫu đều toàn video; logo chỉ lấy loại cố định ở mép trên/dưới
CHINESE_UNICODE_RANGES = [
    (0x4E00, 0x9FFF),      # CJK Unified Ideographs
    (0x3400, 0x4DBF),      # CJK Extension A
    (0xFF00, 0xFFEF),      # Fullwidth forms
    (0x3000, 0x303F),      # CJK Symbols & Punctuation
    (0xFE30, 0xFE4F),      # CJK Compatibility Forms
]
SUBTITLE_SCAN_Y_START_RATIO = 0.70   # quét từ 70% độ cao màn hình xuống
SUBTITLE_SCAN_Y_END_RATIO   = 0.96   # dừng quét ở 96% để tránh thanh trạng thái video
SUBTITLE_SCAN_X_START_RATIO = 0.15   # bỏ qua 15% bên trái (tránh chữ dọc cảnh báo)
SUBTITLE_SCAN_X_END_RATIO   = 0.85   # bỏ qua 15% bên phải (tránh logo góc)
LOGO_CORNER_SIZE_RATIO      = 0.25   # quét vùng 25% ở các góc để tìm logo rộng
AI_REQUEST_TIMEOUT_SECONDS  = 180.0  # tránh treo 10-30 phút khi gateway không phản hồi
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _to_float(value: object, default: float = 0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _frame_indices(value: object, frame_count: int) -> list[int]:
    """Coerce AI-provided frame indices, tolerating JSON floats like 3.0.

    ``str(3.0).isdigit()`` is False, so a naive digit check silently drops every
    float index; the region then loses its start/end seconds and a briefly
    visible watermark ends up blurred across the entire video.
    """
    if not isinstance(value, (list, tuple, set)):
        return []
    indices: set[int] = set()
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if number != number or number in (float("inf"), float("-inf")):
            continue
        index = int(round(number))
        if 0 <= index < frame_count:
            indices.add(index)
    return sorted(indices)


def _target_subtitle_height(height: float) -> int:
    return round(_clamp(height, 6, 13))


def _merge_logo_regions(regions: list[dict]) -> list[dict]:
    merged: list[dict] = []
    # A timeless region is a fixed source logo and must win over a larger
    # time-scoped box that happens to overlap it.  Otherwise a moving mark can
    # accidentally remove the permanent platform logo during de-duplication.
    ordered = sorted(
        regions,
        key=lambda item: (
            not bool(item.get("locally_validated")),
            item.get("start_seconds") is not None,
            -(item["width_percent"] * item["height_percent"]),
        ),
    )
    for region in ordered:
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(merged)
                if region.get("start_seconds") is None
                and existing.get("start_seconds") is None
                and _overlap_ratio(region, existing) > 0.45
            ),
            None,
        )
        if duplicate_index is None:
            merged.append(region)
            continue

        existing = merged[duplicate_index]
        if existing.get("locally_validated"):
            # The persistence detector measured this box from all sampled
            # frames.  Keep its tight geometry instead of unioning it with a
            # looser AI estimate and turning the blur into a wide banner.
            continue
        x1 = min(existing["x_percent"], region["x_percent"])
        y1 = min(existing["y_percent"], region["y_percent"])
        x2 = max(existing["x_percent"] + existing["width_percent"], region["x_percent"] + region["width_percent"])
        y2 = max(existing["y_percent"] + existing["height_percent"], region["y_percent"] + region["height_percent"])
        merged[duplicate_index] = {
            **existing,
            "x_percent": round(x1),
            "y_percent": round(y1),
            "width_percent": round(x2 - x1),
            "height_percent": round(y2 - y1),
        }
    return merged


def _region_kind(region: dict) -> str:
    label = str(region.get("label", "")).lower()
    if "sub" in label or "subtitle" in label or "caption" in label:
        return "subtitle"
    if "logo" in label or "watermark" in label:
        return "logo"
    y = _to_float(region.get("y_percent"))
    width = _to_float(region.get("width_percent"))
    if y >= 65 and width >= 45:
        return "subtitle"
    return "logo"


def _normalize_region(region: dict, kind: str) -> dict | None:
    x = _to_float(region.get("x_percent"))
    y = _to_float(region.get("y_percent"))
    width = _to_float(region.get("width_percent"))
    height = _to_float(region.get("height_percent"))

    if width <= 0 or height <= 0:
        return None

    # Add bleed around detections so blur covers text/logo edges cleanly.
    if kind == "subtitle":
        pad_x = max(2.0, width * 0.06)
        # Black outlines and detached accents often extend above the bright
        # glyph pixels returned by vision models. Use extra asymmetric headroom
        # so the upper strokes cannot poke through the feathered mask.
        pad_top = max(1.5, height * 0.35)
        pad_bottom = max(1.0, height * 0.22)
        x -= pad_x
        y -= pad_top
        width += pad_x * 2
        height += pad_top + pad_bottom
    elif not region.get("locally_validated"):
        pad_x = max(2.0, width * 0.3)
        pad_y = max(2.5, height * 0.5)
        x -= pad_x
        y -= pad_y
        width += pad_x * 2
        height += pad_y * 2

    x = _clamp(x, 0, 99)
    y = _clamp(y, 0, 99)
    width = _clamp(width, 1, 100 - x)
    height = _clamp(height, 1, 100 - y)

    if kind == "subtitle":
        y = _clamp(y, 62, 94)
        height = _clamp(height, 4, 22)
        if y + height > 98:
            y = 98 - height
        label = "Subtitle blur (AI checked)"
    else:
        center_x = x + width / 2
        center_y = y + height / 2
        edge_position = center_y <= 22 or center_y >= 78
        if not edge_position:
            return None
        width = _clamp(width, 4, 55 if region.get("locally_validated") else 45)
        height = _clamp(height, 4, 25)
        # The source warning and platform mark span wider than the bright OCR
        # pixels detected in individual samples. Keep stable top masks wide
        # enough to cover the complete text throughout the video.
        if (
            not region.get("locally_validated")
            and not _has_time_range(region)
            and y <= 15
            and center_x <= 35
        ):
            x = 0
            width = max(width, 47)
            height = min(height, 13)
        elif (
            not region.get("locally_validated")
            and not _has_time_range(region)
            and y <= 15
            and center_x >= 65
        ):
            x = min(x, 70)
            width = 100 - x
            height = min(height, 13)
        label = "Logo/Watermark blur (AI checked)"

    normalized = {
        "x_percent": round(x, 2),
        "y_percent": round(y, 2),
        "width_percent": round(width, 2),
        "height_percent": round(height, 2),
        "label": label,
        "kind": kind,
    }
    if region.get("start_seconds") is not None and region.get("end_seconds") is not None:
        normalized["start_seconds"] = max(0.0, float(region["start_seconds"]))
        normalized["end_seconds"] = max(normalized["start_seconds"], float(region["end_seconds"]))
    if region.get("locally_validated"):
        normalized["locally_validated"] = True
    return normalized


def _overlap_ratio(a: dict, b: dict) -> float:
    ax1, ay1 = a["x_percent"], a["y_percent"]
    ax2 = ax1 + a["width_percent"]
    ay2 = ay1 + a["height_percent"]
    bx1, by1 = b["x_percent"], b["y_percent"]
    bx2 = bx1 + b["width_percent"]
    by2 = by1 + b["height_percent"]
    overlap_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    overlap_h = max(0, min(ay2, by2) - max(ay1, by1))
    overlap_area = overlap_w * overlap_h
    smaller_area = max(1, min(a["width_percent"] * a["height_percent"], b["width_percent"] * b["height_percent"]))
    return overlap_area / smaller_area


def _has_time_range(region: dict) -> bool:
    return region.get("start_seconds") is not None and region.get("end_seconds") is not None


def _time_ranges_overlap(a: dict, b: dict) -> bool:
    """Fixed regions cover every frame; timed regions overlap only in time."""
    if not _has_time_range(a) or not _has_time_range(b):
        return True
    # Adjacent midpoint windows share only their boundary timestamp; treating
    # that as an overlap can delete the box for the sample between them.
    return max(float(a["start_seconds"]), float(b["start_seconds"])) < min(
        float(a["end_seconds"]),
        float(b["end_seconds"]),
    )


def review_and_build_blur_config(regions: list[dict]) -> dict:
    """
    Sanity-check detections and convert them into render-ready settings.
    Subtitle detections become the main blur band; logo detections stay custom boxes.
    """
    normalized: list[dict] = []
    notes: list[str] = []

    for region in regions:
        kind = _region_kind(region)
        if kind == "logo" and _has_time_range(region):
            notes.append("Bo qua watermark/chu chay theo thoi gian; chi giu logo co dinh o mep tren/duoi.")
            continue
        clean = _normalize_region(region, kind)
        if clean is None:
            notes.append("Bo qua mot vung AI vi toa do khong hop ly hoac logo khong nam o goc.")
            continue
        normalized.append(clean)

    subtitle_regions = [r for r in normalized if r["kind"] == "subtitle"]
    logo_regions = [r for r in normalized if r["kind"] == "logo"]

    subtitle_region = None
    if subtitle_regions:
        subtitle_region = max(subtitle_regions, key=lambda r: r["width_percent"] * r["height_percent"])
        notes.append("Da giu bounding box subtitle de tao mask mem bam sat chu.")
    else:
        subtitle_region = {
            "x_percent": 5,
            "y_percent": 80,
            "width_percent": 90,
            "height_percent": 13,
            "label": "Subtitle blur safety band",
            "kind": "subtitle",
        }
        normalized.append(subtitle_region)
        notes.append("AI khong chac vung subtitle, dung dai mo an toan gon o day khung hinh.")

    subtitle_region["y_percent"] = round(_clamp(subtitle_region["y_percent"], 76, 92), 2)
    subtitle_region["height_percent"] = _target_subtitle_height(subtitle_region["height_percent"])
    if subtitle_region["y_percent"] + subtitle_region["height_percent"] > 97:
        subtitle_region["y_percent"] = 97 - subtitle_region["height_percent"]

    custom_boxes: list[dict] = []
    for logo in _merge_logo_regions(logo_regions):
        if logo["y_percent"] <= 15 and not _has_time_range(logo):
            logo["y_percent"] = 0
            logo["height_percent"] = min(logo["height_percent"], 13)
        if (
            subtitle_region
            and not _has_time_range(logo)
            and _overlap_ratio(logo, subtitle_region) > 0.25
        ):
            notes.append("Bo qua mot logo box vi trung voi dai subtitle.")
            continue
        if any(
            _overlap_ratio(logo, existing) > 0.6 and _time_ranges_overlap(logo, existing)
            for existing in custom_boxes
        ):
            notes.append("Gop/bo bot logo box bi trung lap.")
            continue
        custom_boxes.append({
            "x_percent": logo["x_percent"],
            "y_percent": logo["y_percent"],
            "width_percent": logo["width_percent"],
            "height_percent": logo["height_percent"],
            **(
                {
                    "start_seconds": logo["start_seconds"],
                    "end_seconds": logo["end_seconds"],
                }
                if logo.get("start_seconds") is not None and logo.get("end_seconds") is not None
                else {}
            ),
        })
    if logo_regions and len(custom_boxes) < len(logo_regions):
        notes.append("Da gop cac box logo trung lap de giu dung so vung can lam mo.")

    config = {
        "blur_box_enabled": bool(subtitle_region),
        "blur_box_y_percent": subtitle_region["y_percent"] if subtitle_region else 80,
        "blur_box_height_percent": subtitle_region["height_percent"] if subtitle_region else 15,
        "custom_blur_boxes": custom_boxes,
        "subtitle_y_percent": (
            round(_clamp(subtitle_region["y_percent"] + subtitle_region["height_percent"] / 2, 8, 94))
            if subtitle_region else None
        ),
    }

    return {
        "regions": normalized,
        "config": config,
        "review": {
            "ok": bool(subtitle_region or custom_boxes),
            "notes": notes,
        },
    }


def _has_chinese(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if any(lo <= code <= hi for lo, hi in CHINESE_UNICODE_RANGES):
            return True
    return False


def _sample_frames(video_path: str, count: int) -> list[np.ndarray]:
    """Lấy `count` frame phân bố đều trong video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 24

    # Bỏ 5% đầu và 5% cuối
    start = int(total * 0.05)
    end   = int(total * 0.95)
    indices = [start + int((end - start) * i / max(count - 1, 1))
               for i in range(count)]

    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def _sample_frames_with_times(video_path: str, count: int) -> tuple[list[tuple[np.ndarray, float]], float]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    duration = total / fps if total > 0 else 0.0
    if total <= 0 or count <= 0:
        cap.release()
        return [], duration

    # Fixed logos and hard subtitles are best verified over the complete video.
    # Uniform sampling also prevents moving/scrolled text from looking fixed.
    last_index = max(0, total - 1)
    uniform_indices = {
        int(round(value))
        for value in np.linspace(0, last_index, count)
    }
    indices = sorted(uniform_indices)

    # Short clips can collapse several requested timestamps onto the same frame.
    # Fill any missing slots from a denser grid without creating duplicates.
    if len(indices) < min(count, total):
        for value in np.linspace(0, last_index, min(total, count * 4)):
            index = int(round(value))
            if index not in indices:
                indices.append(index)
            if len(indices) >= min(count, total):
                break
        indices.sort()
    samples: list[tuple[np.ndarray, float]] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            samples.append((frame, idx / fps))
    cap.release()
    return samples, duration


def _detect_persistent_edge_logos(frames: list[np.ndarray]) -> list[dict]:
    """
    Validate fixed source logos along the top and bottom edges.

    A platform logo or old channel warning keeps the same edges while the scene
    behind it changes. Moving/scrolled Chinese text does not remain at the same
    coordinates, so it disappears from this temporal-persistence mask.
    """
    if len(frames) < 6:
        return []

    normalized_frames: list[np.ndarray] = []
    for frame in frames:
        if frame is None or frame.size == 0:
            continue
        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            continue
        target_width = min(640, width)
        target_height = max(1, int(round(height * target_width / width)))
        normalized_frames.append(cv2.resize(frame, (target_width, target_height)))

    if len(normalized_frames) < 6:
        return []

    target_height, target_width = normalized_frames[0].shape[:2]
    normalized_frames = [
        frame
        if frame.shape[:2] == (target_height, target_width)
        else cv2.resize(frame, (target_width, target_height))
        for frame in normalized_frames
    ]

    band_height = max(1, int(round(target_height * 0.22)))
    close_width = max(9, int(round(target_width * 0.027)))
    if close_width % 2 == 0:
        close_width += 1
    min_width = max(12, int(round(target_width * 0.04)))
    min_height = max(4, int(round(target_height * 0.012)))
    min_area = max(30, int(round(target_width * target_height * 0.0008)))
    detections: list[dict] = []

    for edge_name, y_offset in (("top", 0), ("bottom", target_height - band_height)):
        edge_maps: list[np.ndarray] = []
        dilate_kernel = np.ones((3, 3), np.uint8)
        for frame in normalized_frames:
            band = frame[y_offset:y_offset + band_height]
            gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 70, 150, L2gradient=True)
            edge_maps.append(cv2.dilate(edges, dilate_kernel) > 0)

        persistence = np.mean(np.stack(edge_maps, axis=0), axis=0)
        mask = (persistence > 0.50).astype(np.uint8) * 255
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (close_width, 5)),
        )
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask)

        for index in range(1, component_count):
            x, y, width, height, area = [int(value) for value in stats[index]]
            touches_outer_edge = (
                y <= target_height * 0.10
                if edge_name == "top"
                else y + height >= band_height - target_height * 0.10
            )
            if width < min_width or height < min_height or area < min_area:
                continue
            if not touches_outer_edge or width > target_width * 0.60:
                continue

            # Add roughly 15 px at 720p before marking this as pre-validated.
            pad_x = max(4, int(round(target_width * 0.012)))
            pad_y = max(3, int(round(target_height * 0.021)))
            global_y = y_offset + y
            x1 = max(0, x - pad_x)
            y1 = max(0, global_y - pad_y)
            x2 = min(target_width, x + width + pad_x)
            y2 = min(target_height, global_y + height + pad_y)
            detections.append({
                "x_percent": round(x1 / target_width * 100, 2),
                "y_percent": round(y1 / target_height * 100, 2),
                "width_percent": round((x2 - x1) / target_width * 100, 2),
                "height_percent": round((y2 - y1) / target_height * 100, 2),
                "label": f"Persistent {edge_name} source logo (local validation)",
                "locally_validated": True,
            })

    return detections


def _track_missing_creator_watermark_samples(
    regions: list[dict],
    frames: list[np.ndarray],
) -> list[dict]:
    """Fill sampled frames where AI omitted a moving creator watermark."""
    if not regions or not frames:
        return regions

    candidates: list[tuple[dict, list[int]]] = []
    for region in regions:
        label = str(region.get("label", "")).lower()
        if not any(token in label for token in ("watermark", "creator", "logo")):
            continue
        indices = _frame_indices(region.get("frame_indices", []), len(frames))
        if not indices:
            continue

        x = _to_float(region.get("x_percent"))
        y = _to_float(region.get("y_percent"))
        width = _to_float(region.get("width_percent"))
        height = _to_float(region.get("height_percent"))
        fixed_top = y <= 15 and (x <= 20 or x + width >= 80)
        if fixed_top or not (8 <= width <= 40 and 1.5 <= height <= 12):
            continue
        candidates.append((region, indices))

    if not candidates:
        return regions

    explicit_creator_candidates = [
        item
        for item in candidates
        if any(
            token in str(item[0].get("label", "")).lower()
            for token in ("creator", "moving", "original", "@", "原创")
        )
    ]
    candidate_pool = explicit_creator_candidates or candidates
    covered_indices = {
        frame_index
        for _, indices in candidate_pool
        for frame_index in indices
    }

    # A mid-frame seed avoids subtitles at the bottom and permanent top logos.
    mid_candidates = [
        item
        for item in candidate_pool
        if 15 <= _to_float(item[0].get("y_percent")) <= 70
    ]
    seed_region, seed_indices = (mid_candidates or candidate_pool)[0]
    seed_index = seed_indices[0]
    seed_frame = frames[seed_index]
    if seed_frame is None or seed_frame.size == 0:
        return regions

    def _resize_for_tracking(frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= 640:
            return frame
        target_height = max(1, int(round(height * 640 / width)))
        return cv2.resize(frame, (640, target_height))

    seed_frame = _resize_for_tracking(seed_frame)
    frame_height, frame_width = seed_frame.shape[:2]
    x1 = int(round(_to_float(seed_region.get("x_percent")) / 100 * frame_width))
    y1 = int(round(_to_float(seed_region.get("y_percent")) / 100 * frame_height))
    x2 = int(round(
        (_to_float(seed_region.get("x_percent")) + _to_float(seed_region.get("width_percent")))
        / 100 * frame_width
    ))
    y2 = int(round(
        (_to_float(seed_region.get("y_percent")) + _to_float(seed_region.get("height_percent")))
        / 100 * frame_height
    ))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_width, x2), min(frame_height, y2)
    if x2 - x1 < 12 or y2 - y1 < 5:
        return regions

    gradient_kernel = np.ones((3, 3), np.uint8)

    def _gradient(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, gradient_kernel)

    template = _gradient(seed_frame[y1:y2, x1:x2])
    if cv2.countNonZero(template) < template.size * 0.04:
        return regions

    tracked = list(regions)
    template_height, template_width = template.shape[:2]
    for frame_index, source_frame in enumerate(frames):
        if frame_index in covered_indices or source_frame is None or source_frame.size == 0:
            continue
        frame = _resize_for_tracking(source_frame)
        if frame.shape[0] < template_height or frame.shape[1] < template_width:
            continue
        response = cv2.matchTemplate(_gradient(frame), template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(response)
        if score < 0.72:
            continue
        x, y = location
        height, width = frame.shape[:2]
        tracked.append({
            "x_percent": round(x / width * 100, 2),
            "y_percent": round(y / height * 100, 2),
            "width_percent": round(template_width / width * 100, 2),
            "height_percent": round(template_height / height * 100, 2),
            "label": "Moving creator watermark (local tracking)",
            "frame_indices": [frame_index],
        })
        covered_indices.add(frame_index)

    return tracked


def _contiguous_runs(indices: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for index in indices:
        if not runs or index != runs[-1][-1] + 1:
            runs.append([index])
        else:
            runs[-1].append(index)
    return runs


def _sample_window_start(times: list[float], index: int) -> float:
    if index <= 0:
        return 0.0
    return max(0.0, (times[index - 1] + times[index]) / 2)


def _sample_window_end(times: list[float], index: int, duration: float) -> float:
    if index >= len(times) - 1:
        return duration
    return min(duration, (times[index] + times[index + 1]) / 2)


# ─── Local Processing Engine ──────────────────────────────────────────────────

def detect_chinese_subtitle_region_local(video_path: str) -> dict | None:
    """
    Quét video, tìm vùng chứa chữ Trung/Hán ở khu vực subtitle thông thường bằng EasyOCR.
    """
    try:
        import easyocr
        reader = easyocr.Reader(["ch_sim", "en"], gpu=False, verbose=False)
    except ImportError:
        logger.warning("easyocr chưa được cài. Bỏ qua nhận diện sub Trung.")
        return None

    frames = _sample_frames(video_path, SAMPLE_COUNT)
    if not frames:
        return None

    h, w = frames[0].shape[:2]
    scan_y_start = int(h * SUBTITLE_SCAN_Y_START_RATIO)
    scan_y_end   = int(h * SUBTITLE_SCAN_Y_END_RATIO)
    scan_x_start = int(w * SUBTITLE_SCAN_X_START_RATIO)
    scan_x_end   = int(w * SUBTITLE_SCAN_X_END_RATIO)

    all_boxes_y: list[int] = []
    all_boxes_h: list[int] = []
    all_boxes_x: list[int] = []
    all_boxes_right: list[int] = []

    for frame in frames:
        roi = frame[scan_y_start:scan_y_end, scan_x_start:scan_x_end]
        results = reader.readtext(roi, detail=1, paragraph=False)
        for (bbox, text, conf) in results:
            if conf < 0.4:
                continue
            if not _has_chinese(text):
                continue
            ys = [pt[1] for pt in bbox]
            xs = [pt[0] for pt in bbox]
            x_left = int(min(xs)) + scan_x_start
            x_right = int(max(xs)) + scan_x_start
            y_top    = int(min(ys)) + scan_y_start
            y_bottom = int(max(ys)) + scan_y_start
            all_boxes_x.append(x_left)
            all_boxes_right.append(x_right)
            all_boxes_y.append(y_top)
            all_boxes_h.append(y_bottom - y_top)

    if not all_boxes_y:
        return None

    median_y = int(np.median(all_boxes_y))
    median_h = int(np.median(all_boxes_h)) if all_boxes_h else 40
    pad = max(8, int(median_h * 0.2))

    y_start = max(0, median_y - pad)
    box_h   = min(h - y_start, median_h + pad * 2)
    x_start = max(0, int(np.percentile(all_boxes_x, 20)) - pad)
    x_end = min(w, int(np.percentile(all_boxes_right, 80)) + pad)

    return {
        "x_percent":      round(x_start / w * 100, 2),
        "y_percent":      round(y_start / h * 100),
        "width_percent":  max(5, round((x_end - x_start) / w * 100, 2)),
        "height_percent": max(5, round(box_h / h * 100)),
        "label":          "Sub chữ Trung (Cục bộ)"
    }


def detect_watermark_region_local(video_path: str) -> dict | None:
    """
    Nhận diện logo/watermark cố định góc màn hình bằng OpenCV.
    """
    frames = _sample_frames(video_path, SAMPLE_COUNT)
    if len(frames) < 4:
        return None

    h, w = frames[0].shape[:2]
    corner_h = int(h * LOGO_CORNER_SIZE_RATIO)
    corner_w = int(w * LOGO_CORNER_SIZE_RATIO)

    corners = {
        "top_left":     (0,          0,          corner_w, corner_h),
        "top_right":    (w-corner_w, 0,          corner_w, corner_h),
        "bottom_left":  (0,          h-corner_h, corner_w, corner_h),
        "bottom_right": (w-corner_w, h-corner_h, corner_w, corner_h),
    }

    best_corner = None
    best_score  = float("inf")

    for name, (cx, cy, cw, ch) in corners.items():
        rois = []
        for frame in frames:
            roi = cv2.cvtColor(frame[cy:cy+ch, cx:cx+cw], cv2.COLOR_BGR2GRAY)
            rois.append(roi.astype(np.float32))

        stack = np.stack(rois, axis=0)
        variance_map = np.var(stack, axis=0)

        patch_h, patch_w = 80, 80
        min_var = float("inf")
        best_patch_x, best_patch_y = 0, 0
        for py in range(0, ch - patch_h, 20):
            for px in range(0, cw - patch_w, 20):
                mean_var = float(np.mean(variance_map[py:py+patch_h, px:px+patch_w]))
                if mean_var < min_var:
                    min_var = mean_var
                    best_patch_x = px
                    best_patch_y = py

        avg_frames = np.mean(stack, axis=0)
        patch_mean = float(np.mean(avg_frames[best_patch_y:best_patch_y+patch_h,
                                              best_patch_x:best_patch_x+patch_w]))

        if 15 < patch_mean < 240 and min_var < best_score:
            best_score  = min_var
            best_corner = (name, cx + best_patch_x, cy + best_patch_y,
                           patch_w, patch_h)

    if best_corner is None or best_score > 800:
        return None

    name, bx, by, bw, bh = best_corner
    pad = 15
    bx = max(0, bx - pad)
    by = max(0, by - pad)
    bw = min(w - bx, bw + pad * 2)
    bh = min(h - by, bh + pad * 2)

    return {
        "x_percent":      round(bx / w * 100),
        "y_percent":      round(by / h * 100),
        "width_percent":  max(5, round(bw / w * 100)),
        "height_percent": max(5, round(bh / h * 100)),
        "label":          f"Logo góc {name} (Cục bộ)"
    }


def auto_detect_blur_regions_local(
    video_path: str,
    detect_sub: bool = True,
    detect_logo: bool = True,
) -> list[dict]:
    """Phân tích video cục bộ sử dụng EasyOCR và OpenCV."""
    results: list[dict] = []

    if detect_sub:
        logger.info("Nhận diện sub Trung cục bộ...")
        region = detect_chinese_subtitle_region_local(video_path)
        if region:
            results.append(region)

    if detect_logo:
        logger.info("Nhận diện logo cục bộ...")
        sampled_frames, _ = _sample_frames_with_times(video_path, AI_SAMPLE_COUNT)
        persistent_regions = _detect_persistent_edge_logos(
            [frame for frame, _ in sampled_frames]
        )
        if persistent_regions:
            results.extend(persistent_regions)
        else:
            # Keep the legacy single-corner detector only as a last fallback.
            region = detect_watermark_region_local(video_path)
            if region:
                results.append(region)

    return results


# ─── Multimodal AI Engine ─────────────────────────────────────────────────────

def _encode_sample_frames(
    samples: list[tuple[np.ndarray, float]],
) -> list[tuple[str, float]]:
    """Downscale and JPEG-encode sampled frames (blocking; run in a thread)."""
    encoded: list[tuple[str, float]] = []
    for frame, timestamp in samples:
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))

        ok, buffer = cv2.imencode(".jpg", frame)
        if ok:
            encoded.append((base64.b64encode(buffer).decode("utf-8"), timestamp))
    return encoded


async def auto_detect_blur_regions_ai(
    video_path: str,
    detect_sub: bool = True,
    detect_logo: bool = True,
) -> list[dict]:
    """
    Sử dụng Multimodal AI qua 9router để nhận diện vùng sub và logo từ các frame ảnh.
    """
    # Decoding ~32 frames, running Canny over all of them and JPEG-encoding them
    # blocks for seconds; keep every OpenCV step off the event loop. This
    # coroutine must stay awaitable from the running loop (routes.py awaits it
    # directly) so the shared AsyncOpenAI client keeps its loop affinity.
    samples, video_duration = await asyncio.to_thread(
        _sample_frames_with_times, video_path, AI_SAMPLE_COUNT
    )
    if not samples:
        return []

    persistent_logo_regions = (
        await asyncio.to_thread(
            _detect_persistent_edge_logos, [frame for frame, _ in samples]
        )
        if detect_logo
        else []
    )

    base64_frames: list[tuple[str, float]] = await asyncio.to_thread(
        _encode_sample_frames, samples
    )
    if not base64_frames:
        return []

    prompt = (
        "You are a professional video editor assistant.\n"
        "Analyze these video frames and detect:\n"
    )
    if detect_sub:
        prompt += (
            "- A TIGHT bounding box around the visible pixels of Chinese hard subtitles in EACH sample frame. "
            "Subtitles are horizontally centered and located "
            "strictly at the bottom center of the screen (typically between y=75% and y=95%). "
            "Do not expand the box to full frame width. CRITICAL: Ignore any vertical text or warnings "
            "running down the sides of the video.\n"
        )
    if detect_logo:
        prompt += (
            "- Detect ONLY fixed graphical source logos/channel/platform marks located along the TOP or BOTTOM "
            "edge of the frame, including the fixed top-left warning/logo and top-right bilibili platform mark. "
            "A valid logo must stay at the same coordinates across many samples. CRITICAL: completely IGNORE "
            "moving or scrolling Chinese creator text such as text beginning with 原创@, even when it appears "
            "near an edge. Also ignore characters, credits, dialogue, and ordinary scene text.\n"
        )
        
    prompt += (
        "\nFor each detected item, calculate coordinates in percentages (0 to 100) relative to the frame size:\n"
        "- x_percent: horizontal offset of top-left corner.\n"
        "- y_percent: vertical offset of top-left corner.\n"
        "- width_percent: width of the region.\n"
        "- height_percent: height of the region.\n"
        "- frame_indices: zero-based indices of sampled frames where this exact region is visible. "
        "This is required for subtitles. Fixed logos may omit frame_indices. Never return moving creator text.\n"
        "\nReturn ONLY a JSON object containing the regions list, like this:\n"
        "{\n"
        "  \"regions\": [\n"
        "    {\"x_percent\": 24, \"y_percent\": 84, \"width_percent\": 52, \"height_percent\": 6, \"label\": \"Sub chữ Trung (AI)\", \"frame_indices\": [0]},\n"
        "    {\"x_percent\": 76, \"y_percent\": 3, \"width_percent\": 22, \"height_percent\": 6, \"label\": \"Fixed top-right platform logo\"}\n"
        "  ]\n"
        "}"
    )

    client = get_async_openai(
        settings.ninerouter_api_key,
        settings.ninerouter_api_url,
        timeout=AI_REQUEST_TIMEOUT_SECONDS,
        max_retries=1,
    )

    content = [{"type": "text", "text": prompt}]
    for frame_index, (img_b64, timestamp) in enumerate(base64_frames):
        content.append({"type": "text", "text": f"Sample frame {frame_index} at {timestamp:.2f} seconds:"})
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}"
            }
        })

    logger.info("Đang gọi Multimodal AI 9router (%s) để nhận diện vùng...", settings.ai_model)
    response = await client.chat.completions.create(
        model=settings.ai_model,
        messages=[
            {"role": "user", "content": content}
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        timeout=AI_REQUEST_TIMEOUT_SECONDS,
    )

    res_text = response.choices[0].message.content
    if not res_text:
        logger.warning("Không nhận được phản hồi từ AI.")
        return persistent_logo_regions

    def _clean_json_text(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    try:
        clean_text = _clean_json_text(res_text)
        data = json.loads(clean_text)
        regions = data.get("regions", [])
        sample_times = [timestamp for _, timestamp in base64_frames]
        expanded_regions: list[dict] = []
        for r in regions:
            raw_label = str(r.get("label", "")).lower()
            if any(
                token in raw_label
                for token in ("creator", "moving", "scroll", "original", "@", "原创")
            ):
                # User only wants fixed edge logos, never the Chinese creator
                # watermark that travels through the picture.
                continue
            if "label" not in r:
                r["label"] = "Vùng làm mờ (AI)"
            elif "sub" in r["label"].lower():
                r["label"] = "Sub chữ Trung (AI)"
            elif "logo" in r["label"].lower() or "watermark" in r["label"].lower():
                r["label"] = "Logo/Watermark (AI)"
            frame_indices = _frame_indices(r.pop("frame_indices", []), len(sample_times))
            if (
                frame_indices
                and len(frame_indices) < max(2, len(sample_times) - 2)
            ):
                for run in _contiguous_runs(frame_indices):
                    item = dict(r)
                    item["start_seconds"] = _sample_window_start(sample_times, run[0])
                    item["end_seconds"] = _sample_window_end(sample_times, run[-1], video_duration)
                    expanded_regions.append(item)
            else:
                expanded_regions.append(r)
        # Gemini can occasionally describe only one of two permanent corner
        # marks.  Add edge-persistence detections from the same sampled frames;
        # the review step below de-duplicates them while preserving fixed boxes.
        expanded_regions.extend(persistent_logo_regions)
        logger.info("AI đã nhận diện thành công: %s", expanded_regions)
        return expanded_regions
    except Exception as e:
        logger.error("Lỗi parse kết quả từ AI: %s", e)
        logger.error("Nội dung phản hồi gốc: %s", res_text)
        return persistent_logo_regions
