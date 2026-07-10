"""
Tự động nhận diện vùng sub chữ Trung và logo/watermark trong video gốc.
Hỗ trợ cả thuật toán xử lý ảnh Cục bộ (Local) và Trí tuệ nhân tạo (AI via 9router).
"""
from __future__ import annotations

import logging
import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from openai import AsyncOpenAI
from app.core import settings

logger = logging.getLogger(__name__)

# ─── Tuning knobs ────────────────────────────────────────────────────────────
SAMPLE_COUNT = 12          # số frame lấy mẫu để phân tích cục bộ
AI_SAMPLE_COUNT = 3        # số frame gửi lên Multimodal AI để nhận diện
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
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _to_float(value: object, default: float = 0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _target_subtitle_height(height: float) -> int:
    return round(_clamp(height, 8, 18))


def _merge_logo_regions(regions: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for region in sorted(regions, key=lambda item: item["width_percent"] * item["height_percent"], reverse=True):
        duplicate_index = next(
            (index for index, existing in enumerate(merged) if _overlap_ratio(region, existing) > 0.45),
            None,
        )
        if duplicate_index is None:
            merged.append(region)
            continue

        existing = merged[duplicate_index]
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
        pad_y = max(1.5, height * 0.25)
        y -= pad_y
        height += pad_y * 2
        x = 0
        width = 100
    else:
        pad_x = max(1.0, width * 0.2)
        pad_y = max(1.0, height * 0.2)
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
        height = _clamp(height, 7, 28)
        if y + height > 98:
            y = 98 - height
        label = "Subtitle blur (AI checked)"
    else:
        center_x = x + width / 2
        center_y = y + height / 2
        near_corner = (
            (center_x <= 35 or center_x >= 65)
            and (center_y <= 35 or center_y >= 65)
        )
        top_watermark = center_y <= 35 and 15 <= center_x <= 85
        side_watermark = center_y <= 75 and (center_x <= 20 or center_x >= 80)
        if not (near_corner or top_watermark or side_watermark):
            return None
        width = _clamp(width, 4, 45)
        height = _clamp(height, 4, 25)
        label = "Logo/Watermark blur (AI checked)"

    return {
        "x_percent": round(x),
        "y_percent": round(y),
        "width_percent": round(width),
        "height_percent": round(height),
        "label": label,
        "kind": kind,
    }


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


def review_and_build_blur_config(regions: list[dict]) -> dict:
    """
    Sanity-check detections and convert them into render-ready settings.
    Subtitle detections become the main blur band; logo detections stay custom boxes.
    """
    normalized: list[dict] = []
    notes: list[str] = []

    for region in regions:
        kind = _region_kind(region)
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
        notes.append("Da can subtitle thanh mot dai mo ngang de che sach chu goc.")
    else:
        subtitle_region = {
            "x_percent": 0,
            "y_percent": 82,
            "width_percent": 100,
            "height_percent": 12,
            "label": "Subtitle blur safety band",
            "kind": "subtitle",
        }
        normalized.append(subtitle_region)
        notes.append("AI khong chac vung subtitle, dung dai mo an toan gon o day khung hinh.")

    subtitle_region["x_percent"] = 0
    subtitle_region["width_percent"] = 100
    subtitle_region["y_percent"] = round(_clamp(subtitle_region["y_percent"], 74, 90))
    subtitle_region["height_percent"] = _target_subtitle_height(subtitle_region["height_percent"])
    if subtitle_region["y_percent"] + subtitle_region["height_percent"] > 97:
        subtitle_region["y_percent"] = 97 - subtitle_region["height_percent"]

    custom_boxes: list[dict] = []
    for logo in _merge_logo_regions(logo_regions):
        if subtitle_region and _overlap_ratio(logo, subtitle_region) > 0.25:
            notes.append("Bo qua mot logo box vi trung voi dai subtitle.")
            continue
        if any(_overlap_ratio(logo, existing) > 0.6 for existing in custom_boxes):
            notes.append("Gop/bo bot logo box bi trung lap.")
            continue
        custom_boxes.append({
            "x_percent": logo["x_percent"],
            "y_percent": logo["y_percent"],
            "width_percent": logo["width_percent"],
            "height_percent": logo["height_percent"],
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

    for frame in frames:
        roi = frame[scan_y_start:scan_y_end, scan_x_start:scan_x_end]
        results = reader.readtext(roi, detail=1, paragraph=False)
        for (bbox, text, conf) in results:
            if conf < 0.4:
                continue
            if not _has_chinese(text):
                continue
            ys = [pt[1] for pt in bbox]
            y_top    = int(min(ys)) + scan_y_start
            y_bottom = int(max(ys)) + scan_y_start
            all_boxes_y.append(y_top)
            all_boxes_h.append(y_bottom - y_top)

    if not all_boxes_y:
        return None

    median_y = int(np.median(all_boxes_y))
    median_h = int(np.median(all_boxes_h)) if all_boxes_h else 40
    pad = max(8, int(median_h * 0.2))

    y_start = max(0, median_y - pad)
    box_h   = min(h - y_start, median_h + pad * 2)

    return {
        "x_percent":      0,
        "y_percent":      round(y_start / h * 100),
        "width_percent":  100,
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
        region = detect_watermark_region_local(video_path)
        if region:
            results.append(region)

    return results


# ─── Multimodal AI Engine ─────────────────────────────────────────────────────

async def auto_detect_blur_regions_ai(
    video_path: str,
    detect_sub: bool = True,
    detect_logo: bool = True,
) -> list[dict]:
    """
    Sử dụng Multimodal AI (Gemini 3.5 Flash qua 9router) để nhận diện vùng sub và logo từ các frame ảnh.
    """
    frames = _sample_frames(video_path, AI_SAMPLE_COUNT)
    if not frames:
        return []

    base64_frames: list[str] = []
    for frame in frames:
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))
        
        ok, buffer = cv2.imencode(".jpg", frame)
        if ok:
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

    if not base64_frames:
        return []

    prompt = (
        "You are a professional video editor assistant.\n"
        "Analyze these video frames and detect:\n"
    )
    if detect_sub:
        prompt += (
            "- The bounding box of Chinese hard subtitles. Subtitles are horizontally centered and located "
            "strictly at the bottom center of the screen (typically between y=75% and y=95%). "
            "CRITICAL: Ignore any vertical text or warnings running down the sides of the video.\n"
        )
    if detect_logo:
        prompt += (
            "- The bounding boxes of persistent source watermarks/logos. Include corner logos, platform marks, "
            "and creator watermark text near the top center/top third of the frame (for example Chinese text "
            "like 原创@...). Do not blur characters, credits, or normal scene text unless it is a persistent "
            "watermark repeated across frames.\n"
        )
        
    prompt += (
        "\nFor each detected item, calculate coordinates in percentages (0 to 100) relative to the frame size:\n"
        "- x_percent: horizontal offset of top-left corner.\n"
        "- y_percent: vertical offset of top-left corner.\n"
        "- width_percent: width of the region.\n"
        "- height_percent: height of the region.\n"
        "\nReturn ONLY a JSON object containing the regions list, like this:\n"
        "{\n"
        "  \"regions\": [\n"
        "    {\"x_percent\": 0, \"y_percent\": 80, \"width_percent\": 100, \"height_percent\": 15, \"label\": \"Sub chữ Trung (AI)\"}\n"
        "  ]\n"
        "}"
    )

    client = AsyncOpenAI(
        api_key=settings.ninerouter_api_key,
        base_url=settings.ninerouter_api_url,
    )

    content = [{"type": "text", "text": prompt}]
    for img_b64 in base64_frames:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}"
            }
        })

    logger.info("Đang gọi Multimodal AI 9router (ag/gemini-3.5-flash-low) để nhận diện vùng...")
    response = await client.chat.completions.create(
        model="ag/gemini-3.5-flash-low",
        messages=[
            {"role": "user", "content": content}
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    res_text = response.choices[0].message.content
    if not res_text:
        logger.warning("Không nhận được phản hồi từ AI.")
        return []

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
        for r in regions:
            if "label" not in r:
                r["label"] = "Vùng làm mờ (AI)"
            elif "sub" in r["label"].lower():
                r["label"] = "Sub chữ Trung (AI)"
                # Bắt buộc độ rộng phụ đề phủ hết chiều ngang màn hình để che sạch sub gốc
                r["x_percent"] = 0
                r["width_percent"] = 100
            elif "logo" in r["label"].lower() or "watermark" in r["label"].lower():
                r["label"] = "Logo/Watermark (AI)"
        logger.info("AI đã nhận diện thành công: %s", regions)
        return regions
    except Exception as e:
        logger.error("Lỗi parse kết quả từ AI: %s", e)
        logger.error("Nội dung phản hồi gốc: %s", res_text)
        return []
