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
            "- The bounding box of channel watermark logo (located strictly in one of the four extreme corners, "
            "very close to the edges. Do not blur normal text, characters, or credits).\n"
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
