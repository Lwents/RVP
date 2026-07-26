from __future__ import annotations

import asyncio
import base64
import json
import re
from pathlib import Path

from openai import AsyncOpenAI

from app.core import settings
from app.services.ai.openai_client import get_async_openai
from app.services.subtitles.timing import SubtitleEvent


async def analyze_video_context(
    video_file: Path | None,
    events: list[SubtitleEvent],
    work_dir: Path,
) -> dict:
    if not video_file or not video_file.exists() or not settings.ninerouter_api_key:
        return {}

    context_file = work_dir / "ai_context.json"
    if context_file.is_file() and context_file.stat().st_size > 0:
        try:
            cached = json.loads(context_file.read_text(encoding="utf-8"))
            if isinstance(cached, dict):
                return cached
        except (OSError, json.JSONDecodeError):
            pass

    transcript = _compact_transcript(events)
    # Seeking/decoding/encoding frames with OpenCV blocks for seconds; keep it
    # off the event loop.
    frames = await asyncio.to_thread(_sample_video_frames, video_file)
    if not transcript and not frames:
        return {}

    client = get_async_openai(
        settings.ninerouter_api_key,
        settings.ninerouter_api_url,
        # _create_context_completion_with_retries already retries three times;
        # SDK retries on top of that would triple every failing request.
        max_retries=0,
    )
    prompt = (
        "Bạn là chuyên gia nhận diện ngữ cảnh phim thuộc mọi quốc gia để hỗ trợ dịch phụ đề và viết review tiếng Việt.\n"
        "Hãy xem frame và transcript ASR thô, rồi suy luận hồ sơ ngữ cảnh giúp dịch chuẩn hơn.\n\n"
        "Yêu cầu:\n"
        "- Nhận diện tên phim/series nếu thấy đủ dấu hiệu, nếu không chắc ghi null hoặc mô tả ngắn.\n"
        "- Mỗi nhân vật phải có đúng một name chuẩn để hiển thị xuyên suốt video; tuyệt đối không dịch nghĩa tên riêng.\n"
        "- Ưu tiên tên chính thức hoặc tên quen thuộc với khán giả Việt; ghi mọi tên gốc, phiên âm, biệt danh và lỗi ASR/OCR vào aliases.\n"
        "- Với Doraemon, dùng chính xác Chaien (không dùng Gian/Jaian) và Suneo (không dùng Xeko/Xê-kô).\n"
        "- characters ưu tiên dạng [{name, aliases, role, relationship}]; aliases luôn là danh sách, không trộn alias vào name.\n"
        "- Đưa quy tắc xưng hô tiếng Việt hợp bối cảnh: vua-thần, cha-con, huynh-đệ, sư đồ, nam-nữ.\n"
        "- Sửa các OCR/ASR tiếng Trung dễ sai thành glossary nguồn -> nghĩa Việt.\n"
        "- Chỉ kết luận quan hệ khi transcript hoặc hình ảnh có bằng chứng trực tiếp; hiểu cả từ đồng nghĩa và cách gọi đa ngôn ngữ, không phụ thuộc một danh sách từ khóa cố định.\n"
        "- Với quan hệ chưa rõ, ghi trung tính: người phụ nữ, người đàn ông, cô ấy, hắn ta, người tình một đêm, nam chính/nữ chính.\n"
        "- Không bịa quá mức. Nếu không chắc, ghi confidence thấp và dùng ghi chú thận trọng.\n\n"
        "Trả về đúng JSON object với key:\n"
        "film_title, genre, setting, characters, relationships, glossary, honorific_rules, translation_notes, confidence.\n"
    )

    content: list[dict] = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": f"Transcript ASR thô (mẫu xuyên suốt video):\n{transcript[:24000]}"},
    ]
    for image_b64 in frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            }
        )

    try:
        response = await _create_context_completion_with_retries(client, content)
        raw = response.choices[0].message.content or "{}"
        context = _loads_json_object(raw)
    except Exception as exc:
        print(f"AI context fallback: {exc}")
        return {}

    context = _sanitize_context(context, transcript)
    if context:
        context_file.write_text(
            json.dumps(context, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return context


async def _create_context_completion_with_retries(client: AsyncOpenAI, content: list[dict], retries: int = 3):
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return await client.chat.completions.create(
                model=settings.ai_model,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                temperature=0.2,
                timeout=90,
            )
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                await asyncio.sleep(min(2 * attempt, 6))
    assert last_error is not None
    raise last_error


def _sample_video_frames(video_file: Path, count: int = 4) -> list[str]:
    try:
        import cv2
    except ImportError:
        return []

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        return []

    positions = [int(total * ratio) for ratio in (0.08, 0.25, 0.55, 0.82)]
    frames: list[str] = []
    for pos in positions[:count]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(total - 1, pos)))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if w > 720:
            scale = 720 / w
            frame = cv2.resize(frame, (720, max(1, int(h * scale))))
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            frames.append(base64.b64encode(buffer).decode("utf-8"))
    cap.release()
    return frames


def _compact_transcript(events: list[SubtitleEvent], max_events: int = 240) -> str:
    """Sample coherent windows across the whole video, not only its opening."""
    if not events:
        return ""
    if len(events) <= max_events:
        selected = list(range(len(events)))
    else:
        window_count = 12
        per_window = max(4, max_events // window_count)
        selected_set: set[int] = set()
        for window in range(window_count):
            center = round(window * (len(events) - 1) / max(window_count - 1, 1))
            start = max(0, min(len(events) - per_window, center - per_window // 2))
            selected_set.update(range(start, min(len(events), start + per_window)))
        selected = sorted(selected_set)[:max_events]

    lines: list[str] = []
    for index in selected:
        text = re.sub(r"\s+", " ", events[index].text).strip()
        if text:
            lines.append(f"#{index + 1}: {text}")
    return "\n".join(lines)


def _loads_json_object(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _sanitize_context(data: object, transcript: str = "") -> dict:
    if not isinstance(data, dict):
        return {}
    allowed = {
        "film_title",
        "genre",
        "setting",
        "characters",
        "relationships",
        "glossary",
        "honorific_rules",
        "translation_notes",
        "confidence",
    }
    return {key: data.get(key) for key in allowed if data.get(key) not in (None, "", [], {})}
