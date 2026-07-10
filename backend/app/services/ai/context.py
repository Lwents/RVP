from __future__ import annotations

import asyncio
import base64
import json
import re
from pathlib import Path

from openai import AsyncOpenAI

from app.core import settings
from app.services.subtitles.timing import SubtitleEvent


async def analyze_video_context(
    video_file: Path | None,
    events: list[SubtitleEvent],
    work_dir: Path,
) -> dict:
    if not video_file or not video_file.exists() or not settings.ninerouter_api_key:
        return {}

    transcript = _compact_transcript(events)
    frames = _sample_video_frames(video_file)
    if not transcript and not frames:
        return {}

    client = AsyncOpenAI(
        api_key=settings.ninerouter_api_key,
        base_url=settings.ninerouter_api_url,
    )
    prompt = (
        "Bạn là chuyên gia nhận diện ngữ cảnh phim/donghua Trung Quốc để hỗ trợ dịch phụ đề tiếng Việt.\n"
        "Hãy xem frame và transcript ASR thô, rồi suy luận hồ sơ ngữ cảnh giúp dịch chuẩn hơn.\n\n"
        "Yêu cầu:\n"
        "- Nhận diện tên phim/series nếu thấy đủ dấu hiệu, nếu không chắc ghi null hoặc mô tả ngắn.\n"
        "- Liệt kê nhân vật/biệt danh/tên Hán Việt nếu nhận ra, vai trò và quan hệ với nhau.\n"
        "- Đưa quy tắc xưng hô tiếng Việt hợp bối cảnh: vua-thần, cha-con, huynh-đệ, sư đồ, nam-nữ.\n"
        "- Sửa các OCR/ASR tiếng Trung dễ sai thành glossary nguồn -> nghĩa Việt.\n"
        "- Không tự nâng quan hệ thành vợ/chồng/kết hôn/cưới trước yêu sau nếu transcript không có từ như 老婆, 妻子, 丈夫, 结婚, 婚约.\n"
        "- Với quan hệ chưa rõ, ghi trung tính: người phụ nữ, người đàn ông, cô ấy, hắn ta, người tình một đêm, nam chính/nữ chính.\n"
        "- Không bịa quá mức. Nếu không chắc, ghi confidence thấp và dùng ghi chú thận trọng.\n\n"
        "Trả về đúng JSON object với key:\n"
        "film_title, genre, setting, characters, relationships, glossary, honorific_rules, translation_notes, confidence.\n"
    )

    content: list[dict] = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": f"Transcript ASR thô:\n{transcript[:9000]}"},
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
        (work_dir / "ai_context.json").write_text(
            json.dumps(context, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return context


async def _create_context_completion_with_retries(client: AsyncOpenAI, content: list[dict], retries: int = 3):
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return await client.chat.completions.create(
                model="ag/gemini-3.5-flash-low",
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


def _compact_transcript(events: list[SubtitleEvent]) -> str:
    lines: list[str] = []
    for event in events[:180]:
        text = re.sub(r"\s+", " ", event.text).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _loads_json_object(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


_MARRIAGE_SOURCE_RE = re.compile(r"(老婆|妻子|夫人|丈夫|老公|结婚|婚姻|婚约|未婚妻|未婚夫|新娘|新郎|成亲|拜堂|夫妻|太太)")
_UNSUPPORTED_MARRIAGE_VI_RE = re.compile(
    r"(cưới trước yêu sau|vợ|chồng|kết hôn|hôn nhân|hôn ước|vị hôn|phu nhân|phu quân)",
    re.IGNORECASE,
)


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
    result = {key: data.get(key) for key in allowed if data.get(key) not in (None, "", [], {})}
    if not _MARRIAGE_SOURCE_RE.search(transcript):
        for key in ("genre", "relationships", "translation_notes", "setting"):
            if isinstance(result.get(key), str):
                result[key] = _UNSUPPORTED_MARRIAGE_VI_RE.sub("", result[key])
        result["relationship_constraints"] = (
            "Transcript chưa có dấu hiệu hôn nhân rõ ràng. Không dùng vợ/chồng/kết hôn/cưới trước yêu sau; "
            "hãy dùng người phụ nữ, người đàn ông, cô ấy, hắn ta, người tình một đêm nếu cần."
        )
    return result
