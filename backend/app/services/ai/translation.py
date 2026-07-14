"""Contextual subtitle translation through the project's configured AI model.

The active engine translates ordered batches directly, preserving timestamps,
Unicode NFC, character identity, relationships, glossary terms, and a rolling
memory across long videos. Legacy Google helpers remain isolated for backwards
compatibility tests but are never selected by ``get_translation_engine``.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from app.core import settings
from app.models.job import ProcessingMode
from app.services.presets import ProcessingProfile, get_processing_profile
from app.services.subtitles.timing import SubtitleEvent


class TranslationError(RuntimeError):
    pass


# ─── Engine classes ───────────────────────────────────────────────────────────

class TranslationEngine:
    async def translate_events(
        self,
        events: list[SubtitleEvent],
        source_language: str,
        target_language: str,
        context: dict | None = None,
        *,
        processing_mode: ProcessingMode | str | None = None,
        checkpoint_file: Path | None = None,
        on_batch_progress: Callable[[int, int], None] | None = None,
    ) -> list[SubtitleEvent]:
        raise NotImplementedError


class PassthroughTranslation(TranslationEngine):
    async def translate_events(
        self,
        events: list[SubtitleEvent],
        source_language: str,
        target_language: str,
        context: dict | None = None,
        *,
        processing_mode: ProcessingMode | str | None = None,
        checkpoint_file: Path | None = None,
        on_batch_progress: Callable[[int, int], None] | None = None,
    ) -> list[SubtitleEvent]:
        return events


class GeminiTranslation(TranslationEngine):
    """Translate directly with the configured contextual AI model.

    Batches are intentionally processed in order. Each request receives the
    global character bible, a small overlap around the current batch, and
    memory returned by the preceding batches. This keeps names, relationships,
    gendered pronouns, and forms of address stable across a long video.
    """

    async def translate_events(
        self,
        events: list[SubtitleEvent],
        source_language: str,
        target_language: str,
        context: dict | None = None,
        *,
        processing_mode: ProcessingMode | str | None = None,
        checkpoint_file: Path | None = None,
        on_batch_progress: Callable[[int, int], None] | None = None,
    ) -> list[SubtitleEvent]:
        if not events:
            return []
        if source_language != "auto" and source_language.lower() == target_language.lower():
            return events
        if not settings.ninerouter_api_key:
            raise TranslationError(
                "Chưa cấu hình API key 9router nên AI không thể dịch phụ đề."
            )

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise TranslationError("Thiếu thư viện openai để gọi AI dịch phụ đề.") from exc

        client = AsyncOpenAI(
            api_key=settings.ninerouter_api_key,
            base_url=settings.ninerouter_api_url,
            # Retry is controlled below so one failing batch cannot silently
            # multiply the SDK retries by the pipeline retries.
            max_retries=0,
        )
        profile = get_processing_profile(processing_mode)
        source_texts = [
            unicodedata.normalize("NFC", _normalize_source_text(event.text.strip()))
            for event in events
        ]
        ranges = _ai_translation_batch_ranges(
            source_texts,
            max_items=profile.translation_batch_items,
            max_chars=profile.translation_batch_chars,
        )
        fingerprint = _translation_fingerprint(
            source_texts,
            source_language,
            target_language,
            context,
            profile,
        )
        translated_texts, memory = _load_translation_checkpoint(
            checkpoint_file,
            fingerprint,
            context,
            ranges,
        )
        completed_batches = sum(end <= len(translated_texts) for _, end in ranges)
        if on_batch_progress:
            on_batch_progress(completed_batches, len(ranges))

        started_at = time.monotonic()
        total_budget = max(
            profile.translation_request_timeout_seconds,
            len(ranges) * profile.translation_batch_budget_seconds,
        )

        for batch_index, (start, end) in enumerate(ranges, start=1):
            if end <= len(translated_texts):
                continue
            if start != len(translated_texts):
                raise TranslationError(
                    "Checkpoint dịch không nằm đúng ranh giới batch; hãy xóa checkpoint và chạy lại."
                )
            elapsed = time.monotonic() - started_at
            if elapsed >= total_budget:
                raise TranslationError(
                    f"Dịch vượt quá ngân sách {int(total_budget)} giây; "
                    f"đã lưu checkpoint {completed_batches}/{len(ranges)} batch để chạy tiếp."
                )
            batch = [
                {"id": index + 1, "source": source_texts[index]}
                for index in range(start, end)
            ]
            previous_context = [
                {
                    "id": index + 1,
                    "source": source_texts[index],
                    "translated": translated_texts[index],
                }
                for index in range(max(0, start - 8), start)
            ]
            next_context = [
                {"id": index + 1, "source": source_texts[index]}
                for index in range(end, min(len(source_texts), end + 4))
            ]
            batch_texts, memory_updates = await _translate_gemini_batch_with_retries(
                client,
                batch,
                source_language,
                target_language,
                context,
                memory,
                previous_context,
                next_context,
                retries=profile.translation_retries,
                request_timeout_seconds=profile.translation_request_timeout_seconds,
            )

            for item, translated in zip(batch, batch_texts):
                source = str(item["source"])
                cleaned = _clean_polished_subtitle(translated, "")
                cleaned = unicodedata.normalize("NFC", cleaned)
                if not cleaned or (
                    target_language.lower() == "vi" and _looks_chinese(cleaned)
                ):
                    raise TranslationError(
                        f"AI còn sót nội dung chưa dịch ở mục {item['id']}: {source[:120]}"
                    )
                translated_texts.append(cleaned)

            _merge_translation_memory(memory, memory_updates)
            memory["recent_translations"] = previous_context[-4:] + [
                {
                    "id": item["id"],
                    "source": item["source"],
                    "translated": translated,
                }
                for item, translated in zip(batch[-8:], batch_texts[-8:])
            ]
            completed_batches = batch_index
            _write_translation_checkpoint(
                checkpoint_file,
                fingerprint,
                translated_texts,
                memory,
                len(source_texts),
            )
            if on_batch_progress:
                on_batch_progress(completed_batches, len(ranges))

        if len(translated_texts) != len(events):
            raise TranslationError("AI trả về thiếu câu dịch so với phụ đề nguồn.")
        return [
            SubtitleEvent(event.start, event.end, translated_texts[index])
            for index, event in enumerate(events)
        ]


class GoogleTranslation(TranslationEngine):
    async def translate_events(
        self,
        events: list[SubtitleEvent],
        source_language: str,
        target_language: str,
        context: dict | None = None,
        *,
        processing_mode: ProcessingMode | str | None = None,
        checkpoint_file: Path | None = None,
        on_batch_progress: Callable[[int, int], None] | None = None,
    ) -> list[SubtitleEvent]:
        if not events:
            return []
        if source_language != "auto" and source_language.lower() == target_language.lower():
            return events

        def translate() -> list[SubtitleEvent]:
            all_texts = [_normalize_source_text(e.text.strip()) for e in events]

            # Xây dựng bản đồ proper nouns từ toàn bộ phụ đề
            proper_noun_map = _build_proper_noun_map(all_texts, source_language)

            # Bảo vệ proper nouns + glossary + interjections
            protected_texts = [_protect_all(t, proper_noun_map) for t in all_texts]

            # Dịch theo batch
            translated_texts: list[str] = []
            for batch in _translation_batches(protected_texts):
                try:
                    translated_texts.extend(_translate_batch(batch, source_language, target_language))
                except Exception as exc:
                    recovered = [
                        _translate_single_or_original(t, source_language, target_language)
                        for t in batch
                    ]
                    if not any(item and item != orig for item, orig in zip(recovered, batch)):
                        raise TranslationError(
                            f"Không dịch được phụ đề bằng Google Translate: {exc}"
                        ) from exc
                    translated_texts.extend(recovered)

            # Sửa các câu còn sót chưa dịch
            translated_texts = _repair_untranslated_lines(
                protected_texts, translated_texts, source_language, target_language
            )

            # Khôi phục proper nouns + glossary
            translated_texts = [_restore_all(t, proper_noun_map) for t in translated_texts]

            # Post-processing sửa lỗi dịch Google sang tiếng Việt
            if target_language.lower() == "vi":
                translated_texts = [_fix_vietnamese_output(t, all_texts[i]) for i, t in enumerate(translated_texts)]
                translated_texts = [_fix_character_names(t) for t in translated_texts]

            return [
                SubtitleEvent(event.start, event.end, translated_texts[i] or event.text)
                for i, event in enumerate(events)
            ]

        translated_events = await asyncio.to_thread(translate)
        if target_language.lower() == "vi":
            translated_events = await _polish_vietnamese_events_with_ai(
                events,
                translated_events,
                source_language,
                context,
            )
        return translated_events


def get_translation_engine() -> TranslationEngine:
    if settings.translation_engine.lower() == "passthrough":
        return PassthroughTranslation()
    # "google" is accepted as a backwards-compatible configuration value so
    # existing .env files immediately stop using the rate-limited unofficial
    # Google endpoint. All real translation now goes directly through AI.
    return GeminiTranslation()


def _ai_translation_batch_ranges(
    texts: list[str],
    max_items: int = 20,
    max_chars: int = 4600,
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    current_chars = 0
    for index, text in enumerate(texts):
        item_chars = len(text) + 60
        if index > start and (
            index - start >= max_items or current_chars + item_chars > max_chars
        ):
            ranges.append((start, index))
            start = index
            current_chars = 0
        current_chars += item_chars
    if start < len(texts):
        ranges.append((start, len(texts)))
    return ranges


def _translation_fingerprint(
    source_texts: list[str],
    source_language: str,
    target_language: str,
    context: dict | None,
    profile: ProcessingProfile,
) -> str:
    payload = {
        "source_texts": source_texts,
        "source_language": source_language,
        "target_language": target_language,
        "context": context or {},
        "model": settings.ai_model,
        "processing_mode": profile.name,
        "batch_items": profile.translation_batch_items,
        "batch_chars": profile.translation_batch_chars,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_translation_checkpoint(
    checkpoint_file: Path | None,
    fingerprint: str,
    context: dict | None,
    ranges: list[tuple[int, int]],
) -> tuple[list[str], dict]:
    fallback = ([], _initial_translation_memory(context))
    if checkpoint_file is None or not checkpoint_file.is_file():
        return fallback
    try:
        data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    if not isinstance(data, dict) or data.get("fingerprint") != fingerprint:
        return fallback
    translated = data.get("translated_texts")
    memory = data.get("memory")
    if not isinstance(translated, list) or not all(isinstance(item, str) for item in translated):
        return fallback
    if not isinstance(memory, dict):
        return fallback
    valid_offsets = {0, *(end for _, end in ranges)}
    if len(translated) not in valid_offsets:
        return fallback
    return list(translated), memory


def _write_translation_checkpoint(
    checkpoint_file: Path | None,
    fingerprint: str,
    translated_texts: list[str],
    memory: dict,
    total_items: int,
) -> None:
    if checkpoint_file is None:
        return
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "fingerprint": fingerprint,
        "model": settings.ai_model,
        "translated_items": len(translated_texts),
        "total_items": total_items,
        "completed": len(translated_texts) == total_items,
        "translated_texts": translated_texts,
        "memory": memory,
    }
    temporary = checkpoint_file.with_suffix(f"{checkpoint_file.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(checkpoint_file)


def _initial_translation_memory(context: dict | None) -> dict:
    return {
        "character_bible": _compact_context(context),
        "characters": [],
        "relationships": [],
        "glossary": {},
        "addressing_rules": [],
        "recent_translations": [],
    }


def _merge_translation_memory(memory: dict, updates: object) -> None:
    if not isinstance(updates, dict):
        return
    for key in ("characters", "relationships", "addressing_rules"):
        incoming = updates.get(key)
        if not isinstance(incoming, list):
            continue
        existing = memory.setdefault(key, [])
        if not isinstance(existing, list):
            existing = []
            memory[key] = existing
        seen = {json.dumps(item, ensure_ascii=False, sort_keys=True) for item in existing}
        for item in incoming[:30]:
            encoded = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if encoded not in seen:
                existing.append(item)
                seen.add(encoded)
        del existing[80:]

    incoming_glossary = updates.get("glossary")
    if isinstance(incoming_glossary, dict):
        glossary = memory.setdefault("glossary", {})
        if not isinstance(glossary, dict):
            glossary = {}
            memory["glossary"] = glossary
        for source, translated in list(incoming_glossary.items())[:50]:
            source_text = str(source).strip()
            translated_text = str(translated).strip()
            if source_text and translated_text:
                glossary[source_text] = translated_text
        if len(glossary) > 120:
            for key in list(glossary)[:-120]:
                glossary.pop(key, None)


async def _translate_gemini_batch(
    client,
    batch: list[dict[str, object]],
    source_language: str,
    target_language: str,
    context: dict | None,
    memory: dict,
    previous_context: list[dict[str, object]],
    next_context: list[dict[str, object]],
    request_timeout_seconds: float = 150,
) -> tuple[list[str], dict]:
    system_prompt = (
        "Bạn là biên dịch viên phụ đề phim nhiều tập, chuyên giữ nhất quán nhân vật và ngữ cảnh. "
        "Hãy dịch TRỰC TIẾP từ ngôn ngữ nguồn sang tiếng Việt tự nhiên; không dùng bản dịch máy trung gian.\n\n"
        "Quy tắc bắt buộc:\n"
        "- Trả đúng JSON object gồm items và memory_updates.\n"
        "- items phải giữ nguyên đủ id, đúng thứ tự và số lượng của CURRENT_ITEMS; không gộp/tách/thêm câu.\n"
        "- Mỗi item có dạng {\"id\":1,\"text\":\"...\"}; text chỉ chứa bản dịch, không giải thích.\n"
        "- Giữ nguyên ý, không bịa tình tiết, không thêm tên người nói nếu câu nguồn không có.\n"
        "- Dùng CHARACTER_BIBLE và MEMORY làm sự thật ưu tiên cho tên, biệt danh, giới tính, quan hệ và xưng hô.\n"
        "- Một nhân vật phải giữ cùng tên Hán Việt/Việt hóa trong toàn bộ video. Không đổi tên giữa các batch.\n"
        "- Chỉ xác định giới tính, vai vế hoặc quan hệ khi câu nguồn/ngữ cảnh có bằng chứng; chưa chắc thì dùng cách gọi trung tính.\n"
        "- PREVIOUS_CONTEXT và NEXT_CONTEXT chỉ giúp hiểu đại từ/người nói; tuyệt đối không dịch chúng vào items.\n"
        "- Câu ngắn, tự nhiên, phù hợp thời lượng phụ đề/lồng tiếng; không để sót chữ Trung, pinyin hoặc chú thích.\n"
        "- memory_updates chỉ ghi thông tin nhân vật mới được CURRENT_ITEMS xác nhận, gồm characters, relationships, glossary, addressing_rules.\n"
    )
    payload = {
        "source_language": source_language,
        "target_language": target_language,
        "character_bible": _compact_context(context),
        "memory": memory,
        "previous_context": previous_context,
        "current_items": batch,
        "next_context": next_context,
    }
    response = await client.chat.completions.create(
        model=settings.ai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.15,
        timeout=request_timeout_seconds,
    )
    content = response.choices[0].message.content or ""
    data = _loads_json_object(content)
    items = data.get("items")
    if not isinstance(items, list):
        raise TranslationError("AI không trả về danh sách items.")

    expected_ids = [int(item["id"]) for item in batch]
    by_id: dict[int, str] = {}
    duplicate_ids: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            item_id = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if item_id in by_id:
            duplicate_ids.add(item_id)
        text = unicodedata.normalize("NFC", str(item.get("text") or "").strip())
        if text:
            by_id[item_id] = text
    missing = [item_id for item_id in expected_ids if item_id not in by_id]
    unexpected = [item_id for item_id in by_id if item_id not in expected_ids]
    if missing or unexpected or duplicate_ids or len(items) != len(batch):
        raise TranslationError(
            "AI trả sai cấu trúc batch "
            f"(thiếu={missing}, thừa={unexpected}, trùng={sorted(duplicate_ids)})."
        )
    translated = [by_id[item_id] for item_id in expected_ids]
    if target_language.lower() == "vi" and any(_looks_chinese(text) for text in translated):
        raise TranslationError("AI còn để sót chữ Trung trong batch dịch.")
    updates = data.get("memory_updates")
    return translated, updates if isinstance(updates, dict) else {}


async def _translate_gemini_batch_with_retries(
    client,
    batch: list[dict[str, object]],
    source_language: str,
    target_language: str,
    context: dict | None,
    memory: dict,
    previous_context: list[dict[str, object]],
    next_context: list[dict[str, object]],
    retries: int = 3,
    request_timeout_seconds: float = 150,
) -> tuple[list[str], dict]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return await asyncio.wait_for(
                _translate_gemini_batch(
                    client,
                    batch,
                    source_language,
                    target_language,
                    context,
                    memory,
                    previous_context,
                    next_context,
                    request_timeout_seconds=request_timeout_seconds,
                ),
                timeout=request_timeout_seconds + 5,
            )
        except TimeoutError as exc:
            last_error = TranslationError(
                f"9router không phản hồi trong {int(request_timeout_seconds)} giây"
            )
        except Exception as exc:
            last_error = exc
        if attempt < retries:
            await asyncio.sleep(min(2 * attempt, 6))
    assert last_error is not None
    first_id = batch[0]["id"] if batch else "?"
    last_id = batch[-1]["id"] if batch else "?"
    raise TranslationError(
        f"AI {settings.ai_model} không dịch được batch {first_id}-{last_id}: {last_error}"
    ) from last_error


# ─── AI Vietnamese subtitle editor ───────────────────────────────────────────

async def _polish_vietnamese_events_with_ai(
    source_events: list[SubtitleEvent],
    draft_events: list[SubtitleEvent],
    source_language: str,
    context: dict | None = None,
) -> list[SubtitleEvent]:
    if not draft_events or not settings.ninerouter_api_key:
        return draft_events

    try:
        from openai import AsyncOpenAI
    except ImportError:
        return draft_events

    polished: list[SubtitleEvent] = []
    failed_batches = 0
    client = AsyncOpenAI(
        api_key=settings.ninerouter_api_key,
        base_url=settings.ninerouter_api_url,
    )

    for batch in _ai_polish_batches(source_events, draft_events):
        try:
            edited = await _polish_vietnamese_batch_with_retries(client, batch, source_language, context)
        except Exception as exc:
            print(f"AI subtitle polish fallback: {exc}")
            failed_batches += 1
            edited = [item["draft"] for item in batch]

        for item, text in zip(batch, edited):
            cleaned = _clean_polished_subtitle(text, item["draft"])
            polished.append(
                SubtitleEvent(
                    item["start"],
                    item["end"],
                    _fix_character_names(cleaned),
                )
            )

    if len(polished) != len(draft_events):
        return draft_events
    if failed_batches and _source_requires_ai_polish(source_events):
        raise TranslationError(
            "AI biên tập phụ đề không chạy được nên không dùng bản dịch thô. "
            "Hãy bật 9router/AI ở localhost:20128 rồi chạy lại."
        )
    return polished


def _ai_polish_batches(
    source_events: list[SubtitleEvent],
    draft_events: list[SubtitleEvent],
    max_items: int = 18,
    max_chars: int = 5200,
) -> list[list[dict[str, object]]]:
    batches: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    current_chars = 0

    for index, (source, draft) in enumerate(zip(source_events, draft_events), start=1):
        source_text = _normalize_source_text(source.text.strip())
        draft_text = draft.text.strip()
        item = {
            "id": index,
            "start": draft.start,
            "end": draft.end,
            "source": source_text,
            "draft": draft_text,
        }
        item_chars = len(source_text) + len(draft_text) + 80
        if current and (len(current) >= max_items or current_chars + item_chars > max_chars):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(item)
        current_chars += item_chars

    if current:
        batches.append(current)
    return batches


async def _polish_vietnamese_batch(
    client,
    batch: list[dict[str, object]],
    source_language: str,
    context: dict | None,
) -> list[str]:
    system_prompt = (
        "Bạn là biên tập viên phụ đề/lồng tiếng Việt chuyên review phim cổ trang Trung Quốc, donghua. "
        "Nhiệm vụ: sửa bản dịch nháp để câu thoại tự nhiên, đúng xưng hô và dễ đọc khi lồng tiếng.\n\n"
        "Quy tắc bắt buộc:\n"
        "- Trả về đúng JSON object: {\"items\":[{\"id\":1,\"text\":\"...\"}]}.\n"
        "- Giữ nguyên số lượng item và id. Không thêm, xoá, gộp, tách dòng.\n"
        "- Không thêm sự kiện mới, không bịa nội dung ngoài câu gốc.\n"
        "- Câu ngắn, hợp phụ đề và giọng đọc; ưu tiên 1-2 câu mỗi item.\n"
        "- Dùng xưng hô cổ trang hợp ngữ cảnh: bệ hạ, thần, phụ hoàng, phụ thân, huynh, tỷ, công tử, đại nhân.\n"
        "- Nếu là vua/hoàng đế nói với quần thần dùng 'trẫm/khanh'; thần tử nói với vua dùng 'thần/bệ hạ'.\n"
        "- Không đổi quan hệ. Tuyệt đối không dịch thành vợ/chồng/kết hôn nếu câu gốc không có 老婆, 妻子, 丈夫, 结婚, 婚约 hoặc nghĩa hôn nhân rõ ràng.\n"
        "- Nếu quan hệ chưa rõ, dùng trung tính: người phụ nữ, người đàn ông, cô ấy, hắn ta, người tình một đêm, nam chính/nữ chính.\n"
        "- Giữ tên riêng nhất quán theo Hán Việt nếu nhận ra: Chu Nguyên Chương, Chu Kỳ Ngọc, Hồng Vũ, Vĩnh Lạc, Mặc Nguyệt, Tần ca, Long tỷ.\n"
        "- Các cụm bị ASR/OCR sai phải sửa theo nghĩa: 照办 = làm theo, 照常 = như thường, 父亲 = phụ thân.\n"
        "- Nếu gặp 海繁的动物/还能翻得动不 hoặc 翻得动: hiểu là 'còn nhào lộn nổi không', tuyệt đối không dịch thành động vật/biển.\n"
        "- Nếu phần CONTEXT có tên phim, nhân vật, quan hệ, vai vế hoặc glossary thì ưu tiên dùng để sửa tên và xưng hô.\n"
        "- Không để sót tiếng Trung, pinyin thô hoặc tên sai kiểu Zhaoban/Zhan Xiang/Qixia/Mo Yue nếu có thể sửa.\n"
    )
    user_prompt = {
        "source_language": source_language,
        "context": _compact_context(context),
        "items": [
            {
                "id": item["id"],
                "source": item["source"],
                "draft_vi": item["draft"],
            }
            for item in batch
        ],
    }
    response = await client.chat.completions.create(
        model=settings.ai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.25,
        timeout=90,
    )
    content = response.choices[0].message.content or ""
    data = _loads_json_object(content)
    items = data.get("items")
    if not isinstance(items, list):
        raise TranslationError("AI polish không trả về items.")

    by_id: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            item_id = int(item.get("id"))
        except Exception:
            continue
        text = str(item.get("text") or "").strip()
        if text:
            by_id[item_id] = text

    result: list[str] = []
    for item in batch:
        item_id = int(item["id"])
        result.append(by_id.get(item_id, str(item["draft"])))
    return result


async def _polish_vietnamese_batch_with_retries(
    client,
    batch: list[dict[str, object]],
    source_language: str,
    context: dict | None,
    retries: int = 3,
) -> list[str]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return await _polish_vietnamese_batch(client, batch, source_language, context)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                await asyncio.sleep(min(2 * attempt, 6))
    assert last_error is not None
    raise last_error


def _source_requires_ai_polish(events: list[SubtitleEvent]) -> bool:
    joined = " ".join(event.text for event in events[:80])
    return _looks_chinese(joined)


def _compact_context(context: dict | None) -> dict:
    if not isinstance(context, dict):
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
        "relationship_constraints",
    }
    compact = {key: context.get(key) for key in allowed if context.get(key)}
    return compact


def _loads_json_object(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _clean_polished_subtitle(text: str, fallback: str) -> str:
    cleaned = _clean_translation(text)
    cleaned = cleaned.replace("\\N", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\n\"'")
    cleaned = re.sub(r"\.{4,}", "...", cleaned)
    cleaned = re.sub(r"\s+([.,!?:;])", r"\1", cleaned)
    cleaned = re.sub(r",([^\s])", r", \1", cleaned)
    if not cleaned or _looks_chinese(cleaned):
        return fallback
    return cleaned


# ─── Google Translate API ─────────────────────────────────────────────────────

def _google_source(language: str) -> str:
    if language == "auto":
        return "auto"
    if language == "zh":
        return "zh-CN"
    return language


def _google_target(language: str) -> str:
    if language == "zh":
        return "zh-CN"
    return language


def _clean_translation(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.replace("\n", " ").split())


def _request_google_translate(
    text: str, source_language: str, target_language: str, timeout: int
) -> object:
    import requests

    params = {
        "client": "gtx",
        "sl": _google_source(source_language),
        "tl": _google_target(target_language),
        "dt": "t",
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
    }
    response = requests.post(
        "https://translate.googleapis.com/translate_a/single",
        params=params,
        data={"q": text},
        headers=headers,
        timeout=timeout,
    )
    if response.status_code == 400 and source_language == "auto" and _looks_chinese(text):
        response = requests.post(
            "https://translate.googleapis.com/translate_a/single",
            params={**params, "sl": "zh-CN"},
            data={"q": text},
            headers=headers,
            timeout=timeout,
        )
    response.raise_for_status()
    return response.json()


def _payload_text(payload: object) -> str:
    if not isinstance(payload, list) or not payload:
        return ""
    segments = payload[0]
    if not isinstance(segments, list):
        return ""
    return "".join(
        part[0] for part in segments if isinstance(part, list) and part and part[0]
    )


def _looks_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _needs_retranslation(text: str, target_language: str) -> bool:
    return target_language.lower() != "zh" and _looks_chinese(text)


# ─── Glossary cố định (thuật ngữ kỹ thuật / thương hiệu) ─────────────────────

_GLOSSARY: dict[str, str] = {
    "ZXQTERM0ZXQ": "AntiGravity",
    "ZXQTERM1ZXQ": "Cursor",
    "ZXQTERM2ZXQ": "Windsurf",
}

_GLOSSARY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\banti\s*gravity\b", re.IGNORECASE), "ZXQTERM0ZXQ"),
    (re.compile(r"\bcursor\b", re.IGNORECASE), "ZXQTERM1ZXQ"),
    (re.compile(r"\bwindsurf\b", re.IGNORECASE), "ZXQTERM2ZXQ"),
]


def _protect_glossary(text: str) -> str:
    for pattern, placeholder in _GLOSSARY_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def _restore_glossary(text: str) -> str:
    restored = text
    for placeholder, term in _GLOSSARY.items():
        restored = restored.replace(placeholder, term)
        restored = restored.replace(placeholder.lower(), term)
        restored = restored.replace(placeholder.title(), term)
    return restored


# Các lỗi ASR/OCR tiếng Trung hay làm sai nghĩa hoặc biến cụm từ thành tên riêng.
_SOURCE_TEXT_FIXES: list[tuple[re.Pattern, str]] = [
    (re.compile("臣定当赵半"), "臣定当照办"),
    (re.compile("定当赵半"), "定当照办"),
    (re.compile("赵半"), "照办"),
    (re.compile("战相"), "照常"),
    (re.compile("太阳照常"), "太阳照常"),
    (re.compile("不清"), "父亲"),
    (re.compile("不苦"), "不哭"),
    (re.compile("莫苦"), "莫哭"),
    (re.compile("一莫哭"), "莫哭"),
    (re.compile("初期"), "朱祁钰"),
    (re.compile("朱言此精花此数"), "朱颜辞镜花辞树"),
    (re.compile("尊是人间留不住"), "最是人间留不住"),
    (re.compile("朱言"), "朱元璋"),
    (re.compile("红武朝"), "洪武朝"),
    (re.compile("从西朝"), "正统朝"),
    (re.compile("有垃圾的"), "永乐朝的"),
    (re.compile("墨月"), "墨月"),
    (re.compile("七夏山"), "栖霞山"),
    (re.compile("秦哥"), "秦哥"),
    (re.compile("龙姐"), "龙姐"),
    (re.compile("海繁的动物"), "还能翻得动不"),
    (re.compile("繁的动"), "翻得动"),
    (re.compile("能连繁"), "能连翻"),
    (re.compile("能繁多少"), "能翻多少"),
    (re.compile("不插气"), "不喘气"),
]


def _normalize_source_text(text: str) -> str:
    if not text:
        return text
    normalized = text
    if _looks_chinese(normalized):
        for pattern, replacement in _SOURCE_TEXT_FIXES:
            normalized = pattern.sub(replacement, normalized)
    return normalized


# ─── Interjections: giữ nguyên không dịch ────────────────────────────────────

# Những từ cảm thán mà Google Translate hay dịch sai
_INTERJECTIONS: dict[str, str] = {
    "ZXQIJ0ZXQ": "Oh",
    "ZXQIJ1ZXQ": "Wow",
    "ZXQIJ2ZXQ": "Hey",
    "ZXQIJ3ZXQ": "Uh",
    "ZXQIJ4ZXQ": "Um",
    "ZXQIJ5ZXQ": "Hmm",
    "ZXQIJ6ZXQ": "Huh",
    "ZXQIJ7ZXQ": "Ugh",
    "ZXQIJ8ZXQ": "Oops",
}

_INTERJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bOh\b", re.IGNORECASE), "ZXQIJ0ZXQ"),
    (re.compile(r"\bWow\b", re.IGNORECASE), "ZXQIJ1ZXQ"),
    (re.compile(r"\bHey\b", re.IGNORECASE), "ZXQIJ2ZXQ"),
    (re.compile(r"\bUh\b", re.IGNORECASE), "ZXQIJ3ZXQ"),
    (re.compile(r"\bUm\b", re.IGNORECASE), "ZXQIJ4ZXQ"),
    (re.compile(r"\bHmm\b", re.IGNORECASE), "ZXQIJ5ZXQ"),
    (re.compile(r"\bHuh\b", re.IGNORECASE), "ZXQIJ6ZXQ"),
    (re.compile(r"\bUgh\b", re.IGNORECASE), "ZXQIJ7ZXQ"),
    (re.compile(r"\bOops\b", re.IGNORECASE), "ZXQIJ8ZXQ"),
]


def _protect_interjections(text: str) -> str:
    for pattern, placeholder in _INTERJECTION_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def _restore_interjections(text: str) -> str:
    for placeholder, word in _INTERJECTIONS.items():
        text = text.replace(placeholder, word)
        text = text.replace(placeholder.lower(), word)
    return text


# ─── Proper noun detection ────────────────────────────────────────────────────

# Từ tiếng Anh thông dụng viết hoa đầu câu — không phải proper noun
_COMMON_WORDS = frozenset({
    # Pronouns & articles
    "i", "a", "the", "in", "on", "at", "by", "to", "of", "or", "and", "but",
    "so", "yet", "for", "nor", "an", "is", "it", "he", "she", "we", "they",
    "you", "my", "your", "his", "her", "our", "its", "this", "that", "these",
    "those", "what", "who", "how", "when", "where", "why", "which",
    # Common verbs
    "do", "did", "does", "be", "are", "was", "were", "will", "would", "could",
    "should", "may", "might", "must", "have", "has", "had", "get", "got",
    "go", "come", "see", "say", "know", "think", "look", "want", "give",
    "use", "find", "tell", "ask", "seem", "feel", "try", "leave", "call",
    "keep", "let", "begin", "show", "hear", "play", "run", "move", "live",
    "believe", "hold", "bring", "happen", "write", "provide", "stand", "lose",
    "pay", "meet", "include", "continue", "set", "learn", "change", "lead",
    "understand", "watch", "follow", "stop", "create", "speak", "read",
    "spend", "grow", "open", "walk", "win", "offer", "remember", "love",
    "consider", "appear", "wait", "serve", "die", "send", "build", "stay",
    "fall", "cut", "reach", "kill", "remain", "suggest", "raise", "pass",
    "sell", "require", "report", "decide", "pull", "need", "put", "help",
    "make", "take", "came", "went", "said", "told", "asked", "knew", "thought",
    # Common adjectives/adverbs
    "yes", "no", "ok", "okay", "please", "sorry", "thank", "thanks",
    "hello", "hi", "bye", "well", "just", "now", "not", "only", "even",
    "also", "still", "again", "never", "always", "maybe", "really", "very",
    "much", "more", "most", "all", "any", "some", "here", "there", "then",
    "than", "because", "if", "when", "while", "before", "after", "until",
    "though", "although", "since", "unless",
    # Titles (chỉ tiêu đề, không phải tên)
    "mr", "mrs", "ms", "dr", "sir", "ma", "miss",
    # Numbers
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "first", "second", "third", "last", "next",
    # Tiếng Trung romanized phổ biến
    "ni", "wo", "ta", "men", "de", "le", "zhe", "na", "ba", "ma", "ne",
    # Tiếng Hàn romanized phổ biến
    "nae", "neo", "uri", "ige", "mwo", "wae", "eodi",
})

# Pattern: từ viết hoa chữ cái đầu (Latin), bao gồm tên ghép kiểu "Ji-ho", "Xiao Ming"
_CAPITALIZED_WORD_RE = re.compile(
    r"\b([A-Z][a-zA-Z]{1,24}(?:[-'][A-Z][a-zA-Z]{1,24})*)\b"
)

# Vị trí đầu câu (sau dấu câu kết thúc hoặc đầu text)
_SENTENCE_END_RE = re.compile(r"[.!?…。！？]\s+")
_LINE_START_RE = re.compile(r"(?:^|\n)\s*")


def _sentence_start_positions(text: str) -> set[int]:
    """Tập hợp các vị trí ký tự đầu tiên của mỗi câu."""
    positions: set[int] = set()
    # Đầu text
    m = re.search(r"\S", text)
    if m:
        positions.add(m.start())
    # Sau dấu câu
    for m in _SENTENCE_END_RE.finditer(text):
        nm = re.search(r"\S", text[m.end():])
        if nm:
            positions.add(m.end() + nm.start())
    # Sau xuống dòng
    for m in re.finditer(r"\n\s*\S", text):
        positions.add(m.end() - 1)
    return positions


def _build_proper_noun_map(
    all_texts: list[str],
    source_language: str,
) -> dict[str, str]:
    """
    Phân tích thống kê toàn bộ phụ đề để phát hiện proper nouns.

    Chiến lược:
    - Từ viết hoa xuất hiện ≥ 2 lần TỔNG CỘNG → bảo vệ (có thể là tên nhân vật)
    - Từ viết hoa xuất hiện ở GIỮA câu (không đầu câu) ≥ 1 lần → bảo vệ
    - Từ được gọi trực tiếp (vocative) như "Will, what are you doing?" → bảo vệ
    - Từ trong _COMMON_WORDS chỉ bảo vệ nếu rõ ràng là tên (vocative pattern)
    """
    # Chỉ áp dụng cho các ngôn ngữ Latin (en, ko romanized, v.v.)
    # Với tiếng Trung thuần CJK thì proper noun giữ nguyên ký tự CJK
    if source_language == "zh":
        return {}

    mid_sentence_count: Counter[str] = Counter()
    total_count: Counter[str] = Counter()
    # Vocative: tên được gọi trực tiếp ("Will, ..." hoặc "..., Will")
    vocative_words: set[str] = set()

    for text in all_texts:
        starts = _sentence_start_positions(text)
        for m in _CAPITALIZED_WORD_RE.finditer(text):
            word = m.group(1)
            if len(word) < 2:
                continue
            # Kiểm tra vocative pattern: "Word, ..." hoặc "..., Word" hoặc "..., Word."
            # Pattern này áp dụng cho cả common words lẫn non-common words
            before = text[:m.start()].rstrip()
            after = text[m.end():].lstrip()
            is_vocative = (
                before.endswith(",")       # "Come here, Will"
                or after.startswith(",")   # "Will, what are you doing?"
                or (m.start() in starts and after.startswith(","))  # đầu câu + dấu phẩy
            )
            if is_vocative:
                vocative_words.add(word)

            # Common words chỉ thêm vào total nếu không phải common
            if word.lower() in _COMMON_WORDS:
                continue
            total_count[word] += 1
            if m.start() not in starts:
                mid_sentence_count[word] += 1

    candidates: set[str] = set()
    for word, count in total_count.items():
        if word.lower() in _COMMON_WORDS:
            continue
        # Xuất hiện ≥ 2 lần tổng cộng → likely proper noun
        if count >= 2:
            candidates.add(word)
        # Xuất hiện ở giữa câu ≥ 1 lần → likely proper noun
        if mid_sentence_count[word] >= 1:
            candidates.add(word)

    # Thêm vocative words (tên được gọi trực tiếp — kể cả common words như "Will", "Mark")
    candidates.update(vocative_words)

    # Tạo placeholder map
    proper_noun_map: dict[str, str] = {}
    counter = len(_GLOSSARY) + len(_INTERJECTIONS)
    for word in sorted(candidates, key=lambda w: (-len(w), w)):  # dài trước, tránh conflict
        placeholder = f"ZXQPN{counter}ZXQ"
        proper_noun_map[placeholder] = word
        counter += 1

    return proper_noun_map


def _protect_all(text: str, proper_noun_map: dict[str, str]) -> str:
    """Bảo vệ: glossary → interjections → proper nouns (theo thứ tự)."""
    text = _protect_glossary(text)
    text = _protect_interjections(text)
    # Proper nouns: thay thế từ dài trước để tránh conflict
    for placeholder, word in proper_noun_map.items():
        text = re.sub(r"\b" + re.escape(word) + r"\b", placeholder, text)
    return text


def _restore_all(text: str, proper_noun_map: dict[str, str]) -> str:
    """Khôi phục ngược lại: proper nouns → interjections → glossary."""
    # Khôi phục proper nouns (bao gồm các biến thể case do Google tạo ra)
    for placeholder, word in proper_noun_map.items():
        text = text.replace(placeholder, word)
        text = text.replace(placeholder.lower(), word)
        text = text.replace(placeholder.title(), word)
    text = _restore_interjections(text)
    text = _restore_glossary(text)
    return text


# ─── Vietnamese post-processing ───────────────────────────────────────────────

# Các lỗi Google Translate thường gặp khi dịch sang tiếng Việt
_VI_FIXES: list[tuple[re.Pattern, str]] = [
    # "Của <Tên>" → "<Tên>'s sẽ được khôi phục đúng, nhưng "của X" sai ngữ pháp Việt
    (re.compile(r"\bcủa\s+([A-Z][a-z]+)\b"), r"\1"),
    # Thưa <Tên> (Google thêm "thưa" trước tên gọi trực tiếp)
    (re.compile(r"\bThưa\s+([A-Z][a-z]+)\b"), r"\1"),
    (re.compile(r"\bthưa\s+([A-Z][a-z]+)\b"), r"\1"),
    # "Nhìn xem" → "Nhìn này" (Google hay dịch Look sai)
    # Bỏ dấu ... cuối câu bị nhân đôi
    (re.compile(r"\.{4,}"), "..."),
    # Bỏ khoảng trắng thừa trước dấu câu
    (re.compile(r"\s+([.,!?:;])"), r"\1"),
    # Đảm bảo khoảng trắng sau dấu phẩy
    (re.compile(r",([^\s])"), r", \1"),
]

_VI_NAME_FIXES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bZhaoban\b", re.IGNORECASE), "làm theo"),
    (re.compile(r"\bZhan\s*Xiang\b", re.IGNORECASE), "như thường"),
    (re.compile(r"Giai đoạn Chiến tranh Mặt trời", re.IGNORECASE), "mặt trời vẫn chiếu như thường"),
    (re.compile(r"\bMo\s*Yue\b", re.IGNORECASE), "Mặc Nguyệt"),
    (re.compile(r"\bQixia\b", re.IGNORECASE), "Tê Hà"),
    (re.compile(r"núi Tê Hà", re.IGNORECASE), "núi Tê Hà"),
    (re.compile(r"\bZhu\s*Yan\b", re.IGNORECASE), "Chu Nguyên Chương"),
    (re.compile(r"\bZhu\s*Qiyu\b", re.IGNORECASE), "Chu Kỳ Ngọc"),
    (re.compile(r"\bZhu\s*Qi\s*Yu(?:zheng)?\b", re.IGNORECASE), "Chu Kỳ Ngọc"),
    (re.compile(r"\bChu\s+Qiyu\b", re.IGNORECASE), "Chu Kỳ Ngọc"),
    (re.compile(r"\bChu\s+Kiyu\b", re.IGNORECASE), "Chu Kỳ Ngọc"),
    (re.compile(r"\bChu\s+Ngọc\b", re.IGNORECASE), "Chu Kỳ Ngọc"),
    (re.compile(r"\bTần ca\b", re.IGNORECASE), "Tần ca"),
    (re.compile(r"\bAnh Tần\b", re.IGNORECASE), "Tần ca"),
    (re.compile(r"\bLong Jie\b", re.IGNORECASE), "Long tỷ"),
    (re.compile(r"\bLong\s+tỷ\b", re.IGNORECASE), "Long tỷ"),
    (re.compile(r"\bBộ trưởng của hoàng đế đã đến\b", re.IGNORECASE), "Bệ hạ, thần đến rồi"),
    (re.compile(r"\bBộ trưởng của hoàng đế\b", re.IGNORECASE), "thần"),
    (re.compile(r"\bHoàng đế\b"), "Bệ hạ"),
    (re.compile(r"\bHongwu\b", re.IGNORECASE), "Hồng Vũ"),
    (re.compile(r"\bYongle\b", re.IGNORECASE), "Vĩnh Lạc"),
    (re.compile(r"\bHaifan\b", re.IGNORECASE), "Hải Phàm"),
    (re.compile(r"triều đại Hồng Vũ", re.IGNORECASE), "triều Hồng Vũ"),
    (re.compile(r"triều đại Vĩnh Lạc", re.IGNORECASE), "triều Vĩnh Lạc"),
    (re.compile(r"triều đại Chính Thống", re.IGNORECASE), "triều Chính Thống"),
    (re.compile(r"thái tử và đại sư", re.IGNORECASE), "Thái tử thái sư"),
    (re.compile(r"Thái tử và đại sư", re.IGNORECASE), "Thái tử thái sư"),
    (re.compile(r"Hãy hứa với chúng tôi một điều", re.IGNORECASE), "Xin hãy hứa với thần một điều"),
    (re.compile(r"Thưa bệ hạ, tôi sẽ làm theo lời ngài", re.IGNORECASE), "Bệ hạ nói gì, thần nhất định sẽ làm theo"),
    (re.compile(r"Xin hãy hứa với ta một chuyện", re.IGNORECASE), "Xin hãy hứa với trẫm một chuyện"),
    (re.compile(r"\bHứa với ta một chuyện", re.IGNORECASE), "Hứa với trẫm một chuyện"),
    (re.compile(r"\bHứa với ta một việc", re.IGNORECASE), "Hứa với trẫm một việc"),
    (re.compile(r"Hãy thay ta bảo vệ", re.IGNORECASE), "Hãy thay trẫm bảo vệ"),
    (re.compile(r"\bThay ta bảo vệ", re.IGNORECASE), "Thay trẫm bảo vệ"),
    (re.compile(r"bảo vệ nó", re.IGNORECASE), "bảo vệ người ấy"),
    (re.compile(r"bảo vệ người\. Dốc", re.IGNORECASE), "bảo vệ người ấy. Dốc"),
    (re.compile(r"để nó sống", re.IGNORECASE), "để người ấy sống"),
    (re.compile(r"Hãy bảo vệ nó, bảo vệ cho tốt", re.IGNORECASE), "Hãy bảo vệ người ấy, bảo vệ cho tốt"),
    (re.compile(r"Bảo nó cứ sống như thường", re.IGNORECASE), "Bảo người ấy cứ sống như thường"),
    (re.compile(r"bảo vệ hắn", re.IGNORECASE), "bảo vệ người ấy"),
    (re.compile(r"Bệ hạ cứ nói, tôi", re.IGNORECASE), "Bệ hạ cứ nói, thần"),
    (re.compile(r"Bệ hạ cứ nói, con", re.IGNORECASE), "Bệ hạ cứ nói, thần"),
    (re.compile(r"Bệ hạ nói gì, tôi", re.IGNORECASE), "Bệ hạ nói gì, thần"),
    (re.compile(r"Bệ hạ nói gì, con", re.IGNORECASE), "Bệ hạ nói gì, thần"),
    (re.compile(r"Hãy bảo vệ anh ấy bằng gần như toàn bộ sức lực của bạn", re.IGNORECASE), "Thần sẽ dốc gần như toàn lực để bảo vệ người ấy"),
    (re.compile(r"Hãy bảo anh ấy làm như bình thường\.?\s*Gọi mặt trời như thường lệ\.?\s*Gọi mặt trời như thường lệ\.?", re.IGNORECASE), "Hãy để mặt trời vẫn chiếu như thường."),
    (re.compile(r"Xin đừng giữ Mặc Nguyệt lên núi Tê Hà", re.IGNORECASE), "Xin hãy ôm Mặc Nguyệt lên núi Tê Hà"),
    (re.compile(r"Làm phiền ôm Mặc Nguyệt lên núi Tê Hà", re.IGNORECASE), "Xin hãy ôm Mặc Nguyệt lên núi Tê Hà"),
    (re.compile(r"Đừng để mọi người tan vỡ", re.IGNORECASE), "Sinh ly tử biệt khiến người ta rơi lệ"),
    (re.compile(r"Đếm xem Tần ca có thể nhân lên bao nhiêu lần", re.IGNORECASE), "Đếm xem Tần ca làm được bao nhiêu lần"),
    (re.compile(r"Tần ca, huynh nói xem, động vật dưới biển\.{0,3}", re.IGNORECASE), "Tần ca, huynh nói xem, huynh còn nhào lộn nổi không?"),
    (re.compile(r"Tần ca, huynh nói xem, .*?dưới biển\.{0,3}", re.IGNORECASE), "Tần ca, huynh nói xem, huynh còn nhào lộn nổi không?"),
    (re.compile(r"Cha ơi con ở đây", re.IGNORECASE), "Phụ thân, con ở đây"),
    (re.compile(r"Được\.\.\. ta sẽ sống thật tốt", re.IGNORECASE), "Được... con sẽ sống thật tốt"),
    (re.compile(r"Cha ơi, sao trời tối thế", re.IGNORECASE), "Phụ thân, sao trời tối thế"),
    (re.compile(r"ngay cả cha tôi cũng là học trò của ông", re.IGNORECASE), "ngay cả phụ hoàng của ta cũng là học trò của ông"),
]


def _fix_vietnamese_output(translated: str, original: str) -> str:
    """Sửa các lỗi Google Translate thường gặp khi dịch sang tiếng Việt."""
    result = translated
    for pattern, replacement in _VI_FIXES:
        result = pattern.sub(replacement, result)
    # Nếu dịch ra toàn chữ thường trong khi gốc có tên riêng → giữ nguyên bản gốc đã clean
    # (trường hợp placeholder không được restore đúng)
    result = result.strip()
    return result or translated


def _fix_character_names(text: str) -> str:
    result = text
    for pattern, replacement in _VI_NAME_FIXES:
        result = pattern.sub(replacement, result)
    result = re.sub(r"\b(làm theo)\s+\1\b", r"\1", result, flags=re.IGNORECASE)
    result = re.sub(r"\s+([.,!?:;])", r"\1", result)
    result = re.sub(r",([^\s])", r", \1", result)
    return result.strip()


# ─── Batch translation ────────────────────────────────────────────────────────

def _translation_batches(texts: list[str], max_chars: int = 900) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for text in texts:
        extra = len(text) + 20  # dự phòng cho delimiter
        if current and current_chars + extra > max_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(text)
        current_chars += extra
    if current:
        batches.append(current)
    return batches


def _translate_batch(
    texts: list[str], source_language: str, target_language: str
) -> list[str]:
    delimiter = "\n###SUB_BREAK###\n"
    query = delimiter.join(texts)
    payload = _request_google_translate(query, source_language, target_language, timeout=30)
    translated = _payload_text(payload)
    parts = [_clean_translation(part) for part in translated.split("###SUB_BREAK###")]
    if len(parts) != len(texts):
        # Fallback: dịch từng câu riêng lẻ
        return [_translate_single(text, source_language, target_language) for text in texts]
    return parts


def _translate_single(
    text: str, source_language: str, target_language: str
) -> str:
    if not text.strip():
        return ""
    chunks = _split_translation_text(text)
    if len(chunks) > 1:
        return _clean_translation(
            " ".join(
                _translate_single(chunk, source_language, target_language)
                for chunk in chunks
            )
        )
    payload = _request_google_translate(text, source_language, target_language, timeout=20)
    return _clean_translation(_payload_text(payload))


def _translate_single_or_original(
    text: str, source_language: str, target_language: str
) -> str:
    try:
        return _translate_single(text, source_language, target_language)
    except Exception:
        return text


def _repair_untranslated_lines(
    originals: list[str],
    translated: list[str],
    source_language: str,
    target_language: str,
) -> list[str]:
    repaired: list[str] = []
    for original, text in zip(originals, translated):
        candidate = text or original
        if _needs_retranslation(candidate, target_language):
            for retry_source in (source_language, "zh", "auto"):
                try:
                    retry = _translate_single(original, retry_source, target_language)
                except Exception:
                    continue
                if retry and not _needs_retranslation(retry, target_language):
                    candidate = retry
                    break
        if _needs_retranslation(candidate, target_language):
            raise TranslationError(
                f"Google Translate còn sót văn bản chưa dịch sang {target_language}: {candidate[:160]}"
            )
        repaired.append(candidate)
    return repaired


def _split_translation_text(text: str, max_chars: int = 450) -> list[str]:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return [cleaned]

    pieces = re.split(r"([。！？!?；;，,、\n])", cleaned)
    chunks: list[str] = []
    current = ""
    for index in range(0, len(pieces), 2):
        piece = pieces[index]
        punctuation = pieces[index + 1] if index + 1 < len(pieces) else ""
        sentence = f"{piece}{punctuation}".strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                sentence[start : start + max_chars]
                for start in range(0, len(sentence), max_chars)
            )
            continue
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks or [cleaned]
