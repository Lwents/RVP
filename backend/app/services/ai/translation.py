import asyncio
import re

from app.core import settings
from app.services.subtitles.timing import SubtitleEvent


class TranslationError(RuntimeError):
    pass


class TranslationEngine:
    async def translate_events(self, events: list[SubtitleEvent], source_language: str, target_language: str) -> list[SubtitleEvent]:
        raise NotImplementedError


class PassthroughTranslation(TranslationEngine):
    async def translate_events(self, events: list[SubtitleEvent], source_language: str, target_language: str) -> list[SubtitleEvent]:
        return events


class GoogleTranslation(TranslationEngine):
    async def translate_events(self, events: list[SubtitleEvent], source_language: str, target_language: str) -> list[SubtitleEvent]:
        if not events:
            return []
        if source_language != "auto" and source_language.lower() == target_language.lower():
            return events

        def translate() -> list[SubtitleEvent]:
            translated_texts: list[str] = []
            texts = [_protect_glossary(event.text.strip()) for event in events]
            for batch in _translation_batches(texts):
                try:
                    translated_texts.extend(_translate_batch(batch, source_language, target_language))
                except Exception as exc:
                    recovered = [_translate_single_or_original(text, source_language, target_language) for text in batch]
                    if not any(item and item != original for item, original in zip(recovered, batch)):
                        raise TranslationError(f"Không dịch được phụ đề bằng Google Translate: {exc}") from exc
                    translated_texts.extend(recovered)
            translated_texts = [_restore_glossary(text) for text in translated_texts]

            return [
                SubtitleEvent(event.start, event.end, translated_texts[index] or event.text)
                for index, event in enumerate(events)
            ]

        return await asyncio.to_thread(translate)


def get_translation_engine() -> TranslationEngine:
    if settings.translation_engine.lower() == "google":
        return GoogleTranslation()
    return PassthroughTranslation()


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


_GLOSSARY = {
    "ZXQTERM0ZXQ": "AntiGravity",
    "ZXQTERM1ZXQ": "Cursor",
    "ZXQTERM2ZXQ": "Windsurf",
}


def _protect_glossary(text: str) -> str:
    protected = text
    protected = re.sub(r"\banti\s*gravity\b", "ZXQTERM0ZXQ", protected, flags=re.IGNORECASE)
    protected = re.sub(r"\bcursor\b", "ZXQTERM1ZXQ", protected, flags=re.IGNORECASE)
    protected = re.sub(r"\bwindsurf\b", "ZXQTERM2ZXQ", protected, flags=re.IGNORECASE)
    return protected


def _restore_glossary(text: str) -> str:
    restored = text
    for placeholder, term in _GLOSSARY.items():
        restored = restored.replace(placeholder, term)
        restored = restored.replace(placeholder.lower(), term)
        restored = restored.replace(placeholder.title(), term)
    return restored


def _translation_batches(texts: list[str], max_chars: int = 900) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for text in texts:
        extra = len(text) + 20
        if current and current_chars + extra > max_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(text)
        current_chars += extra
    if current:
        batches.append(current)
    return batches


def _translate_batch(texts: list[str], source_language: str, target_language: str) -> list[str]:
    delimiter = "\n###SUB_BREAK###\n"
    query = delimiter.join(texts)
    payload = _request_google_translate(query, source_language, target_language, timeout=30)
    translated = _payload_text(payload)
    parts = [_clean_translation(part) for part in translated.split("###SUB_BREAK###")]
    if len(parts) != len(texts):
        return [_translate_single(text, source_language, target_language) for text in texts]
    return parts


def _translate_single(text: str, source_language: str, target_language: str) -> str:
    if not text.strip():
        return ""
    chunks = _split_translation_text(text)
    if len(chunks) > 1:
        return _clean_translation(" ".join(_translate_single(chunk, source_language, target_language) for chunk in chunks))

    payload = _request_google_translate(text, source_language, target_language, timeout=20)
    return _clean_translation(_payload_text(payload))


def _translate_single_or_original(text: str, source_language: str, target_language: str) -> str:
    try:
        return _translate_single(text, source_language, target_language)
    except Exception:
        return text


def _request_google_translate(text: str, source_language: str, target_language: str, timeout: int) -> object:
    import requests

    params = {
        "client": "gtx",
        "sl": _google_source(source_language),
        "tl": _google_target(target_language),
        "dt": "t",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
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
    return "".join(part[0] for part in segments if isinstance(part, list) and part and part[0])


def _looks_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


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
            chunks.extend(sentence[start : start + max_chars] for start in range(0, len(sentence), max_chars))
            continue
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks or [cleaned]
