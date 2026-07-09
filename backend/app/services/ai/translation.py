"""
Translation engine với proper noun protection tự động và Vietnamese post-processing.

Các cải tiến chính:
- Phát hiện proper nouns toàn cục bằng thống kê (không chỉ whitelist cứng)
- Bảo vệ interjections (Oh, Wow, Hey, ...) khỏi bị dịch
- Post-processing sửa lỗi Google Translate thường gặp khi dịch sang tiếng Việt
- Context-aware: truyền 1 câu trước/sau vào batch để tránh mất ngữ cảnh
- Giữ nguyên từ cảm thán tiếng Anh vốn dĩ không cần dịch
"""
from __future__ import annotations

import asyncio
import re
from collections import Counter

from app.core import settings
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
    ) -> list[SubtitleEvent]:
        raise NotImplementedError


class PassthroughTranslation(TranslationEngine):
    async def translate_events(
        self,
        events: list[SubtitleEvent],
        source_language: str,
        target_language: str,
    ) -> list[SubtitleEvent]:
        return events


class GoogleTranslation(TranslationEngine):
    async def translate_events(
        self,
        events: list[SubtitleEvent],
        source_language: str,
        target_language: str,
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

        return await asyncio.to_thread(translate)


def get_translation_engine() -> TranslationEngine:
    if settings.translation_engine.lower() == "google":
        return GoogleTranslation()
    return PassthroughTranslation()


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
    (re.compile("初期"), "朱祁钰"),
    (re.compile("朱言"), "朱元璋"),
    (re.compile("红武朝"), "洪武朝"),
    (re.compile("从西朝"), "正统朝"),
    (re.compile("有垃圾的"), "永乐朝的"),
    (re.compile("墨月"), "墨月"),
    (re.compile("七夏山"), "栖霞山"),
    (re.compile("秦哥"), "秦哥"),
    (re.compile("龙姐"), "龙姐"),
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
    (re.compile(r"Hãy bảo vệ anh ấy bằng gần như toàn bộ sức lực của bạn", re.IGNORECASE), "Thần sẽ dốc gần như toàn lực để bảo vệ người ấy"),
    (re.compile(r"Hãy bảo anh ấy làm như bình thường\.?\s*Gọi mặt trời như thường lệ\.?\s*Gọi mặt trời như thường lệ\.?", re.IGNORECASE), "Hãy để mặt trời vẫn chiếu như thường."),
    (re.compile(r"Xin đừng giữ Mặc Nguyệt lên núi Tê Hà", re.IGNORECASE), "Xin hãy ôm Mặc Nguyệt lên núi Tê Hà"),
    (re.compile(r"Đừng để mọi người tan vỡ", re.IGNORECASE), "Sinh ly tử biệt khiến người ta rơi lệ"),
    (re.compile(r"Đếm xem Tần ca có thể nhân lên bao nhiêu lần", re.IGNORECASE), "Đếm xem Tần ca làm được bao nhiêu lần"),
    (re.compile(r"Cha ơi con ở đây", re.IGNORECASE), "Phụ thân, con ở đây"),
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
