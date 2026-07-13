import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.core import settings
from app.services.ai.context import _sanitize_context
from app.services.ai.translation import GeminiTranslation, get_translation_engine
from app.services.subtitles.timing import SubtitleEvent


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = json.loads(kwargs["messages"][1]["content"])
        items = [
            {"id": item["id"], "text": f"Lục Trạch nói câu {item['id']}"}
            for item in payload["current_items"]
        ]
        updates = {}
        if len(self.calls) == 1:
            updates = {
                "characters": [
                    {
                        "id": "lu_trach",
                        "name": "Lục Trạch",
                        "role": "nam chính",
                        "confidence": 0.98,
                    }
                ],
                "relationships": [],
                "glossary": {"陆泽": "Lục Trạch"},
                "addressing_rules": [],
            }
        content = json.dumps(
            {"items": items, "memory_updates": updates},
            ensure_ascii=False,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions())


class GeminiTranslationTests(unittest.IsolatedAsyncioTestCase):
    async def test_old_google_setting_uses_direct_contextual_ai(self) -> None:
        original_key = settings.ninerouter_api_key
        original_engine = settings.translation_engine
        fake_client = _FakeClient()
        events = [
            SubtitleEvent(float(index), float(index + 1), f"陆泽说第{index + 1}句")
            for index in range(25)
        ]
        try:
            settings.ninerouter_api_key = "test-key"
            settings.translation_engine = "google"
            self.assertIsInstance(get_translation_engine(), GeminiTranslation)
            with patch("openai.AsyncOpenAI", return_value=fake_client), patch(
                "app.services.ai.translation._request_google_translate",
                side_effect=AssertionError("Google must not be called"),
            ):
                translated = await get_translation_engine().translate_events(
                    events,
                    "zh",
                    "vi",
                    context={
                        "characters": [{"name": "Lục Trạch", "source": "陆泽"}],
                        "relationships": [],
                    },
                )
        finally:
            settings.ninerouter_api_key = original_key
            settings.translation_engine = original_engine

        self.assertEqual(len(translated), len(events))
        self.assertEqual(
            [(item.start, item.end) for item in translated],
            [(item.start, item.end) for item in events],
        )
        self.assertTrue(all("Lục Trạch" in item.text for item in translated))
        calls = fake_client.chat.completions.calls
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call["model"] == settings.ai_model for call in calls))
        second_payload = json.loads(calls[1]["messages"][1]["content"])
        self.assertTrue(second_payload["previous_context"])
        self.assertEqual(
            second_payload["memory"]["characters"][0]["name"],
            "Lục Trạch",
        )

    async def test_ai_relationship_terms_are_not_rewritten(self) -> None:
        class _RelationshipCompletions:
            async def create(self, **kwargs):
                content = json.dumps(
                    {
                        "items": [{"id": 1, "text": "Đó là vợ của anh ấy."}],
                        "memory_updates": {
                            "characters": [],
                            "relationships": ["hai người là vợ chồng"],
                            "glossary": {},
                            "addressing_rules": [],
                        },
                    },
                    ensure_ascii=False,
                )
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=_RelationshipCompletions())
        )
        original_key = settings.ninerouter_api_key
        try:
            settings.ninerouter_api_key = "test-key"
            with patch("openai.AsyncOpenAI", return_value=fake_client):
                translated = await GeminiTranslation().translate_events(
                    [SubtitleEvent(0.0, 1.0, "She is his wife.")],
                    "en",
                    "vi",
                    context={"relationships": ["They are married."]},
                )
        finally:
            settings.ninerouter_api_key = original_key

        self.assertEqual(translated[0].text, "Đó là vợ của anh ấy.")

    def test_context_relationship_evidence_is_preserved(self) -> None:
        context = _sanitize_context(
            {
                "characters": [{"name": "Anna"}, {"name": "Minh"}],
                "relationships": "Anna là vợ của Minh.",
            },
            "She is his wife.",
        )
        self.assertEqual(context["relationships"], "Anna là vợ của Minh.")

    def test_ai_model_is_defined_once_in_python_sources(self) -> None:
        app_dir = Path(__file__).resolve().parents[1] / "app"
        occurrences: list[Path] = []
        for source in app_dir.rglob("*.py"):
            if "ag/gemini-pro-agent" in source.read_text(encoding="utf-8"):
                occurrences.append(source)
        self.assertEqual(occurrences, [app_dir / "core.py"])


if __name__ == "__main__":
    unittest.main()
