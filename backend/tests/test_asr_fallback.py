from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.services.ai.asr import (
    FasterWhisperAsr,
    FasterWhisperUnavailable,
    OpenAIWhisperAsr,
    ResilientAsr,
    _result_segment_time,
    get_asr_engine,
)


class AsrFallbackTests(unittest.IsolatedAsyncioTestCase):
    def test_default_engine_is_resilient(self) -> None:
        self.assertIsInstance(get_asr_engine(), ResilientAsr)

    def test_result_segment_time_prefers_word_timestamps(self) -> None:
        segment = {
            "start": 1.0,
            "end": 4.0,
            "words": [
                {"start": 1.15, "end": 1.5, "word": " Xin"},
                {"start": 3.1, "end": 3.65, "word": " chào"},
            ],
        }
        self.assertEqual(_result_segment_time(segment, "start"), 1.15)
        self.assertEqual(_result_segment_time(segment, "end"), 3.65)

    async def test_native_dll_failure_falls_back_to_openai_whisper(self) -> None:
        expected = Path("fallback.srt")
        faster = AsyncMock(
            side_effect=FasterWhisperUnavailable("DLL native bị chặn")
        )
        fallback = AsyncMock(return_value=expected)
        with (
            patch.object(FasterWhisperAsr, "transcribe_to_srt", faster),
            patch.object(OpenAIWhisperAsr, "transcribe_to_srt", fallback),
        ):
            result = await ResilientAsr().transcribe_to_srt(
                Path("audio.wav"), Path("out.srt"), "vi"
            )

        self.assertEqual(result, expected)
        faster.assert_awaited_once()
        fallback.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
