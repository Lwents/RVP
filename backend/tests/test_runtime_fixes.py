import asyncio
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.services.ai.character_names import canonicalize_character_text
from app.services.ai.asr import _load_asr_audio
from app.services.ai.translation import _clean_polished_subtitle, _fix_character_names
from app.services.subtitles.source import download_platform_subtitles
from app.services.subtitles.timing import SubtitleEvent, split_long_subtitle_events


class RuntimeFixTests(unittest.TestCase):
    def test_canonical_name_containing_alias_is_idempotent(self):
        registry = {'Perseverance': 'Tàu thám hiểm Perseverance'}
        expected = 'Tàu thám hiểm Perseverance đang khám phá.'
        result = canonicalize_character_text('Perseverance đang khám phá.', registry)
        self.assertEqual(result, expected)
        for _ in range(3):
            result = canonicalize_character_text(result, registry)
            self.assertEqual(result, expected)

    def test_alias_replacements_preserve_ordinary_prose(self):
        registry = {'Gian': 'Chaien', 'Nobi': 'Nobita', 'Nobita': 'Nobita'}
        self.assertEqual(canonicalize_character_text('Gian gặp Nobi trong thời gian rảnh.', registry),
                         'Chaien gặp Nobita trong thời gian rảnh.')

    def test_pcm_audio_is_read_without_pyav(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / 'audio.wav'
            samples = np.array([-32768, 0, 16384, 32767], dtype='<i2')
            with wave.open(str(audio), 'wb') as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes(samples.tobytes())
            with patch('av.open', side_effect=AssertionError('PyAV must not decode audio')):
                result = _load_asr_audio(audio)
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_allclose(result, [-1.0, 0.0, 0.5, 32767 / 32768])

    def test_decimal_comma_survives_punctuation_cleanup(self):
        text = 'Khoảng 3, 5 tỷ năm trước,đã có nước.'
        self.assertEqual(_clean_polished_subtitle(text, ''), 'Khoảng 3,5 tỷ năm trước, đã có nước.')
        self.assertEqual(_fix_character_names('3,5 tỷ năm,đá cổ.'), '3,5 tỷ năm, đá cổ.')

    def test_long_translation_has_no_flashing_tail(self):
        text = 'Có lẽ đây là tàn tích của một dòng dung nham nay gần như đã bị xói mòn.'
        source = SubtitleEvent(76.276, 79.913, text)
        result = split_long_subtitle_events([source], max_chars=70, max_duration=4.5)
        self.assertGreater(len(result), 1)
        self.assertEqual(' '.join(item.text for item in result), text)
        self.assertEqual(result[0].start, source.start)
        self.assertEqual(result[-1].end, source.end)
        self.assertTrue(all(item.end - item.start >= 0.6 for item in result))
        self.assertTrue(all(len(item.text.split()) > 1 for item in result))
        self.assertTrue(all(left.end <= right.start for left, right in zip(result, result[1:])))

    def test_english_locale_track_is_selected_over_vietnamese(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'subtitles.vi.srt').write_text('wrong target', encoding='utf-8')
            def run(command, timeout):
                self.assertEqual(command[command.index('--sub-langs') + 1], 'en.*')
                self.assertEqual(command[command.index('--sub-format') + 1], 'srt/vtt/best')
                (root / 'subtitles.en-US.srt').write_text('English source', encoding='utf-8')
                (root / 'subtitles.en-orig.srt').write_text('Less accurate automatic source', encoding='utf-8')
                return type('Result', (), {'returncode': 0, 'stderr': '', 'stdout': ''})()
            with patch('app.services.subtitles.source._run_ytdlp', side_effect=run):
                result = asyncio.run(download_platform_subtitles('https://youtu.be/test', root, lambda *args: None, 'en'))
            self.assertEqual(result.name, 'subtitles.en-US.srt')

    def test_auto_language_prefers_original_over_platform_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def run(command, timeout):
                (root / 'subtitles.en.srt').write_text('Platform translation', encoding='utf-8')
                (root / 'subtitles.zh-Hans-orig.srt').write_text('Original source', encoding='utf-8')
                return type('Result', (), {'returncode': 0, 'stderr': '', 'stdout': ''})()
            with patch('app.services.subtitles.source._run_ytdlp', side_effect=run):
                result = asyncio.run(download_platform_subtitles('https://youtu.be/test', root, lambda *args: None))
            self.assertEqual(result.name, 'subtitles.zh-Hans-orig.srt')


if __name__ == '__main__':
    unittest.main()
