from pathlib import Path

from app.core import settings
from app.services.subtitles.ass import write_srt
from app.services.subtitles.timing import SubtitleEvent


class AsrError(RuntimeError):
    pass


class AsrEngine:
    async def transcribe_to_srt(self, audio_file: Path, output_file: Path, language: str) -> Path:
        raise NotImplementedError


class FasterWhisperAsr(AsrEngine):
    async def transcribe_to_srt(self, audio_file: Path, output_file: Path, language: str) -> Path:
        import asyncio

        def transcribe() -> Path:
            _load_nvidia_runtime_dlls()
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise AsrError("Thiếu faster-whisper. Chạy pip install -r requirements.txt trong backend.") from exc

            model = WhisperModel(
                settings.whisper_model,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
                local_files_only=settings.whisper_local_files_only,
            )
            whisper_language = None if language == "auto" else language
            segments, _info = model.transcribe(
                str(audio_file),
                language=whisper_language,
                vad_filter=settings.whisper_vad_filter,
                beam_size=settings.whisper_beam_size,
                word_timestamps=settings.whisper_word_timestamps,
                condition_on_previous_text=False,
            )
            events = [
                SubtitleEvent(
                    start=_segment_start(segment),
                    end=_segment_end(segment),
                    text=segment.text.strip(),
                )
                for segment in segments
                if segment.text.strip()
            ]
            if not events:
                raise AsrError("Whisper không nhận diện được lời thoại để tạo phụ đề.")
            return write_srt(events, output_file)

        return await asyncio.to_thread(transcribe)


def get_asr_engine() -> AsrEngine:
    return FasterWhisperAsr()


def _load_nvidia_runtime_dlls() -> None:
    import os
    import sys

    if not sys.platform.startswith("win"):
        return

    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nvidia_root.exists():
        return

    dll_dirs = [path for path in nvidia_root.glob("*/bin") if path.exists()]
    for path in dll_dirs:
        try:
            os.add_dll_directory(str(path))
        except (AttributeError, OSError):
            pass

    existing = os.environ.get("PATH", "")
    prefixes = [str(path) for path in dll_dirs if str(path) not in existing]
    if prefixes:
        os.environ["PATH"] = ";".join(prefixes + [existing])


def _segment_start(segment) -> float:
    words = getattr(segment, "words", None)
    if not words:
        return segment.start
    first_word = next((word for word in words if getattr(word, "word", "").strip()), None)
    return getattr(first_word, "start", segment.start) if first_word else segment.start


def _segment_end(segment) -> float:
    words = getattr(segment, "words", None)
    if not words:
        return segment.end
    last_word = next((word for word in reversed(words) if getattr(word, "word", "").strip()), None)
    return getattr(last_word, "end", segment.end) if last_word else segment.end
