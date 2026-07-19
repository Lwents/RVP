"""
ASR engine dùng Faster-Whisper.

Cải tiến:
- initial_prompt theo từng ngôn ngữ để cải thiện nhận diện tên riêng
- no_speech_threshold và compression_ratio_threshold để giảm hallucination
- Log số segment nhận diện được để debug
"""
from pathlib import Path

from app.core import settings
from app.models.job import ProcessingMode
from app.services.presets import get_processing_profile
from app.services.subtitles.ass import write_srt
from app.services.subtitles.timing import SubtitleEvent


class AsrError(RuntimeError):
    pass


class FasterWhisperUnavailable(AsrError):
    """The package is installed but one of its native dependencies cannot load."""


# Prompt gợi ý cho Whisper theo từng ngôn ngữ
# Giúp Whisper giữ nguyên tên riêng và không hallucinate
_INITIAL_PROMPTS: dict[str, str] = {
    "en": (
        "A conversation between people. "
        "Proper names, brand names, and technical terms are kept as-is. "
        "Do not translate or paraphrase."
    ),
    "zh": (
        "这是一段对话，包含人名和地名。"
        "请准确转录，保留所有专有名词。"
    ),
    "ko": (
        "사람들 사이의 대화입니다. "
        "고유 명사와 이름은 원래대로 유지하세요."
    ),
    "ja": (
        "これは人々の会話です。"
        "固有名詞や名前はそのまま書き写してください。"
    ),
    "auto": (
        "A natural conversation. Keep all proper names, brand names unchanged. "
        "这是一段对话，请准确转录。"
    ),
}


class AsrEngine:
    async def transcribe_to_srt(
        self,
        audio_file: Path,
        output_file: Path,
        language: str,
        processing_mode: ProcessingMode | str | None = None,
    ) -> Path:
        raise NotImplementedError


class FasterWhisperAsr(AsrEngine):
    async def transcribe_to_srt(
        self,
        audio_file: Path,
        output_file: Path,
        language: str,
        processing_mode: ProcessingMode | str | None = None,
    ) -> Path:
        import asyncio

        profile = get_processing_profile(processing_mode)

        def transcribe() -> Path:
            _load_nvidia_runtime_dlls()
            try:
                from faster_whisper import WhisperModel
                model = WhisperModel(
                    settings.whisper_model,
                    device=settings.whisper_device,
                    compute_type=settings.whisper_compute_type,
                    local_files_only=settings.whisper_local_files_only,
                )
            except ModuleNotFoundError as exc:
                if exc.name == "faster_whisper":
                    raise FasterWhisperUnavailable(
                        "Chưa cài faster-whisper trong môi trường backend. "
                        "Chạy .venv\\Scripts\\python.exe -m pip install -r requirements.txt."
                    ) from exc
                raise FasterWhisperUnavailable(
                    "faster-whisper không tải được dependency native "
                    f"'{exc.name or 'không xác định'}': {exc}."
                ) from exc
            except (ImportError, OSError) as exc:
                raise FasterWhisperUnavailable(
                    "faster-whisper đã được cài nhưng DLL native không tải được: "
                    f"{exc}."
                ) from exc

            whisper_language = None if language == "auto" else language
            initial_prompt = _INITIAL_PROMPTS.get(language, _INITIAL_PROMPTS["auto"])

            segments_iter, info = model.transcribe(
                str(audio_file),
                language=whisper_language,
                vad_filter=settings.whisper_vad_filter,
                beam_size=profile.whisper_beam_size,
                word_timestamps=profile.whisper_word_timestamps,
                condition_on_previous_text=True,   # giúp giữ ngữ cảnh tên nhân vật
                initial_prompt=initial_prompt,
                # Giảm hallucination: segment im lặng bị bỏ
                no_speech_threshold=0.6,
                # Loại bỏ segment bị lặp (hallucination điển hình)
                log_prob_threshold=-1.0,
            )

            events: list[SubtitleEvent] = []
            segment_count = 0
            for segment in segments_iter:
                text = segment.text.strip()
                if not text:
                    continue
                # Bỏ các segment lặp đúng nguyên văn câu trước (hallucination)
                if events and _is_repeated(text, events[-1].text):
                    continue
                events.append(
                    SubtitleEvent(
                        start=_segment_start(segment),
                        end=_segment_end(segment),
                        text=text,
                    )
                )
                segment_count += 1

            print(f"[ASR] Faster-Whisper produced {segment_count} segments from {audio_file.name}")

            if not events:
                raise AsrError("Whisper không nhận diện được lời thoại để tạo phụ đề.")
            return write_srt(events, output_file)

        return await asyncio.to_thread(transcribe)


class OpenAIWhisperAsr(AsrEngine):
    """CPU-safe fallback when Windows policy blocks Faster-Whisper/PyAV DLLs."""

    async def transcribe_to_srt(
        self,
        audio_file: Path,
        output_file: Path,
        language: str,
        processing_mode: ProcessingMode | str | None = None,
    ) -> Path:
        import asyncio

        profile = get_processing_profile(processing_mode)

        def transcribe() -> Path:
            try:
                import torch
                import whisper
            except ImportError as exc:
                raise AsrError(
                    "Không thể chạy Whisper dự phòng; thiếu package whisper hoặc torch: " f"{exc}"
                ) from exc

            requested_device = settings.whisper_device
            device = requested_device if requested_device != "cuda" or torch.cuda.is_available() else "cpu"
            try:
                model = whisper.load_model(settings.whisper_model, device=device)
                result = model.transcribe(
                    str(audio_file),
                    language=None if language == "auto" else language,
                    fp16=False,
                    verbose=False,
                    beam_size=profile.whisper_beam_size,
                    word_timestamps=profile.whisper_word_timestamps,
                    condition_on_previous_text=True,
                    initial_prompt=_INITIAL_PROMPTS.get(language, _INITIAL_PROMPTS["auto"]),
                )
            except Exception as exc:
                raise AsrError(f"Whisper dự phòng không nhận diện được audio: {exc}") from exc

            events: list[SubtitleEvent] = []
            for segment in result.get("segments", []):
                text = str(segment.get("text") or "").strip()
                if not text or (events and _is_repeated(text, events[-1].text)):
                    continue
                events.append(
                    SubtitleEvent(
                        start=_result_segment_time(segment, "start"),
                        end=_result_segment_time(segment, "end"),
                        text=text,
                    )
                )
            if not events:
                raise AsrError("Whisper dự phòng không nhận diện được lời thoại để tạo phụ đề.")
            print(f"[ASR] OpenAI Whisper fallback produced {len(events)} segments from {audio_file.name}")
            return write_srt(events, output_file)

        return await asyncio.to_thread(transcribe)


class ResilientAsr(AsrEngine):
    """Prefer Faster-Whisper, but keep a job alive when its DLLs are blocked."""

    async def transcribe_to_srt(
        self,
        audio_file: Path,
        output_file: Path,
        language: str,
        processing_mode: ProcessingMode | str | None = None,
    ) -> Path:
        try:
            return await FasterWhisperAsr().transcribe_to_srt(
                audio_file, output_file, language, processing_mode
            )
        except FasterWhisperUnavailable as faster_error:
            print("[ASR] Faster-Whisper unavailable; using OpenAI Whisper fallback.")
            try:
                return await OpenAIWhisperAsr().transcribe_to_srt(
                    audio_file, output_file, language, processing_mode
                )
            except AsrError as fallback_error:
                raise AsrError(
                    "Faster-Whisper không khởi động được và Whisper dự phòng cũng thất bại. "
                    f"Faster-Whisper: {faster_error} | Whisper dự phòng: {fallback_error}"
                ) from fallback_error


def get_asr_engine() -> AsrEngine:
    return ResilientAsr()


def _result_segment_time(segment: dict, field: str) -> float:
    words = segment.get("words") or []
    candidates = words if field == "start" else reversed(words)
    for word in candidates:
        value = word.get(field) if isinstance(word, dict) else getattr(word, field, None)
        if value is not None:
            return float(value)
    return float(segment.get(field) or 0.0)


def _is_repeated(text: str, prev_text: str) -> bool:
    """Kiểm tra xem segment hiện tại có bị lặp nguyên văn từ segment trước không."""
    t = text.lower().strip()
    p = prev_text.lower().strip()
    if not t or not p:
        return False
    # Lặp hoàn toàn
    if t == p:
        return True
    # Một trong hai chứa cái kia (lặp một phần lớn)
    if len(t) > 10 and (p.endswith(t) or t.startswith(p)):
        return True
    return False


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
