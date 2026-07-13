import asyncio
import subprocess
from pathlib import Path
from typing import Callable

from app.core import settings
from app.models.job import VoiceGender
from app.services.media.ffmpeg import find_ffmpeg, probe_video_duration
from app.services.subtitles.timing import SubtitleEvent, normalize_events, parse_srt


class VoiceError(RuntimeError):
    pass


class VoiceEngine:
    async def synthesize(
        self,
        text: str,
        output_file: Path,
        voice_gender: VoiceGender,
        progress: Callable[[str, int], None] | None = None,
    ) -> Path:
        raise NotImplementedError

    async def synthesize_srt(
        self,
        subtitle_file: Path,
        output_file: Path,
        voice_gender: VoiceGender,
        duration: float,
        progress: Callable[[str, int], None] | None = None,
    ) -> Path:
        return await self.synthesize(subtitle_text_for_tts(subtitle_file), output_file, voice_gender, progress)


class DisabledVoiceEngine(VoiceEngine):
    async def synthesize(
        self,
        text: str,
        output_file: Path,
        voice_gender: VoiceGender,
        progress: Callable[[str, int], None] | None = None,
    ) -> Path:
        raise VoiceError("Voice engine chưa được cấu hình.")


class EdgeTtsVoiceEngine(VoiceEngine):
    def get_voice(self, gender: VoiceGender) -> str:
        target_lang = settings.target_language.lower()
        if target_lang == "zh":
            return "zh-CN-XiaoxiaoNeural" if gender == VoiceGender.female else "zh-CN-YunxiNeural"
        elif target_lang == "en":
            return "en-US-AriaNeural" if gender == VoiceGender.female else "en-US-GuyNeural"
        else:
            return "vi-VN-HoaiMyNeural" if gender == VoiceGender.female else "vi-VN-NamMinhNeural"

    def get_voice_for_text(self, text: str, gender: VoiceGender) -> str:
        if _cjk_ratio(text) > 0.2:
            return "zh-CN-XiaoxiaoNeural" if gender == VoiceGender.female else "zh-CN-YunxiNeural"
        return self.get_voice(gender)

    async def synthesize(
        self,
        text: str,
        output_file: Path,
        voice_gender: VoiceGender,
        progress: Callable[[str, int], None] | None = None,
    ) -> Path:
        try:
            import edge_tts
        except ImportError as exc:
            raise VoiceError("Thiếu edge-tts. Chạy pip install -r requirements.txt trong backend.") from exc

        normalized = _normalize_tts_text(text)
        if not normalized:
            raise VoiceError("Không có nội dung phụ đề để tạo giọng đọc.")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.exists():
            output_file.unlink()

        chunks = _split_tts_chunks(normalized, settings.tts_chunk_chars)
        part_dir = output_file.parent / "tts_parts"
        part_dir.mkdir(parents=True, exist_ok=True)
        part_files: list[Path] = []

        for index, chunk in enumerate(chunks, start=1):
            if progress:
                percent = 70 + int(((index - 1) / max(len(chunks), 1)) * 8)
                progress(f"Tạo giọng đọc tiếng Việt ({index}/{len(chunks)})", percent)
            part_file = part_dir / f"part_{index:04d}.mp3"
            if part_file.exists():
                part_file.unlink()
            await self._save_chunk(edge_tts, chunk, part_file, self.get_voice_for_text(chunk, voice_gender))
            part_files.append(part_file)

        if progress:
            progress("Ghép các đoạn giọng đọc", 78)
        if len(part_files) == 1:
            part_files[0].replace(output_file)
        else:
            await _concat_mp3_parts(part_files, output_file)

        if not output_file.exists() or output_file.stat().st_size == 0:
            raise VoiceError("TTS chạy xong nhưng không tạo được file audio giọng đọc.")
        return output_file

    async def synthesize_srt(
        self,
        subtitle_file: Path,
        output_file: Path,
        voice_gender: VoiceGender,
        duration: float,
        progress: Callable[[str, int], None] | None = None,
    ) -> Path:
        try:
            import edge_tts
        except ImportError as exc:
            raise VoiceError("Thiếu edge-tts. Chạy pip install -r requirements.txt trong backend.") from exc

        # Keep voice generation aligned with the exact subtitle events burned into the output.
        events = _group_events(normalize_events(parse_srt(subtitle_file)))
        if not events:
            raise VoiceError("Không có phụ đề hợp lệ để tạo giọng đọc theo thời gian.")

        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise VoiceError("Không tìm thấy FFmpeg để dựng timeline giọng đọc.")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.exists():
            output_file.unlink()

        aligned_dir = output_file.parent / "tts_aligned"
        aligned_dir.mkdir(parents=True, exist_ok=True)
        for old_file in aligned_dir.glob("*"):
            old_file.unlink()

        cursor = 0.0
        total = len(events)
        event_specs: list[tuple[int, SubtitleEvent, float, float]] = []
        for index, event in enumerate(events, start=1):
            next_start = events[index].start if index < total else duration
            available = max(0.4, min(event.end, next_start - 0.05, duration) - event.start)
            gap = max(0.0, event.start - cursor)
            event_specs.append((index, event, available, gap))
            cursor = event.start + available

        semaphore = asyncio.Semaphore(4)
        completed_count = 0

        async def render_event(spec: tuple[int, SubtitleEvent, float, float]) -> list[Path]:
            nonlocal completed_count
            index, event, available, gap = spec
            async with semaphore:
                files: list[Path] = []
                if gap > 0.03:
                    silence_file = aligned_dir / f"{index:04d}_silence.wav"
                    await _create_silence(ffmpeg, gap, silence_file)
                    files.append(silence_file)

                speech_file = aligned_dir / f"{index:04d}_speech.wav"
                if not _has_speakable_content(event.text):
                    await _create_silence(ffmpeg, available, speech_file)
                else:
                    raw_file = aligned_dir / f"{index:04d}_raw.mp3"
                    await self._save_chunk(
                        edge_tts,
                        event.text,
                        raw_file,
                        self.get_voice_for_text(event.text, voice_gender),
                    )
                    raw_duration = await _probe_audio_duration(raw_file)
                    speed = max(1.0, raw_duration / available) if raw_duration > 0 else 1.0
                    await _convert_speech(ffmpeg, raw_file, speech_file, speed=speed, max_duration=available)
                files.append(speech_file)

                completed_count += 1
                if progress:
                    percent = 70 + int((completed_count / max(total, 1)) * 8)
                    progress(f"Tạo giọng đọc khớp phụ đề ({completed_count}/{total})", percent)
                return files

        rendered_groups = await asyncio.gather(*(render_event(spec) for spec in event_specs))
        timeline_files = [path for group in rendered_groups for path in group]

        if duration > cursor + 0.03:
            tail_file = aligned_dir / "9999_tail.wav"
            await _create_silence(ffmpeg, duration - cursor, tail_file)
            timeline_files.append(tail_file)

        if progress:
            progress("Ghép timeline giọng đọc theo phụ đề", 78)
        await _concat_wav_parts(timeline_files, output_file)
        if not output_file.exists() or output_file.stat().st_size == 0:
            raise VoiceError("Không tạo được timeline audio giọng đọc.")
        return output_file

    async def _save_chunk(self, edge_tts, text: str, output_file: Path, voice: str) -> None:
        last_error: Exception | None = None
        for attempt in range(1, settings.tts_chunk_retries + 1):
            if output_file.exists():
                output_file.unlink()
            communicate = edge_tts.Communicate(text, voice=voice, rate="+0%", volume="+0%")
            try:
                await asyncio.wait_for(communicate.save(str(output_file)), timeout=settings.tts_chunk_timeout_seconds)
            except asyncio.TimeoutError as exc:
                last_error = exc
            except Exception as exc:
                last_error = exc
            else:
                if output_file.exists() and output_file.stat().st_size > 0:
                    return
                last_error = VoiceError("TTS không tạo được audio cho một đoạn phụ đề.")

            if attempt < settings.tts_chunk_retries:
                await asyncio.sleep(min(2 * attempt, 8))

        if isinstance(last_error, asyncio.TimeoutError):
            raise VoiceError(
                f"TTS quá {settings.tts_chunk_timeout_seconds} giây ở một đoạn phụ đề sau {settings.tts_chunk_retries} lần thử. "
                "Hãy thử chạy lại khi mạng ổn định."
            ) from last_error
        if last_error:
            preview = _tts_error_preview(text)
            raise VoiceError(f"Edge TTS lỗi sau {settings.tts_chunk_retries} lần thử ở đoạn '{preview}': {last_error}") from last_error
        if not output_file.exists() or output_file.stat().st_size == 0:
            raise VoiceError("TTS không tạo được audio cho một đoạn phụ đề.")


def subtitle_text_for_tts(subtitle_file: Path) -> str:
    events = parse_srt(subtitle_file)
    return "\n".join(event.text.strip() for event in events if event.text.strip())


def get_voice_engine() -> VoiceEngine:
    if settings.voice_engine.lower() in {"edge", "edge-tts", "edge_tts"}:
        return EdgeTtsVoiceEngine()
    return DisabledVoiceEngine()


def _normalize_tts_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\\N", "\n").splitlines()]
    return "\n".join(line for line in lines if line)


def _has_speakable_content(text: str) -> bool:
    normalized = _normalize_tts_text(text)
    if not normalized:
        return False
    return any(char.isalnum() or ("\u4e00" <= char <= "\u9fff") for char in normalized)


def _cjk_ratio(text: str) -> float:
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return 0.0
    cjk_count = sum(1 for char in chars if "\u4e00" <= char <= "\u9fff")
    return cjk_count / len(chars)


def _tts_error_preview(text: str) -> str:
    preview = " ".join(text.split())
    if len(preview) > 80:
        preview = f"{preview[:77]}..."
    return preview


def _split_tts_chunks(text: str, max_chars: int) -> list[str]:
    max_chars = max(500, max_chars)
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        extra = len(line) + (1 if current else 0)
        if current and current_length + extra > max_chars:
            chunks.append("\n".join(current))
            current = [line]
            current_length = len(line)
        else:
            current.append(line)
            current_length += extra

    if current:
        chunks.append("\n".join(current))
    return chunks


def _group_events(events: list[SubtitleEvent]) -> list[SubtitleEvent]:
    grouped: list[SubtitleEvent] = []
    current_start: float | None = None
    current_end = 0.0
    current_text: list[str] = []

    for event in events:
        text = _normalize_tts_text(event.text)
        if not text:
            continue
        joined = " ".join(current_text + [text])
        should_flush = (
            current_start is not None
            and (
                event.start - current_end > 1.2
                or len(joined) > 600
                or event.end - current_start > 25
            )
        )
        if should_flush and current_start is not None:
            grouped.append(SubtitleEvent(current_start, current_end, " ".join(current_text)))
            current_start = None
            current_text = []

        if current_start is None:
            current_start = event.start
        current_end = event.end
        current_text.append(text)

    if current_start is not None and current_text:
        grouped.append(SubtitleEvent(current_start, current_end, " ".join(current_text)))
    return grouped


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?", "…", "。", "！", "？"))


async def _concat_mp3_parts(part_files: list[Path], output_file: Path) -> None:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise VoiceError("Không tìm thấy FFmpeg để ghép các đoạn TTS.")
    list_file = output_file.parent / "tts_concat.txt"
    list_file.write_text("\n".join(f"file '{_concat_path(part)}'" for part in part_files), encoding="utf-8")
    command = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(output_file)]
    await _run_audio_command(command, "Không ghép được các đoạn TTS.")


async def _concat_wav_parts(part_files: list[Path], output_file: Path) -> None:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise VoiceError("Không tìm thấy FFmpeg để ghép timeline audio.")
    list_file = output_file.parent / "tts_timeline_concat.txt"
    list_file.write_text("\n".join(f"file '{_concat_path(part)}'" for part in part_files), encoding="utf-8")
    command = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c:a",
        "pcm_s16le",
        str(output_file),
    ]
    await _run_audio_command(command, "Không ghép được timeline giọng đọc.")


async def _create_silence(ffmpeg: str, duration: float, output_file: Path) -> None:
    command = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=44100:cl=mono",
        "-t",
        f"{max(0.03, duration):.3f}",
        "-c:a",
        "pcm_s16le",
        str(output_file),
    ]
    await _run_audio_command(command, "Không tạo được đoạn im lặng cho giọng đọc.")


async def _convert_speech(ffmpeg: str, input_file: Path, output_file: Path, speed: float, max_duration: float) -> None:
    filters = []
    if speed > 1.01:
        filters.append(_atempo_filter(speed))
    filters.append(f"apad=pad_dur={max_duration:.3f}")
    filters.append(f"atrim=0:{max_duration:.3f}")
    filters.append("asetpts=N/SR/TB")
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(input_file),
        "-af",
        ",".join(filters),
        "-ar",
        "44100",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_file),
    ]
    await _run_audio_command(command, "Không chuyển được đoạn TTS theo thời lượng phụ đề.")


async def _probe_audio_duration(path: Path) -> float:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return 0.0
    return await probe_video_duration(ffmpeg, path)


async def _run_audio_command(command: list[str], error_message: str) -> None:
    def run() -> None:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise VoiceError(f"{error_message} {detail}")

    await asyncio.to_thread(run)


def _atempo_filter(speed: float) -> str:
    factors: list[float] = []
    remaining = speed
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    factors.append(max(0.5, min(2.0, remaining)))
    return ",".join(f"atempo={factor:.4f}" for factor in factors)


def _concat_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "'\\''")
