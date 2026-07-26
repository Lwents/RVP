import asyncio
import json
import re
import subprocess
from pathlib import Path
from typing import Callable, TypedDict

from app.core import settings
from app.models.job import VoiceGender
from app.services.media.ffmpeg import find_ffmpeg, probe_video_duration
from app.services.subtitles.timing import SubtitleEvent, normalize_events, parse_srt


# Per-cue FFmpeg work finishes in seconds; the concat of a full timeline is the
# slowest command. The cap only exists so a wedged process cannot hang the job.
AUDIO_COMMAND_TIMEOUT_SECONDS = 600.0


class VoiceError(RuntimeError):
    pass


class VoiceWordTiming(TypedDict):
    text: str
    start: float
    end: float


class VoiceEngine:
    async def synthesize(
        self,
        text: str,
        output_file: Path,
        voice_gender: VoiceGender,
        progress: Callable[[str, int], None] | None = None,
        timing_file: Path | None = None,
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
        timing_file: Path | None = None,
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
        timing_file: Path | None = None,
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
        if timing_file is not None:
            timing_file.parent.mkdir(parents=True, exist_ok=True)
            timing_file.unlink(missing_ok=True)

        chunks = _split_tts_chunks(normalized, settings.tts_chunk_chars)
        part_dir = output_file.parent / "tts_parts"
        part_dir.mkdir(parents=True, exist_ok=True)
        part_files: list[Path] = []
        word_timings: list[VoiceWordTiming] = []
        part_cursor = 0.0

        for index, chunk in enumerate(chunks, start=1):
            if progress:
                percent = 70 + int(((index - 1) / max(len(chunks), 1)) * 8)
                progress(f"Tạo giọng đọc tiếng Việt ({index}/{len(chunks)})", percent)
            part_file = part_dir / f"part_{index:04d}.mp3"
            if part_file.exists():
                part_file.unlink()
            metadata_file = part_dir / f"part_{index:04d}.metadata.jsonl"
            metadata_file.unlink(missing_ok=True)
            await self._save_chunk(
                edge_tts,
                chunk,
                part_file,
                self.get_voice_for_text(chunk, voice_gender),
                metadata_file=metadata_file if timing_file is not None else None,
            )
            part_duration = await _probe_audio_duration(part_file)
            parsed_timings: list[VoiceWordTiming] = []
            if timing_file is not None and metadata_file.is_file():
                parsed_timings = _read_edge_word_timings(metadata_file, part_cursor)
                word_timings.extend(parsed_timings)
            metadata_duration = (
                max(item["end"] for item in parsed_timings) - part_cursor
                if parsed_timings
                else 0.0
            )
            part_cursor += max(0.0, part_duration, metadata_duration)
            part_files.append(part_file)

        if progress:
            progress("Ghép các đoạn giọng đọc", 78)
        if len(part_files) == 1:
            part_files[0].replace(output_file)
        else:
            await _concat_mp3_parts(part_files, output_file)

        if not output_file.exists() or output_file.stat().st_size == 0:
            raise VoiceError("TTS chạy xong nhưng không tạo được file audio giọng đọc.")
        if timing_file is not None and word_timings:
            final_duration = await _probe_audio_duration(output_file)
            scale = final_duration / part_cursor if final_duration > 0 and part_cursor > 0 else 1.0
            payload = {
                "audio_duration": round(final_duration or part_cursor, 6),
                "words": [
                    {
                        "text": item["text"],
                        "start": round(item["start"] * scale, 6),
                        "end": round(item["end"] * scale, 6),
                    }
                    for item in word_timings
                ],
            }
            timing_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
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

        # Keep one TTS request for every subtitle event that is burned into the
        # output.  Grouping adjacent events lets Edge TTS read the text of a
        # later subtitle before that subtitle's on-screen timecode begins.
        events = _voice_timeline_events(parse_srt(subtitle_file), duration)

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

        event_specs, cursor = _voice_timeline_specs(events, duration)
        if not event_specs:
            raise VoiceError("Không có khoảng thời gian phụ đề đủ dài để tạo giọng đọc.")
        total = len(event_specs)

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
                    # Cap speed to keep voice intelligible; atrim in
                    # _convert_speech will clip any leftover audio.
                    speed = min(speed, 1.5)
                    await _convert_speech(ffmpeg, raw_file, speech_file, speed=speed, max_duration=available)
                files.append(speech_file)

                completed_count += 1
                if progress:
                    percent = 70 + int((completed_count / max(total, 1)) * 8)
                    progress(f"Tạo giọng đọc khớp phụ đề ({completed_count}/{total})", percent)
                return files

        rendered_groups = await _gather_cancel_on_error(
            render_event(spec) for spec in event_specs
        )
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

    async def _save_chunk(
        self,
        edge_tts,
        text: str,
        output_file: Path,
        voice: str,
        *,
        metadata_file: Path | None = None,
    ) -> None:
        last_error: Exception | None = None
        for attempt in range(1, settings.tts_chunk_retries + 1):
            if output_file.exists():
                output_file.unlink()
            if metadata_file is not None:
                metadata_file.unlink(missing_ok=True)
            communicate = edge_tts.Communicate(
                text,
                voice=voice,
                rate="+0%",
                volume="+0%",
                boundary="WordBoundary" if metadata_file is not None else "SentenceBoundary",
            )
            try:
                await asyncio.wait_for(
                    communicate.save(
                        str(output_file),
                        str(metadata_file) if metadata_file is not None else None,
                    ),
                    timeout=settings.tts_chunk_timeout_seconds,
                )
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


def load_voice_word_timings(
    timing_file: Path,
    target_duration: float | None = None,
) -> list[VoiceWordTiming]:
    """Load Edge word boundaries and scale them with any later tempo change."""

    if not timing_file.is_file():
        return []
    try:
        payload = json.loads(timing_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict) or not isinstance(payload.get("words"), list):
        return []
    try:
        source_duration = float(payload.get("audio_duration") or 0.0)
    except (TypeError, ValueError):
        source_duration = 0.0
    scale = (
        float(target_duration) / source_duration
        if target_duration is not None and target_duration > 0 and source_duration > 0
        else 1.0
    )
    result: list[VoiceWordTiming] = []
    for item in payload["words"]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        try:
            start = max(0.0, float(item.get("start")) * scale)
            end = max(start + 0.01, float(item.get("end")) * scale)
        except (TypeError, ValueError):
            continue
        if target_duration is not None and target_duration > 0:
            start = min(start, float(target_duration))
            end = min(max(start + 0.01, end), float(target_duration))
        if text and end > start:
            result.append(VoiceWordTiming(text=text, start=start, end=end))
    return sorted(result, key=lambda item: (item["start"], item["end"]))


def _read_edge_word_timings(metadata_file: Path, part_start: float) -> list[VoiceWordTiming]:
    result: list[VoiceWordTiming] = []
    try:
        lines = metadata_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return result
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict) or item.get("type") != "WordBoundary":
            continue
        text = str(item.get("text") or "").strip()
        try:
            offset = max(0.0, float(item.get("offset") or 0.0) / 10_000_000.0)
            duration = max(0.01, float(item.get("duration") or 0.0) / 10_000_000.0)
        except (TypeError, ValueError):
            continue
        if text:
            result.append(
                VoiceWordTiming(
                    text=text,
                    start=part_start + offset,
                    end=part_start + offset + duration,
                )
            )
    return result


def get_voice_engine() -> VoiceEngine:
    if settings.voice_engine.lower() in {"edge", "edge-tts", "edge_tts"}:
        return EdgeTtsVoiceEngine()
    return DisabledVoiceEngine()


def _clamp_events_to_duration(
    events: list[SubtitleEvent],
    duration: float,
) -> list[SubtitleEvent]:
    """Return timeline-safe copies without mutating frozen SubtitleEvent objects."""
    return [
        SubtitleEvent(event.start, min(event.end, duration), event.text)
        for event in events
        if event.start < duration - 0.1
    ]


def _merge_voice_events(
    events: list[SubtitleEvent],
    max_gap: float = 0.3,
    max_chars: int = 120,
    max_duration: float = 8.0,
) -> list[SubtitleEvent]:
    """Merge adjacent subtitle events into single TTS groups for natural speech.

    Subtitle files often split a sentence across multiple display cues (e.g.
    word-level Whisper timestamps or YouTube auto-subs).  When each cue is
    sent to the TTS engine as a separate request, the synthesiser restarts its
    prosody, producing an audible cut mid-sentence.

    This function merges consecutive cues whose gap is small and whose combined
    text does not end a sentence, so the TTS engine reads the whole phrase in
    one natural breath.
    """
    if not events:
        return events

    merged: list[SubtitleEvent] = []
    cur_start = events[0].start
    cur_end = events[0].end
    cur_text = events[0].text.strip()

    for event in events[1:]:
        gap = event.start - cur_end
        combined_text = f"{cur_text} {event.text.strip()}"
        combined_duration = event.end - cur_start

        can_merge = (
            gap <= max_gap
            and len(combined_text) <= max_chars
            and combined_duration <= max_duration
            and not _ends_sentence(cur_text)
        )

        if can_merge:
            cur_end = event.end
            cur_text = combined_text
        else:
            merged.append(SubtitleEvent(cur_start, cur_end, cur_text))
            cur_start = event.start
            cur_end = event.end
            cur_text = event.text.strip()

    merged.append(SubtitleEvent(cur_start, cur_end, cur_text))
    return merged


def _voice_timeline_events(
    events: list[SubtitleEvent],
    duration: float,
) -> list[SubtitleEvent]:
    """Merge adjacent cues into natural speech groups and clamp to video length."""

    return _clamp_events_to_duration(
        _merge_voice_events(normalize_events(events)),
        duration,
    )


def _voice_timeline_specs(
    events: list[SubtitleEvent],
    duration: float,
) -> tuple[list[tuple[int, SubtitleEvent, float, float]], float]:
    """Allocate sequential speech without ever crossing the next cue or video end."""

    minimum_slot = 0.03
    boundary_gap = 0.05
    cursor = 0.0
    specs: list[tuple[int, SubtitleEvent, float, float]] = []
    total = len(events)
    for index, event in enumerate(events, start=1):
        next_start = events[index].start if index < total else duration
        slot_end = min(event.end, next_start, duration)
        slot_duration = slot_end - event.start
        if slot_duration < minimum_slot:
            continue

        # Leave a small pause only when this cue touches the next one.  The
        # pause must come from the cue's real slot, never extend that slot.
        touches_next = index < total and slot_end >= next_start - 1e-9
        separator = (
            min(boundary_gap, max(0.0, slot_duration - minimum_slot))
            if touches_next
            else 0.0
        )
        available = slot_duration - separator
        gap = max(0.0, event.start - cursor)
        specs.append((index, event, available, gap))
        cursor = event.start + available
    return specs, cursor


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
    sentence_units = [
        unit.strip()
        for line in text.splitlines()
        for unit in re.split(r"(?<=[.!?…])\s+", line.strip())
        if unit.strip()
    ]
    pieces: list[str] = []
    for unit in sentence_units:
        if len(unit) <= max_chars:
            pieces.append(unit)
            continue
        current_words: list[str] = []
        for word in unit.split():
            candidate = " ".join([*current_words, word])
            if current_words and len(candidate) > max_chars:
                pieces.append(" ".join(current_words))
                current_words = [word]
            else:
                current_words.append(word)
        if current_words:
            pieces.append(" ".join(current_words))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


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


async def _gather_cancel_on_error(coroutines) -> list:
    """Gather that never leaves siblings running (and unawaited) after a failure."""
    tasks = [asyncio.ensure_future(coroutine) for coroutine in coroutines]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        # Awaiting the cancelled siblings stops them from writing into
        # tts_aligned/ after the job failed and retrieves their exceptions.
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def _run_audio_command(
    command: list[str],
    error_message: str,
    timeout: float = AUDIO_COMMAND_TIMEOUT_SECONDS,
) -> None:
    def run() -> None:
        # synthesize_srt issues one command per subtitle cue; without a timeout a
        # single wedged FFmpeg would stall the whole dubbing job forever.
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            # subprocess.run kills the child before re-raising.
            raise VoiceError(
                f"{error_message} FFmpeg không phản hồi trong {int(timeout)} giây nên đã bị dừng."
            ) from exc
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
