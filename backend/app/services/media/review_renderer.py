from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypedDict

from app.core import settings
from app.models.job import DubbingRequest
from app.services.media.ffmpeg import probe_video_duration, run_command, run_command_with_progress
from app.services.media.renderer import _append_soft_box_blur, _video_output_args
from app.services.subtitles.ass import srt_to_positioned_ass, subtitle_font_dir, write_srt
from app.services.subtitles.timing import SubtitleEvent


class ReviewSceneHint(TypedDict, total=False):
    time_hint: str | None
    start_seconds: float | None
    end_seconds: float | None
    narration: str | None
    voice_start: float | None
    voice_end: float | None
    duration_seconds: float | None


@dataclass(frozen=True)
class ReviewScenePlan:
    """One verified source window fitted to one narration time range."""

    source_start: float
    source_duration: float
    output_duration: float


def build_review_starts(source_duration: float, target_seconds: float, clip_count: int) -> list[float]:
    clip_count = max(1, clip_count)
    clip_length = max(1.0, target_seconds / clip_count)
    max_start = max(0.0, source_duration - clip_length - 0.5)
    if clip_count == 1 or max_start <= 0:
        return [0.0]
    return [min(max_start, (max_start * index) / (clip_count - 1)) for index in range(clip_count)]


def build_review_scenes(
    source_duration: float,
    target_seconds: float,
    scene_hints: list[ReviewSceneHint] | int | None,
) -> list[tuple[float, float]]:
    """Return the source start and output duration for compatibility.

    Rendering needs both the short, verified source span and the longer voice
    span, so :func:`build_review_scene_plans` is authoritative.  Keeping this
    lightweight projection avoids breaking callers that only inspect the
    output schedule.
    """

    return [
        (plan.source_start, plan.output_duration)
        for plan in build_review_scene_plans(source_duration, target_seconds, scene_hints)
    ]


def build_review_scene_plans(
    source_duration: float,
    target_seconds: float,
    scene_hints: list[ReviewSceneHint] | int | None,
) -> list[ReviewScenePlan]:
    if isinstance(scene_hints, int):
        starts = build_review_starts(source_duration, target_seconds, scene_hints)
        clip_seconds = target_seconds / max(len(starts), 1)
        return [ReviewScenePlan(start, clip_seconds, clip_seconds) for start in starts]

    hints = scene_hints or []
    if not hints:
        clip_count = max(8, min(24, math.ceil(target_seconds / 30)))
        starts = build_review_starts(source_duration, target_seconds, clip_count)
        clip_seconds = target_seconds / max(len(starts), 1)
        return [ReviewScenePlan(start, clip_seconds, clip_seconds) for start in starts]

    durations: list[float] = []
    for hint in hints:
        voice_start = _coerce_seconds(hint.get("voice_start"))
        voice_end = _coerce_seconds(hint.get("voice_end"))
        explicit_duration = _coerce_seconds(hint.get("duration_seconds"))
        source_start = _coerce_seconds(hint.get("start_seconds"))
        source_end = _coerce_seconds(hint.get("end_seconds"))
        if explicit_duration is not None and explicit_duration > 0:
            duration = explicit_duration
        elif voice_start is not None and voice_end is not None and voice_end > voice_start:
            duration = voice_end - voice_start
        elif source_start is not None and source_end is not None and source_end > source_start:
            duration = source_end - source_start
        else:
            duration = 3.0
        durations.append(max(0.8, duration))

    # EDL durations are authoritative. Only compensate tiny probe/rounding drift;
    # never stretch each scene by narration character count.
    total_duration = sum(durations)
    if durations and target_seconds > 0 and abs(total_duration - target_seconds) <= max(0.5, target_seconds * 0.03):
        durations[-1] = max(0.8, durations[-1] + target_seconds - total_duration)
    fallback_starts = build_review_starts(source_duration, target_seconds, len(hints))

    scenes: list[ReviewScenePlan] = []
    for index, hint in enumerate(hints):
        start = _coerce_seconds(hint.get("start_seconds"))
        end = _coerce_seconds(hint.get("end_seconds"))
        if start is None or end is None:
            parsed = _parse_time_hint(hint.get("time_hint") or "")
            if parsed:
                start, end = parsed

        fallback_start = fallback_starts[min(index, len(fallback_starts) - 1)]
        start = fallback_start if start is None else start
        end = start + durations[index] if end is None else end
        output_duration = min(max(0.8, durations[index]), max(source_duration, 0.8))

        # The EDL interval is centred on the exact keyframe that multimodal QA
        # accepted.  Expanding a 2.5s verified interval to a 5s voice interval
        # can cross two shot cuts and show an unrelated action at the beginning
        # or end of the sentence.  Preserve that evidence window and fit it to
        # the voice duration during encoding.  This is continuous slow motion,
        # not a visible loop of the same slice.
        safe_source_duration = max(0.01, source_duration)
        requested_start = max(0.0, min(start, safe_source_duration))
        requested_end = max(requested_start, min(end, safe_source_duration))
        evidence_center = (requested_start + requested_end) / 2.0
        evidence_duration = max(0.0, requested_end - requested_start)
        source_clip_duration = min(
            output_duration,
            max(min(0.8, safe_source_duration), evidence_duration),
        )
        clip_start = evidence_center - source_clip_duration / 2.0
        clip_start = max(
            0.0,
            min(clip_start, max(0.0, safe_source_duration - source_clip_duration)),
        )
        scenes.append(
            ReviewScenePlan(
                source_start=clip_start,
                source_duration=source_clip_duration,
                output_duration=output_duration,
            )
        )
    scenes = _partition_repeated_verified_windows(scenes)
    return _quantize_review_plan_durations(scenes, target_seconds)


def _partition_repeated_verified_windows(
    plans: list[ReviewScenePlan],
) -> list[ReviewScenePlan]:
    """Play an identical adjacent evidence window only once.

    Two different atomic facts may be visible in the same verified keyframe.
    Splitting that short source window into chronological pieces keeps both
    sentences grounded without replaying the exact same frames twice.
    """

    result = list(plans)
    index = 0
    while index < len(result):
        end = index + 1
        while end < len(result):
            first = result[index]
            candidate = result[end]
            if (
                abs(candidate.source_start - first.source_start) > 0.05
                or abs(candidate.source_duration - first.source_duration) > 0.05
            ):
                break
            end += 1
        count = end - index
        if count > 1:
            first = result[index]
            slice_duration = first.source_duration / count
            for offset in range(count):
                original = result[index + offset]
                result[index + offset] = ReviewScenePlan(
                    source_start=first.source_start + offset * slice_duration,
                    source_duration=slice_duration,
                    output_duration=original.output_duration,
                )
        index = end
    return result


def _quantize_review_plan_durations(
    plans: list[ReviewScenePlan],
    target_seconds: float,
    fps: int = 30,
) -> list[ReviewScenePlan]:
    """Align every cut boundary to the final CFR timeline without drift."""

    if not plans:
        return plans
    result: list[ReviewScenePlan] = []
    cumulative = 0.0
    previous_frame = 0
    final_frame = max(len(plans), math.ceil(max(0.0, target_seconds) * fps))
    for index, plan in enumerate(plans):
        cumulative += plan.output_duration
        boundary_frame = (
            final_frame
            if index == len(plans) - 1
            else max(previous_frame + 1, round(cumulative * fps))
        )
        output_duration = (boundary_frame - previous_frame) / fps
        result.append(
            ReviewScenePlan(
                source_start=plan.source_start,
                source_duration=plan.source_duration,
                output_duration=output_duration,
            )
        )
        previous_frame = boundary_frame
    return result


async def render_movie_review_video(
    ffmpeg: str,
    source_video: Path,
    narration_audio: Path,
    subtitle_file: Path,
    output_file: Path,
    work_dir: Path,
    target_minutes: int,
    scene_hints: list[ReviewSceneHint] | int | None,
    on_progress: Callable[[int], None],
    render_request: DubbingRequest | None = None,
) -> Path:
    narration_duration = await probe_video_duration(ffmpeg, narration_audio)
    target_seconds = narration_duration if narration_duration > 0 else max(1.0, float(target_minutes) * 60.0)
    source_duration = await probe_video_duration(ffmpeg, source_video)
    scenes = build_review_scene_plans(source_duration, target_seconds, scene_hints)

    segment_dir = work_dir / "review_segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    for old_file in segment_dir.glob("*"):
        old_file.unlink()

    segment_files: list[Path] = []
    for index, plan in enumerate(scenes, start=1):
        percent = 84 + int(((index - 1) / max(len(scenes), 1)) * 8)
        on_progress(percent)
        segment = segment_dir / f"segment_{index:04d}.mp4"
        tempo = plan.output_duration / max(plan.source_duration, 0.01)
        output_frames = max(1, round(plan.output_duration * 30))
        video_filter = (
            "scale=1280:720:force_original_aspect_ratio=decrease,"
            "pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,"
            f"setpts={tempo:.8f}*(PTS-STARTPTS),fps=30,"
            f"tpad=stop_mode=clone:stop=-1,trim=end_frame={output_frames},"
            "setpts=PTS-STARTPTS"
        )
        command = [
            ffmpeg,
            "-y",
            "-ss",
            f"{plan.source_start:.3f}",
            "-t",
            f"{plan.source_duration:.3f}",
            "-i",
            str(source_video),
            "-an",
            "-vf",
            video_filter,
            *_video_output_args(),
            "-movflags",
            "+faststart",
            str(segment),
        ]
        await run_command(command, "Khong cat duoc canh review.")
        segment_files.append(segment)

    concat_file = work_dir / "review_concat.txt"
    concat_file.write_text("\n".join(f"file '{_concat_path(path)}'" for path in segment_files), encoding="utf-8")
    silent_video = work_dir / "review_video_silent.mp4"
    await run_command(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(silent_video),
        ],
        "Khong ghep duoc cac canh review.",
    )

    on_progress(93)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    command = _build_review_output_command(
        ffmpeg,
        silent_video,
        narration_audio,
        subtitle_file,
        output_file,
        work_dir,
        target_seconds,
        render_request,
    )
    await run_command_with_progress(
        command,
        "Khong render duoc video review dau ra.",
        target_seconds,
        93,
        99,
        on_progress,
    )
    return output_file


def _build_review_output_command(
    ffmpeg: str,
    silent_video: Path,
    narration_audio: Path,
    subtitle_file: Path,
    output_file: Path,
    work_dir: Path,
    target_seconds: float,
    request: DubbingRequest | None,
) -> list[str]:
    command = [ffmpeg, "-y", "-i", str(silent_video), "-i", str(narration_audio)]
    filter_parts: list[str] = ["[0:v]setpts=PTS-STARTPTS[vbase]"]
    video_label = "[vbase]"
    video_map = video_label
    next_input_index = 2
    width, height = 1280, 720

    if request:
        if request.blur_box_enabled:
            # Use one fixed, feathered subtitle band for the entire review.
            # This avoids an extra OpenCV detect/inpaint/encode pass and keeps
            # the blur position stable instead of following text per frame.
            video_label = _append_soft_box_blur(
                filter_parts,
                video_label,
                width,
                height,
                [(
                    5.0,
                    float(request.blur_box_y_percent),
                    90.0,
                    float(request.blur_box_height_percent),
                    None,
                    None,
                )],
                "reviewstatic",
                strong_subtitle=True,
            )

        if request.custom_blur_boxes:
            video_label = _append_soft_box_blur(
                filter_parts,
                video_label,
                width,
                height,
                [
                    (
                        box.x_percent,
                        box.y_percent,
                        box.width_percent,
                        box.height_percent,
                        box.start_seconds,
                        box.end_seconds,
                    )
                    for box in request.custom_blur_boxes
                ],
                "reviewcustom",
            )

        subtitle_filters: list[str] = []
        if request.cinematic_bars_enabled:
            bar_h = round(height * request.cinematic_bars_height_percent / 100)
            if bar_h > 0:
                subtitle_filters.append(f"drawbox=x=0:y=0:w=iw:h={bar_h}:color=black:t=fill")
                subtitle_filters.append(f"drawbox=x=0:y=ih-{bar_h}:w=iw:h={bar_h}:color=black:t=fill")

        if request.hard_subtitles:
            ass_file = srt_to_positioned_ass(subtitle_file, work_dir / "review_subtitles.positioned.ass", width, height, request)
            subtitle_filters.append(
                f"subtitles='{_filter_path(ass_file)}':"
                f"fontsdir='{_filter_path(subtitle_font_dir())}':wrap_unicode=1"
            )

        if subtitle_filters:
            filter_parts.append(f"{video_label}{','.join(subtitle_filters)}[vsub]")
            video_label = "[vsub]"

        if request.watermark_file_name and request.logo_enabled:
            watermark = Path(settings.storage_dir) / request.watermark_file_name
            if watermark.exists():
                command.extend(["-i", str(watermark)])
                watermark_index = next_input_index
                next_input_index += 1
                filter_parts.append(f"[{watermark_index}:v]scale={request.logo_width}:-1[wm]")
                x_expr = f"(W-w)*{request.logo_x_percent}/100"
                y_expr = f"(H-h)*{request.logo_y_percent}/100"
                filter_parts.append(f"{video_label}[wm]overlay={x_expr}:{y_expr}[vwm]")
                video_label = "[vwm]"
    else:
        filter_parts.append(
            f"{video_label}subtitles='{_filter_path(subtitle_file)}':"
            f"fontsdir='{_filter_path(subtitle_font_dir())}':wrap_unicode=1:"
            "force_style='FontName=Be Vietnam Pro ExtraBold,FontSize=40,"
            "PrimaryColour=&H0000F2FF,OutlineColour=&H00000000,"
            "Bold=-1,Italic=-1,ScaleX=93,BorderStyle=1,Outline=4,"
            "Shadow=2,Alignment=2,MarginV=58'[v]"
        )
        video_label = "[v]"

    video_map = video_label

    filter_parts.append(f"[1:a]apad,atrim=0:{target_seconds:.3f},asetpts=N/SR/TB[a]")

    command.extend(["-filter_complex", ";".join(filter_parts), "-map", video_map, "-map", "[a]"])
    command.extend(
        [
            "-t",
            f"{target_seconds:.3f}",
            *_video_output_args(),
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(output_file),
        ]
    )
    return command


def write_review_subtitles(
    narration_script: str,
    output_file: Path,
    target_minutes: int,
    scene_hints: list[ReviewSceneHint] | None = None,
    target_seconds: float | None = None,
) -> Path:
    target_seconds = target_seconds if target_seconds and target_seconds > 0 else max(1.0, float(target_minutes) * 60.0)
    chunks = _split_review_subtitle_chunks(narration_script)
    if not chunks:
        chunks = ["Video review phim."]

    if scene_hints and all(
        _coerce_seconds(hint.get("voice_start")) is not None
        and _coerce_seconds(hint.get("voice_end")) is not None
        for hint in scene_hints
    ):
        events: list[SubtitleEvent] = []
        for hint in scene_hints:
            narration = " ".join(str(hint.get("narration") or "").split())
            if not narration:
                continue
            start = float(hint.get("voice_start") or 0.0)
            end = float(hint.get("voice_end") or start + 0.8)
            beat_chunks = _split_review_subtitle_chunks(narration)
            if not beat_chunks:
                continue
            chunk_duration = max(0.15, (end - start) / len(beat_chunks))
            for index, chunk in enumerate(beat_chunks):
                chunk_start = start + index * chunk_duration
                chunk_end = min(end, start + (index + 1) * chunk_duration)
                events.append(SubtitleEvent(start=chunk_start, end=chunk_end, text=chunk))
        return write_srt(events, output_file)

    if scene_hints:
        durations = _allocate_weighted_durations(
            [_narration_weight(hint.get("narration")) for hint in scene_hints],
            target_seconds,
        )
        chunk_counts = _allocate_chunk_counts(len(chunks), durations)
    else:
        durations = [target_seconds]
        chunk_counts = [len(chunks)]

    events: list[SubtitleEvent] = []
    cursor = 0.0
    chunk_index = 0
    for beat_index, beat_duration in enumerate(durations):
        count = chunk_counts[beat_index] if beat_index < len(chunk_counts) else 0
        beat_chunks = chunks[chunk_index : chunk_index + count]
        chunk_index += count
        if not beat_chunks:
            cursor += beat_duration
            continue

        duration_per_chunk = beat_duration / len(beat_chunks)
        for index, chunk in enumerate(beat_chunks):
            start = cursor + index * duration_per_chunk
            end = min(target_seconds, cursor + (index + 1) * duration_per_chunk)
            events.append(SubtitleEvent(start=start, end=end, text=chunk))
        cursor += beat_duration

    if chunk_index < len(chunks):
        tail_start = events[-1].end if events else 0.0
        remaining = chunks[chunk_index:]
        duration_per_chunk = max(0.5, (target_seconds - tail_start) / len(remaining))
        for index, chunk in enumerate(remaining):
            start = min(target_seconds, tail_start + index * duration_per_chunk)
            end = min(target_seconds, tail_start + (index + 1) * duration_per_chunk)
            events.append(SubtitleEvent(start=start, end=end, text=chunk))

    return write_srt(events, output_file)


def _split_review_subtitle_chunks(text: str, max_chars: int = 38) -> list[str]:
    normalized = " ".join(text.replace("\n", " ").split())
    if not normalized:
        return []

    sentences = [part.strip() for part in re.split(r"(?<=[.!?…])\s+", normalized) if part.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = word
            else:
                current = candidate
        if current and current[-1:] in ".!?…" and len(current) >= max_chars * 0.5:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def _concat_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "'\\''")


def _filter_path(path: Path) -> str:
    return path.resolve().as_posix().replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def _narration_weight(text: str | None) -> float:
    normalized = " ".join((text or "").split())
    if not normalized:
        return 1.0
    return max(1.0, float(len(normalized)))


def _allocate_weighted_durations(weights: list[float], target_seconds: float) -> list[float]:
    if not weights:
        return [target_seconds]

    total_weight = sum(max(1.0, weight) for weight in weights)
    if total_weight <= 0:
        return [target_seconds / len(weights) for _ in weights]

    min_duration = min(7.0, max(2.5, target_seconds / len(weights) * 0.45))
    durations = [max(min_duration, target_seconds * max(1.0, weight) / total_weight) for weight in weights]
    total_duration = sum(durations)
    if total_duration <= 0:
        return [target_seconds / len(weights) for _ in weights]

    scale = target_seconds / total_duration
    fitted = [max(2.5, duration * scale) for duration in durations]
    drift = target_seconds - sum(fitted)
    fitted[-1] = max(2.5, fitted[-1] + drift)
    return fitted


def _allocate_chunk_counts(total_chunks: int, durations: list[float]) -> list[int]:
    if not durations:
        return [total_chunks]
    if total_chunks <= 0:
        return [0 for _ in durations]

    total_duration = sum(durations)
    if total_duration <= 0:
        counts = [total_chunks // len(durations) for _ in durations]
    else:
        counts = [int(round(total_chunks * duration / total_duration)) for duration in durations]

    if total_chunks >= len(durations):
        counts = [max(1, count) for count in counts]
    else:
        counts = [0 for _ in durations]
        for index in range(total_chunks):
            counts[index] = 1

    diff = total_chunks - sum(counts)
    index = 0
    while diff != 0 and counts:
        target = index % len(counts)
        if diff > 0:
            counts[target] += 1
            diff -= 1
        elif counts[target] > 0:
            counts[target] -= 1
            diff += 1
        index += 1
    return counts


def _coerce_seconds(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


def _parse_time_hint(value: str) -> tuple[float, float] | None:
    matches = re.findall(r"(\d{1,2}:\d{2}(?::\d{2})?(?:[,.]\d+)?)", value)
    if len(matches) < 2:
        return None
    start = _timestamp_to_seconds(matches[0])
    end = _timestamp_to_seconds(matches[1])
    if start is None or end is None or end <= start:
        return None
    return start, end


def _timestamp_to_seconds(value: str) -> float | None:
    parts = value.replace(",", ".").split(":")
    try:
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return None
    return None
