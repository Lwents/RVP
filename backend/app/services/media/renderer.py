from pathlib import Path

from app.core import settings
from app.models.job import BgmMode, DubbingRequest, LogoPosition
from app.services.media.ffmpeg import copy_mp4, probe_video_duration, probe_video_size, run_ffmpeg_with_progress
from app.services.subtitles.ass import srt_to_positioned_ass


async def render_video(
    ffmpeg: str,
    source_video: Path,
    output_file: Path,
    work_dir: Path,
    request: DubbingRequest,
    on_progress: callable,
    subtitle_file: Path | None = None,
    narration_audio: Path | None = None,
    progress_start: int = 72,
) -> None:
    if _can_stream_copy(request, subtitle_file, narration_audio):
        await copy_mp4(ffmpeg, source_video, output_file)
        on_progress(98)
        return

    width, height = await probe_video_size(ffmpeg, source_video)
    command = [ffmpeg, "-y", "-i", str(source_video)]
    next_input_index = 1
    video_label = "[0:v]"
    filter_parts: list[str] = []

    if request.watermark_file_name:
        watermark = Path(settings.storage_dir) / request.watermark_file_name
        if watermark.exists():
            command.extend(["-i", str(watermark)])
            watermark_index = next_input_index
            next_input_index += 1
            filter_parts.append(f"[{watermark_index}:v]scale={request.logo_width}:-1[wm]")
            filter_parts.append(f"{video_label}[wm]overlay={_overlay_logo_filter(request.logo_position)}[vwm]")
            video_label = "[vwm]"

    narration_index: int | None = None
    if narration_audio and narration_audio.exists():
        command.extend(["-i", str(narration_audio)])
        narration_index = next_input_index
        next_input_index += 1

    if subtitle_file:
        ass_file = srt_to_positioned_ass(subtitle_file, work_dir / "subtitles.positioned.ass", width, height, request)
        subtitle_filters: list[str] = []

        if request.subtitle_box_enabled and request.subtitle_box_opacity > 0:
            box_height = max(24, round(height * request.subtitle_box_height_percent / 100))
            box_y = round((height * request.subtitle_y_percent / 100) - (box_height / 2))
            box_y = max(0, min(height - box_height, box_y))
            alpha = round(request.subtitle_box_opacity / 100, 2)
            subtitle_filters.append(f"drawbox=x=0:y={box_y}:w=iw:h={box_height}:color=black@{alpha}:t=fill")

        subtitle_filters.append(f"subtitles='{_subtitle_filter_path(ass_file)}'")
        filter_parts.append(f"{video_label}{','.join(subtitle_filters)}[vsub]")
        video_label = "[vsub]"

    audio_args = _audio_output_args(request, narration_index is not None)
    map_audio = request.bgm_mode != BgmMode.none
    if filter_parts:
        command.extend(["-filter_complex", ";".join(filter_parts), "-map", video_label])
        if map_audio:
            command.extend(["-map", f"{narration_index}:a" if narration_index is not None else "0:a?"])
    else:
        command.extend(["-map", "0:v"])
        if map_audio:
            command.extend(["-map", f"{narration_index}:a" if narration_index is not None else "0:a?"])

    command.extend(_video_output_args())
    command.extend(audio_args)
    command.extend(["-movflags", "+faststart", str(output_file)])

    await run_ffmpeg_with_progress(
        command,
        "Không render được video đầu ra.",
        duration=await probe_video_duration(ffmpeg, source_video),
        progress_start=progress_start,
        progress_end=98,
        on_progress=on_progress,
    )


async def burn_subtitles(
    ffmpeg: str,
    source_video: Path,
    subtitle_file: Path,
    output_file: Path,
    work_dir: Path,
    request: DubbingRequest,
    on_progress: callable,
) -> None:
    await render_video(ffmpeg, source_video, output_file, work_dir, request, on_progress, subtitle_file)


def _can_stream_copy(request: DubbingRequest, subtitle_file: Path | None, narration_audio: Path | None) -> bool:
    return subtitle_file is None and narration_audio is None and not request.watermark_file_name and request.bgm_mode == BgmMode.demucs


def _audio_output_args(request: DubbingRequest, has_narration: bool) -> list[str]:
    if request.bgm_mode == BgmMode.none:
        return ["-an"]
    if has_narration:
        return ["-c:a", "aac", "-b:a", "192k"]
    if request.bgm_mode == BgmMode.ducking:
        volume = round(10 ** (request.ducking_volume_db / 20), 4)
        return ["-af", f"volume={volume}", "-c:a", "aac", "-b:a", "128k"]
    return ["-c:a", "copy"]


def _video_output_args() -> list[str]:
    encoder = settings.video_encoder.lower()
    if encoder == "h264_nvenc":
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            _nvenc_preset(settings.video_preset),
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            str(settings.video_crf),
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
        ]
    if encoder == "h264_amf":
        return ["-c:v", "h264_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", str(settings.video_crf), "-qp_p", str(settings.video_crf), "-pix_fmt", "yuv420p"]
    if encoder == "h264_qsv":
        return ["-c:v", "h264_qsv", "-global_quality", str(settings.video_crf), "-preset", _qsv_preset(settings.video_preset), "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", settings.video_preset, "-crf", str(settings.video_crf), "-pix_fmt", "yuv420p"]


def _nvenc_preset(preset: str) -> str:
    normalized = preset.lower().strip()
    if normalized in {"p1", "p2", "p3", "p4", "p5", "p6", "p7"}:
        return normalized
    aliases = {
        "ultrafast": "p1",
        "superfast": "p1",
        "veryfast": "p2",
        "faster": "p3",
        "fast": "p3",
        "medium": "p4",
        "slow": "p5",
        "slower": "p6",
        "veryslow": "p7",
    }
    return aliases.get(normalized, "p4")


def _qsv_preset(preset: str) -> str:
    normalized = preset.lower().strip()
    if normalized in {"veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"}:
        return normalized
    aliases = {
        "p1": "veryfast",
        "p2": "faster",
        "p3": "fast",
        "p4": "medium",
        "p5": "slow",
        "p6": "slower",
        "p7": "veryslow",
        "ultrafast": "veryfast",
        "superfast": "veryfast",
    }
    return aliases.get(normalized, "medium")


def _overlay_logo_filter(position: LogoPosition) -> str:
    margin = 18
    positions = {
        LogoPosition.top_right: f"W-w-{margin}:{margin}",
        LogoPosition.top_left: f"{margin}:{margin}",
        LogoPosition.bottom_right: f"W-w-{margin}:H-h-{margin}",
        LogoPosition.bottom_left: f"{margin}:H-h-{margin}",
    }
    return positions[position]


def _subtitle_filter_path(path: Path) -> str:
    value = path.resolve().as_posix()
    value = value.replace(":", r"\:")
    value = value.replace("'", r"\'")
    return value
