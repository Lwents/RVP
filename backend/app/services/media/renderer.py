from pathlib import Path

from app.core import settings
from app.models.job import BgmMode, DubbingRequest
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
    bgm_audio: Path | None = None,
    progress_start: int = 72,
) -> None:
    if _can_stream_copy(request, subtitle_file, narration_audio, bgm_audio):
        await copy_mp4(ffmpeg, source_video, output_file)
        on_progress(98)
        return

    width, height = await probe_video_size(ffmpeg, source_video)
    command = [ffmpeg, "-y", "-i", str(source_video)]
    next_input_index = 1
    video_label = "[0:v]"
    filter_parts: list[str] = []

    if request.blur_box_enabled:
        blur_h = max(2, round(height * request.blur_box_height_percent / 100))
        blur_y = round(height * request.blur_box_y_percent / 100)
        blur_y = max(0, min(height - blur_h, blur_y))
        filter_parts.append(f"{video_label}crop=iw:{blur_h}:0:{blur_y},boxblur=20:5[blurred]")
        filter_parts.append(f"{video_label}[blurred]overlay=0:{blur_y}[vblur]")
        video_label = "[vblur]"

    for i, custom_blur in enumerate(request.custom_blur_boxes):
        cb_w = max(2, round(width * custom_blur.width_percent / 100))
        cb_h = max(2, round(height * custom_blur.height_percent / 100))
        cb_x = round(width * custom_blur.x_percent / 100)
        cb_y = round(height * custom_blur.y_percent / 100)
        
        # Ensure dimensions and coordinates don't exceed video boundaries
        cb_w = min(cb_w, width - cb_x)
        cb_h = min(cb_h, height - cb_y)
        
        filter_parts.append(f"{video_label}crop={cb_w}:{cb_h}:{cb_x}:{cb_y},boxblur=20:5[cblur_{i}]")
        filter_parts.append(f"{video_label}[cblur_{i}]overlay={cb_x}:{cb_y}[vcb_{i}]")
        video_label = f"[vcb_{i}]"



    narration_index: int | None = None
    if narration_audio and narration_audio.exists():
        command.extend(["-i", str(narration_audio)])
        narration_index = next_input_index
        next_input_index += 1

    bgm_index: int | None = None
    if bgm_audio and bgm_audio.exists():
        command.extend(["-i", str(bgm_audio)])
        bgm_index = next_input_index
        next_input_index += 1

    subtitle_filters: list[str] = []
    if request.cinematic_bars_enabled:
        bar_h = round(height * request.cinematic_bars_height_percent / 100)
        if bar_h > 0:
            subtitle_filters.append(f"drawbox=x=0:y=0:w=iw:h={bar_h}:color=black:t=fill")
            subtitle_filters.append(f"drawbox=x=0:y=ih-{bar_h}:w=iw:h={bar_h}:color=black:t=fill")

    if subtitle_file:
        ass_file = srt_to_positioned_ass(subtitle_file, work_dir / "subtitles.positioned.ass", width, height, request)

        if request.subtitle_box_enabled and request.subtitle_box_opacity > 0:
            box_height = max(24, round(height * request.subtitle_box_height_percent / 100))
            box_y = round((height * request.subtitle_y_percent / 100) - (box_height / 2))
            box_y = max(0, min(height - box_height, box_y))
            alpha = round(request.subtitle_box_opacity / 100, 2)
            subtitle_filters.append(f"drawbox=x=0:y={box_y}:w=iw:h={box_height}:color=black@{alpha}:t=fill")

        subtitle_filters.append(f"subtitles='{_subtitle_filter_path(ass_file)}'")
    
    if subtitle_filters:
        filter_parts.append(f"{video_label}{','.join(subtitle_filters)}[vfinal]")
        video_label = "[vfinal]"

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

    if request.output_resolution != "original":
        res_map = {
            "720p": 720,
            "1080p": 1080,
            "1440p": 1440,
            "4k": 2160
        }
        target_h = res_map.get(request.output_resolution)
        if target_h:
            filter_parts.append(f"{video_label}scale=-2:{target_h}[vscaled]")
            video_label = "[vscaled]"

    audio_args = _audio_output_args(request, narration_index is not None)
    audio_label: str | None = None
    if narration_index is not None:
        if request.bgm_mode == BgmMode.none:
            # Only narration, no background
            audio_label = f"{narration_index}:a"
        elif request.bgm_mode == BgmMode.demucs and bgm_index is None:
            # Demucs mode is used to remove the original voice. If separation
            # fails, keep narration only instead of mixing Chinese source audio.
            audio_label = f"{narration_index}:a"
        else:
            # Mix background and narration
            bgm_volume = 1.0
            if request.bgm_mode == BgmMode.ducking:
                bgm_volume = round(10 ** (request.ducking_volume_db / 20), 4)
            bgm_src = f"[{bgm_index}:a]" if bgm_index is not None else "[0:a]"
            filter_parts.append(f"{bgm_src}volume={bgm_volume}[abgm]")
            filter_parts.append(f"[{narration_index}:a]volume=1.0[anar]")
            filter_parts.append(f"[abgm][anar]amix=inputs=2:duration=first:dropout_transition=2[afinal]")
            audio_label = "[afinal]"
    else:
        if request.bgm_mode == BgmMode.demucs:
            audio_label = f"{bgm_index}:a" if bgm_index is not None else None
        elif request.bgm_mode != BgmMode.none:
            audio_label = f"{bgm_index}:a" if bgm_index is not None else "0:a?"

    if request.video_speed != 1.0:
        filter_parts.append(f"{video_label}setpts={1/request.video_speed:.4f}*PTS[vspeed]")
        video_label = "[vspeed]"
        if audio_label:
            # atempo filter requires exactly 1 input stream and produces 1 output stream.
            src_a = audio_label if audio_label.startswith("[") else f"[{audio_label.replace('?', '')}]"
            filter_parts.append(f"{src_a}atempo={request.video_speed:.4f}[aspeed]")
            audio_label = "[aspeed]"

    if filter_parts:
        command.extend(["-filter_complex", ";".join(filter_parts), "-map", video_label])
    else:
        command.extend(["-map", "0:v"])

    if audio_label:
        command.extend(["-map", audio_label])

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


def _can_stream_copy(request: DubbingRequest, subtitle_file: Path | None, narration_audio: Path | None, bgm_audio: Path | None) -> bool:
    return subtitle_file is None and narration_audio is None and bgm_audio is None and not (request.watermark_file_name and request.logo_enabled) and not request.blur_box_enabled and not request.cinematic_bars_enabled and request.output_resolution == "original" and request.bgm_mode == BgmMode.demucs and request.video_speed == 1.0


def _audio_output_args(request: DubbingRequest, has_narration: bool) -> list[str]:
    if request.bgm_mode == BgmMode.none and not has_narration:
        return ["-an"]
    if has_narration or request.video_speed != 1.0:
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


def _subtitle_filter_path(path: Path) -> str:
    value = path.resolve().as_posix()
    value = value.replace(":", r"\:")
    value = value.replace("'", r"\'")
    return value
