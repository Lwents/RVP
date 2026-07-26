import asyncio
import re
import subprocess
from pathlib import Path

from app.core import settings
from app.models.job import BgmMode, DubbingRequest
from app.services.media.ffmpeg import (
    copy_mp4,
    probe_has_audio_stream,
    probe_video_duration,
    probe_video_size,
    run_ffmpeg_with_progress,
)
from app.services.subtitles.ass import srt_to_positioned_ass, subtitle_font_dir

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
    demucs_failed: bool = False,
) -> None:
    if _can_stream_copy(request, subtitle_file, narration_audio, bgm_audio, demucs_failed):
        await copy_mp4(ffmpeg, source_video, output_file)
        on_progress(98)
        return

    width, height = await probe_video_size(ffmpeg, source_video)
    source_has_audio = await probe_has_audio_stream(ffmpeg, source_video)
    command = [ffmpeg, "-y", "-i", str(source_video)]
    next_input_index = 1
    video_label = "[vbase]"
    filter_parts: list[str] = ["[0:v]setpts=PTS-STARTPTS[vbase]"]

    if request.blur_box_enabled:
        # The subtitle band is intentionally static for the whole video. FFmpeg
        # applies one feathered Gaussian-blur mask in the final encode, avoiding
        # the expensive OpenCV OCR/inpaint pass and temporal mask flicker.
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
            "substatic",
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
            "customsoft",
        )



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

        subtitle_filters.append(
            f"subtitles='{_subtitle_filter_path(ass_file)}':"
            f"fontsdir='{_subtitle_filter_path(subtitle_font_dir())}':wrap_unicode=1"
        )
    
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
        elif bgm_index is None and not source_has_audio:
            # Video nguồn không có audio stream: map [0:a] sẽ làm hỏng cả lệnh
            # render, nên chỉ dùng giọng đọc.
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
            if bgm_index is not None:
                audio_label = f"{bgm_index}:a"
            else:
                audio_label = "0:a?" if source_has_audio else None

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

    command.extend(_video_output_args(await resolve_video_encoder(ffmpeg)))
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


def _can_stream_copy(
    request: DubbingRequest,
    subtitle_file: Path | None,
    narration_audio: Path | None,
    bgm_audio: Path | None,
    demucs_failed: bool = False,
) -> bool:
    if demucs_failed:
        # Người dùng yêu cầu bỏ giọng gốc. Nếu Demucs hỏng mà vẫn stream copy
        # thì file đầu ra chính là video gốc kèm nguyên giọng gốc.
        return False
    return subtitle_file is None and narration_audio is None and bgm_audio is None and not (request.watermark_file_name and request.logo_enabled) and not request.blur_box_enabled and not request.cinematic_bars_enabled and request.output_resolution == "original" and request.bgm_mode == BgmMode.demucs and request.video_speed == 1.0


def _gaussian_blur_sigma(width: int, height: int) -> int:
    """Scale a strong content blur from 18px up to 25px."""
    return 25


def _subtitle_blur_sigmas(width: int, height: int) -> tuple[int, int]:
    """Return a stronger horizontal blur that destroys hard-sub glyph edges."""
    scale = max(2 / 3, min(2.0, min(width / 1920, height / 1080)))
    return min(64, round(48 * scale)), min(40, round(28 * scale))


def _append_soft_box_blur(
    filter_parts: list[str],
    video_label: str,
    width: int,
    height: int,
    boxes: list[tuple[float, float, float, float, float | None, float | None]],
    prefix: str,
    pad_boxes: bool = True,
    strong_subtitle: bool = False,
) -> str:
    scale = min(width / 1920, height / 1080)
    padding = max(8, min(30, round(25 * max(scale, 0.35)))) if pad_boxes else 0
    feather = max(6, min(25, round(20 * max(scale, 0.35))))
    mask_sigma = max(2, round(feather / 3))
    expressions: list[str] = []
    for x_percent, y_percent, width_percent, height_percent, start, end in boxes:
        x0 = max(0, round(width * x_percent / 100) - padding)
        y0 = max(0, round(height * y_percent / 100) - padding)
        x1 = min(width - 1, round(width * (x_percent + width_percent) / 100) + padding)
        y1 = min(height - 1, round(height * (y_percent + height_percent) / 100) + padding)
        spatial = f"between(X\\,{x0}\\,{x1})*between(Y\\,{y0}\\,{y1})"
        if start is not None and end is not None and end > start:
            fade = max(0.04, min(0.12, (end - start) / 3))
            gate = (
                f"clip((T-{start:.3f})/{fade:.3f}\\,0\\,1)*"
                f"clip(({end:.3f}-T)/{fade:.3f}\\,0\\,1)"
            )
            expressions.append(f"({spatial})*({gate})")
        else:
            expressions.append(spatial)

    if not expressions:
        return video_label
    combined = expressions[0]
    for expression in expressions[1:]:
        combined = f"max({combined}\\,{expression})"
    filter_parts.append(
        f"{video_label}split=3[{prefix}_base][{prefix}_blur_source][{prefix}_mask_source]"
    )
    if strong_subtitle:
        sigma_x, sigma_y = _subtitle_blur_sigmas(width, height)
        filter_parts.append(
            f"[{prefix}_blur_source]gblur=sigma={sigma_x}:sigmaV={sigma_y}:steps=3"
            f"[{prefix}_blurred]"
        )
    else:
        blur_sigma = _gaussian_blur_sigma(width, height)
        filter_parts.append(
            f"[{prefix}_blur_source]gblur=sigma={blur_sigma}:steps=2[{prefix}_blurred]"
        )
    filter_parts.append(
        f"[{prefix}_mask_source]format=gray,"
        f"geq=lum='255*clip({combined}\\,0\\,1)',"
        f"gblur=sigma={mask_sigma}:steps=2[{prefix}_mask]"
    )
    if strong_subtitle:
        # alphamerge + overlay makes the centre of the mask fully opaque. A
        # limited-range grayscale mask fed to maskedmerge can leak a faint copy
        # of high-contrast hard subtitles even when its nominal value is 255.
        filter_parts.append(
            f"[{prefix}_blurred][{prefix}_mask]alphamerge[{prefix}_rgba]"
        )
        filter_parts.append(
            f"[{prefix}_base][{prefix}_rgba]overlay=0:0:format=auto[{prefix}_out]"
        )
    else:
        filter_parts.append(
            f"[{prefix}_base][{prefix}_blurred][{prefix}_mask]"
            f"maskedmerge=planes=15[{prefix}_out]"
        )
    return f"[{prefix}_out]"


def _render_blur_band(y_percent: int, height_percent: int) -> tuple[int, int]:
    height_percent = min(40, max(2, height_percent))
    y_percent = max(0, min(100 - height_percent, y_percent))
    return y_percent, height_percent


def _audio_output_args(request: DubbingRequest, has_narration: bool) -> list[str]:
    if request.bgm_mode == BgmMode.none and not has_narration:
        return ["-an"]
    if has_narration or request.video_speed != 1.0:
        return ["-c:a", "aac", "-b:a", "192k"]
    if request.bgm_mode == BgmMode.ducking:
        volume = round(10 ** (request.ducking_volume_db / 20), 4)
        return ["-af", f"volume={volume}", "-c:a", "aac", "-b:a", "128k"]
    return ["-c:a", "copy"]


SOFTWARE_ENCODER = "libx264"
# Thứ tự thử khi AUTO_TRANSLATE_VIDEO_ENCODER = "auto".
_HARDWARE_ENCODER_CANDIDATES = ["h264_nvenc", "h264_qsv", "h264_amf"]
_SOFTWARE_ENCODER_FALLBACKS = [SOFTWARE_ENCODER, "libopenh264"]
_SOFTWARE_ALIASES = {"", "auto_software", "libx264", "x264", "software", "cpu", "none"}
_RESOLVED_ENCODERS: dict[str, str] = {}


async def resolve_video_encoder(ffmpeg: str) -> str:
    """Return the encoder that this machine can really use.

    ``h264_nvenc`` chỉ có ý nghĩa khi máy có GPU NVIDIA và driver còn chạy được.
    Liệt kê ``-encoders`` là điều kiện cần chứ chưa đủ, nên phải thử encode thật
    một đoạn testsrc rồi mới tin. Kết quả được cache theo tiến trình.
    """

    configured = (settings.video_encoder or "auto").strip().lower()
    cache_key = f"{ffmpeg}|{configured}"
    cached = _RESOLVED_ENCODERS.get(cache_key)
    if cached:
        return cached

    resolved = await asyncio.to_thread(_probe_video_encoder, ffmpeg, configured)
    _RESOLVED_ENCODERS[cache_key] = resolved
    return resolved


def _probe_video_encoder(ffmpeg: str, configured: str) -> str:
    listed = _listed_encoder_names(ffmpeg)
    software = _resolve_software_encoder(listed)
    if configured in _SOFTWARE_ALIASES:
        return software

    candidates = _HARDWARE_ENCODER_CANDIDATES if configured == "auto" else [configured]
    for candidate in candidates:
        if candidate in _SOFTWARE_ENCODER_FALLBACKS:
            return software
        if listed and candidate not in listed:
            continue
        if _encoder_actually_works(ffmpeg, candidate):
            return candidate
        print(f"Encoder warning: {candidate} không khởi tạo được, chuyển sang {software}.")
    return software


def _resolve_software_encoder(listed: set[str]) -> str:
    """Pick a software encoder that this FFmpeg build actually ships.

    Một số bản dựng (ví dụ ffmpeg-free trên Fedora/RHEL) không có libx264, nên
    fallback phần mềm phải chấp nhận libopenh264 thay vì render thất bại.
    """

    if not listed or SOFTWARE_ENCODER in listed:
        return SOFTWARE_ENCODER
    for candidate in _SOFTWARE_ENCODER_FALLBACKS[1:]:
        if candidate in listed:
            print(f"Encoder warning: FFmpeg này không có {SOFTWARE_ENCODER}, dùng {candidate}.")
            return candidate
    return SOFTWARE_ENCODER


def _listed_encoder_names(ffmpeg: str) -> set[str]:
    try:
        completed = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    output = b"\n".join([completed.stdout or b"", completed.stderr or b""]).decode(errors="ignore")
    return {match.group(1) for match in re.finditer(r"^\s*[VASFXBD.]{6}\s+(\S+)", output, re.MULTILINE)}


def _encoder_actually_works(ffmpeg: str, encoder: str) -> bool:
    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=25:duration=0.2",
                "-frames:v",
                "3",
                "-c:v",
                encoder,
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _video_output_args(encoder: str | None = None) -> list[str]:
    encoder = (encoder or SOFTWARE_ENCODER).lower()
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
    if encoder == "libopenh264":
        # libopenh264 không có -crf/-preset nên quy đổi sang bitrate tương đương.
        return ["-c:v", "libopenh264", "-b:v", _openh264_bitrate(settings.video_crf), "-pix_fmt", "yuv420p"]
    # libx264 dùng -crf + tên preset x264, tương đương -cq/-b:v 0 của nvenc.
    return [
        "-c:v",
        SOFTWARE_ENCODER,
        "-preset",
        _x264_preset(settings.video_preset),
        "-crf",
        str(settings.video_crf),
        "-pix_fmt",
        "yuv420p",
    ]


def _openh264_bitrate(crf: int) -> str:
    if crf <= 20:
        return "8M"
    if crf <= 26:
        return "5M"
    return "3M"


def _x264_preset(preset: str) -> str:
    normalized = preset.lower().strip()
    known = {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
        "placebo",
    }
    if normalized in known:
        return normalized
    aliases = {
        "p1": "ultrafast",
        "p2": "veryfast",
        "p3": "fast",
        "p4": "medium",
        "p5": "slow",
        "p6": "slower",
        "p7": "veryslow",
    }
    return aliases.get(normalized, "medium")


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
