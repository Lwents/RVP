import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import settings
from app.services.ai.asr import get_asr_engine
from app.services.media.ffmpeg import extract_audio, find_ffmpeg


def run(command: list[str], error_message: str, timeout: int | None = None) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{error_message}\n{detail}")


def preview_srt(path: Path, lines: int = 24) -> str:
    return "\n".join(path.read_text(encoding="utf-8").splitlines()[:lines])


async def main() -> None:
    parser = argparse.ArgumentParser(description="Download a YouTube video and test local ASR subtitle generation.")
    parser.add_argument("url")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--work-dir", default="storage/asr-youtube-test")
    args = parser.parse_args()

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("Cannot find FFmpeg.")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    source_file = work_dir / "source.%(ext)s"
    downloaded_file = work_dir / "source.mp4"
    audio_file = work_dir / "source_audio.wav"
    srt_file = work_dir / "subtitles.whisper.srt"

    download_command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--force-ipv4",
        "--socket-timeout",
        "25",
        "--retries",
        "2",
        "--ffmpeg-location",
        str(Path(ffmpeg).parent),
        "-f",
        settings.ytdlp_format,
        "--merge-output-format",
        "mp4",
        "-o",
        str(source_file),
        args.url,
    ]
    if settings.ytdlp_js_runtime:
        download_command.extend(["--js-runtimes", settings.ytdlp_js_runtime])

    print("Downloading source video...")
    run(download_command, "yt-dlp could not download the test video.", timeout=settings.ytdlp_download_timeout_seconds)

    matches = sorted(work_dir.glob("source.*"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not matches:
        raise RuntimeError("No downloaded source file found.")
    if matches[0] != downloaded_file:
        matches[0].replace(downloaded_file)

    print("Extracting 16 kHz mono WAV...")
    if args.seconds:
        run(
            [
                ffmpeg,
                "-y",
                "-t",
                str(args.seconds),
                "-i",
                str(downloaded_file),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(audio_file),
            ],
            "FFmpeg could not extract the requested audio sample.",
        )
    else:
        await extract_audio(ffmpeg, downloaded_file, audio_file)

    print(
        "Running ASR with "
        f"model={settings.whisper_model}, device={settings.whisper_device}, "
        f"compute_type={settings.whisper_compute_type}, language={args.language}..."
    )
    await get_asr_engine().transcribe_to_srt(audio_file, srt_file, args.language)

    print(f"Done: {srt_file.resolve()}")
    print()
    print(preview_srt(srt_file))


if __name__ == "__main__":
    asyncio.run(main())
