import asyncio
import wave
from pathlib import Path

import numpy as np
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model


class SeparationError(RuntimeError):
    pass


async def separate_background_with_demucs(audio_file: Path, output_dir: Path) -> Path:
    return await asyncio.to_thread(_separate_background_sync, audio_file, output_dir)


def _separate_background_sync(audio_file: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    no_vocals_file = output_dir / "no_vocals.wav"
    vocals_file = output_dir / "vocals.wav"

    sample_rate, audio = _read_pcm16_wav(audio_file)
    model = get_model(name="htdemucs")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if sample_rate != model.samplerate:
        raise SeparationError(f"Demucs cần audio {model.samplerate}Hz, file hiện tại là {sample_rate}Hz.")

    if audio.shape[0] == 1 and model.audio_channels == 2:
        audio = np.repeat(audio, 2, axis=0)
    if audio.shape[0] != model.audio_channels:
        raise SeparationError(f"Demucs cần {model.audio_channels} kênh audio, file hiện tại có {audio.shape[0]} kênh.")

    with torch.no_grad():
        wav = torch.from_numpy(audio).float()
        sources = apply_model(
            model,
            wav[None],
            device=device,
            split=True,
            overlap=0.25,
            shifts=0,
            progress=False,
        )[0]

    source_map = {name: sources[index].detach().cpu().numpy() for index, name in enumerate(model.sources)}
    if "vocals" not in source_map:
        raise SeparationError("Demucs không trả stem vocals.")

    no_vocals = sum(stem for name, stem in source_map.items() if name != "vocals")
    _write_pcm16_wav(no_vocals_file, model.samplerate, no_vocals)
    _write_pcm16_wav(vocals_file, model.samplerate, source_map["vocals"])

    if not no_vocals_file.exists() or no_vocals_file.stat().st_size <= 0:
        raise SeparationError("Demucs không tạo được no_vocals.wav.")
    return no_vocals_file


def _read_pcm16_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise SeparationError("Demucs helper chỉ hỗ trợ WAV PCM 16-bit.")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return sample_rate, audio.reshape(-1, channels).T


def _write_pcm16_wav(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio.T * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(pcm.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
