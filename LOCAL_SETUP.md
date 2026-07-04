# Huong dan cai thu vien va chay local

Huong dan nay dung khi chay truc tiep tren Windows, khong dung Docker.

## 1. Cai cong cu can co

### Python

Nen dung Python 3.12.

Tai tai:

```text
https://www.python.org/downloads/
```

Khi cai, tick:

```text
Add python.exe to PATH
```

Kiem tra:

```powershell
python --version
```

### Node.js

Nen dung Node.js LTS.

Tai tai:

```text
https://nodejs.org/
```

Kiem tra:

```powershell
node --version
npm --version
```

### FFmpeg

Du an da co san FFmpeg tai:

```text
tools/ffmpeg-8.1.2-full_build/bin/ffmpeg.exe
```

Khong can cai FFmpeg rieng neu file nay con ton tai.

## 2. Cai thu vien backend

Mo PowerShell trong thu muc du an:

```powershell
cd C:\Users\kirit\Documents\AI_Video
cd backend
```

Tao virtualenv:

```powershell
python -m venv .venv
```

Kich hoat virtualenv:

```powershell
.\.venv\Scripts\Activate.ps1
```

Neu PowerShell chan script, chay:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Sau do kich hoat lai:

```powershell
.\.venv\Scripts\Activate.ps1
```

Cai thu vien Python:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Kiem tra thu vien:

```powershell
pip check
```

## 3. Cai thu vien frontend

Mo PowerShell moi hoac quay ve root:

```powershell
cd C:\Users\kirit\Documents\AI_Video\frontend
npm install
```

Kiem tra build:

```powershell
npm run build
```

## 4. Cau hinh backend

File cau hinh nam tai:

```text
backend/.env
```

Cau hinh quan trong hien tai:

```env
AUTO_TRANSLATE_FFMPEG_PATH=C:\Users\kirit\Documents\AI_Video\tools\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe
AUTO_TRANSLATE_TRANSLATION_ENGINE=google
AUTO_TRANSLATE_VOICE_ENGINE=edge
AUTO_TRANSLATE_WHISPER_MODEL=medium
AUTO_TRANSLATE_WHISPER_DEVICE=cuda
AUTO_TRANSLATE_WHISPER_COMPUTE_TYPE=int8_float16
AUTO_TRANSLATE_VIDEO_ENCODER=h264_nvenc
```

Neu may khong co GPU NVIDIA hoac CUDA loi, doi thanh:

```env
AUTO_TRANSLATE_WHISPER_DEVICE=cpu
AUTO_TRANSLATE_WHISPER_COMPUTE_TYPE=int8
AUTO_TRANSLATE_VIDEO_ENCODER=libx264
AUTO_TRANSLATE_VIDEO_PRESET=veryfast
```

## 5. Cai ho tro GPU NVIDIA tuy chon

Neu muon ASR/render dung GPU:

1. Cai NVIDIA Driver moi nhat.
2. Kiem tra:

```powershell
nvidia-smi
```

3. Backend da cai runtime CUDA qua `requirements.txt`:

```text
nvidia-cublas-cu12
nvidia-cuda-nvrtc-cu12
nvidia-cudnn-cu12
```

Khong can cai CUDA Toolkit rieng neu cac package tren hoat dong binh thuong.

## 6. Chay backend

```powershell
cd C:\Users\kirit\Documents\AI_Video\backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Mo API docs:

```text
http://127.0.0.1:8000/docs
```

## 7. Chay frontend

Mo PowerShell khac:

```powershell
cd C:\Users\kirit\Documents\AI_Video\frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

Mo app:

```text
http://127.0.0.1:5173
```

## 8. Lenh chay nhanh

Backend:

```powershell
cd C:\Users\kirit\Documents\AI_Video\backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
cd C:\Users\kirit\Documents\AI_Video\frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

## 9. Loi thuong gap

### Failed to fetch

Backend chua chay hoac port 8000 bi tat.

Kiem tra:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/docs
```

### Edge TTS loi 500

Day la loi tam thoi tu dich vu Microsoft Edge TTS. Backend da co retry va fallback render video voi audio goc.

### CUDA / GPU loi

Neu loi CUDA, doi `.env` sang CPU:

```env
AUTO_TRANSLATE_WHISPER_DEVICE=cpu
AUTO_TRANSLATE_WHISPER_COMPUTE_TYPE=int8
AUTO_TRANSLATE_VIDEO_ENCODER=libx264
```

### YouTube khong tai duoc video

Thu link khac hoac cau hinh cookie YouTube:

```env
AUTO_TRANSLATE_YTDLP_COOKIES_FILE=C:\path\to\cookies.txt
```

