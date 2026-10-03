# Auto-Translate AI

Video translation and Vietnamese dubbing with a React interface and a FastAPI backend.

- Frontend: React, TypeScript, Vite
- Backend: Python, FastAPI, Pydantic
- Features: video import, translation, Vietnamese voiceover, subtitles, watermark, background processing and job history.
- The interface currently exposes translation only. Movie review is hidden and YouTube publishing has been removed.

## Run Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

API docs: http://127.0.0.1:8000/docs

## Run Frontend

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Frontend: http://127.0.0.1:5173

If your backend uses another URL, create `frontend/.env`:

```env
VITE_API_URL=http://127.0.0.1:8000
```

## Run With Docker

Install Docker Desktop first. Then run from the project root:

```powershell
.\build-docker.ps1
.\start-docker.ps1
```

Or:

```powershell
docker compose build
docker compose up -d
```

Frontend: http://127.0.0.1:5173  
API docs: http://127.0.0.1:8000/docs

The default Docker Compose file is configured for NVIDIA GPU acceleration. If Docker GPU support is not available yet, use the CPU fallback:

```powershell
docker compose -f docker-compose.yml -f docker-compose.cpu.yml build
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

## Notes

The processing pipeline downloads or imports a video, transcribes speech, translates dialogue, and renders voiceover and subtitles. Configure your local AI provider and media tools before processing:

1. Download or load source video.
2. Extract audio with FFmpeg.
3. Transcribe with Whisper or another ASR model.
4. Translate text.
5. Generate Vietnamese TTS.
6. Mix background audio and burn subtitles/watermark.
7. Export the completed video.


## Local configuration and private packages

Copy `backend/.env.example` to `backend/.env` and configure your own local
9router API key and FFmpeg path. The sample contains no account credentials.
Model names must be available in your own 9router installation.

To serve the built interface and API from a single port:

```powershell
cd frontend
npm.cmd run build
cd ../backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8100
```

Open http://127.0.0.1:8100. For this setup, omit `frontend/.env` or set
`VITE_API_URL=http://127.0.0.1:8100` before building.

`desktop_app.py`, `CHAY_APP.bat` and `build-portable.py` support personal Windows
packages. Packaging requires the local launcher executable, Python, Node.js,
FFmpeg and 9router runtimes described by the build script. These binaries are
not included in Git.

The packaging script copies the active 9router profile, including provider
sign-ins and API keys, into `release/`. Treat that folder and its ZIP as private.
Environment files, router profiles, databases, cookies, logs, generated media
and personal release packages are excluded from Git and Docker build contexts.
Credentials are configured separately on each fresh clone.
