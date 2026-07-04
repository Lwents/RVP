# Auto-Translate AI

Full-stack demo for an AI video dubbing workflow.

- Frontend: React, TypeScript, Vite
- Backend: Python, FastAPI, Pydantic
- Features: dubbing configuration form, watermark upload, background job simulation, progress polling, job history

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

The current backend includes a clean mock processing pipeline so the whole app runs immediately. Replace `backend/app/services/processor.py` with real video steps when you are ready:

1. Download or load source video.
2. Extract audio with FFmpeg.
3. Transcribe with Whisper or another ASR model.
4. Translate text.
5. Generate Vietnamese TTS.
6. Mix background audio and burn subtitles/watermark.
7. Publish to configured channels.
cd C:\Users\kirit\Documents\AI_Video\backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
