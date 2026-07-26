import socket
# Force IPv4 to bypass broken local IPv6 network configurations on Windows
orig_getaddrinfo = socket.getaddrinfo
def patched_getaddrinfo(*args, **kwargs):
    responses = orig_getaddrinfo(*args, **kwargs)
    return [r for r in responses if r[0] == socket.AF_INET]
socket.getaddrinfo = patched_getaddrinfo

import sys
if sys.platform.startswith("win"):
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core import settings

import asyncio
import time
from contextlib import asynccontextmanager

async def cleanup_task():
    while True:
        try:
            now = time.time()
            
            # Cleanup old video previews
            videos_dir = Path(settings.storage_dir) / "uploads" / "videos"
            if videos_dir.exists():
                for f in videos_dir.glob("*"):
                    if f.is_file() and now - f.stat().st_mtime > 86400:
                        f.unlink(missing_ok=True)
                        
            # Cleanup old watermarks
            watermarks_dir = Path(settings.storage_dir)
            if watermarks_dir.exists():
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
                    for f in watermarks_dir.glob(ext):
                        if f.is_file() and now - f.stat().st_mtime > 86400:
                            f.unlink(missing_ok=True)
                            
        except Exception:
            pass
        await asyncio.sleep(3600)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(cleanup_task())
    yield
    task.cancel()

app = FastAPI(
    title="Auto-Translate AI API",
    version="1.0.0",
    description="Backend for video dubbing, translation, subtitle, and publishing workflows.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=r"https://.*\.trycloudflare\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

storage_videos = Path(settings.storage_dir) / "uploads" / "videos"
storage_videos.mkdir(parents=True, exist_ok=True)
app.mount("/api/uploads/video", StaticFiles(directory=str(storage_videos)), name="videos")

# Watermarks live in the storage root next to youtube_credentials.json and the
# job index files, so they are served by an explicit allow-listed route in
# app.api.routes rather than a StaticFiles mount over the whole directory.


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
