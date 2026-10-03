"""Build a relocatable Windows x64 folder and ZIP from the local project."""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import sqlite3
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RELEASE = ROOT / "release"
APP = RELEASE / "AutoTranslateAI-Windows-x64"
ZIP = RELEASE / "AutoTranslateAI-Windows-x64.zip"
NINEROUTER_PROFILE = Path(os.environ.get("NINEROUTER_PROFILE", str(Path.home() / ".9router")))
NINEROUTER_SOURCE = ROOT / "tools" / "9router-runtime"


def ignore_caches(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name == "__pycache__" or name.endswith(".pyc")}


def link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def remove_readonly(function, path: str, _error) -> None:
    os.chmod(path, stat.S_IWRITE)
    function(path)


def snapshot_ninerouter(destination: Path) -> str:
    """Copy the active 9router profile, retaining settings and credentials."""
    source_db = NINEROUTER_PROFILE / "db" / "data.sqlite"
    if not source_db.is_file():
        raise SystemExit(f"9router database was not found: {source_db}")

    destination.mkdir(parents=True, exist_ok=True)
    # Copy configuration and identity files, excluding platform-specific runtime
    # artifacts and SQLite's live WAL files. The database is snapshotted below.
    for source in NINEROUTER_PROFILE.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(NINEROUTER_PROFILE)
        if relative.parts[0] in {"db", "runtime", "bin"}:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    db_target = destination / "db" / "data.sqlite"
    db_target.parent.mkdir(parents=True, exist_ok=True)
    source_connection = sqlite3.connect(f"file:{source_db.as_posix()}?mode=ro", uri=True, timeout=30)
    target_connection = sqlite3.connect(db_target)
    try:
        source_connection.backup(target_connection)
    finally:
        target_connection.close()
        source_connection.close()

    # Request bodies and usage history are not needed to run the copied account.
    # Remove them from the personal bundle while retaining all router settings,
    # provider connections, API keys, model choices, and runtime state.
    target_connection = sqlite3.connect(db_target)
    try:
        target_connection.execute("PRAGMA secure_delete=ON")
        tables = {row[0] for row in target_connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in ("requestDetails", "usageHistory", "usageDaily"):
            if table in tables:
                target_connection.execute(f'DELETE FROM "{table}"')
        target_connection.commit()
        target_connection.execute("VACUUM")
        api_key = target_connection.execute(
            "SELECT key FROM apiKeys WHERE isActive=1 ORDER BY createdAt DESC LIMIT 1"
        ).fetchone()
        if not api_key or not api_key[0]:
            raise SystemExit("The 9router profile has no active API key to configure the app.")
        if "\n" in api_key[0] or "\r" in api_key[0]:
            raise SystemExit("The active 9router API key has an unexpected line break.")
        return api_key[0]
    finally:
        target_connection.close()


def build() -> None:
    required = [
        ROOT / "AutoTranslateAI.exe",
        ROOT / "desktop_app.py",
        ROOT / "frontend/dist/index.html",
        ROOT / "backend/.venv/Lib/site-packages/uvicorn",
        ROOT / "backend/.venv/Lib/site-packages/webview",
        ROOT / "tools/python/python.exe",
        ROOT / "tools/ffmpeg/bin/ffmpeg.exe",
        ROOT / "tools/node/node.exe",
        NINEROUTER_SOURCE / "node_modules/9router/app/custom-server.js",
        NINEROUTER_SOURCE / "sqlite-runtime/runtime/node_modules/better-sqlite3/package.json",
        NINEROUTER_SOURCE / "cloudflared-windows-amd64.exe",
        NINEROUTER_SOURCE / "CLOUDFLARED-LICENSE.txt",
        NINEROUTER_PROFILE / "db/data.sqlite",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing build inputs:\n" + "\n".join(missing))

    RELEASE.mkdir(exist_ok=True)
    if APP.exists():
        shutil.rmtree(APP, onexc=remove_readonly)
    APP.mkdir()

    for name in ("AutoTranslateAI.exe", "desktop_app.py", "app_icon.ico", "CHAY_APP.bat"):
        shutil.copy2(ROOT / name, APP / name)

    shutil.copytree(ROOT / "backend/app", APP / "backend/app", ignore=ignore_caches)
    shutil.copytree(ROOT / "frontend/dist", APP / "frontend/dist")
    shutil.copytree(ROOT / "tools/python", APP / "tools/python", ignore=ignore_caches)
    (APP / "tools/python/rvp_packages.pth").unlink(missing_ok=True)
    shutil.copytree(ROOT / "backend/.venv/Lib/site-packages", APP / "tools/python/Lib/site-packages", copy_function=link_or_copy, ignore=ignore_caches)
    shutil.copytree(ROOT / "tools/ffmpeg/bin", APP / "tools/ffmpeg/bin", copy_function=link_or_copy)
    shutil.copytree(ROOT / "tools/node", APP / "tools/node", copy_function=link_or_copy)
    shutil.copytree(NINEROUTER_SOURCE / "node_modules", APP / "9router-runtime/node_modules", copy_function=link_or_copy, ignore=ignore_caches)
    (APP / "9router-runtime/licenses").mkdir(parents=True, exist_ok=True)
    shutil.copy2(NINEROUTER_SOURCE / "CLOUDFLARED-LICENSE.txt", APP / "9router-runtime/licenses/CLOUDFLARED-LICENSE.txt")

    router_profile = APP / "router-data/9router"
    api_key = snapshot_ninerouter(router_profile)
    shutil.copytree(NINEROUTER_SOURCE / "sqlite-runtime/runtime", router_profile / "runtime")
    cloudflared = NINEROUTER_SOURCE / "cloudflared-windows-amd64.exe"
    if cloudflared.read_bytes()[:2] != b"MZ" or cloudflared.stat().st_size < 1024 * 1024:
        raise SystemExit("Bundled Cloudflare Tunnel binary is not a valid Windows executable.")
    (router_profile / "bin").mkdir(parents=True, exist_ok=True)
    shutil.copy2(cloudflared, router_profile / "bin/cloudflared.exe")
    (APP / "backend/storage").mkdir(parents=True)
    (APP / "storage").mkdir()

    (APP / "backend/.env").write_text(
        "AUTO_TRANSLATE_TRANSLATION_ENGINE=gemini\n"
        "AUTO_TRANSLATE_VOICE_ENGINE=edge\n"
        "AUTO_TRANSLATE_WHISPER_MODEL=base\n"
        "AUTO_TRANSLATE_WHISPER_DEVICE=cpu\n"
        "AUTO_TRANSLATE_WHISPER_COMPUTE_TYPE=int8\n"
        "AUTO_TRANSLATE_VIDEO_ENCODER=libx264\n"
        "AUTO_TRANSLATE_VIDEO_PRESET=veryfast\n"
        "AUTO_TRANSLATE_AI_MODEL=ag/gemini-3.8-flash-medium\n"
        "AUTO_TRANSLATE_REVIEW_AI_MODEL=ag/gemini-pro-agent\n"
        "AUTO_TRANSLATE_NINEROUTER_API_URL=http://127.0.0.1:20129/v1\n"
        f"AUTO_TRANSLATE_NINEROUTER_API_KEY={api_key}\n",
        encoding="utf-8",
    )
    (APP / "README.txt").write_text(
        "AutoTranslateAI for Windows x64\n\n"
        "Extract the whole ZIP, then double-click AutoTranslateAI.exe.\n"
        "Keep all folders together. Startup errors are saved to storage\\launcher.log.\n"
        "Windows needs Microsoft Edge WebView2 Runtime to display the app.\n"
        "Your current 9router profile is bundled and starts with the app.\n"
        "It includes provider sign-ins, router settings, API key, model catalog,\n"
        "and Windows runtime files. Your request and usage history is not included.\n"
        "Translation model: ag/gemini-3.8-flash-medium.\n"
        "Review model: ag/gemini-pro-agent.\n"
        "The active 9router API key is configured in backend\\.env.\n"
        "The app connects to its bundled local router; no other PC is required.\n"
        "The profile contains private provider credentials. Do not share this ZIP.\n"
        "A provider may ask you to sign in again after moving to a new computer.\n"
        "Node.js and FFmpeg are included for video downloading and rendering.\n"
        "The Cloudflare Tunnel setting and its Windows runtime are included.\n"
        "The Cloudflare Tunnel license is in 9router-runtime\\licenses.\n"
        "The package uses CPU defaults; GPU options can be set in backend\\.env.\n"
        "Your AutoTranslateAI projects, media and storage data are not included.\n",
        encoding="utf-8",
    )

    temporary_zip = ZIP.with_suffix(".tmp.zip")
    temporary_zip.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for path in sorted(APP.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(RELEASE))
    temporary_zip.replace(ZIP)
    print(f"Folder: {APP}")
    print(f"ZIP: {ZIP} ({ZIP.stat().st_size / (1024 ** 3):.2f} GiB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build()
