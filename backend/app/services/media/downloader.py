import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable
from urllib.parse import quote, urljoin, urlparse

import requests
from app.core import settings
from app.services.media.ffmpeg import find_ffmpeg


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


class DownloadError(RuntimeError):
    pass


def _find_js_runtime() -> str | None:
    if settings.ytdlp_js_runtime:
        return settings.ytdlp_js_runtime

    bundled_node = Path(__file__).resolve().parents[4] / "tools" / "node" / "node.exe"
    if bundled_node.exists():
        return f"node:{bundled_node.as_posix()}"

    node = shutil.which("node")
    if node:
        return f"node:{Path(node).as_posix()}"

    deno = shutil.which("deno")
    if deno:
        return f"deno:{Path(deno).as_posix()}"

    return None


def _platform_name(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if "douyin.com" in host:
        return "Douyin"
    if "tiktok.com" in host:
        return "TikTok"
    if "youtube.com" in host or "youtu.be" in host:
        return "YouTube"
    return "nền tảng này"


def _format_download_error(detail: str, url: str) -> str:
    platform = _platform_name(url)
    message = f"yt-dlp không tải được video từ {platform}. {detail}"
    lowered = detail.lower()
    if "video unavailable" in lowered or "restricted" in lowered:
        message += (
            f"\nVideo này đang bị {platform}/tài khoản/mạng hạn chế. "
            "Nếu mở được video trong trình duyệt, hãy cấu hình cookie hợp lệ bằng "
            "AUTO_TRANSLATE_YTDLP_COOKIES_FILE hoặc AUTO_TRANSLATE_YTDLP_COOKIES_FROM_BROWSER."
        )
    if platform in {"TikTok", "Douyin"} and any(term in lowered for term in ["login", "captcha", "verify", "private", "cookie"]):
        message += (
            f"\n{platform} đôi khi chặn video riêng tư, video cần đăng nhập hoặc link bị kiểm tra bot. "
            "Backend đã tự thử cookie trình duyệt nếu có thể. Nếu vẫn lỗi, hãy đóng Chrome/Edge rồi chạy lại, "
            "xuất cookie Netscape vào AUTO_TRANSLATE_YTDLP_COOKIES_FILE, hoặc tải video về máy rồi chọn File nội bộ."
        )
    if "javascript runtime" in lowered or "js runtime" in lowered:
        message += "\nBackend đã hỗ trợ AUTO_TRANSLATE_YTDLP_JS_RUNTIME, ví dụ node:C:/Program Files/nodejs/node.exe."
    return message


def _without_browser_cookies(command: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for item in command:
        if skip_next:
            skip_next = False
            continue
        if item == "--cookies-from-browser":
            skip_next = True
            continue
        cleaned.append(item)
    return cleaned


def _detail(completed: subprocess.CompletedProcess[str]) -> str:
    return completed.stderr.strip() or completed.stdout.strip()


def _needs_browser_cookies(detail: str, url: str) -> bool:
    lowered = detail.lower()
    return _platform_name(url) in {"Douyin", "TikTok"} and any(
        term in lowered for term in ["fresh cookies", "login", "captcha", "verify", "cookie"]
    )


def _browser_cookie_sources() -> list[str]:
    if not settings.ytdlp_auto_browser_cookies:
        return []
    configured = settings.ytdlp_browser_cookie_sources or ""
    return [source.strip() for source in configured.split(",") if source.strip()]


def _douyin_tool_path() -> Path | None:
    if not settings.douyin_downloader_tool_enabled or not settings.douyin_downloader_tool_path:
        return None
    path = Path(settings.douyin_downloader_tool_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if (path / "run.py").exists():
        return path
    return None


def _external_services() -> list[str]:
    if not settings.douyin_external_downloader_enabled:
        return []
    return [service.strip().lower() for service in (settings.douyin_external_downloader_services or "").split(",") if service.strip()]


def _with_browser_cookies(command: list[str], browser: str) -> list[str]:
    base = _without_browser_cookies(command)
    return [*base[:-1], "--cookies-from-browser", browser, base[-1]]


def _run_ytdlp(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout)
    detail = _detail(completed)
    if completed.returncode != 0 and "--cookies-from-browser" in command and "could not copy chrome cookie database" in detail.lower():
        return subprocess.run(_without_browser_cookies(command), capture_output=True, text=True, check=False, timeout=timeout)
    return completed


def _write_douyin_tool_config(config_file: Path, output_dir: Path) -> None:
    config_file.write_text(
        "\n".join(
            [
                "link: []",
                f"path: {output_dir.as_posix()}",
                "music: false",
                "cover: false",
                "avatar: false",
                "json: true",
                "folderstyle: false",
                'filename_template: "{id}"',
                'folder_template: "{id}"',
                "author_dir: sec_uid",
                "download_pinned: false",
                "mode:",
                "  - post",
                "number:",
                "  post: 1",
                "  like: 0",
                "  allmix: 0",
                "  mix: 0",
                "  music: 0",
                "  collect: 0",
                "  collectmix: 0",
                "increase:",
                "  post: false",
                "  like: false",
                "  allmix: false",
                "  mix: false",
                "  music: false",
                "thread: 1",
                "retry_times: 2",
                'proxy: ""',
                "database: false",
                f"database_path: {(output_dir / 'douyin_downloader.db').as_posix()}",
                "progress:",
                "  quiet_logs: true",
                "transcript:",
                "  enabled: false",
                "browser_fallback:",
                "  enabled: true",
                "  headless: false",
                "  max_scrolls: 20",
                "  idle_rounds: 5",
                "  wait_timeout_seconds: 120",
                "comments:",
                "  enabled: false",
                "live:",
                "  max_duration_seconds: 0",
                "notifications:",
                "  enabled: false",
                "cookies: auto",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _download_with_douyin_tool(url: str, work_dir: Path, timeout: int) -> Path:
    tool_path = _douyin_tool_path()
    if not tool_path:
        raise DownloadError("Douyin Downloader tool chưa được cài trong tools/douyin-downloader.")

    output_dir = work_dir / "douyin_downloader"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_file = work_dir / "douyin_downloader.yml"
    _write_douyin_tool_config(config_file, output_dir)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["TERM"] = "dumb"

    completed = subprocess.run(
        [
            sys.executable,
            str(tool_path / "run.py"),
            "-c",
            str(config_file),
            "-u",
            url,
            "-p",
            str(output_dir),
            "--show-warnings",
        ],
        cwd=tool_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout,
    )
    matches = sorted(output_dir.rglob("*.mp4"), key=lambda item: item.stat().st_mtime, reverse=True)
    if matches:
        destination = work_dir / "source.douyin_tool.mp4"
        shutil.copy2(matches[0], destination)
        return destination

    detail = "\n".join(part for part in [(completed.stderr or "").strip(), (completed.stdout or "").strip()] if part)
    raise DownloadError(
        "Douyin Downloader GitHub fallback không tải được video. "
        "Douyin vẫn đang chặn API/cookie. Chi tiết:\n"
        f"{detail[-4000:]}"
    )


def _run_ytdlp_with_cookie_fallback(command: list[str], url: str, timeout: int) -> subprocess.CompletedProcess[str]:
    completed = _run_ytdlp(command, timeout)
    detail = _detail(completed)
    if completed.returncode == 0 or settings.ytdlp_cookies_file or settings.ytdlp_cookies_from_browser or not _needs_browser_cookies(detail, url):
        return completed

    attempts = [detail]
    for browser in _browser_cookie_sources():
        browser_completed = subprocess.run(
            _with_browser_cookies(command, browser),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        browser_detail = _detail(browser_completed)
        attempts.append(f"[cookies-from-browser {browser}] {browser_detail}")
        if browser_completed.returncode == 0:
            return browser_completed

    return subprocess.CompletedProcess(
        command,
        1,
        stdout="",
        stderr="\n".join(attempt for attempt in attempts if attempt),
    )


def _looks_like_video_response(response: requests.Response, url: str) -> bool:
    content_type = response.headers.get("content-type", "").lower()
    content_disposition = response.headers.get("content-disposition", "").lower()
    return (
        content_type.startswith("video/")
        or "application/octet-stream" in content_type
        or ".mp4" in urlparse(url).path.lower()
        or ".mp4" in content_disposition
    )


def _download_external_video(
    video_url: str,
    destination: Path,
    referer: str | None = None,
    session: requests.Session | None = None,
) -> Path:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    }
    if referer:
        headers["Referer"] = referer
    client = session or requests
    last_error: Exception | None = None
    for _attempt in range(3):
        destination.unlink(missing_ok=True)
        try:
            with client.get(video_url, headers=headers, stream=True, timeout=settings.douyin_external_downloader_timeout_seconds) as response:
                response.raise_for_status()
                if not _looks_like_video_response(response, str(response.url)):
                    raise DownloadError(f"Dịch vụ ngoài trả về nội dung không phải video: {response.headers.get('content-type', 'unknown')}")
                with destination.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
            if destination.stat().st_size <= 0:
                destination.unlink(missing_ok=True)
                raise DownloadError("Dịch vụ ngoài trả về file video rỗng.")
            return destination
        except Exception as exc:
            last_error = exc
            destination.unlink(missing_ok=True)
    raise DownloadError(f"Dịch vụ ngoài tải video bị gián đoạn: {last_error}")


def _download_with_unduhtiktok(url: str, work_dir: Path) -> Path:
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://unduhtiktok.com/vi/douyin/",
        "Origin": "https://unduhtiktok.com",
    }
    timeout = settings.douyin_external_downloader_timeout_seconds

    token_response = session.get(
        "https://unduhtiktok.com/wp-content/plugins/snaptik-app/api/check.php",
        headers=headers,
        timeout=timeout,
    )
    token_response.raise_for_status()

    response = session.post(
        "https://unduhtiktok.com/wp-content/plugins/snaptik-app/api/douyin.php",
        headers=headers,
        json={"url": url},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise DownloadError(f"unduhtiktok trả lỗi: {payload.get('error')}")

    destination = work_dir / "source.external.unduhtiktok.mp4"
    candidates = []
    video_url = payload.get("video")
    if isinstance(video_url, str):
        candidates.append(video_url)
    candidates.extend(_candidate_urls_from_payload(payload))

    seen: set[str] = set()
    candidate_errors: list[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            absolute = urljoin("https://unduhtiktok.com/vi/douyin/", candidate)
            return _download_external_video(
                absolute,
                destination,
                referer="https://unduhtiktok.com/vi/douyin/",
                session=session,
            )
        except Exception as exc:
            candidate_errors.append(str(exc))

    raise DownloadError(
        "unduhtiktok không trả URL video hợp lệ. "
        + "; ".join(candidate_errors[-3:])
        + f" Payload: {str(payload)[:500]}"
    )


def _candidate_urls_from_payload(payload: object) -> list[str]:
    candidates: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("//"):
                text = f"https:{text}"
            if text.startswith("http") and (
                ".mp4" in text.lower()
                or "video" in text.lower()
                or "play" in text.lower()
                or "tos-" in text.lower()
                or "byte" in text.lower()
            ):
                candidates.append(text)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            preferred_keys = [
                "nwm_video_url_HQ",
                "nwm_video_url",
                "hdplay",
                "play",
                "wmplay",
                "url",
                "download",
                "download_url",
                "video_url",
            ]
            for key in preferred_keys:
                if key in value:
                    walk(value[key])
            for item in value.values():
                walk(item)

    walk(payload)
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _download_with_douyin_external_services(url: str, work_dir: Path) -> Path:
    errors: list[str] = []
    encoded = quote(url, safe="")
    custom_urls = [
        item.strip().replace("{url}", encoded).replace("{raw_url}", url)
        for item in (settings.douyin_external_downloader_custom_urls or "").split(";")
        if item.strip()
    ]
    service_urls = {
        "douyinwtf": [
            f"https://api.douyin.wtf/api/download?url={encoded}&prefix=true&with_watermark=false",
            f"https://api.douyin.wtf/api/hybrid/video_data?url={encoded}&minimal=false",
        ],
        "tikwm": [
            f"https://www.tikwm.com/api/?url={encoded}",
            f"https://tikwm.com/api/?url={encoded}",
        ],
        "custom": custom_urls,
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
    }
    for service in _external_services():
        if service == "unduhtiktok":
            try:
                return _download_with_unduhtiktok(url, work_dir)
            except Exception as exc:
                errors.append(f"unduhtiktok failed: {exc}")
                continue
        for api_url in service_urls.get(service, []):
            try:
                response = requests.get(
                    api_url,
                    headers=headers,
                    timeout=settings.douyin_external_downloader_timeout_seconds,
                    allow_redirects=True,
                )
                response.raise_for_status()
                destination = work_dir / f"source.external.{service}.mp4"
                if _looks_like_video_response(response, str(response.url)):
                    destination.write_bytes(response.content)
                    if destination.stat().st_size > 0:
                        return destination
                try:
                    payload = response.json()
                except json.JSONDecodeError:
                    payload = {"raw": response.text}
                for candidate in _candidate_urls_from_payload(payload):
                    try:
                        absolute = urljoin(api_url, candidate)
                        return _download_external_video(absolute, destination, referer=api_url)
                    except Exception as exc:
                        errors.append(f"{service} candidate failed: {exc}")
                errors.append(f"{service} không trả URL video hợp lệ: {str(payload)[:500]}")
            except Exception as exc:
                errors.append(f"{service} failed: {exc}")
    raise DownloadError("Các web/API tải Douyin bên ngoài đều chưa tải được video.\n" + "\n".join(errors[-8:]))


async def prepare_source_video(
    source_url: str | None,
    local_file_path: str | None,
    work_dir: Path,
    progress: Callable[[str, int], None],
) -> Path:
    if local_file_path:
        source = Path(local_file_path)
        if not source.exists() or not source.is_file():
            raise DownloadError(f"Không tìm thấy file video nội bộ: {source}")
        if source.suffix.lower() not in VIDEO_SUFFIXES:
            raise DownloadError("File nội bộ không đúng định dạng video hỗ trợ.")

        destination = work_dir / f"source{source.suffix.lower()}"
        progress("Copy video nguồn vào workspace", 15)
        await asyncio.to_thread(shutil.copy2, source, destination)
        return destination

    if source_url:
        return await download_video(source_url, work_dir, progress)

    raise DownloadError("Cần URL video hoặc đường dẫn file nội bộ.")


async def download_video(url: str, work_dir: Path, progress: Callable[[str, int], None]) -> Path:
    progress(f"Tải video nguồn từ {_platform_name(url)} bằng yt-dlp", 15)
    output_template = str(work_dir / "source.%(ext)s")
    ffmpeg = find_ffmpeg()
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--force-ipv4",
        "--socket-timeout",
        "25",
        "--retries",
        "5",
        "--fragment-retries",
        "5",
        "--no-playlist",
        "--remote-components",
        "ejs:github",
        "--format",
        settings.ytdlp_format,
        "--merge-output-format",
        "mp4",
        "--output",
        output_template,
    ]
    if ffmpeg:
        command.extend(["--ffmpeg-location", str(Path(ffmpeg).parent)])
    js_runtime = _find_js_runtime()
    if js_runtime:
        command.extend(["--js-runtimes", js_runtime])
    if settings.ytdlp_cookies_file:
        command.extend(["--cookies", settings.ytdlp_cookies_file])
    if settings.ytdlp_cookies_from_browser:
        command.extend(["--cookies-from-browser", settings.ytdlp_cookies_from_browser])
    command.append(url)

    def download() -> Path:
        completed = _run_ytdlp_with_cookie_fallback(command, url, timeout=settings.ytdlp_download_timeout_seconds)
        if completed.returncode != 0:
            detail = _detail(completed)
            if _platform_name(url) == "Douyin":
                progress("yt-dlp bị Douyin chặn, thử Douyin Downloader GitHub", 16)
                try:
                    return _download_with_douyin_tool(url, work_dir, settings.ytdlp_download_timeout_seconds)
                except DownloadError as tool_error:
                    detail = f"{detail}\n\n{tool_error}"
                progress("Douyin Downloader GitHub chưa tải được, thử web/API tải ngoài", 17)
                try:
                    return _download_with_douyin_external_services(url, work_dir)
                except DownloadError as external_error:
                    detail = f"{detail}\n\n{external_error}"
            raise DownloadError(_format_download_error(detail, url))
        matches = sorted(work_dir.glob("source.*"), key=lambda item: item.stat().st_mtime, reverse=True)
        for match in matches:
            if match.suffix.lower() in VIDEO_SUFFIXES:
                return match
        raise DownloadError("yt-dlp chạy xong nhưng không tìm thấy file video đầu ra.")

    try:
        return await asyncio.to_thread(download)
    except subprocess.TimeoutExpired as exc:
        raise DownloadError("yt-dlp quá thời gian chờ 180 giây khi tải video.") from exc


async def download_preview_video(url: str, work_dir: Path) -> Path:
    output_template = str(work_dir / "preview.%(ext)s")
    ffmpeg = find_ffmpeg()
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--force-ipv4",
        "--socket-timeout",
        "15",
        "--retries",
        "5",
        "--fragment-retries",
        "5",
        "--no-playlist",
        "--format",
        "bestvideo[height<=480][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=480]+bestaudio/best[height<=480]/best",
        "--merge-output-format",
        "mp4",
        "--output",
        output_template,
    ]
    if ffmpeg:
        command.extend(["--ffmpeg-location", str(Path(ffmpeg).parent)])
    js_runtime = _find_js_runtime()
    if js_runtime:
        command.extend(["--js-runtimes", js_runtime])
    if settings.ytdlp_cookies_file:
        command.extend(["--cookies", settings.ytdlp_cookies_file])
    if settings.ytdlp_cookies_from_browser:
        command.extend(["--cookies-from-browser", settings.ytdlp_cookies_from_browser])
    command.append(url)

    def download() -> Path:
        completed = _run_ytdlp_with_cookie_fallback(command, url, timeout=120)
        if completed.returncode != 0:
            detail = _detail(completed)
            raise DownloadError(f"Không thể tải bản xem trước. yt-dlp báo lỗi: {detail}")
        matches = sorted(work_dir.glob("preview.*"), key=lambda item: item.stat().st_mtime, reverse=True)
        for match in matches:
            if match.suffix.lower() in VIDEO_SUFFIXES:
                return match
        raise DownloadError("Không tìm thấy file preview sau khi tải.")

    try:
        return await asyncio.to_thread(download)
    except subprocess.TimeoutExpired as exc:
        raise DownloadError("Quá thời gian chờ (120s) khi tải bản xem trước.") from exc
