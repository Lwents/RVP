import os
import sys
import threading
import time
import socket
import subprocess
import urllib.request
from pathlib import Path

# Add backend directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(str(BACKEND_DIR))

import uvicorn
import webview


def is_port_available(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            return s.connect_ex(("127.0.0.1", port)) != 0
    except Exception:
        return True


def find_available_port(ports=(8000, 8100, 8088, 8888, 9000)) -> int:
    for p in ports:
        if is_port_available(p):
            return p
    return 8100


server_instance = None
router_process = None
router_log = None


def start_bundled_ninerouter() -> int:
    """Start the packaged 9router with its private, bundled user profile."""
    global router_process, router_log

    runtime_dir = BASE_DIR / "9router-runtime" / "node_modules" / "9router"
    server_path = runtime_dir / "app" / "custom-server.js"
    node_path = BASE_DIR / "tools" / "node" / "node.exe"
    data_root = BASE_DIR / "router-data"
    profile_dir = data_root / "9router"
    runtime_modules = profile_dir / "runtime" / "node_modules"

    if not node_path.exists() or not server_path.exists():
        raise RuntimeError("Bundled 9router runtime is missing. Re-extract the complete app folder.")

    profile_dir.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "storage").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "APPDATA": str(data_root),
        "DATA_DIR": str(profile_dir),
        "HOSTNAME": "127.0.0.1",
        "NODE_PATH": os.pathsep.join((str(runtime_modules), str(runtime_dir / "app" / "node_modules"))),
        "NEXT_TELEMETRY_DISABLED": "1",
    })

    port = next((candidate for candidate in range(20129, 20139) if is_port_available(candidate)), None)
    if port is None:
        raise RuntimeError("No free local port is available for the bundled 9router (20129–20138).")
    env["PORT"] = str(port)
    env["AUTO_TRANSLATE_NINEROUTER_API_URL"] = f"http://127.0.0.1:{port}/v1"

    router_log = (BASE_DIR / "storage" / "9router.log").open("ab")
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    router_process = subprocess.Popen(
        [str(node_path), str(server_path)],
        cwd=str(runtime_dir / "app"),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=router_log,
        stderr=subprocess.STDOUT,
        creationflags=creation_flags,
    )

    models_url = f"http://127.0.0.1:{port}/v1/models"
    try:
        for _ in range(100):
            if router_process.poll() is not None:
                raise RuntimeError("Bundled 9router stopped during startup. See storage\\9router.log.")
            try:
                with urllib.request.urlopen(models_url, timeout=1.0) as response:
                    if response.status == 200:
                        os.environ["AUTO_TRANSLATE_NINEROUTER_API_URL"] = env["AUTO_TRANSLATE_NINEROUTER_API_URL"]
                        return port
            except Exception:
                time.sleep(0.25)
        raise RuntimeError("Bundled 9router did not become ready. See storage\\9router.log.")
    except Exception:
        stop_bundled_ninerouter()
        raise


def stop_bundled_ninerouter() -> None:
    global router_process, router_log
    if router_process and router_process.poll() is None:
        try:
            router_process.terminate()
            router_process.wait(timeout=5)
        except Exception:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(router_process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
    router_process = None
    if router_log:
        router_log.close()
        router_log = None


def run_server(port: int):
    global server_instance
    from app.main import app

    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server_instance = uvicorn.Server(config)
    server_instance.run()


def main():
    router_port = start_bundled_ninerouter()
    print(f"[*] 9router da san sang tai cong {router_port}.", flush=True)
    port = find_available_port()
    print(f"[*] Dang khoi dong Backend tai cong {port}...", flush=True)

    # Start FastAPI backend in a background daemon thread
    server_thread = threading.Thread(target=run_server, args=(port,), daemon=True)
    server_thread.start()

    # Wait for the backend to start listening
    url = f"http://127.0.0.1:{port}"
    for _ in range(40):
        time.sleep(0.15)
        if not is_port_available(port):
            break

    print("[*] Backend da san sang.", flush=True)
    print("[*] Dang khoi tao va mo cua so ung dung Windows (Edge WebView2)...", flush=True)

    webview_storage = BASE_DIR / "storage" / "webview_cache"
    webview_storage.mkdir(parents=True, exist_ok=True)

    app_icon_path = BASE_DIR / "app_icon.ico"
    # Launch native Windows desktop window using Edge WebView2
    try:
        webview.create_window(
            title="Auto-Translate AI - Video Dubbing & Review",
            url=url,
            width=1400,
            height=900,
            min_size=(1050, 720),
            text_select=True,
        )
        webview.start(
            gui="edgechromium",
            storage_path=str(webview_storage),
            icon=str(app_icon_path) if app_icon_path.exists() else None,
            private_mode=False,
        )
    finally:
        if server_instance:
            server_instance.should_exit = True
        stop_bundled_ninerouter()
    print("[*] Da dong ung dung.", flush=True)


if __name__ == "__main__":
    main()
