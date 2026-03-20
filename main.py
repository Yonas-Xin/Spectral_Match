from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start backend and frontend dev servers.")
    parser.add_argument("--backend-python", default="", help="Explicit backend Python executable path.")
    parser.add_argument("--backend-host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--frontend-host", default="127.0.0.1")
    parser.add_argument("--frontend-port", type=int, default=5173)
    parser.add_argument("--no-reload", action="store_true", help="Disable backend auto reload.")
    parser.add_argument("--backend-only", action="store_true", help="Start backend only.")
    parser.add_argument(
        "--auto-install-backend",
        action="store_true",
        help="Auto install backend dependencies when uvicorn/fastapi are missing.",
    )
    return parser.parse_args()


def find_backend_python(backend_dir: Path) -> str:
    win_venv = backend_dir / ".venv" / "Scripts" / "python.exe"
    unix_venv = backend_dir / ".venv" / "bin" / "python"
    if win_venv.exists():
        return str(win_venv)
    if unix_venv.exists():
        return str(unix_venv)
    return sys.executable


def find_node() -> Optional[str]:
    candidates = [shutil.which("node.exe"), shutil.which("node")]
    if os.name == "nt":
        for base in (
            os.getenv("ProgramFiles"),
            os.getenv("ProgramFiles(x86)"),
            os.path.join(os.getenv("LocalAppData", ""), "Programs"),
        ):
            if not base:
                continue
            candidates.append(str(Path(base) / "nodejs" / "node.exe"))
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


def find_npm() -> Optional[str]:
    candidates = [shutil.which("npm.cmd"), shutil.which("npm")]
    if os.name == "nt":
        for base in (os.getenv("ProgramFiles"), os.getenv("ProgramFiles(x86)")):
            if not base:
                continue
            candidates.append(str(Path(base) / "nodejs" / "npm.cmd"))
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


def resolve_frontend_cmd(frontend_dir: Path, host: str, port: int) -> list[str]:
    npm_cmd = find_npm()
    if npm_cmd:
        return [npm_cmd, "run", "dev", "--", "--host", host, "--port", str(port)]

    node_cmd = find_node()
    vite_js = frontend_dir / "node_modules" / "vite" / "bin" / "vite.js"
    if node_cmd and vite_js.exists():
        return [node_cmd, str(vite_js), "--host", host, "--port", str(port)]

    raise RuntimeError(
        "Node.js runtime not found. Please install Node.js LTS and reopen terminal.\n"
        "Windows quick install: winget install OpenJS.NodeJS.LTS"
    )


def stream_logs(prefix: str, proc: subprocess.Popen[str]) -> None:
    if proc.stdout is None:
        return
    for line in proc.stdout:
        print(f"[{prefix}] {line.rstrip()}")


def terminate_process(proc: subprocess.Popen[str], name: str) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        print(f"[main] {name} terminate timeout, killing...")
        proc.kill()
        proc.wait(timeout=3)


def backend_deps_ready(backend_python: str) -> bool:
    cmd = [backend_python, "-c", "import uvicorn, fastapi"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0


def install_backend_deps(backend_python: str, backend_dir: Path) -> int:
    req = backend_dir / "requirements.txt"
    if not req.exists():
        print(f"[main] requirements not found: {req}")
        return 1
    cmd = [backend_python, "-m", "pip", "install", "-r", str(req)]
    print(f"[main] installing backend dependencies: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=backend_dir).returncode


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent
    backend_dir = root_dir / "backend"
    frontend_dir = root_dir / "frontend"

    if not backend_dir.exists():
        print("[main] backend directory not found.")
        return 1
    if not frontend_dir.exists():
        print("[main] frontend directory not found.")
        return 1

    backend_python = args.backend_python.strip() or find_backend_python(backend_dir)
    if not Path(backend_python).exists():
        print(f"[main] backend python not found: {backend_python}")
        return 1

    if not backend_deps_ready(backend_python):
        if args.auto_install_backend:
            code = install_backend_deps(backend_python, backend_dir)
            if code != 0:
                return code
            if not backend_deps_ready(backend_python):
                print("[main] backend dependencies still not ready after install.")
                return 1
        else:
            print("[main] backend dependency missing: uvicorn/fastapi not found in selected Python.")
            print(
                f"[main] fix: {backend_python} -m pip install -r "
                f"{(backend_dir / 'requirements.txt').resolve()}"
            )
            print("[main] or rerun with: python main.py --auto-install-backend")
            return 1

    backend_cmd = [
        backend_python,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        args.backend_host,
        "--port",
        str(args.backend_port),
    ]
    if not args.no_reload:
        backend_cmd.append("--reload")

    frontend_cmd: list[str] | None = None
    if not args.backend_only:
        try:
            frontend_cmd = resolve_frontend_cmd(
                frontend_dir=frontend_dir,
                host=args.frontend_host,
                port=args.frontend_port,
            )
        except RuntimeError as exc:
            print(f"[main] {exc}")
            print("[main] tip: you can run backend only with: python main.py --backend-only")
            return 1

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")

    print(f"[main] backend cmd: {' '.join(backend_cmd)}")
    if frontend_cmd:
        print(f"[main] frontend cmd: {' '.join(frontend_cmd)}")
    else:
        print("[main] frontend disabled (--backend-only).")

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=backend_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )
    frontend_proc: subprocess.Popen[str] | None = None
    if frontend_cmd:
        frontend_proc = subprocess.Popen(
            frontend_cmd,
            cwd=frontend_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )

    threads = [threading.Thread(target=stream_logs, args=("backend", backend_proc), daemon=True)]
    if frontend_proc:
        threads.append(threading.Thread(target=stream_logs, args=("frontend", frontend_proc), daemon=True))
    for t in threads:
        t.start()

    try:
        while True:
            bcode = backend_proc.poll()
            fcode = frontend_proc.poll() if frontend_proc else None
            if bcode is not None:
                if frontend_proc:
                    print(f"[main] backend exited with code {bcode}, stopping frontend...")
                    terminate_process(frontend_proc, "frontend")
                return bcode
            if frontend_proc and fcode is not None:
                print(f"[main] frontend exited with code {fcode}, stopping backend...")
                terminate_process(backend_proc, "backend")
                return fcode
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[main] Ctrl+C received, shutting down...")
        if os.name == "nt":
            targets = [backend_proc] + ([frontend_proc] if frontend_proc else [])
            for proc in targets:
                if proc.poll() is None:
                    try:
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                    except Exception:
                        pass
            time.sleep(1.0)
        if frontend_proc:
            terminate_process(frontend_proc, "frontend")
        terminate_process(backend_proc, "backend")
        print("[main] all processes stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
