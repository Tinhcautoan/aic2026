"""
Video Retrieval System — Launcher
Auto-detects Docker and starts either Docker mode or local mode.
"""

import subprocess
import os
import shutil
from typing import Tuple

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DOCKER_PATHS = [
    r"C:\Program Files\Docker\Docker\resources\bin",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin"),
    os.path.expandvars(r"%ProgramFiles%\Docker\Docker\resources\bin"),
    os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\DockerDesktop\resources\bin"),
]


def _get_docker_env():
    """Return os.environ with Docker bin directories added to PATH."""
    env = os.environ.copy()
    current_path = env.get("PATH", "")
    for p in DOCKER_PATHS:
        if os.path.isdir(p) and p not in current_path:
            current_path = p + os.pathsep + current_path
    env["PATH"] = current_path
    return env


def _find_docker() -> str:
    """Find docker.exe, checking all known paths."""
    for p in DOCKER_PATHS:
        exe = os.path.join(p, "docker.exe")
        if os.path.isfile(exe):
            return exe
    return shutil.which("docker") or "docker"


def check_docker() -> Tuple[bool, str]:
    """Check if Docker daemon is reachable. Retries once with a delay."""
    env = _get_docker_env()
    docker_exe = _find_docker()

    # Verify CLI works
    try:
        r = subprocess.run([docker_exe, "--version"], capture_output=True, timeout=5, env=env)
        if r.returncode != 0:
            return False, docker_exe
    except Exception:
        return False, docker_exe

    # Check daemon - try up to 3 times with 2s delay
    for attempt in range(3):
        for cmd in (["version", "--format", "'{{.Server.Version}}'"], ["ps"], ["info"]):
            try:
                r = subprocess.run(
                    [docker_exe] + cmd, capture_output=True, timeout=20, env=env,
                )
                if r.returncode == 0 and r.stdout.strip():
                    return True, docker_exe
            except (subprocess.TimeoutExpired, Exception):
                break  # try next attempt
        if attempt < 2:
            # Give Docker engine a moment to wake up
            import time
            time.sleep(2)

    return False, docker_exe


def check_docker_compose(docker_exe: str) -> Tuple[bool, str]:
    """Check if docker compose is available."""
    env = _get_docker_env()
    try:
        r = subprocess.run([docker_exe, "compose", "version"], capture_output=True, timeout=5, env=env)
        if r.returncode == 0:
            return True, "docker compose"
    except Exception:
        pass
    # Legacy docker-compose
    legacy = shutil.which("docker-compose") or "docker-compose"
    try:
        r = subprocess.run([legacy, "--version"], capture_output=True, timeout=5, env=env)
        if r.returncode == 0:
            return True, "docker-compose"
    except Exception:
        pass
    return False, ""


def start_docker_services(docker_exe: str):
    """Start Milvus, Redis, and API via docker compose."""
    compose_file = os.path.join(PROJECT_DIR, "docker-compose.yml")
    if not os.path.exists(compose_file):
        print("[ERROR] docker-compose.yml not found!")
        return False

    print()
    print("=" * 60)
    print("  Docker Mode (Milvus + Redis + API)")
    print("=" * 60)

    ok, compose_cmd = check_docker_compose(docker_exe)
    if not ok:
        print("[ERROR] docker compose not found!")
        return False

    env = _get_docker_env()
    cmd = [docker_exe, "compose", "up", "-d"] if compose_cmd == "docker compose" else ["docker-compose", "up", "-d"]

    print(f"  Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=PROJECT_DIR, check=True, env=env)
        print()
        print("  Services starting... (1-2 min first time)")
        print()
        print("  Once ready:")
        print("    UI:     http://localhost:8000/ui")
        print("    Status: http://localhost:8000/api/status")
        print()
        print("  Commands:")
        print("    docker compose ps      - check status")
        print("    docker compose down    - stop all")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] docker compose failed: {e}")
        return False
    except FileNotFoundError:
        print("[ERROR] docker not found. Install Docker Desktop first.")
        return False


def print_instructions():
    """Print step-by-step instructions for using the system."""
    print()
    print("=" * 60)
    print("  HOW TO USE")
    print("=" * 60)
    print()
    print("  Step 1: Upload a video")
    print("    Open http://localhost:8000/process")
    print("    Paste the path to a video file or folder")
    print("    Choose \"Single File\" or \"Entire Folder\"")
    print("    Click Process and wait for it to finish")
    print()
    print("  Step 2: Search your video")
    print("    Open http://localhost:8000/ui")
    print("    Type what you're looking for (e.g. \"a dog running\")")
    print("    Press Enter or click Search")
    print()
    print("  Step 3: View results")
    print("    Matching frames appear with similarity scores")
    print("    Best matches are ranked first")
    print()
    print("  Tip: Check system status at http://localhost:8000/api/status")
    print("=" * 60)
    print()


def start_local_mode():
    """Start the FastAPI server in local/in-memory mode."""
    print()
    print("=" * 60)
    print("  Local Mode (in-memory)")
    print("=" * 60)
    print()
    print("  Data is lost when you stop the server.")
    print("  Loading models on GPU (~60 seconds)...")
    print()
    print_instructions()
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, log_level="warning", reload=False)


def main():
    print()
    print("=" * 60)
    print("  Video Retrieval System")
    print("=" * 60)
    print()

    docker_ok, docker_exe = check_docker()

    if docker_ok:
        print("  Docker: found")
        print()
        print("  1. Docker mode (data survives restart)")
        print("  2. Local mode  (data lost on restart)")
        print()
        try:
            choice = input("  Choose [1]: ").strip() or "1"
        except (KeyboardInterrupt, EOFError):
            choice = "1"

        if choice == "1":
            if start_docker_services(docker_exe):
                print_instructions()
                return
            print("  Falling back to local mode...")
        start_local_mode()
    else:
        print("  Docker: not found or not running")
        print()
        print("  For persistent storage, install Docker Desktop:")
        print("  https://www.docker.com/products/docker-desktop/")
        print()
        print("-" * 60)
        try:
            r = input("  Start in local mode? [Y/n]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            r = "y"
        if r in ("", "y", "yes"):
            start_local_mode()
        else:
            print("  Exiting.")


if __name__ == "__main__":
    main()