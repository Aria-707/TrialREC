"""Starts recognition and the BFF on this PC. The browser uses the BFF port."""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from constants import BFF_HOST, BFF_PORT, RECOGNITION_HOST, RECOGNITION_PORT

ROOT = Path(__file__).resolve().parent


def recognition_is_up():
    try:
        with socket.create_connection((RECOGNITION_HOST, RECOGNITION_PORT), timeout=0.5):
            return True
    except OSError:
        return False


def stop_recognition(process):
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def main():
    print("\n" + "=" * 60, flush=True)
    print("🚀 INICIANDO KIOSCO EN DOS PROCESOS", flush=True)
    print("=" * 60, flush=True)
    print(f"BFF: http://{BFF_HOST}:{BFF_PORT}", flush=True)
    print(f"Reconocimiento: http://{RECOGNITION_HOST}:{RECOGNITION_PORT}", flush=True)
    print("=" * 60 + "\n", flush=True)

    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    recognition = subprocess.Popen(
        [sys.executable, str(ROOT / "recognition.py")],
        cwd=str(ROOT),
        env=child_env,
    )
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            if recognition.poll() is not None:
                print("✖ El reconocimiento se cerró antes de escuchar.")
                return
            if recognition_is_up():
                break
            time.sleep(0.2)
        else:
            print("✖ El reconocimiento no abrió el puerto a tiempo.")
            return

        import bff

        bff.app.run(debug=True, host=BFF_HOST, port=BFF_PORT, use_reloader=False)
    finally:
        stop_recognition(recognition)


if __name__ == "__main__":
    main()
