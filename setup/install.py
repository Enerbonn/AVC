"""Rebuild dependencies without mixing Python ABIs; restore the old env on failure."""
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]


def choose_python():
    # Prefer the existing environment's base interpreter when it still exists.
    env_python = ROOT / ".venv/Scripts/python.exe"
    if env_python.is_file():
        result = subprocess.run([str(env_python), "-c",
                                 "import sys; print(sys._base_executable)"],
                                capture_output=True, text=True)
        candidate = Path(result.stdout.strip()) if result.returncode == 0 else None
        if candidate and candidate.is_file():
            return str(candidate)
    for version in ("-3.12", "-3.13"):
        try:
            result = subprocess.run(["py", version, "-c", "import sys; print(sys.executable)"],
                                    capture_output=True, text=True)
            if result.returncode == 0 and Path(result.stdout.strip()).is_file():
                return result.stdout.strip()
        except FileNotFoundError:
            break
    if sys.version_info < (3, 12):
        raise RuntimeError("Install Python 3.12 or newer before setup.")
    return sys.executable


def main():
    interpreter = choose_python()
    environment = ROOT / ".venv"
    backup = ROOT / (".venv-before-setup-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    if environment.exists():
        environment.rename(backup)
    cache = ROOT / ".local/setup-temp"
    cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TEMP=str(cache), TMP=str(cache))
    python = str(environment / "Scripts/python.exe")
    try:
        for command in ([interpreter, "-m", "venv", str(environment)],
                        [python, "-m", "pip", "install", "--upgrade", "pip"],
                        [python, "-m", "pip", "install", "-r", str(ROOT / "setup/requirements.txt")],
                        [python, str(ROOT / "setup/check_runtime.py")]):
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    except Exception:
        # Keep failed installation evidence, and restore the previous environment.
        if environment.exists():
            environment.rename(ROOT / (backup.name + "-failed"))
        if backup.exists():
            backup.rename(environment)
        raise
    if backup.exists():
        print(f"Previous environment preserved at: {backup}")
    print("AVC dependency checks passed.")


if __name__ == "__main__":
    main()
