"""Fast startup check, also used before accepting a newly installed environment."""
import importlib
import sys


def main():
    for name in ("numpy", "cv2", "skimage", "rapidocr", "onnxruntime",
                 "reportlab", "pypdf", "psutil", "pytesseract", "faster_whisper"):
        try:
            importlib.import_module(name)
        except Exception as exc:
            print(f"AVC dependency check failed: {name}: {exc}", file=sys.stderr)
            print(f"Python: {sys.version.split()[0]} ({sys.executable})", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
