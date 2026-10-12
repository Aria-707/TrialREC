"""Turn `.env` text into the numbers the kiosk imports. python-dotenv reads the file."""
import os
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)


def _text(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Falta la variable de entorno {name}")
    return value


def _int(name):
    return int(_text(name))


def _float(name):
    return float(_text(name))


def _flag(name):
    return _text(name).lower() in ("1", "true", "si", "sí", "yes")


COSINE_THRESHOLD = _float("COSINE_THRESHOLD")
AMBIGUITY_MARGIN = _float("AMBIGUITY_MARGIN")
FACE_SIZE = _int("FACE_SIZE")
RECOGNITION_SECONDS = _int("RECOGNITION_SECONDS")
MINUTES_BEFORE_START = _int("MINUTES_BEFORE_START")
MINUTES_AFTER_START = _int("MINUTES_AFTER_START")
LATE_AFTER_MINUTES = _int("LATE_AFTER_MINUTES")
ON_TIME_MARGIN_MINUTES = _int("ON_TIME_MARGIN_MINUTES")
COUNTDOWN_WINDOW_MINUTES = _int("COUNTDOWN_WINDOW_MINUTES")
SCHEDULER_LOOKAHEAD_MINUTES = _int("SCHEDULER_LOOKAHEAD_MINUTES")
DEMO_MODE = _flag("DEMO_MODE")
DEMO_COURSE_ID = _text("DEMO_COURSE_ID")
DEMO_ID_MIN = _int("DEMO_ID_MIN")
DEMO_ID_MAX = _int("DEMO_ID_MAX")

BFF_HOST = _text("BFF_HOST")
BFF_PORT = _int("BFF_PORT")
RECOGNITION_HOST = _text("RECOGNITION_HOST")
RECOGNITION_PORT = _int("RECOGNITION_PORT")

PHOTO_DIR_NAME = _text("PHOTO_DIR_NAME")
VECTOR_PATH = _text("VECTOR_PATH")
DETECTOR_MODEL_PATH = _text("DETECTOR_MODEL_PATH")
RECOGNIZER_MODEL_PATH = _text("RECOGNIZER_MODEL_PATH")
CLASSROOM_CONFIG_FILE = _text("CLASSROOM_CONFIG_FILE")
ENCRYPTION_KEY_FILE = _text("ENCRYPTION_KEY_FILE")
ENCRYPTION_SALT_FILE = _text("ENCRYPTION_SALT_FILE")
AUDIT_LOG_FILE = _text("AUDIT_LOG_FILE")
