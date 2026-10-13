"""Turn `.env` text into the numbers the kiosk imports. python-dotenv reads the file."""
import os
import re
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


def _optional(name):
    return os.environ.get(name, "").strip()


_CLOCK = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


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
DEMO_COURSE_ID = _optional("DEMO_COURSE_ID")
DEMO_COURSE_NAME = _optional("DEMO_COURSE_NAME")
DEMO_INIT_TIME = _optional("DEMO_INIT_TIME")
DEMO_END_TIME = _optional("DEMO_END_TIME")
if DEMO_MODE:
    _missing_demo = [
        name
        for name, value in (
            ("DEMO_COURSE_ID", DEMO_COURSE_ID),
            ("DEMO_COURSE_NAME", DEMO_COURSE_NAME),
            ("DEMO_INIT_TIME", DEMO_INIT_TIME),
            ("DEMO_END_TIME", DEMO_END_TIME),
        )
        if not value
    ]
    if _missing_demo:
        raise RuntimeError(
            "DEMO_MODE está en true y falta "
            + ", ".join(_missing_demo)
        )
    if not _CLOCK.match(DEMO_INIT_TIME) or not _CLOCK.match(DEMO_END_TIME):
        raise RuntimeError("DEMO_INIT_TIME y DEMO_END_TIME tienen que ser HH:MM")
    if DEMO_END_TIME <= DEMO_INIT_TIME:
        raise RuntimeError("DEMO_END_TIME tiene que ser posterior a DEMO_INIT_TIME")

def _flag_default(name, default=False):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return raw.lower() in ("1", "true", "si", "sí", "yes")


BFF_HOST = _text("BFF_HOST")
BFF_PORT = _int("BFF_PORT")
RECOGNITION_HOST = _text("RECOGNITION_HOST")
RECOGNITION_PORT = _int("RECOGNITION_PORT")
KIOSK_HMAC_SECRET = _text("KIOSK_HMAC_SECRET")
KIOSK_DEBUG = _flag_default("KIOSK_DEBUG", False)
WIREGUARD_HOST = _optional("WIREGUARD_HOST")
FERNET_KEY = _optional("FERNET_KEY")

PHOTO_DIR_NAME = _text("PHOTO_DIR_NAME")
VECTOR_PATH = _text("VECTOR_PATH")
DETECTOR_MODEL_PATH = _text("DETECTOR_MODEL_PATH")
RECOGNIZER_MODEL_PATH = _text("RECOGNIZER_MODEL_PATH")
CLASSROOM_CONFIG_FILE = _text("CLASSROOM_CONFIG_FILE")
ENCRYPTION_KEY_FILE = _text("ENCRYPTION_KEY_FILE")
ENCRYPTION_SALT_FILE = _text("ENCRYPTION_SALT_FILE")
AUDIT_LOG_FILE = _text("AUDIT_LOG_FILE")
