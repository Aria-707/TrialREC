"""Thresholds and local paths. Values are the ones the kiosk already uses."""

CONFIDENCE_THRESHOLD = 70
FACE_SIZE = 150
RECOGNITION_SECONDS = 3
MINUTES_BEFORE_START = 5
MINUTES_AFTER_START = 30
# Attendance field `late` becomes true after this many minutes.
LATE_AFTER_MINUTES = 15
ON_TIME_MARGIN_MINUTES = 1
COUNTDOWN_WINDOW_MINUTES = 5
SCHEDULER_LOOKAHEAD_MINUTES = 6
DEMO_MODE = False
DEMO_COURSE_ID = "00000"
DEMO_ID_MIN = 2000000000
DEMO_ID_MAX = 2999999999

BFF_HOST = "127.0.0.1"
BFF_PORT = 5000
RECOGNITION_HOST = "127.0.0.1"
RECOGNITION_PORT = 5001

PHOTO_DIR_NAME = "data"
MODEL_PATH = "backend/lbph_model.xml"
LABEL_MAP_PATH = "backend/lbph_labels.json"
CLASSROOM_CONFIG_FILE = "classroom_config.txt"
ENCRYPTION_KEY_FILE = "encryption.key"
ENCRYPTION_SALT_FILE = "encryption_salt.key"
AUDIT_LOG_FILE = "audit_log.json"

SPANISH_TO_ENGLISH_WEEKDAYS = {
    "Lunes": "Monday",
    "Martes": "Tuesday",
    "Miércoles": "Wednesday",
    "Jueves": "Thursday",
    "Viernes": "Friday",
    "Sábado": "Saturday",
    "Domingo": "Sunday",
}

ENGLISH_TO_SPANISH_WEEKDAYS = {english: spanish for spanish, english in SPANISH_TO_ENGLISH_WEEKDAYS.items()}
