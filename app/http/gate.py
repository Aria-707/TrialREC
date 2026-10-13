"""Cookie of each browser and the signature between the page and recognition.

The BFF imports this module. It does not import OpenCV or the service account.
"""
import hashlib
import hmac
import json
import time
import uuid

from itsdangerous import BadSignature, URLSafeSerializer

from config import KIOSK_HMAC_SECRET, PROJECT_ROOT

COOKIE_NAME = "kiosk"
SIGNATURE_HEADER = "X-Kiosk-Signature"
CLASSROOM_HEADER = "X-Kiosk-Classroom"
TIME_HEADER = "X-Kiosk-Time"
MAX_SKEW_SECONDS = 120

_serializer = URLSafeSerializer(KIOSK_HMAC_SECRET, salt="kiosk-classroom")
_project_id = None


def dump_classroom_cookie(classroom_name):
    """Signed value stored in the browser. The page cannot read or invent it."""
    payload = {
        "classroom": classroom_name,
        "iat": int(time.time()),
        "sid": uuid.uuid4().hex,
    }
    return _serializer.dumps(payload)


def read_classroom_cookie(raw):
    """Classroom stored in a valid cookie, or an empty string."""
    if not raw:
        return ""
    try:
        payload = _serializer.loads(raw)
    except BadSignature:
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("classroom") or "").strip()


def sign_forward(method, path, query, body, classroom):
    """Headers the page adds when it forwards a request."""
    timestamp = str(int(time.time()))
    signature = _signature(timestamp, classroom, method, path, query, body or b"")
    return {
        TIME_HEADER: timestamp,
        CLASSROOM_HEADER: classroom,
        SIGNATURE_HEADER: signature,
    }


def signature_matches(method, path, query, body, classroom, timestamp, signature):
    """True when the signature matches and the clock is close."""
    if not timestamp or not signature:
        return False
    try:
        sent = int(timestamp)
    except ValueError:
        return False
    if abs(int(time.time()) - sent) > MAX_SKEW_SECONDS:
        return False
    expected = _signature(str(sent), classroom, method, path, query, body or b"")
    return hmac.compare_digest(expected, signature)


def _signature(timestamp, classroom, method, path, query, body):
    digest = hashlib.sha256(body).hexdigest()
    message = "\n".join((
        timestamp,
        classroom,
        method.upper(),
        path,
        query,
        digest,
    )).encode("utf-8")
    return hmac.new(KIOSK_HMAC_SECRET.encode("utf-8"), message, hashlib.sha256).hexdigest()


def token_is_admin(token):
    """True when Firebase signed this token and the account has the admin claim."""
    if not token:
        return False
    try:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token

        claims = id_token.verify_firebase_token(
            token,
            google_requests.Request(),
            audience=_firebase_project_id(),
        )
    except Exception as error:
        print(f"No se pudo comprobar el acceso de administrador: {error.__class__.__name__}")
        return False
    return bool(claims) and claims.get("admin") is True


def _firebase_project_id():
    global _project_id
    if _project_id:
        return _project_id
    path = PROJECT_ROOT / "firebase_config_public.json"
    with open(path, encoding="utf-8") as handle:
        _project_id = json.load(handle)["projectId"]
    return _project_id
