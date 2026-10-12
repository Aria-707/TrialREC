"""Legal consent. The current row is the one with the latest fecha."""
import os
import re
import shutil
from datetime import datetime, timezone

from firebase_config import db

LEGAL_COLLECTION = "aceptacionesLegales"
_FECHA_WITHOUT_DOT = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\d+)([+-]\d{2}:\d{2}|Z)$"
)
_FECHA_FRACTION = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.(\d+)(.*)$"
)
_swept_window = None
_listener = None


def parse_fecha(value):
    """Turn a Firestore timestamp, or the no-dot text, into an aware datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    text = str(value).strip()
    glued = _FECHA_WITHOUT_DOT.fullmatch(text)
    if glued:
        zone = "+00:00" if glued.group(3) == "Z" else glued.group(3)
        text = f"{glued.group(1)}.{glued.group(2)}{zone}"
    elif text.endswith("Z"):
        text = text[:-1] + "+00:00"

    # Python 3.10 accepts a fraction of 3 or 6 digits.
    fraction = _FECHA_FRACTION.fullmatch(text)
    if fraction:
        digits = fraction.group(2)[:6].ljust(6, "0")
        text = f"{fraction.group(1)}.{digits}{fraction.group(3)}"

    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def latest_acceptance(national_id):
    """Return the fields of the newest legal row, or None."""
    documents = (
        db.collection("person")
        .document(national_id)
        .collection(LEGAL_COLLECTION)
        .stream()
    )
    latest = None
    latest_when = None
    for document in documents:
        fields = document.to_dict() or {}
        when = parse_fecha(fields.get("fecha"))
        if when is None:
            continue
        if latest_when is None or when >= latest_when:
            latest_when = when
            latest = fields
    return latest


def biometrico_is_true(national_id):
    """True only when the current row says biometrico is true."""
    current = latest_acceptance(national_id)
    if not current:
        return False
    return current.get("biometrico") is True


def biometrico_is_explicit_false(national_id):
    """True only when the current row says biometrico is false."""
    current = latest_acceptance(national_id)
    if not current:
        return False
    return current.get("biometrico") is False


def has_legal_row(national_id):
    """True when this cédula already has a row to copy."""
    return latest_acceptance(national_id) is not None


def record_acceptance(national_id):
    """Append one acceptance by copying the current row. Never invents the first one."""
    current = latest_acceptance(national_id)
    if not current:
        return False

    payload = dict(current)
    payload["evento"] = "aceptacion"
    payload["biometrico"] = True
    payload["fecha"] = datetime.now().astimezone()
    (
        db.collection("person")
        .document(national_id)
        .collection(LEGAL_COLLECTION)
        .add(payload)
    )
    return True


def delete_revoked_folders(photo_dir):
    """Remove data/{cedula} when the current biometrico is false."""
    removed = []
    if not os.path.isdir(photo_dir):
        return removed

    for name in os.listdir(photo_dir):
        if not name.isdigit():
            continue
        folder = os.path.join(photo_dir, name)
        if not os.path.isdir(folder):
            continue
        if not biometrico_is_explicit_false(name):
            continue
        shutil.rmtree(folder)
        previous = folder + ".previous"
        if os.path.isdir(previous):
            shutil.rmtree(previous)
        removed.append(name)
    return removed


def sweep_and_rebuild():
    """Drop a revoked vector even when the photo folder is already gone."""
    from face import clear_capture, forget_vector, person_ids, photo_dir
    from vector_store import delete_vector_copy

    revoked = set(delete_revoked_folders(photo_dir))
    for national_id in person_ids():
        if biometrico_is_explicit_false(national_id):
            revoked.add(national_id)
    for national_id in revoked:
        clear_capture(national_id)
        forget_vector(national_id)
        delete_vector_copy(national_id)
    if revoked:
        print(f"Barrido: {len(revoked)} consentimiento(s) retirado(s). Vector retirado.")


def sweep_before_class_frame(course_id):
    """Run the sweep once for each class window, before that window's frames."""
    global _swept_window
    today = datetime.now().strftime("%Y-%m-%d")
    window = (course_id, today)
    if _swept_window == window:
        return
    sweep_and_rebuild()
    _swept_window = window


def start_listener():
    """Watch new legal rows while this process is up."""
    global _listener
    if _listener is not None:
        return

    def on_snapshot(_documents, changes, _read_time):
        from face import clear_capture, forget_vector
        from vector_store import delete_vector_copy

        removed_any = False
        seen = set()
        for change in changes:
            person = change.document.reference.parent.parent
            if person is None:
                continue
            national_id = person.id
            if national_id in seen or not national_id.isdigit():
                continue
            seen.add(national_id)
            if not biometrico_is_explicit_false(national_id):
                continue
            dropped_folder = clear_capture(national_id)
            dropped_copy = delete_vector_copy(national_id)
            if forget_vector(national_id) or dropped_folder or dropped_copy:
                removed_any = True
        if removed_any:
            print("Consentimiento retirado. Vector retirado.")

    try:
        _listener = db.collection_group(LEGAL_COLLECTION).on_snapshot(on_snapshot)
        print("✔ Escuchando aceptacionesLegales")
    except Exception as error:
        print(f"⚠️ No se pudo escuchar aceptacionesLegales: {error}")
        _listener = None
