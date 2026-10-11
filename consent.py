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


def record_acceptance(national_id):
    """Append one acceptance. The previous row stays as it was."""
    collection = (
        db.collection("person")
        .document(national_id)
        .collection(LEGAL_COLLECTION)
    )
    current = latest_acceptance(national_id)
    moment = datetime.now().astimezone()
    if current:
        payload = dict(current)
        payload["evento"] = "aceptación"
        payload["biometrico"] = True
        payload["fecha"] = moment
    else:
        payload = {
            "evento": "aceptación",
            "biometrico": True,
            "datosPersonales": True,
            "fecha": moment,
        }
    collection.add(payload)


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
    """Drop revoked photo folders and rebuild the model when something left."""
    from face import photo_dir, rebuild_model

    removed = delete_revoked_folders(photo_dir)
    if not removed:
        return
    print(f"Barrido: {len(removed)} carpeta(s) sin consentimiento. Reconstruyendo el modelo.")
    rebuild_model()


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
        from face import photo_dir, rebuild_model

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
            folder = os.path.join(photo_dir, national_id)
            if not os.path.isdir(folder):
                continue
            shutil.rmtree(folder)
            previous = folder + ".previous"
            if os.path.isdir(previous):
                shutil.rmtree(previous)
            removed_any = True
        if removed_any:
            print("Consentimiento retirado. Reconstruyendo el modelo.")
            rebuild_model()

    try:
        _listener = db.collection_group(LEGAL_COLLECTION).on_snapshot(on_snapshot)
        print("✔ Escuchando aceptacionesLegales")
    except Exception as error:
        print(f"⚠️ No se pudo escuchar aceptacionesLegales: {error}")
        _listener = None
