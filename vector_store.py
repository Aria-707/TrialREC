"""Encrypted SFace vector stored under person/{cedula}/vector/sface."""
import json
from datetime import datetime

from encryption import cipher
from firebase_config import db

MODEL_NAME = "sface-2021dec"
VECTOR_LENGTH = 128


def _reference(national_id):
    return (
        db.collection("person")
        .document(str(national_id))
        .collection("vector")
        .document("sface")
    )


def _same_numbers(left, right):
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return False
    return json.loads(json.dumps(left)) == json.loads(json.dumps(right))


def save_vector_copy(national_id, values):
    """Store one Fernet token. A failure leaves the photos in place."""
    try:
        token = cipher().encrypt(json.dumps(values).encode("utf-8")).decode("ascii")
        _reference(national_id).set({
            "modelo": MODEL_NAME,
            "vector": token,
            "actualizado": datetime.now().astimezone(),
        })
        print("Copia de Firebase escrita.")
        return True
    except Exception:
        print("Copia de Firebase no se pudo escribir.")
        return False


def read_vector_copy(national_id):
    """Decrypt one stored vector. None when it is missing or unreadable."""
    try:
        snapshot = _reference(national_id).get()
        if not snapshot.exists:
            return None
        return _decrypt_fields(cipher(), snapshot.to_dict())
    except Exception:
        return None


def commit_vector_copy(national_id, values):
    """Write the token and accept it only when the read-back matches."""
    if not save_vector_copy(national_id, values):
        return False
    if not _same_numbers(read_vector_copy(national_id), values):
        print("Copia de Firebase no se pudo leer.")
        return False
    print("Copia de Firebase leída.")
    return True


def list_stored_vectors():
    """Decrypt every sface document. A network error returns a reason and no gallery."""
    loaded = {}
    missing = 0
    try:
        fernet = cipher()
        snapshots = list(db.collection_group("vector").stream())
    except Exception:
        return {}, 0, "Firebase no responde"
    for snapshot in snapshots:
        if snapshot.id != "sface":
            continue
        person = snapshot.reference.parent.parent
        if person is None or not str(person.id).isdigit():
            continue
        values = _decrypt_fields(fernet, snapshot.to_dict())
        if values is None:
            missing += 1
            continue
        loaded[str(person.id)] = values
    return loaded, missing, None


def _event_reference(national_id):
    return (
        db.collection("person")
        .document(str(national_id))
        .collection("vector")
        .document("sface-test")
    )


def commit_event_vector(national_id, values):
    """Write sface-test and accept it only when the read-back matches. sface stays."""
    if not save_event_vector(national_id, values):
        return False
    if not _same_numbers(read_event_vector(national_id), values):
        print("Copia de evento no se pudo leer.")
        return False
    print("Copia de evento leída.")
    return True


def save_event_vector(national_id, values):
    try:
        token = cipher().encrypt(json.dumps(values).encode("utf-8")).decode("ascii")
        _event_reference(national_id).set({
            "modelo": MODEL_NAME,
            "vector": token,
            "actualizado": datetime.now().astimezone(),
            "event": True,
        })
        print("Copia de evento escrita.")
        return True
    except Exception:
        print("Copia de evento no se pudo escribir.")
        return False


def read_event_vector(national_id):
    try:
        snapshot = _event_reference(national_id).get()
        if not snapshot.exists:
            return None
        return _decrypt_fields(cipher(), snapshot.to_dict())
    except Exception:
        return None


def list_event_vectors():
    """Decrypt sface-test documents. A network error returns no overlay."""
    loaded = {}
    try:
        fernet = cipher()
        snapshots = list(db.collection_group("vector").stream())
    except Exception:
        print("Firebase no responde al leer los vectores del evento.")
        return {}
    for snapshot in snapshots:
        if snapshot.id != "sface-test":
            continue
        person = snapshot.reference.parent.parent
        if person is None or not str(person.id).isdigit():
            continue
        values = _decrypt_fields(fernet, snapshot.to_dict())
        if values is None:
            continue
        loaded[str(person.id)] = values
    return loaded


def delete_vector_copy(national_id):
    """Remove the Firebase copy when it exists. Missing is not an error."""
    try:
        reference = _reference(national_id)
        if not reference.get().exists:
            return False
        reference.delete()
        print("Copia de Firebase borrada.")
        return True
    except Exception:
        print("Copia de Firebase no se pudo borrar.")
        return False


def _decrypt_fields(fernet, fields):
    if not fields or fields.get("modelo") != MODEL_NAME:
        return None
    token = fields.get("vector")
    if not isinstance(token, str):
        return None
    try:
        raw = fernet.decrypt(token.encode("ascii"))
        values = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(values, list) or len(values) != VECTOR_LENGTH:
        return None
    return values
