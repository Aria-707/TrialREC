"""In-memory event mode. The panel does not write `.env`."""
import secrets
import threading
from datetime import datetime

from firebase_admin import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from config import (
    DEMO_COURSE_ID,
    DEMO_COURSE_NAME,
    DEMO_END_TIME,
    DEMO_INIT_TIME,
    DEMO_MODE,
)
from app.firebase.client import db

_lock = threading.Lock()
_state = {
    "enabled": False,
    "course_id": None,
    "name": "",
    "ini_time": "",
    "end_time": "",
}


def _clock_text(value):
    text = str(value or "").strip()
    if len(text) >= 8 and text[5] == ":":
        return text[:5]
    return text


def display_name(raw):
    """Uppercase, extra spaces removed. Accents stay."""
    return " ".join(str(raw or "").split()).upper()


def status():
    with _lock:
        return dict(_state)


def enabled():
    with _lock:
        return _state["enabled"]


def _remember(is_on, course_id, name, ini_time, end_time):
    with _lock:
        _state["enabled"] = is_on
        _state["course_id"] = course_id
        _state["name"] = name
        _state["ini_time"] = ini_time
        _state["end_time"] = end_time


def _course_id():
    with _lock:
        return _state["course_id"]


def _publish_schedule():
    from app.attendance.scheduler import interrupt_sleep
    from app.attendance.timetable import forget_classroom_slots

    forget_classroom_slots()
    interrupt_sleep()


def _event_map(name, ini_time, end_time, classroom_name):
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "iniTime": ini_time,
        "endTime": end_time,
        "classroom": classroom_name,
    }


def _upsert_course(course_id, name, ini_time, end_time, classroom_name):
    reference = db.collection("courses").document(course_id)
    event = _event_map(name, ini_time, end_time, classroom_name)
    if reference.get().exists:
        reference.update({"nameCourse": name, "event": event})
        return
    reference.set({
        "nameCourse": name,
        "event": event,
        "estudianteID": [],
    })


def _clear_event_field(course_id):
    reference = db.collection("courses").document(course_id)
    if reference.get().exists:
        reference.update({"event": firestore.DELETE_FIELD})


def _fresh_course_id():
    courses = db.collection("courses")
    for _ in range(8):
        course_id = "evt" + secrets.token_hex(4)
        if not courses.document(course_id).get().exists:
            return course_id
    raise RuntimeError("No se pudo crear un id para el evento")


def enable(raw_name, ini_time, end_time, classroom_name):
    """Turn the mode on and write today's event on the course. Does not touch `.env`."""
    from app.recognition.face import overlay_event_vectors
    from app.attendance.timetable import valid_event_clocks

    classroom_name = (classroom_name or "").strip()
    if not classroom_name:
        return False, "Configura un salón antes de activar el modo evento."

    name = display_name(raw_name)
    ini_time = _clock_text(ini_time)
    end_time = _clock_text(end_time)
    if not name or not valid_event_clocks(ini_time, end_time):
        return False, "El nombre, la hora de inicio y la hora final son obligatorios."

    with _lock:
        previous = _state["course_id"]
        was_on = _state["enabled"]

    try:
        if DEMO_MODE:
            course_id = DEMO_COURSE_ID
        elif was_on and previous:
            course_id = previous
        else:
            course_id = _fresh_course_id()
            if previous and previous != course_id:
                _clear_event_field(previous)
        _upsert_course(course_id, name, ini_time, end_time, classroom_name)
    except Exception as error:
        print(f"[✖] No se pudo guardar el evento: {error}")
        return False, "No se pudo guardar el evento."

    _remember(True, course_id, name, ini_time, end_time)
    overlay_event_vectors()
    _publish_schedule()
    print(f"Modo evento activo. Curso {course_id}.")
    return True, ""


def _mark_event_closed(course_id):
    """Keep date and hours. closed stops the camera from opening this event again."""
    reference = db.collection("courses").document(course_id)
    snapshot = reference.get()
    if not snapshot.exists:
        return True
    event = (snapshot.to_dict() or {}).get("event")
    if not isinstance(event, dict) or event.get("closed") is True:
        return True
    reference.update({"event.closed": True})
    return True


def end_event(course_ids):
    """Wipe temporary event data, mark those courses closed, and turn the mode off."""
    from app.recognition.face import reload_stored_gallery

    if not wipe_event_data():
        return False, "No se pudieron borrar los datos del evento."
    try:
        for course_id in course_ids:
            if course_id and not _mark_event_closed(course_id):
                return False, "No se pudo cerrar el evento."
    except Exception as error:
        print(f"[✖] No se pudo cerrar el evento: {error}")
        return False, "No se pudo cerrar el evento."
    if not reload_stored_gallery():
        return False, "No se pudo apagar el modo. Intenta de nuevo."
    with _lock:
        _state["enabled"] = False
    _publish_schedule()
    print("Modo evento apagado.")
    return True, ""


def disable():
    """Turn the enrollment mode off, wipe this event, and load only the real vectors."""
    return end_event([_course_id()])


def boot_from_env(classroom_name):
    """If DEMO_MODE is true and a classroom exists, this boot starts in event mode."""
    if not DEMO_MODE:
        return
    classroom_name = (classroom_name or "").strip()
    if not classroom_name:
        print("DEMO_MODE está en true, pero no hay salón configurado. El modo evento queda apagado.")
        return

    name = display_name(DEMO_COURSE_NAME)
    try:
        _upsert_course(DEMO_COURSE_ID, name, DEMO_INIT_TIME, DEMO_END_TIME, classroom_name)
    except Exception as error:
        print(f"[✖] No se pudo preparar el curso del evento: {error}")
        return

    _remember(True, DEMO_COURSE_ID, name, DEMO_INIT_TIME, DEMO_END_TIME)
    from app.recognition.face import overlay_event_vectors

    overlay_event_vectors()
    _publish_schedule()
    print(f"Modo evento activo al arrancar. Curso {DEMO_COURSE_ID}.")


def wipe_event_data():
    """Delete event persons, sface-test vectors, and temporary legal rows. Courses stay."""
    from app.recognition.face import clear_capture

    removed_vectors = 0
    removed_people = 0
    removed_legal = 0
    try:
        for snapshot in db.collection_group("vector").stream():
            fields = snapshot.to_dict() or {}
            if snapshot.id != "sface-test" and fields.get("event") is not True:
                continue
            snapshot.reference.delete()
            removed_vectors += 1
    except Exception as error:
        print(f"[✖] No se pudieron borrar los vectores del evento: {error}")
        return False

    try:
        for snapshot in db.collection_group("aceptacionesLegales").stream():
            fields = snapshot.to_dict() or {}
            if fields.get("event") is not True:
                continue
            snapshot.reference.delete()
            removed_legal += 1
    except Exception as error:
        print(f"[✖] No se pudieron borrar los consentimientos del evento: {error}")
        return False

    try:
        for snapshot in db.collection("person").where(filter=FieldFilter("event", "==", True)).stream():
            person = snapshot.reference
            national_id = str(snapshot.id)
            for sub_name in ("vector", "aceptacionesLegales"):
                for child in person.collection(sub_name).stream():
                    child.reference.delete()
            person.delete()
            if national_id.isdigit():
                clear_capture(national_id)
            removed_people += 1
    except Exception as error:
        print(f"[✖] No se pudieron borrar las personas del evento: {error}")
        return False

    print(
        "Datos de evento borrados: "
        f"{removed_people} personas, {removed_vectors} vectores, {removed_legal} consentimientos."
    )
    return True


def _assistance_row(person_name):
    row = {
        "estadoAsistencia": "Ausente",
        "horaRegistro": None,
        "late": False,
    }
    if person_name:
        row["namePerson"] = person_name
    return row


def link_existing(national_id, raw_name):
    """Add this cédula to the event roster and to today's list. The person stays as they are."""
    course_id = _course_id()
    if not enabled() or not course_id:
        return False, "El modo evento está apagado."

    national_id = str(national_id)
    person_name = display_name(raw_name)
    reference = db.collection("courses").document(course_id)
    if not reference.get().exists:
        return False, "El curso del evento no existe."

    reference.update({"estudianteID": firestore.ArrayUnion([national_id])})
    course = reference.get().to_dict() or {}
    enrolled = [str(item) for item in course.get("estudianteID", [])]
    today = datetime.now().strftime("%Y-%m-%d")
    assistance = reference.collection("assistances").document(today)
    snapshot = assistance.get()

    if not snapshot.exists:
        rows = {}
        for enrolled_id in enrolled:
            label = person_name if enrolled_id == national_id else ""
            rows[enrolled_id] = _assistance_row(label)
        if national_id not in rows:
            rows[national_id] = _assistance_row(person_name)
        assistance.set(rows)
    else:
        current = (snapshot.to_dict() or {}).get(national_id)
        if not isinstance(current, dict):
            assistance.update({national_id: _assistance_row(person_name)})
        elif person_name and not current.get("namePerson"):
            assistance.update({f"{national_id}.namePerson": person_name})

    from app.attendance.timetable import forget_class_rosters

    forget_class_rosters()
    return True, ""


def _legal_row(national_id, portal_only):
    if portal_only:
        payload = {
            "biometrico": False,
            "datosPersonales": False,
            "terminos": False,
            "evento": "creacion",
            "fecha": datetime.now().astimezone(),
        }
    else:
        payload = {
            "biometrico": True,
            "datosPersonales": False,
            "evento": "aceptacion_biometrico",
            "fecha": datetime.now().astimezone(),
        }
    (
        db.collection("person")
        .document(national_id)
        .collection("aceptacionesLegales")
        .add(payload)
    )


def grant_event_consent(national_id):
    """Append a temporary biometric acceptance. The person and the previous row stay."""
    if not enabled():
        return False, "El modo evento está apagado."

    national_id = str(national_id)
    person = db.collection("person").document(national_id)
    if not person.get().exists:
        return False, "Esa cédula no existe."

    person.collection("aceptacionesLegales").add({
        "biometrico": True,
        "datosPersonales": False,
        "evento": "aceptacion_biometrico",
        "event": True,
        "fecha": datetime.now().astimezone(),
    })
    return True, ""


def create_person(national_id, raw_name, portal_only):
    """Create a person for this event only. A cédula that already exists is left untouched."""
    if not enabled():
        return False, "El modo evento está apagado."

    national_id = str(national_id)
    name = display_name(raw_name)
    course_id = _course_id()
    if not name:
        return False, "El nombre es obligatorio."
    if not course_id:
        return False, "El curso del evento no existe."

    person = db.collection("person").document(national_id)
    if person.get().exists:
        return False, "Esa cédula ya existe."

    person.set({
        "namePerson": name,
        "type": "Estudiante",
        "courses": [course_id],
        "event": True,
    })
    _legal_row(national_id, portal_only)
    return link_existing(national_id, name)
