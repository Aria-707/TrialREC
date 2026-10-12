"""Class schedule, attendance window, and classroom list."""
import re
from datetime import datetime, timedelta

from config import (
    COUNTDOWN_WINDOW_MINUTES,
    LATE_AFTER_MINUTES,
    MINUTES_AFTER_START,
    MINUTES_BEFORE_START,
    ON_TIME_MARGIN_MINUTES,
)
from firebase_config import db

ENGLISH_WEEKDAYS = (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
)
# Firestore stores the Spanish names.
SPANISH_TO_ENGLISH_WEEKDAYS = {
    "Lunes": "Monday",
    "Martes": "Tuesday",
    "Miércoles": "Wednesday",
    "Jueves": "Thursday",
    "Viernes": "Friday",
    "Sábado": "Saturday",
    "Domingo": "Sunday",
}
ENGLISH_TO_SPANISH_WEEKDAYS = {
    english: spanish for spanish, english in SPANISH_TO_ENGLISH_WEEKDAYS.items()
}
_CLOCK_TEXT = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
# Filled on the first read for a classroom, and again when the scheduler confirms the timetable.
_slot_cache = {}


def available_classrooms():
    """Unique classroom names from course groups and from direct schedules."""
    try:
        classrooms = set()
        print("\n=== EXTRAYENDO SALONES DISPONIBLES ===")

        for course_doc in db.collection("courses").get():
            course_id = course_doc.id
            course_data = course_doc.to_dict()

            print(f"\nProcesando curso: {course_id}")
            print(f"  Nombre: {course_data.get('nameCourse', 'Sin nombre')}")

            try:
                groups = db.collection("courses").document(course_id).collection("groups").get()
                if groups:
                    print(f"  ✓ Tiene {len(groups)} grupos")
                    for group_doc in groups:
                        print(f"    Grupo: {group_doc.id}")
                        for slot in group_doc.to_dict().get("schedule", []):
                            classroom_name = slot.get("classroom", "").strip()
                            if classroom_name:
                                classrooms.add(classroom_name)
                                print(f"      → Salón: {classroom_name}")
                    continue
            except Exception as error:
                print(f"  [!] No tiene subcolección groups o error: {error}")

            schedule = course_data.get("schedule", [])
            if schedule:
                print(f"  ✓ Tiene schedule directo con {len(schedule)} horarios")
                for slot in schedule:
                    classroom_name = slot.get("classroom", "").strip()
                    if classroom_name:
                        classrooms.add(classroom_name)
                        print(f"    → Salón: {classroom_name}")

        classroom_list = sorted(list(classrooms))
        print(f"\n=== SALONES ENCONTRADOS: {len(classroom_list)} ===")
        for classroom_name in classroom_list:
            print(f"  • {classroom_name}")
        print("=" * 50 + "\n")
        return classroom_list

    except Exception as error:
        print(f"[✖] ERROR obteniendo los salones de courses: {error}")
        import traceback
        traceback.print_exc()
        return []


_roster_cache = {}


def forget_class_rosters():
    """Drop the in-memory class lists. The next frame reads each course once."""
    _roster_cache.clear()


def forget_classroom_slots():
    """Drop the in-memory timetable. The next question reads Firestore once."""
    _slot_cache.clear()
    forget_class_rosters()


_schedule_listeners = []
_courses_watch = None
_groups_watch = None


def on_schedule_change(callback):
    """Run after a course or group document changes. The scheduler uses this to wake."""
    _schedule_listeners.append(callback)


def start_schedule_watch():
    """Reload the timetable when Firebase changes, without waiting for the next class."""
    global _courses_watch, _groups_watch
    if _courses_watch is not None or _groups_watch is not None:
        return

    def watch(query):
        opened = {"ready": False}

        def on_snapshot(_documents, changes, _read_time):
            if not opened["ready"]:
                opened["ready"] = True
                return
            if not changes:
                return
            forget_classroom_slots()
            print("Horario actualizado en Firebase. Se vuelve a calcular.")
            for callback in list(_schedule_listeners):
                callback()

        return query.on_snapshot(on_snapshot)

    try:
        _courses_watch = watch(db.collection("courses"))
        print("✔ Escuchando courses")
    except Exception as error:
        print(f"⚠️ No se pudo escuchar courses: {error}")
    try:
        _groups_watch = watch(db.collection_group("groups"))
        print("✔ Escuchando groups")
    except Exception as error:
        print(f"⚠️ No se pudo escuchar groups: {error}")


def load_classroom_slots(required_classroom, teacher_id=None, refresh=False):
    """Slots for this classroom. Firestore is read only on the first use or a refresh."""
    if not required_classroom:
        return []
    cache_key = (required_classroom, teacher_id)
    if not refresh and cache_key in _slot_cache:
        return _slot_cache[cache_key]
    if refresh:
        forget_class_rosters()
    slots = _read_classroom_slots(required_classroom, teacher_id)
    _slot_cache[cache_key] = slots
    print(f"Horario en memoria para {required_classroom}: {len(slots)} franjas")
    return slots


def _read_classroom_slots(required_classroom, teacher_id=None):
    """One walk of courses. Groups win over a schedule on the course document."""
    slots = []
    try:
        for course_doc in db.collection("courses").get():
            course_id = course_doc.id
            course_data = course_doc.to_dict() or {}
            if teacher_id and course_data.get("profesorID") != teacher_id:
                continue
            course_name = course_data.get("nameCourse", "Sin nombre")
            event_slot = _event_slot(course_data, course_id, course_name, required_classroom)
            if event_slot:
                slots.append(event_slot)
            try:
                groups = db.collection("courses").document(course_id).collection("groups").get()
                if groups:
                    for group_doc in groups:
                        for raw in group_doc.to_dict().get("schedule", []):
                            stored = _stored_slot(raw, course_id, course_name, required_classroom)
                            if stored:
                                slots.append(stored)
                    continue
            except Exception:
                pass
            for raw in course_data.get("schedule", []):
                stored = _stored_slot(raw, course_id, course_name, required_classroom)
                if stored:
                    slots.append(stored)
    except Exception as error:
        print(f"[✖] ERROR leyendo el horario del salón '{required_classroom}': {error}")
    return slots


def _stored_slot(raw, course_id, course_name, required_classroom):
    classroom_name = (raw or {}).get("classroom", "").strip()
    if classroom_name != required_classroom:
        return None
    return {
        "course_id": course_id,
        "course_name": course_name,
        "day": raw.get("day", ""),
        "iniTime": raw.get("iniTime", "00:00"),
        "endTime": raw.get("endTime", "23:59"),
        "classroom": classroom_name,
    }


def _event_slot(course_data, course_id, course_name, required_classroom):
    """One-day event. A weekly schedule is a different slot."""
    raw = course_data.get("event")
    if not isinstance(raw, dict):
        return None
    classroom_name = (raw.get("classroom") or "").strip()
    if classroom_name != required_classroom:
        return None
    date = str(raw.get("date") or "").strip()
    ini_time = str(raw.get("iniTime") or "").strip()
    end_time = str(raw.get("endTime") or "").strip()
    if not date or not valid_event_clocks(ini_time, end_time):
        return None
    return {
        "course_id": course_id,
        "course_name": course_name,
        "day": "",
        "iniTime": ini_time,
        "endTime": end_time,
        "classroom": classroom_name,
        "event": True,
        "date": date,
    }


def valid_event_clocks(ini_time, end_time):
    """True when both clocks are HH:MM and the end is later the same day."""
    if not _CLOCK_TEXT.match(str(ini_time or "")) or not _CLOCK_TEXT.match(str(end_time or "")):
        return False
    return _minutes(end_time) > _minutes(ini_time)


def _minutes(text):
    hour, minute = (int(part) for part in str(text).split(":"))
    return hour * 60 + minute


def event_window_matches(slot, today, current_time, required_classroom):
    """Start time when this event is open today. The window ends at endTime."""
    if not slot.get("event"):
        return None
    if (slot.get("classroom") or "").strip() != required_classroom:
        return None
    if slot.get("date") != today:
        return None
    try:
        start = _minutes(slot.get("iniTime", "00:00"))
        end = _minutes(slot.get("endTime", "00:00"))
        current = _minutes(current_time)
    except (TypeError, ValueError):
        return None
    if start - MINUTES_BEFORE_START <= current <= end:
        return slot.get("iniTime")
    return None


def event_close_for(required_classroom, course_id, today):
    """endTime when the open course is today's event. None for a weekly class."""
    for slot in load_classroom_slots(required_classroom):
        if slot.get("event") and slot.get("course_id") == course_id and slot.get("date") == today:
            return slot.get("endTime")
    return None


def _event_start(slot, moment):
    """Today's event start, when that minute is still ahead."""
    try:
        day = datetime.strptime(slot.get("date", ""), "%Y-%m-%d")
        hour, minute = (int(part) for part in str(slot.get("iniTime", "")).split(":"))
    except (TypeError, ValueError):
        return None
    candidate = day.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate < moment.replace(second=0, microsecond=0):
        return None
    return candidate


def _event_countdown(slot, current_time, required_classroom):
    """Countdown payload when the event opens within the next few minutes."""
    today = datetime.now().strftime("%Y-%m-%d")
    if slot.get("date") != today:
        return None
    if event_window_matches(slot, today, current_time, required_classroom):
        return None
    try:
        start = _minutes(slot.get("iniTime", "00:00"))
        current = _minutes(current_time)
    except (TypeError, ValueError):
        return None
    minutes_until_open = (start - MINUTES_BEFORE_START) - current
    if minutes_until_open < 0 or minutes_until_open > COUNTDOWN_WINDOW_MINUTES:
        return None
    return {
        "course_id": slot["course_id"],
        "course_name": slot.get("course_name", "Sin nombre"),
        "start_time": slot.get("iniTime"),
        "minutes_until_start": int(minutes_until_open),
        "classroom": required_classroom,
    }


def _clock(moment):
    english_day = moment.strftime("%A")
    spanish_day = ENGLISH_TO_SPANISH_WEEKDAYS.get(english_day, english_day)
    return spanish_day, english_day, moment.strftime("%H:%M")


def active_course_for_classroom(teacher_id=None, required_classroom=None):
    """Course whose registration window is open. Uses the in-memory timetable."""
    try:
        if not required_classroom:
            print("[!] No hay salón configurado - no se puede buscar curso")
            return (None, None)

        now = datetime.now()
        spanish_day, english_day, current_time = _clock(now)
        today = now.strftime("%Y-%m-%d")
        slots = load_classroom_slots(required_classroom, teacher_id=teacher_id)
        for slot in slots:
            if not slot.get("event"):
                continue
            start_text = event_window_matches(slot, today, current_time, required_classroom)
            if start_text:
                return (slot["course_id"], start_text)
        for slot in slots:
            start_text = schedule_matches_classroom(
                [slot], spanish_day, english_day, current_time, required_classroom,
            )
            if start_text:
                return (slot["course_id"], start_text)
        return (None, None)
    except Exception as error:
        print(f"[✖] ERROR obteniendo el curso activo del salón '{required_classroom}': {error}")
        return (None, None)


def class_roster(course_id):
    """Cédulas of the open class. Firestore is read once per course until the timetable refreshes."""
    if not course_id:
        return []
    if course_id in _roster_cache:
        return _roster_cache[course_id]
    try:
        document = db.collection("courses").document(course_id).get()
        raw = (document.to_dict() or {}).get("estudianteID", []) if document.exists else []
        enrolled = [str(item) for item in raw]
    except Exception as error:
        print(f"[✖] ERROR leyendo la lista del curso '{course_id}': {error}")
        return []
    _roster_cache[course_id] = enrolled
    print(f"Lista de la clase {course_id} en memoria: {len(enrolled)} estudiantes")
    return enrolled


def course_name_for(required_classroom, course_id):
    """Name already stored with the slot. Does not read Firestore again."""
    for slot in load_classroom_slots(required_classroom):
        if slot["course_id"] == course_id:
            return slot["course_name"]
    return "Sin nombre"


def milliseconds_until_window_end(start_text, now=None, close_text=None):
    """How long until the open window stops matching. The last minute still counts."""
    now = now or datetime.now()
    if close_text:
        hour, minute = (int(part) for part in close_text.split(":"))
        window_end = now.replace(hour=hour, minute=minute, second=59, microsecond=0)
        return max(0, int((window_end - now).total_seconds() * 1000))
    hour, minute = (int(part) for part in start_text.split(":"))
    start = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    window_end = start + timedelta(minutes=MINUTES_AFTER_START + 1)
    return max(0, int((window_end - now).total_seconds() * 1000))


def milliseconds_until_countdown(required_classroom, now=None):
    """Wait until the on-screen countdown should appear. None when nothing is scheduled."""
    now = now or datetime.now()
    soonest = None
    for slot in load_classroom_slots(required_classroom):
        when = next_occurrence(slot.get("day", ""), slot.get("iniTime", "00:00"), now)
        if when is None:
            continue
        countdown_at = when - timedelta(minutes=MINUTES_BEFORE_START + COUNTDOWN_WINDOW_MINUTES)
        if countdown_at <= now:
            continue
        if soonest is None or countdown_at < soonest:
            soonest = countdown_at
    if soonest is None:
        return None
    return max(0, int((soonest - now).total_seconds() * 1000))


def next_occurrence(day_name, start_text, moment):
    """Next datetime at this weekday and clock, from `moment` truncated to the minute."""
    english = SPANISH_TO_ENGLISH_WEEKDAYS.get(day_name, day_name)
    if english not in ENGLISH_WEEKDAYS:
        return None
    hour, minute = (int(part) for part in str(start_text).split(":"))
    base = moment.replace(second=0, microsecond=0)
    days_ahead = (ENGLISH_WEEKDAYS.index(english) - base.weekday()) % 7
    candidate = (base + timedelta(days=days_ahead)).replace(
        hour=hour, minute=minute, second=0, microsecond=0,
    )
    if candidate < base:
        candidate += timedelta(days=7)
    return candidate


def upcoming_starts(required_classroom, now=None, refresh=False):
    """(course_id, start_datetime) for each slot, soonest first."""
    now = now or datetime.now()
    found = []
    for slot in load_classroom_slots(required_classroom, refresh=refresh):
        if slot.get("event"):
            when = _event_start(slot, now)
        else:
            when = next_occurrence(slot.get("day", ""), slot.get("iniTime", "00:00"), now)
        if when is None:
            continue
        found.append((slot["course_id"], when))
    found.sort(key=lambda item: item[1])
    return found


def schedule_matches_classroom(schedule, spanish_day, english_day, current_time, required_classroom):
    """Return the class start time when day, classroom, and the live window match."""
    for slot in schedule:
        if slot.get("event"):
            continue
        slot_day = slot.get("day", "")
        start_text = slot.get("iniTime", "00:00")
        classroom_name = slot.get("classroom", "").strip()

        if classroom_name != required_classroom:
            continue
        if slot_day != spanish_day and slot_day != english_day:
            continue

        start_clock = datetime.strptime(start_text, "%H:%M")
        now_clock = datetime.strptime(current_time, "%H:%M")
        window_open = start_clock - timedelta(minutes=MINUTES_BEFORE_START)
        window_close = start_clock + timedelta(minutes=MINUTES_AFTER_START)
        if window_open <= now_clock <= window_close:
            return start_text

    return None


def arrival_from_start(start_text, now=None):
    """Screen category and the Firestore late flag, from one clock reading."""
    try:
        if not start_text:
            return "arrived", False

        now = now or datetime.now()
        current_time = now.strftime("%H:%M")
        start_clock = datetime.strptime(start_text, "%H:%M")
        now_clock = datetime.strptime(current_time, "%H:%M")
        difference = (now_clock - start_clock).total_seconds() / 60

        print(f"📊 Diferencia de tiempo: {difference:.1f} minutos")

        is_late = difference > LATE_AFTER_MINUTES
        if difference < -ON_TIME_MARGIN_MINUTES:
            category = "early"
        elif difference <= ON_TIME_MARGIN_MINUTES:
            category = "on_time"
        elif difference <= LATE_AFTER_MINUTES:
            category = "arrived"
        else:
            category = "late"
        return category, is_late

    except Exception as error:
        print(f"Error calculando la categoría de llegada para el inicio {start_text}: {error}")
        return "arrived", False


def next_course(required_classroom):
    """Next class whose registration opens within 5 minutes. Uses the in-memory timetable."""
    try:
        if not required_classroom:
            return None
        now = datetime.now()
        spanish_day, english_day, current_time = _clock(now)
        upcoming = []
        for slot in load_classroom_slots(required_classroom):
            if slot.get("event"):
                found = _event_countdown(slot, current_time, required_classroom)
                if found:
                    upcoming.append(found)
                continue
            found = next_schedule_opening(
                slot, spanish_day, english_day, current_time, required_classroom,
                slot["course_id"], {"nameCourse": slot["course_name"]},
            )
            if found:
                upcoming.append(found)
        if not upcoming:
            return None
        upcoming.sort(key=lambda item: item["minutes_until_start"])
        return upcoming[0]
    except Exception as error:
        print(f"[✖] ERROR buscando el próximo curso del salón '{required_classroom}': {error}")
        return None


def next_schedule_opening(slot, spanish_day, english_day, current_time, required_classroom, course_id, course_data):
    """Return the course when registration opens in 0 to 5 minutes."""
    try:
        slot_day = slot.get("day", "")
        start_text = slot.get("iniTime", "00:00")
        classroom_name = slot.get("classroom", "").strip()

        if classroom_name != required_classroom:
            return None
        if slot_day != spanish_day and slot_day != english_day:
            return None

        now_clock = datetime.strptime(current_time, "%H:%M")
        start_clock = datetime.strptime(start_text, "%H:%M")
        minutes_until_start = (start_clock - now_clock).total_seconds() / 60
        minutes_until_open = minutes_until_start - MINUTES_BEFORE_START

        if minutes_until_open < 0 or minutes_until_open > COUNTDOWN_WINDOW_MINUTES:
            return None

        return {
            "course_id": course_id,
            "course_name": course_data.get("nameCourse", "Sin nombre"),
            "start_time": start_text,
            "minutes_until_start": int(minutes_until_open),
            "classroom": required_classroom,
        }

    except Exception:
        return None
