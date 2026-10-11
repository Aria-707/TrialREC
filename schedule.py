"""Class schedule, attendance window, and classroom list."""
from datetime import datetime, timedelta

from constants import (
    COUNTDOWN_WINDOW_MINUTES,
    ENGLISH_TO_SPANISH_WEEKDAYS,
    LATE_AFTER_MINUTES,
    MINUTES_AFTER_START,
    MINUTES_BEFORE_START,
    ON_TIME_MARGIN_MINUTES,
)
from firebase_config import db


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


def active_course_for_classroom(teacher_id=None, required_classroom=None):
    """Course whose registration window is open in the configured classroom."""
    try:
        now = datetime.now()
        english_day = now.strftime("%A")
        spanish_day = ENGLISH_TO_SPANISH_WEEKDAYS.get(english_day, english_day)
        current_time = now.strftime("%H:%M")

        print("\n=== BUSCANDO CURSO ACTIVO CON SALÓN ===")
        print(f"Día (español): {spanish_day}")
        print(f"Hora actual: {current_time}")
        print(f"Salón requerido: {required_classroom}")

        if not required_classroom:
            print("[!] No hay salón configurado - no se puede buscar curso")
            return (None, None)

        courses_ref = db.collection("courses")
        if teacher_id:
            courses_ref = courses_ref.where("profesorID", "==", teacher_id)

        for course_doc in courses_ref.get():
            course_id = course_doc.id
            course_data = course_doc.to_dict()

            print(f"\n  Verificando curso: {course_id}")
            print(f"  Nombre: {course_data.get('nameCourse')}")

            try:
                groups = db.collection("courses").document(course_id).collection("groups").get()
                if groups:
                    print(f"    → Buscando en {len(groups)} grupos...")
                    for group_doc in groups:
                        start_text = schedule_matches_classroom(
                            group_doc.to_dict().get("schedule", []),
                            spanish_day,
                            english_day,
                            current_time,
                            required_classroom,
                            f"Grupo {group_doc.id}",
                        )
                        if start_text:
                            print(f"  [✔] ¡CURSO ACTIVO ENCONTRADO: {course_id} - Grupo {group_doc.id}!")
                            return (course_id, start_text)
                    continue
            except Exception as error:
                print(f"    [!] No tiene grupos: {error}")

            schedule = course_data.get("schedule", [])
            if schedule:
                start_text = schedule_matches_classroom(
                    schedule,
                    spanish_day,
                    english_day,
                    current_time,
                    required_classroom,
                    "Schedule directo",
                )
                if start_text:
                    print(f"  [✔] ¡CURSO ACTIVO ENCONTRADO: {course_id}!")
                    return (course_id, start_text)

        print("\n[!] No se encontró curso activo para:")
        print(f"    • Día: {spanish_day}")
        print(f"    • Hora: {current_time}")
        print(f"    • Salón: {required_classroom}")
        return (None, None)

    except Exception as error:
        print(f"[✖] ERROR obteniendo el curso activo del salón '{required_classroom}': {error}")
        import traceback
        traceback.print_exc()
        return (None, None)


def schedule_matches_classroom(schedule, spanish_day, english_day, current_time, required_classroom, source):
    """Return the class start time when day, classroom, and the live window match."""
    for slot in schedule:
        slot_day = slot.get("day", "")
        start_text = slot.get("iniTime", "00:00")
        end_text = slot.get("endTime", "23:59")
        classroom_name = slot.get("classroom", "").strip()

        print(f"      [{source}] {slot_day} {start_text}-{end_text} @ {classroom_name}")

        if classroom_name != required_classroom:
            print(f"        ✗ Salón no coincide (esperado: {required_classroom})")
            continue

        print("        ✓ Salón coincide")

        if slot_day != spanish_day and slot_day != english_day:
            print("        ✗ Día no coincide")
            continue

        print("        ✓ Día coincide")

        start_clock = datetime.strptime(start_text, "%H:%M")
        now_clock = datetime.strptime(current_time, "%H:%M")
        window_open = start_clock - timedelta(minutes=MINUTES_BEFORE_START)
        window_close = start_clock + timedelta(minutes=MINUTES_AFTER_START)

        print(f"        Ventana de registro: {window_open.strftime('%H:%M')} - {window_close.strftime('%H:%M')}")

        if window_open <= now_clock <= window_close:
            print("        ✓ Hora dentro del rango de registro")
            return start_text

        print("        ✗ Hora fuera del rango")

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
    """Next class whose registration opens within 5 minutes."""
    try:
        now = datetime.now()
        english_day = now.strftime("%A")
        spanish_day = ENGLISH_TO_SPANISH_WEEKDAYS.get(english_day, english_day)
        current_time = now.strftime("%H:%M")

        print("\n=== BUSCANDO PRÓXIMO CURSO (CONTADOR) ===")
        print(f"Día: {spanish_day}")
        print(f"Hora actual: {current_time}")
        print(f"Salón: {required_classroom}")

        upcoming = []

        for course_doc in db.collection("courses").get():
            course_id = course_doc.id
            course_data = course_doc.to_dict()

            try:
                groups = db.collection("courses").document(course_id).collection("groups").get()
                if groups:
                    for group_doc in groups:
                        for slot in group_doc.to_dict().get("schedule", []):
                            found = next_schedule_opening(
                                slot, spanish_day, english_day,
                                current_time, required_classroom,
                                course_id, course_data,
                            )
                            if found:
                                upcoming.append(found)
                    continue
            except Exception:
                pass

            for slot in course_data.get("schedule", []):
                found = next_schedule_opening(
                    slot, spanish_day, english_day,
                    current_time, required_classroom,
                    course_id, course_data,
                )
                if found:
                    upcoming.append(found)

        if upcoming:
            upcoming.sort(key=lambda item: item["minutes_until_start"])
            closest = upcoming[0]
            print("✔ Próximo curso encontrado:")
            print(f"   Curso: {closest['course_name']}")
            print(f"   Inicia: {closest['start_time']}")
            print(f"   En: {closest['minutes_until_start']} minutos")
            return closest

        print(f"[!] No hay cursos próximos (≤5 min) en {required_classroom}")
        return None

    except Exception as error:
        print(f"[✖] ERROR buscando el próximo curso del salón '{required_classroom}': {error}")
        import traceback
        traceback.print_exc()
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
