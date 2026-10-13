"""Create the day's attendance documents shortly before class starts."""
import os
import threading
from datetime import datetime, timedelta

from config import SCHEDULER_LOOKAHEAD_MINUTES
from firebase_config import db
from schedule import event_finish_moments, upcoming_starts

# One extra minute so the read and the write finish before the 6-minute mark.
WAKE_BEFORE_MINUTES = SCHEDULER_LOOKAHEAD_MINUTES + 1
_wake = threading.Event()

initialized_documents = set()


def create_attendance_document(course_id, day):
    """Create the assistance document with every enrolled student marked Ausente."""
    try:
        print(f"\n{'='*70}")
        print("📋 INICIALIZANDO DOCUMENTO DE ASISTENCIA")
        print(f"{'='*70}")
        print(f"   Curso: {course_id}")
        print(f"   Fecha: {day}")

        document_key = f"{course_id}_{day}"
        if document_key in initialized_documents:
            print("   [!] Ya fue inicializado previamente en esta sesión")
            return True

        assistance_ref = db.collection("courses").document(course_id).collection("assistances").document(day)
        assistance_doc = assistance_ref.get()
        if assistance_doc.exists:
            print("   [!] El documento ya existe en Firebase")
            initialized_documents.add(document_key)
            return True

        course_doc = db.collection("courses").document(course_id).get()
        if not course_doc.exists:
            print(f"   ❌ ERROR: El curso {course_id} no existe")
            return False

        student_ids = course_doc.to_dict().get("estudianteID", [])
        if not student_ids:
            print("   ⚠️ ADVERTENCIA: No hay estudiantes inscritos en el curso")
            print("   ℹ️ Creando documento vacío")
            assistance_ref.set({})
            initialized_documents.add(document_key)
            return True

        print(f"   📚 Estudiantes inscritos: {len(student_ids)}")
        attendance_data = {}
        for student_id in student_ids:
            attendance_data[student_id] = {
                "estadoAsistencia": "Ausente",
                "horaRegistro": None,
                "late": False,
            }

        assistance_ref.set(attendance_data)
        initialized_documents.add(document_key)

        print("   ✅ Documento creado exitosamente")
        print(f"   📊 Total de estudiantes: {len(student_ids)}")
        print("   📝 Estado inicial: Ausente")
        print(f"{'='*70}\n")
        return True

    except Exception as error:
        print(f"   ❌ ERROR inicializando el documento de asistencia del curso {course_id}: {error}")
        import traceback
        traceback.print_exc()
        return False


def clear_old_initialized_documents():
    """Drop in-memory keys from previous days. Does not clear the recognition set."""
    global initialized_documents
    today = datetime.now().strftime("%Y-%m-%d")
    initialized_documents = {doc for doc in initialized_documents if doc.endswith(today)}
    print("\n🧹 Limpieza de registros antiguos completada")
    print(f"   Documentos en memoria: {len(initialized_documents)}")


def interrupt_sleep():
    """Wake the scheduler so a classroom change is calculated immediately."""
    _wake.set()


def _sleep_until_next(seconds):
    """Sleep until the timeout or until interrupt_sleep. Returns True if interrupted."""
    _wake.wait(timeout=max(0, seconds))
    interrupted = _wake.is_set()
    _wake.clear()
    return interrupted


def _watch_classroom(classroom_name, now, serving):
    """Create due lists for one classroom. Returns (ok, next_wake or None, message)."""
    ends = event_finish_moments(classroom_name, refresh=True)
    due_ids = [course_id for course_id, when in ends if when <= now]
    future_ends = [when for _, when in ends if when > now]
    if due_ids and serving:
        import event_mode
        closed, message = event_mode.end_event(due_ids)
        if not closed:
            return False, None, message

    starts = upcoming_starts(classroom_name, now, refresh=False)
    now_minute = now.replace(second=0, microsecond=0)
    due = []
    later = []
    for course_id, when in starts:
        minutes_until = (when - now_minute).total_seconds() / 60
        if 0 <= minutes_until <= WAKE_BEFORE_MINUTES:
            due.append((course_id, when))
        elif minutes_until > WAKE_BEFORE_MINUTES:
            later.append((course_id, when))

    created = set()
    for course_id, when in due:
        if course_id in created:
            continue
        created.add(course_id)
        create_attendance_document(course_id, when.strftime("%Y-%m-%d"))

    if due:
        after_these = max(when for _, when in due) + timedelta(minutes=1)
        following = upcoming_starts(classroom_name, after_these, refresh=False)
    else:
        following = later

    wake_candidates = []
    if following:
        wake_candidates.append(following[0][1] - timedelta(minutes=WAKE_BEFORE_MINUTES))
    if future_ends:
        wake_candidates.append(future_ends[0])
    return True, (min(wake_candidates) if wake_candidates else None), ""


def scheduled_task(classrooms_callback):
    """Sleep until 7 minutes before the next class of any watched classroom."""
    print("\n" + "="*70)
    print("🚀 SCHEDULER DE ASISTENCIA INICIADO")
    print("="*70)
    print(f"   ⏰ Duerme hasta {WAKE_BEFORE_MINUTES} minutos antes de la próxima clase")
    print("   📋 Al despertar confirma el horario y crea la lista de ausentes")
    print("   🧹 Limpieza: al despertar, si cambió el día")
    print("="*70 + "\n")

    last_cleanup = datetime.now().date()

    while True:
        try:
            now = datetime.now()
            if now.date() > last_cleanup:
                clear_old_initialized_documents()
                last_cleanup = now.date()

            classrooms = [name for name in (classrooms_callback() or []) if name]
            if not classrooms:
                print("   ⚠️ No hay salón configurado. Reintento en 60 segundos.")
                if _sleep_until_next(60):
                    continue
                continue

            print(f"\n🔔 [{now.strftime('%Y-%m-%d %H:%M:%S')}] Confirmando horario...")
            serving = os.environ.get("WERKZEUG_RUN_MAIN") == "true" or os.environ.get("KIOSK_DEBUG", "").strip().lower() not in ("1", "true", "si", "sí", "yes")
            wake_candidates = []
            failed = False
            for classroom_name in classrooms:
                print(f"   Salón: {classroom_name}")
                ok, wake_at_room, message = _watch_classroom(classroom_name, now, serving)
                if not ok:
                    print(f"   ⚠️ {message} Reintento en 60 segundos.")
                    failed = True
                    break
                if wake_at_room is not None:
                    wake_candidates.append(wake_at_room)
            if failed:
                _sleep_until_next(60)
                continue
            if not wake_candidates:
                print("   ℹ️ No hay una clase posterior. Reintento en 60 segundos.")
                _sleep_until_next(60)
                continue

            wake_at = min(wake_candidates)
            seconds = (wake_at - datetime.now()).total_seconds()
            if seconds < 1:
                seconds = 1
            print(f"   😴 Próximo despertar: {wake_at.strftime('%Y-%m-%d %H:%M')} ({int(seconds)} s)")
            _sleep_until_next(seconds)

        except Exception as error:
            print(f"\n❌ ERROR en la tarea programada del scheduler de asistencia: {error}")
            import traceback
            traceback.print_exc()
            _sleep_until_next(60)


def start_scheduler(classroom_callback):
    """Start the scheduler thread. The Flask reloader can start it twice."""
    import classroom
    from schedule import on_schedule_change
    classroom.on_classroom_change(lambda _name: interrupt_sleep())
    on_schedule_change(interrupt_sleep)
    thread = threading.Thread(
        target=scheduled_task,
        args=(classroom_callback,),
        daemon=True,
    )
    thread.start()
    return thread
