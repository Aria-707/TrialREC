"""Create the day's attendance documents shortly before class starts."""
import threading
import time
from datetime import datetime

from constants import ENGLISH_TO_SPANISH_WEEKDAYS, SCHEDULER_LOOKAHEAD_MINUTES
from firebase_config import db

initialized_documents = set()


def courses_about_to_start(configured_classroom):
    """Courses in this classroom whose start is within the next 6 minutes."""
    try:
        now = datetime.now()
        english_day = now.strftime("%A")
        spanish_day = ENGLISH_TO_SPANISH_WEEKDAYS.get(english_day, english_day)
        current_time = now.strftime("%H:%M")
        today = now.strftime("%Y-%m-%d")

        to_create = []

        print(f"   📅 Día: {spanish_day} ({english_day})")
        print(f"   🕐 Hora actual: {current_time}")
        print(f"   🎯 Buscando clases que inicien en los próximos {SCHEDULER_LOOKAHEAD_MINUTES} minutos")
        print(f"   🏫 Salón configurado: {configured_classroom}")

        if not configured_classroom:
            print("   [!] No hay salón configurado - saltando verificación")
            return []

        for course_doc in db.collection("courses").get():
            course_id = course_doc.id
            course_data = course_doc.to_dict()

            try:
                groups = db.collection("courses").document(course_id).collection("groups").get()
                if groups:
                    for group_doc in groups:
                        for slot in group_doc.to_dict().get("schedule", []):
                            start_text = schedule_starts_within(
                                slot, spanish_day, english_day, now, configured_classroom
                            )
                            if start_text:
                                to_create.append((course_id, today, start_text))
                                course_name = course_data.get("nameCourse", "Sin nombre")
                                print(f"   ✅ Encontrado: {course_name} ({course_id}) - Inicia a las {start_text}")
                    continue
            except Exception:
                pass

            for slot in course_data.get("schedule", []):
                start_text = schedule_starts_within(
                    slot, spanish_day, english_day, now, configured_classroom
                )
                if start_text:
                    to_create.append((course_id, today, start_text))
                    course_name = course_data.get("nameCourse", "Sin nombre")
                    print(f"   ✅ Encontrado: {course_name} ({course_id}) - Inicia a las {start_text}")

        if not to_create:
            print(f"   ℹ️ No hay cursos iniciando en los próximos {SCHEDULER_LOOKAHEAD_MINUTES} minutos en {configured_classroom}")

        return to_create

    except Exception as error:
        print(f"   ❌ ERROR obteniendo los cursos que están por iniciar: {error}")
        return []


def schedule_starts_within(slot, spanish_day, english_day, now, required_classroom):
    """Return the start time when the class in this classroom begins within the lookahead."""
    try:
        slot_day = slot.get("day", "")
        start_text = slot.get("iniTime", "00:00")
        classroom_name = slot.get("classroom", "").strip()

        if classroom_name != required_classroom:
            return None
        if slot_day != spanish_day and slot_day != english_day:
            return None

        start_clock = datetime.strptime(start_text, "%H:%M").replace(
            year=now.year, month=now.month, day=now.day
        )
        now_clock = now.replace(second=0, microsecond=0)
        minutes_until_start = (start_clock - now_clock).total_seconds() / 60
        if 0 <= minutes_until_start <= SCHEDULER_LOOKAHEAD_MINUTES:
            return start_text
        return None
    except Exception:
        return None


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


def scheduled_task(classroom_callback):
    """Sleep 60 seconds, then create today's list for classes starting within 6 minutes."""
    print("\n" + "="*70)
    print("🚀 SCHEDULER DE ASISTENCIA INICIADO")
    print("="*70)
    print("   ⏰ Verificación: cada 60 segundos")
    print(f"   📋 Inicialización: clases que empiezan en los próximos {SCHEDULER_LOOKAHEAD_MINUTES} minutos")
    print("   🧹 Limpieza: Diariamente a la medianoche")
    print("="*70 + "\n")

    last_cleanup = datetime.now().date()

    while True:
        try:
            now = datetime.now()
            today = now.date()
            if today > last_cleanup:
                clear_old_initialized_documents()
                last_cleanup = today

            print(f"\n🔔 [{now.strftime('%H:%M:%S')}] Verificando clases por iniciar...")
            classroom_name = classroom_callback()
            if classroom_name:
                courses = courses_about_to_start(classroom_name)
                for course_id, day, _start_text in courses:
                    create_attendance_document(course_id, day)
                if not courses:
                    print("   ℹ️ No hay cursos a inicializar en este momento")
            else:
                print("   ⚠️ No hay salón configurado - saltando verificación")
            print("✅ Verificación completada. Próxima verificación en ~60 segundos.\n")

        except Exception as error:
            print(f"\n❌ ERROR en la tarea programada del scheduler de asistencia: {error}")
            import traceback
            traceback.print_exc()

        time.sleep(60)


def start_scheduler(classroom_callback):
    """Start the scheduler thread. The Flask reloader can start it twice."""
    thread = threading.Thread(
        target=scheduled_task,
        args=(classroom_callback,),
        daemon=True,
    )
    thread.start()
    return thread
