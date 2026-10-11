"""Attendance writes and the demo enrollment path."""
from datetime import datetime

from classroom import current_classroom
from consent import biometrico_is_true
from constants import DEMO_COURSE_ID, DEMO_ID_MAX, DEMO_ID_MIN, DEMO_MODE, LATE_AFTER_MINUTES
from firebase_config import db
from schedule import active_course_for_classroom


def normalize_person_name(name):
    """Uppercase name with accents removed, for matching namePerson. Not a folder name."""
    name = name.strip()
    name = " ".join(name.split())
    replacements = {
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "Ñ": "N", "ñ": "n",
    }
    for original, replacement in replacements.items():
        name = name.replace(original, replacement)
    return name.upper()


def lookup_student(national_id):
    """Read person/{cedula}. The cédula is the document id."""
    if DEMO_MODE:
        return {"status": "demo_unavailable"}
    if not str(national_id).isdigit():
        return {"status": "missing"}

    document = db.collection("person").document(str(national_id)).get()
    if not document.exists:
        return {"status": "missing"}

    fields = document.to_dict() or {}
    if fields.get("type") != "Estudiante":
        return {"status": "not_student"}

    return {
        "status": "student",
        "name": fields.get("namePerson") or "",
        "biometrico": biometrico_is_true(str(national_id)),
    }


def record_attendance(national_id, course_id=None, class_start=None):
    """Change Ausente to Presente when this cédula is already on the class list."""
    try:
        national_id = str(national_id)
        classroom_name = current_classroom()

        if not course_id:
            course_id, class_start = active_course_for_classroom(required_classroom=classroom_name)

        if not course_id:
            print("[!] NO SE REGISTRA ASISTENCIA: No hay curso activo")
            return False

        if not national_id.isdigit():
            print("[!] NO SE REGISTRA ASISTENCIA: el rostro no corresponde a una cédula")
            return False

        if not biometrico_is_true(national_id):
            print("[!] NO SE REGISTRA ASISTENCIA: el dato biométrico no está autorizado")
            return False

        today = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M")

        print("\n=== ACTUALIZANDO ASISTENCIA ===")
        print(f"Fecha: {today}")
        print(f"Hora: {current_time}")
        print(f"Curso: {course_id}")

        student_doc = db.collection("person").document(national_id).get()
        if not student_doc.exists or (student_doc.to_dict() or {}).get("type") != "Estudiante":
            print("[✖] ERROR: No se encontró un estudiante con esa cédula")
            return False

        student_id = student_doc.id

        is_late = False
        if class_start:
            try:
                start_clock = datetime.strptime(class_start, "%H:%M")
                now_clock = datetime.strptime(current_time, "%H:%M")
                difference = (now_clock - start_clock).total_seconds() / 60
                if difference > LATE_AFTER_MINUTES:
                    is_late = True
                    print(f"⚠️ TARDE: {int(difference)} minutos")
                else:
                    print("✓ A TIEMPO")
            except Exception:
                pass

        assistance_ref = db.collection("courses").document(course_id).collection("assistances").document(today)
        assistance_doc = assistance_ref.get()

        if not assistance_doc.exists:
            course_doc = db.collection("courses").document(course_id).get()
            if not course_doc.exists:
                print(f"[✖] ERROR: El curso {course_id} no existe")
                return False

            student_ids = course_doc.to_dict().get("estudianteID", [])
            if student_id not in student_ids and not DEMO_MODE:
                print("[!] La cédula no está en la clase activa. No se escribe asistencia.")
                return False

            print("[!] Documento no existe - Creando automáticamente...")
            initial_data = {}
            for enrolled_id in student_ids:
                initial_data[enrolled_id] = {
                    "estadoAsistencia": "Ausente",
                    "horaRegistro": None,
                    "late": False,
                }

            assistance_ref.set(initial_data)
            print(f"[✔] Documento creado con {len(student_ids)} estudiantes")
            assistance_doc = assistance_ref.get()

        existing = assistance_doc.to_dict() or {}

        if student_id not in existing:
            print("[!] La cédula no está en la clase activa. No se escribe asistencia.")
            if not DEMO_MODE:
                return False

            person_courses = (student_doc.to_dict() or {}).get("courses", [])
            if DEMO_COURSE_ID not in person_courses:
                return False

            existing[student_id] = {
                "estadoAsistencia": "Ausente",
                "horaRegistro": None,
                "late": False,
            }
            assistance_ref.update({student_id: existing[student_id]})

        current_status = existing[student_id].get("estadoAsistencia")
        print(f"Estado actual: {current_status}")

        if current_status == "Ausente":
            assistance_ref.update({
                student_id: {
                    "estadoAsistencia": "Presente",
                    "horaRegistro": current_time,
                    "late": is_late,
                }
            })
            print("[✔] ACTUALIZADO: Ausente → Presente")
            print(f"    Hora: {current_time}")
            print(f"    Tarde: {'Sí' if is_late else 'No'}")
            print("=== ACTUALIZACIÓN EXITOSA ===\n")
            return True

        print(f"[!] Ya registrado como: {current_status}")
        return True

    except Exception as error:
        print(f"[✖] ERROR registrando la asistencia: {error}")
        import traceback
        traceback.print_exc()
        return False


def enroll_demo_student(student_name):
    """Demo only: create a person with a random id and course 00000."""
    if not DEMO_MODE:
        print("El modo de prueba está apagado. No se crea un documento en person.")
        return None
    try:
        import random

        normalized_name = normalize_person_name(student_name)

        print(f"\n{'='*60}")
        print("🔥 REGISTRANDO ESTUDIANTE EN FIREBASE (TEMPORAL)")
        print(f"{'='*60}")
        print(f"Nombre original: '{student_name}'")
        print(f"Nombre normalizado: '{normalized_name}'")

        people = db.collection("person")
        query = people.where("type", "==", "Estudiante").get()

        for doc in query:
            stored_name = normalize_person_name(doc.to_dict().get("namePerson", ""))
            if stored_name == normalized_name:
                print(f"[!] Estudiante ya existe en Firebase con ID: {doc.id}")
                print(f"{'='*60}\n")
                return doc.id

        new_id = str(random.randint(DEMO_ID_MIN, DEMO_ID_MAX))
        while people.document(new_id).get().exists:
            new_id = str(random.randint(DEMO_ID_MIN, DEMO_ID_MAX))

        people.document(new_id).set({
            "namePerson": normalized_name,
            "type": "Estudiante",
            "courses": [DEMO_COURSE_ID],
        })

        print("✅ ESTUDIANTE REGISTRADO EXITOSAMENTE")
        print(f"   ID: {new_id}")
        print(f"   Nombre: {normalized_name}")
        print("   Tipo: Estudiante")
        print(f"   Cursos: ['{DEMO_COURSE_ID}']")
        print(f"{'='*60}\n")
        return new_id

    except Exception as error:
        print(f"[✖] ERROR registrando al estudiante de prueba '{student_name}' en Firebase: {error}")
        import traceback
        traceback.print_exc()
        return None
