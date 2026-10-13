"""Attendance writes."""
from datetime import datetime

from app.consent.legal import biometrico_is_true, has_legal_row
from app.firebase.client import db
from app.attendance.timetable import active_course_for_classroom, arrival_from_start


def lookup_student(national_id):
    """Read person/{cedula}. The cédula is the document id."""
    if not str(national_id).isdigit():
        return {"status": "missing"}

    document = db.collection("person").document(str(national_id)).get()
    if not document.exists:
        return {"status": "missing"}

    fields = document.to_dict() or {}
    if fields.get("type") != "Estudiante":
        return {"status": "not_student", "name": fields.get("namePerson") or ""}

    return {
        "status": "student",
        "name": fields.get("namePerson") or "",
        "biometrico": biometrico_is_true(str(national_id)),
        "legal_row": has_legal_row(str(national_id)),
    }


def record_attendance(national_id, course_id=None, class_start=None, classroom_name=None):
    """Change Ausente to Presente when this cédula is already on the class list.

    Returns (recorded, category). The category and the late flag share one clock reading.
    """
    try:
        national_id = str(national_id)
        classroom_name = (classroom_name or "").strip()

        if not course_id:
            if not classroom_name:
                print("[!] NO SE REGISTRA ASISTENCIA: No hay salón configurado")
                return False, None
            course_id, class_start = active_course_for_classroom(required_classroom=classroom_name)

        if not course_id:
            print("[!] NO SE REGISTRA ASISTENCIA: No hay curso activo")
            return False, None

        if not national_id.isdigit():
            print("[!] NO SE REGISTRA ASISTENCIA: el rostro no corresponde a una cédula")
            return False, None

        if not biometrico_is_true(national_id):
            print("[!] NO SE REGISTRA ASISTENCIA: el dato biométrico no está autorizado")
            return False, None

        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M")
        category, is_late = arrival_from_start(class_start, now)

        print("\n=== ACTUALIZANDO ASISTENCIA ===")
        print(f"Fecha: {today}")
        print(f"Hora: {current_time}")
        print(f"Curso: {course_id}")

        student_doc = db.collection("person").document(national_id).get()
        if not student_doc.exists or (student_doc.to_dict() or {}).get("type") != "Estudiante":
            print("[✖] ERROR: No se encontró un estudiante con esa cédula")
            return False, None

        student_id = student_doc.id

        if is_late:
            print("⚠️ TARDE")
        else:
            print("✓ A TIEMPO")

        assistance_ref = db.collection("courses").document(course_id).collection("assistances").document(today)
        assistance_doc = assistance_ref.get()

        if not assistance_doc.exists:
            course_doc = db.collection("courses").document(course_id).get()
            if not course_doc.exists:
                print(f"[✖] ERROR: El curso {course_id} no existe")
                return False, None

            student_ids = course_doc.to_dict().get("estudianteID", [])
            if student_id not in student_ids:
                print("[!] La cédula no está en la clase activa. No se escribe asistencia.")
                return False, None

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
            return False, None

        current_status = existing[student_id].get("estadoAsistencia")
        print(f"Estado actual: {current_status}")

        if current_status == "Ausente":
            assistance_ref.update({
                f"{student_id}.estadoAsistencia": "Presente",
                f"{student_id}.horaRegistro": current_time,
                f"{student_id}.late": is_late,
            })
            print("[✔] ACTUALIZADO: Ausente → Presente")
            print(f"    Hora: {current_time}")
            print(f"    Tarde: {'Sí' if is_late else 'No'}")
            print("=== ACTUALIZACIÓN EXITOSA ===\n")
            return True, category

        print(f"[!] Ya registrado como: {current_status}")
        return True, category

    except Exception as error:
        print(f"[✖] ERROR registrando la asistencia: {error}")
        import traceback
        traceback.print_exc()
        return False, None
