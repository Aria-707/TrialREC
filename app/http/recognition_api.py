"""Live recognition. The classroom comes from the signed request, not the body."""
import base64
import re
import time

import cv2
import numpy as np
from flask import Blueprint, jsonify, request

from app.attendance.record import lookup_student, record_attendance
from app.attendance.timetable import active_course_for_classroom, class_roster
from app.consent.legal import biometrico_is_true, sweep_before_class_frame
from app.http.room import request_classroom
from app.recognition.face import analyze_frame, match_enrolled, recognition_seconds
from app.recognition.session import forget_recognitions_from_other_days, recognized, started
from app.security.audit import log_event

blueprint = Blueprint("recognition_api", __name__)


@blueprint.route("/recognize", methods=["POST"])
def recognize():
    try:
        forget_recognitions_from_other_days()
        classroom_name = request_classroom()
        if not classroom_name:
            return jsonify({
                "status": "error",
                "message": "No hay salón configurado. Configura el salón primero.",
            }), 400

        recognized_students = recognized(classroom_name)
        recognition_started_at = started(classroom_name)

        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"status": "error", "message": "No se recibió imagen"}), 400

        image_data = re.sub(r"^data:image/.+;base64,", "", data["image"])
        image_bytes = base64.b64decode(image_data)
        frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"status": "error", "message": "Imagen inválida"}), 400

        found = analyze_frame(frame)
        if found["status"] in ("no_face", "unavailable"):
            return jsonify({"status": "no_face"})
        if found["status"] == "several":
            return jsonify({"status": "several_faces"})

        box = found["box"]
        course_id, start_time = active_course_for_classroom(required_classroom=classroom_name)
        if course_id:
            sweep_before_class_frame(course_id)

        national_id = None
        cosine = None
        kind = "none"
        if course_id:
            national_id, cosine, kind = match_enrolled(found["feature"])
            roster = {str(person_id) for person_id in class_roster(course_id)}
            if kind == "match" and national_id not in roster:
                national_id = None
                kind = "unknown"
        else:
            print(f"[!] Rostro en {classroom_name}, sin curso activo")
        shown = 0.0 if cosine is None else float(cosine) * 100

        if kind == "match" and national_id:
            student_name = national_id
            if national_id:
                person = lookup_student(national_id)
                if person.get("status") == "student" and person.get("name"):
                    student_name = person["name"]

            if student_name not in recognition_started_at:
                recognition_started_at[student_name] = time.time()
            elif time.time() - recognition_started_at[student_name] >= recognition_seconds:
                if student_name not in recognized_students and course_id and national_id:
                    recorded = False
                    category = None
                    if biometrico_is_true(national_id):
                        recorded, category = record_attendance(
                            national_id, course_id, start_time, classroom_name,
                        )
                        if recorded:
                            recognized_students.add(student_name)
                            log_event(
                                "RECONOCIMIENTO_FACIAL",
                                "Asistencia registrada mediante reconocimiento facial",
                                user=student_name,
                                extra={"curso": course_id},
                            )
                        else:
                            recognized_students.add(student_name)
                    if recorded:
                        return jsonify({
                            "status": "recognized",
                            "student": student_name,
                            "confidence": shown,
                            "box": box,
                            "classroom": classroom_name,
                            "recorded": True,
                            "arrival_category": category,
                        })
                    return jsonify({
                        "status": "recognized",
                        "student": student_name,
                        "confidence": shown,
                        "box": box,
                        "classroom": classroom_name,
                        "recorded": False,
                    })

            return jsonify({
                "status": "recognized",
                "student": student_name,
                "confidence": shown,
                "box": box,
                "classroom": classroom_name,
            })

        return jsonify({
            "status": "unknown",
            "confidence": shown,
            "box": box,
        })
    except Exception as error:
        print(f"Error en /recognize al procesar el fotograma: {error}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": "No se pudo reconocer."}), 500
