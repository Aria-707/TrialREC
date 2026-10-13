"""Whether the screen should wait or open the camera."""
from datetime import datetime

from flask import Blueprint, jsonify

from app.attendance.timetable import (
    active_course_for_classroom,
    course_name_for,
    event_close_for,
    milliseconds_until_countdown,
    milliseconds_until_window_end,
    next_course,
)
from app.http.room import request_classroom
from app.recognition.session import clear_room

blueprint = Blueprint("attendance_api", __name__)


@blueprint.route("/api/active_course", methods=["GET"])
def api_active_course():
    """Tell the screen whether to wait or open the camera."""
    try:
        classroom_name = request_classroom()
        if not classroom_name:
            return jsonify({
                "course_active": False,
                "classroom_configured": False,
                "message": "No hay salón configurado",
            }), 200

        course_id, start_time = active_course_for_classroom(required_classroom=classroom_name)
        if course_id:
            today = datetime.now().strftime("%Y-%m-%d")
            close_text = event_close_for(classroom_name, course_id, today)
            return jsonify({
                "course_active": True,
                "classroom_configured": True,
                "course_id": course_id,
                "start_time": start_time,
                "classroom": classroom_name,
                "course_name": course_name_for(classroom_name, course_id),
                "milliseconds_until_window_end": milliseconds_until_window_end(
                    start_time, close_text=close_text,
                ),
            }), 200

        upcoming = next_course(classroom_name)
        payload = {
            "course_active": False,
            "classroom_configured": True,
            "classroom": classroom_name,
            "next_course": upcoming,
        }
        if not upcoming:
            wait_ms = milliseconds_until_countdown(classroom_name)
            payload["milliseconds_until_next_check"] = 60000 if wait_ms is None else min(wait_ms, 60000)
        return jsonify(payload), 200

    except Exception as error:
        print(f"[✖] ERROR en /api/active_course al consultar el horario del salón: {error}")
        import traceback
        traceback.print_exc()
        return jsonify({"course_active": False, "error": "No se pudo consultar el horario."}), 500


@blueprint.route("/api/clear_recognitions", methods=["POST"])
def api_clear_recognitions():
    """Clear in-memory hits for this browser's classroom."""
    try:
        clear_room(request_classroom())
        print("\n🧹 REGISTROS LIMPIADOS MANUALMENTE")
        print("   Se reintentará el registro de asistencia")
        return jsonify({"success": True, "message": "Registros limpiados correctamente"}), 200
    except Exception as error:
        print(f"[✖] ERROR en /api/clear_recognitions: {error}")
        return jsonify({"success": False, "error": "No se pudieron limpiar los registros."}), 500
