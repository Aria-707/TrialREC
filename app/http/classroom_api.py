"""Classroom list and the room this browser asked to keep."""
from flask import Blueprint, jsonify, request

from app.attendance.timetable import available_classrooms
from app.classroom.rooms import set_classroom
from app.http.room import request_classroom

blueprint = Blueprint("classroom_api", __name__)


@blueprint.route("/api/classrooms", methods=["GET"])
def api_classrooms():
    """List classrooms found in courses."""
    try:
        classrooms = available_classrooms()
        return jsonify({"success": True, "classrooms": classrooms, "total": len(classrooms)}), 200
    except Exception as error:
        print(f"[✖] ERROR en /api/classrooms al leer los salones de courses: {error}")
        return jsonify({"success": False, "error": "No se pudieron cargar los salones."}), 500


@blueprint.route("/api/set_classroom", methods=["POST"])
def api_set_classroom():
    """Save the classroom when its name is in the course list."""
    try:
        data = request.get_json()
        classroom_name = data.get("classroom", "").strip()
        if not classroom_name:
            return jsonify({"success": False, "error": "Nombre de salón requerido"}), 400
        if classroom_name != request_classroom():
            return jsonify({"success": False, "error": "No se pudo guardar el salón."}), 400

        known_classrooms = available_classrooms()
        if classroom_name not in known_classrooms:
            return jsonify({
                "success": False,
                "error": f"El salón '{classroom_name}' no existe en el sistema",
            }), 404

        saved = set_classroom(classroom_name)
        if saved:
            return jsonify({
                "success": True,
                "classroom": classroom_name,
                "message": f"Salón configurado: {classroom_name}",
            }), 200
        return jsonify({"success": False, "error": "No se pudo guardar el salón."}), 500

    except Exception as error:
        print(f"[✖] ERROR en /api/set_classroom al guardar el salón: {error}")
        return jsonify({"success": False, "error": "No se pudo guardar el salón."}), 500


@blueprint.route("/api/current_classroom", methods=["GET"])
def api_current_classroom():
    """Return the classroom signed on this request."""
    try:
        classroom_name = request_classroom()
        return jsonify({
            "success": True,
            "classroom": classroom_name,
            "configured": classroom_name is not None,
        }), 200
    except Exception as error:
        print(f"[✖] ERROR en /api/current_classroom: {error}")
        return jsonify({"success": False, "error": "No se pudo leer el salón."}), 500
