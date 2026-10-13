"""Recognition HTTP app. Importing this does not open the camera or wipe events."""
from flask import Flask, jsonify, request

from app.classroom.rooms import note_classroom
from app.http import (
    attendance_api,
    classroom_api,
    enrollment_api,
    event_api,
    recognition_api,
)
from app.http.gate import SIGNATURE_HEADER, TIME_HEADER, signature_matches
from app.http.room import request_classroom


def create_app():
    """Routes only. Boot is what wipes event data and starts the scheduler."""
    app = Flask(__name__)
    app.register_blueprint(recognition_api.blueprint)
    app.register_blueprint(enrollment_api.blueprint)
    app.register_blueprint(attendance_api.blueprint)
    app.register_blueprint(classroom_api.blueprint)
    app.register_blueprint(event_api.blueprint)

    @app.before_request
    def require_page_signature():
        """Reject anything that did not come from the page on this machine."""
        if not signature_matches(
            request.method,
            request.path,
            request.query_string.decode(),
            request.get_data(),
            request_classroom(),
            request.headers.get(TIME_HEADER, ""),
            request.headers.get(SIGNATURE_HEADER, ""),
        ):
            return jsonify({"error": "No se pudo atender el pedido."}), 403
        room = request_classroom()
        if room:
            note_classroom(room)

    return app


app = create_app()
