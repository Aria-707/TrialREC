"""Event enrollment. One switch for the whole server."""
from flask import Blueprint, jsonify, request

from app.events import mode as event_mode
from app.http.national_id import national_id_from_request
from app.http.room import request_classroom

blueprint = Blueprint("event_api", __name__)


@blueprint.route("/api/event_mode", methods=["GET"])
def api_event_mode():
    """Whether this process is enrolling for an event."""
    return jsonify(event_mode.status()), 200


@blueprint.route("/api/event_mode", methods=["POST"])
def api_set_event_mode():
    """Turn event enrollment on or off. Does not write `.env`."""
    data = request.get_json() or {}
    if data.get("enabled") is True:
        ok, message = event_mode.enable(
            data.get("name", ""),
            data.get("ini_time", ""),
            data.get("end_time", ""),
            request_classroom(),
        )
    else:
        ok, message = event_mode.disable()
    if not ok:
        return jsonify({"ok": False, "error": message}), 400
    payload = event_mode.status()
    payload["ok"] = True
    return jsonify(payload), 200


@blueprint.route("/api/event_person", methods=["POST"])
def api_event_person():
    """Create a person for this event. An existing cédula is not overwritten."""
    data = request.get_json() or {}
    national_id = national_id_from_request(data)
    if not national_id:
        return jsonify({"ok": False, "error": "La cédula solo puede tener números"}), 400
    ok, message = event_mode.create_person(
        national_id,
        data.get("name", ""),
        bool(data.get("portal")),
    )
    if not ok:
        return jsonify({"ok": False, "error": message}), 400
    return jsonify({"ok": True}), 200


@blueprint.route("/api/event_consent", methods=["POST"])
def api_event_consent():
    """Temporary biometric acceptance for someone whose current row says false."""
    data = request.get_json() or {}
    national_id = national_id_from_request(data)
    if not national_id:
        return jsonify({"ok": False, "error": "La cédula solo puede tener números"}), 400
    ok, message = event_mode.grant_event_consent(national_id)
    if not ok:
        return jsonify({"ok": False, "error": message}), 400
    return jsonify({"ok": True}), 200


@blueprint.route("/api/event_link", methods=["POST"])
def api_event_link():
    """Link an existing cédula to today's event without changing their vector."""
    data = request.get_json() or {}
    national_id = national_id_from_request(data)
    if not national_id:
        return jsonify({"ok": False, "error": "La cédula solo puede tener números"}), 400
    ok, message = event_mode.link_existing(national_id, data.get("name", ""))
    if not ok:
        return jsonify({"ok": False, "error": message}), 400
    return jsonify({"ok": True}), 200
