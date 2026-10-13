"""BFF. Serves the screens and forwards the kiosk API to recognition on localhost."""
import urllib.error
import urllib.request

from flask import Flask, Response, redirect, render_template, request, send_file

from kiosk_gate import (
    COOKIE_NAME,
    dump_classroom_cookie,
    read_classroom_cookie,
    sign_forward,
    token_is_admin,
)
from config import RECOGNITION_HOST, RECOGNITION_PORT

app = Flask(__name__)

RECOGNITION_ORIGIN = f"http://{RECOGNITION_HOST}:{RECOGNITION_PORT}"

COOKIE_PATHS = (
    ("GET", "/api/active_course"),
    ("GET", "/api/current_classroom"),
    ("GET", "/api/event_mode"),
    ("POST", "/recognize"),
    ("POST", "/detect_face"),
    ("POST", "/save_photo"),
    ("POST", "/train"),
    ("POST", "/api/lookup_student"),
    ("POST", "/api/begin_capture"),
    ("POST", "/api/clear_recognitions"),
    ("POST", "/api/event_person"),
    ("POST", "/api/event_link"),
    ("POST", "/api/event_consent"),
)


def _json(body, status):
    return Response(body, status=status, content_type="application/json")


def _classroom_cookie():
    return read_classroom_cookie(request.cookies.get(COOKIE_NAME, ""))


def _require_admin():
    header = request.headers.get("Authorization", "")
    token = header[7:].strip() if header.startswith("Bearer ") else ""
    if not token_is_admin(token):
        return _json('{"error": "Correo o contraseña incorrectos."}', 401)
    return None


def _attach_cookie(response, classroom_name):
    response.set_cookie(
        COOKIE_NAME,
        dump_classroom_cookie(classroom_name),
        httponly=True,
        samesite="Strict",
        path="/",
    )
    return response


@app.route("/")
def index():
    """Main screen. Redirects to setup when this browser has no classroom."""
    classroom_name = _classroom_cookie()
    if not classroom_name:
        print("No hay salón en este navegador. Redirigiendo a /setup")
        return redirect("/setup")
    print(f"Salón de este navegador: {classroom_name}")
    return render_template("index.html")


@app.route("/enroll")
def enroll():
    if not _classroom_cookie():
        return redirect("/setup")
    return render_template("enroll.html")


@app.route("/setup")
def setup():
    """First screen, used to pick the classroom."""
    return render_template("classroom_setup.html")


@app.route("/firebase-config")
def firebase_config_route():
    return send_file("./firebase_config_public.json")


def forward_to_recognition(path, classroom_name):
    """Send this request to the recognition process and return its response."""
    full_path = path
    query = request.query_string.decode()
    if query:
        full_path += "?" + query
    payload = None if request.method == "GET" else request.get_data()
    headers = sign_forward(request.method, path, query, payload or b"", classroom_name)
    if request.content_type:
        headers["Content-Type"] = request.content_type
    outgoing = urllib.request.Request(
        RECOGNITION_ORIGIN + full_path,
        data=payload,
        headers=headers,
        method=request.method,
    )
    try:
        with urllib.request.urlopen(outgoing) as incoming:
            content_type = incoming.headers.get("Content-Type", "application/json")
            return Response(incoming.read(), status=incoming.status, content_type=content_type)
    except urllib.error.HTTPError as error:
        content_type = error.headers.get("Content-Type", "application/json")
        return Response(error.read(), status=error.code, content_type=content_type)
    except urllib.error.URLError as error:
        print(f"El reconocimiento no respondió en {path}: {error.reason}")
        return _json('{"error": "El reconocimiento no respondió"}', 502)


def _proxy(path, method):
    def view():
        classroom_name = _classroom_cookie()
        if not classroom_name:
            return _json('{"error": "Configura un salón para continuar."}', 401)
        return forward_to_recognition(path, classroom_name)

    view.__name__ = "proxy_" + method.lower() + "_" + path.strip("/").replace("/", "_")
    return view


@app.route("/api/classrooms", methods=["GET"])
def api_classrooms():
    rejected = _require_admin()
    if rejected:
        return rejected
    return forward_to_recognition("/api/classrooms", "")


@app.route("/api/set_classroom", methods=["POST"])
def api_set_classroom():
    rejected = _require_admin()
    if rejected:
        return rejected
    data = request.get_json(silent=True) or {}
    classroom_name = str(data.get("classroom") or "").strip()
    if not classroom_name:
        return _json('{"success": false, "error": "Nombre de salón requerido"}', 400)
    response = forward_to_recognition("/api/set_classroom", classroom_name)
    if response.status_code == 200:
        _attach_cookie(response, classroom_name)
    return response


@app.route("/api/event_mode", methods=["POST"])
def api_set_event_mode():
    rejected = _require_admin()
    if rejected:
        return rejected
    classroom_name = _classroom_cookie()
    if not classroom_name:
        return _json('{"ok": false, "error": "Configura un salón para continuar."}', 401)
    return forward_to_recognition("/api/event_mode", classroom_name)


for proxied_method, proxied_path in COOKIE_PATHS:
    proxy_view = _proxy(proxied_path, proxied_method)
    app.add_url_rule(
        proxied_path,
        endpoint=proxy_view.__name__,
        view_func=proxy_view,
        methods=[proxied_method],
    )
