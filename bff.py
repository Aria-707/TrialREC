"""BFF. Serves the screens and forwards the kiosk API to recognition on localhost."""
import urllib.error
import urllib.request

from flask import Flask, Response, redirect, render_template, request, send_file
from flask_cors import CORS

from classroom import load_saved_classroom
from constants import RECOGNITION_HOST, RECOGNITION_PORT

app = Flask(__name__)
CORS(app)

RECOGNITION_ORIGIN = f"http://{RECOGNITION_HOST}:{RECOGNITION_PORT}"

GET_PATHS = (
    "/test_course",
    "/api/active_course",
    "/api/classrooms",
    "/api/current_classroom",
)
POST_PATHS = (
    "/recognize",
    "/detect_face",
    "/save_photo",
    "/train",
    "/api/set_classroom",
    "/api/clear_recognitions",
)


@app.route("/")
def index():
    """Main screen. Redirects to setup when no classroom is saved."""
    classroom_name = load_saved_classroom()
    if not classroom_name:
        print("⚠️ No hay salón configurado. Redirigiendo a /setup")
        return redirect("/setup")
    print(f"✔ Salón configurado: {classroom_name}")
    return render_template("index.html")


@app.route("/enroll")
def enroll():
    return render_template("enroll.html")


@app.route("/setup")
def setup():
    """First screen, used to pick the classroom."""
    return render_template("classroom_setup.html")


@app.route("/firebase-config")
def firebase_config_route():
    return send_file("./firebase_config_public.json")


def forward_to_recognition(path):
    """Send this request to the recognition process and return its response."""
    url = RECOGNITION_ORIGIN + path
    if request.query_string:
        url += "?" + request.query_string.decode()
    payload = None if request.method == "GET" else request.get_data()
    headers = {}
    if request.content_type:
        headers["Content-Type"] = request.content_type
    outgoing = urllib.request.Request(url, data=payload, headers=headers, method=request.method)
    try:
        with urllib.request.urlopen(outgoing) as incoming:
            content_type = incoming.headers.get("Content-Type", "application/json")
            return Response(incoming.read(), status=incoming.status, content_type=content_type)
    except urllib.error.HTTPError as error:
        content_type = error.headers.get("Content-Type", "application/json")
        return Response(error.read(), status=error.code, content_type=content_type)
    except urllib.error.URLError as error:
        print(f"✖ El reconocimiento no respondió en {path}: {error.reason}")
        return Response(
            '{"error": "El reconocimiento no respondió"}',
            status=502,
            content_type="application/json",
        )


def _proxy(path):
    def view():
        return forward_to_recognition(path)

    view.__name__ = "proxy_" + path.strip("/").replace("/", "_")
    return view


for proxied_path in GET_PATHS:
    proxy_view = _proxy(proxied_path)
    app.add_url_rule(proxied_path, endpoint=proxy_view.__name__, view_func=proxy_view, methods=["GET"])

for proxied_path in POST_PATHS:
    proxy_view = _proxy(proxied_path)
    app.add_url_rule(proxied_path, endpoint=proxy_view.__name__, view_func=proxy_view, methods=["POST"])
