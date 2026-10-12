"""HTTP routes of the recognition process. The browser reaches them through the BFF."""
import base64
import os
import re
import time
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, send_file
from flask_cors import CORS

from attendance import lookup_student, record_attendance
from audit import log_event
from classroom import current_classroom, load_saved_classroom, set_classroom
from consent import biometrico_is_true, has_legal_row, record_acceptance, sweep_before_class_frame
from encryption import encrypt_file
import event_mode
from face import (
    analyze_frame,
    begin_capture,
    discard_capture,
    match_enrolled,
    photo_dir,
    recognition_seconds,
    clear_capture,
    stored_vector,
    train_incrementally,
)
from firebase_config import db
from schedule import (
    active_course_for_classroom,
    available_classrooms,
    class_roster,
    course_name_for,
    event_close_for,
    milliseconds_until_countdown,
    milliseconds_until_window_end,
    next_course,
)
from vector_store import commit_event_vector, commit_vector_copy

app = Flask(__name__)
CORS(app)

recognized_students = set()
recognition_started_at = {}
recognition_day = None
previous_classroom = None


def forget_recognitions_from_other_days():
    """Drop in-memory hits from a previous date. Today's document stays the source of truth."""
    global recognition_day
    today = datetime.now().strftime("%Y-%m-%d")
    if recognition_day == today:
        return
    recognized_students.clear()
    recognition_started_at.clear()
    recognition_day = today


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


@app.route("/recognize", methods=["POST"])
def recognize():
    """Live recognition. The classroom comes from memory, not from the request body."""
    global recognized_students, recognition_started_at, previous_classroom

    try:
        forget_recognitions_from_other_days()
        classroom_name = current_classroom()
        if not classroom_name:
            return jsonify({
                "status": "error",
                "message": "No hay salón configurado. Configura el salón primero.",
            }), 400

        if previous_classroom is not None and previous_classroom != classroom_name:
            print("\n🔄 CAMBIO DE SALÓN DETECTADO:")
            print(f"   Anterior: '{previous_classroom}'")
            print(f"   Nuevo: '{classroom_name}'")
            print("   Limpiando registros previos...")
            recognized_students.clear()
            recognition_started_at.clear()
            print("   ✅ Registros limpiados - se reintentará asistencia\n")

        previous_classroom = classroom_name

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
                        recorded, category = record_attendance(national_id, course_id, start_time)
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


@app.route("/detect_face", methods=["POST"])
def detect_face_route():
    """Tell the enrollment screen whether a face is in the frame."""
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"face_detected": False}), 200

        image_data = re.sub(r"^data:image/.+;base64,", "", data["image"])
        image_bytes = base64.b64decode(image_data)
        frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"face_detected": False}), 200

        found = analyze_frame(frame)
        if found["status"] == "one":
            return jsonify({
                "face_detected": True,
                "box": found["box"],
                "method": "yunet",
            }), 200
        return jsonify({"face_detected": False}), 200

    except Exception as error:
        print(f"Error en /detect_face al revisar el fotograma de alta: {error}")
        return jsonify({"face_detected": False, "error": "No se pudo revisar el rostro."}), 200


def _national_id_from_request(data):
    """The enrollment id is the cédula, digits only."""
    if not data:
        return None
    national_id = str(data.get("national_id", "")).strip()
    if national_id.isdigit():
        return national_id
    return None


@app.route("/api/lookup_student", methods=["POST"])
def api_lookup_student():
    """Confirm a cédula before any photo is stored."""
    data = request.get_json() or {}
    national_id = _national_id_from_request(data)
    if not national_id:
        return jsonify({"status": "missing"}), 200
    return jsonify(lookup_student(national_id)), 200


@app.route("/api/begin_capture", methods=["POST"])
def api_begin_capture():
    """Park older photos so this capture replaces them."""
    data = request.get_json() or {}
    national_id = _national_id_from_request(data)
    if not national_id:
        return jsonify({"ok": False, "error": "La cédula solo puede tener números"}), 400

    person = lookup_student(national_id)
    if person.get("status") != "student":
        return jsonify({"ok": False, "error": "Esta cédula no sigue el alta"}), 400
    if not person.get("legal_row"):
        return jsonify({
            "ok": False,
            "error": "Un administrador tiene que registrarte antes de este alta.",
        }), 400

    begin_capture(national_id)
    return jsonify({"ok": True}), 200


@app.route("/save_photo", methods=["POST"])
def save_photo():
    """Save one enrollment photo under data/{cedula}."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"ok": False, "error": "No se recibió data"}), 400

        national_id = _national_id_from_request(data)
        photo_b64 = data.get("photo", "")

        if not national_id:
            return jsonify({"ok": False, "error": "La cédula solo puede tener números"}), 400
        if not photo_b64:
            return jsonify({"ok": False, "error": "Foto requerida"}), 400

        person = lookup_student(national_id)
        if person.get("status") != "student":
            return jsonify({"ok": False, "error": "Esta cédula no sigue el alta"}), 400

        folder_name = national_id
        print("\n💾 Guardando foto de alta")

        person_path = os.path.join(photo_dir, folder_name)
        try:
            os.makedirs(person_path, exist_ok=True)
        except Exception as error:
            print(f"❌ Error creando la carpeta {person_path}: {error}")
            return jsonify({"ok": False, "error": "No se pudo guardar la foto."}), 500

        try:
            if "," in photo_b64:
                _header, encoded = photo_b64.split(",", 1)
            else:
                encoded = photo_b64
            image = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return jsonify({"ok": False, "error": "Imagen inválida"}), 400
        except Exception as error:
            print(f"❌ Error decodificando la foto de alta: {error}")
            return jsonify({"ok": False, "error": "No se pudo guardar la foto."}), 500

        timestamp = int(time.time() * 1000000)
        file_path = os.path.join(person_path, f"face_{timestamp}.jpg")

        try:
            found = analyze_frame(image, aligned=True)
            if found["status"] != "one":
                if found["status"] == "several":
                    print("Más de un rostro en el fotograma de alta. No se guarda.")
                else:
                    print("Sin rostro en el fotograma de alta. No se guarda.")
                return jsonify({"ok": False, "error": "No se detectó un rostro"}), 200

            capture_type = "aligned"
            method = "yunet"
            success = cv2.imwrite(file_path, found["aligned"])
            if not success or not os.path.exists(file_path):
                return jsonify({"ok": False, "error": "No se pudo guardar la foto."}), 500

            file_size = os.path.getsize(file_path)
            print(f"✅ Foto guardada: {file_size} bytes ({capture_type})")
            print("🔒 Encriptando imagen...")
            if encrypt_file(file_path):
                print("✔ Imagen encriptada exitosamente")
            else:
                print(f"⚠️ No se pudo encriptar la imagen {file_path}")

            return jsonify({
                "ok": True,
                "path": file_path,
                "capture_type": capture_type,
                "method": method,
                "size": file_size,
            }), 200
        except Exception as error:
            print(f"❌ Error guardando la foto de alta: {error}")
            import traceback
            traceback.print_exc()
            return jsonify({"ok": False, "error": "No se pudo guardar la foto."}), 500

    except Exception as error:
        print(f"❌ ERROR GENERAL guardando la foto de alta: {error}")
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": "No se pudo guardar la foto."}), 500


@app.route("/train", methods=["POST"])
def train():
    """Store one SFace vector from data/{cedula}, then delete those photos."""
    national_id = None
    try:
        data = request.get_json() or {}
        national_id = _national_id_from_request(data)
        if not national_id:
            return jsonify({"success": False, "error": "La cédula solo puede tener números"}), 400

        person = lookup_student(national_id)
        if person.get("status") != "student":
            discard_capture(national_id)
            return jsonify({"success": False, "error": "Esta cédula no sigue el alta"}), 400

        already_consented = biometrico_is_true(national_id)
        if not already_consented and not has_legal_row(national_id):
            discard_capture(national_id)
            return jsonify({
                "success": False,
                "error": "Un administrador tiene que registrarte antes de este alta.",
            }), 400

        person_path = os.path.join(photo_dir, national_id)

        print(f"\n{'='*60}")
        print("🤖 ENTRENANDO MODELO")
        print(f"{'='*60}")

        if not os.path.exists(person_path):
            return jsonify({
                "success": False,
                "error": "No se encontró la carpeta de la cédula",
            }), 404

        files = [name for name in os.listdir(person_path) if name.endswith(".jpg")]
        if len(files) == 0:
            discard_capture(national_id)
            return jsonify({"success": False, "error": "No hay imágenes para entrenar"}), 400

        print(f"Imágenes encontradas: {len(files)}")
        image_files = [os.path.join(person_path, name) for name in files]
        trained = train_incrementally({national_id: image_files})

        if not trained:
            discard_capture(national_id)
            return jsonify({"success": False, "error": "Falló el entrenamiento"}), 500

        values = stored_vector(national_id)
        if event_mode.enabled():
            if values is None or not commit_event_vector(national_id, values):
                return jsonify({
                    "success": False,
                    "error": "No se pudo guardar el vector. Las fotos siguen en el equipo.",
                }), 500
        else:
            if not already_consented and not record_acceptance(national_id):
                return jsonify({
                    "success": False,
                    "error": "Un administrador tiene que registrarte antes de este alta.",
                }), 400
            if values is None or not commit_vector_copy(national_id, values):
                return jsonify({
                    "success": False,
                    "error": "No se pudo guardar el vector. Las fotos siguen en el equipo.",
                }), 500

        clear_capture(national_id)

        print("✅ ENTRENAMIENTO COMPLETADO")
        print(f"   • Imágenes usadas: {len(files)}")
        print("   • Vector guardado en Firebase. Fotos borradas.")
        print(f"{'='*60}\n")

        log_event(
            "REGISTRO_DATOS_BIOMETRICOS",
            "Alta de datos biométricos",
            user=national_id,
            extra={"imagenes": len(image_files)},
        )
        return jsonify({
            "success": True,
            "message": f"Modelo entrenado con {len(image_files)} imágenes",
            "images_trained": len(image_files),
        }), 200

    except Exception as error:
        print(f"✖ ERROR EN ENTRENAMIENTO: {error}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": "No se pudo completar el registro."}), 500


@app.route("/test_course")
def test_course():
    """Dump the active course for this classroom. Still published."""
    classroom_name = current_classroom()
    if not classroom_name:
        return jsonify({
            "success": False,
            "message": "No hay salón configurado",
            "classroom": None,
        })

    course_id, start_time = active_course_for_classroom(required_classroom=classroom_name)
    if not course_id:
        return jsonify({
            "success": False,
            "message": f"No hay curso activo en salón '{classroom_name}'",
            "course_id": None,
            "start_time": None,
            "classroom": classroom_name,
        })

    course_doc = db.collection("courses").document(course_id).get()
    if course_doc.exists:
        return jsonify({
            "success": True,
            "course_id": course_id,
            "start_time": start_time,
            "classroom": classroom_name,
            "course": course_doc.to_dict(),
            "message": f"Curso activo encontrado en {classroom_name}",
        })

    return jsonify({
        "success": False,
        "error": "Curso encontrado pero no existe en BD",
        "course_id": course_id,
        "classroom": classroom_name,
    })


@app.route("/api/active_course", methods=["GET"])
def api_active_course():
    """Tell the screen whether to wait or open the camera."""
    try:
        classroom_name = current_classroom()
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


@app.route("/setup")
def setup():
    """First screen, used to pick the classroom."""
    return render_template("classroom_setup.html")


@app.route("/api/classrooms", methods=["GET"])
def api_classrooms():
    """List classrooms found in courses."""
    try:
        classrooms = available_classrooms()
        return jsonify({"success": True, "classrooms": classrooms, "total": len(classrooms)}), 200
    except Exception as error:
        print(f"[✖] ERROR en /api/classrooms al leer los salones de courses: {error}")
        return jsonify({"success": False, "error": "No se pudieron cargar los salones."}), 500


@app.route("/api/event_mode", methods=["GET"])
def api_event_mode():
    """Whether this process is enrolling for an event."""
    return jsonify(event_mode.status()), 200


@app.route("/api/event_mode", methods=["POST"])
def api_set_event_mode():
    """Turn event enrollment on or off. Does not write `.env`."""
    data = request.get_json() or {}
    if data.get("enabled") is True:
        ok, message = event_mode.enable(
            data.get("name", ""),
            data.get("ini_time", ""),
            data.get("end_time", ""),
        )
    else:
        ok, message = event_mode.disable()
    if not ok:
        return jsonify({"ok": False, "error": message}), 400
    payload = event_mode.status()
    payload["ok"] = True
    return jsonify(payload), 200


@app.route("/api/event_person", methods=["POST"])
def api_event_person():
    """Create a person for this event. An existing cédula is not overwritten."""
    data = request.get_json() or {}
    national_id = _national_id_from_request(data)
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


@app.route("/api/event_consent", methods=["POST"])
def api_event_consent():
    """Temporary biometric acceptance for someone whose current row says false."""
    data = request.get_json() or {}
    national_id = _national_id_from_request(data)
    if not national_id:
        return jsonify({"ok": False, "error": "La cédula solo puede tener números"}), 400
    ok, message = event_mode.grant_event_consent(national_id)
    if not ok:
        return jsonify({"ok": False, "error": message}), 400
    return jsonify({"ok": True}), 200


@app.route("/api/event_link", methods=["POST"])
def api_event_link():
    """Link an existing cédula to today's event without changing their vector."""
    data = request.get_json() or {}
    national_id = _national_id_from_request(data)
    if not national_id:
        return jsonify({"ok": False, "error": "La cédula solo puede tener números"}), 400
    ok, message = event_mode.link_existing(national_id, data.get("name", ""))
    if not ok:
        return jsonify({"ok": False, "error": message}), 400
    return jsonify({"ok": True}), 200


@app.route("/api/set_classroom", methods=["POST"])
def api_set_classroom():
    """Save the classroom when its name is in the course list."""
    try:
        data = request.get_json()
        classroom_name = data.get("classroom", "").strip()
        if not classroom_name:
            return jsonify({"success": False, "error": "Nombre de salón requerido"}), 400

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


@app.route("/api/current_classroom", methods=["GET"])
def api_current_classroom():
    """Return the classroom held in memory."""
    try:
        classroom_name = current_classroom()
        return jsonify({
            "success": True,
            "classroom": classroom_name,
            "configured": classroom_name is not None,
        }), 200
    except Exception as error:
        print(f"[✖] ERROR en /api/current_classroom: {error}")
        return jsonify({"success": False, "error": "No se pudo leer el salón."}), 500


@app.route("/api/clear_recognitions", methods=["POST"])
def api_clear_recognitions():
    """Clear the in-memory recognized set. Called when the classroom changes."""
    global recognized_students, recognition_started_at
    try:
        recognized_students.clear()
        recognition_started_at.clear()
        print("\n🧹 REGISTROS LIMPIADOS MANUALMENTE")
        print("   Se reintentará el registro de asistencia")
        return jsonify({"success": True, "message": "Registros limpiados correctamente"}), 200
    except Exception as error:
        print(f"[✖] ERROR en /api/clear_recognitions: {error}")
        return jsonify({"success": False, "error": "No se pudieron limpiar los registros."}), 500


@app.route("/firebase-config")
def firebase_config_route():
    return send_file("./firebase_config_public.json")

