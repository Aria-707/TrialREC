"""HTTP routes of the recognition process. The browser reaches them through the BFF."""
import base64
import os
import re
import shutil
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
from constants import CONFIDENCE_THRESHOLD, DEMO_MODE, FACE_SIZE
from encryption import encrypt_file
from face import (
    begin_capture,
    detect_face,
    discard_capture,
    image_paths,
    label_dict,
    model_path,
    photo_dir,
    predict_face,
    recognition_seconds,
    rebuild_model,
    train_incrementally,
)
from firebase_config import db
from schedule import (
    active_course_for_classroom,
    available_classrooms,
    next_course,
)

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
    return render_template("enroll.html", demo_mode=DEMO_MODE)


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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_copy = gray.copy()
        faces, _method = detect_face(gray)

        if faces is None or len(faces) == 0:
            return jsonify({"status": "no_face"})

        x, y, width, height = faces[0]
        face_crop = gray_copy[y:y + height, x:x + width]
        face_crop = cv2.resize(face_crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_CUBIC)
        course_id, start_time = active_course_for_classroom(required_classroom=classroom_name)
        if course_id:
            sweep_before_class_frame(course_id)
        label, confidence = predict_face(face_crop)
        box = [int(x), int(y), int(width), int(height)]

        folder_name = image_paths[label] if label < len(image_paths) else None
        if confidence < CONFIDENCE_THRESHOLD and folder_name:
            student_name = folder_name.replace("_", " ")
            national_id = folder_name if folder_name.isdigit() else None
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
                            "confidence": float(confidence),
                            "box": box,
                            "classroom": classroom_name,
                            "recorded": True,
                            "arrival_category": category,
                        })
                    return jsonify({
                        "status": "recognized",
                        "student": student_name,
                        "confidence": float(confidence),
                        "box": box,
                        "classroom": classroom_name,
                        "recorded": False,
                    })
                if not course_id:
                    print(f"[!] Reconocido en {classroom_name}, sin curso activo")

            return jsonify({
                "status": "recognized",
                "student": student_name,
                "confidence": float(confidence),
                "box": box,
                "classroom": classroom_name,
            })

        return jsonify({
            "status": "unknown",
            "confidence": float(confidence),
            "box": box,
        })
    except Exception as error:
        print(f"Error en /recognize al procesar el fotograma: {error}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(error)}), 500


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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces, method = detect_face(gray)

        if faces is not None and len(faces) > 0:
            x, y, width, height = faces[0]
            return jsonify({
                "face_detected": True,
                "box": [int(x), int(y), int(width), int(height)],
                "method": method,
            }), 200
        return jsonify({"face_detected": False}), 200

    except Exception as error:
        print(f"Error en /detect_face al revisar el fotograma de alta: {error}")
        return jsonify({"face_detected": False, "error": str(error)}), 200


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
    if DEMO_MODE:
        return jsonify({
            "status": "demo_unavailable",
            "message": "El modo de prueba todavía no está afinado.",
        }), 200

    data = request.get_json() or {}
    national_id = _national_id_from_request(data)
    if not national_id:
        return jsonify({"status": "missing"}), 200
    return jsonify(lookup_student(national_id)), 200


@app.route("/api/begin_capture", methods=["POST"])
def api_begin_capture():
    """Park older photos so this capture replaces them."""
    if DEMO_MODE:
        return jsonify({
            "ok": False,
            "error": "El modo de prueba todavía no está afinado.",
        }), 400

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

        if DEMO_MODE:
            return jsonify({
                "ok": False,
                "error": "El modo de prueba todavía no está afinado.",
            }), 400

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
            return jsonify({"ok": False, "error": f"Error creando carpeta: {str(error)}"}), 500

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
            return jsonify({"ok": False, "error": f"Error decodificando: {str(error)}"}), 500

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as error:
            return jsonify({"ok": False, "error": f"Error en conversión: {str(error)}"}), 500

        faces = None
        method = "none"
        try:
            faces, method = detect_face(gray)
        except Exception as error:
            print(f"⚠️ Error detectando el rostro al guardar la foto: {error}")
            return jsonify({"ok": False, "error": "No se detectó un rostro"}), 200

        if faces is None or len(faces) == 0:
            print("Sin rostro en el fotograma de alta. No se guarda.")
            return jsonify({"ok": False, "error": "No se detectó un rostro"}), 200

        timestamp = int(time.time() * 1000000)
        file_path = os.path.join(person_path, f"face_{timestamp}.jpg")

        try:
            x, y, width, height = faces[0]
            face_crop = gray[y:y + height, x:x + width]
            final_face = cv2.resize(face_crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_CUBIC)
            capture_type = "crop"

            success = cv2.imwrite(file_path, final_face)
            if not success or not os.path.exists(file_path):
                return jsonify({"ok": False, "error": "Error al guardar archivo"}), 500

            file_size = os.path.getsize(file_path)
            print(f"✅ Foto guardada: {file_size} bytes ({capture_type})")
            print("🔒 Encriptando imagen...")
            if encrypt_file(file_path):
                print("✔ Imagen encriptada exitosamente")
            else:
                print(f"⚠️ No se pudo encriptar la imagen {file_path}")

            log_event(
                "REGISTRO_DATOS_BIOMETRICOS",
                "Captura de imagen facial para el alta",
                user=national_id,
            )
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
            return jsonify({"ok": False, "error": f"Error guardando: {str(error)}"}), 500

    except Exception as error:
        print(f"❌ ERROR GENERAL guardando la foto de alta: {error}")
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"Error general: {str(error)}"}), 500


@app.route("/train", methods=["POST"])
def train():
    """Train LBPH from data/{cedula}. A replacement rebuilds the whole model."""
    national_id = None
    try:
        if DEMO_MODE:
            return jsonify({
                "success": False,
                "error": "El modo de prueba todavía no está afinado.",
            }), 400

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
        previous_path = person_path + ".previous"

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
        if os.path.isdir(previous_path) or national_id in label_dict:
            trained = rebuild_model()
        else:
            trained = train_incrementally({national_id: image_files})

        if not trained:
            discard_capture(national_id)
            return jsonify({"success": False, "error": "Falló el entrenamiento"}), 500

        if os.path.isdir(previous_path):
            shutil.rmtree(previous_path)

        if not already_consented and not record_acceptance(national_id):
            return jsonify({
                "success": False,
                "error": "Un administrador tiene que registrarte antes de este alta.",
            }), 400

        print("✅ ENTRENAMIENTO COMPLETADO")
        print(f"   • Imágenes preservadas: {len(files)}")
        print(f"   • Modelo guardado en: {model_path}")
        print(f"{'='*60}\n")

        return jsonify({
            "success": True,
            "message": f"Modelo entrenado con {len(image_files)} imágenes",
            "images_trained": len(image_files),
        }), 200

    except Exception as error:
        print(f"✖ ERROR EN ENTRENAMIENTO: {error}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(error)}), 500


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
            course_doc = db.collection("courses").document(course_id).get()
            if course_doc.exists:
                course_data = course_doc.to_dict()
                return jsonify({
                    "course_active": True,
                    "classroom_configured": True,
                    "course_id": course_id,
                    "start_time": start_time,
                    "classroom": classroom_name,
                    "course_name": course_data.get("nameCourse", "Sin nombre"),
                }), 200

        upcoming = next_course(classroom_name)
        return jsonify({
            "course_active": False,
            "classroom_configured": True,
            "classroom": classroom_name,
            "next_course": upcoming,
        }), 200

    except Exception as error:
        print(f"[✖] ERROR en /api/active_course al consultar el horario del salón: {error}")
        import traceback
        traceback.print_exc()
        return jsonify({"course_active": False, "error": str(error)}), 500


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
        return jsonify({"success": False, "error": str(error)}), 500


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
        return jsonify({"success": False, "error": "Error al configurar salón"}), 500

    except Exception as error:
        print(f"[✖] ERROR en /api/set_classroom al guardar el salón: {error}")
        return jsonify({"success": False, "error": str(error)}), 500


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
        return jsonify({"success": False, "error": str(error)}), 500


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
        return jsonify({"success": False, "error": str(error)}), 500


@app.route("/firebase-config")
def firebase_config_route():
    return send_file("./firebase_config_public.json")

