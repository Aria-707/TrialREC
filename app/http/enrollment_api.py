"""Enrollment. A cédula already on file can repeat the capture."""
import base64
import os
import re
import time

import cv2
import numpy as np
from flask import Blueprint, jsonify, request

from app.attendance.record import lookup_student
from app.consent.legal import biometrico_is_true, has_legal_row, record_acceptance
from app.events import mode as event_mode
from app.firebase.vectors import commit_event_vector, commit_vector_copy
from app.http.national_id import national_id_from_request
from app.recognition.face import (
    analyze_frame,
    begin_capture,
    clear_capture,
    discard_capture,
    photo_dir,
    stored_vector,
    train_incrementally,
)
from app.security.audit import log_event
from app.security.encryption import encrypt_file

blueprint = Blueprint("enrollment_api", __name__)


@blueprint.route("/detect_face", methods=["POST"])
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


@blueprint.route("/api/lookup_student", methods=["POST"])
def api_lookup_student():
    """Confirm a cédula before any photo is stored."""
    data = request.get_json() or {}
    national_id = national_id_from_request(data)
    if not national_id:
        return jsonify({"status": "missing"}), 200
    return jsonify(lookup_student(national_id)), 200


@blueprint.route("/api/begin_capture", methods=["POST"])
def api_begin_capture():
    """Park older photos so this capture replaces them."""
    data = request.get_json() or {}
    national_id = national_id_from_request(data)
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


@blueprint.route("/save_photo", methods=["POST"])
def save_photo():
    """Save one enrollment photo under data/{cedula}."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"ok": False, "error": "No se recibió data"}), 400

        national_id = national_id_from_request(data)
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


@blueprint.route("/train", methods=["POST"])
def train():
    """Store one SFace vector from data/{cedula}, then delete those photos."""
    national_id = None
    try:
        data = request.get_json() or {}
        national_id = national_id_from_request(data)
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
