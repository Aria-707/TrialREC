"""LBPH model load and face detection."""
import json
import os
import shutil
import threading

import cv2
import numpy as np

from constants import FACE_SIZE, LABEL_MAP_PATH, MODEL_PATH, PHOTO_DIR_NAME, RECOGNITION_SECONDS

photo_dir = os.path.join(os.path.dirname(__file__), PHOTO_DIR_NAME)
model_path = MODEL_PATH
label_map_path = LABEL_MAP_PATH

os.makedirs(photo_dir, exist_ok=True)

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
model_lock = threading.Lock()


def _person_folders():
    """Folders that belong in the model. A parked capture is not one of them."""
    names = []
    for name in os.listdir(photo_dir):
        if name.endswith(".previous"):
            continue
        folder = os.path.join(photo_dir, name)
        if os.path.isdir(folder):
            names.append(name)
    return names


def _apply_labels(stored_labels):
    """Keep the same list object so other modules see the new labels."""
    image_paths.clear()
    if not stored_labels:
        return
    image_paths.extend([None] * (max(stored_labels.values()) + 1))
    for person, label in stored_labels.items():
        image_paths[label] = person


def load_label_map():
    """Return the saved label map, or None when this model has no map yet."""
    if not os.path.exists(label_map_path):
        return None
    with open(label_map_path, encoding="utf-8") as handle:
        stored = json.load(handle)
    return {person: int(label) for label, person in stored.items()}


def save_label_map():
    """Persist label → cédula so a restart keeps the same numbers."""
    stored = {str(label): person for person, label in label_dict.items()}
    os.makedirs(os.path.dirname(label_map_path) or ".", exist_ok=True)
    with open(label_map_path, "w", encoding="utf-8") as handle:
        json.dump(stored, handle)


def _read_gray(image_path):
    from encryption import decrypt_file

    image_bytes = decrypt_file(image_path)
    if image_bytes is None:
        print(f"⚠️ No se pudo desencriptar {image_path}")
        return None
    array = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)


image_paths = []
label_dict = {}
next_label = 0

if os.path.exists(model_path):
    face_recognizer.read(model_path)
    stored_labels = load_label_map()
    if stored_labels is None:
        label_dict = {name: index for index, name in enumerate(_person_folders())}
    else:
        label_dict = stored_labels
    _apply_labels(label_dict)
    next_label = (max(label_dict.values()) + 1) if label_dict else 0

print("Model loaded. Persons:", len([name for name in image_paths if name]))
print("Next label:", next_label)

recognition_seconds = RECOGNITION_SECONDS


def predict_face(face_image):
    """Predict under the model lock. Returns (label, confidence)."""
    with model_lock:
        return face_recognizer.predict(face_image)


def train_incrementally(new_records):
    """Add photos to the existing LBPH model. Photos stay on disk."""
    global face_recognizer, label_dict, next_label, image_paths

    face_images, labels = [], []

    with model_lock:
        for person, paths in new_records.items():
            if person not in label_dict:
                label_dict[person] = next_label
                next_label += 1

            label = label_dict[person]
            for image_path in paths:
                try:
                    image = _read_gray(image_path)
                    if image is not None:
                        face_images.append(image)
                        labels.append(label)
                except Exception as error:
                    print(f"Error leyendo la imagen {image_path}: {error}")

        if not face_images:
            print("No hay imágenes nuevas para entrenar.")
            return False

        face_recognizer.update(face_images, np.array(labels))
        face_recognizer.write(model_path)
        _apply_labels(label_dict)
        save_label_map()
        print(f"Entrenamiento incremental: {len(face_images)} imágenes añadidas.")
        return True


def rebuild_model():
    """Train again from the photos that remain. update() cannot drop a person."""
    global face_recognizer, label_dict, next_label, image_paths

    folders = _person_folders()
    with model_lock:
        kept = {}
        used = set()
        for person, label in label_dict.items():
            if person in folders:
                kept[person] = label
                used.add(label)

        next_free = 0
        for person in folders:
            if person in kept:
                continue
            while next_free in used:
                next_free += 1
            kept[person] = next_free
            used.add(next_free)
            next_free += 1

        face_images, labels = [], []
        for person, label in kept.items():
            folder = os.path.join(photo_dir, person)
            for filename in os.listdir(folder):
                if not filename.endswith(".jpg"):
                    continue
                image_path = os.path.join(folder, filename)
                try:
                    image = _read_gray(image_path)
                    if image is not None:
                        face_images.append(image)
                        labels.append(label)
                except Exception as error:
                    print(f"Error leyendo la imagen {image_path}: {error}")

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        if face_images:
            recognizer.train(face_images, np.array(labels))
            recognizer.write(model_path)
        elif os.path.exists(model_path):
            os.remove(model_path)

        face_recognizer = recognizer
        label_dict.clear()
        label_dict.update(kept)
        _apply_labels(label_dict)
        next_label = (max(kept.values()) + 1) if kept else 0
        save_label_map()
        print(f"Modelo reconstruido con {len(face_images)} imágenes.")
        return bool(face_images) or not folders


def begin_capture(national_id):
    """Park the previous photos so the new hundred replace them."""
    if not national_id.isdigit():
        raise ValueError("La cédula solo puede tener números")

    folder = os.path.join(photo_dir, national_id)
    previous = folder + ".previous"
    if os.path.isdir(previous):
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.rename(previous, folder)
    if os.path.isdir(folder):
        os.rename(folder, previous)
    os.makedirs(folder, exist_ok=True)
    return folder


def discard_capture(national_id):
    """Put the parked photos back when training does not finish."""
    if not national_id.isdigit():
        return
    folder = os.path.join(photo_dir, national_id)
    previous = folder + ".previous"
    if not os.path.isdir(previous):
        return
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.rename(previous, folder)


def detect_face(gray_image):
    """Return (faces, method) or (None, None)."""
    try:
        faces = face_classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30),
        )
        if len(faces) > 0:
            return faces, "normal"

        faces = face_classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(20, 20),
        )
        if len(faces) > 0:
            return faces, "permissive"

        equalized = cv2.equalizeHist(gray_image)
        faces = face_classifier.detectMultiScale(
            equalized,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30),
        )
        if len(faces) > 0:
            return faces, "equalized"

    except Exception as error:
        print(f"Error detectando el rostro en el fotograma: {error}")

    return None, None
