"""LBPH model load and face detection."""
import os

import cv2
import numpy as np

from constants import FACE_SIZE, MODEL_PATH, PHOTO_DIR_NAME, RECOGNITION_SECONDS

photo_dir = os.path.join(os.path.dirname(__file__), PHOTO_DIR_NAME)
model_path = MODEL_PATH

os.makedirs(photo_dir, exist_ok=True)

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

if os.path.exists(model_path):
    face_recognizer.read(model_path)
    image_paths = os.listdir(photo_dir)
    label_dict = {name: index for index, name in enumerate(image_paths)}
    next_label = len(image_paths)
else:
    image_paths = []
    label_dict = {}
    next_label = 0

print("Model loaded. Persons:", image_paths)
print("Label dict:", label_dict, " Next label:", next_label)

recognition_seconds = RECOGNITION_SECONDS


def train_incrementally(new_records):
    """Add photos to the existing LBPH model. Photos stay on disk."""
    global face_recognizer, label_dict, next_label, image_paths

    face_images, labels = [], []

    for person, paths in new_records.items():
        if person not in label_dict:
            label_dict[person] = next_label
            image_paths.append(person)
            next_label += 1

        label = label_dict[person]
        for image_path in paths:
            try:
                from encryption import decrypt_file

                image_bytes = decrypt_file(image_path)
                if image_bytes is None:
                    print(f"⚠️ No se pudo desencriptar {image_path}")
                    continue

                array = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    face_images.append(image)
                    labels.append(label)
            except Exception as error:
                print(f"Error leyendo la imagen {image_path}: {error}")

    if face_images:
        face_recognizer.update(face_images, np.array(labels))
        face_recognizer.write(model_path)
        print(f"Entrenamiento incremental: {len(face_images)} imágenes añadidas.")
        return True

    print("No hay imágenes nuevas para entrenar.")
    return False


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
