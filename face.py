"""SFace vectors and YuNet detection. One normalized vector per person."""
import os
import shutil
import threading

import cv2
import numpy as np

from config import (
    AMBIGUITY_MARGIN,
    COSINE_THRESHOLD,
    DETECTOR_MODEL_PATH,
    PHOTO_DIR_NAME,
    RECOGNITION_SECONDS,
    RECOGNIZER_MODEL_PATH,
    VECTOR_PATH,
)

photo_dir = os.path.join(os.path.dirname(__file__), PHOTO_DIR_NAME)
vector_path = VECTOR_PATH
model_lock = threading.Lock()

os.makedirs(photo_dir, exist_ok=True)

_detector = None
_recognizer = None
_models_ready = False
_models_warned = False
face_vectors = {}
recognition_seconds = RECOGNITION_SECONDS
ALIGNED_SIZE = 112


def _person_folders():
    """Folders that belong in the vector file. A parked capture is not one of them."""
    names = []
    for name in os.listdir(photo_dir):
        if name.endswith(".previous"):
            continue
        folder = os.path.join(photo_dir, name)
        if os.path.isdir(folder):
            names.append(name)
    return names


def _ensure_models():
    """Load the fixed YuNet and SFace weights. Never creates a substitute."""
    global _detector, _recognizer, _models_ready, _models_warned
    if _models_ready:
        return True
    if not os.path.exists(DETECTOR_MODEL_PATH) or not os.path.exists(RECOGNIZER_MODEL_PATH):
        if not _models_warned:
            print("Faltan los pesos de YuNet o SFace en backend/models. No se inventa un modelo.")
            _models_warned = True
        return False
    _detector = cv2.FaceDetectorYN.create(
        DETECTOR_MODEL_PATH, "", (320, 320), 0.9, 0.3, 5000,
    )
    _recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL_PATH, "")
    _models_ready = True
    return True


def _detect(bgr):
    """YuNet rows for this frame. None when the weights are missing."""
    if not _ensure_models():
        return None
    height, width = bgr.shape[:2]
    _detector.setInputSize((width, height))
    _count, faces = _detector.detect(bgr)
    if faces is None or _count == 0:
        return []
    return faces


def _read_color(image_path):
    from encryption import decrypt_file

    image_bytes = decrypt_file(image_path)
    if image_bytes is None:
        print(f"⚠️ No se pudo desencriptar {image_path}")
        return None
    array = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def _mean_vector(features):
    """One normalized vector. Near-duplicate frames collapse instead of stacking."""
    if not features:
        return None
    stacked = np.vstack([feature.reshape(1, -1) for feature in features]).astype(np.float32)
    mean = stacked.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0:
        return None
    return (mean / norm).astype(np.float32)


def _feature_from_stored_unlocked(bgr):
    """Vector of a photo already on disk. Old gray crops fall back to a resize."""
    faces = _detect(bgr)
    if faces is None:
        return None
    if len(faces) == 1:
        aligned = _recognizer.alignCrop(bgr, faces[0])
        return _recognizer.feature(aligned)
    if len(faces) > 1:
        return None
    if bgr.shape[0] <= 200 and bgr.shape[1] <= 200:
        resized = cv2.resize(bgr, (ALIGNED_SIZE, ALIGNED_SIZE))
        return _recognizer.feature(resized)
    return None


def _vector_from_paths_unlocked(paths):
    features = []
    for image_path in paths:
        try:
            image = _read_color(image_path)
            if image is None:
                continue
            feature = _feature_from_stored_unlocked(image)
            if feature is not None:
                features.append(feature)
        except Exception as error:
            print(f"Error leyendo la imagen {image_path}: {error}")
    mean = _mean_vector(features)
    if mean is None:
        return None
    return mean.reshape(-1).tolist()


def _photo_paths(folder):
    return [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.endswith(".jpg")
    ]


def person_count():
    """How many cédulas have a vector in memory."""
    return len(face_vectors)


def person_ids():
    """Cédulas currently loaded for comparison."""
    with model_lock:
        return list(face_vectors.keys())


def has_vector(national_id):
    """True when this cédula is already in memory."""
    return str(national_id) in face_vectors


def stored_vector(national_id):
    """A copy of one vector, for the Firebase write outside the lock."""
    with model_lock:
        values = face_vectors.get(str(national_id))
        if values is None:
            return None
        return list(values)


def choose_match(scored, threshold=COSINE_THRESHOLD, margin=AMBIGUITY_MARGIN):
    """Pick a cédula, or refuse when the best score is low or too close to the next."""
    if not scored:
        return None, None, "none"
    ordered = sorted(scored, key=lambda item: item[0], reverse=True)
    best_score, best_id = ordered[0]
    if best_score < threshold:
        return None, best_score, "unknown"
    if len(ordered) > 1 and best_score - ordered[1][0] < margin:
        return None, best_score, "ambiguous"
    return best_id, best_score, "match"


def analyze_frame(bgr, aligned=False):
    """One face yields a box and either its vector or the aligned crop."""
    with model_lock:
        if not _ensure_models():
            return {"status": "unavailable"}
        faces = _detect(bgr)
        if faces is None or len(faces) == 0:
            return {"status": "no_face"}
        if len(faces) > 1:
            return {"status": "several"}
        row = faces[0]
        result = {
            "status": "one",
            "box": [int(row[0]), int(row[1]), int(row[2]), int(row[3])],
        }
        if aligned:
            result["aligned"] = _recognizer.alignCrop(bgr, row)
        else:
            crop = _recognizer.alignCrop(bgr, row)
            result["feature"] = _recognizer.feature(crop)
        return result


def _match_ids_unlocked(feature, allowed_ids):
    """Score one vector against the given cédulas. Caller holds model_lock."""
    if feature is None or not _ensure_models():
        return None, None, "none"
    scored = []
    for national_id in allowed_ids:
        stored = face_vectors.get(str(national_id))
        if not stored:
            continue
        candidate = np.array(stored, dtype=np.float32).reshape(1, -1)
        score = float(_recognizer.match(
            feature, candidate, cv2.FaceRecognizerSF_FR_COSINE,
        ))
        scored.append((score, str(national_id)))
    return choose_match(scored)


def match_enrolled(feature):
    """Compare one vector with every enrolled cédula. Returns (cédula, cosine, kind)."""
    with model_lock:
        return _match_ids_unlocked(feature, list(face_vectors.keys()))


def train_incrementally(new_records):
    """Store one vector per new cédula. Photos stay on disk."""
    with model_lock:
        if not _ensure_models():
            return False
        added = 0
        for person, paths in new_records.items():
            vector = _vector_from_paths_unlocked(paths)
            if vector is None:
                continue
            face_vectors[str(person)] = vector
            added += 1
        if added == 0:
            print("No hay imágenes nuevas para entrenar.")
            return False
        print(f"Vector guardado para {added} persona(s).")
        return True


def forget_vector(national_id):
    """Drop one cédula from memory. Photos and Firebase are left for the caller."""
    with model_lock:
        key = str(national_id)
        if key not in face_vectors:
            return False
        del face_vectors[key]
        return True


def clear_capture(national_id):
    """Remove this cédula's photos after the Firebase copy has been read back."""
    if not str(national_id).isdigit():
        return False
    folder = os.path.join(photo_dir, str(national_id))
    previous = folder + ".previous"
    removed = False
    if os.path.isdir(folder):
        shutil.rmtree(folder)
        removed = True
    if os.path.isdir(previous):
        shutil.rmtree(previous)
        removed = True
    return removed


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


def _digit_folders():
    return [name for name in _person_folders() if str(name).isdigit()]


def _delete_plaintext_vectors():
    directory = os.path.dirname(vector_path) or "."
    for path in (vector_path, os.path.join(directory, "face_vectors copy.json")):
        if os.path.isfile(path):
            os.remove(path)


def refresh_remote_gallery():
    """Load the gallery from Firebase. A missing folder does not delete that copy."""
    from consent import biometrico_is_explicit_false
    from vector_store import (
        commit_vector_copy,
        delete_vector_copy,
        list_stored_vectors,
    )

    stored, missing, reason = list_stored_vectors()
    if reason:
        with model_lock:
            face_vectors.clear()
        print(f"Galería vacía: {reason}.")
        return

    for national_id in list(stored):
        if biometrico_is_explicit_false(national_id):
            delete_vector_copy(national_id)
            del stored[national_id]

    for name in _digit_folders():
        if biometrico_is_explicit_false(name):
            clear_capture(name)
            delete_vector_copy(name)
            stored.pop(name, None)
            continue
        if name in stored:
            clear_capture(name)
            continue
        if not _ensure_models():
            missing += 1
            continue
        vector = _vector_from_paths_unlocked(
            _photo_paths(os.path.join(photo_dir, name))
        )
        if vector is None or not commit_vector_copy(name, vector):
            missing += 1
            continue
        stored[name] = vector
        clear_capture(name)

    with model_lock:
        face_vectors.clear()
        face_vectors.update(stored)
    if missing:
        print(f"Galería desde Firebase: {len(stored)}. Faltan {missing}.")
    else:
        print(f"Galería desde Firebase: {len(stored)}")
    if missing == 0 and not _digit_folders():
        _delete_plaintext_vectors()
