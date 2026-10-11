"""Recognition process. OpenCV and Firestore, only on localhost."""
import sys

import cv2

import attendance_scheduler
import firebase_config  # noqa: F401  (initializes Firebase before the model)
import face
import classroom
import routes
from constants import RECOGNITION_HOST, RECOGNITION_PORT

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 INICIANDO RECONOCIMIENTO")
    print("=" * 60)
    print(f"📁 Data Path: {face.photo_dir}")
    print(f"🤖 Model Path: {face.model_path}")
    print(f"👥 Personas cargadas: {len(face.image_paths)}")
    print(f"Python: {sys.version}")
    print(f"OpenCV: {cv2.__version__}")

    saved_classroom = classroom.load_saved_classroom()
    if saved_classroom:
        print(f"🏫 Salón configurado: {saved_classroom}")
    else:
        print("⚠️ No hay salón configurado")

    print("=" * 60 + "\n")

    print("\n" + "=" * 60)
    print("⏰ INICIANDO SCHEDULER DE ASISTENCIA")
    print("=" * 60)
    attendance_scheduler.start_scheduler(classroom.classroom_for_scheduler)
    print("✅ Scheduler iniciado en hilo separado")
    print("=" * 60 + "\n")

    routes.app.run(debug=True, host=RECOGNITION_HOST, port=RECOGNITION_PORT)
