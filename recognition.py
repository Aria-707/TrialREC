"""Recognition process. OpenCV and Firestore, only on localhost."""
import os
import sys

import cv2

import attendance_scheduler
import firebase_config  # noqa: F401  (initializes Firebase before the model)
import face
import classroom
import consent
import schedule
import routes
from config import RECOGNITION_HOST, RECOGNITION_PORT

if __name__ == "__main__":
    serving = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    if serving:
        import event_mode
        event_mode.wipe_event_data()
        face.refresh_remote_gallery()

    print("\n" + "=" * 60)
    print("🚀 INICIANDO RECONOCIMIENTO")
    print("=" * 60)
    print(f"📁 Data Path: {face.photo_dir}")
    print("🤖 Galería: Firebase")
    print(f"👥 Personas cargadas: {face.person_count()}")
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

    if serving:
        consent.sweep_and_rebuild()
        import event_mode
        event_mode.boot_from_env(saved_classroom)
        consent.start_listener()
        schedule.start_schedule_watch()

    routes.app.run(debug=True, host=RECOGNITION_HOST, port=RECOGNITION_PORT)
