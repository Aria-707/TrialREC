"""Classroom selected for this kiosk, in memory and in classroom_config.txt."""
import os

from constants import CLASSROOM_CONFIG_FILE

configured_classroom = None


def classroom_for_scheduler():
    """Callback the scheduler uses to read the classroom."""
    return current_classroom()


def load_saved_classroom():
    """Load the classroom saved on disk."""
    global configured_classroom
    try:
        if os.path.exists(CLASSROOM_CONFIG_FILE):
            with open(CLASSROOM_CONFIG_FILE, "r", encoding="utf-8") as handle:
                configured_classroom = handle.read().strip()
                print(f"✔ Salón cargado desde archivo: '{configured_classroom}'")
                return configured_classroom
    except Exception as error:
        print(f"⚠️ Error cargando el salón desde {CLASSROOM_CONFIG_FILE}: {error}")
    return None


def save_classroom(classroom_name):
    """Write the classroom name to disk."""
    try:
        with open(CLASSROOM_CONFIG_FILE, "w", encoding="utf-8") as handle:
            handle.write(classroom_name)
        print(f"💾 Salón guardado: '{classroom_name}'")
        return True
    except Exception as error:
        print(f"❌ Error guardando el salón en {CLASSROOM_CONFIG_FILE}: {error}")
        return False


def set_classroom(classroom_name):
    """Set the active classroom and persist it."""
    global configured_classroom
    try:
        configured_classroom = classroom_name.strip()
        save_classroom(configured_classroom)
        print(f"\n✅ SALÓN CONFIGURADO: '{configured_classroom}'")
        return True
    except Exception as error:
        print(f"[✖] ERROR configurando el salón '{classroom_name}': {error}")
        return False


def current_classroom():
    """Return the classroom currently in memory."""
    return configured_classroom
