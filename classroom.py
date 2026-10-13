"""Classrooms this server is watching. Each browser keeps its own in a cookie."""
import os
import threading

from config import CLASSROOM_CONFIG_FILE

configured_classroom = None
_listeners = []
_known = set()
_known_lock = threading.Lock()


def on_classroom_change(callback):
    """Run after the saved classroom changes. The scheduler uses this to wake."""
    _listeners.append(callback)


def note_classroom(classroom_name):
    """Remember a classroom a browser is using. Wakes listeners only the first time."""
    name = (classroom_name or "").strip()
    if not name:
        return False
    with _known_lock:
        if name in _known:
            return False
        _known.add(name)
    for callback in list(_listeners):
        callback(name)
    return True


def known_classrooms():
    """Classrooms the scheduler should watch."""
    with _known_lock:
        return sorted(_known)


def classroom_for_scheduler():
    """Callback the scheduler uses to read the classrooms."""
    return known_classrooms()


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
    """Remember this classroom for the scheduler. The browser cookie is the source of truth."""
    try:
        name = classroom_name.strip()
        note_classroom(name)
        from schedule import forget_classroom_slots
        forget_classroom_slots()
        print(f"\n✅ SALÓN CONFIGURADO: '{name}'")
        return True
    except Exception as error:
        print(f"[✖] ERROR configurando el salón '{classroom_name}': {error}")
        return False


def current_classroom():
    """Return the classroom currently in memory."""
    return configured_classroom
