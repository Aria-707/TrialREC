"""Audit log. JSON keys stay as stored so the existing file still loads."""
import json
import os
from datetime import datetime

from constants import AUDIT_LOG_FILE


def log_event(event_type, description, user=None, extra=None):
    """Append one audit event. `tipo` and the other keys match the file already on disk."""
    try:
        event = {
            "timestamp": datetime.now().isoformat(),
            "tipo": event_type,
            "descripcion": description,
            "usuario": user,
            "datos_adicionales": extra or {},
        }

        if os.path.exists(AUDIT_LOG_FILE):
            with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as handle:
                logs = json.load(handle)
        else:
            logs = []

        logs.append(event)

        with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as handle:
            json.dump(logs, handle, indent=2, ensure_ascii=False)

        print(f"📝 Evento de auditoría registrado: {event_type}")

    except Exception as error:
        print(f"⚠️ Error registrando el evento de auditoría '{event_type}' en {AUDIT_LOG_FILE}: {error}")


def read_logs(event_type=None, limit=100):
    """Return the latest audit events."""
    try:
        if not os.path.exists(AUDIT_LOG_FILE):
            return []

        with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as handle:
            logs = json.load(handle)

        if event_type:
            logs = [log for log in logs if log["tipo"] == event_type]

        return logs[-limit:]

    except Exception as error:
        print(f"Error leyendo el log de auditoría {AUDIT_LOG_FILE}: {error}")
        return []
