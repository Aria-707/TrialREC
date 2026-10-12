"""Audit log. One JSON object per line. Keys match the events already stored."""
import json
import os
from collections import deque
from datetime import datetime

from config import AUDIT_LOG_FILE


def _convert_array_if_needed():
    """Turn the old JSON array into one object per line, once."""
    if not os.path.exists(AUDIT_LOG_FILE) or os.path.getsize(AUDIT_LOG_FILE) == 0:
        return
    with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as handle:
        first = handle.read(1)
        if first != "[":
            return
        handle.seek(0)
        logs = json.load(handle)
    with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as handle:
        for event in logs:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"📝 Log de auditoría pasado a una línea por evento: {len(logs)}")


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
        _convert_array_if_needed()
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(f"📝 Evento de auditoría registrado: {event_type}")
    except Exception as error:
        print(f"⚠️ Error registrando el evento de auditoría '{event_type}' en {AUDIT_LOG_FILE}: {error}")


def read_logs(event_type=None, limit=100):
    """Return the latest audit events, one line at a time."""
    try:
        if not os.path.exists(AUDIT_LOG_FILE):
            return []
        _convert_array_if_needed()
        kept = deque(maxlen=limit)
        with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if event_type and event.get("tipo") != event_type:
                    continue
                kept.append(event)
        return list(kept)
    except Exception as error:
        print(f"Error leyendo el log de auditoría {AUDIT_LOG_FILE}: {error}")
        return []
