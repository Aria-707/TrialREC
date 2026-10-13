"""Faces already seen in this process, one set per classroom."""
from datetime import datetime

recognized_by_room = {}
started_by_room = {}
recognition_day = None


def recognized(room):
    return recognized_by_room.setdefault(room, set())


def started(room):
    return started_by_room.setdefault(room, {})


def clear_room(room):
    recognized_by_room.pop(room, None)
    started_by_room.pop(room, None)


def forget_recognitions_from_other_days():
    """Drop in-memory hits from a previous date. Today's document stays the source of truth."""
    global recognition_day
    today = datetime.now().strftime("%Y-%m-%d")
    if recognition_day == today:
        return
    recognized_by_room.clear()
    started_by_room.clear()
    recognition_day = today
