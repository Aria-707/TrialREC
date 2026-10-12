"""Class window. These calls do not read or write Firestore."""
from datetime import datetime

from schedule import (
    arrival_from_start,
    event_finish_at,
    event_window_matches,
    next_occurrence,
    schedule_matches_classroom,
    valid_event_clocks,
)

SLOT = {
    "day": "Lunes",
    "iniTime": "10:00",
    "endTime": "11:30",
    "classroom": "E-102",
}


def _match(clock, day="Lunes"):
    return schedule_matches_classroom([SLOT], day, "Monday", clock, "E-102")


def test_window_includes_five_minutes_before_and_thirty_after():
    assert _match("09:55") == "10:00"
    assert _match("10:30") == "10:00"
    assert _match("09:54") is None
    assert _match("10:31") is None


def test_other_day_or_room_does_not_match():
    assert _match("10:00", day="Martes") is None
    assert schedule_matches_classroom(
        [SLOT], "Lunes", "Monday", "10:00", "E-201",
    ) is None


def test_late_starts_after_fifteen_minutes():
    early, early_flag = arrival_from_start("10:00", datetime(2026, 10, 12, 9, 58))
    on_time, on_time_flag = arrival_from_start("10:00", datetime(2026, 10, 12, 10, 0))
    arrived, arrived_flag = arrival_from_start("10:00", datetime(2026, 10, 12, 10, 10))
    late, late_flag = arrival_from_start("10:00", datetime(2026, 10, 12, 10, 16))
    assert (early, early_flag) == ("early", False)
    assert (on_time, on_time_flag) == ("on_time", False)
    assert (arrived, arrived_flag) == ("arrived", False)
    assert (late, late_flag) == ("late", True)


EVENT_SLOT = {
    "event": True,
    "date": "2026-10-12",
    "iniTime": "10:00",
    "endTime": "12:00",
    "classroom": "E-102",
    "day": "",
}


def test_event_window_opens_five_minutes_before_and_closes_at_end():
    assert event_window_matches(EVENT_SLOT, "2026-10-12", "09:55", "E-102") == "10:00"
    assert event_window_matches(EVENT_SLOT, "2026-10-12", "12:00", "E-102") == "10:00"
    assert event_window_matches(EVENT_SLOT, "2026-10-12", "09:54", "E-102") is None
    assert event_window_matches(EVENT_SLOT, "2026-10-12", "12:01", "E-102") is None
    assert event_window_matches(EVENT_SLOT, "2026-10-13", "10:00", "E-102") is None
    assert schedule_matches_classroom(
        [EVENT_SLOT], "Lunes", "Monday", "10:00", "E-102",
    ) is None


def test_closed_event_stays_shut_and_finish_is_the_next_minute():
    closed = dict(EVENT_SLOT)
    closed["closed"] = True
    assert event_window_matches(closed, "2026-10-12", "10:00", "E-102") is None
    assert event_finish_at("2026-10-12", "12:00") == datetime(2026, 10, 12, 12, 1)


def test_event_clocks_need_an_end_after_the_start():
    assert valid_event_clocks("09:00", "10:00") is True
    assert valid_event_clocks("10:00", "10:00") is False
    assert valid_event_clocks("10:00", "09:00") is False
    assert valid_event_clocks("10", "12:00") is False


def test_past_clock_moves_to_next_week():
    moment = datetime(2026, 10, 12, 10, 0)
    assert moment.strftime("%A") == "Monday"
    assert next_occurrence("Lunes", "09:00", moment) == datetime(2026, 10, 19, 9, 0)
    assert next_occurrence("Lunes", "09:00", datetime(2026, 10, 12, 8, 0)) == datetime(
        2026, 10, 12, 9, 0,
    )
