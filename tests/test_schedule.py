"""Class window. These calls do not read or write Firestore."""
from datetime import datetime

from schedule import arrival_from_start, next_occurrence, schedule_matches_classroom

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


def test_past_clock_moves_to_next_week():
    moment = datetime(2026, 10, 12, 10, 0)
    assert moment.strftime("%A") == "Monday"
    assert next_occurrence("Lunes", "09:00", moment) == datetime(2026, 10, 19, 9, 0)
    assert next_occurrence("Lunes", "09:00", datetime(2026, 10, 12, 8, 0)) == datetime(
        2026, 10, 12, 9, 0,
    )
