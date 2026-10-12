"""Cosine cutoff. These calls do not open the camera."""
from face import choose_match


def test_clear_winner_is_a_match():
    national_id, score, kind = choose_match([(0.80, "1"), (0.70, "2")])
    assert kind == "match"
    assert national_id == "1"
    assert score == 0.80


def test_close_second_is_ambiguous():
    national_id, score, kind = choose_match([(0.75, "1"), (0.70, "2")])
    assert kind == "ambiguous"
    assert national_id is None
    assert score == 0.75


def test_low_score_is_unknown():
    national_id, score, kind = choose_match([(0.50, "1")])
    assert kind == "unknown"
    assert national_id is None
    assert score == 0.50


def test_exact_cutoff_is_a_match():
    national_id, score, kind = choose_match([(0.60, "1")])
    assert kind == "match"
    assert national_id == "1"
    assert score == 0.60
