"""Cookie and signature. These tests do not open the camera or write Firestore."""
import time

from kiosk_gate import (
    dump_classroom_cookie,
    read_classroom_cookie,
    sign_forward,
    signature_matches,
)


def test_cookie_round_trip_keeps_the_classroom():
    raw = dump_classroom_cookie("Laboratorio 1")
    assert read_classroom_cookie(raw) == "Laboratorio 1"
    assert read_classroom_cookie("no-es-una-ficha") == ""


def test_signature_accepts_a_fresh_forward_and_rejects_a_changed_classroom():
    headers = sign_forward("POST", "/recognize", "", b'{"image":"a"}', "Sala A")
    assert signature_matches(
        "POST",
        "/recognize",
        "",
        b'{"image":"a"}',
        "Sala A",
        headers["X-Kiosk-Time"],
        headers["X-Kiosk-Signature"],
    )
    assert not signature_matches(
        "POST",
        "/recognize",
        "",
        b'{"image":"a"}',
        "Sala B",
        headers["X-Kiosk-Time"],
        headers["X-Kiosk-Signature"],
    )


def test_old_signature_is_rejected():
    timestamp = str(int(time.time()) - 600)
    from kiosk_gate import _signature

    signature = _signature(timestamp, "Sala A", "GET", "/api/active_course", "", b"")
    assert not signature_matches(
        "GET", "/api/active_course", "", b"", "Sala A", timestamp, signature,
    )


def test_recognition_rejects_an_unsigned_frame():
    import routes

    client = routes.app.test_client()
    blocked = client.post("/recognize", json={"image": "a"})
    assert blocked.status_code == 403

    body = b'{"image":"a"}'
    headers = sign_forward("POST", "/recognize", "", body, "Sala Z")
    headers["Content-Type"] = "application/json"
    signed = client.post("/recognize", data=body, headers=headers)
    assert signed.status_code != 403


def test_browser_without_a_classroom_sees_setup_and_cannot_mark_attendance():
    import bff

    client = bff.app.test_client()
    home = client.get("/")
    assert home.status_code == 302
    assert "/setup" in home.headers["Location"]

    blocked = client.post("/recognize", json={"image": "abc"})
    assert blocked.status_code == 401
    assert client.get("/enroll").status_code == 302

    setup = client.post("/api/set_classroom", json={"classroom": "Sala A"})
    assert setup.status_code == 401

    client.set_cookie("kiosk", "ficha-falsa")
    assert client.get("/").status_code == 302

    client.set_cookie("kiosk", dump_classroom_cookie("Sala A"))
    assert client.get("/").status_code == 200
    enroll = client.get("/enroll")
    assert enroll.status_code == 200
