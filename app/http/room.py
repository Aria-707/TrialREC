"""Classroom name carried on the signed request."""
from flask import request

from app.http.gate import CLASSROOM_HEADER


def request_classroom():
    """Classroom the page signed for this request."""
    return (request.headers.get(CLASSROOM_HEADER) or "").strip()
