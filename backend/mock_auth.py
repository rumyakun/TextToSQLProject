from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "mock_students.json"


def _load_data() -> dict[str, Any]:
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _secret() -> bytes:
    return os.getenv("MOCK_AUTH_SECRET", "dev-cnu-portal-mock-secret").encode("utf-8")


def _sign(student_no: str) -> str:
    return hmac.new(_secret(), student_no.encode("utf-8"), hashlib.sha256).hexdigest()


def make_access_token(student_no: str) -> str:
    return f"mock-cnu.{student_no}.{_sign(student_no)}"


def verify_access_token(token: str) -> str | None:
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != "mock-cnu":
        return None

    student_no = parts[1]
    signature = parts[2]
    if not hmac.compare_digest(signature, _sign(student_no)):
        return None
    return student_no


def authenticate(student_no: str, password: str) -> dict[str, Any] | None:
    data = _load_data()
    expected_hash = None
    for credential in data.get("credentials", []):
        if credential.get("studentNo") == student_no:
            expected_hash = credential.get("passwordHash")
            break

    if not expected_hash or not hmac.compare_digest(expected_hash, _hash_password(password)):
        return None

    return get_student(student_no)


def get_student(student_no: str) -> dict[str, Any] | None:
    data = _load_data()
    for student in data.get("students", []):
        if student.get("studentNo") == student_no:
            return student
    return None
