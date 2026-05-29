from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pathlib import Path
import re

from .keyword_extract import warmup_keyword_normalizer
from .llm import warmup_model
from .mock_auth import authenticate, get_student, make_access_token, verify_access_token
from .process import process
from .db import run_query

load_dotenv()  # load workspace .env if present
load_dotenv(Path(__file__).with_name(".env"), override=False)  # load backend-specific env

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class LoginRequest(BaseModel):
    studentNo: str = Field(min_length=1, max_length=30)
    password: str = Field(min_length=1, max_length=200)


class SignupRequest(BaseModel):
    studentNo: str = Field(min_length=1, max_length=30)
    name: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)
    departmentCode: str = Field(min_length=1, max_length=30)


def _auth_error(message: str = "Authentication failed"):
    return HTTPException(
        status_code=401,
        detail={"error": {"code": "UNAUTHORIZED", "message": message}},
    )


def _student_from_authorization(authorization: str | None):
    if not authorization or not authorization.startswith("Bearer "):
        raise _auth_error("Login is required.")

    token = authorization.removeprefix("Bearer ").strip()
    student_no = verify_access_token(token)
    if not student_no:
        raise _auth_error("Invalid access token.")

    student = get_student(student_no)
    if not student:
        raise _auth_error("Student not found.")
    return student


DAY_ALIASES = {
    "mon": "MON",
    "monday": "MON",
    "월": "MON",
    "월요일": "MON",
    "tue": "TUE",
    "tues": "TUE",
    "tuesday": "TUE",
    "화": "TUE",
    "화요일": "TUE",
    "wed": "WED",
    "wednesday": "WED",
    "수": "WED",
    "수요일": "WED",
    "thu": "THU",
    "thur": "THU",
    "thurs": "THU",
    "thursday": "THU",
    "목": "THU",
    "목요일": "THU",
    "fri": "FRI",
    "friday": "FRI",
    "금": "FRI",
    "금요일": "FRI",
    "sat": "SAT",
    "saturday": "SAT",
    "토": "SAT",
    "토요일": "SAT",
    "sun": "SUN",
    "sunday": "SUN",
    "일": "SUN",
    "일요일": "SUN",
}
DAY_PATTERN = "|".join(
    re.escape(day)
    for day in sorted(DAY_ALIASES, key=len, reverse=True)
)
TIME_RANGE_PATTERN = re.compile(
    r"([0-2]?\d)(?::([0-5]\d))?\s*(?:-|~|–|—|to)\s*([0-2]?\d)(?::([0-5]\d))?",
    re.IGNORECASE,
)


def _normalize_day(value) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return DAY_ALIASES.get(normalized)


def _format_time(hour: str, minute: str | None = None) -> str:
    return f"{int(hour):02d}:{int(minute or '00'):02d}"


def _is_valid_time_range(start: str, end: str) -> bool:
    start_hour, start_minute = [int(part) for part in start.split(":")]
    end_hour, end_minute = [int(part) for part in end.split(":")]
    return (end_hour, end_minute) > (start_hour, start_minute)


def _dedupe_schedule(slots: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for slot in slots:
        key = (slot.get("day"), slot.get("start"), slot.get("end"), slot.get("room"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(slot)
    return deduped


def _parse_credits(value) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else 0


def _compact_value(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "null" else text


def _split_course_list(value) -> list[str]:
    text = _compact_value(value)
    if not text or text in {"없음", "None", "none", "-"}:
        return []
    return [
        part.strip()
        for part in re.split(r"[,;/\n]+", text)
        if part.strip() and part.strip() not in {"없음", "None", "none", "-"}
    ]


def _display_time(value: str) -> str:
    match = re.match(r"^([0-2]?\d):([0-5]\d)(?::[0-5]\d)?$", value)
    if not match:
        return value
    return f"{int(match.group(1)):02d}:{match.group(2)}"


def _parse_schedule_from_fields(row: dict) -> list[dict]:
    day = _normalize_day(row.get("day_of_week") or row.get("day"))
    start_value = row.get("start_time") or row.get("start")
    end_value = row.get("end_time") or row.get("end")
    if not day or start_value is None or end_value is None:
        return []

    start_match = re.search(r"([0-2]?\d)(?::([0-5]\d))?", str(start_value))
    end_match = re.search(r"([0-2]?\d)(?::([0-5]\d))?", str(end_value))
    if not start_match or not end_match:
        return []

    start = _format_time(start_match.group(1), start_match.group(2))
    end = _format_time(end_match.group(1), end_match.group(2))
    if not _is_valid_time_range(start, end):
        return []

    slot = {"day": day, "start": start, "end": end}
    room = row.get("classroom") or row.get("room")
    if room:
        slot["room"] = str(room)
    return [slot]


def _parse_schedule_from_lecture_time(lecture_time: str) -> list[dict]:
    slots = []
    for part in re.split(r"[,;/\n]+", lecture_time):
        days = [
            _normalize_day(match.group(0))
            for match in re.finditer(DAY_PATTERN, part, flags=re.IGNORECASE)
        ]
        days = [day for day in days if day]
        time_match = TIME_RANGE_PATTERN.search(part)
        if not days or not time_match:
            continue

        start = _format_time(time_match.group(1), time_match.group(2))
        end = _format_time(time_match.group(3), time_match.group(4))
        if not _is_valid_time_range(start, end):
            continue

        for day in days:
            slots.append({"day": day, "start": start, "end": end})

    return _dedupe_schedule(slots)


def _schedule_from_row(row: dict, lecture_time: str) -> list[dict]:
    slots = _parse_schedule_from_fields(row)
    if slots:
        return _dedupe_schedule(slots)
    return _parse_schedule_from_lecture_time(lecture_time)


def _lecture_time_from_row(row: dict) -> str:
    lecture_time = _compact_value(row.get("lecture_time") or row.get("table_schedule"))
    if lecture_time:
        return lecture_time

    day = _compact_value(row.get("day_of_week") or row.get("day"))
    start = _compact_value(row.get("start_time") or row.get("start"))
    end = _compact_value(row.get("end_time") or row.get("end"))
    if not day or not start or not end:
        return ""
    start = _display_time(start)
    end = _display_time(end)
    return f"{day} {start}-{end}"


def _location_from_row(row: dict) -> str:
    return _compact_value(row.get("classroom") or row.get("room"))


def _course_detail_items(row: dict) -> list[dict]:
    labels = [
        ("course_year", "학년"),
        ("subject_code", "과목코드"),
        ("section", "분반"),
        ("subject_name", "과목명"),
        ("category", "이수구분"),
        ("credit_hours", "학점"),
        ("target_year", "대상학년"),
        ("professor", "교수"),
        ("capacity", "정원"),
        ("enrolled", "수강인원"),
        ("grading_method", "성적평가"),
        ("eval_type", "평가방식"),
        ("class_mode", "수업방식"),
        ("dept_name", "학과"),
        ("day_of_week", "요일"),
        ("start_time", "시작"),
        ("end_time", "종료"),
        ("table_schedule", "시간표"),
        ("classroom", "강의실"),
        ("prereq_subject_codes", "선수과목 코드"),
        ("prereq_subject_names", "선수과목명"),
    ]
    details = []
    for key, label in labels:
        value = _compact_value(row.get(key))
        if value:
            details.append({"label": label, "value": value})
    return details


def _merge_course_item(existing: dict, item: dict) -> dict:
    seen_slots = {
        (slot.get("day"), slot.get("start"), slot.get("end"), slot.get("room"))
        for slot in existing.get("schedule", [])
    }
    for slot in item.get("schedule", []):
        key = (slot.get("day"), slot.get("start"), slot.get("end"), slot.get("room"))
        if key not in seen_slots:
            existing["schedule"].append(slot)
            seen_slots.add(key)

    existing_times = [time for time in existing.get("lectureTime", "").split(", ") if time]
    for time in [time for time in item.get("lectureTime", "").split(", ") if time]:
        if time not in existing_times:
            existing_times.append(time)
    existing["lectureTime"] = ", ".join(existing_times)

    existing_locations = [location for location in existing.get("locationText", "").split(", ") if location]
    for location in [location for location in item.get("locationText", "").split(", ") if location]:
        if location not in existing_locations:
            existing_locations.append(location)
    existing["locationText"] = ", ".join(existing_locations)

    for key in ("prerequisiteCourseCodes", "prerequisiteCourseNames"):
        existing_values = existing.get(key, [])
        for value in item.get(key, []):
            if value not in existing_values:
                existing_values.append(value)
        existing[key] = existing_values

    seen_details = {
        (detail.get("label"), detail.get("value"))
        for detail in existing.get("details", [])
    }
    for detail in item.get("details", []):
        key = (detail.get("label"), detail.get("value"))
        if key not in seen_details:
            existing["details"].append(detail)
            seen_details.add(key)

    return existing


def _group_course_items(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        item = _to_course_item(row)
        key = item["courseId"]
        grouped[key] = _merge_course_item(grouped[key], item) if key in grouped else item
    return list(grouped.values())


def _to_course_item(row: dict):
    subject_code = row.get("subject_code") or row.get("course_id") or ""
    section = row.get("section") or ""
    course_id = f"{subject_code}-{section}" if section else subject_code
    capacity = row.get("capacity")
    enrolled = row.get("enrolled")

    tags = []
    lecture_time = _lecture_time_from_row(row)
    location = _location_from_row(row)
    if lecture_time:
        tags.append(lecture_time)
    category = _compact_value(row.get("category"))
    if category:
        tags.append(category)

    return {
        "courseId": str(course_id or ""),
        "name": str(row.get("subject_name") or row.get("course_name") or course_id or "Untitled"),
        "departmentCode": str(row.get("dept_code") or ""),
        "departmentName": row.get("dept_name"),
        "credits": _parse_credits(row.get("credit_hours") or row.get("credits")),
        "professor": str(row.get("professor") or "-"),
        "capacity": capacity,
        "enrolled": enrolled,
        "schedule": _schedule_from_row(row, lecture_time),
        "lectureTime": lecture_time,
        "locationText": location,
        "section": str(section),
        "tags": tags,
        "prerequisiteCourseCodes": _split_course_list(row.get("prereq_subject_codes")),
        "prerequisiteCourseNames": _split_course_list(row.get("prereq_subject_names")),
        "details": _course_detail_items(row),
    }


@app.get("/api/v1/health")
def health():
    return {"ok": True}


@app.on_event("startup")
def startup_warmup():
    # Warm up local models once to reduce first-query latency.
    app.state.warmup = {
        "llm": warmup_model(),
        "keyword_normalizer": warmup_keyword_normalizer(),
    }


@app.get("/api/v1/warmup")
def warmup_status():
    return getattr(app.state, "warmup", {"ok": False, "error": "warmup not run"})


@app.post("/api/v1/auth/login")
def login(req: LoginRequest):
    student = authenticate(req.studentNo.strip(), req.password)
    if not student:
        raise _auth_error("학번 또는 비밀번호가 올바르지 않습니다.")

    return {
        "accessToken": make_access_token(student["studentNo"]),
        "expiresIn": 60 * 60 * 8,
        "user": student,
    }


@app.post("/api/v1/auth/signup")
def signup(_: SignupRequest):
    raise HTTPException(
        status_code=400,
        detail={
            "error": {
                "code": "PORTAL_SIGNUP_UNAVAILABLE",
                "message": "포털 연동 계정은 회원가입 없이 충남대 포털 로그인으로 이용합니다.",
            }
        },
    )


@app.post("/api/v1/auth/logout")
def logout():
    return {"ok": True}


@app.get("/api/v1/students/me")
def get_me(authorization: str | None = Header(default=None)):
    return {"user": _student_from_authorization(authorization)}


@app.get("/api/v1/students/me/profile")
def get_my_profile(authorization: str | None = Header(default=None)):
    return _student_from_authorization(authorization)


@app.get("/api/v1/courses")
def list_courses(
    page: int = 1,
    pageSize: int = 100,
    keyword: str | None = None,
):
    page = max(page, 1)
    pageSize = min(max(pageSize, 1), 200)
    offset = (page - 1) * pageSize
    filters = []
    normalized_keyword = (keyword or "").strip()
    if normalized_keyword:
        escaped_keyword = (
            normalized_keyword
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
            .replace("'", "''")
        )
        filters.append(f"subject_name ILIKE '%{escaped_keyword}%' ESCAPE '\\'")

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    sql = f"""
        SELECT
            course_year,
            subject_code,
            section,
            subject_name,
            category,
            credit_hours,
            target_year,
            professor,
            capacity,
            enrolled,
            grading_method,
            eval_type,
            class_mode,
            dept_name,
            table_schedule,
            classroom,
            prereq_subject_codes,
            prereq_subject_names
        FROM v_course_info
        {where_clause}
        ORDER BY subject_code, section
        LIMIT {pageSize} OFFSET {offset}
    """

    try:
        rows = run_query(sql)
    except RuntimeError as e:
        if "DATABASE_URL is not set" in str(e):
            return {
                "items": [],
                "page": page,
                "pageSize": pageSize,
                "total": 0,
                "warning": "DATABASE_URL is not set. Mock course data should be used by the frontend.",
            }
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    items = _group_course_items(rows)[:pageSize]

    return {
        "items": items,
        "page": page,
        "pageSize": pageSize,
        "total": len(items) + offset,
    }


def _run_query(req: QueryRequest):
    try:
        return process(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/chat/query")
def chat_query_api(req: QueryRequest):
    return _run_query(req)


@app.post("/api/v1/query")
def query_api(req: QueryRequest):
    return _run_query(req)
