import logging
import re
import time

from .db import run_query
from .keyword_extract import preprocess_query
from .llm import fix_sql, generate_sql
from .redis_cache import get_cache, set_cache
from .utils import log_query
from .validate import validate_generated_sql


logger = logging.getLogger("uvicorn.error")
SQL_GENERATION_CACHE_VERSION = "course-schedule-v5"
SQL_RESERVED_WORDS = {
    "cross",
    "full",
    "group",
    "having",
    "inner",
    "join",
    "left",
    "limit",
    "offset",
    "order",
    "outer",
    "right",
    "where",
}
SCHEDULE_FILTER_COLUMNS = {
    "classroom",
    "course_schedule",
    "day_of_week",
    "end_time",
    "start_time",
}


def enforce_limit(sql):
    if "limit" not in sql.lower():
        sql += " LIMIT 50"
    return sql


def _strip_trailing_semicolon(sql):
    return sql.strip().removesuffix(";").strip()


def _sql_literal(value):
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def _find_v_course_info_alias(sql):
    match = re.search(
        r"\b(?:from|join)\s+v_course_info(?:\s+as)?\s+([a-z_][a-z0-9_]*)\b",
        sql,
        flags=re.IGNORECASE,
    )
    if match:
        alias = match.group(1)
        if alias.lower() not in SQL_RESERVED_WORDS:
            return alias

    if re.search(r"\b(?:from|join)\s+v_course_info\b", sql, flags=re.IGNORECASE):
        return "v_course_info"

    return None


def _find_top_level_keyword(sql, keywords):
    depth = 0
    in_single_quote = False
    in_double_quote = False
    lowered = sql.lower()
    keyword_patterns = [
        (keyword, re.compile(rf"\b{re.escape(keyword)}\b", flags=re.IGNORECASE))
        for keyword in keywords
    ]

    index = 0
    while index < len(sql):
        char = sql[index]
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif not in_single_quote and not in_double_quote:
            if char == "(":
                depth += 1
            elif char == ")" and depth > 0:
                depth -= 1
            elif depth == 0:
                for keyword, pattern in keyword_patterns:
                    match = pattern.match(lowered, index)
                    if match:
                        return match.start(), keyword
        index += 1

    return None


def _insert_before_query_suffix(sql, clause):
    suffix = _find_top_level_keyword(
        sql,
        ("group by", "having", "order by", "limit", "offset"),
    )
    if not suffix:
        return f"{sql}\n{clause}"

    index = suffix[0]
    return f"{sql[:index]}\n{clause}{sql[index:]}"


def _remove_query_suffix(sql):
    suffix = _find_top_level_keyword(sql, ("order by", "limit", "offset"))
    if not suffix:
        return sql
    return sql[: suffix[0]].strip()


def _extract_limit(sql, default=50):
    match = re.search(r"\blimit\s+(\d+)\b", sql, flags=re.IGNORECASE)
    if not match:
        return default
    return int(match.group(1))


def _where_clause(sql):
    where = _find_top_level_keyword(sql, ("where",))
    if not where:
        return ""

    suffix = _find_top_level_keyword(
        sql[where[0] + len("where") :],
        ("group by", "having", "order by", "limit", "offset"),
    )
    where_start = where[0] + len("where")
    if not suffix:
        return sql[where_start:]
    return sql[where_start : where_start + suffix[0]]


def _has_schedule_filter(sql):
    where = _where_clause(sql)
    if not where:
        return False
    columns = "|".join(re.escape(column) for column in sorted(SCHEDULE_FILTER_COLUMNS))
    return bool(
        re.search(
            rf"(?:\b[a-z_][a-z0-9_]*\.)?\b(?:{columns})\b",
            where,
            flags=re.IGNORECASE,
        )
    )


def _v_course_info_source_ref(sql):
    match = re.search(
        r"\bfrom\s+v_course_info(?P<tail>\s+(?:as\s+)?[a-z_][a-z0-9_]*)?",
        sql,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    tail = (match.group("tail") or "").strip()
    if not tail:
        return "v_course_info"

    alias_match = re.match(r"(?:as\s+)?([a-z_][a-z0-9_]*)$", tail, flags=re.IGNORECASE)
    if not alias_match:
        return "v_course_info"

    alias = alias_match.group(1)
    if alias.lower() in SQL_RESERVED_WORDS:
        return "v_course_info"
    return alias


def expand_schedule_filtered_course_rows(sql):
    base_sql = _strip_trailing_semicolon(sql)
    normalized = base_sql.lower()
    if "v_course_info" not in normalized or not _has_schedule_filter(base_sql):
        return sql
    if re.search(r"\b(group\s+by|having)\b", base_sql, flags=re.IGNORECASE):
        return sql

    source_ref = _v_course_info_source_ref(base_sql)
    if not source_ref:
        return sql

    key_sql = _remove_query_suffix(base_sql)
    key_sql = re.sub(
        r"^\s*select\b.*?\bfrom\b",
        f"SELECT DISTINCT {source_ref}.subject_code, {source_ref}.section FROM",
        key_sql,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )
    expanded_limit = min(max(_extract_limit(base_sql) * 4, 50), 200)

    return f"""
SELECT c.*
FROM v_course_info AS c
JOIN (
    {key_sql}
) AS matched_courses
  ON c.subject_code = matched_courses.subject_code
 AND c.section = matched_courses.section
ORDER BY c.subject_code, c.section, c.day_of_week, c.start_time
LIMIT {expanded_limit}
""".strip()


def exclude_completed_courses_sql(sql, student_id):
    base_sql = _strip_trailing_semicolon(sql)
    course_alias = _find_v_course_info_alias(base_sql)
    if not course_alias:
        raise Exception("v_course_info is required to exclude completed courses")

    predicate = (
        "NOT EXISTS (\n"
        "    SELECT 1\n"
        "    FROM enrollment AS a\n"
        f"    WHERE a.student_id = {_sql_literal(student_id)}\n"
        f"      AND a.subject_code = {course_alias}.subject_code\n"
        ")"
    )

    if _find_top_level_keyword(base_sql, ("where",)):
        return _insert_before_query_suffix(base_sql, f"AND {predicate}")

    return _insert_before_query_suffix(base_sql, f"WHERE {predicate}")


def validate_sql(sql, requested_student_id=None, query=None):
    result = validate_generated_sql(sql, requested_student_id=requested_student_id, query=query)
    if not result["ok"]:
        raise Exception(result["reason"] or "SQL validation failed")
    return True


def build_executable_sql(sql, exclude_completed_courses=False, student_id=None):
    expanded_sql = expand_schedule_filtered_course_rows(sql)
    if exclude_completed_courses:
        return exclude_completed_courses_sql(expanded_sql, student_id)
    return expanded_sql


def _with_query_context(response, query, normalized_query, preprocessing):
    response["query"] = query
    response["normalized_query"] = normalized_query
    response["preprocessing"] = preprocessing
    return response


def process(query, exclude_completed_courses=False, student_id=None):
    started_total = time.perf_counter()
    timings = {
        "cache_lookup_ms": 0,
        "preprocess_ms": 0,
        "llm_ms": 0,
        "validate_ms": 0,
        "db_ms": 0,
        "total_ms": 0,
    }

    t_pre = time.perf_counter()
    preprocessing = preprocess_query(query)
    normalized_query = preprocessing["corrected_query"]
    timings["preprocess_ms"] = int((time.perf_counter() - t_pre) * 1000)

    t0 = time.perf_counter()
    cache_key = f"{SQL_GENERATION_CACHE_VERSION}::{normalized_query}"
    if exclude_completed_courses:
        cache_key = f"{SQL_GENERATION_CACHE_VERSION}::{normalized_query}::exclude_completed::{student_id}"

    cached = get_cache(cache_key)
    timings["cache_lookup_ms"] = int((time.perf_counter() - t0) * 1000)
    if cached:
        cached["cache_hit"] = True
        cached["timings_ms"] = {
            "cache_lookup_ms": timings["cache_lookup_ms"],
            "preprocess_ms": timings["preprocess_ms"],
            "llm_ms": 0,
            "validate_ms": 0,
            "db_ms": 0,
            "total_ms": int((time.perf_counter() - started_total) * 1000),
        }
        return _with_query_context(cached, query, normalized_query, preprocessing)

    sql = None

    try:
        t_llm = time.perf_counter()
        sql = generate_sql(normalized_query)
        timings["llm_ms"] = int((time.perf_counter() - t_llm) * 1000)
        logger.info("LLM generated SQL | query=%r | sql=%s", normalized_query, sql)

        if not sql or sql == "UNKNOWN":
            return {"error": "질문이 모호합니다."}

        sql = enforce_limit(sql)
        t_val = time.perf_counter()
        validate_sql(sql, query=normalized_query)
        timings["validate_ms"] = int((time.perf_counter() - t_val) * 1000)

        executable_sql = build_executable_sql(sql, exclude_completed_courses, student_id)
        if exclude_completed_courses or executable_sql != sql:
            t_val = time.perf_counter()
            validate_sql(
                executable_sql,
                requested_student_id=str(student_id) if exclude_completed_courses else None,
                query=normalized_query,
            )
            timings["validate_ms"] += int((time.perf_counter() - t_val) * 1000)

        try:
            t_db = time.perf_counter()
            result = run_query(executable_sql)
            timings["db_ms"] = int((time.perf_counter() - t_db) * 1000)
        except RuntimeError as db_err:
            if "DATABASE_URL is not set" in str(db_err):
                timings["total_ms"] = int((time.perf_counter() - started_total) * 1000)
                res = _with_query_context(
                    {
                        "sql": executable_sql,
                        "data": [],
                        "warning": "DATABASE_URL is not set. SQL only mode is active.",
                        "cache_hit": False,
                        "timings_ms": timings,
                    },
                    query,
                    normalized_query,
                    preprocessing,
                )
                set_cache(cache_key, res)
                log_query(normalized_query, executable_sql, True)
                return res
            raise

        timings["total_ms"] = int((time.perf_counter() - started_total) * 1000)
        res = _with_query_context(
            {
                "sql": executable_sql,
                "data": result,
                "cache_hit": False,
                "timings_ms": timings,
            },
            query,
            normalized_query,
            preprocessing,
        )

        set_cache(cache_key, res)
        log_query(normalized_query, executable_sql, True)

        return res

    except Exception as e:
        # If we could not generate an initial SQL, surface the error clearly.
        if not sql:
            log_query(normalized_query, "", False)
            return {"error": str(e)}

        try:
            t_fix = time.perf_counter()
            fixed = fix_sql(normalized_query, sql, str(e))
            timings["llm_ms"] += int((time.perf_counter() - t_fix) * 1000)
            logger.info(
                "LLM fixed SQL | query=%r | original_sql=%s | fixed_sql=%s",
                normalized_query,
                sql,
                fixed,
            )

            fixed = enforce_limit(fixed)
            t_val = time.perf_counter()
            validate_sql(fixed, query=normalized_query)
            timings["validate_ms"] += int((time.perf_counter() - t_val) * 1000)

            executable_fixed = build_executable_sql(fixed, exclude_completed_courses, student_id)
            if exclude_completed_courses or executable_fixed != fixed:
                t_val = time.perf_counter()
                validate_sql(
                    executable_fixed,
                    requested_student_id=str(student_id) if exclude_completed_courses else None,
                    query=normalized_query,
                )
                timings["validate_ms"] += int((time.perf_counter() - t_val) * 1000)

            try:
                t_db = time.perf_counter()
                result = run_query(executable_fixed)
                timings["db_ms"] += int((time.perf_counter() - t_db) * 1000)
            except RuntimeError as db_err:
                if "DATABASE_URL is not set" in str(db_err):
                    timings["total_ms"] = int((time.perf_counter() - started_total) * 1000)
                    res = _with_query_context(
                        {
                            "sql": executable_fixed,
                            "data": [],
                            "warning": "DATABASE_URL is not set. SQL only mode is active.",
                            "cache_hit": False,
                            "timings_ms": timings,
                        },
                        query,
                        normalized_query,
                        preprocessing,
                    )
                    set_cache(cache_key, res)
                    log_query(normalized_query, executable_fixed, True)
                    return res
                raise

            timings["total_ms"] = int((time.perf_counter() - started_total) * 1000)
            return _with_query_context(
                {
                    "sql": executable_fixed,
                    "data": result,
                    "cache_hit": False,
                    "timings_ms": timings,
                },
                query,
                normalized_query,
                preprocessing,
            )

        except Exception as e2:
            log_query(normalized_query, sql, False)
            return {"error": str(e2)}
