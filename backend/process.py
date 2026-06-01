import logging
import time

from .db import run_query
from .keyword_extract import preprocess_query
from .llm import fix_sql, generate_sql
from .redis_cache import get_cache, set_cache
from .utils import log_query
from .validate import validate_generated_sql


logger = logging.getLogger("uvicorn.error")
SQL_GENERATION_CACHE_VERSION = "course-schedule-v3"


def enforce_limit(sql):
    if "limit" not in sql.lower():
        sql += " LIMIT 50"
    return sql


def _strip_trailing_semicolon(sql):
    return sql.strip().removesuffix(";").strip()


def _sql_literal(value):
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def exclude_completed_courses_sql(sql, student_id):
    base_sql = _strip_trailing_semicolon(sql)
    return (
        "SELECT b.*\n"
        f"FROM ({base_sql}) AS b\n"
        "WHERE b.subject_code NOT IN (\n"
        "    SELECT a.subject_code\n"
        "    FROM enrollment AS a\n"
        f"    WHERE a.student_id = {_sql_literal(student_id)}\n"
        ")"
    )


def validate_sql(sql, requested_student_id=None, query=None):
    result = validate_generated_sql(sql, requested_student_id=requested_student_id, query=query)
    if not result["ok"]:
        raise Exception(result["reason"] or "SQL validation failed")
    return True


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

        executable_sql = (
            exclude_completed_courses_sql(sql, student_id)
            if exclude_completed_courses
            else sql
        )
        if exclude_completed_courses:
            t_val = time.perf_counter()
            validate_sql(executable_sql, requested_student_id=str(student_id), query=normalized_query)
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

            executable_fixed = (
                exclude_completed_courses_sql(fixed, student_id)
                if exclude_completed_courses
                else fixed
            )
            if exclude_completed_courses:
                t_val = time.perf_counter()
                validate_sql(executable_fixed, requested_student_id=str(student_id), query=normalized_query)
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
