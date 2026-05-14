import os
import threading

import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql

from .keyword_config import BASE_DIR, ENTITY_REFERENCE_MAP, REFERENCE_LIMIT
from .keyword_normalize import normalize_for_match


_references_lock = threading.Lock()
_references: dict[str, list[str]] | None = None


def _load_env() -> None:
    load_dotenv()
    load_dotenv(BASE_DIR / ".env", override=False)


def build_postgres_dsn() -> str:
    _load_env()
    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url:
        return database_url

    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "postgres")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def fetch_distinct_values(conn, table: str, column: str) -> list[str]:
    stmt = sql.SQL(
        """
        SELECT DISTINCT {column}
        FROM {table}
        WHERE {column} IS NOT NULL AND TRIM(CAST({column} AS TEXT)) <> ''
        ORDER BY {column}
        LIMIT %s
        """
    ).format(
        table=sql.Identifier(table),
        column=sql.Identifier(column),
    )
    with conn.cursor() as cur:
        cur.execute(stmt, (REFERENCE_LIMIT,))
        return [str(row[0]).strip() for row in cur.fetchall() if row[0]]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = normalize_for_match(value)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def load_reference_values(force: bool = False) -> dict[str, list[str]]:
    global _references

    with _references_lock:
        if _references is not None and not force:
            return _references

        references: dict[str, list[str]] = {}
        try:
            with psycopg2.connect(build_postgres_dsn()) as conn:
                for label, (table, column) in ENTITY_REFERENCE_MAP.items():
                    try:
                        values = fetch_distinct_values(conn, table, column)
                    except Exception:
                        values = []
                    references[label] = _dedupe(values)
        except Exception:
            references = {label: [] for label in ENTITY_REFERENCE_MAP}

        _references = references
        return _references
