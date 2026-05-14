import datetime
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _default_log_path() -> Path:
    env = os.getenv("QUERY_LOG_PATH", "").strip()
    if env:
        return Path(env).expanduser()
    return _PROJECT_ROOT / "backend-log.txt"


def log_query(q, sql, success, note: str = ""):
    """Append one line. `note` 예: cache_hit, ambiguous 등."""
    log_path = _default_log_path()
    line = f"{datetime.datetime.now()} | {q} | {sql} | {success}"
    if note:
        line += f" | {note}"
    line += "\n"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError as e:
        print(f"[log_query] skip write ({log_path}): {e}", file=sys.stderr)
