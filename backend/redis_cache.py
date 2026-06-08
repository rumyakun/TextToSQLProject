import json
import os
import time
from typing import Any

import redis

_memory_cache: dict[str, tuple[float, Any]] = {}


def get_redis_client():
    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        return None
    try:
        client = redis.Redis.from_url(url)
        client.ping()
        return client
    except Exception:
        return None


def get_cache(key):
    client = get_redis_client()
    if client is not None:
        val = client.get(key)
        if val:
            return json.loads(val)
        return None

    now = time.time()
    item = _memory_cache.get(key)
    if not item:
        return None
    expires_at, value = item
    if expires_at < now:
        _memory_cache.pop(key, None)
        return None
    return value


def set_cache(key, value, ttl_seconds=300):
    ttl_seconds = int(ttl_seconds) if ttl_seconds is not None else None
    client = get_redis_client()
    if client is not None:
        encoded = json.dumps(value, ensure_ascii=False)
        if ttl_seconds and ttl_seconds > 0:
            client.setex(key, ttl_seconds, encoded)
        else:
            client.set(key, encoded)
        return

    expires_at = time.time() + ttl_seconds if ttl_seconds and ttl_seconds > 0 else float("inf")
    _memory_cache[key] = (expires_at, value)
