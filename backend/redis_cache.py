import json
import os
import time
from typing import Any

import redis

_memory_cache: dict[str, tuple[float, Any]] = {}
MAX_MEMORY_CACHE_SIZE = 1000  # Fallback 메모리 캐시 최대 크기


_redis_client = None

def get_redis_client():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
        
    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        return None
    try:
        client = redis.Redis.from_url(url)
        client.ping()
        _redis_client = client
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
        
    # LRU 동작: 접근한 아이템을 가장 최신(맨 뒤)으로 갱신하여 메모리 유지 우선순위 높임
    _memory_cache.pop(key, None)
    _memory_cache[key] = item
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

    # 용량 제한 방어 로직: 캐시 크기가 제한을 넘으면 가장 오래된(맨 앞의) 아이템 삭제
    if len(_memory_cache) > MAX_MEMORY_CACHE_SIZE:
        try:
            oldest_key = next(iter(_memory_cache))
            _memory_cache.pop(oldest_key, None)
        except (StopIteration, RuntimeError):
            pass
