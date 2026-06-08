import hashlib
import json
import logging
import os
import re
import sys
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.query import Query
from redis.exceptions import ResponseError
from transformers import AutoModel, AutoTokenizer

from .keyword_extract import MASKABLE_ENTITY_LABELS
from .redis_cache import get_cache, get_redis_client, set_cache

try:
    from redis.commands.search.index_definition import IndexDefinition, IndexType
except ImportError:
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType


logger = logging.getLogger("uvicorn.error")

CACHE_VERSION = "kure-mask-v1"
DEFAULT_MODEL_NAME = "nlpai-lab/KURE-v1"
DEFAULT_THRESHOLD = 0.9
DEFAULT_TOP_K = 5
DEFAULT_SEARCH_POOL_SIZE = 50
DEFAULT_TTL_SECONDS = 60 * 60 * 24

INDEX_NAME = "idx:sql_generation_vectors"
DOC_PREFIX = "sqlgen:vector:"
PAYLOAD_PREFIX = "sqlgen:payload:"
MASK_RE = re.compile(r"<([A-Z_]+)>")
EXCLUSION_INTENT_RE = re.compile(
    r"(?:제외|빼고|뺀|빼줘|빼서|말고|아닌|아니고|아니라)"
)
DAY_SQL_VALUES = {
    "월": "월",
    "월요일": "월",
    "화": "화",
    "화요일": "화",
    "수": "수",
    "수요일": "수",
    "목": "목",
    "목요일": "목",
    "금": "금",
    "금요일": "금",
    "토": "토",
    "토요일": "토",
    "일": "일",
    "일요일": "일",
}
TIME_SQL_RE = re.compile(r"^\s*(?P<hour>[0-2]?\d):(?P<minute>[0-5]\d)(?::(?P<second>[0-5]\d))?\s*$")
KOREAN_TIME_RE = re.compile(
    r"(?P<hour>[0-2]?\d)\s*시(?:\s*(?P<minute>[0-5]?\d)\s*분?)?"
)

_model_lock = threading.Lock()
_model_bundle: tuple[Any, Any] | None = None
_index_ready = False
_index_lock = threading.Lock()


@dataclass
class VectorCacheHit:
    masked_query: str
    masked_sql: str
    similarity: float
    cache_key: str


def mask_labels(text: str) -> list[str]:
    return [label for label in MASK_RE.findall(text or "") if label in MASKABLE_ENTITY_LABELS]


def mask_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in mask_labels(text):
        counts[label] = counts.get(label, 0) + 1
    return counts


def has_duplicate_mask_type(text: str) -> bool:
    return any(count >= 2 for count in mask_counts(text).values())


def mask_signature(text: str) -> str:
    labels = sorted(set(mask_labels(text)))
    signature_parts = labels if labels else ["__NO_MASK__"]
    excluded_labels = excluded_mask_labels(text)
    if excluded_labels:
        signature_parts.extend(f"__EXCLUDE_{label}__" for label in excluded_labels)
    elif has_exclusion_intent(text):
        signature_parts.append("__EXCLUDE_INTENT__")
    return "|".join(signature_parts)


def has_exclusion_intent(text: str) -> bool:
    return bool(EXCLUSION_INTENT_RE.search(text or ""))


def excluded_mask_labels(text: str) -> list[str]:
    text = text or ""
    excluded_labels: set[str] = set()
    masks = [
        (match.start(), match.end(), match.group(1))
        for match in MASK_RE.finditer(text)
        if match.group(1) in MASKABLE_ENTITY_LABELS
    ]
    if not masks:
        return []

    for exclusion in EXCLUSION_INTENT_RE.finditer(text):
        preceding_masks = [mask for mask in masks if mask[1] <= exclusion.start()]
        if not preceding_masks:
            continue

        nearest_start, nearest_end, nearest_label = preceding_masks[-1]
        between = text[nearest_end:exclusion.start()]
        if len(between.strip()) <= 20:
            excluded_labels.add(nearest_label)

    return sorted(excluded_labels)


def exclusion_signature(text: str) -> str:
    labels = excluded_mask_labels(text)
    if labels:
        return "|".join(labels)
    return "__EXCLUDE_INTENT__" if has_exclusion_intent(text) else "__NO_EXCLUDE_INTENT__"


def is_vector_cacheable(masked_query: str) -> bool:
    return not has_duplicate_mask_type(masked_query)


def extract_mask_values(preprocessing: dict[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    entities = preprocessing.get("corrected_entities") or preprocessing.get("entities") or []
    for entity in sorted(entities, key=lambda item: int(item.get("start", 0))):
        label = entity.get("label")
        if label not in MASKABLE_ENTITY_LABELS or label in values:
            continue
        text = entity.get("corrected_text") or entity.get("text")
        if text is not None:
            values[label] = str(text)
    return values


def _normalize_day_sql_value(value: str) -> str:
    compact = re.sub(r"\s+", "", str(value).strip())
    direct_match = DAY_SQL_VALUES.get(compact)
    if direct_match:
        return direct_match

    for day_text, sql_value in sorted(DAY_SQL_VALUES.items(), key=lambda item: len(item[0]), reverse=True):
        if compact.startswith(day_text):
            return sql_value

    return str(value)


def _normalize_time_sql_value(value: str) -> str:
    text = str(value).strip()
    sql_time = TIME_SQL_RE.match(text)
    if sql_time:
        hour = int(sql_time.group("hour"))
        minute = int(sql_time.group("minute"))
        second = int(sql_time.group("second") or "0")
        return f"{hour:02d}:{minute:02d}:{second:02d}"

    korean_time = KOREAN_TIME_RE.search(text)
    if not korean_time:
        return text

    hour = int(korean_time.group("hour"))
    minute = int(korean_time.group("minute") or "0")
    if ("오후" in text or "저녁" in text or "밤" in text) and 1 <= hour <= 11:
        hour += 12
    elif "오전" in text and hour == 12:
        hour = 0
    elif 1 <= hour <= 7:
        hour += 12

    return f"{hour:02d}:{minute:02d}:00"


def _normalize_mask_sql_value(label: str, value: str) -> str:
    if label == "DAY":
        return _normalize_day_sql_value(value)
    if label == "TIME":
        return _normalize_time_sql_value(value)
    return str(value)


def materialize_masked_sql(masked_sql: str, mask_values: dict[str, str]) -> str:
    sql = masked_sql
    for label, value in mask_values.items():
        normalized_value = _normalize_mask_sql_value(label, value)
        sql = sql.replace(f"<{label}>", normalized_value.replace("'", "''"))
    return sql


def _ttl_seconds() -> int:
    raw = os.getenv("VECTOR_CACHE_TTL_SECONDS", str(DEFAULT_TTL_SECONDS)).strip()
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_TTL_SECONDS


def _threshold() -> float:
    raw = os.getenv("VECTOR_CACHE_THRESHOLD", str(DEFAULT_THRESHOLD)).strip()
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_THRESHOLD


def _top_k() -> int:
    raw = os.getenv("VECTOR_CACHE_TOP_K", str(DEFAULT_TOP_K)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_TOP_K


def _search_pool_size() -> int:
    raw = os.getenv("VECTOR_CACHE_SEARCH_POOL_SIZE", str(DEFAULT_SEARCH_POOL_SIZE)).strip()
    try:
        return max(_top_k(), int(raw))
    except ValueError:
        return DEFAULT_SEARCH_POOL_SIZE


def _vector_cache_enabled() -> bool:
    return os.getenv("VECTOR_CACHE_ENABLED", "1").strip().lower() not in {"0", "false", "no"}


def _ensure_utf8_console():
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if not stream or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def log_vector_cache_similarities(masked_query: str, signature: str, candidates: list[dict]):
    _ensure_utf8_console()
    payload = {
        "masked_query": masked_query,
        "mask_signature": signature,
        "threshold": _threshold(),
        "top_k": _top_k(),
        "candidates": candidates,
    }
    print(f"Vector cache similarities: {json.dumps(payload, ensure_ascii=False)}", flush=True)


def log_vector_cache_skip(masked_query: str, reason: str):
    _ensure_utf8_console()
    payload = {
        "masked_query": masked_query,
        "reason": reason,
    }
    print(f"Vector cache similarities skipped: {json.dumps(payload, ensure_ascii=False)}", flush=True)


def _load_model():
    global _model_bundle
    if _model_bundle is not None:
        return _model_bundle

    with _model_lock:
        if _model_bundle is not None:
            return _model_bundle

        model_name = os.getenv("KURE_EMBEDDING_MODEL", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        _model_bundle = (tokenizer, model)
        return _model_bundle


def embedding_dimension() -> int:
    _, model = _load_model()
    return int(model.config.hidden_size)


def embed_text(text: str) -> bytes:
    tokenizer, model = _load_model()
    with torch.no_grad():
        encoded = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=int(os.getenv("VECTOR_CACHE_MAX_LENGTH", "256")),
            return_tensors="pt",
        )
        output = model(**encoded)
        token_embeddings = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        normalized = F.normalize(pooled, p=2, dim=1)
    return normalized[0].cpu().numpy().astype("float32").tobytes()


def _escape_tag(value: str) -> str:
    return re.sub(r"([,.<>{}\[\]\"':;!@#$%^&*()\-+=~ |])", r"\\\1", value)


def _decode_redis_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _search_signature_documents(client, signature: str) -> list[dict[str, str]]:
    raw_result = client.execute_command(
        "FT.SEARCH",
        INDEX_NAME,
        f"@mask_signature:{{{_escape_tag(signature)}}}",
        "RETURN",
        "3",
        "masked_query",
        "cache_key",
        "mask_signature",
        "LIMIT",
        "0",
        str(_search_pool_size()),
    )

    documents: list[dict[str, str]] = []
    if isinstance(raw_result, dict):
        for item in raw_result.get(b"results", raw_result.get("results", [])):
            attributes = item.get(b"extra_attributes", item.get("extra_attributes", {}))
            documents.append(
                {
                    "id": _decode_redis_value(item.get(b"id", item.get("id", ""))),
                    "masked_query": _decode_redis_value(
                        attributes.get(b"masked_query", attributes.get("masked_query", ""))
                    ),
                    "cache_key": _decode_redis_value(
                        attributes.get(b"cache_key", attributes.get("cache_key", ""))
                    ),
                    "mask_signature": _decode_redis_value(
                        attributes.get(b"mask_signature", attributes.get("mask_signature", ""))
                    ),
                }
            )
        return documents

    for index in range(1, len(raw_result), 2):
        doc_id = _decode_redis_value(raw_result[index])
        fields = raw_result[index + 1]
        attributes = {
            _decode_redis_value(fields[field_index]): _decode_redis_value(fields[field_index + 1])
            for field_index in range(0, len(fields), 2)
        }
        documents.append(
            {
                "id": doc_id,
                "masked_query": attributes.get("masked_query", ""),
                "cache_key": attributes.get("cache_key", ""),
                "mask_signature": attributes.get("mask_signature", ""),
            }
        )

    return documents


def ensure_vector_index() -> bool:
    global _index_ready
    if _index_ready:
        return True

    client = get_redis_client()
    if client is None:
        return False

    with _index_lock:
        if _index_ready:
            return True
        try:
            client.ft(INDEX_NAME).info()
            _index_ready = True
            return True
        except ResponseError:
            pass
        except Exception as exc:
            logger.warning("Redis vector index check failed: %s", exc)
            return False

        try:
            client.ft(INDEX_NAME).create_index(
                fields=[
                    TextField("masked_query"),
                    TextField("cache_key"),
                    TagField("mask_signature"),
                    VectorField(
                        "embedding",
                        "HNSW",
                        {
                            "TYPE": "FLOAT32",
                            "DIM": embedding_dimension(),
                            "DISTANCE_METRIC": "COSINE",
                        },
                    ),
                ],
                definition=IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.HASH),
            )
            _index_ready = True
            return True
        except Exception as exc:
            logger.warning("Redis vector index creation failed: %s", exc)
            return False


def vector_index_exists() -> bool:
    global _index_ready
    if _index_ready:
        return True

    client = get_redis_client()
    if client is None:
        return False

    try:
        client.ft(INDEX_NAME).info()
        _index_ready = True
        return True
    except Exception:
        return False


def _cache_key(masked_query: str, masked_sql: str, signature: str) -> str:
    raw = f"{CACHE_VERSION}\0{signature}\0{masked_query}\0{masked_sql}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def store_vector_cache(masked_query: str, masked_sql: str) -> bool:
    if not _vector_cache_enabled() or not masked_query or not masked_sql:
        return False
    if not is_vector_cacheable(masked_query):
        return False
    if not ensure_vector_index():
        return False

    signature = mask_signature(masked_query)
    cache_key = _cache_key(masked_query, masked_sql, signature)
    payload_key = f"{PAYLOAD_PREFIX}{cache_key}"
    doc_key = f"{DOC_PREFIX}{cache_key}"
    ttl = _ttl_seconds()

    try:
        set_cache(
            payload_key,
            {
                "masked_query": masked_query,
                "masked_sql": masked_sql,
                "mask_signature": signature,
            },
            ttl_seconds=ttl,
        )
        client = get_redis_client()
        if client is None:
            return False
        client.hset(
            doc_key,
            mapping={
                "masked_query": masked_query,
                "masked_sql": masked_sql,
                "cache_key": payload_key,
                "mask_signature": signature,
                "embedding": embed_text(masked_query),
            },
        )
        if ttl > 0:
            client.expire(doc_key, ttl)
        return True
    except Exception as exc:
        logger.warning("Redis vector cache store failed: %s", exc)
        return False


def store_vector_cache_async(masked_query: str, masked_sql: str) -> bool:
    if os.getenv("VECTOR_CACHE_STORE_ASYNC", "1").strip().lower() in {"0", "false", "no"}:
        return store_vector_cache(masked_query, masked_sql)

    thread = threading.Thread(
        target=store_vector_cache,
        args=(masked_query, masked_sql),
        daemon=True,
    )
    thread.start()
    return True


def find_vector_cache(masked_query: str) -> VectorCacheHit | None:
    if not _vector_cache_enabled() or not masked_query:
        log_vector_cache_skip(masked_query, "cache_disabled_or_empty_query")
        return None
    if not is_vector_cacheable(masked_query):
        log_vector_cache_skip(masked_query, "not_vector_cacheable")
        return None
    if not vector_index_exists():
        log_vector_cache_skip(masked_query, "vector_index_not_found")
        return None

    signature = mask_signature(masked_query)
    client = get_redis_client()
    if client is None:
        log_vector_cache_skip(masked_query, "redis_unavailable")
        return None

    try:
        result_docs = _search_signature_documents(client, signature)
    except Exception as exc:
        logger.warning("Redis vector cache search failed: %s", exc)
        log_vector_cache_skip(masked_query, "search_failed")
        return None

    best_hit: VectorCacheHit | None = None
    candidates = []
    current_exclusion_signature = exclusion_signature(masked_query)
    query_embedding = np.frombuffer(embed_text(masked_query), dtype=np.float32)
    for doc in result_docs:
        embedding = client.hget(doc["id"], "embedding")
        if not embedding:
            continue
        candidate_embedding = np.frombuffer(embedding, dtype=np.float32)
        if candidate_embedding.shape != query_embedding.shape:
            continue

        similarity = float(np.dot(query_embedding, candidate_embedding))
        distance = 1.0 - similarity
        payload_key = doc["cache_key"]
        payload = get_cache(payload_key) if payload_key else None
        payload_masked_query = payload.get("masked_query") if payload else None
        intent_matched = (
            payload is not None
            and exclusion_signature(payload_masked_query or "") == current_exclusion_signature
        )
        candidate = {
            "cache_key": payload_key,
            "doc_masked_query": doc["masked_query"],
            "doc_mask_signature": doc["mask_signature"],
            "payload_masked_query": payload_masked_query,
            "payload_mask_signature": payload.get("mask_signature") if payload else None,
            "similarity": round(similarity, 6),
            "distance": round(distance, 6),
            "threshold_passed": similarity >= _threshold(),
            "payload_found": bool(payload),
            "intent_matched": intent_matched,
            "signature_matched": payload.get("mask_signature") == signature if payload else False,
        }
        candidates.append(candidate)

        if similarity < _threshold():
            continue

        if not payload:
            continue

        if payload.get("mask_signature") != signature:
            continue

        if not intent_matched:
            continue

        hit = VectorCacheHit(
            masked_query=payload_masked_query or doc["masked_query"],
            masked_sql=payload.get("masked_sql") or "",
            similarity=similarity,
            cache_key=payload_key,
        )
        if best_hit is None or hit.similarity > best_hit.similarity:
            best_hit = hit

    log_vector_cache_similarities(masked_query, signature, candidates)
    return best_hit
