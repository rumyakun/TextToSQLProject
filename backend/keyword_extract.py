import argparse
import json
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg2
import torch
from dotenv import load_dotenv
from psycopg2 import sql
from rapidfuzz import fuzz, process as fuzz_process
from transformers import AutoModelForTokenClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
load_dotenv()
load_dotenv(BASE_DIR / ".env", override=False)
MODEL_DIR = BASE_DIR / "course_custom_ner_model" / "models" / "course-custom-ner"
REFERENCE_LIMIT = int(os.getenv("KEYWORD_REFERENCE_LIMIT", "20000"))
DEFAULT_MIN_SCORE = float(os.getenv("KEYWORD_CORRECTION_MIN_SCORE", "78"))
SEQUENCE_MIN_SCORE = float(os.getenv("KEYWORD_SEQUENCE_MIN_SCORE", "72"))
SEQUENCE_AMBIGUITY_MARGIN = float(os.getenv("KEYWORD_SEQUENCE_AMBIGUITY_MARGIN", "4"))
SEQUENCE_MAX_QUERY_LENGTH = int(os.getenv("KEYWORD_SEQUENCE_MAX_QUERY_LENGTH", "6"))

## NER 라벨별로 DB에서 허용 후보 값을 가져올 때 사용하는 (릴레이션, 컬럼) 매핑.
## README의 통합 뷰 v_course_info에서 DISTINCT로 읽어, Text-to-SQL이 쓰는 스키마와 동일한 기준으로 교정한다.
## 뷰 이름만 바꾸려면 환경 변수 KEYWORD_REFERENCE_VIEW (기본 v_course_info).
REFERENCE_VIEW_NAME = (os.getenv("KEYWORD_REFERENCE_VIEW", "v_course_info") or "v_course_info").strip()

ENTITY_REFERENCE_MAP = {
    "CATEGORY": (REFERENCE_VIEW_NAME, "category"),
    "COURSE_NAME": (REFERENCE_VIEW_NAME, "subject_name"),
    "DEPARTMENT": (REFERENCE_VIEW_NAME, "dept_name"),
    "CLASS_MODE": (REFERENCE_VIEW_NAME, "class_mode"),
    "EVAL_TYPE": (REFERENCE_VIEW_NAME, "eval_type"),
    "GRADE_METHOD": (REFERENCE_VIEW_NAME, "grading_method"),
}

## 최소 유사도 기준. 라벨별로 다르게 적용하여 과도한 교정을 방지.
## (예: 교수명은 유사도가 높아야 교정, 과목명은 조금 낮아도 교정)
## DEFAULT 값은 78
LABEL_MIN_SCORES = {
    "COURSE_NAME": 70.0,
    "DEPARTMENT": 74.0,
    "CATEGORY": 74.0,
}

## fuzzy matching만으로는 교정이 부적절한 경우가 많아, 자주 발생하는 오탈자나 약어에 대해서는 별도의 허용값을 정의하여 교정 정확도를 높임.
## 일종의 사전을 정의하는 것
COMMON_ALIASES = {
    "DEPARTMENT": {
        "컴공": ["컴퓨터공학과"],
        "컴융": ["컴퓨터융합학부"],
        "인공지능": ["컴퓨터인공지능학부"],
    },
    "CATEGORY": {
        "전필": ["전공(필수)"],
        "전선": ["전공(선택)"],
        "교필": ["교양(필수)"],
        "교선": ["교양(선택)"],
        "일선": ["일반(선택)"],
        "전핵": ["전공(핵심)"],
    },
}

## 한국어 조사 제거를 시도하여 매칭 정확도를 높임. (예: "컴퓨터공학과에서" -> "컴퓨터공학과")
KOREAN_PARTICLES = (
    "으로부터",
    "로부터",
    "에서는",
    "에게서",
    "까지",
    "부터",
    "에서",
    "에게",
    "으로",
    "로",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
    "와",
    "과",
    "도",
    "만",
)

_predictor_lock = threading.Lock()
_predictor: "CourseNERPredictor | None" = None
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


def slot_key(label: str) -> str:
    return label.lower()


class CourseNERPredictor:
    def __init__(self, model_dir: str | Path = MODEL_DIR, max_length: int = 128):
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir).to(self.device)
        self.model.eval()

        with open(self.model_dir / "labels.json", "r", encoding="utf-8") as file:
            label_data = json.load(file)

        self.id_to_label = {
            int(key): value
            for key, value in label_data["id_to_label"].items()
        }

    @torch.no_grad()
    def extract(self, text: str) -> list[dict[str, Any]]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
        )

        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        outputs = self.model(**encoded)
        probs = torch.softmax(outputs.logits[0], dim=-1)
        pred_ids = torch.argmax(probs, dim=-1).tolist()
        pred_scores = torch.max(probs, dim=-1).values.tolist()

        token_predictions = []
        for pred_id, score, (start, end) in zip(pred_ids, pred_scores, offsets):
            if start == end:
                continue
            token_predictions.append(
                {
                    "start": start,
                    "end": end,
                    "label": self.id_to_label[pred_id],
                    "score": float(score),
                }
            )

        return self._merge_bio(text, token_predictions)

    def _merge_bio(self, text: str, token_predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        entities = []
        current = None

        for token in token_predictions:
            label = token["label"]
            if label == "O":
                if current:
                    entities.append(current)
                    current = None
                continue

            if "-" not in label:
                continue

            prefix, entity_type = label.split("-", 1)
            if prefix == "B" or current is None or current["label"] != entity_type:
                if current:
                    entities.append(current)
                current = {
                    "label": entity_type,
                    "start": token["start"],
                    "end": token["end"],
                    "scores": [token["score"]],
                }
                continue

            current["end"] = token["end"]
            current["scores"].append(token["score"])

        if current:
            entities.append(current)

        result = []
        for entity in entities:
            start = entity["start"]
            end = entity["end"]
            avg_score = sum(entity["scores"]) / len(entity["scores"])
            result.append(
                {
                    "text": text[start:end],
                    "label": entity["label"],
                    "start": start,
                    "end": end,
                    "score": round(avg_score, 4),
                }
            )

        return result


def get_predictor(model_dir: str | Path = MODEL_DIR, max_length: int = 128) -> CourseNERPredictor:
    global _predictor

    with _predictor_lock:
        if _predictor is None:
            _predictor = CourseNERPredictor(model_dir=model_dir, max_length=max_length)
        return _predictor


def build_slots(entities: list[dict[str, Any]], text_key: str = "text") -> dict[str, list[str]]:
    slots: OrderedDict[str, list[str]] = OrderedDict()
    for entity in entities:
        key = slot_key(entity["label"])
        slots.setdefault(key, [])

        value = entity.get(text_key) or entity["text"]
        if value not in slots[key]:
            slots[key].append(value)

    return dict(slots)


def normalize_for_match(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"[\s\-_./(){}\[\],:;\"'`~!?]+", "", normalized)
    return normalized


def strip_korean_particle(value: str) -> str:
    stripped = value.strip()
    for particle in KOREAN_PARTICLES:
        if stripped.endswith(particle) and len(stripped) > len(particle) + 1:
            return stripped[: -len(particle)]
    return stripped


def make_match_queries(entity_text: str, label: str) -> list[str]:
    raw = entity_text.strip()
    compact = normalize_for_match(raw)
    particle_stripped = strip_korean_particle(raw)

    variants = [raw, particle_stripped, compact]

    cleaned: list[str] = []
    for variant in variants:
        variant = variant.strip()
        if variant and variant not in cleaned:
            cleaned.append(variant)
    return cleaned


def get_alias_queries(entity_text: str, label: str) -> list[str]:
    raw = entity_text.strip().lower()
    compact = normalize_for_match(entity_text)
    aliases = COMMON_ALIASES.get(label, {})
    alias_values = aliases.get(raw) or aliases.get(compact) or []
    return [value for value in alias_values if value.strip()]


def ordered_subsequence_positions(query: str, candidate: str) -> list[int] | None:
    query_key = normalize_for_match(strip_korean_particle(query))
    candidate_key = normalize_for_match(candidate)
    if (
        len(query_key) < 2
        or len(query_key) > SEQUENCE_MAX_QUERY_LENGTH
        or len(query_key) >= len(candidate_key)
    ):
        return None

    positions: list[int] = []
    start = 0
    for char in query_key:
        found = candidate_key.find(char, start)
        if found < 0:
            return None
        positions.append(found)
        start = found + 1

    return positions


def sequence_match_score(query: str, candidate: str) -> float:
    positions = ordered_subsequence_positions(query, candidate)
    if not positions:
        return 0.0

    query_key = normalize_for_match(strip_korean_particle(query))
    candidate_key = normalize_for_match(candidate)
    span = positions[-1] - positions[0] + 1
    gap_count = max(span - len(query_key), 0)
    length_gap = max(len(candidate_key) - len(query_key), 0)
    prefix_bonus = 6 if positions[0] == 0 else 0
    score = 100 - (gap_count * 4) - (length_gap * 2) + prefix_bonus
    return max(0.0, min(100.0, float(score)))


def collapse_repeated_tokens(value: str) -> str:
    tokens = value.split()
    collapsed: list[str] = []
    for token in tokens:
        if not collapsed or collapsed[-1] != token:
            collapsed.append(token)
    return " ".join(collapsed)


def ambiguous_group_match(ranked: list["MatchResult"]) -> "MatchResult | None":
    if not ranked:
        return None

    best = ranked[0]
    near_matches = [
        item
        for item in ranked
        if item.text and best.score - item.score < SEQUENCE_AMBIGUITY_MARGIN
    ]
    if len(near_matches) < 2:
        return None

    def similarity_to_query(item: MatchResult) -> tuple[float, int]:
        text = item.text or ""
        return (
            float(fuzz.WRatio(best.query, text, processor=normalize_for_match)),
            -len(normalize_for_match(text)),
        )

    selected = max(near_matches, key=similarity_to_query)
    if not selected.text:
        return None

    return MatchResult(
        collapse_repeated_tokens(selected.text),
        selected.score,
        best.query,
        "ambiguous_best",
    )


def find_best_sequence_match(entity_text: str, candidates: list[str]) -> "MatchResult":
    scores_by_candidate: dict[str, MatchResult] = {}

    for query in make_match_queries(entity_text, ""):
        for candidate in candidates:
            score = sequence_match_score(query, candidate)
            previous = scores_by_candidate.get(candidate)
            if previous is None or score > previous.score:
                scores_by_candidate[candidate] = MatchResult(
                    collapse_repeated_tokens(candidate),
                    score,
                    query,
                    "sequence",
                )

    ranked = sorted(scores_by_candidate.values(), key=lambda item: item.score, reverse=True)
    best = ranked[0] if ranked else MatchResult(None, 0.0, entity_text, "sequence")
    second_score = ranked[1].score if len(ranked) > 1 else 0.0

    if not best.text or best.score < SEQUENCE_MIN_SCORE:
        return MatchResult(None, 0.0, entity_text, "sequence")

    if best.score - second_score < SEQUENCE_AMBIGUITY_MARGIN:
        prefix_match = ambiguous_group_match(ranked)
        if prefix_match:
            return prefix_match
        return MatchResult(None, best.score, best.query, "sequence_ambiguous")

    return best


def find_fuzzy_match(query: str, candidates: list[str], method: str) -> "MatchResult":
    matches = fuzz_process.extract(
        query,
        candidates,
        scorer=fuzz.WRatio,
        processor=normalize_for_match,
        limit=10,
    )
    if not matches:
        return MatchResult(None, 0.0, query, method)

    ranked = [
        MatchResult(collapse_repeated_tokens(matched_text), float(score), query, method)
        for matched_text, score, _ in matches
    ]
    best = ranked[0]
    second_score = ranked[1].score if len(ranked) > 1 else 0.0

    if best.score - second_score < SEQUENCE_AMBIGUITY_MARGIN:
        prefix_match = ambiguous_group_match(ranked)
        if prefix_match:
            prefix_match.method = f"{method}_prefix"
            return prefix_match
        return MatchResult(None, best.score, query, f"{method}_ambiguous")

    return best


@dataclass
class MatchResult:
    text: str | None
    score: float
    query: str
    method: str = "none"


def find_best_db_match(entity_text: str, candidates: list[str], label: str) -> MatchResult:
    if not entity_text or not candidates:
        return MatchResult(None, 0.0, entity_text)

    candidate_by_key = {normalize_for_match(candidate): candidate for candidate in candidates}
    best = MatchResult(None, 0.0, entity_text)

    for query in make_match_queries(entity_text, label):
        exact = candidate_by_key.get(normalize_for_match(query))
        if exact:
            return MatchResult(exact, 100.0, query, "exact")

    sequence_match = find_best_sequence_match(entity_text, candidates)
    if sequence_match.text:
        return sequence_match

    for query in get_alias_queries(entity_text, label):
        exact = candidate_by_key.get(normalize_for_match(query))
        if exact:
            return MatchResult(exact, 100.0, query, "alias")

        match = find_fuzzy_match(query, candidates, "alias_fuzzy")
        if match.text and match.score > best.score:
            best = match

    if best.text:
        return best

    for query in make_match_queries(entity_text, label):
        match = find_fuzzy_match(query, candidates, "fuzzy")
        if match.text and match.score > best.score:
            best = match

    return best


def should_apply_match(entity: dict[str, Any], match: MatchResult) -> bool:
    if not match.text:
        return False

    label = entity["label"]
    min_score = LABEL_MIN_SCORES.get(label, DEFAULT_MIN_SCORE)
    if match.score < min_score:
        return False

    source = normalize_for_match(entity["text"])
    target = normalize_for_match(match.text)
    if not source or not target:
        return False

    if source == target:
        return True

    # Very short spans are easy to over-correct unless they are aliases or strong matches.
    if len(source) <= 2 and match.score < 95:
        if match.method == "sequence":
            return True
        aliases = COMMON_ALIASES.get(label, {})
        return source in aliases

    return True


def correct_ner_entities(
    entities: list[dict[str, Any]],
    references: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    references = references if references is not None else load_reference_values()
    corrected_entities = []

    for entity in entities:
        label = entity["label"]
        candidates = references.get(label, [])
        match = find_best_db_match(entity["text"], candidates, label)
        apply_match = should_apply_match(entity, match)
        corrected_text = match.text if apply_match and match.text else entity["text"]

        corrected_entities.append(
            {
                **entity,
                "corrected_text": corrected_text,
                "matched_table": ENTITY_REFERENCE_MAP.get(label, (None, None))[0],
                "matched_column": ENTITY_REFERENCE_MAP.get(label, (None, None))[1],
                "similarity": round(match.score, 2) if match.text else None,
                "match_method": match.method,
                "correction_applied": corrected_text != entity["text"],
            }
        )

    return corrected_entities


def build_corrected_query(query: str, corrected_entities: list[dict[str, Any]]) -> str:
    corrected_query = query
    used_ranges: list[tuple[int, int]] = []

    for entity in sorted(corrected_entities, key=lambda item: item["start"], reverse=True):
        start = int(entity["start"])
        end = int(entity["end"])
        if any(not (end <= used_start or start >= used_end) for used_start, used_end in used_ranges):
            continue

        corrected_text = entity.get("corrected_text", entity["text"])
        corrected_query = corrected_query[:start] + corrected_text + corrected_query[end:]
        used_ranges.append((start, end))

    return corrected_query


def preprocess_query(
    query: str,
    model_dir: str | Path = MODEL_DIR,
    max_length: int = 128,
    fail_open: bool = True,
) -> dict[str, Any]:
    try:
        predictor = get_predictor(model_dir=model_dir, max_length=max_length)
        references = load_reference_values()
        entities = predictor.extract(query)
        corrected_entities = correct_ner_entities(entities, references)
        corrected_query = build_corrected_query(query, corrected_entities)
        return {
            "query": query,
            "corrected_query": corrected_query,
            "changed": corrected_query != query,
            "entities": entities,
            "slots": build_slots(entities),
            "corrected_entities": corrected_entities,
            "corrected_slots": build_slots(corrected_entities, text_key="corrected_text"),
            "error": None,
        }
    except Exception as exc:
        if not fail_open:
            raise
        return {
            "query": query,
            "corrected_query": query,
            "changed": False,
            "entities": [],
            "slots": {},
            "corrected_entities": [],
            "corrected_slots": {},
            "error": str(exc),
        }


def extract_keywords(query: str, model_dir: str | Path = MODEL_DIR, max_length: int = 128) -> dict[str, Any]:
    return preprocess_query(query=query, model_dir=model_dir, max_length=max_length, fail_open=False)


def warmup_keyword_normalizer() -> dict[str, Any]:
    started = time.perf_counter()
    try:
        predictor = get_predictor()
        references = load_reference_values()
        predictor.extract("컴융 데이터베이스 과목 보여줘")
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "ok": True,
            "elapsed_ms": elapsed_ms,
            "device": predictor.device,
            "reference_counts": {
                label: len(values)
                for label, values in references.items()
            },
        }
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "ok": False,
            "elapsed_ms": elapsed_ms,
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Natural language query to normalize")
    parser.add_argument("--model_dir", default=str(MODEL_DIR))
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--json", action="store_true", help="Print full preprocessing result as JSON")
    parser.add_argument("--refresh-references", action="store_true", help="Reload DB reference values")
    args = parser.parse_args()

    if args.refresh_references:
        load_reference_values(force=True)

    result = extract_keywords(
        query=args.query,
        model_dir=args.model_dir,
        max_length=args.max_length,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["corrected_query"])


if __name__ == "__main__":
    main()
