import argparse
import json
import time
from pathlib import Path
from typing import Any

from .keyword_config import MODEL_DIR
from .keyword_matcher import build_corrected_query, build_slots, correct_ner_entities
from .keyword_ner import CourseNERPredictor, get_predictor
from .keyword_references import load_reference_values


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
        corrected_entities = correct_ner_entities(entities, references, query=query)
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
