from collections import OrderedDict
from dataclasses import dataclass
from heapq import nlargest
from typing import Any

from rapidfuzz import fuzz

from .keyword_config import (
    COMMON_ALIASES,
    COURSE_PHRASE_MAX_ENTITIES,
    COURSE_PHRASE_MAX_LENGTH,
    DEFAULT_MIN_SCORE,
    ENTITY_REFERENCE_MAP,
    LABEL_MIN_SCORES,
    PARTIAL_TYPO_LABELS,
    SEQUENCE_AMBIGUITY_MARGIN,
    SEQUENCE_MAX_QUERY_LENGTH,
    SEQUENCE_MIN_SCORE,
    TYPO_AMBIGUITY_MARGIN,
    TYPO_LABEL_MIN_SCORES,
    TYPO_MIN_QUERY_LENGTH,
    TYPO_MIN_SCORE,
)
from .keyword_normalize import (
    collapse_repeated_tokens,
    make_match_queries,
    normalize_for_match,
    normalize_for_typo,
    strip_korean_particle,
)
from .keyword_references import load_reference_values


@dataclass
class MatchResult:
    text: str | None
    score: float
    query: str
    method: str = "none"


def slot_key(label: str) -> str:
    return label.lower()


def build_slots(entities: list[dict[str, Any]], text_key: str = "text") -> dict[str, list[str]]:
    slots: OrderedDict[str, list[str]] = OrderedDict()
    for entity in entities:
        key = slot_key(entity["label"])
        slots.setdefault(key, [])

        value = entity.get(text_key) or entity["text"]
        if value not in slots[key]:
            slots[key].append(value)

    return dict(slots)


def find_alias_match(entity_text: str, candidates: list[str], label: str) -> MatchResult:
    aliases = COMMON_ALIASES.get(label, {})
    if not aliases:
        return MatchResult(None, 0.0, entity_text, "alias_no_match")

    candidate_by_key = {normalize_for_match(candidate): candidate for candidate in candidates}
    alias_targets: list[tuple[str, str]] = []
    for alias, values in aliases.items():
        for value in values:
            target = candidate_by_key.get(normalize_for_match(value))
            if target:
                alias_targets.append((alias, target))

    if not alias_targets:
        return MatchResult(None, 0.0, entity_text, "alias_no_match")

    for query in make_match_queries(entity_text):
        for alias, target in alias_targets:
            if normalize_for_match(query) == normalize_for_match(alias):
                return MatchResult(target, 100.0, query, "alias")

    ranked = sorted(
        (
            MatchResult(target, typo_match_score(query, alias, label), query, "alias_typo")
            for query in make_match_queries(entity_text)
            for alias, target in alias_targets
            if len(normalize_for_match(query)) >= TYPO_MIN_QUERY_LENGTH
        ),
        key=lambda item: item.score,
        reverse=True,
    )
    if not ranked:
        return MatchResult(None, 0.0, entity_text, "alias_no_match")

    best = ranked[0]
    min_score = TYPO_LABEL_MIN_SCORES.get(label, TYPO_MIN_SCORE)
    if best.score < min_score:
        return MatchResult(None, best.score, best.query, "alias_typo_low_score")

    near_matches = [
        item
        for item in ranked[1:5]
        if item.text
        and normalize_for_match(item.text) != normalize_for_match(best.text or "")
        and best.score - item.score < TYPO_AMBIGUITY_MARGIN
    ]
    if near_matches:
        return MatchResult(None, best.score, best.query, "alias_typo_ambiguous")

    return best


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


def exact_containing_matches(query: str, matches: list[MatchResult]) -> tuple[str, list[MatchResult]] | None:
    candidates = [
        variant
        for variant in make_match_queries(query)
        if len(normalize_for_match(variant)) >= 2
    ]
    candidates.sort(key=lambda value: len(normalize_for_match(value)), reverse=True)

    for candidate in candidates:
        candidate_key = normalize_for_match(candidate)
        containing = [
            item
            for item in matches
            if item.text and candidate_key in normalize_for_match(item.text)
        ]
        if len(containing) >= 2:
            return candidate, containing

    return None


def ambiguous_group_match(ranked: list[MatchResult]) -> MatchResult | None:
    if not ranked:
        return None

    best = ranked[0]
    exact_group = exact_containing_matches(best.query, ranked)
    if exact_group:
        shared_text, containing_matches = exact_group
        return MatchResult(shared_text, containing_matches[0].score, best.query, "sequence_exact_shared")

    return None


def find_best_sequence_match(entity_text: str, candidates: list[str]) -> MatchResult:
    scores_by_candidate: dict[str, MatchResult] = {}

    for query in make_match_queries(entity_text):
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

    ranked = nlargest(5, scores_by_candidate.values(), key=lambda item: item.score)
    best = ranked[0] if ranked else MatchResult(None, 0.0, entity_text, "sequence")
    second_score = ranked[1].score if len(ranked) > 1 else 0.0

    if not best.text or best.score < SEQUENCE_MIN_SCORE:
        return MatchResult(None, 0.0, entity_text, "sequence")

    exact_group = exact_containing_matches(best.query, ranked)
    if exact_group:
        shared_text, containing_matches = exact_group
        return MatchResult(shared_text, containing_matches[0].score, best.query, "sequence_exact_shared")

    if best.score - second_score < SEQUENCE_AMBIGUITY_MARGIN:
        prefix_match = ambiguous_group_match(ranked)
        if prefix_match:
            return prefix_match
        return MatchResult(None, best.score, best.query, "sequence_ambiguous")

    return best


def is_typo_candidate_shape(query: str, candidate: str) -> bool:
    query_key = normalize_for_match(strip_korean_particle(query))
    candidate_key = normalize_for_match(candidate)
    if len(query_key) < TYPO_MIN_QUERY_LENGTH or len(candidate_key) < TYPO_MIN_QUERY_LENGTH:
        return False

    max_length_gap = max(2, int(len(query_key) * 0.35))
    if abs(len(candidate_key) - len(query_key)) > max_length_gap:
        return False

    return True


def typo_match_score(query: str, candidate: str, label: str) -> float:
    query_key = normalize_for_match(strip_korean_particle(query))
    candidate_key = normalize_for_match(candidate)
    if not query_key or not candidate_key:
        return 0.0

    ratio = float(fuzz.ratio(query_key, candidate_key))
    if label not in PARTIAL_TYPO_LABELS or len(query_key) > len(candidate_key):
        return ratio

    typo_query_key = normalize_for_typo(query)
    typo_candidate_key = normalize_for_typo(candidate)

    partial = float(fuzz.partial_ratio(query_key, candidate_key))
    typo_ratio = float(fuzz.ratio(typo_query_key, typo_candidate_key))
    typo_partial = float(fuzz.partial_ratio(typo_query_key, typo_candidate_key))
    return max(ratio, partial, typo_ratio, typo_partial)


def find_typo_match(query: str, candidates: list[str], label: str) -> MatchResult:
    query_key = normalize_for_match(strip_korean_particle(query))
    if len(query_key) < TYPO_MIN_QUERY_LENGTH:
        return MatchResult(None, 0.0, query, "typo_ineligible")

    ranked = nlargest(
        5,
        (
            MatchResult(
                collapse_repeated_tokens(candidate),
                typo_match_score(query, candidate, label),
                query,
                "typo",
            )
            for candidate in candidates
            if is_typo_candidate_shape(query, candidate)
            or label in PARTIAL_TYPO_LABELS
        ),
        key=lambda item: (item.score, -(len(normalize_for_match(item.text or "")))),
    )
    if not ranked:
        return MatchResult(None, 0.0, query, "typo_no_match")

    best = ranked[0]
    min_score = TYPO_LABEL_MIN_SCORES.get(label, TYPO_MIN_SCORE)
    if best.score < min_score:
        return MatchResult(None, best.score, query, "typo_low_score")

    near_matches = [
        item
        for item in ranked[1:]
        if item.text and best.score - item.score < TYPO_AMBIGUITY_MARGIN
    ]
    if near_matches:
        best_length = len(normalize_for_match(best.text or ""))
        materially_similar_near_matches = [
            item
            for item in near_matches
            if len(normalize_for_match(item.text or "")) <= max(best_length + 2, int(best_length * 1.5))
        ]
        same_target_matches = [
            item
            for item in materially_similar_near_matches
            if normalize_for_match(item.text or "") == normalize_for_match(best.text or "")
        ]
        same_score_shorter_candidates = [
            item
            for item in materially_similar_near_matches
            if item.score == best.score
            and len(normalize_for_match(item.text or "")) > len(normalize_for_match(best.text or ""))
        ]
        if (
            materially_similar_near_matches
            and len(same_target_matches) != len(materially_similar_near_matches)
            and len(same_score_shorter_candidates) != len(materially_similar_near_matches)
        ):
            return MatchResult(None, best.score, query, "typo_ambiguous")

    return best


def find_best_typo_match(entity_text: str, candidates: list[str], label: str) -> MatchResult:
    best = MatchResult(None, 0.0, entity_text, "typo_no_match")

    for query in make_match_queries(entity_text):
        match = find_typo_match(query, candidates, label)
        if match.text:
            return match
        if match.score > best.score:
            best = match

    return best


def make_corrected_entity(entity: dict[str, Any], match: MatchResult) -> dict[str, Any]:
    apply_match = should_apply_match(entity, match)
    corrected_text = match.text if apply_match and match.text else entity["text"]
    label = entity["label"]

    return {
        **entity,
        "corrected_text": corrected_text,
        "matched_table": ENTITY_REFERENCE_MAP.get(label, (None, None))[0],
        "matched_column": ENTITY_REFERENCE_MAP.get(label, (None, None))[1],
        "similarity": round(match.score, 2) if match.score else None,
        "match_method": match.method,
        "correction_applied": corrected_text != entity["text"],
    }


def find_best_db_match(entity_text: str, candidates: list[str], label: str) -> MatchResult:
    if not entity_text or not candidates:
        return MatchResult(None, 0.0, entity_text)

    candidate_by_key = {normalize_for_match(candidate): candidate for candidate in candidates}

    for query in make_match_queries(entity_text):
        exact = candidate_by_key.get(normalize_for_match(query))
        if exact:
            return MatchResult(exact, 100.0, query, "exact")

    alias_match = find_alias_match(entity_text, candidates, label)
    if alias_match.text:
        return alias_match

    sequence_match = find_best_sequence_match(entity_text, candidates)
    if sequence_match.text:
        return sequence_match

    typo_match = find_best_typo_match(entity_text, candidates, label)
    if typo_match.text:
        return typo_match

    if sequence_match.score >= typo_match.score:
        return sequence_match
    return typo_match


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

    if source.endswith("과") and source[:-1] == target:
        return False

    if len(source) <= 2 and match.score < 95:
        if match.method == "sequence" or "shared" in match.method:
            return True
        aliases = COMMON_ALIASES.get(label, {})
        return source in aliases

    return True


def mergeable_entity_gap(query: str, left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_end = int(left["end"])
    right_start = int(right["start"])
    if left_end > right_start:
        return False
    return query[left_end:right_start] == ""


def labels_for_merged_text(parts: list[dict[str, Any]], text: str) -> list[str]:
    labels: list[str] = []
    compact = normalize_for_match(text)

    if compact.endswith(("학과", "학부", "전공")):
        labels.append("DEPARTMENT")

    for part in parts:
        label = part["label"]
        if label not in labels:
            labels.append(label)

    for label in ENTITY_REFERENCE_MAP:
        if label not in labels:
            labels.append(label)

    return labels


def looks_like_department_text(text: str) -> bool:
    compact = normalize_for_match(text)
    return compact.endswith(("학과", "학꽈", "학부", "학브", "전공"))


def correct_single_entity(entity: dict[str, Any], references: dict[str, list[str]]) -> dict[str, Any]:
    labels = [entity["label"]]
    if looks_like_department_text(entity["text"]) and "DEPARTMENT" not in labels:
        labels.insert(0, "DEPARTMENT")

    best_corrected: dict[str, Any] | None = None
    best_score = -1.0
    for label in labels:
        candidates = references.get(label, [])
        match = find_best_db_match(entity["text"], candidates, label)
        corrected = make_corrected_entity({**entity, "label": label}, match)
        score = float(corrected.get("similarity") or 0.0)
        if best_corrected is None or score > best_score:
            best_corrected = corrected
            best_score = score

    return best_corrected or make_corrected_entity(entity, MatchResult(None, 0.0, entity["text"]))


def find_best_merged_entity(
    query: str,
    parts: list[dict[str, Any]],
    references: dict[str, list[str]],
) -> dict[str, Any] | None:
    start = int(parts[0]["start"])
    end = int(parts[-1]["end"])
    text = query[start:end]
    if len(normalize_for_match(text)) < TYPO_MIN_QUERY_LENGTH:
        return None

    best_corrected: dict[str, Any] | None = None
    best_match: MatchResult | None = None
    has_non_course_filter = any(
        part["label"] in {"DEPARTMENT", "CATEGORY", "CLASS_MODE", "EVAL_TYPE", "GRADE_METHOD"}
        for part in parts
    )

    for label in labels_for_merged_text(parts, text):
        if label == "COURSE_NAME" and not any(part["label"] == "COURSE_NAME" for part in parts):
            continue
        candidates = references.get(label, [])
        if not candidates:
            continue

        merged_entity = {
            "text": text,
            "label": label,
            "start": start,
            "end": end,
            "score": min(float(part.get("score", 0.0)) for part in parts),
            "merged_from": [
                {
                    "text": part["text"],
                    "label": part["label"],
                    "start": part["start"],
                    "end": part["end"],
                }
                for part in parts
            ],
        }
        match = find_best_db_match(text, candidates, label)
        if label == "COURSE_NAME" and match.method == "sequence":
            continue
        if label == "COURSE_NAME" and has_non_course_filter and match.method != "exact":
            continue
        if match.text and should_apply_match(merged_entity, match):
            corrected = make_corrected_entity(merged_entity, match)
            corrected["match_method"] = f"merged_{corrected['match_method']}"
            if best_match is None or match.score > best_match.score:
                best_corrected = corrected
                best_match = match

    return best_corrected


def find_best_course_phrase_entity(
    query: str,
    parts: list[dict[str, Any]],
    references: dict[str, list[str]],
) -> dict[str, Any] | None:
    if len(parts) < 2 or not any(part["label"] == "COURSE_NAME" for part in parts):
        return None
    has_non_course_filter = any(
        part["label"] in {"DEPARTMENT", "CATEGORY", "CLASS_MODE", "EVAL_TYPE", "GRADE_METHOD"}
        for part in parts
    )
    if any(part["label"] in {"CATEGORY", "CLASS_MODE", "EVAL_TYPE", "GRADE_METHOD"} for part in parts):
        return None

    start = int(parts[0]["start"])
    end = int(parts[-1]["end"])
    text = query[start:end].strip()
    if (
        len(normalize_for_match(text)) < TYPO_MIN_QUERY_LENGTH
        or len(text) > COURSE_PHRASE_MAX_LENGTH
    ):
        return None

    candidates = references.get("COURSE_NAME", [])
    if not candidates:
        return None

    phrase_entity = {
        "text": text,
        "label": "COURSE_NAME",
        "start": start,
        "end": end,
        "score": min(float(part.get("score", 0.0)) for part in parts),
        "merged_from": [
            {
                "text": part["text"],
                "label": part["label"],
                "start": part["start"],
                "end": part["end"],
            }
            for part in parts
        ],
    }
    match = find_best_db_match(text, candidates, "COURSE_NAME")
    if (
        not match.text
        or match.method == "sequence"
        or (has_non_course_filter and match.method != "exact")
        or not should_apply_match(phrase_entity, match)
    ):
        return None

    corrected = make_corrected_entity(phrase_entity, match)
    corrected["match_method"] = f"phrase_{corrected['match_method']}"
    return corrected


def find_best_category_phrase_entity(
    query: str,
    parts: list[dict[str, Any]],
    references: dict[str, list[str]],
) -> dict[str, Any] | None:
    if len(parts) != 2 or not any(part["label"] == "CATEGORY" for part in parts):
        return None

    start = int(parts[0]["start"])
    end = int(parts[-1]["end"])
    text = query[start:end].strip()
    match = find_alias_match(text, references.get("CATEGORY", []), "CATEGORY")
    if not match.text:
        return None

    category_entity = {
        "text": text,
        "label": "CATEGORY",
        "start": start,
        "end": end,
        "score": min(float(part.get("score", 0.0)) for part in parts),
        "merged_from": [
            {
                "text": part["text"],
                "label": part["label"],
                "start": part["start"],
                "end": part["end"],
            }
            for part in parts
        ],
    }
    corrected = make_corrected_entity(category_entity, match)
    corrected["match_method"] = f"category_phrase_{corrected['match_method']}"
    return corrected


def correct_ner_entities(
    entities: list[dict[str, Any]],
    references: dict[str, list[str]] | None = None,
    query: str | None = None,
) -> list[dict[str, Any]]:
    references = references if references is not None else load_reference_values()
    corrected_entities = []
    sorted_entities = sorted(entities, key=lambda item: (int(item["start"]), int(item["end"])))
    i = 0

    while i < len(sorted_entities):
        if query:
            phrase_max_end = min(i + COURSE_PHRASE_MAX_ENTITIES, len(sorted_entities))
            consumed = False
            for end_index in range(phrase_max_end, i + 1, -1):
                parts = sorted_entities[i:end_index]
                if len(parts) < 2:
                    continue

                category_phrase = find_best_category_phrase_entity(query, parts, references)
                if category_phrase:
                    corrected_entities.append(category_phrase)
                    i = end_index
                    consumed = True
                    break

                if all(
                    mergeable_entity_gap(query, parts[j], parts[j + 1])
                    for j in range(len(parts) - 1)
                ):
                    merged = find_best_merged_entity(query, parts, references)
                    if merged:
                        corrected_entities.append(merged)
                        i = end_index
                        consumed = True
                        break

                phrase = find_best_course_phrase_entity(query, parts, references)
                if phrase:
                    corrected_entities.append(phrase)
                    i = end_index
                    consumed = True
                    break

            if consumed:
                continue

            merge_max_end = min(i + 3, len(sorted_entities))
            for end_index in range(merge_max_end, i + 1, -1):
                parts = sorted_entities[i:end_index]
                if len(parts) < 2:
                    continue

                if not all(
                    mergeable_entity_gap(query, parts[j], parts[j + 1])
                    for j in range(len(parts) - 1)
                ):
                    continue

                merged = find_best_merged_entity(query, parts, references)
                if merged:
                    corrected_entities.append(merged)
                    i = end_index
                    consumed = True
                    break

            if consumed:
                continue

        entity = sorted_entities[i]
        corrected_entities.append(correct_single_entity(entity, references))
        i += 1

    return corrected_entities


def build_corrected_query(query: str, corrected_entities: list[dict[str, Any]]) -> str:
    selected_entities = []
    used_ranges: list[tuple[int, int]] = []
    for entity in sorted(corrected_entities, key=lambda item: (int(item["start"]), int(item["end"]))):
        start = int(entity["start"])
        end = int(entity["end"])
        if any(not (end <= used_start or start >= used_end) for used_start, used_end in used_ranges):
            continue
        selected_entities.append(entity)
        used_ranges.append((start, end))

    result = []
    cursor = 0
    previous_entity: dict[str, Any] | None = None

    for entity in selected_entities:
        start = int(entity["start"])
        end = int(entity["end"])
        if previous_entity is not None and int(previous_entity["end"]) == start:
            previous_label = previous_entity.get("label")
            current_label = entity.get("label")
            if (
                previous_entity.get("correction_applied")
                and previous_label != current_label
                and previous_label in ENTITY_REFERENCE_MAP
                and current_label in ENTITY_REFERENCE_MAP
            ):
                result.append(" ")

        result.append(query[cursor:start])
        corrected_text = entity.get("corrected_text", entity["text"])
        result.append(corrected_text)
        cursor = end
        previous_entity = entity

    result.append(query[cursor:])
    return "".join(result)
