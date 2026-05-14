import re
from functools import lru_cache

from .keyword_config import KOREAN_PARTICLES


@lru_cache(maxsize=50000)
def normalize_for_match(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"[\s\-_./(){}\[\],:;\"'`~!?]+", "", normalized)
    return normalized


@lru_cache(maxsize=50000)
def decompose_hangul(value: str) -> str:
    result: list[str] = []
    for char in value:
        code = ord(char)
        if 0xAC00 <= code <= 0xD7A3:
            syllable = code - 0xAC00
            cho = syllable // 588
            jung = (syllable % 588) // 28
            jong = syllable % 28
            result.append(chr(0x1100 + cho))
            result.append(chr(0x1161 + jung))
            if jong:
                result.append(chr(0x11A7 + jong))
            continue
        result.append(char)
    return "".join(result)


@lru_cache(maxsize=50000)
def strip_korean_particle(value: str) -> str:
    stripped = value.strip()
    for particle in KOREAN_PARTICLES:
        if particle == "과" and stripped.endswith("학과"):
            continue
        if stripped.endswith(particle) and len(stripped) > len(particle) + 1:
            return stripped[: -len(particle)]
    return stripped


@lru_cache(maxsize=50000)
def normalize_for_typo(value: str) -> str:
    return decompose_hangul(normalize_for_match(strip_korean_particle(value)))


def make_match_queries(entity_text: str) -> list[str]:
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


def collapse_repeated_tokens(value: str) -> str:
    tokens = value.split()
    collapsed: list[str] = []
    for token in tokens:
        if not collapsed or collapsed[-1] != token:
            collapsed.append(token)
    return " ".join(collapsed)
