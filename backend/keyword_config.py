import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv()
load_dotenv(BASE_DIR / ".env", override=False)

MODEL_DIR = BASE_DIR / "course_custom_ner_model" / "models" / "course-custom-ner"
REFERENCE_LIMIT = int(os.getenv("KEYWORD_REFERENCE_LIMIT", "20000"))
DEFAULT_MIN_SCORE = float(os.getenv("KEYWORD_CORRECTION_MIN_SCORE", "78"))
SEQUENCE_MIN_SCORE = float(os.getenv("KEYWORD_SEQUENCE_MIN_SCORE", "72"))
SEQUENCE_AMBIGUITY_MARGIN = float(os.getenv("KEYWORD_SEQUENCE_AMBIGUITY_MARGIN", "4"))
SEQUENCE_MAX_QUERY_LENGTH = int(os.getenv("KEYWORD_SEQUENCE_MAX_QUERY_LENGTH", "6"))
TYPO_MIN_SCORE = float(os.getenv("KEYWORD_TYPO_MIN_SCORE", "84"))
TYPO_AMBIGUITY_MARGIN = float(os.getenv("KEYWORD_TYPO_AMBIGUITY_MARGIN", "6"))
TYPO_MIN_QUERY_LENGTH = int(os.getenv("KEYWORD_TYPO_MIN_QUERY_LENGTH", "4"))
COURSE_PHRASE_MAX_LENGTH = int(os.getenv("KEYWORD_COURSE_PHRASE_MAX_LENGTH", "24"))
COURSE_PHRASE_MAX_ENTITIES = int(os.getenv("KEYWORD_COURSE_PHRASE_MAX_ENTITIES", "4"))

REFERENCE_VIEW_NAME = (os.getenv("KEYWORD_REFERENCE_VIEW", "v_course_info") or "v_course_info").strip()

ENTITY_REFERENCE_MAP = {
    "CATEGORY": (REFERENCE_VIEW_NAME, "category"),
    "COURSE_NAME": (REFERENCE_VIEW_NAME, "subject_name"),
    "DEPARTMENT": (REFERENCE_VIEW_NAME, "dept_name"),
    "CLASS_MODE": (REFERENCE_VIEW_NAME, "class_mode"),
    "EVAL_TYPE": (REFERENCE_VIEW_NAME, "eval_type"),
    "GRADE_METHOD": (REFERENCE_VIEW_NAME, "grading_method"),
}

LABEL_MIN_SCORES = {
    "COURSE_NAME": 70.0,
    "DEPARTMENT": 74.0,
    "CATEGORY": 74.0,
}

COMMON_ALIASES = {
    "DEPARTMENT": {},
    "CATEGORY": {},
}

TYPO_LABEL_MIN_SCORES = {
    "COURSE_NAME": 82.0,
    "DEPARTMENT": 82.0,
    "CATEGORY": 82.0,
    "CLASS_MODE": 86.0,
    "EVAL_TYPE": 86.0,
    "GRADE_METHOD": 86.0,
}
PARTIAL_TYPO_LABELS = {"COURSE_NAME", "DEPARTMENT"}

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
