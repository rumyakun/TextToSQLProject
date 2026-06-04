schema_docs = [
    "v_course_info: integrated course information with day_of_week, start_time, end_time, classroom, prereq_subject_codes, and prereq_subject_names. Query course data from this view only.",
    "course_prerequisite: prerequisite course information",
    "enrollment: student enrollment information",
    "required_subject: graduation requirement information",
    "student: student information",
]


def _tokenize(text: str) -> set[str]:
    return {tok for tok in text.lower().replace(":", " ").replace(",", " ").split() if tok}


def retrieve_schema(query, top_k=2):
    q_tokens = _tokenize(query)
    if not q_tokens:
        return schema_docs[:top_k]

    scores = []
    for doc in schema_docs:
        d_tokens = _tokenize(doc)
        overlap = len(q_tokens.intersection(d_tokens))
        scores.append((doc, overlap))
    scores.sort(key=lambda x: x[1], reverse=True)
    selected = [doc for doc, score in scores[:top_k] if score > 0]
    return selected if selected else schema_docs[:top_k]
