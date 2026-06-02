from .rag import retrieve_schema


def build_prompt(user_query):
    relevant = retrieve_schema(user_query)

    return f"""
You are a strict SQL generator.

Relevant schema:
{relevant}

Full schema:
v_course_info(course_year, subject_code, section, subject_name, category, credit_hours, target_year, professor, capacity, enrolled, grading_method, eval_type, class_mode, dept_name, day_of_week, start_time, end_time, classroom)

Column details:
- course_year: integer (1, 2, 3, 4). Use exact match: course_year = 1.
- credit_hours: string (e.g., '3', '2'). ALWAYS use strings: credit_hours = '3'.
- day_of_week: string containing Korean days (e.g., '월', '목'). Do NOT use integers.
- category: Course category (e.g., '전공(기초)', '전공(핵심)', '교양(필수)'). Use LIKE for partial matches: category LIKE '%전공(기초)%' or category LIKE '%전공%'.
- start_time: Course start time. For '오전' (morning), use start_time < '12:00'. For '오후' (afternoon), use start_time >= '12:00'.
- If the user mentions an exact clock hour, use exact match on start_time: start_time = 'HH:00:00'.
- Map Korean class-hour queries between 1 and 7 o'clock to 13:00:00 through 19:00:00: 1시 -> '13:00:00', 2시 -> '14:00:00', 3시 -> '15:00:00', 4시 -> '16:00:00', 5시 -> '17:00:00', 6시 -> '18:00:00', 7시 -> '19:00:00'.
- class_mode: Do NOT use this column unless the user explicitly asks for online/offline/real-time classes.

Rules:
- ONLY SELECT
- USE ONLY the tables and columns listed in the schema above.
- DO NOT invent or guess table names (e.g., never use 'courses', use 'v_course_info' instead).
- For time information in the user query, use exact start_time equality, not BETWEEN, >=, <=, >, or <, unless the user explicitly asks for a range, before/after, morning, or afternoon.
- For 1시 through 7시, map to afternoon 24-hour time before exact matching: start_time = '13:00:00' through start_time = '19:00:00'.
- ALWAYS include all category information mentioned by the user (e.g., '전공', '교양', '전공(핵심)') in the category filter.
- For string comparisons (like dept_name, subject_name, category), ALWAYS use LIKE '%word%' instead of exact match '='.
- DO NOT add filters for numeric columns (like credit_hours, course_year) unless the user explicitly mentions a value.
- If unclear return UNKNOWN.

User: {user_query}
SQL:
"""
