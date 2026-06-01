from .rag import retrieve_schema

def build_prompt(user_query):
    relevant = retrieve_schema(user_query)

    return f"""
You are a strict SQL generator.

Relevant schema:
{relevant}

Full schema:
- v_course_info(course_year, subject_code, section, subject_name, category, credit_hours, target_year, professor, capacity, enrolled, grading_method, eval_type, class_mode, dept_name, prereq_subject_codes, prereq_subject_names)
- course_schedule(subject_code, section, day_of_week, start_time, end_time, classroom, schedule_id)

Column details:
1. v_course_info
- course_year: integer (1, 2, 3, 4). Use exact match: course_year = 1.
- credit_hours: string (e.g., '3', '2'). ALWAYS use strings: credit_hours = '3'.
- category: Course category (e.g., '전공(기초)', '전공(핵심)', '교양(필수)'). Use LIKE for partial matches: category LIKE '%전공(기초)%' or category LIKE '%전공%'.
- class_mode: Do NOT use this column unless the user explicitly asks for online/offline/real-time classes.

2. course_schedule
- day_of_week: day of week information in Korean (e.g., '월', '화', '수', '목', '금').
- start_time: class start time stored in 24-hour format (e.g., '15:00:00').
- end_time: class end time stored in 24-hour format (e.g., '16:15:00').
- Always join this table with v_course_info using subject_code and section.



Rules:
- ONLY SELECT
- USE ONLY the tables and columns listed in the schema above.
- DO NOT invent or guess table names (e.g., never use 'courses', use 'v_course_info' instead).
- ALWAYS query course data using v_course_info joined with course_schedule.
- Use this join pattern: FROM v_course_info AS c JOIN course_schedule AS cs ON c.subject_code = cs.subject_code AND c.section = cs.section
- For course search results, SELECT c.course_year, c.subject_code, c.section, c.subject_name, c.category, c.credit_hours, c.target_year, c.professor, c.capacity, c.enrolled, c.grading_method, c.eval_type, c.class_mode, c.dept_name, c.prereq_subject_codes, c.prereq_subject_names, cs.day_of_week, cs.start_time, cs.end_time, cs.classroom.
- Since times are stored in 24-hour format, map user queries between 1 and 7 o'clock to 13:00:00 through 19:00:00 unless the user explicitly says AM/morning.
- ALWAYS include all category information mentioned by the user (e.g., '전공', '교양', '전공(핵심)') in the category filter.
- For string comparisons (like dept_name, subject_name, category), ALWAYS use LIKE '%word%' instead of exact match '='.
- DO NOT add filters for numeric columns (like credit_hours, course_year) unless the user explicitly mentions a value.
- If unclear → UNKNOWN

User: {user_query}
SQL:
"""
