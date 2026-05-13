# Course Frontend (+ FastAPI backend)

이 리포지토리는 **Vite(React) 프론트엔드**와 `backend/`의 **FastAPI(text-to-sql) 백엔드**를 함께 담고 있습니다.

## 실행 방법 (개발)

### 프론트 + 백엔드 같이 실행

```bash
npm install
python -m pip install -r backend/requirements.txt
npm run dev:full
```

- 프론트: `http://localhost:5173`
- 백엔드: `http://127.0.0.1:8000`
- 백엔드 헬스체크: `GET /api/v1/health`

### 환경변수

- 프론트: `.env` (예시는 `.env.example`)
  - `VITE_API_BASE_URL=/api/v1` 권장 (Vite proxy로 백엔드로 전달)
- 백엔드: `backend/.env` (예시는 `backend/.env.example`)
  - `OLLAMA_BASE_URL` (기본 `http://127.0.0.1:11434`)
  - `OLLAMA_MODEL` (예: `text2sql-local`)
  - `DATABASE_URL` (예: `postgresql://postgres:1234@localhost:5432/univ`)
  - `REDIS_URL` (선택)

> `DATABASE_URL`이 없으면 `/api/v1/query`는 에러를 반환하지만, 서버는 기동됩니다.

### 로컬 모델(Ollama) 준비

```bash
ollama create text2sql-local -f "C:\충남대\4-1\종합설계\작업\모델\Modelfile"
ollama run text2sql-local
```

- `Modelfile`의 `FROM` 경로는 실제 `gguf` 위치로 맞춰야 합니다.
- API(OpenAI) 기반 백업 코드는 `backend/backup_api/`에 남겨뒀습니다.

## DB 설정

- 현재 설정은 PostgreSQL을 사용한다는 전제 하에 설정되어 있음. 다른 DB를 사용한다면 `backend/db.py` 파일의 코드 일부를 수정
- `backend/db.py`에서 DATABASE_URL은 본인 환경에 맞게 수정. 예: `postgresql://postgres:[PASSWORD]@localhost:[포트]/[DB명]`
- 추가적으로, 현재는 v_courses_info라는 view를 사용하도록 LLM이 설정되어 있기 때문에, DB 내에서 아래 SQL문을 실행하여 View를 설정해야 함.

```sql

CREATE OR REPLACE VIEW v_course_info AS
SELECT
    c.course_year,
    c.subject_code,
    c.section,
    s.subject_name,
    s.category,
    s.credit_hours,
    s.target_year,
    c.professor,
    c.capacity,
    c.enrolled,
    c.grading_method,
    c.eval_type,
    c.class_mode,
    d.dept_name,
    sch.day_of_week,
    sch.start_time,
    sch.end_time,
    sch.classroom
FROM cnu_courses c
JOIN subject s ON c.subject_code = s.subject_code
JOIN department d ON c.dept_code = d.dept_code
LEFT JOIN course_schedule sch ON c.subject_code = sch.subject_code AND c.section = sch.section;

```
