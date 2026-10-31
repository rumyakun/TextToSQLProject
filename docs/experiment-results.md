# 실험·점검 결과 통합 기록

> 2026-10-30 파일 정리: 아래는 작성 당시의 지침·관찰 기록이다. 현재 실행 안내는 [루트 README](../README.md), 설계·결정·현재 상태는 [프로젝트 계획](schema-adaptive-text-to-sql-plan.md)을 따른다. 과거 지침을 새 작업 지시나 현재 성능으로 해석하지 않는다.

## 기록 목차
- [desktop-smoke-results-2026-09-19.md](#desktop-smoke-results-2026-09-19)
- [ollama-adapter-live-smoke-results-2026-09-19.md](#ollama-adapter-live-smoke-results-2026-09-19)
- [phase5-12-synthetic-live-capture-results-2026-09-19.md](#phase5-12-synthetic-live-capture-results-2026-09-19)
- [phase5-13-finetuned-synthetic-exploratory-results-2026-09-19.md](#phase5-13-finetuned-synthetic-exploratory-results-2026-09-19)
- [phase5-14-versioned-queryir-prompt-trial-results-2026-09-19.md](#phase5-14-versioned-queryir-prompt-trial-results-2026-09-19)
- [phase5-19-live-arm-comparison-results-2026-09-28.md](#phase5-19-live-arm-comparison-results-2026-09-28)
- [phase5-20-decontaminated-arm-comparison-results-2026-09-29.md](#phase5-20-decontaminated-arm-comparison-results-2026-09-29)
- [phase5-21-base-instruct-arm-comparison-results-2026-09-29.md](#phase5-21-base-instruct-arm-comparison-results-2026-09-29)
- [phase5-22-symmetric-extraction-results-2026-09-29.md](#phase5-22-symmetric-extraction-results-2026-09-29)
- [phase5-23-schema-variation-results-2026-09-29.md](#phase5-23-schema-variation-results-2026-09-29)
- [phase7-09-ollama-prototype-live-smoke-results-2026-09-28.md](#phase7-09-ollama-prototype-live-smoke-results-2026-09-28)
- [phase7-15-legacy-vs-queryir-comparison-results-2026-09-29.md](#phase7-15-legacy-vs-queryir-comparison-results-2026-09-29)
- [phase7-16-chat-policies-recompare-results-2026-09-29.md](#phase7-16-chat-policies-recompare-results-2026-09-29)
- [phase7-17-anchor-left-join-results-2026-09-29.md](#phase7-17-anchor-left-join-results-2026-09-29)
- [phase8-06-value-dictionary-results-2026-09-30.md](#phase8-06-value-dictionary-results-2026-09-30)
- [phase8-07-value-dictionary-editor-results-2026-09-30.md](#phase8-07-value-dictionary-editor-results-2026-09-30)
- [phase8-08-variant-benchmark-results-2026-09-30.md](#phase8-08-variant-benchmark-results-2026-09-30)
- [phase8-09-ai-mapping-results-2026-09-30.md](#phase8-09-ai-mapping-results-2026-09-30)
- [phase8-10-ai-mapping-v2-results-2026-09-30.md](#phase8-10-ai-mapping-v2-results-2026-09-30)
- [phase8-12-ai-proposal-upload-results-2026-09-30.md](#phase8-12-ai-proposal-upload-results-2026-09-30)
- [phase8-13-chat-value-hints-results-2026-09-30.md](#phase8-13-chat-value-hints-results-2026-09-30)
- [phase8-15-postgres-registration-results-2026-09-30.md](#phase8-15-postgres-registration-results-2026-09-30)
- [phase8-16-registration-list-and-smoke-results-2026-09-30.md](#phase8-16-registration-list-and-smoke-results-2026-09-30)
- [phase8-17-other-school-results-2026-09-30.md](#phase8-17-other-school-results-2026-09-30)
- [phase8-18-target-grade-results-2026-10-01.md](#phase8-18-target-grade-results-2026-10-01)
- [phase8-19-db-extension-fields-results-2026-10-01.md](#phase8-19-db-extension-fields-results-2026-10-01)
- [phase8-20-unknown-conditions-results-2026-10-01.md](#phase8-20-unknown-conditions-results-2026-10-01)
- [phase8-21-code-table-results-2026-10-05.md](#phase8-21-code-table-results-2026-10-05)
- [phase8-21-packed-results-2026-10-06.md](#phase8-21-packed-results-2026-10-06)
- [phase8-21-period-results-2026-10-06.md](#phase8-21-period-results-2026-10-06)
- [phase8-21-wide-table-results-2026-10-02.md](#phase8-21-wide-table-results-2026-10-02)
- [phase8-22-blind-evaluation-results-2026-10-06.md](#phase8-22-blind-evaluation-results-2026-10-06)
- [phase8-23-ai-mapping-results-2026-10-06.md](#phase8-23-ai-mapping-results-2026-10-06)
- [phase8-24-code-table-import-results-2026-10-06.md](#phase8-24-code-table-import-results-2026-10-06)
- [phase8-25-time-expressions-results-2026-10-06.md](#phase8-25-time-expressions-results-2026-10-06)
- [phase9-04-rehearsal-results-2026-10-08.md](#phase9-04-rehearsal-results-2026-10-08)
- [phase9-05-raw-time-and-unasked-select-results-2026-10-08.md](#phase9-05-raw-time-and-unasked-select-results-2026-10-08)
- [phase9-06-numeric-columns-results-2026-10-10.md](#phase9-06-numeric-columns-results-2026-10-10)
- [phase9-07-publish-block-reasons-results-2026-10-10.md](#phase9-07-publish-block-reasons-results-2026-10-10)
- [phase9-08-chat-model-choice-results-2026-10-12.md](#phase9-08-chat-model-choice-results-2026-10-12)
- [phase9-09-openai-chat-model-results-2026-10-12.md](#phase9-09-openai-chat-model-results-2026-10-12)
- [phase9-10-direct-sql-results-2026-10-12.md](#phase9-10-direct-sql-results-2026-10-12)
- [phase9-16-retire-offering-year-results-2026-10-13.md](#phase9-16-retire-offering-year-results-2026-10-13)
- [phase9-21-unmatched-word-notice-results-2026-10-13.md](#phase9-21-unmatched-word-notice-results-2026-10-13)
- [prompts/phase9-10d-results-doc.md](#prompts-phase9-10d-results-doc)

---

<a id="desktop-smoke-results-2026-09-19"></a>

## 원본: desktop-smoke-results-2026-09-19.md

# 데스크톱 smoke test 결과 (2026-09-19)

기준: `docs/development-history.md#desktop-test-handoff`. 합성 질문과 읽기 전용 DB 점검만 수행했다. 실제 질문·모델 응답·SQL·접속 문자열·식별 정보는 이 문서에 기록하지 않았다.

## 환경

- Git: `feat/schema-adaptive-agent`, `251dd91bb56c70cb677d2673a7451b1b47fc9d2d`; 시작 시 작업 트리 깨끗함.
- Windows 데스크톱, Python 3.13.7 (`.venv`), Node 24.15.0, npm 11.17.0, Docker CLI 29.5.3, Ollama 0.34.2.
- `psql` CLI 없음. PostgreSQL 점검은 `.venv`의 `psycopg2`로 수행.
- Docker daemon 접근은 현재 환경에서 거부됨. Compose 실행·컨테이너 통합은 검사하지 않음.
- `.venv`에 `pytest`가 없고 `sqlglot`이 30.11.0이라 처음에는 테스트 1건이 실패했다. `pytest` 9.1.1과 요구사항에 고정된 `sqlglot` 30.18.0을 `.venv`에 설치하고 재실행했다. 저장소 의존성 파일은 변경하지 않음.

## Ollama

- 로컬 `/api/tags` 응답 정상. 설치된 모델은 `text2sql_qwen:latest` 1개이며, 설정된 `OLLAMA_MODEL=text2sql_qwen`으로 호출 성공. 목록의 모델 ID 접두부: `7070730bf2f2`. 실제 weight의 사용 권한·출처는 확인하지 않음.
- 합성 수강 질문 3개와 첫 질문 반복 1회, 고정된 합성 schema context로 `/api/generate` 호출. 4/4 응답 존재, 모두 `SELECT` 문자열 포함, JSON parse 불가, Markdown fence 없음. 첫 호출 약 3.6초, 나머지 약 0.1~0.2초. 이는 출력 형식 관찰이며 정확도·성능 측정값이 아님.
- 기존 모델은 SQL을 반환한다. 새 `QueryIR` 생성 성공으로 해석하지 않음. VRAM·메모리는 측정하지 않음.

## PostgreSQL

- 설정된 로컬 DB에 연결, `SELECT 1` 성공. PostgreSQL 18.3. `information_schema`와 `pg_catalog` 메타데이터 조회 성공.
- 보이는 일반 테이블 8개, view 1개, PK 제약 8개, FK 제약 10개. `v_course_info` 존재, SELECT 권한 있음, 컬럼 20개. `SELECT 1 FROM v_course_info LIMIT 0` 성공; 실제 행은 조회하지 않음.
- 점검 세션에 `transaction_read_only=on`, `statement_timeout=5s` 적용. SSL은 사용 중이지 않음. 연결 계정은 superuser라 계정 자체의 최소 읽기 권한은 증명하지 못함. 운영 DB 권한 정책·실제 데이터 사용 범위도 확인되지 않음.
- DDL/DML 차단 시험은 실행하지 않음. PostgreSQL adaptive compiler/executor 실행 성공으로 해석하지 않음.

## 합성 평가·빌드

- `python -m pytest tests -q`: **254 passed, 3 skipped, 322 subtests passed** (`.venv` Python으로 실행).
- `python -m compileall -q backend tests`, `npm run lint`, `npm run build`: 모두 통과. build에서 오래된 Browserslist 데이터 경고가 나왔으나 빌드는 성공.
- 루트 `python -m pytest --collect-only -q`: **257 tests collected, 1 error**. `backend/test_correction.py`의 `ModuleNotFoundError: No module named 'keyword_extract'`로 중단. adaptive `tests/` 결과와 별개인 기존 수집 문제.
- `python -m backend.adaptive.evaluation --format review-table`: 최초 실행은 Windows 기본 `cp949`의 `UnicodeEncodeError`로 실패. `PYTHONIOENCODING=utf-8` 설정 후 통과; 40건 모두 `NEEDS_HUMAN_REVIEW`.
- `run-smoke`: 합성 gold oracle 120/120 실행·semantic match, 40/40 cross-schema 일관성. 보고서: `.schema-adaptive/reports/desktop-oracle-smoke-20260919.json` (gitignore 경로). 실제 모델 정확도 아님.

## 남은 확인과 다음 작은 작업

- E1·E2·E3·E4 및 L1~L7 사람 승인 상태는 이 smoke test로 변경되지 않음. E5의 이 데스크톱 환경 재현 결과만 위에 기록.
- 실제 모델의 `QueryIR` 출력, PostgreSQL adaptive 실행, 학교 간 일반화, 전체 서비스 end-to-end는 미검증.
- 다음 개발 단위: Ollama adapter의 JSON 반환 및 timeout/empty/provider 오류 변환을 합성 transport로 구현·시험. 실제 DB 질의 전에는 superuser가 아닌 승인된 읽기 전용 계정과 데이터 사용 범위를 확인.

---

<a id="ollama-adapter-live-smoke-results-2026-09-19"></a>

## 원본: ollama-adapter-live-smoke-results-2026-09-19.md

# Ollama adapter 합성 live smoke 결과 (2026-09-19)

기준: `docs/development-history.md#prompts-phase5-08-ollama-adapter-live-smoke`. 합성 질문만 로컬 Ollama에 보냈다. 질문·모델 응답·프롬프트·HTTP body·실제 SQL·DB 행·접속 정보는 저장하거나 이 문서에 기록하지 않았다.

## 환경과 실행

- Windows 데스크톱, `.venv` Python 3.13.7, 로컬 Ollama 0.34.2. 실제 버전·모델 사용 가능 여부는 실행 직전 CLI와 `/api/tags`에서 확인했다.
- 확인된 정확한 모델 tag: `text2sql_qwen:latest`; `/api/tags` digest: `7070730bf2f29aa3e7cbb9b0b566b1c73fd93bfeab41363e4a1f128730a3cb5a`. 모델 weight의 출처·사용 권한은 확인하지 않았다.
- 명시적 `--live`, loopback base URL, timeout 30초, prompt version `query-ir-v1`. 합성 `Ontology`·`PUBLISHED MappingProfile`, 합성 질문 2건을 각 1회 호출했다. retry·fallback·DB 실행·recording 저장은 없었다.

## 안전한 형식 관찰

| case ID | provider 상태 | JSON 파싱·타입 | QueryIR 구조 | semantic field/operator | 경과 시간 |
| --- | --- | --- | --- | --- | ---: |
| `synthetic-01` | 성공 | 성공·object | 유효 | 유효 | 약 6.6초 |
| `synthetic-02` | 성공 | 성공·object | 유효 | 유효 | 약 0.3초 |

provider 실패 코드는 없었다. 이 결과는 **두 합성 입력에서 adapter의 실제 HTTP 전송과 형식 분류가 동작했다**는 뜻이다. 기존 데스크톱 smoke에서 같은 모델이 다른 프롬프트로 SQL 텍스트를 반환한 관찰과 양립한다. QueryIR 의미 정확도, 실제 데이터 질의 정확도, 40×3 점수, 학교 간 일반화나 안정적인 지연은 증명하지 않는다.

## 검증 경계와 다음 작업

- opt-in 경계, 단일 호출, JSON 문자열과 비 JSON 텍스트의 구분, 구조·semantic 오류 코드, 출력 비노출, provider 실패 후 중단을 합성 transport 테스트로 확인했다.
- 실제 응답을 기록·저장·공유하기 전 E3·L5, 모델 성능 수치를 확정하기 전 L1·L3 사람 확인이 필요하다. gold 40건은 모두 `NEEDS_HUMAN_REVIEW` 상태다.
- 다음 작은 작업은 별도 범위의 private live capture CLI·provenance 연결을 설계하고, 실제 응답을 저장하기 전에 데이터·보안 승인 범위를 확인하는 것이다. PostgreSQL 통합과 40×3 평가는 별도 단계다.

---

<a id="phase5-12-synthetic-live-capture-results-2026-09-19"></a>

## 원본: phase5-12-synthetic-live-capture-results-2026-09-19.md

# Phase 5-12 합성 private live capture 결과 (2026-09-19)

> 추가 기록(2026-09-29, Phase 5-20·7-11): 이 실행의 Ollama 호출은 `text2sql_qwen:latest` Modelfile에 내장된 CNU `courses` schema SYSTEM이 주입된 조건(`model-template`)이었다. 수집 당시 provenance에는 이 조건이 드러나지 않는다. no-system 조건과 비교할 때는 ADR-058~060을 따른다.

기준: `docs/development-history.md#prompts-phase5-12-synthetic-live-private-capture`. 완전 합성 질문·ontology·mapping 2건만 로컬 Ollama에 전송했다. 질문·모델 응답·프롬프트·HTTP body·물리 mapping·SQL·DB 행은 이 문서와 공개 로그에 기록하지 않았다.

## 환경과 실행

- 로컬 Ollama 0.34.2, 정확한 model tag `text2sql_qwen:latest`를 `ollama list`와 `/api/tags`에서 확인했다. 선언 digest는 `/api/tags` 값과 일치한다. 모델 출처의 독립 증명은 아니다.
- `capture-targets-v1`의 합성 target 2건, 명시적 base URL·timeout 30초·model run ID·새 private output을 사용했다. 기존 bundle 덮어쓰기·fallback·retry 없음.
- `--check` 먼저 성공: target 2건, 선언 provenance fingerprint `86e32b0572e9d644f4a1a245b9b499d072e292df7698f501acd62db21f8f03d0`. 이후 `--live` 1회 실행해 성공 2건·실패 0건을 private bundle에 저장했다.
- Private target·bundle은 Git ignore와 비추적 상태를 확인했다. `recording-bundle-v1` strict loader로 manifest·artifact hash·건수·`DECLARED_UNVERIFIED` provenance를 검증했다. `RecordedModel`이 입력 순서대로 두 응답을 재생했다.

## 안전한 형식 분류

| case ID | capture | JSON 여부·타입 | QueryIR 구조·semantic 검증 |
| --- | --- | --- | --- |
| `synthetic-live-01` | 성공 | 파싱됨·object | `VALID_QUERY_IR` |
| `synthetic-live-02` | 성공 | 파싱됨·object | `VALID_QUERY_IR` |

응답 본문은 QueryIR로 보정하거나 SQL로 실행하지 않았다. 이 결과는 두 합성 입력의 adapter 전송, private storage, strict load, context-bound replay와 현재 형식·semantic 계약만 검증한다. 모델 정확도, 40×3 benchmark, 학교 간 일반화 또는 안정적인 지연의 증거가 아니다.

E3·L5와 L1·L3는 각 실제 사용·공유·성능 확정 관문까지 대기한다. Gold 40건은 모두 `NEEDS_HUMAN_REVIEW`다.

---

<a id="phase5-13-finetuned-synthetic-exploratory-results-2026-09-19"></a>

## 원본: phase5-13-finetuned-synthetic-exploratory-results-2026-09-19.md

# Phase 5-13 fine-tuned 모델 합성 탐색 평가 (2026-09-19)

> 추가 기록(2026-09-29, Phase 5-20·7-11): 이 실행의 Ollama 호출은 `text2sql_qwen:latest` Modelfile에 내장된 CNU `courses` schema SYSTEM이 주입된 조건(`model-template`)이었다. 수집 당시 provenance에는 이 조건이 드러나지 않는다. no-system 조건과 비교할 때는 ADR-058~060을 따른다.

> 상태: **잠정·미승인 탐색 결과**. Gold 40건 모두 `NEEDS_HUMAN_REVIEW`. 논문용 확정 모델 성능 아님.

## 환경과 실행 범위

- 로컬 Ollama 0.34.2, 정확한 `text2sql_qwen:latest` tag와 `/api/tags` digest 확인. 모델 weight 출처와 선언 provenance는 독립 검증하지 않았다.
- 기존 합성 benchmark v1의 질문 40건 × flat·normalized·legacy fixture 3개로 `capture-targets-v1` 120건을 결정적으로 생성했다. 요청 ID는 평가 runner의 규칙과 동일하다. Gold IR·기대 결과·SQL·DB 행은 capture 입력과 모델 prompt에 넣지 않았다.
- `--check` 120건 통과 후 명시적 `--live` 1회 실행. capture 성공 120건, provider 실패 0건. Retry·fallback·overwrite 없음.
- Git 비추적 private bundle의 manifest·artifact hash·count·`DECLARED_UNVERIFIED` provenance를 strict load했다. 기존 `run-recorded`를 합성 SQLite fixture에만 연결해 오프라인 재생했다. Private target·bundle·상세 report는 Git 비추적 상태다.

## 잠정 집계

| 항목 | 건수 / 전체 120 attempt |
| --- | ---: |
| 예정·완료 attempt | 120 / 120 |
| 구조적 QueryIR exact | 51 / 120 |
| 하네스 실행 성공 | 99 / 120 |
| 합성 gold semantic 일치 | 90 / 120 |
| 비교 불가 | 0 / 120 |

통제된 실패 유형: `IR_VALIDATED:UNKNOWN_FIELD` 21건, `RESULT_VALIDATED:SEMANTIC_MISMATCH` 9건. 실패와 불일치를 전체 120 attempt 분모에서 제외하지 않았다. 모델 응답을 QueryIR로 보정하지 않았다.

이 집계는 합성 fixture와 미승인 gold에 한정된 탐색값이다. 실제 CNU DB 정확도, 다른 학교 스키마 일반화, 최종 fine-tuned 모델 성능을 증명하지 않는다. `run-recorded`는 offline replay이며 provenance도 `DECLARED_UNVERIFIED`다. E3/L5·L1/L3 결정은 각 실제 사용·공유·성능 확정 관문까지 대기한다.

공개 문서·stdout에 질문·응답·프롬프트·물리 mapping·SQL·결과 행·HTTP body·절대 경로를 복제하지 않았다.

---

<a id="phase5-14-versioned-queryir-prompt-trial-results-2026-09-19"></a>

## 원본: phase5-14-versioned-queryir-prompt-trial-results-2026-09-19.md

# Phase 5-14 versioned QueryIR prompt 단일 탐색 비교 (2026-09-19)

> 추가 기록(2026-09-29, Phase 5-20·7-11): 이 실행의 Ollama 호출은 `text2sql_qwen:latest` Modelfile에 내장된 CNU `courses` schema SYSTEM이 주입된 조건(`model-template`)이었다. 수집 당시 provenance에는 이 조건이 드러나지 않는다. no-system 조건과 비교할 때는 ADR-058~060을 따른다.

> 상태: **같은 미승인 합성 개발 세트의 잠정 결과**. Gold 40건 모두 `NEEDS_HUMAN_REVIEW`. 모델 성능 확정 또는 일반화 근거가 아니다.

## 가설과 조건

Phase 5-13 v1의 `UNKNOWN_FIELD` 21건을 private 응답에서 읽기 전용으로 분류했다. 첫 unknown field 기준 ontology 밖 semantic field 12건, entity 접두사 불일치 5건, 물리 column 형태 4건, 대소문자·표기 차이 0건, 그 외 0건, 판별 불가 0건이다. 중복 unknown 참조 0건, 서로 다른 범주가 한 attempt에 공존한 경우 3건이다. 분류 범주 합계는 21건이며 중복·혼합은 별도 보조 건수다.

가설: 허용 semantic field 문자열을 목록에서 정확히 복사하도록 지시하면 `UNKNOWN_FIELD`가 감소한다. 기존 `query-ir-v1` 기본값과 전송 설정을 유지하고, 명시적 `query-ir-v2` 한 가지를 추가했다. v1 template SHA-256은 `9546599f1b1cb0f733c7ff6a7d02c0d47d6dba6a0767263acc37869bb74774d5`, v2는 `e1400acd35e2601cc35f51062b5516c35e14bec41a46ac8eecd9b4d0a3f43b83`이다. 두 버전의 generation config SHA-256은 `48ba68c94123f52c8d0d5f531f69fa29cdea6107a8a2fe69e2d937cfa14126e0`으로 같다.

Ollama 0.34.2의 정확한 `text2sql_qwen:latest` tag와 digest를 다시 확인했다. 합성 40질문 × 3 fixture의 같은 의미 context를 사용하고 새 run ID에 맞게 request ID만 재생성했다. `--check` 후 v2 `--live`를 한 번 실행해 120 target 모두 수집 성공, provider 실패 0건이었다. Private bundle strict load, 선언 provenance 확인, `run-recorded` 오프라인 재생, case×variant 120쌍 대응을 검증했다. 재시도·fallback·응답 보정은 없다. V1/v2 private target·bundle·상세 report는 Git 비추적 상태다.

## Paired 잠정 결과

| 항목 (분모 120) | v1 | v2 | v2만 성공/발생 | v1만 성공/발생 |
| --- | ---: | ---: | ---: | ---: |
| 구조적 QueryIR exact | 51 | 0 | 0 | 51 |
| 하네스 실행 성공 | 99 | 42 | 3 | 60 |
| 합성 gold semantic 일치 | 90 | 36 | 3 | 57 |
| `UNKNOWN_FIELD` | 21 | 72 | 54 | 3 |
| provider 실패 | 0 | 0 | — | — |
| recording 누락 | 0 | 0 | — | — |

V2의 나머지 실패는 `MODEL_INVALID_OUTPUT` 6건, semantic mismatch 6건이다. V2 `UNKNOWN_FIELD` 72건의 첫 unknown 기준 분류는 entity 접두사 불일치 46, ontology 밖 6, 물리 column 형태 5, 그 외 15, 판별 불가 0이다. 중복 참조 27개, 혼합 범주 attempt 50건이다. **가설은 이 실험에서 지지되지 않았다.** V2는 기본값으로 전환하지 않는다.

이 세트는 v1 결과를 본 뒤 v2를 설계한 개발 세트다. 결과는 독립 검증 정확도나 실제 CNU DB 성능을 뜻하지 않는다. Gold 사람 검토와 provenance 독립 검증도 미완료다. E3/L5·L1/L3 팀 결정은 필요한 실제 사용·공유·성능 확정 관문까지 대기한다.

---

<a id="phase5-19-live-arm-comparison-results-2026-09-28"></a>

## 원본: phase5-19-live-arm-comparison-results-2026-09-28.md

# Phase 5-19 로컬 모델 IR·직접 SQL arm paired 비교 결과(잠정, 2026-09-28)

기준: [`docs/development-history.md#prompts-phase5-19-live-arm-comparison`](development-history.md#prompts-phase5-19-live-arm-comparison), 계약은 Phase 5-18·ADR-057. 완전 합성 fixture 3종과 개발 세트 40건만 사용했다. 질문·prompt·모델 출력·SQL·행 값은 이 문서와 report에 기록하지 않았다.

> **잠정치.** gold 40건은 모두 `NEEDS_HUMAN_REVIEW`이며 L1(gold 검토)·L3(결과 검토) 전이다. 논문의 확정 성능이나 우열 결론으로 쓰지 않는다.

> **정정(2026-09-29, Phase 5-20).** 이 실행은 `text2sql_qwen:latest`의 Modelfile에 내장된 CNU `courses` schema SYSTEM이 두 arm의 모든 호출에 주입된 조건이었다(입력 token 114개 차이로 확인). 아래 "CNU schema 재현 경향"은 모델 암기라기보다 이 SYSTEM을 따른 결과였을 가능성이 높으므로 해석으로 쓰지 않는다. 오염을 제거한 재실행 결과(IR 93, SQL 72)는 [`phase5-20-decontaminated-arm-comparison-results-2026-09-29.md`](experiment-results.md#phase5-20-decontaminated-arm-comparison-results-2026-09-29)를 기준으로 한다. 아래 수치는 조건 기록으로만 보존한다.

## 환경

- Windows 데스크톱, 로컬 Ollama `0.34.4`, 기본 loopback URL, timeout 60초, temperature 0, attempt당 1회·retry 없음.
- model tag `text2sql_qwen:latest`(CNU `courses` 단일 schema로 fine-tune된 기존 모델), digest `7070730bf2f29aa3e7cbb9b0b566b1c73fd93bfeab41363e4a1f128730a3cb5a`.
- IR arm prompt `query-ir-v1`(ontology semantic field만), SQL arm prompt `direct-sql-v1`(fixture Catalog의 table·column·PK·FK), 템플릿 SHA-256은 report에 기록.
- 실행: `--check`(호출 0회) 뒤 `--live-synthetic` 1회, 240 호출, 약 1분 46초. report는 Git 비추적 `.schema-adaptive/reports/p519-text2sql-qwen-20260928/`.

## 결과 (분모 = 전체 scheduled attempt 120)

| 지표 | IR arm | SQL arm |
| --- | ---: | ---: |
| 실행 성공 | 99 | 29 |
| 정답(IR: semantic 일치, SQL: 위치 무관 결과 일치) | **90** | **17** |
| IR 구조 exact | 51 | — |
| SQL 정책 통과(valid SQL) | — | 69 |
| UNKNOWN(기권) | — | 0 |

IR arm 수치는 Phase 5-13(같은 tag·prompt)과 정확히 같아 temperature 0에서 재현되었다.

paired count:

| 범위 | 둘 다 정답 | IR만 | SQL만 | 둘 다 오답 | IR 정답 | SQL 정답 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 전체(120) | 12 | 78 | 5 | 25 | 90 | 17 |
| flat(40) | 4 | 26 | 2 | 8 | 30 | 6 |
| normalized(40) | 1 | 29 | 1 | 9 | 30 | 2 |
| legacy(40) | 7 | 23 | 2 | 8 | 30 | 9 |

## 실패 구성

- SQL arm outcome: `SQL_UNKNOWN_TABLE` 51(flat 29, legacy 22), `EXECUTION_ERROR` 40(normalized 38, legacy 2), `INCORRECT_RESULT` 11, `COLUMN_COUNT_MISMATCH` 1, `CORRECT` 17.
- IR arm 실패: `UNKNOWN_FIELD` 21, `SEMANTIC_MISMATCH` 9(Phase 5-14의 기존 분류와 동일).
- 비노출 구조 진단(별도 SQL arm 120 호출, in-memory 분류, 수치가 live와 일치):
  - `SQL_UNKNOWN_TABLE` 51건 전부가 **기존 CNU schema table(`courses`·`v_course_info`·`course_schedule`)만** 참조했다.
  - `EXECUTION_ERROR` 40건 전부가 `no such column`이었다. normalized fixture는 table 검사를 통과했으나 column 이름에서 실패했다.

## 해석과 한계

- ~~관찰: 기존 fine-tuned 모델은 동적 schema를 prompt로 받아도 학습한 CNU table·column을 재현하는 경향이 강했다.~~ **정정됨:** 모델이 내장 SYSTEM으로 CNU schema를 직접 받은 조건이었으므로 암기의 근거가 아니다(Phase 5-20).
- 같은 모델에서 QueryIR 경로는 물리 schema를 모델에 노출하지 않고 compiler가 fixture별 SQL을 만들기 때문에 이 암기 문제를 우회한다. 다만 IR arm에도 `UNKNOWN_FIELD` 21건 등 별도 실패가 있다.
- 공정성 한계:
  - 모델은 CNU 단일 schema SQL로 fine-tune되었으므로 동적 schema SQL은 학습 분포 밖이다. base instruct 모델(§15.1-3)과의 비교가 있어야 "QueryIR 방식 자체"의 효과를 분리할 수 있다. 로컬에는 base tag가 아직 없다.
  - SQL arm 비교는 column 이름을 보지 않고 값만 비교해 IR arm보다 약간 관대하다(ADR-057).
  - 개발 세트 40건은 prompt 탐색(Phase 5-14)에 쓰인 세트다. 독립 holdout 30건은 사람 검토·동결 전이라 실행하지 않았다.
  - 표본이 작고 gold가 미승인이라 통계 검정은 하지 않았다.

## 부수 기록

- Phase 5-19 CLI 테스트 작성 중 SQL provider가 transport factory를 이름으로 묶어 monkeypatch를 우회한 결함이 있었고, 그 테스트 실행 1회가 존재하지 않는 합성 tag로 로컬 Ollama에 120회 요청했다. server log상 모두 `404`(약 1ms, 생성 없음)였다. factory를 module lookup으로 고치고 회귀 테스트를 추가했으며 이후 전체 테스트는 Ollama 호출이 없었다.

## 사람 관문

완전 합성 입력만 사용해 E3·L5 대상 자료가 없다. 수치는 L1/L3 전 잠정치이며 승인 상태는 바꾸지 않았다.

---

<a id="phase5-20-decontaminated-arm-comparison-results-2026-09-29"></a>

## 원본: phase5-20-decontaminated-arm-comparison-results-2026-09-29.md

# Phase 5-20 내장 SYSTEM 제거 후 IR·직접 SQL arm paired 비교 결과(잠정, 2026-09-29)

기준: [`docs/development-history.md#prompts-phase5-20-ollama-system-decontamination`](development-history.md#prompts-phase5-20-ollama-system-decontamination). 계약은 Phase 5-18·ADR-057, 요청 형식은 ADR-058. 완전 합성 fixture 3종과 개발 세트 40건만 사용했고 질문·prompt·출력·SQL·행 값은 기록하지 않았다.

> **잠정치.** gold 40건은 `NEEDS_HUMAN_REVIEW`이며 L1·L3 전이다. 확정 성능·우열 결론으로 쓰지 않는다.

## 오염 확인

- `text2sql_qwen:latest` Modelfile의 SYSTEM(CNU `courses` table·column, 한국어는 CP949 byte)은 `system` 미지정 `/api/generate` 호출마다 주입된다.
- 같은 prompt의 입력 token 수: 기본 호출 161, 등록 TEMPLATE을 SYSTEM 없이 raw로 재현한 호출 47(차이 114). raw 재현 template에 임의 SYSTEM을 넣은 호출과 Ollama의 `system` 지정 호출이 둘 다 49로 같아 template 재현이 byte 단위로 맞음을 확인했다.

## 실행

- 로컬 Ollama `0.34.4`, tag `text2sql_qwen:latest`(digest `7070730b…a3cb5a`, GGUF 메타데이터 `general.name=Model_Gguf`·`quantized_by=Unsloth`로 fine-tune 산출물), timeout 60초, temperature 0, 호출 1회·retry 없음.
- 두 arm 모두 `alpaca-raw-no-system-v1`. 240 호출 약 1분 37초. 최종 코드로 재실행해 모든 수치가 같음을 확인했다. report는 Git 비추적 `.schema-adaptive/reports/p520-text2sql-qwen-nosys-final/`.

## 결과 (분모 120)

| 지표 | 5-19 IR (오염) | **5-20 IR** | 5-19 SQL (오염) | **5-20 SQL** |
| --- | ---: | ---: | ---: | ---: |
| 정답 | 90 | **93** | 17 | **72** |
| 실행 성공 | 99 | 111 | 29 | 92 |
| IR 구조 exact / SQL 정책 통과 | 51 | 51 | 69 | 120 |

paired count(5-20):

| 범위 | 둘 다 정답 | IR만 | SQL만 | 둘 다 오답 | IR 정답 | SQL 정답 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 전체(120) | 68 | 25 | 4 | 23 | 93 | 72 |
| flat(40) | 28 | 3 | 2 | 7 | 31 | 30 |
| normalized(40) | 11 | 20 | 0 | 9 | 31 | 11 |
| legacy(40) | 29 | 2 | 2 | 7 | 31 | 31 |

- IR arm은 세 schema에서 모두 31/40이고 cross-schema 일관성 37/40이다. prompt에 물리 schema가 없어 모델 출력이 schema와 무관하고 compiler가 fixture별 SQL을 만든다. 실패는 `UNKNOWN_FIELD` 9, `SEMANTIC_MISMATCH` 18.
- SQL arm은 단일 table인 flat·legacy에서 IR과 거의 같고(30·31) normalized에서 11/40으로 떨어진다. normalized 실패 29건 중 28건이 `EXECUTION_ERROR`이며, 비노출 진단(40 호출)으로 `no such column` 27·`ambiguous column` 1, 실패 SQL의 JOIN 수는 0개 14·1개 13·3개 1이었다. 정규화 schema에서 column이 속한 table을 잘못 고르거나 필요한 JOIN을 만들지 못하는 경향으로 보인다.

## 해석과 한계

- 관찰: 이 fine-tune 모델·합성 fixture에서 QueryIR 경로의 이점은 **여러 table JOIN이 필요한 정규화 schema에 집중**되었고, 단일 table schema에서는 직접 SQL과 거의 같았다. 이는 "schema 구조가 달라질 때 compiler가 JOIN을 대신 처리한다"는 설계 가설과 일치하는 관찰이다.
- Phase 5-19의 SQL 17/120과 "CNU schema 암기" 해석은 내장 SYSTEM 오염 때문이었을 가능성이 높아 정정한다(5-19 문서 참고).
- 한계:
  - 모델은 CNU 단일 schema SQL로 fine-tune되었다. base instruct 모델 비교(§15.1-3)가 있어야 QueryIR 효과와 fine-tune 효과를 분리할 수 있다.
  - fixture는 같은 6개 개설강좌의 구조 변형 3종뿐이고, 개발 세트 40건은 prompt 탐색(5-14)에 쓰였다. holdout 30건은 미실행.
  - SQL arm은 column 이름을 비교하지 않아 IR arm보다 약간 관대하다.
  - 표본이 작고 gold가 미승인이라 통계 검정을 하지 않았다.
- 후속: runtime `OllamaQueryModel`·prototype 호출도 같은 SYSTEM이 주입되는 조건이다(7-09 포함). runtime 요청 형식은 별도 단계에서 명시해야 한다.

## 사람 관문

완전 합성 입력만 사용했다(E3·L5 대상 없음). 수치는 L1/L3 전 잠정치이며 승인 상태는 바꾸지 않았다.

---

<a id="phase5-21-base-instruct-arm-comparison-results-2026-09-29"></a>

## 원본: phase5-21-base-instruct-arm-comparison-results-2026-09-29.md

# Phase 5-21 base instruct 모델 IR·직접 SQL arm paired 비교 결과(잠정, 2026-09-29)

기준: [`docs/development-history.md#prompts-phase5-21-base-instruct-arm-comparison`](development-history.md#prompts-phase5-21-base-instruct-arm-comparison). 완전 합성 fixture 3종과 개발 세트 40건만 사용했고 질문·prompt·출력·SQL·행 값은 기록하지 않았다.

> **잠정치이며, 이 실행의 IR arm 수치는 방식 비교로 쓸 수 없다.** 아래 "비교 비대칭" 참고. gold 40건도 `NEEDS_HUMAN_REVIEW`(L1·L3 전)다.

## 환경

- 로컬 Ollama `0.34.4`, tag `qwen2.5-coder:7b-instruct`(Ollama 공식 레지스트리, Q4_K_M, manifest digest `d9bb33f2…c8fb446`, model blob `sha256-60e05f21…bfc2463`), 저장 경로 `C:/Users/micea/.ollama/models`.
- 등록 TEMPLATE은 ChatML, 기본 SYSTEM은 일반 문구 `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`
- 요청 형식 `model-template`(사용자 결정). fine-tune 모델의 5-20 조건(`alpaca-raw-no-system-v1`, system 없음)과 system 조건이 다르다.
- `query-ir-v1`·`direct-sql-v1`, temperature 0, timeout 60초, 호출 1회·retry 없음. `--check`(호출 0회) 뒤 live 1회, 240 호출 약 2분. report는 Git 비추적 `.schema-adaptive/reports/p521-qwen25-coder-7b-instruct-template/`.

## 결과 (분모 120)

| 지표 | IR arm | SQL arm |
| --- | ---: | ---: |
| 정답 | 0 | 90 |
| 실행 성공 | 0 | 102 |
| SQL 정책 통과 | — | 108 |
| UNKNOWN(기권) | — | 12 |

- IR arm 실패는 120건 모두 `IR_GENERATED:MODEL_INVALID_OUTPUT`이었다.
- SQL arm variant별 정답: flat 34, normalized 24, legacy 32. 실패는 기권 12(flat 2·normalized 6·legacy 4), `INCORRECT_RESULT` 12, `EXECUTION_ERROR` 6(normalized).

## IR arm 원인 진단(비노출 구조 분류)

IR prompt는 물리 schema를 포함하지 않아 세 fixture에서 같으므로 flat 40건을 별도 진단 호출해 분류했다.

- 40건 모두 **단일 Markdown code fence로만 감싼 출력**이었다(prompt의 "no Markdown" 지시 위반).
- fence 안쪽: 유효 QueryIR(구조·semantic 검증 통과) 28, 구조 검증 실패 8, semantic 검증 실패 4.

## 비교 비대칭

- SQL arm 추출 규칙(`direct-sql-extract-v1`)은 단일 fence를 벗긴다. IR arm은 `OllamaQueryModel`이 JSON 파싱에 실패한 문자열을 그대로 넘겨 fence가 있으면 무조건 실패한다(Phase 5-07 계약).
- fine-tune 모델은 fence 없는 JSON을 내서 이 비대칭이 드러나지 않았지만, base 모델에서는 IR arm 전체를 0으로 만들었다.
- 따라서 이 실행의 IR 0/120 대 SQL 90/120은 **방식의 차이가 아니라 추출 규칙 차이**를 주로 반영한다. 방식 비교에 쓰려면 두 arm에 같은 추출 규칙을 적용한 재실행이 필요하다(결정 대기).

## 관찰(잠정)

- base 모델의 SQL arm(90/120, normalized 24/40)은 fine-tune 모델의 5-20 SQL arm(72/120, normalized 11/40)보다 높았다. 동적 schema SQL에서는 CNU 단일 schema fine-tune이 오히려 불리했을 가능성이 있다. 다만 system 조건이 달라(5-21 일반 system, 5-20 system 없음) 직접 비교에는 한계가 있다.
- fence 안쪽 유효 QueryIR 비율(28/40)은 참고값이며 실행·semantic 정답이 아니다.

## 사람 관문

완전 합성 입력만 사용했다(E3·L5 대상 없음). 수치는 L1/L3 전 잠정치이며 승인 상태는 바꾸지 않았다.

---

<a id="phase5-22-symmetric-extraction-results-2026-09-29"></a>

## 원본: phase5-22-symmetric-extraction-results-2026-09-29.md

# Phase 5-22 대칭 추출 규칙으로 두 모델 IR·직접 SQL arm 재실행 결과(잠정, 2026-09-29)

기준: [`docs/development-history.md#prompts-phase5-22-symmetric-fence-extraction`](development-history.md#prompts-phase5-22-symmetric-fence-extraction). 완전 합성 fixture 3종과 개발 세트 40건만 사용했고 질문·prompt·출력·SQL·행 값은 기록하지 않았다.

> **잠정치.** gold 40건은 `NEEDS_HUMAN_REVIEW`(L1·L3 전)이며 확정 성능·우열 결론으로 쓰지 않는다.

## 조건

- 두 arm 모두 같은 단일 code fence 규칙: SQL arm `direct-sql-extract-v1`, IR arm `single-fence-v1`(같은 탐지 함수, fence가 정확히 1개일 때만 안쪽 사용, 그 외 보정 없음). runtime은 바뀌지 않았다.
- fine-tune: `text2sql_qwen:latest`, `alpaca-raw-no-system-v1`(system 없음).
- base: `qwen2.5-coder:7b-instruct`, `model-template`(일반 기본 system `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`).
- `query-ir-v1`·`direct-sql-v1`, temperature 0, timeout 60초, 호출 1회·retry 없음, 모델당 240 호출(약 1분 30초·2분). report는 Git 비추적 `.schema-adaptive/reports/p522-*`.

## 결과 (분모 120, 괄호는 flat / normalized / legacy 각 40)

| 모델 | IR 정답 | SQL 정답 | 둘 다 | IR만 | SQL만 | 둘 다 오답 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fine-tune | 93 (31/31/31) | 72 (30/11/31) | 68 | 25 | 4 | 23 |
| base | 78 (26/26/26) | 90 (34/24/32) | 70 | 8 | 20 | 22 |

- fine-tune 결과는 Phase 5-20과 모든 수치가 같다(fence 없는 JSON을 내므로 규칙 영향 없음).
- base IR: 실행 성공 84, 구조 exact 36, cross-schema 일관성 28/40. 실패는 출력 구조 무효 24(fence 안쪽 JSON이 QueryIR shape가 아님, variant당 8), `UNKNOWN_FIELD` 9, `INVALID_VALUE_TYPE` 3, `SEMANTIC_MISMATCH` 6.
- base SQL: 실행 성공 102, 정책 통과 108, 기권 12(flat 2·normalized 6·legacy 4), `INCORRECT_RESULT` 12, `EXECUTION_ERROR` 6(normalized).

## 관찰(잠정)

1. **IR arm은 두 모델 모두 schema 구조와 무관하게 같은 정답 수**를 냈다(31·31·31, 26·26·26). prompt에 물리 schema가 없고 compiler가 JOIN을 만들기 때문이다.
2. **직접 SQL arm은 정규화 schema에서 떨어졌다**(fine-tune 30→11, base 34·32→24).
3. base 모델에서는 **단일 table schema에서 직접 SQL이 IR보다 높았고**(flat 34 대 26, legacy 32 대 26), **정규화 schema에서는 비슷했다**(24 대 26). 전체로는 SQL 90 대 IR 78이다.
4. fine-tune 모델의 큰 IR 우위(93 대 72, normalized 31 대 11)는 QueryIR 방식의 효과와 함께 **CNU 단일 schema fine-tune이 동적 schema SQL을 약화시킨 효과**가 섞인 결과로 보인다. base의 SQL(90)이 fine-tune의 SQL(72)보다 높다.
5. base 모델의 IR 실패 상당수(24)는 QueryIR JSON shape를 지키지 못한 출력이다. `query-ir-v1` prompt는 fine-tune 모델에서만 탐색되었으므로(Phase 5-14) base 모델에 불리할 수 있다.

## 해석과 한계

- 현재 근거로 "QueryIR이 직접 SQL보다 정확하다"는 일반 주장은 할 수 없다. 방어 가능한 관찰은 **QueryIR이 schema 구조 변화에 대해 정확도가 일정하고, 직접 SQL은 정규화 schema에서 떨어진다**는 점이며, 이는 schema 적응성 주장과 맞닿아 있다.
- system 조건이 다르다(fine-tune은 system 없음, base는 일반 기본 system).
- fixture는 같은 6개 개설강좌의 구조 변형 3종뿐이고, 개발 세트 40건은 prompt 탐색에 쓰였다. holdout 30건 미실행.
- SQL arm은 column 이름을 비교하지 않아 IR arm보다 약간 관대하다(ADR-057).
- 모델당 1회 실행, 통계 검정 없음.

## 사람 관문

완전 합성 입력만 사용했다(E3·L5 대상 없음). 수치는 L1/L3 전 잠정치이며 승인 상태는 바꾸지 않았다.

---

<a id="phase5-23-schema-variation-results-2026-09-29"></a>

## 원본: phase5-23-schema-variation-results-2026-09-29.md

# Phase 5-23 schema 변형 5종에서 두 모델 IR·직접 SQL arm 일관성 결과(잠정, 2026-09-29)

기준: [`docs/development-history.md#prompts-phase5-23-schema-variation-claim-scope`](development-history.md#prompts-phase5-23-schema-variation-claim-scope), ADR-061·ADR-062. 완전 합성 fixture와 개발 세트 40건만 사용했고 질문·prompt·출력·SQL·행 값은 기록하지 않았다.

> **잠정치.** gold 40건은 `NEEDS_HUMAN_REVIEW`(L1·L3 전)이며 확정 성능·우열 결론으로 쓰지 않는다.

## 조건

- fixture suite `extended-v1`: flat·normalized·legacy(기존) + coded(normalized 구조·약어 이름) + hybrid(2-table 부분 비정규화·영문 동의어 이름). 모두 같은 seed 6개 개설강좌.
- 두 arm 모두 단일 code fence 규칙(`single-fence-v1` / `direct-sql-extract-v1`), `query-ir-v1`·`direct-sql-v1`, temperature 0, timeout 60초, 호출 1회·retry 없음, 모델당 400 호출.
- fine-tune `text2sql_qwen:latest`(`alpaca-raw-no-system-v1`), base `qwen2.5-coder:7b-instruct`(`model-template`).
- report는 Git 비추적 `.schema-adaptive/reports/p523-finetune-ext`, `p523-base-ext`(IR·SQL report, paired comparison, `schema-consistency-v1`).
- core 3종의 variant별 수치는 Phase 5-22와 모두 같았다(재현 확인).

## variant별 정답 (각 분모 40)

| 모델 · arm | flat | normalized | legacy | coded | hybrid | 합계 /200 | spread | 모든 variant 정답 case |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fine-tune · IR | 31 | 31 | 31 | 31 | 31 | 155 | 0 | 31 |
| fine-tune · SQL | 30 | 11 | 31 | 8 | 10 | 90 | 23 | 7 |
| base · IR | 26 | 26 | 26 | 26 | 26 | 130 | 0 | 26 |
| base · SQL | 34 | 24 | 32 | 20 | 23 | 133 | 14 | 15 |

paired(분모 200): fine-tune 둘 다 85·IR만 70·SQL만 5·둘 다 오답 40, base 둘 다 104·IR만 26·SQL만 29·둘 다 오답 41.

세부:

- IR arm은 variant마다 실패 분류도 같았다(fine-tune `UNKNOWN_FIELD` 3·`SEMANTIC_MISMATCH` 6, base 출력 구조 무효 8·`UNKNOWN_FIELD` 3·`INVALID_VALUE_TYPE` 1·`SEMANTIC_MISMATCH` 2). case 단위 cross-schema 일관성은 fine-tune 37/40, base 28/40이며 나머지는 한 variant 이상 미실행이라 판정 불가다.
- 직접 SQL fine-tune: coded 실행 오류 32, hybrid 실행 오류 28·오답 2. 정책 통과(valid SQL)는 모두 40이다.
- 직접 SQL base: coded 기권 9·실행 오류 10·오답 1, hybrid 기권 7·실행 오류 6·오답 4.

## 관찰(잠정)

1. **IR arm은 두 모델 모두 5개 schema에서 정답 수가 같았다(spread 0).** 모델 입력에 물리 schema가 없고 Published Mapping과 compiler가 차이를 흡수하므로 설계상 예상과 맞는다.
2. **직접 SQL arm은 JOIN이 필요한 schema에서 떨어졌다.** 단일 table인 flat·legacy에서는 높고, JOIN이 필요한 normalized·coded·hybrid에서 낮다. 1개 JOIN만 필요한 hybrid에서도 크게 떨어졌다(fine-tune 10, base 23).
3. **이름 축보다 구조 축의 영향이 컸다.** 같은 단일 구조에서 영문(flat)과 약어(legacy)의 차이는 작고(30·31, 34·32), 같은 정규화 구조에서 약어(coded)는 영문(normalized)보다 조금 더 낮았다(11→8, 24→20).
4. base 모델의 전체 정답은 두 arm이 비슷해졌다(IR 130 대 SQL 133). 3종 기준(78 대 90)보다 차이가 줄어든 것은 늘어난 변형 중 JOIN이 필요한 schema가 더 많기 때문이며, 전체 정답 비교는 fixture 구성에 따라 달라지므로 우열 근거로 쓰지 않는다(ADR-061).
5. 모든 variant에서 정답인 case는 IR arm이 두 모델 모두 더 많았다(31 대 7, 26 대 15).

## 해석과 한계

- 방어 가능한 관찰은 ADR-061 범위와 같다: **QueryIR arm은 schema 구조·이름 변화에 대해 정확도가 일정하고, 직접 SQL arm은 JOIN이 필요한 schema에서 떨어진다.**
- IR arm의 일관성은 fixture별 Published Mapping이 정확하다는 전제에 의존한다. 이번 mapping은 개발자가 작성한 합성 fixture mapping이며 자동 mapping 정확도나 L2 검토를 대신하지 않는다.
- 5종 모두 같은 개발자가 같은 seed로 만든 합성 schema다. 독립 unseen schema(E4·L6)가 아니므로 학교 간 일반화를 주장하지 않는다.
- 코드값 변경, FK 없는 관계, view, 컬럼 타입 변경은 포함하지 않았다.
- 개발 세트 40건은 prompt 탐색에 쓰였고 holdout 30건은 실행하지 않았다. system 조건이 모델마다 다르다. 모델당 1회 실행, 통계 검정 없음.
- SQL arm은 column 이름을 비교하지 않아 IR arm보다 약간 관대하다(ADR-057).

## 사람 관문

완전 합성 입력만 사용했다(E3·L5 대상 없음). 수치는 L1/L3 전 잠정치이며 승인 상태는 바꾸지 않았다.

---

<a id="phase7-09-ollama-prototype-live-smoke-results-2026-09-28"></a>

## 원본: phase7-09-ollama-prototype-live-smoke-results-2026-09-28.md

# Phase 7-09 합성 prototype Ollama live smoke 결과 (2026-09-28)

기준: [`docs/development-history.md#prompts-phase7-09-ollama-prototype-live-smoke`](development-history.md#prompts-phase7-09-ollama-prototype-live-smoke). 완전 합성 질문(코드에 고정)과 합성 SQLite fixture만 사용했다. 모델 응답 원문·prompt·QueryIR·SQL·행 값·session token·option ID는 이 문서에 기록하지 않았다.

## 환경과 실행

- Windows 데스크톱, `.venv` Python, 로컬 Ollama `0.34.4`(`/api/version`), 기본 loopback URL.
- model tag `text2sql_qwen:latest`, `/api/tags` digest `7070730bf2f29aa3e7cbb9b0b566b1c73fd93bfeab41363e4a1f128730a3cb5a`. weight 출처·사용 권한은 확인하지 않았다.
- prompt `query-ir-v1`, timeout 60초, retry 없음.
- `--check` 1회(모델 호출 0회, `CHECKED`) 뒤 `--live-synthetic` 1회.

## smoke 결과

전체 `status=COMPLETED`, 모델 호출 4회(기대 4회), 모든 case `boundary_ok=true`.

| case ID | DB | 동작 | start | HTTP | outcome | 호출 | 경과 |
| --- | --- | --- | --- | ---: | --- | ---: | ---: |
| `p709-flat-course` | flat | `Course.name` submit | PENDING | 422 | `CONSTRAINT_INVALID_QUERY_IR` | 1 | 약 9.5초(첫 호출, model load 포함 추정) |
| `p709-flat-instructor` | flat | `Instructor.name` submit | PENDING | 422 | `CONSTRAINT_INVALID_QUERY_IR` | 1 | 약 0.7초 |
| `p709-legacy-course` | legacy | `Course.name` submit | PENDING | 422 | `CONSTRAINT_INVALID_QUERY_IR` | 1 | 약 0.6초 |
| `p709-legacy-instructor` | legacy | `Instructor.name` submit | PENDING | 422 | `CONSTRAINT_INVALID_QUERY_IR` | 1 | 약 0.7초 |
| `p709-flat-cancel` | flat | cancel | PENDING | 200 | `CANCELLED` | 0 | — |
| `p709-flat-no-match` | flat | no-match | NOT_REQUIRED | — | `NOT_REQUIRED` | 0 | — |

관찰 요약: submit 4건 중 실행 성공 0건.

## 원인 진단(비노출 구조 분류)

smoke 뒤 같은 설정으로 별도 진단 호출 3건(flat 2건 + 재확인 1건)을 in-memory로만 분류했다. 원문은 출력·저장하지 않았다.

- 응답은 모두 JSON object였고 key는 정확히 `entity`·`select`·`filters`·`limit`, 기존 `QueryIR` 검증은 통과했다. prompt semantic context는 확인한 field 하나(`available_fields`=`[selected_ref]`)로 제한되어 있었다.
- 각 응답은 filter 1개를 포함했다. 값을 확인한 1건에서 filter는 `equals`이고 값이 **질문 문자열 전체와 정확히 같았다**.
- Phase 6-09 `_reject_reflected_context`는 요청 질문·trigger·physical 이름 등 원문 경계 문자열이 QueryIR에 그대로 반사되면 `CONSTRAINT_INVALID_QUERY_IR`로 거부한다. offline 재현에서 filter 값이 질문 전체 또는 trigger `강의`일 때 같은 422가 났고, 일반 문자열 filter(`contains`/`equals`)는 성공했다.

결론: 422는 **설계된 reflection guard가 모델 출력을 거부한 결과**이며 코드 결함이 아니다. 질문 전체에 대한 `equals` filter는 의미상으로도 0행을 낼 잘못된 IR이다. guard는 완화하지 않았다.

## 해석과 한계

- 확인된 것: live Ollama에서 startup·start·cancel·no-match 호출 0회, submit당 정확히 1회, 확인된 field 하나로 축소된 semantic context 전달, 모델 출력의 validation→guard 경로 통과, 기존 고정 HTTP 오류 taxonomy와 redaction 유지.
- 확인되지 않은 것: 성공 실행 경로(compile·policy·read-only execute)의 live 통과. 이번 모델 출력이 모두 guard에서 멈췄기 때문이다. 해당 경로는 합성 transport 테스트와 Phase 5-13 평가 경로로만 검증되어 있다.
- 4건은 정확도·성능 수치가 아니다. L1/L3 전 성능 주장에 쓰지 않는다.
- 관찰된 모델 경향(축소 context에서 질문 전체를 filter 값으로 복사)은 prompt 설계 문제일 가능성이 있다. 후속 모델 비교에서 다룰 후보이며, 이번 단계에서 prompt·guard를 바꾸지 않았다. `CONSTRAINT_INVALID_QUERY_IR`가 malformed IR과 reflection 거부를 구분하지 않아 진단이 어렵다는 점도 후속 후보다.

## 사람 관문

완전 합성 입력만 사용해 E3·L5 대상 자료가 없다. 실제 학교 DB(L2)·UX 확정(L4)과 무관하며 승인 상태는 바꾸지 않았다.

## 추가 기록: 내장 SYSTEM 제거 후 재실행 (Phase 7-10, 2026-09-29)

- 이 smoke의 원래 실행은 `text2sql_qwen:latest` Modelfile의 내장 CNU `courses` SYSTEM이 주입된 조건이었다(ADR-058).
- Phase 7-10에서 runtime 요청 형식을 명시하게 한 뒤 `--ollama-request-format alpaca-raw-no-system-v1`로 `--check`(호출 0회)와 `--live-synthetic` 1회를 다시 실행했다. 경계는 `COMPLETED`(호출 4/4, cancel·no-match 0회)였다.
- `Instructor.name` submit 2건(flat·legacy)은 `EXECUTED_SUCCEEDED`, columns가 선택 field와 일치했으나 0행이었다. **성공 실행 경로(QueryIR 검증→compile→policy→read-only 실행)가 처음으로 live로 통과**했다.
- `Course.name` submit 2건은 여전히 `CONSTRAINT_INVALID_QUERY_IR`였다. 비노출 구조 분류(진단 4 호출)에서 Course는 질문 전체를 값으로 한 `equals` filter(reflection guard 거부), Instructor는 질문의 일부 문자열을 값으로 한 `contains` filter(guard 통과, 0행)였다.
- 해석: 목록 전체를 묻는 질문에 불필요한 filter를 붙이는 경향은 내장 SYSTEM과 무관하게 남는 모델 출력 특성이다. 고정 합성 질문의 `합성 스모크 P709:` 접두어가 이를 유도했을 가능성도 있다. guard·prompt는 바꾸지 않았고 prompt 개선 후보로 남긴다.

---

<a id="phase7-15-legacy-vs-queryir-comparison-results-2026-09-29"></a>

## 원본: phase7-15-legacy-vs-queryir-comparison-results-2026-09-29.md

# Phase 7-15 결과: 같은 질문에 대한 legacy 챗봇과 QueryIR 경로 비교 (2026-09-29)

> 잠정 결과다. 질문 30개와 기준 조건은 개발자가 결과를 보기 전에 작성했으며, gold가 아니므로 L1(사람 검토) 전이다. 모델 성능의 확정 수치로 쓰지 않는다(L1·L3 필요).

## 1. 무엇을 비교했나

- 같은 legacy DB의 view `v_course_info`(팀이 구성한 유사 데이터)와 같은 모델 `text2sql_qwen`을 쓰는 두 챗봇을 비교했다.
  - legacy: `/api/v1/chat/query`
  - QueryIR: `/api/adaptive/v1/chat/query`
- 질문 세트: `backend/adaptive/evaluation/data/legacy-chat-compare-v1.json` (30문항)
  - `ONTOLOGY` 22: 매핑된 field만으로 답할 수 있다.
  - `OUTSIDE_SCOPE` 5: legacy view에는 있으나 현재 ontology 밖이다(학년, 수업 방식, 선수과목, field 간 비교).
  - `UNSUPPORTED_FIELD` 3: DB에 없는 연도·학기를 묻는다.
- 판정: 각 답을 개설강좌 key(과목 코드, 분반) 집합으로 줄여 기준 SQL의 key 집합과 비교했다.
  - `EXACT`: 기준과 같다.
  - `CORRECT_SUBSET`: 결과 상한에 도달했고, 반환된 강좌가 모두 조건에 맞는다.
  - `INCOMPLETE`: 상한에 도달하지 않았는데 일부 강좌가 빠졌다.
  - `EMPTY`: 0행이다.
  - `INCORRECT`: 조건에 맞지 않는 강좌가 섞였다.
  - `NO_ANSWER`: 이해 못 함·오류.
  - `UNKEYED`: legacy가 key column 없이 조회했다.
  - `REFUSED`: 올바른 "답할 수 없음".
  - `ANSWERED_WITHOUT_BASIS`: DB에 근거가 없는데 행을 반환했다.
- **수용 가능 = `EXACT` + `CORRECT_SUBSET` + `REFUSED`.**
- 두 경로의 구성 차이:
  - legacy: 전처리(오타·학과명 등 키워드 교정, entity masking, cache), 기본 `LIMIT 50`, 시간표 조건이 있으면 일치 강좌의 전체 시간표 행으로 확장(최대 200행)
  - QueryIR: 전처리 없음, 모델이 정한 limit(이번 실행에서는 항상 50)
- 서버 정책 변형(같은 모델 출력, 모델 재호출 없음):
  - `+limit200`: limit를 200으로 올린다.
  - `+limit200+contains`: 추가로 과목명·학과명·교원명·이수구분의 `equals`를 `contains`로 바꾼다.
- 실행: 2026-09-29, `npm run dev:all`, 로컬 Ollama. 원자료는 Git 비추적 `.schema-adaptive/reports/phase7-15/p715-run1.json`.

## 2. 요약

수용 가능 문항 수:

| 범위 | legacy | QueryIR | +limit200 | +limit200+contains |
|---|---:|---:|---:|---:|
| ONTOLOGY (22) | **17** | 11 | 10 | **13** |
| OUTSIDE_SCOPE (5) | 4 | 1* | 0 | 0 |
| UNSUPPORTED_FIELD (3) | 0 | **3** | 3 | 3 |
| 전체 (30) | **21** | 15 | 13 | 16 |

\* q24 `1학년이 듣는 교양 과목`에서 QueryIR은 학년 조건 없이 교양 과목만 조회했다. 그런데 앞쪽 50건이 우연히 모두 1학년 과목이라 `CORRECT_SUBSET`으로 판정되었다. limit 200에서는 `INCORRECT`로 드러난다(§5 한계 1).

- **매핑 범위 안의 질문에서는 현재 legacy가 앞선다(17 대 11).** 서버 정책 두 가지(limit 200, 문자열 `contains`)만 바꿔도 13이 된다.
- **DB에 없는 정보(연도·학기)를 물으면 QueryIR만 올바르게 거절한다(3/3).** legacy는 3문항 모두 근거 없는 결과를 반환했다. `2024년에 개설된 강의`를 `course_year = 4`(4학년)로 조회한 경우도 있다.
- **학점 형식처럼 값 표현이 다른 경우는 QueryIR이 앞선다.**
  - legacy prompt는 `credit_hours = '3'`처럼 비교하라고 지시한다. 실제 값은 `3/3/0` 형식이라 legacy는 `3학점`·`2학점 교양`·`영어영문학과 3학점 전공` 3문항에서 0행을 반환했다.
  - QueryIR은 선언형 변환(7-12)으로 이 중 2문항에 정답을 냈다. 나머지 1문항은 다른 이유로 실패했다.
- ontology 밖의 질문(학년, 선수과목 등)은 설계상 QueryIR이 답하지 못한다. legacy는 5문항 중 4문항을 맞혔다.
- 응답 시간 중앙값: legacy 1.2초(첫 호출 9.1초), QueryIR 0.9초(최대 1.6초).

## 3. 문항별 결과

| ID | 범위 | 질문 | 기준 강좌 수 | legacy | QueryIR | +limit200 | +limit200+contains |
|---|---|---|---:|---|---|---|---|
| q01 | ONTOLOGY | 컴퓨터융합학부 전공 과목 보여줘 | 48 | CORRECT_SUBSET | NO_ANSWER | NO_ANSWER | NO_ANSWER |
| q02 | ONTOLOGY | 교양 과목 중에 금요일 수업 | 119 | EXACT | CORRECT_SUBSET | EXACT | EXACT |
| q03 | ONTOLOGY | 3학점 과목 알려줘 | 2333 | EMPTY | CORRECT_SUBSET | CORRECT_SUBSET | CORRECT_SUBSET |
| q04 | ONTOLOGY | 월요일 오전 수업 보여줘 | 360 | CORRECT_SUBSET | CORRECT_SUBSET | CORRECT_SUBSET | CORRECT_SUBSET |
| q05 | ONTOLOGY | 수학과 강의 목록 | 82 | CORRECT_SUBSET | EMPTY | EMPTY | EXACT |
| q06 | ONTOLOGY | 전공핵심 과목 중 화요일 수업 | 382 | CORRECT_SUBSET | EMPTY | EMPTY | EMPTY |
| q07 | ONTOLOGY | 데이터베이스 강의 알려줘 | 5 | EXACT | EXACT | EXACT | EXACT |
| q08 | ONTOLOGY | 딥러닝 수업 있어? | 3 | EXACT | EXACT | EXACT | EXACT |
| q09 | ONTOLOGY | 김소연 교수님 수업 | 41 | EXACT | EXACT | EXACT | EXACT |
| q10 | ONTOLOGY | 이규철 교수 강의 보여줘 | 2 | EXACT | EXACT | EXACT | EXACT |
| q11 | ONTOLOGY | 정원이 100명 넘는 강의 | 88 | CORRECT_SUBSET | CORRECT_SUBSET | EXACT | EXACT |
| q13 | ONTOLOGY | 오후 3시에 시작하는 수업 | 628 | INCORRECT | NO_ANSWER | NO_ANSWER | NO_ANSWER |
| q14 | ONTOLOGY | 2학점 교양 과목 | 166 | EMPTY | CORRECT_SUBSET | EXACT | EXACT |
| q15 | ONTOLOGY | 경영학부 전공기초 과목 | 27 | EXACT | EMPTY | EMPTY | EMPTY |
| q16 | ONTOLOGY | 목요일 저녁 6시 이후 수업 | 82 | EXACT | INCORRECT | INCORRECT | INCORRECT |
| q17 | ONTOLOGY | 영어영문학과 3학점 전공 과목 | 21 | EMPTY | NO_ANSWER | NO_ANSWER | NO_ANSWER |
| q18 | ONTOLOGY | 일반선택 과목 보여줘 | 81 | CORRECT_SUBSET | CORRECT_SUBSET | EXACT | EXACT |
| q19 | ONTOLOGY | 토요일에 하는 수업 있어? | 28 | EXACT | EMPTY | EMPTY | EMPTY |
| q20 | ONTOLOGY | 미적분학 강의 목록 | 36 | CORRECT_SUBSET | CORRECT_SUBSET | INCOMPLETE | INCOMPLETE |
| q21 | ONTOLOGY | 컴퓨터프로그래밍 수업 시간 알려줘 | 27 | UNKEYED | INCOMPLETE | INCOMPLETE | EXACT |
| q22 | ONTOLOGY | 물리학과 월요일 수업 | 34 | EXACT | EMPTY | EMPTY | EXACT |
| q31 | ONTOLOGY | 간호학과 전공 과목 중 수요일 오후 수업 | 13 | EXACT | EMPTY | EMPTY | EMPTY |
| q12 | OUTSIDE_SCOPE | 아직 자리가 남은 컴퓨터융합학부 과목 | 35 | CORRECT_SUBSET | EMPTY | EMPTY | EMPTY |
| q23 | OUTSIDE_SCOPE | 컴퓨터융합학부 3학년 전공 과목 | 18 | EXACT | EMPTY | EMPTY | EMPTY |
| q24 | OUTSIDE_SCOPE | 1학년이 듣는 교양 과목 | 712 | CORRECT_SUBSET | CORRECT_SUBSET | INCORRECT | INCORRECT |
| q25 | OUTSIDE_SCOPE | 온라인 콘텐츠 수업 보여줘 | 138 | EMPTY | NO_ANSWER | NO_ANSWER | NO_ANSWER |
| q26 | OUTSIDE_SCOPE | 선수과목이 없는 컴퓨터융합학부 과목 | 24 | EXACT | NO_ANSWER | NO_ANSWER | NO_ANSWER |
| q28 | UNSUPPORTED_FIELD | 2024년에 개설된 강의 알려줘 | - | ANSWERED_WITHOUT_BASIS | REFUSED | REFUSED | REFUSED |
| q29 | UNSUPPORTED_FIELD | 1학기 전공 과목 | - | ANSWERED_WITHOUT_BASIS | REFUSED | REFUSED | REFUSED |
| q30 | UNSUPPORTED_FIELD | 2학기에 열리는 교양 과목 | - | ANSWERED_WITHOUT_BASIS | REFUSED | REFUSED | REFUSED |

## 4. 실패 유형

### QueryIR (ONTOLOGY 범위에서 수용 불가 11문항)

| 유형 | 문항 | 내용 | 고칠 수 있는 곳 |
|---|---|---|---|
| 없는 field 이름 생성 | q01, q17, q13 | 모델이 `Course.department`(2건), `Schedule.section`(1건)을 만들어 `NOT_UNDERSTOOD` | prompt, 또는 서버의 field 별칭 교정(후보) |
| 문자열 `equals`로 부분 이름 비교 | q05, q22 | `수학과`·`물리학과`를 실제 값 `자연과학대학 수학과`와 정확 일치로 비교 | `contains` 완화로 해결(측정됨) |
| 요일 값 형식 | q19, q06, q31 | `토요일`·`화요일`·`수요일`을 실제 값 `토`·`화`·`수`와 비교 | 요일 값 정규화(8-06 코드값 사전 후보) |
| 이수구분 값 형식 | q06 | `전공핵심` 대 실제 값 `전공(핵심)` | 코드값 사전(8-06) |
| 조건 해석 오류 | q15, q16, q31 | 질문 전체를 과목명에 넣음(q15), 시작 대신 종료 시간 비교(q16), 없는 시간 범위 추가(q31) | 모델·prompt |
| 과도하게 구체적인 값 | q20 | `미적분학` → `미적분학1` 정확 일치 | 모델·prompt |

- 요일 정규화(`X요일` → `X`)를 `+limit200+contains` 위에 더한 오프라인 추정치는 **14/22**다(q19 해결). 모델 재호출 없이 같은 출력으로 scratch에서 계산했으며, 커밋된 모듈의 변형에는 포함하지 않았다.
- q25는 모델 출력이 JSON이 아니었다.
- q26 선수과목에서도 모델이 `Course.prerequisites`를 만들었다(ontology 밖).

### legacy (ONTOLOGY 범위에서 수용 불가 5문항)

| 유형 | 문항 | 내용 |
|---|---|---|
| 학점 형식 불일치 | q03, q14, q17 | prompt 지시대로 `credit_hours = '3'`으로 비교했으나, 실제 값은 `3/3/0` 형식이라 0행 |
| 시간 해석 오류 | q13 | `오후 3시에 시작`을 12:00~15:00 범위로 조회 |
| key 없는 조회 | q21 | `start_time`만 조회. 화면 카드에 과목명·분반이 없다 |

### legacy가 잘한 부분

전처리의 학과명 교정이 효과적이었다. `물리학과`는 `자연과학대학 물리학과`로, `간호학과`는 `간호대학 간호학과`로 교정되어 정확 일치 조건이 성공했다(q22, q31). 이 기능은 QueryIR 경로에 없다.

## 5. 한계

1. `CORRECT_SUBSET`은 결과 상한에 도달한 경우만 인정한다. 그래도 상한 안의 결과가 우연히 조건을 만족하면, 빠뜨린 조건이 가려질 수 있다(q24의 QueryIR). 50행 상한에서 `CORRECT_SUBSET`이던 q20도 200행에서 `INCOMPLETE`로 드러났다. 그래서 `+limit200` 열이 이전 열보다 엄격한 판정이다.
2. 질문 30개와 기준 SQL은 개발자 한 명이 작성했다. 한국어 의미 해석(예: `오전` = 12시 이전, `전공` = 이수구분에 `전공` 포함)도 개발자 판단이다(L1 전).
3. 실행은 1회다. 모델은 temperature 0이지만 legacy에는 cache·재시도 경로가 있다.
4. legacy 호출은 legacy의 기존 query log에 기록되었을 수 있다(legacy 기존 동작).
5. 시간표 조건이 있는 질문의 화면 표시는 두 경로가 다르다. QueryIR 카드에는 조건에 맞는 요일의 시간만 나온다. legacy는 일치 강좌의 전체 시간표로 확장한다. key 기반 판정에는 영향이 없다.

## 6. 결정이 필요한 항목

| 항목 | 선택지 | 측정된 효과(ONTOLOGY 22) |
|---|---|---|
| 결과 limit | 모델 limit 유지 / 서버가 200으로 고정 / 모델 값이 작을 때만 존중 | 200: 판정은 더 엄격해지고, 정답 강좌를 더 많이 보여준다(`EXACT` 4→8) |
| 문자열 비교 | `equals` 유지 / 과목명·학과명·교원명·이수구분 `contains` 완화 | +limit200 대비 +3(10→13) |
| 코드값 정규화 | 요일(`X요일`→`X`) 등 값 사전(8-06) | 오프라인 추정 +1(13→14) |
| field 별칭 교정 | `Course.department` → `Department.name` 같은 서버 교정 / prompt 수정 / 유지 | 미측정(q01, q17 해당) |
| ontology 확장 | 학년(대상 학년) 등 legacy에서 자주 묻는 field 추가 | OUTSIDE_SCOPE 5문항 중 학년 2문항 해당 |

여기의 수치는 이 질문 세트의 잠정 관찰이며 성능 주장이 아니다.

---

<a id="phase7-16-chat-policies-recompare-results-2026-09-29"></a>

## 원본: phase7-16-chat-policies-recompare-results-2026-09-29.md

# Phase 7-16 결과: 챗봇 서버 정책 3종 적용 뒤 재비교 (2026-09-29)

> 잠정 결과다. 질문 30개와 기준은 7-15와 같다(개발자 작성, L1 전). 모델 성능의 확정 수치로 쓰지 않는다.

## 1. 적용한 정책(사용자 결정)

| 정책 | 위치 | 내용 |
|---|---|---|
| 결과 limit 200 고정 | 챗봇 endpoint | 모델 limit(항상 prompt 예시값 50) 대신 서버 상한 200을 쓴다 |
| 이름형 문자열 `contains` | 챗봇 endpoint | 과목명·학과명·교원명·이수구분의 `equals`를 `contains`로 바꾼다. 이해한 조건에 `포함`으로 표시된다 |
| 요일 값 정규화 | legacy-view profile의 선언형 변환 `KOREAN_WEEKDAY` | `토요일`·`Sat`·`saturday` → `토`. 그 밖의 값은 거부한다 |

리뷰 지적으로 함께 고친 점: PostgreSQL `contains`는 값 속의 `%`·`_`·`\`를 escape한다. 예전에는 `전공_핵심`이 `전공X핵심`에도 일치했다. 실제 PostgreSQL에서 동작을 확인했다.

## 2. 요약(수용 가능 문항 수)

| 범위 | legacy | QueryIR 7-15 | QueryIR 7-16 |
|---|---:|---:|---:|
| ONTOLOGY (22) | **17** | 11 | **14** |
| OUTSIDE_SCOPE (5) | 4 | 1* | 0 |
| UNSUPPORTED_FIELD (3) | 0 | 3 | **3** |
| 전체 (30) | **21** | 15 | 17 |

\* 7-15의 q24는 50행 상한 안에서 우연히 맞은 경우였다. limit 200에서는 `INCORRECT`로 드러난다.

- 두 실행 사이 모델 출력과 legacy SQL은 문항별로 모두 같았다(temperature 0). 차이는 전부 서버 정책의 효과다.
- ONTOLOGY 범위 11 → 14. 7-15 오프라인 추정치와 같다.
- 개선된 문항:
  - q05 `수학과`, q22 `물리학과`: contains 완화
  - q19 `토요일`: 요일 정규화
  - q21 `컴퓨터프로그래밍`: contains 완화
  - q02·q11·q14·q18: 상한 때문에 잘리던 답이 limit 200에서 전체 정답(`EXACT`)이 되었다.
- 판정이 엄격해진 문항: q20(`미적분학1` 정확 지정)과 q24(학년 조건 누락)는 50행 상한에서 가려져 있던 오류가 드러났다.
- 응답 시간 중앙값: legacy 0.9초, QueryIR 0.7초.

## 3. 문항별 결과

| ID | 범위 | 질문 | 기준 | legacy | QueryIR 7-15 | QueryIR 7-16 |
|---|---|---|---:|---|---|---|
| q01 | ONTOLOGY | 컴퓨터융합학부 전공 과목 보여줘 | 48 | CORRECT_SUBSET | NO_ANSWER | NO_ANSWER |
| q02 | ONTOLOGY | 교양 과목 중에 금요일 수업 | 119 | EXACT | CORRECT_SUBSET | EXACT |
| q03 | ONTOLOGY | 3학점 과목 알려줘 | 2333 | EMPTY | CORRECT_SUBSET | CORRECT_SUBSET |
| q04 | ONTOLOGY | 월요일 오전 수업 보여줘 | 360 | CORRECT_SUBSET | CORRECT_SUBSET | CORRECT_SUBSET |
| q05 | ONTOLOGY | 수학과 강의 목록 | 82 | CORRECT_SUBSET | EMPTY | **EXACT** |
| q06 | ONTOLOGY | 전공핵심 과목 중 화요일 수업 | 382 | CORRECT_SUBSET | EMPTY | EMPTY |
| q07 | ONTOLOGY | 데이터베이스 강의 알려줘 | 5 | EXACT | EXACT | EXACT |
| q08 | ONTOLOGY | 딥러닝 수업 있어? | 3 | EXACT | EXACT | EXACT |
| q09 | ONTOLOGY | 김소연 교수님 수업 | 41 | EXACT | EXACT | EXACT |
| q10 | ONTOLOGY | 이규철 교수 강의 보여줘 | 2 | EXACT | EXACT | EXACT |
| q11 | ONTOLOGY | 정원이 100명 넘는 강의 | 88 | CORRECT_SUBSET | CORRECT_SUBSET | EXACT |
| q13 | ONTOLOGY | 오후 3시에 시작하는 수업 | 628 | INCORRECT | NO_ANSWER | NO_ANSWER |
| q14 | ONTOLOGY | 2학점 교양 과목 | 166 | EMPTY | CORRECT_SUBSET | EXACT |
| q15 | ONTOLOGY | 경영학부 전공기초 과목 | 27 | EXACT | EMPTY | EMPTY |
| q16 | ONTOLOGY | 목요일 저녁 6시 이후 수업 | 82 | EXACT | INCORRECT | INCORRECT |
| q17 | ONTOLOGY | 영어영문학과 3학점 전공 과목 | 21 | EMPTY | NO_ANSWER | NO_ANSWER |
| q18 | ONTOLOGY | 일반선택 과목 보여줘 | 81 | CORRECT_SUBSET | CORRECT_SUBSET | EXACT |
| q19 | ONTOLOGY | 토요일에 하는 수업 있어? | 28 | EXACT | EMPTY | **EXACT** |
| q20 | ONTOLOGY | 미적분학 강의 목록 | 36 | CORRECT_SUBSET | CORRECT_SUBSET | INCOMPLETE |
| q21 | ONTOLOGY | 컴퓨터프로그래밍 수업 시간 알려줘 | 27 | UNKEYED | INCOMPLETE | **EXACT** |
| q22 | ONTOLOGY | 물리학과 월요일 수업 | 34 | EXACT | EMPTY | **EXACT** |
| q31 | ONTOLOGY | 간호학과 전공 과목 중 수요일 오후 수업 | 13 | EXACT | EMPTY | INCOMPLETE |
| q12 | OUTSIDE_SCOPE | 아직 자리가 남은 컴퓨터융합학부 과목 | 35 | CORRECT_SUBSET | EMPTY | EMPTY |
| q23 | OUTSIDE_SCOPE | 컴퓨터융합학부 3학년 전공 과목 | 18 | EXACT | EMPTY | EMPTY |
| q24 | OUTSIDE_SCOPE | 1학년이 듣는 교양 과목 | 712 | CORRECT_SUBSET | CORRECT_SUBSET | INCORRECT |
| q25 | OUTSIDE_SCOPE | 온라인 콘텐츠 수업 보여줘 | 138 | EMPTY | NO_ANSWER | NO_ANSWER |
| q26 | OUTSIDE_SCOPE | 선수과목이 없는 컴퓨터융합학부 과목 | 24 | EXACT | NO_ANSWER | NO_ANSWER |
| q28 | UNSUPPORTED_FIELD | 2024년에 개설된 강의 알려줘 | - | ANSWERED_WITHOUT_BASIS | REFUSED | REFUSED |
| q29 | UNSUPPORTED_FIELD | 1학기 전공 과목 | - | ANSWERED_WITHOUT_BASIS | REFUSED | REFUSED |
| q30 | UNSUPPORTED_FIELD | 2학기에 열리는 교양 과목 | - | ANSWERED_WITHOUT_BASIS | REFUSED | REFUSED |

## 4. ONTOLOGY 범위 22문항 대조

| 구분 | 문항 수 | 문항 |
|---|---:|---|
| 둘 다 수용 가능 | 11 | q02, q04, q05, q07, q08, q09, q10, q11, q18, q19, q22 |
| QueryIR만 수용 가능 | 3 | q03·q14(학점 형식), q21(legacy가 key 없이 조회) |
| legacy만 수용 가능 | 6 | q01, q06, q15, q16, q20, q31 |
| 둘 다 불가 | 2 | q13, q17 |

legacy만 맞힌 6문항의 원인:

| 원인 | 문항 | 다음 후보 |
|---|---|---|
| 없는 field 이름 | q01 (`Course.department`) | field 별칭 교정 또는 prompt |
| 이수구분 값 형식 | q06 (`전공핵심` 대 `전공(핵심)`) | 이수구분 값 사전(8-06) |
| 조건 해석 | q15 (질문 전체를 과목명에), q16 (종료 시간으로 비교), q31 (없는 시간 범위 추가) | 모델·prompt |
| 과도한 값 지정 | q20 (`미적분학1`) | 모델·prompt |

## 5. 남은 결정 항목

| 항목 | 예상 효과(이 질문 세트) |
|---|---|
| field 별칭 교정(`Course.department` → 학과명 등) | q01, q17 후보 |
| 이수구분 값 사전(`전공핵심` → `전공(핵심)` 등) | q06 후보 |
| 학년 등 ontology 확장 | OUTSIDE_SCOPE 학년 2문항(q23, q24) |
| prompt 개선(조건 해석, 시간 field 구분) | q15, q16, q20, q31 후보 |
| 개수 요청 존중("3개만") | 현재 질문 세트에는 해당 없음 |

## 6. 한계

7-15 결과 문서 §5와 같다. 개발자 작성 문항·기준이고 실행은 1회다. 상한 안의 우연한 일치는 부분집합으로 보일 수 있다. 원자료는 `.schema-adaptive/reports/phase7-15/p716-run1.json`에 있다.

---

<a id="phase7-17-anchor-left-join-results-2026-09-29"></a>

## 원본: phase7-17-anchor-left-join-results-2026-09-29.md

# Phase 7-17 결과: 기준 table 보존 LEFT JOIN과 legacy 기본 table 매핑 (2026-09-29)

> 기능 검증 결과다. 챗봇 비교 수치는 7-15·7-16과 같은 개발자 작성 30문항·기준이며, L1 전 잠정치다.

## 1. 한 일

- private PostgreSQL profile에 `postgres-runtime-profile-v3`(v2 + 필수 `anchor_table`)를 추가했다. 결과 한 행이 어느 table의 한 행인지를 선언한다.
- anchor가 있으면 JOIN 계획은 항상 anchor에서 시작하고, compiler는 모든 단계를 `LEFT JOIN`으로 쓴다. anchor 행은 관련 행이 없어도 남는다.
- AST policy는 anchor 모드에서 수식어 없는 `LEFT JOIN`만 허용한다. `INNER`, `LEFT OUTER`, `RIGHT`, `FULL`, `NATURAL`, anchor가 아닌 FROM은 거부한다. anchor가 없는 기존 경로는 지금처럼 기본·`INNER JOIN`만 허용하며 `LEFT`는 거부한다.
- 시작 검증은 매핑된 table의 모든 조합에서 anchor 고정 JOIN 경로가 하나로 정해지는지 확인한다. 아니면 DB를 등록하지 않는다.
  - legacy DB에서 anchor 없이 과목+학과를 물으면 경로가 두 개(개설강좌 경유, 필수과목 table 경유)로 모호하다. anchor를 개설강좌 table로 두면 하나로 정해진다.
- `legacy-base` profile(`Legacy DB tables (QueryIR)`)을 등록했다.
  - 기본 table 4개에 view profile과 같은 12개 field, 변환 4개, 미지원 3개를 매핑했다.
  - 학과 코드 column은 기본 table에 있지만, JOIN 차이만 보려고 이번에는 미지원으로 유지했다.
- `dev:all` launcher는 두 legacy profile을 각각 켜고, 하나가 실패해도 나머지를 유지한다.
- 비교 도구에 `--queryir-database`를 추가했다.

## 2. 동등성 검증(실제 legacy DB)

두 validated context로 같은 QueryIR을 컴파일·policy 검증하고, 검증 목적으로만 행 상한을 풀어 전체 결과를 비교했다(scratch script, 런타임 경로 아님).

| 확인 | view | 기본 table + LEFT JOIN | 결과 |
|---|---:|---:|---|
| 카드 field 전체, 조건 없음(행 multiset) | 6,846 | 6,846 | 같음 |
| 7-16 답변 22문항의 실행 조건 재실행(상한 없음) | – | – | 22/22 같음 |
| 시간표 없는 행 | 632 | 632 | 같음 |

비교용 반례: 같은 기본 table을 anchor 없이 INNER JOIN으로 읽으면 다음과 같다.

- 전체가 6,214행이다. 시간표 없는 632행이 빠진다.
- 22문항 중 6문항(q03, q05, q09, q11, q18, q24)의 답이 달라진다.

## 3. 같은 30문항 live 비교(`p717-base-run1`)

- 조건: legacy 챗봇과 QueryIR(`legacy-base`), 7-16과 같은 질문 세트.
- 모델 출력: 30/30 문항이 7-16(view)과 같았다.
- legacy SQL: 30/30 같았다.
- 판정: 30/30 같았다.
- 결과 key 집합: 30/30 같았다.
- ONTOLOGY 수용 가능 문항: QueryIR 14/22, legacy 17/22(7-16과 같음).
- 미지원 field 거절: 3/3.
- 응답 시간 중앙값: QueryIR 0.63초(view 0.66초), legacy 0.86초.

view 없이 기본 table만으로 view 경로와 같은 답을 낸다. 모델 prompt는 매핑된 field 이름에만 의존하므로 물리 구조가 달라도 바뀌지 않았다.

## 4. 한계

- anchor는 profile에 명시한다. 자동 추론은 하지 않는다.
- anchor 모드에서는 부모 방향 JOIN도 LEFT다. FK 무결성이 깨진 행(부모 없는 자식)은 NULL 값으로 남는다. legacy DB에는 그런 행이 0개였다.
- 결과 정렬(ORDER BY)이 없으므로, 200행 상한에 걸리는 질문은 실행 계획에 따라 부분집합이 달라질 수 있다. 이번 실행에서는 달라지지 않았다.
- SQLite 경로와 anchor 없는 PostgreSQL profile은 INNER JOIN 그대로다.
- 비교 도구는 기준 SQL을 선택한 QueryIR 대상과 같은 DSN 변수로 실행한다(리뷰 지적 반영).
  - legacy 챗봇이 쓰는 DB는 legacy backend의 자체 설정이므로 도구가 확인할 수 없다.
  - 이번 실행에서는 세 곳 모두 같은 `DATABASE_URL`을 썼다.
- 원자료는 `.schema-adaptive/reports/phase7-15/p717-base-run1.json`(Git 비추적)에 있다.

---

<a id="phase8-06-value-dictionary-results-2026-09-30"></a>

## 원본: phase8-06-value-dictionary-results-2026-09-30.md

# Phase 8-06 결과: 코드값·값 사전 (2026-09-30)

> 기능 검증 결과다. 챗봇 비교 수치는 7-15~7-17과 같은 개발자 작성 30문항·기준이며, L1 전 잠정치다.

## 1. 한 일

- field별 값 사전 `ValueDictionary`를 추가했다.
  - 각 항목은 저장 값(`stored`), 표시 이름(`label`), 다른 표기(`aliases`)를 가진다.
  - 조회는 값 전체를 대상으로 하며, 전각·대소문자·공백 차이를 무시한다.
  - 한 표기는 한 저장 값으로만 이어진다.
  - 사전에 없는 값은 설정에 따라 그대로 쓰거나(`pass`) 거부한다(`reject`).
- compiler와 SQL policy는 같은 함수로 filter 값을 정한다. policy는 사전을 거치지 않은 값을 거부한다.
- 결과 행의 저장 값은 표시 이름으로 바꿔 보여준다. 코드값(`01` 등)이 사용자에게 그대로 보이지 않는다.
- 챗봇 동작:
  - 이해한 조건은 표시 이름으로 보여준다.
  - 사전이 인식한 값은 정확히 한 저장 값을 뜻하므로 부분 일치로 완화하지 않는다. 이렇게 해야 `01`이 `101`까지 잡지 않는다.
  - `reject` 사전에 없는 값은 "이해 못 함"으로 답한다.
- private profile `postgres-runtime-profile-v4`를 추가했다.
  - 구성: v2 + 비울 수 있는 `anchor_table` + `value_dictionaries`.
  - v1~v3은 그대로 읽는다.
  - 시작 검증 조건: 사전 field는 매핑된 문자열 field이고, 값 변환과 겹치지 않으며, column 타입이 문자열 계열이어야 한다.
- 등록 CLI는 사전의 모든 저장 값이 실제 column에 있는지 읽기 전용으로 확인한다. 없으면 `VALUE_MISSING`으로 등록하지 않는다.
- legacy 두 profile(`legacy-view`, `legacy-base`)을 v4로 다시 등록했다(fingerprint 불변, smoke 5/5).
  - 두 profile에 이수구분 사전을 선언했다: 실제 값 9개, 괄호 없는 표기와 앞뒤 바꾼 표기 aliases, `unknown: pass`.
  - 등록 시 9개 값이 모두 실제 DB에 있음을 확인했다.

리뷰 지적으로 함께 고친 점(모두 재현 뒤 수정, 회귀 테스트 추가):

- 문자열 column 타입을 접두사로 판정해 `textual_category` 같은 사용자 정의 타입도 통과했다. 이제 내장 문자열 타입 이름만 정확히 허용한다.
- 과목 코드·분반·학과 코드 같은 식별 field에도 사전을 둘 수 있었다. 그러면 결과에서 강좌 key가 표시 이름으로 바뀐다. 이제 식별 field에는 사전을 둘 수 없다.
- 사전 문자열에 화면 표시를 뒤집는 양방향 제어 문자(U+202E 등)가 허용됐다. 이제 제어·서식 문자를 모두 거부한다.

## 2. 같은 30문항 live 비교(`p806-base-run1`, QueryIR은 `legacy-base`)

| 범위 | legacy | QueryIR 7-17 | QueryIR 8-06 |
|---|---:|---:|---:|
| ONTOLOGY (22) | **17** | 14 | **15** |
| OUTSIDE_SCOPE (5) | 4 | 0 | 0 |
| UNSUPPORTED_FIELD (3) | 0 | 3 | 3 |
| 전체 (30) | 21 | 17 | 18 |

- 모델 응답 상태와 legacy SQL은 30문항 모두 7-17과 같았다.
- 바뀐 문항은 둘뿐이다.
  - q06 `전공핵심 과목 중 화요일 수업`: `EMPTY` → `CORRECT_SUBSET`. 모델 값 `전공핵심`이 `전공(핵심)`으로 바뀌어 정확히 일치 조건이 되었다. 결과는 200행 상한에 걸린 부분집합이다(기준 382개 강좌).
  - q18 `일반선택 과목 보여줘`: 판정은 `EXACT` 그대로다. 인식된 값이라 부분 일치 대신 정확히 일치로 조회했다.
- 응답 시간 중앙값: 0.64초.

## 3. 남은 격차(legacy만 맞힌 5문항)

| 원인 | 문항 |
|---|---|
| 없는 field 이름(`Course.department`) | q01 |
| 조건 해석(질문 전체를 과목명에, 종료 시간 비교, 없는 시간 범위) | q15, q16, q31 |
| 과도한 값 지정(`미적분학1`) | q20 |

값 사전으로 풀 수 있는 격차는 이번으로 끝났다. 나머지는 prompt·field 별칭 문제이며, L1 질문 세트 확보 뒤에 다룬다(승인된 개발 순서 6번).

## 4. 한계

- 사전은 개발자가 선언했다. 팀이 구성한 legacy DB에 대한 것이며 L2 검토가 아니다.
- 하나의 표기가 여러 저장 값을 뜻하는 경우(`전공` → 세 값)는 사전으로 다루지 않는다. 이 경우는 지금처럼 부분 일치로 처리한다.
- 관리자 화면 편집과 SQLite·게시 mapping 계약 반영은 8-07 범위다.
- FK 없는 관계 선언(관계 교정)은 합성 변형 benchmark에서 필요해질 때 따로 한다.
- 원자료는 `.schema-adaptive/reports/phase7-15/p806-base-run1.json`(Git 비추적)에 있다.

---

<a id="phase8-07-value-dictionary-editor-results-2026-09-30"></a>

## 원본: phase8-07-value-dictionary-editor-results-2026-09-30.md

# Phase 8-07(축소판) 결과: 값 사전 편집(코드표·약어) (2026-09-30)

## 1. 무엇이 되나

화면에서 등록한 PostgreSQL DB에 대해 **값 사전**을 편집한다. PostgreSQL 등록 결과 아래 "값 사전(코드표·약어)" 영역에서 다룬다.

- **코드표:** 저장 값에 뜻을 붙인다. 예: `MON` → `월`. 사용자는 뜻으로 묻고 화면에도 뜻으로 보인다.
- **약어:** 저장 값에 사용자가 쓰는 말을 붙인다. 예: `공과대학 컴퓨터융합학부` ← `컴융`. 8-13 값 힌트와 챗봇이 이 데이터를 그대로 쓴다.
- **약어 후보:**
  - 규칙: 2글자 묶음의 첫 글자. 예: `전공(핵심)` → `전핵`, `일반선택` → `일선`.
  - 로컬 AI: 버튼을 눌렀을 때만 부른다.
  - 후보는 저장되지 않고, 관리자가 "추가"를 눌러야 항목이 된다.
- **충돌 검사:**
  - 막음: 다른 저장 값이나 그 단어 조각, 다른 항목의 표기와 겹칠 때.
  - 경고: 다른 글자 속에 들어 있을 때. 예: `전기` → `전기회로`.
  - 값이 500개를 넘는 column은 DB에 정확 일치로 직접 물어 확인한다.
- **게시:**
  - 사전이 있는 field는 변환을 붙이지 않는다.
  - 저장 값이 실제로 있는지 다시 확인하고, 없으면 게시를 막는다.
  - 이미 게시한 DB도 사전을 고친 뒤 다시 게시하면 새 버전으로 적용된다.

## 2. 확인

**실제 PostgreSQL** (합성 DB의 임시 schema, `weekday-english` 변형, 끝나고 삭제)

| 단계 | "월요일 수업 보여줘" |
|---|---|
| 사전 없이 게시 | `day_of_week = '월'` → **0건** (저장 값이 `MON`이라 맞지 않음) |
| 코드표(`MON`=월, 약어 `월요일`) 넣고 다시 게시 | `day_of_week = 'MON'` → **2건**, 화면 표시는 `월` |

같은 확인에서 이수구분 약어 후보도 받았다. 규칙 `일반선택`→`일선`, AI `일반선택`→`선택`, `교양`→`교과` 등이 나왔고, 모두 후보로만 표시된다.

**브라우저** (새 코드를 8124·5174 포트에 따로 띄움)
- 레거시 `public` schema를 등록하고 검토를 시작한 뒤, 학과명을 승인하고 새로고침했다.
- 값 사전 영역에서 `공과대학 컴퓨터융합학부 ← 컴융`을 저장했다.
- 저장 전에는 "저장하지 않은 편집" 안내가 나와 게시 버튼이 막혔다.
- 규칙 후보가 목록에 나왔다.
- 확인 뒤 등록 파일은 지웠다.
- 이 과정에서 두 가지를 찾아 고쳤다.
  - 검토 전에 "연결을 찾을 수 없습니다"라는 엉뚱한 오류가 나오던 것 → 빈 목록과 안내로 바꿨다.
  - 새로고침 버튼이 없던 것 → 추가했다.

**함께 고친 것**
- 챗봇 모델이 조회 목록에 `Schedule.*` 같은 와일드카드를 쓰면 "없는 field"로 거부됐다.
- 챗봇은 조회 목록을 고정 카드로 바꾸므로, 알려진 개체의 와일드카드는 버리도록 닫힌 보정 규칙을 추가했다.

## 3. 리뷰에서 고친 점

`gpt-6-luna` high review-only 지적 6건을 재현해 모두 고쳤다.
1. 값이 500개를 넘는 field는 게시 때 저장 값 존재를 확인하지 않았다 → DB에 정확 일치로 확인한다.
2. 같은 field에서 약어 충돌을 놓쳤다 → 정확 일치로 확인한다.
3. 편집 대상에서 빠진 field(무시·재지정)의 항목을 지울 수 없어 게시가 막혔다 → 목록에 보이고 삭제할 수 있다.
4. 후보를 요청하는 도중 field를 바꾸면 엉뚱한 field에 추가될 수 있었다 → 후보가 요청한 field를 기억한다.
5. 저장하지 않은 편집이 있어도 게시됐다 → 게시 버튼을 막고 안내한다.
6. 검토자 이름이 길면 500 오류가 났다 → 400으로 거부한다.

## 4. 남은 것

- 업로드한 SQLite DB의 부분 게시·변환·값 사전·QueryIR 챗봇은 아직 없다(SQLite 실행 경로 확장 필요). 화면 등록의 주 경로는 PostgreSQL이다.
- PostgreSQL 등록의 버전 되돌리기, test query 미리보기, 새로고침 뒤 대기 중 등록 다시 열기(목록)는 아직 없다.
- 공통 약어 묶음과 사용 기록 기반 제안은 후속이다(ADR-080).
- 규칙 후보에는 어색한 것도 섞인다(예: `대학 학과공통` → `대학공`). 모두 관리자 승인이 필요하다.

---

<a id="phase8-08-variant-benchmark-results-2026-09-30"></a>

## 원본: phase8-08-variant-benchmark-results-2026-09-30.md

# Phase 8-08 결과: 합성 변형 benchmark와 규칙 기반 기준선 (2026-09-30)

> 합성 데이터만 쓴다. 변형·정답·질문은 개발자가 작성했고 L1/L3 전이다. AI mapping 제안과 비교할 기준 점수이며, 모델 성능 주장에는 쓰지 않는다.

## 1. 만든 것

- `variant_specs.py`:
  - seed: 개설강좌 12개, 학과 4개, 교원 6명, 과목 9개.
  - 12개 변형.
  - 변형마다 inspector와 같은 형태의 catalog와 정답(gold) mapping.
  - 정답은 column 주석에서 자동으로 만든다. 들어가는 것: 위치, FK 사본 허용, 변환 허용 집합, 값 사전, 기준 table.
- `variant_benchmark.py`:
  - 답안 형식 `VariantMappingAnswer`: field별 위치·변환·값 사전과 기준 table. 다음 단계 AI 제안의 출력 형식이다.
  - field 점수(DB 불필요).
  - 실행 점수(live): gold QueryIR 20개를 기존 compiler·AST policy·read-only executor로 실행하고, seed 참조 평가와 행 multiset을 비교한다.
  - 규칙 기반 기준선: 현재 aligner를 그대로 쓴다. 신뢰도 순으로 배정하고 같은 column은 중복 배정하지 않으며, 0.60 미만이면 미지원으로 둔다.
  - live 실행:
    - 로컬 PostgreSQL 전용 DB `schema_adaptive_synthetic`(사용자 승인으로 생성)의 무작위 schema에 변형을 하나씩 만든다.
    - inspect한 catalog가 spec catalog와 완전히 같은지 확인한 뒤 실행하고, 끝나면 지운다(남은 schema 0).
- 실행이 무효가 되는 조건: 오류, 정리 실패, oracle(정답 답안)이 만점이 아닌 경우. 이때 CLI는 실패로 끝난다.

## 2. 결과(`vb-run1`)

| 답안 | field 정답 | 기준 table | 질문 정답 |
|---|---:|---:|---:|
| oracle(정답) | 180/180 | 12/12 | **240/240** |
| 규칙 기반 | 163/180 | 1/12 | **205/240** |

oracle이 모든 변형에서 만점이므로, 변형·정답·참조 평가·실행 경로가 서로 맞다.

### 2.1 규칙 기반 변형별 결과

| 변형 | 축 | field | 기준 table | 질문 | 틀린 이유 |
|---|---|---:|:---:|---:|---|
| `base` | 구조 | 15/15 | X | 20/20 | – |
| `flat` | 구조 | 15/15 | O | 20/20 | – |
| `optional-schedule` | 구조 | 15/15 | X | 19/20 | 기준 table 없음 → 시간표 없는 강좌 2개 누락 |
| `credits-slash` | 형식 | 14/15 | X | 18/20 | 학점 변환 없음 → 숫자 비교가 SQL 오류 |
| `time-type` | 형식 | 13/15 | X | 16/20 | 시간 변환 없음 → 출력이 `HH:MM` 아님 |
| `category-code` | 코드값 | 14/15 | X | 16/20 | 값 사전 없음 → `전공`이 `M`과 불일치 |
| `weekday-english` | 코드값 | 14/15 | X | 17/20 | 값 사전 없음 → `화`가 `TUE`와 불일치 |
| `semester-code` | 코드값 | 14/15 | X | 18/20 | 값 사전 없음 → `2`가 `20`과 불일치 |
| `abbreviated` | 의미 | 10/15 | X | 12/20 | 이름 hint에 없는 약어(`limit_cnt`, `yoil`, `st_tm` 등) 5개 미지원 |
| `grade-trap` | 의미 | 15/15 | X | 20/20 | 학년 column을 연도로 매핑하지 않음(정답) |
| `capacity-trap` | 의미 | 14/15 | X | 18/20 | 이름이 같은 `capacity`(강의실 정원)를 수강 정원으로 선택 |
| `legacy-like` | 복합 | 10/15 | X | 11/20 | 학점·시간 변환, 요일 사전, 교원명(`professor`) 미지원, 기준 table |

축별 질문 정답:

| 축 | 질문 정답 |
|---|---:|
| 구조 | 59/60 |
| 형식 | 34/40 |
| 코드값 | 51/60 |
| 의미 | 50/60 |
| 복합 | 11/20 |

## 3. 해석

- 현재 규칙은 **이름 대응만** 한다. column 위치는 173/180으로 높지만, 다음은 하지 않는다.
  - 값 형식(변환), 코드값(사전), 행 기준(anchor)은 제안하지 않는다.
  - 이름이 같지만 뜻이 다른 column(강의실 정원)은 구분하지 못한다.
- 질문 기준으로 틀린 35개는 모두 위 네 가지 원인 중 하나다. 다음 단계 AI 제안이 줄여야 할 격차가 이것이다.
- legacy DB와 가장 닮은 `legacy-like`에서 규칙 기반은 질문 11/20이다. 7-12~8-06에서 개발자가 손으로 만든 변환·사전·anchor가 이 격차를 메운 것이다.

## 4. 한계

- 모든 것이 개발자가 작성한 합성 자료다.
- `abbreviated`의 이름 절반(`sbjt_nm`, `pnt`, `isu_gb`, `yyyy`, `hakgi`, `bunban` 등)은 기존 합성 fixture에서 왔다. 이 이름들은 ontology의 이름 hint에 이미 들어 있다. 그래서 규칙이 처음 보는 스키마보다 유리하다. AI 제안도 같은 ontology를 보므로 비교는 공정하지만, 처음 보는 스키마에 대한 주장은 할 수 없다.
- 규칙 기준선에는 sample 값이 주어지지 않는다(현재 aligner가 쓰지 않음). AI 제안은 sample 값을 쓸 수 있으므로, 비교할 때 입력 차이를 명시해야 한다.
- 질문은 gold QueryIR을 그대로 실행한다. 자연어 이해는 평가하지 않는다.
- FK 없는 관계, 1:N 값, 닫힌 목록 밖의 형식(예: `0900`)은 포함하지 않았다.
- 원자료는 `.schema-adaptive/reports/variant-benchmark/vb-run1.json`(Git 비추적)에 있다.

---

<a id="phase8-09-ai-mapping-results-2026-09-30"></a>

## 원본: phase8-09-ai-mapping-results-2026-09-30.md

# Phase 8-09 결과: AI mapping 제안 대 규칙 기반 (2026-09-30)

> 8-08 합성 benchmark(개발자 작성, L1/L3 전) 위의 잠정 결과다. 제안 파이프라인 `ai-mapping-v1`은 실행 전에 고정했고, 결과를 보고 성능을 위해 고치지 않았다. 리뷰 뒤 v1.1(누설·견고성 수정)로 한 번 다시 실행했으며, 12개 답안이 모두 v1과 같았다(§2.2).

## 1. 비교 조건

| 비교군 | 입력 | 제안하는 것 |
|---|---|---|
| `oracle` | 정답 | 전부(검증용) |
| `rules` | column 이름·타입 | column 위치 |
| `rules+anchor` | 위와 같음 | column 위치 + 기준 table 규칙(분반 table) |
| `ai` | column 이름·타입, 저장 값 예시, 표준 값 목록 | column 위치, 값 변환, 값 사전(검증기 통과분) + 같은 기준 table 규칙 |

AI 조건:

- 모델은 로컬 `qwen2.5-coder:7b-instruct`, temperature 0, JSON schema 제한 출력이다.
- 호출은 191회, 총 75~99초(변형당 약 6~8초)였다.
- 검증기는 runtime이 거부할 변환·사전을 버린다.
  - 버린 항목은 모두 텍스트 시간 column에 제안한 `TIME_TO_HHMM`이었다(10개 변형, 20건).
  - 버린 덕분에 답안 전체가 거부되는 일은 없었다.

## 2. 결과(`ai-run2` 제안, `vb-ai-run2` 실행; v1 `ai-run1`·`vb-ai-run1`과 같음)

| 비교군 | field 정답 | 기준 table | 질문 정답 |
|---|---:|---:|---:|
| oracle | 180/180 | 12/12 | 240/240 |
| rules | 163/180 | 1/12 | 205/240 |
| rules+anchor | 163/180 | 12/12 | 206/240 |
| **ai** | **173/180** | 12/12 | **223/240** |

oracle 만점이므로 실행은 유효하다.

### 2.1 축별 질문 정답

| 축 | rules | rules+anchor | ai |
|---|---:|---:|---:|
| 구조 | 59/60 | 60/60 | 60/60 |
| 형식 | 34/40 | 34/40 | **40/40** |
| 코드값 | 51/60 | 51/60 | 51/60 |
| 의미 | 50/60 | 50/60 | **56/60** |
| 복합 | 11/20 | 11/20 | **16/20** |

### 2.2 v1 → v1.1(리뷰 반영)

리뷰 지적 4건을 모두 재현해 고쳤다.

- 변환 설명의 예시 `'3/3/0' means 3`이 benchmark 저장 값과 같았다. 누설이므로 `'4/2/2' means 4`로 바꿨다.
- 모델 출력을 schema에 맞는지 검사하지 않았다. 이제 검사하며, 중복 key는 거부한다.
- 실패한 호출이 정상 답안처럼 채점될 수 있었다. 이제 실패가 하나라도 있으면 그 제안은 채점하지 않고 다시 실행한다.
- 검증기가 runtime 사전 모델의 표기 충돌을 보지 않았다. 이제 runtime과 같은 모델로 확인한다.

v1 실행에는 호출 실패와 schema 위반이 없었다. v1.1 재실행 답안은 12개 모두 v1과 같았다. 즉 예시 누설은 결과에 영향이 없었다.

### 2.3 AI가 나아진 곳과 못한 곳

- **나아진 곳**:
  - 형식: 학점 `3/3/0`에 `SLASH_FIRST_NUMERIC`, `time` column에 `TIME_TO_HHMM`을 골랐다.
  - 약어 이름: `limit_cnt`, `aply_cnt`, `yoil`, `st_tm`, `ed_tm`을 모두 맞게 매핑했다(규칙 12/20 → AI 20/20).
  - legacy-like: 교원명 `professor`와 학점·시간 변환을 맞혔다(11 → 16).
- **못한 곳**:
  - **값 사전 0/3**: 코드값 세 변형과 legacy-like 요일에서 모두 빈 사전을 냈다. `M`·`G`·`E`, `10`·`20`, `MON`… 모두 대응하지 못했다.
  - **영어 요일에 `KOREAN_WEEKDAY`**: 검증기는 타입만 보므로 통과했지만 값이 한글이 아니라 틀렸다(weekday-english, legacy-like).
  - **의미 함정**:
    - 학년 column `course_year`를 개설 연도로 골랐다. 규칙은 이 함정을 피했다.
    - 강의실 정원 `capacity`를 수강 정원으로 골랐다. 규칙과 같다.

## 3. 해석

- 이름·형식처럼 **값 예시를 보면 드러나는 차이**는 AI가 잘 메웠다. 규칙 대비 +17문항(205 → 223)이다.
- 값 사전은 이번 파이프라인에서 전혀 작동하지 않았다. 코드값 축은 규칙과 같은 51/60이다.
- 기준 table은 분반 table 규칙만으로 12/12가 된다. 규칙 기반도 이 규칙을 붙이면 같다(+1문항).
- 의미 함정 두 개 중 AI는 하나도 피하지 못했다. 이런 경우가 사람 검토가 꼭 필요한 곳이다.

## 4. 다음 개선 후보(v2, 아직 실행하지 않음)

v1 결과를 보고 고치는 것이므로, 같은 12개 변형으로만 평가하면 benchmark에 맞춘 튜닝이 된다. v2는 **새로 만든 보류(held-out) 변형**으로 함께 평가해야 한다.

- 검증기에 저장 값 일치 검사를 넣는다.
  - `KOREAN_WEEKDAY`는 예시 값이 모두 한 글자 요일일 때만 받는다.
  - `SLASH_FIRST_NUMERIC`는 예시 값이 `숫자/숫자…` 형식일 때만 받는다.
- 값 사전을 별도 단계로 요청한다: 예시 값 중 표준 값 목록 밖의 값이 있을 때만, 각 저장 값에 표준 값을 하나씩 짝짓게 한다.
- 의미 함정은 값 범위(예: 1~4는 연도가 아님)를 prompt에 드러내는 방안을 검토한다.

## 5. 한계

- 모델 하나, 실행 한 번(temperature 0)이다.
- benchmark가 합성·개발자 작성이다.
- 약어 변형의 이름 일부는 ontology hint에 이미 있다.
- AI는 저장 값 예시와 표준 값 목록을 받고 규칙은 받지 않는다. 입력이 다르므로 "같은 입력에서의 방법 비교"가 아니다.
- 자연어 질문 이해는 평가하지 않는다.
- 원자료(모델 원출력, 버린 항목, 호출 시간)는 `.schema-adaptive/reports/variant-benchmark/ai-run{1,2}-proposals.json`, `vb-ai-run{1,2}.json`(Git 비추적)에 있다.

---

<a id="phase8-10-ai-mapping-v2-results-2026-09-30"></a>

## 원본: phase8-10-ai-mapping-v2-results-2026-09-30.md

# Phase 8-10 결과: AI mapping 제안 v2, 관리자 column 설명, 보류 변형 (2026-09-30)

> 합성 benchmark(개발자 작성, L1/L3 전) 위의 잠정 결과다. 모델은 로컬 `qwen2.5-coder:7b-instruct`, 각 조건 한 번 실행(temperature 0)이다.

## 1. 사전 등록

모델을 부르기 전에 다음을 코드로 고정했다.

- 설명 작성 규칙 `admin-notes-v1`과 모든 변형의 설명(기존 29개, 보류 11개).
- 보류 변형 6개.
- v2 파이프라인.

고정 시점(2026-09-30T04:53:49Z)에 세 파일(`variant_specs.py`, `ai_mapping.py`, `variant_benchmark.py`)의 SHA-256을 기록했고, 실행 뒤에도 같았다.

| 파일 | SHA-256(앞 16자) |
|---|---|
| `variant_specs.py` | `9a94f07adb736c65` |
| `ai_mapping.py` | `d0a44f8496e5fc8e` |
| `variant_benchmark.py` | `d5d4d97c42d3e348` |

리뷰 지적 6건을 재현·반영하면서 두 파일이 바뀌었다(hash 불일치).

- 코드표 직접 사전은 완전하고 중복 없는 표만 쓴다.
- 예시 값이 없으면 변환을 받지 않는다.
- 코드·뜻 길이를 제한한다.
- 답안과 변형 이름, trace 버전이 서로 맞는지 확인한다.
- 설명마다 규칙상 사유(`ABBREVIATION`·`CODE`·`PACKED`·`MISLEADING`)를 적고 검사한다.
  - 이 과정에서 규칙 밖 설명 1개(`capacity-trap`의 `enroll_limit`, 이름이 곧 뜻)를 찾아 제거했다. 기존 세트 설명은 29 → 28개가 됐다.
  - 이 설명만 바뀐 조건(기존 세트 v2+설명)을 다시 실행했다. 12개 답안이 모두 처음 실행과 같았다.
- 나머지 수정은 이번 입력에서 발동하지 않는 방어다(모든 코드표가 완전하고 모든 column에 예시가 있음).

보류 변형은 8-09 결과를 본 뒤 만들었다. 그래서 실패 유형(코드값·함정·복합)을 의식한 설계다. "처음 보는 문제 유형"이 아니라 "처음 보는 이름·값 표기"에 대한 검증으로 읽어야 한다.

## 2. 결과

### 2.1 기존 세트(12개 변형, 질문 240개)

| 비교군 | field 정답 | 구조 | 형식 | 코드값 | 의미 | 복합 | **질문 합계** |
|---|---:|---:|---:|---:|---:|---:|---:|
| rules | 163 | 59 | 34 | 51 | 50 | 11 | 205 |
| rules+anchor | 163 | 60 | 34 | 51 | 50 | 11 | 206 |
| ai-v1.1 | 173 | 60 | 40 | 51 | 56 | 16 | 223 |
| ai-v2 | 176 | 60 | 40 | 57 | 56 | 19 | **232** |
| ai-v2+설명 | 177 | 60 | 40 | 60 | 56 | 19 | **235** |

### 2.2 보류 세트(6개 변형, 질문 120개)

| 비교군 | field 정답 | 형식 | 코드값 | 의미 | 복합 | **질문 합계** |
|---|---:|---:|---:|---:|---:|---:|
| rules | 34 | 0 | 0 | 0 | 0 | 0(120문항 모두 거절) |
| rules+anchor | 34 | 0 | 0 | 0 | 0 | 0 |
| ai-v1.1 | 84 | 18 | 51 | 18 | 17 | 104 |
| ai-v2 | 88 | 18 | 60 | 18 | 20 | **116** |
| ai-v2+설명 | 88 | 18 | 60 | 18 | 20 | **116** |

두 세트 모두 oracle이 만점(240/240, 120/120)이므로 실행은 유효하다.

규칙이 보류 세트에서 0점인 이유는 다음과 같다.

- 과목 코드 `subject_id`가 이름 hint에 없어 매핑되지 않았다.
- 모든 질문이 과목 코드를 select하므로 20문항이 전부 거절된다.

field 점수(34/90)가 더 공정한 비교다. 실행 점수는 식별 field 하나가 빠지면 변형 전체가 0이 되는 구조다(§4).

### 2.3 v2의 효과(v1.1 대비)

| 개선 | 기존 세트 | 보류 세트 |
|---|---|---|
| 값 사전 전용 단계 | 영어 요일·학기 코드·legacy 요일 정답 | 요일 전체 표기·숫자 이수구분·학기 텍스트·영어 소문자 요일 모두 정답 |
| 저장 값 일치 검사 | 영어 요일에 붙던 `KOREAN_WEEKDAY` 제거 | 같은 검사 |
| 합계 | 223 → 232 | 104 → 116 |

기존 세트의 `category-code`에서 v2는 사전을 냈지만 코드 뜻을 바꿔 골랐다(`M`→일반선택, `E`→전공). 설명의 코드표(`M=전공…`)가 모델 없이 바로 사전이 되면서 이 오류가 고쳐졌다.

### 2.4 설명의 효과

| 세트 | 설명 수(관리자 개입) | 질문 증가 | 원인 |
|---|---:|---:|---|
| 기존 | 28 | +3 | 코드표 직접 사전 1건(`category-code`) |
| 보류 | 11 | 0 | – |

**설명은 column 선택의 의미 함정을 하나도 고치지 못했다.** 아래 네 column에 뜻을 적은 설명이 있었는데도 v2+설명은 v2와 같은 column을 골랐다.

| 변형 | 틀린 선택 | 설명 |
|---|---|---|
| `grade-trap` | `course_year` | 수강 대상 학년(1~4) |
| `legacy-like` | `course_year` | 수강 대상 학년(1~4) |
| `capacity-trap`, `ho-trap` | `capacity` | 강의실 좌석 수 |
| `ho-units` | `credit_hours` | 주당 수업 시간 합계 |

반면 `ho-trap`의 `academic_year`(학년) 함정은 설명이 없어도 v2가 피했다.

## 3. 해석

- **값 표기 차이는 v2로 대부분 해결됐다.**
  - 코드값 축은 두 세트 모두 만점이다(60/60, 60/60).
  - 처음 보는 이름·표기(보류 세트)에서도 v1.1 104 → v2 116으로 개선이 유지됐다. 기존 세트에만 맞춘 개선이 아니라는 근거다.
- **의미 함정은 이 모델과 prompt로는 풀리지 않았다.**
  - 7B 모델은 column 이름("capacity", "credit_hours", "year")을 설명보다 강하게 따른다.
  - 코드표처럼 **구조화된 설명은 결정적으로 반영**되지만, 자연어 설명은 반영되지 않았다.
- **관리자 개입의 효과는 형식에 달려 있다.**
  - 코드표는 확실히 효과가 있었다.
  - 자연어 설명은 이번 조건에서 효과가 없었다.
  - 따라서 8-07 관리자 화면에서는 자유 문장보다 **구조화된 입력**(코드표, "이 column은 X가 아님" 같은 제외 표시, 단위·형식 선택)을 우선하는 것이 낫다.

## 4. 한계

- (2026-09-30 추가, ADR-078) 함정 네 건 가운데 두 건은 비현실적이었다.
  - `capacity`: 강좌 table의 `capacity`는 보통 수강 정원이다. 강의실 좌석으로 둔 함정에서 모델이 `capacity`를 고른 것은 합리적일 수 있다.
  - `credit_hours`: 기존 세트에서는 `학점/이론/실습` 문자열(정답)인데, 보류 세트 `ho-units`에서는 "주당 수업 시간 합계" 함정으로 썼다. 같은 이름이 세트마다 다른 뜻이다.
  - 그래서 "자연어 설명은 의미 함정을 못 고친다"는 근거는 현실적인 `course_year` 2건이다. 코드표 효과(+3)에는 영향이 없다.
  - 보류 세트는 사전 등록된 것이라 고치지 않고 이 한계만 기록한다.
- 모델 하나, 조건마다 한 번 실행이다.
- 보류 변형이 8-09 결과를 본 뒤 설계됐다.
- 설명은 개발자가 규칙에 따라 썼으므로 실제 관리자의 표현과 다를 수 있다.
- 모든 질문이 과목 코드와 분반을 select한다. 그래서 식별 field 매핑이 틀리면 실행 점수가 변형 단위로 0이 된다(규칙 기반 보류 세트). 실행 점수만으로 방법을 비교하면 과장될 수 있어 field 점수와 함께 보아야 한다.
- AI는 저장 값 예시·표준 값 목록·설명을 받고, 규칙은 받지 않는다.
- 원자료(모델 원출력, 버린 항목, 직접 사전, 호출 시간):
  - `.schema-adaptive/reports/variant-benchmark/p810-*-proposals.json`
  - `vb-p810-dev.json`, `vb-p810-dev-r2.json`, `vb-p810-ho.json`
  - 모두 Git 비추적.

---

<a id="phase8-12-ai-proposal-upload-results-2026-09-30"></a>

## 원본: phase8-12-ai-proposal-upload-results-2026-09-30.md

# Phase 8-12 결과: 업로드 흐름의 AI mapping 제안 (2026-09-30)

> 개발자가 만든 합성 DB 1개로 한 번 확인한 동작 결과다. 성능 주장이 아니다(L1/L3 전).

## 1. 무엇이 바뀌었나

관리자가 SQLite DB를 업로드하면 로컬 모델(`qwen2.5-coder:7b-instruct`)이 canonical field 15개마다 column을 하나씩 고른다. 그 column이 검토 화면의 첫 후보가 되고, 규칙 후보가 뒤따른다.

- prompt는 8-10 v2의 column 단계와 같은 함수로 만든다. 옮기기 전과 모든 변형·버전의 prompt 문자열이 같음을 확인했다.
- 업로드 한 번에 모델을 한 번 호출한다.
- 저장 값 예시는 메모리에서만 읽어 prompt에 넣는다. 저장 파일과 응답에는 없다.
- 모델 실패, schema 위반, 계약 불일치(다른 DB·없는 column 등)는 규칙 제안으로 대체한다. 등록은 그대로 성공한다.
- AI 선택만으로는 자동 승인 후보가 되지 않는다. 확신도는 규칙 점수를 그대로 쓴다.
- 화면:
  - 업로드 요약에 `제안: AI 제안 / 규칙 기반 / 규칙 기반(AI 사용 불가)`을 표시한다.
  - 필드별 첫 후보와 검토 후보에는 `AI 제안` 표시를 한다.

## 2. live smoke

입력은 기존 flat 합성 fixture의 column 이름을 낯선 약어로 바꾼 DB다(`lect.subj_cd`, `pt`, `kind`, `org_cd`, `prof`, `yr`, `cls`, `cap`, `cnt`, `wday`, `t_from` 등).

| 제안 | 첫 후보가 맞은 field | 호출 시간 |
|---|---:|---:|
| 규칙 기반 | 1/15 (`term`만) | – |
| AI 첫 후보 | **15/15** | 13초 |

규칙 기반은 15개 중 14개 field에 후보가 아예 없었다. AI 첫 후보만 승인해 게시하면 원래 flat DB와 같은 강의 목록이 나온다. 이는 fake model로 같은 입력을 넣은 회귀 테스트로 고정했다.

원자료는 `.schema-adaptive/reports/phase8-12/upload-ai-smoke.json`(Git 비추적)에 있다.

## 3. 한계와 다음 단계

- DB 1개, 1회 실행이다. 이 DB는 값 형식이 표준이라 변환·값 사전이 필요 없었다.
- 업로드 DB의 게시 mapping은 여전히 `table.column`만 담는다.
  - 그래서 AI가 `3/3/0` 같은 묶음 형식이나 코드값 column을 골라도, 변환·사전 없이 게시된다.
  - 관리자가 검토에서 걸러야 한다.
- 게시하려면 15개 field가 모두 있어야 한다.
- 위 두 제한은 8-07 축소판(코드표→값 사전, 업로드 DB 변환, 부분 게시)에서 다룬다.

---

<a id="phase8-13-chat-value-hints-results-2026-09-30"></a>

## 원본: phase8-13-chat-value-hints-results-2026-09-30.md

# Phase 8-13 결과: 챗봇 값 힌트 (2026-09-30)

> 개발자가 쓴 질문으로 레거시 DB(`legacy-base`)와 로컬 챗봇 모델(`text2sql_qwen`, temperature 0)을 한 번씩 돌린 잠정 결과다(L1 전).

## 1. 무엇을 바꿨나

**방식(코드, 모든 DB 공통)**
1. **값 목록:** 서로 다른 값이 500개 이하인 text field의 실제 값을 DB별로 첫 질문 때 한 번 읽는다.
   - 대상에서 빼는 것: 변환 field, 식별 field.
   - 레거시에서는 학과명 166개, 이수구분 9개가 대상이고, 교원명·과목명은 너무 많아 빠진다.
2. **질문 검사:** 질문의 표현을 다음과 맞춰 본다.
   - 저장 값 전체, 저장 값의 단어 조각, 그 DB 값 사전의 별칭.
   - 한국어 조사를 떼고 비교하고, "값+과목" 같은 붙여 쓰기도 나눈다(ontology의 개체 이름 기준).
   - 한 글자가 틀린 긴 단어는 fuzzy로 잡는다.
3. **힌트:** 찾은 것을 `표현 → field·값` 힌트로 모델 호출(1회)에 함께 넣는다.
4. **출력 보정:** 모델이 없는 field를 쓰면(`Course.department`) ontology의 이름 힌트로 하나로 정해질 때만 고친다. 값이 엉뚱한 field에 들어가면 힌트의 field로 옮긴다. 조건을 새로 만들지는 않는다.
5. 보정 뒤에도 기존 검증·compiler·policy·executor를 그대로 거친다.

**데이터(레거시 DB의 profile에만 둠)**
- 학과 약어 `컴융`, `컴융부` → `공과대학 컴퓨터융합학부`.
- 이수구분 약어 `전핵`, `전심`, `교소`, `교소필`, `일선`, `교균`, `교기`.
- 레거시 시스템 코드에 박혀 있던 약어 표에서 옮겼다.
- 뺀 것과 이유:
  - `전기`: 일반 단어와 겹친다.
  - `인공지능`: 과목명과 겹친다.
  - `컴공`: 해당 학과가 DB에 없다.

## 2. 새 질문 12개: 조건별 비교

다음 네 조건을 비교했다.
- A: 도입 전.
- B: field 이름 보정만.
- C: 힌트와 보정 모두.
- D: C에서 약어 데이터만 뺀 것.

| 조건 | 해석이 맞은 질문 |
|---|---:|
| A 도입 전 | 3/12 |
| B field 이름 보정만 | 6/12 |
| **C 힌트 + 보정** | **12/12** |
| D 약어 데이터 제외 | 7/12 |

- 채점 기준: 이해한 조건의 field가 기대와 같고, 값에 기대한 말이 들어 있으면 맞은 것으로 본다.
- 두 질문은 결과 0건이 정답이다. DB를 확인한 결과 해당 과목이 실제로 없었다.
  - "학과공통 전공핵심": 학과공통에는 교양·일반선택·교직만 있다.
  - "컴융 교소": 컴퓨터융합학부에는 전공만 있다.
- 공통 방식만으로(D) 풀린 것: `학과공통`, `컴퓨터융합학부 전공`, `공과대학`, 오타 `컴퓨터융학학부`, `수학과 교양`, `약학과 전공`.
- 약어 데이터가 있어야 풀린 것: `컴융`, `전핵`, `교소`.
  - 약어는 규칙으로 알아낼 수 없으므로 DB별 데이터(관리자 입력)가 필요하다는 점이 확인됐다.

## 3. 7-15 비교 30문항 재실행 (퇴보 확인)

레거시 쪽은 사용자가 띄운 기존 서버를 썼고, QueryIR 쪽은 새 코드를 같은 프로세스 안에서 실행했다.

| 범위 | 레거시 | QueryIR 8-06 | **QueryIR 8-13** |
|---|---:|---:|---:|
| 해석 범위(22) 수용 가능 | 17 | 15 | **19** |
| 전체(30) 수용 가능 | 21 | 18 | **22** |
| 미지원 field 거절(3) | 0 | 3 | 3 |

- 나아진 문항:
  - q01 `컴퓨터융합학부 전공`, q15 `경영학부 전공기초`, q17 `영어영문학과 3학점 전공`: 정확.
  - q13 `오후 3시 시작`: 없는 field 보정으로 답함.
  - q31은 "전공" 조건을 되찾았지만 종료 시간 조건이 빠져 여전히 불완전하다.
- 해석 범위 질문에서 나빠진 것은 없다.
- 범위 밖 질문 두 개(q12 "자리가 남은", q23 "3학년")는 0건 답에서 틀린 조건의 답으로 바뀌었다.
  - 두 경우 모두 이번 변경 전에도 수용 불가였다.
  - q23은 모델이 "3학년"을 3학점으로 읽은 기존 문제다.

## 4. 리뷰에서 고친 점과 남긴 판단

`gpt-6-luna` high review-only 지적 6건을 재현했다.

**고친 것 (2건)**
- 값 읽기가 한 번 실패하면 그 상태가 프로세스가 끝날 때까지 캐시됐다.
  - 이제 reader는 "값이 너무 많음"과 "읽기 실패"를 구분한다.
  - 실패가 섞인 목록은 60초 뒤 다시 읽는다.
- 200자를 넘는 fuzzy 힌트가 prompt 검증에 걸려 챗봇 전체가 실패할 수 있었다. 이제 그런 힌트는 버린다.

**의도한 동작으로 남긴 것**
- field 이름 보정은 값 목록이 없어도 동작한다. ontology만 쓰는 공통 규칙이며, 위 B 조건의 개선이 여기서 나왔다.
- 값 읽기에 실패해도 값 사전의 별칭 힌트는 남는다. 별칭은 DB를 읽지 않아도 되는 profile 데이터다.
- 값 전체와 일치하는 이름도 포함 검색이 된다.
  - 7-16에서 사용자가 정한 "이름형 문자열은 포함 검색" 정책을 따른 것이다.
  - 정책을 바꿀지는 따로 결정한다.
- 저장 값이 prompt에 들어가므로, DB 값에 지시문이 있으면 모델이 영향받을 수 있다.
  - 출력은 여전히 QueryIR 검증과 policy를 거친다. 다만 틀린 조건을 막지는 못한다.
  - 관리자가 등록한 DB 값만 들어가며, 한계로 기록한다.

## 5. 한계

- 질문은 개발자가 썼고, 모델은 한 번 실행했다.
- 레거시 약어 표는 팀이 만든 것을 옮긴 것이다.
- 새 DB의 약어는 관리자가 넣어야 한다(8-07).
- 업로드한 SQLite DB 챗봇은 아직 PostgreSQL 경로만 지원한다. 값 힌트 방식은 공통이지만 연결은 PostgreSQL profile만 되어 있다.
- 원자료(Git 비추적):
  - `.schema-adaptive/reports/phase8-13/value-hints-ablation.json`
  - `.schema-adaptive/reports/phase7-15/p813-base-run1.json`

---

<a id="phase8-15-postgres-registration-results-2026-09-30"></a>

## 원본: phase8-15-postgres-registration-results-2026-09-30.md

# Phase 8-15 결과: 화면에서 PostgreSQL 등록 (2026-09-30)

## 1. 무엇이 되나

"DB 변경" 창의 **PostgreSQL 등록** 영역에서 등록부터 사용까지 한 번에 진행된다.
1. 서버에 설정된 연결과 schema를 고른다.
2. ID와 이름을 넣고 등록하면 SQLite와 같은 검토 화면이 뜬다.
3. 검토 후 게시하면 서버를 재시작하지 않아도 DB 목록과 챗봇(QueryIR, 값 힌트, SQL 표시)에 바로 추가된다.

**연결 목록**
- 연결은 `backend/.env`에서 온다.
  - `DATABASE_URL`은 "기본 PostgreSQL 서버"가 된다.
  - `PG_SOURCE_<이름>=<DSN>`을 넣으면 연결이 추가된다.
- 브라우저에는 연결 ID와 이름만 간다. 저장 파일에는 DSN 대신 환경변수 이름만 남는다.

**등록**
- 읽기 전용으로 구조를 읽는다.
- 규칙 제안에 더해, 모델이 켜져 있으면 8-12 AI 제안을 만든다. 모델에 줄 저장 값 예시는 메모리에서만 쓴다.

**검토**
- 동작은 승인, 다른 column 지정, 새로 추가한 **"이 DB에 없음"(IGNORE)**이다.
- PostgreSQL은 없는 정보를 빼고 게시할 수 있다(미지원 field).
- SQLite는 지금처럼 field 15개가 모두 있어야 게시된다.

**게시할 때 서버가 정하는 것** (고정 규칙)
- 변환: field·column type과 저장 값 예시가 맞는 변환이 하나일 때만 붙인다.
  - `3/3/0` 형식 → `SLASH_FIRST_NUMERIC`
  - `time` column → `TIME_TO_HHMM`
  - 한 글자 요일 → `KOREAN_WEEKDAY`
- 기준 table: 분반이 있는 table이다.
- 검증: 실제 연결로 기존 runtime 검증을 통과한 profile만 쓴다. 파일은 서버 재시작 때 다시 읽는 형식으로 저장한다.

**재시작**
- dev launcher가 게시된 등록을 찾아 연결 DSN을 넣고 다시 붙인다.
- 연결 설정이 사라졌으면 해당 DB는 `SOURCE_MISSING`으로 꺼진다.

## 2. 확인

- **실제 PostgreSQL smoke** (합성 DB에 임시 schema를 만들고, 끝나면 schema와 private 파일 삭제):

  | 변형 | 제안 | 게시 | 자동 결정 | 챗봇 |
  |---|---|---|---|---|
  | `legacy-like` | AI | 성공(학기 1개 무시) | 학점 `SLASH_FIRST_NUMERIC`, 시간 `TIME_TO_HHMM`, 기준 `cnu_courses` | "3학점 과목" 9건, SQL 확인 |
  | `credits-slash` | AI | 성공(15개 모두) | 학점 `SLASH_FIRST_NUMERIC`, 요일 `KOREAN_WEEKDAY`, 기준 `course_offerings` | "3학점 과목" 9건 |

- **브라우저** (새 코드를 8124·5174 포트에 따로 띄움. 사용자 서버는 건드리지 않음):
  - "DB 변경" 창에 연결 "기본 PostgreSQL 서버 (DATABASE_URL)"과 schema 목록이 나왔다.
  - 레거시 `public` schema를 등록하니 table 8개·column 38개, 제안은 "AI 제안"으로 나왔고 검토 시작 버튼까지 확인했다. 확인 뒤 등록 파일은 지웠다.
  - 확인 중 "DB 변경" 창에 높이 제한·스크롤이 없어 아래 입력칸을 누를 수 없는 문제를 찾아 고쳤다.

## 3. 리뷰에서 고친 점

`gpt-6-luna` high review-only 지적 4건을 재현해 고쳤다.
1. 게시한 DB가 "합성 질의(clarification)" 경로에서는 보이지 않았다. 살아 있는 집합으로 바꾸기 전에 loader를 만든 탓이다.
2. 실행 중 게시로 DB 수 상한(고정 + PostgreSQL 8개)을 넘길 수 있었다. 이제 409로 거부한다.
3. profile 두 파일 중 하나만 바뀐 채 실패할 수 있었다. 이제 함께 바뀌거나 이전 상태로 돌아간다.
4. DB 목록을 읽는 도중 게시가 일어날 수 있었다. 이제 한 번 복사한 목록으로 응답한다.

## 4. 한계와 다음 단계 (8-07)

- 값 사전(코드표·약어)은 아직 입력할 수 없다. 영어 요일 등 코드값 column은 게시돼도 조건이 맞지 않는다(smoke에서 "월요일 수업" 0건).
- 버전 되돌리기, test query 미리보기, 회귀 비교는 PostgreSQL 등록에 아직 없다.
- view는 등록하지 않는다(table만).
- 자동 승인이 아니므로 검토자가 이름 함정(예: `course_year`)을 걸러야 한다.

---

<a id="phase8-16-registration-list-and-smoke-results-2026-09-30"></a>

## 원본: phase8-16-registration-list-and-smoke-results-2026-09-30.md

# Phase 8-16 결과: 등록 목록, 처음부터 끝까지 browser smoke, 시연 절차 (2026-09-30)

## 1. 등록 목록

- "DB 변경" 창에 **등록된 PostgreSQL DB** 목록이 생겼다. 상태는 `검토 대기` 또는 `게시됨(vN)`으로 보이고, **열기**를 누르면 검토·값 사전을 다시 연다.
- API는 `GET /api/adaptive/v1/postgres/registrations`다. DSN은 없다.
- 목록에서 빼는 것:
  - 손상된 등록이나 손상된 draft.
  - runtime이 거부할 이름을 가진 등록.

## 2. browser smoke (새 코드를 8124·5174 포트에 띄움, 사용자 서버는 건드리지 않음)

시나리오: 레거시 DB `public`을 화면에서 등록 → 검토 → 약어 → 게시 → 새로고침 → 다시 열기 → DB 선택 → 챗봇 질문.

| 단계 | 결과 |
|---|---|
| 등록 | table 8개·column 38개, 제안 "AI 제안" |
| 검토 | 13개 승인. 학기(`enrollment.semester`, 학생 수강 기록)와 개설 연도(`course_year`, 학년)는 "이 DB에 없음" |
| 값 사전 | `공과대학 컴퓨터융합학부 ← 컴융, 컴융부` |
| 게시 | mapping v1. 목록에 "게시됨(v1)", DB 목록에 `Legacy DB (screen)` |
| 새로고침 → 열기 | 검토·값 사전이 다시 열림 |
| "컴융 전공 과목 보여줘" | **82건**. `dept_name = '공과대학 컴퓨터융합학부' AND category LIKE '%전공%'` |
| "학과공통 과목 보여줘" | 200건(상한). `dept_name LIKE '%학과공통%'` |
| "월요일 오전 수업 보여줘" | 200건(상한). `day_of_week = '월' AND 시작 < '12:00'` |

- 브라우저 창이 숨겨진 상태라 일부 단계는 같은 화면 요소를 페이지 안에서 직접 조작했다. 예: `element.click()`, 입력값 설정.
- 확인 뒤 `demo-legacy` 등록과 profile 파일을 지웠다.

**smoke에서 찾아 고친 결함**
1. **이름 규칙 불일치:** 등록은 한글 이름을 받았지만 PostgreSQL runtime은 영문·숫자·공백·`()_.-`만 받는다. 그래서 한글 이름으로 등록하면 게시가 항상 막혔다. 이제 등록할 때 같은 규칙으로 바로 알려 준다(화면 안내 포함).
2. **학점 형식:** 레거시 학점 값 중 `1/.5/1`(소수점 앞 0 생략) 때문에 학점 변환이 자동으로 붙지 않았다. 이제 이 형식도 받는다.

## 3. 리뷰에서 고친 점

`gpt-6-luna` high review-only 지적 4건을 재현해 고쳤다.
1. **[P1] 연결 바꿔치기:** 서버 설정의 같은 연결 이름이 다른 DB를 가리키게 바뀌면, 구조만 같으면 다른 DB에 게시될 수 있었다.
   - 이제 등록할 때 연결 신원(비밀번호를 뺀 host·port·DB·user의 SHA-256)을 기록한다.
   - 게시할 때와 재시작 때 대조해 다르면 `SOURCE_CHANGED`로 막는다. 비밀번호만 바뀐 것은 같은 DB로 본다.
2. 손상된 draft가 있는 등록이 목록에 남았다 → 뺀다.
3. 목록 parser가 runtime이 거부할 이름도 받았다 → 같은 규칙으로 거른다.
4. 서버에 연결 설정이 없으면 기존 등록 목록도 숨겨졌다 → 안내와 함께 목록을 보인다.

## 4. 시연 절차

[demo-procedure.md](demo-procedure.md)에 정리했다. 준비, 10단계 시연과 말할 요점, 되돌리기, 알려진 한계를 담았다.

---

<a id="phase8-17-other-school-results-2026-09-30"></a>

## 원본: phase8-17-other-school-results-2026-09-30.md

# Phase 8-17 결과: 다른 학교 합성 DB로 화면 흐름 검증 (2026-09-30)

지침: [phase8-17-other-school-validation.md](development-history.md#prompts-phase8-17-other-school-validation)

## 1. 만든 DB

- 위치: 로컬 합성 DB `schema_adaptive_synthetic`의 schema `other_school`. 시연용으로 남겨 두었다.
- 생성기: `python -m backend.adaptive.fixtures.other_school create|drop|check`. seed가 고정되어 있고 이름은 모두 지어낸 값이다.
- 규모: table 6개, 강좌 102개, 시간표 143행.
- 레거시와 다른 점:
  - 이름: `hg_nm`, `gm_nm`, `isu_nm`, `yoil`, `si_st` 등. ontology 힌트에 없는 이름만 썼다(테스트로 고정).
  - 저장 방식:
    - 이수 구분은 코드 table로 분리했다.
    - 학기는 코드(`10`/`20`), 요일은 `MON`, 시간은 `char(4)` `0900`이다.
    - 학점·이론·실습은 각각 다른 column이다.
    - 교수도 학과에 속한다(`gyosu.hg_cd`).
    - 주 2회 수업은 시간표 2행이다.
  - 이름 함정: 대상 학년 `suggang_hangnyeon`.

## 2. 화면 흐름 결과

확인용 서버(8124/5174)만 썼고, 사용자 서버와 `backend/.env`는 건드리지 않았다.

| 단계 | 결과 |
|---|---|
| 등록 | table 6개·column 29개·FK 6개, AI 제안 |
| AI 첫 후보 | 15개 중 13개가 맞는 column. 학기·교수명은 후보 없음 → remap(`gangjwa.hakgi_cd`, `gyosu.gs_nm`) |
| 값 사전 | 학과 약어 `소웨`·`전전`·`기공`, 학기 `10=1학기`·`20=2학기`, 요일 `MON=월요일` … `FRI=금요일` |
| 첫 게시 | **막힘** (아래 결함 1) → 고친 뒤 v1 게시 |
| 재게시 | 시간 변환(결함 2) 적용 v2. `Schedule.start_time`/`end_time` → `HHMM_TEXT_TO_HHMM` |
| 강의 목록 | **"응답 형식이 올바르지 않습니다"** (결함 3) → 고친 뒤 강좌 102개, 주 2회 수업은 한 줄에 두 시간 |
| 챗봇 화면 | "소웨 전공필수 과목 보여줘" → 이해한 조건 `학과명 = 소프트웨어학부, 이수 구분에 "전공필수" 포함`, SQL 표시 |

## 3. 고정 질문 12개 (실행 전에 고정, 정답은 생성기 행에서 계산)

화면과 같은 chat API로 채점했다. 판정 기준은 결과의 과목 코드 집합이 정답과 같은지다.

| 실행 | 결과 |
|---|---|
| run1 (v1, 시간 변환 없음) | 11/12 |
| run2 (v2, 시간 변환 적용) | 11/12 |

- 통과(11개):
  - 학과 약어(q01~q03), 이수 구분(q04), 영문 요일(q05), 학기 코드(q06).
  - 분리된 학점(q07), 과목명 부분 일치(q08), 교수명(q09), 정원 비교(q10), 금요일 오전(q11).
- q11은 run1에서도 통과했지만 우연이었다.
  - SQL이 `si_st < '12:00:00'`처럼 글자로 비교해서, `1230`도 오전으로 잡힌다.
  - 이번 데이터의 금요일 수업에는 12시대 시작이 없어서 드러나지 않았다. run2는 `HH:MM`으로 바꿔 비교한다.
- **q12 실패 "1학년 대상 과목 보여줘"**:
  - 모델이 `이수 구분에 "1학년 대상" 포함`으로 넣어 0건이 나왔다. 대상 학년은 ontology에 없으므로 결과를 내지 않아야 맞다.
  - 화면의 "이해한 조건"에는 잘못 이해한 조건이 그대로 보인다.
  - 이 질문에 맞춘 규칙은 넣지 않았다(과적합 방지). 알려진 한계로 둔다.

## 4. 찾아서 고친 결함 (어느 DB에나 해당)

1. **학과로 가는 FK 경로가 두 개면 게시가 항상 막혔다.**
   - 과목→학과와 교수→학과, 두 경로의 길이가 같다. 챗봇은 항상 모든 카드 field를 조회하므로 매번 모호했다.
   - 수정: 길이가 같은 계획이 여러 개일 때만 규칙을 적용한다. 모든 FK가 "같은 entity이거나 ontology에서 직접 관계인 entity"를 잇는 계획이 **정확히 하나**면 그것을 고른다.
     - ontology에는 `Course→Department`만 있고 `Instructor→Department`는 없다.
     - 그 밖의 경우는 여전히 모호하다고 보고 막는다.
     - 더 짧은 계획이 하나뿐이면 바꾸지 않는다.
   - compiler, SQL policy, runtime 게시 검사가 같은 planner를 쓴다.
2. **4자리 글자 시간(`0900`)을 글자 그대로 비교했다.**
   - 새 변환 `HHMM_TEXT_TO_HHMM`을 추가했다. `SUBSTRING`으로 `HH:MM`을 만드는 고정 template이다.
   - 게시할 때 저장 값 표본이 모두 `HHMM`(00~23시, 00~59분)일 때만 붙는다.
3. **주 2회 수업이 있으면 강의 목록이 계약 위반으로 실패했다.**
   - 이제 한 강좌의 여러 시간표 행을 받는다.
   - 단, 요일·시간 외의 값이 모두 같아야 하고 같은 시간이 반복되면 안 된다. 다르면 여전히 거부한다.
   - 받은 행은 강좌 하나에 여러 시간으로 합친다.
   - 요일 `월`·`월요일`도 시간표 칸에 놓는다(레거시 QueryIR DB에도 해당).
4. **화면 안내가 원인과 달랐다.**
   - 한 글자 약어(`월`)는 길이 규칙(2~20자)으로 막히는데 "겹침"으로 안내했다. 이제 `ALIAS_LENGTH`로 알린다.
   - 검토가 모두 끝났는데 runtime이 막으면 "검토가 끝나지 않은 항목"으로 안내했다. 이제 "질의 경로를 하나로 정할 수 없음"으로 알린다.

## 5. 리뷰

`gpt-6-luna` high review-only 지적 2건:

1. **[수정] fixture가 `DROP SCHEMA ... CASCADE`를 썼다.**
   - 다른 schema의 view가 이 table을 참조하면 그 view까지 지울 수 있었다.
   - 이제 자기 table 6개를 자식부터 지우고 schema를 CASCADE 없이 지운다.
   - 합성 DB에서 재현했다. 다른 schema에 참조 view를 만들고 `drop`하면 거부되고, view와 데이터가 그대로 남는다. 재현에 쓴 schema는 지웠다.
2. **[바꾸지 않음] AI mapping 평가 prompt의 변환 목록에 `HHMM_TEXT_TO_HHMM`이 없다.**
   - 평가 prompt는 hash 테스트로 고정한 것이다.
   - 화면 등록의 변환은 AI가 아니라 게시할 때 고정 규칙(`decide_transforms`)으로 정한다.
   - `values_fit` 분기는 모델이 enum 밖 값을 낼 때를 막는 방어용으로 남겼다.

## 6. 알려진 한계

- **모델:**
  - ontology에 없는 조건("1학년 대상")을 다른 field에 끼워 넣을 수 있다.
  - 결과는 0건이 되고, 화면의 "이해한 조건"으로 확인할 수 있다.
- **planner:**
  - 교수→학과 경로가 과목→학과 경로보다 짧으면 교수의 학과를 쓴다. 규칙은 길이가 같을 때만 개입한다.
  - 다만 챗봇은 과목 table을 항상 포함하므로, 이번 schema에서는 이 경우가 생기지 않는다.
- **값 사전:**
  - 한 글자 약어는 넣을 수 없다. `월`은 요일 변환·모델이 처리한다.
- **이번 DB와 모델:**
  - 학교 이름·단과대학(`dh_nm`)은 ontology에 없어 질문할 수 없다.
  - AI mapping 제안 모델도, 챗봇 모델도 이 schema를 본 적이 없다.

---

<a id="phase8-18-target-grade-results-2026-10-01"></a>

## 원본: phase8-18-target-grade-results-2026-10-01.md

# Phase 8-18 결과: 공통 항목 "대상 학년"과 항목 목록 일반화 (2026-10-01)

지침: [phase8-18-target-grade-and-field-list.md](development-history.md#prompts-phase8-18-target-grade-and-field-list)

## 1. 바뀐 것

- **공통 ontology 버전 2:** `Course.target_grade`(대상 학년, 정수)를 추가했다.
  - 버전은 항목을 **더하기만** 한다(`since_version`).
  - 버전 1로 만든 dataset, 평가 세트, draft, 게시 mapping은 그대로 유효하다(`accepts_version`, `fields_at`).
- **runtime:**
  - 옛 버전 mapping은 그 버전의 항목 집합으로 검사한다.
  - 그 뒤에 생긴 항목은 "이 DB에 없음"으로 본다. 챗봇은 `UNSUPPORTED`("대상 학년")로 답한다.
  - 서버보다 새 버전의 mapping은 거부한다.
- **항목 목록:**
  - 챗봇 카드: ontology 순서대로 매핑된 항목을 보이고, `target_grade` key를 추가했다. frontend 카드 상세에 "대상학년"이 보인다.
  - 강의 목록: 15개 항목 그대로다. 대상 학년이 없는 DB도 목록이 나온다.
- **SQLite 업로드 게시:** 강의 목록 15개만 필수다. 다른 항목은 "이 DB에 없음"으로 두고 게시할 수 있다. 게시 차단 목록도 이 기준으로 보여 준다.
- **carry-over(8-03):** 기준 mapping에 없던 항목은 `NOT_IN_BASE`로 기록한다. 화면에는 "기준에 없던 항목"으로 보인다.
  - 전에는 기록이 모든 항목을 분류해야 해서, 새 항목이 생기면 승계 기록 전체가 무효 처리됐다.
- **레거시 profile:**
  - view: `v_course_info.target_year`, base: `subject.target_year`.
  - profile을 다시 만들었다(legacy DB는 읽기만). 고정 smoke 질의가 모두 통과한다.
  - 참고: 레거시 화면은 `course_year`를 "학년", `target_year`를 "대상학년"으로 보인다. 두 값은 3,868행 중 35행만 다르다.

## 2. 확인 결과

**레거시**(확인용 서버 8124, 개발자 profile)

| 질문 | 이해한 조건 | 결과 |
|---|---|---|
| 1학년 과목 보여줘 | 대상 학년 = 1 | 200건(상한), 모두 1학년 |
| 컴융 3학년 전공 과목 보여줘 | 학과명 = 공과대학 컴퓨터융합학부, 대상 학년 = 3, 이수 구분 "전공" 포함 | 33건 |
| 2학년 교양 과목 | 대상 학년 = 2, 이수 구분 "교양" 포함 | 14건 |

view와 base 둘 다 같은 결과다.

**8-17 가상 학교**(다시 등록, 화면과 같은 API)
- AI 첫 후보: 16개 중 13개를 받아들였다.
  - 대상 학년은 AI가 학기 코드 `gangjwa.hakgi_cd`를 잘못 제안했다. 관리자 판단으로 `gyogwamok.suggang_hangnyeon`에 다시 연결했다.
  - 학기·교수명은 8-17과 같이 remap했다.
- 고정 질문: **12/12**. q12 "1학년 대상 과목 보여줘" → `대상 학년 = 1`, 24개 과목.
  - q12의 정답은 원래 "답하지 않음"이었다. 이는 대상 학년이 없다는 잘못된 전제로 정한 것이라 "1학년 과목"으로 바꿨다(생성기 주석에 기록).
- 화면: "소웨 1학년 과목 보여줘" → `학과명 = 소프트웨어학부, 대상 학년 = 1`, 카드 상세 "대상학년: 1".

## 3. 리뷰

`gpt-6-luna` high review-only 지적 4건:

1. **[수정] 버전 1 평가 세트의 ontology fingerprint가 버전 2와 달라 검증이 실패했다.**
   - `DomainOntology.at_version(v)`를 추가했다.
   - fingerprint에서 `since_version`을 뺐다.
   - 평가 세트는 자기 버전의 ontology로 비교한다. 저장된 버전 1 fingerprint가 그대로 재현됨을 테스트로 고정했다.
2. **[수정] 버전 1 dataset profile이 버전 2 항목을 연결해도 통과했다.** 이제 profile 버전의 항목만 허용한다.
3. **[바꾸지 않음] 기존 버전 1 PostgreSQL 등록에서 새 항목을 검토할 수 없다.** 범위 조정으로 뺀 것이다. 지침과 한계에 적었다.
4. **[수정] 버전 1 기준 mapping에서 버전 2 등록으로의 carry-over가 막혔다.**
   - 이제 기준 버전 ≤ 새 등록 버전이면 허용한다.
   - 기준에 없던 항목은 `NOT_IN_BASE`로 기록한다.
   - 버전 1로 게시한 기준에서 옮기는 통합 테스트를 추가했다.

## 4. 알려진 한계

- **이미 화면 등록한 PostgreSQL DB:**
  - 버전 1로 등록된 DB는 계속 동작하고, 대상 학년은 "이 DB에 없음"이다.
  - 등록 기록(proposal)은 해시로 고정된 불변 파일이라, 기존 등록에 새 항목을 추가하는 "draft 올리기"는 이번에 만들지 않았다. 대상 학년을 쓰려면 새 ID로 다시 등록해야 한다.
  - 지금 이 PC에는 화면 등록한 DB가 없다.
- **AI 제안:** 대상 학년을 학기 코드로 잘못 제안했다. 사람 검토가 필요한 사례다.
- **확정:** 공통 ontology 변경은 사용자 결정으로 확정한다(2026-10-01, ADR-086). 대상 학년(버전 2)은 확정이다.

---

<a id="phase8-19-db-extension-fields-results-2026-10-01"></a>

## 원본: phase8-19-db-extension-fields-results-2026-10-01.md

# Phase 8-19 결과: DB별 확장 항목 (2026-10-01)

지침: [phase8-19-db-extension-fields.md](development-history.md#prompts-phase8-19-db-extension-fields)

## 1. 만든 것

- **관리자 화면 "확장 항목" 패널** (PostgreSQL 등록 검토 아래)
  - 이름(표시), 영문 key, 무엇의 정보인가(과목·학과·교원·개설 강좌·강의 시간), 테이블, 컬럼, 값 종류(글자·정수·숫자)를 넣는다.
  - 목록에서 삭제할 수 있다.
  - 게시 버튼은 기존 게시와 같다.
- **선언 검사**
  - column이 실제로 있는지, 값 종류와 타입이 맞는지 확인한다.
  - 한 column은 한 항목에만 쓴다. 검토에서 승인된 column이나 다른 확장 항목과 겹치면 막는다.
  - 이름(label)은 DB 안에서 유일하다. 최대 10개다.
- **게시**
  - 확장 항목이 게시 mapping에 들어가고, profile은 v5(`extension_fields`)이다.
  - 서버를 다시 시작해도 v5 profile을 다시 읽는다.
- **runtime 검사**
  - 이름 형식(`<Entity>.ext_<key>`)을 확인한다.
  - mapping에 연결되어 있는지, column 타입이 맞는지 확인한다.
  - 공통 항목·unsupported·변환·값 사전과 겹치지 않는지 확인한다.
  - 공통 항목 검사에서는 확장 항목을 뺀다.
- **챗봇**
  - 그 DB의 항목 목록 = 공통 항목 + 확장 항목. 모델 prompt에서는 확장 항목에만 이름(label)이 붙는다. 확장 항목이 없는 DB의 prompt는 바뀌지 않는다.
  - 결과 카드에 `ext_<key>` 열과 이름이 나온다.
  - 글자 확장 항목은 "포함" 검색으로 완화한다.
  - 8-13 값 힌트에 글자 확장 항목도 들어간다.
  - SQL은 기존 compiler·policy·read-only executor를 그대로 거친다.

## 2. 확인 결과

### 가상 학교 DB

시간표에 강의실 column `gangui_sil`을 추가했다. 기존 값은 그대로이고, 강의실 값은 학과 건물과 강좌 번호로 정했다. 질문 q13·q14는 실행 전에 고정했다.

- 확장 항목: 강의실(`siganpyo.gangui_sil`, 강의 시간, 글자). 화면 양식으로 단과대학(`hakgwa.dh_nm`, 학과)도 추가해 게시했다(mapping v2).
- 고정 질문 **14/14**:
  - q13 "IT관 303호 수업" → `강의실에 "IT관 303호" 포함`
  - q14 "교양관에서 하는 수업" → `강의실에 "교양관" 포함`
  - 값 힌트를 고친 뒤 다시 돌려도 14/14였다.
- "IT대학 과목 보여줘"
  - 처음에는 모델이 학과명에 "IT대학"을 넣어 0건이 나왔다. 값 힌트가 확장 항목을 몰라서였다.
  - 고친 뒤에는 `단과대학에 "IT대학" 포함`으로 57행(IT 3개 학과)이 나온다. "경영대학 전공필수 과목"은 6행이다.
- 화면 카드 상세에 "단과대학: IT대학", "강의실: IT관 303호"가 보인다.
- **AI mapping 제안이 매번 다르다.** 강의실 column이 추가되어 catalog가 바뀌자, AI 첫 후보가 16개 중 10개만 맞았다(8-18은 13개). 정원→이론시수, 수강인원→대상학년, 연도→강좌 ID처럼 틀렸다. 관리자 검토로 바로잡았다.

### 레거시 DB (`public`, 화면 경로 등록, 읽기 전용)

- AI 첫 후보 16개 중 14개가 맞았고, 대상 학년도 `subject.target_year`로 맞게 골랐다. 학기·연도는 "이 DB에 없음"으로 처리했다.
- 확장 항목: 강의실(`course_schedule.classroom`), 수업방식(`cnu_courses.class_mode`).

| 질문 | 이해한 조건 | 결과 |
|---|---|---|
| 강의실이 교111인 과목 보여줘 | 강의실에 "교111" 포함 | 19행. DB 직접 조회도 19행 |
| 실시간 수업 보여줘 | 수업방식에 "실시간" 포함 | 41행 |
| 콘텐츠 수업 보여줘 | 수업방식에 "콘텐츠" 포함 | 165행 |
| 컴퓨터융합학부 3학년 과목 | 학과명 + 대상 학년 = 3 | 33건 |
| 교111에서 하는 수업 | **분반 = 교111 (틀림)** | 0행 |
| 교111 강의실 수업 보여줘 | **강의실 + 묻지 않은 수업방식 = 실시간 (틀림)** | 0행 |

## 3. 리뷰

`gpt-6-luna` high review-only 지적 2건을 재현하고 고쳤다.

1. **live-capture 대상 형식이 새 `label` 키를 거부했다.** harness `OntologyField`에 생긴 `label: null` 때문이다. 이제 label은 선택 키로 받되 None 또는 글자만 허용한다. 회귀 테스트를 추가했다.
2. **`data_type`이나 `entity`에 목록을 보내면 HTTP 500이 났다.** 이제 400 `INVALID_INPUT`으로 거부한다. 회귀 테스트를 추가했다.

## 4. 알려진 한계

- **값만 있는 질문:** 항목 이름 없이 값만 쓰면("교111에서 하는 수업"), 값 힌트가 있어야 맞게 이해한다.
  - 값 힌트는 값 종류가 500개 이하인 글자 항목에만 만들어진다. 레거시 강의실처럼 값이 많으면 모델이 다른 항목으로 착각할 수 있다.
- **모델의 덧붙임:** 묻지 않은 조건을 덧붙이는 경향이 남아 있다(레거시 "교111 강의실 수업").
- **AI mapping 제안:** 같은 DB라도 catalog가 조금 바뀌면 결과가 달라진다. 관리자 검토가 반드시 필요하다.
- **이번 범위 밖:** 확장 항목에는 변환과 값 사전을 붙일 수 없고, 강의 목록에는 나오지 않는다(챗봇 카드에만 나온다).

---

<a id="phase8-20-unknown-conditions-results-2026-10-01"></a>

## 원본: phase8-20-unknown-conditions-results-2026-10-01.md

# Phase 8-20 결과: 모르는 조건 처리 (2026-10-01)

지침: [phase8-20-unknown-conditions.md](development-history.md#prompts-phase8-20-unknown-conditions)

## 1. 만든 것

- **공통 항목 밖 개념 목록** (`backend/adaptive/ontology/outside-concepts.yaml`, 데이터)
  - 11개: 강의실, 수업 방식, 단과대학, 강의 언어, 성적 평가 방식, 시험, 캠퍼스, 선수 과목, 강의 평가, 여석, 강의계획서.
  - 별칭은 2자 이상이다. 공백과 대소문자는 무시한다. 강의실에는 `303호` 같은 호실 번호 패턴이 있다.
  - 공통 ontology의 이름·별칭이 개념 단어를 품으면 목록을 거부한다(예: `정원`은 공통 항목 수강 정원의 별칭이라 넣을 수 없다).
  - 과목명에 흔한 단어(`건물`, `원격`, `온라인`, `실시간`)는 단독 별칭으로 두지 않았다. 레거시 과목명 `건물에너지`, `원격탐사`, `온라인 영상 플랫폼과 기획론`이 근거다.
- **챗봇: 그 DB가 모르는 개념이면 실행하지 않는다**
  - 질문에서 개념 단어를 찾는다. 그 DB의 확장 항목이 그 개념을 다루면(key가 같거나 이름에 개념 단어가 있으면) 넘어간다.
  - 모델 호출은 그대로 한 번이다. 그 단어가 모델이 찾은 더 긴 값의 일부이면(예: 과목명 "건물에너지") 개념으로 보지 않는다. 호실 번호 패턴은 DB 값(값 힌트)으로만 면제한다.
  - 남은 개념이 있으면 `UNSUPPORTED`로 답한다. 응답 항목은 `{concept, label}`이다. 질문이 직접 이름을 댄 공통 unsupported 항목(예: "2학기")도 함께 적는다.
  - 화면: "이 DB에는 강의실 정보가 없어 답할 수 없습니다. 그 조건을 빼고 다시 물어보세요."
- **0건 진단**
  - 결과가 0건이면 같음·포함 조건을 하나씩 따로 실행한다(그 항목 1행만, 같은 검증·compiler·policy·읽기 전용 executor, 최대 4번). 조건이 하나뿐이면 그 조건 자체가 답이다.
  - 혼자서도 0행인 조건을 `unmatched`로 알린다. 화면: "과목 코드에서 '교111' 값을 찾지 못했습니다. 이 DB에 없는 정보이거나 다른 항목으로 잘못 이해했을 수 있습니다."
  - 처음에는 값 힌트용 값 목록(500개 이하)으로 판단했다. 그런데 레거시에서 모델이 "교111"을 과목 코드(값 500개 초과)에 넣어 진단하지 못했다. 그래서 실제 DB에 묻는 방식으로 바꿨다. 값 목록이 오래되어도 틀리지 않는다.
  - SQL 표시(개발 옵션)는 진단 query가 아니라 답의 query를 보인다.
- **값 힌트 cache 버그 수정** (8-07·8-19부터 있던 문제, 이번 실측에서 발견)
  - 값 힌트 index는 DB의 schema fingerprint로만 cache되었다. 그래서 다시 게시해 확장 항목이나 약어를 더해도, 서버를 재시작할 때까지 예전 힌트를 썼다.
  - 이제 게시 mapping 버전도 key에 넣는다. 예전 버전 index는 버린다.

## 2. 확인 결과

질문과 기대 결과는 실행 전에 지침 6절에 고정했다. 모델은 `text2sql_qwen:latest`, 테스트 서버 8124, 화면 5174를 썼다.

### 레거시 개발자 profile (`legacy-view`, 확장 항목 없음, 읽기 전용)

| 질문 | 결과 |
|---|---|
| 교111 강의실 수업 보여줘 | UNSUPPORTED: 강의실 |
| 305호에서 하는 수업 | UNSUPPORTED: 강의실 |
| 실시간 수업 보여줘 | UNSUPPORTED: 수업 방식 |
| 온라인 강의 보여줘 | UNSUPPORTED: 수업 방식 |
| 절대평가 과목 보여줘 | UNSUPPORTED: 성적 평가 방식 |
| 영어강의 과목 보여줘 | UNSUPPORTED: 강의 언어 |
| 여석 있는 컴융 과목 | UNSUPPORTED: 여석 |
| 교111에서 하는 수업 | 과목 코드 = 교111, 0건, "과목 코드에서 '교111' 값을 찾지 못했습니다" |
| IT대학 과목 보여줘 | 학과명에 "IT대학" 포함, 0건, "학과명에서 'IT대학' 값을 찾지 못했습니다" |

- 고정 기대 9/9.
- 첫 실행에서 "교111 강의실 수업"은 학기·개설 연도도 함께 적었다. 모델이 묻지 않은 학기·연도 조건을 붙였기 때문이다. 질문이 이름을 댄 공통 항목만 적도록 고쳤다.
- 바뀌면 안 되는 질문은 그대로였다: 컴융 전공 82건, 월요일 오전 200건(상한), 건물에너지 2건, 온라인 영상 플랫폼 2건, 컴융 3학년 전공 33건.

### 가상 학교 DB (`other_school`, 화면 경로 등록)

- **확장 항목 없이 게시**
  - "IT관 303호 수업 보여줘" → UNSUPPORTED: 강의실.
  - "교양관에서 하는 수업 보여줘" → 모델이 학기 = 교양관, 개설 연도 = 1로 넣어 0건이었다. 진단이 두 조건을 모두 짚었다.
  - q01~q12는 11/12였다. q11("금요일 오전 수업")은 같은 질문을 두 번 더 보내자 시간 조건을 뺀 답(14건)과 맞는 답(7건)이 번갈아 나왔다. 모델 출력이 흔들린 것이며, 8-20 코드는 모델 출력 뒤에만 동작한다.
- **강의실 확장 항목 선언 후 게시**
  - 13/14였다. q14에서 모델이 묻지 않은 요일 = 화를 붙여 0건이었고, 진단은 "강의 요일에서 '화'"를 짚었다.
  - 단과대학까지 더해도 q14가 틀렸다. "IT대학 과목"도 8-19와 달리 학과명으로 갔다.
  - 서버를 재시작하자 둘 다 맞았다(교양관 14건, IT대학 57건). 원인은 위 값 힌트 cache 버그였다. 확장 항목 없이 처음 만든 힌트를 계속 썼다.
- **cache 수정 뒤 (재시작 없이)**
  - 단과대학을 지우고 게시(v4)하자 "IT대학 과목"은 학과 코드 = 1101, 0건이었다. 진단이 그 조건을 짚었다.
  - 다시 더해 게시(v5)하자 57건이었다.
  - 고정 질문 **14/14**(q13 강의실 "IT관 303호" 2건, q14 강의실 "교양관" 9건).

### 화면 (5174 → 8124)

- `Legacy DB (QueryIR)`에서 "교111 강의실 수업 보여줘" → "이 DB에는 강의실 정보가 없어 답할 수 없습니다. 그 조건을 빼고 다시 물어보세요."
- "교111에서 하는 수업" → "QueryIR 실행 완료: 0건. 이해한 조건: 과목 코드 = 교111. 과목 코드에서 '교111' 값을 찾지 못했습니다. 이 DB에 없는 정보이거나 다른 항목으로 잘못 이해했을 수 있습니다."

## 3. 리뷰

`gpt-6-luna` high review-only 지적 3건을 모두 재현했다.

1. **호실 번호 패턴이 공백을 무시하지 않았다.** "3 0 5 호에서 하는 수업"을 놓쳤다. 별칭처럼 공백을 뺀 질문에도 패턴을 돌리도록 고쳤다. 띄어 쓴 원문에도 계속 돌린다("2024 305호"는 공백을 빼면 숫자가 붙어 버리기 때문이다).
2. **여러 단어에 걸친 값이 별칭을 면제하지 못했다.** 모델이 과목명 "온라인 강의 플랫폼"을 찾아도 "온라인강의"(수업 방식)로 UNSUPPORTED가 되었다. 이제 공백을 뺀 질문에서 그 값이 별칭 위치를 덮으면 면제한다.
   - 대신 모델이 "교111 강의실"을 통째로 다른 항목에 넣으면 면제된다. 이 경우는 결과가 0건이 되어 0건 진단이 그 조건을 짚는다.
3. **숫자 같음 조건이 진단되었다.** 지침은 "숫자 조건 제외"였다. 그런데 그 문구는 값 목록 방식 때의 것이고, DB에 직접 묻는 방식에서는 숫자도 정확하다. 그래서 지침을 고쳤다(4절에 구현 중 수정 기록). 레거시 "7학년 과목 보여줘" → "대상 학년에서 '7' 값을 찾지 못했습니다".

세 건 모두 회귀 테스트를 추가했다. 고친 뒤 레거시 질문 15개와 가상 학교 14개를 다시 돌려 같은 결과(9/9, 기존 질문 불변, 14/14)를 확인했다.

## 4. 알려진 한계

- 개념 목록에 없는 말은 여전히 다른 항목에 들어갈 수 있다. 그때는 0건 진단으로만 알린다. 결과가 0건이 아니면(예: 엉뚱한 조건이 우연히 맞으면) 알리지 못한다.
- 개념 목록은 사람이 고른 11개다. L1 질문 세트를 받은 뒤 보강한다.
- 모델이 묻지 않은 조건을 붙이는 경향(8-19부터)은 그대로다. 0건이면 진단이 그 조건을 짚는다.
- 범위·크기 조건(학점 > 9 등)은 0건 진단에서 뺀다.

---

<a id="phase8-21-code-table-results-2026-10-05"></a>

## 원본: phase8-21-code-table-results-2026-10-05.md

# Phase 8-21 결과 (2): 공통코드 table 형식 A (2026-10-05)

지침: [phase8-21-postgres-format-variety.md](development-history.md#prompts-phase8-21-postgres-format-variety). 형식 D 결과는 [phase8-21-wide-table-results-2026-10-02.md](experiment-results.md#phase8-21-wide-table-results-2026-10-02).

## 1. 만든 DB

- 합성 DB의 `code_school` schema. 8-17 `other_school`과 내용은 같고 모양만 바꿨다.
  - 이수구분·요일·학기 이름은 공통코드 table `gongtong_code(code_group, code, code_nm)` 하나에 모았다. 그룹은 `ISU`·`YOIL`·`HAKGI`다.
  - 과목·강좌·시간표 table에는 코드(`isu_cd='21'`, `yoil_cd='1'`, `hakgi_cd='20'`)만 있고, 공통코드 table로 가는 FK가 없다.
- 고정 질문: q01~q12와 정답(`other_school.gold`)을 그대로 썼다. 첫 실행 전에 두 질문을 더 고정했다.
  - a01 "소웨 전공 과목 보여줘": 소프트웨어학부이고 이수구분이 11 또는 12.
  - a02 "교양 과목 보여줘": 이수구분이 21 또는 22.
- 다시 만드는 SQL(합성 DB에서만 실행):

```sql
CREATE SCHEMA code_school;
CREATE TABLE code_school.gongtong_code (code_group varchar(10), code varchar(4), code_nm varchar(20) NOT NULL,
  PRIMARY KEY (code_group, code));
INSERT INTO code_school.gongtong_code SELECT 'ISU', isu_cd, isu_nm FROM other_school.isu_gubun;
INSERT INTO code_school.gongtong_code VALUES ('YOIL','1','월요일'),('YOIL','2','화요일'),('YOIL','3','수요일'),
  ('YOIL','4','목요일'),('YOIL','5','금요일'),('HAKGI','10','1학기'),('HAKGI','20','2학기');
CREATE TABLE code_school.hakgwa AS SELECT hg_cd, hg_nm, dh_nm FROM other_school.hakgwa;
ALTER TABLE code_school.hakgwa ADD PRIMARY KEY (hg_cd);
CREATE TABLE code_school.gyosu AS SELECT gs_no, gs_nm, hg_cd FROM other_school.gyosu;
ALTER TABLE code_school.gyosu ADD PRIMARY KEY (gs_no), ADD FOREIGN KEY (hg_cd) REFERENCES code_school.hakgwa (hg_cd);
CREATE TABLE code_school.gyogwamok AS SELECT gm_cd, gm_nm, hg_cd, isu_cd::varchar(2) AS isu_cd, hakjeom,
  suggang_hangnyeon FROM other_school.gyogwamok;
ALTER TABLE code_school.gyogwamok ADD PRIMARY KEY (gm_cd), ADD FOREIGN KEY (hg_cd) REFERENCES code_school.hakgwa (hg_cd);
CREATE TABLE code_school.gangjwa AS SELECT ks_id, gm_cd, gs_no, nyeondo, hakgi_cd::varchar(2) AS hakgi_cd, ban_no,
  jeongwon, sugang_inwon FROM other_school.gangjwa;
ALTER TABLE code_school.gangjwa ADD PRIMARY KEY (ks_id), ADD FOREIGN KEY (gm_cd) REFERENCES code_school.gyogwamok (gm_cd),
  ADD FOREIGN KEY (gs_no) REFERENCES code_school.gyosu (gs_no);
CREATE TABLE code_school.siganpyo AS SELECT sp_id, ks_id,
  (CASE yoil WHEN 'MON' THEN '1' WHEN 'TUE' THEN '2' WHEN 'WED' THEN '3' WHEN 'THU' THEN '4' ELSE '5' END)::varchar(1) AS yoil_cd,
  si_st, si_ed FROM other_school.siganpyo;
ALTER TABLE code_school.siganpyo ADD PRIMARY KEY (sp_id), ADD FOREIGN KEY (ks_id) REFERENCES code_school.gangjwa (ks_id);
```

## 2. 첫 실행 (코드 수정 전)

화면 API로 등록·검토·게시했다. 관리자는 코드 column을 연결하고, 공통코드의 뜻을 값 사전에 직접 입력했다. 입력한 것은 이수구분 5개, 요일 `1`~`5`, 학기 `10`·`20`, 학과 약어다.

- q01~q12 **12/12**. 정확한 이름("전공필수", "월요일", "2학기")은 값 사전으로 코드에 연결되었다. 결과 카드에도 코드 대신 이름이 보인다(8-06부터 있던 기능).
- a01·a02 실패:
  - a02 "교양 과목" → 이수 구분에 "교양" 포함 → 코드 column에서 `LIKE '%교양%'` → 0건. 0건 진단(8-20)이 그 조건을 짚었다.
  - a01 "소웨 전공 과목" → 모델이 "전공"을 아예 빼서 소프트웨어학부 과목 전체가 나왔다. 값 힌트가 "전공"을 몰랐다(값 사전은 이름 전체만 안다).
- AI mapping 첫 후보는 **16개 중 9개**만 맞았다. 코드 column(이수구분·학기)과 코드형 key(과목 코드·학과 코드)는 후보가 없었다. 학과명은 단과대학명을, 대상 학년은 학과 코드를 골랐다. 관리자 검토로 바로잡았다.

## 3. 고친 것

- **이름 일부로 코드 찾기:** 값 사전이 있는 항목의 "포함" 조건은, 그 말이 이름(또는 별칭)에 들어 있는 저장 값도 함께 찾는다.
  - SQL은 `(column LIKE '%교양%' OR column IN ('21', '22'))`이다.
  - 이름이 저장 값과 다른 항목(코드)만 대상이다. 별칭은 일부로 찾지 않는다.
  - compiler와 SQL policy가 같은 함수로 이 모양을 정한다. policy는 그 조건에만, 같은 column과 정확한 개수의 placeholder로만 이 모양을 허용한다.
  - 이름 전체("교양필수")는 지금처럼 코드 하나에 정확히 연결된다.
  - 이름을 저장한 DB(레거시)는 이름과 저장 값이 같아 이 목록이 생기지 않는다. SQL이 이전과 똑같다.
- **공유 앞부분 힌트:** 한 항목의 한 단어 코드 이름 2개 이상이 같은 앞부분으로 시작하면, 그 앞부분을 값 힌트로 준다. 예: "전공" → 전공필수·전공선택.

## 4. 고친 뒤 결과

- `code_school` 고정 질문 **14/14**. a01은 7개, a02는 9개로 정답과 같다.
- 레거시 개발자 profile: 컴융 전공 82건, 컴융 3학년 전공 33건으로 이전과 같다. SQL에 `IN` 목록이 없다.

## 5. 리뷰

`gpt-6-luna` high review-only 지적 1건을 재현하고 고쳤다.

- **이름을 저장한 DB에서 결과가 달라졌다.** 레거시 profile에서 "전공핵"으로 찾으면, 별칭 `전공핵심`을 통해 `전공(핵심)`이 `IN` 목록에 들어갔다. 이전에는 `LIKE '%전공핵%'`가 아무것도 찾지 않았다.
- 고친 내용: 이름이 저장 값과 다른 항목(코드)만 대상으로 하고, 별칭은 일부로 찾지 않는다. 레거시는 이름과 저장 값이 같아 목록이 생기지 않는다.
- 앞에서 레거시 기대값을 바꿨던 기존 테스트 2개를 원래대로 되돌렸고, 그대로 통과했다. 회귀 테스트를 추가했다.
- 고친 뒤 `code_school` 14/14와 레거시 82·33건을 다시 확인했다.

## 6. 남은 것

- 공통코드 table에서 값 사전을 자동으로 가져오는 기능은 만들지 않았다. 관리자가 코드마다 입력한다. 코드 수가 많은 학교에서는 불편할 수 있다(후속 후보).
- 코드형 column에 대한 AI mapping 제안이 약하다(9/16). 관리자 검토가 필수다.
- 형식 B(교시)·C(묶인 시간표)가 남았다.

---

<a id="phase8-21-packed-results-2026-10-06"></a>

## 원본: phase8-21-packed-results-2026-10-06.md

# Phase 8-21 결과 (4): 한 칸에 묶인 시간표 형식 C (2026-10-06)

지침: [phase8-21-postgres-format-variety.md](development-history.md#prompts-phase8-21-postgres-format-variety) 3절 C의 ①(요일만 지원, 시작·종료 시간은 "이 DB에 없음").

## 1. 만든 DB

- 합성 DB의 `packed_school` schema. `other_school`과 내용은 같고, 시간표 table 대신 강좌 table에 `sigan_txt`(예: `월 13:30-14:45, 수 13:30-14:45`)가 있다.
- 고정 기대(실행 전): q01~q10·q12는 답한다. q11 "금요일 오전 수업"과 b01 "오후 3시 이후에 시작하는 수업"은 시작 시간이 이 DB에 없으므로 "정보 없음"으로 거절하는 것이 맞다.
- 다시 만드는 SQL(합성 DB에서만 실행):

```sql
CREATE SCHEMA packed_school;
-- isu_gubun, hakgwa, gyosu, gyogwamok: other_school 그대로 복사하고 같은 PK·FK를 건다.
CREATE TABLE packed_school.gangjwa AS SELECT g.*, (SELECT string_agg(
    (CASE s.yoil WHEN 'MON' THEN '월' WHEN 'TUE' THEN '화' WHEN 'WED' THEN '수' WHEN 'THU' THEN '목' ELSE '금' END)
    || ' ' || SUBSTRING(s.si_st FROM 1 FOR 2) || ':' || SUBSTRING(s.si_st FROM 3 FOR 2)
    || '-' || SUBSTRING(s.si_ed FROM 1 FOR 2) || ':' || SUBSTRING(s.si_ed FROM 3 FOR 2), ', ' ORDER BY s.sp_id)
  FROM other_school.siganpyo s WHERE s.ks_id = g.ks_id)::varchar(100) AS sigan_txt
FROM other_school.gangjwa g;
ALTER TABLE packed_school.gangjwa ADD PRIMARY KEY (ks_id), ADD FOREIGN KEY (gm_cd) REFERENCES packed_school.gyogwamok (gm_cd),
  ADD FOREIGN KEY (gs_no) REFERENCES packed_school.gyosu (gs_no);
```

## 2. 첫 실행 (코드 수정 전)

강의 요일을 `sigan_txt`에 연결하고, 시작·종료 시간은 "이 DB에 없음"으로 검토해 게시했다.

- 12/13. q11·b01은 "시작 시간 정보 없음"으로 바르게 거절했다(8-20 이전부터 있던 unsupported 처리).
- q05 "월요일 수업"이 0건이었다. 강의 요일 = '월요일'을 묶인 글자 전체와 같음으로 비교했기 때문이다. 0건 진단(8-20)이 그 조건을 짚었다.

## 3. 고친 것

- **닫힌 변환 `PACKED_WEEKDAY`:** "월요일"을 "월"로 바꾸고 포함으로 찾는다(`LIKE '%월%'`). 연산자는 포함만 허용한다. 챗봇은 같음을 포함으로 바꾼다(이름형 항목과 같은 방식).
- **서버 규칙으로만 붙는다.** 저장 값이 모두 "요일 + 숫자·시각 기호"(`월 09:00-10:15, 수 …`, `월1-2/수3`) 꼴일 때만 붙는다. 과목명처럼 다른 글자가 섞이면 붙지 않는다. 요일 한 글자만 있는 column은 기존 `KOREAN_WEEKDAY`가 맡는다.
- **게시 때 값 표본 읽기:** 게시 때 변환을 정하려고 값을 50개까지 읽는데, 값 종류가 그보다 많으면 아무 변환도 붙지 않았다(묶인 시간표는 80종). 이제 값이 많으면 50개 표본을 읽는다. 표본으로는 `PACKED_WEEKDAY`만 고를 수 있다. 이 변환은 `LIKE`로만 찾아서, 표본 밖의 이상한 값이 있어도 SQL이 실패하지 않는다. 다른 변환(학점 숫자 변환 등)은 지금처럼 전체 값이 필요하다.

## 4. 고친 뒤 결과

- `packed_school` **13/13**(q11·b01은 기대대로 거절).
- 레거시: 컴융 전공 82건, 월요일 오전 200건(상한)으로 이전과 같다.

## 5. 리뷰

`gpt-6-luna` high review-only 지적 1건을 재현하고 고쳤다.

- **표본이 모든 변환 결정에 쓰였다.** 값이 50종 넘는 학점 column에서 표본 50개가 `3/3/0` 꼴이면 `SLASH_FIRST_NUMERIC`이 붙을 수 있었다. 그러면 표본 밖의 이상한 값(`audit` 등)에서 숫자 변환이 실패한다. 이전에는 이런 column에 변환이 붙지 않았다.
- 고친 내용: 표본 값에는 표시(`SampledValues`)를 붙이고, 표본으로는 `PACKED_WEEKDAY`만 고르게 했다. 회귀 테스트를 추가했다. 고친 뒤 `packed_school`을 다시 등록해 13/13을 확인했다.

## 6. 남은 것

- 묶인 시간표에서 시작·종료 시간까지 꺼내는 문자열 해석 변환(지침의 ②)은 하지 않았다. 필요하면 사용자 결정으로 진행한다.
- "금요일 오전 수업"은 요일 부분까지 함께 거절된다(시작 시간이 없으므로). 요일만으로 답하지 않는 것이 맞다고 보았다.

---

<a id="phase8-21-period-results-2026-10-06"></a>

## 원본: phase8-21-period-results-2026-10-06.md

# Phase 8-21 결과 (3): 교시 시간표 형식 B (2026-10-06)

지침: [phase8-21-postgres-format-variety.md](development-history.md#prompts-phase8-21-postgres-format-variety). 앞의 결과: [D 넓은 table](experiment-results.md#phase8-21-wide-table-results-2026-10-02), [A 공통코드](experiment-results.md#phase8-21-code-table-results-2026-10-05).

## 1. 만든 DB

- 합성 DB의 `period_school` schema. `other_school`과 내용은 같고, 시간표만 시각 대신 교시 번호(정수)로 저장한다.
  - 교시는 09:00부터 30분 단위다(1교시 09:00, 2교시 09:30, …).
  - `gyosi_from`은 수업이 시작하는 교시다. 원래 시작 시각과 모두 일치한다(불일치 0건).
  - `gyosi_to`는 수업이 걸치는 마지막 교시다. 원래 종료 시각을 30분 단위로 올린 것이라 종료 시각은 원본과 다르다. 고정 질문 중 종료 시각을 쓰는 것은 없다.
- 고정 질문: q01~q12와 정답을 그대로 썼다. 첫 실행 전에 b01 "오후 3시 이후에 시작하는 수업 보여줘"(시작 ≥ 15:00)를 더 고정했다.
- 다시 만드는 SQL(합성 DB에서만 실행):

```sql
CREATE SCHEMA period_school;
-- isu_gubun, hakgwa, gyosu, gyogwamok, gangjwa: other_school 그대로 복사(CREATE TABLE AS SELECT *)하고 같은 PK·FK를 건다.
CREATE TABLE period_school.siganpyo AS SELECT sp_id, ks_id, yoil,
  ((CAST(SUBSTRING(si_st FROM 1 FOR 2) AS integer) * 60 + CAST(SUBSTRING(si_st FROM 3 FOR 2) AS integer) - 540) / 30 + 1)::integer AS gyosi_from,
  ((CAST(SUBSTRING(si_ed FROM 1 FOR 2) AS integer) * 60 + CAST(SUBSTRING(si_ed FROM 3 FOR 2) AS integer) - 540 + 29) / 30)::integer AS gyosi_to
FROM other_school.siganpyo;
ALTER TABLE period_school.siganpyo ADD PRIMARY KEY (sp_id), ADD FOREIGN KEY (ks_id) REFERENCES period_school.gangjwa (ks_id);
```

## 2. 첫 실행 (코드 수정 전)

- 시작·종료 시간을 정수 교시 column에 연결해도 **게시가 막히지 않았다**. 변환이 없는데도 통과했다.
- b01 "오후 3시 이후"는 **HTTP 500**이었다. 정수 column을 '15:00'과 비교해 SQL이 실패했다.
- 시간과 무관한 질문(q01·q05·q06·q12)은 맞았다.

## 3. 고친 것

- **교시표 = 값 사전.** 시간 항목의 값 사전에서 이름이 모두 `HH:MM`이면 교시표로 본다(1 → 09:00).
  - 저장 위치, 게시, 관리자 화면(값 사전 패널), 결과 카드의 이름 표시는 기존 값 사전 기능을 그대로 쓴다. 카드에는 교시 번호 대신 시각이 보인다.
- **시간 비교 → 교시 목록.** "시작 시간 < 12:00"은 `column IN (12시 전에 시작하는 교시들)`이 된다.
  - compiler와 SQL policy가 같은 함수(`time_label_values`)로 목록을 정한다. policy는 그 조건에만 정확한 `IN` 모양을 허용한다.
- **조건에 맞는 교시가 없으면** SQL 없이 0건으로 답하고 이유를 보인다. 예: "밤 10시 이후" → "종료 시간에서 '22:00:00' 값을 찾지 못했습니다"(모델이 종료 시간으로 이해함).
- **게시 검사**
  - 시간 항목이 정수 column에 연결되었는데 교시표가 없으면, 게시를 막고 그 항목을 알려 준다.
  - runtime도 같은 경우를 거부한다.
- **관리자 입력**
  - 교시표 항목은 "DB에 있는 값" 검사 대신 형식 검사를 한다. 교시는 숫자, 이름은 `HH:MM`이고 별칭은 없다.
  - 아직 쓰지 않는 교시도 넣을 수 있다. 학교 시간표 전체를 한 번에 넣는 것이 자연스럽고, 표에 빠진 교시는 조건에 걸리지 않기 때문이다.
- 값 읽기(값 사전·힌트용)는 column을 글자로 읽는다. 그래야 정수 교시 값도 목록에 보인다.

## 4. 고친 뒤 결과

- 교시표(시작·종료 각 18개)를 넣고 게시: 고정 질문 **11/13**. 틀린 2개는 모두 모델이 조건을 다르게 이해한 것이다.
  - q11 "금요일 오전 수업"은 모델이 시간 조건을 뺐다(8-20·D에서도 본 흔들림).
  - b01 "오후 3시 이후에 시작하는 수업"은 모델이 종료 시간으로 이해했다.
- 교시 계산 자체는 정확했다.
  - "금요일 12시 이전에 시작하는 수업" → 7개(정답).
  - "15시 이후 시작 수업" → 모델이 `>`로 이해해 15:00 시작은 뺀 12건이었다.
- 레거시: 컴융 전공 82건, 월요일 오전 200건(상한), 컴융 3학년 전공 33건으로 이전과 같다.

## 5. 리뷰

`gpt-6-luna` high review-only 지적 1건을 재현하고 고쳤다.

- **`01`처럼 0을 붙인 교시 번호가 입력되었다.** 정수 column은 `1`을 돌려주므로 카드에 `09:00` 대신 `1`이 보였을 것이다. 이제 교시 번호는 `1`처럼 0 없는 정수만 받는다. 같은 교시가 `1`과 `01` 두 항목으로 들어가는 것도 막힌다. 회귀 테스트를 추가했다.

## 6. 남은 것

- "1교시 수업"처럼 교시 번호로 묻는 질문은 다루지 않는다. 공통 항목에 교시가 없다.
- 관리자 화면의 값 사전 패널은 값 목록이 오면 그 안에서만 고르게 한다. 교시표 항목은 서버가 목록을 보내지 않아 자유 입력 칸이 나온다. 화면을 직접 열어 확인하지는 않았다(API 테스트로만 확인).
- 형식 C(한 칸에 묶인 시간표)가 남았다.

---

<a id="phase8-21-wide-table-results-2026-10-02"></a>

## 원본: phase8-21-wide-table-results-2026-10-02.md

# Phase 8-21 결과 (1): 넓은 단일 table 형식 D (2026-10-02)

지침: [phase8-21-postgres-format-variety.md](development-history.md#prompts-phase8-21-postgres-format-variety). 이번 사용량 주기에는 형식 D만 확인했다. A(공통코드)·B(교시)·C(묶인 시간표)는 다음 주기에 한다.

## 1. 만든 DB

- 합성 DB `schema_adaptive_synthetic`의 `wide_school` schema. 8-17 `other_school`의 내용을 join해 table 하나(`sugang_pyeollam`)로 만들었다. 시간표 칸마다 한 행이다(143행, 과목 80개).
- column 이름은 `other_school`과 다르게 지었다(`gwamok_bunho`, `hakgwa_myeong`, `damdang_gyosu`, `sijak` 등). 이수구분·학과명·교수명은 코드가 아니라 이름 그대로 들어 있다.
- 데이터가 같으므로 고정 질문 q01~q12와 정답(`other_school.gold`)을 그대로 썼다. q13·q14(강의실)는 강의실 column이 없어 뺐다.
- 다시 만드는 SQL(합성 DB에서만 실행):

```sql
CREATE SCHEMA wide_school;
CREATE TABLE wide_school.sugang_pyeollam AS
SELECT s.sp_id AS haeng_no, g.gm_cd AS gwamok_bunho, m.gm_nm AS gwamok_myeong, m.hakjeom,
       i.isu_nm AS isu_gubun, m.suggang_hangnyeon AS daesang_hangnyeon,
       h.hg_cd AS hakgwa_cd, h.hg_nm AS hakgwa_myeong, p.gs_nm AS damdang_gyosu,
       g.nyeondo, g.hakgi_cd AS hakgi, g.ban_no AS bunban, g.jeongwon, g.sugang_inwon,
       s.yoil, s.si_st AS sijak, s.si_ed AS jongryo
FROM other_school.siganpyo s
JOIN other_school.gangjwa g ON g.ks_id = s.ks_id
JOIN other_school.gyogwamok m ON m.gm_cd = g.gm_cd
JOIN other_school.isu_gubun i ON i.isu_cd = m.isu_cd
JOIN other_school.hakgwa h ON h.hg_cd = m.hg_cd
JOIN other_school.gyosu p ON p.gs_no = g.gs_no;
ALTER TABLE wide_school.sugang_pyeollam ADD PRIMARY KEY (haeng_no);
-- 지우기: DROP TABLE wide_school.sugang_pyeollam; DROP SCHEMA wide_school;
```

## 2. 확인 방법

화면이 쓰는 API로 테스트 서버(8124)에 등록 → 검토 → 값 사전 → 게시했다. `other_school` 때와 같은 관리자 작업이다.
- 값 사전: 학과 약어 `소웨`·`전전`·`기공`, 학기 `10`·`20`, 요일 `MON`~`FRI`.
- 시간 `0900` 형식의 변환(`HHMM_TEXT_TO_HHMM`)은 서버 규칙이 정했다.

## 3. 결과

- **코드 수정 없이 동작했다.** 등록·검토·게시가 막힘 없이 끝났다(mapping v1).
- AI mapping 첫 후보는 16개 중 14개가 맞았다. 교원명·종료 시간은 후보가 없어 관리자가 지정했다.
- 고정 질문 **11/12**.
  - 틀린 것은 q11 "금요일 오전 수업 보여줘" 하나다. 모델이 시간 조건을 빼서 14개가 나왔다(정답 7개).
  - 같은 질문을 두 번 더 보내자 시간 조건을 뺀 답(14개)과 맞는 답(7개)이 한 번씩 나왔다. "금요일 12시 이전에 시작하는 수업"은 7개로 맞았다.
  - 8-20 가상 학교에서 본 것과 같은 모델 흔들림이다. 넓은 table의 문제가 아니다.
- 넓은 table에서도 시간 변환과 값 사전이 그대로 적용되었다.

## 4. 남은 것

- 형식 A·B·C는 지침 4절 순서로 다음 주기에 한다.
- `wide_school` schema는 합성 DB에 남겨 두었다. 테스트 등록(`wide-school`)은 지웠다.
- 코드를 바꾸지 않았으므로 테스트, Luna 리뷰, 전체 pytest는 하지 않았다.

---

<a id="phase8-22-blind-evaluation-results-2026-10-06"></a>

## 원본: phase8-22-blind-evaluation-results-2026-10-06.md

# Phase 8-22 결과: 팀원 DB 평가 실행 도구 (2026-10-06)

지침: [phase8-22-blind-evaluation-runner.md](development-history.md#prompts-phase8-22-blind-evaluation-runner). 팀원용 안내: [teammate-evaluation-db-guide.md](teammate-evaluation-db-guide.md).

## 1. 만든 것

`python -m backend.adaptive.evaluation.blind_eval check|run|report <평가 ID>`

- **질문 파일:** `.schema-adaptive/blind-evals/<평가 ID>/questions.json`(git 밖). 형식은 `blind-eval-questions-v1`이다. 정답은 과목 코드 목록이거나 `"decline"`이다.
- **check:** 형식만 검사한다. 개수와 해시만 출력하고, 오류는 질문 번호로만 알린다. 질문 내용과 정답 코드는 출력하지 않는다.
- **run**
  - 화면과 같은 챗봇 API(loopback만)로 질문마다 한 번 묻는다.
  - 실행 번호를 먼저 잡고(`run-NNN.json`), 끝나면 결과를 채운다.
  - 첫 실행 때 질문 파일 해시를 잠근다. 잠금 파일이나 앞선 실행과 해시가 다르면 실행을 거부한다.
  - 기록에는 git commit, 모델 이름(`OLLAMA_MODEL`만 읽음), 등록 상태·게시 버전, 질문별 결과(PASS/FAIL/CAPPED/ERROR), 이해한 조건, 빠진·남는 개수를 남긴다. 결과 행은 남기지 않는다.
- **report:** 실행별 집계를 보여 준다. 첫 실행과 재실행(commit)을 구분한다. 빠진 실행 번호, 질문이 다른 실행, 끝나지 않은 실행을 경고한다.

## 2. 동작 확인 (개발용 DB, 평가 수치 아님)

8-21 `code_school`을 다시 등록하고, 고정 질문 14개와 거절 질문 1개("강의실이 IT관인 수업")로 질문 파일을 만들었다.

- check: 15개(코드 14, 거절 1)만 출력.
- run 1: "첫 실행" 15/15. run 2: "재실행" 15/15, 따로 저장됨.
- 기록: 게시 버전 1, commit, 모델 이름이 들어갔다. commit에는 `+dirty`가 붙었다. 사용자의 `package.json` 변경이 커밋되지 않은 상태이기 때문이다.
- 확인 뒤 개발용 평가 폴더와 등록은 지웠다.

## 3. 리뷰

`gpt-6-luna` high review-only 지적 5건을 모두 재현하고 고쳤다. 회귀 테스트를 추가했다.

1. **loopback 밖으로 나갈 수 있었다.** 환경 변수 proxy와 redirect를 따랐다. 이제 proxy와 redirect를 쓰지 않는다.
2. **잠금 파일을 지우면 바뀐 질문으로 "재실행"할 수 있었다.** 이제 앞선 실행 기록의 해시와도 비교한다.
3. **동시에 두 번 실행하면 한 실행의 기록이 사라질 수 있었다.** 이제 묻기 전에 실행 번호를 먼저 잡는다.
4. **평가 폴더가 링크(symlink·junction)이면 private 폴더 밖을 쓸 수 있었다.** 이제 실제 경로를 확인한다. 이 Windows 계정은 symlink를 만들 수 없어 해당 테스트는 건너뛰었다.
5. **report가 빠진 실행을 알리지 않았다.** 이제 빠진 번호, 질문이 다른 실행, 끝나지 않은 실행을 경고한다.

## 4. 남은 것

- 팀원이 DB와 질문 파일을 만들면 Phase 9에서 실제로 돌린다. 그 전에는 AI가 그 파일을 열지 않는다.
- 정답이 150개를 넘는 넓은 질문, 결과가 200행 상한에 닿는 질문은 채점하지 않는다(`CAPPED`). 질문을 좁혀야 한다.

---

<a id="phase8-23-ai-mapping-results-2026-10-06"></a>

## 원본: phase8-23-ai-mapping-results-2026-10-06.md

# Phase 8-23 결과: AI mapping 제안 개선 (2026-10-06)

지침: [phase8-23-ai-mapping-improvement.md](development-history.md#prompts-phase8-23-ai-mapping-improvement)

> 이 수치는 개발 진행 확인용이다. schema 5개와 정답은 모두 AI가 만들었고, 고친 내용도 이 schema를 보며 정했다. 처음 보는 학교에 대한 적응성 근거가 아니다(ADR-086). 평가는 팀원 DB로 한다.

## 1. 측정 도구

`python -m backend.adaptive.evaluation.school_mapping_check --label <이름> --repeat N`

- 화면 등록과 같은 경로로 제안을 만든다: 규칙 → 예시 값 → 로컬 모델(`qwen2.5-coder:7b-instruct`) → 후보 정렬.
- 대상: `other_school`, `wide_school`, `code_school`, `period_school`, `packed_school`. 항목 16개씩, 1회 80개.
- FK로 이어진 column은 같은 값이므로 둘 다 정답으로 친다. 8-21 등록 때 적은 "AI 정답 수"는 이것을 구분하지 않아 조금 낮게 나왔다.

## 2. 결과 (1회당, 3회·2회 반복 모두 같은 결과)

| 단계 | 맞은 수 | 바뀐 것 |
|---|---|---|
| 규칙만 | 5/80 | 영문 이름 규칙이라 로마자 이름을 거의 못 찾음 |
| 기준선(v2) | **55/80** | - |
| + 예시 값 표본 | 65/80 | 값 종류가 10개를 넘는 column도 예시 10개를 prompt에 넣음 |
| + prompt v3 | **75/80** | 항목 이름을 한국어와 로마자로 함께 보임, 한국 수강편람 DB 관례 4줄 |
| + 구조 검사·리뷰 수정(최종) | 75/80 | join할 수 없는 공통코드 table 선택은 "후보 없음"으로. 예시 값은 JSON 목록으로 |

학교별(최종): other 16, wide 16, code 15, period 14, packed 14.

## 3. 원인과 고친 것

1. **예시 값이 빠져 있었다.**
   - prompt는 "예시 값도 보고 판단하라"고 하는데, 값 종류가 10개를 넘는 column은 예시가 하나도 없었다. 교수명·학과명·과목 코드가 그랬다.
   - 학과명(값 11종)에 예시가 없어 단과대학명을 고르는 실수가 났다.
   - 8-21의 표본 읽기로 10개를 넣었다. 예시 값은 지금처럼 loopback 모델 prompt에만 들어가고 저장하지 않는다.
2. **로마자 column 이름을 못 읽었다.**
   - 대상 학년에 학기 코드(`hakgi_cd`)를, 분반에 교수 번호(`gs_no`)를 골랐다.
   - 항목 이름과 별칭을 로마자로 바꿔 함께 보인다(대상 학년 → daesang hangnyeon, 분반 → bunban).
   - 로마자는 국어 로마자 표기법을 계산하는 함수로 만든다. 학교별 목록이 아니다.
3. **한국 수강편람 DB 관례를 몰랐다.** prompt에 일반 관례 4줄을 더했다.
   - 이름이 로마자이거나 줄임말일 수 있다.
   - 값이 코드이고 이름은 코드 table에 있을 수 있다.
   - 시간이 교시 번호일 수 있다.
   - 한 칸에 시간표 전체가 있을 수 있다.
4. **join할 수 없는 선택:** FK가 하나도 없는 table의 column은 버린다(공통코드 table의 이름 column). 단, 선택의 과반이 FK로 이어진 한 table 묶음에 있을 때만 버린다. FK 없는 view만 있는 DB, 한 table DB, 동률이면 그대로 둔다.
   - 처음에는 "글자 항목을 숫자 column에 둔 선택"도 버렸다. 그러나 숫자형 과목 코드·분반처럼 정당한 경우가 있어 리뷰 뒤 뺐다.
- prompt 버전은 `ai-mapping-v3`(화면 등록용)다. benchmark용 v1.1·v2 prompt는 바꾸지 않았다(고정 hash 그대로).

## 4. 남은 틀림 (1회당 5개)

| 학교 | 항목 | 제안 | 정답 | 비고 |
|---|---|---|---|---|
| code_school | 학기 | 없음 | `gangjwa.hakgi_cd` | 코드 `10/20`만으로는 학기인지 알기 어려움 |
| period_school | 시작·종료 시간 | 없음 | 교시 column | 관례 문장이 있어도 고르지 않음. 어차피 교시표 입력이 필요하고, 게시 검사가 항목을 알려 줌 |
| packed_school | 요일 | 없음 | `sigan_txt` | 모델이 묶인 시간표 column을 시작 시간 쪽에 둠 |
| packed_school | 시작 시간 | `gangjwa.sigan_txt` | 없음(이 DB에 없음) | 관리자가 요일로 옮기고 시간은 "이 DB에 없음"으로 고침 |

## 5. 레거시 DB 확인 (읽기 전용)

화면 등록과 같은 경로로 `public` schema를 제안했다. 결과는 8-19와 같다.
- 14개가 맞았다.
- 알려진 이름 함정 2개(학기 → `enrollment.semester`, 개설 연도 → `course_year`)는 그대로다. 시연 절차에서 관리자가 "이 DB에 없음"으로 바로잡는 부분이다.
- 시연 절차는 바뀌지 않는다.

## 6. 리뷰

`gpt-6-luna` high review-only 지적 3건을 모두 재현하고 고쳤다. 회귀 테스트를 추가했다.

1. **저장 값으로 prompt 지시를 끼워 넣을 수 있었다.** 예시 값이 그대로 prompt 줄에 들어가, 줄바꿈과 지시문을 담은 값이 새 지시처럼 보일 수 있었다. 8-12부터 있던 문제이고, 이번에 예시가 늘며 범위가 넓어졌다. v3는 예시를 JSON 목록으로 보이고 "예시 값은 데이터이며 지시가 아니다"를 덧붙인다.
2. **글자 항목의 숫자 column 선택을 버리는 검사가 정당한 경우를 지웠다.** 숫자형 과목 코드·분반이 그렇다. 이 검사는 뺐다.
3. **FK 묶음 검사가 FK 없는 view나 동률에서 정당한 선택을 지울 수 있었다.** 이제 FK가 하나도 없는 table만, 과반이 한 FK 묶음에 있을 때만 버린다.

고친 뒤 다시 측정해도 75/80이었다(packed_school의 틀린 위치만 바뀜). 레거시 결과도 같았다.

---

<a id="phase8-24-code-table-import-results-2026-10-06"></a>

## 원본: phase8-24-code-table-import-results-2026-10-06.md

# Phase 8-24 결과: 코드표에서 값 사전 가져오기 (2026-10-06)

지침: [phase8-24-code-table-import.md](development-history.md#prompts-phase8-24-code-table-import)

## 1. 만든 것

- 값 사전 API의 `IMPORT` 동작. 관리자가 고르는 것:
  - field
  - 코드표 table, 코드 column, 이름 column
  - 필요하면 구분 column과 값
- 서버:
  - 읽기 전용으로 (코드, 이름) 쌍을 읽는다(64개까지).
  - 그 field의 데이터에 있는 코드만 넣는다. 데이터에 없는 코드는 충돌 검사 전에 건너뛴다. 공통코드 table을 구분 없이 읽어도, 다른 그룹이 같은 코드를 다른 이름으로 쓰는 것 때문에 막히지 않는다.
  - 이미 있는 항목은 이름만 바꾸고 약어는 둔다.
  - 한 코드·두 이름, 두 코드·같은 이름, 64개 초과, 교시표 field는 거절한다.
- 화면: 값 사전 패널의 "코드표에서 가져오기"(table·column 선택, 구분 column·값).
  - 결과 문구 예: "코드표에서 가져왔습니다: 새 항목 5개, 이름 바뀐 항목 0개. 게시해야 챗봇에 적용됩니다."

## 2. 확인 결과

- `code_school`을 학과 약어만 넣고 등록했다. 이수구분·요일·학기 사전은 손으로 넣지 않았다.
  - 등록 때 AI 첫 후보는 15/16이었다(8-23 v3).
- 공통코드 table에서 세 그룹을 가져왔다.
  - ISU 5개(11 전공필수 … 30 일반선택), YOIL 5개(1 월요일 …), HAKGI 2개(10 1학기, 20 2학기).
  - 건너뜀 0개, 게시 성공.
- 코드 관련 고정 질문 8/8: q01·q03·q04(이수구분 이름), q05·q11(요일), q06(학기), a01·a02(이름 일부).
- 화면(5174 → 8124):
  - 열어 둔 등록의 값 사전 패널에서 table 목록이 보였다.
  - `gongtong_code`, `code`, `code_nm`, `code_group = ISU`로 가져오기를 눌렀다. "새 항목 0개, 이름 바뀐 항목 0개"(이미 있음)가 나왔다.
  - 같은 화면의 강의 목록 오류는 레거시 백엔드(8000)를 켜지 않아서 생긴 것이고 이번 변경과 무관하다.

## 3. 리뷰

`gpt-6-luna` high review-only 지적 3건을 모두 재현하고 고쳤다. 회귀 테스트를 추가했다.

1. **맞는 코드가 하나도 없는 가져오기가 "충돌"로 거절되었다.** 빈 사전이 값 사전 검사에 걸렸기 때문이다. 이제 바뀐 것이 없으면 저장하지 않고 "건너뜀 N개"로 답한다.
2. **화면 parser가 있을 수 없는 개수를 받았다**(예: 새 항목 65개). 이제 세 개수의 합이 64(한 번에 읽는 최대 쌍 수)를 넘으면 응답을 거부한다.
3. **한국어나 공백이 든 table·column 이름이 있으면 table 목록 전체가 실패해 가져오기 칸이 사라졌다**(`"코드 표"`). 8-19 확장 항목 parser의 영문 이름 규칙 때문이고, 확장 항목 패널도 같이 영향을 받았다. 이제 PostgreSQL이 허용하는 이름(1~63자, 제어 문자 없음)을 받는다.

---

<a id="phase8-25-time-expressions-results-2026-10-06"></a>

## 원본: phase8-25-time-expressions-results-2026-10-06.md

# Phase 8-25 결과: 챗봇의 시간 표현 안정화 (2026-10-06)

지침: [phase8-25-time-expressions.md](development-history.md#prompts-phase8-25-time-expressions)

- 2026-10-06 사용자 결정으로 진행했다. "질문 이해 개선은 L1 질문 세트를 확보한 뒤"의 예외다(ADR-096).
- 아래 수치는 AI가 만든 개발용 합성 DB와 개발자가 고정한 질문 위의 것이다. L1/L3 전 잠정치이고 적응성 주장에 쓰지 않는다.

## 1. 만든 것

- `backend/adaptive/api/time_hints.py`: 질문의 시간 표현을 닫힌 한국어 규칙으로 읽는다.
  - 예: "오전" → 시작 시간 < 12:00. "오후 3시 이후에 시작하는" → 시작 시간 ≥ 15:00. "4시 전에 끝나는" → 종료 시간 < 16:00. "6시까지" → 종료 시간 ≤ 18:00.
  - 확실히 읽을 수 없으면 조건을 만들지 않고 모델에 맡긴다.
    - "3시 전 수업"(시작인지 종료인지 모름), "3시 넘어서", "3시~5시", "3시간", 조건 셋 이상.
- 쓰는 곳(모델 호출은 그대로 한 번):
  1. **힌트.** 8-13 값 힌트 줄에 연산자까지 넣어 모델에 준다. 연산자 있는 힌트가 없으면 prompt는 이전과 같다.
  2. **고침.** 모델의 시작·종료 시간 조건을 질문의 조건에 맞춘다.
     - 개수가 같으면 field·연산자·값을 바꾼다.
     - 모델이 시간 조건을 하나도 내지 않았으면 더한다. 8-13의 "조건을 더하지 않는다"에 대한 유일한 예외다.
     - 개수가 다르면 두고, 고친 출력도 QueryIR 검증·compiler·policy·executor를 그대로 지난다.
  3. **없는 시간.** 그 DB에 시작·종료 시간이 없으면 모델 출력과 관계없이 "정보 없음"이다.
- 호환:
  - 힌트를 받지 않는 예전 모델 인터페이스에는 힌트를 주지 않고 고침만 한다.
  - 레거시 비교 도구의 재생 모델은 힌트 인자를 받게 했다.

## 2. 확인 결과 (테스트 포트 8124, 합성 DB, 레거시 profile 읽기 전용)

- 실행 전에 시간 질문 7개(t01~t07)와 정답을 고정했다. 정답은 `other_school` 데이터 기준이다.

| DB | 결과 | 비고 |
| --- | --- | --- |
| `other_school` | **19/19** (q01~q12, t01~t07) | 8-20에서 흔들리던 q11 통과 |
| `other_school` 반복 | **20/20** (q11·t01·t02·t05를 5번씩) | 흔들림 없음 |
| `period_school` | 18/19 | t03 아래 참고 |
| `packed_school` | **19/19** | 시간 질문 8개(q11, t01~t07)는 모두 "정보 없음", 나머지 11개 정답 |

- `period_school` t03("오후 4시 전에 끝나는 수업")은 고정 정답보다 3과목 적었다(62 대 59).
  - 이 DB는 종료 시각을 교시로 올림해 저장한다(15:40·15:50에 끝나는 수업 → 16:00에 끝나는 교시).
  - 그래서 이 DB 데이터로는 59과목이 맞다. 챗봇 답은 59과목이고, 빠진 3과목이 정확히 그 수업이다.
  - 고정 정답이 `other_school` 기준이라 생긴 차이로, 챗봇 오류가 아니다. 점수는 고정한 대로 18/19로 적는다.
- 레거시 profile: "월요일 오전 수업" 200건(상한), 컴융 전공 82건, 컴융 3학년 전공 33건. 모두 이전과 같다.
  - "오후 3시 이후에 시작하는 수업"은 시작 시간 ≥ 15:00으로 이해했다.

## 3. 리뷰

`gpt-6-luna` high review-only 지적 4건을 모두 재현하고 고쳤다. 회귀 테스트를 추가했다.

1. **부정·모순 표현이 두 조건이 되었다.** "오전 수업 말고 오후 수업만"이 시작 < 12:00과 시작 ≥ 12:00이 되어, 모델이 시간 조건을 빼면 서로 어긋나는 조건 둘이 더해질 수 있었다. 이제 부정 말(`말고`·`빼고`·`제외`·`아닌`·`않는`·`아니`)이 있거나 한 항목의 조건이 겹치는 시간이 없으면 조건을 만들지 않는다.
2. **"밤 12시"를 정오로 읽었다.** 이제 `오후 12시`만 12:00이고, `밤`·`저녁 12시`는 읽지 않는다.
3. **`value_hints`를 위치 인자로만 받는 모델에 이름으로 넘겼다.** 이제 그런 모델은 힌트를 받지 않는 모델로 보고 고침만 한다.
4. **힌트의 연산자를 그 field가 허용하는지 보지 않았다**(예: 과목명에 `less_than`). 이제 prompt에 넣기 전에 field별 허용 연산자로 검사한다.

고친 뒤 테스트 서버를 다시 띄워 같은 질문을 돌렸다. `other_school` 19/19, `period_school` 18/19(같은 t03), `packed_school` 19/19, 레거시 200·82·33건으로 같았다. 테스트 등록 세 개는 지웠다. 합성 DB schema는 그대로 둔다.

---

<a id="phase9-04-rehearsal-results-2026-10-08"></a>

## 원본: phase9-04-rehearsal-results-2026-10-08.md

# Phase 9-04 결과: 팀원 평가 리허설 학교 (2026-10-08)

지침: [phase9-04-rehearsal-school.md](development-history.md#prompts-phase9-04-rehearsal-school)

> AI가 만든 학교와 질문·정답이다. 개발용이며 적응성 근거가 아니다(ADR-086). 사람 관문은 승인하지 않는다.
> 지침대로 이번 단계에서는 코드를 고치지 않았다. 고칠지와 방법은 사용자가 정한다(5절).

## 1. 준비

- schema: `python -m backend.adaptive.fixtures.rehearsal_school create`(합성 DB `schema_adaptive_synthetic`). dept 11행, lecture 102행, lecture_time 143행.
- 질문 14개와 정답을 첫 실행 전에 고정했다(`rehearsal-v1`, 질문 파일 sha256 `6bd52b03651253ac6fbf5d4c634b2650471ee7c9b7786e5852704d6a29c56ea8`). 정답은 강의번호(`lec_no`) 목록 13개, 거절 1개.

| ID | 질문 | 정답 수 | 보는 모양 |
| --- | --- | --- | --- |
| r01 | 소웨 전공필수 과목 | 3 | 약어 + 영문 코드 |
| r02 | 교양 과목 | 11 | 영문 코드 여러 개(이름 일부) |
| r03 | 3학점 과목 | 57 | `3-2-2` 학점 |
| r04 | 2학기 개설 과목 | 54 | `2026-2` 학기 |
| r05 | 월요일 수업 | 40 | 한 글자 요일 |
| r06 | 금요일 오전 수업 | 7 | 시간 범위 칸 |
| r07 | 오후 3시 이후에 시작하는 수업 | 23 | 시간 범위 칸 |
| r08 | 데이터베이스 과목 | 3 | 과목명 일부 |
| r09 | 정원 60명 이상인 강의 | 29 | 숫자 비교 |
| r10 | 1학년 대상 과목 | 34 | 대상 학년 |
| r11 | SW1041 과목 | 1 | 분반이 붙은 과목코드 |
| r12 | (교수명) 교수님 강의 | 2 | 강의 table의 교수명 |
| r13 | 강의 평가가 좋은 과목 | 거절 | 공통 항목 밖 개념 |
| r14 | 전전 3학점 전공 과목 | 9 | 약어 + 학점 + 코드 여러 개 |

## 2. 실행 전 예상 (첫 실행 전에 적음)

- 될 것: r01·r02(값 사전에 영문 코드 뜻을 넣으면), r05(한 글자 요일은 이미 지원), r08, r09, r10, r12, r13.
- 안 될 것:
  - r03·r14: `3-2-2`를 숫자로 읽는 변환이 없다(지금은 `/` 구분만).
  - r04: `2026-2`에서 학기를 꺼내는 변환이 없다.
  - r06·r07: `09:00-10:15`에서 시작 시간을 꺼내는 변환이 없다.
  - r11: 과목코드 칸이 `SW1041-01`이라 `SW1041`과 같음 비교가 맞지 않는다.
- 게시: 학점·시작·종료 시간을 글자 칸에 연결하면 게시가 막힐 수 있다. 그러면 관리자는 "이 DB에 없음"을 고르게 되고, 해당 질문은 "정보 없음"이 된다.
- (실행 뒤 덧붙임) 이 예상은 반만 맞았다. 막힌 이유는 값 형식이 아니라 같은 칸 규칙이었고, 학점·종료 시간은 막히지 않았다(3·5절).

## 3. 관리자 단계 (테스트 포트 8124)

| 단계 | 결과 |
| --- | --- |
| 등록·AI 제안 | AI 첫 후보 15/16(분반은 "후보 없음"이 맞음). 학기·연도 → `term`, 시작·종료 → `time_range`, 학점 → `credit_txt` |
| 검토 | 분반을 "이 DB에 없음"으로. 나머지는 AI 제안대로 승인(연도·시작·종료도 같은 칸을 제안했고 그대로 둠) |
| 값 사전 10개 | 학과 약어 3(`소웨`·`전전`·`기공`), 이수구분 영문 코드 5(`MR`=전공필수 …), 학기 2(`2026-1`=1학기, `2026-2`=2학기) |
| 게시 1차 | **막힘**: 연도, 시작 시간. 원인은 값 형식이 아니라 **"한 칸은 한 항목만" 규칙**이다(`plan_publish`). 같은 칸에 연결된 두 항목 중 이름 순으로 먼저인 학기·종료 시간이 칸을 차지하고, 연도·시작 시간이 막혔다. 화면에는 막힌 항목만 나오고 이 이유는 나오지 않는다 |
| 관리자 조치 | 두 항목을 "이 DB에 없음"으로 바꿈 |
| 게시 2차 | 성공(mapping v1). 변환은 요일(`KOREAN_WEEKDAY`)만. **학점(`3-2-2`)과 종료 시간(`09:00-10:15`)은 변환 없이 게시됨.** 글자 칸에 연결한 숫자·시간 항목의 값 형식은 게시 때 검사하지 않는다 |

관리자 작업: 검토 16건, 값 사전 10건, 게시 2번(막힌 뒤 "이 DB에 없음" 2건).

## 4. 첫 실행 (고정, `run-001`)

`blind_eval run rehearsal-v1 --base-url http://127.0.0.1:8124` → **6/14 통과**(실패 6, 오류 2).

| ID | 결과 | 원인 |
| --- | --- | --- |
| r01·r02·r05·r08·r10 | 통과 | |
| r13 | 통과(거절) | 강의 평가 → "정보 없음" |
| r03·r14 | 오류(HTTP 500) | 학점 글자 칸 `3-2-2`를 숫자 3과 비교해 SQL 실행이 실패 |
| r04·r09·r12 | 실패("분반 정보 없음") | 모델이 보여 줄 항목에 **분반만** 넣었다. 이 DB는 분반이 없어서 답 전체가 거절됨. 조건은 맞았다(`2026-2`, 정원 ≥ 60, 교수명) |
| r06·r07 | 실패("시작 시간 정보 없음") | 관리자가 게시를 위해 시작 시간을 "이 DB에 없음"으로 둠 |
| r11 | 실패(0건) | `SW1041`과 `SW1041-01`이 같지 않음. 0건 안내("과목 코드에서 'SW1041' 값을 찾지 못했습니다")는 나옴 |

분석은 첫 실행 뒤 같은 질문을 다시 보내, 모델 출력(QueryIR)을 테스트 서버에만 기록해 확인했다.

## 5. 발견한 문제와 처리 후보 (사용자 결정 대기)

> 2026-10-08 갱신: 1·2번은 사용자 결정으로 9-05에서 고쳤다([결과](experiment-results.md#phase9-05-raw-time-and-unasked-select-results-2026-10-08), 재실행 9/14). 3~6번은 그대로다.

심각한 순서다.

1. **조용히 틀린 답(종료 시간).** 종료 시간이 `09:00-10:15` 글자 칸에 변환 없이 게시되었다. "4시 전에 끝나는 수업"을 물으면 글자 비교(`time_range < '16:00'`)를 해서 **정답 79과목에 틀린 17과목이 섞인 96과목**을 경고 없이 보였다. 게시 검사는 글자 칸에 연결한 시간 항목의 값 형식을 보지 않는다(변환이 정해지지 않으면 그대로 글자 비교).
   - 같은 칸 규칙 때문에 시작·종료 중 이름 순으로 먼저인 **종료 시간**이 칸을 차지했다. 학생 질문은 시작 시간이 더 흔해서 관리자에게는 반대가 더 유용했을 것이다.
   - 후보: 게시 검사에서 변환 없는 글자 칸의 시간 항목 값이 `HH:MM` 꼴이 아니면 막는다(최소 수정). 더 나아가 범위 칸에서 시작·끝을 꺼내는 닫힌 변환을 더하고, 그 변환에 한해 한 칸을 두 항목이 쓰게 한다.
2. **분반이 없는 DB에서 답 대부분이 거절됨.** 모델이 보여 줄 항목(`select`)에 분반을 자주 넣는다. 그런데 거절 규칙은 select의 항목도 본다. 챗봇 카드는 select를 어차피 고정 항목으로 바꾸므로, 질문이 분반을 말하지 않았다면 거절할 이유가 없다. 8-20의 "모델이 혼자 더한 항목은 탓하지 않는다"와 같은 원리다.
   - 후보: select에만 있고 질문이 말하지 않은 "이 DB에 없음" 항목은 거절하지 않고 select에서 뺀다. 질문이 그 항목을 말하면("분반 알려줘") 지금처럼 거절한다. 빼서 select가 비면 쓸 수 있는 항목 하나를 넣는다(QueryIR은 select가 비면 안 된다).
3. **학점 글자 칸이 게시되고 실행 때 오류.** `3-2-2`를 숫자 항목에 연결해도 게시가 막히지 않았고, 질문하면 HTTP 500이었다.
   - 후보: 숫자 항목은 칸이 숫자 형식이거나, 검증된 글자→숫자 변환이 있을 때만 게시한다. 값이 `3`처럼 숫자로 읽혀도 글자 칸이면 SQL에서 글자=숫자 비교가 실패하므로 값 검사만으로는 부족하다. `3-2-2`는 `/` 변환(`SLASH_FIRST_NUMERIC`, SQL이 `/`로 고정)을 넓히는 것으로는 안 되고, `-` 구분용 별도 변환이 필요하다.
4. **분반이 붙은 과목코드.** `SW1041`로 찾으면 0건이다. 안내는 나오지만 답을 못 한다.
   - 후보: 그대로 두고 한계로 적기. 또는 조건에만 쓰는 "구분자까지 앞부분 일치"(`SW1041` → `SW1041-%`, `SW10410`은 제외). 칸 전체에 거는 변환은 화면에 보이는 과목코드까지 바꾸므로 맞지 않는다.
5. **연도·학기가 한 칸.** 학기는 관리자가 값 사전(`2026-2`=2학기)으로 풀었다. 연도는 "한 칸은 한 항목만" 규칙으로 막혀 "이 DB에 없음"이 되었다. 연도 질문이 드물어 우선순위는 낮다.
6. **게시가 막힌 이유가 안 보인다.** 화면은 막힌 항목 이름만 보인다. "다른 항목이 이미 이 칸을 쓴다"는 이유를 보이면 관리자가 어느 쪽을 남길지 고를 수 있다.

학교별 데이터로 이미 풀린 것: 영문 이수구분 코드, 학과 약어, 학기(`2026-1/2`), 한 글자 요일.

## 6. 리뷰

`gpt-6-luna` high review-only(표준 입력을 닫고 실행). 지적을 코드로 확인해 반영했다.

- 게시가 막힌 원인은 같은 칸 규칙이다(처음 결과 문서는 값 형식 때문이라고 잘못 적었다. 리뷰 전에 코드로 확인해 고쳤고, 리뷰도 같은 결론).
- 처리 후보 세 가지를 다듬었다. 분반 select 제외는 질문이 말하면 거절을 유지하고 select가 비지 않게 한다. 학점은 값 검사만으로 부족하고 별도 `-` 변환이 필요하다. 과목코드는 조건에만 쓰는 구분자 앞부분 일치로 한다.
- fixture 테스트 보강: 바뀐 모든 칸(학기, 요일, 교수명, 정원·수강 인원, 대상 학년, 강의실, 분반)을 `other_school`과 대조하고, loopback이지만 다른 DB를 가리키는 연결도 거부하는지 본다.
- 79/96 수치는 테스트 서버에서 실제로 질문해 얻은 값이다(리뷰는 실행 기록이 없어 직접 확인하지 못했다고 적었다).

## 7. 팀원 평가 과정 리허설에서 본 것

- `blind_eval check → run → report`는 안내대로 동작했다. 질문 파일 잠금과 첫 실행 고정도 확인했다.
- 관리자 쪽에서 팀원이 겪을 일:
  - 게시가 막히면 어떤 항목인지 화면에 나온다. 그러나 "왜 막혔는지(같은 칸)"와 "막히지 않은 항목도 위험할 수 있다(값 형식 미검사)"는 알 수 없다(문제 1·3·6).
  - 팀원 안내 문서에 "게시가 막히면 그 항목을 '이 DB에 없음'으로 두고 다시 게시한다"를 적어 둘 필요가 있다.
- 첫 실행의 실패 원인을 보려면 질문과 모델 출력을 봐야 한다. 팀원 DB에서는 이 순간 그 DB가 "처음 보는 학교"가 아니게 되므로, 피드백용과 최종 평가용을 나누는 것이 맞다(중간 점검 논의 1번에 덧붙일 내용).

## 8. 정리

- 테스트 등록(`rehearsal-school`)은 지웠다. schema `rehearsal_school`과 평가 기록 `.schema-adaptive/blind-evals/rehearsal-v1/`(git 밖)은 남긴다.
- 다시 만들기: `python -m backend.adaptive.fixtures.rehearsal_school create`, 지우기: `drop`.

---

<a id="phase9-05-raw-time-and-unasked-select-results-2026-10-08"></a>

## 원본: phase9-05-raw-time-and-unasked-select-results-2026-10-08.md

# Phase 9-05 결과: 리허설 문제 1·2 고치기 (2026-10-08)

지침: [phase9-05-raw-time-and-unasked-select.md](development-history.md#prompts-phase9-05-raw-time-and-unasked-select) · 앞 단계: [phase9-04 결과](experiment-results.md#phase9-04-rehearsal-results-2026-10-08)

> 사용자 결정(2026-10-08): 기존 레거시 결과가 이상해지지 않는 범위에서 1·2번을 고친다. 수치는 AI가 만든 개발용 질문·DB 위의 것이다.

## 1. 고친 것

1. **게시 검사(`plan_publish`).** 시간 항목이 변환 없이 글자 칸에 남으면 그 칸의 값이 모두 0을 채운 `HH:MM`이어야 한다. 아니면 그 항목으로 게시를 막는다. `time` 형식 칸, `0900` 꼴, 교시표 항목은 그대로다.
2. **챗봇 거절 규칙(`_drop_unasked_select`).** select에만 있고 질문이 말하지 않은 "이 DB에 없음" 항목은 거절하지 않고 select에서 뺀다. 질문이 말하거나 조건에 쓰면 지금처럼 거절한다. select가 비면 쓸 수 있는 카드 항목 하나를 넣는다.

## 2. 레거시 결과 (고치기 전·후, 테스트 포트 8124)

- 레거시 시간 칸은 `time` 형식(`course_schedule`, `v_course_info`)이라 1번 검사 대상이 아니다.
- 비교 30문항 + 고정 확인 6개(월요일 오전, 컴융 전공, 컴융 3학년 전공, 교111 강의실, 2학기 개설, 학과 코드)를 고치기 전에 기록하고, 고친 뒤 다시 물었다.
- **36/36 같음**: 상태, 행 수, 과목 코드 목록, 거절 사유가 모두 같았다.
  - 그대로인 결과의 예: 컴융 전공 82건, 컴융 3학년 전공 33건, 월요일 오전 200건(상한).
  - 학기·개설 연도·학과 코드를 묻는 질문은 여전히 "정보 없음"이다(질문이 말하거나 조건에 씀).

## 3. 리허설 학교 재실행 (`run-002`, 재실행으로 기록)

- 같은 관리자 순서로 다시 등록했다. 게시가 세 번 만에 됐다.
  - 1차: 연도·시작 시간이 막혔다(같은 칸 규칙).
  - 2차: **종료 시간이 막혔다(이번 1번 검사)**.
  - 관리자가 세 항목을 "이 DB에 없음"으로 바꾼 뒤 게시됐다.
- `blind_eval run` → **9/14**(첫 실행 6/14). 첫 실행 기록 `run-001`은 그대로다.

| ID | 첫 실행 | 재실행 | 이유 |
| --- | --- | --- | --- |
| r04 2학기 | 실패 | **통과** | 2번: 분반 select를 빼고 답함 |
| r09 정원 60명 이상 | 실패 | **통과** | 2번 |
| r12 교수님 강의 | 실패 | **통과** | 2번 |
| r03·r14 학점 | 오류 | 오류 | 3번(학점 글자 칸), 이번 범위 밖 |
| r06·r07 시작 시간 | 실패 | 실패 | 시작 시간 "이 DB에 없음"(범위 칸 변환 없음) |
| r11 과목코드 | 실패 | 실패 | 4번(분반이 붙은 코드), 이번 범위 밖 |

- "4시 전에 끝나는 수업 보여줘": 고치기 전에는 정답 79과목에 틀린 17과목이 섞인 96과목이었다. 이제 **"종료 시간 정보 없음"**으로 답한다.

## 4. 남은 것

- 문제 3(학점 글자 칸 → 실행 오류), 4(분반이 붙은 과목코드), 5(연도·학기 한 칸), 6(게시 막힘 이유 미표시)은 그대로다.
- 게시 막힘이 한 번에 다 나오지 않는다. 같은 칸 규칙 → 시간 형식 검사 순서로 나뉘어 나와서, 관리자가 게시를 여러 번 눌러야 했다(문제 6과 함께 볼 것).
- 범위 칸(`09:00-10:15`)에서 시작·끝을 꺼내는 변환은 아직 없다. 그런 DB에서는 시간 질문이 "정보 없음"이다.

## 5. 리뷰

`gpt-6-luna` high review-only(표준 입력을 닫고 실행). 지적 3건을 코드로 확인했다.

1. **이름 없이 묻는 연도가 빠질 수 있었다.** "지난해 개설된 3학점 과목"에서 모델이 연도를 select에만 넣으면, "지난해"는 연도의 이름·별칭이 아니라서 연도를 빼고 모든 연도의 3학점 과목을 답했을 것이다. 고침: 연도를 이름 없이 묻는 말(`작년`·`지난해`·`올해`·`내년`·`연도`·`2025년` 등)이 있으면 빼지 않고 이전처럼 거절한다. 공통 ontology 별칭은 바꾸지 않았다(사용자 결정 대상).
2. **레거시에서도 결과가 바뀔 수 있다.** 맞다. 바뀌는 경우는 질문이 말하지 않은 학기·연도·학과 코드를 모델이 select에만 넣어 **이전에 "정보 없음"이던 답**뿐이다. 이전에 답하던 결과는 바뀌지 않는다. 사용자 조건("잘 나오던 것이 이상해지지 않을 것")과 맞고, 1번 보완으로 연도를 이름 없이 묻는 경우는 이전처럼 거절한다. 실제 36문항은 고친 뒤(1·3번 보완 포함)에도 36/36 같았다.
3. **빈 글자 시간 값이 검사를 지나갔다.** 값 읽기는 빈 글자를 빼므로 `""`이 섞인 칸이 게시될 수 있었고, `"" < "10:00"`이 맞아 틀린 행이 나올 수 있었다. 읽기에 실패해도 빈 목록이라 통과했다. 고침: 빈 글자를 따로 묻고(기존 정확 일치 확인 `present`), 있거나 확인할 수 없으면 막는다. 읽은 값이 하나도 없어도 막는다.

회귀 테스트를 더했다(빈 글자·읽을 값 없음 → 막힘, 연도를 이름 없이 묻는 질문 3개 → 거절).

---

<a id="phase9-06-numeric-columns-results-2026-10-10"></a>

## 원본: phase9-06-numeric-columns-results-2026-10-10.md

# Phase 9-06 결과: 숫자 항목을 글자 칸에 그대로 게시하지 않기 (2026-10-10)

지침: [phase9-06-numeric-columns.md](development-history.md#prompts-phase9-06-numeric-columns) · 앞 단계: [9-04](experiment-results.md#phase9-04-rehearsal-results-2026-10-08), [9-05](experiment-results.md#phase9-05-raw-time-and-unasked-select-results-2026-10-08)

> 사용자 결정(2026-10-09): 추천대로 리허설 문제 3을 고친다. 조건: 기존 레거시 결과가 이상해지지 않을 것. 수치는 AI가 만든 개발용 질문·DB 위의 것이다.

## 1. 고친 것

- 게시 검사(`plan_publish`). 공통 항목 중 숫자 항목(학점, 대상 학년, 개설 연도, 정원, 수강 인원)은 다음 중 하나일 때만 게시한다.
  - 칸이 숫자 형식이다(`integer`, `numeric(3,1)`, `double precision` 등).
  - 검증된 글자→숫자 변환이 정해졌다(`3/3/0` 꼴의 `SLASH_FIRST_NUMERIC`).
- 아니면 그 항목으로 게시를 막는다. (처음에는 "값 사전이 있으면 예외"도 두었으나, 숫자 항목에는 값 사전을 쓸 수 없어 리뷰 뒤 뺐다.)
- 글자→숫자·시간 변환(`3/3/0`, `0900`)이 정해진 칸에 빈 글자(`""`)가 있으면 게시를 막는다(리뷰 지적). 관리자는 그 항목을 "이 DB에 없음"으로 둔다.
- `3-2-2`를 실제로 읽는 변환은 만들지 않았다. DB별 확장 항목은 대상이 아니다.

## 2. 레거시 확인

- 고치기 전 읽기 전용 확인: 레거시 public schema의 정원·수강 인원·대상 학년·개설 연도는 `integer`, 학점(`credit_hours`)은 글자 칸이지만 `3/3/0` 꼴이다. 합성 학교 다섯 곳의 숫자 항목도 모두 숫자 형식이다.
- 화면 등록 경로(테스트 포트 8124)로 레거시 public schema를 등록하고 AI 첫 후보를 모두 승인해 게시했다. **1번에 게시되었다.** 학점은 `SLASH_FIRST_NUMERIC`, 시작·종료 시간은 `TIME_TO_HHMM`이다.
- 이번 변경은 게시 단계만 바꾸므로 이미 게시된 레거시 profile과 챗봇 답에는 영향이 없다. 확인으로 물은 질문:
  - 3학점 과목: 화면 등록본·레거시 profile 모두 답함(200건 상한).
  - 컴융 전공 82건, 컴융 3학년 전공 33건, 월요일 오전 200건, 정원 60명 이상 200건. 이전과 같다.
- 9-05의 36문항 고치기 전 기록은 날짜가 바뀌며 scratchpad와 함께 지워져 그대로 다시 대조하지는 않았다. 바뀐 코드가 게시 경로뿐이라 챗봇 답에 닿지 않는다.

## 3. 리허설 학교 재실행 (`run-003`, 재실행)

- 같은 관리자 순서로 다시 등록했다. 게시 막힘이 단계별로 나와 네 번 만에 게시되었다.
  - 1차: 연도·시작 시간(같은 칸 규칙).
  - 2차: 종료 시간(9-05 시간 검사).
  - 3차: **학점(이번 숫자 검사)**.
- `blind_eval run` → **9/14, 오류 0**(재실행 9-05 `run-002`는 9/14, 오류 2).
  - r03 "3학점 과목", r14 "전전 3학점 전공 과목": HTTP 500 → "이 DB에는 학점 정보가 없어 답할 수 없습니다".
  - 점수는 그대로다. 정답이 강의 목록이라 거절은 실패로 센다. 대신 서버 오류가 없어졌다.

## 4. 남은 것

- 문제 4(분반이 붙은 과목코드), 5(연도·학기 한 칸), 6(게시 막힘 이유 미표시)은 그대로다.
- 게시 막힘이 같은 칸 → 시간 → 숫자 순서로 나뉘어 나와, 관리자가 게시를 네 번 눌렀다. 문제 6과 함께, 막힌 항목을 한 번에 모두 보이는 것이 다음 후보다.
- `3-2-2`·`09:00-10:15`처럼 묶인 값을 읽는 변환은 없다. 그런 학교에서는 해당 질문이 "정보 없음"이다.

## 5. 리뷰

`gpt-6-luna` high review-only(표준 입력을 닫고 단독 실행). 지적 3건을 코드로 확인했다.

1. **빈 글자가 변환 검사를 지나갔다.** 값 읽기는 빈 글자를 빼므로, `3/3/0`과 `""`이 섞인 학점 칸에도 `SLASH_FIRST_NUMERIC`이 정해지고 질문하면 `CAST('' AS NUMERIC)`에서 실패했을 것이다. `0900` 꼴 시간도 같다. 고침: 이 두 변환이 정해진 칸은 빈 글자를 따로 묻고, 있거나 확인할 수 없으면 막는다. 레거시 학점 칸에는 빈 글자가 없음을 읽기 전용으로 확인했고(2588행·6846행 모두 0), 고친 뒤 레거시 화면 등록과 `other_school` 등록이 모두 1번에 게시되었다.
2. **PostgreSQL 도메인 형식의 숫자 칸이 막힌다.** 카탈로그가 `format_type`의 도메인 이름을 그대로 적기 때문이다. 카탈로그 형식을 바꾸면 schema fingerprint 등 다른 곳에 닿아 이번에는 고치지 않고 한계로 둔다. 팀원 안내에 "숫자 칸은 `integer`·`numeric` 형식으로 만든다"를 적었다.
3. **값 사전 예외가 쓸모없었다.** 숫자 항목에는 값 사전을 만들 수 없고 런타임도 거부한다. 예외가 있으면 막힌 항목 이름 없이 게시가 실패한다. 고침: 예외를 뺐다.

- 리뷰가 확인한 것: 레거시 profile(`legacy-view`)은 게시 경로를 거치지 않아 영향이 없다. 이미 게시된 예전 profile은 다시 검사하지 않는다(소급 적용 없음).
- 회귀 테스트를 더했다(빈 글자 + 변환 → 막힘 2건, 숫자 항목 값 사전 예외 없음).

---

<a id="phase9-07-publish-block-reasons-results-2026-10-10"></a>

## 원본: phase9-07-publish-block-reasons-results-2026-10-10.md

# Phase 9-07 결과: 게시가 막힌 항목과 이유를 한 번에 보이기 (2026-10-10)

지침: [phase9-07-publish-block-reasons.md](development-history.md#prompts-phase9-07-publish-block-reasons) · 앞 단계: [9-06](experiment-results.md#phase9-06-numeric-columns-results-2026-10-10)

> 사용자 결정(2026-10-10): 추천대로 리허설 문제 6을 고친다.

## 1. 고친 것

- **서버(`plan_publish`)**: 모든 게시 검사를 끝까지 돌리고 막힌 항목마다 이유를 모은다.
  - 응답: 기존 `fields`는 그대로 두고 `reasons`를 더했다. 각 항목에 `field`, 한국어 `label`, `reason`이 있고, 같은 칸이면 `other`·`other_label`이 붙는다.
  - 이유 코드: `PENDING`, `COLUMN_MISSING`, `COLUMN_SHARED`, `NOTHING_MAPPED`, `DICTIONARY_UNUSED`, `DICTIONARY_STALE`, `PERIOD_TABLE_NEEDED`, `TIME_FORMAT`, `NUMBER_FORMAT`, `BLANK_VALUES`.
  - 런타임 검증 실패(항목 없는 거절)와 SQLite 경로의 응답은 그대로다.
- **화면**: `reasons`를 엄격히 읽고 항목마다 한 줄씩 이유와 할 일을 한국어로 보인다. 예:

  ```text
  게시하지 못했습니다. 아래 항목을 고친 뒤 다시 게시하세요.
  - 학점: 숫자 칸이 아닙니다(예: 3-2-2 같은 글자). 숫자 형식 칸에 연결하거나 '이 DB에 없음'으로 두세요.
  - 개설 연도: 학기와 같은 칸을 씁니다. 한 칸에는 한 항목만 연결할 수 있으니 하나를 '이 DB에 없음'으로 두세요.
  - 종료 시간: 시간 값이 09:00 꼴이 아닙니다(예: 09:00-10:15, 9:00, 빈 값). '이 DB에 없음'으로 두세요.
  - 시작 시간: 종료 시간과 같은 칸을 씁니다. 한 칸에는 한 항목만 연결할 수 있으니 하나를 '이 DB에 없음'으로 두세요.
  ```

  - 검토·값 사전·확장 항목 패널의 오류 칸은 줄바꿈을 보인다.

## 2. 확인

- 리허설 학교를 같은 관리자 순서로 등록하고 게시를 한 번 눌렀다. **막힌 항목 넷(학점, 개설 연도, 종료 시간, 시작 시간)이 이유와 함께 한 번에 나왔다.** 이전에는 같은 항목을 찾는 데 게시를 세 번 눌러야 했다.
- 화면(5174 → 8124)의 값 사전 패널 "사전 적용해 게시"에서도 위 메시지가 줄바꿈된 채 보였다(`white-space: pre-line`).
  - 앱 창이 최소화되어 클릭은 페이지 안 JavaScript로 했고, 결과는 화면 글자와 스크린샷으로 확인했다.
- 막힘이 없을 때의 게시와 기존 오류 문구는 그대로다(테스트).

## 3. 리뷰

`gpt-6-luna` high review-only(표준 입력을 닫고 단독 실행). 지적 3건을 코드로 확인해 모두 고쳤다.

1. **낡은 값 사전이 다른 막힘을 가렸다.** 값 사전 값이 DB에 없으면 계획 전에 바로 돌려주어, 검토 안 함·숫자 형식 같은 다른 이유가 보이지 않았다. 고침: 그 항목을 `DICTIONARY_STALE`로 계획에 넘겨 다른 이유와 함께 돌려준다. 이런 항목이 있으면 게시는 되지 않는다.
2. **화면이 모르는 이유 코드를 받아들였다.** 고침: 서버의 이유 목록에 있는 코드만 받는다. 두 목록이 어긋나지 않는지 서버 테스트로 확인한다(목록 일치, 코드마다 문장이 있음).
3. **화면이 빠지거나 겹친 이유 목록을 받아들였다.** 고침: 막힌 항목마다 이유가 정확히 하나여야 한다.

회귀 테스트를 더했다(낡은 값 사전 + 검토 안 함 + 숫자 형식이 한 번에 나옴, 모르는 코드·빠진 이유·겹친 이유 거부).

---

<a id="phase9-08-chat-model-choice-results-2026-10-12"></a>

## 원본: phase9-08-chat-model-choice-results-2026-10-12.md

# Phase 9-08 결과: 적응형 챗봇의 기본 모델을 일반 모델로 바꾸고, 쓸 모델을 고르게 하기 (2026-10-12)

지침: [phase9-08-chat-model-choice.md](development-history.md#prompts-phase9-08-chat-model-choice) · 앞 단계: [9-07](experiment-results.md#phase9-07-publish-block-reasons-results-2026-10-10)

> 사용자 결정(2026-10-12):
> - 특정 DB로 학습한 모델을 적응형 챗봇의 기본으로 쓰지 않는다. 규칙으로 덮지 말고 모델을 바꾼다.
> - 어떤 모델을 쓸지 사용자가 고를 수 있게 한다.

> 아래 수치는 개발자가 고정한 질문과 AI가 만든 개발용 DB 위의 잠정치다. L1/L3 전이며 모델 우열의 결론으로 쓰지 않는다.

## 1. 왜 바꿨나

- 리허설 학교에서 "컴융 전공 보여줘"가 `과목 코드 = 1101-7019`가 되었다. 이 코드는 리허설 학교에 없고, 레거시 DB에만 있다.
- "소웨 전공 보여줘"도 `과목 코드 = 1101-7001`이 되었다.
- 레거시로 미세 조정한 모델(`text2sql_qwen`)이 레거시 값을 외워서 낸 것이다(계획 문서 "위험 3 CNU 데이터 암기").

## 2. 만든 것

### 2.1 모델 선택지

| id | 화면 이름 | 모델 | 요청 방식 |
| --- | --- | --- | --- |
| `general`(기본) | 일반 모델 | `CHAT_MODEL`, 없으면 `qwen2.5-coder:7b-instruct` | model template, QueryIR JSON schema, prompt `query-ir-v3` |
| `legacy-tuned` | 레거시 학습 모델(비교용) | `OLLAMA_MODEL`(레거시 챗봇과 같은 값) | 이전과 같은 요청(`alpaca-raw-no-system-v1`, `query-ir-v1`, schema 없음) |

- 서버가 선택지를 정한다. 브라우저는 모델 이름이 아니라 id만 보낸다. 모르는 id는 `CHAT_INVALID_REQUEST`다.
- `GET /api/adaptive/v1/chat/models`가 선택지(id, 화면 이름, 모델 이름)를 알려 준다. 연결 주소·`.env` 값은 보내지 않는다.
- 질문 요청의 `model`은 선택이다. 없으면 기본 모델이다.
- 답 형식은 바꾸지 않았다. 모델 호출은 질문당 한 번이고, QueryIR 검증·compiler·policy·executor도 그대로다.
- 한 모델이 시작에 실패해도 다른 모델로 시작한다. `OLLAMA_MODEL`이 없어도 일반 모델로 챗봇이 켜진다.
- 레거시 챗봇(8000)은 바꾸지 않았다. 계속 `OLLAMA_MODEL`을 쓴다.
- 예전 명확화 흐름(SQLite 합성 DB)은 기본 모델을 쓴다.

### 2.2 일반 모델이 제대로 답하게 한 것

처음에는 일반 모델을 그대로 붙였다. 레거시 30문항 중 17문항이 `MODEL_INVALID_OUTPUT`이었다. 원인과 고침은 다음과 같다(모두 일반 모델에만 적용된다).

1. **JSON 모양.**
   - 원인: 일반 모델은 답을 Markdown 코드 블록으로 감싸거나, 힌트가 있으면 `{"항목": 값}` 같은 다른 모양을 냈다. 미세 조정 모델은 학습으로 QueryIR 모양을 익혔다.
   - 고침: Ollama structured output으로 QueryIR JSON schema를 함께 보낸다(`queryir-schema-v1`).
     - schema는 공통 항목 이름, 항목별 연산자, 항목별 값 형식(숫자 항목은 숫자)만 정한다.
     - 항목 목록은 이 DB에 없는 항목도 포함한다. 그래야 "2024년 강의"가 "개설 연도 정보 없음"으로 거절된다.
     - 조회 항목은 겹치지 않게 한다(QueryIR 규칙).
     - 조건은 12개까지만 받는다. 제한이 없으면 같은 조건을 되풀이하다 시간 초과(120초)가 났다. 질문이 말하는 조건(3~4개)보다 훨씬 많아 실제 질문은 잘리지 않는다.
   - 모양만 정한다. 검증은 그대로 서버가 한다.
2. **질문에 없는 조건.**
   - 원인: 일반 모델은 "2023년", "Spring", "정원 50 이하" 같은 조건을 스스로 더했다.
   - 고침: prompt `query-ir-v3`. v2에 두 문장을 더했다.
     - "질문이 말한 조건만 넣는다."
     - "값 힌트가 있으면 그 항목과 값을 쓴다."
3. **학과 이름이 분반 칸으로.**
   - 원인: 일반 모델은 학과를 `Course.department`(없는 항목)로 쓰려 했고, schema 안에서는 `CourseOffering.section`으로 갔다.
   - 고침: 8-13의 값 힌트 고침을 넓혔다. 모델이 코드·분반 같은 식별 항목에 넣은 값이 **값 힌트 하나와 정확히 같으면** 그 힌트의 항목으로 옮긴다.
     - 코드처럼 생긴 값, 힌트 이름의 일부, 힌트가 없는 값은 옮기지 않는다.
     - 레거시 학습 모델의 레거시 답 30개는 이 고침 전후로 같았다.

## 3. 다시 잰 결과 (테스트 포트 8124, 같은 질문, 모델만 바꿈)

### 3.1 레거시 비교 30문항 (`legacy-view`, 기준 SQL은 읽기 전용으로 실행)

| 결과 | 일반 모델 | 레거시 학습 모델 |
| --- | ---: | ---: |
| 정확히 맞음 | 18 | 17 |
| 맞는 부분 집합(200건 상한) | 5 | 5 |
| 일부만 맞음 | 0 | 1 |
| 틀린 행 | 2 | 2 |
| 답 없음 | 2 | 2 |
| "정보 없음"으로 거절(정답) | 3 | 3 |
| **맞음 합계** | **23** | **22** |

- 다른 문항은 하나뿐이다. "미적분학 강의 목록"에서 레거시 학습 모델은 `미적분학1`로 좁혀 68건(정답 72건)을 냈고, 일반 모델은 72건으로 맞았다.
- 레거시 학습 모델의 답 30개는 9-08 전 첫 측정과 모두 같았다.
- 레거시 확인 질문 "컴퓨터융합학부 전공 과목 보여줘"는 두 모델 모두 82건이다(화면에서도 확인).
- 일반 모델의 단계별 변화(같은 30문항, 맞음 합계): 그대로 붙임 10 → schema 19 → 항목 목록 넓힘·식별 항목 고침 20 → prompt v3 20 → 값 형식 schema 23 → 리뷰 반영(조건 상한 12, 조회 항목 중복 금지) 23.

### 3.2 리허설 학교 14문항 (사용자가 등록한 `rehearsal-school`, 재실행으로 기록)

| 실행 | 모델 | 결과 |
| --- | --- | --- |
| `run-004` | 일반 모델 | 6/14 |
| `run-005` | 레거시 학습 모델 | 6/14 |
| `run-006` | 일반 모델(리뷰 반영 후) | 6/14 |

- 두 모델은 같은 6문항을 맞혔다.
- 나머지는 등록 설정(값 사전 없음, 학점·시간 "이 DB에 없음") 때문에 둘 다 틀렸다.
- 이 등록은 사용자가 화면에서 한 것이라 9-06의 `run-003`(9/14, AI가 한 등록)과 비교할 수 없다.
- 틀리는 모양이 달랐다. "소웨 전공필수 과목"에서:
  - 레거시 학습 모델은 레거시 과목 코드 `1101-7001`을 지어냈다.
  - 일반 모델은 이수 구분에서 "소웨 전공필수"를 찾아 0건이었다.

### 3.3 `other_school` 12문항 (테스트 등록 `p908-other`: 관리자 정답 연결, 값 사전 15개, 확인 후 지움)

| 모델 | 결과 | 틀린 문항 |
| --- | --- | --- |
| 일반 모델 | 10/12 | q02 전전, q03 기공 전공선택 |
| 레거시 학습 모델 | 12/12 | 없음 |

- 일반 모델이 틀린 두 문항은 모두 **학과 약어**다. 값 힌트(전전 → 전자전기공학부)를 따르지 않았고, 질문에 없는 조건(2023년, Spring, 분반 A, 정원 범위, 3학년)을 더했다. q02는 같은 조건 6개를 두 번 되풀이해 상한 12개에서 멈췄다.
- 리뷰 반영 전에는 9/12였다(q01 소웨 전공필수도 틀림). 같은 모델·질문에서 한 문항이 바뀌었으니 이 정도 차이는 흔들림으로 본다.
- 이 DB에서는 레거시 학습 모델이 외운 값을 내지 않았고, 힌트를 잘 따랐다.

## 4. 해석 (잠정)

- 일반 모델로 바꾸니 레거시 값 암기(없는 과목 코드를 지어냄)는 사라졌다. 레거시 질문에서는 레거시 학습 모델과 같은 수준이었다(23 대 22).
- 하지만 다른 학교의 약어 질문에서는 일반 모델이 더 약했다(10 대 12). 약어 힌트를 따르는 힘과 조건을 지어내지 않는 힘은 미세 조정 모델이 낫다.
- 즉 "미세 조정 모델 = 레거시 전용"만은 아니다. 미세 조정은 한국어 수강 질문 → QueryIR의 모양과 습관도 가르쳤고, 그 대가로 레거시 값을 외웠다.
- 결정에 필요한 다음 근거:
  - 팀원 DB 평가를 두 모델로 각각 실행하는 것. 평가 도구가 `--model`을 받는다.
  - 다른 일반 모델(예: 코딩 전용이 아닌 `qwen2.5:7b-instruct`)은 설치와 내려받기가 필요해 시험하지 않았다.

## 5. 확인

- 서버 테스트(새 `tests/adaptive/test_chat_model_choice.py`)와 화면 테스트(`tests/frontend/phase9-08-chat-model-choice.test.ts`)를 더했다. 서버 테스트가 보는 것:
  - 선택지와 목록 API, `model` 필드
  - JSON schema 모양
  - prompt v3
  - 레거시 학습 모델 요청 불변
  - 시작 실패 시 다른 모델로 시작
  - 식별 항목 고침
  - 평가 도구의 모델 기록
- 화면(5174 → 8124):
  - 새 챗봇에 "모델" 고르기가 보이고 기본값은 "일반 모델 · qwen2.5-coder:7b-instruct"다.
  - 레거시 학습 모델로 바꾸면 대화에 "다음 질문부터 이 모델이 답합니다"가 나온다.
  - 두 모델 모두 "컴퓨터융합학부 전공 과목 보여줘"에 82건이었다.
  - 앱 창이 최소화되어 고르기와 질문은 페이지 조작으로 했고, 결과는 화면 글자로 확인했다.
- 원자료(질문별 답)는 Git이 추적하지 않는 `.schema-adaptive/reports/phase9-08/`에 있다.

## 6. 리뷰

`gpt-6-luna` high review-only(표준 입력을 닫고 단독 실행). 지적 5건을 하나씩 확인했다.

1. **[바꾸지 않음] 모델 이름이 화면 목록과 환경 점검 출력에 나온다.**
   - 의도한 것이다. 모델 이름은 비밀이 아니고, 어느 모델이 답하는지 사용자가 보고 고르는 것이 이번 결정의 목적이다.
   - 환경 점검은 9-03부터 `OLLAMA_MODEL`을 보여 왔다. 접속 정보(DSN)·주소·비밀번호는 여전히 나가지 않는다.
2. **[고침] 조건 6개 상한이 질문을 자를 수 있었다.**
   - 고침: 상한을 12개로 올렸다.
   - 상한을 아예 없애 보니 `other_school` q02에서 같은 조건을 되풀이하다 120초 시간 초과가 났다. 그래서 상한은 남겼다.
   - 질문이 말하는 조건(3~4개)보다 훨씬 많아 실제 질문은 잘리지 않는다. 13개 이상을 말하는 질문은 알려진 한계다.
3. **[고침] schema가 조회 항목 중복을 허용했다(QueryIR는 거부).** 고침: `uniqueItems`를 넣었다.
4. **[재현 안 됨] 화면의 모델 id 정규식이 끝 줄바꿈을 받는다는 지적.**
   - JavaScript `$`는 `m` 플래그가 없으면 끝 줄바꿈 앞에서 맞지 않는다. Node로 끝에 줄바꿈이 붙은 `general`이 거부되는 것을 확인했다.
   - 회귀 테스트만 더했다.
5. **[고침] 평가 도구가 모델 목록을 못 읽어도 모델 없이 진행했다.**
   - 고침: 목록 경로가 없는 예전 서버(404)만 모델 없이 진행한다.
   - 그 밖의 실패(500, 연결 실패, 모양이 다름)는 `MODELS_UNAVAILABLE`로 멈춘다.

고친 뒤 일반 모델을 다시 쟀다. 레거시 23, `other_school` 10/12, 리허설 `run-006` 6/14. 위 표는 이 값이다.

---

<a id="phase9-09-openai-chat-model-results-2026-10-12"></a>

## 원본: phase9-09-openai-chat-model-results-2026-10-12.md

# Phase 9-09 결과: 챗봇 모델 선택지에 ChatGPT API 추가 (2026-10-12)

지침: [phase9-09-openai-chat-model.md](development-history.md#prompts-phase9-09-openai-chat-model) · 앞 단계: [9-08](experiment-results.md#phase9-08-chat-model-choice-results-2026-10-12)

> 사용자 결정(2026-10-12):
> - 로컬 7B 모델만으로는 한계가 있어 ChatGPT API 선택지를 더한다. API 키는 사용자가 넣는다.
> - 기본 모델은 `gpt-5-mini`다.
> - 값 힌트도 보낸다.
> - "Legacy 기준선"(예전 레거시 챗봇)은 그대로 둔다.

## 1. 쓰는 법

1. `backend/.env`에 한 줄을 넣는다. 키는 이 파일에만 둔다. 이 파일은 Git이 추적하지 않는다.
   ```text
   OPENAI_API_KEY=sk-...
   ```
2. 다른 모델을 쓰려면 한 줄을 더 넣는다(선택).
   ```text
   OPENAI_MODEL=gpt-5
   ```
3. adaptive 서버를 다시 켠다(`npm run dev:all`).
   - 상태 줄에 `chat_model.openai=enabled`가 나오면 켜진 것이다.
   - 키 모양이 이상하면 `disabled (INVALID_CONFIG)`가 나온다. 키 값은 출력하지 않는다.
4. 챗봇 창에서 QueryIR 경로 DB(예: "Legacy DB (QueryIR)")를 고르고, "모델"에서 "ChatGPT(외부 API) · gpt-5-mini"를 고른다.
   - "Legacy 기준선"은 예전 레거시 챗봇이라 모델 고르기가 없다. 의도한 것이고, 사용자 결정으로 그대로 둔다.
5. 평가 도구에서도 고를 수 있다.
   ```bash
   python -m backend.adaptive.evaluation.blind_eval run <평가 ID> --model openai
   ```

## 2. 만든 것

- `backend/adaptive/harness/openai_chat.py`: `OpenAIQueryModel`.
  - 질문당 Chat Completions 한 번(`https://api.openai.com/v1/chat/completions`). 재시도는 없다.
  - prompt는 일반 모델과 같은 `query-ir-v3`이고, 값 힌트도 같다.
  - 9-08의 QueryIR JSON schema를 `response_format`으로 보낸다. strict는 아니다(schema의 일부 키워드를 strict가 받지 않는다).
  - `temperature`는 보내지 않는다. gpt-5 계열은 기본값만 받는다.
  - 고정 주소만 쓴다. proxy와 redirect는 따르지 않고, 응답은 1MB까지 읽는다. TLS 인증서는 Python 기본 검증을 쓴다.
  - 실패는 고정 코드만 남는다.
    - 시간 초과(60초)는 `TIMEOUT`이다.
    - HTTP 오류, 거절(refusal), 이상한 모양은 `PROVIDER_ERROR`이다.
    - 빈 답은 `EMPTY_RESPONSE`이다.
    - 화면에는 "모델 응답 시간이 초과되었습니다" 또는 "QueryIR 모델을 사용할 수 없습니다"로 보인다.
  - JSON이 아닌 글은 다른 모델처럼 "질문을 이해하지 못함"이 된다.
- 선택지 `openai`("ChatGPT(외부 API)").
  - `OPENAI_API_KEY`가 있을 때만 맨 뒤에 붙는다. 기본 모델은 그대로 일반 모델이다.
  - Ollama가 없어도 ChatGPT 선택지만으로 챗봇이 켜진다.
  - 예전 명확화 흐름(SQLite 합성 DB)은 계속 로컬 Ollama 모델을 쓴다.
- 환경 점검: `OPENAI_API_KEY`가 있는지만 보인다.
- 화면: 선택지 목록을 서버에서 받으므로 자동으로 나온다. 모델 실패 안내만 고쳤다("Ollama가 켜져 있는지, ChatGPT라면 API 키와 사용 한도를 확인").

## 3. 보내는 것과 안 보내는 것

| 보냄 | 보내지 않음 |
| --- | --- |
| 질문, 공통 항목 목록(이름·형식·연산자·이름표), 값 힌트(질문 단어와 맞는 DB 값: 학과명·교수명·이수 구분 등), QueryIR schema | 물리 table·column 이름, DSN, DB 행 데이터, API 키 외의 `.env` 값 |

- 값 힌트에는 질문에 나온 교수명 같은 레거시 DB 값이 들어갈 수 있다. 사용자가 이를 알고 보내기로 정했다.
- 데모 자료를 밖에 낼 때의 교수명 처리(L5)는 기존 규칙을 그대로 따른다.

## 4. 확인

- 서버 테스트(`tests/adaptive/test_openai_chat_model.py`)는 가짜 전송으로 본다.
  - 요청 모양(주소, 헤더, 모델, prompt v3, 값 힌트, schema, `temperature` 없음, 물리 이름 없음)
  - JSON 읽기와 실패 코드 8가지
  - 키가 오류·repr·목록·답·상태 줄 어디에도 나오지 않음
  - 고정 주소 밖은 거부
  - 키가 없거나 모양이 이상할 때의 동작
  - 화면 고르기로 고른 ChatGPT만 한 번 불림
  - 명확화 흐름은 로컬 모델
- 실제 호출은 하지 않았다. 비용이 들므로 키는 사용자가 넣고 화면에서 시험한다.

## 5. 리뷰

`gpt-6-luna` high review-only(표준 입력을 닫고 단독 실행). 지적 3건을 코드로 확인하고 모두 고쳤다. 회귀 테스트를 더했다.

1. **시작 설정에 담긴 API 키가 repr로 보였다.** 출력되는 곳은 없었지만, 키를 `_Secret`으로 감싸 repr·str에 `***`만 보이게 했다.
2. **`raise ... from None`이어도 원래 예외가 `__context__`에 남았다.** 전송 오류에 요청 내용이 실려 있으면 따라가서 볼 수 있었다. 이제 실패는 except 밖에서 새로 만들어, 원래 예외가 남지 않는다.
3. **`finish_reason`을 보지 않았다.** 잘린 답(`length`), 걸러진 답(`content_filter`), 도구 호출(`tool_calls`)도 실행될 수 있었다. 이제 `stop`만 답으로 받는다.

---

<a id="phase9-10-direct-sql-results-2026-10-12"></a>

## 원본: phase9-10-direct-sql-results-2026-10-12.md

# Phase 9-10 결과: 등록 PostgreSQL 직접 SQL과 QueryIR 비교 (2026-10-12)

지침: [phase9-10d-results-doc.md](experiment-results.md#prompts-phase9-10d-results-doc) · 설계: 계획 문서 ADR-100 · 앞 단계: [9-08](experiment-results.md#phase9-08-chat-model-choice-results-2026-10-12)

> 아래 수치와 실패 분류는 Claude가 기존 측정 기록과 실패 문항 재생성으로 확인해 제공한 것이다. 이번 작업은 문서 기록만 하며 새 측정·모델 호출·서버 조작·테스트·커밋은 하지 않았다.
>
> 개발 질문 56개는 QueryIR 규칙을 맞춘 질문이라 QueryIR에 유리하다. L1/L3 전 잠정치이며 적응성 근거가 아니다. 판정 근거는 에이전트가 미리 보지 않은 팀원 DB와 고정 질문의 첫 실행이다. 어떤 사람 관문도 승인하지 않았다.

## 1. 측정 조건

- 2026-10-12 사용자 승인으로 한 번 측정했다. 두 방식 모두 로컬 `qwen2.5-coder:7b-instruct`, temperature 0, 질문당 모델 호출 한 번이다. OpenAI 측정은 사용자 결정으로 제외했다.
- 테스트 서버는 8124였고 측정 뒤 종료했다. 사용자 서버는 건드리지 않았다.
- 직접 SQL(`direct-sql-pg-v1`)은 등록만 한 DB를 쓴다. 관리자 mapping·변환·값 사전은 주지 않았다. 등록 catalog, column별 예시 최대 10개/40자, 실제 값 힌트를 쓴다.
- QueryIR은 기존 profile 또는 게시 mapping·값 사전·compiler 경로를 쓴다. 두 방식은 온보딩 비용이 다른 조건이다.

| 질문 세트 | 직접 SQL 등록 | QueryIR 조건 |
| --- | --- | --- |
| 레거시 30 | public schema를 `p910-legacy`로 새로 등록(table 8·column 38·FK 12, `student`·`enrollment` 포함), 읽기만 | 9-08과 같은 `legacy-view` profile. 기준 SQL은 `p910-legacy` 연결로 실행(CLI에 `--reference-database` 추가) |
| `other_school` 12 | `p910-other`(table 6·column 30·FK 6) | 다시 돌리지 않고 9-08 결과 10/12 재사용. 같은 모델·질문이며 9-08 이후 QueryIR runtime 변경 없음. 9-08 기록에는 응답 시간이 없음 |
| 리허설 14 | 기존 `rehearsal-school` | 게시된 `rehearsal-school` |

새 등록 `p910-legacy`·`p910-other`는 `PENDING_REVIEW`로 남겼고 게시·관리자 작업은 하지 않았다.

두 방식은 같은 채점기를 쓴다. 거절은 `blind_eval`과 같은 규칙(`UNSUPPORTED`, `NOT_UNDERSTOOD`)으로 채점했다. 레거시 QueryIR 26은 9-08의 23(정확 18 + 상한 부분 집합 5)에 거절 정답 3을 더한 것과 같다. 9-08의 23에서 정확도가 3문항 개선되었다는 뜻은 아니다.

## 2. 결과

| 질문 세트 | 직접 SQL 맞음 | QueryIR 맞음 |
| --- | ---: | ---: |
| 레거시 30 | 10 | 26 |
| `other_school` 12 | 5 | 10 (9-08) |
| 리허설 14 | 2 | 6 |
| 합 56 | 17 | 42 |

| 지표 | 직접 SQL (레거시 / other / 리허설) | QueryIR (레거시 / 리허설) |
| --- | --- | --- |
| 조용한 오답 | 2 / 1 / 10 | 2 / 3 (other 9-08: 틀린 2문항) |
| 실행 실패 | 15 / 0 / 0 (모호한 column 등 11, `course_code` 별칭 없음 4) | 0 / 0 |
| 정책 거절·생성 실패 | 0 | 0 |
| 거절이 맞음 / 답할 수 있는데 거절 | 0/3, 0/6, 1/2 | 3/2, 1/5 |
| 상한(200행) 도달 | 0 | 5 / 0 |
| 지연 중앙값 / p90 (ms) | 674/1104, 514/880, 508/844 | 1469/2030, 1231/1494 |

QueryIR `other_school`의 지연은 9-08 기록에 없어 비교하지 않는다. QueryIR 지연은 HTTP 왕복·JSON 출력·compile을 포함한다.

## 3. 직접 SQL 실패 원인

직접 SQL 오답은 39건(56 − 17)이다. 아래 분류는 실패 문항만 같은 조건으로 다시 생성해 SQL을 확인한 것이다. temperature 0이지만 재생성이 완전히 같지는 않았다(`other_school` q12는 기록은 거절, 재생성은 SQL). 그래서 분류별 개수는 대략적인 값이며 원래 측정의 채점 결과를 대체하지 않는다.

1. **SQL 작성 실수 약 11건.** JOIN 뒤 table 없이 쓴 `subject_code`가 모호한 경우 9건(레거시 q01·q14·q15·q17·q18·q23·q24·q29·q30, 조건은 대부분 맞음), 없는 column 2건(q06 `c.category`, q20 `cnu_courses.subject_name`).
2. **지시 미준수 4건.** 과목 코드를 `AS course_code`로 내지 않음(레거시 q02·q04·q13·q16, 조건은 그럴듯함).
3. **과목 코드 식별 오인 약 9건.** 리허설에서 `lecture_id`(정수 PK)를 과목 코드로 냄(r02~r07·r09·r10·r12). 조건은 대부분 맞음(예: r09 `cap >= 60`, r10 `grade_year = 1`). 이 DB의 과목 코드는 분반이 붙은 `lec_no`에서 나오며, QueryIR은 관리자 mapping·변환이 이를 알려 준다. “과목 코드가 무엇인지”는 DB별 지식이다.
4. **값 위치·형식 못 찾음 약 5건.** 과목명 질문. 레거시 q07(데이터베이스)·q08(딥러닝)·q21(컴퓨터프로그래밍)은 거절, other q08·리허설 r08은 `과목명 = '데이터베이스'` 정확 일치. 과목명 column은 서로 다른 값이 500개를 넘거나 값 힌트에 잡히지 않는다(값 힌트는 값 500개 이하 글자 column만 색인).
5. **약어·코드 의미(값 사전 없음) 6건.** other q01(소웨)·q02(전전)·q03(기공)·q06(2학기 = `hakgi_cd` 20), 리허설 r01(소웨)·r14(전전). 설계상 직접 SQL에는 관리자 값 사전을 주지 않았다.
6. **의미 오해 약 3~4건.** 레거시 q03(학점 `3/0/0` 정확 일치), q28(대상 학년 `course_year`를 개설 연도로 씀), other q12(1학년 대상을 학기 코드로).

## 4. 해석 (잠정, 개발용)

- 등록만 한 직접 SQL + 7B는 17/56, QueryIR은 42/56이다. 이 질문은 QueryIR 규칙을 맞춘 개발 질문이며 L1/L3 전 잠정치다. 적응성 판정은 팀원 DB 첫 실행으로 한다.
- 직접 SQL 실패의 절반 이상(위 1·2·3, 약 24건)은 SQL 작성 습관과 과목 코드 식별이다. 4는 사용자가 제안한 “DB 값 전체 검색” 아이디어가 겨냥하는 경우다. 5는 값 사전 같은 DB별 지식이 필요하다.
- 응답 시간을 기록한 레거시·리허설에서 직접 SQL은 약 2배 빠르다. QueryIR 지연은 HTTP 왕복·JSON 출력·compile을 포함하므로 모델 생성 시간만의 비교는 아니다.
- 이 실패 사례로 직접 SQL을 고친 뒤 같은 56개로 다시 재면 그 수치는 개발 질문에 맞춘 것이다. 개선 여부는 팀원 DB로 판단한다.
- 제품 구조 변경은 사용자 결정 사항이다. 이번 결과로 어떤 결정도 내리지 않았다. Legacy 기준선은 유지한다.

## 5. 측정 기록

원자료는 Git 밖의 private 경로에 있다. 이 문서에는 제공된 요약과 실패 분류만 기록한다.

| 경로 | 방식·질문 세트 |
| --- | --- |
| `.schema-adaptive/direct-sql-pg/run-001` | 직접 SQL·레거시 |
| `.schema-adaptive/direct-sql-pg/run-002` | 직접 SQL·`other_school` |
| `.schema-adaptive/direct-sql-pg/run-003` | QueryIR·레거시 |
| `.schema-adaptive/blind-evals/rehearsal-v1/run-007` | 직접 SQL·리허설 |
| `.schema-adaptive/blind-evals/rehearsal-v1/run-008` | QueryIR·리허설 |

`other_school` QueryIR은 [9-08 결과 문서](experiment-results.md#phase9-08-chat-model-choice-results-2026-10-12) §3.3의 10/12를 재사용한다.

---

<a id="phase9-16-retire-offering-year-results-2026-10-13"></a>

## 원본: phase9-16-retire-offering-year-results-2026-10-13.md

# Phase 9-16 결과 — 개설 연도 사용 중단 (2026-10-13)

사용자 지침 `docs/development-history.md#prompts-phase9-16-retire-offering-year`를 기존 미커밋 작업 위에 구현했다. 공통 ontology v3의 `CourseOffering.year`는 정의를 보존하고 사용 중단한다(ADR-101, 사용자 결정). 학기와 레거시 기준선은 유지한다. 버전 1·2 gold·fixture·dataset profile·기존 게시 파일 및 gold `review_status`는 수정하지 않았다.

현재 등록·규칙/AI 제안·검토 표·게시·값 사전/확장 대상·모델 prompt·QueryIR schema·값 힌트에서는 year가 빠진다. 기존 v2 mapping에 연결되어 있어도 현재 runtime은 unsupported다. 연도 질문은 `UNSUPPORTED`, 질문에 없는데 모델이 year를 생성하면 `UNKNOWN_FIELD`로 거절한다. 과거 draft의 required/pending year는 게시를 막지 않는다. SQLite 강의 목록은 year를 제외한 14개 항목을 사용하며, 기존 catalog DTO의 year key는 null로 유지한다.

## 검증

- backend 관련 18개 파일: 최종 확인한 **481 passed**, 22 subtests. 전체 pytest 없음.
  - ontology·mapping·chat·hint·fake AI·gold·fixture runner·variant benchmark 12개 파일: 305 passed, 22 subtests.
  - PostgreSQL 등록·SQLite 게시 2개 파일: 51 passed.
  - 확장·값 사전·PostgreSQL runtime·legacy adaptive profile 4개 파일: 124 passed; 마지막 adaptive smoke의 기존 `MAPPING_MISSING` 기대를 현재 `UNKNOWN_FIELD`로 갱신한 뒤 해당 테스트만 재실행하여 1 passed. 레거시 챗봇 구현은 변경하지 않았다.
- frontend 관련 5개 파일: **39 passed**. `npx tsc -b` 통과.
- `git diff --check` 통과. 변경 파일은 UTF-8.
- fake model/reader/executor/transport 및 임시 디렉터리만 사용했다. 테스트 프로세스에만 `.venv/phase916-temp` 아래 tempfile·mkdir mode=0777을 적용하고 plugin 자동 로딩/cacheprovider를 끄는 기존 Windows sandbox 우회를 사용했다. 등록 첫 실행에서 일시적인 500이 발생했다. 기존 Windows 임시 파일 접근 문제를 고려해 임시 경로 안의 rename PermissionError만 최대 20회/50ms 재시도한 최종 51개 실행은 통과했고, 재시도 3회를 관측했다. 첫 500과 해당 rename 오류의 동일 원인 여부는 별도 확정하지 않았다. 제품 권한 처리는 변경하지 않았다.
- 사용자 서버(8000·8123·5173), 사용자 `.schema-adaptive` 데이터, legacy DB를 조작하지 않았다. live 모델 호출·commit·`backend/.env` 값 출력 없음.
- AGENTS.md의 Luna/opencode 독립 리뷰는 사용자 no-live-model 지침과 충돌하므로 실행하지 않고 대기로 기록했다. 팀 공유 필요, 사람 관문 승인 없음.

## 기대 결과가 달라지는 평가 문항 — 기록 재현과 현재 챗봇 구분 (9-24 보완)

오프라인 benchmark·oracle·gold/녹화 replay·capture targets는 현재 runtime 버전이 아니라 기록이 선언한 `ontology.at_version(v)`로 재현한다. 선언이 없는 과거 기록은 v2로 본다. 금번 fixture gold는 v1을 선언하므로 연도 문항도 그대로 실행되며 core oracle 120건, extended oracle 200건의 기대값을 보존한다. 실제 챗봇 HTTP를 호출하는 `blind_eval` 등은 현재 v3 그대로이며 연도 질문을 거절한다. gold·fixture·녹화·review_status를 바꾸지 않는다.

| 평가 세트 | 문항 | 연도 사용 | 오프라인 기록 재현 | 현재 챗봇 v3로 다시 측정 |
|---|---|---|---|---|
| `fixture_v1_cases.yaml` | P2-009 | 2026년 조건 및 year projection | v1 기존 정답 재현 | 정보 없음/거절 |
| `fixture_v1_cases.yaml` | P2-028 | 2026년 이후 조건 및 year projection(원래 0건) | v1 기존 0건 재현 | 정보 없음/거절 |
| `fixture_v1_independent_cases.yaml` | P2-049 | 2026년·1학기·교양 조건 | v1 기존 정답 재현 | 연도 조건 때문에 거절 |
| `fixture_v1_independent_cases.yaml` | P2-063 | 2026년·정원 35 조건 | v1 기존 정답 재현 | 연도 조건 때문에 거절 |
| synthetic variant benchmark | v09 | year=2025 조건 및 year projection | profile의 v1로 기존 결과 재현 | 정보 없음/거절 |
| synthetic variant benchmark | v19 | year=2026 및 semester=1 조건 | profile의 v1로 기존 결과 재현 | 정보 없음/거절 |

9-16 당시 현재 v3를 넣은 fixture runner의 38/40·variant `FAILED_UNKNOWN_FIELD` 기록은 당시 구현의 관측값이다. 9-24부터 오프라인 평가 정책을 위와 같이 바로잡았다. variant `grade-trap`처럼 원래 year 미지원인 경우는 기록 버전에서 `MAPPING_MISSING` 거절을 기존 scorer로 채점한다. 버전 선언 없는 variant 규칙/AI 매핑 benchmark와 학교형 매핑 check는 v2로 재현한다. 새 제품 등록·검토는 v3다. 이 검증은 oracle/fake 모델 기반이며 모델 성능 측정이나 사람 관문 승인이 아니다.

`legacy_chat_compare` q28 “2024년에 개설된 강의 알려줘”는 원래 `UNSUPPORTED_FIELD`이며 기대 결과 변화가 없다. 현재 adaptive 챗봇의 연도 거절과 모델 생성 year의 `UNKNOWN_FIELD`는 기존 v3 회귀 테스트로 유지한다.

## 이번 단계에서 변경한 파일

기존 미커밋 파일의 다른 변경은 보존했다. 아래는 이번 단계에서 실제로 편집하거나 추가한 파일만이다.

```text
backend/adaptive/ontology/contracts.py
backend/adaptive/ontology/course-domain.yaml
backend/adaptive/mapping/rules.py
backend/adaptive/mapping/ai_proposal.py
backend/adaptive/mapping/review.py
backend/adaptive/api/admin_test_queries.py
backend/adaptive/api/chat_query.py
backend/adaptive/api/legacy_view_profile.py
backend/adaptive/api/postgres_registration.py
backend/adaptive/api/postgres_runtime.py
backend/adaptive/api/review_publish.py
backend/adaptive/api/synthetic_registry.py
backend/adaptive/api/value_hints.py
backend/adaptive/registration/postgres_dictionaries.py
backend/adaptive/registration/postgres_extensions.py
backend/adaptive/registration/postgres_registrations.py
backend/adaptive/registration/reviews.py
backend/adaptive/evaluation/reference.py
backend/adaptive/evaluation/runner.py
backend/adaptive/evaluation/validator.py
src/adaptive-prototype/MappingReviewPanel.tsx
src/adaptive-prototype/reviewFieldName.ts
src/adaptive-prototype/testQueries.ts
src/adaptive-prototype/databases.ts
tests/adaptive/test_retire_offering_year.py
tests/adaptive/test_target_grade.py
tests/adaptive/test_rule_based_mapping.py
tests/adaptive/test_postgres_registration.py
tests/adaptive/test_private_mapping_publish.py
tests/adaptive/test_evaluation_runner.py
tests/adaptive/test_variant_benchmark.py
tests/adaptive/test_legacy_view_partial_mapping.py
tests/frontend/phase6-16-databases.test.ts
docs/schema-adaptive-text-to-sql-plan.md
docs/human-validation-checkpoints.md
docs/next-session-handoff.md
docs/phase9-16-retire-offering-year-results-2026-10-13.md
```

## 9-24 전체 테스트 정리

### 분류 기준과 집계

1 = 과거 기록 버전 재현, 2 = ADR-101에 따른 현재 v3 기대 갱신, 3 = 정상 경로를 깨뜨린 회귀/충돌. 제품 runtime·HTTP 평가·레거시 기준선은 유지했다.

프롬프트의 12개 파일별 건수 합은 26이다. 마지막 전체 pytest에서 목록에 없던 `test_multi_schema_fixtures.py`의 flat/normalized/legacy 3개 subtest가 추가로 `KeyError: CourseOffering.year`를 재현했다. 이 셋을 더한 원래 v3 정리 대상 **29건은 1: 13 / 2: 9 / 3: 7**이다. 실제 13개 파일/26개 테스트 함수·파라미터 사례 + 3개 subtest로 구분한다. 원래 기대 배열·gold·fixture 정의를 바꾸지 않고 해당 테스트의 임시 bundle 생성에만 과거 v2 ontology를 지정했다.

12파일 초기 재현은 **26 failed, 193 passed, 1 skipped, 3 setup errors, 28 subtests passed**(336.39초)였다. 일부 원래 실패는 업로드 setup 500에 가려졌고 `test_fresh_init_draft_can_be_replaced_but_reviewed_one_cannot`, `test_damaged_record_is_hidden_and_does_not_affect_execution`, `test_base_version_change_between_compare_and_copy_conflicts`에서 추가 임시 업로드 오류를 관측했다. 이 실행 환경 관측은 위 29건 집계에 넣지 않는다. 후속 targeted/전체 실행의 검토 준비 500도 최종 검증 절에 별도로 기록한다.

모든 테스트 이름은 `tests/adaptive/` 기준이다. 수정 위치의 backend 경로는 `backend/adaptive/` 기준이다.

| 테스트 이름 | 분류 | 한 줄 이유 | 고친 곳/처리 |
|---|---:|---|---|
| `test_admin_test_queries.py::test_every_query_matches_seed_on_three_schemas` | 2 | v1 fixture의 year REMAP을 현재 v3 검토로 보내던 준비 코드 제거 | `test_admin_test_queries.py::_publish_fixture` |
| `test_arm_live_cli.py::test_live_run_with_oracle_transport_writes_three_reports` | 1 | 선언된 gold v1로 두 평가 arm을 재현해 120건 유지 | `evaluation/runner.py, direct_sql.py` |
| `test_arm_live_cli.py::test_no_system_format_applies_to_both_arms` | 1 | 선언된 gold v1로 두 평가 arm을 재현해 120건 유지 | `evaluation/runner.py, direct_sql.py` |
| `test_arm_live_cli.py::test_ir_extract_rule_controls_fenced_ir[single-fence-v1-120]` | 1 | 선언된 gold v1로 두 평가 arm을 재현해 120건 유지 | `evaluation/runner.py, direct_sql.py` |
| `test_capture_targets.py::test_targets_match_runner_context_and_exclude_gold` | 1 | capture의 모델 문맥도 gold의 v1 항목 집합을 사용 | `evaluation/capture_targets.py, test_capture_targets.py` |
| `test_direct_sql_arm.py::test_oracle_sql_arm_is_self_consistent_and_pairs_with_ir_oracle` | 1 | IR/SQL oracle의 gold 버전을 일치시켜 120쌍 재현 | `evaluation/runner.py, direct_sql.py` |
| `test_mapping_carry_over.py::test_same_schema_carries_every_field_and_publishes` | 2 | v3 carried 항목은 year를 뺀 14개 | `test_mapping_carry_over.py` |
| `test_mapping_carry_over.py::test_renamed_column_leaves_only_that_field_stale` | 2 | v3에서 이름 변경까지 제외하면 REMAPPED 13개 | `test_mapping_carry_over.py` |
| `test_mapping_carry_over.py::test_structure_change_marks_every_field_stale` | 2 | v3 stale 항목은 year를 뺀 15개 | `test_mapping_carry_over.py` |
| `test_mapping_carry_over.py::test_a_version_one_base_carries_over_to_a_version_two_target` | 2 | 현재 target은 v3이며 retired year를 옮기지 않음; 테스트 이름 갱신 | `test_mapping_carry_over.py` |
| `test_postgres_live_smoke.py::CleanupAndRedactionTest::test_success_cleanup_once_and_safe_summary` | 3 | smoke가 retired year 후보를 필수로 요구해 정상 경로가 실패 | `harness/postgres_live_smoke.py::SMOKE_EXPECTED_MAPPING` |
| `test_postgres_live_smoke.py::CatalogMappingQueryTest::test_mapping_top_candidates_and_no_publish` | 3 | retired year 후보 조회의 KeyError 제거 | `harness/postgres_live_smoke.py::SMOKE_EXPECTED_MAPPING` |
| `test_postgres_live_smoke.py::CatalogMappingQueryTest::test_query_mismatch_fails_closed` | 3 | 매핑 단계에서 잘못 중단되지 않아 QUERY_FAILED 검사 도달 | `harness/postgres_live_smoke.py::SMOKE_EXPECTED_MAPPING` |
| `test_raw_columns_and_select.py::test_an_unasked_missing_select_field_is_dropped_not_declined` | 2 | ADR-101의 모델 생성 year UNKNOWN_FIELD가 의도된 v3 동작; 일반 미지원 select 제거 검사는 유지 | `test_raw_columns_and_select.py` |
| `test_recorded_model.py::RecordedModelRunnerTests::test_runner_preserves_recorded_failures_missing_attempts_and_privacy` | 1 | gold v1로 녹화 mapping hash와 year 실행을 재현해 기존 37건 유지 | `evaluation/runner.py` |
| `test_schema_compare.py::test_same_schema_carries_over` | 2 | 현재 v3 field impact는 year를 뺀 14개 | `test_schema_compare.py` |
| `test_schema_compare.py::test_dropped_foreign_key_blocks_carry_over` | 2 | historical fixture의 year를 현재 검토에서 REMAP하지 않음; FK 차단 기대 유지 | `test_schema_compare.py::_publish_hybrid` |
| `test_schema_variation_suite.py::test_oracle_smoke_is_complete_on_all_five_variants` | 1 | gold v1의 extended 200건·40문항 정답/일관성을 fake transport로 재현 | `evaluation/runner.py, direct_sql.py` |
| `test_schema_variation_suite.py::test_schema_consistency_counts` | 1 | gold v1의 extended 200건·40문항 정답/일관성을 fake transport로 재현 | `evaluation/runner.py, direct_sql.py` |
| `test_schema_variation_suite.py::test_extended_live_run_with_oracle_transport[extra0-p523-ext-ir-synthetic-p519-latest.ext]` | 1 | gold v1의 extended 200건·40문항 정답/일관성을 fake transport로 재현 | `evaluation/runner.py, direct_sql.py` |
| `test_schema_variation_suite.py::test_extended_live_run_with_oracle_transport[extra1-p523-ext-nosys-fence-ir-synthetic-p519-latest.nosys.fence.ext]` | 1 | gold v1의 extended 200건·40문항 정답/일관성을 fake transport로 재현 | `evaluation/runner.py, direct_sql.py` |
| `test_synthetic_active_database.py::test_registry_lists_two_databases_strict` | 2 | 현재 catalog projection에서 year가 빠지는 것이 의도된 v3 동작 | `test_synthetic_active_database.py` |
| `test_upload_ai_proposal.py::test_ai_choice_is_first_with_rule_score_and_ai_evidence` | 3 | stale AI year key만 버리고 나머지 후보/점수/근거 유지; retired 항목 검사는 v3로 갱신 | `mapping/ai_proposal.py, test_upload_ai_proposal.py(첫 테스트의 retired 항목만)` |
| `test_upload_ai_proposal.py::test_none_and_reused_columns_keep_the_rule_candidates` | 3 | retired key 때문에 유효한 AI 전체 응답이 버려지던 회귀 | `mapping/ai_proposal.py, test_upload_ai_proposal.py(첫 테스트의 retired 항목만)` |
| `test_upload_ai_proposal.py::test_examples_reach_the_prompt_but_never_the_stored_files` | 3 | 유효한 AI 후보가 사라져 IndexError가 나던 회귀 | `mapping/ai_proposal.py, test_upload_ai_proposal.py(첫 테스트의 retired 항목만)` |
| `test_upload_ai_proposal.py::test_ai_proposal_upload_review_publish_and_catalog` | 3 | retired key만 제거해 정상 경로의 AI_UNAVAILABLE 회귀 해결 | `mapping/ai_proposal.py, test_upload_ai_proposal.py(첫 테스트의 retired 항목만)` |
| `test_multi_schema_fixtures.py::MultiSchemaFixtureTest::test_aligner_finds_all_expected_semantic_fields [variant=FLAT]` | 1 | 빠진 3개 subtest: 버전 선언 없는 과거 fixture 매핑 baseline은 v2로 생성/재현; 기존 year 기대 유지 | `test_multi_schema_fixtures.py::setUp/test_rebuild_is_repeatable (ontology 인자만)` |
| `test_multi_schema_fixtures.py::MultiSchemaFixtureTest::test_aligner_finds_all_expected_semantic_fields [variant=NORMALIZED]` | 1 | 빠진 3개 subtest: 버전 선언 없는 과거 fixture 매핑 baseline은 v2로 생성/재현; 기존 year 기대 유지 | `test_multi_schema_fixtures.py::setUp/test_rebuild_is_repeatable (ontology 인자만)` |
| `test_multi_schema_fixtures.py::MultiSchemaFixtureTest::test_aligner_finds_all_expected_semantic_fields [variant=LEGACY]` | 1 | 빠진 3개 subtest: 버전 선언 없는 과거 fixture 매핑 baseline은 v2로 생성/재현; 기존 year 기대 유지 | `test_multi_schema_fixtures.py::setUp/test_rebuild_is_repeatable (ontology 인자만)` |

추가 회귀: `test_offline_ontology_version.py`의 기록 버전/미선언 v2·fingerprint·잘못된 선언 거절·stale AI 값 제거/원본 불변/활성 schema 엄격성·학교형 benchmark의 fake adapter 검사, `test_v2_year_mapping_does_not_affect_v3_compare_or_carry_over`의 실제 v2 year 연결→year 물리 칸 이름 변경→v3 비교/이월/게시 및 base 파일 불변 검사. 기존 현재 v3 챗봇 연도 거절 테스트는 그대로 유지한다.

### 9-24에서 변경한 파일

이 목록은 작업 시작 시점의 미커밋 상태와 비교한 9-24 변경만이며, gold·fixture·녹화 파일은 없다.

```text
backend/adaptive/evaluation/ai_mapping.py
backend/adaptive/evaluation/capture_targets.py
backend/adaptive/evaluation/contracts.py
backend/adaptive/evaluation/direct_sql.py
backend/adaptive/evaluation/ontology_version.py
backend/adaptive/evaluation/runner.py
backend/adaptive/evaluation/school_mapping_check.py
backend/adaptive/evaluation/variant_benchmark.py
backend/adaptive/harness/postgres_live_smoke.py
backend/adaptive/mapping/ai_proposal.py
backend/adaptive/registration/reviews.py
docs/phase9-16-retire-offering-year-results-2026-10-13.md
docs/schema-adaptive-text-to-sql-plan.md
tests/adaptive/test_admin_test_queries.py
tests/adaptive/test_capture_targets.py
tests/adaptive/test_evaluation_runner.py
tests/adaptive/test_mapping_carry_over.py
tests/adaptive/test_multi_schema_fixtures.py
tests/adaptive/test_offline_ontology_version.py
tests/adaptive/test_raw_columns_and_select.py
tests/adaptive/test_schema_compare.py
tests/adaptive/test_synthetic_active_database.py
tests/adaptive/test_upload_ai_proposal.py
tests/adaptive/test_variant_benchmark.py
```

### 최종 검증 (2026-10-13)

- **전체 `pytest tests -q`: 2341 passed, 11 skipped, 9 failed, 608 subtests passed**, 2046.05초(34분 6초). 한 번 실행했다. 실패 9개는 뒤늦게 확인한 역사적 fixture 매핑 3 subtest와 검토 준비 단계의 `MAPPING_REVIEW_INTERNAL_ERROR` 6개였다.
- fixture 테스트는 기존 기대 배열과 데이터 정의를 유지하고 임시 bundle 생성/재생성의 ontology 인자만 v2로 지정했다. 이후 **그 전체 테스트 파일 + 위 6개 정확한 실패 사례: 10 passed, 12 subtests passed**, 67.43초. 제품 코드·기대값을 추가로 바꾸지 않은 6개 검토 준비 오류는 재실행에서 재현되지 않아 원인을 확정하지 않았다. 전체 실행 자체가 한 번에 green이었다고 보고하지 않는다. 12개 지정 파일·추가 fixture 파일·9-16/9-23 관련 backend 테스트는 전체 실행 및 정확한 실패 재실행을 합쳐 모두 통과를 확인했다.
- 전체 실행에서 재시도한 6개: `test_dictionary_import.py::test_invalid_imports_are_refused[overrides5]`, `test_mapping_version_rollback.py::test_interrupted_activation_is_repaired_by_republish`, `test_postgres_registration.py::test_register_review_publish_and_serve_without_restart`, `test_schema_compare.py::test_invalid_requests_fail_closed`, `test_value_dictionary_editor.py::test_editable_fields_and_values`, `test_value_dictionary_editor.py::test_a_too_short_alias_is_not_reported_as_a_collision`.
- 선행 focused 검증: replay·variant benchmark·smoke·AI·capture 5개 파일 **121 passed, 17 subtests passed**; 등록/비교/시험 질의/current runtime/신규 회귀 6개 파일 **70 passed, 2 temporary upload failures**(해당 사례는 뒤 전체 실행에서 통과); 독립 신규 버전/AI 회귀 파일 **17 passed**. 기존 비교 함수를 파일 수정 없이 별도 테스트 프로세스 메모리에만 로드했을 때 새 v2→v3 year 회귀는 `carry_over_executable=False`로 실패하고 수정된 함수로는 통과했다.
- 관련 frontend 7개 파일 **56 passed**; 로컬 TypeScript `tsc -b` 통과. UTF-8 strict decode와 `git diff --check` 통과.
- pytest plugin 자동 로딩/cacheprovider를 끄고, 테스트 프로세스에만 workspace의 git-ignored `.venv` 아래 `--basetemp`·`tempfile.tempdir`·그 아래 mkdir mode=0777을 적용했다. 기존 Windows 임시 파일 문제 대응으로 테스트 임시 경로 안의 실제 `os.rename` PermissionError만 최대 20회/50ms 재시도했다(전체 실행 7회, 최종 재실행 1회 관측). 최종 재실행은 더 짧은 `.venv/p924-<pid>` 경로를 사용했다. 제품 파일 권한/저장 정책은 이 문제 때문에 바꾸지 않았다. 검증 뒤 이번 작업의 basetemp·진단/보조 파일을 모두 삭제했다.
- 사용자 서버·사용자 `.schema-adaptive` 데이터·레거시 기준선·기존 미커밋 변경을 유지했다. gold·fixture 데이터/정의·녹화·gold review_status 수정, live 모델 호출, commit, `backend/.env` 값 출력 없음. fixture **테스트 소스의 ontology 인자** 변경과 원본 fixture 데이터/정의 변경을 구분한다. AGENTS.md의 Luna/opencode 독립 리뷰는 사용자 no-live-model 제약 때문에 실행하지 않았고 대기다. 사람 관문 승인 없음.

---

<a id="phase9-21-unmatched-word-notice-results-2026-10-13"></a>

## 원본: phase9-21-unmatched-word-notice-results-2026-10-13.md

# Phase 9-21: 연결되지 않은 낱말 알림 (2026-10-13)

사용자 지침: [phase9-21-unmatched-word-notice.md](development-history.md#prompts-phase9-21-unmatched-word-notice). 설계: 계획 문서 ADR-102.

적응형 성공 응답에 선택 필드 `unmatched_terms`(최대 5개)를 추가하고 결과 위에 짧은 값 사전 안내를 표시한다. 거절 답에는 추가하지 않고 빈 결과의 기존 `unmatched` 진단과 함께 표시할 수 있다. 조건 생성이나 추가 모델 호출·DB 조회는 없다. 레거시 챗봇과 Phase 9-20을 포함한 기존 미커밋 작업을 보존했다.

## 오프라인 검증 조건과 한계

- 사용자 서버·사용자 `.schema-adaptive` 데이터를 읽거나 변경하지 않았다. live 모델 호출·commit·`backend/.env` 값 출력 없음. 파일은 UTF-8.
- 과거 9-08/9-10 raw 응답이 사용자 private 데이터에 있으므로 열지 않았다. 대신 체크인된 `legacy-chat-compare-v1.json` reference SQL의 의미에 맞춘 고정 QueryIR 조건과 가짜 모델·executor를 사용했다. q12는 표현 불가능한 여석 조건을 빼고 학과 조건만 사용했다. q25·q26은 기존 outside-concept 거절, q28은 사용 중단 연도 거절, q29·q30은 학기 미지원 거절이다. q23·q24는 현재 지원하는 대상 학년 조건을 사용했다.
- 이 결과는 개발 질문 기준 잠정치다. 실제 모델이 같은 조건을 생성한다는 증거, 모델 성능 평가, 미관측 학교 평가 또는 L1/L3 승인으로 쓰지 않는다. 값 힌트에 연결된 단어도 이해함으로 보므로 누락 조건 전체를 찾는 기능은 아니다.
- 값 사전 없이 “컴융 전공 보여줘”에서 모델이 전공 조건만 반환하면 `['컴융']`을 알린다. 사전에 컴융 별칭이 있고 학과·전공 조건을 반환하면 알림이 없다. 한 번의 모델 호출과 기존 compiler·policy·executor 경로를 테스트로 확인했다.

## 30문항 고정 판정

테스트 `tests/adaptive/test_unmatched_terms.py`의 `LEGACY_CASES`에 조건·상태·알림 낱말을 문항별로 고정했다. 30개 중 1개에 알림, 나머지 29개는 알림 없음(이 중 5개는 거절 답). 정상 지원 24개는 잘못된 알림 0개다.

| ID | 질문 | 상태 | 알림 낱말 |
|---|---|---|---|
| q01 | 컴퓨터융합학부 전공 과목 보여줘 | 성공 | 없음 |
| q02 | 교양 과목 중에 금요일 수업 | 성공 | 없음 |
| q03 | 3학점 과목 알려줘 | 성공 | 없음 |
| q04 | 월요일 오전 수업 보여줘 | 성공 | 없음 |
| q05 | 수학과 강의 목록 | 성공 | 없음 |
| q06 | 전공핵심 과목 중 화요일 수업 | 성공 | 없음 |
| q07 | 데이터베이스 강의 알려줘 | 성공 | 없음 |
| q08 | 딥러닝 수업 있어? | 성공 | 없음 |
| q09 | 김소연 교수님 수업 | 성공 | 없음 |
| q10 | 이규철 교수 강의 보여줘 | 성공 | 없음 |
| q11 | 정원이 100명 넘는 강의 | 성공 | 없음 |
| q13 | 오후 3시에 시작하는 수업 | 성공 | 없음 |
| q14 | 2학점 교양 과목 | 성공 | 없음 |
| q15 | 경영학부 전공기초 과목 | 성공 | 없음 |
| q16 | 목요일 저녁 6시 이후 수업 | 성공 | 없음 |
| q17 | 영어영문학과 3학점 전공 과목 | 성공 | 없음 |
| q18 | 일반선택 과목 보여줘 | 성공 | 없음 |
| q19 | 토요일에 하는 수업 있어? | 성공 | 없음 |
| q20 | 미적분학 강의 목록 | 성공 | 없음 |
| q21 | 컴퓨터프로그래밍 수업 시간 알려줘 | 성공 | 없음 |
| q22 | 물리학과 월요일 수업 | 성공 | 없음 |
| q31 | 간호학과 전공 과목 중 수요일 오후 수업 | 성공 | 없음 |
| q12 | 아직 자리가 남은 컴퓨터융합학부 과목 | 성공 | 아직, 자리 |
| q23 | 컴퓨터융합학부 3학년 전공 과목 | 성공 | 없음 |
| q24 | 1학년이 듣는 교양 과목 | 성공 | 없음 |
| q25 | 온라인 콘텐츠 수업 보여줘 | 거절 | 없음 |
| q26 | 선수과목이 없는 컴퓨터융합학부 과목 | 거절 | 없음 |
| q28 | 2024년에 개설된 강의 알려줘 | 거절 | 없음 |
| q29 | 1학기 전공 과목 | 거절 | 없음 |
| q30 | 2학기에 열리는 교양 과목 | 거절 | 없음 |

q12의 “자리가 남은”은 기존 outside-concept 목록의 “남은자리”와 어순이 달라 기존 거절 규칙에 걸리지 않는다. 이 규칙은 바꾸지 않았다. 학과만 조회하는 가짜 응답에 연결되지 않은 “아직”, “자리”를 알린다. 조사 제거로 “남은”은 1글자 “남”이 되어 검사 대상에서 빠진다.

## 테스트와 리뷰

관련 backend 5개 파일의 최초 실행은 240 passed·3 failed였다. 실패는 새 테스트의 사전 조건 출력·조사 제거 결과·q12 거절 기대가 실제 기존 계약과 다른 경우로, 제품 규칙은 변경하지 않고 고정 출력과 기대를 바로잡았다. 새 테스트 파일 재실행 45 passed, 일반어 목록 고정·항목 별칭/연산/시간 규칙 검증 2개 추가 후 47 passed. 기존 관련 4개 파일 198개와 합쳐 총 245개 통과를 확인했다. frontend 관련 3개 파일 19 passed, `npx tsc -b`·대상 ESLint 통과.

Windows sandbox의 기본 pytest 실행·collect-only는 출력 없이 지연되어 해당 테스트 세션만 Ctrl-C로 종료했다. 테스트 프로세스에만 `.venv` 아래 새 임시 경로·mkdir mode=0777·pytest plugin 자동 로딩 및 cacheprovider 비활성화를 적용한 실행으로 검증했다. 제품 권한 코드는 변경하지 않았다.

`git diff --check`와 변경 8개 파일 UTF-8 decoding 확인도 통과했다. Luna/opencode 독립 리뷰는 사용자 no-live-model 제약과 충돌하여 실행하지 않고 대기로 기록한다. 사람 관문 승인 없음.

---

<a id="prompts-phase9-10d-results-doc"></a>

## 원본: prompts/phase9-10d-results-doc.md

# Phase 9-10 결과 기록 지침 (문서만)

코드는 바꾸지 않는다. 아래 측정 결과를 문서로 남긴다. 수치와 분류는 Claude가 측정 기록과 실패 문항 재생성으로 확인한 것이다. 새로 측정하거나 모델을 부르지 않는다. 커밋하지 않는다.

## 만들 파일

`docs/experiment-results.md#phase9-10-direct-sql-results-2026-10-12` (기존 결과 문서 문체를 따른다).

## 측정 조건 (2026-10-12, 사용자 승인, 한 번)

- 모델: 두 방식 모두 로컬 `qwen2.5-coder:7b-instruct`, temperature 0, 질문당 호출 한 번. OpenAI 측정은 사용자 결정으로 제외.
- 테스트 서버 8124(측정 뒤 종료). 사용자 서버는 건드리지 않음.
- 직접 SQL(`direct-sql-pg-v1`): 등록만 한 DB. 레거시는 public schema를 `p910-legacy`로 새로 등록(table 8·column 38·FK 12, `student`·`enrollment` 포함, 읽기만), `other_school`은 `p910-other`(table 6·column 30·FK 6), 리허설은 기존 `rehearsal-school`. 게시·관리자 작업 없음.
- QueryIR: 레거시는 9-08과 같은 `legacy-view` profile(기준 SQL은 `p910-legacy` 연결로 실행, CLI에 `--reference-database` 추가). 리허설은 게시된 `rehearsal-school`. `other_school` QueryIR은 다시 돌리지 않고 9-08 결과(10/12, 같은 모델·질문, 9-08 이후 QueryIR runtime 변경 없음)를 쓴다. 9-08 기록에는 응답 시간이 없다.
- 채점: 두 방식 같은 채점기. 거절은 `blind_eval`과 같은 규칙(`UNSUPPORTED`, `NOT_UNDERSTOOD`). 레거시 QueryIR 26은 9-08의 23(정확 18 + 상한 부분 집합 5)에 거절 정답 3을 더한 것과 같다.
- 기록: `.schema-adaptive/direct-sql-pg/run-001`(직접·레거시), `run-002`(직접·other_school), `run-003`(QueryIR·레거시), `.schema-adaptive/blind-evals/rehearsal-v1/run-007`(직접), `run-008`(QueryIR).

## 결과

| 질문 세트 | 직접 SQL 맞음 | QueryIR 맞음 |
| --- | ---: | ---: |
| 레거시 30 | 10 | 26 |
| `other_school` 12 | 5 | 10 (9-08) |
| 리허설 14 | 2 | 6 |
| 합 56 | 17 | 42 |

| 지표 | 직접 SQL (레거시 / other / 리허설) | QueryIR (레거시 / 리허설) |
| --- | --- | --- |
| 조용한 오답 | 2 / 1 / 10 | 2 / 3 (other 9-08: 틀린 2문항) |
| 실행 실패 | 15 / 0 / 0 (모호한 column 등 11, `course_code` 별칭 없음 4) | 0 / 0 |
| 정책 거절·생성 실패 | 0 | 0 |
| 거절이 맞음 / 답할 수 있는데 거절 | 0/3, 0/6, 1/2 | 3/2, 1/5 |
| 상한(200행) 도달 | 0 | 5 / 0 |
| 지연 중앙값 / p90 (ms) | 674/1104, 514/880, 508/844 | 1469/2030, 1231/1494 |

## 직접 SQL 실패 원인 (약 39건, 실패 문항만 같은 조건으로 다시 생성해 SQL 확인)

temperature 0이지만 재생성이 완전히 같지는 않았다(other_school q12는 기록은 거절, 재생성은 SQL). 그래서 개수는 "약"으로 적는다.

1. SQL 작성 실수 약 11: JOIN 뒤 table 없이 쓴 `subject_code`가 모호함 9(레거시 q01·q14·q15·q17·q18·q23·q24·q29·q30, 조건은 대부분 맞음), 없는 column 2(q06 `c.category`, q20 `cnu_courses.subject_name`).
2. 지시 미준수 4: 과목 코드를 `AS course_code`로 내지 않음(레거시 q02·q04·q13·q16, 조건은 그럴듯함).
3. 과목 코드 식별 오인 약 9: 리허설에서 `lecture_id`(정수 PK)를 과목 코드로 냄(r02~r07·r09·r10·r12). 조건은 대부분 맞음(예: r09 `cap >= 60`, r10 `grade_year = 1`). 이 DB의 과목 코드는 분반이 붙은 `lec_no`에서 나오며, QueryIR은 관리자 mapping·변환이 이를 알려 준다. "과목 코드가 무엇인지"는 DB별 지식이다.
4. 값 위치·형식 못 찾음 약 5: 과목명 질문. 레거시 q07(데이터베이스)·q08(딥러닝)·q21(컴퓨터프로그래밍)은 거절, other q08·리허설 r08은 `과목명 = '데이터베이스'` 정확 일치. 과목명 column은 서로 다른 값이 500개를 넘거나 값 힌트에 잡히지 않는다(값 힌트는 값 500개 이하 글자 column만 색인).
5. 약어·코드 의미(값 사전 없음) 6: other q01(소웨)·q02(전전)·q03(기공)·q06(2학기 = `hakgi_cd` 20), 리허설 r01(소웨)·r14(전전). 설계상 직접 SQL에는 관리자 값 사전을 주지 않았다.
6. 의미 오해 약 3~4: 레거시 q03(학점 `3/0/0` 정확 일치), q28(대상 학년 `course_year`를 개설 연도로 씀), other q12(1학년 대상을 학기 코드로).

## 해석 (잠정, 개발용)

- 이 개발 질문(56개)은 QueryIR 규칙을 맞춘 질문이라 QueryIR에 유리하다. L1/L3 전 잠정치이며 적응성 근거가 아니다. 판정은 팀원 DB 첫 실행이다.
- 등록만 한 직접 SQL + 7B는 17/56, QueryIR은 42/56이다.
- 직접 SQL 실패의 절반 이상(1·2·3, 약 24건)은 SQL 작성 습관과 과목 코드 식별이다. 4는 사용자가 제안한 "DB 값 전체 검색" 아이디어가 겨냥하는 경우다. 5는 값 사전 같은 DB별 지식이 필요하다.
- 직접 SQL은 응답이 약 2배 빠르다(QueryIR은 HTTP 왕복·JSON 출력·compile 포함).
- 이 실패 사례로 직접 SQL을 고친 뒤 같은 56개로 다시 재면 그 수치는 개발 질문에 맞춘 것이다. 개선 여부는 팀원 DB로 판단한다.
- 제품 구조 변경은 사용자 결정 사항이다. 이번 결과로 어떤 결정도 내리지 않았다.

## 함께 고칠 문서

- `docs/schema-adaptive-text-to-sql-plan.md`: ADR-100에 "결과" 항목(위 표 요약과 결과 문서 링크), 로드맵 9-10의 live 측정 항목 완료, 변경 이력에 2026-10-12 9-10 측정 항목, 15.1.1에 `--reference-database` 한 줄.
- `docs/human-validation-checkpoints.md` 9-10 메모에 결과 한 줄과 "개발 질문·L1/L3 전 잠정치, 관문 승인 없음".
- `docs/development-history.md#next-session-handoff` 9-10 줄에 결과 한 줄과 남은 등록 `p910-legacy`·`p910-other`(PENDING_REVIEW, 게시 안 함)를 적는다.
- 끝나면 바꾼 파일을 짧게 적는다.
