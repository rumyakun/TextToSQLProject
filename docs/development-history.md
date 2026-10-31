# 개발 지침·인수인계 통합 기록

> 2026-10-30 파일 정리: 아래는 작성 당시의 지침·관찰 기록이다. 현재 실행 안내는 [루트 README](../README.md), 설계·결정·현재 상태는 [프로젝트 계획](schema-adaptive-text-to-sql-plan.md)을 따른다. 과거 지침을 새 작업 지시나 현재 성능으로 해석하지 않는다.

## 기록 목차
- [desktop-test-handoff.md](#desktop-test-handoff)
- [midterm-review-2026-10-08.md](#midterm-review-2026-10-08)
- [next-session-handoff.md](#next-session-handoff)
- [prompts/phase1-01-jsonl-loader-quality-report.md](#prompts-phase1-01-jsonl-loader-quality-report)
- [prompts/phase1-02-sql-normalization-ast-analysis.md](#prompts-phase1-02-sql-normalization-ast-analysis)
- [prompts/phase1-03-sql-to-query-ir-conversion.md](#prompts-phase1-03-sql-to-query-ir-conversion)
- [prompts/phase1-04-identifier-pseudonymization.md](#prompts-phase1-04-identifier-pseudonymization)
- [prompts/phase2-01-common-evaluation-questions.md](#prompts-phase2-01-common-evaluation-questions)
- [prompts/phase5-01-lightweight-evaluation-runner.md](#prompts-phase5-01-lightweight-evaluation-runner)
- [prompts/phase5-02-recorded-model-replay-core.md](#prompts-phase5-02-recorded-model-replay-core)
- [prompts/phase5-03-recorded-evaluation-cli.md](#prompts-phase5-03-recorded-evaluation-cli)
- [prompts/phase5-04-recording-capture-session-core.md](#prompts-phase5-04-recording-capture-session-core)
- [prompts/phase5-05-private-recording-storage-provenance.md](#prompts-phase5-05-private-recording-storage-provenance)
- [prompts/phase5-06-provider-neutral-capture-orchestrator.md](#prompts-phase5-06-provider-neutral-capture-orchestrator)
- [prompts/phase5-07-ollama-querymodel-adapter.md](#prompts-phase5-07-ollama-querymodel-adapter)
- [prompts/phase5-08-ollama-adapter-live-smoke.md](#prompts-phase5-08-ollama-adapter-live-smoke)
- [prompts/phase5-09-private-capture-provenance-synthetic.md](#prompts-phase5-09-private-capture-provenance-synthetic)
- [prompts/phase5-10-live-capture-boundary.md](#prompts-phase5-10-live-capture-boundary)
- [prompts/phase5-11-capture-preflight-approval.md](#prompts-phase5-11-capture-preflight-approval)
- [prompts/phase5-12-synthetic-live-private-capture.md](#prompts-phase5-12-synthetic-live-private-capture)
- [prompts/phase5-13-finetuned-synthetic-evaluation.md](#prompts-phase5-13-finetuned-synthetic-evaluation)
- [prompts/phase5-14-versioned-queryir-prompt-trial.md](#prompts-phase5-14-versioned-queryir-prompt-trial)
- [prompts/phase5-15-prompt-failure-audit-evaluation-freeze.md](#prompts-phase5-15-prompt-failure-audit-evaluation-freeze)
- [prompts/phase5-16-independent-synthetic-evaluation-draft.md](#prompts-phase5-16-independent-synthetic-evaluation-draft)
- [prompts/phase5-17-independent-draft-coverage-oracle-smoke.md](#prompts-phase5-17-independent-draft-coverage-oracle-smoke)
- [prompts/phase5-18-direct-sql-baseline-arm.md](#prompts-phase5-18-direct-sql-baseline-arm)
- [prompts/phase5-19-live-arm-comparison.md](#prompts-phase5-19-live-arm-comparison)
- [prompts/phase5-20-ollama-system-decontamination.md](#prompts-phase5-20-ollama-system-decontamination)
- [prompts/phase5-21-base-instruct-arm-comparison.md](#prompts-phase5-21-base-instruct-arm-comparison)
- [prompts/phase5-22-symmetric-fence-extraction.md](#prompts-phase5-22-symmetric-fence-extraction)
- [prompts/phase5-23-schema-variation-claim-scope.md](#prompts-phase5-23-schema-variation-claim-scope)
- [prompts/phase6-01-bounded-synthetic-repair.md](#prompts-phase6-01-bounded-synthetic-repair)
- [prompts/phase6-02-synthetic-critique-disposition.md](#prompts-phase6-02-synthetic-critique-disposition)
- [prompts/phase6-03-synthetic-clarification-gate.md](#prompts-phase6-03-synthetic-clarification-gate)
- [prompts/phase6-04-synthetic-review-queue.md](#prompts-phase6-04-synthetic-review-queue)
- [prompts/phase6-05-private-review-snapshot-synthetic.md](#prompts-phase6-05-private-review-snapshot-synthetic)
- [prompts/phase6-06-synthetic-agent-workflow.md](#prompts-phase6-06-synthetic-agent-workflow)
- [prompts/phase6-07-synthetic-durable-review-queue.md](#prompts-phase6-07-synthetic-durable-review-queue)
- [prompts/phase6-08-synthetic-ambiguity-detection.md](#prompts-phase6-08-synthetic-ambiguity-detection)
- [prompts/phase6-09-clarification-query-constraint.md](#prompts-phase6-09-clarification-query-constraint)
- [prompts/phase6-10-clarification-execution-adapter.md](#prompts-phase6-10-clarification-execution-adapter)
- [prompts/phase6-11-clarification-session-boundary.md](#prompts-phase6-11-clarification-session-boundary)
- [prompts/phase6-12-clarification-http-boundary.md](#prompts-phase6-12-clarification-http-boundary)
- [prompts/phase6-13-synthetic-clarification-asgi-prototype.md](#prompts-phase6-13-synthetic-clarification-asgi-prototype)
- [prompts/phase6-14-synthetic-clarification-react-flow.md](#prompts-phase6-14-synthetic-clarification-react-flow)
- [prompts/phase6-15-synthetic-prototype-browser-smoke.md](#prompts-phase6-15-synthetic-prototype-browser-smoke)
- [prompts/phase6-16-active-database-selection.md](#prompts-phase6-16-active-database-selection)
- [prompts/phase6-17-private-sqlite-registration.md](#prompts-phase6-17-private-sqlite-registration)
- [prompts/phase6-18-loopback-sqlite-upload.md](#prompts-phase6-18-loopback-sqlite-upload)
- [prompts/phase6-19-private-mapping-publish-activation.md](#prompts-phase6-19-private-mapping-publish-activation)
- [prompts/phase7-01-postgres-adapter-capabilities.md](#prompts-phase7-01-postgres-adapter-capabilities)
- [prompts/phase7-02-postgres-query-runtime.md](#prompts-phase7-02-postgres-query-runtime)
- [prompts/phase7-03-synthetic-postgres-live-smoke.md](#prompts-phase7-03-synthetic-postgres-live-smoke)
- [prompts/phase7-04-server-owned-postgres-runtime-registry.md](#prompts-phase7-04-server-owned-postgres-runtime-registry)
- [prompts/phase7-05-private-postgres-startup.md](#prompts-phase7-05-private-postgres-startup)
- [prompts/phase7-06-postgres-frontend-selection.md](#prompts-phase7-06-postgres-frontend-selection)
- [prompts/phase7-07-ollama-constraint-provider.md](#prompts-phase7-07-ollama-constraint-provider)
- [prompts/phase7-08-server-owned-ollama-startup.md](#prompts-phase7-08-server-owned-ollama-startup)
- [prompts/phase7-09-ollama-prototype-live-smoke.md](#prompts-phase7-09-ollama-prototype-live-smoke)
- [prompts/phase7-10-runtime-ollama-request-format.md](#prompts-phase7-10-runtime-ollama-request-format)
- [prompts/phase7-11-request-format-in-model.md](#prompts-phase7-11-request-format-in-model)
- [prompts/phase7-12-legacy-view-partial-mapping.md](#prompts-phase7-12-legacy-view-partial-mapping)
- [prompts/phase7-13-legacy-view-chat-endpoint.md](#prompts-phase7-13-legacy-view-chat-endpoint)
- [prompts/phase7-14-chatpopup-queryir-routing.md](#prompts-phase7-14-chatpopup-queryir-routing)
- [prompts/phase7-15-legacy-vs-queryir-comparison.md](#prompts-phase7-15-legacy-vs-queryir-comparison)
- [prompts/phase7-16-chat-policies-and-recompare.md](#prompts-phase7-16-chat-policies-and-recompare)
- [prompts/phase7-17-anchor-left-join.md](#prompts-phase7-17-anchor-left-join)
- [prompts/phase8-01-mapping-version-rollback.md](#prompts-phase8-01-mapping-version-rollback)
- [prompts/phase8-02-schema-fingerprint-compare.md](#prompts-phase8-02-schema-fingerprint-compare)
- [prompts/phase8-03-stale-carry-over.md](#prompts-phase8-03-stale-carry-over)
- [prompts/phase8-04-test-query-preview.md](#prompts-phase8-04-test-query-preview)
- [prompts/phase8-05-verified-query-regression.md](#prompts-phase8-05-verified-query-regression)
- [prompts/phase8-06-value-dictionary.md](#prompts-phase8-06-value-dictionary)
- [prompts/phase8-07-value-dictionary-editor.md](#prompts-phase8-07-value-dictionary-editor)
- [prompts/phase8-08-variant-benchmark.md](#prompts-phase8-08-variant-benchmark)
- [prompts/phase8-09-ai-mapping-proposal.md](#prompts-phase8-09-ai-mapping-proposal)
- [prompts/phase8-10-ai-mapping-v2.md](#prompts-phase8-10-ai-mapping-v2)
- [prompts/phase8-12-ai-proposal-upload.md](#prompts-phase8-12-ai-proposal-upload)
- [prompts/phase8-13-chat-value-hints.md](#prompts-phase8-13-chat-value-hints)
- [prompts/phase8-14-chat-sql-display.md](#prompts-phase8-14-chat-sql-display)
- [prompts/phase8-15-postgres-registration.md](#prompts-phase8-15-postgres-registration)
- [prompts/phase8-16-registration-list-and-demo.md](#prompts-phase8-16-registration-list-and-demo)
- [prompts/phase8-17-other-school-validation.md](#prompts-phase8-17-other-school-validation)
- [prompts/phase8-18-target-grade-and-field-list.md](#prompts-phase8-18-target-grade-and-field-list)
- [prompts/phase8-19-db-extension-fields.md](#prompts-phase8-19-db-extension-fields)
- [prompts/phase8-20-unknown-conditions.md](#prompts-phase8-20-unknown-conditions)
- [prompts/phase8-21-postgres-format-variety.md](#prompts-phase8-21-postgres-format-variety)
- [prompts/phase8-22-blind-evaluation-runner.md](#prompts-phase8-22-blind-evaluation-runner)
- [prompts/phase8-23-ai-mapping-improvement.md](#prompts-phase8-23-ai-mapping-improvement)
- [prompts/phase8-24-code-table-import.md](#prompts-phase8-24-code-table-import)
- [prompts/phase8-25-time-expressions.md](#prompts-phase8-25-time-expressions)
- [prompts/phase9-01-l1-review-table.md](#prompts-phase9-01-l1-review-table)
- [prompts/phase9-02-l4-materials.md](#prompts-phase9-02-l4-materials)
- [prompts/phase9-03-env-check.md](#prompts-phase9-03-env-check)
- [prompts/phase9-04-rehearsal-school.md](#prompts-phase9-04-rehearsal-school)
- [prompts/phase9-05-raw-time-and-unasked-select.md](#prompts-phase9-05-raw-time-and-unasked-select)
- [prompts/phase9-06-numeric-columns.md](#prompts-phase9-06-numeric-columns)
- [prompts/phase9-07-publish-block-reasons.md](#prompts-phase9-07-publish-block-reasons)
- [prompts/phase9-08-chat-model-choice.md](#prompts-phase9-08-chat-model-choice)
- [prompts/phase9-09-openai-chat-model.md](#prompts-phase9-09-openai-chat-model)
- [prompts/phase9-10-direct-sql-postgres.md](#prompts-phase9-10-direct-sql-postgres)
- [prompts/phase9-10b-review-fixes.md](#prompts-phase9-10b-review-fixes)
- [prompts/phase9-10c-luna-fixes.md](#prompts-phase9-10c-luna-fixes)
- [prompts/phase9-11-published-review-contract.md](#prompts-phase9-11-published-review-contract)
- [prompts/phase9-11b-activatable.md](#prompts-phase9-11b-activatable)
- [prompts/phase9-12-db-admin-layout.md](#prompts-phase9-12-db-admin-layout)
- [prompts/phase9-13-postgres-test-queries.md](#prompts-phase9-13-postgres-test-queries)
- [prompts/phase9-14-review-row-cleanup.md](#prompts-phase9-14-review-row-cleanup)
- [prompts/phase9-15-bulk-approve.md](#prompts-phase9-15-bulk-approve)
- [prompts/phase9-16-retire-offering-year.md](#prompts-phase9-16-retire-offering-year)
- [prompts/phase9-17-reserved-database-id.md](#prompts-phase9-17-reserved-database-id)
- [prompts/phase9-18-delete-registration.md](#prompts-phase9-18-delete-registration)
- [prompts/phase9-19-database-id-message.md](#prompts-phase9-19-database-id-message)
- [prompts/phase9-20-copy-dictionaries.md](#prompts-phase9-20-copy-dictionaries)
- [prompts/phase9-21-unmatched-word-notice.md](#prompts-phase9-21-unmatched-word-notice)
- [prompts/phase9-22-admin-style.md](#prompts-phase9-22-admin-style)
- [prompts/phase9-23-luna-fixes.md](#prompts-phase9-23-luna-fixes)
- [prompts/phase9-24-v3-fallout.md](#prompts-phase9-24-v3-fallout)
- [prompts/phase9-25-teammate-guide.md](#prompts-phase9-25-teammate-guide)
- [prompts/phase9-26-dictionary-refresh.md](#prompts-phase9-26-dictionary-refresh)
- [prompts/phase9-27-renames.md](#prompts-phase9-27-renames)
- [prompts/phase9-28-onboarding-docs.md](#prompts-phase9-28-onboarding-docs)
- [synthetic-clarification-prototype.md](#synthetic-clarification-prototype)
- [synthetic-clarification-team-handoff.md](#synthetic-clarification-team-handoff)
- [team-discussion-brief-2026-10-06.md](#team-discussion-brief-2026-10-06)

---

<a id="desktop-test-handoff"></a>

## 원본: desktop-test-handoff.md

# 데스크톱 테스트 인수인계: Schema-adaptive Text-to-SQL

> 기준일: 2026-09-19 · 구현 브랜치: `feat/schema-adaptive-agent` · 이 문서는 **실행 결과 보고서가 아니라 테스트 출발점**이다.
>
> 설계·진행 상태의 원본: [`schema-adaptive-text-to-sql-plan.md`](schema-adaptive-text-to-sql-plan.md) · 사람 승인 관문: [`human-validation-checkpoints.md`](human-validation-checkpoints.md)

## 1. 프로젝트가 무엇인가

기존 졸업·경연 프로젝트는 충남대학교 수강정보에 특화된 한국어 Text-to-SQL 서비스다. React/FastAPI 화면에서 질문을 받아 로컬 Ollama 모델이 PostgreSQL용 SQL을 만들고, 수강정보를 조회한다. 기존 모델과 서비스는 **비교 기준선(legacy baseline)** 으로 보존한다.

확장 프로젝트의 목표는 대학 수강정보 도메인 안에서 **학교마다 다른 DB 스키마에 적응하는 시스템**이다. 새 DB를 Catalog로 분석하고, ontology와 물리 컬럼의 Mapping Proposal을 생성한다. 관리자가 의미를 교정·승인한 `Published Mapping`만 질의에 사용한다. 모델은 물리 SQL이 아니라 공통 `QueryIR`을 생성하고, 결정적 compiler가 학교별 JOIN과 parameterized SQL을 만든다. AST 정책과 읽기 전용 실행이 이를 검증한다. 수강 추천은 핵심 연구 주제가 아니다.

```text
새 DB → Catalog → Schema Aligner → 관리자 교정·publish
질문 → 모델의 QueryIR → Published Mapping + FK JoinPlan → SQL compiler
     → AST read-only policy → 읽기 전용 실행 → semantic 결과
```

기존 모델은 SQL을 생성한다. 새 경로의 `QueryModel`은 **QueryIR을 생성해야 한다**. Ollama에 접속된다는 사실만으로 기존 fine-tuned 모델을 새 QueryIR 경로의 성공 모델이라고 간주하지 않는다.

## 2. 현재 구현과 검증 경계

| 영역 | 구현 상태 | 아직 증명하지 못한 것 |
| --- | --- | --- |
| 데이터·IR | JSONL 품질 report, PostgreSQL SQL AST 분석, 승인 source profile 기반 SQL→QueryIR 변환, HMAC 가명화 도구 | 변환·가명화 결과의 외부 공개 승인, 모든 legacy SQL의 자동 변환 |
| 스키마 적응 | 공통 ontology, SQLite Catalog·aligner, 관리자 교정 CLI와 publish gate | 실제 다른 학교 컬럼·코드 의미에 대한 mapping 정확도 |
| 질의 실행 | FK JoinPlan, JOIN 가능한 SQLite compiler, AST allowlist, parameter binding·읽기 전용 SQLite executor | PostgreSQL용 adapter/compiler/policy의 실행 성공 |
| 평가 입력 | 한 synthetic seed의 flat·normalized·legacy SQLite 3개, 한국어 gold 40건 | 독립 unseen school, 40건의 독립 사람 승인 |
| 모델 평가 기반 | FakeModel runner, RecordedModel replay/CLI, capture session, private bundle storage, provider-neutral orchestrator | 실제 Ollama adapter·수집 CLI, live 모델 정확도·지연·VRAM |
| 제품 통합 | 기존 FastAPI/React legacy 경로 보존 | 새 adaptive 경로의 기존 API/화면 통합, repair loop·관리자 UI·schema evolution |

마지막 확인된 노트북 테스트(2026-09-18): `python -m pytest tests -q`에서 **254 passed, 3 skipped, 322 subtests passed**. 프론트 lint/build와 Python `compileall`도 직전 점검에서 통과했다. 이는 데스크톱의 새 환경 통과를 보증하지 않는다. 세 skip은 Windows에서 symlink 생성 권한이 없었던 보안 테스트다. 2026-10-17 Phase 9-27에서 assertion 없는 NER 보정 예제를 `backend/scripts/demo_ner_correction.py`로 옮겨 pytest 수집 대상에서 제외했다. 예제 실행은 `python -m backend.scripts.demo_ner_correction`이며, 전체 pytest 최종 확인은 별도로 수행한다.

`run-smoke`는 gold IR을 그대로 돌려주는 **oracle smoke**다. 실제 모델 정확도가 아니다. `run-recorded`는 외부에서 이미 만든 recording을 재생할 뿐 live 모델을 호출하지 않으며, recording의 provenance도 독립 증명이 아니다.

## 3. 데스크톱으로 가져갈 것과 가져가지 말 것

- fork: `https://github.com/miceadf/TextToSQLProject`, 브랜치 `feat/schema-adaptive-agent`. 마지막 인수인계 커밋 `2d666d6`. 데스크톱에서 `git status -sb`와 `git rev-parse HEAD`로 실제 체크아웃 상태를 다시 확인한다. 이 문서가 커밋되면 HEAD는 달라지므로 위 해시는 **기준선 식별용**이다.
- 원본: `https://github.com/rumyakun/TextToSQLProject`. 2026-09-19 확인 당시 원본 `master`의 `3ac6d8f`가 위 작업 브랜치의 조상이었다. 새 테스트 시점의 원본 변경 여부는 다시 확인해야 한다.
- Git에는 소스, 합성 fixture와 평가 질문만 둔다. 실제 CNU `dataset.csv`, DB dump, GGUF/model weight, 실제 질문·응답 recording, 가명화 key, 접속 비밀값은 Git으로 옮기지 않는다.
- 현재 저장소에는 실제 fine-tuned Ollama 모델 artifact가 포함되지 않는다. 데스크톱의 기존 로컬 모델 tag/Modelfile/GGUF 위치와 사용 권한을 별도로 확인한다. `models/Modelfile`만으로 모델 파일이 있다는 뜻은 아니다.
- 기록 파일·민감 report가 필요하면 `.schema-adaptive/` 등 gitignore된 private 경로를 사용한다. 그 경로도 보안·접근권한 검토 없이 공유하지 않는다.

## 4. 첫 세션: 환경 재현과 기준선 확인

작업 디렉터리를 저장소 루트로 둔다. 아래 명령은 새 코드 작성이나 DB 변경을 요구하지 않는다.

```powershell
git status -sb
git rev-parse HEAD
python --version
node --version
npm --version
docker --version
ollama --version
psql --version
```

클린 clone/가상환경에서 의존성을 설치할 때는 `backend/requirements.txt`, `package-lock.json`, `Dockerfile.backend`의 Python/Node 환경을 확인한다. `requirements.txt`는 일부 의존성만 버전이 고정되어 있어 노트북·데스크톱 결과가 다를 수 있다. 설치 실패나 버전 차이를 기록하고, 임의로 전체 의존성을 업데이트하지 않는다.

```powershell
python -m pytest tests -q
python -m compileall -q backend tests
npm ci
npm run lint
npm run build
```

위 테스트의 실제 pass/skip/fail을 기록한다. `npm ci`는 의존성 설치가 필요할 때만 실행한다. 루트 전체 수집도 별도 확인하되 현재 알려진 legacy import 실패는 adaptive 테스트 실패와 구분한다.

```powershell
python -m pytest --collect-only -q
```

참고: 현재 `docker-compose.yml`에는 PostgreSQL 서비스가 없고 backend의 `OLLAMA_BASE_URL`, `DATABASE_URL`이 비어 있다. README의 오래된 Docker 설명을 그대로 믿지 않는다. Docker `up`만으로 PostgreSQL·모델·스키마가 준비됐다고 판정하지 않는다. 기존 API의 `/api/v1/health` 통과도 실제 모델·DB 질의 성공의 증거가 아니다.

## 5. 첫 데스크톱 smoke test: 범위를 작게 유지

### A. Ollama 자체 확인 — DB와 분리

1. `ollama list`로 실제 설치된 모델 tag를 확인한다. 기존 fine-tuned 모델과 base 모델을 혼동하지 않는다.
2. Ollama 서버의 로컬 `/api/tags` 접근과 `ollama show <확인한-tag>`를 확인한다. 모델 tag, 가능하면 모델 digest, Ollama 버전, Modelfile/설정 버전을 기록한다. 비밀 경로·환경변수·프롬프트 원문은 공유 문서에 복제하지 않는다.
3. **합성 수강 질문 3~5개**만 별도 수동 호출한다. 질문과 schema context를 고정하고 응답이 raw SQL인지, JSON/QueryIR인지, Markdown fence·설명문이 섞이는지 확인한다. JSON parse 가능 여부와 필드 구조를 관찰하되 의미를 자동 보정하지 않는다.
4. 첫 호출과 재호출의 대략적인 시간·메모리/VRAM, model-not-found·timeout·빈 응답의 실제 형태를 기록한다. 이 관찰은 adapter 설계 입력이며 benchmark 점수가 아니다.

현재 새 adaptive 경로의 Ollama adapter와 live capture CLI는 **없다**. `backend/llm.py`는 legacy SQL 생성용이며 오류 본문·프롬프트 취급도 새 private recording 계약과 다르다. 이를 `QueryModel` 구현처럼 재사용하거나 실제 응답을 `run-recorded`에 바로 넣지 않는다. 다음 작은 개발 단계는 계획 문서의 **Ollama adapter 최소 호출·오류 변환 계약**이다.

### B. PostgreSQL 연결·메타데이터 확인 — adaptive 실행과 분리

1. 개발용 DB 또는 승인된 읽기 전용 계정·replica에서 PostgreSQL 버전, DB/role, 접속 방식(TLS·host/port), `SELECT 1` 성공을 확인한다. 실제 계정 비밀번호나 DSN을 로그·채팅·커밋에 남기지 않는다.
2. 메타데이터만 조회해 예상 테이블/view, PK/FK, 타입, nullable, 권한을 확인한다. `information_schema`와 `pg_catalog` 조회 가능 여부도 확인한다. 이때 실제 행이나 학생 정보는 출력하지 않는다.
3. `v_course_info` 등 legacy가 기대하는 view가 실제 DB에 있는지 확인한다. 없는 view를 smoke test 중 즉흥적으로 생성하지 않는다. README의 view DDL은 과거 환경의 예시이며 현재 DB에 그대로 적용해도 된다는 승인 정보가 아니다.
4. 읽기 전용 계정의 권한과 `statement_timeout` 적용 가능성을 DB 담당자와 확인한다. 단순 `SELECT 1`은 권한·행 수준 보안·실제 질의 안전을 증명하지 않는다. 운영 DB에서 DDL/DML 차단 시험을 위해 실제 쓰기 명령을 실행하지 않는다.

예를 들어 `psql` 접속 설정을 안전하게 준비한 뒤 아래 읽기 전용 명령만 사용할 수 있다. 연결 문자열·비밀번호는 명령 인자나 결과 공유 문서에 적지 않는다.

```powershell
psql -X -c "SELECT current_database(), current_user, current_setting('server_version')"
psql -X -c "SELECT 1"
psql -X -c "SELECT table_schema, table_name, table_type FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog', 'information_schema') ORDER BY table_schema, table_name LIMIT 30"
```

현재 새 경로에는 `PostgresAdapter`와 PostgreSQL JOIN compiler/executor가 없다. DB 연결 성공을 schema-adaptive end-to-end 성공으로 보고하지 않는다. SQLite SQL을 PostgreSQL에 그대로 실행해 성능을 판정하지 않는다.

### C. 합성 평가 경로 재확인 — 모델 성능과 분리

`python -m backend.adaptive.evaluation --format review-table`로 40개 synthetic gold의 검토 표를 확인한다. `run-smoke`는 합성 fixture의 실행 파이프라인 점검에만 사용한다. 보고서를 저장할 때는 민감 자료가 아닌 별도 출력 경로를 지정하고 기존 파일 덮어쓰기를 피한다.

```powershell
python -m backend.adaptive.evaluation run-smoke --output .schema-adaptive/reports/desktop-oracle-smoke.json
```

동일 seed의 3개 SQLite 스키마는 **타 대학 3곳이 아니다**. 40개 질문은 모두 `NEEDS_HUMAN_REVIEW`다. `run-recorded`에는 기존 `recorded-model-v1` JSON만 입력할 수 있으며, 새 private bundle의 `manifest.json`을 직접 받는 CLI는 아직 없다.

## 6. 이번 smoke test의 성공·중단 기준

- 성공: 새 환경에서 핵심 테스트가 재현되고, Ollama 모델과 PostgreSQL에 각각 **독립적으로 연결**되며, 모델 출력 형식·DB 메타데이터·권한 범위·환경 차이가 기록된다.
- 중단: 모델 tag/weight가 없거나, DB 연결·권한이 불명확하거나, 실제 CNU 자료 사용 범위가 미승인인 경우. 실패를 임의 데이터·다른 모델·관리자 계정으로 대체하지 않는다.
- 성공으로 **표기하지 않을 것**: 실제 모델 QueryIR 정확도, PostgreSQL adaptive 실행 성공, 학교 간 일반화, 40문항 benchmark, 개인정보 공개 안전성.
- 결함 발견 시: 재현 조건, 구성 버전, 실패 단계·안전한 오류 코드, 관련 커밋과 다음 작은 수정 단위를 기록한다. 실제 질문·SQL·응답·DSN은 이슈 본문에 붙이지 않는다.

## 7. 사람 확인과 후속 개발

가능하면 지금 E1(ontology/행 단위), E2(gold 5~10건), E4(독립 평가 스키마 확보 가능성), E5(깨끗한 clone 재현)을 확인한다. 실제 CNU 자료 활용·반출 전 **E3**, 실제 응답 기록·저장·외부 시연 전 **L5**의 데이터·보안 범위를 확인한다. 실제 학교 DB mapping을 publish·질의하기 전 **L2**, 모델 성능 수치를 확정하기 전 **L1·L3**, 학교 간 일반화 주장 전 **L6**가 필요하다. 승인되지 않은 상태를 코드나 문서에서 자동으로 완료 처리하지 않는다.

환경 smoke test 후의 권장 순서:

1. Ollama adapter의 JSON 반환·timeout/empty/provider error 변환을 합성 transport로 테스트한다.
2. 데스크톱에서 소수 합성 질문으로 adapter live smoke를 수행한다.
3. 별도 단계로 private capture CLI와 provenance 입력·검증을 연결한다.
4. `PostgresAdapter`·PostgreSQL dialect/read-only 정책을 별도 구현하고, 승인된 DB에서 소수 질의를 검증한다.
5. gold 사람 검토와 독립 평가 스키마를 확보한 뒤 공정한 모델 비교·논문 실험을 진행한다.

### 다음 대화에 전달할 짧은 문장

> `docs/development-history.md#desktop-test-handoff`와 `docs/schema-adaptive-text-to-sql-plan.md`, `docs/human-validation-checkpoints.md`를 먼저 읽어줘. `feat/schema-adaptive-agent`에서 Phase 5-06까지 완료했지만 Ollama/PostgreSQL adaptive adapter는 아직 없다. 먼저 데스크톱의 환경·모델 tag·DB 읽기 권한·합성 질문 출력 형식만 smoke test하고, 실제 데이터 기록이나 성능 점수 확정은 하지 말아줘. 테스트 결과와 한계, 다음 작은 수정 단위를 문서에 남겨줘.

### 세션 기록 양식

```text
날짜 / OS·Python·Node·PostgreSQL·Ollama 버전 / Git 커밋:
모델 tag·digest(가능한 범위) / prompt·설정 버전:
DB 종류·버전·role의 읽기 권한·metadata 접근(비밀값 제외):
백엔드 tests 통과·실패·skip / 프론트 lint·build / 루트 pytest 수집:
Ollama 합성 질문 수 / 응답 형식·JSON parse 여부 / 오류 유형:
PostgreSQL 연결·SELECT 1·metadata·view 확인 결과:
사람 검토 E1~E5 및 L1~L6 상태(승인자를 추측하지 않음):
미해결 위험 / 다음 작업 / 재검토 필요 여부:
```

실험 기록에는 질문·응답·SQL·개인정보·접속 비밀값을 기본적으로 넣지 않는다. 실제 데이터가 필요한 경우 접근·보관·삭제 정책을 먼저 확정한다.

---

<a id="midterm-review-2026-10-08"></a>

## 원본: midterm-review-2026-10-08.md

# 중간 점검: 다른 학교 DB에 적응하는 수강 챗봇 (2026-10-08)

> 팀원 공유용. 이 프로젝트를 처음 보는 사람도 읽을 수 있게 썼다. AI가 저장소 문서를 모아 정리했고, 결정이나 승인을 담지 않는다.
> 아래 숫자는 모두 개발 중에 확인한 잠정치다(3절 끝 참고).

## 0. 한 줄 요약

우리 학교 DB에만 맞춰져 있던 수강 챗봇을, **다른 학교 DB에도 모델을 다시 학습하지 않고 연결할 수 있게** 바꾸는 중이다. 만드는 일은 거의 끝났다. 남은 것은 "정말 다른 학교에서도 되는지"를 **사람이 확인하는 일**이다.

## 1. 무엇을 만드나

### 기존 방식과 문제

- 학생이 "컴융 전공 과목 보여줘"라고 물으면, AI 모델이 DB에 보낼 명령문(SQL)을 직접 썼다.
- 이 모델은 우리 학교 DB의 표 이름·칸 이름·코드 값을 외워서 학습했다.
- 다른 학교는 표 이름(`gangjwa`, `course_offering` …), 칸 이름, 코드(`11` = 전공필수 …)가 모두 다르다. 그래서 그대로 쓸 수 없다. 다른 학교에 쓰려면 그 학교 데이터로 다시 학습시켜야 하는 구조였다.

### 새 방식 (비유)

기존 모델은 **우리 집 냉장고 위치를 외운 사람**이다. 새 방식은 **어느 집에 가든 먼저 "무엇이 어디 있는지" 지도를 받아서 찾는 사람**이다. 이 지도를 "mapping"이라고 부른다.

1. **관리자가 학교 DB를 등록한다.** AI가 "과목명은 이 칸, 학점은 이 칸" 하고 지도 초안을 제안하고, 관리자가 확인해 확정(게시)한다.
2. **학생이 질문하면** AI는 SQL을 쓰지 않는다. "어떤 조건으로 무엇을 보여 줄지"만 정해진 양식(QueryIR)으로 정리한다.
3. **프로그램이** 지도를 보고 그 학교에 맞는 SQL을 만든다. 안전한지 검사한 뒤 DB를 읽기만 해서 답한다.

### 용어

| 용어 | 뜻 |
| --- | --- |
| DB, table, column | 데이터를 담은 엑셀 파일 묶음, 그 안의 시트, 시트의 열이라고 보면 된다 |
| SQL | DB에 "이런 조건의 행을 보여 줘"라고 묻는 명령문 |
| 공통 항목 | 학교와 무관하게 정해 둔 기준 정보 16가지(과목명, 학점, 이수 구분, 요일, 시작 시간 등). 학교에 따라 없는 항목은 "이 DB에 없음"으로 둔다 |
| mapping(지도) | "공통 항목이 이 학교 DB의 어느 칸에 있는지"를 적은 표. 학교마다 하나 |
| QueryIR | AI가 질문을 정리하는 양식. 예: `학과 = 컴퓨터융합학부`, `이수 구분에 "전공" 포함` |
| 값 사전 | 학교마다 다른 말을 맞추는 표. 약어(`소웨` = 소프트웨어학부), 코드 뜻(`11` = 전공필수), 교시표(`1교시` = 09:00) |
| 게시 | 관리자가 검토를 마친 지도를 챗봇에 적용하는 것 |
| 로컬 모델 | 인터넷 서비스가 아니라 우리 PC에서 돌리는 AI 모델(Ollama). 챗봇용 1개, 지도 제안용 1개 |

## 2. 지금 되는 것

### 관리자 화면

- 화면에서 PostgreSQL DB를 고르고 등록한다. AI가 공통 항목마다 맞는 칸을 제안한다(20~30초).
- 항목마다 **승인 / 다시 지정 / 이 DB에 없음**을 고른다.
- 값 사전에 약어·코드 뜻·교시표를 넣는다. 학교 DB에 코드표가 있으면 **"코드표에서 가져오기"**로 한꺼번에 넣는다.
- 공통 항목에 없는 그 학교만의 정보(강의실, 수업 방식 등)를 **확장 항목**으로 더한다.
- 게시하면 서버를 다시 켜지 않아도 챗봇에서 바로 쓸 수 있다.

### 학생 챗봇

- 평소 말투로 질문하면 결과와 함께 **"이해한 조건"** 카드가 나온다. 챗봇이 질문을 어떻게 이해했는지 확인할 수 있다.
- "이 DB에 없음"으로 표시한 항목이나, 미리 정해 둔 개념 11개(강의실 등)를 물으면 "이 DB에는 강의실 정보가 없어 답할 수 없습니다"처럼 거절한다. 엉뚱한 결과를 내지 않으려는 것이다. 목록에 없는 말은 다른 항목으로 잘못 이해할 수 있다.
- 결과가 0건이면 어느 조건에 맞는 값이 없었는지 알려 준다. 같음·포함 조건이 4개 이하일 때만 하고, "3학점 이상" 같은 범위 조건은 보지 않는다.
- "오전", "오후 3시 이후에 시작하는", "4시 전에 끝나는" 같은 시간 표현은 정해진 규칙으로 읽는다. 모델이 빠뜨리면 규칙이 보완한다.

### 이런 DB 모양에서 확인했다

| 모양 | 예 |
| --- | --- |
| 여러 table로 나뉜 DB, 로마자 이름 | `gangjwa`, `gyogwamok`, `hg_nm` |
| 한 table에 다 넣은 DB | 수강편람 엑셀을 그대로 옮긴 모양 |
| 코드만 저장하고 뜻은 공통코드 table에 | `isu_cd = 11`, 공통코드에 `11 = 전공필수` |
| 시간 대신 교시 번호 | `1교시`, `2교시` … |
| 한 칸에 시간표를 묶어 저장 | `월1-2,수3` |

### 평가·협업 도구

- **팀원 DB 평가 도구:** 팀원이 정한 질문을 개발자가 보지 않은 채 실행하고, 첫 결과를 그대로 남긴다.
- **L1 검토 표:** 사람이 질문·정답 70개를 판정하는 표.
- **L4 사용성 점검 자료:** 개발에 참여하지 않은 사람이 써 보는 진행 자료.
- **환경 점검 명령:** 팀원 PC에서 무엇이 빠졌는지 알려 준다(`.venv\Scripts\python.exe -m backend.adaptive.env_check`).

### 안전 장치

- 학교 DB는 **읽기만** 한다. 쓰기·삭제 명령은 만들 수도, 실행할 수도 없다.
- DB 접속 정보(주소·비밀번호)는 화면과 기록에 나오지 않는다.
- AI가 낸 결과는 검사를 통과해야 실행된다. AI가 쓴 SQL을 그대로 실행하는 길은 없다.

## 3. 얼마나 잘 되나 (개발 중 확인)

| 확인 | 결과 |
| --- | --- |
| 레거시 DB, 질문 30개(개발자 작성) | 새 방식 22개, 기존 시스템 21개가 맞음 |
| 여러 table·로마자 이름 학교 | 19/19 |
| 한 table 학교 | 11/12 |
| 공통코드 학교 | 14/14 |
| 교시 번호 학교 | 18/19 (틀린 1개는 정답 기준 차이) |
| 묶인 시간표 학교 | 19/19 (시간 질문은 기대대로 "정보 없음") |
| AI 지도 제안이 첫 후보로 맞힌 항목 | 80개 중 55개 → 75개 |
| 자동 테스트 | 1,996개 통과, 실패 0 |

각 값은 그 기능을 만든 단계에서 잰 것이다(레거시 30문항은 2026-09-30, 여러 table·교시·묶인 시간표 학교는 2026-10-06 시간 표현 개선 뒤, 나머지는 2026-10-02~06).

**이 숫자를 읽는 법.** 위 학교 DB 5개는 AI가 만든 가짜 학교이고, 질문과 정답도 개발자 쪽(AI 포함)이 만들었다. 문제를 낸 쪽과 푼 쪽이 같으므로 **"다른 학교에 적응한다"는 근거가 아니다.** 아직 사람이 질문·정답을 검토하지도 않았다. 논문에 쓸 숫자는 팀원이 만든 평가 DB의 첫 결과를 사람이 검토한 뒤에 나온다.

## 4. 더 고려해야 할 것

### 챗봇 품질

- 조건이 많으면 **하나를 빠뜨릴 수 있다.** "이해한 조건" 카드로 확인해야 한다.
- **한 질문씩만** 이해한다. "그중에서 3학점만"처럼 앞 질문을 이어 물으면 앞 조건이 남지 않는다.
- 애매한 질문을 **되묻지 않는다.** 대신 이해한 조건을 보여 준다.
- "나한테 맞는 과목 추천" 같은 개인 맞춤 요청은 하지 않는다.
- 같은 질문도 실행마다 답이 달라질 수 있다. 시간 표현은 규칙으로 보완했지만 다른 표현은 여전하다.

### 관리자 기능

- PostgreSQL 등록에는 **이전 버전으로 되돌리기**와 **게시 전 시험 질문**이 아직 없다.
- 한 칸에 묶인 시간표(`월1-2,수3`)에서는 요일만 찾고, **시간은 읽지 않는다.**
- AI 지도 제안은 틀릴 수 있다. 예: "대상 학년"을 학기 코드 칸으로 제안했다. **관리자 검토는 꼭 필요하다.**

### 평가와 논문

- **독립 평가 DB가 아직 없다.** 그래서 지금은 "다른 학교에 적응한다"고 말할 수 없다.
- 사람 검토(L1)가 0/70이다.
- 기존 모델과 새 방식, 학습한 모델과 기본 모델의 비교 수치도 모두 잠정치다.
- 실제 다른 학교 DB는 받지 않기로 했으므로(2026-10-01 결정), 논문 한계에 **"모든 평가 데이터는 모사 데이터"**라고 적어야 한다(독립 학교 평가 기준. 레거시 DB는 공개 수강편람으로 만든 실제 학교 데이터다).

### 재현성

- 지금까지 **개발 PC 한 대**에서만 확인했다. 팀원 PC에서도 되는지(L7)는 아직 모른다.
- AI 모델 두 개(각 7B)를 PC에서 직접 돌리므로 PC 사양이 필요하다.

## 5. 팀원과 논의할 것

### 정해야 할 것

1. **평가 DB 분담과 일정.** 누가 어느 학교를 흉내 내어 언제까지 만들지. 2~3개가 목표다. 이번에 확인한 다섯 모양 밖의 모양이 섞일수록 좋다.
2. **L1 판정 일정.** 질문·정답 70개를 사용자가 먼저, 그다음 팀원이 본다.
3. **L4 일정과 참가자.** 개발에 참여하지 않은 3~5명, 한 사람에 40분 안팎.
4. **묶인 시간표의 시간까지 읽을지.** 읽으면 답할 수 있는 질문이 늘지만, 학교마다 표기가 달라 잘못 읽을 위험이 있다.
5. **공통 항목을 더할지.** 지금 16개다. 강의실·여석 같은 것은 학교별 확장 항목이나 "정보 없음"으로 처리한다.
6. **"정보 없음"으로 알아듣는 개념 11개**(강의실, 수업 방식, 단과대학, 강의 언어, 성적 평가 방식, 시험, 캠퍼스, 선수 과목, 강의 평가, 여석, 강의계획서)에 더할 것이 있는지.
7. **논문에서 주장할 범위.** "흉내 낸 학교 DB 사이의 적응"으로 좁히는 데 동의하는지.
8. **(선택) 다음 개발 우선순위.** 4절의 한계 중 무엇을 먼저 할지. 예: 이어 묻기, 애매한 질문 되묻기, 등록 되돌리기.

### 지켜 줄 규칙 (평가 DB를 만들 때)

1. 학교명·교수명은 지어낸다.
2. 질문과 정답은 DB를 만든 사람이 먼저 고정한다.
3. 개발자(AI 포함)는 평가 전까지 그 DB와 질문을 보지 않는다. 질문 파일을 채팅이나 저장소에 올리지 않는다.
4. 첫 실행 결과를 그대로 기록하고, 그 뒤 고친 결과는 따로 표시한다.

### 이미 정해져 공유만 할 것

- 실제 다른 학교 DB는 받지 않는다. 레거시 DB와 흉내 낸 DB만 쓴다(2026-10-01).
- PostgreSQL DB만 대상으로 한다(2026-10-02).
- 시간 표현 개선은 원래 L1 뒤에 할 일이었지만, 학교와 무관한 규칙이라 먼저 했다(2026-10-06).
- 공통 항목에 "대상 학년"을 더했다(2026-10-01).

## 6. 앞으로의 순서

1. 팀 논의(5절).
2. L1 판정과 평가 DB 만들기를 함께 진행한다.
3. 팀원 PC에서 환경 점검과 재현(L7)을 한다.
4. 평가를 실행하고 첫 결과를 고정한다.
5. 결과를 사람이 검토한다(L3).
6. 사용성 점검(L4)을 한다.
7. 논문 수치와 한계를 정리한다.

## 7. 자료 위치

| 자료 | 위치 |
| --- | --- |
| 설계와 모든 결정 기록 | [schema-adaptive-text-to-sql-plan.md](schema-adaptive-text-to-sql-plan.md) |
| 사람 확인 관문(L1~L7) | [human-validation-checkpoints.md](human-validation-checkpoints.md) |
| 팀원용 평가 DB·질문 만들기 | [teammate-evaluation-db-guide.md](teammate-evaluation-db-guide.md) |
| L1 검토 표 | [l1-review/fixture-v1-review.md](l1-review/fixture-v1-review.md) |
| L4 진행 자료 | [l4-usability/facilitator-guide.md](l4-usability/facilitator-guide.md) |
| 시연 절차 | [demo-procedure.md](demo-procedure.md) |

---

<a id="next-session-handoff"></a>

## 원본: next-session-handoff.md

## 사람용 현재 요약 (2026-10-17)

- 현재 기능: PostgreSQL 등록·항목 연결 검토·게시·값 사전·확장 항목·시험 질의·등록 삭제, DB별 QueryIR 챗봇과 레거시 기준선.
- 실행: 루트 README의 준비·설정 후 환경 점검 → `npm run dev:all`(5173·8000·8123).
- 읽기: [문서 목차](README.md) → 계획 문서 개요·범위·구조 → 팀원 DB 안내 → 시연 절차.
- 남은 사람 작업: 사용자·팀원 L1/L3 검토, 개발 외부인 L4, 팀원 평가용 모사 DB·질문·정답 고정과 평가.
- 알려진 한계: PostgreSQL 전용, SQLite 등록은 검토 대기만 지원; 복합 질문의 조건 누락 가능, 성능은 사람 검토 전 잠정치.
- 현재 ontology v3에는 개설 연도가 없고 학기는 유지한다. 아래는 과거 상태가 포함된 누적 기록이다.

# 새 Codex 대화용 짧은 인수인계

사람 관문(2026-10-01 사용자 결정, ADR-086): 실제 학교 DB는 없다는 전제로 재정의했다. 개발을 막는 관문은 없다. 세부는 `human-validation-checkpoints.md`의 2026-10-01 절에 있다. 남은 사람 작업은 L1/L3 검토(Phase 9 진입 때 표 준비), L4, 팀원의 평가용 모사 DB 제작이다.

현재 기준: 2026-10-01, Phase 8-20 모르는 조건 처리까지 완료(ADR-088). 사용자가 정한 순서(8-18 → 8-19 → 8-20)는 끝났다. 2026-10-02 사용자 결정(ADR-089): PostgreSQL 전용, 다른 PostgreSQL DB에 대한 적응성만 범위다(업로드 SQLite 확장 안 함). 진행 중: 8-21 형식이 다른 PostgreSQL DB 대응 점검(지침 `docs/development-history.md#prompts-phase8-21-postgres-format-variety`). 형식 D(넓은 단일 table)는 코드 수정 없이 동작 확인(11/12, `docs/experiment-results.md#phase8-21-wide-table-results-2026-10-02`, 합성 DB `wide_school` schema 유지). 형식 A(공통코드)도 완료(ADR-090, `docs/experiment-results.md#phase8-21-code-table-results-2026-10-05`, 합성 DB `code_school` 유지): 코드 column + 관리자 값 사전, 이름 일부로 코드 여러 개 찾기(`LIKE OR IN`). 형식 B(교시)도 완료(ADR-091, `docs/experiment-results.md#phase8-21-period-results-2026-10-06`, 합성 DB `period_school` 유지): 교시표 = `HH:MM` 이름의 값 사전, 시간 비교 → 교시 `IN` 목록. 형식 C(묶인 시간표)도 ①로 완료(ADR-092, `docs/experiment-results.md#phase8-21-packed-results-2026-10-06`, 합성 DB `packed_school` 유지): 요일은 `PACKED_WEEKDAY` 포함 검색, 시간은 "이 DB에 없음". 8-21 형식 점검 끝. 8-22 팀원 DB 평가 실행 도구 완료(ADR-093, `python -m backend.adaptive.evaluation.blind_eval check|run|report <ID>`, 팀원용 안내 `docs/teammate-evaluation-db-guide.md`). AI는 팀원의 질문 파일을 평가 실행 전까지 열지 않는다. Phase 9 진입으로 L1 검토 표를 준비했다(ADR-094): `docs/l1-review/fixture-v1-review.md`(읽기)·`.csv`(판정 입력, 승인/수정/보류·의견·검토자·검토일). 기록은 `python -m backend.adaptive.evaluation.l1_review record docs/l1-review/fixture-v1-review.csv`. AI는 판정하지 않고 gold review_status를 바꾸지 않는다. 다음: 사용자의 L1 판정 → 팀원 검토, 팀원 모사 DB·질문 파일(팀원 질문 표는 `l1_review build --blind <ID>`). Phase 9-09 ChatGPT API 선택지 완료(ADR-099, 사용자 결정; `backend/.env`에 `OPENAI_API_KEY`(선택 `OPENAI_MODEL`, 기본 `gpt-5-mini`)를 넣고 서버를 다시 켜면 "모델"에 "ChatGPT(외부 API)"가 생김; 질문·공통 항목·값 힌트만 보냄; 실제 호출 측정은 사용자 시험 대기; `docs/experiment-results.md#phase9-09-openai-chat-model-results-2026-10-12`). Phase 9-08 챗봇 기본 모델을 일반 모델로 바꾸고 모델 고르기 완료(ADR-098, 사용자 결정; `general`=`CHAT_MODEL` 또는 `qwen2.5-coder:7b-instruct` + QueryIR JSON schema + prompt `query-ir-v3`, `legacy-tuned`=`OLLAMA_MODEL` 요청 불변; 화면 "모델" 고르기, `GET /api/adaptive/v1/chat/models`, `blind_eval run --model <id>`; 레거시 30문항 23 대 22, `other_school` 10/12 대 12/12, 리허설 6/14 대 6/14, `docs/experiment-results.md#phase9-08-chat-model-choice-results-2026-10-12`; 남은 판단: 팀원 DB를 두 모델로 평가한 뒤 기본 모델 재검토). Phase 9-07 게시 막힘 항목·이유 한 번에 보이기 완료(응답 `reasons`, 화면 한국어 한 줄씩; 남은 결정: 리허설 문제 4·5). Phase 9-06 리허설 문제 3 고침 완료(숫자 항목은 숫자 칸·검증된 변환일 때만 게시, 변환 칸의 빈 글자 막음; 리허설 재실행 9/14·오류 0; 남은 결정: 문제 4~6, 게시 막힘을 한 번에 보이기). Phase 9-05 리허설 문제 1·2 고침 완료(ADR-097: 글자 칸 시간 항목은 `HH:MM`일 때만 게시, 말하지 않은 select의 없는 항목은 빼고 답함; 레거시 36/36 불변, 리허설 재실행 9/14; 남은 결정: 문제 3~6). Phase 9-04 리허설 학교 완료(`rehearsal_school`, 첫 실행 6/14, 발견 6건은 사용자 결정 대기: 종료 시간 조용한 오답·분반 select 거절·학점 글자 칸 500·분반 붙은 코드·연도/학기 한 칸·게시 막힘 이유, `docs/experiment-results.md#phase9-04-rehearsal-results-2026-10-08`). 중간 점검 자료 `docs/development-history.md#midterm-review-2026-10-08`(2026-10-08, 팀원 공유용). Phase 9-03 팀원용 환경 점검 `python -m backend.adaptive.env_check` 완료(팀원 안내 1-1절). 가이드 시간 문장은 QueryIR 화면에서만 바꿈(사용자 결정, 완료). opencode Luna 리뷰는 표준 입력을 닫고 시간 제한을 걸어 돌린다: `timeout 1200 opencode run ... < /dev/null`. 원인(2026-10-08 확인): `opencode run`은 표준 입력이 pipe면 끝(EOF)까지 읽은 뒤 시작하므로, 닫히지 않는 입력에 붙으면 모델 호출 전에 무한히 기다린다(2026-10-06·08 두 번 멈춤). Phase 9-02 L4 진행 자료 완료(`docs/l4-usability/`: 진행자 안내·과제 카드·관찰 CSV·사후 질문·요약 양식; 진행·판단은 팀). 8-25 챗봇 시간 표현 안정화 완료(ADR-096, `api/time_hints.py`: 닫힌 규칙 → 연산자 힌트·고침·빠진 조건 추가·없는 시간 항목은 정보 없음; 사용자 결정으로 L1 전 예외). 팀 논의 자료 `docs/development-history.md#team-discussion-brief-2026-10-06` 작성(결정·승인 없음, 팀이 정할 것 6개 정리). 8-24 코드표에서 값 사전 가져오기 완료(값 사전 패널 "코드표에서 가져오기", API `IMPORT`). 8-23 AI mapping 제안 v3 완료(ADR-095, 개발용 schema 5개 55→75/80, 측정은 `python -m backend.adaptive.evaluation.school_mapping_check --label <이름> --repeat 2`; 합성 DSN 필요). 후속 후보: 묶인 시간표의 시간 해석(②), 공통코드 table에서 값 사전 가져오기, 코드형 column AI 제안 개선. 후속 후보: 공통코드 table에서 값 사전 자동 가져오기, 코드형 column AI 제안 개선. 테스트 서버 helper는 scratchpad가 지워지므로 결과 문서의 SQL과 절차로 다시 만든다. 프로토타입 완성 단계(ADR-078) 종료. 시연은 [`demo-procedure.md`](demo-procedure.md) 참고(8-07 값 사전, 8-15 PostgreSQL 등록, 8-14 챗봇 SQL 표시, 8-13 챗봇 값 힌트, 8-12 업로드 AI 제안, 8-10 AI 제안 v2, 8-09 AI 제안, 8-08 benchmark, 8-06 값 사전, 7-17 LEFT JOIN, 8-05까지의 관리자 기능 포함). 사용자 결정으로 benchmark 개선과 8-11은 프로토타입 이후로 미뤘다(ADR-078). 최신 상태는 항상 Git과 코드로 다시 확인한다. 설계와 로드맵의 원본은 [`schema-adaptive-text-to-sql-plan.md`](schema-adaptive-text-to-sql-plan.md), 독립 평가 세트 설계 기준은 [`independent-evaluation-set-design.md`](independent-evaluation-set-design.md), 안전한 잠정 비교는 [`phase5-14-versioned-queryir-prompt-trial-results-2026-09-19.md`](experiment-results.md#phase5-14-versioned-queryir-prompt-trial-results-2026-09-19)다.

현재 기준 추가(2026-10-12): Phase 9-10 PostgreSQL 직접 SQL 비교군 구현(평가 전용, ADR-100; catalog·10개/40자 예시·실제 값 힌트, 관리자 값 사전 없음; private 코드/개수/hash/elapsed_ms 기록, blind arm 보고, other_school QueryIR 비교 CLI). 9-10b: OpenAI 평가 허용은 같은 날 철회·직접 SQL OpenAI 경로 제거, 비교는 로컬 general만; 검사 뒤 200행 wrapper와 PostgreSQL 식별자 대소문자 규칙 보완. 가짜 targeted 테스트·9-10c 제공 Luna 리뷰 지적 4건 재현·수정 완료. 사용자 승인으로 한 번 측정한 로컬 7B 결과: 직접 SQL 17/56·QueryIR 42/56(레거시 10/30·26/30, other_school 5/12·10/12 — QueryIR은 9-08 재사용·응답 시간 없음, 리허설 2/14·6/14; 레거시 QueryIR은 거절 정답 3을 포함). [결과 문서](experiment-results.md#phase9-10-direct-sql-results-2026-10-12). 개발 질문·L1/L3 전 잠정치, 관문 승인·제품 구조 결정 없음. 테스트 서버 8124는 측정 뒤 종료했고 사용자 서버는 건드리지 않았다. 남은 등록 `p910-legacy`·`p910-other`는 PENDING_REVIEW·게시 안 함(관리자 작업 없음), 리허설은 기존 `rehearsal-school`이다. 이번 문서 기록은 새 측정·모델 호출(Luna/opencode 포함)·서버 조작·테스트·commit 없이 수행했다.

## 프로젝트와 현재 경계

**현재 개발 방침:** 팀의 사람 판단(E3/L5, L1/L3)은 지금 요청하거나 완료 처리하지 않는다. 합성 입력의 로컬 구현·테스트·private 검증을 계속한다. 실제 CNU 자료의 추가 사용·공유, 실제 데이터 배포·외부 시연, 모델 성능 수치 확정에 도달할 때 해당 관문을 적용한다. 개발 중 같은 결정을 반복해서 묻지 않는다.

- 기존 React/FastAPI + Ollama SQL 생성 + PostgreSQL 서비스는 legacy 기준선으로 보존한다. 새 경로는 `QueryIR`과 runtime ontology, Published Mapping, 읽기 전용 실행 하네스를 사용한다.
- Phase 5-07: Ollama `QueryModel` adapter와 합성 transport 테스트 완료. Phase 5-08: 로컬 `text2sql_qwen:latest`에 합성 질문 2건만 비저장 호출했고 둘 다 JSON object/현재 QueryIR 검증을 통과했다. 모델 정확도 증거는 아니다.
- Phase 5-09: 합성 provider의 capture→private bundle→strict load, 저장 전 preflight, synthetic-only CLI, Ollama 선언형 provenance hash 완료.
- Phase 5-10: `.schema-adaptive`의 versioned private target 파일을 읽는 별도 `backend.adaptive.harness.live_capture` CLI 구현. 명시적 `--live`와 전체 사전 검증, 선언 provenance, 기존 private bundle을 연결했다. 합성 transport 테스트만 실행했다.
- Phase 5-11: `--check`와 `--live`가 동일한 preflight 결과를 사용한다. `--check`는 호출·저장 0회이고 합성 테스트로 경계를 검증했다. E3·L5 미기입 양식은 `PENDING_HUMAN_REVIEW`다.
- Phase 5-12: 완전 합성 질문 2건을 로컬 Ollama에 `--live` 1회로 전송하고 응답 2건을 Git 비추적 private bundle에 저장했다. strict load·`RecordedModel` 재생과 JSON object·QueryIR 검증을 통과했다.
- Phase 5-13: 합성 benchmark v1 40×3 target을 결정적으로 생성해 정확한 로컬 fine-tuned tag로 120건을 한 번씩 수집했다. 잠정 집계는 구조 exact 51, 실행 성공 99, semantic 일치 90이다. Gold 40건은 모두 `NEEDS_HUMAN_REVIEW`이므로 확정 성능 아님.
- Phase 5-14: v1 `UNKNOWN_FIELD` 21건을 비노출 분류한 뒤 `query-ir-v2`를 같은 합성 40×3 세트에서 수집·비교했다. v2는 구조 exact 0, 실행 성공 42, semantic 일치 36, `UNKNOWN_FIELD` 72로 악화. v1을 기본값으로 유지한다.
- Phase 5-15: paired 비교 무결성 계약, 합성 테스트, 읽기 전용 strict-load 재검증(v2 `UNKNOWN_FIELD` 72건 범주 재집계) 완료. 미승인 합성 개발 세트의 진단 집계이며 확정 성능 아님. 독립 합성 평가 세트 설계 기준 문서 작성. gold 40건 `NEEDS_HUMAN_REVIEW` 유지. 새 prompt 탐색·live 호출·확정 성능 발표 없음.
- Phase 5-16: 독립 합성 평가 세트 30건 초안(P2-041~070) 작성·자동 검증 완료, 사람 검토·동결 전. 정식 30~50건 조건 통과, 3개 fixture에서 gold IR 90회 실행이 기대 결과와 일치(자기 일관성, 모델 점수 아님). 개발 세트와 격리 검증 완료. 기존 40건과 새 30건 모두 `NEEDS_HUMAN_REVIEW` 유지. Ollama 호출·성능 평가·승인 없음.
- Phase 5-17: 독립 세트 문항 정리(P2-057·P2-059, 격리 유지)·단독 커버리지(30건·15개 field·결과 0/1/여러 건·JOIN 0~3단계)·FakeModel oracle smoke 90건 전건 일치(모델 점수 아님) 완료. 제품 코드 변경 없음. 새 30건은 사람 검토·동결 전 question holdout 초안 유지. Ollama 호출·성능 평가·승인 없음.
- Phase 6-01: 합성 입력의 제한된 QueryIR repair 경계(`BoundedRepairCoordinator`, 최초 1회 + repair 1회) 구현·합성 검증 완료. 재시도 허용 code/stage 고정, provider에는 고정 code만 전달, 후보 전체 안전 경로 재통과. request·ontology·mapping은 깊은 복사 스냅샷으로 고정하고, 각 모델은 격리 래퍼를 거쳐 harness 검증 컨텍스트와 분리된 복사본만 받아 모델 변이가 검증·컴파일·실행 컨텍스트에 영향을 주지 않는다. 사용자 확인·관리자 검토 큐는 미구현. 성능·정확도 수치 없음. 새 30건과 기존 40건 `NEEDS_HUMAN_REVIEW` 유지.
- Phase 6-02: 합성 `RepairSummary` 전용 결정적 critique 분류(`critique-disposition-v1`) 구현·합성 검증 완료. 성공은 `EXECUTED_UNVERIFIED`+`repair_attempted`·`recovered_after_repair` 구분, 실패는 고정 (code, stage) 6개 disposition 분류, 위조·미지·모순은 `CRITIQUE_INVALID_SUMMARY`로 거부. repair 없으면 최종=초기, repair 표시는 Phase 6-01 재시도 허용 쌍에서만 수락. 원문·질문·SQL·행·경로 비노출, 호출 0회. 전체 loop·모호 질문 확인·관리자 검토 큐는 미구현. 성능·정확도 수치 없음.
- Phase 6-03: 합성 모호성 신호·사용자 확인 경계(`clarification-gate-v2`) 구현·합성 검증 완료. caller 제공 명시적 신호만 pending에 묶고(2개 이상 서로 다른 option ID·semantic 식별자), 신호 없으면 pending·차단 없음. `resolve`는 caller 원본 신호를 재요구해 정렬 option ID·semantic_ref 전체 일치를 확인(일관성 확인이며 암호학적 진위 증명 아님). 정확한 한 선택만 `CONFIRMED`, 명시적 취소는 별도 `CANCELLED`. pending·결과에는 원시 식별자를 저장하지 않고 domain-separated fingerprint만 저장. 전 공개 경계는 dump-then-validate 재검증으로 위조 인스턴스를 고정 code로 거부. 원문·물리·SQL·행·경로·secret 비노출, 모델·DB·네트워크 호출 0회. 자동 감지·QueryIR 보정·하네스 자동 실행·UI·관리자 큐는 미구현. `모호한 질문 감지 및 사용자 확인` 전체 항목은 미완료 유지. 성능·정확도 수치 없음. 실제 CNU 자료·private bundle 미사용.
- 중간 점검 결함 수정: critique는 위조된 repair 횟수·code 타입을 고정 오류로 거부한다. clarification은 `clarification-gate-v2`에서 caller option ID를 fingerprint로 바꿔 pending·결과에 원문을 보관하지 않는다. v1 pending은 수락하지 않는다. 실제 UI·저장소 migration은 없다.
- Phase 6-04: 합성 관리자 검토 큐 경계(`review-queue-v1`) 구현·합성 검증 완료. 원본 `RepairSummary`를 내부 `critique_repair_summary()`로 분류해 운영 실패 5개 disposition만 메모리 등록하고 성공·`REQUEST_INVALID`은 `NOT_ELIGIBLE` 미등록. 항목은 version·run/request fingerprint·고정 code/stage/disposition·repair flag·idempotency key만 저장. 동일 재등록은 `ALREADY_PRESENT`, 같은 키 다른 결과는 `REVIEW_QUEUE_CONFLICT` 거부, ID 오류는 `REVIEW_QUEUE_INVALID_INPUT` 차단, 위조 요약은 `CRITIQUE_INVALID_SUMMARY` 그대로 차단. 반환·목록 방어적 복사, 첫 등록 순서. coordinator·critique·clarification·harness 변경 없음, disk/DB/network/Ollama/private 접근 0회. 실제 운영 저장·UI·승인·full loop는 미완료. 성능·정확도 수치 없음. 실제 CNU 자료·private bundle 미사용.
- Phase 6-05: 합성 검토 큐의 private snapshot 저장·strict load(`review-queue-snapshot-v1`) 구현·합성 검증 완료. 독립 리뷰 보완으로 code/stage/disposition triple 일치를 단일 critique 분류표(`disposition_for`)로 저장 전·strict load 모두 강제하고, 재서명한 위조 snapshot(`INVALID_REQUEST` 둔갑·stage 있는 `REPAIR_PROVIDER_ERROR`·`recovered_after_repair=True` 포함) 거부 테스트 5개 추가. private 경로를 부모 생성 후·게시 직전에 재검사하고 race 비보장을 문서화. `list_entries()` 고정 메타데이터만 명시적 `.schema-adaptive` 단일 JSON에 저장하고 queue version·항목 수·순서·canonical payload SHA-256으로 strict load. 경로 검사는 기존 recording bundle 동일 validator의 최소 공개 wrapper를 재사용하고 별도 약한 검사를 두지 않음. no-overwrite 기본·원자 게시·임시 정리, 빈 큐 명시 저장 허용, loader는 내부 queue를 자동 수정하지 않음. `운영 실패를 관리자 검토 큐에 저장` 전체 항목은 실제 운영 연결 전까지 미완료. 성능·정확도 수치 없음. 실제 CNU 자료·private recording bundle·Ollama·PostgreSQL 미사용.
- Phase 6-06: 합성 agent workflow 통합(`agent-workflow-v1`) 구현·합성 검증 완료. `BoundedRepairCoordinator.run_detailed()`가 재실행 없이 최종 `HarnessResult`와 `RepairSummary`를 함께 반환하고 기존 `run()`은 위임만 하며 호출 예산을 유지한다. terminal 실패·repair 후 성공/실패는 최종 결과를 보존하고 `REPAIR_PROVIDER_ERROR`는 `final_result=None`으로 부재를 명시한다. `AgentWorkflowCoordinator`는 명시적 `model_run_id`·request·ontology·Published mapping·model·repair provider·주입 queue로 상세 실행 → 내부 critique → 운영 실패 후보 등록을 순서대로 수행하며 `CritiqueResult` 주입 API는 없다. 성공·0행 성공·`REQUEST_INVALID`은 미등록, queue ID는 단일 공개 helper(`validate_queue_identifier`)로 실행 전 검증하고 conflict·내부 불일치는 고정 code로 fail closed하며 기존 entry를 보존한다. 실행 후 queue 등록 실패의 잔여 경계와 side effect를 문서화했다. 합성 테스트 45개로 호출 횟수·최종 결과 보존·queue side effect·비노출·방어적 복사와 위조 nested 계약을 검증했다. 독립 검토에서 summary/result 불일치, 위조 request instance, 비운영 disposition의 등록 상태 위조를 추가 차단했다. `validate → execute → critique → repair loop 구현`만 완료 처리하고 모호한 질문 감지·실제 운영 영속 큐·교정 반영은 미완료. 40건·30건 `NEEDS_HUMAN_REVIEW`와 E3/L5·L1/L3 대기 유지. 실제 Ollama·PostgreSQL·CNU 자료·UI·운영 영속 큐 미사용.
- Phase 6-07: 합성 영속 검토 큐 lifecycle(`durable-review-workflow-v1`) 구현·합성 검증 완료. current snapshot을 전체 계약으로 strict load·복원한 뒤 기존 agent workflow를 한 번 실행하고 `ENQUEUED`만 서로 다른 새 private snapshot에 no-overwrite 게시한다. `ALREADY_PRESENT`·성공·0행 성공·`REQUEST_INVALID`은 무쓰기다. caller queue/critique/disposition 주입 없음. 경로·복원 오류는 실행 전 차단하고 conflict·내부·storage 오류를 분리한다. 반환은 기존 workflow result와 `WRITTEN`/`UNCHANGED`, hash·entry count뿐이다. 독립 검토에서 cross-module private 접근, 내부 오류 오분류, 게시 후 재읽기/결과 생성 실패 경계, hard-link temp 정리 실패 뒤 invalid target 잔류를 수정했다. 신규 합성 테스트 22개, 인접 회귀 101개 통과·1개 skip. overwrite·current pointer·동시성/잠금·운영 DB/UI·관리자 승인·자동 retry는 미구현. 실제 Ollama·PostgreSQL·CNU 자료·private recording bundle·성능 측정 없음.
- Phase 6-08: versioned literal ruleset 기반 합성 모호성 감지(`ambiguity-detector-v1`)와 기존 `clarification-gate-v2` prepare/resolve 연결 완료. 0 match 통과, 단일 match만 opaque signal 생성, multi match conflict, 모든 공개 입력 재검증, Published mapping·DB·ontology ref 검증, resolve 재감지와 context/ruleset drift fail closed를 합성 테스트로 확인했다. 독립 검토에서 canonical 전체 ruleset fingerprint를 option ID에 결합해 trigger·미매치 rule 변경도 과거 pending과 불일치하게 보완했다. 순서만 바뀐 동등 ruleset은 안정적이다. literal 규칙은 NLU가 아니며 QueryIR 반영·실제 UI·session/expiry는 미구현. 실제 Ollama·PostgreSQL·CNU 자료·성능 측정 없음.
- Phase 6-09: 확인 선택의 합성 QueryIR constraint 전달(`clarification-query-constraint-v1`) 완료. 원래 context/ruleset/pending/choice를 기존 detector/gate로 재확인하고 선택 fingerprint를 entity/field semantic ref로 승격해 constraint-aware provider에 방어적 복사로 1회 전달한다. 반환 QueryIR는 기존 validator와 entity 정확 일치 또는 field select/filter 사용 규칙을 통과해야 한다. 독립 검토에서 최소 constraint 필드, 고정 violation code, known raw context reflection 차단과 semantic/physical 이름 겹침 과차단 회귀를 보완했다. 관련 테스트 83개 통과. compile/execute/repair·실제 UI·session/expiry·실제 Ollama/PostgreSQL/CNU 자료·성능 측정 없음.
- Phase 6-10: 확인 선택의 합성 read-only 실행 adapter(`clarification-execution-adapter-v1`) 완료. Phase 6-09 재확인·constraint-aware 생성 뒤 검증된 QueryIR 하나를 내부 one-shot model로 기존 `ExecutionHarness`에 전달해 request/context/IR 검증, compile, SQL policy, read-only execute, result 검증을 그대로 수행한다. caller의 IR/constraint/model/result/SQL 주입 API 없음. compile 이후 기존 실패 code/stage·IR 보존, 초기 단계 실패·IR/request 불일치·상태 모순은 내부 오류로 fail closed. 독립 검토에서 결과 모델 자체에 nested 불변식을 추가했다. 신규 12개·인접 105개 테스트 통과. 실제 UI·session/expiry·repair·Ollama/PostgreSQL/CNU 자료·성능 측정 없음.
- Phase 6-11: 합성 clarification session(`clarification-session-v1`) 완료. `start`가 no-match `NOT_REQUIRED` 또는 hashed bearer token 기반 `PENDING`을 만들고, `submit`은 TTL 확인과 lock 기반 원자 소비 후 취소 재확인 또는 Phase 6-10 실행을 호출한다. 동시 제출·재사용·만료·downstream 실패 후 재실행을 차단한다. 독립 검토에서 clock 실패 소비, lazy expiry 정리, token 길이, deadline 정밀도, package export를 보완했다. 신규 13개·Phase 6-08/09/10 포함 관련 71개 통과. active raw context는 메모리에 남으며 persistence·cross-process/restart·background cleanup·HTTP/UI·인증/rate limit은 없음. 실제 Ollama/PostgreSQL/CNU 자료·성능 측정 없음.
- Phase 6-12: 합성 clarification HTTP sub-app(`clarification-http-v1`) 완료. 독립 FastAPI factory가 server-owned context loader와 같은 session coordinator/provider/read-only executor를 start/submit POST에 연결한다. HTTP caller의 context/IR/model/result 주입 없음. start는 token+실제 submit 가능한 opaque option만, submit은 성공 columns/JSON rows 또는 실패 fixed code/stage만 반환하며 SQL·IR·constraint·trace·raw ID·fingerprint·message를 제거한다. 오류는 fixed code→400/404/409/410/422/502/503/500 표로 변환한다. 독립 검토에서 Phase 6-11 option 이중 fingerprint 결함과 HTTP 재감지 잔여 session 위험을 수정하고 token/nested/row/error 불변식을 강화했다. 신규 HTTP 13개·session 14개, Phase 6-08~12 총 122개 관련 테스트 통과. legacy app mount·React·auth·실제 Ollama/PostgreSQL/CNU 자료·성능 측정 없음.
- Phase 6-13: 실행 가능한 합성 clarification ASGI prototype(`synthetic-clarification-prototype-v1`) 완료. `DEFAULT_SEED` 임시 flat SQLite, canonical `Course.name`·`Instructor.name` context, 결정론 constraint provider를 Phase 6-12 factory에 조립했다. loopback-only module CLI는 제한 port만 받고 reload/access log를 끄며 `.env`·실서비스 설정을 읽지 않는다. 독립 검토에서 CLI 임시 DB 이중 생성·invalid port 반사·실패 cleanup 지연과 semantic 명칭을 수정했다. 신규 7개와 Phase 6-08~12 관련 92개 테스트 통과. legacy app·React·인증·실제 Ollama/PostgreSQL/CNU 자료·성능 측정 없음.
- Phase 6-14: 별도 `/adaptive-prototype` React 흐름 완료. strict runtime parser, `NOT_REQUIRED`/`PENDING`, option 선택·cancel, 실행 성공·실패, generic table, 메모리 token, stale/abort 처리를 구현하고 adaptive 전용 Vite proxy·실행 script를 추가했다. 독립 검토에서 pending session 고아 상태, token 조기 삭제, 느슨한 submit/code/stage, proxy 환경변수 충돌, system Python script와 unknown error suffix 오분류를 수정했다. Node 16개·targeted ESLint·frontend build·backend 7개 통과. 실제 browser 실기동 smoke는 미수행. legacy app·실제 Ollama/PostgreSQL/CNU 자료·성능 측정 없음.
- Phase 6-15: backend+Vite loopback 실제 기동과 Chrome headless DOM·network smoke 완료. 두 option, 강의명·교수자명 각 6행 결과, cancel, no-match, terminal 재제출 control 제거, legacy `/` 유지, 별도 실패 proxy의 redacted 일반 오류를 확인했다. 독립 검토에서 비-2xx 비-JSON proxy 오류를 계약 오류로 오분류하던 client를 code 없는 일반 server 오류로 수정하고 회귀 테스트를 추가했다. 팀 실행·관찰표는 `docs/development-history.md#synthetic-clarification-team-handoff`에 있다. 합성 내부 prototype 범위의 모호성 사용자 확인 구현은 완료지만 L4 사람 검토·실제 auth/provider/PostgreSQL/CNU 자료·배포/외부 시연은 미완료다.
- Phase 6-16: 메인 화면 `DB 변경`에서 legacy 기준선과 server-owned 합성 flat·legacy-schema DB를 선택한다. 선택은 sessionStorage에 두고 strict registry로 재검증하며, 합성 course catalog와 clarification start가 같은 DB ID를 쓴다. catalog는 canonical 15개 field의 Published 상태 mapping, safe compiler/policy, read-only executor를 통과하고 schema fingerprint drift·미등록/중복 ID·변조 response는 redacted fail-closed다. DB 변경은 이전 요청과 선택 강의·경고·hover·modal 상태를 폐기하며 legacy API 실패도 오류·재시도로 표시한다. 관련 backend 39개·frontend 24개, ESLint/build, Chrome smoke 7개 흐름(flat/legacy 각 6개 동일 강의, reload 유지, 동일 DB clarification 6행, unsupported query, legacy backend 부재 오류)이 통과했다. 파일 업로드·mapping publish·실제 DB/Ollama·L4는 미완료다.
- Phase 6-17: standalone SQLite를 단일 read fd에서 32 MiB 제한·header/hash·sidecar/WAL 검사 후 Git ignored private root에 immutable/no-overwrite로 등록한다. private copy만 read-only immutable quick check하고 sample 없는 catalog·mapping proposal을 생성하며 strict loader가 exact 파일·hash/size·JSON 계약·fingerprint를 재검사한다. 상태는 `PENDING_REVIEW`이고 Phase 6-16 registry에는 들어가지 않는다. 독립 검토에서 redirect parent 선검증, source identity, exact JSON, bounded load, fsync를 보강했다. 신규 18개와 관련 기존 15개 통과, Windows source symlink 권한 1개 skip. upload HTTP/UI·mapping 검토/publish·실제 DB/Ollama·L4는 미완료다.
- Phase 6-18: 메인 `DB 변경` modal에 standalone SQLite raw-body upload를 추가했다. HTTP 경계가 `Content-Length`와 실제 stream을 모두 32 MiB로 제한하고 exclusive temp를 정리한 뒤 Phase 6-17 register→strict load만 호출한다. 성공은 `PENDING_REVIEW` sample-free schema/mapping summary만 표시하고 active/pending DB 선택·sessionStorage·course UI·Phase 6-16 registry/catalog/clarification은 바꾸지 않는다. `gpt-6-luna` high review-only에서 찾은 modal 취소 후 upload 고착을 메인에서 재현·수정했고 strict response count/status·빈 query segment도 보강했다. 관련 backend 31개 통과·1개 기존 Windows symlink 권한 skip, frontend 17개·ESLint·build 통과. 실제 browser upload smoke는 미수행이며 mapping 검토/publish·실행 registry 편입·실제 DB/Ollama·L4는 미완료다.
- Phase 6-19: 업로드 성공 아래에서 server-owned proposal을 private draft로 `INIT`하고 field별 `APPROVE`/`REMAP` 뒤 명시적으로 `PUBLISH`한다. immutable mapping version과 `activation.json` 최종 commit을 strict reload한 DB만 동적 registry에 나타나며 catalog·clarification은 Published Mapping 기반 schema-adaptive compiler/AST policy/read-only executor를 쓴다. publish 전·불완전·변조 항목은 숨고 active DB는 자동 변경되지 않는다. Luna review-only의 재게시 원자성·HTTP body 상한·publish/detail 상태 전이 3건을 메인에서 확인·수정했다. 신규 backend 13개, 관련 backend 36개·1 skip·7 subtests, frontend 27개·ESLint·build 통과. 실제 browser 통합 smoke와 L2/L4는 미수행이다.
- Phase 7-01: immutable `dbms-capability-v1`로 SQLite는 inspector/compiler/policy/executor 지원, PostgreSQL은 inspector만 지원한다고 명시했다. `PostgresAdapter`는 server-owned DSN과 PostgreSQL 12+ 단일 schema를 read-only transaction·2초 timeout·parameterized `pg_catalog` query로 검사하며 sample을 수집하지 않는다. table/column/PK/composite FK/generated/identity와 기존 logical affinity를 deterministic catalog로 만든다. Luna review-only의 미등록 ID 반사, cross-schema FK, shallow-frozen list를 고정 오류·schema filter·tuple로 수정했다. 관련 39개·5 subtests와 전체 pytest 725개·5 skip·452 subtests, compileall, diff check 통과. 실 PostgreSQL·CNU·Ollama·runtime 질의·UI는 사용하지 않았다.
- Phase 7-02: 기존 QueryIR subset용 PostgreSQL join compiler(`%s`), postgres dialect AST allowlist, PostgreSQL 12+ read-only executor를 추가하고 `ExecutionHarness` compiler/executor 타입을 protocol로 일반화했다. executor는 server-owned config, read-only transaction, config/executor 중 짧은 timeout, 200행 상한, SQLSTATE `57014`, cleanup과 고정 redaction을 강제한다. PostgreSQL capability의 compiler/policy/executor는 fake connector로 검증한 component 준비 상태로 `true`다. Luna review-only의 timeout 미적용과 외부 `HarnessFailure` redaction 우회를 수정했다. 관련 82개·69 subtests와 전체 pytest 747개·5 skip·459 subtests, compileall, diff check 통과. 실 PostgreSQL·CNU·Ollama·registry/UI 연결은 없다.
- Phase 7-03: `DEFAULT_SEED` 기반 PostgreSQL fixture와 explicit opt-in live smoke CLI를 추가했다. loopback·전용 DB DSN만 허용하고 실행별 생성 schema에서 inspector→mapping→graph→compiler/policy/executor 두 QueryIR를 검증한 뒤 cleanup한다. Luna review-only의 setup 실패 cleanup·느슨한 FK 검증을 수정했고, 실제 PostgreSQL 18에서 드러난 FK catalog query 호환성과 runtime schema 고정을 `generate_subscripts`·transaction-local `search_path`로 보완했다. 관련 57개·18 subtests와 전체 pytest 762개·5 skip·470 subtests, compileall, diff check 통과. 임시 cluster/schema 제거 완료. 실제 CNU·기존 PostgreSQL 데이터·Ollama·registry/UI 연결은 없다.
- Phase 7-04: Python server code 전용 PostgreSQL profile을 startup에서 sample 없는 catalog·승인 mapping·fingerprint·ontology·physical graph로 검증하고 기존 registry/catalog/clarification에 DB별 compiler/policy/executor router로 연결했다. request context와 실행 직전 fingerprint drift를 재검사하며 DSN/schema/physical/SQL/driver text를 공개하지 않는다. Luna review-only의 compiler/policy redaction과 catalog executor 우회를 수정했고 실제 PostgreSQL 18 HTTP smoke에서 registry·catalog 6행·clarification 성공 및 `Decimal` 직렬화 수정을 확인했다. 전체 pytest 772개·5 skip·470 subtests, compileall, diff check 통과. 기본 app에는 profile이 없어 PostgreSQL 연결 0회다. 실제 CNU·Ollama·frontend 변경 없음.
- Phase 7-05: loopback CLI가 database ID별 고정 private profile/mapping을 기존 Git ignored·untracked guard와 256 KiB bounded read로 strict load한다. exact JSON·mapping SHA-256·명시적 `PUBLISHED`·별도 제한된 DSN 환경변수를 요구하고, 전부 성공한 profile만 Phase 7-04 factory에 주입한다. option 없는 기본 app은 loader/environment/PostgreSQL 접근 0회다. Luna actionable 결함은 없었고, 메인 검토에서 제한 전 전체 파일 read를 공용 reader의 `max_bytes`로 수정했다. 관련 51개·2 skip, 전체 pytest 780개·5 skip·470 subtests, compileall, diff check 통과. 실제 PostgreSQL·CNU·Ollama·frontend 변경 없음.
- Phase 7-06: frontend strict registry에 `schema=postgres`와 backend 동등 ASCII label 검증을 추가했다. 메인 `DB 변경`은 server registry만 렌더링하고 명시적 적용·상태 초기화·sessionStorage membership 재검증을 유지하며, 동일 ID를 catalog와 clarification에 전달한다. profile이 registry에서 사라지거나 registry가 invalid/unavailable이면 legacy로 fallback한다. DSN·physical metadata·SQL browser 노출 없음. Luna의 trailing newline 지적은 현재 Node에서 재현되지 않아 미적용했다. Node 30개, ESLint, build, 전체 pytest 780개·5 skip·470 subtests, compileall, diff check 통과. 실제 PostgreSQL browser smoke·Ollama·CNU 자료 없음.
- Phase 7-07: 기존 `OllamaQueryModel`을 clarification `ConstraintAwareProvider`에 연결했다. 확인된 entity/field 하나에 맞춰 ontology·Published Mapping의 semantic context만 축소하고 모델을 1회 호출한다. 응답은 기존 QueryIR constraint 검증·compiler·AST policy·DB별 executor를 우회하지 않는다. factory 명시 주입만 허용하고 기본 deterministic provider·network 0회를 유지한다. Luna가 찾은 nested ontology/mapping extra-key 허용을 메인에서 재현해 exact nested shape 검사로 수정했다. 관련 113개, 전체 pytest 816개·5 skip·470 subtests, compileall, diff check 통과. live Ollama·CLI 설정·raw SQL fallback·재학습·실제 PostgreSQL/CNU·frontend 변경 없음.
- Phase 7-08: prototype CLI `--ollama-model-tag`(필수 스위치)·`--ollama-base-url`(127.0.0.1/[::1] loopback만)·`--ollama-timeout`(1~120초)으로만 Ollama provider를 켠다. startup은 proxy·redirect 차단 transport를 가진 기존 `OllamaQueryModel`을 조립만 하고, option 없으면 기존 결정론 provider다. synthetic transport와 loopback 임시 server로만 검증했다. 상세는 계획 문서 변경 이력과 ADR-055.
- Phase 7-09: `python -m backend.adaptive.api.ollama_prototype_smoke --check|--live-synthetic --ollama-model-tag <tag> --ollama-request-format <format>`로 합성 case 6건을 in-process 실행한다. 로컬 `text2sql_qwen:latest` live 1회에서 경계는 `COMPLETED`(호출 4/4)였지만 submit 4건 모두 모델이 질문 전체를 `equals` filter 값으로 복사해 Phase 6-09 reflection guard가 `CONSTRAINT_INVALID_QUERY_IR`로 거부했다. guard는 유지했다. 이 실행은 내장 SYSTEM 주입 조건이었고 7-10 재실행에서 성공 실행 경로가 live 통과했다. 결과는 `docs/experiment-results.md#phase7-09-ollama-prototype-live-smoke-results-2026-09-28`, ADR-056.
- Phase 5-18: 평가 전용 직접 SQL arm(`evaluation/direct_sql.py`)과 QueryIR paired 비교(`evaluation/arm_comparison.py`)를 추가했다. 동적 schema prompt·보정 없는 추출·read-only SELECT 정책·catalog 읽기 제한 executor·위치 무관 결과 비교를 쓰며 oracle 40×3이 120/120으로 자기 일관성을 확인했다. 실제 모델 수치는 아직 없다. ADR-057.
- Phase 5-19: `arm_live_cli`로 두 arm을 live 실행했으나 모델 내장 CNU SYSTEM이 주입된 조건이어서 수치·해석을 5-20에서 정정했다.
- Phase 5-20: `--ollama-request-format alpaca-raw-no-system-v1`로 내장 SYSTEM을 제거하고 재실행했다. 잠정 IR 정답 93/120, SQL 72/120(둘 다 68·IR만 25·SQL만 4·둘 다 오답 23), normalized에서 IR 31 대 SQL 11. 결과 문서 `docs/experiment-results.md#phase5-20-decontaminated-arm-comparison-results-2026-09-29`, ADR-058.
- Phase 7-10: prototype·smoke CLI에서 Ollama를 켤 때 `--ollama-request-format`(`model-template`|`alpaca-raw-no-system-v1`)이 필수다. `text2sql_qwen`에는 `alpaca-raw-no-system-v1`을 쓴다. no-system 재실행에서 성공 실행 경로가 처음 live 통과했으나 `Course.name`은 질문 전체 `equals` filter 복사로 guard 거부가 남았다. ADR-059.
- Phase 7-11: `OllamaQueryModel(request_format=...)`이 형식을 생성자에서 한 번 적용하고 provenance에 선언한다(`model-template` hash 유지). non-callable transport는 모든 형식에서 생성 시점에 거부된다. 5-12~5-14 recording·결과는 내장 SYSTEM 주입 조건이었다. ADR-060.
- Phase 5-21: `qwen2.5-coder:7b-instruct`(`model-template`) 실행에서 SQL 정답 90/120, IR 0/120. base 모델이 IR 출력을 모두 code fence로 감쌌고(안쪽 유효 QueryIR 28/40) IR arm에는 fence 제거 규칙이 없어 방식 비교로 쓸 수 없다. 결과 문서 `docs/experiment-results.md#phase5-21-base-instruct-arm-comparison-results-2026-09-29`.
- Phase 5-22: `arm_live_cli --ir-extract single-fence-v1`로 두 arm에 같은 fence 규칙을 적용했다. fine-tune IR 93·SQL 72, base IR 78(26/26/26)·SQL 90(34/24/32). IR은 schema 구조와 무관하게 일정하고 직접 SQL은 정규화 schema에서 떨어진다. 결과 문서 `docs/experiment-results.md#phase5-22-symmetric-extraction-results-2026-09-29`.
- Phase 5-23: 주장을 'schema 변화에 대한 정확도 일관성'으로 한정(ADR-061)하고 `coded`·`hybrid` 변형을 더한 `extended-v1` suite(ADR-062, `arm_live_cli --fixture-suite extended-v1`)로 재실행했다. IR은 두 모델 모두 5종 spread 0(31×5, 26×5), 직접 SQL은 fine-tune 30·11·31·8·10, base 34·24·32·20·23. 기본 `core-v1`은 이전과 같다. 결과 문서 `docs/experiment-results.md#phase5-23-schema-variation-results-2026-09-29`.
- Phase 8-01: 관리자 UI 첫 단계. mapping 검토 패널에 버전 이력과 이전 immutable 버전 재활성화(`ACTIVATE`, 2단계 확인)를 추가했다. `activation.json`만 바뀌고, `current.json`의 게시 digest 기록에 있는 버전만 publish 수준 재검증 뒤 활성화된다. 시작된 clarification session은 활성 mapping이 바뀌면 실행이 거부된다. ADR-063. Phase 8 분할과 다음 단계(8-02 schema fingerprint 비교)는 `docs/development-history.md#prompts-phase8-01-mapping-version-rollback`.
- Phase 8-02: 새 등록(검토 대기 포함)을 기준 게시·활성 등록과 비교하는 `GET /api/adaptive/v1/registrations/compare`와 업로드 결과 아래 비교 패널. table·column·FK 변화, 활성 mapping field 영향, 그대로 옮긴 mapping의 실행 검사 통과 여부를 보여주며 파일을 쓰지 않는다. ADR-064.
- Phase 8-03: `CARRY_OVER`로 새 등록의 첫 검토를 기준 활성 mapping 승계로 시작한다. 영향 없는 field만 사유 있는 REMAP, 영향 field는 PENDING·`STALE`(`succession.json`), publish 조건 불변. 패널에 승계 시작·STALE 표시. ADR-065.
- Phase 8-04: 게시·활성 private 등록에서만 서버 고정 test query 5개(`GET /api/adaptive/v1/registrations/test-queries`, `.../test-query?database_id=..&query_id=..`)를 기존 실행 경로로 돌려 최대 5행·행 수·결과 hash를 보여준다. 게시 상태 검토 패널 안의 `TestQueryPanel`. ADR-066.
- Phase 8-05: VERIFY(`POST .../verified-queries`)는 서버 실행 hash를 기준선으로 기록하고, 회귀(`GET .../regression?database_id=..&mapping_version=..`)는 활성·비활성 게시 버전에서 고정 query를 다시 실행해 `MATCH`/`CHANGED`/`FAILED`/`UNVERIFIED`를 보여준다. publish·재활성화 직후 자동 실행, 버전 이력의 회귀 확인 버튼. 경고만. ADR-067.
- 현재 로컬에는 fine-tuned tag만 확인되어 base instruct 평가는 정확한 tag가 별도로 준비될 때 진행한다. E3·L5 팀 결정 대기와 합성 개발은 병행한다.
- `docs/development-history.md#desktop-test-handoff`는 초기 데스크톱 이전 시점의 기록이다.

- Phase 7-12: legacy 챗봇의 view `v_course_info`를 profile v2로 등록한다. mapping 12개, 미지원 3개(`CourseOffering.year`·`semester`·`Department.code`), 변환은 학점 `SLASH_FIRST_NUMERIC`와 시간 `TIME_TO_HHMM`이다. private profile은 `.schema-adaptive/postgres-profiles/legacy-view/`에 있다. DSN은 `SCHEMA_ADAPTIVE_POSTGRES_DSN_LEGACY_VIEW` 환경변수로 주고, CLI는 `python -m backend.adaptive.api.legacy_view_profile --check | --write --live | --smoke --live`다. 부분 profile의 catalog는 닫혀 있다(7-14에서 화면 연결). legacy prompt의 `credit_hours = '3'` 비교가 0행이라는 관찰은 7-15 비교 후보다.

- Phase 7-13: `POST /api/adaptive/v1/chat/query` `{database_id, query}` → `adaptive-chat-query-v1`. 상태는 `SUCCEEDED`(카드 행과 understood filter), `UNSUPPORTED`(미지원 field label), `NOT_UNDERSTOOD`(고정 code)다. prototype을 `--postgres-profile-id legacy-view`와 Ollama 옵션으로 띄워야 동작하며, Ollama가 없으면 503이다. 시간 filter 값은 `HH:MM`으로 정규화하고 표현 불가 값은 거부한다. 문자열 `equals` 경향(`김 교수님` → 0행)은 7-15 비교 항목이다.

- Phase 7-14: `ChatPopup`은 DB에 따라 경로를 고른다(Legacy 기준선 → 기존 SQL, postgres → QueryIR, 그 밖 → 미지원 안내). `dev:all`은 `dev_launcher`로 legacy `.env`의 DB·Ollama를 재사용한다(상태 한 줄 출력). 부분 mapping catalog는 409 `SYNTHETIC_CATALOG_UNSUPPORTED`다.

- Phase 7-15: `python -m backend.adaptive.evaluation.legacy_chat_compare --live`(`dev:all` 실행 중, DSN 환경변수 필요)로 30문항을 비교한다. 원자료는 `.schema-adaptive/reports/phase7-15/`에, 요약은 `--summarize`로 본다. ONTOLOGY 22문항 수용 가능은 legacy 17, QueryIR 11, `limit200+contains` 13이다. 미지원 field 거절은 QueryIR 3/3, legacy 0/3이다. 결정 항목(limit, `contains` 완화, 요일·이수구분 값 정규화, field 별칭, 학년 ontology 확장)은 결과 문서 §6에 있다.

- Phase 7-16: 챗봇은 limit 200 고정, 이름형 문자열 `contains`, profile 선언 `KOREAN_WEEKDAY` 요일 정규화를 쓴다. PostgreSQL `contains`는 LIKE 와일드카드를 escape한다. 재비교 결과 ONTOLOGY 수용 가능은 QueryIR 14, legacy 17이다. 남은 후보(field 별칭, 이수구분 값 사전, 학년 ontology, prompt)는 결과 문서 §5에 있다.

- Phase 7-17: profile v3 `anchor_table`이 있으면 모든 JOIN이 anchor에서 시작하는 `LEFT JOIN`이다(ADR-073). `legacy-base`(기본 table 4개) profile은 `python -m backend.adaptive.api.legacy_view_profile --write|--smoke --live --target base`로 등록·확인한다. DSN 변수는 `SCHEMA_ADAPTIVE_POSTGRES_DSN_LEGACY_BASE`이고 `dev:all`이 자동으로 설정한다. view 경로와 답이 같다.

- Phase 8-06: profile v4 `value_dictionaries`로 field별 값 사전을 선언한다(ADR-074). legacy 두 profile은 이수구분 사전을 가진 v4다. 재등록은 `--write --live --replace --target view|base`이고, 저장 값이 DB에 없으면 `VALUE_MISSING`이다. ONTOLOGY 수용 가능은 QueryIR 15, legacy 17이다.

- Phase 8-08: `python -m backend.adaptive.evaluation.variant_benchmark --check`는 오프라인 field 점수다. `--live-synthetic --label <id>`는 `SCHEMA_ADAPTIVE_SYNTHETIC_POSTGRES_DSN`(loopback, DB `schema_adaptive_synthetic`)이 필요하다. 로컬 전용 DB는 2026-09-30 사용자 승인으로 생성했다. oracle 240/240, 규칙 205/240이다. 다음 AI 제안은 `VariantMappingAnswer`를 낸다.

- Phase 8-09: `python -m backend.adaptive.evaluation.ai_mapping --propose --label <id>`는 로컬 Ollama `qwen2.5-coder:7b-instruct`를 부른다(loopback). 그 결과 파일을 `variant_benchmark --live-synthetic --ai-proposals <file>`에 넘겨 실행한다. 결과는 AI 223/240, 규칙 205/240이다. 값 사전과 의미 함정은 미해결이다.

- Phase 8-10:
  - 제안 실행: `ai_mapping --propose --suite dev|holdout --pipeline v1.1|v2 [--notes] --label <id>`.
  - 채점 실행: `variant_benchmark --live-synthetic --suite <s> --ai-proposals NAME=PATH …`.
  - 결과: 기존 235/240, 보류 116/120(v2+설명).
  - 의미 함정은 미해결이다.

- Phase 8-07(축소판):
  - PostgreSQL 등록 결과 아래 "값 사전(코드표·약어)"에서 저장 값의 뜻·약어를 넣고 "사전 적용해 게시"로 반영한다.
  - 후보: 규칙 / 규칙+AI. 충돌은 막고 다른 글자 속 포함은 경고한다.
  - 저장: `.schema-adaptive/postgres-registrations/<id>/dictionaries.json`.
- Phase 8-15:
  - "DB 변경" 창에서 PostgreSQL 연결·schema를 골라 등록하고, 같은 검토 화면에서 검토·게시하면 재시작 없이 목록·챗봇에 추가된다.
  - 연결은 `backend/.env`의 `DATABASE_URL`(기본)과 `PG_SOURCE_<이름>=<DSN>`이다.
  - 저장 위치: `.schema-adaptive/postgres-registrations/<id>`(등록·draft·게시 기록), `.schema-adaptive/postgres-profiles/<id>`(profile 쌍).
  - 없는 정보는 "이 DB에 없음"(IGNORE)으로 빼고 게시한다. 변환·기준 table은 서버가 정한다.
- Phase 8-14:
  - 챗봇 화면의 SQL 패널에 QueryIR 경로의 실행 SQL(값을 채운 보기용)이 나온다.
  - `dev:all`은 기본으로 켜져 있고 `backend/.env`의 `SHOW_SQL=0`으로 끈다. 직접 실행할 때는 `--show-sql`을 붙인다.
- Phase 8-13:
  - 챗봇이 DB 실제 값(값 500개 이하 text field)과 값 사전 별칭으로 질문을 검사해 모델에 힌트를 주고, 없는 field 이름·잘못 들어간 값을 닫힌 규칙으로 고친다.
  - 레거시 약어는 `legacy_view_profile.py`의 값 사전 데이터에 있다. 바꾸면 `--write --live --replace --target base|view`로 private profile을 다시 쓴다.
  - 새 질문 12/12, 7-15 해석 범위 19/22(레거시 17). 사용 중인 서버는 재시작해야 반영된다.
- Phase 8-12:
  - 업로드 시 로컬 모델이 column을 고르고 첫 후보로 둔다. 실패하면 규칙 제안을 쓴다.
  - 켜는 법: `dev:all` launcher는 loopback Ollama면 기본으로 켠다(`backend/.env`의 `MAPPING_MODEL`로 tag 변경). 직접 실행 시 `--mapping-model-tag qwen2.5-coder:7b-instruct`.
  - 화면: 업로드 요약에 `제안: AI 제안/규칙 기반/규칙 기반(AI 사용 불가)`, 검토 후보에 `AI 제안`.
  - live smoke(이름 변경 합성 DB 1개): AI 15/15, 규칙 1/15.

## 실행

팀 논의용 전체 실행은 `npm run dev:all`(legacy `:8000` + 합성 prototype `:8123` + Vite `:5173`). legacy 강좌 DB는 팀이 구성한 유사 데이터라는 사용자 확인이 있다(`docs/development-history.md#synthetic-clarification-team-handoff`).

## 다음 작은 작업

- 다음 순서는 legacy 챗봇 연결이다.
  - 7-13: 완료.
  - 7-14: 완료.
  - 7-15: 완료.
  - 7-16: 3종 정책 적용·재비교 완료.
  - 7-17: LEFT JOIN·기본 table 매핑 완료.
  - 8-06: 값 사전 완료.
  - 8-08: 합성 변형 benchmark·규칙 기준선 완료.
  - 8-09: AI mapping 제안 비교 완료(v1.1).
  - 8-10: AI 제안 v2·관리자 설명·보류 변형 완료.
  - 8-12: 업로드 흐름 AI 제안 연결 완료.
  - 8-13: 챗봇 값 힌트 완료(사용자 요청으로 8-07보다 먼저).
  - 8-07 축소판 완료(PostgreSQL 값 사전).
  - 8-16 등록 목록·browser smoke·시연 절차 완료.
  - 8-18 대상 학년(ontology 버전 2, 추가형 호환) 완료. 버전 1로 화면 등록한 PostgreSQL DB는 대상 학년이 "이 DB에 없음"이며, 쓰려면 새 ID로 다시 등록한다.
  - 8-17 다른 학교 합성 DB(`other_school`, 합성 DB에 남겨 둠) 화면 검증 완료. 다시 만들기: DSN을 `SCHEMA_ADAPTIVE_SYNTHETIC_POSTGRES_DSN`에 두고 `python -m backend.adaptive.fixtures.other_school create`. 화면에서 쓰려면 `backend/.env`에 `PG_SOURCE_OTHER_SCHOOL=<합성 DB DSN>`을 사용자가 직접 추가한다. ontology에 없는 조건 문제는 8-18(대상 학년)·8-20(모르는 조건)에서 다뤘다.
  - 8-19 DB별 확장 항목 완료(ADR-087). 관리자가 화면 "확장 항목" 패널에서 그 DB만의 항목(예: 강의실)을 선언하고 게시한다. profile v5 `extension_fields`.
  - 8-20 모르는 조건 처리 완료(ADR-088).
    - 공통 항목 밖 개념 목록 `backend/adaptive/ontology/outside-concepts.yaml`(데이터, 11개). 질문이 이 개념을 말하는데 그 DB에 확장 항목이 없으면 실행하지 않고 `UNSUPPORTED`(`{concept, label}`)로 답한다.
    - 0건이면 같음·포함 조건을 하나씩 1행 query로 다시 실행해, 혼자서도 0행인 조건을 `unmatched`로 알린다.
    - 값 힌트 cache key에 게시 mapping 버전을 넣었다. 그전에는 다시 게시해도 서버 재시작 전까지 예전 힌트를 썼다.
    - 결과: [`phase8-20-unknown-conditions-results-2026-10-01.md`](experiment-results.md#phase8-20-unknown-conditions-results-2026-10-01).
  - 다음 순서(ADR-089): 8-21 형식이 다른 PostgreSQL DB 대응 점검 → 팀원 DB 평가 실행 도구 → Phase 9 진입(L1 검토 표 준비, 팀원 모사 DB). 뒤로 미룬 것: PostgreSQL 버전 되돌리기·test query, 확장 항목의 변환·값 사전, 개념 목록 보강. 업로드 SQLite DB 확장은 범위 밖이다.
  - 후속: 업로드 SQLite DB 확장, PostgreSQL 버전 되돌리기·test query, 공통 약어 묶음·사용 기록 제안.
    - 8-07 축소판 범위: 검토, 코드표→값 사전, test query 미리보기, 버전. 업로드 DB의 변환·값 사전·부분 게시(`unsupported`)도 포함한다.
    - 8-13 뒤 추가: 값 사전 별칭(약어) 입력. 업로드한 SQLite DB도 QueryIR 챗봇·값 힌트를 쓰도록 연결한다(지금은 PostgreSQL profile만).
    - 약어 입력에는 규칙 후보(단어별 첫 글자)·로컬 AI 후보·충돌 검사·관리자 승인을 붙인다(ADR-080). 공통 약어 묶음과 사용 기록 기반 제안은 후속이다.
    - 그 뒤: 8-11(이론·실습 시간 등), L1 뒤 prompt·별칭, Phase 9.
0. Phase 8 남은 항목:
   - FK 없는 관계 선언(benchmark에서 필요할 때).
   - 8-07 관리자 화면 통합(값 사전 편집 포함).
   - 8-01~8-05 관리자 흐름의 browser smoke(private storage 격리 방법 필요).
1. base 모델에 대한 QueryIR prompt 적합성(출력 구조 실패 variant당 8건) 비노출 진단. 주장 범위는 ADR-061로 한정됐고 확정 수치는 L1·L3 뒤다.
2. 목록 전체 질문에 불필요한 filter를 붙이는 모델 출력 경향(Course는 질문 전체 `equals`로 guard 거부, Instructor는 부분 문자열 `contains`로 0행)과 `CONSTRAINT_INVALID_QUERY_IR`의 reflection 거부 미구분은 prompt·진단 후보다. 내장 SYSTEM 제거 후에도 남았다. guard 완화는 하지 않는다.
3. Phase 6-20의 합성 SQLite browser end-to-end smoke는 별도 잔여 작업이며 L4 승인이 아니다.
4. 실 PostgreSQL 학교 DB publish·질의 전 L2, 실제 CNU 자료 사용·공유 전 E3, 외부 시연 전 L5를 적용한다.
5. 운영 검토 큐 current pointer·동시성/잠금과 새 30건 사람 검토·모델 성능 관문은 별도다.

## 새 대화의 최소 시작 순서

1. 저장소 루트에서 `AGENTS.md`와 이 문서, 계획 문서의 **Phase 7 현재 체크포인트·ADR-048/049/050/051/052/053/054/055/056/057/058/059/060/061/062/063/064/065/066/067·다음 시작점**, `docs/human-validation-checkpoints.md`의 E3/L1/L2/L3/L4/L5를 읽는다.
2. `git status --short`, 최근 commit, 관련 코드·테스트를 확인한다. 문서는 방향을 알려주지만 구현 상태와 테스트 통과를 대신하지 않는다. Phase 5-18·5-19·5-20·7-08·7-09 범위는 `docs/development-history.md#prompts-phase5-18-direct-sql-baseline-arm`, `docs/development-history.md#prompts-phase5-19-live-arm-comparison`, `docs/development-history.md#prompts-phase5-20-ollama-system-decontamination`, `docs/development-history.md#prompts-phase7-10-runtime-ollama-request-format`, `docs/development-history.md#prompts-phase7-11-request-format-in-model`, `docs/development-history.md#prompts-phase5-21-base-instruct-arm-comparison`, `docs/development-history.md#prompts-phase5-22-symmetric-fence-extraction`, `docs/development-history.md#prompts-phase5-23-schema-variation-claim-scope`, `docs/development-history.md#prompts-phase8-01-mapping-version-rollback`, `docs/development-history.md#prompts-phase8-02-schema-fingerprint-compare`, `docs/development-history.md#prompts-phase8-03-stale-carry-over`, `docs/development-history.md#prompts-phase8-04-test-query-preview`, `docs/development-history.md#prompts-phase8-05-verified-query-regression`, `docs/development-history.md#prompts-phase7-08-server-owned-ollama-startup`, `docs/development-history.md#prompts-phase7-09-ollama-prototype-live-smoke`에 둔다. 이 sandbox에서는 pytest 기본 임시 경로 쓰기가 막힐 수 있어 `--basetemp`와 `-p no:cacheprovider`를 쓴다.
3. 현재 작업에 필요한 파일만 더 읽고 해당 범위의 테스트부터 실행한다. 이전 대화 전체나 저장소 전체를 다시 요약할 필요는 없다.

실제 질문·응답, DB 행·dump, 모델 weight, 비밀값, private recording은 Git이나 공개 문서에 복사하지 않는다.

2026-10-13 Phase 9-16: 사용자 결정으로 공통 ontology v3의 개설 연도 사용 중단(ADR-101), v1/v2 기록·gold·학기·레거시 기준선 유지; 현재 runtime 연도 질문 거절, targeted 결과·평가 영향은 docs/phase9-16-retire-offering-year-results-2026-10-13.md, Luna/opencode는 no-live-model 제약으로 대기, 팀 공유 필요·사람 관문 승인 없음.

---

<a id="prompts-phase1-01-jsonl-loader-quality-report"></a>

## 원본: prompts/phase1-01-jsonl-loader-quality-report.md

# 다음 개발 프롬프트: Phase 1 JSONL loader와 데이터 품질 report

다음 단계인 기존 학습 데이터셋 JSONL loader와 결정적 데이터 품질 report를 개발해줘.

작업 전 반드시 다음을 확인해.

- `docs/schema-adaptive-text-to-sql-plan.md` 전체
- 기존 `dataset.csv` 분석 결과와 데이터 활용 방침
- 현재 `QueryIR` 계약과 테스트 구조
- 저장소의 Conventional Commit 기록
- 작업 시작 시 working tree 상태

## 목표

확장자는 `.csv`지만 실제로는 한 줄에 JSON 객체 하나가 저장된 기존 학습 데이터셋을 안전하게 읽고, 원본을 변경하지 않으면서 중복·충돌·파싱 오류를 재현 가능한 JSON report로 생성한다.

이 단계는 데이터셋을 바로 학습하거나 Query IR로 변환하는 단계가 아니다. 이후 SQL AST 분석과 변환 작업이 신뢰할 수 있는 입력 계층을 만드는 것이 목적이다.

## 입력 데이터 원칙

- 사용자가 제공한 실제 데이터셋 경로는 실행 시 인자로만 받는다.
- 실제 데이터셋이나 실제 질문·SQL 원문을 저장소에 복사하거나 커밋하지 않는다.
- 구현 전에 실제 레코드의 key와 값 타입을 read-only로 확인한다.
- 문서에 기록된 통계는 검증 기준이지 코드에 하드코딩할 값이 아니다.
- 실제 파일을 사용할 수 없는 테스트 환경에서는 작은 합성 JSONL fixture를 사용한다.
- UTF-8과 UTF-8 BOM을 명시적으로 처리하고, 지원하지 않는 인코딩이나 잘못된 JSON은 줄 번호가 포함된 issue로 보고한다.

현재 문서에 기록된 참고 통계:

- 전체 레코드 5,385개
- 고유 질문 5,354개
- 고유 SQL 5,124개
- 중복 질문 그룹 19개
- 같은 질문에 서로 다른 SQL이 연결된 충돌 그룹 3개

## 이번 범위

- versioned dataset record Pydantic 계약 정의
- source line number와 원본 field를 보존하는 streaming JSONL loader
- 빈 줄, malformed JSON, 객체가 아닌 JSON, 필수 field 누락, 잘못된 field type에 대한 구조화된 issue
- 전체·유효·무효 레코드 수 집계
- 고유 instruction·질문·SQL 수 집계
- 정확히 동일한 record 중복 탐지
- 동일 질문에 같은 SQL이 반복되는 duplicate group 탐지
- 동일 질문에 서로 다른 SQL이 연결된 conflict group 탐지
- 결과 순서가 항상 같은 deterministic JSON report
- report schema version, 입력 파일 fingerprint, 생성 도구 version 기록
- 사람이 검토할 수 있는 CLI 진입점
- 합성 fixture와 단위·CLI 테스트
- 계획 문서의 Phase 1 체크박스, 현재 체크포인트, 변경 이력 갱신

## 정규화 경계

- 원본 문자열은 변경하지 않고 보존한다.
- 중복 식별용 question key는 Unicode NFC, 줄바꿈 통일, 앞뒤 공백 제거까지만 허용한다.
- SQL은 이번 단계에서 의미 정규화하거나 AST로 재작성하지 않는다.
- SQL duplicate/conflict 판정은 원본 SQL에 줄바꿈 통일과 앞뒤 공백 제거만 적용한 key를 사용한다.
- 대소문자, 내부 공백, quote, literal 값이 다른 SQL을 임의로 같은 SQL로 취급하지 않는다.
- normalization 전후 값이 달라졌는지는 report에서 추적 가능해야 한다.

## 개인정보와 report 정책

- 기본 JSON report에는 전체 실제 질문·SQL 원문을 무제한으로 복제하지 않는다.
- group 식별에는 안정적인 hash와 source line 목록을 사용한다.
- 사람이 검토할 preview가 필요하면 명시적인 옵션과 제한된 group·variant 개수·문자 길이를 적용한다.
- 입력 파일 절대 경로는 report에 저장하지 않고 파일명 또는 사용자가 지정한 dataset id만 기록한다.
- fingerprint는 파일 내용 기반으로 계산해 같은 입력을 재현 가능하게 식별한다.

## 오류 정책

- 한 줄의 오류 때문에 기본 분석 전체를 중단하지 않는다.
- 각 issue는 최소한 code, line number, field, message를 가진다.
- 파일을 열 수 없음, 지원하지 않는 report version, 출력 실패는 명시적 fatal error로 처리한다.
- issue와 group 정렬 기준을 계약으로 고정해 같은 입력이 같은 report를 생성하게 한다.
- timestamp처럼 실행마다 달라지는 값은 결정성 비교 대상 report core와 분리하거나 주입 가능하게 한다.

## CLI 예시

구체적인 module 경로는 현재 패키지 구조를 조사해 정하되, 사용 방식은 다음 수준으로 단순하게 유지한다.

```powershell
python -m backend.adaptive.dataset inspect `
  --input "C:\path\to\dataset.csv" `
  --dataset-id cnu-original-v1 `
  --output ".schema-adaptive\reports\dataset-quality.json"
```

CLI는 원본 파일을 수정하지 않아야 하며, 기존 report를 덮어쓰는 정책은 명시적으로 정한다. 가능하면 임시 파일 작성 후 원자 교체를 사용한다.

## 필수 테스트

- 정상 JSONL 여러 행 loading
- `.csv` 확장자의 JSONL loading
- UTF-8 BOM 처리
- 빈 줄 처리
- malformed JSON과 정확한 line number
- JSON object가 아닌 행 거부
- 필수 field 누락과 잘못된 type
- 동일 record 중복
- 동일 질문·동일 SQL duplicate group
- 동일 질문·서로 다른 SQL conflict group
- NFC·줄바꿈·경계 공백 normalization
- SQL 내부 공백이나 대소문자를 과도하게 합치지 않음
- deterministic issue/group/report 순서
- 같은 입력의 동일 fingerprint와 report core
- preview 기본 비활성화 및 개수·길이 제한
- CLI JSON 출력과 입력 파일 불변
- input/output 동일 경로·symlink·hardlink에서 `--force`를 사용해도 원본 보존
- escaped lone surrogate를 행 issue로 처리하고 다음 행 분석 지속
- 기존 전체 테스트 회귀 없음

## 제외 범위

- SQL 의미 정규화
- sqlglot 기반 SQL AST 분석
- 기존 SQL을 Query IR로 변환
- 변환 실패 수동 검토 목록
- 실제 식별값 가명화
- train/validation/test split
- 공통 평가 질문 30~50개 작성
- 기대 semantic result 작성
- evaluation runner
- RecordedModel 또는 실제 모델 실행
- Ollama/PostgreSQL
- repair loop와 관리자 UI

## 설계 원칙

- 실제 레코드 key를 확인하기 전에 schema를 추측하지 않는다.
- loader, 분석기, report serializer, CLI 책임을 분리한다.
- Pydantic 계약으로 record, issue, duplicate/conflict group, summary, report를 명시한다.
- 통계 계산은 입력 순서와 무관하게 결정적이어야 한다.
- 메모리 사용을 불필요하게 키우지 않되 5천여 건 규모에서 과도한 구조를 만들지 않는다.
- 기존 legacy 시스템과 현재 schema-adaptive 실행 경로를 변경하지 않는다.

## 서브 에이전트

- 메인 에이전트가 먼저 실제 데이터 key, normalization 경계, 개인정보 report 정책을 정의한다.
- 독립 구현을 분배한다면 loader/contracts와 report/tests의 수정 파일을 겹치지 않게 한다.
- 메인 에이전트가 CLI 통합, 실제 통계 확인, 전체 테스트를 담당한다.
- 구현 후 별도 리뷰 에이전트가 malformed input, 결정성, 중복·충돌 정의, 개인정보 노출, 원본 불변성을 점검한다.
- 리뷰 지적 수정 후 전체 테스트를 다시 실행한다.

## 완료 기준

- 실제 `dataset.csv`를 read-only로 끝까지 분석 가능
- 문서의 기존 통계와 실제 report 차이를 확인하고 차이가 있으면 원인을 기록
- 오류 행이 구조화된 issue로 남고 나머지 유효 행 분석은 계속됨
- duplicate와 conflict가 서로 구분됨
- 같은 입력·설정은 같은 report core와 fingerprint 생성
- 실제 원문 데이터가 저장소에 추가되지 않음
- 기존 테스트 회귀 없음
- `compileall` 통과
- `git diff --check` 통과
- working tree clean

## 문서

- Phase 1의 JSONL loader와 데이터 품질 report 상태 갱신
- 실제 통계와 문서의 기존 통계 비교 결과 기록
- normalization·개인정보·report 결정이 중요하면 ADR 추가
- 다음 체크포인트를 Phase 1의 SQL 정규화 규칙과 sqlglot AST 분석으로 지정
- 날짜가 포함된 변경 이력 추가

## 커밋

- 과거 Conventional Commit 형식을 따른다.
- contracts/loader, report/CLI, 테스트, 문서를 기능별로 분리 커밋한다.
- 각 커밋 전 staged diff를 확인한다.
- 실제 데이터셋과 생성된 민감 report를 커밋하지 않는다.
- 원격 push는 하지 않는다.

이번 단계까지만 개발하고 SQL AST 분석, SQL→Query IR 변환, 가명화, 평가 질문 작성과 evaluation runner는 다음 작업으로 남겨줘.

---

<a id="prompts-phase1-02-sql-normalization-ast-analysis"></a>

## 원본: prompts/phase1-02-sql-normalization-ast-analysis.md

# 다음 개발 프롬프트: Phase 1 SQL 정규화와 AST 분석

다음 단계인 기존 학습 데이터셋의 SQL 정규화 규칙과 sqlglot 기반 AST 분석기를 개발해줘.

작업 전 반드시 다음을 확인해.

- `docs/schema-adaptive-text-to-sql-plan.md` 전체
- `backend/adaptive/dataset`의 record, loader, quality report, CLI와 저장 안전장치
- 현재 `QueryIR`, ontology operator, compiler 계약
- `backend/requirements.txt`의 sqlglot 고정 버전과 기존 AST policy
- 기존 `dataset.csv`의 SQL 통계와 데이터 활용 방침
- 저장소의 Conventional Commit 기록
- 작업 시작 시 working tree 상태

## 목표

기존 `DatasetRecord.sql` 원문을 변경하지 않고 PostgreSQL dialect로 AST 분석해, 보수적인 canonical fingerprint와 SQL feature 분포, 현재 Query IR로의 변환 가능성 및 수동 검토 사유를 결정적이고 재현 가능한 JSON report로 만든다.

이번 단계는 SQL을 실행하거나 Query IR로 실제 변환하는 단계가 아니다. 다음 변환기의 입력 계약과 지원 범위를 실측으로 결정하는 것이 목적이다.

## 입력과 dialect 원칙

- 입력은 구현된 `load_jsonl_dataset()`의 유효 `DatasetRecord`를 사용한다.
- 실제 데이터셋 경로는 CLI 인자로만 받고 저장소에 복사하지 않는다.
- SQL 원문과 source line을 보존하되 기본 report에는 SQL·질문·literal 원문을 넣지 않는다.
- source dialect는 명시적인 설정으로 받고 기본값은 기존 시스템 DB에 맞춘 `postgres`로 한다.
- parser와 serializer 모두 같은 명시적 dialect를 사용한다.
- sqlglot 버전은 현재 고정된 버전을 사용하고 report metadata에 기록한다.
- parse error나 지원하지 않는 SQL 한 건 때문에 전체 분석을 중단하지 않는다.

현재 문서의 검증 대상 통계:

- 유효 SQL 5,385개
- 고유 SQL 5,124개
- 참조 table은 모두 `courses`
- JOIN 0개
- subquery 0개
- aggregate function 675개
- `GROUP BY` 4개
- `ORDER BY` 122개
- `LIMIT` 130개
- `capacity - enrolled` 형태 산술식 256개

이 값은 검증 기준이지 코드에 하드코딩할 값이 아니다. 새 분석기의 AST 정의와 기존 수치가 다르면 어느 정의가 다른지 설명하고 문서에 기록한다.

## 이번 범위

- versioned SQL analysis Pydantic 계약
- sqlglot PostgreSQL dialect parse
- 빈 SQL, parse error, 다중 statement, SELECT가 아닌 statement의 구조화된 issue
- 원문을 변경하지 않는 보수적 canonical SQL 생성 규칙
- canonical SQL 재parse와 AST round-trip 검증
- canonical SQL hash와 literal-redacted shape hash 생성
- statement·table·column·projection·predicate·function feature 추출
- aggregate, DISTINCT, GROUP BY, HAVING, ORDER BY, LIMIT/OFFSET 집계
- JOIN, subquery, CTE, set operation, window function 집계
- arithmetic, comparison, boolean operator와 wildcard 집계
- 현재 Query IR 후보 가능성 분류와 reason code 생성
- feature와 분류 결과의 전체·그룹별 결정적 통계
- 개인정보를 복제하지 않는 versioned JSON report
- 기존 dataset CLI에 독립 subcommand 또는 명확한 새 CLI 진입점 추가
- 합성 fixture 기반 parser·normalizer·classifier·CLI 테스트
- 실제 `dataset.csv` read-only 분석과 기존 통계 비교
- 계획 문서의 Phase 1 체크박스, 현재 체크포인트, ADR와 변경 이력 갱신

## SQL 정규화 계약

정규화 결과는 분석과 fingerprint 용도이며 실행 SQL이나 새로운 정답 SQL로 사용하지 않는다.

- raw SQL은 `DatasetRecord`에서 그대로 보존한다.
- canonicalization은 sqlglot이 parse한 AST를 동일 dialect로 결정적으로 직렬화한다.
- identifier 이름, identifier qualification, alias, literal 값과 literal type을 보존한다.
- predicate, projection, JOIN, GROUP BY, ORDER BY 순서를 임의로 바꾸지 않는다.
- commutative expression이나 boolean 조건을 재정렬하지 않는다.
- quote 제거·추가로 identifier 의미가 달라지지 않도록 한다.
- comments와 formatting 차이는 canonical fingerprint에서 제외할 수 있지만 적용 규칙을 계약과 테스트로 고정한다.
- canonical SQL을 다시 parse한 AST가 원래 분석 AST와 동등한지 확인한다.
- round-trip이 불일치하면 성공으로 숨기지 말고 구조화된 issue로 분류한다.
- canonical hash는 identifier와 literal을 포함한다.
- shape hash는 literal 값만 type-aware placeholder로 치환한 복사 AST에서 계산한다.
- shape hash를 duplicate 정답 판정이나 실행 동등성의 근거로 사용하지 않는다.
- 원본 AST를 shape 생성 과정에서 in-place로 변경하지 않는다.

## AST feature 계약

문자열 정규식이 아니라 AST node와 clause를 근거로 최소한 다음을 기록한다.

- statement kind와 root node
- table과 column 참조 수
- projection 수, wildcard, alias, expression projection
- WHERE 존재 여부와 predicate 수
- `AND`, `OR`, `NOT`
- `=`, `!=`, `<`, `<=`, `>`, `>=`, `LIKE`, `IN`, `BETWEEN`, `IS NULL`
- aggregate function 종류와 개수
- 일반 function 종류와 개수
- arithmetic operator 종류와 개수
- DISTINCT, GROUP BY, HAVING, ORDER BY, LIMIT, OFFSET
- JOIN 종류와 개수
- subquery, CTE, UNION/INTERSECT/EXCEPT, window function
- placeholder와 literal type별 개수

동일 node를 부모와 자식 순회에서 중복 집계하지 않도록 feature 정의와 count 단위를 문서화한다. table·column·function 이름을 report에 포함한다면 정렬하고 literal이나 질문 원문과 섞지 않는다.

## Query IR 가능성 분류

각 SQL을 다음 상태 중 하나로 결정적으로 분류한다.

- `CURRENT_IR_CANDIDATE`: 현재 Query IR의 select, AND filter, 지원 operator, limit만으로 표현 가능해 보이는 단일 SELECT
- `IR_EXTENSION_REQUIRED`: 안전한 SELECT지만 aggregate, DISTINCT, GROUP BY, ORDER BY, 산술식, 추가 operator 등 현재 IR에 없는 표현이 필요함
- `MANUAL_REVIEW`: AST만으로 의도 보존 여부를 확정하기 어렵거나 모호한 alias·LIKE·projection·literal 처리가 있음
- `UNSUPPORTED_OR_UNSAFE`: parse 실패, 다중 statement, SELECT 외 statement 또는 현재 연구 범위 밖의 구조

분류는 최소한 다음 조건을 지킨다.

- 단일 table이라는 이유만으로 자동 변환 가능하다고 판단하지 않는다.
- 단순 column projection과 지원되는 literal filter만 `CURRENT_IR_CANDIDATE` 후보가 된다.
- 현재 Query IR의 지원 operator는 ontology와 실제 계약에서 읽고 별도로 중복 하드코딩하지 않는다.
- `LIKE '%값%'`과 prefix/suffix/exact LIKE를 구분한다.
- OR, NOT, IN, BETWEEN, IS NULL은 현재 계약으로 표현할 수 있는지 명시적으로 판정한다.
- aggregate, GROUP BY, HAVING, DISTINCT, ORDER BY와 산술 projection은 조용히 손실하지 않는다.
- `SELECT *`, function projection, alias 의존 결과명은 명시적인 reason code를 가진다.
- LIMIT 값이 현재 Query IR 범위를 벗어나면 후보로 승인하지 않는다.
- 한 SQL에 여러 사유가 있으면 모든 reason code를 결정적 순서로 보존한다.
- 이 분류는 자동 변환 성공 판정이 아니라 다음 변환 작업의 후보 선별 결과임을 report에 명시한다.

## 개인정보와 report 정책

- 기본 report에는 실제 SQL, 질문, string/numeric literal 원문을 저장하지 않는다.
- record 식별은 source line, canonical hash, shape hash를 사용한다.
- parse error 메시지에 SQL 원문이나 literal이 포함되지 않도록 정제한다.
- feature 예시가 필요하면 명시적 preview 옵션과 group·record·문자 길이 상한을 적용한다.
- preview에서도 literal redaction을 기본으로 하고 원문 preview는 별도 명시 없이는 허용하지 않는다.
- 입력 절대 경로와 실행 시각은 deterministic report core에 포함하지 않는다.
- dataset content fingerprint, report schema version, analyzer version, sqlglot version, dialect는 기록한다.
- 실제 분석 report와 실제 데이터 원문은 저장소에 커밋하지 않는다.

## 오류와 안전 정책

- SQL을 어떤 데이터베이스에서도 실행하지 않는다.
- sqlglot parse·transform·serialize 예외를 record-level issue로 변환한다.
- 한 record의 실패 이후 다음 record 분석을 계속한다.
- 빈 SQL과 whitespace-only SQL을 구분 가능한 issue로 남긴다.
- 한 record에 다중 statement가 있으면 첫 statement만 선택하지 않는다.
- DML, DDL, transaction, PRAGMA, ATTACH와 database control statement를 분석 성공으로 취급하지 않는다.
- input/output 동일 경로·symlink·hardlink 보호와 atomic non-overwrite 정책은 기존 dataset CLI 안전장치를 재사용한다.
- 같은 설정과 입력은 동일한 report core 및 정렬 순서를 생성해야 한다.

## CLI 예시

현재 dataset CLI 구조를 우선 재사용한다.

```powershell
python -m backend.adaptive.dataset analyze-sql `
  --input "C:\path\to\dataset.csv" `
  --dataset-id cnu-original-v1 `
  --dialect postgres `
  --output ".schema-adaptive\reports\sql-analysis.json"
```

기존 `inspect` subcommand 동작과 JSON 계약을 깨지 않는다. 출력 파일은 기본적으로 덮어쓰지 않고 `--force`에서만 원자 교체한다.

## 필수 테스트

- 단순 단일 table SELECT parse와 canonical round-trip
- whitespace·keyword formatting만 다른 SQL의 같은 canonical hash
- identifier, alias, quote, literal이 다른 SQL을 과도하게 병합하지 않음
- literal 값만 다른 SQL의 다른 canonical hash와 같은 shape hash
- string, integer, decimal, boolean, NULL literal type 구분
- 원본 AST와 raw SQL 불변
- predicate·projection 순서 미변경
- comment 처리 규칙
- 빈 SQL과 malformed SQL
- 다중 statement
- INSERT/UPDATE/DELETE와 DDL/database control statement
- aggregate와 일반 function 구분
- DISTINCT, GROUP BY, HAVING, ORDER BY, LIMIT/OFFSET
- JOIN 종류, subquery, CTE, set operation, window function
- arithmetic과 boolean/comparison operator 집계
- wildcard와 expression projection
- 현재 IR 후보, IR 확장 필요, 수동 검토, 미지원 분류
- LIKE contains와 prefix/suffix pattern 구분
- 복수 reason code의 결정적 순서
- parse error message의 원문·literal 비노출
- report의 deterministic order와 metadata
- preview 기본 비활성화와 redaction·개수·길이 제한
- CLI output 안전성과 input 불변
- 동일 경로·symlink·hardlink 출력 차단
- 실제 데이터 5,385건 전체 분석
- 기존 문서의 JOIN/subquery/aggregate/GROUP BY/ORDER BY/LIMIT/산술 통계 비교
- 기존 전체 테스트 회귀 없음

## 제외 범위

- SQL 실행 또는 실행 결과 비교
- SQL을 실제 Query IR로 변환
- 자동 변환된 Query IR 저장
- 변환 실패 수동 검토 목록 확정
- Query IR 모델 변경
- ontology나 compiler 변경
- SQL 의미 동등성 증명
- aggregate·ORDER BY·OR 등을 지원하기 위한 IR 확장
- 질문 template 분류와 train/test split
- 실제 식별값 가명화
- 공통 평가 질문·기대 결과 작성
- evaluation runner
- 모델 실행, Ollama, PostgreSQL 연결
- repair loop와 관리자 UI

## 설계 원칙

- parser, canonicalizer, feature extractor, IR compatibility classifier, report serializer, CLI 책임을 분리한다.
- 모든 결과와 오류를 Pydantic 계약으로 명시한다.
- 분류 reason code는 문자열 설명만 두지 말고 enum으로 정의한다.
- sqlglot 내부 class 이름을 그대로 장기 저장 계약으로 노출하지 않는다.
- feature 집계 정의를 unit test로 고정한다.
- 원문 SQL의 의미를 바꿀 수 있는 rewrite를 하지 않는다.
- 현재 Query IR이 표현하지 못하는 기능을 자동으로 버리지 않는다.
- 기존 legacy 시스템, compiler, AST execution policy를 변경하지 않는다.

## 서브 에이전트

- 메인 에이전트가 먼저 canonicalization 계약, feature count 단위, IR 분류 상태와 reason code를 정의한다.
- 한 에이전트는 parser·canonicalizer core를 담당한다.
- 다른 에이전트는 feature extractor·classifier를 담당한다.
- 별도 에이전트는 합성·적대적 테스트를 담당한다.
- 수정 파일은 에이전트별로 겹치지 않게 한다.
- 메인 에이전트가 report·CLI 통합, 실제 5,385건 분석과 기존 통계 비교를 담당한다.
- 구현 후 별도 리뷰 에이전트가 canonical 의미 보존, literal redaction, node 중복 집계, 분류 누락, 결정성, 원본 불변성을 점검한다.
- 리뷰 지적 수정 후 실제 데이터 분석과 전체 테스트를 다시 실행한다.

## 완료 기준

- 실제 dataset 5,385개 SQL을 끝까지 분석하고 개별 오류가 전체 실행을 중단하지 않음
- source dialect와 sqlglot version이 report에 기록됨
- raw SQL과 원본 AST를 변경하지 않음
- canonical hash와 literal-redacted shape hash의 역할이 분리됨
- canonical round-trip 불일치가 명시적으로 보고됨
- SQL feature count 정의가 계약과 테스트로 고정됨
- 네 가지 Query IR 후보 상태와 모든 reason code가 결정적임
- 기존 통계와 새 AST 통계를 비교하고 차이를 설명함
- 기본 report와 parse error에 실제 SQL·질문·literal 원문이 노출되지 않음
- 실제 데이터와 실제 분석 report가 저장소에 추가되지 않음
- 기존 `inspect` CLI 회귀 없음
- 기존 전체 테스트 회귀 없음
- `compileall` 통과
- `git diff --check` 통과
- working tree clean

## 문서

- Phase 1의 SQL 정규화 규칙과 sqlglot AST 분석 체크박스 갱신
- 실제 AST feature 통계와 기존 문서 수치 비교 결과 기록
- canonical hash, shape hash, IR 분류 경계가 중요한 결정이면 ADR 추가
- 다음 체크포인트를 기존 SQL→Query IR 변환기와 수동 검토 목록으로 지정
- 날짜가 포함된 변경 이력 추가

## 커밋

- 과거 Conventional Commit 형식을 따른다.
- parser/canonicalizer, feature/classifier, report/CLI, 테스트, 문서를 기능별로 분리 커밋한다.
- 각 커밋 전 staged diff를 확인한다.
- 실제 dataset, SQL 원문, literal preview, 생성된 실제 report를 커밋하지 않는다.
- 원격 push는 하지 않는다.

이번 단계까지만 개발하고 SQL→Query IR 변환, Query IR 확장, 가명화, 평가 질문 작성과 evaluation runner는 다음 작업으로 남겨줘.

---

<a id="prompts-phase1-03-sql-to-query-ir-conversion"></a>

## 원본: prompts/phase1-03-sql-to-query-ir-conversion.md

# 다음 개발 프롬프트: Phase 1 SQL→Query IR 변환과 수동 검토 목록

다음 단계인 기존 학습 데이터셋의 SQL→Query IR 변환기와 결정적인 수동 검토 목록을 개발해줘.

작업 전 반드시 다음을 확인해.

- `docs/schema-adaptive-text-to-sql-plan.md` 전체
- `docs/development-history.md#prompts-phase1-01-jsonl-loader-quality-report`
- `docs/development-history.md#prompts-phase1-02-sql-normalization-ast-analysis`
- `backend/adaptive/dataset`의 loader, 품질 report, SQL parser, AST feature와 호환성 분류 계약
- 현재 `QueryIR`, `FilterIR`, ontology field·operator·data type 계약
- 기존 single-table 및 JOIN-capable compiler와 Query IR validator 동작
- 기존 데이터셋의 `courses` 물리 컬럼과 course-domain ontology의 대응 가능 범위
- 저장소의 Conventional Commit 기록
- 작업 시작 시 working tree 상태

## 목표

기존 `DatasetRecord.sql`을 PostgreSQL AST로 해석하고, 명시적으로 승인된 legacy dataset schema profile을 이용해 현재 `QueryIR`로 손실 없이 표현 가능한 부분만 결정적으로 변환한다.

자동 변환할 수 없는 레코드는 임의로 보정하거나 버리지 않고 구조화된 수동 검토 항목으로 남긴다. 이 단계의 목적은 Query IR 학습·평가 자료의 신뢰할 수 있는 원천을 만드는 것이며 Query IR 자체의 표현 범위를 넓히는 것이 아니다.

## 현재 기준선

직전 SQL AST 분석 결과는 다음과 같다.

- 전체 SQL: 5,385개
- parse 및 canonical round-trip issue: 0개
- 고유 canonical SQL: 5,085개
- literal-redacted shape: 1,667개
- `CURRENT_IR_CANDIDATE`: 3,395개
- `IR_EXTENSION_REQUIRED`: 1,115개
- `MANUAL_REVIEW`: 875개
- `UNSUPPORTED_OR_UNSAFE`: 0개

이 수치는 변환기의 검증 기준이지 코드에 하드코딩할 값이 아니다. `CURRENT_IR_CANDIDATE`도 자동 변환 성공을 보장하지 않으며 source mapping, literal type, 질문 충돌과 변환 후 검증에서 추가로 거부될 수 있다.

실제 데이터의 AST에는 정상 물리 컬럼 외에 PostgreSQL에서 identifier로 해석되는 double-quoted 값 후보가 존재한다. 예를 들어 문자열 의도로 작성된 `"컴퓨터공학과"`도 PostgreSQL AST에서는 column identifier가 된다. 이를 문자열 literal로 추측해 자동 변환하지 않는다.

## 이번 범위

- versioned legacy dataset schema profile 계약
- `courses` 물리 컬럼에서 ontology semantic field로 가는 명시적 mapping 정의
- source profile의 root entity, dialect, table allowlist, ontology version 검증
- versioned SQL→Query IR conversion result와 review item Pydantic 계약
- 이전 SQL analyzer의 canonical AST와 호환성 결과 재사용
- 단순 column projection의 semantic field 변환
- `AND` filter 순서와 비교 operator의 Query IR 변환
- 안전한 literal 추출과 ontology data type 검증
- 정확한 contains 형태의 `LIKE '%value%'` 변환
- 명시적 `LIMIT` 보존과 LIMIT이 없는 경우의 실행 안전 기본값 정책
- source intent signature와 변환된 IR intent signature 비교
- 기존 Query IR validator를 이용한 최종 계약 검증
- exact duplicate, 질문·SQL duplicate, 질문별 SQL conflict 정보 결합
- 자동 변환 성공, 실행 정책 적용 성공, 수동 검토, 미지원 상태 분류
- reason code와 source line 기반 결정적 수동 검토 목록
- 개인정보를 복제하지 않는 conversion summary JSON report
- 명시적으로 요청한 경우에만 쓰는 private converted JSONL artifact
- 기존 dataset CLI에 독립 subcommand 추가
- 합성 fixture 기반 converter·report·CLI 테스트
- 실제 5,385건 read-only 변환 시도와 상태·사유 통계 기록
- 계획 문서의 Phase 1 체크박스, 현재 체크포인트, ADR와 변경 이력 갱신

## Legacy dataset schema profile

물리 컬럼명을 ontology field로 변환할 때 이름 유사도를 다시 계산하거나 추측하지 않는다. versioned profile에 명시적으로 승인된 mapping만 사용한다.

초기 profile은 최소한 다음 정보를 가진다.

- profile schema version과 profile id
- source dialect: `postgres`
- source table allowlist: `courses`
- root entity: `CourseOffering`
- ontology version
- 물리 column → semantic field mapping
- 각 mapping의 expected source type 또는 허용 literal type
- 검토 상태와 근거

초기 검토 대상 mapping:

```text
courses.course_year   → CourseOffering.year
courses.subject_code  → Course.code
courses.section       → CourseOffering.section
courses.subject_name  → Course.name
courses.category      → Course.category
courses.credit_hours  → Course.credits
courses.professor     → Instructor.name
courses.capacity      → CourseOffering.capacity
courses.enrolled      → CourseOffering.enrolled
courses.dept_name     → Department.name
```

`lecture_time`, `class_mode` 등 현재 ontology에서 하나의 semantic field로 손실 없이 대응하지 않는 컬럼은 억지로 매핑하지 않는다. `lecture_time`을 `Schedule.start_time` 또는 `Schedule.end_time`으로 임의 분해하지 않는다.

이 profile은 학습 데이터 변환용 source metadata다. 운영 실행에 사용하는 `PublishedMappingProfile`인 것처럼 가장하거나 runtime publish gate를 우회하지 않는다.

## 변환 상태 계약

각 유효 dataset record는 다음 중 정확히 하나의 상태를 가진다.

- `CONVERTED_EXACT`: 현재 Query IR로 source intent를 그대로 표현 가능
- `CONVERTED_WITH_EXECUTION_POLICY`: source SQL에는 없던 안전 LIMIT 기본값만 적용됨
- `REVIEW_REQUIRED`: 사람의 의미 판단이나 교정이 필요함
- `UNSUPPORTED`: 현재 연구 범위 또는 Query IR 계약 밖이라 이번 단계에서 변환하지 않음

상태 우선순위와 reason code 순서를 계약으로 고정한다. 한 레코드에 여러 문제가 있으면 첫 문제만 남기지 말고 모든 reason code를 enum 순서로 보존한다.

이전 analyzer 상태는 입력 근거지만 최종 변환 상태와 동일하지 않다.

- `CURRENT_IR_CANDIDATE`만 자동 변환 경로에 진입할 수 있다.
- `MANUAL_REVIEW`는 `REVIEW_REQUIRED`로 전달하되 더 구체적인 변환 사유를 추가할 수 있다.
- `IR_EXTENSION_REQUIRED`는 현재 Query IR을 변경하지 않고 `UNSUPPORTED` 또는 의미 판단이 필요한 경우 `REVIEW_REQUIRED`로 보낸다.
- `UNSUPPORTED_OR_UNSAFE`는 변환하지 않는다.

## projection 변환 규칙

- projection은 단순 physical column만 자동 변환한다.
- source projection 순서를 Query IR `select` 순서로 보존한다.
- profile에 없는 table·column은 `UNMAPPED_SOURCE_COLUMN`으로 검토 목록에 남긴다.
- wildcard, expression projection, aggregate, function, alias 의존 결과명은 자동 변환하지 않는다.
- 같은 semantic field가 중복 projection되면 Query IR의 unique select 계약을 위반하므로 조용히 제거하지 않는다.
- double-quoted identifier가 profile에 없는 경우 실제 문자열 값으로 추측하지 않는다.
- projection이 여러 ontology entity의 field를 포함해도 root entity는 projection 첫 항목으로 추측하지 않고 profile의 `CourseOffering`을 사용한다.

## filter와 literal 변환 규칙

- top-level `AND`만 flatten하며 원래 predicate 순서를 보존한다.
- 왼쪽이 승인된 physical column이고 오른쪽이 단일 literal인 비교만 자동 변환한다.
- `=`, `!=`, `<`, `<=`, `>`, `>=`는 현재 ontology `Operator`에서 읽어 변환한다.
- operator가 해당 ontology field의 `allowed_operators`에 없으면 변환하지 않는다.
- literal은 string, integer, decimal, boolean만 현재 `FilterIR.value` 범위에서 처리한다.
- `NULL`, column-to-column 비교, expression, placeholder, cast가 필요한 값은 자동 변환하지 않는다.
- ontology data type과 literal type이 정확히 맞지 않으면 문자열 숫자 변환 같은 coercion을 조용히 적용하지 않는다.
- 음수와 decimal의 부호·정밀도를 손실하지 않는다. 현재 `float` 변환이 정밀도를 보존하지 못하면 수동 검토로 보낸다.
- 비교식의 column과 literal 위치가 뒤집힌 형태는 operator 반전 의미를 명시적으로 검증하지 않는 한 자동 변환하지 않는다.

`LIKE`는 다음 조건을 모두 만족할 때만 `Operator.CONTAINS`로 변환한다.

- 왼쪽이 string ontology field로 매핑된 column
- 오른쪽이 string literal
- pattern이 정확히 `%value%`
- 내부 `%`, `_`, escape 문자가 없음
- value가 빈 문자열이 아님

prefix, suffix, exact LIKE, `ILIKE`, complex wildcard와 double-quoted identifier RHS는 수동 검토 대상으로 남긴다.

## LIMIT 정책

- source SQL에 1~200 범위 integer literal LIMIT이 있으면 그대로 보존한다.
- 범위 밖, expression, placeholder, OFFSET 결합 LIMIT은 자동 변환하지 않는다.
- source SQL에 LIMIT이 없으면 현재 Query IR 계약상 configurable safe default를 적용한다. 초기값은 기존 계약과 같은 50이다.
- 기본 LIMIT 적용은 원본 SQL과 완전히 같은 의미라고 기록하지 않고 `CONVERTED_WITH_EXECUTION_POLICY`와 `DEFAULT_LIMIT_APPLIED` provenance를 남긴다.
- 안전 LIMIT과 사용자 의도 LIMIT을 report에서 구분한다.

## 의미 보존 검증

SQL 문자열을 다시 생성해 text equality만 비교하지 않는다. 데이터베이스에서 SQL을 실행하지도 않는다.

지원 subset에 대해 source AST에서 다음 intent signature를 만든다.

- root entity
- semantic projection 순서
- semantic filter field·operator·typed literal 순서
- source LIMIT 또는 default-limit provenance

생성한 `QueryIR`에서 같은 signature를 다시 만들고 정확히 비교한다. 불일치하면 변환 성공으로 숨기지 않고 `INTENT_SIGNATURE_MISMATCH`로 실패시킨다.

최종 Query IR은 기존 Pydantic model과 ontology/mapping validator를 통과해야 한다. validator 통과를 위해 source profile에서 runtime 검증용 mapping view를 만들 수 있지만, 이를 운영용 Published Mapping으로 저장하거나 publish하지 않는다.

## 중복·충돌 정책

- 원본 record와 source line을 임의로 제거하거나 재정렬하지 않는다.
- exact duplicate와 동일 질문·동일 SQL duplicate는 변환 통계에서 별도로 표시한다.
- 질문별 SQL conflict group의 레코드는 자동 학습 artifact 승인 대상으로 취급하지 않고 `QUESTION_SQL_CONFLICT`로 검토 목록에 보낸다.
- conflict를 다수결, 첫 레코드 또는 SQL hash 순으로 자동 해결하지 않는다.
- 중복 제거된 학습셋 생성은 이번 단계 범위가 아니다.

## 개인정보와 artifact 정책

- 기본 conversion report와 review list에는 질문, SQL, literal, 완전한 Query IR value를 저장하지 않는다.
- record 식별은 source line, canonical hash, shape hash와 안정적인 group hash를 사용한다.
- review item은 reason code와 redacted 구조 정보만 기본 저장한다.
- parse·validation error message에 원문이나 literal이 포함되지 않도록 정제한다.
- literal-redacted preview는 명시적 옵션과 record 수·문자 길이 상한에서만 허용한다.
- 질문과 literal을 포함한 converted JSONL은 별도의 명시적 `--private-output` 옵션에서만 생성한다.
- private artifact는 `.schema-adaptive/`처럼 gitignore된 위치를 기본 권장하고 저장소 tracked 경로에는 쓰지 못하게 보호한다.
- private artifact에는 schema version, dataset fingerprint, profile id/version, ontology version과 source line을 포함한다.
- 입력 절대 경로와 실행 시각은 deterministic report core에 넣지 않는다.
- 실제 dataset, 실제 converted JSONL, 실제 review report를 커밋하지 않는다.

## 오류와 안전 정책

- SQL을 어떤 데이터베이스에서도 실행하지 않는다.
- raw SQL을 rewrite해 정답 SQL로 저장하지 않는다.
- unknown table, unknown column, ambiguous quoted identifier와 type mismatch를 추측으로 해결하지 않는다.
- 한 record 변환 실패가 전체 분석을 중단하지 않는다.
- conversion 예외는 원문 비노출 record-level issue로 변환한다.
- input, public report, private output 사이의 동일 경로·symlink·hardlink를 모두 차단한다.
- 출력은 기본 non-overwrite이며 `--force`에서만 원자 교체한다.
- public report 저장 성공 후 private output 저장 실패처럼 artifact가 부분적으로 섞이지 않도록 commit 순서 또는 bundle manifest 정책을 명시한다.
- 같은 입력, profile, ontology와 설정은 동일한 report core·review 순서·private JSONL 순서를 생성해야 한다.

## CLI 예시

현재 dataset CLI 구조를 재사용한다.

```powershell
python -m backend.adaptive.dataset convert-ir `
  --input "C:\path\to\dataset.csv" `
  --dataset-id cnu-original-v1 `
  --profile backend/adaptive/dataset/profiles/cnu-courses-v1.yaml `
  --ontology backend/adaptive/ontology/course-domain.yaml `
  --output ".schema-adaptive\reports\ir-conversion.json"
```

민감한 변환 결과가 실제로 필요한 경우에만 다음 옵션을 추가한다.

```powershell
  --private-output ".schema-adaptive\private\cnu-query-ir-v1.jsonl"
```

기존 `inspect`와 `analyze-sql` subcommand 및 JSON 계약을 깨지 않는다.

## 필수 reason code

최소한 다음 사유를 enum으로 정의하고 결정적 순서를 고정한다.

- `ANALYZER_NOT_CURRENT_IR_CANDIDATE`
- `QUESTION_SQL_CONFLICT`
- `SOURCE_PROFILE_MISMATCH`
- `UNKNOWN_SOURCE_TABLE`
- `UNMAPPED_SOURCE_COLUMN`
- `AMBIGUOUS_QUOTED_IDENTIFIER`
- `UNSUPPORTED_PROJECTION`
- `DUPLICATE_SEMANTIC_PROJECTION`
- `UNSUPPORTED_BOOLEAN_STRUCTURE`
- `UNSUPPORTED_OPERATOR`
- `OPERATOR_NOT_ALLOWED_FOR_FIELD`
- `NON_LITERAL_FILTER_VALUE`
- `LITERAL_TYPE_MISMATCH`
- `UNSAFE_LITERAL_COERCION`
- `UNSUPPORTED_LIKE_PATTERN`
- `EMPTY_CONTAINS_VALUE`
- `INVALID_SOURCE_LIMIT`
- `DEFAULT_LIMIT_APPLIED`
- `QUERY_IR_VALIDATION_FAILED`
- `INTENT_SIGNATURE_MISMATCH`

이전 analyzer reason code를 문자열로 복사해 새 enum인 것처럼 만들지 않는다. 필요하면 원래 analyzer status와 reason code를 별도 provenance field로 보존한다.

## 필수 테스트

- 승인된 `courses` source profile loading과 version 검증
- profile의 unknown/duplicate physical column과 unknown semantic field 거부
- ontology version·dialect·table 불일치 거부
- 단순 projection의 semantic field 변환과 순서 보존
- 여러 ontology entity field를 포함해도 root entity가 결정적임
- `AND` filter flatten과 predicate 순서 보존
- 여섯 비교 operator의 변환
- ontology field별 operator allowlist 검증
- string, integer, decimal, boolean literal 처리
- NULL, placeholder, cast, expression과 column-to-column 비교 거부
- literal type mismatch와 unsafe coercion 차단
- 정확한 contains LIKE 변환
- prefix/suffix/exact/complex/empty LIKE 검토 전환
- double-quoted 값 후보를 string literal로 자동 변환하지 않음
- unmapped `lecture_time`, `class_mode` 등 명시적 검토
- 명시 LIMIT 보존
- LIMIT 없는 SQL의 default 50과 `CONVERTED_WITH_EXECUTION_POLICY`
- 범위 밖·비정수 LIMIT 및 OFFSET 거부
- wildcard, alias, function, aggregate, arithmetic projection 거부
- OR, NOT, IN, BETWEEN, IS NULL 거부 또는 이전 분석 상태 전달
- source와 Query IR intent signature 동등성
- 기존 Query IR validator 통과
- exact duplicate와 question/SQL duplicate provenance 보존
- question conflict 전체를 review 대상으로 분류
- 복수 reason code의 enum 순서 보존
- record와 review list의 source line 결정성
- 기본 report에 질문·SQL·literal·Query IR value 비노출
- preview redaction 및 개수·길이 제한
- private output opt-in과 tracked path 차단
- input/output/private-output 동일 경로·symlink·hardlink 차단
- atomic non-overwrite 및 `--force`
- malformed record 이후 계속 변환
- 실제 5,385건 전체 변환 시도 및 상태 합계 일치
- 기존 `inspect`, `analyze-sql`, 전체 테스트 회귀 없음

## 제외 범위

- Query IR Pydantic 모델 변경
- aggregate, DISTINCT, GROUP BY, ORDER BY, OR, 산술식을 위한 IR 확장
- legacy double-quoted identifier를 문자열 literal로 자동 수정
- 수동 검토 항목의 관리자 승인·교정 UI
- conflict 자동 해결
- 중복 제거 또는 train/validation/test split
- 실제 식별값 가명화
- 변환 결과를 학습 데이터로 최종 배포
- SQL 실행 또는 PostgreSQL 연결
- semantic execution equivalence 평가
- 공통 평가 질문·기대 결과 작성
- evaluation runner
- 모델 실행, Ollama, repair loop

## 설계 원칙

- source profile, AST subset extractor, converter, intent verifier, report serializer, private writer, CLI 책임을 분리한다.
- 모든 저장 계약과 오류·상태·reason code를 Pydantic과 enum으로 명시한다.
- analyzer의 후보 판정을 그대로 신뢰하지 않고 converter가 실제 AST와 profile을 다시 검증한다.
- physical→semantic mapping은 profile에서만 읽고 column 이름을 추측하지 않는다.
- 기존 Query IR과 ontology operator·data type을 중복 정의하지 않는다.
- source projection·filter 순서와 literal type을 보존한다.
- 자동 변환 범위를 넓히기 위해 의미가 불명확한 SQL을 임의 보정하지 않는다.
- 변환 성공률뿐 아니라 review 비율과 reason 분포도 연구 결과로 남긴다.
- 기존 legacy 시스템, compiler, AST execution policy를 변경하지 않는다.

## 서브 에이전트

- 메인 에이전트가 먼저 source profile 계약, root entity, LIMIT provenance, 변환 상태와 reason code를 정의한다.
- 한 에이전트는 source profile loader와 validation을 담당한다.
- 다른 에이전트는 AST subset→Query IR converter와 intent signature 검증을 담당한다.
- 별도 에이전트는 report·private artifact 안전성 및 합성·적대적 테스트를 담당한다.
- 에이전트별 수정 파일이 겹치지 않게 한다.
- 메인 에이전트가 CLI 통합, 실제 5,385건 분석과 문서 갱신을 담당한다.
- 구현 후 별도 리뷰 에이전트가 quoted identifier 오인, literal coercion, operator 방향, LIMIT 의미, conflict 누락, 개인정보 노출과 결정성을 검사한다.
- 리뷰 지적 수정 후 실제 데이터 변환과 전체 테스트를 다시 실행한다.

## 완료 기준

- 실제 5,385개 record가 모두 conversion result 또는 구조화된 review/unsupported 결과를 가짐
- 자동 변환은 승인된 source profile과 `CURRENT_IR_CANDIDATE`에만 제한됨
- 변환된 Query IR이 기존 Pydantic·ontology validator를 통과함
- projection·filter 순서와 typed literal 의미가 intent signature에서 보존됨
- LIMIT 기본값 적용과 exact 변환이 구분됨
- double-quoted 값, unmapped field와 type mismatch가 자동 보정되지 않음
- conflict group이 자동 학습 후보로 통과하지 않음
- 상태 합계가 5,385개와 일치하고 reason 통계가 결정적임
- 실제 데이터 기반 상태·사유 통계를 계획 문서에 기록함
- 기본 report에 실제 질문·SQL·literal·IR value가 노출되지 않음
- 실제 dataset과 생성 artifact가 저장소에 추가되지 않음
- 기존 dataset CLI 회귀 없음
- 기존 전체 테스트 회귀 없음
- `compileall` 통과
- `git diff --check` 통과
- working tree clean

## 문서

- Phase 1의 기존 SQL→Query IR 변환과 수동 검토 목록 체크박스 갱신
- 실제 변환 상태와 주요 review reason 통계 기록
- source profile, root entity, LIMIT provenance, conflict 차단 정책이 중요하면 ADR 추가
- 다음 체크포인트를 Phase 1의 실제 식별값 가명화 도구로 지정
- 날짜가 포함된 변경 이력 추가

## 커밋

- 과거 Conventional Commit 형식을 따른다.
- source profile/contracts, converter/verifier, report/CLI, 테스트, 문서를 기능별로 분리 커밋한다.
- 각 커밋 전 staged diff를 확인한다.
- 실제 dataset, 질문·SQL·literal·private Query IR artifact와 실제 review report를 커밋하지 않는다.
- 원격 push는 하지 않는다.

이번 단계까지만 개발하고 Query IR 확장, 수동 검토 UI, 가명화, 평가 질문 작성과 evaluation runner는 다음 작업으로 남겨줘.

---

<a id="prompts-phase1-04-identifier-pseudonymization"></a>

## 원본: prompts/phase1-04-identifier-pseudonymization.md

# 다음 개발 프롬프트: Phase 1 실제 식별값 가명화

다음 단계인 기존 학습 데이터셋의 실제 식별값 가명화 도구를 개발해줘.

작업 전 반드시 다음을 확인해.

- `docs/schema-adaptive-text-to-sql-plan.md` 전체
- `docs/development-history.md#prompts-phase1-01-jsonl-loader-quality-report`
- `docs/development-history.md#prompts-phase1-02-sql-normalization-ast-analysis`
- `docs/development-history.md#prompts-phase1-03-sql-to-query-ir-conversion`
- `backend/adaptive/dataset`의 loader, conversion contracts, source profile, converter, report, storage와 CLI
- `PrivateConvertedRecord`, `QueryIR`, `FilterIR`, `DuplicateProvenance`, `LimitProvenance` 계약
- course-domain ontology의 semantic field와 data type
- `.gitignore`의 `.schema-adaptive/` 정책
- 저장소의 Conventional Commit 기록
- 작업 시작 시 working tree 상태

## 목표

SQL→Query IR 변환 단계에서 생성한 private converted JSONL을 입력으로 받아, 실제 교수명·학과명·과목명·과목코드처럼 특정 학교를 식별하거나 원본 데이터와 연결할 수 있는 값을 안정적인 가명으로 교체한다.

질문의 표현과 Query IR filter value를 항상 함께 바꾸고, 중복·충돌 관계와 실행 의도는 보존한다. 자동으로 안전성을 확인할 수 없는 record는 부분적으로 가명화한 결과를 내보내지 않고 구조화된 수동 검토 목록으로 보낸다.

이번 결과물은 재식별 위험이 전혀 없는 익명 데이터가 아니라 연구·개발용 pseudonymized artifact다. 외부 공개 가능 여부는 별도의 라이선스·개인정보 검토 전까지 확정하지 않는다.

## 현재 기준선

직전 SQL→Query IR 변환 결과는 다음과 같다.

- 전체 dataset record: 5,385개
- `CONVERTED_EXACT`: 11개
- `CONVERTED_WITH_EXECUTION_POLICY`: 3,201개
- private converted record: 3,212개
- `REVIEW_REQUIRED`: 1,059개
- `UNSUPPORTED`: 1,114개
- 질문별 SQL conflict 영향 record: 6개이며 자동 변환 차단됨

이 수치는 입력 검증 기준이지 코드에 하드코딩할 값이 아니다. 이번 단계는 private converted record만 처리한다. 이전 단계에서 review 또는 unsupported로 분류되어 Query IR이 없는 record를 억지로 복원하거나 가명화하지 않는다.

## 용어와 보안 경계

- 문서와 코드에서는 가능한 한 `anonymization`보다 `pseudonymization`을 사용한다.
- 같은 secret key와 profile에서 같은 semantic field·원본 값은 항상 같은 가명을 가져야 한다.
- 다른 semantic field의 같은 문자열은 기본적으로 서로 다른 namespace에서 가명화한다.
- 원본 값의 일반 SHA-256처럼 사전 대입에 취약한 unkeyed hash를 가명으로 사용하지 않는다.
- 가명화 key와 원본→가명 대응표는 public report, 로그, 예외, Git history에 남기지 않는다.
- 가명화 결과만으로 법적·통계적 익명성을 보장한다고 주장하지 않는다.
- 원본 private converted JSONL과 key file은 계속 민감 자료로 취급한다.

## 이번 범위

- versioned pseudonymization profile과 Pydantic 계약
- semantic field별 `PSEUDONYMIZE`, `PRESERVE`, `REVIEW` 정책
- HMAC-SHA-256 기반 결정적 pseudonym token 생성
- secret key file과 명시적 key id 검증
- 입력 private JSONL streaming loader와 version 검증
- 전체 입력을 먼저 읽어 sensitive value lexicon 구성
- 질문·instruction과 Query IR filter value의 동기화된 치환
- Unicode NFC와 결정적 longest-match-first 치환
- ASCII code의 token boundary와 한글 문자열 치환 규칙
- 동일 surface의 semantic namespace 충돌과 겹치는 match 탐지
- record 단위 all-or-nothing 가명화
- 가명화 후 원본 sensitive value 잔존 검사
- Query IR 구조·field·operator·비민감 literal·LIMIT 보존 검증
- source line, SQL hash, duplicate/conflict group id를 공개 artifact에 직접 복사하지 않는 keyed lineage token
- duplicate와 conflict 관계의 동등성 보존 검증
- versioned pseudonymized JSONL 계약
- 개인정보를 복제하지 않는 summary report와 private review report
- 원자적 artifact bundle 저장과 manifest
- 기존 dataset CLI에 독립 `pseudonymize-ir` subcommand 추가
- 합성 fixture 기반 단위·적대적·CLI 테스트
- 실제 3,212개 private converted record의 read-only/in-memory 가명화 시도와 상태·사유 통계 기록
- 계획 문서의 Phase 1 체크박스, 현재 체크포인트, ADR와 변경 이력 갱신

## 초기 semantic field 정책

초기 profile은 최소한 다음 정책을 명시한다.

```text
Course.name               → PSEUDONYMIZE, namespace=course-name
Course.code               → PSEUDONYMIZE, namespace=course-code
Department.name           → PSEUDONYMIZE, namespace=department-name
Instructor.name           → PSEUDONYMIZE, namespace=instructor-name
Course.category           → PRESERVE
Course.credits            → PRESERVE
CourseOffering.year       → PRESERVE
CourseOffering.section    → PRESERVE
CourseOffering.capacity   → PRESERVE
CourseOffering.enrolled   → PRESERVE
```

- 정책에 없는 semantic field를 자동으로 `PRESERVE`로 간주하지 않는다.
- unknown field, 중복 field policy, 잘못된 namespace와 ontology version 불일치를 거부한다.
- `PSEUDONYMIZE`는 기본적으로 string field에만 허용한다.
- 숫자·boolean·date/time을 가명화하려면 별도 형식 보존 규칙과 명시적 profile 변경이 필요하다.
- `Course.category`는 초기에는 질의 의미를 보존하기 위해 유지하지만, 실제 값 분포와 학교 고유 용어 여부를 report에 집계하고 공개 전 재검토 대상으로 문서화한다.
- `CourseOffering.section`도 초기에는 유지하되 학교·학생을 식별하는 값이 포함되지 않는다는 전제와 재검토 필요성을 기록한다.

## secret key와 token 계약

- Python 표준 라이브러리의 HMAC-SHA-256을 사용한다.
- key는 CLI의 `--key-file`로만 읽고 command-line 문자열, 환경변수, source code에 직접 넣지 않는다.
- key file은 최소 32 byte의 무작위 binary 값이어야 한다.
- 빈 key, 짧은 key, 읽을 수 없는 key, 저장소에서 Git tracked인 key file을 거부한다.
- `--key-id`는 사람이 key rotation을 식별하는 비밀이 아닌 label이며 output metadata에 기록할 수 있다.
- HMAC 입력은 tool version, profile id/version, semantic namespace, Unicode NFC 원본 값으로 domain separation한다.
- token 형식은 versioned contract로 고정한다. 예: `<INSTRUCTOR_AB12CD34EF56>`.
- 최소 token 길이와 alphabet을 계약으로 명시하고 질문과 JSON에서 안전하게 표현한다.
- truncated token collision을 반드시 검사한다.
- collision이 있으면 입력 순서에 의존해 번호를 붙이지 말고, 충돌한 모든 token의 digest 길이를 결정적으로 늘려 유일하게 만든다.
- 같은 key·profile·input에서는 입력 record 순서와 관계없이 같은 token을 생성해야 한다.
- 다른 key 또는 namespace에서는 같은 원본 값도 다른 token을 생성해야 한다.
- key 원문, HMAC 중간값과 원본→token 사전은 report나 로그에 출력하지 않는다.

## 입력 계약

- 입력은 이전 단계의 `PrivateConvertedRecord` JSONL만 허용한다.
- `schema_version`, dataset fingerprint, profile id/version, ontology version과 record별 conversion status를 검증한다.
- `CONVERTED_EXACT`와 `CONVERTED_WITH_EXECUTION_POLICY`만 처리한다.
- record의 Query IR은 현재 Pydantic 계약과 ontology validator를 다시 통과해야 한다.
- 같은 artifact 내부의 dataset/profile/ontology metadata 불일치를 fatal error로 처리한다.
- malformed JSON, object가 아닌 행, version 불일치와 잘못된 field type은 줄 번호가 포함된 issue로 남긴다.
- 기본 정책은 record-level 오류 이후 계속 읽되, key·profile·artifact metadata처럼 전체 결과의 일관성을 깨는 오류는 fatal로 처리한다.
- 원본 input file을 수정하지 않는다.

## sensitive value lexicon

- 각 record의 Query IR filter 중 profile이 `PSEUDONYMIZE`로 지정한 field의 string value만 원본 sensitive value 후보로 수집한다.
- 원본 값과 NFC 정규화 값을 모두 추적하되 public report에는 저장하지 않는다.
- empty 또는 whitespace-only value는 자동 처리하지 않는다.
- NFC 이후 서로 다른 원본 값이 같은 값으로 합쳐지는 normalization collision을 명시적으로 탐지한다.
- lexicon key는 semantic namespace와 normalized value다.
- 같은 normalized surface가 여러 namespace에서 서로 다른 token을 갖는 경우 질문 text만으로 namespace를 결정할 수 없으므로 `AMBIGUOUS_SURFACE_NAMESPACE`로 처리한다.
- 원본 값 길이, 문자 종류와 field별 고유 값 개수는 구간화한 통계만 public report에 기록하고 실제 값은 기록하지 않는다.
- 질문에만 있고 Query IR에는 없는 새 고유명사를 NER나 추측으로 lexicon에 추가하지 않는다.
- 이번 단계에서 탐지할 수 없는 잔여 고유명사가 있을 수 있음을 report와 문서에 명시한다.

## 질문·instruction 치환 규칙

- output text는 Unicode NFC와 LF 줄바꿈으로 통일하며 이 변화를 provenance로 기록한다.
- lexicon의 normalized surface를 길이 내림차순, namespace, digest 순으로 정렬해 longest-match-first로 처리한다.
- 같은 위치에서 여러 후보가 일치하거나 match 범위가 겹쳐 결과가 달라질 수 있으면 임의 선택하지 않는다.
- ASCII letter·digit·underscore로 구성된 과목코드는 더 긴 identifier 내부를 잘못 치환하지 않도록 양쪽 token boundary를 검사한다.
- 한글 교수명·학과명·과목명은 NFC exact substring만 자동 치환한다.
- 띄어쓰기 제거, 조사 제거, 오타 교정, fuzzy matching, 형태소 분석을 조용히 적용하지 않는다.
- record의 sensitive Query IR filter value가 question과 instruction 어디에도 나타나지 않으면 의미 대응을 확인할 수 없으므로 `SENSITIVE_VALUE_NOT_FOUND_IN_TEXT`로 검토한다.
- question에 같은 sensitive value가 여러 번 등장하면 모든 occurrence를 같은 token으로 교체한다.
- instruction에도 lexicon의 sensitive value가 나타나면 같은 규칙으로 교체한다.
- 치환 후 question·instruction·Query IR 어디에도 해당 record 및 전역 lexicon의 원본 sensitive value가 남지 않았는지 검사한다.
- 단순 substring 검사는 짧은 값에서 false positive를 만들 수 있으므로 lexicon과 같은 boundary 규칙을 재사용한다.
- 자동 검증에 실패한 record는 일부 field만 바뀐 output으로 내보내지 않는다.

## Query IR 의미 보존 규칙

- `entity`, `select` 순서, filter 순서, filter field, operator와 LIMIT를 변경하지 않는다.
- `PSEUDONYMIZE` field의 string filter value만 대응 token으로 교체한다.
- `PRESERVE` field의 value는 타입과 값을 그대로 유지한다.
- `REVIEW` field가 record에 나타나면 자동 output을 만들지 않는다.
- sensitive field의 non-string value는 추측하거나 문자열로 coercion하지 않는다.
- 가명화 전후 Query IR에서 값 이외의 구조가 동일한지 structural signature로 검증한다.
- 가명화된 question과 Query IR에는 동일한 token 문자열을 사용한다.
- 기존 conversion status와 LIMIT provenance를 보존한다.
- 가명 token을 원본 SQL로 다시 compile하거나 실행하지 않는다.

## lineage와 중복·충돌 정책

- 공개 가능한 pseudonymized record에는 원본 `source_line`, canonical SQL hash, shape hash와 기존 group id를 그대로 넣지 않는다.
- 각 값은 별도 HMAC domain을 사용한 stable lineage token으로 변환한다.
- record id는 dataset id/fingerprint와 source line을 포함한 keyed 입력에서 생성한다.
- canonical/shape lineage가 필요하면 원래 hash 자체 대신 keyed token만 저장한다.
- exact duplicate와 question-SQL duplicate group 관계는 group id를 keyed token으로 바꿔 보존한다.
- conflict record는 이전 단계에서 private converted artifact에 없어야 한다. 발견하면 자동 처리하지 않고 입력 계약 오류로 보고한다.
- 같은 원본 group의 record가 같은 transformed group id를 갖고 다른 group과 섞이지 않는지 검증한다.
- output record 순서는 source line 대신 생성 전 입력 순서와 무관한 stable record id 순으로 결정한다.
- 원본 source line과 변환된 record id의 대응은 local private review artifact에만 둘 수 있으며 public summary나 pseudonymized JSONL에는 넣지 않는다.

## 출력 artifact

최소한 다음 세 산출물을 구분한다.

1. `pseudonymized-query-ir.jsonl`
   - 가명화에 성공한 record만 포함
   - versioned schema, dataset id, pseudonymization profile, key id, stable record id 포함
   - 원본 source line·SQL hash·원본 group id·원본 sensitive value 제외
2. `pseudonymization-summary.json`
   - 상태·reason·field별 집계와 입력/output fingerprint
   - 질문, instruction, Query IR value, token과 원본 식별값 제외
3. `pseudonymization-review.json`
   - local private review용
   - source line, reason code와 원문 비노출 구조 정보만 포함
   - `.schema-adaptive/` 아래에만 저장하고 commit 금지

원본→가명 mapping table은 기본 산출물로 만들지 않는다. 같은 key와 profile로 다시 계산할 수 있게 한다. 별도 mapping export가 정말 필요하다면 이번 범위에 임의로 추가하지 말고 보안 설계를 먼저 문서화한다.

## 상태와 reason code

각 입력 record는 다음 중 정확히 하나의 상태를 가진다.

- `PSEUDONYMIZED_EXACT`
- `REVIEW_REQUIRED`
- `UNSUPPORTED`

최소한 다음 reason code를 enum으로 정의하고 순서를 고정한다.

- `INPUT_SCHEMA_VERSION_MISMATCH`
- `INPUT_METADATA_MISMATCH`
- `NON_CONVERTED_INPUT`
- `QUESTION_SQL_CONFLICT_PRESENT`
- `PROFILE_VERSION_MISMATCH`
- `ONTOLOGY_VERSION_MISMATCH`
- `UNKNOWN_SEMANTIC_FIELD`
- `FIELD_POLICY_MISSING`
- `FIELD_REQUIRES_REVIEW`
- `NON_STRING_SENSITIVE_VALUE`
- `EMPTY_SENSITIVE_VALUE`
- `NORMALIZATION_COLLISION`
- `AMBIGUOUS_SURFACE_NAMESPACE`
- `OVERLAPPING_TEXT_MATCH`
- `SENSITIVE_VALUE_NOT_FOUND_IN_TEXT`
- `ORIGINAL_VALUE_REMAINS`
- `PSEUDONYM_COLLISION_RESOLVED`
- `QUERY_IR_STRUCTURE_CHANGED`
- `QUESTION_IR_TOKEN_MISMATCH`
- `OUTPUT_VALIDATION_FAILED`
- `PSEUDONYMIZATION_ERROR`

한 record에 여러 사유가 있으면 첫 사유만 남기지 않고 enum 순서로 모두 보존한다. `PSEUDONYM_COLLISION_RESOLVED`처럼 안전하게 해결된 provenance와 실패 reason을 상태 계약에서 구분한다.

## report와 개인정보 정책

- public summary에는 질문, instruction, 원본 Query IR value, 가명 token, key 원문과 source line을 저장하지 않는다.
- error message는 입력 문자열이나 key material을 포함하지 않는 고정된 문구를 사용한다.
- field별 고유 sensitive value 수, 성공·review·unsupported 수와 reason 분포만 결정적으로 기록한다.
- sample이나 preview는 이번 단계의 public report에서 제공하지 않는다.
- local review report에도 원본 sensitive value를 복제하지 않고 source line과 reason만 둔다.
- input 절대 경로와 실행 시각은 deterministic report core에 포함하지 않는다.
- dataset fingerprint, input artifact fingerprint, profile id/version, ontology version, algorithm version과 key id를 기록한다.
- pseudonymized artifact도 별도 공개 검토 전까지 Git에 commit하지 않는다.

## 저장과 실패 원자성

- input, key, pseudonymized output, summary, review와 manifest 사이의 동일 경로·symlink·hardlink를 모두 차단한다.
- key file을 어떤 출력으로도 덮어쓰지 않는다.
- 실제 artifact는 `.schema-adaptive/` 아래처럼 gitignore된 경로에만 쓴다.
- Git tracked target을 명시적으로 거부한다.
- 기본은 non-overwrite이며 `--force`에서만 교체한다.
- 모든 파일을 temporary generation directory에 완전히 쓰고 fsync·검증한 뒤 manifest를 마지막에 원자 교체한다.
- 일부 output만 새 generation이고 나머지는 이전 generation인 상태를 성공으로 노출하지 않는다.
- 실패 시 이전 manifest와 이전 generation은 그대로 사용할 수 있어야 한다.
- key나 원본 문자열이 temporary filename, 예외 또는 stdout/stderr에 들어가지 않게 한다.

## CLI 예시

현재 dataset CLI에 독립 subcommand를 추가하되 기존 command가 항상 원본 JSONL loader를 먼저 실행하는 구조는 command별로 안전하게 분리한다.

```powershell
python -m backend.adaptive.dataset pseudonymize-ir `
  --input ".schema-adaptive\private\cnu-query-ir-v1.jsonl" `
  --dataset-id cnu-anonymized-v1 `
  --profile backend/adaptive/dataset/profiles/cnu-pseudonymization-v1.yaml `
  --ontology backend/adaptive/ontology/course-domain.yaml `
  --key-file ".schema-adaptive\secrets\pseudonymization-v1.key" `
  --key-id local-key-v1 `
  --output-dir ".schema-adaptive\pseudonymized\cnu-v1"
```

- 기존 `inspect`, `analyze-sql`, `convert-ir` subcommand와 JSON 계약을 깨지 않는다.
- CLI stdout에는 성공 여부, dataset id, generation id, 비민감 summary만 출력한다.
- key path, key fingerprint, 원본 값과 가명 token을 stdout/stderr에 출력하지 않는다.

## 필수 테스트

- versioned pseudonymization profile loading
- unknown·duplicate field policy, unknown namespace와 ontology version 불일치 거부
- 초기 10개 semantic field가 빠짐없이 명시됨
- key file 최소 길이, empty/short key와 tracked key 거부
- 같은 key·profile·field·원본 값의 같은 token
- 다른 key, namespace 또는 원본 값의 다른 token
- token alphabet·길이·prefix 계약
- truncated HMAC collision의 결정적 길이 확장
- 입력 private JSONL 정상 loading과 metadata 일관성
- malformed JSON, non-object, 잘못된 schema version과 field type
- non-converted status와 conflict provenance 거부
- sensitive filter value와 question의 동시 치환
- instruction에 포함된 sensitive value 치환
- 같은 값의 복수 occurrence 전체 치환
- 여러 sensitive value의 longest-match-first 처리
- ASCII 과목코드 boundary 처리
- 한글 NFC exact match
- 띄어쓰기·오타·형태 변형을 임의로 fuzzy match하지 않음
- semantic namespace가 모호한 같은 surface 차단
- 겹치는 text match의 결정적 처리 또는 review 전환
- question에서 찾지 못한 sensitive IR value 검토 전환
- empty·non-string sensitive value 차단
- `PRESERVE` field의 type과 값 유지
- `REVIEW` field 포함 record 차단
- entity·select·filter order·field·operator·LIMIT와 provenance 불변
- question과 Query IR token 일치
- 원본 sensitive value 잔존 검사
- 한 record 실패 시 부분 가명화 output 없음
- 같은 입력·key·profile의 byte-identical JSONL과 report core
- 입력 순서를 바꿔도 token과 stable record id가 같음
- source line, 원본 SQL hash와 원본 group id가 output에 직접 남지 않음
- duplicate group 관계가 transformed lineage에서도 유지됨
- public report와 error에 원본 값·token·key 비노출
- output bundle manifest의 generation 일관성
- non-overwrite, `--force`, 중간 실패 rollback
- input/key/output의 동일 경로·symlink·hardlink 차단
- 기존 `inspect`, `analyze-sql`, `convert-ir` CLI 회귀 없음
- 실제 private converted record 3,212개 전체 처리 시도와 상태 합계 일치
- 실제 output에서 lexicon 기반 원본 sensitive value 잔존 0건
- 기존 전체 테스트 회귀 없음

## 실제 데이터 검증 정책

- 실제 dataset과 private converted artifact는 read-only로 다룬다.
- 실제 key는 저장소에 만들거나 commit하지 않는다.
- 개발 검증에는 temporary directory의 ephemeral 32-byte key를 사용하고 종료 후 폐기한다.
- 실제 3,212개 변환 record를 처리하되 output artifact와 review report는 commit하지 않는다.
- 성공률만 기록하지 말고 field별 고유 값 수, review reason과 text-match 실패율을 기록한다.
- lexicon 기반 leak scan 0건은 “알려진 원본 값이 남지 않았다”는 뜻일 뿐, 알려지지 않은 고유명사까지 완전 제거했다는 뜻이 아님을 문서에 명시한다.
- `Course.category`, `CourseOffering.section`, instruction의 학교 고유 정보 잔존 가능성을 별도 검토 항목으로 기록한다.

## 제외 범위

- review/unsupported SQL을 Query IR로 복구
- legacy double-quoted identifier 자동 수정
- SQL 원문 재작성 또는 SQL 실행
- NER, LLM, 외부 API를 이용한 고유명사 추측
- fuzzy matching, 형태소 기반 자동 보정
- 법적·통계적 익명성 보장
- 원본→가명 mapping table의 평문 export
- 관리자 review UI
- Query IR 모델 확장
- aggregate, DISTINCT, GROUP BY, ORDER BY, OR, 산술식 지원
- 중복 제거와 train/validation/test split
- 공통 평가 질문·기대 결과 작성
- evaluation runner
- Ollama/PostgreSQL 연결
- repair loop

## 설계 원칙

- profile/contracts, key provider, token generator, private loader, lexicon builder, text rewriter, IR transformer, leak auditor, report, storage와 CLI 책임을 분리한다.
- 상태, reason, lineage와 artifact를 Pydantic 및 enum으로 명시한다.
- 원본 식별값을 로그·예외·public artifact에 넣지 않는다.
- HMAC domain separation과 token format을 unit test로 고정한다.
- field policy가 없는 값을 추측으로 보존하거나 가명화하지 않는다.
- 질문과 Query IR 중 하나만 바뀐 record를 성공으로 내보내지 않는다.
- 실패한 record는 부분 결과 없이 review로 보낸다.
- 기존 conversion artifact와 Query IR 계약을 변경하지 않는다.
- 기존 legacy 시스템과 schema-adaptive 실행 경로를 변경하지 않는다.
- 가명화율뿐 아니라 review 비율과 알려진 값 잔존 검사 결과를 연구 지표로 남긴다.

## 서브 에이전트

- 메인 에이전트가 먼저 field policy, HMAC domain/token format, text match 경계, lineage 변환과 all-or-nothing 상태 계약을 정의한다.
- 한 에이전트는 profile·contracts·key provider·token generator를 담당한다.
- 다른 에이전트는 private loader·lexicon·text/IR transformer·leak auditor를 담당한다.
- 별도 에이전트는 artifact bundle·CLI 안전성과 합성·적대적 테스트를 담당한다.
- 에이전트별 수정 파일이 겹치지 않게 한다.
- 메인 에이전트가 통합, 실제 3,212건 검증, 계획 문서 갱신을 담당한다.
- 구현 후 별도 리뷰 에이전트가 key 노출, unkeyed hash, token collision, namespace 혼동, 부분 치환, question-IR 불일치, 원본 값 잔존, lineage linkability, 결정성과 저장 실패 원자성을 검사한다.
- 리뷰 지적 수정 후 실제 데이터 검증과 전체 테스트를 다시 실행한다.

## 완료 기준

- private converted JSONL의 모든 행이 성공·review·unsupported 중 하나로 결정적으로 분류됨
- 초기 sensitive field 네 종류가 HMAC 기반 stable token으로 교체됨
- 같은 semantic value의 question·instruction·Query IR 표현이 일치함
- 비민감 Query IR 구조와 typed value가 보존됨
- 자동 처리 실패 record가 부분 가명화 output에 포함되지 않음
- 공개 artifact에 source line, 원본 SQL hash, 원본 group id와 sensitive value가 직접 남지 않음
- duplicate 관계가 keyed lineage token으로 보존됨
- 같은 입력·key·profile은 byte-identical artifact와 report core 생성
- 실제 private converted record 전체 상태 합계가 3,212개와 일치함
- 알려진 sensitive lexicon 기준 원본 값 잔존 0건
- key, 실제 dataset, private input, 실제 pseudonymized output과 review report가 저장소에 추가되지 않음
- 기존 dataset CLI 회귀 없음
- 기존 전체 테스트 회귀 없음
- `compileall` 통과
- `git diff --check` 통과
- working tree clean

## 문서

- Phase 1의 실제 식별값 가명화 도구 체크박스 갱신
- 실제 처리 상태, field별 고유 값 수, review reason과 leak audit 결과 기록
- pseudonymization 용어, HMAC key 관리, field policy, lineage token과 공개 한계에 대한 ADR 추가
- 다음 체크포인트를 Phase 2의 공통 평가 질문 30~50개와 기대 semantic result 작성으로 지정
- 날짜가 포함된 변경 이력 추가

## 커밋

- 과거 Conventional Commit 형식을 따른다.
- profile/contracts/key, transformer/auditor, report/storage/CLI, 테스트, 문서를 기능별로 분리 커밋한다.
- 각 커밋 전 staged diff를 확인한다.
- key, 실제 dataset, private converted JSONL, 실제 pseudonymized JSONL, 실제 review report를 commit하지 않는다.
- 원격 push는 하지 않는다.

이번 단계까지만 개발하고 수동 검토 UI, Query IR 확장, 데이터 split, 평가 질문·기대 결과와 evaluation runner는 다음 작업으로 남겨줘.

---

<a id="prompts-phase2-01-common-evaluation-questions"></a>

## 원본: prompts/phase2-01-common-evaluation-questions.md

# 다음 개발 프롬프트: Phase 2 공통 평가 질문과 기대 semantic 결과

다음 단계인 **공통 평가 질문 30~50개와 기대 semantic result 집합**을 개발해줘.

작업 전 반드시 다음을 확인해.

- `docs/schema-adaptive-text-to-sql-plan.md` 전체와 현재 Phase 2 체크포인트
- `backend/adaptive/fixtures`의 `FixtureSeed v1`, `LogicalOffering`, flat·normalized·legacy builder와 canonical logical row 계약
- `backend/adaptive/ontology/course-domain.yaml`의 field, data type, 허용 operator
- 현재 `QueryIR`·`FilterIR`, validator, JOIN planner, compiler, AST policy, read-only executor 계약
- 기존 cross-schema 실행 테스트와 Published Mapping 생성 방식
- `docs/development-history.md#prompts-phase1-04-identifier-pseudonymization` 및 CNU 원본·가명 데이터 사용 경계
- 저장소의 Conventional Commit 기록과 작업 시작 시 working tree 상태

## 목표

같은 한국어 질문을 세 가지 물리 스키마에 동일하게 제시할 수 있도록, synthetic `FixtureSeed v1`에만 근거한 versioned 평가 세트를 만든다. 각 사례에 사람이 검토할 수 있는 gold Query IR과 물리 DB 이름에 의존하지 않는 기대 semantic 결과를 붙여, 후속 evaluation runner와 모델 비교의 신뢰할 수 있는 기준 입력으로 사용한다.

이 단계의 성공은 모델 정확도 측정이 아니다. 질문·의도·정답 데이터와 그 검증 계약을 고정하는 것이다.

## 데이터 및 평가 경계

- 기본 대상은 현재 `DEFAULT_SEED`의 개설강좌 6건과 15개 논리 field다. 세 DB는 데이터가 같고 구조만 다르다.
- 평가 질문은 한국어로 새로 작성하며, 실제 CNU `dataset.csv`의 질문·SQL·literal이나 가명화 private artifact를 복사하지 않는다.
- seed에 없는 강좌·학과·교수·학기·시간을 사실처럼 답에 넣지 않는다. 존재하지 않는 값을 사용하는 **빈 결과** 사례는 명시적으로 표시한다.
- 원본 CNU 값에 대한 leak 검사는 이 synthetic 세트를 외부 공개할 수 있다는 보증이 아니다.
- 이 세 스키마는 개발 fixture다. 이를 독립 unseen school이나 실제 다른 대학 일반화 실험으로 부르지 않는다.
- Phase 1 가명화 review 684건을 승인된 평가 질문으로 자동 편입하지 않는다.
- 6건 seed로 만들기 어려운 연산·분포는 질문을 무의미하게 반복해 숫자를 채우지 말고 coverage gap으로 기록한다. 이번 단계에서 `FixtureSeed v1`이나 legacy 시스템을 조용히 바꾸지 않는다.

## 이번 범위

- versioned `EvaluationCase`·`ExpectedSemanticResult` Pydantic 계약
- 고정 ID를 가진 한국어 공통 평가 질문 30~50개
- 각 사례의 gold Query IR, 기대 semantic row multiset, 기대 개설강좌 key와 provenance
- fixture seed·ontology·Query IR version/fingerprint 연결
- 질문 유형·field·operator·JOIN 난도·결과 cardinality coverage metadata
- 중복 질문·중복 의도·동의어 paraphrase 식별 및 과다 반복 방지
- 독립적인 gold set 정적 validator와 결정적 JSON/YAML loader
- 세 스키마에서 **gold Query IR**의 실행 결과가 기대 semantic 결과와 일치하는지 검증하는 테스트
- 오류가 있는 gold 사례, stale seed, 미지원 연산을 명시적으로 거부하는 적대적 테스트
- 평가 세트의 machine-readable coverage report와 사람용 검토 표
- 계획 문서의 Phase 2 체크박스, 현재 체크포인트, 필요한 ADR와 날짜별 변경 이력 갱신

## 평가 사례 계약

각 사례는 최소한 다음 정보를 가진다.

- `schema_version`, 안정적인 `case_id` (`P2-001` 등), `question_ko`
- `seed_version` 또는 seed content fingerprint, ontology version
- `gold_query_ir`: 현재 Pydantic `QueryIR` 그대로 사용; 별도 유사 모델을 만들지 않음
- `expected_offering_keys`: 사람이 지정한 논리 개설강좌 key 목록
- `expected_semantic_rows`: `gold_query_ir.select`의 semantic field를 key로 한 typed row 목록
- `expected_cardinality`, answerability/zero-result 표시
- 검토 상태·검토 근거와 작성자/검토자 식별(실제 개인정보가 아닌 repository label)
- 난도·coverage tag: filter 수, operator, 참조 entity, 필요한 physical table 수/JOIN step 수, 결과 행 수, paraphrase family 등

질문은 세 물리 스키마에 공통으로 사용한다. 질문 문자열이나 gold Query IR 안에 `course_offerings`, `sbjt_nm`, `prof_nm` 같은 물리 table·column 이름을 넣지 않는다. `gold_query_ir.entity`는 질문의 논리 root entity를 명시하고 첫 projection field에서 추측하지 않는다.

평가 사례의 정답은 생성된 SQL 문자열이 아니라 **semantic 결과**다. `expected_offering_keys`는 검토용 행 provenance이고, 실제 결과 비교는 선택한 semantic field의 typed row multiset으로 한다. 동일 projection 값이 두 개설강좌에서 반복되면 중복 행 수를 보존한다. `ORDER BY`가 없는 현 Query IR에 순서를 요구하지 않는다. 문자열은 정확히 비교하고, 숫자는 ontology data type에 따른 명시적 비교 규칙을 사용하며 `bool`을 숫자로 혼동하지 않는다. 기대 결과를 테스트 대상 compiler나 SQL 실행 결과에서 자동 복사해 gold로 삼지 않는다.

## 질문 구성 원칙

30~50개 모두 현재 Query IR과 ontology가 표현할 수 있고 현재 seed에서 답을 판정할 수 있어야 한다. 단순 어미 변경만으로 사례 수를 채우지 않는다.

최소한 다음 축을 균형 있게 포함한다.

- 단일 field·다중 field projection, 여러 entity field를 함께 선택하는 사례
- filter 없음, 단일 filter, `AND` 다중 filter
- string `equals`·`contains`, 지원되는 정수/수치 비교 operator의 유효 사례
- 과목명·과목코드·학과명·교수명·연도·학기·분반·정원·수강인원·요일·시간 등 현재 fixture에 존재하는 field의 실제 coverage
- 정규화 스키마에서 직접 JOIN과 2단계 이상 JOIN이 필요한 field 조합
- 0건·1건·여러 건의 결과, 같은 과목의 서로 다른 분반, 같은 projection 값의 중복 가능성
- 값의 경계: 정원과 수강인원 일치, 숫자 threshold의 경계, `contains`의 부분 문자열
- 같은 의미의 한국어 paraphrase는 일부 포함하되 family ID로 묶어 독립 난도로 과장하지 않음

목표 개수보다 **서로 다른 논리 의도와 검증 가능한 정답**을 우선한다. field 또는 operator가 seed 분포상 구별력을 갖지 못한다면 coverage report에 그 한계를 적고, 허위 다양성을 만들지 않는다. 30개를 의미 있게 만들 수 없다면 부족한 축과 필요한 seed 보강안을 제시하고 완료로 표시하지 않는다. 현재 IR에 없는 잔여석 산술, OR, 집계, GROUP BY, ORDER BY, DISTINCT, 시간표 충돌 계산은 실행 가능한 gold 세트에 섞지 않는다. 필요하면 별도 `out_of_scope_examples` 문서에만 기록하고 30~50개 완료 수에 포함하지 않는다.

## gold 작성·검토 절차

1. `DEFAULT_SEED.canonical_rows()`와 ontology에서 사용 가능한 값·field·operator를 추출해 사람이 읽는 작업용 표를 만든다.
2. 사례별 한국어 질문과 gold Query IR을 작성한다. 질문이 여러 해석을 허용하면 보수적으로 수정하거나 제외한다.
3. gold Query IR의 Pydantic 및 ontology validator를 통과시킨다. 1~200 `LIMIT`을 명시하고 기대 행이 잘리는 사례는 현재 세트에서 피한다.
4. 사람이 seed를 보고 `expected_offering_keys`와 `expected_semantic_rows`를 기록한다. 예상 SQL이나 실행 결과에서 정답을 역생성하지 않는다.
5. 별도의 consistency 검사로 key 존재·중복, row shape/type, cardinality, seed 값 일치를 확인한다.
6. 세 스키마의 Published Mapping·Catalog FK·JoinPlan을 이용해 같은 gold Query IR을 compile·AST validate·read-only execute하고 각 결과 multiset을 gold와 비교한다.
7. 작성자와 다른 검토자가 질문 의도와 정답을 확인한다. 별도 사람이 없는 개발 환경에서는 `review_status=NEEDS_HUMAN_REVIEW`를 유지하고 자동 테스트 통과를 사람 승인으로 표기하지 않는다.

## 결정성·버전·저장 정책

- 파일 내 사례 정렬은 `case_id` 기준, row와 key 정렬 규칙은 계약으로 고정한다.
- 같은 seed·ontology·평가 세트 내용은 동일 fingerprint와 coverage report core를 만든다. 생성 시각·절대 경로는 core에 넣지 않는다.
- 질문 수정, gold IR 수정, 기대 결과 수정은 case ID 재사용 정책과 세트 version 변경 기준을 문서화한다. 이미 사용한 benchmark의 의미를 조용히 바꾸지 않는다.
- 합성 질문과 gold는 저장소에 커밋할 수 있다. 실제 dataset, private converted/pseudonymized artifact, key, 실제 질문·SQL·값은 포함하지 않는다.
- 사람용 검토 표에 개인정보나 실제 CNU 원문이 섞이지 않았는지 확인한다.

## 필수 테스트

- 30~50개 고유 case ID와 안정적 정렬
- 한국어 질문, 중복/공백/빈 질문, 동일 의도 paraphrase family 검증
- 모든 gold Query IR의 Pydantic·ontology·Published Mapping 유효성
- projection 순서, filter operator·typed literal, LIMIT 계약
- gold expected row의 key/field/type/cardinality·중복 행 multiset 정합성
- empty·single·multi-result와 분반 중복 처리
- flat·normalized·legacy 각각에서 gold Query IR의 기대 semantic 결과 일치
- 단일 테이블에서는 JOIN 없음, normalized에서는 필요한 FK 경로만 사용
- 예상 결과를 SQL 문자열 비교나 단순 행 정렬로 잘못 판정하지 않음
- fixture seed version/fingerprint 또는 ontology version drift 명시적 실패
- 없는 offering key, 없는 semantic field, 허용되지 않은 operator, 틀린 예상 행을 테스트에서 탐지
- 같은 입력의 byte-identical 평가 세트·coverage report core
- synthetic 자료만 저장소에 포함되고 실제 CNU/private 값이 새 파일에 들어가지 않음
- 기존 전체 테스트 회귀 없음, `compileall`, `git diff --check` 통과

## 제외 범위

- 모델 응답을 수집하거나 점수화하는 Phase 5 evaluation runner
- `RecordedModel`, Ollama, PostgreSQL, 기존 fine-tuned 모델 실행
- 모델별 정확도·응답시간·메모리 수치 측정
- 독립 unseen school 확보 또는 일반화 성능 주장
- 학습/검증/test split과 기존 데이터셋으로부터의 질문 자동 생성
- Query IR 모델·ontology operator·SQL compiler 확장
- 집계·정렬·산술·OR 등을 현재 IR에 억지로 삽입
- 관리자 UI, repair loop, schema evolution 실험

## 서브 에이전트

- 메인 에이전트가 먼저 평가 사례 계약, seed 고정 정책, 결과 multiset 비교 규칙과 review status를 정의한다.
- 독립 작업을 분배한다면 사례 작성·gold 정답 검토·검증기/테스트를 수정 파일이 겹치지 않게 나눈다.
- 메인 에이전트가 세 스키마 통합과 전체 테스트, coverage gap 및 문서 갱신을 담당한다.
- 구현 후 별도 리뷰 에이전트가 질문-IR 의미 일치, 정답의 독립성, 누락된 JOIN/중복 행, seed 과적합과 개인정보 경계를 검사한다.
- 리뷰 지적 수정 후 전체 테스트를 다시 실행한다.

## 완료 기준

- 30~50개의 사람이 읽고 의미를 확인할 수 있는 synthetic 한국어 평가 사례가 versioned 파일에 존재
- 각 사례에 gold Query IR과 기대 semantic 결과·logical offering provenance가 있음
- 모든 사례가 현재 Query IR 지원 범위에 있고 결과가 세 fixture에서 동일함
- coverage report가 field/operator/JOIN/cardinality와 부족한 축을 정직하게 보여줌
- 자동 정합성 검사가 사람의 검토 상태를 대신하지 않음
- CNU 원본 또는 private 가명 artifact가 benchmark에 섞이지 않음
- 기존 테스트 회귀 없음, `compileall`·`git diff --check` 통과, working tree clean

## 문서와 커밋

- Phase 2의 `공통 평가 질문 30~50개 선정`과 `기대 결과 집합 작성`은 실제 완료 후에만 체크한다.
- seed 규모, coverage, 사람 검토 상태, 결과 비교 규칙과 한계를 계획 문서에 기록한다.
- gold의 독립성·version 정책이 중요한 결정이면 ADR을 추가하고 날짜별 변경 이력을 갱신한다.
- 다음 체크포인트는 Phase 5 `RecordedModel` 또는 evaluation runner 중 이 평가 세트에 직접 의존하는 작은 단위로 정한다. 실제 모델 baseline 측정은 데스크톱 작업으로 남긴다.
- 과거 Conventional Commit 형식에 맞춰 계약/loader, 평가 사례, 검증·테스트, 문서를 분리 커밋한다. 각 커밋 전 staged diff를 확인하고 원격 push는 하지 않는다.

이번 단계까지만 개발하고 모델 실행·평가 runner·Query IR 확장·관리자 UI는 다음 작업으로 남겨줘.

---

<a id="prompts-phase5-01-lightweight-evaluation-runner"></a>

## 원본: prompts/phase5-01-lightweight-evaluation-runner.md

# 다음 개발 프롬프트: Phase 5 경량 evaluation runner

다음 단계인 **FakeModel·gold Query IR·세 SQLite fixture 기반 경량 evaluation runner**를 개발해줘.

작업 전 반드시 다음을 확인해.

- `docs/schema-adaptive-text-to-sql-plan.md` 전체, Phase 5 체크박스와 현재 체크포인트
- `docs/development-history.md#prompts-phase2-01-common-evaluation-questions`와 `backend/adaptive/evaluation`의 40개 gold 사례, loader, 정적 validator, typed multiset comparator, coverage report
- `backend/adaptive/harness`의 `QueryModel`, `FakeModel`, `ExecutionHarness`, `HarnessResult`·오류·trace 계약
- `SchemaAdaptiveSQLiteCompiler`, `SchemaAdaptiveAstPolicy`, `SQLiteReadOnlyExecutor`, `to_harness_mapping`, fixture별 Published Mapping과 Catalog FK graph
- 기존 gold cross-schema 실행 테스트와 저장소 Conventional Commit 기록
- 작업 시작 시 working tree 상태

## 목표

하나의 질문 세트를 flat·normalized·legacy 스키마에 동일하게 적용하고, **모델 출력 Query IR → 검증 → JOIN 계획·SQL 컴파일 → AST policy → read-only 실행 → gold semantic 결과 비교**를 사례별로 재현 가능하게 기록하는 runner를 만든다.

이번 단계는 runner와 지표의 동작 검증이다. `FakeModel`이 gold IR을 그대로 되돌리는 oracle smoke 결과는 모델 정확도나 학교 간 일반화 성능이 아니다. 모든 gold 사례가 `NEEDS_HUMAN_REVIEW`이므로 사람 승인 전 수치를 논문의 확정 성능으로 표기하지 않는다.

## 이번 범위

- versioned `EvaluationRunConfig`, `CaseAttempt`, `EvaluationRunReport` Pydantic 계약
- gold set·seed·ontology·Query IR·mapping/catalog fingerprint 및 variant 검증
- `QueryModel` 주입을 받는 결정적 runner core; 질문을 모델에 전달하고 임의의 gold IR을 내부에서 정답 예측으로 대체하지 않음
- 세 fixture 각각의 Published Mapping·FK graph와 현재 `ExecutionHarness`의 schema-adaptive compiler·AST policy·read-only executor 결합
- 각 `(case_id, variant)`의 구조적 IR 비교, 실행 상태, semantic row multiset 비교와 failure stage/code 기록
- 세 스키마 간 모델 출력 의도 일관성 및 결과 일관성의 명시적 판정
- 전체·variant별·coverage tag별 분모와 성공·실패·비교 불가 수 집계
- 결정적 JSON report core와 사람이 읽는 요약 출력
- gold 응답 `FakeModel`의 oracle smoke와 의도적으로 틀린 응답·실패 응답을 쓰는 합성 테스트
- 기존 평가 CLI에 분명한 별도 subcommand를 추가하거나 독립 CLI 진입점 제공
- 계획 문서의 Phase 5 evaluation runner 상태, 다음 체크포인트, 필요 시 ADR, 날짜별 변경 이력 갱신

## 실행·비교 계약

- 평가 단위는 `(benchmark version, case_id, fixture variant, model run id)`다. 사례는 고정 ID 순서, variant는 명시한 순서로 실행한다.
- `HarnessRequest.request_id`의 UUID 기본값은 사용하지 않고 사례·variant에서 결정적인 ID를 명시한다.
- 모델에는 사례의 **질문**과 기존 harness가 제공하는 ontology·Published Mapping runtime view만 전달한다. gold IR·기대 row·offering key를 모델 입력이나 prompt context에 넣지 않는다.
- gold IR을 반환하는 `FakeModel`은 `oracle-smoke`로 별도 표시한다. 그 결과는 runner 정합성 확인이며 benchmark score 집계나 논문 성능 표에 넣지 않는다.
- IR 비교는 원시 JSON 문자열이 아니라 현재 `QueryIR`의 typed 구조를 기준으로 한다. projection 순서, filter field/operator/typed value, 명시·기본 LIMIT 정책을 보존하고 비교 규칙을 테스트로 고정한다. `AND` filter 순서를 의미상 동일하게 볼지 여부는 별도 structural-exact 지표와 혼동하지 않게 명시한다.
- semantic 비교는 기존 `semantic_row_multiset()`을 재사용한다. 행 순서는 무시하되 중복 행 수, ontology type, 문자열 정확성을 보존한다.
- 모델 출력 projection field 집합이 gold와 다르면 semantic 결과를 같은 답으로 억지 비교하지 않는다. `NOT_COMPARABLE_PROJECTION`과 구조적 IR 불일치를 별도로 기록한다.
- 정상 실행했으나 값 또는 행 수가 다르면 `SEMANTIC_MISMATCH`다. 0건 gold도 명시적으로 비교한다.
- 모델 출력의 구조 오류, ontology·mapping 검증 실패, JOIN ambiguity, AST 차단, timeout과 SQL 실행 실패는 각각 기존 harness stage/error code로 집계한다. 오류를 0건 정답으로 계산하지 않는다.
- 예상한 `(case, variant)` 모두 하나의 결과를 가져야 한다. 일부 실패를 누락하거나 성공한 사례만 분모로 쓰지 않는다.
- 같은 모델 응답이 세 variant에서 서로 다른 Query IR 또는 semantic 결과를 만들면 `CROSS_SCHEMA_INCONSISTENCY`를 별도로 드러낸다. 단순 SQL 문자열 차이는 불일치가 아니다.

## 지표와 report 정책

최소한 다음을 **서로 다른 지표**로 정의한다.

- `structural_ir_exact`: gold Query IR과 동일한 typed 구조
- `semantic_correct`: 정상 실행 결과의 typed row multiset이 gold와 동일
- `cross_schema_consistent`: 동일 질문의 세 variant 실행 결과가 비교 가능하고 서로 동일
- `execution_success`: AST 정책과 read-only 실행을 포함한 harness 성공
- `not_comparable`: projection 불일치 등 정답 비교 조건 미충족
- `failure_by_stage_and_code`: 모델 출력부터 결과 검증까지의 실패 분포

각 지표의 분모는 전체 예정 attempt와 비교 가능 attempt를 구분해 기록하고, 실패·비교 불가를 조용히 제외한 높은 정확도처럼 표시하지 않는다. `semantic_correct`는 실행 성공률과 분리해 보고한다. case-level 세 스키마 일관성은 3개 attempt가 모두 존재할 때만 판정한다.

report에는 benchmark·seed·ontology·Query IR fingerprint, fixture catalog·Published Mapping fingerprint/version, runner/report schema version, model label과 `oracle_smoke` 여부, 예정·완료 attempt 수, 각 지표의 count·denominator, case별 상태와 reason을 넣는다. 실행 시각·임시 경로·UUID·측정 시간은 결정적 report core에서 제외한다. SQL·전체 row·모델 응답 원문은 기본 report에 저장하지 않는다. 합성 fixture용 제한된 debug artifact가 필요하면 명시적 opt-in과 길이 상한을 둔다.

현재 harness trace의 `elapsed_ms`는 wall-clock 값이라 byte-identical core에 넣지 않는다. 실제 지연·GPU·메모리 benchmark는 이번 단계에서 측정하거나 주장하지 않는다.

## 오류·안전 경계

- 모델 예외 또는 비정상 출력 한 건 때문에 전체 평가를 중단하지 않는다. runner 경계에서 구조화된 안전한 오류로 바꾸되 원문 응답·예외의 민감 문자열을 기본 report에 복제하지 않는다.
- gold set의 seed/ontology/Query IR drift나 fixture mapping/catalog drift는 **run 시작 전 fatal error**로 처리한다. 잘못된 기준으로 점수를 만들지 않는다.
- Mapping Proposal을 실행에 사용하지 않는다. synthetic fixture의 명시적 Published Mapping만 사용한다.
- `ExecutionHarness`·compiler·policy·executor의 안전 계층을 우회하거나 모델 생성 raw SQL을 직접 실행하지 않는다.
- 입력으로 실제 CNU dataset·private 가명 artifact를 요구하지 않는다. 그런 입력을 report에 혼합하지 않는다.
- 실패한 attempt의 SQL이나 row를 성공 결과처럼 기록하지 않고, 결과가 부분적으로 만들어졌더라도 집계 규칙을 명확히 한다.

## CLI 예시

구체적 module 경로와 옵션은 기존 구조를 조사해 정하되 사용 방식은 다음 수준으로 유지한다.

```powershell
python -m backend.adaptive.evaluation run-smoke `
  --gold backend/adaptive/evaluation/data/fixture_v1_cases.yaml `
  --variants flat,normalized,legacy `
  --output .schema-adaptive/reports/fixture-runner-smoke.json
```

CLI의 기본 provider는 명시적인 `oracle-smoke`여야 하며 report에 그 사실을 눈에 띄게 기록한다. gold 답을 사용한 smoke run에는 `model_accuracy`라는 이름을 붙이지 않는다. 기존 `--format json|review-table` inspection CLI 동작을 깨지 않는다. 출력은 기존 dataset CLI처럼 기본 덮어쓰기 금지, 명시적 `--force`에서만 원자 교체하고 입력 gold와 출력의 동일 경로를 차단한다.

## 필수 테스트

- 40개 사례 × 3 variant의 예정·완료 attempt 수와 안정적 순서
- oracle-smoke의 모든 정상 attempt 실행 및 gold semantic multiset 일치; 결과를 실제 모델 성능으로 표기하지 않음
- 같은 질문에서 variant별 물리 SQL은 달라도 semantic 결과가 같음
- 틀린 field·filter·typed value·LIMIT·projection 순서에 대한 structural IR 판정
- projection 불일치의 `NOT_COMPARABLE_PROJECTION`과 semantic mismatch의 분리
- gold 0건을 빈 정상 결과와 비교하고 실패를 빈 정답으로 오인하지 않음
- duplicate semantic row를 set으로 축약하지 않음
- malformed Query IR, 미매핑 field, JOIN 실패, AST 거부, 실행 오류, model 예외의 stage/code 및 다음 사례 계속 실행
- 한 variant만 다른 결과를 낼 때 case-level cross-schema inconsistency 검출
- 실패 포함 전체 분모와 variant·coverage별 count 합계 정합성
- seed·ontology·Query IR·mapping/catalog drift의 run 시작 전 차단
- SQL·row·모델 원문과 임시 경로가 기본 report에 없음
- 같은 입력·고정 응답의 byte-identical report core와 결정적 request ID
- CLI 기존 inspection 회귀, 출력 덮어쓰기·동일 경로 보호, 기존 전체 테스트 회귀 없음

## 제외 범위

- `RecordedModel`, Ollama, PostgreSQL, 기존 fine-tuned/base instruct 모델 실행
- 새 자연어 모델 prompt 설계나 few-shot context 실험
- 실제 모델 정확도·응답시간·메모리 benchmark와 논문 성능 주장
- gold의 사람 승인 자동 처리
- 새로운 평가 질문·seed·ontology·Query IR 또는 SQL compiler 확장
- 독립 unseen school 일반화 실험
- repair loop, 관리자 UI, schema evolution
- 실제 CNU 데이터셋·private 가명 artifact 사용

## 서브 에이전트

- 메인 에이전트가 먼저 지표의 분모, projection 비교 가능성, `oracle_smoke` 표기, 결정적 report core 계약을 정의한다.
- 독립 구현을 분배한다면 runner core, report/CLI, 적대적 테스트를 수정 파일이 겹치지 않게 나눈다.
- 메인 에이전트가 세 fixture 통합과 전체 테스트·문서 갱신을 담당한다.
- 구현 후 별도 리뷰 에이전트가 gold 누출, oracle 결과의 정확도 오인, 누락 attempt, failure denominator, 비교 불가 처리, AST·read-only 우회와 개인정보 노출을 점검한다.
- 리뷰 지적 수정 후 전체 테스트를 재실행한다.

## 완료 기준

- FakeModel 주입으로 40개 질문을 세 fixture에서 끝까지 반복 실행 가능
- 모델 출력과 gold를 혼동하지 않고 구조적 IR·semantic 정답·실행 성공·세 스키마 일관성을 분리 보고
- 실패와 비교 불가를 전체 분모에서 숨기지 않음
- oracle smoke와 `NEEDS_HUMAN_REVIEW` 경고가 report·CLI에 명시됨
- 결정적 report core, 안정적 failure taxonomy, 안전한 기본 출력 정책 검증
- 기존 inspection CLI와 legacy 시스템 보존
- 기존 전체 테스트 회귀 없음, `compileall`·`git diff --check` 통과, working tree clean

## 문서와 커밋

- Phase 5 `evaluation runner 구현`은 runner·테스트가 완료된 뒤에만 체크한다. `RecordedModel`과 실제 모델 평가 체크박스는 유지한다.
- 계획 문서에 runner 지표 분모, oracle smoke 경계, gold 사람 검토 상태, 실제 모델 평가를 위한 데스크톱 체크포인트를 기록한다.
- 지표·report 결정이 중요하면 ADR을 추가하고 날짜별 변경 이력을 갱신한다.
- 다음 시작점은 `RecordedModel`의 안전한 재생 계약 또는 사람 gold 검토 절차 중 의존성이 적은 단위로 지정한다.
- 과거 Conventional Commit 형식에 맞춰 runner 계약/core, report/CLI, 테스트, 문서를 분리 커밋한다. 각 커밋 전 staged diff를 확인하고 원격 push는 하지 않는다.

이번 단계까지만 개발하고 실제 모델 실행·성능 주장·repair loop·관리자 UI는 다음 작업으로 남겨줘.

---

<a id="prompts-phase5-02-recorded-model-replay-core"></a>

## 원본: prompts/phase5-02-recorded-model-replay-core.md

# 다음 개발 프롬프트: Phase 5-02 RecordedModel 재생 코어

다음 단계인 `RecordedModel`의 **오프라인 재생 코어만** 개발해줘. 이번 작업은 작은 단위로 끝낸다. 완료 후 멈추고 후속 작업을 자동으로 시작하지 마.

## 먼저 확인

- `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 5, 현재 체크포인트, 개인정보·평가 방침
- `backend/adaptive/harness/model.py`의 `QueryModel`·`FakeModel`, `HarnessRequest`와 Query IR 검증 경로
- `backend/adaptive/evaluation/runner.py`, `run_contracts.py`, `run_report.py`의 결정적 request ID·실패 집계 방식
- `docs/human-validation-checkpoints.md`의 조기 확인과 실제 모델 평가 전 승인 관문
- 작업 시작 시 `git status --short`, 최근 Conventional Commit 형식

## 목표와 경계

이미 **외부에서 기록된** 모델 응답을 파일에서 읽어 `QueryModel.generate_ir()`로 그대로 재생한다. 모델을 호출하거나 gold Query IR로 빈 응답을 채우지 않는다. 기록 파일의 구조와 재생 일치 여부만 검증하며, 이번 단계 결과를 모델 성능으로 주장하지 않는다.

현재 노트북에는 실행 중인 Ollama 서버와 PostgreSQL이 없다. 40개 synthetic gold 사례는 모두 `NEEDS_HUMAN_REVIEW`다. 세 SQLite 스키마도 독립된 다른 학교가 아니다. 이 경계는 테스트·문서에서 유지한다.

## 이번 범위

- versioned JSON recording 계약과 엄격한 loader. 입력 경로는 명시적으로 받는다.
- 각 기록의 결정적 `request_id`, `database_id`, 질문의 SHA-256 digest, runtime mapping의 안정적 fingerprint, 재생 결과를 저장한다. 모델 run 식별자와 recording schema version도 명시한다.
- 성공 기록은 **원래 JSON 값의 구조와 타입을 보존**해 반환한다. 잘못된 Query IR도 임의 수정하지 않아 기존 하네스가 `MODEL_INVALID_OUTPUT`으로 판정할 수 있게 한다.
- 기록된 모델 실패와 누락된 request는 gold로 대체하지 않고 명시적인 안전한 실패로 처리한다. runner에서 예정 attempt가 사라지지 않도록 한다.
- 중복 key, 잘못된 version, malformed JSON, 입력 계약 위반, question/database/mapping 불일치를 거부한다. 실패 메시지에 질문·응답·literal 원문을 넣지 않는다.
- 고정된 합성 recording으로 `RecordedModel`을 기존 `run_evaluation()`에 **Python API로만 주입**하는 소수의 통합 테스트를 추가한다. 기존 runner·CLI의 지표 계약은 바꾸지 않는다.
- 계획 문서의 Phase 5 진행 상태와 날짜별 변경 이력을 갱신한다. `RecordedModel` 체크박스는 이번 계약·테스트가 실제 완료된 경우에만 체크한다.

## 기록 파일 안전성

- 실제 모델 응답에는 질문·식별값이 들어갈 수 있으므로 실제 recording은 private artifact로 취급한다. 저장소에 실제 응답, 실제 CNU 데이터, 민감한 Query IR을 추가하지 않는다.
- 저장소에 넣는 테스트 자료는 **합성 값만** 사용한다. 기본 report에는 recording 원문을 복제하지 않는다.
- loader는 읽기 전용이다. 자동 다운로드, 모델 호출, 원본 파일 수정, recording 생성 기능은 넣지 않는다.
- `oracle_smoke=False`는 모델 성능의 증거가 아니다. 출처가 검증되지 않은 recording을 실제 모델 실행으로 주장하지 않도록 문서에 제한을 적는다.

## 최소 테스트

- 동일 입력의 결정적 재생; 질문·DB·mapping이 다른 기록의 오사용 차단
- JSON 객체·문자열·비정상 IR 등 원래 응답 타입 보존과 기존 검증기 실패 전달
- 기록된 오류·누락 항목이 runner의 실패 attempt로 남고 뒤 사례가 계속 실행됨
- 중복 ID, 잘못된 version, malformed JSON, 비정상 field type 거부
- 오류 메시지·기본 report에서 실제 질문·응답 값 비노출
- 합성 recording을 통한 flat/normalized/legacy 중 작은 사례 집합의 재생; 전체 40×3 재측정은 요구하지 않음
- 관련 단위 테스트, 기존 `tests/` 회귀, `compileall`, `git diff --check` 통과

## 제외 범위

- Ollama·PostgreSQL 연결과 실제 응답 수집
- recording 생성·내보내기 CLI, evaluation CLI의 새 subcommand
- 전체 40×3 benchmark 실행, 모델 정확도·지연·메모리 측정
- gold 사람 승인, Query IR·ontology·mapping·compiler 변경
- 관리자 UI, repair loop, 레거시 API 연결
- 앞선 점검에서 발견된 루트 `pytest`의 `backend/test_correction.py` import 문제 수정(별도 작업으로 기록만)

## 사용량 절감 규칙

- 이번 크기에서는 **서브 에이전트를 만들지 않는다**. 압축된 서브 에이전트 답변은 메인 대화 맥락을 줄일 수 있지만, 에이전트 실행 자체의 사용량을 없애지는 않는다.
- 현재 범위에 필요한 파일만 읽고 수정한다. 관련 테스트를 먼저 실행하고 전체 `tests/`는 최종 1회 실행한다. 긴 성공 로그는 요약한다.
- 예상 밖의 구조 변경이 필요하면 범위를 넓히지 말고 이유와 다음 작은 단계를 보고한다.
- 완료 보고는 변경 파일, 테스트 결과, 남은 위험·다음 체크포인트만 간결하게 쓴다.

## 이번 단계의 사람 확인

합성 recording 재생 코어의 자동 테스트를 위해 40개 gold의 정식 사람 승인을 요구하지 않는다. 완료 보고에는 `docs/human-validation-checkpoints.md`의 E1~E4 조기 확인 필요 여부를 짧게 알리고, 실제 모델 recording을 평가 결과로 사용하기 전 L1·L3, 실제 자료를 기록·공유하기 전 E3·L5가 필요함을 명시한다. 사람 승인 상태를 코드나 문서에서 임의로 완료 처리하지 않는다.

## 완료·커밋

기능과 테스트가 완료되면 과거 Conventional Commit 형식으로 적절히 분리 커밋한다. 커밋 전 staged diff를 확인한다. 원격 push는 하지 않는다. 작업 트리를 깨끗하게 마친다.

다음 작업은 **recording 수집·실행 CLI와 provenance 검증**으로 남긴다. 이번에는 재생 코어만 구현해줘.

---

<a id="prompts-phase5-03-recorded-evaluation-cli"></a>

## 원본: prompts/phase5-03-recorded-evaluation-cli.md

# 다음 개발 프롬프트: Phase 5-03 RecordedModel 평가 CLI

다음 단계인 **기존 recording을 evaluation runner에 연결하는 읽기 전용 CLI**만 개발해줘. 이번 작업이 끝나면 멈추고 recording 수집 기능은 시작하지 마.

## 먼저 확인

- `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 5와 현재 체크포인트
- `docs/human-validation-checkpoints.md`의 E3, L1, L3, L5
- `backend/adaptive/harness/recorded.py`
- `backend/adaptive/evaluation/__main__.py`, `run_cli.py`, `runner.py`, `run_contracts.py`, `run_report.py`
- 관련 테스트와 최근 Conventional Commit 형식
- 시작 시 `git status --short`

## 목표

외부에서 이미 생성된 `recorded-model-v1` JSON을 `RecordedModel`로 읽고, 기존 synthetic gold와 flat·normalized·legacy fixture를 대상으로 `run_evaluation()`을 실행하는 CLI를 추가한다.

이 CLI는 모델을 호출하거나 recording을 생성하지 않는다. recording의 `model_run_id`와 각 요청의 context fingerprint를 검증해 재생할 뿐이다. 결과는 재생 평가 report이며, recording 출처나 실제 모델 성능의 증명은 아니다.

## 이번 범위

- `python -m backend.adaptive.evaluation run-recorded` subcommand
- 명시적 `--recording`, `--gold`, `--variants`, `--model-label`, `--output`, `--force` 인자
- `load_model_recording()`과 `RecordedModel` 재사용
- `EvaluationRunConfig.model_run_id`를 recording 값에서만 설정
- `oracle_smoke=False` 고정; 사용자가 이를 바꾸는 옵션은 만들지 않음
- 기존 fixture bundle과 `run_evaluation()` 재사용
- report의 model run ID·label·variant가 실행 입력과 일치하는지 검증
- recording·gold·output 경로 충돌 차단과 기존 atomic non-overwrite 저장 정책 재사용
- 사람이 읽는 summary에 다음 경고 표시
  - offline replay이며 live model 실행이 아님
  - gold가 `NEEDS_HUMAN_REVIEW`이면 확정 성능이 아님
  - recording의 출처는 파일 내부 주장만으로 증명되지 않음
- 합성 recording 기반 CLI 테스트
- 계획 문서의 Phase 5 상태, 다음 체크포인트, 날짜별 변경 이력 갱신

## 실행 계약

- recording 전체 계약이 잘못되면 실행 전 fatal error로 종료하고 report를 만들지 않는다.
- 누락 record, recorded provider failure, question/database/mapping mismatch는 개별 attempt 실패다. 평가 전체를 중단하거나 gold로 대체하지 않는다.
- 모델 실패가 포함된 정상 evaluation run도 report 저장에 성공한다. 이는 CLI 인프라 실패와 구분한다.
- `model_run_id`를 CLI에서 별도로 받지 않는다. recording의 값과 다른 ID를 덮어씌울 경로를 만들지 않는다.
- `model_label`은 사람이 읽는 제한된 식별자일 뿐 provenance 증명이 아니다. 기존 `EvaluationRunConfig` 형식을 따른다.
- variant와 case 순서, request ID, report core 결정성은 기존 runner 계약을 유지한다.
- 기존 `run-smoke`와 inspection CLI 동작을 깨지 않는다.

## 개인정보·저장 안전

- 실제 recording은 private artifact다. 저장소에 복사하거나 수정하지 않는다.
- CLI와 report는 recording 경로, 질문, 응답 JSON, literal, SQL, 결과 row를 출력하지 않는다.
- 오류 메시지에도 recording payload를 포함하지 않는다.
- output은 gold 또는 recording과 같은 파일, symlink, hardlink가 될 수 없다.
- 기본은 덮어쓰기 금지다. `--force`에서만 원자 교체한다.
- 테스트 recording은 합성 질문·값만 사용한다.

## 최소 테스트

- 합성 recording으로 작은 case×variant 집합 실행 및 report 생성
- report의 `model_run_id`가 recording과 같고 `oracle_smoke=False`임
- 성공 response가 기존 harness와 semantic 비교까지 전달됨
- 누락·recorded failure·context mismatch가 attempt로 남고 뒤 사례가 계속됨
- attempt 실패가 있어도 정상 run report는 저장됨
- malformed recording과 잘못된 schema version은 report 생성 전 실패
- output과 recording/gold의 동일 경로 및 same-file 충돌 차단
- 기본 non-overwrite와 `--force` atomic replacement
- stdout·오류·report에 질문·응답·절대 recording 경로가 노출되지 않음
- 기존 inspection과 `run-smoke` 회귀 없음
- 관련 테스트 후 전체 `tests/` 최종 1회, `compileall`, `git diff --check` 통과

## 제외 범위

- Ollama 또는 다른 provider 호출
- recording 생성·수집·내보내기·수정
- 모델 설정, prompt, seed, temperature, 모델 파일 hash를 새 recording schema에 추가
- PostgreSQL 및 실제 CNU 데이터 사용
- 전체 40×3 실제 모델 benchmark, 지연·GPU·메모리 측정
- 모델 정확도 또는 학교 일반화 성능 주장
- gold 사람 승인 상태 변경
- Query IR·ontology·mapping·compiler·runner 지표 변경
- 관리자 UI, repair loop, 레거시 API 연결
- 루트 `pytest`의 `backend/test_correction.py` import 문제 수정

## 사용량 절감 규칙

- 범위가 작으므로 **서브 에이전트를 사용하지 않는다**.
- 위에 지정한 파일과 직접 관련된 테스트만 먼저 읽는다. 저장소 전체 재분석은 하지 않는다.
- 기존 loader, runner, 저장 helper를 재사용하고 새 추상화는 꼭 필요할 때만 만든다.
- 개발 중에는 관련 테스트만 실행하고 전체 `tests/`는 마지막에 한 번 실행한다. 프론트엔드 파일을 바꾸지 않으므로 frontend lint/build는 실행하지 않는다.
- recording schema 변경이나 수집기 구현이 필요해지면 범위를 넓히지 말고 이유와 후속 단계로 보고한다.
- 완료 보고는 변경 파일, 검증 결과, 사람 확인, 남은 한계만 짧게 쓴다.

## 사람 확인

합성 recording CLI 테스트에는 새 사람 승인이 필요 없다. 실제 recording을 만들거나 공유하기 전 E3·L5를 확인한다. 재생 report를 모델 성능으로 사용하기 전 L1·L3를 통과해야 한다. 승인 상태를 자동으로 변경하지 않는다.

## 완료·커밋

- CLI가 existing `RecordedModel`과 evaluation runner를 우회 없이 연결함
- 실패 attempt를 숨기지 않는 결정적 privacy-safe report 생성
- 기존 CLI·runner·legacy 시스템 회귀 없음
- `compileall`, `git diff --check` 통과, working tree clean
- 과거 Conventional Commit 형식으로 CLI, 테스트, 문서를 적절히 분리 커밋
- 각 커밋 전 staged diff 확인, 원격 push 금지

다음 작업은 **live provider의 응답을 안전하게 recording으로 수집하는 writer/capture 계약**으로 남긴다. 이번에는 읽기 전용 평가 CLI만 구현해줘.

---

<a id="prompts-phase5-04-recording-capture-session-core"></a>

## 원본: prompts/phase5-04-recording-capture-session-core.md

# 다음 개발 프롬프트: Phase 5-04 Recording capture session 코어

다음 단계인 **메모리 내 recording capture session 코어만** 개발해줘. 완료 후 멈추고 파일 저장, CLI, Ollama 호출은 시작하지 마.

## 먼저 확인

- `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 5와 현재 체크포인트
- `docs/human-validation-checkpoints.md`의 E3, L1, L3, L5
- `backend/adaptive/harness/model.py`, `contracts.py`, `recorded.py`
- `tests/adaptive/test_recorded_model.py`
- 최근 Conventional Commit 형식과 시작 시 `git status --short`

## 목표

외부 provider adapter가 전달한 성공 JSON 또는 명시적 실패 코드를 기존 `recorded-model-v1`의 `ModelRecording`으로 안전하고 결정적으로 조립하는 capture session을 만든다.

이번 단계는 provider를 호출하지 않는다. 파일도 쓰지 않는다. recording provenance를 증명하지도 않는다. 후속 storage·provider 계층이 재사용할 작은 코어 계약을 만드는 것이 목적이다.

## 이번 범위

- `model_run_id`로 시작하는 `RecordingCaptureSession` 계약
- `record_success(request, mapping, response)`
- `record_failure(request, mapping, failure_code)`
- request에서 `request_id`, `database_id`, question fingerprint를 생성
- runtime Mapping Profile에서 mapping fingerprint를 생성
- 기존 `RecordedSuccess`, `RecordedFailure`, `ModelRecording` 계약 재사용
- capture 순서를 보존한 결정적 `finalize()`
- 성공 response의 JSON 구조·타입과 값을 수정 없이 보존
- caller가 capture 후 원본 response를 변경해도 session 내부 기록은 변하지 않도록 방어적 복사
- 중복 request ID, 비정상 JSON 값, 비유한 숫자, 잘못된 실패 코드의 명시적 거부
- finalize 후 추가 capture 차단
- 빈 session finalize 차단
- 오류 메시지에서 질문·응답·literal 원문 비노출
- 단위 테스트와 계획 문서 갱신

## capture 계약

- hash를 호출자가 직접 전달하게 하지 않는다. session이 기존 `question_fingerprint()`와 `mapping_fingerprint()`로 계산한다.
- 성공 response는 Pydantic `JsonValue`로 손실 없이 검증 가능한 값만 허용한다.
- 문자열 response는 잘못된 Query IR일 수 있어도 유효한 JSON 값이므로 그대로 기록한다. Query IR 유효성 검사는 capture 계층이 대신하지 않는다.
- `NaN`, `Infinity`, bytes, 임의 객체 등 JSON recording으로 재현할 수 없는 값은 문자열로 변환하지 않고 안전한 capture 오류로 거부한다.
- provider 예외를 session이 추측해 분류하지 않는다. 호출 계층이 기존 `RecordedFailureCode` 중 하나를 명시적으로 전달한다.
- 같은 request ID를 다시 기록해 기존 결과를 덮어쓰거나 first/last wins로 처리하지 않는다.
- 성공과 실패가 같은 request ID로 들어와도 중복으로 거부한다.
- capture 순서가 `ModelRecording.records` 순서다. 자동 정렬하지 않는다.
- `finalize()`는 session을 seal하고 불변 `ModelRecording`을 반환한다. 반복 호출 허용 여부는 하나로 정해 테스트로 고정하되, 서로 다른 결과를 만들면 안 된다.
- 원본 request, mapping, response를 변경하지 않는다.

## 오류와 개인정보 경계

- 별도의 안전한 `RecordingCaptureError` 또는 동등한 통제 오류를 정의한다.
- 오류 code가 필요하면 enum으로 고정하고 원문 예외·payload를 message에 복제하지 않는다.
- 질문·response·literal·mapping sample value를 로그로 출력하지 않는다.
- capture session은 recording 내용을 report나 stdout으로 출력하지 않는다.
- 이번 단계에서는 thread safety나 여러 프로세스 동시 capture를 지원한다고 주장하지 않는다.

## 최소 테스트

- 성공 object, string, number, boolean, null, list의 JSON 타입 보존
- failure code별 `RecordedFailure` 생성
- question/database/mapping identity hash 정확성
- capture 순서와 동일 입력의 결정적 `ModelRecording`
- capture 후 caller response mutation이 결과를 바꾸지 않음
- 중복 request ID 거부와 기존 record 불변
- 성공 후 실패 및 실패 후 성공의 중복 차단
- 비유한 숫자, bytes, 임의 객체 거부
- 잘못된 failure code 거부
- 빈 finalize와 finalize 후 capture 차단
- 오류 문자열에 실제 질문·응답·literal 비노출
- 생성한 `ModelRecording`을 기존 `RecordedModel`로 정상 재생 가능
- 기존 recorded model·evaluation tests 회귀 없음
- 관련 테스트 후 전체 `tests/` 최종 1회, `compileall`, `git diff --check` 통과

## 제외 범위

- recording JSON 파일 저장, atomic write, overwrite 정책
- provenance manifest 또는 bundle
- model name·digest, prompt hash, temperature, seed, provider version 수집
- QueryModel wrapper, provider 예외 자동 분류
- Ollama·HTTP·PostgreSQL 연결
- capture·evaluation CLI 변경
- `run-recorded` 변경
- 실제 질문·응답 수집과 40×3 benchmark
- Query IR 수정·자동 보정·검증 정책 변경
- 관리자 UI, repair loop, 레거시 API 연결
- 루트 `pytest`의 기존 import 문제 수정

## 사용량 절감 규칙

- 작은 코어 작업이므로 **서브 에이전트를 사용하지 않는다**.
- 위에 지정한 파일과 직접 관련된 테스트만 읽는다. 저장소 전체 구조를 다시 조사하지 않는다.
- 기존 recording Pydantic 계약과 fingerprint 함수를 재사용한다.
- 새 파일은 가능하면 코어 1개와 테스트 1개 이내로 제한한다.
- 개발 중에는 관련 테스트만 실행하고 전체 `tests/`는 마지막에 한 번만 실행한다. 프론트엔드 검사는 실행하지 않는다.
- storage나 provider 변경이 필요해지면 범위를 넓히지 말고 후속 단계로 보고한다.
- 완료 보고는 변경 파일, 검증 결과, 남은 한계와 사람 확인만 간결하게 작성한다.

## 사람 확인

합성 response 기반 코어 테스트에는 새 사람 승인이 필요 없다. 실제 모델 응답을 capture하기 전 E3·L5를 확인한다. recording 기반 결과를 모델 성능으로 확정하기 전 L1·L3를 통과해야 한다. 승인 상태는 자동 변경하지 않는다.

## 완료·커밋

- 기존 `recorded-model-v1`과 호환되는 결정적 capture session 완성
- payload를 수정·보정·노출하지 않고 성공·실패를 기록
- 기존 replay와 evaluation 경로 회귀 없음
- `compileall`, `git diff --check` 통과, working tree clean
- 과거 Conventional Commit 형식으로 core, 테스트, 문서를 적절히 분리 커밋
- 각 커밋 전 staged diff 확인, 원격 push 금지

다음 작업은 **private recording atomic storage와 provenance manifest 계약**으로 남긴다. 그 이후에 provider adapter와 실제 수집 CLI를 개발해줘.

---

<a id="prompts-phase5-05-private-recording-storage-provenance"></a>

## 원본: prompts/phase5-05-private-recording-storage-provenance.md

# 다음 개발 프롬프트: Phase 5-05 Private recording storage와 provenance manifest

다음 단계인 **private recording atomic storage와 선언형 provenance manifest만** 개발해줘. 완료 후 멈추고 provider 호출·수집 CLI·evaluation 연동은 시작하지 마.

## 먼저 확인

- `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 5와 현재 체크포인트
- `docs/human-validation-checkpoints.md`의 E3, L1, L3, L5
- `backend/adaptive/harness/recorded.py`, `recording_capture.py`
- `backend/adaptive/dataset/pseudonym_storage.py`의 generation·manifest-last·private path 정책
- `tests/adaptive/test_recorded_model.py`, `test_recording_capture.py`
- 최근 Conventional Commit 형식과 시작 시 `git status --short`

## 목표

완성된 `ModelRecording`을 저장소의 gitignore된 private workspace에 결정적으로 저장하고, payload를 복제하지 않는 versioned provenance manifest와 함께 다시 검증해 읽을 수 있게 한다.

이번 provenance는 caller가 제공한 정보를 구조화하고 fingerprint로 묶는 **선언형 metadata**다. 실제로 특정 모델·설정에서 생성되었다는 암호학적 증명이나 독립 검증은 아니다.

## 이번 범위

- versioned `RecordingProvenance` Pydantic 계약
- versioned `RecordingBundleManifest`와 artifact metadata 계약
- 결정적 canonical JSON과 SHA-256 fingerprint
- `ModelRecording` 성공·실패·전체 record 수 집계
- content-addressed immutable generation
- recording artifact를 먼저 쓰고 manifest를 마지막 commit marker로 게시
- 기본 overwrite 금지와 명시적 overwrite 정책
- 저장 직후 byte hash·Pydantic 계약·record count 재검증
- bundle loader에서 manifest, recording hash, generation ID와 metadata 교차 검증
- `.schema-adaptive` 아래 private 경로만 허용
- symlink·git-tracked target·불완전 bundle·변조 artifact 차단
- 오류 메시지에서 recording payload·질문·응답·절대 경로 비노출
- 합성 recording 기반 storage·loader 테스트
- 계획 문서의 Phase 5 상태와 다음 체크포인트 갱신

## provenance 계약

최소한 다음 정보를 가진다. 실제 필드명은 기존 명명 규칙에 맞추되 의미는 유지한다.

- provenance schema version
- `verification_status=DECLARED_UNVERIFIED` 고정
- provider kind: `SYNTHETIC`, `OLLAMA`, `OPENAI_COMPATIBLE`, `OTHER` 중 하나
- provider ID와 provider/adapter version
- model ID
- 선택적 model artifact SHA-256
- prompt template SHA-256
- generation configuration SHA-256
- collector/tool version

provider ID, model ID, version은 제한된 길이의 안전한 label이어야 한다. 로컬 파일 경로, URL credential, prompt 원문, 환경변수, 사용자 이름을 넣지 않는다. hash 입력의 원문이나 비밀값은 manifest에 저장하지 않는다.

같은 recording과 같은 provenance는 동일한 generation ID와 manifest core를 만든다. timestamp, 임시 경로, UUID, hostname처럼 실행마다 달라지는 값은 넣지 않는다. 실제 수집 시각이 필요하면 후속 실행 log에서 별도로 다루고 deterministic bundle identity와 섞지 않는다.

## bundle 저장 계약

기존 pseudonymization bundle의 검증된 패턴을 재사용하되 recording에 필요한 최소 구조만 만든다.

```text
<private-output>/
  manifest.json
  generations/
    <generation-id>/
      recording.json
```

- generation ID는 recording bytes와 provenance canonical metadata를 포함한 material의 SHA-256으로 계산한다.
- `recording.json`은 기존 `ModelRecording`의 결정적 UTF-8 JSON 직렬화다.
- manifest에는 상대 경로, byte 수, SHA-256, recording schema version, model run ID, record count와 provenance만 둔다.
- 질문·response JSON·literal·SQL·absolute path를 manifest에 복제하지 않는다.
- generation은 불변이다. 같은 ID가 이미 있으면 모든 bytes와 계약이 정확히 같은 경우만 재사용한다.
- 새 generation을 임시 directory에 완성·검증한 뒤 publish한다.
- `manifest.json`을 마지막에 원자적으로 게시한다. 기본은 기존 manifest를 덮어쓰지 않는다.
- overwrite가 허용되더라도 기존 generation을 수정하지 않고 commit marker만 검증된 generation으로 교체한다.
- 저장 실패 시 이전 정상 manifest를 손상시키거나 일부 generation을 정상 bundle로 보이게 하지 않는다.

## private path와 안전 정책

- caller가 `repository_root`와 output directory를 명시한다.
- output은 repository의 `.schema-adaptive`보다 아래에 있어야 한다.
- repository root, `.schema-adaptive` 자체, tracked path에는 쓰지 않는다.
- private root, output, generations, artifact, manifest의 symlink redirect를 거부한다.
- git tracking 여부를 확인할 수 없으면 안전하게 실패한다.
- loader는 manifest가 가리키는 경로가 bundle 밖으로 탈출하지 못하게 한다.
- 예상하지 않은 artifact, 잘못된 hash/count/version/model run ID를 명시적으로 거부한다.
- 오류에는 payload, 절대 경로, 내부 예외 원문을 넣지 않는다.

## loader 계약

- loader 입력은 bundle directory 또는 manifest path 중 하나로 명확히 고정한다.
- manifest가 없거나 generation이 불완전하면 실패한다.
- recording bytes의 hash와 크기를 먼저 확인하고 기존 strict `load_model_recording()`과 동일한 JSON 계약으로 검증한다.
- manifest의 model run ID·schema version·record count가 실제 recording과 정확히 같아야 한다.
- provenance fingerprint와 generation ID를 다시 계산한다.
- 성공 시 검증된 `ModelRecording`과 manifest를 반환하되 payload를 stdout에 출력하지 않는다.
- loader는 bundle을 수정하거나 자동 복구하지 않는다.

## 최소 테스트

- 합성 `ModelRecording`과 provenance의 정상 bundle 저장·load
- 같은 입력의 동일 canonical bytes, hashes, generation ID와 manifest core
- 성공·실패·전체 count 정확성
- provenance 필수 field, enum, label 길이, SHA-256 형식 검증
- `DECLARED_UNVERIFIED` 이외 상태 거부
- prompt·configuration 원문 대신 hash만 저장됨
- 기본 overwrite 차단과 명시적 overwrite의 manifest-only 교체
- 기존 immutable generation이 변경되지 않음
- recording, manifest, hash, byte count, model run ID, record count 변조 탐지
- incomplete generation과 missing manifest 거부
- bundle 밖 상대 경로와 path traversal 거부
- `.schema-adaptive` 밖, repository root, symlink, tracked target 차단
- 저장 실패 시 이전 manifest 보존 및 임시 artifact 정리
- 오류·manifest에 질문·응답·literal·절대 경로 비노출
- loader가 기존 `RecordedModel`에서 재생 가능한 recording 반환
- 기존 capture·replay·evaluation 테스트 회귀 없음
- 관련 테스트 후 전체 `tests/` 최종 1회, `compileall`, `git diff --check` 통과

## 제외 범위

- Ollama·HTTP·PostgreSQL 연결
- provider adapter와 예외 분류
- live response 수집·capture orchestration
- storage·capture·evaluation CLI 추가 또는 변경
- `run-recorded`에서 bundle 직접 읽기
- 실제 recording과 실제 CNU 데이터 사용
- 암호화, 서명, remote attestation, 비밀키 관리
- latency·GPU·메모리 측정과 모델 성능 주장
- Query IR·mapping·compiler·runner 변경
- 관리자 UI, repair loop, 레거시 API 연결
- 루트 `pytest`의 기존 import 문제 수정

## 사용량·서브 에이전트 규칙

- 기본은 메인 에이전트 단독 구현이다. provenance 계약과 atomic storage가 같은 bundle 불변식을 공유하므로 병렬 builder는 만들지 않는다.
- 다만 Windows path·symlink·atomic publish·tamper 경계가 예상보다 커져 독립 보안 검토가 실질적으로 필요한 경우에만 **리뷰 에이전트 1명**을 사용할 수 있다. 이때 코드를 분담하지 말고 완성된 diff의 경로 탈출·부분 저장·payload 노출만 검토하게 한다.
- 이는 서브 에이전트의 전면 금지가 아니다. 후속 provider adapter와 수집 CLI가 서로 독립적인 구현 단위로 커지면 제한적으로 분담할 수 있다.
- 지정한 파일과 직접 관련된 storage 테스트만 읽고 저장소 전체를 다시 분석하지 않는다.
- pseudonym storage 패턴을 참고하되 recording에 불필요한 구조를 복사하지 않는다.
- 개발 중에는 관련 테스트만 실행하고 전체 `tests/`는 마지막에 한 번만 실행한다. 프론트엔드 검사는 실행하지 않는다.
- provider 호출이나 CLI 변경이 필요하면 범위를 넓히지 말고 후속 단계로 보고한다.
- 완료 보고는 변경 파일, 검증 결과, 남은 한계와 사람 확인만 간결하게 작성한다.

## 사람 확인

합성 recording storage 테스트에는 새 사람 승인이 필요 없다. 실제 recording을 저장·반출·공유하기 전 E3·L5를 확인한다. bundle provenance는 선언 정보이며 승인 증거가 아니다. recording 기반 결과를 모델 성능으로 확정하기 전 L1·L3를 통과해야 한다.

## 완료·커밋

- private atomic bundle 저장과 strict loader가 결정적으로 동작
- manifest와 generation이 recording·provenance에 content-bound됨
- payload 비노출과 path·symlink·tamper 안전 정책 검증
- 기존 capture·replay·evaluation 회귀 없음
- `compileall`, `git diff --check` 통과, working tree clean
- 과거 Conventional Commit 형식으로 contracts, storage/loader, 테스트, 문서를 적절히 분리 커밋
- 각 커밋 전 staged diff 확인, 원격 push 금지

다음 작업은 **provider-neutral capture orchestrator와 실패 분류 계약**으로 남긴다. Ollama adapter와 실제 수집 CLI는 그 이후 데스크톱 단계에서 개발해줘.

---

<a id="prompts-phase5-06-provider-neutral-capture-orchestrator"></a>

## 원본: prompts/phase5-06-provider-neutral-capture-orchestrator.md

# 다음 개발 프롬프트: Phase 5-06 Provider-neutral capture orchestrator

다음 단계인 **provider-neutral recording capture orchestrator와 명시적 실패 분류 계약만** 개발해줘. 완료 후 멈추고 파일 저장, CLI, Ollama·HTTP 호출은 시작하지 마.

## 먼저 확인

- `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 5와 현재 체크포인트
- `docs/human-validation-checkpoints.md`의 E3, L1, L3, L5
- `backend/adaptive/harness/model.py`, `contracts.py`, `recorded.py`
- `backend/adaptive/harness/recording_capture.py`
- `backend/adaptive/harness/recording_bundle_contracts.py`, `recording_bundle_storage.py`
- `tests/adaptive/test_recording_capture.py`
- 최근 Conventional Commit 형식과 시작 시 `git status --short`

## 목표

외부 provider adapter를 주입받아 정해진 request를 순서대로 호출하고, 각 성공 JSON 또는 명시적 실패를 기존 `RecordingCaptureSession`에 기록하는 작은 orchestration 코어를 만든다.

이번 단계는 provider-neutral 경계와 실패 의미만 고정한다. 실제 provider를 호출하는 adapter, recording 파일 저장, provenance 생성, CLI는 구현하지 않는다.

## 이번 범위

- provider 호출을 나타내는 최소 Protocol 또는 callable 계약
- 기존 `HarnessRequest`, `Ontology`, `MappingProfile`을 묶는 capture target 계약
- 기존 `RecordedFailureCode`만 사용하는 typed provider failure 계약
- 입력 순서를 보존하는 `capture_recording(...)` 또는 동등한 orchestration API
- 각 target마다 provider를 정확히 한 번 호출
- 성공 반환값을 보정하지 않고 `RecordingCaptureSession.record_success()`로 전달
- 통제된 provider 실패를 `record_failure()`로 변환하고 다음 target 계속 처리
- 예상하지 못한 provider 예외를 payload 비노출 `PROVIDER_ERROR`로 기록하고 계속 처리
- invalid JSON/non-finite response의 명시적 실패 처리
- 중복 request ID와 request/mapping database 불일치의 호출 전 차단
- 완료 시 기존 `ModelRecording` 반환
- 단위 테스트와 계획 문서 갱신

## provider와 실패 계약

- provider는 `request`, `ontology`, `mapping`을 받고 Python JSON 값 또는 JSON으로 검증될 값을 반환한다.
- 문자열, 숫자, boolean, `null`, list, object는 Query IR로 유효한지와 무관하게 성공 응답으로 기록한다. Query IR 검증은 이 계층의 책임이 아니다.
- `None`은 JSON `null`이므로 orchestrator가 빈 응답으로 추측하지 않는다. 실제 빈 응답은 adapter가 typed `EMPTY_RESPONSE` 실패로 알려야 한다.
- timeout, empty response, cancellation, 일반 provider failure는 기존 `RecordedFailureCode.TIMEOUT`, `EMPTY_RESPONSE`, `CANCELLED`, `PROVIDER_ERROR` 중 하나로만 전달한다.
- 예외 문자열, HTTP body, prompt, 질문, 모델 응답을 실패 code나 message에 복제하지 않는다.
- 문자열 matching으로 timeout이나 provider 종류를 추측하지 않는다.
- provider가 bytes, 임의 객체, `NaN`, `Infinity` 등 기록할 수 없는 값을 반환하면 문자열로 바꾸지 않고 해당 attempt를 `PROVIDER_ERROR`로 남긴다.
- 한 attempt의 실패는 다음 attempt를 막지 않는다. 반면 입력 계약 자체가 잘못된 경우에는 provider를 한 번도 호출하지 않고 전체 orchestration을 거부한다.

## 결정성과 실행 정책

- target 순서가 provider 호출 순서이자 `ModelRecording.records` 순서다. 자동 정렬하지 않는다.
- retry, backoff, concurrency, timeout 측정은 이번 단계에서 구현하지 않는다.
- request ID를 자동 생성하거나 교체하지 않는다. caller가 결정적인 ID를 제공해야 한다.
- gold Query IR, 기대 결과 또는 fallback response를 사용하지 않는다.
- mapping fingerprint와 question fingerprint는 기존 `RecordingCaptureSession`이 계산하게 하며 중복 구현하지 않는다.
- provider 반환값, request, ontology, mapping을 수정하지 않는다.
- 전체 target이 실패하더라도 각 실패 record가 존재하면 유효한 `ModelRecording`을 반환한다.
- 빈 target 목록은 provider 호출 없이 명시적으로 거부한다.

## 오류·개인정보 경계

- orchestration 입력 오류와 provider attempt 실패를 구분한다.
- 통제 오류가 필요하면 고정 enum code를 사용하고 안전한 message만 제공한다.
- 질문, response, literal, prompt, mapping sample value, provider 예외 원문을 로그·stdout·오류 문자열에 넣지 않는다.
- recording을 파일에 쓰거나 bundle storage를 자동 호출하지 않는다.
- model/provider provenance를 생성하거나 검증했다고 주장하지 않는다.

## 최소 테스트

- object·string·number·boolean·null·list 성공 응답의 순서와 타입 보존
- typed timeout·empty·cancelled·provider failure가 해당 `RecordedFailureCode`로 기록됨
- 실패 다음 target도 호출되고 전체 record 수가 유지됨
- 예상 밖 provider 예외가 원문 비노출 `PROVIDER_ERROR`가 되고 계속 진행됨
- invalid JSON 값과 non-finite number가 보정 없이 `PROVIDER_ERROR`로 기록됨
- 문자열 형태의 잘못된 Query IR도 성공 JSON으로 보존됨
- provider는 target당 정확히 한 번 호출되고 retry하지 않음
- 중복 request ID, database 불일치, 빈 target을 provider 호출 전에 거부
- 입력 순서와 동일 입력의 결정적 `ModelRecording`
- request·ontology·mapping·response 원본 불변
- 오류 문자열에 실제 질문·응답·provider 예외 원문 비노출
- 생성 결과를 기존 `RecordedModel`로 재생 가능
- 기존 capture·bundle·replay·evaluation 테스트 회귀 없음
- 관련 테스트 후 전체 `tests/` 최종 1회, `compileall`, `git diff --check` 통과

## 제외 범위

- Ollama·HTTP·OpenAI-compatible provider adapter
- 실제 timeout 구현, retry, concurrency, rate limit
- recording bundle 저장·load 변경
- provenance manifest 생성·수정
- capture 또는 evaluation CLI
- 실제 모델 응답과 CNU 자료 수집
- 전체 40×3 benchmark와 모델 성능 측정
- Query IR 검증·수정·자동 보정
- PostgreSQL, 관리자 UI, repair loop, 레거시 API 연결
- 루트 `pytest`의 기존 import 문제 수정

## 사용량·서브 에이전트 규칙

- 이번 범위는 orchestrator 코어 1개와 테스트 1개 수준이므로 **메인 에이전트 단독 구현을 기본값**으로 한다.
- 이는 서브 에이전트 전면 금지가 아니다. 조사 과정에서 provider 경계와 기존 capture 계약을 함께 바꿔야 하는 예상 밖의 독립 작업이 발견되어 품질상 분리가 필요할 때만 리뷰 에이전트 1명을 사용할 수 있다.
- 지정한 파일과 직접 관련된 테스트만 먼저 읽고 저장소 전체를 다시 분석하지 않는다.
- 기존 `RecordingCaptureSession`과 `RecordedFailureCode`를 재사용하고 유사 계약을 중복 생성하지 않는다.
- 개발 중에는 관련 테스트만 실행하고 전체 `tests/`는 마지막에 한 번만 실행한다. 프론트엔드 검사는 실행하지 않는다.
- storage, CLI, 실제 adapter 변경이 필요해지면 범위를 넓히지 말고 후속 단계로 보고한다.
- 완료 보고는 변경 파일, 검증 결과, 남은 한계와 사람 확인만 간결하게 작성한다.

## 사람 확인

합성 provider 테스트에는 새 사람 승인이 필요 없다. 실제 모델을 호출·기록·저장하기 전 E3·L5를 확인한다. recording 기반 결과를 모델 성능으로 확정하기 전 L1·L3를 통과해야 한다. 승인 상태는 자동 변경하지 않는다.

## 완료·커밋

- provider-neutral orchestrator가 성공·실패를 누락 없이 기존 `ModelRecording`으로 조립
- attempt 실패 후 계속 진행하면서 입력 계약 오류는 호출 전에 차단
- payload 비노출과 결정적 순서 검증
- 기존 capture·bundle·replay·evaluation 회귀 없음
- `compileall`, `git diff --check` 통과, working tree는 이 프롬프트 작성 전 변경을 제외하고 깨끗함
- 과거 Conventional Commit 형식으로 core, 테스트, 문서를 적절히 분리 커밋
- 각 커밋 전 staged diff 확인, 원격 push 금지

다음 작업은 **Ollama adapter의 최소 호출·오류 변환 계약**으로 남긴다. 실제 수집 CLI와 40×3 모델 평가는 그 이후 데스크톱 단계에서 개발해줘.

---

<a id="prompts-phase5-07-ollama-querymodel-adapter"></a>

## 원본: prompts/phase5-07-ollama-querymodel-adapter.md

# 다음 개발 프롬프트: Phase 5-07 Ollama QueryModel adapter

**Ollama `QueryModel` adapter의 최소 호출·오류 변환 계약만** 구현해줘. 합성 transport 테스트까지 완료하고 멈춰. 실제 응답 수집·저장 CLI, 40×3 평가, PostgreSQL 통합은 다음 단계로 남겨줘.

## 먼저 확인

- `AGENTS.md`, `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 5·다음 시작점
- `docs/development-history.md#desktop-test-handoff`, `docs/experiment-results.md#desktop-smoke-results-2026-09-19`
- `docs/human-validation-checkpoints.md`의 E3, L1, L3, L5
- `backend/adaptive/harness/model.py`, `contracts.py`, `recording_orchestrator.py`, `recorded.py`, `engine.py`
- `backend/llm.py`의 legacy 경계와 `tests/adaptive/test_recording_orchestrator.py`
- 시작 시 `git status --short`와 최근 Conventional Commit 형식

## 목표와 계약

기존 `QueryModel.generate_ir(request, ontology, mapping)` 계약을 구현하는 Ollama adapter를 추가한다. 합성 transport를 주입해 네트워크 없이 호출·파싱·실패 분류를 검증할 수 있어야 한다. 기존 legacy `backend/llm.py`의 SQL 생성기와 프롬프트를 재사용하거나 변경하지 않는다.

- adapter 설정은 base URL, 모델 tag, timeout을 명시적으로 받는다. `.env` 읽기는 호출자 책임으로 두고 모델명을 추측·치환하지 않는다. secret·DSN을 설정이나 오류에 넣지 않는다.
- 한 요청은 현재 로컬 Ollama에서 확인한 `/api/generate`에 `stream: false`로 보낸다. 호출당 retry·fallback·동시 실행은 없다.
- 결정적이고 version을 식별할 수 있는 최소 프롬프트를 사용한다. 질문과 runtime ontology의 entity·field·자료형·허용 연산자, Published Mapping에서 사용 가능한 **semantic field 이름**만 포함한다. 물리 table/column, 실제 행, gold IR, 기대 결과, 비밀 설정은 보내지 않는다. 출력 형식은 공통 `QueryIR` JSON으로 지시한다.
- `temperature: 0` 등 실제 전송 설정을 테스트로 고정한다. 프롬프트와 설정의 원문을 로그·예외에 넣지 않는다. 향후 provenance에 필요한 식별자나 hash가 있다면 코드에서 재현 가능하게 하되 이번 단계에서 manifest를 생성하지 않는다.
- HTTP 응답의 envelope는 별도로 검사한다. 성공 envelope의 `response`가 비어 있거나 공백뿐이면 기존 `RecordingProviderError(EMPTY_RESPONSE)`를 발생시킨다.
- 비어 있지 않은 모델 출력이 유효한 JSON이면 Python JSON 값으로 반환한다. 객체, 배열, 문자열, 숫자, boolean, `null`의 타입을 보존한다. 중복 key·비유한 수 등 JSON 의미가 불확실한 출력은 유효한 객체로 조용히 바꾸지 않는다.
- 모델 출력이 SQL이나 설명문처럼 JSON이 아니면 **원래 텍스트를 문자열로 반환**한다. 기존 orchestrator가 이를 성공 JSON 문자열로 보존하고, 후속 `QueryIR` 검증이 `MODEL_INVALID_OUTPUT`으로 판정하게 한다. SQL을 QueryIR로 번역·보정하거나 provider 장애로 위장하지 않는다.
- 네트워크 timeout은 `RecordingProviderError(TIMEOUT)`, 실제 cancellation은 `CANCELLED`, HTTP 오류·연결 실패·잘못된 envelope 등 provider 호출 오류는 `PROVIDER_ERROR`로 변환한다. 취소 신호를 지원하지 않는 transport라면 취소를 임의 추측하지 말고 지원 범위를 명시한다.
- provider 예외 원문, HTTP body, 모델 출력, 질문, 프롬프트, 로컬 경로를 오류 문자열·stdout·기본 로그에 포함하지 않는다. 오류 원인은 고정 code로만 외부에 전달한다.

## 최소 테스트

- 합성 transport가 정확한 URL, 모델 tag, `stream: false`, timeout, 안정적인 프롬프트·설정을 받음
- 정상 JSON 객체와 다른 JSON 타입의 보정 없는 반환
- legacy 형태 SQL 텍스트·Markdown fence·설명문을 그대로 문자열로 반환하고, 실제 harness에서 유효한 QueryIR로 오인하지 않음
- 빈 문자열·공백 응답은 `EMPTY_RESPONSE`; JSON `null`은 성공 값으로 구분
- timeout, 연결 실패, HTTP 오류, malformed envelope, 취소의 명시적 실패 code
- 모델명 자동 fallback과 재시도가 없고 호출당 transport 실행 1회
- 예외·stdout에 질문, 모델 응답, HTTP body, 프롬프트, 로컬 경로가 노출되지 않음
- 기존 `capture_recording()`에 연결했을 때 성공 값과 typed failure가 입력 순서대로 기록되고, 실패 뒤 다음 target을 처리함
- `QueryIR` 구조 검증은 adapter 밖에서 수행됨. JSON 객체여도 잘못된 field·operator는 기존 validator가 거부함
- 관련 테스트를 먼저 실행하고 마지막에 `python -m pytest tests -q`, `python -m compileall -q backend tests`, `git diff --check` 실행. Windows에서는 `.venv/Scripts/python.exe` 사용 가능

## 제외 범위

- 실제 CNU 질문·SQL·DB 행을 모델로 전송
- 실제 모델 응답 recording 저장, private bundle·provenance·수집 CLI 변경
- `run-recorded` 또는 40×3 benchmark 실행과 모델 정확도 수치 발표
- 모델 weight·Modelfile 변경, 재학습, prompt 탐색·성능 최적화
- PostgreSQL adapter, 기존 FastAPI/React 연결, repair loop, 관리자 UI
- 루트 `pytest`의 기존 `backend/test_correction.py` import 문제 해결

## 사람 확인과 완료 보고

합성 transport 테스트에는 새 사람 승인이 필요 없다. 추후 실제 응답을 기록·저장·공유하기 전 E3·L5를 확인하고, 모델 성능 확정 전 L1·L3를 확인한다. 기존 gold 40건의 `NEEDS_HUMAN_REVIEW` 상태를 바꾸지 않는다.

완료 보고에는 변경 파일, 테스트 결과, 실제 Ollama를 호출했는지 여부, 현재 SQL 생성 모델의 한계, 다음 작은 작업만 적어줘. 설계·로드맵 상태가 바뀌면 `docs/schema-adaptive-text-to-sql-plan.md`의 해당 본문·체크박스·변경 이력을 갱신해줘. 과거 형식의 Conventional Commit으로 적절히 커밋하되 push는 하지 마.

---

<a id="prompts-phase5-08-ollama-adapter-live-smoke"></a>

## 원본: prompts/phase5-08-ollama-adapter-live-smoke.md

# 다음 개발 프롬프트: Phase 5-08 Ollama adapter 합성 live smoke

`OllamaQueryModel`을 **로컬 Ollama와 소수의 합성 입력으로만** 호출해 Phase 5-07의 실제 전송·응답 분류 경계를 확인해줘. 구현이 필요한 경우에는 명시적으로 실행해야 하는 작은 smoke 진입점과 그 합성 테스트만 추가해줘. 실제 응답 recording·저장 CLI, DB 실행, 40×3 평가는 하지 마.

## 먼저 확인

- `AGENTS.md`, `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 5·다음 시작점·변경 이력
- `docs/development-history.md#desktop-test-handoff`, `docs/experiment-results.md#desktop-smoke-results-2026-09-19`, `docs/human-validation-checkpoints.md`의 E3·L1·L3·L5
- `backend/adaptive/harness/ollama.py`, `contracts.py`, `validator.py`, `engine.py`, `tests/adaptive/test_ollama_query_model.py`
- 시작 시 `git status --short`, 최근 Conventional Commit, 로컬 `ollama list`와 `/api/tags`에서 실제 사용 가능한 **정확한 모델 tag**. 없거나 서버가 응답하지 않으면 다른 모델로 대체하지 말고 환경 미준비로 보고해줘.

## 작은 구현·실행 범위

1. 현재 adapter의 기본 urllib transport가 로컬 `/api/generate`에서 동작하는지 확인한다. 모델 tag, base URL, timeout은 호출자가 명시하며, 모델명 추측·fallback·retry는 하지 않는다. `backend/llm.py`와 legacy 프롬프트·API는 변경하지 않는다.
2. 필요하다면 수동 실행 전용 smoke 진입점을 추가한다. **명시적 `--live` 없이는 HTTP 호출하지 않고**, 합성 질문 2~3개와 고정된 합성 runtime `Ontology`·`PUBLISHED MappingProfile`만 사용한다. 물리 mapping은 합성 이름을 쓰되 adapter 전송에는 semantic 이름만 들어가야 한다. 실제 DB 연결, SQL 실행, gold IR·기대 결과 사용은 금지한다.
3. 호출당 adapter 실행은 한 번만 한다. 원 응답을 파일·recording·report·로그·stdout에 저장하지 않는다. 각 합성 case의 비식별 ID, 모델 tag·가능하면 digest, prompt version, 설정의 안전한 식별 정보, JSON 파싱 여부·JSON 타입, `QueryIR` 구조 검증 여부, semantic field/operator 검증 결과 코드, provider 실패 code, 대략적인 경과 시간만 요약한다. 질문·응답·프롬프트·HTTP body·로컬 경로·DSN은 출력하지 않는다. 모델 출력이 SQL이나 Markdown이면 원문을 보이지 않고 `NON_JSON_TEXT`로만 분류한다.
4. `QueryIR` 검증은 adapter 밖에서 한다. 모델 출력이 JSON 객체라도 잘못된 구조·field·operator를 성공으로 세지 않는다. SQL 텍스트를 QueryIR로 번역하거나 프롬프트를 즉흥적으로 바꾸지 않는다. 기존 fine-tuned SQL 생성 모델이 QueryIR을 못 만들 수 있음을 실패가 아닌 관찰로 기록한다.
5. 순수 합성 transport 테스트에서 opt-in 경계, 단일 호출, 출력 비노출, JSON/비 JSON/typed failure의 분류를 확인한다. 네트워크 오류나 모델 부재는 fixed code만 표시하고 실제 live smoke를 성공으로 표시하지 않는다. timeout 이후 강제 취소를 구현한 것으로 주장하지 않는다.
6. **실제 live 호출이 가능할 때만** 합성 질문으로 smoke를 실행한다. 결과는 안전한 요약만 새 문서에 남긴다. 실행 불가하면 원인 코드·미검증 상태를 남기고 멈춘다. 모델 weight, Modelfile, 데이터셋, legacy 시스템은 변경하지 않는다.

## 제외 범위와 사람 확인

- 실제 CNU 질문·SQL·DB 행·식별값을 모델로 보내지 않는다.
- `capture_recording()`, private bundle, provenance manifest, 수집·저장 CLI, `run-recorded`, 40×3 benchmark, PostgreSQL adapter, FastAPI/React 연결은 건드리지 않는다.
- 이번 **합성 입력의 비저장 live smoke**는 E3·L5의 실제 데이터·응답 recording 승인을 대신하지 않는다. 이후 실제 응답을 기록·저장·공유하기 전 E3·L5를 확인하고, 모델 성능 수치를 확정하기 전 L1·L3를 확인한다. gold 40건의 `NEEDS_HUMAN_REVIEW`는 유지한다.
- 모델 정확도, 학교 간 일반화, PostgreSQL adaptive 통합 성공 수치를 발표하지 않는다.

## 검증·완료 보고

- 관련 합성 테스트를 먼저 실행한다. 마지막에 `.venv/Scripts/python.exe -m pytest tests -q`, `.venv/Scripts/python.exe -m compileall -q backend tests`, `git diff --check`를 실행한다. 루트 `pytest`의 기존 `backend/test_correction.py` import 문제는 범위 밖이다.
- `docs/schema-adaptive-text-to-sql-plan.md`에서 실제 완료한 범위만 체크하고 다음 시작점·변경 이력을 갱신한다. live smoke 결과 문서에는 환경·실행 여부·안전한 형식 분류·실패 코드·미검증 사항만 기록한다.
- 완료 보고에 변경 파일, 테스트, 실제 Ollama 호출 여부, SQL 생성 모델의 QueryIR 한계, 다음 작은 작업을 적는다. 과거 형식의 Conventional Commit으로 커밋하되 push는 하지 않는다.

---

<a id="prompts-phase5-09-private-capture-provenance-synthetic"></a>

## 원본: prompts/phase5-09-private-capture-provenance-synthetic.md

# 다음 개발 프롬프트: Phase 5-09 private capture·provenance 합성 연결

기존 `capture_recording()`과 `write_recording_bundle()`을 작은 수집 경계로 연결해줘. **이번 단계는 합성 provider와 합성 target으로만** capture→private bundle 저장→strict load를 검증한다. Ollama provenance를 실제 adapter 설정에서 재현 가능하게 계산하는 계약을 준비하되, 로컬 Ollama를 호출하거나 실제 모델 응답을 기록·저장하지 마.

## 먼저 확인

- `AGENTS.md`, `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 5·ADR-023/024·다음 시작점
- `docs/human-validation-checkpoints.md`의 E3·L1·L3·L5와 `docs/experiment-results.md#ollama-adapter-live-smoke-results-2026-09-19`
- `backend/adaptive/harness/ollama.py`, `recording_orchestrator.py`, `recording_capture.py`, `recorded.py`, `recording_bundle_contracts.py`, `recording_bundle_storage.py`
- 기존 recording·bundle·Ollama 테스트, 시작 시 `git status --short`, 최근 Conventional Commit 형식

## 목표와 경계

1. 재사용 가능한 작은 함수 또는 coordinator를 만든다. 명시적 `model_run_id`, 주입된 `QueryModel`, 검증된 `CaptureTarget` 목록, `RecordingProvenance`, repository root, private output 경로를 받아 `capture_recording()` 결과를 기존 `write_recording_bundle()`에 전달한다. batch·provenance·저장 target의 입력 오류는 provider 호출·파일 생성 전에 차단한다. 기존 storage 경로 검사에 작은 공개 preflight가 필요하면 같은 validator를 공유하고 검사 로직을 복제하지 않는다. 성공·typed 실패를 기존 순서와 JSON 타입 그대로 보존한다.
2. 로컬에서 수동으로 실행할 진입점이 필요하면 **합성 fixture 전용**으로 만든다. 명시적 `--synthetic` 없이는 provider를 호출하거나 파일을 쓰지 않는다. 이 단계에는 `--live`, 실제 질문 파일 import, gold 40건 자동 수집, `.env` 자동 로드 기능을 넣지 않는다. synthetic bundle은 기존 storage가 허용하는 `.schema-adaptive` private 경로에만 쓰고, 기본 overwrite는 금지한다.
3. Ollama provenance helper는 `ProviderKind.OLLAMA`와 기존 `RecordingProvenance`를 사용한다. provider version, 정확한 모델 tag, 선택적 모델 digest는 호출자가 명시한다. `verification_status`는 항상 `DECLARED_UNVERIFIED`다. `/api/tags` digest를 독립적으로 증명된 weight hash라고 주장하지 않는다. tag fallback, digest 추측, 빈 값을 임의 버전으로 채우는 동작은 금지한다.
4. `prompt_template_sha256`은 **질문·runtime ontology·mapping을 채우기 전의 versioned template**에서, `generation_config_sha256`은 **실제 전송 설정**에서 결정적으로 계산한다. 현재 adapter의 prompt version·template·`stream: false`·`temperature: 0`과 하나의 코드 출처를 공유하도록 필요한 최소 리팩터링만 한다. hash 대상, UTF-8 인코딩, canonical JSON 규칙, timeout과 모델 tag를 어느 필드에 넣는지 문서화하고 테스트로 고정한다. 프롬프트·설정 원문, 질문, 응답, 물리 mapping, DSN을 manifest·stdout·기본 로그·오류 문자열에 넣지 않는다.
5. 기존 `recording-bundle-v1` 저장·strict load 계약을 재사용한다. storage 경로 방어·원자 게시·overwrite 정책을 새 코드에서 우회하거나 중복 구현하지 않는다. `run-recorded`와 평가 보고서를 자동 실행하지 않는다. 합성 bundle도 논문 성능 자료로 표시하지 않는다.

## 최소 합성 테스트

- 주입 provider의 성공 JSON 값과 `TIMEOUT`·`EMPTY_RESPONSE`·`CANCELLED`·`PROVIDER_ERROR`가 입력 순서대로 기록되고 strict load 후 `RecordedModel`에서 재생됨
- 한 target당 provider 1회; 실패 후 다음 target 처리; preflight 실패 시 provider 0회·출력 파일 0개
- 합성 진입점의 opt-in 경계와 Ollama HTTP 호출 0회. 테스트에서 네트워크 함수를 실패하도록 막아도 통과
- private 경로 밖, tracked target, 기존 bundle 덮어쓰기, 잘못된 provenance는 기존 storage 오류로 거부되고 이전 manifest가 보존됨
- 같은 template·전송 설정은 같은 SHA-256, version·template·설정 변경은 다른 SHA-256. 요청별 질문이나 mapping이 달라도 template hash는 동일
- manifest의 `DECLARED_UNVERIFIED`, record count, content hash가 strict loader에서 확인됨
- 질문·모델 응답·HTTP body·프롬프트·설정 원문·로컬 경로가 manifest·stdout·기본 로그·통제 오류에 노출되지 않음
- 관련 테스트를 먼저 실행하고 마지막에 `.venv/Scripts/python.exe -m pytest tests -q`, `.venv/Scripts/python.exe -m compileall -q backend tests`, `git diff --check` 실행

## 제외 범위와 사람 확인

- 실제 Ollama 호출, 실제 응답 저장·공유, CNU 질문·DB 행·SQL, 40×3 평가, PostgreSQL 통합
- 모델 weight·Modelfile 변경, prompt 탐색, QueryIR 자동 보정, legacy `backend/llm.py` 변경
- `run-recorded` 변경, bundle schema migration, FastAPI/React 연결, 루트 `pytest`의 기존 `backend/test_correction.py` import 문제 해결
- 합성 recording 테스트에는 새 사람 승인이 필요 없다. **실제 응답을 기록·저장·공유하기 전 E3·L5**의 데이터·보안 범위를 확인한다. **모델 성능 확정 전 L1·L3**를 확인하고 gold 40건의 `NEEDS_HUMAN_REVIEW`를 유지한다. synthetic provenance나 2건 live smoke를 승인·정확도 증거로 사용하지 않는다.

## 완료 보고

설계·로드맵 상태가 바뀌면 `docs/schema-adaptive-text-to-sql-plan.md`의 본문·체크박스·변경 이력을 갱신한다. 변경 파일, 테스트, 실제 Ollama 호출 여부, private 저장 경계, 승인 전 남은 작업만 보고해줘. 과거 형식의 Conventional Commit으로 적절히 커밋하되 push는 하지 마.

---

<a id="prompts-phase5-10-live-capture-boundary"></a>

## 원본: prompts/phase5-10-live-capture-boundary.md

# 다음 개발 프롬프트: Phase 5-10 live capture 입력·실행 경계

실제 Ollama recording을 위한 **입력 검증과 명시적 실행 경계**를 구현해줘. 이번 단계에서는 합성 입력·합성 transport로만 검증하고, 실제 Ollama 호출이나 실제 응답 저장은 실행하지 마. E3·L5 확인 전에는 운영 데이터 수집을 시작하지 않는다.

## 먼저 확인

- `AGENTS.md`, `docs/development-history.md#next-session-handoff`, `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 5·ADR-023/024·다음 시작점
- `docs/human-validation-checkpoints.md`의 E3·L1·L3·L5와 `docs/experiment-results.md#ollama-adapter-live-smoke-results-2026-09-19`
- `backend/adaptive/harness/ollama.py`, `private_capture.py`, `synthetic_capture_cli.py`, `recording_orchestrator.py`, `recording_bundle_storage.py`, `recording_bundle_contracts.py`, `contracts.py`
- 관련 테스트, 시작 시 `git status --short`와 최근 Conventional Commit. `docs/development-history.md#desktop-test-handoff`는 초기 데스크톱 계획의 과거 기록이므로 현재 구현 상태는 계획 문서와 코드로 확인한다.

## 구현 범위

1. 수동 실행용 live capture 진입점을 작게 추가한다. **명시적 `--live`가 없으면 provider 호출과 bundle 쓰기가 0회**여야 한다. 입력은 호출자가 명시한 private target 파일, 정확한 model tag, base URL, timeout, Ollama version, 선택적 digest, model run ID, repository root, private output 경로만 받는다. `.env` 자동 읽기, 모델 추측·fallback·retry, gold 40건 자동 import는 금지한다.
2. private target 파일의 형식과 버전을 명시하고, 기존 `HarnessRequest`·`Ontology`·`PUBLISHED MappingProfile`·`CaptureTarget` 계약으로 엄격히 검증한다. 전체 batch의 형식, 중복 ID, database/mapping 일치, 비어 있는 입력, provenance, 저장 경로와 overwrite 정책을 **첫 provider 호출·파일 생성 전에** 검사한다. 질문·ontology·mapping 원문과 로컬 경로는 오류 문자열·stdout·기본 로그에 넣지 않는다. 입력 파일은 실행용 private 자료이며 Git에 추가하지 않는다.
3. 승인된 live 실행 경로에서는 기존 `OllamaQueryModel`, `declared_provenance()`, `capture_private_bundle()` 및 `recording-bundle-v1` storage를 그대로 연결한다. provider 호출은 target당 1회이며 typed 실패 뒤 다음 target을 처리한다. 성공·실패와 JSON 타입·순서는 기존 recording 계약을 따른다. 별도 SQL 생성·QueryIR 보정·DB 연결·평가 실행은 넣지 않는다.
4. CLI의 성공·실패 출력은 고정 상태/code와 안전한 건수·선언형 provenance 식별 정보만 허용한다. 질문, 모델 응답, HTTP body, 프롬프트, 물리 mapping, DSN, 절대 경로, private manifest 내용을 출력하지 않는다. 실제 recording은 기존 `.schema-adaptive` private 저장 경로에만 원자 게시하고 기본 덮어쓰기를 금지한다. 실행 flag 자체를 E3·L5의 사람 승인으로 간주하지 않는다.
5. 기존 synthetic-only CLI의 의미와 legacy `backend/llm.py`는 보존한다. 별도 진입점 또는 명확히 분리된 명령을 사용해 실수로 synthetic과 live가 섞이지 않게 한다.

## 합성 검증

- opt-in 누락, 잘못된 입력·중복 ID·비공개 경로 위반·기존 bundle·잘못된 provenance: provider 0회, 새 출력 0개, 기존 manifest 보존
- 버전 있는 private target 파일의 strict parsing; 유효한 synthetic target의 순서·필드 보존
- 주입한 합성 transport로 정확한 model tag·URL·timeout·`stream: false`·설정, target당 호출 1회, fallback/retry 0회 확인
- JSON 객체·다른 JSON 타입·비 JSON 텍스트·`TIMEOUT`·`EMPTY_RESPONSE`·`CANCELLED`·`PROVIDER_ERROR`의 capture→strict load 결과 확인
- 오류·stdout·기본 로그·manifest metadata에서 질문·응답·프롬프트·HTTP body·물리 mapping·로컬 경로 비노출 확인. recording 본문 자체는 private artifact이므로 이 비노출 검사의 대상과 구분한다.
- 관련 테스트를 먼저 실행하고 마지막에 `.venv/Scripts/python.exe -m pytest tests -q`, `.venv/Scripts/python.exe -m compileall -q backend tests`, `git diff --check` 실행

## 실행 금지·사람 확인

- **이번 단계에서 실제 `--live` 실행 금지.** 합성 transport 테스트만 한다. 실제 CNU 질문·SQL·DB 행·식별값을 Ollama에 보내거나 실제 모델 응답을 recording으로 저장·공유하지 않는다.
- 실제 응답 수집 전 E3·L5에서 데이터 사용 목적, 접근 권한, 보관·반출·삭제와 recording 노출 경로를 사람과 확인한다. 확인 결과가 문서화되지 않았다면 live capture는 후속 작업으로 남긴다.
- `run-recorded`, 40×3 benchmark, PostgreSQL adapter, FastAPI/React 연결, 모델 weight·Modelfile·legacy 프롬프트 변경은 제외한다. 성능 확정 전 L1·L3를 확인하고 gold 40건의 `NEEDS_HUMAN_REVIEW`를 유지한다.

## 완료 보고

실제 구현 범위만 `docs/schema-adaptive-text-to-sql-plan.md`의 본문·체크박스·변경 이력에 반영한다. 변경 파일, 테스트 결과, 실제 Ollama 호출 여부(이번 단계는 0회), E3·L5 전에 남은 작업, 다음 작은 작업을 보고해줘. 과거 형식의 Conventional Commit으로 커밋하되 push는 하지 마.

---

<a id="prompts-phase5-11-capture-preflight-approval"></a>

## 원본: prompts/phase5-11-capture-preflight-approval.md

# 다음 개발 프롬프트: Phase 5-11 live capture 사전 점검·승인 자료

Phase 5-10의 live capture 경계에 **호출·저장 없는 사전 점검**을 추가하고 E3·L5 사람 검토에 필요한 자료를 준비해줘. 이번 단계는 합성 target과 합성 transport로만 검증한다. 실제 Ollama 호출, 실제 질문·응답 recording, 40×3 평가는 실행하지 마.

## 먼저 확인

- `AGENTS.md`, `docs/development-history.md#next-session-handoff`, `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 5·ADR-023/024/025·다음 시작점
- `docs/human-validation-checkpoints.md`의 E3·L1·L3·L5, `docs/experiment-results.md#ollama-adapter-live-smoke-results-2026-09-19`
- `backend/adaptive/harness/live_capture.py`, `private_capture.py`, `recording_bundle_storage.py`, `recording_orchestrator.py`, `ollama.py`와 관련 테스트
- 시작 시 `git status --short`, 최근 Conventional Commit. 기존 `docs/development-history.md#desktop-test-handoff`의 구형 상태 설명보다 현재 계획·코드를 우선한다.

## 목표

1. CLI에 명시적 사전 점검 모드를 추가한다(예: `--check`). `--check`와 `--live`는 서로 배타적이어야 한다. `--check`는 versioned private target 파일, 모델 설정·선언 provenance, model run ID, output 경로·overwrite 정책을 live와 **같은 검증 함수**로 점검한다. 검증 로직을 복제하지 않는다. 기본 실행도 계속 provider 호출·파일 생성 0회다.
2. 사전 점검은 Ollama HTTP 요청, provider `generate_ir()`, bundle 파일 생성·수정, 실제 응답 수집을 절대 하지 않는다. 경로·입력 파일을 검증할 때 불필요한 내용을 출력하거나 새로운 공개 artifact를 만들지 않는다. 출력은 고정 상태/code, target 건수, 모델 tag·provenance의 안전한 식별값 정도로 제한한다. 질문·응답·프롬프트·물리 mapping·HTTP body·DSN·절대 경로를 stdout·기본 로그·통제 예외에 넣지 않는다.
3. live 실행도 동일한 preflight 결과를 사용하게 하되 Phase 5-10의 target당 단일 호출·typed failure·private atomic storage·기본 overwrite 금지 계약을 보존한다. 기본 urllib transport의 능동 취소나 독립적인 모델 출처 검증을 새로 주장하지 않는다. `--check` 성공을 실제 데이터 사용 승인으로 표시하지 않는다.
4. `docs/live-capture-human-review.md`에 E3·L5 검토용 **미기입 결정 양식**을 작성한다. 데이터 출처·사용 목적, 승인 주체, 전송할 질문 범위, 모델·저장 위치 접근 권한, recording·로그·백업·반출·보존 기간·삭제 절차, 허용/보류 결정과 근거를 사람이 채울 수 있게 한다. 현재 상태를 `PENDING_HUMAN_REVIEW`로 표시하고, 실제 질문·응답·경로·비밀값을 문서에 복사하지 않는다. `--live` flag나 합성 smoke를 승인 증거로 쓰지 않는다.

## 합성 테스트

- `--check` 성공·실패 모두 HTTP 0회, provider 0회, 새 출력 파일 0개; 기존 bundle/manifest 불변
- live와 check가 동일한 batch·provenance·path preflight를 사용하고, 잘못된 target·중복 ID·DRAFT mapping·모델 설정·tracked/nonprivate output을 호출 전에 거부
- `--check`와 `--live` 동시 지정 거부; 명시적 모드 없이 실행하면 호출·저장 없음
- 기존 live 합성 transport의 단일 호출·recording 순서·typed failure·strict load 회귀 통과
- 성공·오류 출력, 로그, 문서에 합성 비밀 문자열·절대 경로 비노출
- 관련 테스트를 먼저 실행하고 마지막에 `.venv/Scripts/python.exe -m pytest tests -q`, `.venv/Scripts/python.exe -m compileall -q backend tests`, `git diff --check` 실행

## 경계와 완료 보고

- 이번 단계에서 실제 `--live` 실행, 실제 CNU 질문·SQL·DB 행 전송, 실제 모델 응답 저장·공유를 하지 않는다. 승인 양식은 AI가 대신 승인하지 않는다. 실제 응답 수집 전 E3·L5의 사람이 범위를 확인하고 문서화해야 한다.
- `run-recorded`, 40×3 benchmark, PostgreSQL, FastAPI/React, 모델 weight·Modelfile·legacy `backend/llm.py` 변경은 제외한다. 성능 확정 전 L1·L3를 확인하고 gold 40건의 `NEEDS_HUMAN_REVIEW`를 유지한다.
- 실제 완료 범위만 계획 문서의 본문·체크박스·변경 이력과 `docs/development-history.md#next-session-handoff`에 반영한다. 변경 파일·테스트·실제 Ollama 호출 여부·승인 대기 항목·다음 작은 작업을 보고해줘. 과거 형식의 Conventional Commit으로 커밋하되 push는 하지 마.

---

<a id="prompts-phase5-12-synthetic-live-private-capture"></a>

## 원본: prompts/phase5-12-synthetic-live-private-capture.md

# 다음 개발 프롬프트: Phase 5-12 합성 입력의 첫 private live capture

Phase 5-11의 `--check`와 `--live` 경계를 **로컬 Ollama + 합성 질문 2건**으로 작게 검증해줘. 질문·ontology·mapping은 전부 합성으로 만들고, 모델 응답은 기존 private bundle에만 저장한다. E3·L5의 팀 결정은 실제 CNU 자료 사용·공유·배포·외부 시연 전에 남겨둔다. 합성 개발을 그 결정 때문에 중단하지 마.

## 먼저 확인

- `AGENTS.md`, `docs/development-history.md#next-session-handoff`, `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 5·ADR-023/024/025·다음 시작점
- `docs/human-validation-checkpoints.md`의 E3·L1·L3·L5와 `docs/live-capture-human-review.md`
- `backend/adaptive/harness/live_capture.py`, `ollama.py`, `private_capture.py`, `recording_bundle_storage.py`, `recorded.py`, `validator.py`와 관련 테스트
- 시작 시 `git status --short`, 최근 Conventional Commit, `ollama list`와 `/api/tags`의 정확한 model tag·서버 상태. tag가 없거나 서버가 응답하지 않으면 다른 모델로 대체하지 말고 환경 미준비로 보고한다.

## 실행 범위

1. `.schema-adaptive` 아래 Git 비추적 `capture-targets-v1` 파일에 **완전 합성 질문 2건**과 고정 합성 runtime ontology·`PUBLISHED` mapping을 준비한다. 물리 table/column 이름도 합성으로 한다. 실제 CNU 질문·dataset·SQL·DB 행·식별값, gold IR·기대 결과를 쓰지 않는다.
2. 정확한 model tag, base URL, timeout, Ollama version, 선택적 digest, model run ID, 새 private output 경로를 명시한다. `--check`를 먼저 실행하고 성공 상태·target 건수·선언 provenance fingerprint만 확인한다. 실패하거나 기존 bundle이 있으면 중단한다. fallback·retry·overwrite는 하지 않는다.
3. `--live`를 한 번 실행한다. target당 adapter 호출은 한 번으로 제한한다. 원 응답은 stdout·기본 로그·공개 문서에 복제하지 않고 기존 private bundle에만 보관한다. 실제 호출이 불가능하면 고정 실패 code와 미검증 상태만 보고한다.
4. strict loader로 manifest·hash·건수·`DECLARED_UNVERIFIED` provenance를 확인하고 `RecordedModel`에서 입력 순서, JSON 타입 또는 typed failure를 재생한다. QueryIR 구조·semantic field/operator 검증은 adapter 밖에서 한다. SQL·Markdown·설명문을 QueryIR로 번역·보정하지 않는다.
5. 비식별 case ID별 성공/실패 code, JSON 여부·타입, QueryIR 검증 code, 안전한 경과 시간 정도만 결과 문서에 남긴다. 질문·응답·프롬프트·HTTP body·물리 mapping·절대 경로·DSN은 출력·커밋하지 않는다. private target과 bundle은 Git 비추적 상태를 확인한다.

## 검증·완료 보고

- 관련 테스트를 먼저 실행한다. 코드가 바뀌면 마지막에 `.venv/Scripts/python.exe -m pytest tests -q`, `.venv/Scripts/python.exe -m compileall -q backend tests`, `git diff --check`를 실행한다. 문서만 바뀌어도 `git diff --check`와 Git 비추적 상태를 확인한다.
- 합성 2건의 결과를 모델 정확도, 학교 간 일반화, 40×3 benchmark 성공으로 해석하지 않는다. `run-recorded` 평가, PostgreSQL·실제 DB 연결, 모델 weight·Modelfile·legacy `backend/llm.py` 변경은 제외한다.
- E3·L5는 실제 CNU 자료 사용·공유·배포·외부 시연 전에 팀이 결정한다. 이번 합성 로컬 수집은 그 결정을 대신하지 않는다. 성능 확정 전 L1·L3를 확인하고 gold 40건의 `NEEDS_HUMAN_REVIEW`를 유지한다.
- 실제 완료 범위만 `docs/schema-adaptive-text-to-sql-plan.md`와 `docs/development-history.md#next-session-handoff`에 반영한다. 변경 파일, 테스트, 실제 Ollama 호출 여부, 안전한 형식 분류, 남은 한계를 보고해줘. 공개 가능한 코드·문서만 과거 형식의 Conventional Commit으로 커밋하고 push는 하지 마.

---

<a id="prompts-phase5-13-finetuned-synthetic-evaluation"></a>

## 원본: prompts/phase5-13-finetuned-synthetic-evaluation.md

# 다음 개발 프롬프트: Phase 5-13 기존 fine-tuned 모델의 합성 탐색 평가

Phase 5의 첫 미완료 항목인 **기존 fine-tuned 모델 평가**를 시작해줘. 로컬 `text2sql_qwen:latest`를 정확한 tag로 사용하고, 저장소의 **합성 fixture 40건 × 3스키마**에서만 수집→strict load→오프라인 `run-recorded`를 연결한다. Gold 40건은 `NEEDS_HUMAN_REVIEW`이므로 결과는 **탐색용 잠정 수치**이며 확정 성능이나 논문 결과로 발표하지 마. E3/L5·L1/L3의 팀 결정을 지금 요청하지 않는다.

## 먼저 확인

- `AGENTS.md`, `docs/development-history.md#next-session-handoff`, `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 5·ADR-023/024/025·다음 시작점
- `docs/human-validation-checkpoints.md`의 E3/L5·L1/L3, `docs/experiment-results.md#phase5-12-synthetic-live-capture-results-2026-09-19`
- `backend/adaptive/evaluation/{contracts.py,loader.py,runner.py,run_report.py,recorded_cli.py,fixture_mapping.py}`, `backend/adaptive/harness/{live_capture.py,recorded.py,recording_bundle_storage.py,ollama.py}` 및 관련 테스트
- 시작 시 `git status --short`, 최근 Conventional Commit, `ollama list`·`/api/tags`의 정확한 model tag와 서버 상태. 모델이 없으면 fallback 없이 환경 미준비로 보고한다.

## 구현·실행 범위

1. 기존 benchmark v1의 합성 질문 40건과 flat·normalized·legacy fixture 각각의 runtime ontology·`PUBLISHED` mapping으로 `capture-targets-v1` target 120건을 결정적으로 만든다. request ID는 `run-recorded`와 같은 `deterministic_request_id()`를 재사용한다. target에는 질문과 runtime context만 넣고 **gold IR·기대 결과·실제 DB 행·SQL을 모델 prompt나 capture 파일에 넣지 않는다.** target 파일은 Git 비추적 `.schema-adaptive` 아래에만 둔다.
2. 전체 target의 개수·순서·request ID·database/mapping 일치·fingerprint drift를 합성 FakeModel/transport 테스트로 먼저 검증한다. 별도 생성기가 필요하면 최소 함수/CLI를 추가하고 기존 loader·preflight를 재사용한다. 질문·mapping·gold를 stdout·기본 로그에 출력하지 않는다.
3. 정확한 tag·base URL·timeout·Ollama version·선택적 digest·model run ID·새 private output을 명시하고 `--check`를 먼저 수행한다. 통과할 때만 명시적 `--live`를 한 번 실행한다. target당 호출 1회, retry·fallback·overwrite 없음. 120건은 시간이 걸릴 수 있으므로 실행 전에 전체 소요와 timeout을 확인하되, 일부만 실행하고 40×3 완료로 표시하지 않는다. 서버나 모델이 불가하면 중단 사유와 미실행 상태를 남긴다.
4. private bundle을 strict load해 count·hash·provenance를 확인한 뒤 기존 `run-recorded`에서 오프라인 재생한다. 실행은 합성 SQLite fixture에만 연결한다. 성공·실패·누락을 예정된 120 attempt 분모에서 숨기지 않는다. 기존 runner/report의 계산 계약은 유지하고, 필요하지 않은 benchmark·schema migration은 하지 않는다.
5. 원 응답·질문·프롬프트·SQL·결과 행·물리 mapping·HTTP body·절대 경로를 공개 문서·stdout·기본 로그에 복제하지 않는다. private target, bundle, 상세 report는 Git 비추적 경로에 둔다. 공개 문서에는 환경·실행 여부·시도 건수·안전한 실패 유형과 **잠정/미승인** 표시, 미검증 사항만 남긴다. 모델 결과를 QueryIR로 자동 보정하지 않는다.

## 테스트와 경계

- target 생성 120건과 각 variant의 정확한 request ID·ontology·Published Mapping, gold 비포함, fingerprint drift 차단
- `--check` 실패 시 Ollama 호출·파일 생성 0회; 합성 transport에서 target당 호출 1회, typed failure 뒤 다음 target 처리
- strict load 후 `run-recorded`가 동일 context에서 재생하고 실패·누락을 분모에 남김; report·stdout 비노출
- 관련 테스트를 먼저 실행하고 마지막에 `.venv/Scripts/python.exe -m pytest tests -q`, `.venv/Scripts/python.exe -m compileall -q backend tests`, `git diff --check` 실행
- 실제 CNU 자료·DB·식별값, PostgreSQL 통합, base instruct 모델 비교, prompt 탐색, LoRA 학습, legacy `backend/llm.py` 변경은 제외한다.
- Gold 40건의 `NEEDS_HUMAN_REVIEW`를 유지한다. 잠정 수치를 최종 점수·논문 주장으로 바꾸기 전 L1·L3를 확인한다. 실제 CNU 자료 사용·공유 또는 실제 데이터 배포·외부 시연 전 E3·L5를 확인한다. 지금은 팀 결정을 요청하지 않는다.

## 완료 보고

`docs/schema-adaptive-text-to-sql-plan.md`의 본문·체크박스·변경 이력과 `docs/development-history.md#next-session-handoff`를 **실제 완료 범위만** 갱신한다. 현재 fine-tuned 모델 평가 항목은 검토 전 잠정 실험만 했다면 완료 체크하지 않는다. 변경 파일·테스트·실제 Ollama 호출 수·120 attempt 완료 여부·안전한 잠정 결과의 한계·다음 작은 작업을 보고해줘. 공개 가능한 코드·문서만 과거 형식의 Conventional Commit으로 커밋하고 push는 하지 마.

---

<a id="prompts-phase5-14-versioned-queryir-prompt-trial"></a>

## 원본: prompts/phase5-14-versioned-queryir-prompt-trial.md

# 다음 개발 프롬프트: Phase 5-14 QueryIR field 실패 분석·versioned prompt 1회 비교

Phase 5-13의 합성 40×3 잠정 결과에서 `UNKNOWN_FIELD` 21건을 **private 응답 비노출** 상태로 분류하고, 그 근거로 기존 fine-tuned 모델의 QueryIR 동적 schema prompt를 **한 가지 versioned 변형**과 비교해줘. 같은 40건은 이미 개발에 사용된 평가 세트이므로 새 비교도 탐색용 개발 결과다. 확정 정확도나 일반화 성능으로 발표하지 마.

## 먼저 확인

- `AGENTS.md`, `docs/development-history.md#next-session-handoff`, `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 5·평가 설계·다음 시작점·ADR-023/024/025
- `docs/experiment-results.md#phase5-13-finetuned-synthetic-exploratory-results-2026-09-19`, `docs/human-validation-checkpoints.md`의 E3/L5·L1/L3
- `backend/adaptive/harness/ollama.py`, `live_capture.py`, `recorded.py`, `recording_bundle_storage.py`, `validator.py`; `backend/adaptive/evaluation/capture_targets.py`, `runner.py`, `run_report.py`, `recorded_cli.py`와 관련 테스트
- 시작 시 `git status --short`, 최근 Conventional Commit, `ollama list`·`/api/tags`의 정확한 tag. 현재 fine-tuned tag 외 base instruct 모델이 없으면 대체 모델을 추측·다운로드하지 않는다.

## 범위

1. Phase 5-13의 private recording·상세 report를 **읽기 전용**으로 검사한다. `UNKNOWN_FIELD` 21건의 원인을 원 응답을 공개하지 않고 고정 범주로 집계한다. 예: ontology 밖 semantic field, entity 접두사 불일치, 물리 column 형태, 대소문자/표기 차이, 그 외. 분류 실패·중복·판별 불가도 별도 건수로 남긴다. 공개 문서에는 원 field 이름, 질문, 값, SQL, 모델 출력, private 경로를 쓰지 않는다. 기존 bundle·report를 수정하지 않는다.
2. 이 집계로 **검증 가능한 가설 하나**를 정하고 `query-ir-v1`을 그대로 보존한 채 `query-ir-v2`처럼 식별되는 prompt 변형을 추가한다. 기존 default 동작은 v1로 유지하고 새 버전은 명시적으로 선택할 때만 쓴다. semantic field를 허용 목록에서 정확히 복사하도록 지시하되 물리 table/column, 행, gold IR·기대 결과, 비밀 설정을 prompt에 넣지 않는다. SQL/Markdown을 QueryIR로 자동 번역·보정하지 않는다.
3. prompt version·template hash는 실제 선택한 template에서, generation config hash는 실제 전송 설정에서 계산한다. 같은 version/settings는 동일 hash, v1/v2는 다른 hash여야 한다. 호출당 1회, `stream: false`, `temperature: 0`, timeout·model tag 명시, fallback·retry 없음. 기존 `backend/llm.py` 및 모델 weight·Modelfile은 변경하지 않는다.
4. 합성 transport 테스트와 `--check`를 먼저 통과시킨 뒤, 모델과 서버가 가능하면 **같은 합성 40×3 target**, 같은 정확한 fine-tuned tag, 새 model run ID·private output으로 v2를 한 번 수집한다. strict load→오프라인 `run-recorded`를 수행하고 v1/v2를 같은 case×variant 단위로 대응시켜 구조 exact, 하네스 실행, semantic 일치, `UNKNOWN_FIELD`, provider 실패·누락을 같은 120 attempt 분모로 비교한다. v1 recording과 세트 자체는 수정하지 않는다.
5. 공개 결과는 prompt/version/hash의 안전한 식별 정보, 전체 건수, 실패 범주, paired 변화 요약과 **동일 개발 세트에 맞춘 탐색 결과**라는 한계만 남긴다. 질문·응답·프롬프트 원문·filter 값·물리 mapping·SQL·결과 행·HTTP body·절대 경로를 stdout·기본 로그·공개 문서에 쓰지 않는다. private input·bundle·상세 report는 Git 비추적 경로에만 둔다.

## 최소 검증

- v1 기본 동작·기존 provenance hash 회귀, 명시적 v2 선택·서로 다른 template hash, 전송 설정 불변
- 합성 transport에서 v1/v2 prompt의 semantic field만 전송되고 물리 mapping·gold·비밀 문자열은 전송되지 않음
- private 실패 분류의 합계·중복·판별 불가 처리와 공개 출력 비노출
- `--check` 실패 시 호출·파일 생성 0회; v2 target당 단일 호출과 strict load·동일 ID paired 비교
- 관련 테스트를 먼저 실행하고 마지막에 `.venv/Scripts/python.exe -m pytest tests -q`, `.venv/Scripts/python.exe -m compileall -q backend tests`, `git diff --check` 실행

## 제외 범위·완료 보고

- 실제 CNU 자료·DB, PostgreSQL, base instruct 모델 설치·비교, 반복 prompt 탐색, LoRA 학습, legacy 경로 변경은 제외한다. 로컬 모델이 없거나 서버가 응답하지 않으면 v2 live 비교를 성공으로 표시하지 말고 합성 테스트 결과와 환경 미준비 code만 보고한다.
- Gold 40건은 `NEEDS_HUMAN_REVIEW`를 유지한다. 이번 v1/v2 결과는 같은 미승인 개발 세트의 잠정 비교이며 Phase 5의 모델 평가 체크박스를 확정 완료로 바꾸지 않는다. E3/L5·L1/L3 팀 결정을 지금 요청하지 않는다.
- 실제 완료 범위만 `docs/schema-adaptive-text-to-sql-plan.md`의 본문·체크박스·변경 이력, 필요한 ADR, `docs/development-history.md#next-session-handoff`에 반영한다. 변경 파일·테스트·실제 Ollama 호출 여부·paired 잠정 결과·한계·다음 작은 작업을 보고해줘. 공개 가능한 코드·문서만 Conventional Commit으로 커밋하고 push는 하지 마.

---

<a id="prompts-phase5-15-prompt-failure-audit-evaluation-freeze"></a>

## 원본: prompts/phase5-15-prompt-failure-audit-evaluation-freeze.md

# 다음 개발 프롬프트: Phase 5-15 v2 실패 감사·비교 계약 고정

Phase 5-14의 `query-ir-v2` 악화 원인을 private 응답 비노출 상태로 점검하고, v1/v2 paired 비교가 서로 다른 fixture·모델·recording을 잘못 묶지 않도록 계약을 보강해줘. **새 prompt 버전이나 live 모델 호출은 만들지 않는다.** 같은 미승인 40건에 반복 적합하지 않도록 다음 독립 합성 평가의 분리 기준을 문서화한다.

## 먼저 확인

- `AGENTS.md`, `docs/development-history.md#next-session-handoff`, `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 5·평가 설계·다음 시작점·ADR-023/024/025
- `docs/experiment-results.md#phase5-13-finetuned-synthetic-exploratory-results-2026-09-19`, `docs/experiment-results.md#phase5-14-versioned-queryir-prompt-trial-results-2026-09-19`, `docs/human-validation-checkpoints.md`의 L1/L3·E3/L5
- `backend/adaptive/evaluation/field_failure_analysis.py`, `paired_prompt_analysis.py`, `run_contracts.py`, `capture_targets.py`, `runner.py`와 관련 테스트; `backend/adaptive/harness/ollama.py`, `recorded.py`, `recording_bundle_storage.py`
- 시작 시 `git status --short`, 최근 Conventional Commit. Phase 5-13/14 private bundle·report가 없으면 공개 결과만으로 세부 원인을 추측하지 말고 미검증으로 표시한다.

## 목표

1. 기존 v1/v2 private bundle·상세 report를 **읽기 전용**으로 strict load한다. `UNKNOWN_FIELD`가 v1 21건에서 v2 72건으로 늘어난 이유를 고정된 범주·집계로 재확인한다. 예컨대 완전한 entity 접두사, 누락/변형된 접두사, ontology 밖 field, 물리 column 형태, 그 외를 서로 구분한다. 원 field 이름·모델 응답·질문·값은 stdout·공개 문서·기본 로그에 쓰지 않는다. 증거가 없는 인과 해석은 하지 않는다.
2. `paired_prompt_summary()`와 실패 분류 함수가 report·recording·target의 **같은 benchmark, seed, ontology, QueryIR 계약, fixture variant/mapping context, model tag, 질문 fingerprint, request ID 집합**을 비교하는지 점검한다. 부족한 검사가 있으면 기존 fingerprint와 strict-loaded bundle의 provenance를 재사용해 작은 입력 검증을 추가한다. report의 `model_label`만으로 model tag를 증명하지 않는다. 단순히 두 결과가 120건이라는 이유만으로 비교를 허용하지 않는다. 원문을 담지 않는 고정 error code로 거부한다. 공개 API 변경은 최소화한다.
3. v1 기본 prompt와 v2의 bytes/hash/전송 설정, 기존 private recording·report를 수정하지 않는다. `query-ir-v3` 생성, prompt 탐색, 응답 보정, 모델 weight·Modelfile 변경, 실제 Ollama 호출은 하지 않는다.
4. 별도 문서에 다음 평가용 **독립 합성 질문 세트**의 설계 기준을 고정한다. 기존 40건의 질문 문장·paraphrase family를 재사용하지 않고, 작성·검토·동결 순서, seed/ontology/fixture fingerprint, 모델·prompt version 고정, 사람 gold 검토 상태, 개발 세트와 독립 세트의 결과를 구분하는 방법을 정의한다. 같은 합성 스키마를 쓴다면 이는 질문 holdout이지 unseen school이 아님을 명시한다. 이번 단계에서 새 세트를 모델에 실행하거나 최종 성능을 발표하지 않는다.

## 최소 합성 테스트

- 정상 v1/v2 paired 비교와 기존 공개 집계의 회귀
- 다른 benchmark/seed, fixture metadata, mapping·question fingerprint, model tag, 중복·누락 request ID, variant context가 어긋난 합성 report·recording은 비교 전 거부
- v2 실패 분류에서 범주 합계·중복·혼합·판별 불가 건수가 일관되고 원문이 출력되지 않음
- private 파일 수정 0회, 네트워크·Ollama 호출 0회, SQL·행·응답·질문·경로 비노출
- 관련 테스트를 먼저 실행하고 마지막에 `.venv/Scripts/python.exe -m pytest tests -q`, `.venv/Scripts/python.exe -m compileall -q backend tests`, `git diff --check` 실행

## 경계·완료 보고

- 이 결과는 v2 실패 원인과 비교 무결성 점검이다. 같은 40건의 잠정 점수를 확정 성능으로 바꾸지 않는다. 기존 fine-tuned·동적 prompt 평가 체크박스는 사람 gold 검토 전 완료 처리하지 않는다.
- 현재 로컬에는 fine-tuned tag만 확인됐다. base instruct tag를 추측·다운로드·대체하지 않는다. E3/L5·L1/L3 팀 결정을 지금 요청하지 않고 gold 40건의 `NEEDS_HUMAN_REVIEW`를 유지한다.
- 실제 완료 범위만 `docs/schema-adaptive-text-to-sql-plan.md` 본문·체크박스·변경 이력, 필요 시 ADR, `docs/development-history.md#next-session-handoff`에 반영한다. 변경 파일·테스트·실제 모델 호출 여부·안전한 진단 집계·독립 평가 준비 상태·다음 작은 작업을 보고해줘. 공개 가능한 코드·문서만 Conventional Commit으로 커밋하고 push는 하지 마.

---

<a id="prompts-phase5-16-independent-synthetic-evaluation-draft"></a>

## 원본: prompts/phase5-16-independent-synthetic-evaluation-draft.md

# Phase 5-16: 독립 합성 평가 세트 초안과 교차 검증

기존 40건 개발 세트와 분리된 합성 질문 30건의 평가 세트 초안을 작성하고 자동 검증해줘. 이번 단계는 질문·gold QueryIR·기대 결과의 작성과 합성 fixture 검증까지만 수행한다. Ollama 등 실제 모델 호출, 성능 평가, 사람 승인, 최종 동결은 하지 마.

## 먼저 확인

- 시작 시 `git status --short`와 최근 Conventional Commit 형식
- `AGENTS.md`
- `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 5·평가 설계·다음 시작점·변경 이력
- `docs/development-history.md#next-session-handoff`
- `docs/independent-evaluation-set-design.md`
- `docs/human-validation-checkpoints.md`의 E3·L1·L3·L5
- `backend/adaptive/evaluation/contracts.py`, `loader.py`, `validator.py`, `reference.py`, `coverage.py`
- 기존 `backend/adaptive/evaluation/data/fixture_v1_cases.yaml`과 gold·cross-schema 테스트
- `backend/adaptive/fixtures/seed.py`, ontology, 합성 fixture builder

기존 40건과 Phase 5-13/14의 모델 결과는 변경하지 마. 새 질문을 만들 때 과거 모델의 오답 유형에 맞춰 질문을 선택하지 마. 기존 세트는 새 초안을 만든 후 겹침을 검사하는 데 사용해줘.

## 목표 1: 새 평가 세트 초안 30건

- 기존 `fixture-evaluation-set-v1` 계약을 재사용하고 별도 YAML 파일로 30건을 작성해줘. 기존 40건 파일은 수정하지 마.
- 새 `benchmark_id`를 사용하고 case ID도 기존 40건과 겹치지 않게 해줘. 예를 들어 `P2-041`부터 연속된 ID를 사용할 수 있다.
- `FixtureSeed v1`과 현재 ontology를 재사용한다. 따라서 이 세트는 **독립 질문 세트(question holdout)**이며 독립 학교나 unseen schema가 아니다.
- 각 case에 한국어 질문, 명시적인 gold QueryIR, typed 기대 행·offering key·cardinality·answerability, intent/paraphrase family, 작성 근거를 기록해줘. 정답을 런타임에 seed에서 자동 생성해 gold로 대체하지 마. 파일 안의 gold는 사람이 읽고 검토할 수 있어야 한다.
- 모든 case는 `NEEDS_HUMAN_REVIEW`이고 `reviewer_id`가 없어야 한다. 작성자가 스스로 승인하거나 `HUMAN_APPROVED`로 표시하지 마.
- `CourseOffering` root, 현재 semantic field와 허용 operator만 사용한다. 물리 table/column, 실제 CNU 질문·데이터·SQL을 포함하지 마.
- 결과가 0건·1건·여러 건인 질문과 단일·복수 filter, 서로 다른 semantic field를 포함해줘. 의미 없는 수치 채우기용 질문은 만들지 마.

## 목표 2: 기존 40건과의 교차 격리 검사

현재 `validate_evaluation_set()`은 한 세트 안의 일관성을 검사한다. 새 세트와 기존 세트 사이의 겹침을 검사하는 작은 재사용 함수를 추가해줘.

- 서로 다른 `benchmark_id`를 요구한다.
- case ID 중복을 거부한다.
- NFC 정규화와 공백 정리 후 같은 질문 문장을 거부한다.
- `paraphrase_family` ID 재사용을 거부한다.
- 이번 초안에서는 기존 `_intent_identity()` 기준의 동일 QueryIR intent도 거부한다.
- 검사 실패는 원문 질문이나 gold 값을 포함하지 않는 고정 code로 알린다.
- 이 자동 검사는 의미가 비슷한 다른 문장을 모두 찾아내지는 못한다. 독립적인 사람 검토가 남았다는 한계를 문서에 적어줘.
- 범용 validator나 기존 loader의 계약을 불필요하게 바꾸지 마.

## 목표 3: 합성 검증

- 새 30건이 기존 `load_evaluation_set()`과 `validate_evaluation_set()`의 **정식 30~50건 조건**을 통과해야 한다. 검증을 위해 `require_full=False`로 우회하지 마.
- seed fingerprint·ontology fingerprint·QueryIR fingerprint, 중복 질문, 물리 식별자 노출, 자료형·operator, offering key·기대 행을 기존 validator로 검사해줘.
- 동일 seed로 생성한 flat·normalized·legacy fixture에서 모든 gold IR이 안전하게 컴파일·실행되고 명시적인 기대 semantic 결과와 일치하는지 확인해줘. 기존 cross-schema 테스트의 실행 경로를 재사용해줘.
- 교차 검사 테스트에는 질문·case ID·paraphrase family·intent 각각의 충돌과 정상 분리를 포함해줘.
- 합성 검증 결과는 **gold와 fixture의 자기 일관성**이다. FakeModel을 사용한다면 oracle smoke라고만 표시하고 모델 정확도 수치로 발표하지 마.

## 문서와 사람 확인 경계

- `docs/independent-evaluation-set-design.md`에 새 세트의 상태를 “30건 작성·자동 검증 완료, 사람 검토·동결 전”으로 기록해줘. 작성 과정, 격리 기준, 자동 검사로 증명하지 못하는 부분을 간결하게 남겨줘.
- `docs/schema-adaptive-text-to-sql-plan.md`의 관련 본문·체크박스·다음 시작점·변경 이력과 `docs/development-history.md#next-session-handoff`를 실제 완료 범위에 맞게 갱신해줘. 평가 설계의 중요한 결정이 바뀌면 ADR도 갱신해줘.
- 기존 gold 40건과 새 30건 모두 `NEEDS_HUMAN_REVIEW`를 유지해줘. E3/L5·L1/L3 팀 결정을 지금 요청하지 마.
- 새 세트를 “동결”, “사람 승인”, “독립 성능 검증 완료”, “학교 간 일반화 검증”으로 표현하지 마. 다음 단계가 필요하다면 사람 결정 이전에 진행 가능한 합성 작업을 먼저 명시해줘.

## 제외 범위

- 실제 CNU 자료·private recording·`.env` 읽기 또는 출력
- Ollama 호출, 응답 저장, `run-recorded`, 새 모델 점수, prompt 선택·최적화
- 기존 40건 또는 Phase 5-13/14 결과 변경
- 모델 weight·Modelfile, PostgreSQL, FastAPI/React, legacy SQL 생성기 변경

## 검증·완료 보고

관련 테스트를 먼저 실행하고 마지막에 다음을 실행해줘.

- `.venv/Scripts/python.exe -m pytest tests -q`
- `.venv/Scripts/python.exe -m compileall -q backend tests`
- `git diff --check`

Windows 임시 폴더의 PermissionError가 재현되면 접근 가능한 `TEMP/TMP`를 사용한 사실을 정확히 보고해줘. 실패한 명령을 통과했다고 쓰지 마.

완료 보고에는 새 세트의 case 수와 상태, 교차 격리 검사 결과, 세 fixture 검증 결과, 테스트, 남은 사람 검토·동결 작업을 적어줘. 공개 가능한 합성 코드·문서만 과거 형식의 Conventional Commit으로 커밋하고 push는 하지 마.

---

<a id="prompts-phase5-17-independent-draft-coverage-oracle-smoke"></a>

## 원본: prompts/phase5-17-independent-draft-coverage-oracle-smoke.md

# Phase 5-17: 독립 합성 세트 문항 정리·커버리지·oracle smoke

Phase 5-16에서 작성한 독립 합성 평가 세트 30건의 문항 표현을 점검하고, 기존 커버리지 도구와 `FakeModel` oracle smoke로 세트의 준비 상태를 확인해줘. **사람 검토·동결·실제 모델 평가는 이번 단계에 포함하지 않는다.** 기존 기능으로 충분하면 새 실행기나 보고서 형식을 만들지 마.

## 먼저 확인

- 시작 시 `git status --short`와 최근 Conventional Commit을 확인하고 기존 미커밋 변경을 보존한다.
- `AGENTS.md`, `docs/development-history.md#next-session-handoff`, `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 5·ADR-028·다음 시작점·변경 이력, `docs/independent-evaluation-set-design.md`를 읽는다.
- `docs/human-validation-checkpoints.md`의 E2·E3·L1·L3·L5를 확인한다. 사람 결정을 지금 요청하거나 승인으로 표시하지 않는다.
- `backend/adaptive/evaluation/data/fixture_v1_independent_cases.yaml`, 기존 40건 YAML, `isolation.py`, `coverage.py`, `__main__.py`, `run_cli.py`, `runner.py`, `run_contracts.py`, 관련 테스트를 확인한다.
- 현재 도구가 `--input`으로 독립 세트 커버리지를 만들고 `run-smoke --gold`로 독립 세트를 실행할 수 있는지 확인한다. `run-recorded`는 실행하지 않는다.

## 작업 1: 문항 표현 점검

- P2-057의 “13:00 이후”는 `greater_than 13:00`과 경계 포함 여부가 모호하다. 의도한 배타적 조건이 한국어 질문에 분명히 드러나도록 고친다. 기존 gold IR과 기대 결과가 그대로 적절하면 질문만 수정한다.
- P2-059의 “요일에 'UE'가 들어가는”은 내부 요일 코드에 의존해 일반적인 학생 질문으로 부자연스럽다. 자연스러운 한국어 질문과 정확히 대응하는 허용 `QueryIR` 조건으로 고친다. `Schedule.day`의 의미와 seed 값을 확인하고 질문·gold IR·기대 결과·intent family가 서로 맞도록 한다. 단순 문구 교체로 의미가 맞지 않으면 필요한 필드만 함께 수정한다. 기존 개발 세트와의 intent 격리도 유지한다.
- 다른 문항에서도 질문과 `QueryIR`의 경계 조건·단위·코드 표현이 어긋난 곳이 있으면 같은 기준으로 **명백한 오류만** 정리하고 변경 이유를 기록한다. 과거 모델 오답에 맞춰 질문을 고르거나 새 질문을 추가하지 않는다.
- 수정 후 독립 세트 30건의 ID·benchmark ID, 기존 40건, `NEEDS_HUMAN_REVIEW`·reviewer 공란 상태를 보존한다. 초안 변경으로 gold set fingerprint가 달라지면 새로운 값을 기존 도구에서 계산하되 동결 fingerprint로 부르지 않는다. 검토 전 초안의 문구 수정에 불필요한 benchmark version 변경은 하지 않는다.

## 작업 2: 독립 세트 커버리지와 oracle smoke

- 기존 `load_evaluation_set()`의 정식 30~50건 계약으로 새 YAML을 로드한다. `require_full=False` 우회는 하지 않는다. `check_set_isolation()`을 다시 실행하고 flat·normalized·legacy의 gold IR 실행 결과가 명시된 기대 semantic rows와 일치하는지도 재검증한다.
- 기존 `build_coverage_report()` 또는 `python -m backend.adaptive.evaluation --input ...`을 재사용해 **독립 세트 단독** 커버리지 집계를 만든다. 최소한 case 수, 사용 semantic field·operator, filter 수 분포, 결과 0/1/여러 건, normalized JOIN 단계, review 상태와 `coverage_gaps`를 확인한다. 개발 세트와 합산하지 않고, 빈 범주·편중은 숨기지 않는다. 커버리지의 case별 원문 표나 전체 JSON을 공개 문서에 복사할 필요는 없다.
- 기존 `run-smoke --gold <독립 세트 경로> --output <임시 비추적 경로>`로 `FakeModel` oracle smoke를 실행한다. 세 variant × 30건 = 90 attempts, `config.oracle_smoke=true`, 모든 attempt의 구조·실행·semantic 비교 상태와 실패 코드를 확인한다. gold IR을 응답으로 주는 자기 일관성 점검임을 명시한다. 실패 시 모델 점수로 해석하지 말고 case·variant와 기존 고정 code를 이용해 원인을 조사한다.
- CLI에 범용적으로 재현되는 결함이 확인된 경우에만 작은 수정과 의미 있는 회귀 테스트를 추가한다. 새 모델 adapter, report schema, 평가 지표, 저장 경로 정책을 만들지 않는다. 검증용 임시 report는 Git에 추가하지 않는다.

## 테스트와 완료 기준

- 문항 수정에 맞는 테스트와 격리·cross-schema 테스트를 먼저 실행한다. 기존 커버리지 및 runner/CLI 관련 테스트도 실행한다.
- 독립 세트 단독 커버리지와 90-attempt oracle smoke를 실제로 실행해 결과를 확인한다. 기대와 다르면 실패 원인을 보고하고 성공으로 표시하지 않는다.
- 마지막에 `.venv/Scripts/python.exe -m pytest tests -q`, `.venv/Scripts/python.exe -m compileall -q backend tests`, `git diff --check`를 실행한다. Windows 기본 임시폴더 `PermissionError`가 재현되면 접근 가능한 `TEMP/TMP`와 필요한 pytest 옵션을 명시해 재실행하고, 실패한 명령과 우회 실행 결과를 구분해 보고한다.
- `docs/independent-evaluation-set-design.md`에는 수정 근거, **독립 세트만의 안전한 커버리지 집계**, oracle smoke 결과와 한계를 기록한다. `docs/schema-adaptive-text-to-sql-plan.md`의 관련 본문·체크박스·다음 시작점·변경 이력, `docs/development-history.md#next-session-handoff`를 실제 완료 범위에 맞게 갱신한다. 중요한 설계 결정이 바뀔 때만 ADR을 추가·수정한다.
- 완료 보고에는 변경 파일, 문항 수정 근거, 격리·3개 fixture·커버리지·oracle smoke 결과, 전체 테스트, 남은 위험을 적는다. 공개 가능한 합성 코드·문서만 최근 형식의 Conventional Commit으로 커밋하고 **push는 하지 않는다**.

## 경계

- 기존 40건 gold, Phase 5-13/14 결과, private bundle·recording, 실제 CNU 질문·DB 행·SQL·비밀값을 수정·노출하지 않는다.
- Ollama 호출, prompt 탐색, `run-recorded`, 실제 모델 점수, PostgreSQL·FastAPI/React·legacy 변경은 하지 않는다.
- 새 30건은 계속 **사람 검토·동결 전 question holdout 초안**이다. 같은 seed·ontology를 썼으므로 unseen school 또는 학교 간 일반화 검증으로 표현하지 않는다. Oracle smoke의 90건 성공을 모델 정확도로 발표하지 않는다.
- L1·E2 사람 검토와 동결은 남긴다. E3/L5·L1/L3의 팀 결정도 해당 실제 사용·성능 확정 관문까지 보류한다.

---

<a id="prompts-phase5-18-direct-sql-baseline-arm"></a>

## 원본: prompts/phase5-18-direct-sql-baseline-arm.md

# Phase 5-18 구현 지침: 직접 SQL baseline arm과 QueryIR paired 비교 계약

## 목표

§15.1 비교 시스템 중 "기존 모델 + 동적 schema prompt → 직접 SQL"(이하 **SQL arm**)을 기존 QueryIR 평가(이하 **IR arm**)와 **같은 합성 질문·같은 fixture·같은 분모**로 비교할 수 있게 한다. 이번 단계는 계약·runner·비교 report를 합성 provider로만 검증한다. live Ollama 실행과 수치 기록은 Phase 5-19로 분리한다.

## 범위

1. 신규 `backend/adaptive/evaluation/direct_sql.py`
   - 동적 schema prompt(`direct-sql-v1`), 출력 추출, raw SQL 정책, 실행, 위치 무관 결과 비교, runner, report.
2. 신규 `backend/adaptive/evaluation/arm_comparison.py`
   - IR arm `EvaluationRunReport`와 SQL arm report의 paired 비교.
3. 신규 `backend/adaptive/evaluation/direct_sql_ollama.py`
   - Phase 7-08 `OllamaStartupConfig`·loopback transport를 재사용하는 SQL 생성 provider(1회 호출).
4. 신규 테스트 `tests/adaptive/test_direct_sql_arm.py`.

기존 IR runner·report 계약, runtime harness·compiler·policy·executor, prototype, frontend, legacy는 변경하지 않는다. `_prepare_fixture_contexts`와 `SQLiteReadOnlyExecutor`는 재사용만 한다.

## 격리 원칙

- raw SQL 실행은 **평가 전용**이다. runtime HTTP·prototype·clarification 경로에 연결하지 않고, 이 모듈을 runtime에서 import하지 않는다(ADR-048/049의 raw SQL fallback 금지 유지).
- 평가 대상은 개발 세트 `fixture_v1_cases.yaml` 40건이다. 독립 holdout 30건(P2-041~070)은 사람 검토·동결 전이므로 이 단계와 5-19에서 실행하지 않는다.
- 모델에는 gold IR·gold 결과를 주지 않는다.

## prompt 계약 (`direct-sql-v1`)

- 입력: fixture의 `DatabaseCatalog`와 질문만. sample 값은 넣지 않는다(IR arm과 같은 조건).
- 결정적 문자열: 테이블 이름순, 각 테이블 `name(col TYPE, ...)`(catalog 선언 순서), PK 목록, FK `table.col -> ref_table.ref_col`.
- 규칙은 도메인 힌트 없이 일반 규칙만: SQLite dialect, 단일 SELECT만, 나열된 table·column만, 이름을 만들지 말 것, SQL만 출력, 판단 불가 시 `UNKNOWN`.
- 템플릿 SHA-256을 report에 기록한다. legacy `backend/prompt.py`의 CNU 전용 규칙은 복사하지 않는다(고정 schema arm은 별도 후보).

## 출력 추출 (`direct-sql-extract-v1`)

결정적이고 보정하지 않는다.

1. 문자열이 아니면 `NON_STRING_OUTPUT`, 공백뿐이면 `EMPTY_OUTPUT`.
2. 정확히 하나의 fenced block(```` ```sql ```` 또는 ```` ``` ````)이 있으면 그 내용만 사용한다. fence가 둘 이상이면 `AMBIGUOUS_OUTPUT`.
3. 앞의 `SQL:` label 한 번과 앞뒤 공백, 끝의 `;`를 제거한다.
4. 결과가 대소문자 무관 `UNKNOWN`이면 `ABSTAIN`.
5. 그 외 문자열은 그대로 정책으로 넘긴다. 설명 문장 제거·키워드 추정 같은 보정은 하지 않는다.

## raw SQL 정책

`sqlglot` sqlite dialect로 검사한다. 실패는 고정 code만 기록한다.

- parse 실패 `SQL_PARSE_ERROR`, statement 수 ≠ 1 `SQL_MULTIPLE_STATEMENTS`.
- root는 `Select` 또는 `Union`/`Intersect`/`Except`만. 그 외 `SQL_NOT_SELECT`.
- 트리 어디에도 DML/DDL/`Command`/`Pragma`/transaction 노드가 없어야 한다(`SQL_FORBIDDEN_NODE`).
- 모든 `Table`은 catalog table이거나 **자기 조상 chain의 WITH가 정의한** CTE 이름이어야 하고 schema/catalog 한정자가 없어야 한다(`SQL_UNKNOWN_TABLE`). 중첩 CTE 이름으로 바깥 참조를 위장할 수 없다. `sqlite_` 접두 table·CTE 이름은 항상 거부한다.
- 정책 통과 뒤에도 기존 `SQLiteReadOnlyExecutor`(`mode=ro`, `query_only`, authorizer, 2초, 200행)를 상속한 평가 전용 executor로만 실행하며, authorizer가 `SQLITE_READ`를 해당 fixture catalog table로 제한한다(방어 2중화). parameter는 없다.

## 결과 비교 (`positional-row-multiset-v1`)

- column 이름은 비교하지 않는다(SQL arm은 physical alias를 쓰므로).
- 결과 column 수는 gold `select` 수와 같아야 한다. 다르면 `COLUMN_COUNT_MISMATCH`.
- 결과 column 이름이 중복되면 dict row가 값을 잃으므로 `RESULT_DUPLICATE_COLUMNS`로 판정 불가 처리한다.
- 각 값 정규화: `None`→`N:`(모든 값과 구별), bool→`V:1`/`V:0`, int·float·Decimal→`V:`+유한 `Decimal` normalize 문자열(3과 3.0 동일), str→`V:`+그대로. 그 외 타입은 비교 불가.
- 각 행을 정규화 값의 정렬 tuple로 만들고 행 multiset을 gold 행(같은 방식, gold select 순서 값)과 비교한다. 행 순서·column 순서는 무시하고 중복 행은 보존한다.
- 한계(문서화): 행 안 값 위치를 무시하므로 같은 값 집합의 column 뒤바뀜은 구분하지 않는다. 문자열 `'3'`과 숫자 3은 같은 값으로 본다.

## SQL arm report (`direct-sql-run-v1`)

- config: `model_run_id`, `model_label`, variants, `oracle_smoke`, prompt·extract·comparison version, 템플릿 SHA-256.
- IR report와 같은 benchmark ID·version·gold set fingerprint·seed fingerprint·fixture catalog fingerprint(variant별)를 기록한다.
- attempt: `case_id`, `variant`, `request_id`(IR arm과 같은 결정적 규칙), `outcome`(`CORRECT`|`INCORRECT_RESULT`|위 고정 code|`PROVIDER_ERROR`|`PROVIDER_TIMEOUT`|`EXECUTION_ERROR`|`EXECUTION_TIMEOUT`|`RESULT_TOO_LARGE`), `sql_fingerprint`(추출 SQL SHA-256, 있을 때), `result_fingerprint`(실행 성공 시).
- 질문·prompt·raw 출력·SQL 원문·행 값은 report에 넣지 않는다.
- 지표(모든 scheduled attempt 분모, ADR-021): `valid_sql`(정책 통과), `execution_success`, `execution_correct`, `abstain`, outcome별 count. 전체와 variant별.

## paired 비교 (`arm-comparison-v1`)

- 두 report의 benchmark ID·version·gold fingerprint·seed fingerprint·variant 집합·variant별 catalog fingerprint·`oracle_smoke`·(case, variant) attempt 집합이 정확히 같지 않으면 비교를 거부한다(ADR-027 확장). 또한 모든 case가 모든 variant를 가진 전체 schedule이어야 하고, 각 report의 전체·variant별 분모와 attempt 수가 일치해야 하며, 각 attempt의 request ID가 그 report의 run ID 기준 결정적 규칙과 같아야 한다.
- 한계: IR arm은 예측 select field 집합이 gold와 같아야 비교 가능(`NOT_COMPARABLE_PROJECTION`)하지만 SQL arm은 column 이름을 비교할 수 없어 column 수와 값만 본다. SQL arm이 이 점에서 약간 관대하다.
- IR arm 정답은 `semantic_correct is True`, SQL arm 정답은 `outcome == CORRECT`.
- 출력: 전체·variant별 `both_correct`, `ir_only`, `sql_only`, `neither`, 각 arm 정답 수와 분모, 두 arm의 `model_label`과 동일 여부, `oracle_smoke` 여부, `gold_human_review_pending`.
- 통계 검정 결과나 우열 결론 문장은 만들지 않는다. discordant count(b=ir_only, c=sql_only)만 기록한다.

## Ollama SQL provider

- `OllamaDirectSqlModel(config: OllamaStartupConfig, *, transport=None)`: Phase 7-08 `build_loopback_transport` 재사용, payload는 `{"model", "prompt", "stream": false, "options": {"temperature": 0}}`, 호출 1회, retry 없음.
- envelope가 dict이고 `error`가 없고 `response`가 문자열이면 그 문자열을 반환한다. 그 외는 `PROVIDER_ERROR`, timeout 계열은 `PROVIDER_TIMEOUT` 고정 예외. 원문 비보존.
- repr에 URL·tag를 넣지 않는다.

## 테스트 (합성 provider만)

- prompt 결정성, table·column·FK 포함, sample·질문 외 원문 비포함, 템플릿 hash 고정.
- 추출 규칙 각 분기.
- 정책: DML/DDL/PRAGMA/ATTACH/다중 statement/`sqlite_master`/schema 한정/unknown table 거부, CTE·JOIN·subquery·UNION 허용.
- **oracle 자기 일관성**: gold IR을 기존 compiler로 컴파일하고 parameter를 안전한 literal로 치환한 SQL을 fake provider가 반환할 때 40×3 = 120 attempt가 모두 `CORRECT`.
- 비교기: column 순서 뒤바뀜·행 순서 무관 정답, 행 누락·중복·추가 column·NULL·3 대 3.0.
- provider 실패·timeout·비문자열이 고정 outcome으로 기록되고 분모에 남음.
- paired 비교: 정상 집계, fingerprint·variant·attempt 집합 불일치 거부.
- Ollama SQL provider: 단일 호출, payload, envelope 오류 분류, 비노출.
- runtime 격리: `backend/adaptive/api`·`harness` 모듈이 `direct_sql`을 import하지 않음.
- 회귀: `tests/adaptive/test_evaluation_runner.py` 등 기존 평가 테스트.

## 금지

- live Ollama 호출, 실제 CNU·학교 DB, holdout 30건 실행.
- runtime 경로에서 raw SQL 실행, SQL 보정·재시도·repair.
- 기존 IR runner·report schema 변경, 성능 주장, 사람 승인 상태 변경.

---

<a id="prompts-phase5-19-live-arm-comparison"></a>

## 원본: prompts/phase5-19-live-arm-comparison.md

# Phase 5-19 구현 지침: 로컬 모델 IR·직접 SQL arm live paired 비교(잠정)

## 목표

Phase 5-18 계약으로 같은 로컬 모델 tag를 IR arm(`query-ir-v1`)과 SQL arm(`direct-sql-v1`)에 각각 연결해 개발 세트 40×3을 한 번씩 실행하고 paired count를 **잠정 관찰값**으로 기록한다. gold 40건이 `NEEDS_HUMAN_REVIEW`이므로 확정 성능·우열 주장이 아니다(L1/L3 전).

## 범위

1. 신규 `backend/adaptive/evaluation/arm_live_cli.py`와 테스트 `tests/adaptive/test_arm_live_cli.py`.
2. 메인이 로컬 Ollama로 `--check` 후 `--live-synthetic` 1회 실행, 결과 문서 `docs/experiment-results.md#phase5-19-live-arm-comparison-results-2026-09-28`.

기존 runner·Phase 5-18 모듈·runtime은 변경하지 않는다.

## CLI 계약

```text
python -m backend.adaptive.evaluation.arm_live_cli (--check | --live-synthetic)
    --ollama-model-tag TAG [--ollama-base-url URL] [--ollama-timeout SECONDS]
    [--repository-root PATH] --output-name NAME
```

- mode는 정확히 하나. Ollama 설정은 Phase 7-08 `parse_ollama_startup_config`로 검증하고 환경변수를 읽지 않는다. parser는 Phase 7-08과 같은 strict parser(반복·축약·unknown 거부, 입력 비반사).
- gold는 코드에 고정된 개발 세트 `fixture_v1_cases.yaml`만 쓴다. gold 경로 option은 없으며 holdout은 실행할 수 없다.
- 두 arm 모두 같은 config(tag·URL·timeout), temperature 0, attempt당 1회, retry 없음. IR arm은 `build_startup_ollama_model`, SQL arm은 `OllamaDirectSqlModel`.
- `model_label`은 tag의 `:`를 `-`로 바꾼 값, `model_run_id`는 `p519-ir-<label>`·`p519-sql-<label>`(패턴 불일치 시 설정 오류).
- 출력: `<repository-root>/.schema-adaptive/reports/<NAME>/`에 `ir-report.json`, `sql-report.json`, `comparison.json`. NAME은 `[a-z0-9][a-z0-9-]{0,63}`, 디렉터리가 이미 있으면 거부(no-overwrite). `.schema-adaptive`·`reports`·출력 디렉터리가 symlink이거나 저장소 밖으로 해석되면 거부한다. 세 파일은 같은 부모의 staging 디렉터리에 쓴 뒤 한 번의 rename으로 게시하고, 실패 시 staging을 지워 부분 결과를 남기지 않는다. report에는 5-18 계약대로 원문이 없다. mode flag도 반복을 거부한다.
- stdout에는 한 줄 JSON 요약만: 각 arm totals(IR: execution_success·semantic_correct, SQL: valid_sql·execution_success·execution_correct·abstain), paired totals와 variant별 count, 경고. 경로·URL은 출력하지 않는다.
- `--check`: 설정·gold·fixture bundle 준비만 하고 모델 호출 0회. fixture는 OS 임시 디렉터리에 만들고 정리하며 저장소·출력 경로 쓰기는 0회다(fixture 생성 가능 여부 확인이 check의 목적).
- 실패: 설정 오류 `{"status":"CONFIG_ERROR"}` exit 2, 그 외 `{"status":"RUN_ERROR"}` exit 1. 원문 예외 없음.

## 테스트 (synthetic transport만)

- `--check` 호출 0회·파일 0개.
- 두 arm에 synthetic transport 주입(IR은 gold IR, SQL은 oracle SQL 반환) 시 세 report가 생성되고 paired both_correct 120, stdout 한 줄.
- 기존 output 디렉터리·잘못된 NAME·mode 오류·잘못된 Ollama 설정은 고정 오류, 호출 0회.
- report·stdout에 질문·SQL·URL 비포함.

## live 실행

- 로컬 `/api/tags`의 정확한 tag, 기본 loopback URL, timeout 60초. 결과가 나빠도 재시도·prompt 조정 없이 그대로 기록한다.
- 결과 문서에는 환경(Ollama version, tag, digest), 두 arm의 지표, paired count, 해석 한계만 적는다.

## 사람 관문

완전 합성 fixture·질문만 사용한다(E3·L5 대상 없음). 수치는 L1(gold 검토)·L3(결과 검토) 전 잠정치이며 논문의 확정 성능이나 우열 결론으로 쓰지 않는다.

---

<a id="prompts-phase5-20-ollama-system-decontamination"></a>

## 원본: prompts/phase5-20-ollama-system-decontamination.md

# Phase 5-20 구현 지침: Ollama 내장 SYSTEM 오염 제거와 paired 비교 재실행

## 배경

`text2sql_qwen:latest`의 Modelfile에는 Alpaca TEMPLATE(CRLF)과 CNU `courses` schema를 담은 SYSTEM이 등록되어 있다(SYSTEM의 한국어는 UTF-8이 아닌 CP949 byte로 저장됨). `system`을 지정하지 않은 `/api/generate` 호출은 이 SYSTEM을 매번 주입한다. token 수 측정에서 기본 호출 161 대 SYSTEM 없는 raw 호출 47로 114 token 차이를 확인했고, raw 재현 template에 임의 SYSTEM을 넣은 호출이 Ollama의 `system` 지정 호출과 같은 49 token이어서 template byte 재현을 확인했다. Phase 5-08·5-12~5-14·5-19·7-09의 live 호출은 모두 이 SYSTEM이 주입된 조건이었다.

## 범위

1. 신규 `backend/adaptive/evaluation/ollama_request_format.py`
   - `model-template`: 기존 동작(Ollama가 등록 TEMPLATE·SYSTEM 적용).
   - `alpaca-raw-no-system-v1`: `raw: true`, 등록 TEMPLATE을 SYSTEM 빈 값으로 byte 단위 재현한 문자열로 prompt를 감싼다. payload 원본은 복사 후 변경하고, 이미 `raw`·`system`·`template`이 있으면 거부한다.
2. `arm_live_cli.py`에 `--ollama-request-format`(기본 `model-template`)을 추가하고 **두 arm에 같은 wrapper**를 적용한다.
   - 기본 형식은 Phase 5-19의 label·run ID를 그대로 유지한다.
   - 다른 형식은 label에 `.nosys`를 붙이고 run ID prefix를 `p520-nosys-`로 바꿔 tag와 형식 조합이 기본 형식의 ID와 충돌하지 않게 한다.
3. runtime(`OllamaQueryModel`·prototype)은 이번 단계에서 바꾸지 않는다. runtime 호출도 같은 오염 조건이라는 점을 후속 항목으로 기록한다.

## 테스트

- wrapper의 정확한 byte 문자열, 원본 비변이, 중복 key 거부.
- CLI 기본 형식은 `raw` 없음·5-19 label 유지, no-system 형식은 240 호출 모두 `raw: true`와 정확한 prefix/suffix, 잘못된 형식은 설정 오류.
- 기본 형식 tag가 no-system 형식의 run ID와 충돌하지 않음.

## live 실행과 문서

- `--ollama-request-format alpaca-raw-no-system-v1`로 개발 세트 40×3을 1회 실행하고, 최종 코드로 1회 재실행해 재현을 확인한다.
- Phase 5-19 결과 문서의 "CNU schema 암기" 해석을 오염 가능성으로 정정하고, 원래 수치는 조건과 함께 보존한다.
- 수치는 gold 미승인 잠정치다(L1/L3 전).

---

<a id="prompts-phase5-21-base-instruct-arm-comparison"></a>

## 원본: prompts/phase5-21-base-instruct-arm-comparison.md

# Phase 5-21 실행 지침: base instruct 모델 IR·직접 SQL arm paired 비교(잠정)

## 목표

Phase 5-20은 CNU 단일 schema로 fine-tune된 `text2sql_qwen:latest`만 비교해, QueryIR 경로의 이점이 방식 자체 때문인지 fine-tune 편향 때문인지 분리할 수 없었다(§15.1-3). 같은 계열 base instruct 모델로 같은 두 arm을 실행해 이를 분리하는 첫 관찰을 기록한다.

## 조건

- 모델: `qwen2.5-coder:7b-instruct`(Ollama 공식 레지스트리, Q4_K_M, 사용자 승인, 저장 경로 `C:/Users/micea/.ollama/models`).
- 등록 TEMPLATE은 ChatML이고 기본 SYSTEM은 일반 문구(`You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`)다. CNU schema 같은 도메인 내용은 없다.
- 요청 형식: 사용자 결정에 따라 `model-template`(공식 기본 system 포함, base 모델의 자연스러운 사용 조건). fine-tune 모델의 5-20 조건(`alpaca-raw-no-system-v1`, system 없음)과 system 조건이 다르다는 점을 결과에 명시한다.
- 나머지는 5-19/5-20과 동일: 개발 세트 40×3, `query-ir-v1`·`direct-sql-v1`, temperature 0, timeout 60초, 호출 1회·retry 없음, holdout 미실행.

## 실행

코드 변경 없이 기존 CLI를 사용한다.

```text
python -m backend.adaptive.evaluation.arm_live_cli --check|--live-synthetic \
    --ollama-model-tag qwen2.5-coder:7b-instruct --ollama-timeout 60 \
    --ollama-request-format model-template --output-name <name>
```

결과가 나빠도 prompt·추출·guard를 조정하지 않는다. 실패 구성은 원문 없는 구조 분류로만 진단한다.

## 기록

결과 문서 `docs/experiment-results.md#phase5-21-base-instruct-arm-comparison-results-2026-09-29`에 환경(버전·tag·digest·TEMPLATE/SYSTEM 요약), 두 arm 지표, paired count, 5-20 fine-tune 결과와의 대조, 해석 한계를 적는다. 수치는 L1/L3 전 잠정치다.

---

<a id="prompts-phase5-22-symmetric-fence-extraction"></a>

## 원본: prompts/phase5-22-symmetric-fence-extraction.md

# Phase 5-22 구현 지침: arm 간 대칭 code fence 추출과 두 모델 재실행

## 배경

Phase 5-21에서 base 모델은 IR 출력 40건을 모두 단일 Markdown code fence로 감쌌고 안쪽 28건은 유효 QueryIR이었다. SQL arm(`direct-sql-extract-v1`)은 단일 fence를 벗기지만 IR arm은 벗기지 않아 IR 0/120이 되었다. 사용자는 평가 계층에서 IR arm에도 같은 규칙을 적용하는 방식을 선택했다.

## 범위

1. `direct_sql.py`의 fence 탐지를 공개 함수 `find_code_fences(text) -> list[str]`로 노출하고 `extract_sql`이 그대로 사용한다(동작 불변).
2. 신규 평가 전용 `backend/adaptive/evaluation/ir_extract.py`
   - `IR_EXTRACT_NONE = "none"`, `IR_EXTRACT_SINGLE_FENCE = "single-fence-v1"`.
   - `SingleFenceIrModel(delegate)`: delegate 출력이 문자열이고 `find_code_fences`가 정확히 1개를 찾으면 안쪽을 strip해 `ollama.py`와 같은 strict JSON 규칙(duplicate key·non-finite 거부)으로 파싱해 반환한다. 파싱 실패, fence 0개·2개 이상, 비문자열은 원래 출력을 그대로 반환해 기존 분류(`MODEL_INVALID_OUTPUT` 등)를 유지한다. 그 외 보정은 없다.
   - runtime(`api`·`harness`)은 이 모듈을 import하지 않는다.
3. `arm_live_cli`에 `--ir-extract {none,single-fence-v1}`(기본 `none`, 반복 거부). `single-fence-v1`은 IR arm model만 감싼다(SQL arm은 원래 같은 규칙).
   - 기본값에서는 label·run ID가 5-19~5-21과 같다.
   - `single-fence-v1`은 label에 `.fence`를 붙이고 run ID prefix를 `p522-`로 시작해 기존 run과 섞이지 않게 한다(형식이 no-system이면 `p522-nosys-fence-`, `model-template`이면 `p522-fence-`).
   - stdout 요약에 `ir_extract`를 추가한다.
4. 재실행: fine-tune(`text2sql_qwen:latest`, `alpaca-raw-no-system-v1`)과 base(`qwen2.5-coder:7b-instruct`, `model-template`)를 `single-fence-v1`로 각 1회. 결과가 나빠도 규칙·prompt를 조정하지 않는다.

## 테스트

- `find_code_fences` 도입 전후 `extract_sql` 동작 동일(기존 테스트).
- wrapper: 단일 fence JSON→dict, 안쪽 비JSON·duplicate key·NaN→원본 문자열, fence 0·2개→원본, 비문자열·dict는 그대로, delegate 1회 호출.
- CLI: fence로 감싼 oracle IR을 내는 synthetic transport에서 `none`은 IR 0, `single-fence-v1`은 IR 120; label·run ID·요약 필드; 잘못된 값·반복은 설정 오류.
- runtime 격리 테스트에 `ir_extract` 추가.

## 기록

결과 문서 `docs/experiment-results.md#phase5-22-symmetric-extraction-results-2026-09-29`: 두 모델의 두 arm 지표, paired count, 5-20·5-21과의 대조, 한계. 수치는 L1/L3 전 잠정치다.

---

<a id="prompts-phase5-23-schema-variation-claim-scope"></a>

## 원본: prompts/phase5-23-schema-variation-claim-scope.md

# Phase 5-23 구현 지침: 주장 범위 축소와 schema 변형 fixture 확장

## 배경

Phase 5-22 대칭 추출 재실행에서 IR arm은 두 모델 모두 세 fixture에서 같은 정답 수를 냈고(fine-tune 31·31·31, base 26·26·26), 직접 SQL arm은 정규화 schema에서 떨어졌다(fine-tune 30→11, base 34·32→24). base 모델 전체 정답은 직접 SQL 90이 IR 78보다 높다. 따라서 "QueryIR이 직접 SQL보다 정확하다"는 일반 주장은 근거가 없고, 방어 가능한 주장은 **schema 구조 변화에 대한 정확도 일관성**이다. 사용자는 이 방향으로 평가 설계를 정리하고 schema 변형 fixture를 늘리는 것을 승인했다.

현재 fixture 3종은 이름 축(영문 snake vs 약어)과 구조 축(단일 table vs 5-table 정규화)이 섞여 있다: flat(영문·단일), normalized(영문·정규화), legacy(약어·단일). 약어·정규화 조합과 중간 구조가 없어 "정규화 schema에서 직접 SQL이 떨어진다"가 이름 때문인지 JOIN 때문인지 분리할 수 없다.

## 범위

### 1. 문서: 주장 범위와 평가 설계

- 계획 문서 §15.3에 구현된 변형과 지원하지 않는 변형(코드값 변경, FK 없는 관계, view, 컬럼 타입 변경)을 구분해 적는다. 지원하지 않는 변형은 ontology value mapping·FK 추론·L2 검토가 필요하므로 이번 범위에서 제외한다.
- §16.2에 연구 질문 "QueryIR 방식은 같은 질문의 정확도를 물리 schema 구조 변화에 대해 일정하게 유지하는가"를 추가하고, §16.4에 "QueryIR이 직접 SQL보다 정확하다"는 일반 우열 주장을 하지 않는다는 제한을 적는다.
- 일관성 지표를 정의한다: arm별 variant 정답 수의 최대−최소(spread), 모든 variant에서 정답인 case 수(all-variant correct).
- ADR-061(주장 범위)과 ADR-062(확장 fixture suite 계약)를 추가한다.

### 2. 새 fixture 변형 2종 (같은 seed, 같은 논리 행)

- `coded`: normalized와 같은 5-table 구조·FK에 약어 이름을 쓴다. `tb_dept(dept_cd, dept_nm)`, `tb_prof(prof_no, prof_nm)`, `tb_sbjt(sbjt_cd, sbjt_nm, pnt, isu_gb, dept_cd→tb_dept)`, `tb_lect(lect_no, sbjt_cd→tb_sbjt, prof_no→tb_prof, yyyy, hakgi, bunban, limit_cnt, aply_cnt)`, `tb_lect_tm(tm_no, lect_no→tb_lect UNIQUE, yoil, st_tm, ed_tm)`. legacy와 이름 계열이 같고 normalized와 구조가 같아 이름 축과 구조 축을 분리한다.
- `hybrid`: 2-table 부분 비정규화. `subjects(subject_code, subject_title, credit_hours, course_type)`, `class_sections(section_id, subject_code→subjects, dept_code, dept_name, professor, term_year, term, section_no, seat_limit, seats_taken, weekday, time_from, time_to)`. 학과가 과목이 아니라 분반 행에 있어 §15.3의 "컬럼의 다른 테이블 이동"·"테이블 병합"을 1개 JOIN 수준으로 재현한다. 다른 영문 동의어 이름을 써 기존 fixture와 이름이 겹치지 않는다.
- 두 변형의 Published Mapping은 `fixture_mapping.py`에 명시적으로 추가하고 catalog 대조를 그대로 거친다. 자동 proposal은 기존처럼 만들기만 하고 publish하지 않는다.

### 3. suite 계약 (기본 동작 불변)

- `FixtureVariant`에 `CODED`, `HYBRID`를 추가한다. suite는 `core-v1`(flat·normalized·legacy)과 `extended-v1`(core + coded·hybrid) 두 개로 고정한다.
- `build_fixture_bundle(..., variants=CORE_VARIANTS)`: 기본값은 core 3종이며 생성 파일·manifest가 이전과 같다. 허용 값은 두 suite의 variant 집합뿐이다.
- runner·직접 SQL runner는 bundle에 들어 있는 variant를 enum 순서로 준비하고, bundle variant 집합이 두 suite 중 하나와 정확히 같을 때만 진행한다. `config.variants`가 bundle에 없는 variant를 요구하면 실패한다.
- `run_contracts.FixtureVariant` literal에 두 값을 추가하고 `variants` 최대 길이를 5로 늘린다. 기존 report는 그대로 유효하므로 schema version은 바꾸지 않는다.
- case 단위 cross-schema consistency는 attempt 수가 설정 variant 수와 같고 3 이상일 때만 계산한다(3 variant 설정의 기존 결과와 같다).
- 런타임(`api`·`harness`·registry)은 bundle을 쓰지 않으므로 prototype registry에 새 변형이 나타나지 않는다. `capture_targets`는 core 3종 고정을 유지한다.

### 4. 평가 CLI

- `arm_live_cli --fixture-suite {core-v1,extended-v1}`(기본 `core-v1`, 반복·빈 값·unknown 거부).
- `extended-v1`은 label에 `.ext`를 붙이고 run ID prefix를 `p523-ext`로 시작한다(no-system이면 `-nosys`, fence면 `-fence`를 이어 붙인다). core 설정의 label·run ID는 5-19~5-22와 같다.
- stdout 요약에 `fixture_suite`와 arm별 일관성 지표(`spread`, `all_variants_correct`)를 추가한다. 일관성 지표는 report에서 계산하는 평가 전용 순수 함수로 둔다.

### 5. 실행

- 두 모델을 `extended-v1`·`single-fence-v1`로 각 1회 실행한다(fine-tune `alpaca-raw-no-system-v1`, base `model-template`; 모델당 400 호출). 결과가 나빠도 prompt·규칙·fixture를 조정하지 않는다.

## 테스트

- 새 builder: seed canonical 행과 같은 내용(runner preflight), FK·UNIQUE, catalog table 수.
- 기본 bundle은 core 3종만 만들고 extended는 5종; 허용되지 않는 variant 조합 거부.
- fixture mapping: 두 변형의 catalog 대조 통과, 잘못된 database ID 거부.
- oracle smoke: IR runner와 직접 SQL runner 모두 extended 200/200, IR cross-schema consistent 40/40.
- config: 5 variant 허용, 중복·6개 거부; bundle에 없는 variant 요구 시 실패.
- consistency: 5 attempt 계산, 설정보다 적은 attempt는 `INSUFFICIENT_VARIANTS`.
- CLI: extended label·run ID·요약 필드, oracle transport에서 IR·SQL 200, 잘못된 suite·반복은 설정 오류, core 기본값 불변.
- 일관성 지표 함수: spread·all-variant correct 계산, 불일치 report 거부.

## 기록

결과 문서 `docs/experiment-results.md#phase5-23-schema-variation-results-2026-09-29`: 두 모델 × 두 arm × 5 variant 정답, spread, all-variant correct, paired count, 5-22와 core 3종 비교, 한계(합성 fixture 5종 모두 개발자 작성, 같은 seed, 개발 세트 40건은 prompt 탐색에 사용, 1회 실행). 수치는 L1/L3 전 잠정치다.

---

<a id="prompts-phase6-01-bounded-synthetic-repair"></a>

## 원본: prompts/phase6-01-bounded-synthetic-repair.md

# Phase 6-01: 합성 입력의 제한된 QueryIR repair 경계

기존 `ExecutionHarness`를 재사용해 **검증 실패한 QueryIR에 한해 최대 한 번** 수정 후보를 받는 작은 coordinator를 구현해줘. `FakeModel` 또는 주입한 합성 repair provider로만 호출·검증한다. 실제 Ollama, 평가 세트 점수, 사용자 화면, 운영 DB는 이번 단계에 연결하지 마. Phase 5의 사람 검토·동결 대기는 그대로 유지한다.

## 먼저 확인

- 시작 시 `git status --short`, 최근 Conventional Commit, 미커밋 변경을 확인하고 보존한다.
- `AGENTS.md`, `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 6·다음 시작점·ADR-022/028·변경 이력, `docs/development-history.md#next-session-handoff`, `docs/human-validation-checkpoints.md`의 E2/E3/L1/L3/L5를 읽는다.
- `backend/adaptive/harness/engine.py`, `contracts.py`, `model.py`, `validator.py`, `errors.py`, SQLite compiler·AST policy·read-only executor, `tests/adaptive/test_execution_harness.py`, `test_schema_adaptive_harness.py`를 확인한다.
- 현재 `ExecutionHarness.run()`의 단계별 실패 code와 `HarnessResult`를 먼저 파악한다. legacy `backend/llm.py`는 변경하지 않는다.

## 목표와 계약

1. 기존 `ExecutionHarness` 바깥에 작은 coordinator와 명시적 repair provider 계약을 둔다. 최초 시도는 기존 `QueryModel`과 하네스로 실행한다. repair provider에는 원래 `HarnessRequest`, runtime ontology, `PUBLISHED` mapping, **고정된 실패 code**만 전달한다. gold IR·기대 행·SQL·DB 행·원래 모델 출력·예외 원문·전체 trace는 전달하지 않는다. provider가 반환한 후보도 기존 `QueryIR` validator, compiler, AST policy, read-only executor를 모두 다시 통과해야 한다. SQL 직접 수정·실행 경로는 만들지 않는다.
2. 최대 시도 수는 최초 1회 + repair 1회로 고정한다. 최초 성공이면 repair 호출 0회. `MODEL_INVALID_OUTPUT`은 `IR_GENERATED` 단계에서만, `UNKNOWN_ENTITY`, `UNKNOWN_FIELD`, `INVALID_OPERATOR`, `INVALID_VALUE_TYPE`은 `IR_VALIDATED` 단계에서만 repair 1회를 허용한다. 오류 **code와 stage를 함께** 검사한다. `INVALID_REQUEST`, mapping/context/JOIN/compiler/SQL policy/실행/결과 실패, provider 예외·취소는 terminal이다. repair 실패 뒤 추가 retry·fallback은 없다. 빈 결과 0행은 성공으로 취급하고 repair하지 않는다.
3. 동일 request ID·database ID·질문과 동일 ontology·mapping version을 두 시도에서 사용한다. repair 후보가 요청이나 context를 바꾸지 못하게 한다. 기존 `QueryModel.generate_ir()`와 `ExecutionHarness.run()`의 공개 계약은 유지한다. 생성한 임시 모델 wrapper가 있다면 호출당 후보 1개만 반환한다.
4. 결과는 최초/최종 시도의 상태, stage, 고정 failure code, repair 시도 여부·횟수를 구분해 확인할 수 있게 한다. 기존 `HarnessResult`의 SQL·행·원문 메시지를 새 요약에 복사하지 않는다. 성공은 최종 시도 성공일 때만 성공이다. repair로 성공한 결과를 최초 모델의 정확도나 평가 점수로 세지 않는다. 기본 stdout·로그·통제 오류에는 질문, 원 응답, SQL, 행, 비밀값, 로컬 경로를 넣지 않는다.
5. 실제 provider 호출용 CLI, prompt, 자동 prompt 수정, 임의 예외를 prompt에 이어 붙이는 동작은 구현하지 않는다. provider 예외가 나도 원문을 노출하거나 무한 재시도하지 않도록 고정 code 또는 기존의 안전한 실패 경계로 처리한다. 실제 cancellation은 취소로 전파하고 성공·repair 성공으로 위장하지 않는다.

## 최소 합성 테스트

- 첫 시도 성공, 0행 성공: repair 0회·하네스 1회.
- 구조 오류 및 각 허용 semantic 오류: repair 정확히 1회, 성공 후보는 기존 하네스 전체 안전 경로를 다시 통과해 최종 성공. 최초 실패와 최종 성공을 구분한다.
- repair 후보도 잘못되면 최종 실패, 총 2회에서 중지. provider 자체 실패·취소는 재시도하지 않고 안전하게 종료 또는 전파한다.
- 요청·mapping 오류, 모호한 JOIN, unsafe SQL, 실행 timeout·오류, 결과 schema 오류는 repair 0회. **오류 code가 우연히 같아도 stage가 다르면** 재시도하지 않는다.
- 원본 request/context가 변형되지 않음. repair provider에는 고정 code만 전달되고 gold·원 출력·SQL·행·trace가 전달되지 않음. 원문·경로·비밀값이 새 요약·stdout·기본 로그·통제 오류에 없음.
- 안전성 회귀: invalid IR은 SQL 실행 0회, repair 결과도 AST policy와 read-only executor를 우회하지 않음, DB 변경 0건. 실제 모델·네트워크 호출 0회.
- 결과 횟수·순서와 테스트 분모를 명시한다. `FakeModel` oracle을 쓴 결과는 모델 성능으로 표시하지 않는다.

## 범위와 문서

- Phase 6의 전체 `validate → execute → critique → repair` 로드맵을 이 작은 coordinator만으로 완료 처리하지 않는다. 이번 구현 범위와 남은 critique·사용자 확인·관리자 검토 큐를 분리해 기록한다.
- `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 6 본문·실제 완료 범위의 체크박스·다음 시작점·날짜별 변경 이력을 갱신한다. 재시도 정책이 중요한 결정이면 ADR을 추가한다. `docs/development-history.md#next-session-handoff`도 갱신한다.
- 새 30건과 기존 40건의 `NEEDS_HUMAN_REVIEW`를 유지한다. 사람 결정은 지금 요청하지 않는다. 실제 CNU 자료 추가 사용·공유 전 E3, 실제 데이터 배포·외부 시연 전 L5, 최종 모델 성능 주장 전 L1/L3를 적용한다.
- 독립 평가 세트 실행, Ollama/live capture, private recording, `run-recorded`, prompt 탐색, PostgreSQL, FastAPI/React, legacy 시스템 변경은 제외한다.

## 검증·완료 보고

- 새 coordinator의 합성 테스트와 관련 하네스 회귀 테스트를 먼저 실행한다. 마지막에 `.venv/Scripts/python.exe -m pytest tests -q`, `.venv/Scripts/python.exe -m compileall -q backend tests`, `git diff --check`를 실행한다. Windows 임시폴더 `PermissionError`가 나면 우회 환경과 실패·재실행 결과를 구분해 기록한다.
- 변경 파일, 재시도 허용 code/stage, 호출 횟수·안전성 테스트 결과, 기존 하네스 회귀, 미구현 범위와 위험을 보고한다. 성능·정확도 수치를 만들지 않는다. 새 코드가 요청·context를 오염시키거나 안전 검증을 우회하는 등 품질 저하 징후가 있으면 숨기지 말고 구체적 증거와 함께 보고한다.
- 과거 형식의 Conventional Commit으로 커밋하되 push는 하지 않는다.

---

<a id="prompts-phase6-02-synthetic-critique-disposition"></a>

## 원본: prompts/phase6-02-synthetic-critique-disposition.md

# Phase 6-02: 합성 결과의 결정적 critique 분류

Phase 6-01의 `BoundedRepairCoordinator`가 반환하는 `RepairSummary`를 **고정된 후속 처리 분류**로 바꾸는 작은 순수 함수를 구현해줘. 이번 critique는 모델이 답을 맞혔는지 판정하지 않는다. 성공은 안전한 실행이 끝났다는 뜻만 갖고, 실패는 code·stage에 따라 후속 검토 대상을 분류한다. 합성 입력과 테스트만 사용한다.

## 먼저 확인

- `git status --short`, 최근 Conventional Commit, 미커밋 변경을 확인하고 보존한다.
- `AGENTS.md`, `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 6·ADR-029·다음 시작점·변경 이력, `docs/development-history.md#next-session-handoff`, `docs/human-validation-checkpoints.md`의 E2/E3/L1/L3/L5를 읽는다.
- `backend/adaptive/harness/repair.py`, `contracts.py`, `engine.py`, `errors.py`, `__init__.py`, `tests/adaptive/test_bounded_repair.py`와 관련 하네스 테스트를 확인한다.
- 기존 실행 오류 taxonomy와 단계별 실제 발생 위치를 확인한다. 문서의 예정 기능과 현재 코드의 완료 상태를 혼동하지 않는다.

## 구현 범위

1. `RepairSummary`만 받는 결정적 critique 함수와 작은 결과 계약을 추가한다. 결과에는 version 식별자, 최종 disposition, 고정된 failure code·stage, 최초 성공인지 repair 후 성공인지 정도만 넣는다. `HarnessResult`, 질문, QueryIR, SQL, 결과 행, 원문 예외·메시지, trace, 경로, 비밀값을 입력·출력·로그에 포함하지 않는다. 파일·DB·네트워크 접근도 하지 않는다.
2. 성공한 최초 실행과 repair 후 성공은 둘 다 `EXECUTED_UNVERIFIED`로 분류하되 repair 사용 여부는 구분한다. 0행 성공도 같다. **정답, 사용자 만족, 모델 정확도, 안전한 답변 내용**을 추론하지 않는다.
3. 실패는 최종 code와 stage의 허용 조합을 명시적으로 분류한다. 최소 범주는 `REQUEST_INVALID`(입력 계약), `MODEL_REVIEW`(IR 구조·semantic 오류 및 repair provider 실패), `CONTEXT_REVIEW`(DB·Published Mapping·JOIN/관계 문제), `SAFETY_BLOCKED`(SQL policy 거부), `EXECUTION_FAILURE`(timeout·실행·결과 검증), `INTERNAL_REVIEW`(컴파일 등 내부 결함 또는 분류 불가)다. `REPAIR_PROVIDER_ERROR`의 stage 없음은 명시적 경우로 다룬다. 실제 `HarnessErrorCode` 전체와 새 provider code를 살펴 빠진 code가 조용히 성공·사용자 질문으로 바뀌지 않게 한다. code만 맞고 stage가 어긋나는 합성 입력, 알 수 없는 code, 서로 모순된 status·code·repair 횟수는 고정 오류 code로 거부하거나 `INTERNAL_REVIEW`로 닫는다. 오류 원문은 반영하지 않는다.
4. 자동 사용자 확인 요청이나 관리자 queue 등록은 하지 않는다. 분류 결과는 **후속 작업 제안**일 뿐 승인·publish·repair 재시도 지시가 아니다. 모호한 질문은 현재 구조화된 신호가 없으므로 `INVALID_REQUEST`, 0행, 모델 오류만 보고 모호하다고 추측하지 않는다. 별도 ambiguity 신호와 사용자 확인 계약은 다음 단계로 남긴다.
5. Phase 6-01의 재시도 목록·최대 1회·context 격리, `ExecutionHarness`, `QueryModel`, legacy 경로를 변경하지 않는다. 결과가 모델 성능 report로 흘러가는 새 경로를 만들지 않는다.

## 최소 합성 테스트

- 실제 `BoundedRepairCoordinator` 결과로 최초 성공, repair 후 성공, 최초 실패, repair 실패, `REPAIR_PROVIDER_ERROR`를 분류한다. 성공 두 종류 모두 `EXECUTED_UNVERIFIED`이며 repair 여부는 다르게 남는다.
- 성공 0행은 성공 분류. `INVALID_REQUEST`는 사용자 **질문 모호성**으로 취급하지 않는다.
- mapping/JOIN, SQL policy 차단, 실행 timeout·오류·결과 schema 오류, 내부 컴파일 오류의 대표 code·stage를 분류한다. safety 차단과 시스템 실패가 model failure로 섞이지 않는다.
- 존재하는 모든 `HarnessErrorCode`와 `REPAIR_PROVIDER_ERROR`의 처리 범위를 테스트로 고정한다. 새 code가 생기면 정책 검토 없이 조용히 기본 성공 처리되지 않는다.
- 위조된 stage, 알 수 없는 code, 모순된 `RepairSummary`는 fail-closed이며 고정 오류만 보인다. 요약·예외·stdout·기본 로그에 질문·SQL·결과 행·로컬 경로·임의 오류 원문이 없다.
- 분류는 순수 함수이고 model/provider/executor/DB/네트워크 호출 0회다. 기존 repair·하네스 관련 테스트도 유지한다.

## 문서·검증·보고

- `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 6 본문·실제 완료 범위의 체크박스·다음 시작점·변경 이력을 갱신하고, 분류 정책을 고정하는 ADR을 추가한다. `docs/development-history.md#next-session-handoff`도 짧게 갱신한다. 전체 `validate → execute → critique → repair` loop, 모호한 질문 확인, 관리자 검토 큐는 아직 완료로 표시하지 않는다.
- 관련 합성 테스트를 먼저 실행하고 마지막에 `.venv/Scripts/python.exe -m pytest tests -q`, `.venv/Scripts/python.exe -m compileall -q backend tests`, `git diff --check`를 실행한다. Windows 임시폴더 ACL 실패는 실패한 명령과 재실행 결과를 구분해 보고한다. 임시 경로를 만들면 사용 후 정리하고, 재귀 삭제 전 절대경로가 저장소 내부인지 확인한다.
- E3/L5·L1/L3 팀 결정을 지금 요청하지 않는다. 기존 40건과 독립 30건의 `NEEDS_HUMAN_REVIEW`를 유지한다. 실제 데이터·private bundle·Ollama·PostgreSQL·FastAPI/React·prompt 탐색·평가 점수는 범위 밖이다.
- 변경 파일, code별 분류 계약, 테스트, 미구현 범위, 위험을 보고한다. 코드 품질 저하나 요구사항 누락 징후도 숨기지 않는다. 최근 형식의 Conventional Commit으로 커밋하되 push는 하지 않는다.

---

<a id="prompts-phase6-03-synthetic-clarification-gate"></a>

## 원본: prompts/phase6-03-synthetic-clarification-gate.md

# Phase 6-03: 합성 모호성 신호·사용자 확인 경계

명시적인 모호성 신호가 있을 때 실행을 보류하고, 사용자가 제시된 선택지 하나를 확인했는지 검증하는 **작은 합성 계약**을 구현해줘. 이번 단계는 자동 모호성 감지나 사용자 화면이 아니다. 기존 `INVALID_REQUEST`, 0행, 모델 오류, `CritiqueDisposition`에서 모호성을 추측하지 않는다.

## 먼저 확인

- 시작 시 `git status --short`, 최근 Conventional Commit을 확인하고 기존 미커밋 변경을 보존한다.
- `AGENTS.md`, `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 6·ADR-029/030·다음 시작점, `docs/development-history.md#next-session-handoff`, `docs/human-validation-checkpoints.md`의 E2/E3/L1/L3/L5를 읽는다.
- `backend/adaptive/harness/contracts.py`, `repair.py`, `critique.py`, `engine.py`, `__init__.py`와 해당 합성 테스트를 읽고 실제 호출 순서를 확인한다.

## 범위와 계약

1. 별도 `clarification` 모듈에 versioned, 명시적 `AmbiguitySignal`과 `PendingClarification`, 사용자 `ClarificationChoice` 및 결정적 결과/고정 오류 code를 둔다. 신호는 caller가 제공하며 질문의 의미를 모델이나 문자열 규칙으로 추론하지 않는다. 후보는 최소 2개이고 서로 다른 안정적인 option ID와 **semantic** 설명 식별자를 갖는다. 물리 table/column, SQL, 결과 행, 자유 형식 사용자 답변은 후보에 넣지 않는다. 선택지의 표시 문구가 필요하면 이번 단계에는 출력하지 않고 ID만 다룬다.
2. `prepare`는 유효한 `HarnessRequest`, runtime `Ontology`, `PUBLISHED MappingProfile`, 명시적 신호를 받아 pending 상태를 만든다. 신호가 없으면 pending을 만들거나 호출을 막지 않는다. request ID, 질문 fingerprint, ontology/mapping의 결정적 fingerprint, 허용 option ID를 pending에 묶어 둔다. pending·결과·예외·stdout·기본 로그에 질문 원문, physical mapping, SQL, 행, 경로, secret을 넣지 않는다. hash는 canonical UTF-8 JSON에서 계산하고 원문을 저장하지 않는다.
3. `resolve`는 원래 request·ontology·mapping과 pending, 선택을 재검증한다. request ID·질문·ontology·mapping 중 하나라도 달라지거나, option ID가 없거나 중복·변조되면 고정 오류로 fail-closed한다. 정확히 한 개의 제시된 option ID를 선택한 경우에만 `CONFIRMED`와 선택 ID를 반환한다. `CANCELLED`는 별도 고정 상태로 반환하고 임의의 기본 선택지를 고르지 않는다. 임의 free-text 답변, 빈 답변, 여러 선택은 자동 확인으로 인정하지 않는다. 호출자가 재사용 가능한 stateless 순수 계약이라면 한 번만 쓰는 보증을 주장하지 말고 한계를 문서화한다.
4. 질문 원문을 바꾸거나 QueryIR을 보정·생성하지 않는다. 확인 결과만으로 `ExecutionHarness`나 `BoundedRepairCoordinator`를 자동 실행하지 않는다. 실제 UI/API, 관리자 큐, 저장소, DB, 네트워크, Ollama, legacy 경로를 변경하지 않는다. 기존 `CritiqueResult`와 오류 taxonomy도 변경하지 않는다. `INVALID_REQUEST` 등 기존 실패를 이 경계로 자동 승격시키지 않는다.
5. 입력 모델의 Pydantic 오류와 통제 오류의 문자열에 사용자 입력이 새지 않도록 공개 함수의 경계를 설계한다. 기존 타입의 raw `ValidationError`를 그대로 노출하는 패턴을 피하고 고정 code만 외부에 전달한다.

## 최소 합성 테스트

- 명시적 신호가 없을 때 pending·모델·DB 호출 0회. 신호가 있을 때 2개 이상 option과 published mapping에서 pending 생성, 정확한 한 선택 확인 및 취소 분리.
- 질문, request ID, ontology, mapping 변경마다 이전 pending 거부. 동일 입력은 같은 fingerprint, 순서가 의미 없는 set/dict 순서 변경은 동일 fingerprint.
- 후보 0/1개, 중복 ID, 존재하지 않는 semantic field/entity, 미발행 mapping, 물리 이름 주입, 미지 option, 다중·빈·자유 텍스트 답변을 거부.
- `INVALID_REQUEST`, 0행, model invalid output, `MODEL_REVIEW`만으로 신호가 자동 생성되지 않음.
- 원문 질문·물리 이름·임의 악성 문자열·로컬 경로가 pending·결과·예외·stdout·기본 로그에 없음. model/provider/executor/DB/network 호출 0회.
- 관련 테스트 먼저, 마지막에 `.venv/Scripts/python.exe -m pytest tests -q`, `.venv/Scripts/python.exe -m compileall -q backend tests`, `git diff --check`. Windows 임시폴더 ACL 우회가 필요하면 원 명령 실패와 우회 실행을 구분해 보고한다.

## 문서와 완료

- 실제 완료 범위만 plan Phase 6 본문·체크박스·다음 시작점·변경 이력에 반영하고, 중요한 계약 결정은 ADR로 남긴다. `docs/development-history.md#next-session-handoff`를 갱신한다. `모호한 질문 감지 및 사용자 확인` 전체 항목은 자동 감지와 실제 상호작용이 없으므로 완료 표시하지 않는다.
- E3/L5·L1/L3의 팀 결정을 지금 요청하지 않는다. 40건·30건 `NEEDS_HUMAN_REVIEW` 유지. 실제 CNU 자료와 private bundle을 사용하지 않는다.
- 변경 파일, 검증 결과, 남은 한계, 코드 품질 징후를 보고한다. Conventional Commit으로 커밋하되 push는 하지 않는다.

---

<a id="prompts-phase6-04-synthetic-review-queue"></a>

## 원본: prompts/phase6-04-synthetic-review-queue.md

# Phase 6-04: 합성 관리자 검토 큐 계약

Phase 6-01의 `RepairSummary`와 Phase 6-02의 `critique_repair_summary()`를 연결해 **운영 실패 후보만 메모리 내 검토 큐에 등록**하는 작은 합성 경계를 구현해줘. 이번 단계는 실제 운영 저장·관리자 UI·사람 승인 흐름이 아니다. Phase 6-03 `clarification-gate-v2`의 사용자 확인 결과를 관리자 승인으로 해석하지 않는다.

## 먼저 확인

- 시작 시 `git status --short`, 최근 Conventional Commit을 확인하고 기존 미커밋 변경을 보존한다.
- `AGENTS.md`, `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 6·ADR-029/030/031·다음 시작점, `docs/development-history.md#next-session-handoff`, `docs/human-validation-checkpoints.md`의 E2/E3/L1/L3/L5를 읽는다.
- `backend/adaptive/harness/repair.py`, `critique.py`, `clarification.py`, `contracts.py`, `__init__.py` 및 관련 합성 테스트를 확인한다. 최신 `019b07d`의 위조 요약 거부와 option ID 비노출 수정을 보존한다.

## 구현 계약

1. 새 작은 `review_queue` 모듈에 `InMemoryReviewQueue` 또는 동등한 주입 가능한 메모리 구현을 둔다. 입력은 원본 `RepairSummary`, 명시적 request ID·model run ID다. **내부에서 기존 `critique_repair_summary()`를 호출**해 분류하며, caller가 임의의 `CritiqueResult`를 넣거나 disposition을 조작하는 API를 만들지 않는다. 기존 coordinator·critique·clarification·harness 동작을 바꾸지 않는다.
2. `EXECUTED_UNVERIFIED`와 `REQUEST_INVALID`는 등록하지 않는다. `MODEL_REVIEW`, `CONTEXT_REVIEW`, `SAFETY_BLOCKED`, `EXECUTION_FAILURE`, `INTERNAL_REVIEW`만 운영 실패 후보로 등록한다. `REPAIR_PROVIDER_ERROR`도 기존 critique 분류에 따른다. 등록 결과는 `ENQUEUED`, 동일 항목의 `ALREADY_PRESENT`, 또는 `NOT_ELIGIBLE`을 명시한다. 이는 관리자에게 보여줄 **검토 후보**일 뿐 승인·publish·수정·재시도 지시가 아니다.
3. 큐 항목은 version, run/request ID의 domain-separated SHA-256, 최종 고정 code·stage·disposition, repair 시도/회복 여부와 안정적인 idempotency key만 가진다. 질문·QueryIR·SQL·행·원문 예외·물리 mapping·path·secret·raw run/request ID는 저장·출력·기본 로그에 넣지 않는다. hash 규칙은 canonical UTF-8 기준으로 문서화하고 테스트로 고정한다. ID 원문에 로컬 경로나 임의 secret 문자열이 들어와도 항목/결과/통제 오류에는 드러나지 않아야 한다.
4. 같은 run/request 조합에 같은 결과를 재등록하면 항목 1개만 유지한다. **같은 키에 다른 결과**가 오면 이전 항목을 덮어쓰지 말고 고정 오류 code로 거부한다. 입력 검증·critique 실패 시 큐는 변하지 않는다. 항목 반환과 목록 조회는 외부 변이가 내부 상태를 바꿀 수 없도록 방어적으로 처리한다. 순서는 첫 등록 순서다.
5. 이 모듈은 메모리 전용이며 disk/DB/network/Ollama/private bundle 접근 0회다. 영속성·동시성·여러 프로세스 간 dedupe·관리자 작업 상태를 보장하지 않는다. 실제 운영 큐 저장 체크박스와 전체 `validate → execute → critique → repair` loop 체크박스는 완료하지 않는다.

## 최소 합성 테스트

- 실제 `BoundedRepairCoordinator`의 성공, 사용자 입력 실패, model IR 오류·repair provider 실패, context·policy·execution·internal 실패의 대표 요약을 큐 정책에 통과시킨다. 성공·입력 실패는 항목 0개, 운영 실패는 고정 메타데이터만 1개.
- 동일 항목 재등록은 `ALREADY_PRESENT`, 같은 run/request의 다른 결과는 고정 conflict code이며 기존 항목 불변. 다른 run/request는 순서대로 별개 항목.
- 위조 `RepairSummary`의 2회 repair 주장·잘못된 code 타입은 기존 `CRITIQUE_INVALID_SUMMARY` 경계에서 차단되고 큐 0개. 입력 ID 오류도 등록 전 차단.
- 조회·반환 객체 변이로 큐 내부를 바꿀 수 없음. 질문·SQL·행·경로·secret형 raw ID가 항목·결과·예외·stdout·기본 로그에 없음.
- 모델/provider/executor/DB/network 호출은 큐 동작에서 0회. 관련 테스트를 먼저 돌리고 마지막에 `.venv/Scripts/python.exe -m pytest tests -q`, `.venv/Scripts/python.exe -m compileall -q backend tests`, `git diff --check` 실행. Windows 임시폴더 ACL 오류는 원 명령 실패와 우회 환경의 성공을 구분해 보고한다.

## 문서·완료

- plan의 Phase 6 본문·실제 완료 범위 체크박스·다음 시작점·변경 이력을 갱신하고 새 계약 결정은 ADR로 남긴다. `docs/development-history.md#next-session-handoff`도 갱신한다. 실제 운영 실패 저장·관리자 UI·사람 승인·full loop를 완료 표시하지 않는다.
- 실제 CNU 자료·private bundle·Ollama 호출·PostgreSQL·legacy·모델 점수는 범위 밖이다. 40건·30건 `NEEDS_HUMAN_REVIEW`와 E3/L5·L1/L3 대기 상태를 유지하고 지금 팀 결정을 요청하지 않는다.
- 변경 파일, 테스트, 남은 위험, 코드 품질 징후를 보고한다. Conventional Commit으로 커밋하되 push는 하지 않는다.

---

<a id="prompts-phase6-05-private-review-snapshot-synthetic"></a>

## 원본: prompts/phase6-05-private-review-snapshot-synthetic.md

# Phase 6-05: 합성 검토 큐의 private snapshot 저장·strict load

Phase 6-04 `InMemoryReviewQueue`의 **고정 메타데이터 항목만** 명시적 private 파일에 저장하고 strict load하는 작은 합성 경계를 구현해줘. 실제 운영 큐, 관리자 UI, 실제 CNU 질문·응답·DB 행, live 모델 호출은 이번 단계에 없다. 영속 snapshot은 관리자 승인이나 교정 결과가 아니다.

## 먼저 확인

- 시작 시 `git status --short`, 최근 Conventional Commit을 확인하고 미커밋 변경을 보존한다.
- `AGENTS.md`, `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 6·ADR-030/031/032·다음 시작점, `docs/development-history.md#next-session-handoff`, `docs/human-validation-checkpoints.md`의 E2/E3/L1/L3/L5를 읽는다.
- `backend/adaptive/harness/review_queue.py`, `recording_bundle_storage.py`의 repository/private path 방어·원자 게시·strict load, `recording_bundle_contracts.py`, 해당 테스트를 읽는다. 기존 recording bundle 저장·legacy 시스템은 변경하지 않는다.

## 저장 계약

1. 별도 `review_queue_storage` 모듈을 둔다. 명시적인 `repository_root`와 `.schema-adaptive` 아래 private output **파일 경로**, `InMemoryReviewQueue`를 받아 snapshot을 쓴다. source는 `queue.list_entries()`의 고정 필드만 허용한다. 질문·SQL·행·원문 예외·물리 mapping·raw run/request ID·경로·secret을 파일 내용·stdout·기본 로그·통제 오류에 넣지 않는다. 새로운 수집·enqueue·모델·DB 실행 경로는 만들지 않는다.
2. versioned 단일 JSON snapshot에 queue version, 항목 수, 순서 있는 항목, **hash 필드를 제외한 payload의 canonical JSON bytes SHA-256**을 넣는다. UTF-8, `sort_keys=True`, compact separators, `ensure_ascii=False`, `allow_nan=False` 규칙을 문서화·테스트한다. 내용 hash는 변조 감지일 뿐 작성자 인증이 아님을 명시한다. strict loader는 duplicate key, 비유한 값, 추가/누락 필드, count/hash 불일치, 중복 run/request 조합, 중복 idempotency key, 잘못된 code·stage·disposition·version을 고정 오류로 거부한다. loader는 파일 안의 경로를 신뢰하지 않는다.
3. 저장 전 private 경로·Git ignored/tracked·symlink/junction/reparse·hardlink 정책을 **기존 recording bundle의 동일 validator를 재사용**해 검사한다. 새 코드에서 별도의 약한 경로 검사 복제본을 만들지 않는다. 필요한 최소 공개 wrapper만 추가할 수 있지만 기존 bundle 계약·동작은 보존한다. 파일과 부모 경로의 redirect/escape를 막는다. 경로·OS 예외 원문은 고정 오류 code로 변환한다.
4. no-overwrite가 기본값이다. 같은 path에 기존 snapshot이 있으면 이전 바이트를 보존하고 고정 오류로 거부한다. 임시 파일 fsync 후 원자 게시하고 실패 시 임시 파일을 정리한다. `overwrite`·in-place update·실제 운영 동시성/잠금은 이번 단계에서 구현하지 않는다. 빈 큐 snapshot도 명시적 저장이면 허용하되 내용은 고정 메타데이터만 유지한다.
5. strict load 결과는 검증된 snapshot 계약/항목의 방어적 복사본만 반환하고 내부 queue를 자동 수정하지 않는다. 실제 운영 영속 큐 저장·관리자 UI·승인 workflow 체크박스는 미완료로 둔다.

## 합성 테스트

- 합성 queue의 성공 미등록 + 실패 2~3건을 snapshot 저장→strict load하면 순서·고정 code/stage/disposition·hash가 일치. 빈 queue도 검증.
- 같은 입력은 같은 canonical bytes/hash. 항목 순서 또는 고정 metadata 변경은 hash 변경. 잘못된 count/hash·중복 JSON key·비유한 값·중복 ID/key·미지 code/stage/disposition/version·추가 필드 strict 거부.
- private 경로 밖·Git tracked/unignored·symlink/junction/reparse/hardlink·기존 파일 덮어쓰기·게시 실패 시 이전 파일 보존과 임시 파일 정리. 기존 bundle path/storage 테스트 회귀 유지.
- 원시 질문·SQL·행·비밀형 ID·로컬 경로가 저장 바이트·결과·예외·stdout·기본 로그에 없음. provider/model/DB/network 호출 0회. 실패는 fixed code만.
- 관련 테스트 먼저, 마지막에 `.venv/Scripts/python.exe -m pytest tests -q`, `.venv/Scripts/python.exe -m compileall -q backend tests`, `git diff --check`. 기본 Windows temp ACL 실패와 접근 가능한 임시 경로 재실행을 구분해 보고한다.

## 문서·보고

- 실제 완료 범위만 plan Phase 6 본문·체크박스·다음 시작점·변경 이력에 반영하고 중요한 storage 결정은 ADR로 남긴다. `docs/development-history.md#next-session-handoff` 갱신. `운영 실패를 관리자 검토 큐에 저장` 전체 항목은 실제 운영 연결 전까지 완료 표시하지 않는다.
- 40건·30건의 `NEEDS_HUMAN_REVIEW`와 E3/L5·L1/L3 대기를 유지하고 지금 팀 결정을 요청하지 않는다. 실제 CNU 자료, private recording bundle, Ollama, PostgreSQL, FastAPI/React, 모델 점수는 사용하지 않는다.
- 변경 파일, 테스트, 실제 쓰기 경로가 합성 private인 점, 남은 위험·미구현 사항과 코드 품질 징후를 보고한다. Conventional Commit으로 커밋하되 push는 하지 않는다.

---

<a id="prompts-phase6-06-synthetic-agent-workflow"></a>

## 원본: prompts/phase6-06-synthetic-agent-workflow.md

# Phase 6-06: 합성 agent workflow 통합

기존 `BoundedRepairCoordinator`, `critique_repair_summary()`, `InMemoryReviewQueue`를 하나의 합성 실행 흐름으로 연결해줘. 이번 단계에서는 최종 `HarnessResult`를 잃지 않으면서 **validate → execute → bounded repair → critique → failure queue registration**을 한 번의 명시적 coordinator 호출로 수행한다. 실제 Ollama·PostgreSQL·CNU 자료·UI·운영 영속 큐는 사용하지 않는다.

## 먼저 확인

- 시작 시 `git status --short`, 최근 Conventional Commit을 확인하고 기존 변경을 보존한다.
- `AGENTS.md`, `docs/schema-adaptive-text-to-sql-plan.md`의 Phase 6·ADR-029~033·다음 시작점, `docs/development-history.md#next-session-handoff`, `docs/human-validation-checkpoints.md`의 E2/E3/L1/L3/L5를 읽는다.
- `backend/adaptive/harness/repair.py`, `critique.py`, `review_queue.py`, `review_queue_storage.py`, `engine.py`, `contracts.py`, 해당 테스트를 읽는다. legacy `backend/llm.py`와 기존 FastAPI/React 경로는 변경하지 않는다.

## 구현 계약

1. `BoundedRepairCoordinator`에 상세 결과 계약을 추가한다. 최초 실행 또는 repair 실행에서 얻은 **최종 `HarnessResult`와 기존 `RepairSummary`를 함께** 반환하는 immutable model이어야 한다. 기존 `run()`의 반환형·동작은 보존하고 새 상세 실행 API에 위임하게 한다. 모델·repair provider·executor 호출 횟수를 늘리거나 같은 요청을 재실행해서 결과를 복원하지 않는다.
2. terminal initial failure와 repair 후 성공/실패는 해당 최종 harness result를 보존한다. repair provider 자체가 예외를 내 `REPAIR_PROVIDER_ERROR`가 된 경우에는 그 code를 가짜 `HarnessErrorCode`나 가짜 harness result로 만들지 말고, 상세 계약에서 harness result 부재를 명시적으로 표현한다. 취소 `BaseException` 전파, 최대 repair 1회, 전체 안전 경로 재통과, context deep-copy 격리 계약은 유지한다.
3. 별도 작은 `AgentWorkflowCoordinator`를 추가한다. 명시적 `model_run_id`, request, ontology, Published Mapping, model, repair provider, 주입된 `InMemoryReviewQueue`를 받아 상세 repair 실행 → `critique_repair_summary()` → 운영 실패 후보 queue 등록을 순서대로 수행한다. 성공과 `REQUEST_INVALID`은 기존 큐 계약대로 저장하지 않는다. caller가 `CritiqueResult`나 disposition을 주입할 수 없어야 한다.
4. workflow 반환 계약에는 version, 최종 `RepairSummary`, `CritiqueResult`, queue registration 상태/항목, 가능한 경우 최종 `HarnessResult`를 포함한다. 성공 행과 0행 성공도 보존하되 정확도·정답·사용자 만족을 추론하지 않는다. 실패의 raw exception, 질문, SQL, 행, 물리 mapping, local path, secret을 새 요약·queue·오류·stdout·기본 로그에 복사하지 않는다. 기존 `HarnessResult` 자체의 SQL/rows 계약은 바꾸지 않는다.
5. 실행 전에 가능한 workflow 입력과 queue identity를 검증해 잘못된 `model_run_id`, valid request의 queue-incompatible request ID, 잘못된 queue/type 때문에 모델·repair·DB 실행 후 실패하지 않게 한다. queue의 ID 규칙은 하나의 코드 출처를 공개 helper로 재사용하고 복제하지 않는다. invalid `HarnessRequest`는 기존 harness/critique의 `REQUEST_INVALID` 결과를 유지하며 queue에 넣지 않는다. 입력 객체와 반환 객체의 사후 mutation이 coordinator/queue 내부 상태를 바꾸지 않게 한다.
6. queue conflict나 예상 밖 내부 계약 불일치는 고정 code로 fail closed하고 기존 queue entry를 보존한다. 실행 완료 뒤 queue 등록이 실패할 수 있는 남은 경계와 side effect를 문서화한다. 자동 snapshot write, overwrite, retry, clarification 생성/해결, QueryIR 보정, mapping/glossary/verified-query 변경은 하지 않는다.
7. 기존 `ExecutionHarness`, `QueryModel`, public `BoundedRepairCoordinator.run()`, critique taxonomy, review queue 분류 및 snapshot schema의 기존 계약을 깨지 않는다.

## 합성 테스트

- 최초 성공, 0행 성공: 모델 1회, repair 0회, `EXECUTED_UNVERIFIED`, queue 미등록, 최종 columns/rows 유지.
- retryable 초기 실패 후 repair 성공: 모델 1회 + repair provider 1회 + repair candidate 실행 1회, recovered flag와 최종 결과 유지, queue 미등록.
- retryable 실패 후 repair 후보도 실패: 최종 code/stage 기준 critique 후 queue 1건 등록. 같은 workflow 재등록은 `ALREADY_PRESENT`, 다른 결과의 같은 run/request는 conflict이며 기존 entry 보존.
- terminal validation/context/policy/execution 실패: repair 없음, 기존 disposition에 맞는 운영 실패만 queue 등록. `INVALID_REQUEST`는 미등록.
- repair provider 예외: `REPAIR_PROVIDER_ERROR`, harness result 없음, `MODEL_REVIEW` queue 등록. 실제 cancellation은 전파하고 queue 변화 없음.
- invalid model run ID·queue-incompatible valid request ID·잘못된 queue/type는 모델/provider/executor 0회, queue 0개. 입력 mutation provider를 써도 caller request/ontology/mapping과 검증 context 불변.
- workflow 반환 모델과 queue entry의 방어적 복사, 실행당 중복 호출 없음, 실패 뒤 별도 다음 요청 처리 가능.
- 새 workflow 계층이 파일·네트워크·Ollama를 호출하지 않으며 snapshot 자동 저장 0회. 질문·원문 응답·SQL·행·secret/path가 새 summary/queue/error/stdout/log에 노출되지 않음. 단, 성공/실패 `HarnessResult`는 기존 계약대로 명시적 호출자 반환값임을 테스트와 문서에서 구분한다.
- 기존 repair·critique·queue·snapshot 테스트 회귀. 관련 테스트를 먼저 실행하고 마지막에 `.venv/Scripts/python.exe -m pytest tests -q`, `.venv/Scripts/python.exe -m compileall -q backend tests`, `git diff --check` 실행. Windows 기본 temp ACL 오류와 접근 가능한 임시 경로 재실행을 구분해 보고한다.

## 문서·완료 기준

- 실제 완료 범위만 plan Phase 6 본문·체크박스·다음 시작점·변경 이력에 반영한다. 이 통합으로 계약상 전체 loop가 완성되었다면 `validate → execute → critique → repair loop 구현`만 완료 처리한다. `모호한 질문 감지 및 사용자 확인`, 실제 운영 영속 큐, 교정 반영은 미완료로 둔다. 중요한 결과 보존·side-effect 결정은 ADR로 남긴다.
- `docs/development-history.md#next-session-handoff`를 갱신한다. 40건·30건 `NEEDS_HUMAN_REVIEW`, E3/L5·L1/L3 대기를 유지하고 지금 팀 결정을 요청하지 않는다.
- 실제 Ollama·PostgreSQL·CNU 데이터·private recording bundle·live 모델·성능 측정·FastAPI/React를 건드리지 않는다.
- 변경 파일, 호출 횟수, 테스트, 최종 결과 보존 방식, queue side effect와 남은 위험을 보고한다. Conventional Commit으로 커밋하되 push는 하지 않는다.

---

<a id="prompts-phase6-07-synthetic-durable-review-queue"></a>

## 원본: prompts/phase6-07-synthetic-durable-review-queue.md

# Phase 6-07: 합성 영속 검토 큐 lifecycle

기존 `InMemoryReviewQueue`, `review-queue-snapshot-v1`, `AgentWorkflowCoordinator`를 이어서 **strict load/복원 → 합성 workflow 실행 → 변경된 큐를 새 immutable snapshot에 게시**하는 한 기능 흐름을 구현한다. 기존 snapshot을 덮어쓰거나 실제 운영 DB·UI·관리자 승인 흐름을 만들지 않는다. 실제 Ollama·PostgreSQL·CNU 자료·private recording bundle을 사용하지 않는다.

## 먼저 확인

- 시작 시 `git status --short`와 최근 commit을 확인하고 기존 변경을 보존한다.
- `AGENTS.md`, `docs/development-history.md#next-session-handoff`, 계획 문서의 Phase 6·다음 시작점·ADR-029~034, `docs/human-validation-checkpoints.md`의 E3/L1/L3/L5를 따른다.
- `backend/adaptive/harness/review_queue.py`, `review_queue_storage.py`, `agent_workflow.py`, recording bundle의 private path validator와 관련 테스트만 읽는다. legacy `backend/llm.py`, FastAPI/React 경로는 변경하지 않는다.

## 구현 계약

1. strict-loaded `ReviewQueueSnapshot`에서 새 `InMemoryReviewQueue`를 복원하는 좁은 공개 경계를 추가한다. snapshot entry 순서, idempotency key, `(model_run_id_sha256, request_id_sha256)` identity를 그대로 보존한다. raw run/request ID, `RepairSummary`, 질문, SQL, 행을 역추정하지 않는다.
2. 복원은 caller가 만든 Pydantic instance를 그대로 신뢰하지 않는다. plain JSON으로 dump 후 재검증하고 기존 snapshot의 code/stage/disposition/idempotency/count/hash 계약을 가능한 단일 validator에서 재사용한다. 위조 entry, 중복 identity/key, 잘못된 version은 고정 오류로 fail closed한다. 복원 실패 시 부분 queue를 반환하거나 변경하지 않는다.
3. 작은 durable lifecycle coordinator를 추가한다. 입력은 현재 snapshot 경로(없으면 명시적 empty 시작), 다음 snapshot 경로, repository root와 기존 `AgentWorkflowCoordinator.run()` 입력이다. 실행 전 현재/다음 경로, 서로 다른 경로, private·Git ignored/tracked·symlink/junction/reparse/hardlink 정책, 다음 경로 no-overwrite를 검사한다.
4. 현재 snapshot이 있으면 strict load 후 queue를 복원하고, 없으면 새 empty queue를 만든다. 그 queue를 기존 `AgentWorkflowCoordinator`에 주입한다. caller가 queue, `CritiqueResult`, disposition, 복원 entry를 직접 주입해 lifecycle 불변식을 우회하는 API는 두지 않는다.
5. workflow 결과가 `ENQUEUED`일 때만 변경된 queue를 다음 **새** snapshot에 원자 게시한다. `ALREADY_PRESENT`와 `NOT_ELIGIBLE`이면 새 snapshot을 쓰지 않는다. overwrite, in-place update, current-pointer 교체, 자동 retry를 구현하지 않는다.
6. 반환 계약은 version, 기존 `AgentWorkflowResult`, persistence 상태(`WRITTEN` 또는 `UNCHANGED`), 가능한 경우 새 snapshot의 content hash 같은 비식별 metadata만 포함한다. raw local path, 질문, SQL, 행, 물리 mapping, secret, raw ID를 새 결과·오류·stdout·기본 로그에 넣지 않는다. 기존 `HarnessResult` 반환 계약은 바꾸지 않는다.
7. 경로/복원/type 오류는 모델·repair provider·executor 호출 전에 고정 code로 차단한다. 실행 뒤 snapshot 게시 실패는 고정 storage code로 fail closed한다. 이 경우 읽기 전용 모델/실행 side effect는 이미 발생했고 호출자에게 반환되지 않은 in-memory queue 변경은 process-local로 남을 수 있으나 기존 source snapshot과 기존 target bytes는 보존됨을 문서화한다.
8. 같은 source snapshot과 같은 run/request/result 재실행은 기존 entry를 찾아 `ALREADY_PRESENT`·`UNCHANGED`가 되어야 한다. 같은 identity의 다른 결과는 기존 queue conflict로 실패하고 source snapshot·target을 바꾸지 않는다.
9. 기존 `review-queue-snapshot-v1`, `review-queue-v1`, `agent-workflow-v1`, `ExecutionHarness`, `BoundedRepairCoordinator.run()`, critique taxonomy와 호출 예산을 깨지 않는다. 새 snapshot schema가 꼭 필요하지 않으면 추가하지 않는다.

## 합성 테스트

- empty 시작 + eligible failure: workflow 호출 예산 유지, `ENQUEUED`, next snapshot 1개 게시, strict load 후 entry 정확히 1개.
- 기존 snapshot + 새 eligible failure: 기존 순서/entry 보존, 새 entry 뒤에 추가, source bytes 불변, next snapshot에 2개.
- 기존 entry 동일 재등록: `ALREADY_PRESENT`, `UNCHANGED`, next 파일 생성 없음.
- success·0행 success·`REQUEST_INVALID`: `NOT_ELIGIBLE`, `UNCHANGED`, next 파일 생성 없음.
- 같은 run/request의 다른 결과: conflict, source 불변, next 파일 없음.
- 위조 snapshot/model instance, duplicate identity/key, hash/count/version/entry semantic mismatch: 모델/provider/executor 0회, 부분 복원 없음.
- source=target, tracked/non-ignored/outside path, symlink/junction/reparse/hardlink, 이미 존재하는 target: 실행 전 차단. 기존 private path validator를 재사용하고 약한 복제 validator를 만들지 않는다.
- 게시 직전 target 생성, write/link/fsync 계열 실패: 기존 source/target 보존, private temp path·secret 비노출, 고정 오류. 실제 filesystem 보장 범위를 과장하지 않는다.
- 반환 객체와 복원 queue의 방어적 복사. 파일·네트워크·Ollama·PostgreSQL 호출 없음.
- 관련 테스트만 실행한다. 전체 pytest/compileall/diff-check는 Codex 독립 검토와 수정 뒤 별도로 한 번씩 실행하므로 여기서 실행하지 않는다.

## 보고 형식

긴 diff·전체 로그 금지. 변경 파일, 핵심 계약, 관련 테스트 수와 결과, 실패 원인, 남은 위험만 요약한다. 계획·ADR·체크리스트·handoff·commit은 Codex가 독립 검토 후 처리하므로 수정하지 않는다.

---

<a id="prompts-phase6-08-synthetic-ambiguity-detection"></a>

## 원본: prompts/phase6-08-synthetic-ambiguity-detection.md

# Phase 6-08: 합성 모호성 감지와 clarification 연결

기존 `clarification-gate-v2` 앞에 versioned 합성 ambiguity rule set 기반의 결정적 감지 경계를 추가한다. **질문 감지 → 기존 `prepare()` pending 생성 → 같은 규칙으로 재감지 → 기존 `resolve()` 확인/취소**를 한 기능 흐름으로 닫는다. 자유형 모델 추론, QueryIR 생성·수정, harness 실행, 실제 UI·사용자 세션은 구현하지 않는다.

## 먼저 확인

- 시작 시 `git status --short`와 최근 commit을 확인하고 기존 변경을 보존한다.
- `AGENTS.md`, `docs/development-history.md#next-session-handoff`, 계획 문서의 Phase 6·다음 시작점·ADR-031/034/035, `docs/human-validation-checkpoints.md`의 E3/L1/L3/L5를 따른다.
- `backend/adaptive/harness/clarification.py`, runtime ontology/mapping 계약, 해당 합성 테스트만 읽는다. legacy FastAPI/React·`backend/llm.py`는 변경하지 않는다.

## 구현 계약

1. 별도 작은 모듈에 `ambiguity-detector-v1` 계약을 둔다. caller가 제공하는 frozen rule set은 고정 version, 고유 rule ID, 1개 이상 literal trigger phrase, 서로 다른 semantic option 2~16개를 가진다. option은 opaque option key와 ontology entity/field `semantic_ref`만 허용한다. display 문장, 물리 table/column, SQL, 결과 행, 자유 답변 필드는 두지 않는다.
2. 모든 공개 입력은 Pydantic instance를 그대로 신뢰하지 않고 warning 비노출 plain JSON dump 후 재검증한다. rule ID/option key/trigger는 길이·문자·중복을 제한한다. Unicode NFKC + casefold + whitespace 정규화를 한 코드 출처로 사용한다. empty/control-only trigger, 정규화 후 중복 rule/trigger/option, 미지 semantic ref, unpublished mapping, request/mapping database 불일치는 고정 오류로 차단한다.
3. rule match는 정규화한 질문에 정규화 literal trigger 중 하나가 포함될 때만 성립한다. trigger는 2자 이상으로 제한해 한 글자 오탐을 줄인다. 이 규칙 기반 match는 자연어 의미 이해나 완전한 모호성 탐지가 아님을 문서화한다.
4. match 0개면 signal/pending 없이 통과한다. 정확히 1개면 해당 rule의 option으로 기존 `AmbiguitySignal`을 결정적으로 만든다. option ID는 raw rule ID/option key를 노출하지 않는 domain-separated SHA-256 기반 64자리 lowercase hex로 만들고 동일 입력·순서 변화에서 안정적이어야 한다. match 2개 이상은 서로 다른 ambiguity 축을 한 선택지 집합으로 합치지 않고 고정 conflict로 fail closed한다.
5. `prepare_detected_clarification()`은 감지 결과가 없으면 `None`, 하나면 기존 `clarification.prepare()`를 호출해 `PendingClarification`을 반환한다. caller가 별도 `AmbiguitySignal`을 주입할 수 없어야 한다.
6. `resolve_detected_clarification()`은 동일 request/ontology/mapping/rule set으로 다시 감지해 trusted signal을 재구성한 뒤 기존 `clarification.resolve()`를 호출한다. no-match, multi-match, rule/option/trigger/context drift는 확인으로 승격하지 않는다. 기존 pending option 전체 일치·질문/context fingerprint·정확히 한 선택·별도 취소 계약을 유지한다.
7. detector 결과·pending·result·고정 오류·stdout/log에 raw 질문, raw request/database/rule/option key, trigger phrase, 물리 mapping, SQL, 행, path, secret을 새로 저장하거나 반영하지 않는다. runtime rule set 자체는 caller-owned 입력이며 저장하지 않는다. low-entropy phrase/hash 추측 방지나 rule 작성자 인증을 주장하지 않는다.
8. 기존 `clarification-gate-v2`, `AmbiguitySignal`, `PendingClarification`, `ClarificationResult`, fingerprint 규칙과 오류 taxonomy를 깨지 않는다. detector 오류는 고정 code만 노출하고 기존 clarification 오류는 원래 고정 code로 유지한다.
9. 모델·repair provider·executor·파일·DB·network·Ollama·PostgreSQL·CNU 자료·private bundle을 호출하지 않는다. 기존 실패 code를 모호성으로 자동 승격하지 않는다. QueryIR 수정·재실행·UI 표시·single-use/expiry/session 저장은 후속 단계다.

## 합성 테스트

- match 0개: pending `None`, 차단·외부 호출 없음.
- match 1개: 결정적 signal 생성, 기존 pending 생성, 정확한 한 선택 `CONFIRMED`, 명시적 취소 `CANCELLED`.
- rule/option 입력 순서가 달라도 같은 option ID·정렬 pending·결과. 다른 rule/option 의미는 다른 ID.
- match 2개 이상: conflict, pending 없음. 서로 다른 ambiguity 축을 합치지 않음.
- trigger NFKC/case/whitespace 정규화, literal 포함 정책, 한 글자·blank/control trigger 거부.
- 중복 rule ID, 정규화 중복 trigger, option key/semantic ref 중복, 미지 entity/field, unpublished mapping, database mismatch 거부.
- prepare 뒤 질문·ontology·mapping·rule trigger/option 변경, match 소멸·추가 시 confirmation 거부. 기존 pending을 새 규칙으로 성공시키지 않음.
- forged `model_copy(update=...)`·`model_construct` request/ontology/mapping/rules/pending/choice 거부.
- raw path/secret형 request/database/rule/option/trigger가 pending/result/error/stdout/log에 없음. 단, caller-owned rule set·request 자체는 입력이므로 검사 출력에 직렬화하지 않는다.
- 기존 clarification 35개와 detector 신규 테스트 회귀. 관련 테스트만 실행한다. 전체 pytest/compileall/diff-check는 Codex 독립 검토 후 각각 한 번 실행한다.

## 완료 기준·보고

- 긴 diff·전체 로그 금지. 변경 파일, 감지 규칙, 호출 0회, 테스트 수·결과, 실패 원인, 남은 위험만 한국어로 요약한다.
- 계획·ADR·체크리스트·handoff·commit은 Codex 독립 검토 뒤 처리하므로 수정하지 않는다.
- 전체 `모호한 질문 감지 및 사용자 확인` 체크박스는 실제 UI/상호작용과 선택의 QueryIR 반영이 없으므로 완료 처리하지 않는다. Phase 6-08 하위 항목만 완료로 추가한다.

---

<a id="prompts-phase6-09-clarification-query-constraint"></a>

## 원본: prompts/phase6-09-clarification-query-constraint.md

# Phase 6-09: 확인 선택의 QueryIR constraint 전달

`ambiguity-detector-v1`과 `clarification-gate-v2`가 확인한 선택을 후속 QueryIR 생성에 명시적으로 전달하고, 생성 결과가 그 선택을 지켰는지 실행 전에 검증하는 합성 경계를 구현한다. **재감지·재확인 → semantic constraint 생성 → constraint-aware provider 1회 호출 → QueryIR·constraint 검증**을 한 기능 흐름으로 닫는다. 기존 `QueryModel`·`ExecutionHarness`·repair loop, 실제 UI·DB·Ollama·legacy는 변경하지 않는다.

## 먼저 확인

- 시작 시 `git status --short`와 최근 commit을 확인하고 기존 변경을 보존한다.
- `AGENTS.md`, `docs/development-history.md#next-session-handoff`, 계획 문서 Phase 6·다음 시작점·ADR-031/036, `docs/human-validation-checkpoints.md`의 E3/L1/L3/L5를 따른다.
- `clarification.py`, `ambiguity_detector.py`, `contracts.py`, `validator.py`, `model.py`와 해당 합성 테스트만 읽는다.

## 구현 계약

1. 별도 작은 모듈에 `clarification-query-constraint-v1` 계약을 둔다. 기존 `QueryModel.generate_ir(request, ontology, mapping)` protocol과 `ExecutionHarness`는 변경하지 않는다. 새 합성 provider protocol만 `generate_ir(request, ontology, mapping, constraint)`를 받는다.
2. 공개 coordinator는 request, runtime ontology, Published mapping, detector ruleset, pending, 원래 `ClarificationChoice`, constraint-aware provider를 입력받는다. caller가 `ClarificationResult`, `AmbiguitySignal`, semantic_ref, constraint를 직접 주입하는 API는 두지 않는다.
3. 모델 호출 전에 기존 `resolve_detected_clarification()`을 호출해 동일 context/ruleset에서 선택을 다시 확인한다. 같은 규칙으로 signal을 재생성하고, 확인 결과의 opaque selected fingerprint와 정확히 일치하는 option의 ontology entity/field `semantic_ref`만 constraint로 승격한다. 취소·no/multi match·pending/choice/context/ruleset drift·위조 입력은 모델 호출 전에 fail closed한다. 기존 ambiguity/clarification 고정 오류는 그대로 보존한다.
4. constraint는 frozen strict Pydantic 계약으로 version, request/database domain fingerprint, ontology/mapping fingerprint와 선택된 semantic_ref만 가진다. raw request/database/option/rule/trigger, 질문, 물리 table/column, SQL, 행, path, secret, display/free text를 저장하지 않는다. 모든 공개 입력과 nested 반환은 warning 비노출 plain JSON dump 후 재검증한다.
5. provider에는 재검증한 request/ontology/mapping/constraint의 방어적 깊은 복사만 전달한다. 정확히 1회 호출하고 provider 변이가 caller 객체나 사후 검증 context에 영향을 주지 못하게 한다. provider 예외는 원문을 반영하지 않는 새 고정 오류로 변환한다.
6. provider 반환은 `QueryIR`로 strict하게 검증한 뒤 기존 `validate_query_ir()`로 ontology/mapping 의미 검증을 통과해야 한다. model output/semantic validation 실패는 원문·값을 포함하지 않는 새 고정 오류로 닫고 기존 harness error/result를 위조하지 않는다.
7. 선택 constraint 준수 규칙은 작고 명시적으로 고정한다. 선택이 entity ref이면 `QueryIR.entity`가 정확히 같아야 한다. field ref이면 해당 ref가 `select` 또는 `filters[].field`에 최소 한 번 있어야 한다. 불일치는 고정 constraint violation이며 실행·compile·repair로 넘기지 않는다. 이는 질문 의미의 완전한 정답 판정이나 만족도 증거가 아니다.
8. 성공 반환은 versioned frozen 결과로 검증된 constraint와 QueryIR만 가진다. provider 호출 횟수나 자유 설명은 반환하지 않는다. 파일·DB·network·executor·compiler·repair·queue·snapshot을 호출하지 않는다.
9. 새 error code는 invalid input/context, cancelled, provider failure, invalid QueryIR, constraint violation처럼 고정 taxonomy로 제한한다. 입력·provider 예외 문자열을 message/log/stdout에 반영하지 않는다. 기존 ambiguity/clarification 오류 taxonomy는 감싸지 말고 유지한다.
10. package 공개 API를 추가하고 모듈 docstring에 신뢰 경계, entity/field 준수 규칙, bearer pending·규칙 기반 탐지 한계, 실행/UI/session/expiry 부재를 기록한다.

## 합성 테스트

- entity 선택과 field 선택이 constraint로 전달되고 준수 QueryIR가 성공한다.
- field는 select 또는 filter 사용을 각각 허용하며 미사용, 다른 entity는 constraint violation이다.
- cancel, no/multi match, request/ontology/mapping/ruleset/pending/choice drift·위조는 provider 0회다.
- provider 1회, 입력 방어적 복사, provider 변이 격리, provider 예외 고정 오류를 검증한다.
- invalid QueryIR와 기존 validator 실패를 고정 오류로 막고 compiler/executor/repair 호출 0회를 검증한다.
- constraint/result/error/stdout/log에 raw 질문·식별자·option/rule/trigger·물리 mapping·SQL·행·path·secret이 없는지 검사한다.
- Pydantic `model_copy(update=...)`·`model_construct` 위조와 serializer warning 비노출을 검사한다.
- 기존 ambiguity detector·clarification gate 회귀를 함께 실행한다.

OpenCode에서는 관련 테스트만 실행한다. 전체 pytest, compileall, `git diff --check`, 계획·ADR·handoff 갱신, commit은 Codex 독립 검토 뒤에만 수행한다. 상위 `모호한 질문 감지 및 사용자 확인` 체크박스는 실제 UI·실행 경로 통합 전까지 미완료로 유지한다.

---

<a id="prompts-phase6-10-clarification-execution-adapter"></a>

## 원본: prompts/phase6-10-clarification-execution-adapter.md

# Phase 6-10: 확인된 QueryIR의 안전 실행 adapter

Phase 6-09의 constraint 검증을 통과한 QueryIR를 기존 `ExecutionHarness`의 **request/context 검증 → QueryIR 재검증 → compile → SQL policy → read-only execute → result 검증** 경로에 우회 없이 전달하는 합성 adapter를 구현한다. 확인·생성·실행을 한 명시적 호출로 닫되 기존 `ExecutionHarness`, `QueryModel`, compiler, policy, executor 계약은 변경하지 않는다. 실제 UI·PostgreSQL·Ollama·CNU 자료·repair는 제외한다.

## 먼저 확인

- 시작 시 `git status --short`와 최근 commit을 확인하고 기존 변경을 보존한다.
- `AGENTS.md`, `docs/development-history.md#next-session-handoff`, 계획 문서 Phase 6·다음 시작점·ADR-031/036/037, `docs/human-validation-checkpoints.md`의 E3/L1/L3/L5를 따른다.
- `clarification_query_constraint.py`, `engine.py`, `compiler.py`, `policy.py`, `executor.py`, `contracts.py`와 관련 합성 테스트만 읽는다. legacy FastAPI/React와 기존 Ollama/PostgreSQL 경로는 변경하지 않는다.

## 구현 계약

1. 별도 작은 모듈에 `clarification-execution-adapter-v1`을 둔다. 공개 coordinator는 원래 request, runtime ontology, Published mapping, detector ruleset, pending, `ClarificationChoice`, constraint-aware provider, read-only executor를 받는다. caller가 `QueryIR`, `ConstrainedGenerationResult`, constraint, `QueryModel`, `HarnessResult`를 직접 주입하는 API는 두지 않는다.
2. provider/executor/compiler/policy의 필수 callable과 모든 context를 provider 호출 전에 검사한다. 공개 Pydantic 입력은 warning 비노출 plain JSON dump 후 재검증하고 깊은 snapshot을 만든다. invalid dependency/context, cancel, no/multi match, pending/choice/ruleset/context drift는 provider·compiler·policy·executor 호출 전에 fail closed한다. 기존 ambiguity/clarification/constraint 오류는 감싸지 않고 보존한다.
3. snapshot만 Phase 6-09 `run_constrained_generation()`에 전달한다. provider는 정확히 1회 호출되며 성공하면 검증된 `ConstrainedGenerationResult`를 dump-then-validate로 다시 고정한다. provider가 caller나 실행 snapshot을 변이하지 못하게 한다.
4. adapter 내부 전용 one-shot `QueryModel`만 만든다. 이 모델은 외부 주입할 수 없고 Phase 6-09의 검증된 QueryIR 한 개를 가진다. `ExecutionHarness`가 전달한 request/ontology/mapping이 adapter snapshot과 완전히 같을 때만 QueryIR의 방어적 복사본을 정확히 1회 반환한다. 두 번째 호출이나 context drift는 원문 없는 고정 내부 오류로 닫는다.
5. 기존 `ExecutionHarness`를 그대로 생성해 snapshot request/ontology/mapping으로 정확히 1회 `run()`한다. 기본 compiler는 기존 `SingleTableSQLiteCompiler`, 기본 policy는 기존 `ConservativeSqlPolicy`를 사용한다. 합성 테스트를 위해 기존 protocol과 호환되는 compiler/policy 주입은 허용하되 prebuilt SQL/result를 받지 않는다. read-only executor는 주입받지만 adapter가 `execute()`를 직접 호출하지 않는다.
6. 성공과 compile/policy/execution/result 실패는 기존 `HarnessResult`·`HarnessErrorCode`·stage를 그대로 보존한다. 자동 repair, 재생성, 재확인 반복, 성공 추정, critique/queue 등록을 하지 않는다. constraint/provider 단계 실패에는 executor가 호출되지 않는다.
7. versioned frozen 결과는 `adapter_version`, 재검증된 `ConstrainedGenerationResult`, 재검증된 `HarnessResult`만 가진다. harness의 `query_ir`는 generation의 QueryIR와 정확히 같아야 한다. 이미 생성·검증된 QueryIR이므로 harness 결과가 request/context/IR-generated/IR-validated 단계에서 실패하거나 query_ir가 없거나 달라지면 고정 adapter 내부 오류로 fail closed한다. compile 이후의 정상 harness 실패는 결과로 반환한다.
8. 결과의 `HarnessResult`는 기존 계약상 raw request ID, compiled SQL, columns/rows 또는 고정 오류 message를 포함할 수 있다. adapter가 이를 숨겼다고 주장하지 않는다. 새 adapter 오류·로그·stdout에는 질문, raw 식별자, option/rule/trigger, 물리 mapping, SQL, 행, path, secret, provider/OS 예외 문자열을 반영하지 않는다. 결과 보관·표시·접근 통제는 후속 API/UI 경계다.
9. adapter 자체 오류는 `EXECUTION_ADAPTER_INVALID_INPUT`과 `EXECUTION_ADAPTER_INTERNAL_ERROR`처럼 최소 고정 code만 둔다. 기존 하위 경계 오류는 그대로 전파한다. provider·compiler·policy·executor 호출 횟수를 결과에 넣지 않는다.
10. 파일·network·Ollama·PostgreSQL·queue·snapshot을 adapter가 직접 접근하지 않는다. 테스트용 기존 SQLite read-only executor 호출은 명시적 주입을 통해서만 발생한다. 모듈 docstring에 신뢰 경계, one-shot 모델, 기존 harness 재사용, HarnessResult의 raw content, 실행 후 side effect와 UI/session/expiry 부재를 기록한다.

## 합성 테스트

- entity/field 선택 각각에서 provider 1회 → 내부 model 1회 → compiler/policy/executor 1회로 성공하고 결과 QueryIR가 generation과 같다.
- 기존 `SQLiteReadOnlyExecutor`와 임시 합성 SQLite fixture를 사용해 실제 read-only 성공·0행 성공을 검증한다.
- cancel, no/multi match, request/ontology/mapping/ruleset/pending/choice 위조·drift, invalid dependency는 모든 downstream 호출 0회다.
- constraint 위반·provider 오류·invalid QueryIR는 compiler/policy/executor 0회다.
- compiler, policy, executor, result-schema 실패는 기존 code/stage와 검증된 QueryIR를 보존하고 재생성·repair하지 않는다.
- 악성 compiler SQL이 policy를 우회하지 못하고 executor 0회인지 검사한다.
- provider mutation과 executor가 받은 compiled query 변이가 caller/context/generation 결과를 바꾸지 않는지 검사한다.
- 내부 one-shot 모델의 두 번째 호출·context drift와 위조 nested 결과를 고정 내부 오류로 막는다. 공개 API로 모델/IR/result 주입이 불가능한지도 확인한다.
- adapter 오류·stdout/log에 raw 질문·식별자·option/rule/trigger·물리 mapping·SQL·행·path·secret·예외 원문이 없는지 검사한다. 기존 성공 `HarnessResult`가 SQL/rows를 포함한다는 경계는 명시적으로 확인한다.
- 기존 Phase 6-08/09 clarification·constraint 회귀를 함께 실행한다.

OpenCode에서는 관련 테스트만 실행한다. 전체 pytest, compileall, `git diff --check`, 계획·ADR·handoff 갱신, commit은 Codex 독립 검토 뒤에만 수행한다. 상위 `모호한 질문 감지 및 사용자 확인`은 실제 API/UI·session/expiry 전까지 미완료로 유지한다.

---

<a id="prompts-phase6-11-clarification-session-boundary"></a>

## 원본: prompts/phase6-11-clarification-session-boundary.md

# Phase 6-11: 합성 clarification session/API 경계

Phase 6-08~10의 **ambiguity 감지 → pending → 선택 constraint → 기존 read-only 실행**을 framework-neutral 메모리 session 하나로 연결한다. 실제 HTTP/FastAPI·React UI·운영 저장소·Ollama·PostgreSQL·CNU 자료는 제외한다. 이 단계의 목표는 bearer session의 만료·일회성·취소·동시 제출 경계를 합성 입력으로 닫는 것이다.

## 먼저 확인

- 시작 시 `git status --short`와 최근 commit을 확인하고 기존 변경을 보존한다.
- `AGENTS.md`, `docs/development-history.md#next-session-handoff`, 계획 문서 Phase 6·다음 시작점·ADR-031/036/037/038, `docs/human-validation-checkpoints.md`의 E3/L1/L3/L5를 따른다.
- `clarification.py`, `ambiguity_detector.py`, `clarification_query_constraint.py`, `clarification_execution_adapter.py`와 관련 합성 테스트만 읽는다. legacy FastAPI/React와 기존 Ollama/PostgreSQL 경로는 변경하지 않는다.

## 구현 계약

1. 별도 작은 모듈에 `clarification-session-v1`을 둔다. 메모리 coordinator는 `start(request, ontology, Published mapping, ruleset)`와 `submit(session_token, choice, provider, executor, 선택적 합성 compiler/policy)`의 두 공개 흐름만 제공한다. caller가 pending·signal·semantic ref·QueryIR·constraint·model·SQL·result·저장 record를 주입하는 API는 두지 않는다.
2. `start`는 모든 Pydantic 입력을 warning 비노출 plain JSON dump 후 재검증하고 깊은 snapshot을 만든 뒤 기존 detector/`prepare_detected_clarification()`을 호출한다. no-match는 `NOT_REQUIRED`이며 token/pending/options 없이 끝난다. 단일 match만 `PENDING`으로 저장하고 실제 choice에 그대로 제출할 수 있는 detector의 opaque option ID/semantic ref를 방어적 복사로 반환한다. 결과 validator는 각 selectable option ID를 기존 domain fingerprint로 변환했을 때 내부 `PendingClarification` option과 정확히 일치하는지 강제한다. multi-match·invalid context/ruleset은 기존 고정 오류를 그대로 전파하며 session을 만들지 않는다.
3. session token은 기본적으로 `secrets.token_urlsafe(32)` 이상의 예측 불가능한 opaque bearer 값이다. 내부 dict와 terminal tombstone에는 raw token 대신 domain-separated SHA-256만 저장한다. token과 request/question/database/physical mapping은 로그·오류에 반영하지 않는다. 합성 테스트용 clock/token factory 주입은 허용하되 결과 계약에 callable이나 호출 횟수를 넣지 않는다. token 충돌은 기존 session을 덮지 않고 고정 내부 오류로 닫는다.
4. coordinator 생성 시 양의 유한 TTL을 제한된 범위(예: 1초~1일)로 고정한다. deadline은 주입 monotonic clock으로 계산한다. `submit` 시 deadline 이상이면 active record를 제거하고 `EXPIRED` tombstone을 남긴 뒤 고정 `CLARIFICATION_SESSION_EXPIRED`로 거부한다. 만료 전에는 연장하지 않는다. 실제 wall-clock timestamp나 raw token을 결과에 저장하지 않는다.
5. `submit`은 token 형식과 `ClarificationChoice` 구조를 먼저 재검증한다. 알려지지 않은 token은 `..._NOT_FOUND`, 이미 처리 중/완료/취소/실패한 token은 `..._ALREADY_CONSUMED`로 고정 거부한다. active record 획득과 terminal `CONSUMED` 전환은 `threading.Lock` 아래 원자적으로 수행하며 downstream 호출 전에 끝낸다. 동시에 두 번 제출하면 최대 한 호출만 provider/compiler/policy/executor에 도달한다.
6. syntactically valid한 첫 submit은 성공·취소·unknown option·provider/compile/policy/executor 실패와 무관하게 영구 소비된다. downstream 예외 뒤 session을 active로 되돌리거나 자동 retry/repair하지 않는다. 이는 중복 read side effect를 막는 at-most-once dispatch이며, 프로세스 crash 후 exactly-once를 주장하지 않는다.
7. 취소는 빈 selection+`cancelled=True`만 허용한다. 저장한 snapshot/pending으로 기존 `resolve_detected_clarification()`을 호출해 context와 choice를 재확인하고 `CANCELLED` 결과를 반환한다. provider/compiler/policy/executor는 호출하지 않는다. selection과 cancel을 함께 주거나 invalid choice면 기존 clarification 오류를 그대로 전파하되 session은 소비된 상태다.
8. 확인 선택은 저장한 snapshot/ruleset/pending과 caller choice만 `run_clarified_execution()`에 전달한다. context/pending은 submit caller에게서 다시 받지 않는다. 기존 adapter가 constraint-aware provider → 내부 one-shot model → validation/compile/policy/read-only execute/result validation을 수행하게 하며 우회 실행은 금지한다. adapter 성공과 compile 이후 실패 `HarnessResult`를 모두 `EXECUTED` submission 결과에 보존한다.
9. versioned frozen `start`/`submit` 결과는 상태별 불변식을 자체 validator로 강제한다. `NOT_REQUIRED`는 token/pending 없음, `PENDING`은 token/pending 있음. `CANCELLED`는 기존 `ClarificationResult`만 있고 execution 없음, `EXECUTED`는 `ClarificationExecutionResult`만 있다. caller가 forged nested 결과를 직접 구성하지 못하도록 dump 재검증과 request fingerprint 일치를 확인한다.
10. session 자체 오류 taxonomy는 `INVALID_INPUT`, `NOT_FOUND`, `EXPIRED`, `ALREADY_CONSUMED`, `INTERNAL_ERROR`의 고정 code만 둔다. 기존 ambiguity/clarification/constraint/adapter 오류는 원문 없이 그대로 전파한다. 모듈은 stdout/log를 쓰지 않는다.
11. active session record는 raw 질문과 context를 프로세스 메모리에 보관한다는 한계를 docstring에 명시한다. submit/expiry 때 active record에서 제거하지만 Python 메모리의 즉시 안전 삭제를 주장하지 않는다. persistence·cross-process 공유·restart 복구·distributed lock·인증/권한·rate limit·HTTP cookie/CSRF·UI label은 범위 밖이다.

## 합성 테스트

- no-match `NOT_REQUIRED`: session/token 없음, provider/executor 호출 없음.
- 단일 match `PENDING`: opaque token과 기존 pending options 반환, caller 입력 변이와 내부 snapshot 격리.
- entity/field 선택 각각 `submit` 1회로 Phase 6-10 실행 성공. provider/compiler/policy/executor 각각 1회, 결과 QueryIR binding 유지.
- cancel은 `CANCELLED`, downstream 0회, 재사용은 `ALREADY_CONSUMED`.
- TTL 경계 직전 성공, deadline에서 `EXPIRED`, provider/executor 0회, 이후에도 재활성화 없음.
- unknown/malformed token, forged choice, unknown/multi/cancel+selection을 고정 오류로 막고 원문 비노출. 첫 syntactically valid invalid-choice submit 뒤 재사용 차단 확인.
- provider/constraint/compiler/policy/executor/result 실패 뒤에도 소비 유지. compile 이후 기존 HarnessResult code/stage 보존.
- 두 thread 동시 submit에서 정확히 하나만 provider/executor에 도달하고 다른 하나는 consumed 오류.
- token factory 충돌이 기존 active/terminal record를 덮지 않음.
- forged start/submit nested result와 public API의 pending/context/IR/result 주입 부재 확인.
- 오류·stdout/log에 token, 질문, raw 식별자, option/rule/trigger, physical mapping, SQL, rows, path, secret, 하위 예외 원문이 없음.
- Phase 6-08/09/10 관련 회귀를 함께 실행한다.

OpenCode에서는 관련 테스트만 실행한다. 전체 pytest, compileall, `git diff --check`, 계획·ADR·handoff 갱신, commit은 Codex 독립 검토 뒤에만 수행한다. 실제 UI/API route가 없으므로 상위 `모호한 질문 감지 및 사용자 확인` 체크는 미완료로 유지한다.

---

<a id="prompts-phase6-12-clarification-http-boundary"></a>

## 원본: prompts/phase6-12-clarification-http-boundary.md

# Phase 6-12: 합성 clarification HTTP 경계

Phase 6-11 `ClarificationSessionCoordinator.start/submit`을 legacy `backend/main.py`와 분리된 작은 FastAPI application factory로 노출한다. 목표는 두 POST endpoint, server-owned dependency lifecycle, 고정 오류→HTTP status 매핑, 응답 redaction을 합성으로 검증하는 것이다. 실제 legacy app mount, React UI, 인증, Ollama, PostgreSQL, CNU 자료는 제외한다.

## 먼저 확인

- `git status --short`, 최근 commit, `AGENTS.md`, handoff, 계획 Phase 6·ADR-039·다음 시작점, E3/L1/L3/L5를 따른다.
- `backend/main.py`는 route/path 관례만 확인하고 수정·import하지 않는다. import 시 legacy warmup/DB/model side effect가 생길 수 있으므로 새 모듈은 독립적이어야 한다.
- `clarification_session.py`와 Phase 6-08~11 관련 테스트만 읽는다.

## 구현 계약

1. 새 `backend/adaptive/api` package에 `clarification-http-v1` application factory를 둔다. `create_clarification_http_app(...) -> FastAPI`가 독립 app을 만들며 legacy global `app`, CORS, auth, startup warmup을 변경하지 않는다. endpoint는 정확히 `POST /api/adaptive/v1/clarifications/start`, `POST /api/adaptive/v1/clarifications/submit`이다.
2. factory는 동일한 Phase 6-11 coordinator 인스턴스, context loader, constraint-aware provider, read-only executor와 선택적 compiler/policy를 한 번 주입받아 두 endpoint closure가 재사용한다. 요청마다 coordinator를 새로 만들면 안 된다. 필수 callable은 app 생성 시 검증하고 잘못된 구성은 fixed `CLARIFICATION_HTTP_INVALID_CONFIGURATION`만 가진 구성 오류로 fail closed한다. 전역 mutable singleton은 만들지 않는다.
3. context loader는 `load(database_id)`로 server-owned runtime `Ontology`, Published `MappingProfile`, `DetectorRuleSet`을 반환한다. HTTP caller가 ontology/mapping/rules/pending/constraint/IR/model/executor/result를 주입할 필드는 없다. 반환 context는 dump-then-validate하고 mapping database ID가 요청과 일치해야 한다. loader 예외·부재·위조 context는 원문 없이 `CLARIFICATION_HTTP_CONTEXT_UNAVAILABLE`/HTTP 503으로 응답하고 session/provider/executor를 호출하지 않는다.
4. start request body는 `request_id`, `database_id`, `question`만 허용하고 strict extra-forbid·기존 길이 제한을 적용한다. malformed JSON, validation 오류, extra field는 body/token/question을 반영하지 않는 `CLARIFICATION_HTTP_INVALID_REQUEST`/HTTP 400으로 반환한다. FastAPI 기본 422 detail이 raw input을 돌려주지 않도록 이 app 전용 `RequestValidationError` handler를 둔다.
5. start endpoint는 server-loaded context와 request만 coordinator `start`에 전달한다. `NOT_REQUIRED` 응답은 version/status만, `PENDING` 응답은 version/status/session token과 option ID·semantic ref만 제공한다. request/database/ontology/mapping fingerprint, pending 전체, raw question/identifier/rule/trigger/physical mapping은 HTTP 응답에서 제외한다.
6. submit request body는 `session_token`, `selection`(opaque option ID 목록), `cancelled`만 허용한다. token은 body에만 있고 URL/header/log에 옮기지 않는다. endpoint는 `ClarificationChoice`를 구성해 같은 coordinator의 `submit`을 호출한다. caller context는 받지 않는다.
7. cancel 결과는 version/status `CANCELLED`만 반환한다. executed 결과는 version/status `EXECUTED`, 기존 harness `SUCCEEDED|FAILED`, columns/rows 또는 fixed error code/stage만 제공한다. SQL, QueryIR, constraint/generation, trace, request ID, fingerprint, 내부 error message는 항상 제외한다. 성공 rows는 JSON 값으로 재검증한다. 0행 성공을 실패로 바꾸지 않는다. compile/policy/executor/result-schema 실패는 HTTP 200의 redacted `EXECUTED`/`FAILED` domain result로 보존한다.
8. 오류 매핑은 단일 명시 표로 고정한다. 최소 규칙: malformed/input/invalid choice/ruleset=400, session not found=404, expired=410, consumed/conflict/context drift=409, constraint invalid/violation=422, provider error=502, context unavailable=503, adapter/session/internal·미지 오류=500. 응답은 항상 `{"error":{"code":"FIXED_CODE"}}` shape이며 message/detail/exception/input을 넣지 않는다. 기존 하위 fixed code를 가능한 그대로 사용하고 HTTP layer가 새 자유 문자열을 만들지 않는다.
9. endpoint 본문은 알려진 도메인 예외와 최종 unexpected exception을 모두 잡아 fixed response로 변환하여 exception 원문이 framework log/response로 탈출하지 않게 한다. app/모듈 자체는 print/log를 하지 않는다. access log 정책·proxy log·TLS·auth는 범위 밖임을 docstring에 명시한다.
10. response model은 strict/frozen/extra-forbid이며 상태별 불변식을 자체 validator로 강제한다. direct forged response에서 token/options 상태 불일치, success+error, failed+rows, cancelled+execution payload, SQL/IR/trace extra field를 거부한다.
11. 이 단계는 synthetic sub-application일 뿐 legacy app에 include/mount하지 않는다. 파일·network·Ollama·PostgreSQL·CNU 자료에 직접 접근하지 않는다. SQLite 실행은 테스트가 명시적으로 주입한 read-only executor를 통해서만 발생한다.

## 합성 테스트

- app factory의 정확한 두 POST route와 legacy `backend.main` 미import/미변경을 확인한다.
- no-match start 200 `NOT_REQUIRED`; 단일 match 200 `PENDING` 후 같은 app/coordinator의 submit 성공. entity/field, 0행 성공 포함.
- PENDING 응답은 token/options만, EXECUTED 성공은 columns/rows만 필요한 범위로 반환하며 SQL/IR/trace/fingerprint/raw ID/context는 없다.
- cancel 200 후 재사용 409, expired 410, unknown 404, 동시 제출은 하나만 200이고 다른 하나 409.
- malformed JSON, missing/extra/type/길이 오류가 400 fixed body이고 raw question/token/path/secret을 반영하지 않는다.
- loader 부재/예외/잘못된 DB·Draft mapping·위조 context는 503, coordinator/provider/executor 0회.
- ambiguity conflict/context drift, invalid/unknown/multi choice, constraint violation/provider error를 명시 표대로 매핑한다.
- compiler/policy/executor/result-schema 실패는 HTTP 200 domain failure로 기존 code/stage만 보존하고 SQL/message/trace/rows는 숨긴다.
- unexpected loader/coordinator/provider/response serialization 예외도 500 fixed body이며 stdout/log에 원문이 없다.
- factory dependency 검증, singleton coordinator 재사용, 공개 body/API에 forbidden injection field 부재를 검사한다.
- response model direct forgery와 row의 비 JSON 값을 fail closed한다.
- Phase 6-08/09/10/11 관련 회귀를 함께 실행한다.

OpenCode에서는 관련 테스트만 실행한다. 전체 pytest, compileall, `git diff --check`, 계획·ADR·handoff 갱신, commit은 Codex 독립 검토 뒤에만 수행한다. 실제 legacy mount와 React 사용자 흐름이 없으므로 상위 `모호한 질문 감지 및 사용자 확인` 체크는 미완료로 유지한다.

---

<a id="prompts-phase6-13-synthetic-clarification-asgi-prototype"></a>

## 원본: prompts/phase6-13-synthetic-clarification-asgi-prototype.md

# Phase 6-13 구현 지침: 합성 clarification ASGI 프로토타입

## 목표

Phase 6-12의 `create_clarification_http_app()`에 완전 합성 dependency를 조립해 팀원이 로컬 HTTP로 start → option 선택 → submit → 읽기 전용 결과를 재현할 수 있는 독립 ASGI 프로토타입을 만든다.

이 단계는 배포용 서비스가 아니다. legacy app, 실제 CNU 자료, Ollama, PostgreSQL, React, 외부 네트워크와 연결하지 않는다.

## 구현 범위

1. `backend/adaptive/api/synthetic_prototype.py`를 추가한다.
   - 공개 version 상수를 둔다.
   - 고정 synthetic database ID 하나만 허용한다.
   - 기존 `DEFAULT_SEED`와 `build_flat_database()`로 임시 SQLite DB를 만든다.
   - 최소 runtime `Ontology`, `Published MappingProfile`, `DetectorRuleSet`을 코드 내부 합성 상수로 만든다.
   - context loader는 고정 database ID만 받고 매 호출마다 방어적 복사본을 반환한다.
   - deterministic constraint-aware provider는 선택된 semantic field에 대응하는 하나의 유효 `QueryIR`만 반환한다. 질문에서 임의 SQL이나 물리 식별자를 해석하지 않는다.
   - 기존 `ClarificationSessionCoordinator`, `SQLiteReadOnlyExecutor`, `SingleTableSQLiteCompiler`, `ConservativeSqlPolicy`, Phase 6-12 factory만 조립한다.
   - 임시 directory 수명은 app 수명에 묶고 종료 시 정리한다. 중복 종료도 안전해야 한다.
   - `create_synthetic_clarification_prototype()` factory와 import 가능한 `app`을 제공한다.

2. 안전한 로컬 실행 진입점을 제공한다.
   - `python -m backend.adaptive.api.synthetic_prototype`로 실행한다.
   - host는 `127.0.0.1`로 고정한다. 외부 bind option을 제공하지 않는다.
   - port만 제한된 정수로 선택 가능하게 한다.
   - reload를 끄고 access log를 끈다.
   - `.env`, 실제 DB URL, Ollama 설정을 읽지 않는다.
   - 시작 시 질문·token·SQL·행·경로·secret을 출력하지 않는다.

3. `docs/development-history.md#synthetic-clarification-prototype`에 짧은 재현 절차를 작성한다.
   - 안전 실행 명령, 고정 database ID, start/submit JSON 예시를 제공한다.
   - start 응답에서 받은 실제 `session_token`과 선택한 `option_id`를 submit에 넣도록 설명한다.
   - 합성/로컬/무인증/비배포 경계를 분명히 적는다.
   - 결과를 모델 정확도나 실제 데이터 검증으로 해석하지 않는다.

4. 필요하면 `backend/adaptive/api/__init__.py`에 factory/version/database ID만 공개한다. module-level `app`을 package import에서 재수출해 불필요한 임시 DB side effect를 만들지 않는다.

## 계약과 안전 조건

- Phase 6-12 endpoint와 HTTP response/error 계약을 변경하지 않는다.
- legacy `backend.main`을 import·수정·mount하지 않는다.
- 외부 입력으로 ontology, mapping, rules, provider, executor, QueryIR, SQL, DB path를 주입받지 않는다.
- SQLite는 기존 read-only executor로만 질의한다.
- 합성 DB 밖 파일, 네트워크, 환경변수, private bundle에 접근하지 않는다.
- provider는 선택 constraint를 정확히 준수한다. 알 수 없는 semantic ref는 fixed provider failure로 닫는다.
- prototype 생성 실패와 CLI 설정 오류는 원문 데이터나 로컬 경로를 노출하지 않는 고정 오류로 종료한다.
- public response는 기존 Phase 6-12 redaction을 그대로 유지한다.
- session은 기존 메모리 TTL/at-most-once 한계를 그대로 가진다. 인증, rate limit, cross-process persistence를 추가하거나 주장하지 않는다.

## 테스트

`tests/adaptive/test_synthetic_clarification_prototype.py`에 합성 테스트를 추가한다.

- factory가 만든 app으로 start → 각 option 선택 → submit 성공을 end-to-end 검증한다.
- 반환 row/column이 `DEFAULT_SEED`의 해당 semantic field와 일치함을 검증한다.
- cancel, token 재사용, no-match `NOT_REQUIRED`, 알 수 없는 database ID를 검증한다.
- 고정 dependency와 방어적 context 복사, read-only DB, 종료 정리와 중복 종료를 검증한다.
- CLI가 loopback/reload off/access-log off만 사용하는지 실제 서버를 열지 않고 monkeypatch로 검증한다.
- import가 legacy, Ollama, PostgreSQL, `.env`를 불러오지 않는지 검증한다.
- 응답·stdout·stderr·log에 SQL, token, 임시 경로, 질문 원문, secret이 새지 않는지 검증한다.
- Phase 6-12 관련 테스트도 함께 실행한다.

## 작업 제한

- 구현 중 신규 테스트와 Phase 6-08~12 관련 테스트만 실행한다.
- 전체 pytest, compileall, `git diff --check`, commit, push는 실행하지 않는다.
- 계획 문서, ADR, checklist, changelog, handoff는 수정하지 않는다.
- 긴 diff와 전체 테스트 로그를 응답에 출력하지 않는다. 변경 파일, 핵심 계약, 실패 원인, 테스트 요약, 남은 위험만 한국어로 간결히 보고한다.

---

<a id="prompts-phase6-14-synthetic-clarification-react-flow"></a>

## 원본: prompts/phase6-14-synthetic-clarification-react-flow.md

# Phase 6-14 구현 지침: 합성 clarification React 흐름

## 목표

Phase 6-13 합성 ASGI prototype의 두 endpoint를 사용하는 별도 React 화면을 만든다. 팀원이 브라우저에서 질문 시작 → ambiguity option 선택 또는 취소 → submit 결과를 확인할 수 있어야 한다.

legacy `App`·`MainPage`·`ChatPopup` 흐름은 기준선으로 그대로 유지한다. 실제 Ollama, PostgreSQL, CNU 자료, auth와 연결하지 않는다.

## 진입점과 실행

- `/adaptive-prototype` 경로에서만 새 화면을 렌더한다. 다른 경로는 기존 `<App />`을 그대로 렌더한다.
- 새 router dependency를 추가하지 않는다. 현재 `window.location.pathname`을 작은 진입점 분기로 사용한다.
- Vite dev proxy에 `/api/adaptive`를 추가하고 기본 target은 `http://127.0.0.1:8123`으로 한다. 기존 `/api/v1` proxy는 변경하지 않는다.
- `package.json`에 backend prototype 단독 실행과 backend+Vite 동시 실행 script를 추가한다. backend는 Phase 6-13 module CLI를 그대로 사용하고 외부 bind/reload를 추가하지 않는다.
- 새 dependency를 설치하지 않는다.

## API client와 runtime 계약

- 새 코드는 `src/adaptive-prototype/` 아래 격리한다.
- start path: `/api/adaptive/v1/clarifications/start`
- submit path: `/api/adaptive/v1/clarifications/submit`
- database ID: `synth-clarification-prototype`
- TypeScript 타입 선언만 믿지 말고 모든 성공·오류 JSON을 runtime에서 엄격히 검사한다.
- 허용 start 상태는 `NOT_REQUIRED`, `PENDING`뿐이다. PENDING은 43~256자 URL-safe token과 2개 이상의 서로 다른 lowercase hex64 option ID, 유효 semantic ref가 필요하다.
- 허용 submit 상태는 `CANCELLED`, `EXECUTED`뿐이다. EXECUTED는 `SUCCEEDED` columns/rows 또는 고정 code/stage 실패 계약을 검사한다.
- HTTP 오류는 `{error:{code:string}}`만 수용한다. response text, exception message, request body, token을 UI·console·storage에 반사하지 않는다.
- fetch는 주입 가능하게 작성해 Node 내장 테스트에서 호출 횟수·body·응답 검증을 확인한다.
- `localStorage`, `sessionStorage`, cookie, auth header를 사용하지 않는다.

## React 상태와 안전 경계

- 상태를 명시적으로 분리한다: 입력 가능, start 진행, clarification pending, submit/cancel 진행, not-required, cancelled, executed success, executed failure, transport/contract error.
- 중복 클릭과 동시에 여러 요청을 차단한다.
- 각 네트워크 작업은 `AbortController`로 관리하고 unmount/reset 시 중단한다.
- session token은 pending 동안 메모리에만 보관한다. submit/cancel 시도 직전에 UI에서 재사용 불가능하게 만들고 성공·실패·abort 후 복원하지 않는다. 다시 시작하도록 안내한다.
- stale response가 새 요청 상태를 덮지 못하게 operation identity를 둔다.
- option label은 알려진 `Course.name`, `Instructor.name`에 한국어 표시명을 사용한다. 알 수 없는 semantic ref는 원문을 표시하지 않고 안전한 일반 label로 표시한다.
- 결과는 backend의 columns 순서로 generic table에 표시한다. cell은 제한된 JSON renderer로 표시하고 HTML 삽입을 쓰지 않는다.
- fixed error code는 제한된 한국어 메시지 표로 변환한다. 미지 code는 일반 오류로 표시하고 원문을 노출하지 않는다.
- 화면 상단에 `합성 데이터`, `로컬 전용`, `모델 정확도 아님`, `인증 없음`을 명확히 표시한다.
- SQL, QueryIR, token, physical mapping, 내부 trace를 표시하지 않는다.

## 테스트와 확인

새 dependency 없이 Node 24 내장 test runner와 type stripping을 사용한다.

- pure runtime parser/client/state helper를 `.ts`로 분리한다.
- `tests/frontend/`에 Phase 6-14 전용 테스트를 둔다.
- 정상 start PENDING/NOT_REQUIRED, submit success/failure/cancel을 검증한다.
- malformed/extra/duplicate option/잘못된 token/row-column mismatch/미지 상태를 거부한다.
- fetch URL·method·header·body가 정확한지, auth header와 storage 사용이 없는지 검증한다.
- error/exception/token 원문을 결과 message로 반사하지 않는지 검증한다.
- 관련 Node 테스트, 변경 frontend 파일 대상 ESLint, `npm run build`, Phase 6-13 backend prototype 테스트만 실행한다.
- 실제 browser 조작과 frontend/backend 실기동 smoke는 Phase 6-15로 남긴다.

## 작업 제한

- legacy `App.tsx`, `MainPage`, `ChatPopup`, `/api/v1` 동작을 변경하지 않는다. `main.tsx`의 경로 분기만 최소 변경한다.
- backend Phase 6-13/HTTP 계약을 변경하지 않는다.
- 실제 서비스, 외부 네트워크, Ollama, PostgreSQL, CNU 자료를 사용하지 않는다.
- 계획 문서, ADR, checklist, changelog, handoff는 수정하지 않는다.
- 전체 pytest, 전체 ESLint, compileall, `git diff --check`, commit, push는 실행하지 않는다.
- 긴 diff와 전체 로그를 출력하지 않는다. 변경 파일·핵심 계약·실패 원인·테스트 요약·남은 위험만 한국어로 간결히 보고한다.

---

<a id="prompts-phase6-15-synthetic-prototype-browser-smoke"></a>

## 원본: prompts/phase6-15-synthetic-prototype-browser-smoke.md

# Phase 6-15 구현 지침: 합성 prototype browser smoke와 팀 공유 handoff

## 목표

Phase 6-14의 `/adaptive-prototype` 화면과 Phase 6-13 loopback ASGI backend를 실제로 함께 실행할 수 있는 팀 공유 단위로 정리한다. 이 단계는 합성 fixture 전용 내부 prototype의 재현성과 경계를 닫는 작업이다. legacy 시스템 교체, 실제 서비스 연결, 외부 시연 승인을 뜻하지 않는다.

## 작업 범위

1. 현재 구현과 실행 script를 필요한 범위만 확인한다.
2. `docs/development-history.md#synthetic-clarification-team-handoff`를 새로 작성한다.
   - 사전 조건
   - 단일 실행 명령과 접속 URL
   - 종료 방법
   - 브라우저 smoke 시나리오 표
   - 합성 전용·loopback-only·무인증·비영속 session 등 경계와 알려진 한계
   - 실제 browser 관찰 전 상태는 `PENDING_BROWSER_SMOKE`로 적는다.
3. 기존 `docs/development-history.md#synthetic-clarification-prototype`에서 팀 handoff 문서를 연결한다.
4. 팀원이 재현하는 데 막히는 명백한 script 또는 문서 결함만 최소 수정한다. 새 제품 기능이나 운영 연결은 추가하지 않는다.

## 브라우저 smoke 시나리오

실제 통과 여부는 Codex가 browser를 직접 조작한 뒤 기록한다. OpenCode는 통과했다고 추정하지 않는다.

- `강의 목록 보여줘` 시작 시 두 option 표시
- `강의명 기준` 선택 후 합성 결과 표 표시
- 새 흐름에서 `교수자명 기준` 선택 후 합성 결과 표 표시
- 명시적 취소가 `CANCELLED`로 종료
- trigger가 없는 질문이 `NOT_REQUIRED`로 종료
- backend 연결 실패가 안전한 일반 오류로 표시되고 token, SQL, 경로, 원시 응답을 노출하지 않음
- terminal 상태에서 UI가 같은 session 재제출을 허용하지 않음; backend token at-most-once 계약은 기존 자동 테스트 증거를 함께 명시
- `/` legacy 경로가 기존 화면을 유지

## 금지 범위

- legacy app 동작·API·인증 변경 금지
- 실제 CNU 데이터, `.env`, PostgreSQL, Ollama, private recording 접근 금지
- 외부 bind, 배포, 계정·권한·rate limit 구현 금지
- 계획 문서, ADR, 체크리스트, 변경 이력, `docs/development-history.md#next-session-handoff` 수정 금지
- browser를 실제로 조작했다고 주장 금지
- 전체 pytest, compileall, commit, push 금지

## 검증

변경과 직접 관련된 최소 검증만 실행한다.

- adaptive prototype Node 테스트
- 변경한 frontend 파일에 대한 ESLint(해당할 때만)
- frontend production build
- synthetic prototype backend 관련 pytest

전체 로그나 긴 diff는 출력하지 않는다. 마지막 응답은 다음만 한국어로 간결히 보고한다.

- 변경 파일
- 공유 계약과 안전 경계
- 실패 원인(있을 때)
- 테스트 요약
- 실제 browser smoke에 남긴 항목과 위험

---

<a id="prompts-phase6-16-active-database-selection"></a>

## 원본: prompts/phase6-16-active-database-selection.md

# Phase 6-16 구현 지침: 합성 활성 DB 선택과 메인 목록·질의 동기화

## 목표

메인 수강신청 화면에서 사용 DB를 선택하고, 선택한 합성 DB가 강의 목록과 `/adaptive-prototype` clarification 질의에 동일하게 적용되는 하나의 기능 흐름을 완성한다.

이번 단계는 **미리 등록된 합성 DB 선택**까지만 구현한다. 파일 업로드·임의 경로·PostgreSQL 등록·실제 CNU 데이터·Ollama live 호출·인증·전역 current DB·관리자 mapping publish UI는 구현하지 않는다.

## 필수 계약

1. 기존 legacy `/api/v1/courses`와 기존 로그인·수강신청 화면 동작을 제거하거나 교체하지 않는다. 기본 선택은 legacy 기준선이다.
2. 합성 prototype backend는 최소 두 개의 서버 소유 등록 DB를 제공한다.
   - 동일 `DEFAULT_SEED`를 사용한 `flat` schema
   - 동일 `DEFAULT_SEED`를 사용한 `legacy` schema
   - 서로 다른 물리 테이블·컬럼과 각 DB의 Published Mapping을 사용한다.
3. backend에 strict JSON 응답의 합성 DB 목록 endpoint와 canonical course catalog endpoint를 추가한다.
   - caller가 파일 경로·mapping·SQL·물리 식별자를 주입할 수 없어야 한다.
   - catalog는 seed를 그대로 반환하지 말고, 선택 DB를 기존 read-only executor와 Published Mapping/안전한 compiler 경로로 조회한 결과를 canonical course DTO로 변환한다.
   - unknown DB, mapping/schema drift, 실행 실패는 고정 code의 redacted 오류로 닫는다.
   - 두 schema의 canonical course 결과가 같음을 테스트한다.
4. 기존 clarification start는 request의 `database_id`를 서버 registry에서 해석한다. session token은 시작 DB·context fingerprint에 계속 묶여야 하며 submit에서 다른 DB로 바꿀 API를 만들지 않는다.
5. frontend에 strict parser/client를 둔다. 응답의 version, 허용 key, DB ID 형식, canonical course 필드·타입·row 수를 runtime 검증한다. raw server body나 내부 식별자·경로·SQL을 UI 오류에 반사하지 않는다.
6. 메인 화면 header에 현재 DB와 `DB 변경` 버튼을 추가한다.
   - modal/list에서 legacy 기준선 또는 등록 합성 DB를 선택하고 명시적으로 적용한다.
   - legacy 선택은 기존 `coursesApi.getCourses()`를 그대로 쓴다.
   - 합성 선택은 adaptive catalog endpoint를 쓰며 검색 keyword도 선택 DB 요청에 포함한다.
   - loading/error/retry를 표시하고 실패를 빈 목록으로 위장하지 않는다.
   - DB 변경 시 선택 강의·경고·hover 등 이전 DB에 종속된 UI 상태를 초기화한다.
7. 활성 선택은 서버 전역 mutable state가 아니라 browser `sessionStorage`에 저장한다. 값은 registry 응답으로 재검증하고, 손상·미등록 값은 legacy로 fail closed한다. localStorage·cookie·backend global current DB를 쓰지 않는다.
8. `/adaptive-prototype` 진입 링크 또는 버튼은 선택한 합성 DB ID를 명시적으로 전달한다. adaptive page는 같은 검증된 선택을 표시하고 `startClarification()`에 전달한다. legacy가 선택된 경우 합성 질의 지원 없음이 분명해야 한다.
9. 기존 fixed `DATABASE_ID` 의존을 제거하되 기존 Phase 6-12~15 응답·token·redaction 계약은 유지한다.
10. prototype app 종료 시 두 임시 DB 모두 정리한다. import side effect, 외부 network, `.env`, PostgreSQL, Ollama 접근을 추가하지 않는다.

## 구현 품질

- registry/context는 방어적 복사 또는 immutable view로 노출한다.
- DB ID와 label은 제한된 길이·문자 집합으로 검증한다.
- SQL identifier는 caller 입력으로 조립하지 않는다.
- canonical DTO 변환은 한 곳에 모으고 frontend legacy `Course` 타입으로 매핑 가능한 명확한 계약을 둔다.
- React async 요청은 abort/stale response 방지를 유지한다.
- 새 코드가 기존 legacy API client의 auth/base URL 계약을 오염시키지 않게 별도 adaptive client를 쓴다.

## 테스트 범위

구현 중 관련 테스트만 실행한다.

- backend: registry 목록, 두 DB catalog 동등성, keyword, unknown DB, drift/failure redaction, clarification 두 DB 분기, cleanup
- frontend: strict parser/client, sessionStorage fail-closed, DB 적용·상태 초기화에 필요한 순수 state/helper, dynamic clarification database ID
- 기존 Phase 6-13~15 backend/frontend 관련 회귀
- 변경 frontend 파일 ESLint와 frontend build

전체 pytest, 전체 compileall, 전체 `git diff --check`, 문서 계획/ADR/handoff 수정, commit, push는 하지 않는다. 마지막에는 긴 diff나 전체 로그 없이 변경 파일, 핵심 계약, 실패 원인, 실행한 테스트와 결과, 남은 위험만 한국어로 요약한다.

---

<a id="prompts-phase6-17-private-sqlite-registration"></a>

## 원본: prompts/phase6-17-private-sqlite-registration.md

# Phase 6-17 구현 지침: private SQLite 등록 경계

## 목표

사용자가 제공한 독립형 SQLite 파일을 backend core가 안전하게 검증·복사하고, 표본값 없이 schema catalog와 rule-based mapping proposal을 생성해 private storage에 하나의 immutable registration으로 게시한다. 이 단계의 결과는 `PENDING_REVIEW`이며, 사람 검토와 명시적 publish 전에는 Phase 6-16 database registry, course catalog, clarification 실행 경로에서 선택하거나 질의할 수 없다.

## 범위

- 새 `backend/adaptive/registration` package에 strict contract, 고정 오류 코드, storage/service를 둔다.
- public core API는 repository root, source SQLite path, `database_id`, display label을 받아 등록하고 strict load할 수 있어야 한다. HTTP upload endpoint와 브라우저 UI는 만들지 않는다.
- 기본 private root는 `.schema-adaptive/registered-databases`다. 기존 recording bundle의 Git root·ignored/private containment·symlink/junction/reparse 방어 원칙을 재사용하거나 같은 강도의 공용화된 helper로 사용한다. 기존 API를 약화하지 않는다.
- `database_id`는 Phase 6-16과 같은 정규화된 소문자 `^[a-z0-9][a-z0-9-]{0,63}$`만 허용하고 Windows 예약 장치명과 trailing dot/space를 거부한다. 기존 합성 registry ID와 충돌해도 거부한다. label도 공백-only, 제어문자, 과도한 길이를 거부한다.
- source는 regular file이어야 하며 symlink/junction/reparse를 거부한다. 빈 파일과 32 MiB 초과 파일을 거부하고, 확장자만 신뢰하지 말고 SQLite header와 실제 read-only immutable open/`PRAGMA quick_check`를 검증한다. `-wal`, `-shm`, `-journal` sidecar가 등록 전후에 있거나 복사본의 journal mode가 WAL이면 독립 snapshot이 아니므로 fail closed한다.
- source를 수정하거나 write-open하지 않는다. source는 한 번 연 binary fd에서 fstat/header/32 MiB 상한/streaming copy/SHA-256을 처리하고, 경로를 다시 열어 복사하지 않는다. 상한은 stat뿐 아니라 읽는 중에도 강제한다. 복사본 검사는 `mode=ro&immutable=1` 및 `PRAGMA query_only=ON`으로 sidecar 생성을 막는다. 필요하면 기존 `SQLiteAdapter`에 default-off immutable 옵션을 추가하되 기존 호출 동작은 바꾸지 않는다.
- `SQLiteAdapter`나 sqlite가 상세 메시지를 포함한 예외를 내더라도 registration 경계에서 원래 message/경로를 폐기하고 고정 오류 코드로만 재매핑한다. 오류 문자열·manifest에 source 절대경로, row/sample value, SQLite 상세 오류 payload를 노출하지 않는다.
- `SQLiteAdapter.inspect(..., include_samples=False)`, `load_default_ontology()`, `RuleBasedSchemaAligner.align()`을 사용한다. 새 병렬 schema/mapping 구현을 만들지 않는다.
- `registration.json` 자체를 manifest이자 commit marker로 정의한다. schema version, `PENDING_REVIEW` 상태, database ID/label, content hash/bytes, schema fingerprint, ontology version, 나머지 각 artifact의 상대 경로·hash·bytes를 기록한다. 허용 파일은 정확히 `database.sqlite3`, `catalog.json`, `proposal.json`, `registration.json` 네 개며 registration manifest는 staging 안에서 마지막에 쓴다.
- registration 디렉터리는 immutable/no-overwrite다. 같은 database ID 재등록은 기존 자료를 덮어쓰지 않는 고정 conflict로 닫는다. staging은 private root와 같은 volume 아래에 만들고, database ID별 `O_CREAT|O_EXCL` lock으로 cooperating writer를 직렬화한 뒤 final 부재를 재검증하고 staging directory를 단일 rename한다. lock과 staging은 성공·실패 모두 정리한다. 동시 등록도 최대 하나만 성공해야 한다.
- strict loader는 private containment, redirect와 지원 플랫폼의 hardlink, 허용 파일 집합, size/hash, Pydantic contract, database ID, catalog/proposal fingerprint·ontology 일치, 저장 DB 재검사의 schema fingerprint를 검증한다. 저장 후 변조·추가 파일·경로 이탈은 fail closed한다. hardlink link-count를 신뢰할 수 없는 플랫폼에서는 배타 생성·containment·reparse 검사가 주 방어선이다.
- 등록 성공만으로 Phase 6-16 fixed registry를 수정하지 않는다. 기존 `/api/adaptive/databases`, course catalog, clarification 동작과 legacy baseline은 그대로 둔다.

## 오류 계약

- 외부에는 payload 없는 고정 enum 문자열만 노출하는 전용 예외를 둔다. 최소 분류는 `INVALID_INPUT`, `UNSAFE_SOURCE`, `SOURCE_TOO_LARGE`, `INVALID_SQLITE`, `PRIVATE_STORAGE_INVALID`, `REGISTRATION_CONFLICT`, `ARTIFACT_MISMATCH`, `STORAGE_ERROR`, `INSPECTION_FAILED`, `MAPPING_FAILED`다.
- 내부 `sqlite3`, catalog, validation, OS, Git 검사 예외의 메시지나 경로를 그대로 연결하지 않는다.

## 테스트

새 집중 테스트에서 최소 다음을 검증한다.

1. 합성 flat/legacy 중 하나의 독립 SQLite를 등록하면 copy hash·catalog fingerprint·proposal이 결정적으로 일치하고 sample values/source path가 artifact와 예외에 없다.
2. strict load round-trip이 성공하고 상태가 `PENDING_REVIEW`이며 Phase 6-16 registry 응답/선택 가능 목록은 전후 동일하다.
3. empty, header 위조, corrupt DB, 초과 크기, sidecar 동반 source를 각 고정 오류로 거부한다.
4. source redirect와 private target redirect/path escape/Git tracked 또는 not-ignored 경계를 거부한다. 지원 환경에서는 hardlink도 거부한다.
5. 기존 ID 충돌, artifact 변조/누락/추가, DB schema 변조, manifest의 path/hash/fingerprint/ontology/ID 변조를 fail closed한다.
6. inspection/mapping/write 실패 후 published directory와 staging residue가 없고, 동시 동일 ID 등록은 하나만 성공한다.
7. 관련 catalog/mapping/Phase 6-16 backend 테스트만 실행한다. 전체 pytest, compileall, `git diff --check`, 문서 갱신, commit, push는 하지 않는다.

## 변경 금지

- legacy system, 기존 합성 fixture 내용, frontend, 실제 CNU 자료, PostgreSQL, Ollama live 호출을 변경하거나 사용하지 않는다.
- mapping review/publish를 자동 승인하지 않는다. `AUTO_APPROVE_CANDIDATE`는 제안 신호일 뿐 등록 상태를 publish로 바꾸지 않는다.
- 계획/ADR/handoff는 구현 agent가 수정하지 않는다.

## 완료 보고

긴 diff와 전체 로그를 출력하지 않는다. 변경 파일, 핵심 계약, 실행한 관련 테스트와 결과, 실패 원인/남은 위험만 한국어로 간결히 보고한다. 사용량 통계는 조회하거나 보고하지 않는다.

---

<a id="prompts-phase6-18-loopback-sqlite-upload"></a>

## 원본: prompts/phase6-18-loopback-sqlite-upload.md

# Phase 6-18 구현 지침: loopback SQLite upload와 검토 대기 요약

## 목표

메인 화면의 `DB 변경` modal에서 사용자가 standalone SQLite 파일을 선택하고, 제한된 loopback HTTP upload 경계를 통해 기존 Phase 6-17 private registration을 호출한다. 성공한 등록은 계속 `PENDING_REVIEW`로 격리하며, schema catalog와 mapping proposal의 표본값 없는 요약만 읽기 전용으로 표시한다. 등록 직후 course catalog나 clarification 실행 DB로 선택할 수 없어야 한다.

## 완결 범위

```text
DB 변경 modal
→ database ID·label·SQLite 파일 선택
→ 제한된 raw-body loopback upload
→ 임시 파일 cleanup 보장
→ Phase 6-17 register_database()
→ Phase 6-17 load_registration() strict reload
→ PENDING_REVIEW schema/mapping 요약 반환
→ modal 안 읽기 전용 표시
```

- 기존 synthetic prototype FastAPI app에 등록 전용 POST endpoint를 추가한다. endpoint는 raw SQLite body를 streaming으로 읽고 `database_id`와 `label`만 명시적으로 받는다. multipart parser가 route 진입 전에 무제한 spool하지 않도록 multipart upload를 사용하지 않는다.
- `Content-Type`은 고정 SQLite upload media type만 허용한다. 누락·중복 query parameter, 예상하지 않은 query parameter, 비어 있는 body, 잘못된 `Content-Length`, 32 MiB 초과 선언 또는 실제 streaming 초과를 fail closed한다.
- request body를 한 번에 메모리에 적재하지 않는다. 제한된 chunk로 exclusive 임시 regular file에 쓰며 32 MiB를 HTTP 경계에서도 강제한다. 성공·실패·client disconnect 모두 임시 파일을 정리한다. 임시 경로나 원래 filename은 응답·로그·예외에 노출하지 않는다.
- HTTP boundary는 기존 `backend.adaptive.registration.register_database()`와 `load_registration()`을 그대로 호출한다. 별도 SQLite header/schema/mapping 검사나 약한 loader를 만들지 않는다.
- 등록 성공 응답은 strict reload 결과에서만 만든다. 상태는 정확히 `PENDING_REVIEW`여야 한다. 응답은 version, database ID, label, status, schema fingerprint와 sample 없는 schema/mapping proposal 요약만 포함한다. source path, private path, row/sample value, SQL, SQLite 상세 오류, traceback을 포함하지 않는다.
- schema 요약은 table/column/FK 개수와 읽기 전용 table 구조 요약만 허용한다. mapping 요약은 proposal의 semantic field, 검토 상태, 상위 후보의 물리 table/column·confidence/reasons처럼 이미 proposal에 있는 metadata만 허용한다. 새 자동 승인·교정·publish 판단을 만들지 않는다.
- 모든 HTTP 실패는 고정 공개 code와 고정 status mapping만 반환한다. Phase 6-17 `RegistrationErrorCode`의 원문 예외나 내부 message를 반사하지 않는다. validation, OS, SQLite, catalog, mapping, JSON 오류도 redacted generic code로 닫는다.
- endpoint 성공 전후 `/api/adaptive/v1/databases` 응답은 byte-equivalent semantic content를 유지한다. 새 registration ID는 Phase 6-16 `SyntheticDatabaseRegistry`, course catalog, clarification loader/executor에 추가하지 않는다.
- 메인 `DB 변경` modal에 별도 `SQLite 등록(검토 대기)` 영역을 둔다. database ID, label, `.sqlite`/`.sqlite3` file 입력과 등록 button을 제공한다. extension은 UX 힌트일 뿐 backend 신뢰 근거가 아니다.
- frontend client는 raw `File` body를 전송하고 strict 성공/error parser를 사용한다. 예상하지 않은 key/type/status/version은 contract 오류로 처리한다. 서버 body·path·SQL·상세 message를 사용자 문자열에 반사하지 않는다.
- 업로드 중 중복 submit을 막고 modal close/unmount에서 request를 abort한다. 새 시도는 이전 결과·오류를 명시적으로 교체하며 stale response가 최신 UI를 덮지 못하게 한다.
- 성공 시 active/pending DB 선택, sessionStorage, course list, 선택 강의, clarification link를 바꾸지 않는다. registry refetch로 등록 DB를 선택 가능하게 만들지 않는다. 성공 요약에는 `검토 전에는 조회·질의에 사용할 수 없음`을 명시한다.
- mapping 수정·승인·publish, 실행 registry 편입, 등록 목록 관리·삭제는 구현하지 않는다.

## 보안·격리 계약

- Phase 6-17의 32 MiB 제한, standalone SQLite 검사, Git ignored private storage, immutable/no-overwrite, read-only immutable schema 검사, sample 없는 artifact, strict reload를 약화하지 않는다.
- `PENDING_REVIEW` 이외 상태를 성공 응답으로 반환하지 않는다.
- Phase 6-16 fixed registry와 Published Mapping만 실행 권한을 가진다. registration 성공은 권한 상승이 아니다.
- loopback CLI host `127.0.0.1`, reload off, access log off 계약을 유지한다. CORS나 외부 bind를 추가하지 않는다.
- legacy `/api/v1/courses`와 기존 메인 화면 legacy 흐름을 변경하지 않는다.
- 실제 CNU 자료, PostgreSQL, Ollama live 호출을 사용하지 않는다.

## 테스트

새 backend 집중 테스트에서 최소 다음을 검증한다.

1. 합성 standalone SQLite raw upload 성공, `PENDING_REVIEW` strict summary, sample/path/SQL 비노출.
2. 등록 전후 Phase 6-16 database registry 동일, 새 ID의 catalog·clarification 접근 거부.
3. empty, 잘못된 content type/parameter/content length, 선언·실제 32 MiB 초과, malformed/corrupt/sidecar 성격 입력, 기존 ID 충돌의 고정 HTTP mapping.
4. registration/strict reload 실패와 예상 밖 예외의 redaction, 임시 파일 cleanup.
5. duplicate/unknown parameter와 response contract extra field를 fail closed.

새 frontend 집중 테스트에서 최소 다음을 검증한다.

1. strict response/error parser와 raw File request 계약.
2. idle/uploading/success/error/retry 상태 전이, 중복 submit 차단, abort/stale 차단.
3. 성공 후 active/pending selection과 sessionStorage·course UI를 바꾸지 않고 `PENDING_REVIEW` 요약만 표시.
4. 등록 DB가 선택 radio와 clarification link에 추가되지 않음.
5. backend 상세 body나 예외 문자열 비노출.

OpenCode는 신규 및 직접 관련 Phase 6-16/17 backend·frontend 테스트, 변경 frontend lint/build만 실행한다. 전체 pytest, compileall, `git diff --check`, 문서 갱신, commit, push는 하지 않는다.

## 변경 금지

- legacy system, 실제 CNU 자료, PostgreSQL, Ollama live 호출, 기존 합성 fixture 내용을 변경하지 않는다.
- mapping review/edit/approve/publish, Phase 6-16 registry 편입, 실제 실행을 추가하지 않는다.
- 계획/ADR/change log/handoff는 구현 agent가 수정하지 않는다.

## 완료 보고

긴 diff와 전체 로그를 출력하지 않는다. 변경 파일, 핵심 계약, 실패 원인, 실행한 관련 테스트 결과, 남은 위험만 한국어로 간결히 보고한다. 모델별 사용량이나 통계는 조회·보고하지 않는다.

---

<a id="prompts-phase6-19-private-mapping-publish-activation"></a>

## 원본: prompts/phase6-19-private-mapping-publish-activation.md

# Phase 6-19 구현 지침: private mapping 검토·publish와 실행 registry 활성화

## 목표

Phase 6-18에서 등록한 합성 standalone SQLite의 `PENDING_REVIEW` mapping proposal을 메인 `DB 변경` 흐름에서 명시적으로 검토·수정하고 publish한다. strict publish가 성공한 registration만 Phase 6-16 registry에 편입해 course catalog와 clarification에서 선택·실행할 수 있게 한다. registration 성공, draft 생성, 일부 field 승인만으로는 실행 권한이 생기지 않는다.

## 완결 흐름

```text
strict PENDING_REVIEW registration
→ private review draft 최초 생성
→ field별 candidate 승인 또는 catalog column으로 remap
→ 모든 prototype 필수 semantic field 검토 완료
→ schema/ontology/fingerprint/physical column 재검증
→ immutable Published Mapping 게시
→ strict published activation reload
→ Phase 6-16 registry에 private entry 노출
→ 같은 published mapping으로 catalog·clarification read-only 실행
```

이번 단계는 완전 합성 flat 또는 legacy-schema standalone SQLite로 end-to-end 검증한다. 실제 CNU 자료나 실제 학교 mapping을 사용하지 않으며 L2를 승인 처리하지 않는다.

## backend review·storage 계약

- 기존 `MappingReviewService`, `MappingDraft`, `PublishedMappingProfile`, `to_harness_mapping()`, default ontology를 재사용한다. 병렬 mapping 판단이나 자동 승인을 만들지 않는다.
- review state는 immutable Phase 6-17 registration 디렉터리를 수정하지 않고 별도 Git ignored private root(예: `.schema-adaptive/registration-reviews/<database_id>`)에 둔다.
- private root containment, Git ignored/untracked, redirect/reparse/hardlink 방어는 Phase 6-17과 같은 강도로 적용한다. source/private path와 내부 예외는 공개 응답에 노출하지 않는다.
- draft init은 strict `load_registration()`의 proposal/catalog에서만 생성한다. 동일 registration에 대한 같은 init은 idempotent해야 하며 기존 review를 덮어쓰지 않는다. registration fingerprint·ontology가 달라졌으면 fail closed한다.
- field mutation은 server-owned proposal/catalog/draft를 매번 strict reload한 뒤 기존 review service를 호출한다. caller는 draft 전체, proposal, catalog, status, published mapping을 주입할 수 없다.
- 허용 action은 이번 UI 흐름에 필요한 `INIT`, `APPROVE`, `REMAP`, `PUBLISH`로 제한한다. `APPROVE`는 proposal candidate index만 받는다. `REMAP`은 strict registration catalog에 실제 존재하는 table/column만 받으며 reason을 요구한다. reviewer/published_by는 짧은 비공백·제어문자 없는 synthetic-local 식별자만 받는다.
- database ID별 cooperating lock으로 init/mutation/publish를 직렬화한다. mutation은 strict old state에서 새 draft를 원자 교체하고 실패 시 기존 bytes를 보존한다.
- publish는 기존 `MappingReviewService.publish()` 검증에 더해 Phase 6-16 canonical course DTO의 15개 `CANONICAL_SELECT` field가 모두 Published Mapping에 정확히 존재하는지 요구한다. 일부 field ignore/reject, pending, duplicate physical column, stale fingerprint, ontology drift, 없는 column, ambiguous/disconnected JOIN은 활성화하지 않는다.
- publish version은 server가 결정한다. immutable `mapping-vNNNN.json`과 atomic current pointer/activation manifest를 manifest-last 방식으로 게시한다. 같은 publish retry는 idempotent하거나 fixed conflict로 닫고 기존 published bytes를 덮어쓰지 않는다.
- strict activation loader는 registration strict reload, review artifact 허용 파일 집합·bounded exact JSON·hash/version, Published Mapping contract, database/schema/ontology fingerprint, physical relation graph와 canonical 15-field 실행 가능성을 다시 검증한다.
- process restart 뒤에도 strict published activation만 registry에 복원한다. malformed/tampered/stale/incomplete activation 하나가 있으면 해당 ID만 숨기고 fixed registry 동작과 다른 valid activation은 유지한다. 임의 디렉터리나 draft만 존재하는 ID는 노출하지 않는다.

## HTTP 계약

- 기존 loopback synthetic prototype에 작은 review API를 둔다. GET detail과 strict JSON action POST 정도의 최소 surface를 사용한다.
- GET은 draft/published review metadata, semantic field, 현재 decision, proposal candidates, remap에 허용되는 catalog table/column metadata만 반환한다. sample value, row, SQL, path, SQLite detail, raw exception은 반환하지 않는다.
- POST body는 `extra=forbid`, strict type, action별 discriminated contract를 적용한다. duplicate JSON key, unknown action/field/candidate/column, 과도한 문자열, status·version·draft injection을 거부한다.
- 외부 오류는 payload 없는 fixed code와 고정 HTTP status만 반환한다. publish block은 field별 내부 message를 그대로 반사하지 말고 고정 code와 안전한 미완료 field ID 목록 정도만 허용한다.
- publish 응답은 strict activation reload 뒤에만 `PUBLISHED`와 registry eligibility를 표시한다. response 생성 실패나 reload 실패 시 실행 registry에 노출하지 않는다.

## Phase 6-16 실행 registry 연결

- fixed flat·legacy synthetic entries와 legacy baseline을 그대로 보존한다. private published entry는 `schema: "private"` 같은 명시적 variant로만 추가한다. backend와 frontend registry parser를 함께 strict 갱신한다.
- registry snapshot은 server-owned fixed entries + strict valid activations로 결정적으로 정렬한다. caller action body가 registry entry나 label/schema를 직접 주입하지 못한다.
- private entry의 DB path는 strict registration loader가 확인한 `database.sqlite3`만 사용한다. 등록 source path나 UI file object를 재사용하지 않는다.
- catalog와 clarification은 동일한 activated database ID, strict Published Mapping, schema fingerprint, default ontology를 사용한다.
- private mapping은 기존 `SchemaAdaptiveSQLiteCompiler`, physical relation graph, `SchemaAdaptiveAstPolicy`, `SQLiteReadOnlyExecutor`를 사용한다. caller SQL/path/mapping 주입 금지. JOIN ambiguity·schema drift·mapping drift·invalid canonical row는 redacted fail closed한다.
- clarification의 deterministic provider 범위는 기존 `Course.name`·`Instructor.name` 그대로 유지한다. 새 모델·Ollama 호출을 추가하지 않는다.
- publish 전 ID, draft-only ID, tampered/stale activation은 `/databases`·catalog·clarification 모두에서 계속 미노출/실행 불가다.

## frontend 상태 전이

- Phase 6-18 upload 성공 요약 아래 `mapping 검토 시작`을 제공한다. strict INIT/GET 성공 뒤 field별 current decision과 candidates를 읽기 전용 metadata로 표시한다.
- reviewer 입력, candidate 승인, table/column remap+reason, publish button을 제공한다. 임의 JSON·path·SQL 입력 UI는 만들지 않는다.
- action 중 중복 submit 차단, modal close/unmount abort, operation generation으로 stale response 차단을 적용한다. 새 action은 이전 오류를 교체하되 성공한 server state를 추측해 optimistic publish하지 않는다.
- publish 성공 뒤 database registry를 다시 불러오고, strict registry 응답에 새 ID가 실제 포함된 경우에만 선택 radio를 표시한다. 자동 active 선택·sessionStorage 변경·course UI 초기화는 하지 않는다.
- publish 전에는 기존 upload summary와 review state가 있어도 radio/clarification link를 만들지 않는다.
- 오류는 고정 한국어 문구만 표시하고 backend body/detail/path/SQL을 반사하지 않는다.

## 테스트

backend 집중 테스트에서 최소 다음을 검증한다.

1. strict registration → init → 15 field approve/remap → publish → strict activation round-trip.
2. registration 직후·draft init·일부 승인·publish 실패까지 registry/catalog/clarification 미노출.
3. publish 후 registry에 정확히 한 번 노출, catalog canonical 6행과 clarification 기존 두 option 중 선택 결과 실행.
4. process restart rehydrate 후 같은 private entry와 결과 유지.
5. stale schema, registration/proposal/draft/published/activation 변조·추가/누락, ontology mismatch, 없는/중복 column, ambiguous/disconnected JOIN을 fail closed하고 registry에 숨김.
6. duplicate/unknown JSON key/action, forged status/version/draft/mapping/path/SQL 주입, concurrent same-ID mutation/publish, storage failure의 fixed redaction과 기존 state 보존.
7. fixed flat·legacy registry/catalog/clarification과 legacy baseline 회귀 없음.

frontend 집중 테스트에서 최소 다음을 검증한다.

1. review detail/action/publish strict parser와 fixed error redaction.
2. idle/loading/action/publish/success/error/retry, duplicate 차단, abort/stale 차단.
3. publish 전 radio·clarification link 미노출, 성공 후 registry refetch 결과에 있을 때만 표시.
4. publish 성공이 active DB/sessionStorage/course UI를 자동 변경하지 않음.
5. `schema: private` registry strict 수락과 unknown variant 거부.

OpenCode는 신규 및 직접 관련 Phase 6-16/17/18 mapping·registration·registry backend/frontend 테스트, 변경 frontend lint/build만 실행한다. 전체 pytest, compileall, `git diff --check`, 문서 갱신, commit, push는 하지 않는다.

## 변경 금지

- legacy system, 실제 CNU 자료, PostgreSQL, Ollama live 호출, 기존 합성 fixture 내용을 변경하거나 사용하지 않는다.
- 자동 approve/publish를 만들지 않는다. `AUTO_APPROVE_CANDIDATE`도 사용자 action 전에는 `PENDING`이다.
- 실제 학교 DB publish·질의에 필요한 L2를 완료 처리하지 않는다.
- 계획/ADR/change log/handoff는 구현 agent가 수정하지 않는다.

## 완료 보고

긴 diff와 전체 로그를 출력하지 않는다. 변경 파일, 핵심 계약, 실패 원인, 관련 테스트 결과, 남은 위험만 한국어로 간결히 보고한다. 모델별 사용량·통계는 조회·보고하지 않는다.

---

<a id="prompts-phase7-01-postgres-adapter-capabilities"></a>

## 원본: prompts/phase7-01-postgres-adapter-capabilities.md

# Phase 7-01 PostgreSQL adapter와 capability 계약 구현 지침

## 목적

기존 SQLite schema-adaptive 경계를 보존하면서, 시스템이 지원하는 DBMS와 QueryIR 기능을 명시하는 공통 capability 계약 및 PostgreSQL read-only schema inspector를 한 개의 완결된 합성 개발 단위로 구현한다.

이번 단계는 실제 CNU DB, 실제 PostgreSQL 서버, Ollama live 호출을 사용하지 않는다. PostgreSQL compiler/policy/executor, runtime 질의 API, React UI, connection profile 영속 저장은 후속 단계다.

## 고정 결정

- 초기 지원 DBMS는 `sqlite`, `postgres`뿐이다. Oracle/MySQL은 미지원으로 명시한다.
- 모델 출력의 장기 경계는 DB 독립적 QueryIR이다. raw SQL runtime fallback은 추가하지 않는다.
- DBMS별 adapter는 schema inspection, compiler, policy, executor capability를 명시해야 한다.
- SQLite의 기존 catalog JSON, schema fingerprint, mapping proposal, registration strict reload를 변경하지 않는다.
- PostgreSQL은 server-owned 설정과 단일 허용 schema만 검사한다. 기본 schema는 `public`이다.
- sample 수집은 기본 비활성화이며 이번 PostgreSQL adapter에서는 `include_samples=True`도 명시적으로 거부한다.
- 오류·반환값에 DSN, host, port, user, password, SQL 원문, exception 원문을 노출하지 않는다.

## 구현 범위

### 1. DBMS capability 계약

- versioned frozen Pydantic 계약을 추가한다.
- 최소 필드:
  - adapter/dialect: `sqlite` 또는 `postgres`
  - schema inspection 지원 여부
  - compiler/policy/executor 지원 여부
  - 지원 QueryIR 기능: select, filters, joins, limit
  - 미지원 기능: order_by, group_by, aggregate, subquery, cte, write
  - read-only only, parameterized only
  - max rows 및 timeout 상한
- SQLite 현재 구현 capability와 PostgreSQL Phase 7-01 capability를 각각 결정적으로 반환한다.
- PostgreSQL은 이번 단계에서 inspector만 `true`, compiler/policy/executor는 `false`다.
- 알 수 없는 adapter/dialect는 fallback하지 않고 고정 오류로 거부한다.

### 2. PostgreSQL schema adapter

- 기존 `DatabaseAdapter.inspect(database_id, include_samples=False)` protocol을 구현한다.
- constructor는 database ID별 server-owned connection config와 주입 가능한 connector를 받는다.
- production connector는 기존 설치 dependency인 `psycopg2`를 지연 import한다. import·연결 실패를 고정 catalog 오류로 변환한다.
- config는 최소 DSN과 허용 schema를 가지되 repr/str/오류에 DSN을 노출하지 않는다. database ID·schema 이름은 엄격히 검증한다.
- connect timeout과 statement timeout 상한을 강제하고, inspection transaction을 read-only로 설정한다.
- `pg_catalog`의 parameterized metadata query만 사용해 ordinary/partitioned table, column 순서·declared type·nullable·default·generated/identity, PK 순서, FK와 composite FK sequence를 수집한다.
- system schema, view/materialized view, foreign table, 다른 schema를 제외한다.
- table/column/constraint 결과는 결정적으로 정렬한다.
- PostgreSQL type을 기존 mapping aligner가 이해하는 논리 affinity로 보수적으로 변환한다. SQLite 공개 API와 기존 affinity serialization은 깨뜨리지 않는다.
- catalog dialect는 `postgres`이며 fingerprint는 sample/연결정보를 제외한 schema 구조에서 결정적으로 계산한다.
- cursor/connection은 성공·실패 모두 닫고 rollback/cleanup 실패 원문을 노출하지 않는다.

### 3. 안전·호환성

- legacy `backend/main.py`, `backend/db.py`, 기존 Ollama/SQL 생성 경로를 수정하지 않는다.
- 기존 SQLite adapter 동작·fingerprint·테스트를 그대로 통과시킨다.
- DSN을 public model, catalog, capability, exception message, 로그에 넣지 않는다.
- caller SQL, 임의 metadata query, 여러 schema, sample 값 수집을 허용하지 않는다.
- PostgreSQL server major version은 연결 metadata로 확인하되 이번 단계에서 지원 범위를 과장하지 않는다. synthetic fake connector 결과만 검증했음을 문서화한다.

## 테스트 범위

- capability exact contract, sqlite/postgres 값, unknown dialect fail-closed
- fake psycopg2-style connection/cursor로 PostgreSQL catalog round-trip
- 단일 schema 격리, system/view 제외, table/column 순서 결정성
- PK·composite FK sequence, generated/identity, type affinity
- parameterized metadata query와 read-only/timeout 설정
- unknown database ID, invalid config/schema, connector/import/query failure의 고정 오류
- cleanup 성공·실패, DSN/password/SQL/exception redaction
- `include_samples=True` 차단 및 query 0회
- 기존 SQLite catalog/fingerprint, mapping aligner, relation planning 회귀

## OpenCode 실행 제한

- 관련 파일과 테스트만 확인한다.
- 관련 backend 테스트만 실행한다.
- 전체 pytest, compileall, 문서 계획/ADR/handoff 갱신, commit, push를 하지 않는다.
- 실제 PostgreSQL/Ollama/CNU 데이터에 연결하지 않는다.
- 마지막 보고는 변경 파일, 핵심 계약, 실패 원인, 관련 테스트 요약, 남은 위험만 한국어로 간결히 작성한다.

---

<a id="prompts-phase7-02-postgres-query-runtime"></a>

## 원본: prompts/phase7-02-postgres-query-runtime.md

# Phase 7-02 PostgreSQL QueryIR 실행 경계 구현 지침

## 목적

Phase 7-01의 PostgreSQL schema inspector와 `dbms-capability-v1` 위에, 기존 DB 독립 QueryIR subset을 PostgreSQL용 parameterized SQL로 컴파일하고 AST allowlist와 read-only executor를 통과시키는 합성 runtime 단위를 구현한다.

이번 단계는 fake psycopg2-style connector로만 검증한다. 실제 PostgreSQL 서버, 실제 CNU 자료, Ollama live 호출, React/HTTP registry 연결은 사용하거나 변경하지 않는다.

## 고정 결정

- 모델 출력은 기존 QueryIR만 허용한다. raw SQL runtime fallback을 추가하지 않는다.
- 지원 subset은 `select`, `filters`, `joins`, `limit`이며 `order_by`, `group_by`, aggregate, subquery, CTE, write는 계속 미지원이다.
- PostgreSQL SQL은 compiler-owned이며 식별자는 double quote escaping, 값과 LIMIT은 psycopg2 `%s` placeholder로만 바인딩한다.
- Phase 7-02 완료 뒤 PostgreSQL capability의 compiler/policy/executor를 `true`로 올리되, 이는 합성 component 준비 상태다. 실제 server 연결이나 UI registry 노출을 뜻하지 않는다.
- SQLite compiler/policy/executor의 SQL, placeholder, fingerprint, 공개 API와 기존 테스트를 깨뜨리지 않는다.
- 모든 PostgreSQL 실행 오류는 고정 code/message로 redaction한다. DSN, host, user, password, SQL, parameter, DB driver exception 원문을 반환·로그하지 않는다.

## 구현 범위

### 1. PostgreSQL compiler

- 기존 `JoinCompilationRequest`, Published Mapping, physical relation graph, deterministic alias/Join plan 계약을 그대로 사용한다.
- PostgreSQL 전용 join-capable compiler와 기존 harness protocol adapter를 추가한다.
- projection alias, table/column identifier는 기존 안전한 double-quote escaping을 재사용한다.
- filter 순서와 LIMIT 순서를 보존하고 placeholder는 정확히 `%s`를 사용한다.
- `contains`는 기존처럼 값만 `%value%`로 바인딩하고 SQL에 literal을 삽입하지 않는다.
- ambiguous/invalid/self join, mapping drift는 기존 고정 Harness code/stage로 fail closed한다.

### 2. PostgreSQL AST policy

- `sqlglot`의 `postgres` dialect로 compiler output을 parse한다.
- 한 개 SELECT, 정확한 projection/table/alias/JOIN predicate/filter/parameterized LIMIT만 허용하는 기존 allowlist 의미를 보존한다.
- `%s`가 sqlglot에서 anonymous placeholder로 정규화되는 경계를 명시적으로 테스트한다.
- comment, multi-statement, write, CTE/subquery/function, extra table/column, literal filter/LIMIT, placeholder/parameter drift를 거부한다.
- SQLite policy 기본 dialect와 회귀를 보존한다.

### 3. PostgreSQL read-only executor

- database ID별 Phase 7-01 `PostgresConnectionConfig`와 주입 가능한 connector를 받는다. production connector는 `psycopg2`를 지연 import한다.
- database ID/config 타입, timeout·max rows 상한을 엄격히 검증한다.
- 연결 metadata에서 PostgreSQL 12 이상을 확인한다.
- transaction을 read-only로 설정하고 statement timeout을 서버 소유 config에서 적용한 뒤 compiler-owned SQL을 tuple parameters로 정확히 한 번 실행한다.
- cursor description 순서를 column 이름으로 사용하고 `max_rows + 1`만 읽어 초과 시 `RESULT_TOO_LARGE`로 거부한다. tuple row와 mapping row를 결정적으로 dict로 변환한다.
- SQLSTATE `57014` 또는 동등한 query-cancel 예외만 `EXECUTION_TIMEOUT`, 나머지는 `EXECUTION_ERROR`로 고정 redaction한다.
- 성공·실패 모두 cursor close, rollback, connection close를 수행하고 cleanup 예외 원문을 노출하지 않는다.
- 미등록 DB는 caller 입력을 반사하지 않는 고정 `DATABASE_NOT_FOUND` 오류로 닫는다.

### 4. 통합·capability

- `ExecutionHarness`가 SQLite 구체 타입 대신 기존 compiler/policy/executor protocol을 정확히 표현하도록 최소 일반화하되 기존 호출과 결과 계약을 바꾸지 않는다.
- PostgreSQL compiler + postgres AST policy + executor를 기존 `ExecutionHarness`에 주입한 fake end-to-end 성공을 검증한다.
- Phase 7-02 완료 상태의 PostgreSQL capability에서 inspector/compiler/policy/executor가 모두 `true`인지 검증한다.
- 이 단계에서 synthetic active DB registry, registration/upload/publish, legacy backend, React를 변경하지 않는다.

## 테스트 범위

- PostgreSQL single/multi-table compiler exact SQL, `%s` count/order, quoting, contains, composite/reverse JOIN
- PostgreSQL AST 정상 SQL과 comment/multi/write/CTE/subquery/function/extra table·column/literal/placeholder drift 거부
- fake connector executor 성공, tuple/mapping row, zero row, exact query/params 1회
- read-only transaction, timeout, server version, max rows, missing DB
- connector/import/cursor/execute/fetch/cleanup/query cancel 실패의 고정 code/stage 및 secret/SQL/parameter redaction
- 기존 harness fake end-to-end PostgreSQL 성공·result schema mismatch
- SQLite join compiler, AST policy, executor, schema-adaptive harness와 Phase 7-01 capability/inspector 회귀

## OpenCode 실행 제한

- 관련 파일과 테스트만 확인한다.
- 관련 backend 테스트만 실행한다.
- 전체 pytest, compileall, 계획/ADR/handoff 갱신, commit, push를 하지 않는다.
- 실제 PostgreSQL/Ollama/CNU 데이터에 연결하지 않는다.
- 긴 diff·전체 로그를 출력하지 않는다. 마지막 보고는 변경 파일, 핵심 계약, 실패 원인, 테스트 요약, 남은 위험만 한국어로 간결히 작성한다.

---

<a id="prompts-phase7-03-synthetic-postgres-live-smoke"></a>

## 원본: prompts/phase7-03-synthetic-postgres-live-smoke.md

# Phase 7-03 합성 PostgreSQL 실제 연결 smoke 구현 지침

## 목적

Phase 7-01/02의 PostgreSQL inspector→mapping proposal→physical relation graph→QueryIR compiler→AST policy→read-only executor를 완전 합성 schema/data가 들어 있는 실제 로컬 PostgreSQL에서 한 번 연결 검증하는 독립 smoke를 구현한다.

이번 단계는 backend/CLI 통합 검증만 한다. React·HTTP DB registry·Ollama·legacy 경로는 변경하지 않는다. 실제 CNU 자료나 기존 PostgreSQL database/schema를 읽거나 수정하지 않는다.

## 고정 경계

- 실행에는 명시적 `--live-synthetic`가 필요하다. 기본 실행은 provider/DB 연결·DDL·파일 쓰기 0회로 종료한다.
- DSN은 CLI 인자나 파일이 아니라 server-owned 환경변수 `SCHEMA_ADAPTIVE_SYNTHETIC_POSTGRES_DSN`에서만 읽는다.
- DSN은 `postgresql://` URI, loopback host(`127.0.0.1`, `localhost`, `::1`), database 이름 `schema_adaptive_synthetic`만 허용한다. multi-host, Unix socket, 다른 database, fragment, 알 수 없는 query option은 연결 전에 거부한다.
- 비밀번호·DSN·host/port·생성 schema 이름·SQL·질문·row/sample·driver exception 원문을 stdout/stderr/result/error에 출력하거나 저장하지 않는다.
- 기존 database나 schema를 재사용하지 않는다. 프로세스가 생성한 고유 `schema_adaptive_smoke_<hex>` schema만 사용하고 `finally`에서 그 정확한 schema만 제거한다.
- raw SQL model fallback, 실제 Ollama 호출, actual CNU data, browser/UI, registry 활성화는 없다.

## 구현 범위

### 1. 합성 PostgreSQL fixture

- 기존 `DEFAULT_SEED` 6개 강좌와 normalized SQLite fixture의 논리 구조를 재사용한다.
- PostgreSQL용 departments/instructors/courses/course_offerings/schedules DDL과 parameterized insert를 제공한다.
- identifier는 `psycopg2.sql.Identifier`, 값은 `%s` binding으로만 구성한다. schema 이름을 문자열 보간하지 않는다.
- 생성 schema는 프로세스 내부에서 cryptographic random suffix로 만들고 엄격한 prefix/pattern을 재검증한다.
- bootstrap connection은 synthetic 전용 database에서 schema 생성·seed·cleanup에만 사용한다. 각 단계 실패는 고정 오류로 redaction한다.
- fixture setup 뒤 commit하고, cleanup은 성공·실패 모두 별도 안전 경계에서 `DROP SCHEMA <exact generated> CASCADE` 후 commit한다. 생성 전 실패에는 drop하지 않는다.

### 2. live smoke orchestration

- 작은 module CLI(권장: `python -m backend.adaptive.harness.postgres_live_smoke`)를 추가한다.
- preflight가 flag·환경변수·URI·loopback·database·query option을 모두 검증한 뒤에만 psycopg2를 지연 import하고 첫 연결을 수행한다.
- 생성 schema를 Phase 7-01 `PostgresConnectionConfig`에 server-owned로 넣어 `PostgresAdapter.inspect()`를 실행한다.
- catalog가 `postgres`, sample 0개, 예상 table/PK/FK 구조, PostgreSQL 12+를 만족하는지 확인한다.
- `RuleBasedSchemaAligner`와 default ontology로 proposal을 생성하고, smoke에 필요한 semantic field의 상위 후보가 예상 physical field인지 확인한다. 자동 publish로 과장하지 않는다.
- 확인된 합성 expected mapping으로 메모리 `PublishedMappingProfile`을 만들고 catalog fingerprint와 ontology version을 결합한다.
- physical graph, `SchemaAdaptivePostgresCompiler`, `SchemaAdaptivePostgresPolicy`, `PostgresReadOnlyExecutor`, 기존 `ExecutionHarness`를 연결한다.
- `FakeModel`의 고정 QueryIR로 최소 두 query를 실행한다: multi-table join/filter 성공과 0-row 성공. 결과는 `DEFAULT_SEED`의 기대값과 내부 비교한다.
- 성공 출력은 version/status, table·mapping·query count, schema fingerprint 정도의 안전한 JSON 한 건만 허용한다. row/question/SQL/schema/DSN은 출력하지 않는다.
- 실패 출력은 고정 code만 사용하고 exit code는 0/2처럼 결정적으로 구분한다. traceback과 원문 예외를 출력하지 않는다.

### 3. 검증 가능 구조

- DSN/preflight, fixture setup/cleanup, orchestration은 connector/clock/token 또는 필요한 최소 dependency를 주입 가능하게 해 unit test가 실제 network 없이 경계를 검증할 수 있게 한다.
- 실제 live test는 명시적 환경변수와 flag가 있을 때만 실행하고 기본 pytest에서는 skip하거나 network 0회 unit test만 수행한다.
- live smoke 중 생성한 schema가 성공·중간 실패 뒤 남지 않는지 실제 PostgreSQL에서 메인 Codex가 별도 확인할 수 있게 한다. 단, 공개 출력에는 schema 이름을 넣지 않는다.

## 테스트 범위

- no-flag/no-env/invalid scheme·host·database·multi-host·query option의 연결 0회 거부
- DSN·password·host·port·schema·SQL·row·driver error redaction
- generated schema prefix/pattern 및 identifier composition, parameterized inserts
- setup 성공/부분 실패/inspection/mapping/query 실패의 cleanup 정확히 1회, 생성 전 실패 cleanup 0회
- catalog expected tables, samples 0, PK/FK/composite order
- mapping top candidate 검증과 임의 auto-publish 없음
- 두 QueryIR의 compiler→policy→executor 결과 내부 비교 및 안전한 summary
- 기존 Phase 7-01/02 fake tests와 SQLite compiler/policy/executor 회귀

## OpenCode 실행 제한

- 관련 파일과 테스트만 확인한다.
- unit/fake 관련 backend 테스트만 실행한다.
- 실제 PostgreSQL container/server를 시작하거나 연결하지 않는다. live smoke는 메인 Codex가 구현 검토 뒤 수행한다.
- 전체 pytest, compileall, 계획/ADR/handoff 갱신, commit, push를 하지 않는다.
- legacy, React, HTTP registry, Ollama를 변경하지 않는다.
- 긴 diff·전체 로그를 출력하지 않는다. 마지막 보고는 변경 파일, 핵심 계약, 실패 원인, 테스트 요약, 남은 위험만 한국어로 간결히 작성한다.

---

<a id="prompts-phase7-04-server-owned-postgres-runtime-registry"></a>

## 원본: prompts/phase7-04-server-owned-postgres-runtime-registry.md

# Phase 7-04 구현 지침: server-owned PostgreSQL runtime registry 연결

## 목표

Phase 7-03에서 실제 합성 PostgreSQL로 검증한 inspector·mapping·compiler·policy·executor를 기존 adaptive prototype의 database registry, course catalog, clarification 실행 경로에 연결한다. 이번 단계는 backend 조립만 다루며 UI, Ollama, 실제 학교 DB는 사용하지 않는다.

## 범위

- Python server code만 만들 수 있는 strict immutable PostgreSQL profile 계약을 추가한다.
- profile은 `database_id`, 안전한 `label`, `PostgresConnectionConfig`, 명시적으로 승인된 `PublishedMappingProfile`을 가진다.
- `create_synthetic_clarification_prototype()`가 선택적 profile 목록과 테스트용 adapter/executor dependency를 받을 수 있게 한다.
- 기본 인자에서는 PostgreSQL profile이 0개이며 import·app 생성 중 PostgreSQL 연결도 0회다.
- profile이 있으면 startup에서 PostgreSQL catalog를 sample 없이 검사하고 다음을 확인한다.
  - profile·catalog·published mapping의 database ID 일치
  - catalog fingerprint와 mapping fingerprint 일치
  - ontology version 일치
  - course catalog의 canonical 15개 field가 모두 정확히 한 physical column에 mapping됨
  - 모든 mapped table/column이 catalog에 존재
  - physical relation graph 생성 가능
- 검증된 profile만 기존 `/api/adaptive/v1/databases`에 `schema: "postgres"`로 나타난다.
- `/api/adaptive/v1/courses`는 database ID에 따라 기존 SQLite 또는 PostgreSQL compiler·AST policy·read-only executor를 선택한다.
- clarification start/submit도 같은 server-owned mapping과 database별 compiler/policy/executor를 사용한다.
- PostgreSQL context load, catalog query, clarification execute 직전에 catalog를 다시 inspect해 fingerprint drift를 확인한다. drift·연결 실패는 기존 fixed redacted unavailable 오류로 닫는다.

## 보안·격리 계약

- DSN, host, port, user, password, schema, physical table/column, SQL, bound parameter, driver exception text를 HTTP 응답·예외·repr·log·app public state에 노출하지 않는다.
- HTTP request는 PostgreSQL profile, DSN, schema, mapping, catalog, compiler, policy, executor를 주입할 수 없다.
- raw SQL fallback을 추가하지 않는다.
- profile ID는 기존 fixed SQLite ID와 중복될 수 없다. 전체 fixed registry 상한과 deterministic ordering을 유지한다.
- private SQLite activation 경로와 legacy app을 변경하지 않는다.
- PostgreSQL profile을 제공하지 않은 기존 SQLite registry/catalog/clarification 동작과 응답 shape는 그대로 유지한다.
- fail-open, 자동 mapping proposal publish, client 제공 mapping, 실제 CNU 데이터 접근을 금지한다.

## 권장 구조

- PostgreSQL profile validation과 runtime context를 새 작은 backend module로 분리한다.
- compiler/policy/executor는 mapping/database ID로 라우팅하는 얇은 server-owned router를 사용한다.
- 기존 `SyntheticDatabaseRegistry`는 `postgres` entry를 허용하되 credential 객체를 보관하거나 직렬화하지 않는다.
- 테스트 injection은 connector 자체보다 validated adapter/executor 또는 factory 수준으로 제한하고 production 기본값은 기존 `PostgresAdapter`·`PostgresReadOnlyExecutor`를 사용한다.

## 테스트

관련 테스트만 실행한다.

- 기본 앱 생성과 기존 SQLite 경로에서 PostgreSQL dependency 호출 0회
- valid synthetic PostgreSQL profile의 registry listing
- catalog의 PostgreSQL compiler/policy/executor routing과 canonical DTO
- clarification start/submit의 PostgreSQL routing
- startup ID/fingerprint/ontology/mapping/table/column/graph 불일치 거부
- 요청 전 schema drift와 inspector/executor 실패의 fail-closed 처리
- duplicate ID, unknown ID, request profile/DSN/schema/mapping 주입 거부
- 응답·오류·repr에 credential, schema, physical identifier, SQL, driver text 비노출
- 기존 Phase 6-16/19 SQLite registry/catalog/clarification 회귀

실제 PostgreSQL server, Docker, Ollama, frontend test, 전체 pytest, compileall, git commit은 실행하지 않는다.

## 완료 보고

긴 diff와 전체 로그를 출력하지 않는다. 변경 파일, 핵심 계약, 관련 테스트 결과, 실패 원인, 남은 위험만 한국어로 간결히 보고한다. 문서의 계획·ADR·handoff는 수정하지 않는다.

---

<a id="prompts-phase7-05-private-postgres-startup"></a>

## 원본: prompts/phase7-05-private-postgres-startup.md

# Phase 7-05 구현 지침: private PostgreSQL startup profile

## 목표

Phase 7-04의 server-owned PostgreSQL runtime profile을 loopback synthetic prototype CLI 시작 경계에 연결한다. Git 비추적 private metadata와 별도 환경변수의 DSN을 엄격히 읽고, 이미 승인·publish된 immutable mapping만 runtime registry에 주입한다.

이번 단계는 startup configuration 한 흐름만 완결한다. frontend parser/UI, SQLite upload 변경, mapping 편집·승인·publish, Ollama 연결, 실제 CNU 데이터, 외부 PostgreSQL, legacy 시스템은 변경하거나 사용하지 않는다.

## 구현 범위

1. `backend/adaptive/api` 아래에 private PostgreSQL profile loader를 추가한다.
2. CLI에 다음 server/operator 전용 옵션을 추가한다.
   - `--repository-root PATH`: private 파일 검증 기준 저장소. 기본값은 현재 작업 디렉터리.
   - 반복 가능한 `--postgres-profile-id DATABASE_ID`: 로드할 profile ID.
3. profile ID마다 다음 고정 경로만 읽는다.
   - `.schema-adaptive/postgres-profiles/<database_id>/profile.json`
   - `.schema-adaptive/postgres-profiles/<database_id>/mapping.json`
4. 성공한 profile들을 `create_synthetic_clarification_prototype(postgres_profiles=...)`에 전달한다.
5. profile 옵션이 없으면 환경변수, private 파일, PostgreSQL driver/connection을 읽거나 호출하지 않고 기존 동작을 그대로 유지한다.

## private storage 계약

- `recording_bundle_storage.preflight_private_snapshot_path` / `read_private_snapshot_bytes`의 기존 repository/private guard를 재사용한다. 별도 약한 검사기를 만들지 않는다.
- 두 JSON 모두 저장소 내부, Git ignored, Git untracked여야 한다. symlink/junction/reparse/hardlink/redirect/escape를 기존 guard가 거부하게 유지한다.
- profile ID는 기존 PostgreSQL database ID 형식만 허용한다. ID로 고정 경로를 조합하므로 absolute path, traversal, separator를 허용하지 않는다.
- 각 파일은 작고 제한된 크기여야 한다. 기존 read helper에 크기 제한이 없으면 loader에서 읽은 직후 명시적 상한(각 256 KiB 이하)을 적용한다. 초과 파일은 내용 parsing 전 거부한다.
- UTF-8만 허용하고 duplicate JSON key를 거부한다.
- 파일이나 DSN을 수정·복사·생성하지 않는다.

## `profile.json` strict 계약

정확히 다음 필드만 허용한다. 누락, 추가 필드, 타입 coercion을 거부한다.

```json
{
  "version": "postgres-runtime-profile-v1",
  "database_id": "synthetic-postgres",
  "label": "Synthetic PostgreSQL",
  "schema": "public",
  "dsn_env": "SCHEMA_ADAPTIVE_POSTGRES_DSN_SYNTHETIC",
  "mapping_sha256": "64자리 소문자 hex"
}
```

- `database_id`는 CLI ID 및 mapping의 `database_id`와 모두 같아야 한다.
- `label`은 기존 `PostgresRuntimeProfile`의 안전한 label 형식을 만족해야 한다.
- `schema`는 기존 `PostgresConnectionConfig` schema 형식을 만족해야 한다.
- `dsn_env`는 `SCHEMA_ADAPTIVE_POSTGRES_DSN_` prefix와 대문자 영숫자/underscore만 허용하고 길이를 제한한다. 환경변수 이름이나 값은 HTTP/app public state/log/error/repr/stdout/stderr에 노출하지 않는다.
- `mapping_sha256`는 원본 `mapping.json` bytes의 SHA-256과 일치해야 한다.
- 동일 profile ID, DSN 환경변수 이름, mapping 대상의 중복은 startup 전에 거부한다.
- 입력 ID 순서와 무관하게 결과 순서는 database ID 기준으로 결정적이어야 한다.

## `mapping.json` strict 계약

- `PublishedMappingProfile`의 정확한 JSON shape만 허용한다. 모델의 기본 coercion/extra-ignore에 기대지 말고 recursive strict transport validation 또는 동등한 검증으로 extra/missing/wrong scalar type/duplicate key를 거부한다.
- `status`는 명시적으로 `PUBLISHED`여야 한다. 누락된 status의 모델 default를 허용하지 않는다.
- `database_id`, 양의 정수 `version`/`ontology_version`, fingerprint, non-empty `published_by`, timezone-aware ISO datetime, fields object를 검증한다.
- 각 field key와 value shape를 엄격히 검증하고 빈 semantic field/table/column/reviewer를 거부한다. `reason`은 string 또는 null만 허용한다.
- hash 확인 뒤 strict parse하며, 최종적으로 `PublishedMappingProfile`을 생성한다.
- Phase 7-04 runtime validation과 request-time schema fingerprint drift check는 그대로 유지한다. startup profile loader가 이를 우회하지 않는다.

## DSN 및 오류 redaction

- DSN은 `profile.json`에 절대 저장하지 않는다. 승인된 `dsn_env`가 지목하는 process environment에서만 읽는다.
- missing/empty DSN을 거부한다. DSN 문자열, 비밀번호, host, port, user, schema, 실제 path, mapping physical identifier, 원본 parser/driver 예외를 출력하지 않는다.
- loader public failure는 고정된 단일 메시지의 전용 예외로 정규화한다.
- CLI startup 실패도 기존 `synthetic prototype invalid configuration` 같은 고정 문구만 stderr에 내고 non-zero로 종료한다. 사용자 입력값을 argparse 오류에 반사하지 않는다.
- object `repr`/`str`은 redacted 상태를 유지한다.

## CLI 경계

- global module-level `app = create_synthetic_clarification_prototype()`에는 private profile을 자동 주입하지 않는다. ASGI import는 현재와 같은 synthetic-only 기본값이다.
- CLI `main`에서만 profile을 로드한다. 모든 profile strict load가 성공한 뒤 app을 조립하고 uvicorn을 시작한다.
- startup validation 실패 시 uvicorn은 시작하지 않는다.
- host는 계속 `127.0.0.1` 고정, reload/access log는 계속 비활성화한다.
- cleanup은 성공 startup과 이후 종료에서 기존처럼 한 번 실행된다. app 생성 전 실패에는 존재하지 않는 cleanup을 호출하지 않는다.

## 테스트

관련 테스트만 실행한다. 실제 PostgreSQL/Ollama/network를 사용하지 않는다.

- profile 옵션 없는 CLI가 loader/env/PostgreSQL을 건드리지 않고 기존 loopback 계약을 유지.
- 정상 private ignored/untracked metadata + mapping + env로 `PostgresRuntimeProfile` 생성, CLI factory 전달.
- 여러 ID의 결정적 정렬 및 duplicate ID/env 거부.
- tracked/not-ignored/outside-repo/symlink/reparse/hardlink/path escape 거부(기존 guard를 과도하게 중복 테스트하지 말고 연결 계약 중심).
- oversize, invalid UTF-8, duplicate JSON key, extra/missing/wrong type, 잘못된 version/ID/schema/env 이름/hash/status/time/field shape 거부.
- missing/empty env 거부.
- profile과 mapping ID 불일치 거부.
- startup 실패 시 app factory/uvicorn 미호출.
- DSN, secret marker, path, schema, physical identifier, parser 원문이 exception/repr/stdout/stderr/log에 없음.
- 기존 Phase 7-04 registry/clarification routing과 CLI/import isolation 관련 테스트 회귀 없음.

## 금지

- 실제 CNU 데이터, 실제 운영 PostgreSQL, Ollama live 호출, Docker 추가 금지.
- frontend/UI와 upload/review/publish flow 변경 금지.
- legacy 시스템 변경 금지.
- 계획/ADR/handoff/변경 이력 수정, 전체 pytest, compileall, commit, push 금지. 이는 메인 Codex가 담당한다.

## 완료 보고

긴 diff와 전체 로그를 출력하지 않는다. 한국어로 변경 파일, 핵심 계약, 관련 테스트 결과, 실패 원인, 남은 위험만 간결히 보고한다.

---

<a id="prompts-phase7-06-postgres-frontend-selection"></a>

## 원본: prompts/phase7-06-postgres-frontend-selection.md

# Phase 7-06 구현 지침: PostgreSQL frontend 선택 연결

## 목표

Phase 7-05 CLI startup에서 strict load·검증되어 Phase 7-04 registry에 노출된 PostgreSQL profile을 메인 `DB 변경`과 `/adaptive-prototype` 흐름에서 선택할 수 있게 한다. frontend는 registry의 안전한 `database_id`·`label`·`schema=postgres`만 신뢰하며, 선택한 동일 database ID를 기존 course catalog와 clarification start에 전달한다.

이번 단계는 frontend registry/selection 계약만 완결한다. backend profile loader·runtime, private artifact, DSN, mapping, SQL compiler/policy/executor, SQLite upload/review/publish, Ollama provider, 실제 CNU/학교 DB, legacy API는 변경하거나 사용하지 않는다.

## 구현 범위

1. `src/adaptive-prototype/databases.ts`의 strict registry 계약에 `postgres` schema variant를 추가한다.
2. 기존 `RegistryDatabase` union과 parser가 `flat | legacy | private | postgres`만 허용하도록 한다. 알 수 없는 schema는 계속 전체 registry fail-closed로 거부한다.
3. PostgreSQL label은 Phase 7-04 backend의 안전한 ASCII label 계약과 같은 제한을 적용한다. `private`의 한국어 가능 label 예외를 PostgreSQL에 확대하지 않는다.
4. 메인 `DB 변경` modal은 server-owned registry 항목만 렌더링한다. PostgreSQL entry는 다른 실행 가능 DB처럼 radio 선택·명시적 `적용` 후 활성화한다.
5. 선택 적용 시 기존 Phase 6-16 상태 초기화 계약을 그대로 유지한다: 이전 catalog request abort/stale 무효화, 선택 강의·경고·hover·modal/course 상태 초기화, sessionStorage에는 검증된 database ID만 저장한다.
6. 활성 PostgreSQL ID를 course catalog query와 `/adaptive-prototype?database=...` 링크에 전달한다. prototype page는 registry를 다시 strict load하고 URL ID가 실제 registry에 있을 때만 같은 ID로 clarification start를 보낸다.
7. registry refresh에서 저장된 PostgreSQL ID가 사라지거나 registry가 invalid/unavailable이면 legacy 기준선으로 fail closed한다.

## 보안·노출 계약

- browser가 허용하거나 표시할 PostgreSQL 정보는 registry의 exact `database_id`, 안전한 `label`, literal `schema=postgres`뿐이다.
- DSN, DSN 환경변수 이름, host, port, username, password, 물리 schema/table/column, mapping, fingerprint, SQL, driver 오류를 새 response·state·storage·URL·DOM·log에 추가하지 않는다.
- frontend가 PostgreSQL profile을 생성·수정·등록하거나 임의 DSN/path를 전송하는 기능을 추가하지 않는다.
- request body/query에는 기존 검증된 database ID만 포함한다. raw SQL fallback이나 dialect 선택 입력을 추가하지 않는다.
- server error body/detail을 반사하지 않는 기존 redaction을 유지한다.

## UI 계약

- 기존 registry 순서와 radio rendering을 보존한다. frontend 하드코딩 PostgreSQL entry를 만들지 않는다.
- 사용자는 registry에 실제 존재하는 PostgreSQL entry만 선택할 수 있다.
- `적용` 전에는 active DB, sessionStorage, catalog, clarification link가 바뀌지 않는다.
- 적용 후 current DB label과 catalog/clarification link는 동일 PostgreSQL ID를 사용한다.
- 별도 upload/review UI나 credential 입력란을 만들지 않는다.
- 필요하면 schema 표시를 사람이 읽기 쉬운 안전한 label로 바꿀 수 있지만, raw server metadata를 추가 표시하지 않는다.

## 테스트

관련 frontend 테스트만 실행한다. 실제 PostgreSQL/Ollama/network/browser를 사용하지 않는다.

- strict registry parser가 정상 `postgres` entry를 수락한다.
- extra key, unknown schema, invalid ID, unsafe/Unicode/control/blank/too-long PostgreSQL label, duplicate ID를 거부한다.
- `private` label의 기존 한국어 허용과 flat/legacy 계약은 회귀하지 않는다.
- 저장된 PostgreSQL ID가 현재 registry에 있을 때 복원되고, 사라지면 legacy로 fallback한다.
- write/read sessionStorage가 검증된 PostgreSQL ID만 보존한다.
- catalog client가 PostgreSQL ID를 query parameter로 보내고 echoed ID 불일치를 거부한다.
- clarification client가 동일 PostgreSQL ID를 body에 보낸다.
- prototype URL parser와 registry membership 경계가 invalid/unknown ID를 실행에 사용하지 않는다.
- MainPage가 registry 기반 radio rendering, 명시적 적용, 상태 초기화, 동일 ID link를 유지하는지 필요한 최소 회귀를 추가한다.
- 관련 Node tests, 변경 파일 ESLint, frontend build를 실행한다.

## 금지

- backend Python 코드와 API response shape 변경 금지.
- private profile/mapping/DSN 파일 생성·수정·읽기 금지.
- 실제 PostgreSQL/CNU 데이터/Ollama/network/browser 호출 금지.
- legacy 시스템, SQLite upload/review/publish 흐름 변경 금지.
- 계획/ADR/handoff/사람 체크포인트 수정, 전체 Python pytest, compileall, commit, push 금지. 이는 메인 Codex가 담당한다.

## 완료 보고

긴 diff와 전체 로그를 출력하지 않는다. 한국어로 변경 파일, 핵심 계약, 관련 테스트 결과, 실패 원인, 남은 위험만 간결히 보고한다.

---

<a id="prompts-phase7-07-ollama-constraint-provider"></a>

## 원본: prompts/phase7-07-ollama-constraint-provider.md

# Phase 7-07 구현 지침: Ollama constraint-aware provider 합성 통합

## 목표

기존 `OllamaQueryModel`을 adaptive clarification의 `ConstraintAwareProvider` 경계에 연결한다. 사용자가 선택한 semantic constraint를 server-owned bridge가 검증하고, Ollama prompt에 전달되는 ontology/mapping semantic view를 해당 entity 또는 field로 제한한 뒤 DB 독립 QueryIR을 정확히 한 번 생성한다. 생성된 QueryIR은 기존 Phase 6-09/10 validation·constraint compliance·compiler·policy·read-only executor 경로를 그대로 통과한다.

이번 단계는 injectable synthetic transport로 통합 계약만 완결한다. CLI Ollama 설정, 기본 provider 전환, live Ollama 호출, raw SQL fallback, model 재학습, 실제 PostgreSQL/CNU 자료, frontend, legacy 시스템은 변경하거나 사용하지 않는다.

## 구현 범위

1. `backend/adaptive/harness`에 `OllamaQueryModel` 전용 constraint-aware bridge를 추가한다.
2. bridge는 네 인자 `generate_ir(request, ontology, mapping, constraint)`를 제공하고 내부 `OllamaQueryModel.generate_ir(request, restricted_ontology, restricted_mapping)`을 최대 1회 호출한다.
3. `create_synthetic_clarification_prototype`에 명시적 server-owned Ollama model 주입 인자를 추가한다.
   - 인자가 없으면 기존 `_FixedConstraintProvider`를 그대로 사용한다.
   - 정확한 `OllamaQueryModel` 계열만 bridge로 감싼다. 임의 HTTP/request 입력으로 provider를 고를 수 없다.
   - global module-level `app`과 기본 CLI는 model을 주입하지 않아 Ollama import 외 network 호출 0회를 유지한다.
4. 성공적으로 주입된 bridge를 기존 `create_clarification_http_app` provider 자리에만 연결한다. compiler/policy/executor/router/registry/catalog는 변경하지 않는다.

## bridge strict 계약

- public 입력 네 개를 plain JSON round-trip으로 strict 재검증한다. forged/constructed Pydantic instance, extra field, coercion, wrong type을 fail closed한다.
- mapping은 `PUBLISHED`이고 request와 같은 database ID여야 한다.
- constraint version과 request ID hash, database ID hash, ontology fingerprint, mapping fingerprint가 현재 입력과 정확히 일치해야 한다.
- `selected_semantic_ref`는 ontology와 mapping 양쪽에 존재하는 entity 또는 field만 허용한다.
- field constraint라면 restricted ontology/mapping에는 해당 field와 그 entity만 둔다.
- entity constraint라면 해당 entity와 `Entity.` prefix의 mapped fields만 둔다. field가 하나도 없으면 호출 전 거부한다.
- semantic view는 결정적 순서/값을 가지며 원본 request/ontology/mapping/constraint를 변경하지 않는다. model에는 방어적 deep copy만 전달한다.
- model 호출은 검증 성공 뒤 정확히 1회, retry/model substitution/repair/raw SQL fallback 없이 수행한다.
- model output은 가공·보정하지 않고 기존 `run_constrained_generation`이 QueryIR 구조·semantic·constraint compliance를 검증하게 한다.

## prompt·정보 비노출 계약

- 기존 `OllamaQueryModel` prompt builder가 physical `MappingProfile`을 직렬화하지 않는 계약을 유지한다.
- synthetic transport가 받은 prompt에는 선택 범위의 canonical semantic entity/field와 operator/type만 있어야 한다.
- DSN, host, port, username, password, physical schema/table/column, mapping fingerprint, request/database fingerprint, option ID, session token, SQL을 prompt·public exception·repr·HTTP response·log에 노출하지 않는다.
- question은 기존 `OllamaQueryModel` 계약대로 prompt에 포함될 수 있으나 오류·repr·HTTP response/log에 반사하지 않는다.
- bridge public failure는 고정 code/message의 전용 오류로 정규화한다. underlying transport/model/parser 예외 원문을 보존하거나 출력하지 않는다.
- bridge와 app state의 repr/str에 model tag/base URL/transport/question/credential이 나타나지 않게 한다.

## DBMS 독립성

- bridge는 SQLite/PostgreSQL dialect를 알지 못하며 SQL을 받거나 반환하지 않는다.
- 동일 semantic ontology/mapping/constraint라면 database ID가 달라도 같은 제한 semantic prompt shape를 만든다.
- 이후 기존 database ID router가 SQLite 또는 PostgreSQL compiler/policy/executor를 선택한다. bridge가 router를 우회하지 않는다.
- PostgreSQL profile의 startup validation과 request-time schema drift 재검사를 약화하지 않는다.

## factory·HTTP 상태 전이

- app startup은 model transport를 호출하지 않는다.
- clarification `start`는 기존처럼 ambiguity pending만 만들며 model transport 호출 0회다.
- 정확한 option `submit`에서만 model transport 1회 후 기존 실행 경로를 탄다.
- cancel, invalid/expired/reused token, context drift, invalid selection은 model 호출 0회다.
- provider timeout/error/empty/non-JSON/invalid QueryIR/constraint violation은 기존 fixed HTTP error taxonomy로 redacted 변환하며 session은 기존 one-shot 소비 계약을 유지한다.
- SQLite와 주입 PostgreSQL context 모두 같은 bridge를 공유하되 request의 database ID와 mapping/context를 혼합하지 않는다.

## 테스트

관련 Python 테스트만 실행한다. 모든 Ollama 응답은 injected synthetic transport이며 network·실 DB를 사용하지 않는다.

- field constraint가 prompt semantic context를 정확히 한 field로 제한하고 valid QueryIR을 반환.
- entity constraint가 해당 entity의 mapped fields만 포함.
- physical table/column·DSN-like marker·option/token/hash가 prompt/envelope/error/repr/log에 없음.
- SQLite와 PostgreSQL ID의 semantic prompt가 dialect/physical identifier 없이 동등한 shape를 사용.
- request/mapping ID mismatch, stale mapping, forged input, fingerprint/version mismatch, unknown/unmapped ref는 transport 0회 고정 오류.
- model mutation이 caller context에 영향 없음.
- transport success/non-JSON/empty/timeout/error가 호출 예산과 fixed redaction을 지킴.
- default factory는 기존 fixed provider와 network 0회를 유지.
- injected Ollama model factory는 startup/start 0회, submit 1회, cancel/invalid submit 0회.
- generated QueryIR은 기존 constraint validation 및 DB별 compiler/policy/executor를 우회하지 않음.
- Phase 6-09/10/11/12/13과 Phase 7-04 관련 회귀 테스트 실행.

## 금지

- Ollama live 호출, network, model tag 조회, 재학습, 성능 측정 금지.
- CLI/env/config flag 추가와 기본 provider 전환 금지.
- raw SQL prompt/output fallback 또는 dialect별 prompt 추가 금지.
- frontend, PostgreSQL profile/startup loader, SQLite registration/review/publish, legacy 변경 금지.
- 실제 CNU/학교 DB와 private artifact 사용 금지.
- 계획/ADR/handoff/사람 체크포인트 수정, 전체 pytest, compileall, commit, push 금지. 이는 메인 Codex가 담당한다.

## 완료 보고

긴 diff와 전체 로그를 출력하지 않는다. 한국어로 변경 파일, 핵심 계약, 관련 테스트 결과, 실패 원인, 남은 위험만 간결히 보고한다.

---

<a id="prompts-phase7-08-server-owned-ollama-startup"></a>

## 원본: prompts/phase7-08-server-owned-ollama-startup.md

# Phase 7-08 구현 지침: server-owned Ollama startup 설정과 명시적 provider 선택

> Phase 7-10 이후: Ollama를 켤 때 `--ollama-request-format`(`model-template`|`alpaca-raw-no-system-v1`)이 필수다. 최신 계약은 [`phase7-10-runtime-ollama-request-format.md`](development-history.md#prompts-phase7-10-runtime-ollama-request-format)를 따른다.

## 목표

loopback prototype CLI(`python -m backend.adaptive.api.synthetic_prototype`)가 제한된 server-owned Ollama 설정을 명시적 option으로만 받아 기존 `OllamaQueryModel`을 조립하고, Phase 7-07 factory 인자 `ollama_model=`로 주입한다. option이 없으면 기존 결정론 `_FixedConstraintProvider`, environment·Ollama 접근 0회, network 0회를 그대로 유지한다. startup은 설정 검증과 adapter 조립만 하고 모델을 호출하지 않는다. 실제 호출은 clarification `submit`의 확인된 option 실행 시점에만 일어난다.

이번 단계는 synthetic transport로 계약만 완결한다. live Ollama 호출, availability 확인·warm-up, model 다운로드·재학습, 성능 측정은 하지 않는다.

## 구현 위치

1. 신규 `backend/adaptive/api/ollama_startup.py`
   - 설정 계약, strict parser, loopback 전용 transport, model builder.
2. `backend/adaptive/api/synthetic_prototype.py`
   - CLI option 추가, `main()` 연결, app state의 provider 종류 기록.
3. 신규 테스트 `tests/adaptive/test_ollama_startup.py`.

`OllamaQueryModel`(`backend/adaptive/harness/ollama.py`), `OllamaConstraintProvider`(`backend/adaptive/harness/ollama_constraint.py`), compiler/policy/executor/router, PostgreSQL startup loader, frontend, legacy는 변경하지 않는다. `ollama.py`의 private JSON helper(`_reject_duplicate_keys`, `_reject_non_finite`, `_finite_float`)는 import해 재사용할 수 있으나 해당 파일을 수정하지 않는다.

## 설정 계약 (`ollama-startup-config-v1`)

CLI option 세 개. 모두 선택이며 `--ollama-model-tag`가 있어야만 Ollama가 활성화된다.

| option | 계약 | 기본값 |
| --- | --- | --- |
| `--ollama-model-tag` | `recording_bundle_contracts.SAFE_LABEL_PATTERN`을 `re.fullmatch`로 만족하는 ASCII 문자열. 공백·개행·빈 값 거부 | 없음(활성화 스위치) |
| `--ollama-base-url` | 정확히 `http://127.0.0.1:<port>` 또는 `http://[::1]:<port>`. `<port>`는 선행 0 없는 10진수 1~65535. trailing slash·path·query·fragment·userinfo·`localhost`·기타 host 이름·`https`·대문자 scheme 거부 | `http://127.0.0.1:11434` |
| `--ollama-timeout` | `^[1-9][0-9]{0,2}(\.[0-9]{1,3})?$` 형식의 10진 문자열이며 1 이상 120 이하. 부호·지수·`nan`·`inf`·공백 거부 | `30` |

- prompt version은 `query-ir-v1`로 고정한다(ADR-026). option을 추가하지 않는다.
- `--ollama-base-url` 또는 `--ollama-timeout`이 `--ollama-model-tag` 없이 주어지면 설정 오류다.
- 세 option 모두 한 번만 허용한다. 같은 option 반복(`--x a --x b`, `--x=a --x=b` 포함)은 마지막 값 채택이 아니라 설정 오류다.
- parser는 `allow_abbrev=False`로 option 축약(`--ollama-model` 등)을 거부한다. 기존 `--port`, `--repository-root`, `--postgres-profile-id` 동작은 축약 금지 외에는 바꾸지 않는다.
- Ollama 설정은 환경변수·`.env`·설정 파일을 읽지 않는다. CLI 인자만 사용한다.
- 모든 argparse 오류(unknown option, 반복, 형식 오류)는 고정 문구 `synthetic prototype invalid configuration`만 stderr에 출력하고 exit code 2로 끝난다. argparse 기본 메시지처럼 입력값이나 unknown 인자를 반사하지 않는다. 이를 위해 `ArgumentParser.error`를 재정의한 subclass를 쓴다. `--help`는 그대로 둔다.

설정 객체:

- `OllamaStartupConfig`: frozen dataclass(또는 동등 immutable). field는 `base_url: str`, `model_tag: str`, `timeout_seconds: float`.
- `repr`/`str`은 고정 문자열 `OllamaStartupConfig(version='ollama-startup-config-v1')`만 반환한다. URL·port·tag·timeout을 포함하지 않는다.
- `parse_ollama_startup_config(model_tag, base_url, timeout) -> OllamaStartupConfig`: 원문 문자열만 받는다. `bool`, 숫자, `None`(기본값 적용 전의 model tag), bytes 등 문자열이 아닌 값은 coercion 없이 거부한다.
- 실패는 전용 `OllamaStartupConfigError`(고정 code `OLLAMA_STARTUP_CONFIG_INVALID`, 고정 message)로만 알린다. `raise ... from None`으로 원인 예외를 끊고 입력값을 속성·args·message에 보존하지 않는다.

## loopback 전용 transport

기존 `OllamaQueryModel`의 기본 transport는 `urllib.request.urlopen`이라 시스템/환경 proxy(`HTTP_PROXY` 등, Windows registry proxy)를 따르고 POST에 대한 301/302/303 redirect를 GET으로 따라간다. startup 경로는 이 기본 transport를 쓰지 않고 다음 transport를 `transport=`로 주입한다.

- `build_loopback_transport(config)`가 반환하는 callable은 `(url, payload, timeout)` 계약을 지킨다.
- 호출 시 `url`이 `config.base_url + "/api/generate"`와 정확히 같지 않으면 요청 없이 고정 오류를 낸다.
- opener는 `urllib.request.build_opener(ProxyHandler({}), <redirect 거부 handler>)`로 만든다. redirect 거부 handler는 `HTTPRedirectHandler` subclass로 모든 redirect를 따르지 않는다(`redirect_request`가 `None` 반환 또는 고정 오류). 결과적으로 proxy 환경변수를 읽지 않고 외부 host로 요청을 보내지 않는다.
- 요청 body는 기존 `_default_transport`와 같은 `json.dumps(payload, ensure_ascii=False, allow_nan=False)` UTF-8, `Content-Type: application/json`, `POST`이다.
- 응답은 최대 1 MiB까지만 읽는다(`read(limit + 1)`로 초과 감지 시 고정 오류). 2xx 외 status는 고정 오류다.
- 응답 JSON은 기존 helper로 duplicate key·non-finite를 거부해 parse한다.
- timeout은 기존처럼 `opener.open`의 `timeout`에 전달한다. timeout 계열(`TimeoutError`, `socket.timeout`, reason이 timeout인 `URLError`)은 원문 없는 새 `TimeoutError()`로 올려 기존 `OllamaQueryModel`이 `TIMEOUT`으로 분류하게 한다. 그 외 연결·HTTP·parse 오류는 원문 없는 고정 `OllamaLoopbackTransportError`로 바꿔 `PROVIDER_ERROR`로 분류되게 한다. 원인 예외는 연결하지 않는다.
- transport는 로그를 남기지 않고, repr에 URL·tag를 포함하지 않는다.
- 이 transport는 새 Ollama adapter가 아니다. prompt 생성·envelope 검사·QueryIR JSON parse는 모두 기존 `OllamaQueryModel`이 맡는다.

## model 조립

- `build_startup_ollama_model(config, *, transport=None) -> OllamaQueryModel`
  - `config`는 정확히 `OllamaStartupConfig` 인스턴스만 허용한다.
  - `transport`가 `None`이면 `build_loopback_transport(config)`를 사용하고, 테스트는 synthetic transport를 주입한다.
  - `OllamaQueryModel(config.base_url, config.model_tag, config.timeout_seconds, transport=..., prompt_version="query-ir-v1")`만 호출한다. 생성자 예외는 `OllamaStartupConfigError`로 정규화한다.
  - 이 함수와 transport 생성은 network·파일·환경변수 접근 0회다.

## CLI·factory 연결

- `main()` 순서:
  1. argparse strict parse(pure).
  2. Ollama option이 있으면 `parse_ollama_startup_config` → `build_startup_ollama_model`. 실패 시 고정 문구 stderr, exit 2, PostgreSQL loader·factory·uvicorn 호출 0회.
  3. 기존 PostgreSQL profile loading(Phase 7-05) 그대로.
  4. factory 호출. Ollama model이 있을 때만 `ollama_model=model` keyword를 추가한다. option이 전혀 없으면 factory를 **인자 없이** 호출해야 한다(기존 테스트 `seen["kwargs"] == {}` 유지). PostgreSQL만 있을 때는 기존과 같이 `postgres_profiles=`만 전달한다.
  5. ready 문구는 기존 고정 문자열을 유지하며 URL·tag·timeout을 출력하지 않는다.
- Ollama model 테스트 주입을 위해 `main`에 공개 인자를 추가하지 않는다. 테스트는 `ollama_startup` 모듈 함수(예: `build_loopback_transport`)를 monkeypatch해 synthetic transport를 넣는다.
- factory(`create_synthetic_clarification_prototype`)는 Phase 7-07 검증 로직을 그대로 쓴다. 추가 변경은 `app.state.synthetic_prototype_provider_kind`에 `"deterministic"` 또는 `"ollama"` 고정 문자열을 기록하는 것뿐이다. model·config·URL·tag를 app state에 새로 두지 않는다.
- module-level `app = create_synthetic_clarification_prototype()`은 그대로다. import 시 Ollama·환경변수 접근이 없어야 한다.
- 같은 provider 하나를 SQLite(flat·legacy·private)와 PostgreSQL context가 공유하고 database ID router가 compiler/policy/executor를 고른다. provider가 router를 우회하지 않는다.

## 호출 시점 계약

- import, parser 생성, config parse, model 조립, factory, app startup: transport 호출 0회.
- clarification `start`(pending 생성, no-match 포함): 0회.
- 정확한 option `submit`: 정확히 1회. 이후 기존 Phase 7-07 constraint 축소 → Phase 6-09 QueryIR·constraint 검증 → compiler → AST policy → read-only executor.
- cancel, invalid/expired/reused token, invalid selection, context drift: 0회.
- transport timeout/error/empty/non-JSON/invalid QueryIR/constraint 위반은 기존 고정 HTTP 오류 taxonomy로 redacted 변환한다.

## 비노출 계약

stdout/stderr, 예외 message·args·repr, HTTP 응답, app state의 새 항목, 로그에 다음이 나타나지 않는다: base URL·host·port, model tag, timeout 원문, 잘못된 CLI 입력값, transport/urllib 예외 원문, prompt, question, PostgreSQL DSN, physical schema/table/column, SQL.

## 테스트 (`tests/adaptive/test_ollama_startup.py`)

모든 모델 응답은 synthetic transport다. 외부 network, 실제 Ollama, 실제 PostgreSQL, CNU 자료를 쓰지 않는다. redirect/proxy 검증에 한해 테스트 안에서 `127.0.0.1` 임시 `http.server`를 띄우는 것은 허용한다(외부 host로의 실제 연결은 금지이며, redirect 대상 외부 host는 연결되지 않아야 함을 증명하는 용도다).

1. parser·config
   - 정상: tag만(기본 URL·timeout), `[::1]` URL, 경계 timeout `1`·`120`·`0.5` 거부·`120.001` 거부.
   - 거부: 빈/공백/개행 tag, 비 ASCII tag, `localhost`, `0.0.0.0`, 외부 host, `https`, trailing slash, path/query/fragment/userinfo, port 0·65536·선행 0·누락, timeout `0`·`-1`·`1e2`·`nan`·`inf`·` 30`, 문자열 아닌 값.
   - 반복 option, 축약 option, tag 없는 URL/timeout, unknown option은 exit 2와 고정 문구만 출력하고 입력값 비반사.
2. 조립
   - config repr/str 고정, error message·args·`__cause__`/`__context__` 비노출.
   - builder가 `OllamaQueryModel`을 반환하고 prompt version `query-ir-v1`, transport 호출 0회.
3. transport
   - URL 불일치 시 요청 없이 고정 오류.
   - `HTTP_PROXY`/`http_proxy`/`ALL_PROXY` 환경변수가 있어도 proxy를 쓰지 않음.
   - loopback 임시 server의 302 응답을 따라가지 않고 고정 오류.
   - 1 MiB 초과 응답, 비 2xx, duplicate key, non-finite, 비 JSON envelope 거부. 오류 원문 비노출.
4. CLI
   - option 없음: factory kwargs `{}`, 환경변수·transport 접근 0회, uvicorn loopback host.
   - Ollama option: factory에 `ollama_model` 전달, startup 동안 transport 0회, PostgreSQL option과 함께 쓰면 두 kwargs 모두 전달.
   - 설정 오류: PostgreSQL loader·factory·uvicorn 호출 0회.
   - stdout/stderr에 URL·tag·timeout 비노출.
5. 실제 factory 통합(synthetic transport 주입)
   - `provider_kind` 기본 `"deterministic"`, 주입 시 `"ollama"`.
   - startup·start 0회, submit 1회, cancel·재사용 token 0회.
   - prompt의 semantic context가 확인된 field/entity로 제한됨(Phase 7-07 계약 재확인).
   - SQLite와 주입 PostgreSQL context(기존 7-07 테스트의 fake adapter/executor 방식 재사용)가 같은 provider를 쓰되 database ID별 router 결과를 유지.
   - transport 실패가 기존 고정 HTTP 오류로 변환되고 응답에 URL·tag·예외 원문 없음.

회귀로 함께 실행할 관련 테스트:

- `tests/adaptive/test_ollama_constraint_provider.py`
- `tests/adaptive/test_ollama_query_model.py`
- `tests/adaptive/test_synthetic_clarification_prototype.py`
- `tests/adaptive/test_private_postgres_startup.py`

## 금지

- live Ollama 호출, model tag 조회, availability 확인·warm-up, 다운로드·재학습, 성능 측정.
- 환경변수·설정 파일 기반 Ollama 설정, HTTP/browser에서 model URL/tag/timeout 지정.
- raw SQL prompt/output fallback, dialect별 prompt, 모델 출력 보정.
- `OllamaQueryModel`·`OllamaConstraintProvider`·Phase 7-07 계약 완화 또는 수정.
- frontend, PostgreSQL profile/startup loader, SQLite registration/review/publish, legacy 변경.
- 실제 CNU/학교 DB, private artifact 사용.
- 계획/ADR/handoff/사람 체크포인트 문서 수정, 전체 pytest, compileall, commit, push. 이는 메인(Claude Code)이 담당한다.

---

<a id="prompts-phase7-09-ollama-prototype-live-smoke"></a>

## 원본: prompts/phase7-09-ollama-prototype-live-smoke.md

# Phase 7-09 구현 지침: 합성 prototype Ollama end-to-end live smoke

> Phase 7-10 이후: Ollama를 켤 때 `--ollama-request-format`(`model-template`|`alpaca-raw-no-system-v1`)이 필수다. 최신 계약은 [`phase7-10-runtime-ollama-request-format.md`](development-history.md#prompts-phase7-10-runtime-ollama-request-format)를 따른다.

## 목표

Phase 7-08 server-owned Ollama 설정으로 조립한 합성 clarification prototype을 로컬 Ollama에 실제로 연결해, 완전 합성 질문으로 `start → option 선택 → submit → QueryIR 검증 → compile → policy → read-only 실행` 전체 경계가 live 환경에서 유지되는지 확인한다. 이 단계는 **형식·경계 smoke**이며 모델 정확도·성능 평가가 아니다.

## 범위

1. 신규 `backend/adaptive/api/ollama_prototype_smoke.py`: opt-in CLI와 순수 실행 함수.
2. 신규 `tests/adaptive/test_ollama_prototype_smoke.py`: synthetic transport 테스트.
3. 구현·검토 뒤 메인이 로컬 Ollama로 1회 실행하고 결과를 `docs/experiment-results.md#phase7-09-ollama-prototype-live-smoke-results-2026-09-28`에 기록한다.

Phase 7-07/7-08 모듈(`ollama_startup.py`, `ollama_constraint.py`, `ollama.py`), prototype factory·CLI, compiler/policy/executor, frontend, legacy, PostgreSQL 코드는 변경하지 않는다.

## CLI 계약

```text
python -m backend.adaptive.api.ollama_prototype_smoke (--check | --live-synthetic)
    --ollama-model-tag TAG [--ollama-base-url URL] [--ollama-timeout SECONDS]
```

- `--check`와 `--live-synthetic` 중 정확히 하나가 필수다. 둘 다 없거나 둘 다 있으면 설정 오류다.
- Ollama option 세 개는 Phase 7-08의 `parse_ollama_startup_config`와 같은 계약(loopback URL, strict tag, 1~120초)을 재사용한다. 반복·축약·unknown option은 Phase 7-08 prototype CLI와 같이 고정 문구로 거부하고 입력을 반사하지 않는다.
- 설정은 CLI 인자만 쓴다. 환경변수·설정 파일·private artifact를 읽지 않는다.
- model은 `build_startup_ollama_model(config, transport=<counting wrapper>)`로 조립한다. wrapper는 Phase 7-08 `build_loopback_transport(config)`가 만든 transport에 그대로 위임하고 호출 횟수만 센다. payload·응답을 보관하지 않는다.
- app은 `create_synthetic_clarification_prototype(ollama_model=model)`만 사용한다. PostgreSQL profile은 주입하지 않는다(PostgreSQL routing은 7-04 live·7-07/7-08 합성 테스트가 담당).
- HTTP는 `fastapi.testclient.TestClient`로 in-process 호출한다. port를 열지 않는다. 종료 시 `app.state.synthetic_prototype_cleanup`을 반드시 호출한다.

## 고정 합성 case

질문은 모두 코드에 고정된 완전 합성 문자열이다. 실제 과목명·교수명·CNU 자료를 쓰지 않는다. ambiguity rule trigger는 prototype의 `강의`다.

| case ID | DB | 동작 | 기대 model 호출 |
| --- | --- | --- | ---: |
| `p709-flat-course` | flat | `Course.name` option submit | 1 |
| `p709-flat-instructor` | flat | `Instructor.name` option submit | 1 |
| `p709-legacy-course` | legacy | `Course.name` option submit | 1 |
| `p709-legacy-instructor` | legacy | `Instructor.name` option submit | 1 |
| `p709-flat-cancel` | flat | pending 뒤 cancel | 0 |
| `p709-flat-no-match` | flat | trigger 없는 질문, `NOT_REQUIRED` | 0 |

- 각 case는 고유 request ID를 쓰고 case마다 새 session을 만든다. retry·재제출·model 대체는 없다.
- start 단계 호출은 항상 0이어야 한다.

## case별 공개 관찰 항목

case마다 다음만 기록한다.

- `case_id`, `database`(`flat`|`legacy`), `action`, `selected_ref`(submit case만)
- `start_status`(`PENDING`|`NOT_REQUIRED`), `http_status`
- `outcome`: 성공 시 `EXECUTED_SUCCEEDED`, 실행 실패 시 기존 고정 execution code, HTTP 오류 시 기존 고정 error code, cancel은 `CANCELLED`, no-match는 `NOT_REQUIRED`
- `columns_match_constraint`(성공 case만): 결과 columns가 `[selected_ref]`와 정확히 같은지
- `row_count`(성공 case만), `model_calls`(case 동안 증가분), `elapsed_ms`(정수)
- `boundary_ok`: 호출 예산 일치, 응답이 기존 strict shape·고정 code taxonomy 안에 있음, 성공 columns가 비어 있지 않고 canonical semantic field(`Course.name`·`Instructor.name`)만 포함, 응답 본문에 model tag·base URL·질문 원문·`SELECT`·합성 mapping의 physical table/column 이름이 없음
- `columns_match_constraint`는 **관찰값**이며 `boundary_ok`에 넣지 않는다. Phase 6-09 계약은 선택 field가 `select` 또는 `filter`에 쓰이면 준수로 보므로, columns가 `[selected_ref]`와 다른 성공은 경계 위반이 아니라 모델 출력 정밀도 관찰이다.

질문·prompt·model 원문 응답·QueryIR·SQL·행 값·session token·option ID·base URL은 출력·저장하지 않는다.

## 전체 결과

stdout에 한 줄 JSON만 출력한다.

- `status`: 모든 case의 `boundary_ok`가 참이고 총 호출이 정확히 4이면 `COMPLETED`, 아니면 `BOUNDARY_VIOLATION`.
- `summary`: submit 4건 중 `EXECUTED_SUCCEEDED` 수, `columns_match_constraint` 참 수, 총 `model_calls`.
- 메타데이터: smoke version `ollama-prototype-smoke-v1`, `model_tag`, `prompt_version`, `timeout_seconds`. base URL은 출력하지 않는다.
- model 출력이 틀려 422/502가 나도 경계가 지켜졌다면 `COMPLETED`다. 성공 수는 관찰값일 뿐 정확도 주장이 아니다.
- exit code: `COMPLETED` 0, `BOUNDARY_VIOLATION` 1, 설정 오류·미실행 2.
- `--check`: 설정·model·app 조립 뒤 provider 종류가 `ollama`인지 확인하고 cleanup한다. 호출 0회, `{"status":"CHECKED",...,"model_calls":0}`, exit 0.
- 설정·조립 실패: 고정 `{"status":"CONFIG_ERROR"}`만 stdout에 출력하고 exit 2. 원문 예외 없음.
- 예기치 않은 내부 오류: 고정 `{"status":"SMOKE_ERROR"}`, exit 1, cleanup 보장.

## 테스트 (synthetic transport만)

`ollama_startup.build_loopback_transport`를 monkeypatch해 synthetic transport를 주입한다. network·실제 Ollama를 쓰지 않는다.

- 정상 transport: `COMPLETED`, 총 호출 4, cancel·no-match·start 0회, 성공 4건, `columns_match_constraint` 4건.
- 전부 실패하는 transport(timeout·비 JSON·constraint 위반 혼합): 고정 code로 기록되고 `COMPLETED`, 성공 0건.
- `--check`: 호출 0회, transport 생성은 되지만 호출 없음, cleanup 호출.
- mode 누락·중복, 잘못된 URL·tag·timeout, 반복 option: exit 2, 고정 출력, 입력 비반사.
- stdout에 질문 원문·base URL·token·option ID·`SELECT`·행 값이 없음.
- boundary 위반 감지: 호출을 두 번 하는 가짜 provider 등 예산 위반을 만들면 `BOUNDARY_VIOLATION`과 exit 1.
- 예외 경로에서도 cleanup이 호출됨.
- 회귀: `tests/adaptive/test_ollama_startup.py`, `tests/adaptive/test_ollama_constraint_provider.py`, `tests/adaptive/test_synthetic_clarification_prototype.py`.

## live 실행 (메인만)

- 구현·Luna 검토·전체 테스트가 끝난 뒤 메인이 `--check` 후 `--live-synthetic`을 각 1회 실행한다.
- 대상은 `/api/tags`로 확인한 로컬 정확한 tag와 기본 loopback URL, timeout 60초다. 실패해도 재시도하지 않고 그대로 기록한다.
- 결과 문서에는 환경(Ollama version, tag, digest), 공개 관찰 항목 표, 해석 한계만 적는다. 원문 응답·prompt·QueryIR·SQL·행은 기록하지 않는다.

## 사람 관문

완전 합성 질문과 합성 fixture만 쓰므로 E3·L5 대상 자료가 없다. 관찰된 성공 수는 L1/L3 이전의 비확정 관찰이며 성능 주장에 쓰지 않는다. 실제 학교 DB(L2)·UX 확정(L4)과 무관하다. 승인 상태를 바꾸지 않는다.

## 금지

- 실제 CNU·학교 DB·private artifact, PostgreSQL 연결, 외부 host.
- retry, model 대체, raw SQL fallback, 모델 출력 보정.
- 응답·prompt 저장, recording bundle 생성.
- 기존 Phase 7-07/7-08 계약·prototype factory 변경, frontend·legacy 변경.

---

<a id="prompts-phase7-10-runtime-ollama-request-format"></a>

## 원본: prompts/phase7-10-runtime-ollama-request-format.md

# Phase 7-10 구현 지침: runtime Ollama 요청 형식 명시

## 배경

Phase 5-20에서 `text2sql_qwen:latest`의 내장 SYSTEM(CNU `courses` schema)이 `system`을 지정하지 않은 모든 `/api/generate` 호출에 주입됨을 확인했다(ADR-058). 평가 CLI는 요청 형식을 명시하게 했지만 runtime(Phase 7-07 provider, 7-08 startup, 7-09 smoke)은 여전히 기본 형식으로 호출한다. `alpaca-raw-no-system-v1`은 등록 TEMPLATE이 Alpaca인 모델에서만 유효하므로 모든 모델의 기본값으로 둘 수 없다.

## 범위

1. 요청 형식 모듈을 runtime 계층 `backend/adaptive/harness/ollama_request_format.py`로 옮긴다. `backend/adaptive/evaluation/ollama_request_format.py`는 호환 re-export만 남긴다. runtime은 evaluation을 import하지 않는다.
2. `OllamaStartupConfig`에 `request_format`을 추가하고 버전을 `ollama-startup-config-v2`로 올린다.
   - `parse_ollama_startup_config(model_tag, base_url=None, timeout=None, request_format=None)`: `None`은 `model-template`(기존 호출자 호환). 값은 `model-template`·`alpaca-raw-no-system-v1`만, 문자열 외 거부.
   - `build_startup_ollama_model(config, transport=None)`는 기본 loopback transport든 주입 transport든 **config의 형식을 한 번만** 적용한다.
   - repr/str은 고정 문자열이며 형식·URL·tag를 넣지 않는다.
3. 평가 쪽 `OllamaDirectSqlModel`도 `config.request_format`을 직접 적용한다. `arm_live_cli`는 형식을 config로만 전달하고 transport를 미리 감싸지 않는다(이중 적용 방지). 5-19·5-20 label·run ID 규칙은 그대로다.
4. prototype CLI(`synthetic_prototype`)와 smoke CLI(`ollama_prototype_smoke`)에 `--ollama-request-format`을 추가한다.
   - **Ollama를 켤 때 필수**다. 없으면 고정 설정 오류(prototype: stderr 고정 문구·exit 2, smoke: `CONFIG_ERROR`). 반복·축약은 기존 strict parser 규칙대로 거부한다.
   - Ollama를 켜지 않으면(`--ollama-model-tag` 없음) 이 option도 줄 수 없다.
   - smoke 출력 JSON에 `request_format`을 추가한다.
5. startup·import에서 network 0회, clarification start 0회, submit 1회 계약은 그대로다. Phase 7-07 constraint 축소와 QueryIR 검증 경계는 바꾸지 않는다.

## 테스트

- config: 형식 기본값·허용값·거부값, repr 고정, 위조 config 거부.
- builder: no-system 형식에서 transport가 받는 payload가 `raw: true`·정확한 Alpaca 문자열, `model-template`에서는 변화 없음, 주입 transport에도 한 번만 적용.
- prototype CLI: Ollama option에 형식 누락 시 exit 2, 형식만 단독 지정 시 exit 2, 정상 시 factory에 전달된 model이 형식을 적용.
- smoke CLI: 형식 필수, 출력에 형식 포함, synthetic transport로 `raw: true` 확인.
- 평가: `arm_live_cli` 두 arm 모두 no-system에서 240 호출 전부 `raw: true`(이중 적용 없음), 기본 형식 label·run ID 불변.
- 회귀: Phase 7-07/7-08/7-09/5-18/5-19/5-20 관련 테스트.

## live 확인

구현·검토·전체 테스트 뒤 7-09 smoke를 `--ollama-request-format alpaca-raw-no-system-v1`로 1회 실행한다. 성공 실행 경로가 live로 통과하는지와 7-09의 질문 복사 경향이 남는지 관찰만 기록한다. 결과가 나빠도 guard·prompt는 바꾸지 않는다.

## 금지

- 모든 모델에 대한 no-system 기본값, startup에서 `ollama show`·network 호출.
- raw SQL fallback, guard 완화, frontend·legacy·PostgreSQL profile 변경.
- 성능 주장, 사람 승인 상태 변경.

---

<a id="prompts-phase7-11-request-format-in-model"></a>

## 원본: prompts/phase7-11-request-format-in-model.md

# Phase 7-11 구현 지침: 요청 형식을 모델 안으로 옮기고 provenance에 반영

## 배경 (Codex 독립 검증 결과, 226a633 기준)

1. `alpaca-raw-no-system-v1`에서 non-callable transport가 wrapper에 가려 startup에서 거부되지 않고 호출 시점까지 지연된다. `model-template`은 startup에서 거부되어 형식별 동작이 다르다.
2. `model-template`과 `alpaca-raw-no-system-v1` 모델의 `declared_provenance()`가 완전히 같다. 요청 형식이 transport wrapper에서만 적용되어 `OllamaQueryModel`이 형식을 모르기 때문이다. 지금 capture CLI는 형식 option이 없어 이 경로에 닿지 않지만, no-system 수집을 시작하면 오염 조건(5-12·5-13 recording)과 구별할 수 없다.

두 결함의 공통 원인은 형식을 모델 밖에서 적용하는 구조다.

## 범위

1. `apply_request_format(transport, request_format)`는 형식과 무관하게 **먼저** `callable(transport)`를 검사하고 실패 시 고정 `RequestFormatError`를 낸다.
2. `OllamaQueryModel(..., request_format=None)`
   - `None`은 `model-template`. 허용값 외는 고정 `ValueError("INVALID_OLLAMA_REQUEST_FORMAT")`.
   - 기존 transport callable 검사 뒤 생성자 안에서 형식을 한 번 적용한다. `request_format` 속성으로 노출한다.
   - `declared_provenance()`의 generation config에 형식을 반영한다. **`model-template`은 기존 hash를 그대로 유지**(5-15 paired 계약과 기존 bundle 호환)하고, 그 외 형식만 `request_format`, `raw: true`, 재현 template SHA-256을 추가한다. URL·질문·prompt·SYSTEM 원문은 넣지 않는다.
3. `build_startup_ollama_model`은 transport를 감싸지 않고 `request_format=config.request_format`을 모델에 넘긴다. 형식 적용 지점은 runtime에서 모델 생성자 하나다.
4. 평가 SQL provider(`OllamaDirectSqlModel`)는 기존처럼 `apply_request_format`을 쓰며, 진입 callable 검사로 같은 fail-fast 동작을 얻는다.
5. 5-12·5-13 private recording이 내장 SYSTEM 주입 조건이었고 provenance에 드러나지 않는다는 점을 문서에 기록한다. `live_capture`에 형식 option을 추가하는 일은 이번 범위가 아니다.

## 테스트

- non-callable transport: 두 형식 모두 `apply_request_format`·`OllamaQueryModel`·`build_startup_ollama_model`·`OllamaDirectSqlModel`에서 생성 시점에 고정 오류.
- provenance: `model-template` 값이 변경 전 기록값(`generation_config_sha256=48ba68c9…`)과 동일, no-system은 다르고 같은 설정끼리는 동일, URL·prompt 원문 비포함.
- 형식 이중 적용 없음: startup builder·smoke·arm CLI 경로에서 no-system prompt prefix가 정확히 1회.
- 회귀: Phase 5-15·5-18~5-20·7-07~7-10 관련 테스트.

## 금지

live Ollama 호출, model 다운로드, `live_capture` 기능 추가, guard·prompt 변경, 성능 주장, 사람 승인 상태 변경.

---

<a id="prompts-phase7-12-legacy-view-partial-mapping"></a>

## 원본: prompts/phase7-12-legacy-view-partial-mapping.md

# Phase 7-12 구현 지침: legacy view의 부분 mapping과 선언형 값 변환

## 배경

목표는 legacy 챗봇(AI Course Assistant)이 쓰는 legacy PostgreSQL 강좌 DB를 QueryIR 경로로도 질의해, 같은 질문의 결과가 legacy와 비슷하거나 더 나은지 확인하는 것이다(7-13 챗봇 endpoint, 7-14 화면 연결, 7-15 비교). 7-12는 그 전제인 **legacy DB 등록**이다.

- 대상은 legacy 챗봇이 SQL을 보내는 view `v_course_info`다. 두 경로가 같은 행 집합을 보므로, 결과 차이는 질문 이해 방식(QueryIR와 직접 SQL)에서만 생긴다(사용자 결정 A, 2026-09-29).
- 사용자 결정은 두 가지다. 부분 mapping을 허용하고, 추가 field 없이 15개 canonical field만으로 비교한다.
- 사용자 확인(2026-09-29): legacy 강좌 DB는 실제 충남대 테이블이 아니라 팀이 구성한 유사 스키마·데이터다. 이 DB의 mapping은 개발자가 작성한 fixture mapping과 같은 지위이므로, 실제 학교 DB의 L2 검토를 대신하지 않는다.

## 확인된 차이(구조·집계만 확인, 행 값 비노출)

| canonical field | view column | 처리 |
|---|---|---|
| Course.code | `subject_code` | 그대로 |
| Course.name | `subject_name` | 그대로 |
| Course.credits | `credit_hours` (varchar, "학점/이론/실습") | 변환 `SLASH_FIRST_NUMERIC` |
| Course.category | `category` | 그대로 |
| Department.code | 없음 | 미지원 |
| Department.name | `dept_name` | 그대로 |
| Instructor.name | `professor` | 그대로 |
| CourseOffering.year | 없음(`course_year`는 1~5 학년이다) | 미지원 |
| CourseOffering.semester | 없음 | 미지원 |
| CourseOffering.section | `section` | 그대로 |
| CourseOffering.capacity / enrolled | `capacity` / `enrolled` | 그대로 |
| Schedule.day | `day_of_week` (한 글자 한국어 요일) | 그대로 |
| Schedule.start_time / end_time | `start_time` / `end_time` (TIME) | 변환 `TIME_TO_HHMM` |

view는 시간표가 없는 강좌를 LEFT JOIN으로 포함한다(6,846행, 개설강좌 4,003개, 시간표 없는 행 632개). 따라서 view 한 개로 매핑하면 compiler의 INNER JOIN 한계가 드러나지 않는다.

## 범위

### 1. 선언형 변환 (`backend/adaptive/mapping/transforms.py`)

- 닫힌 목록 `FieldTransform`:
  - `SLASH_FIRST_NUMERIC`
    - PostgreSQL 식: `CAST(SPLIT_PART(col, '/', 1) AS NUMERIC)`
    - 허용 field 타입: ontology `number`
    - 허용 column 타입: 문자열
  - `TIME_TO_HHMM`
    - PostgreSQL 식: `TO_CHAR(col, 'HH24:MI')`
    - 허용 field 타입: ontology `time`
    - 허용 column 타입: `time`로 시작하는 선언 타입
- 변환된 field에는 비교 연산자(`equals`, 크기 비교)만 허용하고 `contains`는 compile 단계에서 거부한다.
- 결과값 정규화: `SLASH_FIRST_NUMERIC`의 `Decimal`은 정수면 `int`, 아니면 `float`로 바꾼다.
- AI나 사용자가 식을 쓰지 않는다. 식은 이 모듈의 고정 template에서만 나온다.

### 2. PostgreSQL inspector의 view 선택 포함

- `PostgresConnectionConfig(include_views=False)` 옵션을 추가한다. `True`이면 `relkind IN ('r','p','v')`로 조회한다. 기본값에서는 기존 query와 fingerprint가 그대로 유지된다.
- view에는 PK·FK가 없으므로 relation graph에는 독립 table로 들어간다.
- 한계: column 구성이 같은 채로 view 정의만 바뀌면 fingerprint가 달라지지 않는다. 이 한계는 문서에 기록한다.

### 3. runtime profile의 부분 mapping (`postgres_runtime.py`)

- `PostgresRuntimeProfile(field_transforms={}, unsupported_fields=())`
  - 기본값에서는 기존과 동일하게 canonical 15개 전부를 요구한다.
- 검증(`_validate_one`, fail closed):
  - mapped ∪ unsupported = canonical 15이고, 둘은 겹치지 않는다.
  - 변환 대상 field는 mapped에 속해야 한다.
  - 변환은 ontology 타입과 실제 column 선언 타입 모두와 호환되어야 한다.
- compiler와 policy에 같은 변환을 넘긴다.
- `ValidatedPostgresContext.unsupported_fields()`를 둔다. 7-13에서 "지원하지 않는 항목" 안내에 쓴다.
- `DatabaseRoutingExecutor`는 변환된 field의 결과값만 정규화한다.
- 부분 profile의 catalog(`/courses`)는 지금처럼 15개 전체를 요구하므로 `CONTEXT_UNAVAILABLE`로 닫힌다. 화면 연결은 7-14에서 다룬다.

### 4. compiler·AST policy

- `JoinCompilationRequest.field_transforms`: 변환 field의 projection과 WHERE 좌변을 변환식으로 렌더링한다. placeholder·LIMIT 계약은 그대로다.
- `AstPolicyContext.field_transforms`: 변환 field의 projection과 predicate 좌변은 **mapping column에 선언 변환을 적용해 파싱한 AST와 정확히 같아야** 한다.
  - 검증된 변환 subtree 안의 노드(Cast, 함수, literal)만 허용 노드 검사에서 제외한다.
  - 그 밖의 위치에 나오는 함수·Cast·literal은 기존처럼 거부한다.
  - 변환이 없는 context는 기존 검사와 byte 단위로 같은 경로를 탄다.

### 5. private startup profile v2 (`postgres_startup_profile.py`)

- `postgres-runtime-profile-v2`: v1 key에 `include_views`(bool), `field_transforms`(object), `unsupported_fields`(정렬된 고유 문자열 배열)를 더한다.
- v1은 변경하지 않는다.
- 알 수 없는 변환·key·타입은 고정 오류로 거부한다.

### 6. legacy view 등록 CLI (`backend/adaptive/api/legacy_view_profile.py`)

- 서버 코드 상수:
  - database ID `legacy-view`
  - label `Legacy DB (QueryIR)`
  - DSN 환경변수 `SCHEMA_ADAPTIVE_POSTGRES_DSN_LEGACY_VIEW`
  - 위 표의 mapping과 변환·미지원 목록
- `--check`: 연결 없이 profile 파일의 존재와 strict load 가능 여부만 확인한다.
- `--write --live`
  - view 포함 inspection으로 fingerprint를 얻는다.
  - view column의 존재와 타입을 확인한다.
  - `.schema-adaptive/postgres-profiles/legacy-view/{profile.json,mapping.json}`을 원자적으로 쓴다.
  - 기존 파일이 다르면 `--replace` 없이 거부한다.
- `--smoke --live`
  - 고정 QueryIR를 기존 검증 → compiler → AST policy → read-only executor 경로로 실행한다.
  - 행 값은 출력하지 않고, query별 상태·행 수·형식 검사 결과만 출력한다.
- 출력은 고정 code와 수치뿐이다. DSN, SQL, 물리 이름, 행 값은 출력하지 않는다.

## 제외

- 챗봇 endpoint(7-13), 화면 연결(7-14), legacy와의 비교(7-15).
- LEFT JOIN, 기본 table mapping(B), 코드값 사전(8-06), AI mapping 제안.
- legacy backend·`v_course_info`·legacy DB의 변경. legacy DB에는 읽기 전용 연결만 사용한다.

## 보안 계약(유지)

- server-owned DSN 환경변수를 쓰고, read-only transaction·timeout·행 제한을 유지한다.
- 모델 호출은 1회이며 raw SQL fallback은 없다. QueryIR 검증·compiler·policy·executor를 우회하지 않는다.
- 변환은 닫힌 목록과 고정 template만 쓴다.
- 부분 mapping은 명시적 미지원 목록으로만 허용한다(누락은 여전히 오류다).
- SQLite 등록·upload·review 경로와 v1 profile은 변경하지 않는다.

## 테스트

- 변환
  - compiler 출력이 변환식을 포함하고, AST policy가 이를 통과한다.
  - 변환 subtree 밖의 literal·함수와 다른 변환식·다른 column·`contains`는 거부한다.
  - 변환이 없는 기존 query의 SQL은 이전과 같다.
- runtime 검증
  - 부분 profile이 통과하려면 합집합이 15개여야 한다.
  - 누락, 중복, 미지원 field의 mapping, 알 수 없는 field는 거부한다.
  - 변환 타입이 ontology·column과 맞지 않으면 거부한다.
- inspector: 기본값에서 view가 제외되고 query가 동일하다. `include_views`일 때 view를 포함한다.
- startup profile: v2 strict load와 거부 사례를 확인하고, v1 회귀가 없음을 확인한다.
- 실행 결과: 변환 결과값 정규화(`Decimal` → 수)를 확인한다.
- CLI: 기본 실행은 연결 0회이고 출력에 민감값이 없다.
- live(수동): legacy DB에 `--write --live` 후 `--smoke --live`를 실행한다. 학점 필터 결과가 view의 동일 조건 직접 계산과 같은지 scratch에서 대조하고, 행 값은 기록하지 않는다.

---

<a id="prompts-phase7-13-legacy-view-chat-endpoint"></a>

## 원본: prompts/phase7-13-legacy-view-chat-endpoint.md

# Phase 7-13 구현 지침: legacy view 대상 QueryIR 챗봇 질의 endpoint

## 배경

7-12에서 legacy 챗봇의 view `v_course_info`를 PostgreSQL profile v2(`legacy-view`)로 등록했다. 7-13은 그 DB에 대해 **자연어 질문 하나를 QueryIR 경로로 답하는 endpoint**를 prototype 서버에 추가한다. 7-14에서 `ChatPopup`이 이 endpoint를 부르고, 7-15에서 legacy 챗봇과 같은 질문으로 비교한다.

legacy 챗봇은 모델이 쓴 SQL을 view에 그대로 실행하고 결과 행을 강좌 카드로 보여준다. 새 경로도 결과는 강좌 카드로 보여주지만, 다음 세 가지를 지킨다.

- 모델은 QueryIR만 만든다.
- 실행은 기존 검증 → compiler → AST policy → read-only executor 경로만 거친다.
- SQL과 물리 이름은 응답에 넣지 않는다.

## 범위

### 1. `backend/adaptive/api/chat_query.py`

- `POST /api/adaptive/v1/chat/query`, 요청 `{"database_id", "query"}`
  - strict JSON이어야 하고, 추가 key·중복 key는 거부한다.
  - body는 기존 review 경계 한도 이하, 질문은 1~500자다.
- 대상: 서버가 검증한 PostgreSQL context만 받는다. 합성 SQLite DB는 기존 clarification 흐름을 쓰고, 이 endpoint에서는 404로 처리한다.
- 흐름(모델 호출은 정확히 1회, 재시도·fallback 없음)
  1. drift 검사(`is_postgres_context_current`). 실패하면 404를 돌려준다.
  2. 전체 runtime ontology와 context의 published mapping으로 `generate_ir`를 1회 호출한다. prompt는 매핑된 field만 `available_fields`로 알려준다.
  3. QueryIR 구조 검증에 실패하면 `NOT_UNDERSTOOD`다.
  4. select·filter가 context의 `unsupported_fields()`를 참조하면 실행하지 않고 `UNSUPPORTED`로 답한다. 이때 field의 한국어 label을 함께 보낸다.
  5. 반사 guard: QueryIR 안의 문자열이 질문 원문 전체, database ID, 비의미 물리 table·column 이름과 같으면 `NOT_UNDERSTOOD`다. 질문 전체를 `equals` filter로 복사해 0행이 되는 기존 관찰 경향을 막는다.
  6. 카드 projection: 모델의 filter·entity·limit은 그대로 둔다. select만 **서버 고정 카드 field**(canonical 순서 중 매핑된 field 전체)로 바꾼다. legacy 챗봇이 view의 모든 column을 조회해 카드로 보여주는 것과 같은 조건이다. 이 재작성은 서버 규칙이며 모델 출력에 새 filter를 더하지 않는다.
  7. 재작성한 QueryIR을 기존 `ExecutionHarness`로 실행한다. 고정 출력 model, context compiler·policy, drift 재검사·결과 정규화를 하는 routing executor를 쓴다. 검증 단계 실패(`UNKNOWN_FIELD` 등)는 `NOT_UNDERSTOOD`, 실행 단계 실패는 오류로 처리한다.
- 응답 `adaptive-chat-query-v1`(HTTP 200)
  - `status`: `SUCCEEDED` | `UNSUPPORTED` | `NOT_UNDERSTOOD`
  - `SUCCEEDED`:
    - `columns`: canonical DTO key로, 예: `course_name`, `credits`, `start_time`.
    - `rows`: 결과 행.
    - `row_count`, `limit`.
    - `understood.filters`: field label·연산자·값이다. 질문을 어떻게 이해했는지 SQL 대신 보여주는 용도다.
  - `UNSUPPORTED`: `unsupported`(field·label).
  - `NOT_UNDERSTOOD`: `failure_code`(고정 code).
- 오류 응답 `{"error":{"code"}}`
  - 400 `CHAT_INVALID_REQUEST`
  - 404 `CHAT_CONTEXT_UNAVAILABLE`
  - 503 `CHAT_MODEL_UNAVAILABLE`: prototype에 Ollama가 설정되지 않았거나 모델 호출이 실패한 경우다. timeout 등은 `failure_code`에 담는다.
  - 500 `CHAT_EXECUTION_FAILED`: `failure_code`를 담는다.
- 값은 JSON 안전 타입만 쓴다. `Decimal`은 수로 바꾸고, 그 밖의 타입이 나오면 실행 실패로 닫는다.
- 질문·SQL·물리 이름·행 값·예외 text는 log에 쓰지 않는다.

### 2. prototype 조립

- `create_synthetic_clarification_prototype(...)`가 route를 등록한다. model은 `--ollama-model-tag`로 만든 기존 `OllamaQueryModel`이며, 없으면 503을 돌려준다.
- 기존 route·clarification·catalog 동작은 바꾸지 않는다.

## 제외

- `ChatPopup` 연결·`dev:all` DSN 주입(7-14), legacy와 비교(7-15).
- 다중 턴 대화, clarification 연결, 수강 이력 제외 옵션(legacy의 `excludeCompletedCourses`), 선수과목·강의실 등 15개 밖 field.
- legacy backend·view·DB 변경.

## 보안 계약(유지)

- 모델 호출은 1회이고 raw SQL fallback은 없다.
- QueryIR 검증·compiler·AST policy·read-only executor를 우회하지 않는다.
- 명시적으로 게시된 PostgreSQL profile만 실행하고, server-owned DSN과 drift 검사를 유지한다.
- browser에는 SQL·물리 이름·DSN을 보내지 않는다.
- 카드 projection은 select만 서버 고정 목록으로 바꾸고, filter는 모델 출력 그대로 기존 검증을 받는다.

## 테스트

- fake model(호출 수 기록)과 fake PostgreSQL context로 확인한다.
  - 성공 시 카드 column·행 정규화·understood filter를 확인한다.
  - 모델 호출 1회를 확인한다.
  - 미지원 field는 실행 0회로 `UNSUPPORTED`가 되는지 확인한다.
  - 질문 반사·물리 이름 반사·비JSON·잘못된 구조·허용되지 않은 연산자는 `NOT_UNDERSTOOD`가 되는지 확인한다.
  - drift·unknown·SQLite DB는 404, model 없음·모델 실패는 503, 실행 실패는 500인지 확인한다.
  - 요청 형식(추가 key, 빈 질문, 500자 초과, 중복 key, 비JSON)이 거부되는지 확인한다.
- 응답과 오류에 SQL·물리 이름·DSN·예외 text가 없는지 확인한다.
- 기존 prototype·registry 테스트에 회귀가 없는지 확인한다.
- live(수동): legacy DB와 로컬 Ollama(`text2sql_qwen:latest`, `alpaca-raw-no-system-v1`)로 몇 개 질문을 보낸다. 상태·행 수만 기록하며 모델 성능 주장이 아니다.

---

<a id="prompts-phase7-14-chatpopup-queryir-routing"></a>

## 원본: prompts/phase7-14-chatpopup-queryir-routing.md

# Phase 7-14 구현 지침: ChatPopup의 DB별 경로 선택과 dev:all 연결

## 배경

7-13에서 prototype 서버에 QueryIR 챗봇 endpoint(`POST /api/adaptive/v1/chat/query`)를 만들었다. 7-14는 팀원이 기존 화면에서 이 경로를 바로 써 볼 수 있게 두 가지를 연결한다.

- `DB 변경`에서 고른 DB에 따라 AI Course Assistant가 알맞은 경로로 질문한다.
- `npm run dev:all` 하나로 legacy DB와 로컬 Ollama를 쓰는 QueryIR 경로까지 켜진다.

## 범위

### 1. frontend 경로 선택

- `src/adaptive-prototype/chatQuery.ts`
  - `chatTargetFor(activeDbId, registry)`는 DB 종류에 따라 경로를 정한다.

    | 선택한 DB | 경로 |
    |---|---|
    | Legacy 기준선 | `legacy`(기존 `/api/v1/chat/query` 그대로) |
    | registry `schema === 'postgres'` | `queryir` |
    | 그 밖의 합성·private SQLite | `unsupported` |

    `unsupported`는 legacy DB로 질의하지 않고 안내만 한다. 이전에는 DB를 바꿔도 legacy SQL이 legacy DB로 나갔다.
  - `adaptive-chat-query-v1` strict parser: 카드 key 허용 목록, 행 수·limit 일관성, understood filter 형식을 검사한다. SQL·물리 이름·추가 key는 거부한다.
  - client는 `{database_id, query}`만 보낸다. 질문 길이·제어 문자는 서버와 같은 규칙으로 검사한다.
  - 오류는 고정 한국어 문구로 보여준다. Ollama 미설정, timeout, DB 사용 불가, 형식 오류를 구분한다.
  - 카드 key를 기존 카드 렌더러가 읽는 legacy 행 key로 바꾼다. 예: `course_name` → `subject_name`, `credits` → `credit_hours`.
- `ChatPopup`
  - `chatTarget` prop을 받고, header에 현재 경로(`Legacy 기준선 · 기존 SQL 경로` / `… · QueryIR 경로` / `… · 챗봇 미지원`)를 표시한다.
  - QueryIR 경로의 응답 처리
    - 결과는 기존 카드와 시간표에 그대로 넣는다.
    - SQL 대신 "이해한 조건 (QueryIR)"을 보여준다.
    - `UNSUPPORTED`·`NOT_UNDERSTOOD`는 안내 문구로 보여준다.
    - 수강 이력 제외 옵션은 비활성화한다.
  - 대상 DB가 바뀌면 진행 중인 요청을 취소하고 이전 결과를 지운다. 대화가 시작된 뒤에만 변경 알림을 남긴다.
  - 늦게 도착한 이전 DB의 응답은 두 경로 모두에서 버린다.
- `MainPage`: 부분 mapping PostgreSQL DB는 catalog가 `SYNTHETIC_CATALOG_UNSUPPORTED`(409)로 답한다. 이때만 목록 대신 "AI Course Assistant로 질문해 조회" 안내를 보인다. 그 밖의 catalog 오류는 기존처럼 오류와 재시도 버튼을 보인다.

### 2. catalog 응답 구분(`synthetic_registry.py`)

- `unsupported_fields()`가 있는 PostgreSQL context의 catalog는 DB 조회 없이 409 `SYNTHETIC_CATALOG_UNSUPPORTED`를 돌려준다.
- 완전 mapping PostgreSQL과 SQLite의 응답은 변경하지 않는다.

### 3. `dev:all` 연결(`backend/adaptive/api/dev_launcher.py`)

- `backend/.env`에서 `DATABASE_URL`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL` 세 key만 읽는다. legacy backend와 같은 DB·같은 모델을 쓰게 하기 위해서다.
- private legacy-view profile이 있고 DSN 환경변수가 비어 있을 때만, DSN을 프로세스 안에서 `SCHEMA_ADAPTIVE_POSTGRES_DSN_LEGACY_VIEW`로 넘긴다. 출력·파일 기록은 하지 않는다.
- Ollama는 loopback URL일 때만 `alpaca-raw-no-system-v1` 형식으로 켠다.
- 각 부분은 독립적으로 강등된다. profile 오류, DB 연결·drift 실패, 모델 구성 실패 중 하나가 생겨도 나머지는 유지된다.
- 출력은 고정 상태 한 줄 하나다. 예: `adaptive dev launcher: legacy-view=enabled, ollama=enabled`.
- `package.json`
  - `dev:adaptive-prototype:legacy` script를 추가한다.
  - `dev:all`이 이 script를 쓰게 바꾼다.
  - 기존 `dev:adaptive-prototype`·`dev:adaptive-full`은 그대로 둔다.

## 제외

- legacy 챗봇과의 정량 비교(7-15)와 그 결과에 따른 문자열 `equals` 완화·limit 정책 결정.
- 부분 mapping DB의 목록 화면(catalog DTO 확장).
- `/adaptive-prototype` clarification 화면의 QueryIR 챗봇 연결.
- legacy backend 변경.

## 보안 계약(유지)

- browser에는 SQL·물리 이름·DSN을 보내지 않는다. 모델 호출은 질문당 1회이고 raw SQL fallback은 없다.
- DSN은 server-owned다. launcher는 legacy backend가 이미 쓰는 `.env` 값을 같은 프로세스 경계 안에서만 넘기고, 출력하지 않는다.
- QueryIR 경로의 질문과 인증 token은 legacy endpoint로 보내지 않는다.

## 테스트

- launcher
  - `.env`에서 세 key만 읽는지 확인한다.
  - 부분별 활성·비활성 사유를 확인한다(profile 없음, DSN 없음, 비loopback·잘못된 모델).
  - DB 실패에도 Ollama가, 모델 실패에도 legacy-view가 유지되는지 확인한다.
  - 출력에 값이 없고, 포트가 범위 안인지 확인한다.
- catalog: 부분 profile은 409 `SYNTHETIC_CATALOG_UNSUPPORTED`다.
- chat 서버: 1,000자를 넘는 결과 문자열은 닫힌다. frontend와 같은 제한이다.
- frontend
  - 경로 선택, parser 수용·거부, client 요청 body·오류 문구를 확인한다.
  - 카드 key 변환과 답변 설명 문구를 확인한다.
  - 소스 연결을 확인한다: layout effect 무효화, 두 경로의 응답 대상 확인, 409 전용 안내.
- browser(수동): `dev:all`에서 다음을 확인한다.
  - 합성 flat → 챗봇 미지원 표시.
  - Legacy DB (QueryIR) → 목록 안내와 질문 결과 카드·이해한 조건·미지원 안내.
  - Legacy 기준선 → 기존 SQL 경로와 SQL 패널.

---

<a id="prompts-phase7-15-legacy-vs-queryir-comparison"></a>

## 원본: prompts/phase7-15-legacy-vs-queryir-comparison.md

# Phase 7-15 구현 지침: 같은 질문으로 legacy 챗봇과 QueryIR 경로 비교

## 배경

7-12~7-14로 legacy 챗봇의 view `v_course_info`를 QueryIR 경로로도 질의할 수 있게 되었다. 7-15의 목표는 팀원이 legacy 대비 기존 기능이 유지되는지 판단할 근거를 만드는 것이다. 이를 위해 **같은 질문 세트**를 두 경로에 보내고 결과를 기준 조건과 대조한다.

두 경로의 구성 차이(공정 비교를 위해 보고서에 명시):

| | legacy 챗봇 | QueryIR 경로 |
|---|---|---|
| 모델 | `text2sql_qwen` | `text2sql_qwen`(no-system 요청 형식) |
| 모델 출력 | SQL | QueryIR |
| 전처리 | 오타·키워드 교정, entity masking, vector cache | 없음 |
| 후처리 | 기본 `LIMIT 50`, 시간표 조건이 있으면 일치 강좌의 전체 시간표 행으로 확장(최대 200행), SQL 검증·재시도 | 반사 guard, 서버 고정 카드 projection, 모델 limit |
| 표현 범위 | view의 모든 column | canonical 15개 중 매핑된 12개 |

## 범위

### 1. 질문 세트 `backend/adaptive/evaluation/data/legacy-chat-compare-v1.json`

- 개발자가 작성한 30문항(`legacy-chat-compare-questions-v1`)이다. 결과를 보기 전에 고정하며, gold가 아니므로 사람 검토(L1) 전 상태다.
- 범위(`scope`)
  - `ONTOLOGY`(22): 매핑된 field만으로 답할 수 있다.
  - `OUTSIDE_SCOPE`(5): legacy는 답할 수 있으나 현재 ontology 밖이다(학년, 수업 방식, 선수과목, field 간 비교).
  - `UNSUPPORTED_FIELD`(3): 이 DB에 없는 연도·학기를 묻는다.
- 기준: 앞의 두 범위는 기준 SQL `SELECT DISTINCT subject_code, section FROM v_course_info WHERE …`의 개설강좌 key 집합이다. `UNSUPPORTED_FIELD`는 기준 없이 "답할 수 없음"이 올바른 동작이다.

### 2. 비교 모듈 `backend/adaptive/evaluation/legacy_chat_compare.py`

- 질문 세트를 strict load한다. 기준 SQL은 고정 prefix이고 `;`·주석이 없어야 한다.
- 판정은 순수 함수로 한다. 두 경로 모두 개설강좌 key(과목 코드, 분반) 집합 R을 기준 집합 G와 비교한다.
  - 기준 있음:
    - `EXACT`: R = G
    - `CORRECT_SUBSET`: R ≠ ∅이고 R ⊂ G. 결과 상한으로 잘린 정답 일부다.
    - `EMPTY`: R = ∅, G ≠ ∅
    - `INCORRECT`: 조건에 맞지 않는 결과를 포함한다.
    - `NO_ANSWER`: 미지원·이해 못 함·오류
    - `UNKEYED`: legacy 결과에 key column이 없다.
  - 기준 없음:
    - `REFUSED`: 올바른 거절
    - `ANSWERED_WITHOUT_BASIS`: 근거 없이 행을 반환했다.
    - `EMPTY_ANSWER`
    - `NO_ANSWER`
  - "수용 가능"은 `EXACT`·`CORRECT_SUBSET`·`REFUSED`다. 정밀도(|R∩G|/|R|)와 재현율(|R∩G|/|G|)을 함께 기록한다.
- live 실행(`--live`, loopback만 사용)
  - legacy `POST http://127.0.0.1:8000/api/v1/chat/query`와 QueryIR `POST http://127.0.0.1:8123/api/adaptive/v1/chat/query`에 질문을 보낸다.
  - 기준 SQL은 server-owned DSN 환경변수로 읽기 전용 실행한다.
  - QueryIR 성공 문항은 같은 모델 출력(이해한 filter)을 서버 정책만 바꿔 다시 실행한다. 모델은 다시 부르지 않고, 기존 `answer_chat_query` 경로를 쓴다.
    - `limit200`: limit를 200으로 올린다.
    - `limit200_contains`: 추가로 문자열 field(과목명·학과명·교원명·이수구분)의 `equals`를 `contains`로 바꾼다.
  - 원자료(legacy SQL, key 집합)는 Git 비추적 `.schema-adaptive/reports/phase7-15/`에만 저장한다. stdout은 요약 수치만 출력한다.
- `--summarize RUN.json`: 문항별 판정과 경로·변형별 집계를 계산한다. 행 값과 SQL은 출력하지 않는다.

### 3. 결과 문서 `docs/experiment-results.md#phase7-15-legacy-vs-queryir-comparison-results-2026-09-29`

- 문항별 판정, 범위별 집계, 두 가지 서버 정책 변형의 효과, 관찰된 실패 유형, 한계를 적는다.
- 결정 항목도 정리한다: 문자열 `equals` 완화, limit 정책, 학년 등 ontology 확장.

## 제외

- 서버 정책(limit, `equals` 완화)의 실제 적용. 사용자 결정 뒤에 한다.
- ontology 확장(학년 등), legacy 변경.
- 모델 성능 확정 수치. 질문과 기준이 개발자 작성이라 L1·L3 전 잠정치다.

## 보안·데이터

- legacy DB는 팀이 구성한 유사 데이터다(사용자 확인).
- 질문 세트에는 legacy DB의 학과·과목·교원 이름이 들어간다. 커밋 문서에는 문항과 판정·집계만 적고 행 값과 SQL은 적지 않는다.
- legacy 경로 호출은 legacy의 기존 query log에 기록될 수 있다(legacy 기존 동작).

## 테스트

- 질문 세트 strict load와 거부 사례: 중복 ID, scope·기준 불일치, 허용되지 않은 SQL 형태.
- 판정 함수의 각 분기와 정밀도·재현율.
- 변형 filter 재작성: `equals`→`contains` 대상 field만 바뀌는지.
- 집계.
- fake transport로 live 실행 흐름을 확인한다: 저장 경로, stdout에 값·SQL 없음.

---

<a id="prompts-phase7-16-chat-policies-and-recompare"></a>

## 원본: prompts/phase7-16-chat-policies-and-recompare.md

# Phase 7-16 구현 지침: 챗봇 서버 정책 3종 적용과 재비교

## 배경

7-15 비교(개발자 작성 30문항)에서 매핑 범위 22문항 중 수용 가능한 답은 legacy 17, QueryIR 11이었다. 같은 모델 출력에 서버 정책을 바꿔 재실행한 결과는 다음과 같았다.

- limit 200 + 문자열 `contains`: 13
- 요일 정규화 추가(오프라인 추정): 14

사용자는 이 세 가지 적용과 재비교를 결정했다(2026-09-29).

## 범위

### 1. 결과 limit 200 고정(`api/chat_query.py`)

- 챗봇 경로는 모델이 낸 limit 대신 서버 상한 `CHAT_ROW_LIMIT = 200`을 쓴다. executor 최대와 같다.
- 모델의 limit은 이번 실행에서 항상 prompt 예시값 50이었다. 표시 행 수는 화면 규칙이므로 서버가 정한다.
- 한계: "3개만 보여줘" 같은 개수 요청은 반영하지 않는다(문서에 명시).

### 2. 이름형 문자열 field의 `equals` → `contains`(`api/chat_query.py`)

- 대상 field는 `Course.name`, `Department.name`, `Instructor.name`, `Course.category`다. 모델이 `equals`를 쓰면 서버가 `contains`로 바꾼다.
- 코드형 field(`Course.code`, `CourseOffering.section`)와 요일은 바꾸지 않는다.
- 바뀐 연산자는 "이해한 조건"에 그대로 보인다. 예: `학과명에 "수학과" 포함`.
- 바뀐 QueryIR은 기존 QueryIR 검증·compiler·AST policy를 그대로 거친다.

### 3. 요일 값 정규화: profile 선언형 변환 `KOREAN_WEEKDAY`(`mapping/transforms.py`)

- 요일 표기는 DB마다 다를 수 있다(`월`, `MON`, `1` 등). 그래서 챗봇 전역 규칙이 아니라 **profile이 선언하는 닫힌 목록 변환**으로 둔다. ADR-068의 확장이며, 8-06 코드값 교정의 첫 사례다.
- `KOREAN_WEEKDAY`
  - SQL 식: column 그대로(identity).
  - filter 값: `월`·`월요일`·`mon`·`monday`(대소문자 무시) 등을 한 글자 `월`로 바꾼다. 월~일 7개 요일을 모두 다룬다.
  - 그 밖의 값(`주말` 등)은 표현 불가로 거부한다: compiler는 `INVALID_VALUE_TYPE`, 챗봇은 `NOT_UNDERSTOOD`.
  - 허용 field 타입: ontology `string`. 허용 column 타입: 문자열.
  - 허용 연산자: `equals`, `not_equals`, `contains`.
- 변환별 허용 연산자를 함수로 둔다(`transform_operators`). 기존 두 변환의 허용 연산자는 바뀌지 않는다.
- compiler와 policy는 같은 정규화를 적용한다. identity 변환도 AST exact 비교를 통과해야 한다.
- legacy-view profile(`legacy_view_profile.py`)의 선언에 `Schedule.day: KOREAN_WEEKDAY`를 더한다. private profile 파일은 `--write --live --replace`로 갱신한다(fingerprint 불변).

### 4. 재비교

- `legacy_chat_compare --live`를 같은 질문 세트로 다시 실행한다(`p716-run1`).
- 7-15의 `p715-run1`과 문항별로 비교한다. 모델 출력은 temperature 0이다.
- 정책이 서버에 들어갔으므로 `queryir` 열 자체가 정책 적용 결과다. 기존 변형 열(`+limit200`, `+limit200+contains`)은 같은 filter의 재실행이라 거의 같아진다.
- 결과 문서: `docs/experiment-results.md#phase7-16-chat-policies-recompare-results-2026-09-29`.

## 제외

- 이수구분 값 사전(`전공핵심` ↔ `전공(핵심)`), 모델이 만든 field 이름의 별칭 교정, 학년 등 ontology 확장, prompt 변경.
- 개수 요청("3개만")을 존중하는 limit 규칙.

## 보안 계약(유지)

- 모델 호출은 1회이고 raw SQL fallback은 없다. 정책은 QueryIR 수준의 결정적 재작성이며, 결과는 기존 검증·compiler·policy·executor를 그대로 거친다.
- 변환은 닫힌 목록이고, 식은 고정 template에서만 나온다. SQL·물리 이름은 응답에 넣지 않는다.

## 테스트

- 챗봇 서버 정책
  - limit이 200으로 고정되는지 확인한다.
  - 이름형 field `equals`가 `contains`로 바뀌고, 코드형·수치형 field는 유지되는지 확인한다.
  - understood 표시가 실제 실행 조건과 같은지 확인한다.
- `KOREAN_WEEKDAY`
  - 정규화 표와 거부 값을 확인한다.
  - compiler와 policy에 같은 값이 bind되는지 확인한다.
  - 허용 연산자(`contains` 허용)와 기존 변환의 연산자 불변을 확인한다.
  - identity 변환의 AST 검증(다른 column·함수 거부)을 확인한다.
- profile·runtime: `Schedule.day` 변환 선언이 타입 검사를 통과하는지 확인한다.
- live: 재비교를 실행하고 7-15 대비 문항별 변화를 확인한다.

---

<a id="prompts-phase7-17-anchor-left-join"></a>

## 원본: prompts/phase7-17-anchor-left-join.md

# Phase 7-17 지침: 기준 table 보존 LEFT JOIN과 legacy 기본 table 매핑

작성일: 2026-09-29. 사용자 결정(개발 순서 1번, B안).

## 1. 목적

지금 compiler는 FK 경로를 INNER JOIN으로만 잇는다. 정규화된 실제 스키마에서는 "시간표가 없는 강좌"처럼 선택적 관계가 흔하다. INNER JOIN만 쓰면 이런 행이 오류 없이 결과에서 빠진다.

legacy DB는 view(`v_course_info`)가 이 문제를 LEFT JOIN으로 숨기고 있다(시간표 없는 개설강좌 632개). 7-17은 view 없이 기본 table에 직접 매핑해도 같은 답이 나오게 한다.

- 이후 AI mapping 비교 실험에서 mapping 오류와 JOIN 한계가 섞이지 않게 한다.
- 이 단계는 관계 해석 기능만 추가한다. 모델·prompt·챗봇 정책은 바꾸지 않는다.

## 2. 핵심 결정

### 2.1 결과 행의 기준 table(anchor)을 profile이 선언한다

- private profile `postgres-runtime-profile-v3`는 v2 key에 `anchor_table`을 더한다. 값은 문자열이며 필수다.
- v1·v2 형식과 동작은 바꾸지 않는다. v2 profile에 `anchor_table`이 있으면 거부한다(엄격 key 집합).
- `anchor_table`의 의미: 결과 한 행은 이 table의 한 행(및 그에 딸린 자식 행)이다. legacy에서는 개설강좌 table이다.

### 2.2 anchor가 있으면 모든 JOIN이 anchor에서 시작하는 LEFT JOIN이다

- 계획: anchor를 required table에 항상 포함하고, JOIN 계획의 시작점으로 고정한다. 나머지는 기존 최소 FK 경로 탐색 그대로다.
- SQL: 모든 단계를 `LEFT JOIN`으로 쓴다.
  - 부모 방향(자식→부모 FK)에서는 FK 무결성이 있으면 INNER와 결과가 같다.
  - 자식 방향(부모→자식)에서는 자식이 없는 anchor 행이 NULL과 함께 남는다.
  - 즉 "anchor 행은 모두 남고, 관련 값은 있으면 붙는다"가 된다.
- WHERE 조건은 그대로다. LEFT JOIN된 column에 조건이 걸리면 NULL 행은 빠진다. 이는 view에서 같은 조건을 쓸 때와 같은 의미다.
- anchor가 없는 profile(SQLite 경로, 기존 PostgreSQL profile)은 INNER JOIN 그대로다.

### 2.3 AST policy는 모양을 정확히 확인한다

- anchor 모드:
  - FROM은 anchor table이어야 한다.
  - 모든 JOIN은 수식어 없는 `LEFT JOIN`이어야 한다.
  - `INNER`, `LEFT OUTER`, `RIGHT`, `FULL`, `CROSS`, `NATURAL`, `USING`은 거부한다.
- 비 anchor 모드: 지금처럼 기본 JOIN 또는 `INNER JOIN`만 허용한다. `LEFT`는 거부한다.
- policy는 compiler와 독립적으로 JOIN 계획을 다시 세운다. anchor도 같은 profile 값에서 받는다.

### 2.4 시작 시 경로 모호성을 막는다

- anchor profile은 시작 검증에서 두 가지를 확인한다.
  - anchor table이 catalog에 있고, 매핑된 table 중 하나여야 한다.
  - 매핑된 table의 공집합이 아닌 모든 부분집합(최대 8개 table)에 대해 anchor 고정 JOIN 계획이 하나로 정해져야 한다.
- 하나라도 AMBIGUOUS이거나 연결되지 않으면 context를 만들지 않는다(fail closed).
- legacy DB에는 같은 table 쌍을 잇는 다른 경로(예: 필수과목 table)가 있다. anchor를 고정하면 경로가 하나로 정해져야 한다.

### 2.5 legacy 기본 table profile

- `legacy-base`(label `Legacy DB tables (QueryIR)`, DSN 변수 `SCHEMA_ADAPTIVE_POSTGRES_DSN_LEGACY_BASE`)를 추가한다.
- anchor는 개설강좌 table이다.
- view profile과 같은 12개 field를 매핑한다.
  - 과목 field는 과목 table, 학과명은 학과 table, 교원·분반·정원·신청은 개설강좌 table, 요일·시간은 시간표 table이다.
  - 변환 4개와 미지원 3개도 view profile과 같다.
- 기본 table에는 학과 코드 column이 있다. 하지만 이번 단계는 view와 같은 조건에서 JOIN 차이만 보려는 것이므로 `Department.code`는 미지원으로 유지한다. 이 판단은 결과 문서에 적는다.
- `legacy_view_profile.py`는 spec 구조로 일반화한다.
  - CLI에 `--target view|base`를 둔다. 기본값은 `view`다.
  - 기존 view 동작, 오류 code, 파일 경로는 바꾸지 않는다.
  - 등록 전 각 매핑 column의 table 존재와 선언 타입을 확인한다.

### 2.6 dev launcher와 비교 도구

- `dev:all` launcher:
  - 두 profile이 각각 있으면 legacy DSN을 두 DSN 변수에 in-process로 넣는다.
  - 상태는 profile별로 한 줄에 표시한다.
  - 한 profile이 잘못되어도 다른 profile과 Ollama는 계속 쓴다.
- 비교 도구에 `--queryir-database legacy-view|legacy-base`를 추가한다(기본 `legacy-view`). 실행 기록에 대상 DB를 남긴다.

## 3. 검증

1. 단위 테스트:
   - planner anchor 고정: 모호성 해소, 알 수 없는 anchor.
   - compiler LEFT JOIN 출력.
   - policy 허용·거부 모양 전체.
   - v3 profile 엄격 load, v2에 추가 key가 있으면 거부.
   - 시작 시 모호성 거부.
   - base spec 등록·smoke, launcher 두 profile 상태, 비교 도구 옵션.
2. 실제 legacy DB 동등성:
   - base profile로 컴파일한 "카드 전체 field, 조건 없음" 관계를 조회한다. view profile의 같은 관계와 행 multiset이 같아야 한다(6,846행).
   - 조건은 두 경로에서 같은 column 값과 같은 변환에 대한 행 단위 함수다. 따라서 전체 관계가 같으면 어떤 조건에서도 결과가 같다.
   - 이 검증 script는 scratchpad에 두고, 결과만 문서에 기록한다.
3. 같은 30문항 live 비교를 `legacy-base`로 다시 실행한다.
   - 모델 출력이 7-16과 같은지 확인한다.
   - 판정 분포가 view와 같은지 확인한다.
   - LIMIT 200에 걸리는 문항은 ORDER BY가 없어서 부분집합이 달라질 수 있다. 이 차이는 따로 표시한다.
4. 전체 pytest, compileall, `git diff --check`를 실행한다.

## 4. 범위 밖

- 학과 코드 추가 매핑과 ontology 확장.
- anchor 자동 추론. 이번에는 profile에 명시한다.
- nullable FK의 부모 방향 INNER 선택.
- 다중 anchor와 self join.
- SQLite 경로의 LEFT JOIN.
- 결과 정렬(ORDER BY).
- 모델·prompt 변경.

## 5. 지켜야 할 제약

- private profile과 비교 원자료는 Git 비추적 경로에만 둔다.
- DSN·SQL·물리 이름을 브라우저나 stdout에 내보내지 않는다.
- 모델 호출은 1회이며 raw SQL fallback은 없다.
- 실행은 read-only이고 timeout과 행 상한 200을 유지한다.
- legacy 시스템은 바꾸지 않는다.
- 사람 관문(E3, L1–L5)은 승인으로 표시하지 않는다.
- push하지 않는다.

---

<a id="prompts-phase8-01-mapping-version-rollback"></a>

## 원본: prompts/phase8-01-mapping-version-rollback.md

# Phase 8-01 구현 지침: mapping 버전 이력과 이전 버전 재활성화(rollback)

## Phase 8 분할 계획

Phase 6-17~6-19에서 합성 SQLite의 private 등록·loopback upload·`PENDING_REVIEW` 요약·필드 검토(`INIT/APPROVE/REMAP`)·명시적 publish·activation commit까지 구현했다. 로드맵 Phase 8 8개 항목을 기존 보안 계약을 유지하는 작은 단계로 나눈다. 순서는 위험이 작고 기존 immutable 구조를 그대로 쓰는 것부터다.

| 단계 | 로드맵 항목 | 내용 | 전제 |
| --- | --- | --- | --- |
| 8-01 | mapping publish·rollback 화면 | 버전 이력 표시, 이전 immutable 버전 재활성화, 활성 기록 | 이번 단계 |
| 8-02 | schema fingerprint 비교 | 같은 논리 DB의 새 등록(후속 파일) 또는 PostgreSQL profile 재검사 결과를 기존 활성 mapping과 table·column 단위로 비교 | 등록 불변성 유지(덮어쓰기 대신 새 등록) |
| 8-03 | 영향받은 mapping `STALE` 처리 | 비교 결과 영향 field만 재검토 대상으로 표시, 영향 없는 field 승계 제안(자동 publish 없음) | 8-02 |
| 8-04 | test query·결과 미리보기 | **publish된** mapping에서만 고정 합성 test QueryIR 실행 미리보기(행 수 제한·redaction) | 미게시 mapping 실행 금지 유지 |
| 8-05 | verified query 자동 회귀 | 승인된 test query 집합을 publish·재활성화 전 자동 실행해 결과 변화 표시 | 8-04 |
| 8-06 | 코드값·관계 교정 | ontology value mapping·관계 교정 계약 | 설계 ADR 선행, L2 전 실제 DB 금지 |
| 8-07 | DB 등록·스캔 화면 통합 | 메인 `DB 변경` 안의 prototype 흐름을 별도 관리자 화면으로 모음 | 8-01~8-05 |

실제 학교 DB mapping publish·질의는 계속 L2 전 금지다. 모든 단계는 합성 입력으로만 검증한다.

## 8-01 범위

### 1. storage (`backend/adaptive/registration/reviews.py`)

- `activation.json`이 유일한 실행 권한 기록이라는 기존 계약을 유지한다. `current.json`은 최신 publish 버전 포인터로 남기고 재활성화에서 바꾸지 않는다.
- `list_mapping_versions(repository_root, database_id)`: `mapping-vNNNN.json`을 버전 순으로 strict 읽고 버전별 `version`, `published_by`, `published_at`, `activatable`(아래 재활성화 검증 통과 여부), `active`(activation 버전과 같음), `latest`(current 버전과 같음)를 반환한다. 최근 50개까지만. 파일 원문·물리 이름·경로는 반환하지 않는다.
- `activate_mapping_version(repository_root, database_id, mapping_version, activated_by)`:
  - 입력 검증(database ID·버전 1~9999·reviewer), 등록 strict reload, per-DB lock.
  - 한 번이라도 publish된 상태(`current.json`·`activation.json`·draft 존재)가 아니면 `ARTIFACT_MISMATCH`.
  - 대상 mapping 파일을 strict 읽고 `database_id`·`version`·`PUBLISHED`·등록 schema fingerprint·현재 ontology version·canonical 전체 coverage·catalog 실행 가능성을 publish와 같은 기준으로 재검증한다. 실패는 `PUBLISH_BLOCKED`.
  - 이미 같은 버전이 활성이면 쓰지 않고 그대로 반환한다(idempotent).
  - 새 `ActivationManifest`(대상 mapping SHA-256·bytes, 등록 content hash·label)를 원자적으로 교체하고 `load_activation`으로 재검증한 결과를 반환한다.
- 활성 기록: `ActivationManifest`에 선택 필드 `activated_by`, `activated_at`(UTC ISO 8601)을 추가한다. 기존 manifest(필드 없음)도 그대로 유효하다. publish는 `activated_by=published_by`로 기록한다.
- publish idempotent 경로 보완: draft가 최신 `current`와 같아 새 버전을 만들지 않을 때, 활성 버전이 다르면(재활성화 뒤 또는 current·activation 사이 중단) 최신 버전을 재활성화 검증 뒤 활성화한다. "publish를 누르면 최신 검토 결과가 실행된다"를 보장한다.

### 2. HTTP (`backend/adaptive/api/review_publish.py`)

- `POST` action `ACTIVATE`: `{"action","database_id","mapping_version","activated_by"}` strict body. 성공 응답은 publish 응답과 같은 형태(`status: PUBLISHED`, `mapping_version`=활성 버전, `registry_eligible: true`). 실패는 기존 고정 code(`PUBLISH_BLOCKED`는 fields 없이).
- `GET` detail에 게시된 DB이면 `active_mapping_version`(activation strict load 성공 시 버전, 실패 시 null)과 `mapping_versions` 목록을 추가한다. 미게시 detail shape는 바꾸지 않는다.
- registry·catalog·clarification은 이미 요청마다 activation을 strict reload하므로 추가 변경 없이 재활성화 버전을 쓴다.

### 3. frontend (`src/adaptive-prototype`)

- `mappingReview.ts`: detail parser가 새 두 필드를 strict 검증(버전 정수, 중복 없음, active 최대 1개이고 `active_mapping_version`과 일치, latest는 `mapping_version`과 일치, 불리언 타입), `activateMappingVersion` client, `ACTIVATE` 응답 parser.
- `MappingReviewPanel.tsx`: 게시된 DB에서 버전 이력(버전·게시자·게시 시각·활성/최신 표시)과 활성이 아닌 `activatable` 버전의 "이 버전으로 되돌리기" 버튼. 버튼은 2단계 확인(선택 → "재활성화 확인")을 거치고, 검토자 입력이 필요하다. 성공 뒤 detail을 다시 불러오고 `onPublished`로 registry 갱신을 알린다.
- legacy 화면은 바꾸지 않는다.

## 보안 계약(유지)

미게시 draft 실행 금지, 명시적 publish·재활성화만 실행 권한 변경, immutable mapping 파일 불변, 등록 파일 불변, 실행 전 strict reload, browser에 물리 경로·SQL·행·sample·traceback 비노출, 고정 오류 code, 요청 body 64 KiB·duplicate key 거부.

## 테스트

- storage: publish v1→remap→publish v2→v1 재활성화→`load_activation` v1·`current` v2 유지; 같은 버전 재활성화 idempotent(파일 미변경); 없는 버전·0·문자열·범위 밖 거부; 변조된 대상 파일(schema fingerprint·coverage·JSON 손상) `PUBLISH_BLOCKED`/`ARTIFACT_MISMATCH`이고 기존 활성 유지; 미게시 DB 재활성화 거부; 재활성화 뒤 같은 draft publish 시 최신 버전 재활성화; 기존 `activated_by` 없는 manifest 로드; 이력 목록 필드·순서·active/latest.
- HTTP: ACTIVATE strict body(extra key·타입·duplicate key 거부), 성공 응답 shape, registry가 재활성화 버전을 사용, GET 새 필드, 오류 응답에 경로·원문 없음.
- frontend(node --test): parser 새 필드 검증·거부 사례, `activateMappingVersion` 요청 body와 응답 검증, 기존 detail(미게시) 호환.

---

<a id="prompts-phase8-02-schema-fingerprint-compare"></a>

## 원본: prompts/phase8-02-schema-fingerprint-compare.md

# Phase 8-02 구현 지침: schema fingerprint 비교와 활성 mapping 영향 분석(읽기 전용)

## 배경

private SQLite 등록은 불변이므로(Phase 6-17) 같은 논리 DB의 schema가 바뀌면 관리자는 새 파일을 **새 database ID로 등록**한다. 이때 기존에 게시·활성화된 mapping을 새 등록에 그대로 쓸 수 있는지, 어떤 field가 영향을 받는지 알아야 8-03(`STALE` 처리·영향 field만 재검토)을 설계할 수 있다. 8-02는 두 등록의 schema를 비교하고 활성 mapping의 영향을 계산하는 **읽기 전용** 단계다. 어떤 파일도 쓰지 않고 실행 권한도 바꾸지 않는다.

## 범위

### 1. 순수 비교 (`backend/adaptive/catalog/compare.py`, 신규)

- `compare_catalogs(base, candidate) -> CatalogDiff`(`catalog-diff-v1`). dialect 무관 순수 함수로, PostgreSQL catalog에도 쓸 수 있다.
  - fingerprint 두 개와 `identical`(fingerprint 동일).
  - 추가·삭제된 table.
  - 공통 table의 column 변화: `ADDED`·`REMOVED`·`TYPE_CHANGED`(declared type 또는 affinity)·`NULLABILITY_CHANGED`·`PRIMARY_KEY_CHANGED`.
  - FK 변화: `ADDED`·`REMOVED`(table·column·참조 table·참조 column 기준).
  - 이름은 정확히 일치할 때만 같은 것으로 본다. 이름 변경 추정은 하지 않는다(삭제 + 추가로 보고).
  - `sample_values`·`default_sql`·hidden/generated 원문은 비교하거나 반환하지 않는다(값 노출 방지).
  - 변경 목록은 합계 500개까지만 반환하고 `truncated`로 표시한다. 순서는 결정적(table, column, 변경 종류)이다.
- `mapping_impact(profile, candidate) -> list[FieldImpact]`: 활성 mapping의 field마다 `UNCHANGED`·`TYPE_CHANGED`·`COLUMN_MISSING`·`TABLE_MISSING`. base와 candidate의 column 속성(declared type·affinity·nullable)을 비교한다.

### 2. 등록 간 비교 (`backend/adaptive/registration/reviews.py`)

- `compare_registrations(repository_root, base_database_id, candidate_database_id) -> RegistrationComparison`:
  - base는 strict `load_activation`이 성공하는 게시·활성 등록이어야 한다(실행 중 mapping 기준).
  - candidate는 strict `load_registration`이 성공하는 등록(검토 대기 포함)이다. base와 같은 ID는 거부.
  - diff, field 영향, `carry_over_executable`(base mapping을 candidate ID·fingerprint로 옮겼을 때 canonical 전체 coverage·catalog 실행 가능성 검사를 통과하는지; 모든 field가 존재하고 ontology version이 같을 때만 계산, 아니면 false), 두 ontology version 일치 여부.
  - 등록·검토·activation artifact 쓰기·lock 없음(strict reload의 기존 임시 검증 사본은 예외로 생성 뒤 삭제).

### 3. HTTP (`GET /api/adaptive/v1/registrations/compare?base_database_id=..&candidate_database_id=..`)

- 두 query parameter만 정확히 허용(중복·추가 key 거부). 응답은 `registration-compare-v1` metadata만: fingerprint, 요약 수, 변경 목록, field 영향, `carry_over_executable`, `ontology_version_match`, base 활성 mapping 버전. physical table·column 이름은 기존 관리자 검토 화면과 같은 범위로 포함하되 경로·행·sample·SQL·traceback은 포함하지 않는다.
- 고정 오류: `SCHEMA_COMPARE_INVALID_REQUEST`(400, 형식 오류·같은 ID), `SCHEMA_COMPARE_NOT_FOUND`(404, base 미활성·등록 없음), `SCHEMA_COMPARE_INTERNAL_ERROR`(500).

### 4. frontend

- `src/adaptive-prototype/schemaCompare.ts`: strict parser와 client.
- `SchemaComparePanel.tsx`: 업로드 직후 검토 대기 DB 아래에서, registry의 private 활성 DB 중 하나를 기준으로 골라 비교 결과(동일 여부, table·column·FK 변화, field 영향, 그대로 옮겨 실행 가능 여부)를 읽기 전용으로 보여준다. 기준 후보는 MainPage가 이미 가진 registry에서 `schema === 'private'`이고 자기 자신이 아닌 항목만 전달한다.
- legacy 화면 동작은 바꾸지 않는다(MainPage에는 기존 업로드 결과 블록 안에 패널 한 개만 추가).

## 제외

- mapping 승계·`STALE` 표시·후속 등록 연결 기록(8-03).
- PostgreSQL profile 재검사 연결(순수 함수만 공유).
- 이름 변경 추정, 코드값 비교.

## 보안 계약(유지)

등록·mapping·activation 파일 불변, 비교는 실행 권한을 바꾸지 않음, 미게시 candidate의 SQL 실행 없음(`carry_over_executable`은 compile·AST policy 검사까지만 하고 실행하지 않는다), sample·행·경로 비노출, 고정 오류 code.

## 테스트

- 순수 비교: 동일 catalog, table 추가·삭제, column 추가·삭제·타입·nullable·PK 변화, FK 추가·삭제, 결정적 순서, 500개 상한과 `truncated`, sample·default 값 비노출.
- field 영향: 각 상태, flat→normalized 같은 구조 변경에서 `TABLE_MISSING`.
- 등록 비교: 동일 schema의 새 등록은 `identical`·전 field `UNCHANGED`·`carry_over_executable` true; column 이름 변경·삭제는 해당 field `COLUMN_MISSING`·false; FK 삭제로 JOIN 불가한 normalized는 field는 `UNCHANGED`여도 false; base 미게시·같은 ID·없는 ID는 거부; 파일 변경 없음.
- HTTP: query strictness, 응답 shape, 경로·sample 비노출.
- frontend: parser 거부 사례와 client 요청 URL.

---

<a id="prompts-phase8-03-stale-carry-over"></a>

## 원본: prompts/phase8-03-stale-carry-over.md

# Phase 8-03 구현 지침: 영향 field `STALE` 표시와 영향 없는 field 승계로 검토 시작

## 배경

8-02는 새 등록과 기준(게시·활성) 등록의 schema 차이와 활성 mapping 영향을 읽기 전용으로 보여준다. 8-03은 그 결과로 **새 등록의 검토를 시작**하는 명시적 관리자 행위를 추가한다. 영향이 없는 field는 기준 mapping을 승계하고, 영향받은 field만 `STALE`로 표시해 다시 검토하게 한다. 자동 publish·실행 권한 변경은 없다.

로드맵 "영향받은 mapping을 `STALE` 처리"는 등록 불변 모델에서 다음 뜻으로 구현한다: 기준 등록의 파일·mapping은 그대로 유효하고(같은 파일에서는 계속 실행 가능), schema가 바뀐 새 등록에서 기준 mapping의 해당 field가 **그대로 쓸 수 없는 상태(STALE)** 로 기록된다.

## 범위

### 1. storage (`backend/adaptive/registration/reviews.py`)

- `carry_over_review(repository_root, database_id, base_database_id, reviewer) -> MappingDraft`:
  - `database_id`(대상)와 `base_database_id`는 달라야 한다. 기준은 strict activation load가 되어야 한다(8-02 `compare_registrations` 재사용).
  - ontology version이 다르면 `PUBLISH_BLOCKED`(승계 불가).
  - 대상은 한 번도 publish되지 않아야 한다(`current.json` 없음). 첫 검토에만 쓴다.
  - 대상 draft가 없으면 proposal에서 새 draft를 만든다. 이미 있으면 모든 field가 `PENDING`일 때만(검토 행위 전) 대체한다. 그 외는 `REVIEW_CONFLICT`.
  - 영향이 `UNCHANGED`인 field만 기준 활성 mapping의 table·column으로 `REMAP`하고 사유를 `carried over from <base> mapping-vN; schema unchanged`로 남긴다. reviewer는 요청자다.
  - `TYPE_CHANGED`·`COLUMN_MISSING`·`TABLE_MISSING` field는 `PENDING`으로 두고 `STALE`로 기록한다.
  - 승계 기록 `succession.json`(`mapping-succession-v1`): 대상 ID, 기준 ID·활성 mapping 버전·mapping SHA-256·schema fingerprint, 대상 schema fingerprint, `carried_fields`, `stale_fields`(field·영향 상태), `initiated_by`, `initiated_at`. 새 파일로만 쓰고 덮어쓰지 않는다.
  - 순서: draft 교체 → 승계 기록 생성. 중간에 끊긴 경우 같은 요청 재시도는 draft가 이 승계 결과와 정확히 같으면 기록만 마저 쓴다(idempotent). 승계 기록이 이미 있으면 같은 기준·버전·field 분류일 때만 그대로 반환하고, 다르면 `REVIEW_CONFLICT`.
  - review 디렉터리 파일 허용 목록에 `succession.json`을 추가한다. 이 파일은 실행 권한과 무관하며 `load_activation`은 내용을 쓰지 않는다.
- `load_succession(repository_root, database_id) -> MappingSuccession | None`: strict load, 대상 ID·fingerprint가 현재 등록과 맞지 않거나 손상되면 None(표시만 생략).
- publish는 바뀌지 않는다: 모든 field 검토 완료, 실행 가능성 검사를 그대로 요구한다. 승계 field도 관리자가 다시 remap·approve할 수 있다.

### 2. HTTP

- `POST` action `CARRY_OVER`: `{"action","database_id","base_database_id","reviewer"}` strict body, 성공 시 기존 detail 응답.
- detail(GET·POST 응답)에 승계 기록이 있으면 `succession`: `{base_database_id, base_mapping_version, carried_fields, stale_fields: [{semantic_field, status}]}`. 없으면 key 없음. 기존 detail shape는 그대로다.
- 고정 오류 매핑: 같은 ID·형식 오류 400, 기준 미활성·대상 없음 404, 이미 검토 중·게시됨·기록 불일치 409, ontology 불일치 422(`PUBLISH_BLOCKED`, fields 없음).

### 3. frontend

- `mappingReview.ts`: detail parser가 선택 `succession`을 strict 검증(ID 형식, 버전, carried·stale 서로소, stale 상태 값, 모든 이름이 detail field에 존재), `carryOverReview` client.
- `MappingReviewPanel.tsx`: 검토 시작 전 화면에 "기준 DB mapping 승계로 시작"(기준 선택 + 검토자 필수). 검토 화면에서 stale field에 `STALE · 재검토 필요(사유)`, 승계 field에 `승계됨` 표시, 상단에 승계 요약.
- `MainPage.tsx`: 패널에 registry의 private 활성 DB 목록(자기 자신 제외)을 전달하는 prop 하나만 추가.

## 제외

- 기준 등록을 superseded로 표시하거나 registry에서 자동 교체하는 일(명시적 publish 뒤 관리자가 선택).
- PostgreSQL live schema drift의 `STALE` 연결(현재는 `is_postgres_context_current`로 fail closed).
- 이름 변경 추정 승계, 코드값 비교.

## 보안 계약(유지)

미게시 draft 실행 금지, 승계는 draft만 바꾸고 실행 권한을 만들지 않음, 명시적 publish만 실행 권한 부여, 등록·mapping·activation 불변, sample·행·경로·SQL 비노출, 고정 오류 code.

## 테스트

- 같은 schema: 전 field 승계, stale 없음, 바로 publish 가능하고 publish 결과 mapping이 기준과 같은 column.
- column 이름 변경: 해당 field만 stale·PENDING, publish는 `PUBLISH_BLOCKED`(field 목록에 stale field), remap 뒤 publish 성공.
- 구조 변경(flat→hybrid): 전 field stale.
- 전제 위반: 같은 ID, 기준 미게시, 대상 게시됨, 검토 행위가 있는 draft, 다른 기준으로 재승계는 거부; 같은 요청 재시도 idempotent; draft만 쓰이고 기록이 없는 중단 상태 복구.
- 손상된 승계 기록은 detail에서 생략되고 activation·registry에 영향 없음.
- HTTP strict body, detail `succession` shape, 경로 비노출.
- frontend parser 거부 사례와 client body.

---

<a id="prompts-phase8-04-test-query-preview"></a>

## 원본: prompts/phase8-04-test-query-preview.md

# Phase 8-04 구현 지침: 게시·활성 mapping의 고정 test query 결과 미리보기

## 배경

8-01~8-03으로 관리자는 mapping을 게시·재활성화·승계할 수 있다. 게시 뒤 "이 mapping이 실제로 그럴듯한 결과를 내는가"를 확인할 방법이 없다. 8-04는 **서버가 소유한 고정 합성 test query**를 **게시·활성화된 private 등록**에서만 기존 실행 경로로 돌려 결과 일부를 보여준다. 결과 hash는 8-05(verified query 회귀)에서 버전 간 비교에 쓴다.

## 범위

### 1. 고정 test query 집합 (`admin-test-queries-v1`)

서버 코드에 고정된 QueryIR 5개(filter 없음, limit 200, canonical field만):

| ID | entity | select |
| --- | --- | --- |
| `offerings` | CourseOffering | Course.code, Course.name, CourseOffering.section, Instructor.name |
| `courses` | CourseOffering | Course.code, CourseOffering.section, Course.name, Course.credits, Course.category |
| `departments` | CourseOffering | Course.code, CourseOffering.section, Department.code, Department.name |
| `schedules` | CourseOffering | Course.code, CourseOffering.section, Schedule.day, Schedule.start_time, Schedule.end_time |
| `seats` | CourseOffering | Course.code, CourseOffering.section, CourseOffering.capacity, CourseOffering.enrolled |

compiler는 entity가 아니라 select field의 JOIN tree로 행 단위를 정하므로(합성 확인: 과목 field만 select하면 flat 6행, normalized·coded 5행; 학과만이면 3행), 모든 query가 개설강좌 단위 field(`CourseOffering.section`)를 포함해 행 단위를 개설강좌로 고정한다. 그래야 같은 seed의 schema 간·버전 간 결과를 비교할 수 있다.

브라우저는 query ID만 고른다. SQL·QueryIR·filter 값을 보낼 수 없다.

### 2. 실행 (`backend/adaptive/api/admin_test_queries.py`, 신규)

- 대상은 strict `load_activation`이 성공하는 게시·활성 private 등록뿐이다(검토 대기·미게시·fixture·PostgreSQL 제외).
- 기존 `ExecutionHarness`(고정 IR을 돌려주는 `FakeModel`, `SchemaAdaptiveSQLiteCompiler`, `SchemaAdaptiveAstPolicy`, 활성 DB 경로만 가진 `SQLiteReadOnlyExecutor`)로 QueryIR 검증→compile→AST policy→read-only 실행을 모두 거친다.
- 결과 행은 JSON scalar(문자열·정수·유한 실수·bool·null)만 허용하고 BLOB 등은 실패로 처리한다. 행을 값 기준으로 결정적 정렬한 뒤 최대 5행만 반환하고, 전체 행 수(`row_count`, 200에서 잘리면 `row_count_capped`)와 정렬된 전체 결과의 SHA-256(`result_sha256`, NULL과 문자열 구분)을 함께 준다.
- column은 semantic field 이름으로만 반환한다. SQL·물리 table/column 이름·경로는 반환하지 않는다.

### 3. HTTP

- `GET /api/adaptive/v1/registrations/test-queries`: 고정 목록 `{version, queries: [{query_id, title, entity, select}]}`.
- `GET /api/adaptive/v1/registrations/test-query?database_id=..&query_id=..`: 두 key만 허용. 응답 `admin-test-query-result-v1`: `database_id, mapping_version, query_id, columns, rows, row_count, row_count_capped, result_sha256`.
- 고정 오류: `ADMIN_TEST_QUERY_INVALID_REQUEST`(400, 형식·알 수 없는 query), `ADMIN_TEST_QUERY_NOT_FOUND`(404, 활성 게시 등록 아님), `ADMIN_TEST_QUERY_FAILED`(422, 실행 경로 실패; harness 고정 오류 code만 `failure_code`로 동봉), `ADMIN_TEST_QUERY_INTERNAL_ERROR`(500).

### 4. frontend

- `src/adaptive-prototype/testQueries.ts`: strict parser·client.
- `TestQueryPanel.tsx`: `MappingReviewPanel`의 게시 상태이고 활성 버전이 있을 때 표시. query 선택 → 실행 → 결과 표(최대 5행)·전체 행 수·실행 mapping 버전. 재활성화·publish 뒤에는 결과를 비운다.

## 제외

- 임의 QueryIR·filter 입력, 비활성 버전 미리보기, 자연어 질문 실행.
- verified query 저장·회귀 비교(8-05).
- 합성 fixture·PostgreSQL 대상.

## 보안 계약(유지)

미게시 mapping 실행 금지, 활성 mapping만 실행, 기존 QueryIR 검증·compiler·AST policy·read-only executor 우회 없음, raw SQL 입력 없음, SQL·물리 이름·경로·traceback 비노출, 행 수 제한. 결과 값은 기존 catalog 화면과 같은 수준으로 관리자에게 보이므로 실제 학교 DB에서는 L2(연결 전)·L5(실데이터 시연 전) 관문을 그대로 적용한다.

## 테스트

- flat·coded·hybrid 합성 등록을 게시한 뒤 5개 query 모두 seed 기대 행 수·결과와 일치, 버전 간 hash 비교(같은 mapping → 같은 hash, 다른 column mapping → 다른 hash).
- 결정적 정렬과 5행 제한, NULL과 'NULL' 문자열 hash 구분.
- 미게시·검토 대기·없는 DB 404, 알 수 없는 query·추가 key 400, 재활성화 뒤 새 활성 버전 사용.
- 응답에 SQL·물리 이름·경로 없음.
- frontend parser 거부 사례와 client URL.

---

<a id="prompts-phase8-05-verified-query-regression"></a>

## 원본: prompts/phase8-05-verified-query-regression.md

# Phase 8-05 구현 지침: verified query 기준선과 버전 회귀 비교

## 배경

8-04는 활성 mapping에서 고정 test query를 실행해 결과 일부와 순서 무관 결과 hash를 보여준다. 8-05는 관리자가 **확인한 결과를 기준선(verified)으로 기록**하고, publish·재활성화 전후에 같은 query를 다시 실행해 **결과가 바뀐 query를 표시**하는 회귀 비교를 추가한다. 회귀 결과는 경고 정보이며 publish·재활성화를 자동으로 막거나 실행하지 않는다.

## 범위

### 1. 게시 버전 로드 (`backend/adaptive/registration/reviews.py`)

- `load_published_version(repository_root, database_id, mapping_version) -> LoadedActivation`형 결과: `current.json`의 게시 digest 기록에 있는 버전이고 파일 bytes가 digest와 같으며, 재활성화와 같은 검사(`_check_activatable`: 등록 fingerprint, 현재 ontology, canonical coverage, 실행 가능성)를 통과할 때만 반환한다. 파일을 쓰지 않고 activation을 바꾸지 않는다.

### 2. verified 기준선 (`verified-queries.json`, `verified-queries-v1`)

- review 디렉터리의 관리자 metadata(실행 권한과 무관). 항목: `query_id`, `mapping_version`, `result_sha256`, `row_count`, `verified_by`, `verified_at`. query ID당 최신 1개. 행 값은 저장하지 않는다.
- `verify_test_query(repository_root, database_id, query_id, verified_by)`: **서버가 활성 버전에서 query를 실행한 결과 hash**를 기록한다(클라이언트가 hash를 보내지 않는다). per-DB lock, 원자적 교체.
- 파일 허용 목록에 `verified-queries.json`을 추가한다. 손상된 기준선은 없는 것으로 보고 회귀를 `UNVERIFIED`로 표시한다(실행에는 영향 없음).

### 3. 회귀 비교 (`backend/adaptive/api/admin_test_queries.py`)

- `run_regression(repository_root, database_id, mapping_version)`: 모든 고정 query에 대해 상태를 계산한다.
  - `MATCH`: 기준선 hash와 같음, `CHANGED`: 다름(행 수 함께), `FAILED`: 실행 경로 실패(harness 고정 code), `UNVERIFIED`: 기준선 없음(실행 결과 hash는 표시).
  - 대상 버전은 활성 버전 또는 게시 digest 기록이 있는 비활성 버전(재활성화 전 확인용). 고정 query만, 같은 실행 경로로 실행한다.
- HTTP:
  - `POST /api/adaptive/v1/registrations/verified-queries` `{"action":"VERIFY","database_id","query_id","verified_by"}` → 갱신된 기준선 목록.
  - `GET /api/adaptive/v1/registrations/regression?database_id=..&mapping_version=..` → `verified-query-regression-v1`: 대상 버전, 활성 버전, query별 상태·기준 버전·hash·행 수, 요약(match/changed/failed/unverified 수).
  - 고정 오류: 형식 400, 활성·게시 버전 아님 404, 실행 전 오류 500, lock 충돌 409.

### 4. frontend

- `testQueries.ts`: 기준선·회귀 parser와 client.
- `TestQueryPanel.tsx`: 결과 옆 "이 결과를 verified로 기록"(검토자 필수).
- `MappingReviewPanel.tsx`: 버전 이력의 비활성 버전에 "회귀 확인"(재활성화 전 비교), publish·재활성화 직후 새 활성 버전 회귀 자동 실행과 `CHANGED`·`FAILED` 경고 표시.

## 제외

- 회귀 결과로 publish·재활성화를 자동 차단하거나 자동 되돌리기.
- 임의 query의 verified 등록, 등록 간(승계 기준 DB ↔ 새 DB) 비교.
- 미게시 draft에서의 실행.

## 보안 계약(유지)

미게시 draft 실행 금지, 고정 QueryIR만 기존 검증·compiler·AST policy·read-only executor로 실행, 비활성 게시 버전 실행은 관리자 회귀 비교의 고정 query에 한하고 사용자 질의 실행 권한(activation)은 바꾸지 않음, 행 값 저장 없음, SQL·물리 이름·경로 비노출, 고정 오류 code.

## 테스트

- verify 뒤 같은 버전 회귀는 전부 `MATCH`; 과목명 column을 바꾼 v2는 해당 query만 `CHANGED`, 나머지 `MATCH`; 재활성화 전 비활성 v1 회귀 가능.
- 기준선 없는 query `UNVERIFIED`, 손상된 기준선은 `UNVERIFIED`로 처리하고 activation·registry 영향 없음.
- digest 기록에 없는 orphan·교체된 버전, 미게시·검토 대기 DB는 404. 클라이언트가 hash를 보낼 수 없음(추가 key 400).
- 파일 허용 목록 회귀: 기준선이 있어도 `load_activation`·INIT·publish 정상.
- frontend parser·client 거부 사례.

---

<a id="prompts-phase8-06-value-dictionary"></a>

## 원본: prompts/phase8-06-value-dictionary.md

# Phase 8-06 지침: 코드값·값 사전(value dictionary)

작성일: 2026-09-30. 사용자 승인 개발 순서 2번.

## 1. 목적

같은 의미의 값이 DB마다 다르게 저장된다.

- 형식 차이: legacy DB의 이수구분은 `전공(핵심)`인데 사용자와 모델은 `전공핵심`이라고 쓴다.
- 코드 차이: 다른 학교는 이수구분을 `01`·`02` 같은 코드로 저장할 수 있다. 이런 코드는 사용자가 알 수 없다.

7-12·7-16의 닫힌 목록 변환은 **형식이 규칙적인 값**(학점 `3/3/0`, 시간, 요일)만 다룬다. 이수구분처럼 **값 목록 자체가 DB마다 다른 경우**는 규칙으로 풀 수 없다. 사람이 검토한 대응표(값 사전)가 필요하다.

이 단계는 이후 합성 변형 benchmark와 AI mapping 제안이 "무엇을 제안해야 하는가"의 형식을 정한다. AI가 제안하더라도 결과물은 이 값 사전 형식이고, 게시 전에 사람이 검토한다.

## 2. 핵심 결정

### 2.1 값 사전은 field별로 선언하는 닫힌 대응표다

`ValueDictionary`(검증 모델, 추가 key 금지):

- `entries`: 1~64개. 각 항목의 구성은 다음과 같다.
  - `stored`: DB에 저장된 실제 값.
  - `label`: 사용자에게 보일 이름.
  - `aliases`: 사용자가 쓸 수 있는 다른 표기, 최대 16개.
- `unknown`: 사전에 없는 값을 어떻게 처리할지 정한다.
  - `pass`: 그대로 쓴다. 이름형처럼 목록 밖 검색도 의미가 있는 경우다.
  - `reject`: "이해 못 함"으로 거부한다. 코드 column처럼 목록 밖 값이 무의미한 경우다.
- `reviewer`, `reason`: 누가 왜 선언했는지 남긴다.

조회 key는 NFKC 정규화, 소문자화, 공백 제거를 거친 문자열이다. `stored`, `label`, `aliases`가 모두 key가 된다.

- 한 key는 정확히 한 `stored`로만 이어져야 한다. 충돌하면 사전 자체를 거부한다.
- `stored` 값은 서로 달라야 한다.

### 2.2 적용 위치

- **filter 값**: compiler와 AST policy가 같은 함수 `bind_filter_value`로 값을 정한다.
  - filter 값 전체가 사전 key와 일치하면 `stored`로 바꾼다.
  - 일치하지 않으면 `unknown` 설정에 따라 그대로 쓰거나 거부한다.
  - 연산자(`equals`, `not_equals`, `contains`)는 바꾸지 않는다. `contains`는 바뀐 값을 escape해서 감싼다.
  - 부분 문자열 치환은 하지 않는다. 값 전체가 key일 때만 바꾼다.
- **결과 값**: runtime이 결과 행의 `stored`를 `label`로 바꾼다. 코드 값이 그대로 사용자에게 보이지 않게 하기 위해서다.
- **챗봇 이해 조건 표시**: 사전으로 바뀐 값은 `label`로 보여준다.

### 2.3 선언 위치와 검증

- private PostgreSQL profile `postgres-runtime-profile-v4`를 추가한다.
  - v2 key + `anchor_table`(문자열 또는 null) + `value_dictionaries`(field → 사전)로 구성한다.
  - v1·v2·v3은 그대로 읽는다.
- 시작 검증:
  - 사전 field는 매핑된 field여야 한다.
  - ontology 타입이 string이어야 한다.
  - 값 변환(transform)과 같은 field에 둘 수 없다.
  - 매핑 column의 선언 타입이 문자열 계열이어야 한다.
  - 시작 검증은 DB 값을 읽지 않는다(sample-less 유지).
- 등록 CLI(`legacy_view_profile --write --live`)는 각 `stored` 값이 실제 column에 있는지 read-only로 확인한다. 없으면 `VALUE_MISSING`으로 등록하지 않는다. 출력은 고정 code뿐이다.

### 2.4 legacy 적용

- 두 legacy profile(`legacy-view`, `legacy-base`)에 `Course.category` 사전을 선언한다(`unknown: pass`).
  - 실제 값 9개(`전공(핵심)`, `전공(기초)`, `전공(심화)`, `교양(소양)`, `교양(균형)`, `교양(소양필수)`, `교양(기초)`, `일반선택`, `교직`)를 등록한다.
  - label은 값 그대로다.
  - 괄호 없는 표기(`전공핵심` 등)와 앞뒤를 바꾼 표기(`핵심전공` 등)를 aliases로 둔다.
- 두 profile을 v4로 다시 쓴다(`--replace`). fingerprint는 바뀌지 않는다.

## 3. 검증

1. 단위 테스트:
   - 사전 모델 검증(key 충돌, 중복 stored, 크기).
   - 조회 규칙(공백·전각 정규화).
   - `pass`와 `reject` 동작.
   - compiler와 policy의 bind 일치, policy가 치환 전 값을 거부하는지.
   - 결과 decode, 챗봇 표시와 `reject` 시 `NOT_UNDERSTOOD`.
   - v4 엄격 load.
   - 시작 검증 거부 조건.
   - 등록 시 `VALUE_MISSING`.
   - 코드값 합성 예(`01` → `전공`)의 입력·출력 양방향.
2. 실제 legacy DB:
   - 두 profile을 다시 등록하고 smoke를 실행한다.
   - 같은 30문항 live 비교를 `legacy-base`로 실행한다.
   - 기대: q06 개선, 다른 문항은 변화 없음.
3. 전체 pytest, compileall, `git diff --check`를 실행한다.

## 4. 범위 밖

- FK 없는 관계 선언(관계 교정). 합성 변형 benchmark에서 필요해지면 별도 단계로 한다.
- 관리자 화면에서 값 사전 편집(8-07).
- SQLite 경로와 게시 mapping 계약(8-01~8-05)에 값 사전을 넣는 일.
- 하나의 표기가 여러 저장 값을 뜻하는 경우(1:N). 예: `전공` → 세 값. 이 경우는 지금처럼 `contains`로 처리한다.
- 모델·prompt 변경.
- 학년 등 ontology 확장.

## 5. 지켜야 할 제약

- private profile은 Git 비추적 경로에만 둔다.
- DSN·SQL·물리 이름은 출력하지 않는다.
- 모델 호출은 1회이며 raw SQL fallback은 없다.
- 실행은 read-only이고 행 상한 200을 유지한다.
- legacy 시스템은 바꾸지 않는다.
- 사람 관문은 승인으로 표시하지 않는다. 이 사전은 팀이 구성한 legacy DB에 대한 개발자 선언이며 L2 검토가 아니다.
- push하지 않는다.

---

<a id="prompts-phase8-07-value-dictionary-editor"></a>

## 원본: prompts/phase8-07-value-dictionary-editor.md

# Phase 8-07(축소판) 지침: 값 사전 편집(코드표·약어)과 약어 후보

작성일: 2026-09-30. 사용자 결정: 8-07 진행. 프로토타입 우선(ADR-078). 약어는 자동 후보 + 관리자 승인(ADR-080).

## 1. 범위

- **포함:** 화면에서 등록한 PostgreSQL DB의 값 사전 편집. 게시할 때 profile에 적용한다(ADR-074의 `ValueDictionary`).
  - **코드표:** 저장 값 → 뜻(label). 예: `01` → `전공`.
  - **약어:** 저장 값 → 사용자가 쓰는 다른 말. 예: `공과대학 컴퓨터융합학부` ← `컴융`.
  - **약어 후보:** 규칙과 로컬 AI로 만들고, 충돌 검사를 붙인다. 관리자가 골라 추가한다.
- **제외 (후속, 문서에 기록):**
  - 업로드 SQLite DB의 부분 게시·변환·값 사전·QueryIR 챗봇. SQLite 실행 경로 확장이 필요하다.
  - PostgreSQL 등록의 버전 되돌리기와 test query 미리보기.
  - 공통 약어 묶음, 사용 기록 기반 제안.

## 2. 편집 대상 field

draft에서 승인·재지정된 field 가운데 다음을 모두 만족하는 것이다.
- ontology type이 문자열이다.
- column type이 text 계열이다.
- 식별 field가 아니다(`Course.code`, `CourseOffering.section`, `Department.code`).

runtime 규칙과 같다. 서로 다른 저장 값은 최대 500개까지 읽는다. 넘으면 저장 값 목록은 보여 주지 않고, 저장 값을 직접 입력하게 한다(존재 확인은 게시 때 한다).

## 3. 저장

- 편집 중인 사전은 `.schema-adaptive/postgres-registrations/<id>/dictionaries.json`에 둔다(draft 성격, 원자적 교체).
- 항목은 `stored`, `label`, `aliases`이며 기존 `ValueEntry` 규칙을 따른다.
  - 공백·제어 문자 금지, 100자 이하, 약어 16개 이하, field당 64항목 이하.
  - 한 표기가 두 저장 값으로 이어지면 거부한다.
- 모르는 값은 그대로 통과(`unknown=pass`)한다.

## 4. 약어 후보

- **규칙(한국어 공통):** 저장 값을 문자·숫자가 아닌 곳에서 단어로 나눈다. 단어가 모두 2·4글자이고 2글자 묶음이 2~3개일 때, 묶음마다 첫 글자를 이어 붙인다.
  - 예: `전공(핵심)` → `전핵`, `일반선택` → `일선`, `교양(소양필수)` → `교소필`.
  - 코드에 DB 값은 없다.
- **AI(선택):** 버튼을 눌렀을 때만 로컬 mapping 모델을 부른다(JSON schema로 key는 저장 값). 저장 값마다 학생들이 쓰는 약어를 최대 2개 받는다. 저장 값 60개씩 나눠 부른다.
- **충돌 검사:**
  - **막음:** 다른 저장 값·그 단어 조각·다른 항목의 표기와 같을 때(모든 편집 대상 field 기준).
  - **경고만:** 다른 text column(과목명·교원명 등, 최대 5,000개)의 값 속에 들어 있을 때. 예: `전기` → `전기회로`.
- 후보는 저장하지 않는다. 관리자가 "추가"를 눌러야 항목이 된다.

## 5. 게시

- 사전이 있는 field는 변환을 붙이지 않는다. runtime이 둘을 함께 허용하지 않는다.
- 게시 때 각 `stored`가 실제 저장 값에 있는지 다시 확인한다. 500개 이하 field만 확인할 수 있으며, 없으면 막는다.
- profile v4 `value_dictionaries`에 넣고 runtime 검증을 통과해야 쓴다.
- 이미 게시된 DB는 사전을 고친 뒤 다시 게시하면 새 버전으로 적용된다.

## 6. API와 화면

- `GET /api/adaptive/v1/postgres/dictionaries?database_id=`: field별 저장 값·항목.
- `POST` 같은 경로:
  - `SET`: 항목 추가·수정. 응답에 경고를 담는다.
  - `REMOVE`: 항목 삭제.
  - `SUGGEST`: 후보와 충돌·경고. `ai: true`면 AI도 부른다.
- 화면: PostgreSQL 등록 결과 아래에 "값 사전(코드표·약어)" 영역을 둔다.
  - field 선택 → 저장 값 선택 → 뜻·약어 입력 → 저장.
  - 약어 후보 받기(규칙 / 규칙+AI) → 후보별 "추가".
  - "사전 적용해 게시" 버튼.

## 7. 테스트와 확인

- 단위:
  - 규칙 후보, 충돌 막음·경고, 편집 대상 판정.
  - 저장·strict reload, 표기 충돌 거부.
  - 게시 시 사전 적용과 변환 제외, 저장 값 없음 차단.
- API: SET·REMOVE·SUGGEST, 잘못된 입력.
- frontend parser·client.
- live: 합성 schema에서 코드값 column(예: 영어 요일)에 코드표를 넣고 게시한 뒤, "월요일 수업"이 답하는지 확인하고 정리한다.

## 8. 제약

기존 제약을 그대로 따른다.
- DSN 비노출, 읽기 전용, 게시된 mapping만 실행.
- 모델 호출은 관리자 버튼으로만 한다.
- 사람 관문을 표시하지 않고, push하지 않는다.

---

<a id="prompts-phase8-08-variant-benchmark"></a>

## 원본: prompts/phase8-08-variant-benchmark.md

# Phase 8-08 지침: 합성 변형 benchmark와 규칙 기반 mapping 기준선

작성일: 2026-09-30. 사용자 승인 개발 순서 3번. 사용자 승인: 로컬 PostgreSQL에 전용 DB `schema_adaptive_synthetic` 생성(2026-09-30).

## 1. 목적

AI mapping 제안(다음 단계)을 공정하게 평가할 고정 benchmark를 먼저 만든다. 같은 의미의 수강 데이터를 여러 방식으로 저장한 합성 PostgreSQL 변형을 두고, mapping 제안이 얼마나 맞는지 측정한다. 이번 단계는 benchmark와 현재 규칙 기반 aligner의 기준 점수까지다.

- benchmark는 AI 구현 전에 고정한다. AI 결과를 보고 benchmark를 고치지 않는다.
- 규칙 기반 기준선은 **현재 aligner 그대로** 측정한다. benchmark에 맞춘 새 규칙은 만들지 않는다.

## 2. 구성

### 2.1 seed

- `variant-benchmark-seed-v1`의 구성:
  - 개설강좌 12개, 학과 4개, 교원 6명, 과목 9개.
  - 연도 2개, 학기 2개, 이수구분 3개(`전공`, `교양`, `일반선택`), 요일 6개(한 글자 `월`~`토`).
- 표준 표현은 runtime 출력 형식과 같다: 학점은 숫자, 시간은 `HH:MM`, 요일은 한 글자.
- 기존 fixture seed(`MON` 등)와 40문항 세트는 바꾸지 않는다.

### 2.2 변형(12개, 축별)

| 축 | 변형 | 차이 |
|---|---|---|
| 구조 | `base` | 정규화 5 table, 영어 이름, 표준 값. 대조군 |
| 구조 | `flat` | 한 table |
| 구조 | `optional-schedule` | 개설강좌 2개에 시간표가 없음. 기준 table(anchor) 필요 |
| 형식 | `credits-slash` | 학점이 `3/3/0` 문자열 → `SLASH_FIRST_NUMERIC` |
| 형식 | `time-type` | 시간이 PostgreSQL `time` → `TIME_TO_HHMM` |
| 코드값 | `category-code` | 이수구분이 `M`·`G`·`E` → 값 사전 |
| 코드값 | `weekday-english` | 요일이 `MON`… → 값 사전 |
| 코드값 | `semester-code` | 학기가 `10`·`20` → 값 사전 |
| 의미 | `abbreviated` | 약어 이름(`tb_sbjt.sbjt_nm`, `yoil` 등) |
| 의미 | `grade-trap` | `course_year`는 학년(1~4)이고 연도 column이 없다 → 연도 미지원 |
| 의미 | `capacity-trap` | `capacity`는 강의실 정원이고 수강 정원은 `enroll_limit` |
| 복합 | `legacy-like` | 학점 문자열, `time`, 영어 요일, 시간표 없는 강좌, 학년 column |

### 2.3 정답(gold)

변형 spec의 column 주석에서 자동으로 만든다. field마다 다음을 둔다.

- 위치: table·column, 또는 미지원. 다른 table의 같은 값을 가리키는 FK column(예: 개설강좌의 과목 코드)도 정답으로 인정한다.
- 값 변환: 허용 집합. 한 글자 요일 column은 변환 없음과 `KOREAN_WEEKDAY`를 모두 인정한다.
- 값 사전: 저장 값 → 표준 값.
- 변형 단위 기준 table: 개설강좌 table. 한 table 변형은 그 table 또는 없음.

## 3. 채점

### 3.1 mapping 답안

답안(`VariantMappingAnswer`)은 AI 제안의 출력 형식이기도 하다.

- field별 위치, 값 변환, 값 사전.
- 기준 table.

### 3.2 field 단위 점수(DB 불필요)

- column, 변환, 값 사전, 세 가지 모두의 정답 수를 셌다(변형당 15 field).
- 값 사전은 두 조건을 모두 만족해야 정답이다.
  - 정답의 모든 표준 값이 제안 사전에서 정답 저장 값으로 풀린다.
  - 제안 사전의 모든 저장 값이 정답 저장 값 안에 있다.
- 기준 table은 따로 센다.

### 3.3 실행 점수(live)

- 고정 gold QueryIR 20개를 쓴다. 모두 과목 코드와 분반을 select하여 개설강좌 단위 결과가 되게 한다.
- 답안으로 runtime profile을 만들고, 기존 compiler·AST policy·read-only executor로 실행한다.
- 결과 행 multiset이 seed 참조 평가(compiler와 무관한 Python 평가)와 같으면 정답이다.
- 정답에서 미지원인 field를 쓰는 질문은 거절(`MAPPING_MISSING`)이 정답이다.
- runtime이 답안을 거부하면(같은 column 중복, 타입 불일치 등) 그 변형의 질문은 모두 오답이다.
- 정답 답안(oracle)은 모든 변형에서 field·실행 모두 100%여야 한다. 이것이 benchmark 자체의 검증이다.

### 3.4 규칙 기반 기준선

- 현재 `RuleBasedSchemaAligner`를 쓴다(이름·타입, sample 없음).
- field마다 상위 후보 3개 중 이미 쓰인 column이 아닌 첫 후보를 신뢰도 순으로 고른다. 신뢰도가 검토 기준 0.60 미만이면 미지원으로 둔다.
- 변환, 값 사전, 기준 table은 제안하지 않는다(규칙에 없음).

## 4. 실행과 저장

- 오프라인: `python -m backend.adaptive.evaluation.variant_benchmark --check`.
  - spec 검증, gold 생성, oracle과 규칙 기준선의 field 점수를 계산한다.
  - DB 연결은 없다.
- live: `--live-synthetic`. 두 조건이 필요하다.
  - `SCHEMA_ADAPTIVE_SYNTHETIC_POSTGRES_DSN`: 7-03과 같은 loopback, `schema_adaptive_synthetic` 전용 검증.
  - label.
- live 실행의 흐름:
  - 변형마다 무작위 schema를 만들고 끝나면 지운다.
  - inspect한 catalog가 spec catalog와 구조적으로 같은지 확인한다.
  - oracle과 규칙 답안을 실행한다.
  - 규칙 답안은 live catalog로 다시 만들어 오프라인 답안과 같은지 확인한다.
- 결과 JSON은 `.schema-adaptive/reports/variant-benchmark/`(Git 비추적)에 둔다. stdout에는 집계만 낸다.
- 합성 자료이므로 결과 문서에는 변형별 점수와 실패 원인을 적는다.

## 5. 범위 밖

- AI 제안 구현(다음 단계).
- sample 값을 쓰는 규칙 추가.
- FK 없는 관계, 1:N 값, 표현할 수 없는 형식(예: `0900`).
- SQLite 경로.
- 실제 DB.
- 질문의 자연어 이해. 이번 평가는 gold QueryIR을 그대로 실행하므로 모델 호출이 없다.

## 6. 제약

- 합성 전용 DB와 무작위 schema만 쓰고 정리한다. legacy DB는 건드리지 않는다.
- DSN과 driver 오류 문구는 출력하지 않는다.
- 실행은 기존 read-only executor를 거친다.
- 사람 관문을 승인으로 표시하지 않는다. gold는 개발자 작성이며 L1/L3 전이다.
- push하지 않는다.

---

<a id="prompts-phase8-09-ai-mapping-proposal"></a>

## 원본: prompts/phase8-09-ai-mapping-proposal.md

# Phase 8-09 지침: AI mapping 제안과 규칙 기반 비교

작성일: 2026-09-30. 사용자 승인 개발 순서 4번.

## 1. 목적

8-08 benchmark(12개 변형, 고정)에서 로컬 LLM이 낸 mapping 제안을 규칙 기반과 같은 방식으로 채점한다. benchmark·정답·질문은 바꾸지 않는다.

AI의 역할은 **제안**까지다. 제안은 검증기를 통과한 뒤에도 사람 검토 전에는 게시되지 않는다. 이번 단계는 평가만 하며 runtime 게시 경로는 바꾸지 않는다.

## 2. 제안 파이프라인

모델은 로컬 Ollama `qwen2.5-coder:7b-instruct`(범용 instruct)를 쓴다.

- 이유: 챗봇용 fine-tune 모델은 QueryIR 생성 전용이다.
- 연결: 기존 loopback 전용 transport(proxy·redirect 차단, 응답 크기 상한)를 쓴다.
- 생성 설정: temperature 0, JSON schema 제한 출력.

단계는 다음과 같다.

1. **column 선택(변형당 1회)**
   - 입력:
     - field 15개의 이름·설명·타입.
     - 모든 column의 `table.column`·선언 타입·저장 값 예시(최대 5개).
   - 출력: field마다 후보 column 하나 또는 `none`. schema의 enum이 실제 column만 허용한다.
   - 두 field가 같은 column을 고르면 표준 field 순서상 먼저인 field만 남긴다.
2. **값 처리(매핑된 field마다 1회)**
   - 입력:
     - field 설명·타입, column 선언 타입, 저장 값 예시(최대 10개).
     - 닫힌 목록 변환의 설명.
     - 표준 값 목록(아래)이 있는 field는 그 목록.
   - 출력: 변환 하나 또는 `none`. 표준 값 목록이 있는 field는 `{stored, canonical}` 쌍의 값 사전도 낸다. schema의 enum이 저장 값과 표준 값만 허용한다.
3. **결정적 검증기**(runtime과 같은 규칙):
   - 타입이 맞지 않는 변환은 버린다(ontology 타입·column 선언 타입).
   - 값 사전은 다음 경우에 버린다.
     - 식별 field이거나 column이 문자열이 아닐 때.
     - 모든 항목이 저장 값 = 표준 값(무의미)일 때.
     - 저장 값이나 표준 값이 목록 밖일 때, 또는 key가 중복될 때.
   - 변환과 사전을 함께 내면 변환만 남긴다.
4. **기준 table**: `CourseOffering.section`이 매핑된 table로 정하는 결정적 규칙이다(개설강좌 단위). AI가 고르지 않는다.
   - 같은 규칙을 규칙 기반에도 붙인 비교군(`rules+anchor`)을 둔다. 그래야 AI의 기여와 anchor 규칙의 기여를 나눌 수 있다.

### 표준 값 목록 `canonical-values-v1`

- 이수구분: `전공`·`교양`·`일반선택`
- 요일: `월`~`일`
- 학기: `1`·`2`

시스템이 쓰는 표준 표현이다. ontology YAML은 고치지 않는다(기존 평가 세트의 ontology fingerprint 보존). 제안기 입력으로만 둔다.

## 3. 비교군과 채점

- 비교군:
  - `oracle`(정답).
  - `rules`(8-08과 같음).
  - `rules+anchor`.
  - `ai`(검증기와 anchor 규칙 포함).
- 채점은 8-08과 같다: field 점수, 기준 table, live 질문 240개. oracle 만점이 아니면 실행 전체가 무효다.
- 입력 차이: 규칙은 이름·타입만 보고, AI는 저장 값 예시와 표준 값 목록도 본다. 이 차이를 결과에 적는다.
- 모델 원출력, 검증기가 버린 항목, 호출 수·시간은 Git 비추적 보고서(`.schema-adaptive/reports/variant-benchmark/`)에 둔다.
- AI 답안은 저장해 두었다가 DB 실행을 다시 할 때 재사용한다(모델을 다시 부르지 않음).

## 4. 범위 밖

- 제안을 관리자 검토 화면에 연결하는 일(8-07).
- 모델 fine-tune과 여러 모델 비교.
- prompt 반복 튜닝. prompt는 v1로 한 번 고정하고 실행한다. 결과를 보고 고치면 새 버전으로 기록한다.
- 실제 DB에 적용하는 일(L2 전 금지).

## 5. 제약

- 모델 호출은 loopback만 쓰고, 합성 catalog와 합성 값만 보낸다.
- DSN·비밀값을 prompt나 출력에 넣지 않는다.
- 합성 DB는 무작위 schema만 쓰고 정리한다.
- 사람 관문은 승인으로 표시하지 않는다. 결과는 개발자 작성 benchmark 위의 잠정치이며 L1/L3 전이다.
- push하지 않는다.

---

<a id="prompts-phase8-10-ai-mapping-v2"></a>

## 원본: prompts/phase8-10-ai-mapping-v2.md

# Phase 8-10 지침: AI mapping 제안 v2와 관리자 column 설명, 보류 변형 평가

작성일: 2026-09-30. 사용자 결정(8-09 결과 뒤): v2 → 8-11(ontology 확장 field·매개변수형 변환) → 8-07 순서.

## 1. 목적

8-09에서 드러난 약점 가운데 **값을 보면 알 수 있는 것**은 파이프라인을 개선해 줄인다. **사람만 아는 것**은 관리자가 DB를 등록할 때 주는 column 설명으로 반영한다.

결과를 보고 고치는 것이므로 기존 12개 변형만으로 평가하지 않는다. v2를 실행하기 **전에** 보류(held-out) 변형과 설명 작성 규칙을 고정하고, 두 세트 모두에서 v1.1과 비교한다.

## 2. 고정 순서(사전 등록)

1. 설명 작성 규칙과 보류 변형 6개, 모든 변형의 설명을 코드로 고정하고 테스트한다.
2. v2 파이프라인을 fake model 테스트로만 확인한다(실제 모델 호출 없음).
3. 실제 모델 실행:
   - v1.1은 보류 세트에서만 실행한다. 기존 세트는 8-09 결과를 쓴다.
   - v2와 v2+설명은 두 세트 모두에서 실행한다.
   - 각 실행은 한 번이며, 결과를 보고 파이프라인을 고치지 않는다.
4. live 실행을 한다. oracle이 만점이어야 유효하다.

## 3. 관리자 column 설명

### 3.1 형식

column마다 선택적으로 둔다.

- 한국어 설명 한 문장(최대 200자).
- 코드표: `저장 값=뜻` 쌍, 최대 32개.
- 제어·서식 문자는 거부한다.

### 3.2 작성 규칙(`admin-notes-v1`)

관리자는 다음 column에만 설명을 쓴다.

1. 이름만으로 뜻을 알기 어려운 약어.
2. 코드값 column. 이 경우 코드표도 쓴다.
3. 여러 값을 한 문자열에 합친 형식.
4. 이름과 실제 뜻이 다른 column.

지키는 점:

- 설명에는 canonical field 이름(`Course.credits` 등)과 변환 이름을 쓰지 않는다. 사람 말로 뜻만 쓴다.
- 코드표의 뜻은 관리자가 평소 쓰는 말로 쓴다(`10=1학기`, `MON=월요일`). 표준 값 목록에 맞추지 않는다.
- 설명 개수는 관리자 개입량으로 함께 보고한다.
- (리뷰 반영) 설명마다 규칙상 사유를 적고 테스트로 확인한다.
  - 사유는 `ABBREVIATION`, `CODE`, `PACKED`, `MISLEADING` 중 하나다.
  - `CODE`만 코드표를 가진다.
  - `PACKED`는 저장 값에 `/`가 있어야 한다.
  - `MISLEADING`은 정답 위치가 아닌 column이어야 한다.

### 3.3 안전

- 설명은 prompt에 JSON 데이터로 넣는다. 설명 속 지시문은 따르지 않도록 명시한다.
- 모델 출력은 지금처럼 schema와 검증기를 통과해야 한다.
- 설명은 제안의 근거일 뿐, 게시 전 검토를 대신하지 않는다.

## 4. 보류 변형 `variant-benchmark-holdout-v1`(6개)

기존 변형과 다른 이름 체계(`subject_master`, `lecture`, `lecture_time` 등)를 쓴다. seed와 질문 20개는 같다.

| 변형 | 축 | 차이 |
|---|---|---|
| `ho-weekday-full` | 코드값 | 요일이 `월요일`… → 값 사전 |
| `ho-category-numeric` | 코드값 | 이수구분이 `1`·`2`·`3` → 값 사전 |
| `ho-semester-text` | 코드값 | 학기가 `1학기`·`2학기` → 값 사전 |
| `ho-units` | 형식+의미 | 학점은 `unit_info`의 `학점/이론/실습` 문자열이다. `credit_hours`는 주당 수업 시간 합계(함정) |
| `ho-trap` | 의미 | `academic_year`는 학년(1~4)(함정)이고 개설 연도는 `open_yr`. `capacity`는 강의실 좌석(함정)이고 수강 정원은 `max_enroll` |
| `ho-mixed` | 복합 | `time` 타입, 영어 소문자 요일, 약어(`crs_ttl`, `instr`, `dpt_ttl`), 학점 문자열, 시간표 없는 강좌 |

이 변형은 8-09 결과를 본 뒤 만들었다. 그래서 실패 유형(코드값·함정·복합 형식)을 의식해 설계했다. 결과 문서에 이 점을 적는다.

## 5. v2 파이프라인(`ai-mapping-v2`)

v1.1에서 바뀌는 점:

1. **저장 값 일치 검사(검증기)**:
   - `KOREAN_WEEKDAY`는 예시 값이 모두 한 글자 요일일 때만 받는다.
   - `SLASH_FIRST_NUMERIC`는 예시 값이 모두 `숫자/숫자…`일 때만 받는다.
   - `TIME_TO_HHMM`은 지금처럼 time 타입일 때만 받는다.
2. **값 사전 전용 단계**:
   - 조건: 표준 값 목록이 있는 문자열 field이고, 예시 값 중 목록 밖 값이 있을 때만 부른다.
   - 방식: 예시 값마다 표준 값을 하나씩 고르게 한다. schema의 key는 예시 값, 값은 표준 값 enum이다.
   - 값 처리 호출에서는 사전을 더 이상 받지 않는다.
3. **설명 입력(`+notes` 비교군)**:
   - column 선택과 값 처리, 사전 호출에 해당 설명을 데이터로 넣는다.
   - 코드표의 뜻이 표준 값과 정확히 일치하고(공백·전각 정규화), 코드가 모두 예시 값에 있으면 모델 없이 사전으로 바로 쓴다. 아니면 모델에 참고로 준다.

기준 table 규칙, 중복 column 처리, 실패 호출 제외는 v1.1과 같다.

## 6. 비교군과 보고

- 비교군:
  - `oracle`.
  - `rules`.
  - `rules+anchor`.
  - `ai-v1.1`(기존 세트는 8-09 제안 재사용).
  - `ai-v2`.
  - `ai-v2+notes`.
- 두 세트(기존 12, 보류 6)별로 field 점수와 질문 정답을 보고한다.
- 설명 개수를 함께 보고한다.
- 원자료는 Git 비추적 보고서에 둔다.

## 7. 범위 밖

- ontology 확장 field, 이론·실습 시간(8-11).
- 관리자 화면에서 설명 입력(8-07).
- 모델 교체.

## 8. 제약

8-09와 같다. loopback 모델, 합성 자료만 쓴다. 사람 관문은 표시하지 않으며, push하지 않는다.

---

<a id="prompts-phase8-12-ai-proposal-upload"></a>

## 원본: prompts/phase8-12-ai-proposal-upload.md

# Phase 8-12 지침: 업로드 흐름에 AI mapping 제안 연결

작성일: 2026-09-30. 사용자 결정: 합성 benchmark 개선을 멈추고 프로토타입을 완성한다. 이론·실습 시간(8-11)은 넣지 않는다.

## 1. 배경과 범위 조정(프로토타입 타협)

8-08~8-10은 원래 table을 변형한 합성 benchmark의 점수를 올리는 데 집중했다. 그 결과 제가 만든 함정에 맞추는 과적합이 생겼다(예: `capacity`를 강의실 좌석으로 둔 함정, 세트마다 뜻이 다른 `credit_hours`). 그래서 다음처럼 범위를 조정한다.

- 멈추는 것:
  - benchmark 개선과 새 보류 세트.
  - 8-11 일반화 설계(DB별 ontology 확장, 매개변수형 변환, 화면 field 목록 일반화).
  - prompt 개선, 추가 모델 비교, Phase 9.
- 프로토타입 완성 순서:
  1. **8-12(이번):** 업로드한 DB의 mapping 초안에 AI 제안을 연결한다.
  2. **8-07 축소판:** 최소 관리자 화면.
  3. **처음부터 끝까지 browser smoke와 시연 절차.**

## 2. 목표

관리자가 SQLite DB를 업로드하면 로컬 모델이 canonical field마다 column을 고르고, 그 column이 검토 화면의 첫 후보가 된다. 모델이 없거나 실패하면 지금처럼 규칙 기반 후보만 쓴다. 검토·게시 절차와 실행 경로는 바꾸지 않는다.

## 3. 설계

### 3.1 무엇을 쓰는가

- 8-10 v2의 **column 선택 단계만** 쓴다. prompt 문장은 v2와 같다(`ai-mapping-v2/columns`).
- 값 처리(변환, 값 사전)는 쓰지 않는다. 업로드 DB의 게시 mapping은 지금 `table.column`만 담기 때문이다.
  - 변환·값 사전의 업로드 지원은 8-07 코드표 입력과 함께 다룬다.
- 모델: 로컬 loopback Ollama, 기본 `qwen2.5-coder:7b-instruct`, temperature 0, JSON schema 제한 출력. 한 번 업로드에 한 번 호출한다.

### 3.2 입력

- 모델 입력은 column별로 이름, 선언 type, 저장 값 예시(최대 10개, 100자 이하)다.
- 예시 값은 등록 중 메모리에서만 읽는다. catalog·proposal 파일과 HTTP 응답에는 넣지 않는다.
  - 기존 규칙(저장 artifact에 sample 없음)을 유지한다.
  - schema fingerprint는 sample과 무관하다.

### 3.3 결과 조립

- field별 후보 순서는 AI가 고른 column, 그다음 규칙 후보(같은 column은 한 번만)다. 최대 4개다.
- AI 후보:
  - `confidence`는 같은 column의 규칙 점수를 그대로 쓴다. 모델은 보정된 확신도를 주지 않기 때문이다.
  - `review_status`는 규칙 점수가 자동 승인 기준(0.90) 이상이면 `AUTO_APPROVE_CANDIDATE`, 아니면 `NEEDS_REVIEW`다. AI 선택만으로 자동 승인 후보가 되지 않는다.
  - 근거(evidence)는 규칙의 세 근거에 `AI_PROPOSAL` 한 줄을 더한다. 문장은 고정 형식이며 예시 값을 넣지 않는다.
- 한 column은 한 field에만 쓴다. 중복이면 뒤 field의 AI 선택을 버린다.
- AI가 "none"을 고른 field는 규칙 후보를 그대로 둔다.
- `unmapped_fields`는 AI 후보가 없고 규칙 최고 점수가 0.60 미만인 field다.

### 3.4 실패와 대체

- 모델 설정이 없으면 규칙 제안만 쓴다(지금과 같음).
- 다음 경우에는 규칙 제안으로 대체하고 등록은 계속한다.
  - 호출 실패, 시간 초과, JSON 오류, schema 위반.
  - 조립 결과가 계약 검사를 통과하지 못함: database ID·fingerprint·ontology 버전이 같아야 하고, field 목록이 같아야 하며, 모든 후보 column이 catalog에 있어야 한다.
- 대체 사유는 저장하지 않는다. 응답의 `proposer`로만 구분한다.

### 3.5 공개 응답

- 업로드 응답 버전을 `private-sqlite-upload-v2`로 올린다.
- `mapping.proposer`:
  - `AI`: AI 후보가 하나라도 있음.
  - `RULES`: 모델 설정 없음.
  - `AI_UNAVAILABLE`: 모델 설정은 있으나 AI 후보가 없음(실패 또는 전부 none).
- 근거 signal에 `AI_PROPOSAL`을 더한다.
- 검토 화면 후보에 `source`(`AI`/`RULES`)를 더한다.
- 물리 이름은 지금처럼 관리자 화면에만 보인다. SQL·예시 값·모델 원출력은 공개하지 않는다.

### 3.6 서버 동작

- 등록(모델 호출 포함)은 threadpool에서 실행한다. 한 업로드가 모델을 기다리는 동안 다른 요청이 막히지 않게 하기 위해서다.
- 설정:
  - `synthetic_prototype` CLI: `--mapping-model-tag TAG`(명시적 opt-in). base URL은 `--ollama-base-url`과 공유하고, 없으면 `http://127.0.0.1:11434`를 쓴다.
  - dev launcher: loopback `OLLAMA_BASE_URL`이 유효하면 켠다. 모델 tag는 `backend/.env`의 `MAPPING_MODEL`이 있으면 그것을, 없으면 기본 tag를 쓴다. 상태 줄에 `mapping_ai=enabled|disabled (…)`를 표시한다.

## 4. 평가 코드와의 관계

- `OllamaJsonModel`·`ProposalError`·`schema_conforms`와 column prompt 조립은 제품 모듈 `backend/adaptive/mapping/ai_proposal.py`로 옮긴다. `evaluation/ai_mapping.py`는 그것을 가져다 쓴다.
- 8-09·8-10의 prompt 문장이 바뀌지 않았음을 테스트로 고정한다(모든 변형·버전에서 옮기기 전과 같은 문자열).

## 5. 테스트

- fake model로 확인할 것:
  - AI 첫 후보, 규칙 후보 dedupe, 최대 4개.
  - confidence·status 규칙, 중복 column 버림, none 처리.
  - 실패·schema 위반·계약 위반 시 규칙으로 대체.
  - 예시 값이 prompt에만 들어가고 저장 파일·응답에는 없음.
- 업로드 HTTP:
  - `proposer` 세 값, v2 응답, threadpool 실행.
  - 기존 오류 code 유지.
- 검토 요약의 `source`.
- CLI·launcher 설정 파싱.
- frontend parser: v2, `AI_PROPOSAL`, `proposer`, `source`.
- 실제 모델 smoke:
  - 합성 SQLite(기존 flat 합성 DB 형태) 한 개를 로컬 모델로 한 번 등록하고, 제안 결과를 private 보고서로 확인한다.
  - 점수 주장은 하지 않는다.

## 6. 제약

- 기존 제약을 그대로 유지한다.
  - 32 MiB, Git 비추적 private 저장, immutable 등록, strict reload, PENDING_REVIEW 격리, 게시된 mapping만 실행.
  - 모델 출력은 제안일 뿐이며 게시 전 사람 검토를 대신하지 않는다.
  - 사람 관문을 표시하지 않고, push하지 않는다.
- 범위 밖:
  - 업로드 DB의 변환·값 사전.
  - 일부 field를 `unsupported`로 두는 게시(지금은 15개 field 모두 필요).
  - 관리자 설명 입력.
  - 이 셋은 8-07에서 다룬다.

---

<a id="prompts-phase8-13-chat-value-hints"></a>

## 원본: prompts/phase8-13-chat-value-hints.md

# Phase 8-13 지침: 챗봇 값 힌트(DB 값 기반 질문 해석 보조)

작성일: 2026-09-30.

사용자 결정:
- 기능하게 만드는 것이 먼저다. 실제 학교 데이터 노출 방지는 당분간 고려 대상에서 뺀다.
- 모델을 특정 DB에 묶지 않는 원칙은 유지한다.
- 약어·fuzzy 매칭이 특정 DB에 묶이지 않도록 설계한다.

## 1. 문제

레거시 DB 챗봇(QueryIR 경로)에서 다음 질문이 실패했다.
- "컴융 전공 과목"
- "학과공통 과목"
- "컴퓨터융합학부 전공 과목"

실제 모델 출력으로 원인을 확인했다.
- 모델은 `컴융→컴퓨터융합학부`를 안다. 레거시 데이터로 fine-tuning했기 때문이다.
- 하지만 학과를 이수구분 field에 넣거나, 없는 field(`Course.department`)를 지어낸다. "전공" 조건은 버린다.
- prompt에는 field 이름·type·연산자만 있고 DB 값이 없다.
- 모델 뒤의 보정도 좁다. 값 사전의 정확 일치, 이름형 field의 포함 검색, 시간·요일 정규화뿐이다.
- 레거시 시스템에는 모델 앞에 NER + 약어 표 + DB 값 fuzzy 매칭 단계가 있었다. 약어 표는 코드에 박혀 있다.

prompt에 "질문 표현 → field·실제 값" 힌트를 넣은 시험에서는 세 질문 모두 의도대로 출력됐다.

## 2. 원칙: 방식은 코드, DB별 지식은 데이터

- **코드(모든 DB 공통):**
  - 등록된 DB에서 값 목록을 읽는다.
  - 질문에서 그 값과 일치·부분 일치·별칭·오타 근접 표현을 찾는다.
  - 찾은 것을 모델 힌트로 주고, 모델 출력의 field를 ontology 기준으로 고친다.
  - 코드에는 특정 DB 이름·값·약어를 두지 않는다.
- **데이터(DB마다):**
  - 약어·별칭은 그 DB의 값 사전(8-06 `ValueDictionary.aliases`)에 둔다.
  - 레거시 DB는 레거시 약어 표를 그 DB의 profile 데이터로 옮긴다.
  - 새 DB는 관리자가 넣는다(8-07).
- **모델:** 재학습하지 않는다. 지식은 실행 시점의 DB별 데이터에서 온다.

## 3. 설계

### 3.1 값 목록(`value index`)

- 대상 field는 아래 조건을 모두 만족하는 것이다.
  - mapped 상태이고, ontology type이 string이다.
  - 변환이 없다.
  - 식별 field가 아니다(`Course.code`, `CourseOffering.section`, `Department.code`).
  - 서로 다른 값이 **500개 이하**다. 넘으면 그 field는 목록에서 뺀다.
  - 레거시 기준으로는 학과명 166개, 이수구분 9개가 대상이다. 교원명 1,395개, 과목명 2,331개는 빠진다.
- 읽는 방법:
  - 읽기 전용 연결, statement timeout을 쓴다.
  - SQL은 `SELECT DISTINCT` 한 가지 모양만 쓰고, 식별자는 게시된 mapping의 물리 이름을 `psycopg2.sql.Identifier`로 넣는다.
  - 모델 실행 경로(QueryIR → compiler → policy)와는 별도인 서버 내부 조회다.
- 시점: DB별로 첫 챗봇 질문 때 한 번 읽고, schema fingerprint와 함께 캐시한다.
- 읽기에 실패하면 힌트 없이 동작한다. 챗봇 답변을 막지 않는다.

### 3.2 질문 검사(규칙 기반, 공통)

- 정규화: 값 사전과 같은 `lookup_key`(NFKC, casefold, 공백 제거)를 쓴다.
- field마다 다음 키를 만든다.
  - 저장 값 전체.
  - 저장 값을 문자·숫자가 아닌 곳(공백, 괄호 등)에서 나눈 **연속 단어 조각**(2자 이상). 예: `공과대학 컴퓨터융합학부` → `공과대학`, `컴퓨터융합학부`. `전공(핵심)` → `전공`, `핵심`.
  - 값 사전의 별칭.
- 질문은 공백·문장부호로 단어를 나누고, 끝의 한국어 조사를 뗀다(공통 조사 목록).
- 연속 1~3단어의 조합을 긴 것부터 키와 비교하고, 겹치지 않게 고른다.
- 결과:
  - 한 저장 값에만 맞으면 힌트 값은 그 저장 값이다.
  - 여러 값에 맞으면 힌트 값은 맞은 조각이다(예: `전공`, `공과대학`). 이 값은 이후 포함 검색으로 처리된다.
- 오타 근접(fuzzy):
  - 맞는 것이 없는 4자 이상 단어만 대상이다.
  - 4자 이상 키와 `difflib` 유사도 0.85 이상이고, 최고점이 하나일 때만 인정한다.
  - 기준값은 레거시 수치를 옮기지 않은 공통 보수값이다.
- 힌트는 최대 8개다.

### 3.3 모델 입력

- 힌트가 있을 때만 Context 다음 줄에 넣는다: `Value hints (text in the question and the field/value it refers to): [...]`.
- 모델 호출은 여전히 한 번이다.

### 3.4 모델 출력 보정(QueryIR 검증 전, 닫힌 규칙)

1. **없는 field 이름 고치기(ontology 기반):** `E.x`가 ontology에 없으면 다음 둘 중 **하나로만** 정해질 때 그 field로 바꾼다.
   - `x`가 어떤 field의 `physical_name_hints`에 있다.
   - `x`가 어떤 entity의 `table_name_hints`에 있고, 그 entity에 `.name` field가 있다.
   - 예: `Course.department` → `Department.name`, `Course.professor` → `Instructor.name`.
2. **값으로 field 옮기기:** 다음을 모두 만족하면 filter를 힌트의 field로 옮긴다.
   - filter의 field가 값 목록 대상이다.
   - 그 field의 어떤 저장 값에도 filter 값이 포함되지 않는다.
   - filter 값이 정확히 한 힌트의 표현·값과 맞는다(같거나 힌트 값에 포함됨).
3. 조건을 새로 만들어 넣지는 않는다.
4. 보정 뒤에도 기존 QueryIR 검증·미지원 field·compiler·policy·executor를 모두 거친다.
5. 응답 형식은 바꾸지 않는다. 보정 결과는 "이해한 조건"에 그대로 보인다.

### 3.5 레거시 DB 데이터

- 학과명 사전(새로 추가): `공과대학 컴퓨터융합학부` ← `컴융`, `컴융부`.
- 이수구분 별칭 추가: `전핵`, `전심`, `교소`, `교소필`, `일선`(→`일반선택`), `교균`, `교기`.
- 레거시 표에서 뺀 것과 이유:
  - `전기`: 일반 단어(전기공학)와 겹친다.
  - `인공지능`: 과목명과 겹친다.
  - `컴공`: 대상 학과가 DB에 없다.
- private profile 두 개(`legacy-base`, `legacy-view`)를 다시 쓴다(`--write --live --replace`).

## 4. 검증

- 단위 테스트(합성 값):
  - 조각·별칭·조사·fuzzy·모호·겹침·상한.
  - field 이름 보정(유일할 때만), 값 기반 이동 조건.
  - 코드에 DB별 값이 없음.
- 챗봇 경로 테스트:
  - fake model로 힌트가 prompt에 들어가는지 확인한다.
  - 보정 뒤에도 검증을 거치는지, reader 실패 시 힌트 없이 동작하는지 확인한다.
- live 확인(레거시 DB, 로컬 모델):
  - 새 질문 유형: 약어, 학과공통, 학과+이수구분, 대학 단위, 이수구분 약어, 오타.
  - 7-15 비교 30문항을 다시 실행해 퇴보가 없는지 본다.
  - 레거시 약어 데이터를 뺀 상태의 결과도 함께 본다(공통 방식과 약어 데이터의 기여를 분리).
- 모든 결과는 개발자 질문 기준 잠정치다(L1 전).

## 5. 제약

- 기존 제약을 유지한다.
  - 모델 호출 1회, raw SQL fallback 없음.
  - QueryIR 검증·compiler·policy·executor 우회 없음.
  - 브라우저에는 DSN·물리 이름·SQL을 노출하지 않는다.
  - legacy DB는 읽기만 한다.
  - 사람 관문을 표시하지 않고, push하지 않는다.
- 이번 결정으로 저장 값이 로컬 모델 prompt에 들어간다. 실제 CNU 자료에 쓰기 전에는 E3 판단이 필요하다(기록만 하고 막지 않는다).

---

<a id="prompts-phase8-14-chat-sql-display"></a>

## 원본: prompts/phase8-14-chat-sql-display.md

# Phase 8-14 지침: 챗봇 실행 SQL 표시(개발 옵션)

작성일: 2026-09-30.

사용자 요청: 레거시 챗봇처럼 QueryIR 경로도 실제 SQL을 보여 달라. 그래야 잘 작동하는지 끝까지 확인할 수 있다.

## 1. 결정

- 지금까지는 "브라우저에 SQL·물리 이름을 보내지 않는다"는 원칙이었다. 이번에 사용자 결정으로 **서버 개발 옵션**을 켰을 때만 성공한 챗봇 답에 실행 SQL을 붙인다.
- 기본값:
  - `synthetic_prototype`는 끔. `--show-sql`로 켠다.
  - `npm run dev:all` launcher는 개발용이므로 켬. `backend/.env`의 `SHOW_SQL=0`으로 끈다. 상태 줄에 `chat_sql=shown|hidden`을 표시한다.

## 2. 무엇을 보여 주나

- executor에 실제로 들어간 compile 결과다. 모델 출력이 아니라, 검증·policy를 통과해 실행된 것이다.
- 읽기 쉽게 bound 값을 SQL literal로 채워 넣은 **보기용 문장**이다.
  - 문자열은 `'...'`로 감싸고 작은따옴표는 두 번 쓴다.
  - 숫자와 `TRUE`/`FALSE`는 그대로 쓴다. `%%`는 `%`로 바꾼다.
- 실제 DB에는 여전히 문장과 값을 따로 보낸다(parameter binding). 화면 표시만 합친 것이다.
- `UNSUPPORTED`·`NOT_UNDERSTOOD`·오류 응답에는 SQL을 넣지 않는다.
- 보기용 문장을 만들다 실패하면 SQL만 빼고 답은 그대로 준다. 길이는 최대 20,000자다.

## 3. 화면

- 기존 레거시용 SQL 패널을 같이 쓴다.
- QueryIR 경로에서는 제목을 "Executed SQL (QueryIR · 값은 보기용으로 채워 넣음)"으로 표시한다.
- frontend parser는 성공 응답에서만 선택 key `sql`(비어 있지 않은 문자열, 20,000자 이하)을 받는다.

## 4. 테스트

- 보기용 문장: literal 규칙, 매개변수 개수 불일치는 거부.
- 옵션을 껐을 때는 응답이 이전과 같다. 켰을 때는 `sql` 외에는 같다.
- 실행하지 않은 답에는 SQL이 없다.
- launcher 기본값과 `SHOW_SQL=0`, CLI `--show-sql`.
- frontend parser 허용·거부.

## 5. 영향

- 옵션을 켜면 브라우저에 물리 table·column 이름과 값이 보인다.
- 외부 시연이나 실제 자료 사용 전에는 끄거나 L5/E3 판단이 필요하다(기록만 하고 막지 않는다).

---

<a id="prompts-phase8-15-postgres-registration"></a>

## 원본: prompts/phase8-15-postgres-registration.md

# Phase 8-15 지침: 화면에서 PostgreSQL DB 등록(서버 연결 목록 방식)

작성일: 2026-09-30.

사용자 결정:
- 접속 정보는 B, 즉 서버 연결 목록으로 받는다. 비밀번호는 브라우저를 거치지 않는다.
- 8-15와 8-07의 순서는 개발자 판단에 맡긴다. 8-15를 먼저 한다. 그래야 8-07 검토 화면을 SQLite와 PostgreSQL 공통으로 한 번에 만들 수 있다.

## 1. 목표

"DB 변경" 화면에서 PostgreSQL 연결과 schema를 골라 등록한다. 등록한 DB는 SQLite와 같은 검토 화면에서 검토한 뒤 게시하며, 게시하면 서버를 재시작하지 않아도 DB 목록·챗봇(QueryIR, 값 힌트, SQL 표시)에 바로 붙는다.

## 2. 연결 목록(서버 소유)

- 연결 목록은 `backend/.env`에서 만든다.
  - `DATABASE_URL`은 연결 `default`("기본 PostgreSQL 서버")가 된다.
  - `PG_SOURCE_<NAME>=<DSN>`은 연결 `<name>`(소문자, `_`→`-`)이 된다.
- 브라우저에는 연결 ID와 이름만 보낸다. DSN·계정·비밀번호는 보내지 않는다.
- schema 목록은 읽기 전용으로 조회한다. `pg_catalog`, `information_schema`, `pg_toast`, `pg_temp*`는 뺀다.

## 3. 등록

- 입력: 연결 ID, schema, database ID, 이름. view 포함 여부는 이번에 다루지 않는다(table만).
- 순서:
  1. 읽기 전용으로 구조를 확인한다(기존 `PostgresAdapter`). table이 없으면 거부한다.
  2. 규칙 제안을 만들고, 모델이 설정돼 있으면 8-12 AI column 제안을 만든다.
     - AI 제안에는 저장 값 예시가 필요하다. column마다 서로 다른 값을 최대 10개 읽어(8-13 reader) 메모리에서만 쓴다.
     - column이 300개를 넘으면 예시 없이 규칙만 쓴다.
  3. private 저장소 `.schema-adaptive/postgres-registrations/<id>/`에 `registration.json`, `catalog.json`, `proposal.json`을 쓴다.
     - 등록 기록은 immutable이고, strict reload한다.
     - Git이 추적하지 않는 경로여야 한다.
     - 저장 파일에는 예시 값을 넣지 않는다.
- database ID는 다음 어느 것과도 겹치면 거부한다: SQLite 등록, 기존 PostgreSQL profile, 고정 합성 ID.
- 응답은 SQLite 업로드 응답(v2)과 같은 모양이다. 버전은 `private-postgres-registration-v1`이다.

## 4. 검토와 게시

- 검토 API 경로(`/registrations/review`)는 같다. database ID가 PostgreSQL 등록이면 서버가 PostgreSQL 처리로 넘긴다. 응답 모양도 같아서 화면을 그대로 쓴다.
- 동작: `INIT`, `APPROVE`, `REMAP`, `IGNORE`(새로 추가), `PUBLISH`.
  - `IGNORE`는 "이 DB에 이 정보가 없음"이며 사유가 필요하다. SQLite에서도 draft에는 기록되지만, SQLite 게시는 지금처럼 15개 field가 모두 있어야 한다.
  - `ACTIVATE`·`CARRY_OVER`·test query·회귀 비교는 PostgreSQL 등록에서 아직 지원하지 않는다(고정 오류 code).
- 게시 조건:
  - 모든 field가 승인·재지정·무시 중 하나여야 한다. 대기 상태가 남으면 거부한다.
  - 연결된 field가 하나 이상 있어야 한다.
- 게시 때 서버가 정하는 것(모델 없음, 결정적 규칙):
  - **변환:** field type과 column type이 맞고 저장 값 예시(최대 50개)가 형식에 맞는 변환이 정확히 하나일 때만 붙인다. 예: `time` column → `TIME_TO_HHMM`, `3/3/0` 형식 → `SLASH_FIRST_NUMERIC`, 한 글자 요일 → `KOREAN_WEEKDAY`.
  - **기준 table:** `CourseOffering.section`의 table이다(8-09 규칙). 분반이 없으면 기준 table도 없다.
  - **미지원 field:** 무시한 field다.
  - **값 사전:** 없음(8-07에서 코드표·약어 입력).
- 검증:
  - 실제 연결로 기존 runtime 검증(`build_validated_contexts`)을 통과해야 한다. fingerprint, 형식, 기준 table 경로 유일성을 본다.
  - 실패하면 `PUBLISH_BLOCKED`다.
- 저장:
  - `.schema-adaptive/postgres-profiles/<id>/`에 profile v4와 mapping 쌍을 쓴다. 기존 startup loader가 읽는 형식이며, DSN 환경변수 이름은 `SCHEMA_ADAPTIVE_POSTGRES_DSN_REG_<ID>`다.
  - 등록 폴더에 게시 기록 `published.json`(버전·게시자·시각)을 쓴다.
  - 다시 게시하면 버전이 올라가고 profile을 교체한다.

## 5. 실행 중 추가

- 게시에 성공하면 검증된 context를 실행 중인 서버의 PostgreSQL context 집합에 추가한다. 추가만 하고, 같은 ID가 있으면 교체한다.
- DB 목록·강의 목록·챗봇·값 힌트 reader가 같은 집합을 본다.
- 서버를 재시작하면 dev launcher가 게시된 등록을 찾아 연결 목록의 DSN을 해당 환경변수로 넣고 profile을 읽는다.

## 6. 화면

- "DB 변경" 창에 "PostgreSQL 등록" 영역을 추가한다: 연결 선택 → schema 선택 → ID·이름 → 등록.
- 결과 요약과 검토 패널은 SQLite 업로드와 같다.
- 검토 패널에 "이 DB에 없음(IGNORE)" 버튼과 사유 입력을 추가한다.

## 7. 테스트

- 단위(fake connector·adapter):
  - 연결 목록 파싱, DSN 비노출, schema 목록.
  - 등록 ID 충돌, strict reload, 예시 값 비저장.
  - IGNORE, 게시 조건, 변환·기준 table 결정, 실패 시 파일 없음.
  - 실행 중 추가 뒤 DB 목록·챗봇 연결.
- frontend parser와 client.
- live:
  - 합성 DB(`schema_adaptive_synthetic`)에 임시 schema를 만든다.
  - 화면 API로 등록 → 검토 → 게시 → 챗봇 질문까지 확인한다.
  - 끝나면 schema와 private 파일을 지운다.

## 8. 제약

- 읽기 전용이며 legacy DB는 수정하지 않는다.
- DSN은 브라우저·stdout에 내보내지 않는다.
- 게시된 mapping만 실행하고, 모델 호출 수·검증 경로는 변경하지 않는다.
- 사람 관문을 표시하지 않고, push하지 않는다.

---

<a id="prompts-phase8-16-registration-list-and-demo"></a>

## 원본: prompts/phase8-16-registration-list-and-demo.md

# Phase 8-16 지침: 등록 목록(다시 열기), 처음부터 끝까지 browser smoke, 시연 절차

작성일: 2026-09-30. 사용자 결정: 진행. 순서는 ADR-078의 프로토타입 마지막 단계를 따른다.

## 1. 등록 목록

- 문제: 페이지를 새로고침하면 검토 중이던 PostgreSQL 등록으로 돌아갈 길이 없다.
- API `GET /api/adaptive/v1/postgres/registrations`는 화면 등록 PostgreSQL DB 목록을 돌려준다.
  - 항목: `database_id`, `label`, `source_id`, `schema`, `status`(`PENDING_REVIEW`/`PUBLISHED`), `mapping_version`(게시 전 `null`).
  - DSN과 물리 이름은 넣지 않는다. schema 이름은 관리자가 고른 값이라 보여 준다.
  - 손상된 등록은 목록에서 빼고, 요청 전체를 실패시키지 않는다.
- 화면: PostgreSQL 등록 영역에 "등록된 PostgreSQL DB" 목록을 두고, 항목마다 "열기" 버튼으로 검토·값 사전 패널을 다시 연다.

## 2. 처음부터 끝까지 browser smoke

- 새 코드를 확인용 포트(adaptive 8124, 화면 5174)에 띄운다. 사용자 서버(8000/8123/5173)는 건드리지 않는다.
- 시나리오: 레거시 DB를 화면에서 등록한다(읽기 전용).
  1. DB 변경 → PostgreSQL 등록(연결 `default`, schema `public`).
  2. 검토: 후보를 보고 승인하거나 "이 DB에 없음"을 누른다(이름 함정 확인).
  3. 값 사전: 학과 약어 `컴융`.
  4. 게시.
  5. 새로고침한 뒤 목록에서 다시 열리는지 확인한다.
  6. DB 선택 → 강의 목록·챗봇 "컴융 전공 과목" → 이해한 조건과 SQL을 확인한다.
- 끝나면 확인용 등록·profile 파일을 지우고 확인용 서버를 끈다.
- 결과와 발견한 문제를 결과 문서에 적는다. 막는 문제는 고친다.

## 3. 시연 절차 문서

`docs/demo-procedure.md`에 둔다.
- 준비: `backend/.env`, Ollama 모델 2개, 서버 실행과 상태 줄 읽는 법.
- 시연 순서: 위 시나리오와 말할 요점.
- 되돌리기: 등록 파일 삭제 위치.
- 주의: SQL 표시(`SHOW_SQL=0`), 실제 학교 자료 사용 전 E3/L5, 외부 시연 전 L5.

## 4. 제약

- 읽기 전용이며 legacy DB는 수정하지 않는다.
- DSN은 브라우저에 보내지 않는다.
- 사람 관문을 표시하지 않고, push하지 않는다.

---

<a id="prompts-phase8-17-other-school-validation"></a>

## 원본: prompts/phase8-17-other-school-validation.md

# Phase 8-17 지침: 다른 학교 합성 DB로 화면 흐름 검증

작성일: 2026-09-30. 사용자 결정: 진행("다른 학교 합성 DB를 직접 만들어 검증").

## 1. 목적

- 지금까지 화면에서 끝까지 확인한 DB는 레거시 DB 하나다. 챗봇 모델은 레거시 데이터로 fine-tuning되었다.
- 그래서 "모델이 특정 DB에 묶이지 않는다"는 주장을 레거시가 아닌 DB에서 같은 화면 흐름으로 확인한다.
- 결과가 나빠도 기록한다. 어디가 깨지는지가 다음 작업의 근거다.

## 2. 합성 DB: 가상 학교 schema `other_school`

- 위치: 로컬 전용 DB `schema_adaptive_synthetic`(2026-09-30 사용자 승인). 레거시 DB는 건드리지 않는다.
- 생성기: `backend/adaptive/fixtures/other_school.py`.
  - seed를 고정해 몇 번을 돌려도 같은 DB가 나온다.
  - 학교명은 넣지 않는다. 학과·교수·과목 이름은 지어낸 값이다.
  - CLI: `python -m backend.adaptive.fixtures.other_school create|drop|check`.
  - DSN은 `SCHEMA_ADAPTIVE_SYNTHETIC_POSTGRES_DSN`에서만 읽는다(7-03과 같은 loopback·DB 이름 검증). 출력에 DSN을 넣지 않는다.
  - `create`는 같은 이름 schema가 있으면 지우고 다시 만든다. 그 schema 하나만 다룬다.
- 레거시와 일부러 다르게 둔다.

| 항목 | 레거시 | 다른 학교 |
|---|---|---|
| 이름 | 영문 단어(`dept_name`) | ontology 힌트에 없는 로마자 약어(`hg_nm`, `gm_nm`) |
| 학과 | 과목 table의 문자열 | 학과 table(`hakgwa`)과 단과대학 column 분리 |
| 이수 구분 | 글자("전공핵심") | 코드(`11`, `12`, `21` …)와 코드 table |
| 학점 | `3/3/0` 한 칸 | 학점·이론·실습 column 분리 |
| 학기 | 글자 | 코드(`10`, `20`) |
| 요일 | `월` | `MON` |
| 시간 | time type | `char(4)` `0900` |
| 시간표 | 강좌당 1행 | 강좌당 1~2행 |
| 이름 함정 | `course_year` = 학년 | `suggang_hangnyeon` = 대상 학년 |

- 결과가 200건 상한에 걸리지 않게 규모를 정한다.

## 3. 질문과 정답은 실행 전에 고정

- 생성기 module에 질문 12개와 정답 판정을 둔다. 정답은 DB가 아니라 생성기의 Python 행에서 계산한다.
- 판정: `SUCCEEDED`이고 결과 행의 과목 코드 집합이 정답 집합과 같으면 통과다.
- "1학년 대상 과목"처럼 ontology에 없는 질문은 결과를 내지 않으면(`UNSUPPORTED`/`NOT_UNDERSTOOD`) 통과다.
- 결과를 본 뒤 질문이나 정답을 바꾸지 않는다.

## 4. 화면 흐름

- 확인용 서버: adaptive 8124, 화면 5174. 사용자 서버(8000/8123/5173)는 건드리지 않는다.
- 연결은 확인용 서버 프로세스에만 `PG_SOURCE_OTHER_SCHOOL`로 넘긴다. `backend/.env`는 고치지 않는다.
- 순서: 등록 → 검토(함정은 "이 DB에 없음") → 값 사전(학교 약어와 코드 이름) → 게시 → 챗봇.
- 값 사전은 이 학교 관리자가 넣을 법한 것만 넣는다: 학과 약어, 코드의 이름(이수 구분, 학기, 요일).
- 질문 12개는 화면과 같은 chat API로 보내 채점한다. 대표 질문은 화면에서도 확인한다.

## 5. 고치는 기준

- 어느 DB에나 해당하는 결함만 고친다. 이 schema 이름이나 값을 코드에 넣지 않는다.
- 막히지 않는 한계는 결과 문서와 시연 절차의 알려진 한계에 적는다.

## 6. 정리

- 확인용 등록·profile 파일을 지우고 확인용 서버를 끈다.
- 합성 schema `other_school`은 시연용으로 남긴다. 지우는 명령은 `drop`이다.

## 7. 제약

- legacy DB는 읽지도 쓰지도 않는다.
- DSN을 브라우저·출력에 보내지 않는다.
- 사람 관문을 표시하지 않고, push하지 않는다.

---

<a id="prompts-phase8-18-target-grade-and-field-list"></a>

## 원본: prompts/phase8-18-target-grade-and-field-list.md

# Phase 8-18 지침: 공통 항목 "대상 학년"과 항목 목록 일반화

작성일: 2026-10-01. 사용자 결정: 진행(순서 8-18 대상 학년·목록 일반화 → 8-19 DB별 확장 항목 → 8-20 모르는 조건 처리).

## 1. 배경

- "1학년 과목 보여줘"는 학생이 흔히 묻는 조건이고, 레거시 챗봇도 답했다.
  - 레거시는 `course_year`로 "학년", `subject.target_year`로 "대상학년"을 표시했다.
- 공통 ontology 15개 항목에 대상 학년이 없어, 새 시스템은 이 조건을 다른 항목에 끼워 넣었다(8-17 q12).
- 레거시 DB와 8-17 가상 학교 DB 모두 이 정보가 있다. 어느 학교에나 있을 만한 공통 항목이다.

## 2. 공통 ontology 추가(버전 2)

- `Course.target_grade`: label "대상 학년", 정수.
  - 별칭: 학년, 대상학년, 수강학년.
  - 연산자: 같음과 크기 비교.
  - `required: false`.
- 새 항목에는 `since_version: 2`를 단다.
- ontology 버전 2는 버전 1에 항목만 더한 것이다(**추가형 호환**).
  - 버전 1로 만든 dataset, 평가 세트, mapping draft, 게시 mapping은 그대로 유효하다.
  - 판단 기준은 `1 ≤ 기록된 버전 ≤ 현재 버전`이다.
  - 옛 mapping에서 버전 1 이후 항목은 "이 DB에 없음"으로 본다. 챗봇은 `UNSUPPORTED`로 답한다.
- 버전 1 항목 집합으로 검사하던 곳(runtime 게시 검사, 검토 publish 검사)은 mapping이 기록한 버전의 항목 집합으로 검사한다.

## 3. 항목 목록

- **강의 목록(catalog):** 15개 항목 그대로다(`CATALOG_FIELDS`). 대상 학년은 강의 목록 계약에 넣지 않는다.
- **챗봇 카드:** ontology 순서대로 매핑된 항목을 보인다. DTO key는 고정 표(기존 15개) + 새 항목(`target_grade`)이다.
  - frontend도 이 key를 받고, 카드 상세에 "대상 학년"을 보인다.
- **SQLite 업로드 게시:** 강의 목록 15개가 모두 있어야 한다는 조건은 유지한다. 대상 학년은 있어도 되고 없어도 된다.

## 4. 이미 등록한 DB

- **PostgreSQL 화면 등록:** 버전 1 등록은 그대로 동작하고, 대상 학년은 "이 DB에 없음"이다.
  - (구현 중 조정) 등록 기록(proposal)은 해시로 고정된 불변 파일이다. 그래서 draft 올리기는 새 저장 구조가 필요해 이번 범위에서 뺐다. 대상 학년을 쓰려면 새 ID로 다시 등록한다.
- **SQLite 업로드 carry-over:** 버전 1 기준 mapping에서 버전 2 등록으로 옮길 수 있다. 기준에 없던 항목은 `NOT_IN_BASE`로 기록하고 검토를 기다린다.
- **레거시 profile(개발자 mapping):**
  - view: `v_course_info.target_year`.
  - base: `subject.target_year`.
  - profile을 다시 만든다. legacy DB는 catalog와 값만 읽는다.

## 5. 검증

- 레거시 profile: "1학년 과목 보여줘"가 `target_year = 1`로 답하는지.
- 8-17 가상 학교: 검토에서 `suggang_hangnyeon`을 대상 학년에 연결하고 q12를 다시 돌린다.
  - q12 정답을 "답하지 않음"에서 "1학년 과목"으로 바꾼다. 처음 정답이 기능 부재를 전제로 잘못 정해진 것이므로 문서에 그렇게 적는다.
  - 나머지 11개가 그대로인지 확인한다.
- 대상 학년이 없는 DB에서 학년 질문은 `UNSUPPORTED`("대상 학년")로 답해야 한다.

## 6. 제약

- 공통 ontology는 팀이 관리한다. 이번 추가는 팀 논의 때 공유할 항목으로 기록한다.
- QueryIR 검증, compiler, SQL policy, executor는 우회하지 않는다.
- 사람 관문은 표시하지 않고, push하지 않는다.

---

<a id="prompts-phase8-19-db-extension-fields"></a>

## 원본: prompts/phase8-19-db-extension-fields.md

# Phase 8-19 지침: DB별 확장 항목 (8-11 축소판)

작성일: 2026-10-01. 사용자 결정: 진행(8-18 → 8-19 → 8-20 순서).

## 1. 목적

- 강의실처럼 어떤 학교에서는 중요하지만 공통 ontology에 넣을 정도는 아닌 정보를, 관리자가 그 DB에만 추가한다.
- 코드는 고치지 않고, 관리자가 화면에서 데이터로 선언한다.
- 공통 항목과 똑같이 QueryIR 검증, compiler, SQL policy, read-only executor를 거친다.

## 2. 확장 항목 선언

- 대상: 화면 등록 PostgreSQL DB.
- 선언하는 값:
  - **label:** 화면과 모델에 보이는 이름. 예: 강의실. 1~20자.
  - **key:** 영문 소문자로 시작하고 영문 소문자·숫자·밑줄만 쓴다. 최대 31자.
  - **entity:** 공통 entity 5개 중 하나. join 경로 판단에 쓴다.
  - **table, column:** 등록 catalog에 실제로 있는 것.
  - **data_type:** `string` | `integer` | `number`.
- 항목 이름은 `<Entity>.ext_<key>`이다(예: `Schedule.ext_classroom`). 공통 항목과 겹치지 않는다.
- 검사:
  - column 타입과 data_type이 맞아야 한다(문자 ↔ string, 정수 ↔ integer, 수치 ↔ number).
  - 한 column은 한 항목에만 연결한다. 검토에서 승인된 column이나 다른 확장 항목과 겹치면 막는다.
  - DB당 최대 10개.
- 저장 위치: 등록 폴더의 `extensions.json`(변경 가능). 게시해야 챗봇에 반영된다. 값 사전과 같은 방식이다.

## 3. 게시와 runtime

- 게시하면 확장 항목이 게시 mapping의 field로 들어간다. profile은 v5이고 `extension_fields`(label, data_type)를 담는다.
- runtime 검사:
  - 이름 형식과 entity를 확인한다.
  - mapping에 있는지 확인한다.
  - column 타입이 맞는지 확인한다.
  - 변환·값 사전·unsupported 목록과 겹치지 않는지 확인한다.
- 공통 항목 검사(`mapped | unsupported == 버전의 항목`)에서는 확장 항목을 뺀다.

## 4. 챗봇

- 그 DB의 runtime ontology = 공통 항목 + 확장 항목.
  - 연산자: string은 equals·contains, 숫자는 equals와 크기 비교.
- 모델 prompt의 field 목록에서 확장 항목에만 label을 붙인다. 확장 항목이 없는 DB의 prompt는 바뀌지 않는다.
- 문자열 확장 항목의 equals는 이름형 항목처럼 contains로 완화한다.
- 결과 카드:
  - DTO key는 `ext_<key>`이다.
  - 응답에 `extensions: [{key, label}]`를 추가한다.
  - frontend는 이 목록에 있는 key만 받고, 카드 상세에 label로 보인다.

## 5. 화면

- PostgreSQL 등록 검토 아래에 "확장 항목" 패널을 둔다.
  - 목록과 삭제.
  - 추가 양식: label, key, entity, table, column, data_type.
- 게시는 기존 게시 버튼과 같다. 패널에 "게시해야 챗봇에 반영" 안내를 둔다.

## 6. 검증

- 8-17 가상 학교 DB의 시간표에 강의실 column을 추가한다. 질문과 정답을 실행 전에 고정한다.
- 레거시 DB(`public`)를 화면 경로로 등록하고 강의실을 확장 항목으로 추가해 확인한다(읽기 전용).

## 7. 제약

- 확장 항목에는 변환과 값 사전을 붙이지 않는다(이번 범위).
- 강의 목록 계약은 그대로 둔다.
- push하지 않는다.

---

<a id="prompts-phase8-20-unknown-conditions"></a>

## 원본: prompts/phase8-20-unknown-conditions.md

# Phase 8-20 지침: 모르는 조건 처리

작성일: 2026-10-01. 사용자 결정: 진행(8-18 → 8-19 → 8-20 순서).

## 1. 목적

- 질문에 그 DB가 모르는 조건이 들어 있으면, 엉뚱한 항목에 끼워 넣은 결과를 보여 주지 않고 "이 DB에는 그 정보가 없다"고 답한다.
  - 예: 확장 항목이 없는 DB에서 "교111 강의실 수업" → "이 DB에는 강의실 정보가 없어 답할 수 없습니다."
- 조건을 다른 항목에 넣어 0건이 나왔을 때는 그 이유를 보여 준다.
  - 예: "교111에서 하는 수업" → 분반 = 교111, 0건 → "분반에서 '교111' 값을 찾지 못했습니다."
- 모델 호출은 한 질문에 한 번이다. 추가 모델 호출, raw SQL, 검증·compiler·policy·executor 우회는 없다.

## 2. 공통 항목 밖 개념 목록 (데이터)

- 파일: `backend/adaptive/ontology/outside-concepts.yaml`. 코드가 아니라 데이터다.
- 항목마다:
  - **key:** 영문 소문자 시작, 소문자·숫자·밑줄.
  - **label:** 답변에 보이는 이름. 예: 강의실.
  - **aliases:** 질문에서 찾을 말. 2자 이상. 공백과 대소문자는 무시한다.
  - **patterns(선택):** 정규식. 예: `303호` 같은 호실 번호.
- 검사:
  - key·label·alias는 서로 겹치지 않는다.
  - 공통 ontology의 entity·field 이름과 별칭과 겹치지 않는다(공통 항목은 기존 unsupported 처리가 맡는다).
- 처음 목록: 강의실, 수업 방식, 단과대학, 강의 언어, 성적 평가 방식, 시험, 캠퍼스, 선수 과목, 강의 평가, 여석, 강의계획서.
- 일반 단어 하나로 과목명에 흔히 들어가는 말(`건물`, `원격`, `온라인`, `실시간`)은 단독 별칭으로 두지 않는다. `온라인수업`처럼 붙은 말로만 둔다. 레거시 과목명 `건물에너지`, `원격탐사`, `온라인 영상 플랫폼과 기획론`이 근거다.

## 3. 챗봇 판단 순서

1. 모델 호출 전에 질문에서 개념 별칭과 패턴을 찾는다.
2. 그 DB의 확장 항목이 그 개념을 다루면(label이나 key가 개념의 label·alias·key와 맞으면) 그 개념은 뺀다.
3. 모델을 한 번 호출하고, 기존 repair를 적용한다.
4. 찾은 말이 다음에 들어 있으면 그 개념은 뺀다(값의 일부로 쓰인 말이다).
   - 모델이 만든 조건 값. 예: 과목명에 "건물에너지" 포함.
   - 값 힌트가 DB 값으로 찾은 말.
5. 남은 개념이 있으면 실행하지 않고 `UNSUPPORTED`로 답한다.
   - 기존 형식에 항목 `{concept, label}`을 더한다. 공통 항목의 `{field, label}`과 함께 올 수 있다.
   - 모델 출력이 잘못되었어도 남은 개념이 있으면 `UNSUPPORTED`가 우선한다.

## 4. 0건 진단

> 2026-10-01 구현 중 수정: 처음에는 값 힌트용 값 목록(500개 이하)으로 판단하려 했다. 레거시 실측에서 모델이 "교111"을 과목 코드(값 500개 초과)에 넣어 판단할 수 없었다. 그래서 아래처럼 DB에 직접 묻는 방식으로 바꿨다. 숫자 같음 조건도 정확히 판단되므로 진단에 넣는다(Luna 리뷰 3번).

- 실행 결과가 0건일 때만 본다. 답의 실행은 바꾸지 않는다.
- 같음·포함 조건을 하나씩 따로 실행한다.
  - 그 항목 1행만 고르는 QueryIR이다(`limit 1`).
  - 같은 검증·compiler·policy·읽기 전용 executor를 거친다. 최대 4번이다.
  - 조건이 하나뿐이면 그 조건 자체가 답이므로 다시 실행하지 않는다.
- 혼자서도 0행인 조건을 `unmatched`에 넣는다. 진단 실행이 실패하면 그 조건은 뺀다.
- 범위·크기 조건(`>`, `<` 등)은 진단하지 않는다.
- 응답: `SUCCEEDED`에 선택 키 `unmatched: [{field, label, value}]`(0건일 때만).
- 화면: "분반에서 '교111' 값을 찾지 못했습니다. 이 DB에 없는 정보이거나 다른 항목으로 잘못 이해했을 수 있습니다."
- SQL 표시(개발 옵션)는 진단이 아니라 답의 query를 보인다.

## 5. 화면

- `UNSUPPORTED` 항목에 `concept` 형식을 받는다. 문구: "이 DB에는 강의실 정보가 없어 답할 수 없습니다. 그 조건을 빼고 다시 물어보세요."
- 0건이고 `unmatched`가 있으면 위 문구를 더한다.

## 6. 검증 (실행 전 고정)

- 레거시 개발자 profile(`legacy-view`, 확장 항목 없음):
  - `UNSUPPORTED` 기대: "교111 강의실 수업 보여줘"(강의실), "305호에서 하는 수업"(강의실), "실시간 수업 보여줘"(수업 방식), "온라인 강의 보여줘"(수업 방식), "절대평가 과목 보여줘"(성적 평가 방식), "영어강의 과목 보여줘"(강의 언어), "여석 있는 컴융 과목"(여석).
  - 0건이면 `unmatched` 기대: "교111에서 하는 수업", "IT대학 과목 보여줘".
  - 바뀌면 안 되는 것: "컴융 전공 과목 보여줘", "월요일 오전 수업 보여줘", "건물에너지 과목 보여줘", "온라인 영상 플랫폼 과목 보여줘", "컴융 3학년 전공 과목 보여줘".
- 가상 학교 DB:
  - 확장 항목 없이 게시: "IT관 303호 수업 보여줘" → `UNSUPPORTED`(강의실). q01~q12는 그대로 통과.
  - 강의실 확장 항목을 선언하고 게시: q13·q14 포함 14/14 유지.

## 7. 제약

- 개념 목록에 없는 말은 여전히 다른 항목에 들어갈 수 있다. 0건 진단으로만 알린다.
- 범위·크기 조건은 0건 진단에서 뺀다.
- push하지 않는다.

---

<a id="prompts-phase8-21-postgres-format-variety"></a>

## 원본: prompts/phase8-21-postgres-format-variety.md

# Phase 8-21 지침: 형식이 다른 PostgreSQL DB 대응 점검

작성일: 2026-10-02. 사용자 결정: PostgreSQL 전용, 다른 PostgreSQL DB에 대한 적응성만 범위(ADR-089). 이번 사용량 주기에는 이 지침과 형식 D(넓은 단일 table) 확인까지만 한다.

## 1. 목적

- 팀원이 다른 대학의 공개 수강편람을 흉내 내면 나올 만한 형식을, Phase 9 평가 전에 미리 확인하고 막는다.
- 팀원의 평가 DB를 대신하지 않는다. 여기서 만드는 DB는 AI가 만든 개발용이다(ADR-086).

## 2. 지금 코드의 전제 (확인한 것)

- join: FK만 따라간다(`relations/planner.py`). FK 없는 관계나 "코드그룹 = 'X'" 같은 추가 조건은 표현할 수 없다.
- 변환: 닫힌 목록 4개(`SLASH_FIRST_NUMERIC`, `TIME_TO_HHMM`, `KOREAN_WEEKDAY`, `HHMM_TEXT_TO_HHMM`). 매개변수가 없다.
- 공통 항목 하나는 column 하나에 연결한다. 한 column을 여러 항목으로 나눌 수 없다.
- 값 사전(8-06·8-07)과 코드표→값 사전(8-10)은 있다.

## 3. 점검할 형식

데이터는 모두 8-17 `other_school`과 같은 내용으로 만들고, 모양만 바꾼다. 그래서 고정 질문 q01~q12와 정답(`other_school.gold`)을 그대로 쓴다. 정답은 과목 코드 집합으로 채점한다.

| 형식 | 모양 | 예상 | 대응 후보(구현 전 결정) |
|---|---|---|---|
| A. 공통코드 table | `common_code(group_cd, code, code_nm)` 하나에 이수구분·요일·학기 코드를 모음. 강좌 table에는 코드만 있음 | 코드 column에 연결하면 값은 코드로만 비교된다. join은 FK가 없어 불가 | ① 공통코드 행을 그 항목의 값 사전으로 가져오기(8-10 코드표→사전 재사용, join 없음). ② 조건 붙은 관계 선언. **①을 먼저 시험한다.** |
| B. 교시 시간표 | `siganpyo(yoil, gyosi_from, gyosi_to)`, 교시 번호만 저장. 교시→시각 table이 따로 있을 수도 있음 | 시작·종료 시간 연결 불가("오전 수업" 실패) | 관리자가 교시→시각 표를 데이터로 선언하는 매개변수형 변환 `PERIOD_TO_HHMM`. 같은 table을 두 번 join하는 방식은 피한다. |
| C. 한 칸에 묶인 시간표 | `sigan_txt = '월1-2,수3'` 또는 `'월09:00-10:30,수09:00-10:30'` | 요일·시간 연결 불가 | ① 요일만 "포함" 조건으로 지원하고 시작·종료 시간은 "이 DB에 없음". ② 문자열 해석 변환. **①로 시작하고, ②는 사용자 결정.** |
| D. 넓은 단일 table | 모든 정보가 한 table(시간표 칸마다 한 행) | 레거시 view와 같은 경우라 동작 예상 | 이번 주기에 확인만 한다. |

## 4. 진행 순서

1. **D 확인(이번 주기).** 합성 DB에 `wide_school` schema를 만든다. `other_school`의 table을 join한 한 table이며, scratch SQL로 만든다. 화면 경로 API로 등록·검토·게시하고 q01~q12를 돌린다. 깨지는 것만 기록한다.
2. A: 공통코드 schema → 등록 → ①로 값 사전 → q01~q12.
3. B: 교시 schema → `PERIOD_TO_HHMM` 설계·구현 → q05·q11(요일·오전) 중심.
4. C: 묶인 시간표 schema → ① 부분 지원 확인 → ② 여부는 사용자에게 묻는다.
5. 형식마다 깨진 것만 고친다. 고치는 단위마다 회귀 테스트, Luna 리뷰, 문서 갱신을 한다.

## 5. 제약

- 레거시 DB는 읽기만 한다. 합성 DB(`schema_adaptive_synthetic`)만 쓴다.
- 모델 호출은 질문당 한 번이다. 검증·compiler·policy·read-only executor 우회는 없다.
- 공통 ontology는 바꾸지 않는다(바꿔야 하면 사용자 결정).
- push하지 않는다.

---

<a id="prompts-phase8-22-blind-evaluation-runner"></a>

## 원본: prompts/phase8-22-blind-evaluation-runner.md

# Phase 8-22 지침: 팀원 DB 평가 실행 도구

작성일: 2026-10-06. 사용자 결정: 진행(ADR-089 순서: 8-21 → 평가 실행 도구 → Phase 9).

## 1. 목적

- 2026-10-01 사용자 결정(ADR-086)의 독립 학교 평가 규칙을 도구로 지킨다.
  - ② 질문과 정답은 DB를 만든 팀원이 먼저 고정한다.
  - ③ AI(개발자)는 평가 전까지 그 DB와 질문을 보지 않는다.
  - ④ 첫 실행 결과는 그대로 남기고, 그 뒤 수정은 따로 표시한다.
- 평가는 화면과 같은 챗봇 API로 한다. 모델 호출은 질문당 한 번이다.

## 2. 팀원이 하는 일

1. 다른 대학 공개 수강편람을 참고해 PostgreSQL DB를 흉내 내어 만든다. 학교명과 교수명은 지어낸다.
2. 화면에서 그 DB를 등록하고, 검토와 게시를 직접 한다. 관리자 검토가 곧 L2다.
3. 질문 파일을 쓴다(아래 형식). 정답은 과목 코드 목록이거나 `"decline"`이다. `"decline"`은 그 DB에 없는 정보라서 거절이 맞는 질문이다.
   - 정답 코드는 팀원이 자기 DB에서 직접 SQL로 뽑는다.
4. 파일을 `.schema-adaptive/blind-evals/<평가 ID>/questions.json`에 둔다. 이 폴더는 git에 올라가지 않는다.
5. `check`로 형식만 확인한다. 이 명령은 질문 내용을 출력하지 않는다.

## 3. 질문 파일 형식 (`blind-eval-questions-v1`)

```json
{
  "format": "blind-eval-questions-v1",
  "database_id": "school-a",
  "author": "팀원 이름",
  "fixed_at": "2026-10-10",
  "questions": [
    {"id": "a01", "text": "컴공 전공필수 과목 보여줘", "expect": {"codes": ["CS101", "CS205"]}},
    {"id": "a02", "text": "강의실이 공학관인 수업", "expect": "decline"}
  ]
}
```

- 질문은 1~200개. `id`는 영문·숫자·밑줄·하이픈으로 쓰고 서로 달라야 한다. `text`는 1~500자다.
- `codes`는 과목 코드(챗봇 결과 카드의 과목 코드)이며 1~150개다. 결과가 너무 많은 질문은 좁혀서 쓴다(챗봇은 최대 200행).

## 4. 실행과 기록

- `python -m backend.adaptive.evaluation.blind_eval check <평가 ID>`
  - 형식 검사만 한다. 개수와 오류 위치(질문 번호)만 출력한다.
- `python -m backend.adaptive.evaluation.blind_eval run <평가 ID> [--base-url http://127.0.0.1:8123]`
  - 질문을 화면과 같은 챗봇 API로 보낸다. 주소는 loopback만 받는다.
  - 결과를 `run-001.json`, `run-002.json` …으로 저장한다. 이미 있는 파일은 덮어쓰지 않는다.
  - 첫 실행 때 질문 파일의 SHA-256을 고정한다(`questions.lock.json`). 그 뒤 질문 파일이 바뀌면 실행을 거부한다(질문을 고치려면 새 평가 ID로 한다).
  - 각 실행에 git commit, 모델 이름(`backend/.env`의 `OLLAMA_MODEL`), 시각, 등록 게시 버전을 함께 남긴다.
- `python -m backend.adaptive.evaluation.blind_eval report <평가 ID>`
  - 실행별 집계를 출력한다. 첫 실행은 "첫 실행"으로, 그 뒤는 "재실행(commit …)"으로 표시한다.

## 5. 채점

- 정답이 코드 목록인 질문
  - `PASS`: 결과 카드의 과목 코드 집합이 정답과 같다.
  - `FAIL`: 다르다. 빠진 개수와 남는 개수를 남긴다.
  - `CAPPED`: 결과가 200행 상한에 닿아 비교할 수 없다.
- 정답이 `"decline"`인 질문: `UNSUPPORTED` 또는 `NOT_UNDERSTOOD`면 `PASS`다.
- 오류 응답(HTTP 오류, 연결 실패)은 `ERROR`다. 실행은 멈추지 않는다.
- 기록에는 응답 상태, 행 수, 이해한 조건, 빠진·남는 코드 수만 남긴다. 결과 행 전체는 저장하지 않는다.

## 6. 제약

- 질문 파일과 실행 기록은 `.schema-adaptive/` 아래에만 둔다. 문서에는 집계만 쓴다.
- 이 도구는 DB에 직접 접속하지 않는다. 화면과 같은 API만 쓴다.
- AI가 만든 개발용 DB(8-17, 8-21)로는 도구 동작만 확인한다. 독립 평가 수치로 쓰지 않는다.
- push하지 않는다.

---

<a id="prompts-phase8-23-ai-mapping-improvement"></a>

## 원본: prompts/phase8-23-ai-mapping-improvement.md

# Phase 8-23 지침: AI mapping 제안 개선

작성일: 2026-10-06. 사용자 결정: 진행.

## 1. 목적

- 화면 등록 때 AI가 항목마다 고르는 첫 후보가 더 자주 맞게 한다. 관리자가 고칠 양을 줄이는 것이 목표다.
- 관리자 검토는 그대로 필수다. 제안은 검토의 출발점일 뿐이다.

## 2. 측정

- 도구: `python -m backend.adaptive.evaluation.school_mapping_check --label <이름> [--repeat N] [--rules-only]`
  - 화면 등록과 같은 경로로 제안을 만든다: 규칙 → 예시 값 → 로컬 모델 → 후보 정렬.
  - 대상은 AI가 만든 개발용 schema 5개다: `other_school`, `wide_school`, `code_school`, `period_school`, `packed_school`.
  - 항목마다 첫 후보가 개발자 정답과 같은지 센다.
- FK로 이어진 column은 같은 값을 담으므로 둘 다 정답으로 본다(예: `gangjwa.gm_cd`와 `gyogwamok.gm_cd`). 8-21 등록 때의 "AI 정답 수"는 이것을 구분하지 않았다.
- 모델 출력이 흔들리므로 3번 반복해 합계로 본다.
- **이 수치는 개발 진행 확인용이다.** 정답과 schema 모두 AI가 만들었으므로 처음 보는 학교에 대한 적응성 근거가 아니다(ADR-086). 평가는 팀원 DB로 한다.

## 3. 바꿀 후보 (측정을 보고 고른다)

1. **예시 값 빠짐:** 값 종류가 10개를 넘는 column은 예시 값이 하나도 prompt에 들어가지 않는다. 이름·코드·교수명 column이 그렇다. 8-21의 표본 읽기로 10개 예시를 넣는다.
2. 그 밖의 원인은 기준선의 틀린 항목을 보고 정한다. 공통 ontology(항목 이름·별칭·설명)는 바꾸지 않는다(바꿔야 하면 사용자 결정).

## 4. 지킬 것

- 개발용 schema 5개에 맞춘 특수 규칙(이 schema의 column 이름을 코드에 넣는 것 등)은 만들지 않는다.
- 예시 값은 loopback 모델 prompt에만 들어가고 저장하지 않는다(8-12와 같음).
- prompt가 바뀌면 version을 올린다.
- push하지 않는다.

---

<a id="prompts-phase8-24-code-table-import"></a>

## 원본: prompts/phase8-24-code-table-import.md

# Phase 8-24 지침: 코드표에서 값 사전 가져오기

작성일: 2026-10-06. 사용자 결정: 진행.

## 1. 목적

- 공통코드 table을 쓰는 학교에서, 관리자가 코드마다 값 사전을 입력하는 대신 코드표를 골라 한꺼번에 넣게 한다.
- 관리자 편의 기능이다. 챗봇의 답하는 능력은 바뀌지 않는다(8-21 A에서 값 사전만 있으면 14/14).

## 2. 동작

- 관리자가 고르는 것:
  - 어느 field의 사전인지
  - 코드표 table, 코드 column, 이름 column
  - 필요하면 구분 column과 값(예: `code_group = ISU`)
- 서버(읽기 전용):
  - 그 table에서 (코드, 이름) 쌍을 글자로 읽는다. 한 번에 64개까지다. 더 많으면 거절하고 구분 column으로 좁히라고 안내한다.
  - 그 field의 실제 데이터에 있는 코드만 넣는다. 나머지는 건너뛴 수로 알린다.
  - 이미 있는 항목은 이름(표시)만 바꾸고 약어는 그대로 둔다.
  - 거절하는 경우:
    - 한 코드에 이름이 둘이거나, 두 코드가 같은 이름이다(값 사전 규칙상 한 표기는 한 값).
    - 64개를 넘는다.
    - 교시표 field다(8-21 B 규칙이 따로 있음).
- 넣은 항목은 기존 값 사전과 같다. 게시해야 챗봇에 반영되고, 게시 때 저장 값 확인도 그대로 받는다.
- 화면: 값 사전 패널 안의 "코드표에서 가져오기". table·column 목록은 확장 항목 API가 주는 catalog를 쓴다.

## 3. 검증

- 8-21 `code_school`을 값 사전 입력 없이 등록한 뒤, 세 그룹(ISU·YOIL·HAKGI)을 가져와 게시한다. 코드 관련 고정 질문을 다시 돌린다.
- 화면 패널에서도 한 번 가져와 본다.

## 4. 제약

- 새 SQL은 읽기 전용 값 읽기뿐이다. table·column은 등록 catalog에 있는 것만 받는다.
- push하지 않는다.

---

<a id="prompts-phase8-25-time-expressions"></a>

## 원본: prompts/phase8-25-time-expressions.md

# Phase 8-25 지침: 챗봇의 시간 표현 안정화

## 배경

- 2026-10-06 사용자 결정으로 진행한다. 계획의 "질문 이해 개선은 L1 질문 세트를 확보한 뒤"에 대한 예외다.
  - 예외 이유: 시간 표현은 학교·DB와 무관한 한국어 규칙이다. 개발용 질문에 맞춘 개선이 아니다.
- 지금까지 본 흔들림(모델 출력, temperature 0에서도 질문에 따라 다름):
  1. "금요일 오전 수업"에서 "오전"을 빼먹는다(8-20 q11, 같은 질문에 맞는 답·틀린 답이 번갈아 나옴).
  2. "오후 3시 이후에 시작하는 수업"을 종료 시간 조건으로 이해한다(8-21 B b01).
  3. "이후"를 초과(`>`)로 받아 15:00에 시작하는 수업이 빠진다.

## 목표

질문 속 시간 표현을 닫힌 규칙으로 찾고, 그 뜻을 모델에 힌트로 준다. 모델 출력이 그 뜻과 다르면 고친다. 모델 호출은 그대로 한 번이다.

## 규칙(언어 규칙, DB 데이터 아님)

1. **시각.** `N시`, `N시 M분`, `N시 반`, `HH:MM`. 앞에 `오전`·`오후`·`아침`·`저녁`·`밤`이 붙을 수 있다.
   - `오후`·`저녁`·`밤` + 1~11시 → 12를 더한다.
   - `오전`·`아침` → 그대로(오전 12시는 해석하지 않는다).
   - 앞말이 없으면: 13~23시는 그대로, 8~12시는 그대로, 1~7시는 오후로 본다(수업 시간대).
   - `시간`·`시험`·`시수`처럼 `시` 뒤에 다른 낱말이 이어지면 시각이 아니다.
2. **방향.**
   - `이후`·`부터`·`후`·`뒤` → 이상(`>=`). 그 시각에 시작하는 수업도 포함한다.
   - `이전`·`전` → 미만(`<`).
   - `까지` → 이하(`<=`).
   - 방향 말이 없으면 같음(`=`).
3. **항목.**
   - 시각 앞뒤 가까이에 `시작`이 있으면 시작 시간, `끝`·`종료`·`마치`가 있으면 종료 시간이다. 둘 다 있으면 가까운 쪽이다.
   - 둘 다 없으면 `까지`는 종료 시간, `이후`·`부터`·`후`·`뒤`·방향 없음은 시작 시간이다. `전`·`이전`은 뜻이 갈려 힌트를 만들지 않는다.
4. **오전·오후만 있을 때.** `오전` → 시작 시간 < 12:00. `오후` → 시작 시간 >= 12:00. 항목 말(`끝나는`)이 가까이 있으면 그 항목이다.
5. 찾은 조건이 셋 이상이면 힌트를 만들지 않는다(모델에 맡긴다).
6. `야간`·`저녁 수업`처럼 시각이 없는 말은 이번에 다루지 않는다.

## 쓰는 곳

1. **힌트.** 8-13 값 힌트 줄에 `{"text","field","operator","value"}`로 넣는다. 시간 힌트를 앞에 두고, 합쳐서 8개까지다. 연산자 없는 힌트만 있을 때 prompt는 지금과 한 글자도 다르지 않다.
2. **고침(닫힌 규칙).** 모델 출력의 시간 조건(시작·종료 시간 field)과 찾은 조건을 맞춘다.
   - 개수가 같으면 field·연산자·값을 찾은 조건으로 바꾼다. 같은 값끼리 먼저 짝짓고, 나머지는 순서대로 짝짓는다.
   - 모델이 시간 조건을 하나도 내지 않았으면 찾은 조건을 더한다. 8-13의 "조건을 더하지 않는다"에 대한 예외다. 질문에 적힌 시간이고 닫힌 규칙으로만 정해지며, 화면의 "이해한 조건" 카드에 보인다.
   - 개수가 다르면(모델이 더 많거나 적음) 그대로 둔다.
3. **이 DB에 없는 시간.** 찾은 조건의 항목이 그 DB에 없으면(`이 DB에 없음`) 모델 출력과 관계없이 "정보 없음"으로 답한다. 예: 묶인 시간표 DB의 "금요일 오전 수업".
4. 고친 출력도 QueryIR 검증·compiler·policy·read-only executor를 그대로 지난다.

## 하지 않는 것

- 시간 말고 다른 조건(학점, 정원 등)은 건드리지 않는다.
- 공통 ontology를 바꾸지 않는다.
- 레거시 기존 시스템 경로는 건드리지 않는다.

## 확인

1. 단위 테스트: 규칙 표, 고침 규칙, prompt 줄, chat 경로(빠진 조건 추가, 종료→시작, 초과→이상, 없는 항목은 정보 없음).
2. live(테스트 포트 8124, 합성 DB와 레거시 profile 읽기 전용):
   - `other_school` q11을 여러 번 보내 흔들림이 없어졌는지.
   - `period_school` b01, `packed_school` q11·b01(거절 기대).
   - 시간 표현이 든 새 질문 몇 개를 **실행 전에** 정답과 함께 고정해 둔다.
   - 레거시: "월요일 오전 수업"(200건 상한), 컴융 전공 82건 불변.
3. Luna review-only → 확인된 지적만 고침 → 전체 테스트 → 문서·커밋.

---

<a id="prompts-phase9-01-l1-review-table"></a>

## 원본: prompts/phase9-01-l1-review-table.md

# Phase 9-01 지침: L1 검토 표 준비

작성일: 2026-10-06. 사용자 결정: 진행(ADR-086 "L1 검토 표는 Phase 9 진입 때 준비", ADR-089 순서).

## 1. 목적

- L1은 gold 질문마다 사람이 확인하는 관문이다. 확인할 것은 한국어 뜻, gold 조건, 정답 행, 중복 행, 빈 결과다.
- AI는 검토 표만 준비한다. 판정은 사용자가 먼저 하고, 그다음 팀원이 한다. AI는 어떤 사례도 승인으로 바꾸지 않는다.

## 2. 대상

1. 합성 benchmark 70개(`fixture_v1_cases.yaml` 40개, `fixture_v1_independent_cases.yaml` 30개). 같은 합성 seed(개설 강좌 6개) 위에 있다.
2. 팀원 질문 파일(Phase 8-22). 표는 팀원의 private 폴더에만 만들고, 내용을 출력하지 않는다. AI는 평가 실행 전까지 열지 않는다.

## 3. 표

- Markdown(`docs/l1-review/fixture-v1-review.md`): 읽는 용도. 맨 앞에 seed 표, 그다음 사례마다 아래를 보인다.
  - 질문, gold 조건(한국어 이름), 보여 줄 항목, 정답 행 수와 행.
  - 자동 확인 메모: 빈 결과, 같은 내용 행, 질문 표현과 조건 값의 차이, 경계값(초과/이상), 같은 뜻 다른 표현 묶음.
- CSV(`docs/l1-review/fixture-v1-review.csv`, Excel용 UTF-8 BOM): 판정·의견·검토자·검토일 칸을 비워 둔다.
- 두 파일 모두 gold 파일 지문(SHA-256)을 담는다. gold가 바뀌면 이전 판정은 이어받지 않는다.

## 4. 판정 기록

- `python -m backend.adaptive.evaluation.l1_review record <CSV>`
  - 판정은 승인·수정·보류 중 하나다. 판정한 행에는 검토자를 적는다. 수정·보류에는 의견을 적는다.
  - 빈 판정은 "미검토"로 센다.
  - 기록(`<CSV>.record.json`)에는 gold 지문, 판정 개수, 사례별 판정이 들어간다.
- gold의 `review_status`는 이 도구가 바꾸지 않는다. 승인을 gold에 반영하는 것은 기록을 보고 사용자가 정한다.

## 5. 제약

- 합성 seed만 쓴다. 실제 학생 정보나 비밀값은 없다.
- push하지 않는다.

---

<a id="prompts-phase9-02-l4-materials"></a>

## 원본: prompts/phase9-02-l4-materials.md

# Phase 9-02 지침: L4 사용성 점검 진행 자료

## 배경

- L4는 "개발에 참여하지 않은 최종 사용자가 화면 공개 전에 써 보는" 관문이다([human-validation-checkpoints.md](human-validation-checkpoints.md)).
- 2026-10-01 사용자 결정(ADR-086): L4는 그대로 둔다. 개발에 참여하지 않은 사람 몇 명이 써 본다. 사람 섭외와 진행은 팀이 한다.
- 이번 단계는 그 진행에 쓸 자료만 만든다. 코드 변경은 없다. AI는 L4를 진행하거나 통과로 표시하지 않는다.

## 만들 것 (`docs/l4-usability/`)

1. **진행자 안내**: 준비(환경, 계정 없음, SQL 표시 끔), 참가자 조건, 진행 순서와 대본, 과제별 관찰 요점, 끝난 뒤 되돌리기, 결과 정리와 관문 기록.
2. **참가자 과제 카드**: 질문 문장을 주지 않고 목표만 준다. 참가자가 스스로 어떻게 묻는지가 관찰 대상이다.
   - 학생 과제: 검색, 시간 조건, 복합 조건, 없는 정보, 0건, 개인 추천 요청, 이해한 조건 읽기.
   - 관리자 과제(선택): 지어낸 합성 학교 DB를 등록·검토·약어·게시하고 질문.
3. **관찰 기록 양식(CSV)**: 참가자 ID, 과제, 결과(완료/도움 받아 완료/포기/잘못된 결과를 믿음), 시간, 헷갈린 곳, 개입, 심각도.
4. **사후 질문**: 5점 척도 몇 문항과 열린 질문.
5. **결과 정리 양식**: 문제 목록(심각도), 고칠 것과 한계로 남길 것, 관문 기록 줄.

## 원칙

- 데이터: 학생 과제는 레거시 DB(공개 수강편람 기반, ADR-086), 관리자 과제는 지어낸 `other_school`. 외부 공개 자료가 아니므로 교수명을 가리지 않아도 되지만, 결과 정리에는 교수명을 쓰지 않는다.
- 개발 옵션의 SQL 표시는 끈다(`SHOW_SQL=0`). 최종 사용자 화면이 대상이다.
- 기록에는 이름·학번 같은 개인정보를 남기지 않는다. 참가자는 P1, P2 …로 적는다. 채운 기록 원본은 git 밖(`.schema-adaptive/l4/`)에 두고, 저장소에는 개인정보 없는 요약만 올린다.
- 통과 판단은 사람이 한다. 기준은 체크포인트 문서의 L4 기준(주요 오해·위험 행동을 고치거나 미해결 제한으로 명시)이다.

## 확인

- 문서 안의 화면 이름·문구·명령이 지금 코드와 맞는지 확인한다(`ChatPopup.tsx`, `chatQuery.ts`, `demo-procedure.md`).
- Luna review-only로 사실 오류와 개인정보·관문 문구를 확인한다.
- 코드 변경이 없으므로 pytest는 돌리지 않는다. diff check만 한다.

---

<a id="prompts-phase9-03-env-check"></a>

## 원본: prompts/phase9-03-env-check.md

# Phase 9-03 지침: 팀원용 환경 점검 명령

## 배경

- 팀원이 평가용 DB를 만들고 화면에서 등록·평가하려면(ADR-086, [teammate-evaluation-db-guide.md](teammate-evaluation-db-guide.md)) 각자 PC에서 `npm run dev:all`과 평가 도구가 돌아가야 한다.
- 지금은 무엇이 빠졌는지 서버를 켜 보고 오류로 알아내야 한다. 명령 하나로 확인하고 해결 방법을 알려 준다.
- L7(재현성) 준비에도 쓴다. 이 명령의 통과가 L7 통과는 아니다.

## 명령

```bash
python -m backend.adaptive.env_check
```

## 확인할 것 (읽기 전용)

1. Python 버전, 저장소의 `.venv`, 필요한 Python 패키지(설치 여부만, import하지 않음).
2. Node·npm, `node_modules`(vite, concurrently).
3. `backend/.env`: `DATABASE_URL`, `OLLAMA_MODEL` 등 키가 있는지. 값은 출력하지 않는다.
4. PostgreSQL:
   - `DATABASE_URL`(레거시 DB)에 읽기 전용 세션으로 접속해 `SELECT 1`. public schema의 table·view 수.
   - `PG_SOURCE_*` 연결마다 접속해 schema 이름 목록(팀원이 만든 schema가 보이는지).
   - `SCHEMA_ADAPTIVE_SYNTHETIC_POSTGRES_DSN`(있으면): loopback·`schema_adaptive_synthetic` 규칙 검사 후 접속.
5. Ollama: base URL이 loopback인지, `/api/tags`에 챗봇 모델과 mapping 모델이 있는지. proxy를 쓰지 않는다.
6. 포트 8000·8123·5173이 이미 쓰이는지(켜진 서버가 있으면 경고만).
7. `.schema-adaptive/`에 쓸 수 있는지(임시 파일을 만들고 바로 지운다).

## 원칙

- DSN·비밀번호는 출력하지 않는다. 모델 이름과 연결 이름(`PG_SOURCE_` 뒤 이름)은 확인에 필요해 보이되, 이상한 글자가 있으면 가린다. 접속 실패는 오류 종류 이름만 보인다(오류 문장에 호스트·사용자가 들어갈 수 있다).
- 쓰기 SQL을 실행하지 않는다. 세션은 읽기 전용, 접속 시간 제한 5초.
- 결과는 줄마다 `[OK]`·`[WARN]`·`[FAIL]`과 해결 방법. `FAIL`이 하나라도 있으면 종료 코드 1.
- 의존성(접속, HTTP, 포트, 명령 실행)은 주입할 수 있게 해 테스트에서 실제 네트워크를 쓰지 않는다.

## 확인

- 단위 테스트: 모두 정상, `.env` 없음, 접속 실패 문장에 비밀이 있어도 출력에 없음, loopback 아닌 Ollama·합성 DSN, 모델 없음, 포트 사용 중, 종료 코드.
- 실제 PC에서 한 번 실행해 결과를 확인한다(값이 출력되지 않는지 포함).
- 팀원 안내([teammate-evaluation-db-guide.md](teammate-evaluation-db-guide.md))에 명령을 적는다.
- Luna review-only → 확인된 지적만 고침 → 전체 테스트.

---

<a id="prompts-phase9-04-rehearsal-school"></a>

## 원본: prompts/phase9-04-rehearsal-school.md

# Phase 9-04 지침: 팀원 평가 리허설 학교

## 배경

- 2026-10-08 사용자 요청: 팀원 평가 전에 AI가 새 PostgreSQL DB를 하나 만들어 시험해 본다.
- 이 DB와 질문·정답은 AI가 만들었으므로 **개발용**이다. 적응성 근거(E4/L6)가 아니다(ADR-086).
- 목적:
  1. 팀원이 할 평가 과정(DB → 질문 고정 → 등록·검토·게시 → `blind_eval` 첫 실행 → 분석 → 재실행)을 그대로 해 보고 막히는 곳을 찾는다.
  2. 지금까지 다루지 않은 저장 모양에서 무엇이 안 되는지 찾는다.

## 리허설 학교 `rehearsal_school` (합성 DB 전용)

`backend/adaptive/fixtures/rehearsal_school.py` (`create|drop|check`). 내용은 `other_school`과 같은 과목이고 저장 모양만 다르다.

| 새 모양 | 예 |
| --- | --- |
| 학점이 `학점-이론-실습` 글자 | `3-2-2` |
| 연도와 학기를 한 칸에 | `2026-1` |
| 시작~종료를 한 칸에 | `09:00-10:15` |
| 과목코드에 분반이 붙음 | `SW1001-01` |
| 이수구분이 영문 코드 | `MR` 전공필수, `ME` 전공선택, `GR` 교양필수, `GE` 교양선택, `FE` 일반선택 |
| 요일 한 글자, 교수명이 강의 table에 | `월`, `prof_name` |

## 순서 (팀원 안내와 같게)

1. 합성 DB에 schema를 만든다.
2. 질문 14개와 정답(강의번호 `lec_no` 목록 또는 거절)을 **첫 실행 전에** 고정한다. 파일: `.schema-adaptive/blind-evals/rehearsal-v1/questions.json`(git 밖). `blind_eval check`로 형식을 확인한다.
3. 실행 전 예상을 결과 문서에 먼저 적는다.
4. 테스트 포트(8124)에서 관리자처럼 등록한다.
   - AI 첫 후보를 기록한다.
   - 정답 지도로 승인·다시 지정한다.
   - 값 사전에 약어·영문 코드 뜻을 넣는다.
   - 게시한다. 막히면 관리자가 할 수 있는 일(이 DB에 없음)만 하고, 막힌 내용과 관리자 작업 수를 기록한다.
5. `blind_eval run rehearsal-v1 --base-url http://127.0.0.1:8124`로 첫 실행을 고정한다. `report`로 확인한다.
6. 틀린 질문을 분석해 "학교별 데이터로 풀 것"과 "일반 형식이라 코드로 풀 후보"로 나눈다.
7. **코드는 고치지 않는다.** 고칠지와 방법은 사용자가 결과를 보고 정한다.
8. 테스트 등록을 지운다. schema는 다른 합성 schema처럼 남긴다.

## 지키는 것

- 레거시 DB는 건드리지 않는다. 합성 DB(`schema_adaptive_synthetic`)에만 쓴다.
- 사용자 서버(8000·8123·5173)는 건드리지 않는다.
- 결과는 개발용 수치로만 적는다. 사람 관문은 승인하지 않는다.
- Luna 리뷰는 `timeout 1200 ... < /dev/null`로 돌린다(AGENTS.md).

---

<a id="prompts-phase9-05-raw-time-and-unasked-select"></a>

## 원본: prompts/phase9-05-raw-time-and-unasked-select.md

# Phase 9-05 지침: 리허설에서 찾은 문제 1·2 고치기

## 배경

- 9-04 리허설 학교에서 찾은 문제 중 1·2번을 사용자 결정(2026-10-08)으로 고친다. 조건: **기존 레거시 결과가 바뀌지 않아야 한다.**
- 1번: 종료 시간이 `09:00-10:15` 글자 칸에 변환 없이 게시되어, 글자 비교로 조용히 틀린 답을 냈다.
- 2번: 모델이 보여 줄 항목(`select`)에 분반만 넣으면, 분반이 없는 DB에서 답 전체가 "정보 없음"이 되었다.

## 고칠 것

1. **게시 검사.** 시간 항목이 변환 없이 글자 칸에 남으면, 그 칸의 값이 모두 0을 채운 `HH:MM`이어야 한다. 아니면 그 항목으로 게시를 막는다.
   - 대상 밖: `time` 형식 칸(`TIME_TO_HHMM`), `0900` 꼴(`HHMM_TEXT_TO_HHMM`), 교시표가 있는 항목, 빈 칸.
   - 값이 많아 표본만 읽은 경우에도 표본에 맞지 않는 값이 있으면 막는다.
2. **챗봇 거절 규칙.** select에만 있고(조건에는 없고) 질문이 이름·별칭으로 말하지 않은 "이 DB에 없음" 항목은 거절하지 않고 select에서 뺀다.
   - 질문이 그 항목을 말하거나("학기 알려줘") 조건에 쓰면 지금처럼 거절한다.
   - 빼서 select가 비면 쓸 수 있는 카드 항목 하나를 넣는다(QueryIR은 select가 비면 안 된다). 카드는 어차피 select를 고정 항목으로 바꾼다.

## 확인

1. 회귀 테스트: 범위 글자·0 없는 시각·초 단위·글자·섞인 값은 막힘, `HH:MM` 글자는 변환 없이 게시, `0900`은 기존 변환. 말하지 않은 select 항목은 답, 말하거나 조건에 쓴 항목은 거절, 모르는 항목은 그대로 "이해하지 못함".
2. 레거시: 고치기 **전**에 테스트 포트(8124)에서 비교 30문항과 고정 확인 6개를 기록하고, 고친 뒤 같은 질문의 상태·행 수·과목 코드 목록·거절 사유가 모두 같은지 본다. 레거시 시간 칸의 형식(`time`)도 확인한다.
3. 리허설 학교를 같은 관리자 순서로 다시 등록·게시하고 `blind_eval`을 재실행한다(첫 실행 기록은 그대로).
4. Luna review-only(`timeout 1200 ... < /dev/null`) → 확인된 지적만 고침 → 전체 테스트 → 문서·커밋.

---

<a id="prompts-phase9-06-numeric-columns"></a>

## 원본: prompts/phase9-06-numeric-columns.md

# Phase 9-06 지침: 숫자 항목을 글자 칸에 그대로 게시하지 않기 (리허설 문제 3)

## 배경

- 9-04 리허설 문제 3: 학점을 `3-2-2` 글자 칸에 연결해도 게시가 막히지 않았다. 학점을 물으면 SQL이 글자 칸과 숫자를 비교하다 실패해 HTTP 500이 났다.
- 2026-10-09 사용자 결정: 추천대로 진행한다. 조건은 9-05와 같다(기존 레거시 결과가 이상해지지 않을 것).
- 9-04 리뷰 결론: 값이 숫자로 읽히는지만 보는 검사는 부족하다. `3`처럼 숫자로 읽히는 값도 글자 칸이면 SQL에서 글자=숫자 비교가 실패한다.

## 고칠 것

- 게시 검사(`plan_publish`)에서 공통 항목 중 숫자 항목(학점, 대상 학년, 개설 연도, 정원, 수강 인원)은 다음 중 하나일 때만 게시한다. 아니면 그 항목으로 게시를 막는다.
  1. 칸이 숫자 형식이다(`integer`, `numeric` 등).
  2. 검증된 글자→숫자 변환이 정해졌다(지금은 `3/3/0` 꼴의 `SLASH_FIRST_NUMERIC`).
  - (리뷰 뒤 갱신) 값 사전 예외는 두지 않는다. 숫자 항목에는 값 사전을 쓸 수 없다.
- (리뷰 뒤 추가) 글자→숫자·시간 변환이 정해진 칸에 빈 글자가 있으면 막는다.
- `3-2-2`를 실제로 읽는 새 변환은 이번에 만들지 않는다. 관리자는 그 항목을 "이 DB에 없음"으로 둔다.
- DB별 확장 항목은 이번 대상이 아니다.

## 레거시 확인 (읽기 전용, 고치기 전에)

- 레거시 public schema: 정원·수강 인원·대상 학년·개설 연도는 `integer`, 학점(`credit_hours`)은 글자 칸이지만 `3/3/0` 꼴이라 `SLASH_FIRST_NUMERIC`이 붙는다. 막힐 항목이 없다.
- 합성 학교 다섯 곳의 숫자 항목도 모두 숫자 형식이다.

## 확인

1. 회귀 테스트: `3-2-2`·`3` 글자 칸은 막힘, 숫자 칸과 `3/3/0` 글자 칸은 게시, 값 사전이 있는 항목은 그대로.
2. 레거시 36문항(9-05와 같은 질문)이 고친 뒤에도 같은지 본다. 화면 등록 경로로 레거시 public schema를 등록·게시해 막히지 않는지도 본다.
3. 리허설 학교를 다시 등록해 학점이 게시에서 막히는지 보고, `blind_eval`을 재실행한다(첫 실행 기록 유지).
4. Luna review-only(`timeout 1200 ... < /dev/null`, 다른 명령 뒤에 잇지 않음) → 확인된 지적만 고침 → 전체 테스트 → 문서·커밋.

---

<a id="prompts-phase9-07-publish-block-reasons"></a>

## 원본: prompts/phase9-07-publish-block-reasons.md

# Phase 9-07 지침: 게시가 막힌 항목과 이유를 한 번에 보이기 (리허설 문제 6)

## 배경

- 9-04·9-06 리허설에서 게시 막힘이 "같은 칸 → 시간 형식 → 숫자 형식" 순서로 하나씩 나와, 관리자가 게시를 네 번 눌렀다.
- 화면은 이유와 관계없이 "아직 검토가 끝나지 않은 항목이 있습니다"만 보였다. 막힌 항목 이름도 보이지 않았다.
- 2026-10-10 사용자 결정: 추천대로 진행한다. 팀원이 직접 DB를 등록할 때 가장 먼저 겪을 불편이다.

## 고칠 것

1. **서버(`plan_publish`).** 모든 게시 검사를 끝까지 돌리고, 막힌 항목마다 이유를 모아 한 번에 돌려준다.
   - 이유: 검토 안 함, 칸 없음, 같은 칸(다른 항목 이름 포함), 연결된 항목 없음, 쓰지 않는 값 사전, 교시표 필요, 시간 형식, 숫자 형식, 빈 값, 값 사전 값이 DB에 없음.
   - 응답의 기존 `fields`(이름 목록)는 그대로 두고 `reasons`(항목, 한국어 이름, 이유, 같은 칸이면 다른 항목)를 더한다.
   - 런타임 검증 실패(이름 없는 거절)와 SQLite 경로는 그대로다.
2. **화면.** `reasons`를 엄격히 읽고(모르는 key·막히지 않은 항목·잘못된 이유 코드는 응답 형식 오류), 항목마다 한 줄씩 한국어로 이유와 할 일을 보인다. 오류 칸은 줄바꿈을 보인다.

## 확인

1. 서버 테스트: 한 번의 게시가 같은 칸·시간 형식·숫자 형식을 함께 알림, 검토 안 함과 다른 이유가 함께 나옴, 정상 게시는 그대로.
2. 화면 테스트: 읽기·거부, 한 줄씩 메시지, 조사(와/과), 이유 없을 때 기존 문구.
3. 리허설 학교를 등록해 게시 한 번으로 막힌 항목 넷이 모두 나오는지 API와 화면(5174→8124)에서 본다.
4. tsc·ESLint·vite build·Node 테스트, Luna review-only(`timeout 1200 ... < /dev/null`, 단독 실행), 전체 pytest.

---

<a id="prompts-phase9-08-chat-model-choice"></a>

## 원본: prompts/phase9-08-chat-model-choice.md

# Phase 9-08 지침: 적응형 챗봇의 기본 모델을 일반 모델로 바꾸고, 쓸 모델을 고르게 하기

## 배경

- 2026-10-12, 리허설 학교에서 "컴융 전공 보여줘"가 `과목 코드 = 1101-7019`로 바뀌었다.
  - 이 코드는 리허설 학교에 없고, 레거시 DB(인문대학 국어국문학과)에만 있다.
  - 레거시 DB로 미세 조정한 챗봇 모델(`OLLAMA_MODEL`, `text2sql_qwen`)이 레거시 값을 외워서 낸 것이다(계획 문서의 "위험 3 CNU 데이터 암기").
- 2026-10-12 사용자 결정:
  - 특정 DB로 학습한 모델을 적응형 챗봇의 기본으로 쓰는 것은 취지와 맞지 않는다. 규칙을 덧붙이지 말고 모델을 바꾼다.
  - 어떤 모델을 쓸지 사용자가 고를 수 있게 한다.
- 계획 문서 10.1·10.2절: 미세 조정 모델은 기준선과 "초기 후보"일 뿐이다. 목표는 "하나의 공통 모델 + 학교별 지도"다.

## 고칠 것

1. **모델 선택지(서버가 정함).**
   - `general`(일반 모델, 기본):
     - `CHAT_MODEL`, 없으면 `qwen2.5-coder:7b-instruct`.
     - 요청 형식은 `model-template`.
   - `legacy-tuned`(레거시 학습 모델, 비교용):
     - 레거시 챗봇과 같은 `OLLAMA_MODEL`.
     - 요청 형식은 지금과 같은 `alpaca-raw-no-system-v1`.
   - 두 이름이 같으면 `legacy-tuned` 하나만 둔다.
   - 하나가 실패하면 다른 하나로 시작한다.
   - `OLLAMA_MODEL`이 없어도 일반 모델로 챗봇이 켜진다.
2. **서버 API.**
   - `GET /api/adaptive/v1/chat/models` → `{version, default, models:[{id, label, tag}]}`.
     - 모델 이름(tag)은 비밀이 아니고 env_check도 보인다.
     - DSN·URL은 보내지 않는다.
   - 질문 요청에 `model`(선택지 id)을 선택으로 더한다.
     - 없으면 기본 모델이다.
     - 모르는 id는 `CHAT_INVALID_REQUEST`다.
     - 브라우저는 모델 이름이 아니라 서버가 정한 id만 보낸다.
   - 답 형식은 바꾸지 않는다.
   - 모델 호출은 여전히 질문당 한 번이다. QueryIR 검증·compiler·policy·executor는 그대로다.
3. **기존 흐름.**
   - 예전 명확화 흐름(SQLite 합성 DB)은 기본 모델을 쓴다.
   - 레거시 챗봇(8000, `backend/llm.py`)은 바꾸지 않는다. 계속 `OLLAMA_MODEL`이다.
4. **화면.**
   - 새 챗봇(QueryIR 경로)에서 선택지가 둘 이상이면 "모델" 고르기를 보인다.
   - 기본값은 서버의 기본이다.
   - 답을 만드는 동안에는 바꿀 수 없다.
   - 모델을 바꾸면 대화에 한 줄로 알린다.
5. **평가 도구.**
   - `blind_eval run --model <id>`로 모델을 고른다.
   - 기록의 `model`에는 서버가 알려 준 id와 이름을 남긴다. `.env`를 읽지 않는다.
6. **환경 점검.**
   - `CHAT_MODEL`(없으면 기본값)이 Ollama에 있는지 본다.
   - `OLLAMA_MODEL`은 "레거시 학습 모델"로 보인다.

## 확인

1. 서버 테스트:
   - 선택지 구성(기본값, 같은 이름, `OLLAMA_MODEL` 없음, 하나 실패)
   - 모델 목록 API
   - `model` 필드: 없음·알려진 id·모르는 id·잘못된 형
   - 고른 모델만 한 번 불림
   - 기존 단일 모델 호출은 그대로
2. 화면 테스트:
   - 목록 읽기·거부
   - 요청 본문에 `model`
   - 하나뿐이면 고르기를 숨김
3. 다시 재기. 같은 질문을 두 모델로 각각 묻고 나란히 표로 남긴다.
   - 레거시 고정 질문
   - 가짜 학교 다섯 곳
   - 리허설 학교(재실행으로 기록)
   - 리허설 학교의 "컴융 전공 보여줘"
4. 마무리:
   - tsc·ESLint·vite build·Node 테스트
   - Luna review-only(`timeout 1200 ... < /dev/null`, 단독 실행)
   - 전체 pytest
   - ADR, 계획 문서 10절, 변경 기록

---

<a id="prompts-phase9-09-openai-chat-model"></a>

## 원본: prompts/phase9-09-openai-chat-model.md

# Phase 9-09 지침: 챗봇 모델 선택지에 ChatGPT API 추가

## 배경

- 9-08에서 일반 모델(`qwen2.5-coder:7b-instruct`)은 레거시 질문에서 미세 조정 모델과 비슷했다. 다른 학교의 학과 약어 질문에서는 약했다.
- 2026-10-12 사용자 결정: 로컬 7B 모델만으로는 한계가 있다. ChatGPT API 선택지를 더한다. API 키는 사용자가 넣는다.
- 같은 날 사용자 결정:
  - 기본 모델은 `gpt-5-mini`다. `OPENAI_MODEL`로 바꿀 수 있다.
  - 값 힌트(질문 단어와 맞는 DB 값)도 보낸다. 다른 모델과 같은 조건으로 비교하기 위해서다.
  - "Legacy 기준선"(예전 레거시 챗봇)은 그대로 둔다. 모델 고르기는 QueryIR 경로에만 있다.

## 고칠 것

1. **선택지 `openai`("ChatGPT(외부 API)").**
   - `backend/.env`에 `OPENAI_API_KEY`가 있을 때만 생긴다.
   - 모델은 `OPENAI_MODEL`이고, 없으면 `gpt-5-mini`다.
2. **모델 호출.**
   - `https://api.openai.com/v1/chat/completions`로 질문당 한 번 부른다.
   - prompt는 일반 모델과 같은 `query-ir-v3`이고, 값 힌트도 같다.
   - QueryIR JSON schema를 `response_format`으로 보낸다(strict 아님).
   - 고정 주소만 쓴다. proxy와 redirect는 따르지 않고, 응답 크기에 상한을 둔다.
   - 실패는 고정 코드(`TIMEOUT`, `PROVIDER_ERROR`)로만 알린다.
   - 이후 QueryIR 검증·compiler·policy·executor는 그대로다.
3. **비밀 유지.**
   - API 키는 서버 안에서만 쓴다.
   - 화면, 상태 줄, 오류, 로그, 평가 기록에 나가지 않는다.
   - 환경 점검은 "있음"만 보인다.
4. **보내는 것.**
   - 보내는 것: 질문, 공통 항목 목록(이름·형식·연산자), 값 힌트.
   - 보내지 않는 것: 물리 table·column 이름, DSN.
5. 화면은 바꾸지 않는다. 선택지 목록을 서버에서 받으므로 자동으로 보인다.

## 확인

1. 가짜 전송으로 서버 테스트를 한다.
   - 요청 모양(주소, 헤더, 모델, schema, 힌트)
   - JSON 읽기
   - 실패 코드
   - 키가 어디에도 나오지 않음
   - 키 없을 때 선택지 없음
   - 화면 고르기로 고른 모델만 한 번 불림
2. 실제 호출은 사용자가 키를 넣은 뒤 화면에서 한다. 비용이 들므로 AI가 키를 넣거나 측정을 돌리지 않는다.
3. Luna review-only, 전체 pytest, compileall, diff check를 돌린다.
4. ADR-099와 문서를 갱신한다.

---

<a id="prompts-phase9-10-direct-sql-postgres"></a>

## 원본: prompts/phase9-10-direct-sql-postgres.md

# Phase 9-10 지침: PostgreSQL 직접 SQL 비교군 (평가 전용)

## 배경

- 사용자의 원래 의도는 "등록한 DB의 실제 구조와 데이터 모양을 보고 모델이 SQL을 만든다"였다. 지금 제품은 공통 항목 mapping → QueryIR → compiler 구조다.
- 두 방식을 같은 조건에서 비교해, 제품 구조는 결과를 본 뒤 사용자가 정한다.
- 지금 직접 SQL 비교군(`backend/adaptive/evaluation/direct_sql.py`, Phase 5-19~22)은 쓸 수 없다.
  - SQLite 합성 fixture에서만 돈다(`_CatalogBoundExecutor`). 제품은 PostgreSQL 전용이다(ADR-089).
  - prompt에 table·column·선언형·PK/FK만 있다. 예시 값과 값 힌트가 없다.
  - 채점이 gold 행 위치 비교라 column이 하나만 더 있어도 틀린다.
  - QueryIR 쪽은 정답 mapping(`fixture_published_mapping`)을 받았다.
- 2026-10-12 사용자 결정(모두 제안대로):
  1. 9-10을 진행한다.
  2. 직접 SQL 쪽 context: table·column·자료형·PK/FK + column마다 예시 값 10개까지 + 값 힌트.
  3. OpenAI로도 직접 SQL 쪽을 돌릴 수 있다. 이때 실제 table·column 이름과 예시 값이 외부로 나간다. 평가 전용 범위에서 ADR-099의 "물리 이름·DB 행을 보내지 않음"을 넓히는 사용자 결정이다. 제품 챗봇의 `openai` 선택지가 보내는 것은 그대로다.
  4. 개발용 비교 질문: 레거시 30, `other_school` 12, `rehearsal_school` 14(합 56).

## 지킬 것 (바꾸지 않음)

- 평가 전용이다. runtime HTTP·prototype·명확화·챗봇 코드는 새 모듈을 import하지 않는다(ADR-048/049 raw SQL fallback 금지). 이를 확인하는 테스트를 둔다(기존 `direct_sql` import 금지 테스트가 있으면 같은 방식으로 넓힌다).
- 화면, 제품 API, 기존 QueryIR 경로, 기존 SQLite `direct_sql.py` 동작은 바꾸지 않는다. 기존 테스트가 그대로 통과해야 한다.
- legacy DB(postgres, public schema)는 읽기 전용이다. DDL·쓰기 없음. 테스트용 schema는 합성 DB `schema_adaptive_synthetic`에만 만든다.
- DSN·비밀번호·API 키는 출력·기록·repr·오류에 나가지 않는다. `backend/.env` 값을 출력하지 않는다.
- 기록에는 SQL 원문, 모델 원문, 결과 행 값, prompt를 남기지 않는다. 결과 코드, 개수, fingerprint(sha256), 시간만 남긴다. (콘솔에 질문 id별 결과 코드는 출력해도 된다.)
- 질문당 모델 호출 한 번, 재시도 없음.
- 공통 ontology는 바꾸지 않는다.
- 사용자 서버(8000·8123·5173)를 건드리지 않는다. 실제 모델 호출(live 측정)은 하지 않는다. 테스트는 가짜 모델·가짜 전송으로 한다. 합성 DB를 쓰는 기존 live 테스트 방식이 있으면 그 방식만 따른다.
- 커밋하지 않는다. 사용자의 `package.json`, `package-lock.json`, `.claude/`는 건드리지 않는다.

## 만들 것

### 1. `backend/adaptive/evaluation/direct_sql_pg.py`

**context 만들기** (등록된 DB 하나, `database_id`로)
- 저장된 등록 정보에서 catalog(table·column·자료형·PK/FK, schema 이름)와 연결 설정을 읽는다. 등록·런타임이 이미 쓰는 loader를 재사용한다. 관리자 mapping·값 사전·변환은 쓰지 않는다(직접 SQL 쪽은 등록만 한 상태를 가정).
- 예시 값: `PostgresDistinctValueReader`의 `sample=True` 경로로 column마다 최대 10개. 값 하나는 40자에서 자른다. 숫자·날짜·시간 column도 포함한다. 읽을 수 없는 column은 건너뛴다.
- 값 힌트(물리 버전): 질문 단어를 낮은 cardinality 글자 column의 실제 값과 맞춘다. `api/value_hints.py`의 `ValueIndex` 일치 방식(전체 값, 값의 단어 조각, 가까운 철자)을 재사용하되 **관리자 값 사전(약어)은 넣지 않는다**. 힌트는 `"<질문 단어>" -> <schema>.<table>.<column> = '<값>'` 꼴로 보인다. 개수 상한은 QueryIR 쪽 값 힌트와 같게 한다.
- context는 결정적(정렬 고정)이어야 한다.

**prompt `direct-sql-pg-v1`** (영어, 기존 `DIRECT_SQL_TEMPLATE` 문체)
- "PostgreSQL database that stores a university course catalog."
- schema-qualified table 이름, column(자료형), PK/FK, column별 예시 값, 값 힌트.
- 규칙: 읽기 전용 SELECT 하나; 목록의 table·column만; 과목 코드 column을 첫 column으로 `AS course_code`로; 분반 column이 있으면 `AS section`으로 함께; 질문이 말한 조건만; 답할 수 없으면 `UNKNOWN`; SQL만, Markdown·설명 없이.
- 질문은 `build_direct_sql_prompt`처럼 분할 치환(중괄호 해석 금지). template sha256을 기록한다.

**SQL 검사** (PostgreSQL 방언, sqlglot `read="postgres"`)
- `extract_sql`과 결과 코드 체계를 재사용한다.
- 문장 하나, 루트는 Select/Union/Intersect/Except, 쓰기·DDL·SET·COPY·트랜잭션 노드 금지.
- table은 그 등록의 schema에 있는 catalog table이거나 자기 CTE만. schema 생략은 그 등록 schema로 본다. `pg_catalog`, `information_schema`, 다른 schema 금지.
- 위험 함수 금지: 이름이 `pg_`·`lo_`·`dblink`로 시작하는 함수, `current_setting`, `set_config`, `query_to_xml` 계열, `txid_`·`nextval`·`setval`. 이름 비교는 대소문자 무시.

**실행**
- `PostgresReadOnlyExecutor`(READ ONLY 트랜잭션, statement timeout)를 그 등록의 연결 설정으로 쓴다. 행 상한은 챗봇과 같은 200.

**답 모양**
- 기존 채점기가 그대로 쓰도록 챗봇 응답과 같은 dict를 만든다: `status`(`SUCCEEDED` / 거절은 `UNSUPPORTED` / 실패는 `FAILED`), `rows`(`course_code`, 있으면 `section`), `failure_code`(직접 SQL 결과 코드).
- `course_code` column이 없으면 `FAILED` + 고정 코드.

**모델**
- `general`: `CHAT_MODEL` 또는 `qwen2.5-coder:7b-instruct`. 기존 loopback Ollama 전송과 `GENERATION_OPTIONS`를 재사용한다(`direct_sql_ollama.py` 참고). 모델 이름 결정은 `api/chat_models.py`와 같은 규칙.
- `openai`: `OPENAI_MODEL` 또는 `gpt-5-mini`, `backend/.env`의 `OPENAI_API_KEY`. `harness/openai_chat.py`의 고정 HTTPS 주소·proxy/redirect 없음·응답 상한·`finish_reason == "stop"`·고정 실패 코드를 재사용한다. `response_format`은 쓰지 않고 SQL 글자를 받는다. 키는 header에만.
- 모델 id와 이름은 기록한다.

### 2. 실행 CLI

`python -m backend.adaptive.evaluation.direct_sql_pg run --set <legacy|other_school|blind:<eval_id>> --database <database_id> --model <general|openai> [--out <dir>]`

- `legacy`: `legacy_chat_compare`의 질문 세트·기준 SQL 키·`score_case`로 채점한다(QueryIR 쪽과 같은 채점).
- `other_school`: `fixtures/other_school.py`의 `QUESTIONS`와 `grade`.
- `blind:<eval_id>`: `blind_eval`의 `load_questions`와 `grade`. 이 질문 파일은 팀원 DB에도 쓰이므로 `blind_eval`의 잠금 규칙(첫 실행 hash 고정, 덮어쓰지 않음)을 따른다. 직접 SQL 실행은 `blind_eval` run 기록과 같은 디렉터리에 같은 번호 체계로 남기고 `arm: "direct-sql"`을 적는다. 기존 기록은 `arm`이 없으면 QueryIR로 읽는다. `report`는 arm별로 나눠 보인다.
- 질문마다 `elapsed_ms`(모델 호출 + 실행)를 기록한다.
- 요약: 맞음, 조용한 오답(`SUCCEEDED`인데 틀림), 생성 실패(빈 출력·파싱), 정책 거절, 실행 실패, 거절(UNKNOWN)이 맞음/틀림, 상한 도달, 지연 중앙값·p90.
- 기록은 Git이 무시하는 `.schema-adaptive/` 아래에 둔다.

### 3. QueryIR 쪽 지연 기록

- `blind_eval run`과 `legacy_chat_compare`의 live 실행 기록에 질문별 `elapsed_ms`를 더한다(HTTP 왕복). 다른 필드와 채점은 바꾸지 않는다.
- `other_school` 12문항을 QueryIR 쪽으로 돌리는 기존 경로를 찾아(9-08 §3.3) 같은 요약 지표가 나오는지 확인한다. 경로가 없으면 위 CLI에 `--arm queryir`(챗봇 HTTP API 사용, `--base-url` 필요)를 더해 같은 채점기로 돌린다.

## 확인 (테스트)

- 가짜 모델·가짜 reader·가짜 executor로:
  - context가 결정적이고, 예시 값 10개·40자 제한, 값 사전 미사용, schema-qualified 이름.
  - prompt 분할 치환, template sha256 고정.
  - 검사: 쓰기·DDL·여러 문장·다른 schema·`pg_catalog`·위험 함수 거절, CTE·schema 생략 허용.
  - 답 dict가 세 채점기에서 그대로 채점됨(맞음, 조용한 오답, UNKNOWN 거절, `course_code` 없음).
  - OpenAI 전송: 주소·header·모델·`response_format` 없음, 키가 기록·repr·오류에 없음, `finish_reason` 확인.
  - 기록에 SQL·행·prompt·키가 없음.
  - `blind_eval` 잠금·번호·`arm` 읽기 호환.
  - runtime 모듈이 `direct_sql_pg`를 import하지 않음.
- 새 테스트와 관련 기존 테스트(`test_direct_sql_arm`, `test_arm_live_cli`, `blind_eval`, `legacy_chat_compare`, `openai_chat` 관련)만 실행한다. 전체 pytest는 돌리지 않는다.

## 문서

`AGENTS.md` 규칙대로:
- `docs/schema-adaptive-text-to-sql-plan.md`:
  - ADR-100 추가(위 사용자 결정 1~4, 평가 전용, 보내는 데이터 범위, 값 사전 미사용 이유: 직접 SQL 쪽은 "등록만" 비용을 재기 위함).
  - 15절(평가 설계)에 9-10 비교 조건과 지표, 17절에 OpenAI 평가 전용 전송 범위.
  - 로드맵에 9-10 항목, 변경 이력에 2026-10-12 항목. live 측정은 "사용자 승인 뒤 한 번" 대기로 적는다.
- `docs/development-history.md#next-session-handoff` 현재 기준 문단에 9-10 한 줄.
- `docs/human-validation-checkpoints.md`에 9-10 메모: 사용자 결정이며, 개발 질문 56개는 QueryIR 규칙을 맞춘 질문이라 A 쪽에 유리하고, 판정 근거는 팀원 DB 첫 실행이다. 어떤 관문도 승인하지 않는다.

## 끝나면

- 바꾼 파일 목록, 실행한 테스트와 결과, 남은 한계를 짧게 적는다.

---

<a id="prompts-phase9-10b-review-fixes"></a>

## 원본: prompts/phase9-10b-review-fixes.md

# Phase 9-10 보완 지침: 검토 지적 고침과 OpenAI 제외

9-10 구현(`backend/adaptive/evaluation/direct_sql_pg.py` 등, 미커밋)에 대한 Claude 검토 결과다. 아래만 고친다. 다른 동작은 바꾸지 않는다. 9-10 지침(`docs/development-history.md#prompts-phase9-10-direct-sql-postgres`)의 "지킬 것"은 그대로다(live 모델 호출 없음, 사용자 서버 안 건드림, 커밋 없음, `.env` 값 출력 없음, 전체 pytest 안 돌림).

## 1. OpenAI 직접 SQL 경로 제거 (2026-10-12 사용자 결정)

- 사용자는 ChatGPT Plus만 쓰고 API 추가 비용을 원하지 않는다. OpenAI를 이용한 측정은 모두 제외한다. 비교는 로컬 `general` 모델로만 한다.
- `OpenAIDirectSqlModel`, `make_model`의 `openai` 분기, 관련 테스트를 지운다.
- CLI `--model`: `direct-sql` arm은 `general`만, `queryir` arm은 `general`과 `legacy-tuned`만 받는다. `openai`는 거절한다(테스트).
- 제품 챗봇의 9-09 `openai` 선택지(ADR-099)는 그대로 둔다.
- 문서:
  - ADR-100의 사용자 결정 3을 "같은 날 사용자가 철회: 추가 비용 없음(ChatGPT Plus는 API 사용량을 포함하지 않음). OpenAI 측정 제외, 직접 SQL OpenAI 경로 제거"로 고친다. 결정·이유 문장에서 OpenAI 전송 내용을 지운다.
  - 17절의 9-10 OpenAI 전송 예외 항목을 지우거나 "철회"로 바꾼다.
  - 15.1.1의 CLI 예시와 `human-validation-checkpoints.md` 9-10 메모, `next-session-handoff.md` 한 줄에서 OpenAI를 지우고 철회를 적는다.

## 2. 행 상한 공정성 (확인된 결함)

- 현재 직접 SQL 결과가 200행을 넘으면 executor가 `RESULT_TOO_LARGE`를 내고 `FAILED`가 된다. 챗봇(QueryIR)은 200행까지 보여 주고 `row_count`·`limit`으로 상한 도달을 알린다. 레거시 채점은 상한 도달을 `CORRECT_SUBSET`으로 인정하고, `blind_eval`은 `CAPPED`로 센다. 그래서 큰 결과 질문에서 직접 SQL만 불리하다.
- 고침: 정책 검사를 통과한 SQL을 실행할 때 `SELECT * FROM (<sql>) AS direct_sql_result LIMIT 200`으로 감싸 최대 200행을 받는다. 200행이면 `SUCCEEDED`, `row_count` 200, `limit` 200(챗봇과 같은 모양). 감싸기는 검사 뒤에만 하고, 검사는 원래 SQL로 한다. `sql_sha256`은 원래 SQL의 것을 유지한다.
- 회귀 테스트: 가짜 executor가 감싼 SQL을 받는지; 200행 결과가 레거시 채점에서 기준의 부분 집합이면 `CORRECT_SUBSET`(인정)으로, `blind_eval` 채점에서 `CAPPED`로 되는지; 199행은 그대로 채점되는지.

## 3. 따옴표 없는 이름의 대소문자 (확인된 결함)

- `check_sql`은 table·schema 이름을 대소문자 그대로 비교한다. PostgreSQL은 따옴표 없는 이름을 소문자로 접으므로, 모델이 `FROM Courses`처럼 쓰면 PostgreSQL에서는 맞는 SQL인데 `SQL_UNKNOWN_TABLE`로 거절된다.
- 고침: identifier가 따옴표 없으면 소문자로 접은 뒤 비교하고, 따옴표가 있으면 그대로 비교한다(schema·table·CTE 모두).
- 회귀 테스트: 따옴표 없는 대문자 table·schema 허용, 따옴표 있는 다른 대소문자는 거절.

## 확인

- `tests/adaptive/test_direct_sql_pg.py`와 9-10에서 돌린 관련 테스트 파일만 실행한다.
- 끝나면 바꾼 파일, 테스트 결과를 짧게 적는다.

---

<a id="prompts-phase9-10c-luna-fixes"></a>

## 원본: prompts/phase9-10c-luna-fixes.md

# Phase 9-10 보완 지침 2: Luna 리뷰 확인 지적 고침

Luna(`gpt-6-luna` high) 리뷰와 Claude 검토에서 나와 Claude가 재현한 4건만 고친다. 9-10 지침의 "지킬 것"은 그대로다(live 모델 호출 없음, 사용자 서버 안 건드림, 커밋 없음, `.env` 값 출력 없음, 전체 pytest 안 돌림). 각 고침에 회귀 테스트를 둔다.

## 1. 함수 정책 구멍 (High, 재현됨)

- `direct_sql_pg.check_sql`은 위험 함수 이름만 막는다. `SELECT public.leak_code() AS course_code FROM courses`(sqlglot: `Dot` + `Anonymous:leak_code`)와 `SELECT leak_code() ...`가 통과한다. 사용자 정의 함수는 다른 schema를 읽거나 부작용을 낼 수 있다.
- 고침:
  - schema로 한정한 함수 호출(`exp.Dot` 안의 함수)은 모두 거절한다.
  - `exp.Anonymous`(sqlglot이 모르는 함수)는 코드에 고정한 PostgreSQL 내장 함수 허용 목록에 있을 때만 허용한다. 목록은 글자·날짜·숫자·배열 처리에 흔한 내장 함수로 좁게 잡는다(예: `split_part`, `string_to_array`, `regexp_split_to_array`, `array_to_string`, `unnest`, `btrim`, `initcap`, `to_char`, `date_part`). sqlglot이 아는 함수(`Count`, `Lower` 등 typed 노드)는 기존 금지 이름 검사만 한다.
  - 거절 코드는 기존 `SQL_FORBIDDEN_NODE`.
- 테스트: 위 두 SQL 거절, 허용 목록 함수와 `count`/`lower` 허용.

## 2. 200행 감싸기와 끝 주석 (재현됨)

- `SELECT code AS course_code FROM courses -- c`를 `SELECT * FROM (<sql>) AS direct_sql_result LIMIT 200`으로 감싸면 닫는 괄호가 주석에 들어가 실행 오류가 된다.
- 고침: `"SELECT * FROM (\n" + sql + "\n) AS direct_sql_result LIMIT 200"`처럼 원래 SQL 앞뒤에 줄바꿈을 둔다.
- 테스트: 끝에 `--` 주석이 있는 SQL을 감싼 결과에서 닫는 괄호가 주석 밖에 있음.

## 3. 거절 채점의 arm 간 차이 (Medium, 재현됨)

- `direct_sql_pg` CLI의 legacy 채점은 `status == "UNSUPPORTED"`만 거절로 본다. 직접 SQL의 `UNKNOWN`은 `UNSUPPORTED`가 되어 `REFUSED`(인정)인데, QueryIR의 `NOT_UNDERSTOOD`는 `NO_ANSWER`가 된다. 직접 SQL은 두 경우를 구분할 수 없으므로 같은 질문에서 직접 SQL만 유리하다.
- 고침: 이 CLI의 legacy 채점에서 두 arm 모두 `blind_eval.DECLINE_STATUSES`(`UNSUPPORTED`, `NOT_UNDERSTOOD`)를 거절로 본다. `blind_eval`과 같은 규칙이다. `legacy_chat_compare` 자체 채점(9-08 수치의 기준)은 바꾸지 않는다.
- 문서: 15.1.1에 "이 CLI의 legacy 거절 규칙은 `blind_eval`과 같고, 9-08 `legacy_chat_compare` 수치와 직접 비교하지 않는다"를 한 줄 적는다.
- 테스트: 거절 대상 질문(scope `UNSUPPORTED_FIELD`)에서 QueryIR `NOT_UNDERSTOOD`와 직접 SQL `UNKNOWN`이 같은 결과.

## 4. blind 지연 측정 위치 (Low)

- `blind_eval.run`이 `elapsed_ms`를 채점 뒤에 잰다. `ask` 반환 직후(채점 전)에 재도록 옮긴다. 답에 `elapsed_ms`가 이미 있으면 그 값을 쓴다(직접 SQL과 같은 기준).
- 테스트: 채점 시간이 지연에 들어가지 않음(가짜 시계나 순서 확인).

## 확인

- `tests/adaptive/test_direct_sql_pg.py`, `test_blind_eval`, `test_legacy_chat_compare`만 실행한다.
- 변경 이력의 9-10b 항목 아래에 "9-10c Luna 지적 4건 고침"을 한 줄 더한다(로드맵의 Luna 리뷰 항목은 완료로).
- 끝나면 바꾼 파일과 테스트 결과를 짧게 적는다.

---

<a id="prompts-phase9-11-published-review-contract"></a>

## 원본: prompts/phase9-11-published-review-contract.md

# Phase 9-11 지침: 게시된 PostgreSQL DB의 "mapping 검토 시작" 오류 고침

## 증상 (사용자 보고, Claude 재현)

- 게시된 PostgreSQL 등록(`rehearsal-school`, mapping v2)에서 관리자 화면의 "mapping 검토 시작"을 누르면 "응답 형식이 올바르지 않습니다. 다시 시도해 주세요."가 나온다.
- 재현: 사용자 서버의 읽기 전용 `GET /api/adaptive/v1/registrations/review?database_id=rehearsal-school` 응답을 `src/adaptive-prototype/mappingReview.ts`의 `parseReviewDetail`에 넣으면 `null`이다. INIT 응답도 같은 형식이라 같은 이유로 실패한다.

## 원인 (확인됨)

`backend/adaptive/api/postgres_registration.py`의 review 상세(약 222~229행)가 만드는 `mapping_versions` 항목이 화면 계약과 다르다. 두 가지를 모두 고쳐야 통과한다(하나만 고치면 여전히 `null`).

1. `published_at`이 마이크로초를 포함한다(`2026-10-12T12:39:49.389950Z`). 화면은 `UTC_TIMESTAMP_PATTERN = /^...T\d\d:\d\d:\d\dZ$/`(초까지)만 받는다. 저장 형식은 `registration/postgres_registrations.py` 약 927행 `isoformat()`이다.
2. 현재 버전에 `activatable: False`를 보낸다. 화면은 `active && !activatable`을 잘못된 이력으로 본다(`parseVersionHistory`). SQLite 경로(`registration/reviews.py`)는 활성 버전을 `activatable: True`로 보낸다.

## 고칠 것

- 서버 응답을 기존 화면 계약에 맞춘다. 화면 parser는 바꾸지 않는다.
  - review 상세의 `published_at`은 초 단위 UTC `YYYY-MM-DDTHH:MM:SSZ`로 보낸다. 이미 저장된 `published.json`(마이크로초 포함)도 그대로 읽혀야 하므로 응답을 만들 때 자른다. 저장 형식은 바꾸지 않는다(기존 파일과 다른 읽는 곳 보존).
  - 현재 버전 항목은 `activatable: True`로 보낸다. PostgreSQL 경로에 이전 버전 재활성화(ACTIVATE)가 없다면 그 동작은 그대로 두고, 화면에 재활성화 버튼이 새로 생기지 않는지 확인한다(활성 버전에는 버튼이 없어야 한다).
- 다른 응답(게시 결과, 등록, 챗봇)은 바꾸지 않는다.

## 테스트

- 서버: 게시된 PostgreSQL 등록의 review GET과 INIT 응답에서 `published_at`이 초 단위 `Z` 형식이고 활성 버전이 `activatable: True`인지(마이크로초가 있는 저장 파일로).
- 계약: 그 서버 응답 모양을 `parseReviewDetail`이 받아들이는지(기존 frontend 테스트 방식대로 `tests/frontend/`에 fixture 하나).
- 관련 기존 테스트 파일만 실행한다. 전체 pytest는 돌리지 않는다.

## 지킬 것

- 사용자 서버(8000·8123·5173)와 사용자 데이터(`.schema-adaptive/postgres-registrations/rehearsal-school` 등)를 건드리지 않는다. 확인이 필요하면 테스트용 임시 디렉터리만 쓴다.
- live 모델 호출 없음, 커밋 없음, `backend/.env` 값 출력 없음.
- 문서: `docs/schema-adaptive-text-to-sql-plan.md` 변경 이력에 2026-10-12 Phase 9-11 한 줄(증상, 원인 두 가지, 서버 쪽 고침), 로드맵에 9-11 항목.
- 끝나면 바꾼 파일과 테스트 결과를 짧게 적는다.

---

<a id="prompts-phase9-11b-activatable"></a>

## 원본: prompts/phase9-11b-activatable.md

# Phase 9-11 보완: 활성 버전의 `activatable` 처리 방향 바꾸기

9-11 첫 고침(미커밋)에 Claude 검토 지적 1건.

## 문제 (확인됨)

- 9-11은 PostgreSQL review 상세의 활성 버전을 `activatable: True`로 보냈다.
- 화면 `MappingReviewPanel.tsx`(약 215~224행)는 `activatable`이면 "회귀 확인" 버튼을 보인다. 이 버튼이 부르는 `/api/adaptive/v1/registrations/regression`(`api/admin_test_queries.py` `run_regression`)은 SQLite review 저장소(`review_storage.load_published_version`)만 읽어 PostgreSQL 등록에서는 오류가 난다. 새로 누를 수 있는 버튼이 오류를 내게 된다.

## 고칠 것

- 서버는 원래대로 PostgreSQL 활성 버전에 `activatable: False`를 보낸다(재활성화·회귀 확인을 지원하지 않는다는 뜻 그대로). `published_at` 초 단위 고침은 유지한다.
- 화면 계약을 넓힌다: `parseVersionHistory`의 `if (parsed.active && !parsed.activatable) return null;`을 없앤다. 활성 버전이 `activatable: false`인 것을 허용한다. 다른 검사는 그대로 둔다.
- 화면 표시: `!activatable`일 때 붙는 " · 현재 스키마에서 실행 불가"는 활성이 아닌 버전에만 붙인다. 활성 버전이면 "실행 중"만 보이고 "회귀 확인"·"이 버전으로 되돌리기" 버튼은 보이지 않는다(기존 조건 그대로).
- SQLite 경로 응답과 동작은 바꾸지 않는다.

## 테스트

- 9-11 테스트를 위 방향에 맞게 고친다: 서버 응답은 활성 버전 `activatable: False`·초 단위 `published_at`; frontend fixture가 `parseReviewDetail`을 통과; 활성·`activatable: false` 항목에 버튼과 "실행 불가" 문구가 없음(기존 frontend 테스트 방식으로 가능한 범위).
- SQLite 쪽 기존 이력 테스트가 그대로 통과하는지.
- 관련 테스트 파일만 실행한다.

## 정리

- 저장소 루트의 `.tmp-phase9-11-tests`(첫 실행이 남긴 테스트 임시 폴더)는 Claude가 지운다. 이번 실행은 임시 경로를 저장소 밖이나 기존 git-ignored 위치에 둔다.
- 변경 이력 9-11 줄을 고친 방향으로 바꾼다.
- 사용자 서버·데이터 안 건드림, live 모델 호출 없음, 커밋 없음. 끝나면 바꾼 파일과 테스트 결과를 짧게 적는다.

---

<a id="prompts-phase9-12-db-admin-layout"></a>

## 원본: prompts/phase9-12-db-admin-layout.md

# Phase 9-12 지침: "DB 변경" 관리 화면을 보기 쉽게 다시 배치

## 배경 (2026-10-12 사용자 요청)

- 메인 화면 "DB 변경" 버튼이 여는 창(`src/pages/MainPage.tsx` 약 613행, `max-w-sm` 세로로 긴 dialog)에 DB 고르기, SQLite 등록, PostgreSQL 등록(`PostgresRegistrationPanel`), mapping 검토(`MappingReviewPanel`)와 그 안의 값 사전(`ValueDictionaryPanel`, `CodeTableImport`), 확장 항목(`ExtensionFieldPanel`), 시험 질의(`TestQueryPanel`), 버전 이력이 한 줄로 쌓여 있다. 챗봇처럼 좁고 길어서 무엇이 어디 있는지 알아보기 어렵다.
- 사용자 결정: 지금은 **가시성**이 목표다. 기존 스타일과 맞추는 것은 다음 단계에서 한다. 기능·API·검증 규칙은 바꾸지 않는다(배치·제목·안내 문구만).

## 바꿀 것

1. **창 크기와 구조**
   - dialog를 넓고 높게: 화면의 약 90%(예: `max-w-6xl w-[95vw] h-[90vh]`), 안쪽만 스크롤.
   - 위쪽에 탭(또는 왼쪽 메뉴) 셋: **사용할 DB** / **DB 등록** / **검토·게시**. 지금 어느 탭인지 분명히 보인다.
   - 화면 폭이 좁으면(약 768px 미만) 세로로 쌓인다.
2. **사용할 DB 탭**: 지금의 DB 목록 라디오, 취소·적용. 각 DB에 상태(게시됨 vN)를 배지로.
3. **DB 등록 탭**
   - 등록 양식(연결, Schema, Database ID, Label, 등록)을 한 줄 또는 2열로.
   - "등록된 PostgreSQL DB"를 표로: ID | 이름 | 연결/Schema | 상태 배지(검토 대기 / 게시됨 vN) | "검토 열기" 버튼. "검토 열기"는 검토·게시 탭으로 그 DB를 연다.
   - SQLite 등록은 범위 밖(ADR-089)이므로 이 탭 맨 아래 접힌 영역("기타: SQLite 파일 등록")으로 옮긴다. 기능은 그대로 둔다.
4. **검토·게시 탭** (DB 하나를 연 상태)
   - 맨 위: DB 이름·상태·현재 버전, 그리고 단계 표시 `① 항목 연결 확인 → ② 값 사전(선택) → ③ 확장 항목(선택) → ④ 게시`.
   - 본문 2열:
     - 왼쪽(넓게): **항목 연결 표**. 열: 공통 항목(한국어 이름과 key) | 연결된 칸(`table.column` 또는 "이 DB에 없음") | 상태 배지(확정/대기/없음) | 동작(후보 승인, 다른 칸으로 변경, 이 DB에 없음). 지금 `MappingReviewPanel`의 항목별 UI를 표 행으로 옮긴다.
     - 오른쪽: 값 사전, 확장 항목, 시험 질의, 버전 이력을 접고 펼 수 있는 상자(또는 하위 탭)로. 각 상자 제목 아래 한 줄 설명(무엇을 하는 곳인지, 꼭 해야 하는지).
   - 게시 버튼과 게시 막힘 이유 목록(9-07 `reasons`)은 아래쪽에 크게, 막힌 항목마다 한 줄.
5. **문구**: 처음 쓰는 사람이 읽을 수 있게 짧은 한국어로. 내부 용어는 괄호로만(예: "칸(column)", "공통 항목(field)"). 기존 오류·안내 문구의 뜻은 바꾸지 않는다.

## 바꾸지 않을 것

- 서버 API, 요청·응답 parser(`mappingReview.ts` 등 `*.ts` 계약), 검증·게시 규칙, 챗봇 화면, 레거시 기준선, adaptive-prototype 페이지.
- 기존 frontend 테스트가 확인하는 동작. 문구를 바꿔 테스트가 깨지면 테스트의 의도(동작)를 유지하는 범위에서만 기대 문구를 고친다.

## 확인

- `npx tsc -b`, `npm run lint`, `npx vite build`, 그리고 `tests/frontend/`의 관련 테스트(`phase6-19-mapping-review`, `phase8-24-dictionary-import`, `phase9-07-publish-block-reasons`, `phase9-11-published-review-contract` 등)를 `node --test`로.
- 서버를 띄우지 않는다(사용자 서버 8000·8123·5173도 건드리지 않는다). 화면 확인은 Claude가 따로 한다.
- 커밋하지 않는다. `backend/.env` 값 출력 없음.
- 문서: `docs/schema-adaptive-text-to-sql-plan.md` 로드맵에 9-12, 변경 이력에 2026-10-12 한 줄(사용자 요청, 배치만 바꿈, 스타일 맞추기는 다음 단계).
- 끝나면 바꾼 파일, 실행한 확인과 결과, 새 배치를 5줄 이내로 요약한다.

---

<a id="prompts-phase9-13-postgres-test-queries"></a>

## 원본: prompts/phase9-13-postgres-test-queries.md

# Phase 9-13 지침: PostgreSQL 등록 DB에서도 시험 질의·기준 저장·회귀 확인이 되게 하기

## 배경 (2026-10-13 사용자 결정: B안)

- 관리자 화면 "검토·게시"의 **시험 질의**(test query 미리보기·실행), **기준 결과 저장**(verified queries), **회귀 확인**(regression)은 SQLite 등록 저장소만 읽는다(`backend/adaptive/api/admin_test_queries.py`: `run_test_query` → `review_storage.load_activation`, `verify_test_query`, `run_regression` → `review_storage.load_published_version`).
- PostgreSQL 등록(`rehearsal-school`, mapping-v3 게시됨)에서 시험 질의를 누르면 "게시·활성화된 DB에서만 test query를 실행할 수 있습니다", 그리고 "회귀 확인 실패: ..."가 나온다(사용자 스크린샷). 게시는 정상이다.
- 사용자 결정: 숨기지 말고 PostgreSQL에서도 동작하게 한다.

## 만들 것

1. **시험 질의 실행 (PostgreSQL)**
   - `TEST_QUERIES`(offerings, courses, departments, schedules, seats)를 PostgreSQL 등록의 **게시된 현재 버전**으로 실행한다.
   - 챗봇과 같은 PostgreSQL 실행 경로를 재사용한다: 게시된 runtime profile(`.schema-adaptive/postgres-profiles/<id>/`)과 `api/postgres_runtime.py`의 검증된 context(`ValidatedPostgresContext`, 변환·값 사전·anchor·"이 DB에 없음" 항목 포함), PostgreSQL compiler·`SchemaAdaptivePostgresPolicy`·`PostgresReadOnlyExecutor`. 새 SQL 생성 경로를 만들지 않는다. 모델 호출 없음(고정 QueryIR).
   - "이 DB에 없음"인 항목이 select에 있으면, 챗봇의 규칙과 같게 그 항목을 빼고 실행하고 결과에 "빠진 항목" 목록을 알린다. 남는 항목이 없으면 고정 코드로 "이 DB에서는 실행할 수 없음"을 돌려준다(오류 500이 아니라 설명 가능한 결과).
   - 응답 모양은 SQLite 경로와 같게 한다(화면 parser 그대로 통과). 필요한 필드가 하나 더 있으면(빠진 항목) 화면 parser와 표시를 함께 넓힌다.
   - 행 미리보기·상한은 기존 `MAX_PREVIEW_ROWS`·`MAX_RESULT_ROWS` 그대로.
2. **기준 결과 저장과 회귀 확인 (PostgreSQL)**
   - 기준 저장: 현재 버전의 시험 질의 결과를 서버가 계산해 기준으로 저장한다. 저장 위치는 그 등록 디렉터리(`.schema-adaptive/postgres-registrations/<id>/` 아래 새 파일). 형식과 검증은 SQLite 기준 저장과 같은 규칙(검토자, 결과 fingerprint, 원본 행 값을 남기지 않는 기존 원칙이 있으면 그대로).
   - 회귀 확인: PostgreSQL은 이전 버전 파일을 따로 보관하지 않으므로 **현재 버전**에 대해서만 기준과 비교한다. 다른 버전 번호를 요청하면 고정 코드로 거절한다.
   - 이에 맞춰 review 상세의 PostgreSQL 활성 버전 `activatable`을 `True`로 되돌린다(9-11b에서 False로 둔 이유였던 "회귀 확인 오류"가 사라지므로). 활성 버전에는 "되돌리기" 버튼이 원래 안 보이므로 재활성화 기능은 새로 생기지 않는다. 9-11b의 화면 계약 완화(활성·`activatable: false` 허용)는 그대로 둔다.
3. **라우팅**: 같은 HTTP 경로(`/registrations/test-queries`, `/test-query`, `/verified-queries`, `/regression`)에서 `database_id`가 PostgreSQL 등록이면 PostgreSQL 경로로, 아니면 기존 SQLite 경로로 보낸다(`PostgresRegistrationService.handles` 같은 기존 판별 재사용). SQLite 동작은 바꾸지 않는다.
4. **화면**: 서버 응답이 같으면 바꿀 것이 없다. 빠진 항목 표시가 필요하면 시험 질의 결과 위에 "이 DB에 없어 뺀 항목: ..." 한 줄을 보인다. 자동으로 회귀 확인을 부르는 곳이 있으면 PostgreSQL에서도 정상 동작하는지 확인한다.

## 테스트

- 가짜 PostgreSQL executor·connector와 임시 디렉터리의 게시된 profile로:
  - 시험 질의 5종이 게시 mapping으로 컴파일·실행되고 SQLite와 같은 응답 모양.
  - "이 DB에 없음" 항목이 빠지고 알려짐; 모두 빠지면 고정 코드.
  - 기준 저장 → 회귀 확인 일치/불일치; 다른 버전 번호 거절.
  - review 상세 활성 버전 `activatable: True`, `published_at` 초 단위 유지.
  - SQLite 경로 기존 테스트 그대로 통과.
- frontend: 바뀐 parser·표시가 있으면 해당 테스트.
- 관련 테스트 파일만 실행한다. 전체 pytest는 돌리지 않는다.

## 지킬 것

- 사용자 서버(8000·8123·5173)와 사용자 데이터(`.schema-adaptive/postgres-registrations/rehearsal-school` 등)를 건드리지 않는다. 테스트는 임시 디렉터리만.
- legacy DB 읽기 전용, live 모델 호출 없음, 커밋 없음, `backend/.env` 값 출력 없음, DSN 노출 없음.
- 문서: `docs/schema-adaptive-text-to-sql-plan.md` 로드맵 9-13, 변경 이력 2026-10-13 한 줄(사용자 결정 B, 현재 버전만 회귀 확인).
- 끝나면 바꾼 파일과 테스트 결과를 짧게 적는다.

---

<a id="prompts-phase9-14-review-row-cleanup"></a>

## 원본: prompts/phase9-14-review-row-cleanup.md

# Phase 9-14 지침: 검토·게시 탭 정리 (항목 연결 표 단순화)

## 배경 (2026-10-13 사용자 요청·스크린샷)

9-12에서 "DB 변경" 창을 탭 구조로 바꿨다. 사용자가 써 보니 "검토·게시" 탭의 항목 연결 표가 여전히 직관적이지 않다. 확인된 문제:

1. 한 줄에 조작이 전부 펼쳐져 있다: AI 후보 정보, 후보 선택+후보 승인, 테이블·칸 선택, 사유, "다른 칸으로 변경", "이 DB에 없음". 확정된 줄에도 그대로 보인다.
2. **실수 위험(결함)**: 테이블·칸 드롭다운의 처음 값이 현재 연결된 칸이 아니라 목록의 첫 값(`dept`/`college`)이다. "수강 인원 = lecture.enrolled"로 확정된 줄에서 그대로 "다른 칸으로 변경"을 누르면 `dept.college`로 바뀐다.
3. 후보 줄의 "검토 필요"는 AI 후보 자체의 표시인데, 줄 상태가 "확정"이어도 보여서 헷갈린다.
4. 검토자 입력이 패널마다 따로 있고, 검토자 아래에 "DB 이름 · 검토 대기"가 한 번 더 나온다.
5. 오른쪽 상자(값 사전, 확장 항목, 버전 이력, 시험 질의)의 설명이 "무엇을 언제 하는지" 드러나지 않는다.

기능·API·검증 규칙은 바꾸지 않는다. 화면 배치·문구·기본값만 바꾼다. 스타일 통일은 아직 하지 않는다.

## 바꿀 것

1. **항목 연결 표의 한 줄** (`MappingReviewPanel.tsx`)
   - 기본 표시: 공통 항목(한국어 이름 + key) | 연결된 칸(또는 "이 DB에 없음") | 상태 배지 | 버튼.
   - AI 후보는 "AI 추천: `table.column` (점수 0.2)" 한 줄로. 후보가 여럿이면 드롭다운은 승인할 때만. 후보 없음은 "AI 추천 없음".
   - 버튼:
     - 대기 줄: [추천대로 승인] [다른 칸으로] [이 DB에 없음].
     - 확정/없음 줄: [다시 정하기] 하나. 누르면 위 세 버튼이 나온다.
   - [다른 칸으로]를 누를 때만 테이블·칸 선택과 사유 입력, [저장] [취소]가 펼쳐진다. **테이블·칸의 처음 값은 현재 연결된 칸**(없으면 AI 추천, 그것도 없으면 빈 선택 "선택하세요" — 빈 선택이면 저장 불가).
   - [이 DB에 없음]을 누를 때만 사유 입력과 [확인] [취소]가 펼쳐진다.
   - 사유가 필요한 규칙은 그대로(빈 사유면 저장 불가). 사유 입력란 옆에 "왜 바꾸는지 짧게" 같은 안내.
   - 펼친 편집은 한 번에 한 줄만.
2. **검토자**: "검토·게시" 탭 맨 위에서 한 번만 입력하고, 값 사전·확장 항목·시험 질의 기준 저장·게시에 같은 값을 쓴다. 비어 있으면 버튼들이 비활성인 이유를 "먼저 검토자 이름을 넣어 주세요"로 보인다. 중복된 "DB 이름 · 상태" 줄을 없앤다.
3. **오른쪽 상자 설명** (제목 아래 한 줄, 쉬운 말):
   - 값 사전: "DB에 코드로 저장된 값의 뜻과, 사람들이 쓰는 줄임말을 적습니다. 예: GE = 교양, 소웨 = 소프트웨어학부. 적은 뒤 ‘사전 적용해 게시’를 눌러야 챗봇에 반영됩니다."
   - 확장 항목: "공통 항목에 없는 이 DB만의 정보(예: 강의실)를 챗봇이 알 수 있게 추가합니다. 추가 후 게시해야 반영됩니다."
   - 버전 이력: "게시한 버전과 지금 챗봇이 쓰는 버전입니다. PostgreSQL DB는 현재 버전만 보관합니다."
   - 시험 질의: "게시한 연결로 정해진 조회를 실행해 결과가 맞는지 봅니다. 맞으면 ‘기준으로 저장’ 해 두고, 나중에 다시 게시했을 때 ‘회귀 확인’으로 결과가 바뀌었는지 비교합니다."
4. **게시 영역**: 게시 버튼 옆에 "게시하면 챗봇이 이 연결을 씁니다(새 버전)." 한 줄. 막힌 이유 목록(9-07)은 그대로.

## 테스트

- frontend: 기존 `tests/frontend/` 관련 테스트 통과(동작 의도가 같으면 기대 문구만 고친다). 새로: 다른 칸 편집의 처음 값이 현재 연결 칸인지, 빈 선택·빈 사유면 저장 불가인지(기존 테스트 방식으로 가능한 범위, 순수 함수로 뺄 수 있으면 빼서 테스트).
- `npx tsc -b`, 관련 파일 eslint, `npx vite build`.
- 서버를 띄우지 않는다. 화면 확인은 Claude가 한다.

## 지킬 것

- 사용자 서버·데이터 안 건드림, live 모델 호출 없음, 커밋 없음, `backend/.env` 값 출력 없음.
- 문서: `docs/schema-adaptive-text-to-sql-plan.md` 로드맵 9-14, 변경 이력 2026-10-13 한 줄(드롭다운 처음 값 결함 포함).
- 끝나면 바꾼 파일과 확인 결과를 짧게 적는다.

---

<a id="prompts-phase9-15-bulk-approve"></a>

## 원본: prompts/phase9-15-bulk-approve.md

# Phase 9-15 지침: 검토·게시 탭에 "AI 추천 모두 승인" 버튼

## 배경 (2026-10-13 사용자 요청)

- 레거시를 새로 등록해 검토하려는데, 항목마다 [추천대로 승인]을 누르는 것이 번거롭다. 일괄 승인 기능을 원한다.

## 만들 것 (화면만, 서버 API 변경 없음)

- 항목 연결 표 위에 버튼 **"AI 추천 모두 승인"**.
- 대상: 상태가 **대기(PENDING)** 이고 AI/규칙 후보가 하나 이상 있는 항목만. 각 항목의 첫 후보(지금 [추천대로 승인]이 쓰는 후보와 같은 것)를 기존 `approveReviewField`(action `APPROVE`)로 승인한다. 이미 확정·변경·없음인 항목과 후보가 없는 항목은 건드리지 않는다.
- 누르면 먼저 확인: "대기 중인 N개 항목을 AI 추천대로 승인합니다. 후보가 없는 M개는 그대로 둡니다. 승인 뒤에도 항목마다 [다시 정하기]로 고칠 수 있습니다." [승인] [취소].
- 실행은 한 항목씩 순서대로(동시에 보내지 않음). 진행 중 "k/N 승인 중", 버튼들 비활성. 한 항목이 실패하면 멈추고 "k개 승인, X 항목에서 실패: <기존 오류 문구>"를 보인다. 끝나면 "N개 승인 완료. 후보 없는 항목: …(이름 목록)"과 함께 표를 최신 상태로.
- 검토자가 비어 있으면 버튼 비활성(기존 안내 문구 그대로). 대상이 0개면 비활성.
- 게시는 자동으로 하지 않는다. 게시 규칙(9-06 숫자 칸, ADR-097 시간 칸)은 그대로이므로, 잘못된 추천은 게시할 때 막힌 이유로 보인다.

## 테스트

- 대상 고르기를 순수 함수로 빼서 테스트: 대기+후보 있음만 포함, 확정·없음·후보 없음 제외, 첫 후보 index 사용.
- 순차 실행: 가짜 fetch로 N번 순서대로 호출, 중간 실패 시 멈춤과 메시지.
- 기존 frontend 테스트 통과. `npx tsc -b`, 관련 파일 eslint.

## 지킬 것

- 서버·사용자 데이터 안 건드림, live 모델 호출 없음, 커밋 없음, `backend/.env` 값 출력 없음.
- 문서: `docs/schema-adaptive-text-to-sql-plan.md` 로드맵 9-15, 변경 이력 2026-10-13 한 줄(사용자 요청, 화면만, 게시 규칙은 그대로).
- 끝나면 바꾼 파일과 확인 결과를 짧게 적는다.

---

<a id="prompts-phase9-16-retire-offering-year"></a>

## 원본: prompts/phase9-16-retire-offering-year.md

# Phase 9-16 지침: 공통 항목 "개설 연도"(CourseOffering.year) 사용 중단 — 공통 ontology 버전 3

## 사용자 결정 (2026-10-13, 공통 ontology 변경 = 사용자 결정으로 기록)

- 수강 강의를 찾을 때 개설 연도로 검색하는 경우는 없다. 공통 항목에서 "개설 연도"를 뺀다(레거시만 보고 내린 결론이 아님).
- 방법은 **A. 사용 중단**: 새 ontology 버전에서 사용 중단으로 표시해 등록·검토·AI 추천·챗봇에서 사라지게 하고, 이미 게시된 mapping과 평가 정답 데이터는 그대로 읽히게 둔다. 항목 정의 삭제(B)는 하지 않는다.
- **학기(CourseOffering.semester)는 그대로 둔다.**
- 배경: 레거시 등록에서 AI가 개설 연도를 `cnu_courses.course_year`(실제 값 1~5, 학년)에 연결했다. 연도 질문을 학년 칸으로 비교하는 조용한 오답 위험이었다.

## 현재 구조 (확인됨)

- `backend/adaptive/ontology/course-domain.yaml` `version: 2`, 필드마다 `since_version`(Phase 8-18: 버전은 필드를 더하기만 함). `CourseOffering.year`는 `required: true`.
- `ontology/contracts.py`: `since_version`, `accepts_version`, `fields_at(version)`.
- `CourseOffering.year` 참조: `api/chat_query.py`, `api/legacy_view_profile.py`, `api/synthetic_registry.py`, 평가 데이터·fixture(`evaluation/data/*.yaml`, `fixture_mapping.py`, `reference.py`, `variant_*`), `dataset/profiles/*.yaml`, frontend `reviewFieldName.ts`, `MappingReviewPanel.tsx`, `testQueries.ts`, 그리고 여러 테스트.

## 만들 것

1. **ontology 계약**: 필드에 `retired_in_version: int | None`(기본 None)를 더한다. ontology `version: 3`으로 올리고 `CourseOffering.year`에 `retired_in_version: 3`. 
   - "버전 v에서 쓰는 필드" = `since_version <= v` 이고 (`retired_in_version`이 없거나 `> v`).
   - `accepts_version`은 1~3을 받는다. v1·v2로 만든 기록(게시 mapping, 평가 gold, fixture)은 그대로 유효하다.
   - 사용 중단 필드는 `required`로 게시를 막지 않는다.
2. **현재 버전(3) 기준으로 사라지게**:
   - 새 등록의 규칙·AI mapping 제안, 검토 표(`review.fields`), 게시 검사, 값 사전·확장 항목 대상 목록에서 빠진다.
   - 챗봇: 모델에 주는 공통 항목 목록·QueryIR JSON schema(`queryir-schema-v1` 열거)·값 힌트 대상에서 빠진다. 질문이 연도를 말하면(예: "2024년에 개설된 강의") 기존 "이 DB에 없음/정보 없음" 거절과 같은 경로로 답한다(조용히 무시하지 않는다). 모델이 그래도 year를 내면 QueryIR 검증에서 기존 "지원하지 않는 항목"과 같이 처리한다.
   - **이미 게시된 mapping이 year를 연결해 두었어도** runtime(현재 ontology 3)에서는 year를 "이 DB에 없음"으로 취급한다(챗봇·시험 질의·게시 재검사 공통). 저장된 파일은 바꾸지 않는다.
   - frontend: 검토 표·이름표(`reviewFieldName.ts`)에서 year가 서버 응답에 없으면 자연히 안 보이게. 하드코딩된 목록이 있으면 정리한다.
3. **바꾸지 않을 것**: 평가 gold·fixture·데이터셋 profile의 year 기록(v1/v2 기록으로 유효), legacy 기준선 챗봇, 학기 항목, 다른 필드.
   - 평가 도구가 현재 버전으로 year 질문을 채점할 때 기대 결과가 달라지면(예: 합성 benchmark에서 year 조건이 있는 문항), 그 문항 목록과 이유를 보고서에 적고 gold는 고치지 않는다. AI는 gold `review_status`를 바꾸지 않는다.
4. **레거시 기준 확인**: `legacy_chat_compare` 30문항 중 q28 "2024년에 개설된 강의 알려줘"는 원래 거절이 정답(`UNSUPPORTED_FIELD`)이다. 이 경로가 계속 거절인지 테스트로 확인한다(모델 호출 없이 가능한 범위: 가짜 모델 출력 또는 기존 unknown-condition 경로).

## 테스트

- ontology: v1·v2·v3의 `fields_at`/활성 필드, retired 필드가 v3에서 빠짐, v2 기록 유효.
- 등록·검토·게시: 새 PostgreSQL/SQLite 검토에 year 없음, year가 연결된 v2 게시 mapping을 읽어도 runtime에서 year는 unsupported, `required`로 게시가 막히지 않음.
- 챗봇: 모델 항목 목록·JSON schema에 year 없음, 연도 질문은 거절 경로.
- 기존 관련 테스트 중 year를 쓰는 것은 의도에 맞게 고친다(현재 버전 동작을 확인하던 테스트는 v3 동작으로, 기록 호환을 확인하던 테스트는 그대로).
- 관련 테스트 파일만 실행한다(전체 pytest는 Claude가 커밋 전에 한 번). frontend는 관련 테스트와 `npx tsc -b`.

## 지킬 것

- 사용자 서버(8000·8123·5173)와 사용자 데이터(`.schema-adaptive/` 아래 등록·profile·평가 기록)를 건드리지 않는다. 테스트는 임시 디렉터리.
- legacy DB 읽기 전용, live 모델 호출 없음, 커밋 없음, `backend/.env` 값 출력 없음.
- 문서(`AGENTS.md` 규칙):
  - `docs/schema-adaptive-text-to-sql-plan.md`: ADR-101(공통 ontology v3, 개설 연도 사용 중단, **사용자 결정**, 이유, A안을 고른 이유, 학기는 유지), 로드맵 9-16, 변경 이력 2026-10-13, ontology를 설명하는 절이 있으면 v3 반영.
  - `docs/human-validation-checkpoints.md`: "공통 ontology v3(개설 연도 사용 중단)는 사용자 결정이며 팀 공유가 필요하다" 한 줄. 관문 승인 없음.
  - `docs/development-history.md#next-session-handoff` 한 줄.
- 끝나면 바꾼 파일, 테스트 결과, 기대 결과가 달라진 평가 문항(있으면)을 짧게 적는다.

---

<a id="prompts-phase9-17-reserved-database-id"></a>

## 원본: prompts/phase9-17-reserved-database-id.md

# Phase 9-17 지침: 예약된 DB id(`legacy`) 등록 막기와 DB 목록 부분 실패 처리

## 증상 (사용자 보고, Claude 재현)

- 사용자가 관리자 화면 "DB 등록"에서 레거시 public schema를 Database ID `legacy`로 등록·게시했다. 그 뒤 메인 화면과 "DB 변경" 창이 "DB 목록을 불러오지 못했습니다. legacy 기준선으로 동작합니다."를 보이고 rehearsal 등 모든 DB가 사라졌다.
- 서버 `GET /api/adaptive/v1/databases`는 200으로 6개(`synth-clarification-prototype`, `synth-clarification-legacy`, `legacy-base`, `legacy-view`, `rehearsal-school`, `legacy`)를 정상으로 준다.
- 원인: frontend `src/adaptive-prototype/databases.ts`의 `LEGACY_SELECTION_ID = 'legacy'`(Legacy 기준선 선택지의 내부 id)와 등록 id `legacy`가 겹쳐 `parseDatabaseRegistry`가 목록 전체를 `null`로 돌린다. 그 응답에서 `legacy` 항목만 빼면 parser가 통과한다(재현).

## 고칠 것

1. **등록에서 예약 id 거절 (서버)**: PostgreSQL 등록(`api/postgres_registration.py`/`registration/postgres_registrations.py`)과 SQLite 업로드 등록의 database_id 검증에 예약 id 목록을 더한다. 최소 `legacy`. Legacy 기준선·화면이 예약해 쓰는 다른 id가 있으면 함께(코드에서 찾아 목록으로). 거절은 기존 INVALID_INPUT 계열 고정 코드 + 화면에서 "이 ID는 시스템이 쓰는 이름이라 쓸 수 없습니다" 같은 안내.
2. **화면 등록 양식**: 같은 예약 id를 제출 전에 막고 안내한다.
3. **DB 목록 부분 실패**: `parseDatabaseRegistry`가 예약 id와 겹치거나 형식이 잘못된 항목 **하나 때문에 전체를 버리지 않게** 한다. 그 항목만 빼고 나머지를 쓰며, 뺀 항목이 있으면 "일부 DB(`legacy`)는 이름이 예약어와 겹쳐 목록에서 뺐습니다" 같은 안내를 한 줄 보인다. 응답 전체 형식이 틀린 경우(최상위 키·version 등)는 지금처럼 실패로 둔다. 중복 id 등 다른 엄격 검사는 그대로.
4. 이미 저장된 `legacy` 등록 데이터는 건드리지 않는다(사용자가 지울지 결정).

## 테스트

- 서버: id `legacy`로 등록 시도 → 고정 코드 거절, 파일이 생기지 않음. 다른 id는 그대로 성공.
- frontend: 예약 id 항목이 섞인 목록에서 나머지가 파싱되고 경고가 나옴; 최상위 형식 오류는 여전히 null; 등록 양식이 `legacy`를 막음.
- 관련 테스트 파일만 실행, `npx tsc -b`.

## 지킬 것

- 사용자 서버·데이터(`.schema-adaptive/postgres-registrations/legacy` 포함) 안 건드림, live 모델 호출 없음, 커밋 없음, `backend/.env` 값 출력 없음. UTF-8로 쓴다.
- 문서: `docs/schema-adaptive-text-to-sql-plan.md` 로드맵 9-17, 변경 이력 2026-10-13 한 줄.
- 끝나면 바꾼 파일과 테스트 결과를 짧게 적는다.

---

<a id="prompts-phase9-18-delete-registration"></a>

## 원본: prompts/phase9-18-delete-registration.md

# Phase 9-18 지침: 등록된 PostgreSQL DB 삭제 기능

## 배경 (2026-10-13 사용자 요청)

- 관리자 화면 "DB 등록" 탭의 "등록된 PostgreSQL DB" 표에서 등록을 지울 방법이 없다. 지금은 `.schema-adaptive/postgres-registrations/<id>`와 `postgres-profiles/<id>` 폴더를 손으로 지우고 서버를 다시 켜야 한다.
- 당장의 사례: 예약 id와 겹친 `legacy` 등록(9-17)을 지우고 다른 id로 다시 등록하려 한다.

## 만들 것

1. **서버 API** (`api/postgres_registration.py`): `DELETE /api/adaptive/v1/postgres/registrations?database_id=<id>` (또는 기존 경로 규칙에 맞는 POST action). 본문/파라미터 엄격 검증.
   - 그 id의 PostgreSQL 등록 폴더(`postgres-registrations/<id>`: registration·catalog·proposal·draft·published·dictionaries·extensions·verified-queries 등)와 게시 profile 폴더(`postgres-profiles/<id>`)를 지운다.
   - **안전**: 경로는 기존 `pg._private_dir` 같은 검증 함수로만 만든다(경로 탈출·symlink 거절, 폴더가 symlink면 지우지 않고 거절). `postgres-registrations`에 없는 id(예: `legacy-view`, `legacy-base` profile, 합성 DB, SQLite 등록)는 고정 코드 NOT_FOUND로 거절한다. 예약 id(`legacy`)라도 이미 저장된 PostgreSQL 등록이면 지울 수 있어야 한다(9-17 정리용).
   - 실행 중 서버 상태에서도 빠지게: live chat context(`LivePostgresContexts`), 값 reader 등록, DB 목록(`/api/adaptive/v1/databases`), 등록 목록에서 즉시 사라진다(재시작 불필요). 등록 lock 안에서 처리.
   - 평가 기록(`.schema-adaptive/blind-evals/...`, `direct-sql-pg/...`)은 지우지 않는다.
   - 응답: `{database_id, deleted: true}` 같은 작은 고정 형식. 오류는 고정 코드만(경로·DSN 노출 없음).
2. **화면** (`PostgresRegistrationPanel.tsx`): 표의 각 줄 "동작"에 [삭제] 버튼.
   - 누르면 확인 상자: "`<id>`를 삭제합니다. 등록 정보, 검토·게시 내용, 값 사전, 확장 항목, 시험 질의 기준이 모두 지워지고 되돌릴 수 없습니다. 확인하려면 ID를 그대로 입력하세요." ID 입력이 정확히 같을 때만 [삭제] 활성.
   - 성공하면 표·DB 목록·"사용할 DB"를 새로 고친다. 지금 선택된 DB를 지웠으면 Legacy 기준선으로 돌아간다(기존 "DB 목록 실패 시 legacy" 동작과 같은 방식).
   - 검토·게시 탭에 그 DB가 열려 있었으면 닫는다.

## 테스트

- 서버(임시 디렉터리, 가짜 connector): 게시된 등록 삭제 → 두 폴더 없음, live context·DB 목록·등록 목록에서 빠짐, 같은 id로 재등록 가능(예약 id 제외). 없는 id·profile 전용 id·합성 id·경로 탈출·symlink 거절. 평가 기록 폴더는 남음.
- frontend: 확인 ID가 다르면 삭제 비활성, 성공 후 목록 갱신, 선택 중 DB 삭제 시 legacy 기준선으로.
- 관련 테스트 파일만, `npx tsc -b`.

## 지킬 것

- 사용자 서버·데이터(특히 `.schema-adaptive/postgres-registrations/legacy`, `rehearsal-school`)를 건드리지 않는다. 지우는 시험은 임시 디렉터리에서만.
- live 모델 호출 없음, 커밋 없음, `backend/.env` 값 출력 없음. UTF-8로 쓴다.
- 문서: `docs/schema-adaptive-text-to-sql-plan.md` 로드맵 9-18, 변경 이력 2026-10-13 한 줄(사용자 요청, PostgreSQL 등록만, 평가 기록은 남김).
- 끝나면 바꾼 파일과 테스트 결과를 짧게 적는다.

---

<a id="prompts-phase9-19-database-id-message"></a>

## 원본: prompts/phase9-19-database-id-message.md

# Phase 9-19 지침: DB ID 입력 오류 안내 고침

## 증상 (사용자 보고, Claude 확인)

- "DB 등록" 탭에서 DB 식별자(ID)를 `legacy_test`로 넣고 등록하자 "서버 응답 형식이 올바르지 않습니다."가 나왔다.
- 원인: ID 규칙은 `^[a-z0-9][a-z0-9-]{0,63}$`(`backend/adaptive/registration/contracts.py:27`)라 밑줄이 안 된다. `src/adaptive-prototype/postgresRegistration.ts` 약 134행이 요청 전에 `isDatabaseId`로 막으면서 `contract` 오류를 던지고, 화면은 이를 "서버 응답 형식" 문구로 보인다. 서버에는 요청이 가지 않았다.

## 고칠 것 (화면만)

- 등록 입력 검증 실패는 `contract`가 아니라 입력 오류 종류로 구분해 안내한다.
  - ID 규칙 위반: "ID는 영문 소문자·숫자·하이픈(-)만 쓸 수 있고, 첫 글자는 영문 소문자나 숫자여야 합니다(64자 이내). 예: legacy-pg"
  - 표시 이름 규칙 위반: 기존 label 규칙에 맞는 안내(허용 문자와 길이).
  - 예약 id는 9-17 안내 그대로.
- ID 입력란 아래에 규칙 한 줄 안내를 항상 보인다. 입력 중 규칙에 맞지 않으면 입력란 옆에 바로 알리고 [등록]을 비활성으로(선택 사항이지만 권장).
- 진짜 서버 응답 형식 오류(`contract`)의 문구는 그대로 둔다.
- SQLite 업로드 양식에도 같은 ID 검증이 있으면 같은 방식으로.

## 테스트

- 밑줄·대문자·빈 값·65자 ID → 입력 오류 문구, 요청 안 감. 올바른 ID → 요청 감. 응답 형식 오류 → 기존 문구.
- 관련 frontend 테스트만, `npx tsc -b`.

## 지킬 것

- 서버·사용자 데이터 안 건드림, 커밋 없음, UTF-8로 쓴다.
- 문서: `docs/schema-adaptive-text-to-sql-plan.md` 변경 이력 2026-10-13 한 줄(9-19).
- 끝나면 바꾼 파일과 테스트 결과를 짧게 적는다.

---

<a id="prompts-phase9-20-copy-dictionaries"></a>

## 원본: prompts/phase9-20-copy-dictionaries.md

# Phase 9-20 지침: 다른 DB의 값 사전 가져오기

## 배경 (2026-10-13 사용자 결정)

- 사용자가 레거시 public schema를 `legacy-test`로 새로 등록·게시했다. 챗봇에서 "컴융 전공 보여줘"가 "컴융"을 못 알아봤다(이해한 조건: 이수 구분 "전공"만, 200건).
- 원인(확인됨): 기본 제공 `legacy-view`/`legacy-base` profile에는 개발자가 옮긴 값 사전이 있다(`api/legacy_view_profile.py` `VALUE_DICTIONARIES`: 이수 구분 9개 + 학과명 `공과대학 컴퓨터융합학부` = 컴융·컴융부). 새 등록에는 값 사전이 없다.
- 사용자 결정: 다른 DB의 값 사전을 새 등록으로 가져오는 기능을 만든다.

## 만들 것

1. **서버**: PostgreSQL 등록의 값 사전에 "다른 DB에서 가져오기" 동작.
   - 가져올 수 있는 출처: 게시된 PostgreSQL 등록(`postgres-registrations/<id>` 값 사전)과 서버 시작 profile(`legacy-view`, `legacy-base`의 `value_dictionaries`). 출처 목록 API(id·이름·항목별 개수만)와 가져오기 API(`source_database_id`, 대상 `database_id`, 검토자).
   - 항목(공통 field) 이름이 같은 사전만 대상으로 한다. 대상 DB에서 그 항목이 연결되어 있고(이 DB에 없음·사용 중단 항목 제외) 값 사전 편집 대상(`editable_fields`)인 경우만.
   - 항목마다 **출처의 저장 값이 대상 DB의 그 칸에 실제로 있는 값일 때만** 가져온다(기존 값 사전 검증과 같은 방식: 대상 칸의 distinct 값 읽기, 기존 `check_alias` 등 별칭 검증 재사용). 없는 값은 건너뛰고 개수와 이유를 돌려준다.
   - 대상에 같은 저장 값 항목이 이미 있으면 별칭을 합친다(중복 제거). 뜻(label)은 대상 것이 있으면 유지.
   - 결과는 대상의 **편집 중 사전**에 들어간다(자동 게시 안 함). 기존 "사전 적용해 게시"로 반영한다.
   - 응답: 항목별 가져온 수·건너뛴 수(이유 코드). 값 자체는 화면에서 사전 표로 보이므로 응답에 다시 싣지 않아도 된다.
2. **화면** (`ValueDictionaryPanel.tsx`): "다른 DB 사전 가져오기" 줄 — 출처 선택(자기 자신 제외), [가져오기]. 끝나면 "학과명 1개, 이수 구분 9개 가져옴. 건너뜀 0개" 같은 요약과 사전 표 새로고침. 검토자 필요(탭 상단 공통 값).

## 테스트

- 서버(임시 디렉터리, 가짜 reader): `legacy-view` 같은 profile 출처와 등록 출처 각각에서 가져오기; 대상에 없는 값 건너뜀; 연결 안 된·없음·사용 중단 항목 제외; 기존 항목 별칭 합침; 자동 게시 안 됨; 잘못된 출처·자기 자신 거절.
- frontend: 출처 목록·요약 표시.
- 관련 테스트만, `npx tsc -b`.

## 지킬 것

- 사용자 서버·데이터(`legacy-test`, `rehearsal-school` 등) 안 건드림, live 모델 호출 없음, 커밋 없음, `backend/.env` 값 출력 없음. UTF-8로 쓴다.
- 문서: `docs/schema-adaptive-text-to-sql-plan.md` 로드맵 9-20, 변경 이력 2026-10-13 한 줄(사용자 결정).
- 끝나면 바꾼 파일과 테스트 결과를 짧게 적는다.

---

<a id="prompts-phase9-21-unmatched-word-notice"></a>

## 원본: prompts/phase9-21-unmatched-word-notice.md

# Phase 9-21 지침: 이해하지 못한 낱말을 조용히 빼지 않고 알리기

## 배경 (2026-10-13 사용자 결정)

- `legacy-test`(값 사전 없음)에서 "컴융 전공 보여줘" → 모델이 "컴융"을 버리고 `이수 구분 ∋ 전공`만으로 200건을 냈다. 질문의 조건 하나가 조용히 빠진 답이다(조용한 오답에 가까움). 화면의 "이해한 조건"을 봐야만 알아챌 수 있다.
- 사용자 결정: 질문의 낱말 중 어떤 항목·값에도 연결되지 않은 것이 있으면 조용히 빼지 않고 알린다.
- 기존 장치: `api/unknown_conditions.py`(8-20: 공통 항목 밖 개념 목록 `outside-concepts.yaml`, 빈 결과 진단), `api/value_hints.py`(값 힌트, `lookup_key`, `ValueIndex`), `api/time_hints.py`.

## 만들 것 (규칙은 닫힌 형태로, 모델 추가 호출 없음)

1. **이해 못 한 낱말 찾기** (질문당 한 번, 모델 호출 뒤):
   - 질문을 공백 단위 낱말로 나누고, 끝의 흔한 조사·어미(예: 은/는/이/가/을/를/의/에/에서/으로/로/과/와/도/만, 보여줘/알려줘/찾아줘/있어/뭐야 등)를 떼어 낸 2글자 이상 한글·영문 낱말만 본다.
   - 다음 중 하나라도 해당하면 "이해함"으로 본다:
     - 이해한 조건(QueryIR filters)의 값·연산과 `lookup_key` 기준으로 같거나 포함 관계.
     - 값 힌트가 그 낱말에서 찾은 값이 있음(그 DB의 값·값 사전 별칭 포함).
     - 공통 항목·확장 항목의 이름·별칭(ontology label/aliases), 엔터티 이름·별칭, `domain_nouns`, 시간 표현 규칙(`time_hints`)이 읽은 낱말, 숫자+단위(3학점, 2학년, 60명, 오후 3시 등).
     - 고정 일반어 목록(과목, 수업, 강의, 강좌, 목록, 전체, 모든, 정보, 이상, 이하, 이후, 이전, 오전, 오후, 저녁, 시간, 요일 이름 등). 목록은 코드 상수로 두고 테스트로 고정한다.
   - 남는 낱말이 "이해 못 한 낱말"이다.
2. **답에 알리기**: 답을 막지는 않는다. 응답에 `unmatched_terms`(최대 5개)를 더하고, 화면은 결과 위에 한 줄 "‘컴융’을(를) 이해하지 못해 조건에서 뺐습니다. 값 사전에 줄임말을 넣으면 알아들을 수 있습니다." 같은 안내를 보인다(관리자 안내 문구는 짧게).
   - 이미 거절(UNSUPPORTED/NOT_UNDERSTOOD)이면 덧붙이지 않는다.
   - 기존 8-20 outside-concept 거절 규칙과 충돌하지 않게: outside concept로 거절된 경우는 그대로 거절.
3. **화면 parser**: 챗봇 응답 parser가 새 필드를 선택 필드로 받는다(없어도 통과).
4. 레거시 기준선 챗봇(8000)은 바꾸지 않는다.

## 테스트 (가짜 모델 출력으로)

- "컴융 전공 보여줘" + 값 사전 없음 + 모델 출력 `category ∋ 전공` → `unmatched_terms = ["컴융"]`.
- 같은 질문 + 값 사전에 컴융 → 학과 조건이 있고 `unmatched_terms` 없음.
- 레거시 비교 30문항(`backend/adaptive/evaluation/data/legacy-chat-compare-v1.json`)의 질문을 9-08/9-10에서 쓰던 고정 결과(있으면) 또는 gold 성격의 조건과 함께 넣어, **정상 질문에서 잘못된 알림이 거의 없는지** 확인한다. 각 질문의 판정(알림 낱말)을 표로 테스트에 고정하고, 잘못된 알림이 있으면 일반어 목록·조사 처리로 고치되 특정 질문 맞춤 예외는 두지 않는다. 이 확인은 개발 질문 기준이라는 점을 결과 문서에 적는다.
- 시간·숫자 표현("오후 3시 이후", "3학점", "60명 이상"), 요일, 학년이 알림을 내지 않는지.
- 관련 테스트만, frontend 관련 테스트와 `npx tsc -b`.

## 지킬 것

- 모델 호출 수 그대로(질문당 1번), QueryIR 검증·compiler·policy·executor 우회 없음, 조건을 새로 만들지 않음.
- 사용자 서버·데이터 안 건드림, live 모델 호출 없음, 커밋 없음, `backend/.env` 값 출력 없음. UTF-8로 쓴다.
- 문서: `docs/schema-adaptive-text-to-sql-plan.md`에 ADR-102(사용자 결정, 알림만·막지 않음, 이유: 조용한 조건 누락), 로드맵 9-21, 변경 이력 2026-10-13. 결과는 개발 질문 기준 잠정치로 적는다.
- 끝나면 바꾼 파일, 테스트 결과, 30문항 중 알림이 난 질문과 낱말을 짧게 적는다.

---

<a id="prompts-phase9-22-admin-style"></a>

## 원본: prompts/phase9-22-admin-style.md

# Phase 9-22 지침: 관리 창 스타일 정리와 검토자 기억

## 배경 (2026-10-13 사용자 요청)

1. 값 사전의 [가져오기]·[사전 적용해 게시]가 눌리지 않았다. 원인: 검토자 이름이 비어 있었다. 9-14 이후 검토자는 "검토·게시" 탭 맨 위 한 곳(`MappingReviewPanel`의 `reviewer` state)에서만 입력하는데, 새로고침하면 비고, 값 사전 상자 안에는 "먼저 검토자 이름을 넣어 주세요" 문구만 있어 어디에 넣는지 모른다.
2. "DB 변경" 관리 창(9-12에서 배치만 바꾸고 스타일 통일은 미룸)의 버튼 대부분이 외곽선·배경이 없어 글자와 구분되지 않는다. 이제 스타일을 정리한다.

## 고칠 것 (화면만, API·동작·검증 규칙 변경 없음)

### A. 검토자
- 검토자 이름을 브라우저에 기억한다(`localStorage`, 키 하나, 앱 공통). 읽기·쓰기는 모두 try/catch로 감싸고 실패해도 화면이 정상 동작해야 한다(사생활 창·차단 환경). 처음 열 때 기억된 값으로 채운다.
- 검토자가 비어 있을 때 하위 상자(값 사전·확장 항목·시험 질의·게시)의 안내 문구를 "맨 위 ‘검토자’ 칸에 이름을 넣어 주세요"로 바꾸고, 문구를 누르면 그 입력칸으로 스크롤·focus한다.
- 비활성 버튼에는 왜 비활성인지 `title`(마우스 올림 설명)을 단다(검토자 없음 / 저장하지 않은 편집 있음 / 출처 미선택 등 실제 조건대로).

### B. 스타일 통일 (메인 화면의 기존 Tailwind 스타일에 맞춘다)
- 공용 class 상수 파일(예: `src/adaptive-prototype/adminStyles.ts`)을 만들고 관리 창 안의 모든 컴포넌트에서 쓴다: `MainPage.tsx`의 DB 창, `PostgresRegistrationPanel`, `MappingReviewPanel`, `MappingReviewRow`, `ValueDictionaryPanel`, `CodeTableImport`, `ExtensionFieldPanel`, `TestQueryPanel`, `SchemaComparePanel`, SQLite 등록 영역.
  - 기본 버튼(주요 동작: 등록, 검토 시작, 게시, 사전 적용해 게시, 저장): 메인 화면 "적용" 버튼과 같은 파란 채움(`bg-blue-600 text-white hover:bg-blue-700 rounded-lg h-9 px-3 text-sm font-semibold`).
  - 보조 버튼(검토 열기, 다른 칸으로, 다시 정하기, 가져오기, 추가, 새로고침, 취소 등): 메인 "취소"와 같은 흰 바탕+외곽선(`border border-slate-200 bg-white text-slate-700 hover:bg-slate-50`).
  - 위험 버튼(삭제, 이 DB에 없음): 빨간 외곽선(`border-red-200 text-red-700 hover:bg-red-50`), 확인 단계의 최종 삭제는 빨간 채움.
  - 비활성: 흐리게(`disabled:opacity-50 disabled:cursor-not-allowed`), 눌림 표시 없음.
  - 입력·선택(input/select/textarea): 메인 검색창 계열(`h-9 rounded-lg border border-slate-200 bg-white px-3 text-sm focus:border-blue-400 focus:ring-4 focus:ring-blue-100`).
  - 상자(section/fieldset): 흰 바탕, `rounded-xl border border-slate-200 p-4`, 제목 `text-sm font-semibold text-slate-900`, 설명 `text-xs text-slate-500`.
  - 상태 배지: 확정=초록, 대기=노랑, 없음=회색, 게시됨=파랑, 검토 대기=노랑 계열의 연한 배경 배지.
  - 안내·오류·성공 메시지: 기존 amber/red/emerald 연한 상자 스타일로 통일.
- 표(등록 목록, 항목 연결 표): 머리글 연한 회색 배경, 줄 구분선, 버튼은 한 줄로 정렬(좁으면 줄바꿈).
- 글자·문구는 바꾸지 않는다(A의 안내 문구 제외).

## 확인

- 관련 frontend 테스트(동작 의도가 같으면 class 문자열 기대만 고친다), `npx tsc -b`, 관련 파일 eslint.
- 서버를 띄우지 않는다. 화면 확인은 Claude가 한다.

## 지킬 것

- 서버·사용자 데이터 안 건드림, live 모델 호출 없음, 커밋 없음, `backend/.env` 값 출력 없음. UTF-8로 쓴다.
- 문서: `docs/schema-adaptive-text-to-sql-plan.md` 로드맵 9-22, 변경 이력 2026-10-13 한 줄(사용자 요청: 스타일 통일·검토자 기억).
- 끝나면 바꾼 파일과 확인 결과를 짧게 적는다.

---

<a id="prompts-phase9-23-luna-fixes"></a>

## 원본: prompts/phase9-23-luna-fixes.md

# Phase 9-23 지침: Luna 리뷰(9-11~9-22) 확인 지적 고침 + 남은 회귀 테스트

Luna(`gpt-6-luna` high) 리뷰 지적 4건을 Claude가 확인했다. 각 고침에 회귀 테스트를 둔다. 다른 동작은 바꾸지 않는다.

## 1. 삭제 뒤 값 힌트 cache가 남음 (High, 코드로 확인)

- `api/postgres_registration.py` `delete()`(약 242~248행)는 live context와 reader 설정만 지우고 `api/value_hints.py` `ChatValueIndexes` cache(key = database_id, schema fingerprint, mapping version)는 지우지 않는다. 같은 id·같은 구조로 다시 등록해 v1을 게시하면 이전 등록의 값·별칭이 힌트로 다시 쓰인다.
- 고침: 등록 삭제 때 그 database_id의 cache 항목을 모두 지운다(`ChatValueIndexes`에 `forget(database_id)` 같은 메서드, 삭제 경로에서 호출). 서버가 cache 객체에 접근하는 기존 방식을 따른다.
- 테스트: cache에 올라간 등록 삭제 → 같은 id 재등록·같은 key → 새로 build됨(이전 값 미사용).

## 2. `Path.is_junction()` 없는 Python에서 삭제 실패 (Low, 확인)

- `registration/postgres_registrations.py` `delete_registration`(약 358~372행)이 `Path.is_junction()`을 부른다(Python 3.12+). `env_check.py`는 Python 3.10 이상을 지원한다. 3.10/3.11에서는 `AttributeError` → `STORAGE_ERROR`.
- 고침: 이웃 코드의 호환 처리처럼 `getattr(path, "is_junction", None)`이 없으면 Windows에서는 `os.path.isjunction`(3.12+) 또는 reparse point 속성(`os.lstat(...).st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT`)으로 판정하는 helper를 쓴다. 판정할 수 없으면 안전하게 거절하지 말고, symlink 검사는 그대로 유지한다.
- 테스트: `is_junction`이 없는 Path를 흉내 낸 경우에도 일반 삭제가 성공하고, reparse point로 표시된 경로는 거절.

## 3. 사용 중단 필드에 대한 검토 변경 요청이 받아들여짐 (Low, 코드로 확인)

- `mapping/review.py` `approve`(약 45~71행)와 remap/ignore가 ontology 현재 버전의 사용 중단 필드(`CourseOffering.year`)를 거르지 않는다. v2 proposal에 year가 있으면 화면에 안 보이는데도 API로 APPROVE가 저장된다.
- 고침: 검토 변경(APPROVE/REMAP/IGNORE)과 일괄 경로에서 `ontology.retired_fields`에 든 필드는 기존 "알 수 없는 필드" 오류와 같은 고정 코드로 거절한다. 읽기(v2 초안 로드)는 그대로 된다.
- 테스트: v2 초안에서 year APPROVE/REMAP/IGNORE → 거절, 다른 필드는 정상.

## 4. 요청 동사의 잘못된 알림 (Low, 재현함)

- `api/unmatched_terms.py`: "교양 과목 정리해줘" + 조건 `Course.category ∋ 교양` → `['정리해줘']`. "보여줘/알려주세요/찾아주세요"는 알림 없음.
- 고침: 특정 낱말 추가가 아니라 **요청 어미 일반 규칙**으로: 낱말이 `해줘`, `해 줘`, `줘`, `주세요`, `해주세요`, `해봐`, `할래`, `해` 등 요청·명령 어미로 끝나면 그 어미를 뗀 뒤 남는 어간이 일반어이거나 비면 알리지 않는다. 어간 자체(예: "정리")는 일반 동사 목록(정리, 추천, 검색, 확인, 비교, 나열, 소개 등 짧은 고정 목록)에 있으면 알리지 않는다. "컴융 전공 보여줘"의 "컴융" 알림은 그대로 나와야 한다.
- 테스트: 위 4개 질문 + "컴융 전공 정리해줘" → `['컴융']`; 9-21의 30문항 고정 판정 표가 그대로인지.

## 5. 남은 회귀 테스트: `direct_sql_pg --reference-database`

- 9-10 측정 중 Claude가 `backend/adaptive/evaluation/direct_sql_pg.py`에 `--reference-database`를 더했다(QueryIR arm이 profile `legacy-view`를 쓸 때 기준 SQL을 다른 등록 연결로 실행). 테스트가 없다.
- 테스트: 가짜 `load_context`로 `--arm queryir --set legacy --database legacy-view --reference-database X`가 기준 계산에 X를 쓰는지, 옵션이 없으면 `--database`를 쓰는지. 기존 동작은 바꾸지 않는다.

## 확인과 지킬 것

- 관련 테스트 파일만 실행한다(전체 pytest는 Claude가 커밋 전에 한 번).
- 사용자 서버·데이터 안 건드림, live 모델 호출 없음, 커밋 없음, `backend/.env` 값 출력 없음. UTF-8로 쓴다.
- 문서: `docs/schema-adaptive-text-to-sql-plan.md` 변경 이력 2026-10-13 "9-23 Luna 지적 4건 고침 + reference-database 테스트" 한 줄, 로드맵에서 9-11~9-22의 Luna 리뷰 대기 표시가 있으면 완료로.
- 끝나면 바꾼 파일과 테스트 결과를 짧게 적는다.

---

<a id="prompts-phase9-24-v3-fallout"></a>

## 원본: prompts/phase9-24-v3-fallout.md

# Phase 9-24 지침: ontology v3(개설 연도 사용 중단) 뒤 전체 pytest 실패 29건 정리

## 상황

- 9-16에서 공통 ontology를 v3으로 올리고 `CourseOffering.year`를 사용 중단했다(ADR-101, 사용자 결정). 당시 관련 테스트만 돌렸다.
- Claude가 커밋 전 전체 `pytest tests`를 돌렸더니(`--basetemp`를 저장소 밖 임시 폴더로) **29 failed, 2303 passed, 11 skipped**. 실패 파일(개수): `test_admin_test_queries`(1), `test_arm_live_cli`(3), `test_capture_targets`(1), `test_direct_sql_arm`(1), `test_mapping_carry_over`(4), `test_postgres_live_smoke`(3), `test_raw_columns_and_select`(1), `test_recorded_model`(1), `test_schema_compare`(2), `test_schema_variation_suite`(4), `test_synthetic_active_database`(1), `test_upload_ai_proposal`(4).
- 대표 오류: `assert 114 == 120`(oracle/benchmark 사례 수), `{'count': 190} != {'count': 200}`, `KeyError: 'CourseOffering.year'`, 항목 수 `15 == 16`/`14 == 15`, `'AI_UNAVAILABLE' == 'AI'`, `'MAPPING_FAILED' != 'QUERY_FAILED'`, `'NOT_UNDERSTOOD' == 'SUCCEEDED'`, 응답 400, `IndexError`.
- 기본 임시 폴더 `C:\Users\micea\AppData\Local\Temp\pytest-of-micea`는 권한 문제로 쓸 수 없다. pytest는 `--basetemp`를 작업 공간 안의 git-ignored 위치나 샌드박스 임시 위치로 지정해 돌린다(끝나면 지운다).

## 원칙 (실패마다 하나로 분류해서 처리)

1. **과거 기록 재현(평가·benchmark·oracle·gold 재생·녹화 replay·capture targets)**: gold·fixture·녹화가 만들어진 ontology 버전(v1/v2)으로 평가한다. 즉 오프라인 평가 경로는 현재 버전이 아니라 **기록이 선언한 버전**(`ontology.at_version(v)`)을 써서, 기존 수치(120건 등)가 그대로 재현되게 한다. gold·fixture·녹화 파일은 바꾸지 않는다. 기록에 버전이 없으면 v2로 본다(9-16 이전 기록). 실제 챗봇 HTTP를 부르는 평가(`blind_eval` 등)는 그대로 현재 runtime(v3)이다. 9-16 결과 문서의 "기대 결과가 달라지는 평가 문항" 절을 이 원칙에 맞게 고친다(오프라인 benchmark는 기록 버전으로 재현, 현재 챗봇으로 다시 재면 연도 문항은 거절).
2. **제품 동작이 9-16에서 의도대로 바뀐 것**(새 등록·검토의 항목 수, carry-over·schema compare에서 year가 빠짐 등): 테스트 기대를 v3 동작으로 고치고, 테스트 이름이나 주석에 이유를 한 줄 남긴다. 기존 v2 base에서 v3 target으로 carry-over할 때 year는 옮기지 않는 것이 맞다.
3. **충돌·회귀**(`KeyError: 'CourseOffering.year'`, `IndexError`, 정상 경로가 `AI_UNAVAILABLE`·`MAPPING_FAILED`·400·`NOT_UNDERSTOOD`로 바뀐 것 등): 테스트를 고치지 말고 코드를 고친다. 예: AI 제안에 사용 중단 필드가 섞여 와도 그 필드만 버리고 나머지는 쓰기, 필수 필드 목록에서 사용 중단 필드 제외, 시험 질의·live smoke 경로가 year를 가정하지 않기.

## 할 일

- 실패 29건을 하나씩 위 1·2·3으로 분류해 고친다. 분류표(테스트 이름 → 분류 → 한 줄 이유 → 고친 곳)를 `docs/experiment-results.md#phase9-16-retire-offering-year-results-2026-10-13` 끝에 "9-24 전체 테스트 정리" 절로 더한다.
- 12개 실패 파일 전부와 9-16·9-23에서 바꾼 파일의 테스트를 다시 돌려 모두 통과시킨다. 가능하면 `pytest tests -q`(전체, 약 40분)를 마지막에 한 번 돌려 결과를 적는다. 시간이 모자라면 실패 파일과 관련 파일까지만 돌리고 그렇다고 적는다.
- 사용 중단 원칙(ADR-101)이 바뀌는 것이 아니다. 제품 runtime은 v3 그대로다.

## 지킬 것

- gold `review_status`·gold·fixture·녹화 파일 수정 금지. 사용자 서버·데이터 안 건드림, live 모델 호출 없음, 커밋 없음, `backend/.env` 값 출력 없음. UTF-8로 쓴다.
- 문서: `docs/schema-adaptive-text-to-sql-plan.md` ADR-101에 "오프라인 평가는 기록 버전으로 재현" 한 줄, 변경 이력 2026-10-13 "9-24" 한 줄.
- 끝나면 분류 요약(1·2·3 각 개수), 바꾼 파일, 테스트 결과를 짧게 적는다.

---

<a id="prompts-phase9-25-teammate-guide"></a>

## 원본: prompts/phase9-25-teammate-guide.md

# Phase 9-25 지침: 팀원 평가 안내서를 현재 화면에 맞게 갱신 (문서만)

## 배경

- `docs/teammate-evaluation-db-guide.md`(91줄)는 Phase 8-22에 썼고, 그 뒤 관리 화면과 규칙이 많이 바뀌었다(9-11~9-24, 커밋 `c0f46f0`). 팀원이 첫 평가에서 헤매면 등록 비용 측정이 흐려진다.
- 코드는 바꾸지 않는다. 안내서의 다른 절(지켜야 할 것, 환경 점검, DB 만들기, 질문 파일, 실행)은 사실이 바뀐 곳만 고친다. 특히 "AI는 질문 파일을 평가 전까지 보지 않는다", "첫 실행을 그대로 기록" 규칙은 그대로 둔다.

## 반영할 현재 사실 (코드와 docs/schema-adaptive-text-to-sql-plan.md 변경 이력으로 확인해서 쓴다)

1. **창 구조**: 메인 화면 "DB 변경" → 탭 3개(사용할 DB / DB 등록 / 검토·게시).
2. **등록**: 연결·schema 선택, DB 식별자(ID) 규칙(영문 소문자·숫자·하이픈, 첫 글자 영문 소문자/숫자, 64자 이내, 밑줄 불가, `legacy` 같은 예약 id 불가), 표시 이름 규칙. 등록 목록 표의 상태 배지(검토 대기/게시됨 vN), [검토 열기], [삭제](ID를 그대로 입력해 확인, 되돌릴 수 없음, 평가 기록은 남음). 잘못 등록했으면 삭제 후 다른 id로 다시 등록.
3. **검토·게시 탭 순서**:
   1. 맨 위 "검토자"에 이름(브라우저가 기억).
   2. "항목 연결 검토 시작".
   3. "AI 추천 모두 승인"(대기 + 추천 있는 항목만, 게시 안 함) → **표를 꼭 훑어본다**: 이름만 보고 고른 잘못된 연결이 있을 수 있다(예: 학년 칸을 개설 연도로, 수강 기록 table의 학기를 강좌 학기로). 고칠 줄은 [다시 정하기] → [다른 칸으로](처음 값은 현재 칸) 또는 [이 DB에 없음](사유 필요).
   4. 숫자·시간이 글자 칸에 섞여 있으면(예: `3-2-2`, `09:00-10:15`) 게시가 "막힌 이유"로 알려 준다. 정해진 변환이 없으면 "이 DB에 없음"으로 둔다.
   5. **값 사전**: 코드값의 뜻(예: GE = 교양)과 학교 줄임말(예: 소웨 = 소프트웨어학부)을 넣는다. "다른 DB 사전 가져오기"(대상 칸에 실제로 있는 값만), "코드표에서 가져오기", 약어 후보(규칙/규칙+AI; 학교 고유 줄임말은 AI가 잘 못 맞힘). 넣은 뒤 "사전 적용해 게시".
   6. 확장 항목(공통 항목에 없는 이 DB만의 정보, 선택).
   7. 게시 → "시험 질의"로 결과 확인, 맞으면 "기준으로 저장"(이후 회귀 확인).
4. **공통 항목**: 공통 ontology v3, 개설 연도는 사용 중단(ADR-101) — DB에 연도 칸이 있어도 질문 항목이 아니다. 학기는 그대로.
5. **챗봇**: 모델 선택(일반 모델 기본), 이해하지 못한 낱말은 답 위에 알림(ADR-102), "이해한 조건"을 보고 결과를 판단.
6. **평가 실행**: `blind_eval check|run|report`와 `--model`. 첫 실행이 잠긴다는 점. QueryIR과 직접 SQL 비교 실행(`python -m backend.adaptive.evaluation.direct_sql_pg run --set blind:<ID> --database <ID> --model general`)은 Claude/개발자가 돌리는 것이며 팀원이 할 일은 아님을 한 줄로.
7. **기록해 달라는 것**(등록 비용 측정용): 등록부터 게시까지 걸린 시간, 고친 항목 수, 값 사전에 넣은 줄 수, 막혀서 물어본 곳. 짧은 표 양식 하나.

## 형식

- 짧은 한국어, 순서가 있는 단계는 번호 목록. 화면에 실제로 보이는 버튼 이름을 그대로(코드의 문구를 확인해서) 쓴다.
- 길이는 지금의 두 배를 넘기지 않는다.
- `docs/schema-adaptive-text-to-sql-plan.md` 변경 이력에 2026-10-14 "9-25 팀원 안내서 갱신" 한 줄.

## 지킬 것

- 문서만 바꾼다. 서버·데이터·코드 안 건드림, 커밋 없음, UTF-8.
- 끝나면 바꾼 절 목록을 짧게 적는다.

---

<a id="prompts-phase9-26-dictionary-refresh"></a>

## 원본: prompts/phase9-26-dictionary-refresh.md

# Phase 9-26 지침: 검토·승인·게시 뒤 값 사전·확장 항목 상자를 자동으로 새로 고침

## 증상 (Claude가 테스트 화면에서 재현, 2026-10-14)

- 새 PostgreSQL 등록 → 검토·게시 탭 → "항목 연결 검토 시작" → "AI 추천 모두 승인" → "검토한 항목 연결 게시"(v1)까지 했는데, 오른쪽 **값 사전** 상자는 처음 열렸을 때 그대로 "값 사전: 승인된 문자열 field가 없습니다(검토에서 먼저 승인)."를 보였다. 상자 안 [새로고침]을 눌러야 출처·항목 선택과 [가져오기]가 나타났다.
- 원인(추정, 확인할 것): `PostgresRegistrationPanel.tsx`가 `<ValueDictionaryPanel key={\`dictionary-${opened}\`} …>`로 한 번만 mount하고, 패널은 mount 때만 사전 정보를 읽는다. `MappingReviewPanel`의 승인·변경·없음·일괄 승인·게시로 검토 상태가 바뀌어도 알 길이 없다. `ExtensionFieldPanel`(연결된 항목·table 목록에 의존)도 같은 문제일 수 있다.

## 고칠 것 (화면만, API 변경 없음)

- `MappingReviewPanel`이 검토 상태가 바뀔 때마다 바뀌는 값(예: 검토 detail의 결정 상태 요약 + `mapping_version`/`active_mapping_version`을 묶은 문자열이나 증가하는 revision 번호)을 `reviewTools` 쪽으로 넘긴다(지금 `reviewTools(reviewer)` 형태를 확장).
- `ValueDictionaryPanel`과 `ExtensionFieldPanel`은 그 값이 바뀌면 자기 정보를 다시 읽는다. 단, **저장하지 않은 편집(dirty)이 있으면 덮어쓰지 않고** "검토 내용이 바뀌었습니다. [새로고침]" 안내만 보인다.
- 값 사전 쪽 "사전 적용해 게시"와 확장 항목 게시가 성공하면 `MappingReviewPanel`도 최신 상태(버전)를 다시 읽는다(이미 그렇다면 그대로).
- [새로고침] 버튼은 그대로 둔다.

## 테스트

- frontend: revision 값이 바뀌면 다시 읽기 호출이 한 번 일어나고, dirty면 다시 읽지 않고 안내가 나오는지(기존 테스트 방식으로 가능한 범위, 필요하면 판단 로직을 순수 함수로 빼서 테스트). 기존 관련 테스트 통과.
- `npx tsc -b`, 관련 파일 eslint, 전체 `node --test tests/frontend/*.test.ts`.

## 지킬 것

- 서버·사용자 데이터 안 건드림, live 모델 호출 없음, 커밋 없음. UTF-8로 쓴다.
- 문서: `docs/schema-adaptive-text-to-sql-plan.md` 변경 이력 2026-10-14 "9-26" 한 줄.
- 끝나면 바꾼 파일과 테스트 결과를 짧게 적는다.

---

<a id="prompts-phase9-27-renames"></a>

## 원본: prompts/phase9-27-renames.md

# Phase 9-27 지침: 팀원이 읽기 쉽게 이름 정리 (B·C, 코드)

## 배경 (2026-10-17 사용자 결정)

팀원이 저장소를 받아 처음 실행·읽을 때 헷갈리는 이름을 정리한다. 동작은 바꾸지 않는다. Codex 읽기 전용 감사(`feat/schema-adaptive-agent`, 커밋 `bbb38b3` 기준)의 지적 중 사용자가 B·C 전부를 승인했다.

## 바꿀 것

### B. 작은 이동·삭제
1. `backend/test_correction.py`(assertion 없는 예제 스크립트, 저장소 루트에서 `pytest`를 치면 `ModuleNotFoundError: keyword_extract`로 수집 실패) → `backend/scripts/demo_ner_correction.py`로 옮긴다. `if __name__ == "__main__":` 아래로 실행 코드를 넣고, import는 `python -m backend.scripts.demo_ner_correction`로 돌 수 있게 패키지 기준으로 고친다(필요하면 `backend/scripts/__init__.py`). 이 파일을 import하는 곳은 없다. 문서 참조가 있으면 새 경로로.
2. 참조 없는 `src/assets/vite.svg`, `src/assets/react.svg` 삭제(삭제 전에 참조 0개를 다시 확인).
3. `src/adaptive-prototype/AdaptivePrototypePage.tsx` → 역할에 맞는 이름(실제 역할을 코드로 확인해 정한다; 합성 clarification 시연 화면이면 `ClarificationDemoPage.tsx`). import 1곳(`src/main.tsx`)과 테스트 문자열 참조 갱신.

### C. 큰 이름 변경
1. `backend/adaptive/api/synthetic_prototype.py` → `backend/adaptive/api/app_factory.py` (dev 서버의 적응형 FastAPI 앱 조립: 합성 기본 구성 + PostgreSQL 등록·QueryIR 챗봇).
2. `backend/adaptive/api/synthetic_registry.py` → `backend/adaptive/api/database_registry.py` (DB 목록·course catalog; 합성 DB와 게시된 PostgreSQL DB를 함께 다룸).
3. `src/adaptive-prototype/` → `src/adaptive/`.

각 변경에서:
- 모든 import(제품·테스트), `package.json` 스크립트, `python -m ...` CLI 경로, 테스트 안의 모듈 경로 문자열·monkeypatch 대상 문자열, `vite`/`tsconfig`/eslint 설정의 경로를 함께 바꾼다.
- **바꾸지 말 것**: 저장된 계약 버전 문자열(예: `synthetic-database-registry-v1`), DB ID(`synth-clarification-prototype` 등), HTTP 경로(`/api/adaptive/v1/...`), 브라우저 URL 경로(`/adaptive-prototype` 등 페이지 경로), `localStorage` 키, `.schema-adaptive/` 저장 형식. 이것들은 사용자 데이터·북마크·기록과 연결된다.
- 옛 모듈 이름으로 import하는 외부 사용자는 없으므로 호환 shim은 두지 않는다. 단 `npm run` 스크립트 이름(`dev:all`, `dev:adaptive-prototype` 등)은 사용자가 쓰므로 **스크립트 이름은 그대로** 두고 내용만 새 경로로.
- 이름이 바뀐 모듈의 맨 위 설명(docstring/주석)에서 "synthetic-only", "no PostgreSQL" 같은 지금과 다른 설명을 실제 역할로 고친다(설명 정리는 다음 단계에서도 하지만, 이름과 설명이 어긋나지 않게 여기서 최소한 맞춘다).
- 역사 문서(`docs/prompts/`, 날짜별 결과 문서, 계획 문서의 지난 변경 이력·ADR 본문)는 고치지 않는다(그 시점 기록).

## 확인

- `git grep`으로 옛 이름(`synthetic_prototype`, `synthetic_registry`, `adaptive-prototype/`, `AdaptivePrototypePage`, `test_correction`)이 코드·설정·현재 안내 문서(README·팀원 안내서·demo-procedure)에 남지 않았는지 확인한다. 역사 문서와 URL 경로 `/adaptive-prototype`는 예외로 목록에 적는다.
- `npx tsc -b`, `npx eslint src`, `node --test tests/frontend/*.test.ts`(EPERM이면 `--test-isolation=none`), `npm run build`.
- 백엔드: 바뀐 모듈을 import하는 테스트 파일 전부 + `python -m compileall -q backend`. 전체 pytest는 Claude가 마지막에 한 번 돌린다. pytest는 `--basetemp`를 git-ignored 위치로.
- 사용자 서버(8000·8123·5173)·데이터 안 건드림, live 모델 호출 없음, 커밋 없음, `backend/.env` 값 출력 없음. UTF-8로 쓴다. 파일 이동은 `git mv`로.
- 끝나면 바꾼 이름 표(옛 → 새, 영향 파일 수), 남겨 둔 옛 이름 예외 목록, 테스트 결과를 짧게 적는다.

---

<a id="prompts-phase9-28-onboarding-docs"></a>

## 원본: prompts/phase9-28-onboarding-docs.md

# Phase 9-28 지침: 팀원 첫 실행·읽기용 문서와 주석 정리 (A, 문서·주석만)

## 배경 (2026-10-17 사용자 결정)

9-27에서 이름을 정리했다(`api/app_factory.py`, `api/database_registry.py`, `src/adaptive/`, `backend/scripts/demo_ner_correction.py` 등 — 실제 결과는 git 상태로 확인). 이제 문서와 주석을 현재 코드에 맞춘다. 코드 동작은 바꾸지 않는다.

## 바꿀 것

1. **`README.md` 새로 쓰기** (한국어, 200줄 이내):
   - 이 저장소가 무엇인지 3~4줄: 레거시 Text-to-SQL 챗봇(기준선, 미세 조정 모델, 포트 8000)과 스키마 적응형 QueryIR 시스템(관리자가 DB 등록·mapping 검토·게시, 챗봇이 공통 항목으로 질문 해석, 포트 8123). 흐름 그림(텍스트) 한 개.
   - 준비: Python 버전(`env_check` 기준), Node 버전, Ollama와 필요한 모델 이름(`qwen2.5-coder:7b-instruct`, 레거시 모델은 `models/Modelfile` 저장소 상대 경로), PostgreSQL.
   - 설치: `.venv` 만들기, 그 Python으로 `backend/requirements.txt`, requirements에 없는 것(예: `torch`, `pytest`)이 있으면 명시, `npm install`. Windows 기준 명령 + 다른 OS 대안 한 줄.
   - 설정: `backend/.env.example`을 복사해 `backend/.env`를 만들고 값 채우기. 변수 표(이름·필수 여부·용도만, 값 없음): `DATABASE_URL`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `CHAT_MODEL`, `MAPPING_MODEL`, `SHOW_SQL`, `REDIS_URL`(쓰면), `OPENAI_API_KEY`·`OPENAI_MODEL`(선택, 유료·외부 전송), `PG_SOURCE_<이름>`. 실제 쓰는 변수 이름은 `api/dev_launcher.py`와 레거시 백엔드 코드에서 확인.
   - 실행: `npm run dev:all` → 5173(화면), 8000(레거시), 8123(적응형). 서버 시작 전 `python -m backend.adaptive.env_check`(OK/WARN/FAIL 의미).
   - 사용 흐름 한 단락: 화면 "DB 변경" → 등록 → 검토·게시 → 사용할 DB 선택 → 챗봇. 자세한 건 `docs/teammate-evaluation-db-guide.md`.
   - 테스트: `.venv\Scripts\python.exe -m pytest tests -q`(약 40분, 임시 폴더 권한 오류 시 `--basetemp`), `node --test tests/frontend/*.test.ts`, `npx tsc -b`, `npm run build`. `npm run test:adaptive-prototype`은 일부만 돈다는 점.
   - 코드 둘러보기: 주요 폴더 표(`backend/`(레거시), `backend/adaptive/{api,registration,mapping,ontology,compiler,policy,harness,evaluation,catalog,relations}`, `src/adaptive/`, `src/pages/MainPage.tsx`, `tests/`, `docs/`) 각 한 줄.
   - Docker: 지금 compose는 레거시용이라는 점(실제 compose 내용을 확인해 사실대로).
   - 기존 README의 틀린 내용(개인 PC 절대 경로, `v_courses_info` 오타 등)은 고치거나 뺀다. 레거시 DB 구성 설명이 필요하면 사실만 짧게.
2. **`backend/.env.example` 추가**: 위 변수 이름과 한 줄 주석만, 값은 비우거나 안전한 예시(`OLLAMA_BASE_URL=http://127.0.0.1:11434` 같은 비밀 아닌 기본값만). DSN·키 예시에 실제 값처럼 보이는 문자열을 넣지 않는다. `.gitignore`가 `.env.example`을 무시하지 않는지 확인.
3. **`docs/README.md` 추가** (문서 목차): 처음 읽을 순서(루트 README → 계획 문서의 개요·현재 범위·구조 절 → `teammate-evaluation-db-guide.md` → `demo-procedure.md`), 평가 담당 문서(`human-validation-checkpoints.md`, `independent-evaluation-set-design.md` 등 실제 있는 것), `docs/prompts/`와 날짜별 결과 문서는 "작업 기록"이라는 설명, 9-27의 옛 이름 → 새 이름 표(역사 문서에 옛 이름이 나오므로).
4. **`docs/development-history.md#next-session-handoff` 맨 위**에 사람용 요약 절(현재 기능 / 실행·읽기 경로 / 남은 사람 작업 / 알려진 한계, 10줄 이내)을 더한다. 아래 기존 기록은 그대로.
5. **`docs/demo-procedure.md` 갱신**: 현재 화면(탭 3개, AI 추천 모두 승인, 표 훑어보기, 값 사전 가져오기, 시험 질의, 삭제)과 ontology v3(개설 연도 없음)에 맞춘다. 이전 관찰 수치는 "당시 관찰값"으로 표시.
6. **모듈 설명(주석)**:
   - 설명이 없는 핵심 모듈에 2~4줄: `backend/adaptive/mapping/{contracts,rules,review,repository,bridge,cli}.py`, `backend/adaptive/ontology/{contracts,loader}.py`(입력·출력·책임, 제안→검토→게시 흐름, ontology 버전별 활성 항목).
   - 지금 동작과 어긋나는 설명 고침: `api/chat_query.py` 맨 위("SQL 없음" ↔ 개발 옵션 SQL 표시), `api/dev_launcher.py` 맨 위(세 값만 재사용·같은 모델 → 현재 모델 선택지), `api/__init__.py`·`evaluation/__init__.py` 소개(초기 합성 기능만 설명), `api/postgres_runtime.py` "Phase 8-21: period numbers" 주석에 목적 한 줄, `src/adaptive/{contracts,state,mappingReview,postgresRegistration}.ts` 맨 위 역할 한 줄.
   - Phase 번호 주석은 지우지 않는다(이력 추적용). 기능 설명이 빠진 곳에만 설명을 더한다.
7. `docs/schema-adaptive-text-to-sql-plan.md` 변경 이력 2026-10-17 "9-27 이름 정리, 9-28 팀원용 문서·주석 정리" 한 줄씩, 구조 절에 모듈 이름이 나오면 새 이름으로.

## 확인과 지킬 것

- 코드 동작 변경 없음(주석·docstring·문서만). `python -m compileall -q backend`, `npx tsc -b`, `npx eslint src`가 통과해야 한다.
- README의 명령·변수·포트·파일 경로는 모두 실제 코드·설정과 대조해 맞는 것만 쓴다. 확인 못 한 내용은 쓰지 않는다.
- 비밀 값·DSN·키를 문서에 쓰지 않는다. `backend/.env`를 열어 값을 보지 않는다(변수 이름은 코드에서 확인).
- 서버·데이터 안 건드림, 모델 호출 없음, 커밋 없음. UTF-8로 쓴다.
- 끝나면 바꾼 파일과 확인 결과를 짧게 적는다.

---

<a id="synthetic-clarification-prototype"></a>

## 원본: synthetic-clarification-prototype.md

# 합성 Clarification 프로토타입 재현 절차

합성 전용 로컬 ASGI 프로토타입이다. 배포용이 아니며 인증이 없다.

## 안전 실행

```bash
.venv/Scripts/python.exe -m backend.adaptive.api.app_factory --port 8123
```

- host는 `127.0.0.1` 고정이며 외부 bind 옵션이 없다.
- port는 `8000-8999` 정수만 허용된다. reload와 access log는 꺼져 있다.
- 실제 DB URL, 모델 서버 설정, 환경 파일을 읽지 않는다.

## React 합성 화면

backend와 Vite를 loopback에서 함께 실행한다.

```powershell
npm run dev:adaptive-full
```

브라우저에서 `http://127.0.0.1:5173/adaptive-prototype`을 연다. 다른 경로는 기존 legacy 화면을 유지한다. 새 화면은 `/api/adaptive` proxy만 사용하며 legacy `/api/v1`이나 로그인 token을 사용하지 않는다.

Phase 6-15에서 실제 Chrome DOM·network browser smoke를 완료했다. 합성 내부 팀 논의용 준비 상태이며 외부 시연 완료나 최종 사용자 검증(L4)으로 표시하지 않는다.

팀 공유 실행 절차와 browser smoke 표는 [`docs/development-history.md#synthetic-clarification-team-handoff`](development-history.md#synthetic-clarification-team-handoff)를 사용한다.

## 고정 Database ID

```text
synth-clarification-prototype
```

이 ID 외의 database ID는 실패로 닫힌다.

## Start 예시

```json
{
  "request_id": "req-proto-001",
  "database_id": "synth-clarification-prototype",
  "question": "강의 목록 보여줘"
}
```

`POST /api/adaptive/v1/clarifications/start` 로 전송한다.
질문에는 `강의` 트리거가 포함되어야 `PENDING` 과 2개 option이 반환된다.

## Submit 예시

start 응답에서 받은 실제 값을 그대로 사용한다.

```json
{
  "session_token": "<start 응답의 session_token>",
  "selection": ["<선택한 option의 option_id>"],
  "cancelled": false
}
```

`POST /api/adaptive/v1/clarifications/submit` 로 전송한다.
`session_token` 을 손으로 만들거나 다른 세션 값을 재사용하지 않는다.

## 경계

- 합성 fixture(`DEFAULT_SEED` 기반 임시 SQLite)만 사용한다. 로컬 전용이다.
- 인증, rate limit, 영속 세션이 없다. 메모리 TTL/at-most-once 한계가 그대로 있다.
- 응답은 Phase 6-12 redaction 계약을 유지한다. SQL, 토큰 재사용 값, 임시 경로, 질문 원문은 응답에 포함되지 않는다.
- 결과는 합성 읽기 전용 재현용이며 모델 정확도나 실제 데이터 검증으로 해석하지 않는다.

---

<a id="synthetic-clarification-team-handoff"></a>

## 원본: synthetic-clarification-team-handoff.md

# 합성 Prototype 팀 공유 Handoff (Phase 6-15)

Phase 6-14 `/adaptive-prototype` 화면과 Phase 6-13 loopback ASGI backend를 함께 실행하는 팀 공유용 내부 prototype 정리다. 합성 fixture 전용이며 legacy 교체, 실제 서비스 연결, 외부 시연 승인이 아니다.

## 사전 조건

- 저장소 루트에서 실행한다.
- Python 가상환경 `.venv`와 npm 의존성이 준비되어 있어야 한다.
- `8123`과 `5173` loopback port가 비어 있어야 한다.
- 실제 CNU 데이터, `.env`, PostgreSQL, Ollama, private recording이 필요하지 않으며 사용하지 않는다.

## legacy와 함께 전체 실행 (2026-09-29 추가)

팀 논의에서 기존 legacy 화면과 스키마 적응 기능을 한 화면에서 함께 보여줄 때 쓴다.

```powershell
npm run dev:all
```

- legacy backend(`:8000`, venv Python), 합성 prototype backend(`:8123`), Vite frontend(`:5173`)를 함께 실행하고 `Ctrl+C`로 함께 종료한다.
- legacy 쪽은 기존처럼 `backend/.env`의 PostgreSQL·Ollama 설정이 필요하다(Redis는 없으면 cache 없이 동작).
- 접속: `http://localhost:5173/` (legacy 목록·AI Course Assistant, `DB 변경`의 합성 DB 선택·SQLite 업로드), `합성 질의 열기` → `/adaptive-prototype`.
- 2026-09-29 기계적 browser 확인: legacy 목록 200건, legacy assistant 질의 응답("SQL 실행 완료: 50건"), `DB 변경` → 합성 flat 6건 목록 전환, `강의 목록 보여줘` → 두 의미 선택 → 강의명 6행. SQLite 업로드 이후 관리자 흐름은 browser로 확인하지 않았다.
- QueryIR 챗봇 비교(Phase 7-14): `dev:all`은 `backend/.env`의 DB·Ollama 설정을 재사용해 `Legacy DB (QueryIR)`를 켠다. 시작 로그의 `adaptive dev launcher: legacy-view=..., ollama=...` 줄에서 상태를 확인한다. legacy-view profile은 한 번만 만들면 되고, 명령은 `python -m backend.adaptive.api.legacy_view_profile --write --live`이며 DSN 환경변수가 필요하다. 화면에서 `DB 변경` → `Legacy DB (QueryIR)` → AI Course Assistant로 질문하면, QueryIR 경로 답과 "이해한 조건"이 보인다. `Legacy 기준선`으로 바꾸면 같은 질문을 기존 SQL 경로로 볼 수 있다. 합성 SQLite DB에서는 챗봇이 미지원으로 안내된다.
- 사용자 확인(2026-09-29): legacy DB는 충남대 수강신청 테이블을 그대로 가져온 것이 아니라 팀이 유사한 스키마·데이터로 직접 구성한 것이다. 따라서 이 전체 실행은 실제 개인정보 노출을 전제로 하지 않는다.

## 실행

단일 명령:

```powershell
npm run dev:adaptive-full
```

접속 URL:

```text
http://127.0.0.1:5173/adaptive-prototype
```

위 명령은 합성 prototype backend(port `8123`, host `127.0.0.1` 고정)와 Vite frontend를 함께 실행한다. 개별 실행이 필요할 때만 아래 명령을 사용한다.

```powershell
npm run dev:adaptive-prototype
npm run dev:frontend
```

## 종료 방법

- 실행 터미널에서 `Ctrl+C`를 누른다.
- `dev:adaptive-full`은 `concurrently -k`로 두 프로세스를 함께 종료한다.

## 브라우저 smoke 시나리오

2026-09-27 설치된 Chrome headless의 실제 DOM·network 경로로 관찰했다. 이는 기계적 browser smoke이며 사람의 시각·사용성 검토(L4)는 아니다.

| # | 시나리오 | 기대 관찰 | 상태 |
| - | -------- | --------- | ---- |
| 1 | `강의 목록 보여줘` 시작 | `강의명 기준`·`교수자명 기준` 두 option 표시 | `PASS_BROWSER_SMOKE` |
| 2 | `강의명 기준` 선택 후 제출 | 합성 강의명 결과 6행 표시 | `PASS_BROWSER_SMOKE` |
| 3 | 새 흐름에서 `교수자명 기준` 선택 후 제출 | 합성 교수자명 결과 6행 표시 | `PASS_BROWSER_SMOKE` |
| 4 | 명시적 취소 | `취소되었습니다` 종료 안내 표시 | `PASS_BROWSER_SMOKE` |
| 5 | trigger 없는 질문 | 추가 확인 불필요 안내 표시 | `PASS_BROWSER_SMOKE` |
| 6 | backend 연결 실패 | 안전한 일반 오류 표시; token, SQL, 경로, endpoint, 원시 응답 노출 없음 | `PASS_BROWSER_SMOKE` |
| 7 | terminal 상태 후 같은 session 재제출 시도 | option·제출·취소 control 제거로 재제출 불가 | `PASS_BROWSER_SMOKE` |
| 8 | `/` 경로 접속 | 기존 `Course Registration` 화면 유지 | `PASS_BROWSER_SMOKE` |

browser smoke는 기본 Vite proxy와, adaptive target만 사용하지 않는 loopback port로 바꾼 별도 Vite process를 사용해 연결 실패도 재현했다. backend token at-most-once의 직접 `409` 계약은 기존 자동 테스트(`tests/adaptive/test_synthetic_clarification_prototype.py`)가 담당하고, browser에서는 terminal UI가 같은 session의 제출 control을 제거하는 것을 확인했다.

## 공유 계약과 안전 경계

- 합성 전용: 고정 database ID `synth-clarification-prototype`와 `DEFAULT_SEED` 기반 임시 SQLite만 사용한다.
- loopback-only: backend host는 `127.0.0.1` 고정이며 외부 bind 옵션이 없고, port는 `8000-8999`만 허용된다.
- 무인증: 로그인 token, 계정, 권한, rate limit이 없다.
- 비영속 session: 메모리 session이며 TTL/at-most-once 한계가 있다.
- 응답 redaction: SQL, session token 재사용 값, 임시 경로, 질문 원문, 원시 응답을 UI와 응답에 노출하지 않는다.
- frontend는 `/api/adaptive` proxy만 사용하며 legacy `/api/v1`과 인증·storage를 사용하지 않는다.
- 결과는 합성 읽기 전용 재현용이며 모델 정확도나 실제 데이터 검증이 아니다.

## 알려진 한계

- 실제 화면의 시각 품질·용어 이해·사용자 행동은 L4 사람 검토 전이다.
- 장시간 서버 운용, 영속 session, 인증/rate limit, 실제 Ollama/PostgreSQL/CNU 자료, 성능 측정은 범위 밖이다.
- `E3/L5`와 `L1/L3` 사람 관문 결정 없이 실제 데이터 사용·공유, 배포·외부 시연, 최종 성능 주장을 하지 않는다.

---

<a id="team-discussion-brief-2026-10-06"></a>

## 원본: team-discussion-brief-2026-10-06.md

# 팀 논의 자료: 지금까지 만든 것과 팀이 할 일 (2026-10-06)

> AI가 저장소 문서를 모아 정리했다. 결정이나 승인을 담지 않는다. 아래 숫자는 모두 개발용 잠정치다(4절).
> 근거 문서: [schema-adaptive-text-to-sql-plan.md](schema-adaptive-text-to-sql-plan.md), [human-validation-checkpoints.md](human-validation-checkpoints.md), 각 단계 결과 문서.

## 1. 한눈에

- **목표:** 학교마다 다른 수강편람 DB에, 모델을 다시 학습하지 않고 연결되는 수강 챗봇.
- **범위(2026-10-02 사용자 결정, ADR-089):** PostgreSQL 전용. 다른 PostgreSQL DB에 적응하는지만 본다.
- **데이터(2026-10-01 사용자 결정, ADR-086):** 실제 다른 학교 DB는 받지 않는다. 다음 둘만 쓴다.
  - 레거시 DB: 팀이 공개 수강편람으로 만든 DB.
  - 흉내 낸 DB: 다른 학교 공개 정보를 참고해 만든 DB.
- **상태:** 관리자 등록부터 게시, 챗봇, 평가 도구까지 만들었다. 남은 큰일은 사람 검토(L1·L3·L4)와 팀원의 평가용 DB다.

## 2. 지금 되는 것

### 관리자: DB 하나를 챗봇에 연결하기

1. **등록.** 서버에 등록된 연결에서 schema를 고른다. 로컬 AI(`qwen2.5-coder:7b-instruct`)와 규칙이 공통 항목 16개(과목 코드, 과목명, 학점, 이수 구분, 대상 학년, 학과, 교원명, 연도, 학기, 분반, 정원, 수강 인원, 요일, 시작·종료 시간 등)마다 맞는 column을 제안한다.
2. **검토.** 항목마다 승인, 다시 지정, "이 DB에 없음" 중 하나를 고른다. 그 DB를 만든 사람이 검토자다. 이 검토가 곧 의미 검토(L2)다.
3. **값 사전.**
   - 약어: `소웨` → 소프트웨어학과.
   - 코드의 뜻: `11` → 전공필수.
   - 교시표: `1` → `09:00`.
   - 코드표 table이 있으면 "코드표에서 가져오기"로 한꺼번에 넣는다.
4. **확장 항목.** 공통 항목에 없는 column(강의실, 수업 방식 등)을 그 DB만의 항목으로 선언한다.
5. **게시.** 실제 DB의 구조·저장 값과 맞는지 검사를 통과해야 챗봇에 적용된다. 게시마다 버전이 남는다. 게시 전에 고정 질의를 실행해 보는 기능은 아직 없다(SQLite에만 있음).

### 학생: 챗봇

- 질문 하나에 모델 호출은 한 번이다. 모델은 SQL을 직접 쓰지 않고 "어떤 조건으로 무엇을 보여 줄지"를 정해진 구조(QueryIR)로 낸다.
- 그 구조를 검사한 뒤 프로그램이 SQL을 만들고, 모양을 다시 검사하고, 읽기 전용으로 실행한다.
- 화면에 "이해한 조건"이 카드로 나온다. 개발 옵션을 켜면 실행 SQL도 보인다.
- 그 DB에 없는 정보는 "이 DB에는 강의실 정보가 없습니다"처럼 거절한다.
- 결과가 0건이면 어느 조건에 맞는 값이 없는지 알려 준다. 같음·포함 조건이 4개 이하일 때만 하고, 범위 조건("3학점 이상" 등)은 보지 않는다.

### 평가 도구

- **팀원 DB 평가:** `python -m backend.adaptive.evaluation.blind_eval check|run|report <평가 ID>`.
  - 첫 실행 때 질문 파일이 잠긴다.
  - 첫 결과는 덮어쓰지 않는다.
- **L1 판정 기록:** `python -m backend.adaptive.evaluation.l1_review record <CSV>`.
  - 판정 형식만 확인한다.
  - AI가 판정하거나 승인 상태를 바꾸지 않는다.

## 3. 확인한 DB 형식

레거시 DB를 뺀 다섯 DB는 모두 AI가 만든 개발용 합성 DB다. 팀원 평가 DB를 대신하지 않는다.

| DB | 형식 | 결과(잠정) | 결과 문서 |
| --- | --- | --- | --- |
| 레거시 DB | 팀이 만든 기존 DB | 30문항 비교: 기존 시스템 21, 새 방식 22. 모르는 조건 처리 9/9 | phase8-13, phase8-20 |
| `other_school` | 여러 table, 로마자 이름, 코드형 학기 | 11/12 | phase8-17 |
| `wide_school` | 넓은 단일 table | 11/12 | phase8-21 wide-table |
| `code_school` | 공통코드 table(코드만 저장) | 14/14 | phase8-21 code-table |
| `period_school` | 교시 번호 시간표 | 11/13 | phase8-21 period |
| `packed_school` | 한 칸에 묶인 시간표(`월1-2,수3`) | 13/13. 시간 질문 2개는 기대대로 거절 | phase8-21 packed |

- AI mapping 제안: 위 합성 schema 5개에서 맞힌 항목이 55/80에서 75/80이 되었다(phase8-23).
- 코드표 가져오기: `code_school`에서 코드 관련 질문 8/8(phase8-24).

## 4. 숫자를 읽는 법

- **질문과 정답을 개발자 쪽(AI 포함)이 만들었다.** 아직 사람이 검토하지 않았다(L1 전).
- **합성 DB도 AI가 만들었다.** 문제를 낸 쪽과 푼 쪽이 같아서, "다른 학교 DB에 적응한다"는 근거가 될 수 없다.
- **모델 답이 흔들린다.** 같은 질문도 실행마다 다를 수 있다.
  - "금요일 오전 수업"에서 "오전"을 빼먹는다.
  - "3시 이후 시작"을 종료 시간 조건으로 이해한다.
  - 이 두 시간 표현 문제는 8-25에서 규칙으로 보완했다(아래 6절 5).
- **논문에 쓸 수 있는 숫자는 따로 나온다.** 팀원 평가 DB의 첫 실행 결과를 L1·L3 검토한 뒤의 숫자다. 논문 한계에는 "모든 평가 데이터는 모사 데이터"라고 적는다.

## 5. 팀에게 부탁할 일

| 일 | 누가 | 할 일 | 자료 |
| --- | --- | --- | --- |
| L1 판정 | 사용자 먼저, 그다음 팀원 | 합성 benchmark 질문 70개의 뜻과 정답을 보고 승인·수정·보류를 적는다. | [l1-review/fixture-v1-review.md](l1-review/fixture-v1-review.md)(읽기), 같은 이름의 `.csv`(입력) |
| 평가용 DB 2~3개 | 팀원 | 다른 대학 공개 수강편람을 참고해 DB를 흉내 내어 만든다. 질문과 정답을 먼저 고정한다. | [teammate-evaluation-db-guide.md](teammate-evaluation-db-guide.md) |
| L3 | 사용자, 팀원 | 평가 실행 결과(맞음·틀림 판정)를 검토한다. | 평가 DB 실행 뒤 |
| L4 | 팀이 섭외 | 개발에 참여하지 않은 사람 몇 명이 관리자 화면과 챗봇을 써 본다. | 진행 자료는 준비 예정 |

평가용 DB를 만들 때 지킬 것(ADR-086):

1. 학교명·교수명은 지어낸다.
2. 질문과 정답은 DB를 만든 사람이 먼저 고정한다.
3. 개발자(AI 포함)는 평가 전까지 그 DB와 질문을 보지 않는다. 질문 파일을 채팅이나 저장소에 올리지 않는다.
4. 첫 실행 결과를 그대로 기록하고, 그 뒤 고친 결과는 따로 표시한다.

## 6. 팀이 정할 것

1. **평가 DB 분담.** 누가 어느 학교를 언제까지 만들지 정한다. 이번에 확인한 네 형식 밖의 형식이 섞일수록 평가 가치가 크다.
2. **묶인 시간표의 시간 해석(8-21 C ②).** 지금은 `월1-2,수3` 같은 칸에서 요일만 찾고, 시간 질문은 "이 DB에 없음"으로 거절한다. 문자열을 해석해 시간까지 꺼낼지 정한다. 꺼내면 답할 수 있는 질문이 늘지만, 학교마다 표기가 달라 잘못 해석할 위험이 있다.
3. **공통 항목(공통 ontology).**
   - 대상 학년 추가(버전 2)는 사용자가 이미 확정했다. 팀에 공유한다.
   - 공통 항목을 더할지 정한다. 지금 강의실·여석 같은 것은 DB별 확장 항목이거나 "이 DB에 없음"이다.
4. **공통 항목 밖 개념 목록.** 챗봇이 "이 DB에 없음"으로 알아듣는 개념 11개(강의실, 수업 방식, 단과대학, 강의 언어, 성적 평가 방식, 시험, 캠퍼스, 선수 과목, 강의 평가, 여석, 강의계획서)에 더할 것이 있는지 본다.
5. **시간 표현 안정화를 L1 전에 할지.** 계획에는 "질문 이해 개선은 L1 질문 세트를 확보한 뒤"로 되어 있다. 개발용 질문에 맞춘 개선을 막으려는 원칙이다. "오전", "N시 이후 시작"은 학교와 무관한 일반 규칙이라 예외로 먼저 할 수 있다.
   - **2026-10-06 사용자 결정으로 진행했다(8-25, ADR-096).** 팀에는 공유만 하면 된다. 결과: `docs/experiment-results.md#phase8-25-time-expressions-results-2026-10-06`.
6. **L4 일정과 대상.** 몇 명을, 언제, 어떤 과제로 할지 정한다.

## 7. 지켜 온 안전 원칙

- 레거시 DB는 읽기만 한다. 원본 엑셀과 DB 덤프는 공개된 곳에 올리지 않는다.
- DB 접속 정보는 화면과 로그에 나오지 않는다.
- 모델이 낸 구조는 검사를 통과해야 SQL이 된다. SQL을 그대로 실행하는 우회 경로는 없다.
- 학교 밖에 공개하는 화면에서는 교수명을 가린다(L5).
