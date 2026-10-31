# 문서 읽기 안내

## 처음 읽을 순서

1. [루트 README](../README.md): 준비·설치·설정·실행·코드 위치.
2. [프로젝트 계획](schema-adaptive-text-to-sql-plan.md): 2~3절 개요, 6절 목표·현재 범위, 7~8절 공통 개념·구조, 13절 구현 현황. 현재 PostgreSQL 전용 범위와 ontology v3는 이후 ADR·변경 이력도 함께 본다.
3. [팀원 평가 DB 안내](teammate-evaluation-db-guide.md): DB 등록·검토·게시와 팀원이 만드는 평가 DB 작업.
4. [시연 절차](demo-procedure.md): 현재 화면에서 진행하는 시연 순서와 한계.

설계와 결정, 로드맵, 현재 상태의 기준은 계획 문서다. 과거 인수인계는 [개발 기록](development-history.md#next-session-handoff)에 보존한다.

## 평가 담당 문서

| 문서 | 읽는 목적 |
|---|---|
| [사람 검토 기준](human-validation-checkpoints.md) | 사용자 결정과 남은 L1/L3·L4 등 사람 작업 구분 |
| [독립 평가 세트 설계](independent-evaluation-set-design.md) | 개발자가 보지 않은 학교형 DB·질문·정답의 평가 설계 |
| [팀원 평가 DB 안내](teammate-evaluation-db-guide.md) | DB와 질문 준비·고정·평가 실행·비용 기록 |
| [L1 검토 안내](l1-review/fixture-v1-review.md) | 사람이 gold 질문·정답을 검토하는 순서 |
| [L4 진행자 안내](l4-usability/facilitator-guide.md) | 개발 외부인 사용성 과제·관찰·설문 진행 |
| [모델 녹화 사람 검토](live-capture-human-review.md) | 녹화된 응답을 사람이 검토할 때의 기준 |

평가용 모사 DB·질문·정답은 팀원이 고정하며, agent는 평가 전 질문·정답을 보지 않는다. 사람 승인 상태는 사람이 결정한다.

## 문서 유지 원칙

2026-10-30에 Markdown 182개를 14개로 통합했다. 단계마다 새 문서를 만들지 않고 다음 위치를 갱신한다.

| 내용 | 갱신할 문서 |
|---|---|
| 설치·실행·테스트 | [루트 README](../README.md) |
| 설계·결정·로드맵·변경 이력 | [프로젝트 계획](schema-adaptive-text-to-sql-plan.md) |
| 과거 구현 지침·인수인계·팀 논의 | [개발 기록](development-history.md) |
| 실험 조건·관찰값·한계 | [결과 기록](experiment-results.md) |

통합 기록의 목차에는 옛 파일명과 고정 링크가 있다. 기존 지침·측정 조건·한계는 보존하며, 과거 지침을 새 작업 지시나 과거 결과를 현재 성능으로 해석하지 않는다. L1 검토 CSV와 L4 과제·설문 등 사람이 직접 사용하는 양식은 별도 파일로 유지한다.

테스트도 단계별 새 파일 대신 기능별 기존 파일을 확장한다. 프론트엔드는 `tests/frontend/`의 7개 기능별 모음, Python은 `tests/adaptive/`의 기능별 모음을 쓴다. 공유 fixture와 테스트 간 import가 있는 Python 모듈은 기존 경계를 유지한다. 테스트 명령은 루트 README와 같다.

## 이름 변경

Phase 9-27(2026-10-17)에서 다음 이름을 바꿨다. 역사 문서에는 옛 이름이 남아 있다.

| 옛 이름 | 현재 이름 |
|---|---|
| `backend/adaptive/api/synthetic_prototype.py` | `backend/adaptive/api/app_factory.py` |
| `backend/adaptive/api/synthetic_registry.py` | `backend/adaptive/api/database_registry.py` |
| `src/adaptive-prototype/` | `src/adaptive/` |
| `src/adaptive-prototype/AdaptivePrototypePage.tsx` | `src/adaptive/ClarificationDemoPage.tsx` |
| `backend/test_correction.py` | `backend/scripts/demo_ner_correction.py` |

`/adaptive-prototype` 브라우저 URL, HTTP 계약, npm 스크립트 이름은 유지된다. 소스 폴더 이름과 구분해서 읽는다.
