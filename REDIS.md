## 캐싱 레이어 로직 설명

  ### 목적

  동일하거나 의미적으로 유사한 자연어 질의에 대해 매번 LLM을 호출하지 않고, Redis 기반 벡터 캐시를 통해 이전에 생성된
  SQL을 재사용한다.
  이를 통해 응답 시간을 줄이고 LLM 호출 비용을 낮춘다.

  ---

  ### 전체 처리 흐름

  1. 사용자가 자연어 질의를 입력한다.
  2. NER 전처리를 통해 질의 내 주요 엔티티를 추출한다.
  3. 추출된 엔티티를 `<DEPARTMENT>`, `<CATEGORY>`, `<DAY>`, `<TIME>` 등으로 마스킹한다.
  4. 마스킹된 질의를 기준으로 Redis 벡터 캐시를 조회한다.
  5. 캐시 hit 시:
     - 저장된 masked SQL을 가져온다.
     - 현재 질의에서 추출한 실제 엔티티 값으로 SQL을 복원한다.
     - 필요한 후처리 SQL을 적용한다.
     - DB에 실행한다.
  6. 캐시 miss 시:
     - LLM을 호출해 SQL을 생성한다.
     - SQL 검증 및 실행을 수행한다.
     - 생성된 SQL을 masked SQL 형태로 변환해 Redis에 저장한다.

  ---


  ### 질의 마스킹

  질의 내 DB 조건으로 사용될 수 있는 엔티티는 NER 결과를 기반으로 마스킹한다.

  예시:

  ```text
  컴융 전핵 보여줘

  NER 결과:

  [
    { "text": "컴융", "label": "DEPARTMENT" },
    { "text": "전핵", "label": "CATEGORY" }
  ]

  masked query:

  <DEPARTMENT> <CATEGORY> 보여줘
 ```

  이 masked query를 캐시 검색의 기준으로 사용한다.

  ---

  ### SQL 마스킹

  LLM이 생성한 SQL도 캐시에 저장하기 전에 엔티티 값을 마스킹한다.

```
  예시 SQL:

  SELECT *
  FROM v_course_info
  WHERE dept_name = '공과대학 컴퓨터융합학부'
    AND category LIKE '%전공(핵심)%'
  LIMIT 50

  masked SQL:

  SELECT *
  FROM v_course_info
  WHERE dept_name = '<DEPARTMENT>'
    AND category LIKE '%<CATEGORY>%'
  LIMIT 50
```

  이렇게 저장하면 다른 학과/이수구분 질의에도 같은 SQL 구조를 재사용할 수 있다.

  ---

  ### 캐시 저장 구조

  Redis에는 두 종류의 key를 저장한다.

  #### Payload key

  캐시 hit 시 실제로 사용할 SQL 정보를 저장한다.

  sqlgen:payload:{cache_key}
```
  저장 내용:

  {
    "masked_query": "<DEPARTMENT> <CATEGORY> 보여줘",
    "masked_sql": "SELECT * FROM v_course_info WHERE dept_name = '<DEPARTMENT>' AND category LIKE '%<CATEGORY>%' LIMIT
    50",
    "mask_signature": "CATEGORY|DEPARTMENT"
  }
```

  #### Vector key

  벡터 검색 및 후보 조회를 위한 hash를 저장한다.

  sqlgen:vector:{cache_key}

  저장 내용:

  masked_query
  masked_sql
  cache_key
  mask_signature
  embedding

  ———

  ### Mask Signature

  캐시 후보를 좁히기 위해 masked query에 포함된 마스크 라벨 집합으로 signature를 만든다.

  예시:

  ```<DEPARTMENT> <CATEGORY> 보여줘```

  signature:

  CATEGORY|DEPARTMENT

  다른 종류의 조건을 가진 SQL이 잘못 매칭되지 않도록, 동일한 signature를 가진 캐시만 후보로 비교한다.

  ———


  ### 캐시 조회 로직

  1. 현재 질의를 masked query로 변환한다.
  2. masked query의 mask signature를 계산한다.
  3. Redis에서 동일한 signature를 가진 vector 문서를 조회한다.
  4. 각 후보의 embedding과 현재 masked query embedding의 cosine similarity를 계산한다.
  5. similarity가 threshold 이상인 후보 중 가장 높은 후보를 선택한다.

  기본 threshold:

  VECTOR_CACHE_THRESHOLD=0.9

  ———

  ### Redis 검색 방식

  
  1. Redis FT.SEARCH raw command로 동일 mask_signature 문서 조회
  2. 각 후보 hash에서 embedding 로드
  3. Python에서 cosine similarity 직접 계산


  ———

  ### 캐시 Hit 처리

  캐시 hit가 발생하면 저장된 masked SQL을 현재 질의의 실제 엔티티 값으로 복원한다.
```
  예시:

  masked SQL:

  WHERE dept_name = '<DEPARTMENT>'
    AND category LIKE '%<CATEGORY>%'

  현재 질의 엔티티:

  {
    "DEPARTMENT": "공과대학 컴퓨터융합학부",
    "CATEGORY": "전공(핵심)"
  }

  복원 SQL:

  WHERE dept_name = '공과대학 컴퓨터융합학부'
    AND category LIKE '%전공(핵심)%'
```
  ———


  ### 캐시 Miss 처리

  캐시 miss 시에는 기존 LLM 생성 흐름을 그대로 사용한다.

  1. LLM으로 SQL 생성
  2. SQL 검증
  3. 필요 시 후처리 SQL 적용
  4. DB 실행
  5. 기본 SQL을 masked SQL로 변환
  6. Redis vector cache에 저장

  ———

  ### 로그

  캐시 조회 시 다음 정보를 로그로 남긴다.
```
  {
    "cache_hit": true,
    "cache_type": "redis-vector",
    "query": "공과대학 컴퓨터융합학부 전공(핵심) 보여줘",
    "masked_query": "<DEPARTMENT> <CATEGORY> 보여줘",
    "lookup_ms": 123,
    "cache_key": "sqlgen:payload:...",
    "similarity": 1.0
  }
```

  캐시 후보 비교 로그에는 다음 정보가 포함된다.
```
  {
    "masked_query": "<DEPARTMENT> <CATEGORY> 보여줘",
    "mask_signature": "CATEGORY|DEPARTMENT",
    "threshold": 0.9,
    "candidates": [
      {
        "payload_masked_query": "<DEPARTMENT> <CATEGORY> 보여줘",
        "similarity": 1.0,
        "threshold_passed": true,
        "intent_matched": true,
        "signature_matched": true
      }
    ]
  }
```
  ———
### 문제점
동일한 개체명이 두개 이상 나올 경우엔, 마스킹된 sql 템플릿 각 파라미터에 어느 위치에 값을 삽입할지 판단할 수 없어 캐싱 대상에서 제외했음.
```
예:
자연어 질의문: 전심이랑 3시 전핵 보여줘 -> <CATEGORY>이랑 <TIME> <CATEGORY> 보여줘
sql 템플릿: SELECT DISTINCT v_course_info.subject_code, v_course_info.section FROM v_course_info WHERE (start_time = '<TIME>' AND category LIKE '%<CATEGORY>%') OR category LIKE '%<CATEGORY>%'
```
=> 각 CATEGORY 값이 어느 위치에 들어가야 할지 판별 불가능 
- 해결방안
1. 현행 유지
2. 경량화된 분류 모델을 활용하여 마스킹 자연어 질의문의 각 값을 sql 템플릿 파라미터에 어느 위치에 삽입할지 결정하는 로직 추가 

