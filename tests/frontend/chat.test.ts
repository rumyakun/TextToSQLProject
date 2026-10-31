import test, { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { CHAT_QUERY_PATH, ChatQueryClientError, chatErrorMessage, chatRowsToLegacyRows, structuredTimeText, TIME_TBA_TEXT, answerWarning, chatTargetFor, describeAnswer, parseChatQueryResponse, postChatQuery, describeUnmatched, QUERYIR_TIME_GUIDE, CHAT_MODELS_PATH, CHAT_MODELS_VERSION, describeChatModel, fetchChatModels, parseChatModels, describeUnmatchedTerms } from '../../src/adaptive/chatQuery.ts'
import type { ChatPostLike, ChatGetLike } from '../../src/adaptive/chatQuery.ts'
import { LEGACY_SELECTION_ID } from '../../src/adaptive/databases.ts'
import { customChatModelErrorMessage, normalizeOllamaTag, parseCustomChatModels, builtinModelStatus, validateApiBaseUrl, validApiModel } from '../../src/adaptive/customChatModels.ts'

// phase7-14-chat-queryir
{
  const DB = 'legacy-view';
  const COLUMNS = [
    'course_code', 'course_name', 'credits', 'category', 'department_name', 'instructor_name',
    'section', 'capacity', 'enrolled', 'day', 'start_time', 'end_time',
  ];
  const ROW = {
    course_code: '0001-0001', course_name: '자료구조', credits: 3, category: '전공(핵심)',
    department_name: '컴퓨터학과', instructor_name: '홍길동', section: '01', capacity: 40,
    enrolled: 12, day: '월', start_time: '13:00', end_time: '14:15',
  };
  
  it('formats structured time labels and uses the shared Korean TBA sentinel', () => {
    assert.equal(structuredTimeText('월', '10:30', '12:00'), '월 10:30-12:00');
    assert.equal(structuredTimeText('월', null, '12:00'), TIME_TBA_TEXT);
    assert.equal(TIME_TBA_TEXT, '시간 미정');
    assert.equal(structuredTimeText('월', '10:00:00', '12:00:00'), '월 10:00-12:00');  // legacy chatbot rows
    const labels = [structuredTimeText('월', '10:30', '12:00'), structuredTimeText('수', '10:30', '12:00')];
    assert.equal(labels.join(', '), '월 10:30-12:00, 수 10:30-12:00');
    // Luna 9-35: the chat cards must actually use the helper (rowToCourse is not exported).
    const popup = readFileSync('src/components/ChatPopup.tsx', 'utf-8');
    assert.ok(popup.includes('structuredTimeText('));
    assert.ok(popup.includes('part === TIME_TBA_TEXT'));
    // Luna 9-35: the legacy caption separator survived an encoding slip as "?".
    assert.ok(popup.includes("'기본 DB · 기존 챗봇'"));
  });
  
  it('warns when no filters were understood and preserves unmatched-word notices', () => {
    const noConditions = parseChatQueryResponse(succeeded({ understood: { filters: [] } }), DB);
    assert.ok(noConditions);
    assert.equal(answerWarning(noConditions), '질문에서 조건을 찾지 못해 전체 강좌를 보여 줍니다. 학과·과목명·요일 같은 조건을 넣어 다시 물어보세요.');
  
    const both = parseChatQueryResponse(succeeded({ understood: { filters: [] }, unmatched_terms: ['컴공'] }), DB);
    assert.ok(both);
    assert.match(answerWarning(both), /질문에서 조건을 찾지 못해 전체 강좌를 보여 줍니다/);
    assert.match(answerWarning(both), /‘컴공’.*이해하지 못해 조건에서 뺐습니다/);
  });
  
  it('renders each selectable database row as a label', () => {
    const main = readFileSync('src/pages/MainPage.tsx', 'utf-8');
    assert.match(main, /registry\.map\(\(entry\) => \(\s*<label[\s\S]*?<input\s+type="radio"[\s\S]*?databaseLabel\(registry, entry\.database_id\)/);
    assert.match(main, /<label className="flex items-center gap-2 rounded-lg border border-slate-200 px-3 py-2 text-sm">\s*<input\s+type="radio"[\s\S]*?기본 DB/);
  });
  
  function succeeded(overrides: Record<string, unknown> = {}) {
    return {
      version: 'adaptive-chat-query-v1',
      database_id: DB,
      status: 'SUCCEEDED',
      columns: COLUMNS,
      rows: [ROW],
      row_count: 1,
      limit: 50,
      understood: { filters: [{ field: 'Course.credits', label: '학점', operator: 'equals', value: 3 }] },
      ...overrides,
    };
  }
  
  const REGISTRY = [
    { database_id: 'synth-clarification-prototype', label: 'Synthetic flat (2026 seed)', schema: 'flat' as const },
    { database_id: DB, label: 'Legacy DB (QueryIR)', schema: 'postgres' as const },
  ];
  
  describe('phase7-14 chat target', () => {
    it('routes legacy, postgres, and other databases', () => {
      assert.deepEqual(chatTargetFor(LEGACY_SELECTION_ID, REGISTRY), { kind: 'legacy' });
      // Phase 9-34: the legacy view is shown as the 기본 DB (its server label stays ASCII).
      assert.deepEqual(chatTargetFor(DB, REGISTRY), { kind: 'queryir', databaseId: DB, label: '기본 DB (적응형 챗봇)' });
      assert.equal(chatTargetFor('synth-clarification-prototype', REGISTRY).kind, 'unsupported');
      assert.equal(chatTargetFor('unknown-db', REGISTRY).kind, 'unsupported');
    });
  });
  
  describe('phase7-14 response parser', () => {
    it('accepts the three answer shapes', () => {
      const ok = parseChatQueryResponse(succeeded(), DB);
      assert.equal(ok?.status, 'SUCCEEDED');
      const unsupported = parseChatQueryResponse(
        { version: 'adaptive-chat-query-v1', database_id: DB, status: 'UNSUPPORTED',
          unsupported: [{ field: 'CourseOffering.year', label: '개설 연도' }] },
        DB,
      );
      assert.equal(unsupported?.status, 'UNSUPPORTED');
      const notUnderstood = parseChatQueryResponse(
        { version: 'adaptive-chat-query-v1', database_id: DB, status: 'NOT_UNDERSTOOD', failure_code: 'REFLECTED_INPUT' },
        DB,
      );
      assert.equal(notUnderstood?.status, 'NOT_UNDERSTOOD');
    });
  
    it('phase8-14: keeps the executed SQL only as an optional string on success', () => {
      const plain = parseChatQueryResponse(succeeded(), DB);
      assert.ok(plain?.status === 'SUCCEEDED' && plain.sql === undefined);
      const withSql = parseChatQueryResponse(succeeded({ sql: "SELECT 1 WHERE x = '3'" }), DB);
      assert.ok(withSql?.status === 'SUCCEEDED' && withSql.sql === "SELECT 1 WHERE x = '3'");
      for (const sql of ['', 3, null, 'x'.repeat(20001)]) {
        assert.equal(parseChatQueryResponse(succeeded({ sql }), DB), null);
      }
      assert.equal(
        parseChatQueryResponse(
          { version: 'adaptive-chat-query-v1', database_id: DB, status: 'NOT_UNDERSTOOD', failure_code: 'X', sql: 'SELECT 1' },
          DB,
        ),
        null,
      );
    });
  
    it('rejects physical columns and inconsistent answers', () => {
      const cases: Record<string, unknown>[] = [
        succeeded({ columns: [...COLUMNS.slice(0, -1), 'credit_hours'] }),
        succeeded({ rows: [{ ...ROW, credit_hours: '3/3/0' }] }),
        succeeded({ rows: [{ ...ROW, credits: Number.NaN }] }),
        succeeded({ row_count: 2 }),
        succeeded({ limit: 0, row_count: 1 }),
        succeeded({ database_id: 'other-db' }),
        succeeded({ version: 'adaptive-chat-query-v2' }),
        succeeded({ understood: { filters: [{ field: 'v_course_info.credit_hours', label: 'x', operator: 'equals', value: 3 }] } }),
        succeeded({ understood: { filters: [{ field: 'Course.credits', label: '학점', operator: 'like', value: 3 }] } }),
        succeeded({ columns: [...COLUMNS, 'course_name'] }),
        { version: 'adaptive-chat-query-v1', database_id: DB, status: 'UNSUPPORTED', unsupported: [] },
        { version: 'adaptive-chat-query-v1', database_id: DB, status: 'NOT_UNDERSTOOD', failure_code: 'boom text' },
        { version: 'adaptive-chat-query-v1', database_id: DB, status: 'OTHER' },
      ];
      for (const value of cases) {
        assert.equal(parseChatQueryResponse(value, DB), null, JSON.stringify(value).slice(0, 100));
      }
    });
  });
  
  describe('phase7-14 client', () => {
    it('posts only the database id and trimmed question', async () => {
      const calls: { url: string; body: unknown }[] = [];
      const postFn: ChatPostLike = async (url, init) => {
        calls.push({ url, body: JSON.parse(init.body) });
        return { ok: true, status: 200, json: async () => succeeded() };
      };
      const answer = await postChatQuery(postFn, DB, '  3학점 전공 과목  ');
      assert.equal(answer.status, 'SUCCEEDED');
      assert.deepEqual(calls, [{ url: CHAT_QUERY_PATH, body: { database_id: DB, query: '3학점 전공 과목' } }]);
      await assert.rejects(postChatQuery(postFn, DB, '   '), ChatQueryClientError);
      await assert.rejects(postChatQuery(postFn, DB, '가'.repeat(501)), ChatQueryClientError);
      await assert.rejects(postChatQuery(postFn, '../x', '질문'), ChatQueryClientError);
      await assert.rejects(postChatQuery(postFn, DB, '\u0000질문'), ChatQueryClientError);
      await assert.rejects(postChatQuery(postFn, DB, '질문\u001b'), ChatQueryClientError);
      assert.equal(calls.length, 1);
      await postChatQuery(postFn, DB, '줄바꿈\n질문');
      assert.equal(calls.length, 2);
    });
  
    it('maps server errors to fixed messages', async () => {
      const failing = (status: number, error: Record<string, unknown>): ChatPostLike =>
        async () => ({ ok: false, status, json: async () => ({ error }) });
      const cases: [ChatPostLike, RegExp][] = [
        [failing(503, { code: 'CHAT_MODEL_UNAVAILABLE' }), /Ollama/],
        [failing(503, { code: 'CHAT_MODEL_UNAVAILABLE', failure_code: 'TIMEOUT' }), /시간이 초과/],
        [failing(404, { code: 'CHAT_CONTEXT_UNAVAILABLE' }), /사용할 수 없습니다/],
        [failing(500, { code: 'CHAT_EXECUTION_FAILED', failure_code: 'EXECUTION_ERROR' }), /오류가 발생/],
      ];
      for (const [postFn, pattern] of cases) {
        await assert.rejects(postChatQuery(postFn, DB, '질문'), (error: unknown) => {
          assert.match(chatErrorMessage(error), pattern);
          return true;
        });
      }
      const transport: ChatPostLike = async () => {
        throw new TypeError('fetch failed');
      };
      await assert.rejects(postChatQuery(transport, DB, '질문'), (error: unknown) => {
        assert.match(chatErrorMessage(error), /개발 서버가 실행 중인지 확인해 주세요/);
        return true;
      });
    });
  });
  
  describe('phase7-14 rendering helpers', () => {
    it('renames card keys to the legacy row keys the course cards read', () => {
      const [row] = chatRowsToLegacyRows([ROW]);
      assert.equal(row['subject_code'], '0001-0001');
      assert.equal(row['subject_name'], '자료구조');
      assert.equal(row['credit_hours'], 3);
      assert.equal(row['dept_name'], '컴퓨터학과');
      assert.equal(row['professor'], '홍길동');
      assert.equal(row['day_of_week'], '월');
      assert.equal(row['section'], '01');
      assert.equal(row['start_time'], '13:00');
      assert.equal('course_name' in row, false);
    });
  
    it('describes answers without SQL', () => {
      const ok = parseChatQueryResponse(succeeded(), DB);
      assert.ok(ok);
      assert.match(describeAnswer(ok), /조회 완료: 1행\. 이해한 조건: 학점 = 3/);
      const capped = parseChatQueryResponse(
        succeeded({ rows: Array(50).fill(ROW), row_count: 50 }),
        DB,
      );
      assert.ok(capped);
      assert.match(describeAnswer(capped), /최대 50행까지만 조회되어 일부 강좌가 빠졌을 수 있습니다/);
      assert.match(describeAnswer(capped, { courseCount: 15 }), /조회 완료: 강좌 15개/);
      const merged = parseChatQueryResponse(
        succeeded({ rows: Array(17).fill(ROW), row_count: 17, limit: 100 }), DB,
      );
      assert.ok(merged);
      assert.match(describeAnswer(merged, { courseCount: 15 }), /조회 완료: 강좌 15개/);
      const belowLimit = parseChatQueryResponse(
        succeeded({ rows: Array(49).fill(ROW), row_count: 49 }), DB,
      );
      assert.ok(belowLimit);
      assert.doesNotMatch(describeAnswer(belowLimit), /일부 강좌가 빠졌을 수 있습니다/);
      assert.match(describeAnswer(capped), /일부 강좌가 빠졌을 수 있습니다/);
      assert.match(
        describeAnswer({ status: 'UNSUPPORTED', unsupported: [{ field: 'CourseOffering.semester', label: '학기' }] }),
        /학기 정보가 없어/,
      );
    });
  });
  
  describe('phase7-14 wiring', () => {
    it('ChatPopup routes QueryIR targets away from the legacy SQL endpoint', () => {
      const popup = readFileSync(new URL('../../src/components/ChatPopup.tsx', import.meta.url), 'utf8');
      assert.match(popup, /chatTarget\.kind === 'queryir'/);
      assert.match(popup, /postChatQuery\(/);
      assert.match(popup, /chatTarget\.kind === 'unsupported'/);
      const main = readFileSync(new URL('../../src/pages/MainPage.tsx', import.meta.url), 'utf8');
      assert.match(main, /chatTarget=\{chatTarget\}/);
      assert.match(main, /chatTargetFor\(activeDbId, registry\)/);
      // Only the dedicated partial-mapping code shows the chatbot hint; other errors keep retry.
      assert.match(main, /SYNTHETIC_CATALOG_UNSUPPORTED/);
      assert.match(main, /catalogOnlyViaChat = catalogError === CATALOG_VIA_CHAT_MESSAGE/);
      // Target changes invalidate in-flight answers synchronously, on both paths.
      assert.match(popup, /useLayoutEffect\(\(\) => \{\r?\n {4}if \(targetKeyRef\.current === targetKey\) return/);
      assert.equal((popup.match(/targetKeyRef\.current !== requestKey/g) ?? []).length >= 4, true);
    });
  });
}

// phase8-18-target-grade
{
  const base = {
    version: 'adaptive-chat-query-v1',
    database_id: 'other-school',
    status: 'SUCCEEDED',
    row_count: 1,
    limit: 200,
    understood: { filters: [{ field: 'Course.target_grade', label: '대상 학년', operator: 'equals', value: 1 }] },
  };
  
  describe('phase8-18 target grade in chat cards', () => {
    it('accepts the target_grade column when the database maps it', () => {
      const parsed = parseChatQueryResponse(
        { ...base, columns: ['course_code', 'target_grade'], rows: [{ course_code: 'SW1001', target_grade: 1 }] },
        'other-school',
      );
      assert.equal(parsed?.status, 'SUCCEEDED');
      assert.ok(parsed?.status === 'SUCCEEDED' && parsed.filters[0]?.label === '대상 학년');
    });
  
    it('still rejects unknown card columns', () => {
      const parsed = parseChatQueryResponse(
        { ...base, columns: ['course_code', 'classroom'], rows: [{ course_code: 'SW1001', classroom: 'A101' }] },
        'other-school',
      );
      assert.equal(parsed, null);
    });
  });
  
  describe('phase8-19 extension columns', () => {
    const answer = (extra: Record<string, unknown>) => ({
      ...base,
      columns: ['course_code', 'ext_classroom'],
      rows: [{ course_code: 'SW1001', ext_classroom: 'IT관 303호' }],
      ...extra,
    });
  
    it('accepts an ext_ column named in the extensions list', () => {
      const parsed = parseChatQueryResponse(answer({ extensions: [{ key: 'ext_classroom', label: '강의실' }] }), 'other-school');
      assert.ok(parsed?.status === 'SUCCEEDED' && parsed.extensions?.[0]?.label === '강의실');
    });
  
    it('rejects ext_ columns without a label, unnamed labels, and bad keys', () => {
      assert.equal(parseChatQueryResponse(answer({}), 'other-school'), null);
      assert.equal(
        parseChatQueryResponse(answer({ extensions: [{ key: 'ext_classroom', label: '강의실' }, { key: 'ext_other', label: '기타' }] }), 'other-school'),
        null,
      );
      assert.equal(parseChatQueryResponse(answer({ extensions: [{ key: 'classroom', label: '강의실' }] }), 'other-school'), null);
    });
  });
}

// phase8-20-unknown-conditions
{
  const DB = 'legacy-view';
  const base = { version: 'adaptive-chat-query-v1', database_id: DB };
  const empty = {
    ...base,
    status: 'SUCCEEDED',
    columns: ['course_name'],
    rows: [],
    row_count: 0,
    limit: 200,
    understood: { filters: [{ field: 'CourseOffering.section', label: '분반', operator: 'equals', value: '교111' }] },
  };
  const unmatched = [{ field: 'CourseOffering.section', label: '분반', value: '교111' }];
  
  describe('phase8-20 outside concepts', () => {
    it('accepts concept entries beside field entries', () => {
      const parsed = parseChatQueryResponse(
        {
          ...base,
          status: 'UNSUPPORTED',
          unsupported: [
            { field: 'CourseOffering.semester', label: '학기' },
            { concept: 'classroom', label: '강의실' },
          ],
        },
        DB,
      );
      assert.ok(parsed && parsed.status === 'UNSUPPORTED');
      assert.deepEqual(parsed.unsupported[1], { concept: 'classroom', label: '강의실' });
      assert.equal(describeAnswer(parsed), '이 DB에는 학기, 강의실 정보가 없어 답할 수 없습니다. 그 조건을 빼고 다시 물어보세요.');
    });
  
    it('rejects malformed concept entries', () => {
      for (const item of [
        { concept: 'Class Room', label: '강의실' },
        { concept: 'classroom', field: 'Course.name', label: '강의실' },
        { concept: 'classroom' },
        { concept: 'classroom', label: '' },
      ]) {
        assert.equal(parseChatQueryResponse({ ...base, status: 'UNSUPPORTED', unsupported: [item] }, DB), null);
      }
    });
  });
  
  describe('phase8-20 empty-result diagnosis', () => {
    it('explains an empty result', () => {
      const parsed = parseChatQueryResponse({ ...empty, unmatched }, DB);
      assert.ok(parsed && parsed.status === 'SUCCEEDED');
      assert.deepEqual(parsed.unmatched, unmatched);
      assert.match(describeAnswer(parsed), /0행.*분반에서 '교111' 값을 찾지 못했습니다/);
      assert.equal(describeUnmatched([]), '');
    });
  
    it('keeps the plain answer without a diagnosis', () => {
      const parsed = parseChatQueryResponse(empty, DB);
      assert.ok(parsed && parsed.status === 'SUCCEEDED');
      assert.equal(parsed.unmatched, undefined);
      assert.doesNotMatch(describeAnswer(parsed), /찾지 못했습니다/);
    });
  
    it('rejects a diagnosis on a non-empty result or in a bad shape', () => {
      const row = { course_name: '자료구조' };
      assert.equal(parseChatQueryResponse({ ...empty, rows: [row], row_count: 1, unmatched }, DB), null);
      assert.equal(parseChatQueryResponse({ ...empty, unmatched: [] }, DB), null);
      assert.equal(parseChatQueryResponse({ ...empty, unmatched: [{ ...unmatched[0], extra: 1 }] }, DB), null);
      assert.equal(parseChatQueryResponse({ ...empty, unmatched: [{ ...unmatched[0], field: 'section' }] }, DB), null);
    });
  
    it('ChatPopup shows the diagnosis as a warning', () => {
      const popup = readFileSync(new URL('../../src/components/ChatPopup.tsx', import.meta.url), 'utf8');
      assert.match(popup, /setLastWarning\(answerWarning\(answer\)\)/);
    });
  });
}

// phase8-25-time-guide
{
  const LEGACY_LINE = '시간은 24시간 형식으로 적어 주세요. 예: 오전 2시는 2시, 오후 2시는 14시.';
  
  describe('phase8-25 question guide on the QueryIR chat', () => {
    it('describes the closed time rules, not the legacy 24-hour advice', () => {
      assert.match(QUERYIR_TIME_GUIDE, /3시 → 15시/);
      assert.match(QUERYIR_TIME_GUIDE, /오후 3시 이후에 시작하는/);
      assert.doesNotMatch(QUERYIR_TIME_GUIDE, /오전 2시는 2시/);
    });
  
    it('only the QueryIR target shows it; the legacy screen keeps its line', () => {
      const popup = readFileSync(new URL('../../src/components/ChatPopup.tsx', import.meta.url), 'utf8');
      assert.match(popup, /chatTarget\.kind === 'queryir'\s*\?\s*QUERYIR_TIME_GUIDE\s*:\s*'시간은 24시간 형식으로/);
      assert.equal(popup.split(LEGACY_LINE).length, 2);
    });
  });
}

// phase9-08-chat-model-choice
{
  const LISTING = {
    version: CHAT_MODELS_VERSION,
    default: 'general',
    models: [
      { id: 'general', label: '일반 모델', tag: 'qwen2.5-coder:7b-instruct' },
      { id: 'legacy-tuned', label: '레거시 학습 모델(비교용)', tag: 'text2sql_qwen:latest' },
    ],
  };
  
  function withModels(models: unknown, extra: Record<string, unknown> = {}): unknown {
    return { ...LISTING, models, ...extra };
  }
  
  describe('Phase 9-08 chat model listing', () => {
    it('reads the listing, default first', () => {
      const listing = parseChatModels(LISTING);
      assert.deepEqual(listing, { defaultId: 'general', models: LISTING.models });
      assert.deepEqual(parseChatModels({ version: CHAT_MODELS_VERSION, default: null, models: [] }), {
        defaultId: null,
        models: [],
      });
      const single = { version: CHAT_MODELS_VERSION, default: 'default', models: [{ id: 'default', label: '기본 모델', tag: null }] };
      assert.equal(parseChatModels(single)?.models[0].tag, null);
    });
  
    it('refuses anything off the contract', () => {
      const bad: unknown[] = [
        null,
        [],
        { ...LISTING, version: 'v0' },
        { ...LISTING, extra: 1 },
        { ...LISTING, default: 'legacy-tuned' },
        { ...LISTING, default: null },
        { version: CHAT_MODELS_VERSION, default: 'general', models: [] },
        withModels([{ id: 'General', label: 'x', tag: null }], { default: 'General' }),
        withModels([{ id: 'general\n', label: 'x', tag: null }], { default: 'general\n' }),
        withModels([{ id: 'general', label: ' ', tag: null }]),
        withModels([{ id: 'general', label: 'x', tag: 'a b' }]),
        withModels([{ id: 'general', label: 'x', tag: null, url: 'http://127.0.0.1:11434' }]),
        withModels([LISTING.models[0], LISTING.models[0]]),
        // Phase 9-31: up to 7 choices (3 built-in + 4 added); 8 is off the contract.
        withModels(Array.from({ length: 8 }, (_unused, index) => ({ id: `m${index}`, label: 'x', tag: null })), {
          default: 'm0',
        }),
      ];
      for (const value of bad) assert.equal(parseChatModels(value), null, JSON.stringify(value));
    });
  
    it('fetches the listing and reports failures as client errors', async () => {
      const seen: string[] = [];
      const getFn: ChatGetLike = async (url, init) => {
        seen.push(`${init.method} ${url}`);
        return { ok: true, status: 200, json: async () => LISTING };
      };
      assert.equal((await fetchChatModels(getFn)).defaultId, 'general');
      assert.deepEqual(seen, [`GET ${CHAT_MODELS_PATH}`]);
      const failing: ChatGetLike = async () => ({ ok: false, status: 404, json: async () => ({}) });
      await assert.rejects(fetchChatModels(failing), ChatQueryClientError);
      const odd: ChatGetLike = async () => ({ ok: true, status: 200, json: async () => ({ models: [] }) });
      await assert.rejects(fetchChatModels(odd), ChatQueryClientError);
    });
  
    it('names a model with its tag when the server gives one', () => {
      assert.equal(describeChatModel(LISTING.models[1]), '레거시 학습 모델(비교용) · text2sql_qwen:latest');
      assert.equal(describeChatModel({ id: 'default', label: '기본 모델', tag: null }), '기본 모델');
    });
  });
  
  describe('Phase 9-08 chat request', () => {
    const bodies: unknown[] = [];
    const postFn: ChatPostLike = async (_url, init) => {
      bodies.push(JSON.parse(init.body));
      return { ok: false, status: 400, json: async () => ({ error: { code: 'CHAT_INVALID_REQUEST' } }) };
    };
  
    it('sends the chosen model id, and none when nothing was chosen', async () => {
      bodies.length = 0;
      await assert.rejects(postChatQuery(postFn, 'legacy-view', '질문', { model: 'legacy-tuned' }), ChatQueryClientError);
      await assert.rejects(postChatQuery(postFn, 'legacy-view', '질문'), ChatQueryClientError);
      assert.deepEqual(bodies, [
        { database_id: 'legacy-view', query: '질문', model: 'legacy-tuned' },
        { database_id: 'legacy-view', query: '질문' },
      ]);
    });
  
    it('never sends a model tag or other text as the model', async () => {
      bodies.length = 0;
      for (const model of ['text2sql_qwen:latest', '', 'General', 'a'.repeat(40), 'general\n']) {
        await assert.rejects(postChatQuery(postFn, 'legacy-view', '질문', { model }), (error: unknown) => {
          return error instanceof ChatQueryClientError && error.kind === 'contract';
        });
      }
      assert.deepEqual(bodies, []);
    });
  });
  
  describe('Phase 9-08 chat popup', () => {
    const source = readFileSync(new URL('../../src/components/ChatPopup.tsx', import.meta.url), 'utf8');
  
    it('shows the model picker only for the QueryIR chat with more than one choice', () => {
      assert.match(source, /isQueryIr && chatModels !== null && chatModels\.length > 1/);
      assert.match(source, /data-testid="chat-model"/);
      assert.match(source, /disabled=\{loading\}\s*\n\s*data-testid="chat-model"/);
    });
  
    it('sends the pick with each QueryIR question', () => {
      assert.match(source, /\.\.\.\(chatModel === null \? \{\} : \{ model: chatModel \}\)/);
    });
  });
  
  describe('Phase 9-09 model failure message', () => {
    it('names the ChatGPT API as well as Ollama', () => {
      const text = chatErrorMessage(new ChatQueryClientError('server', 503, 'CHAT_MODEL_UNAVAILABLE', 'PROVIDER_ERROR'));
      assert.match(text, /Ollama/);
      assert.match(text, /ChatGPT라면 API 키/);
    });
  });
}

// phase9-21-unmatched-word-notice
{
  const base = { version: 'adaptive-chat-query-v1', database_id: 'legacy-view' };
  const succeeded = {
    ...base, status: 'SUCCEEDED', columns: ['course_name'],
    rows: [{ course_name: '자료구조' }], row_count: 1, limit: 200,
    understood: { filters: [] },
  };
  
  it('accepts an optional notice on non-empty and empty results', () => {
    for (const body of [succeeded, { ...succeeded, rows: [], row_count: 0 }]) {
      const answer = parseChatQueryResponse({ ...body, unmatched_terms: ['컴융'] }, 'legacy-view');
      assert.ok(answer?.status === 'SUCCEEDED');
      assert.deepEqual(answer.unmatchedTerms, ['컴융']);
      const old = parseChatQueryResponse(body, 'legacy-view');
      assert.ok(old?.status === 'SUCCEEDED');
      assert.equal(old.unmatchedTerms, undefined);
    }
  });
  
  it('rejects malformed, duplicate, oversized and refusal notices', () => {
    for (const terms of [null, '컴융', [], ['가'], ['컴융', '컴융'], [42], ['컴융\n'], ['컴융😀'], ['a'.repeat(501)], ['aa', 'bb', 'cc', 'dd', 'ee', 'ff']]) {
      assert.equal(parseChatQueryResponse({ ...succeeded, unmatched_terms: terms }, 'legacy-view'), null);
    }
    for (const body of [
      { ...base, status: 'UNSUPPORTED', unsupported: [{ concept: 'classroom', label: '강의실' }] },
      { ...base, status: 'NOT_UNDERSTOOD', failure_code: 'UNKNOWN_FIELD' },
    ]) assert.equal(parseChatQueryResponse({ ...body, unmatched_terms: ['컴융'] }, 'legacy-view'), null);
  });
  
  it('preserves empty-result diagnosis beside word notices', () => {
    const answer = parseChatQueryResponse({ ...succeeded, rows: [], row_count: 0,
      unmatched_terms: ['컴융'], unmatched: [{ field: 'Course.category', label: '이수 구분', value: '전공' }],
    }, 'legacy-view');
    assert.ok(answer?.status === 'SUCCEEDED');
    assert.equal(answer.unmatched?.[0].value, '전공');
    assert.deepEqual(answer.unmatchedTerms, ['컴융']);
  });
  
  it('shows a brief dictionary notice above results and keeps the legacy path', () => {
    assert.equal(describeUnmatchedTerms([]), '');
    assert.equal(describeUnmatchedTerms(['컴융']), '‘컴융’을(를) 이해하지 못해 조건에서 뺐습니다. 값 사전에 줄임말을 넣으면 알아들을 수 있습니다.');
    assert.match(describeUnmatchedTerms(['컴융', '소웨']), /‘컴융’, ‘소웨’/);
    const popup = readFileSync(new URL('../../src/components/ChatPopup.tsx', import.meta.url), 'utf8');
    assert.match(popup, /setLastWarning\(answerWarning\(answer\)\)/);
    const answer = parseChatQueryResponse({ ...succeeded, unmatched_terms: ['컴융'] }, 'legacy-view');
    assert.ok(answer?.status === 'SUCCEEDED');
    assert.match(answerWarning(answer), /‘컴융’을\(를\) 이해하지 못해 조건에서 뺐습니다/);
    assert.ok(popup.indexOf('{lastWarning &&') < popup.indexOf('data-testid="chat-understood"'));
  });
}

// phase9-31-custom-chat-models
{
  const valid = {
    version: 'adaptive-custom-chat-models-v1', installed: ['qwen2.5:7b'], unavailable: false, builtin: [{id:'general',label:'일반 모델',tag:'base:latest',provider:'ollama',default:true,installed:true}], limit: 4,
    custom: [{ id: 'custom-qwen2-5-7b', label: '개인 모델', tag: 'qwen2.5:7b', mode: 'general', added_by: '관리자', added_at: '2026-10-27T00:00:00+00:00', active: true, provider: 'ollama', base_url: null, has_key: false }],
  }
  
  describe('phase9-31 custom chat model contracts', () => {
    it('normalizes Ollama tags and identifies builtin aliases for filtering', () => {
      assert.equal(normalizeOllamaTag('text2sql_qwen'), 'text2sql_qwen:latest')
      assert.equal(normalizeOllamaTag('qwen2.5-coder:7b-instruct'), 'qwen2.5-coder:7b-instruct')
      assert.equal(normalizeOllamaTag('hf.co/user/model'), 'hf.co/user/model:latest')
      assert.equal(normalizeOllamaTag('host:5000/ns/model'), 'host:5000/ns/model:latest')
      const builtinTags = ['text2sql_qwen']
      assert.equal(builtinTags.some((tag) => normalizeOllamaTag(tag) === normalizeOllamaTag('text2sql_qwen:latest')), true)
    })
  
    it('accepts a valid listing and rejects malformed entries', () => {
      assert.deepEqual(parseCustomChatModels(valid)?.custom[0], valid.custom[0])
      assert.equal(parseCustomChatModels({ ...valid, unknown: true }), null)
      assert.equal(parseCustomChatModels({ ...valid, custom: [{ ...valid.custom[0], mode: 'other' }] }), null)
      assert.equal(parseCustomChatModels({ ...valid, custom: [{ ...valid.custom[0], active: 'true' }] }), null)
      assert.equal(parseCustomChatModels({ ...valid, installed: ['bad tag'] }), null)
    })
  
    it('maps fixed server codes to Korean messages', () => {
      assert.match(customChatModelErrorMessage('NOT_INSTALLED'), /설치된 모델/)
      assert.match(customChatModelErrorMessage('LIMIT_REACHED'), /최대 4개/)
      assert.match(customChatModelErrorMessage('UNKNOWN'), /처리하지 못했습니다/)
    })
  })
}

// phase9-32-custom-chat-models
{
  test('Phase 9-32: OpenAI-compatible API base URL rule', () => {
    for (const url of ['https://api.example.com/v1', 'http://127.0.0.1:11434/v1', 'http://localhost:1234/v1', 'http://[::1]:1234/v1']) assert.equal(validateApiBaseUrl(url), true, url)
    for (const url of ['http://example.com/v1', 'https://user:pass@example.com/v1', 'https://example.com/v1?x=1', 'https://example.com/v1#frag']) assert.equal(validateApiBaseUrl(url), false, url)
  })
  
  test('Phase 9-32: strict API entry parsing and model-name validation', () => {
    assert.equal(validApiModel('meta-llama/Llama-3.1-8B-Instruct'), true)
    assert.equal(validApiModel('bad model'), false)
    assert.equal(validApiModel('model@v1'), false)
    for (const name of ['gpt-4o', 'meta-llama/Llama-3.1-8B-Instruct', 'model:v1']) assert.equal(validApiModel(name), true)
    const base = { version: 'adaptive-custom-chat-models-v1', installed: [], unavailable: false, builtin: [{id:'general',label:'일반 모델',tag:'qwen:latest',provider:'ollama',default:true,installed:true}], limit: 4 }
    const item = { id: 'custom-llama', label: 'Llama', tag: 'meta-llama/Llama-3.1-8B-Instruct', mode: 'general', added_by: 'reviewer', added_at: '2026-10-27T00:00:00+00:00', active: true, provider: 'openai-compatible', base_url: 'https://api.example.com/v1', has_key: true }
    assert.equal(parseCustomChatModels({ ...base, custom: [item] })?.custom[0]?.has_key, true)
    assert.equal(parseCustomChatModels({ ...base, custom: [{ ...item, unexpected: true }] }), null)
    assert.equal(parseCustomChatModels({ ...base, custom: [{ ...item, base_url: 'http://example.com/v1' }] }), null)
    assert.equal(parseCustomChatModels({ ...base, custom: [], builtin: [...base.builtin, {...base.builtin[0], id:'second', default:true}] }), null)
    assert.equal(parseCustomChatModels({ ...base, custom: [], builtin: [{...base.builtin[0], secret:'no'}] }), null)
    assert.equal(parseCustomChatModels({ ...base, custom: [], builtin: [{...base.builtin[0], installed:'yes'}] }), null)
  })
  
  test('Phase 9-33: builtin model status mapping', () => {
    assert.equal(builtinModelStatus({provider:'ollama',installed:true}), '설치됨')
    assert.equal(builtinModelStatus({provider:'ollama',installed:false}), 'Ollama에 없음')
    assert.equal(builtinModelStatus({provider:'ollama',installed:null}), '확인 불가')
    assert.equal(builtinModelStatus({provider:'openai',installed:null}), '외부 API')
  })
}
