import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { TEST_QUERIES_PATH, TEST_QUERY_PATH, TestQueryClientError, fetchTestQueryList, parseTestQueryList, parseTestQueryResult, runTestQuery, toTestQueryMessage, REGRESSION_PATH, VERIFIED_QUERIES_PATH, fetchRegression, parseRegression, parseVerifiedList, verifyTestQuery } from '../../src/adaptive/testQueries.ts'
import type { TestQueryFetchLike, TestQueryPostLike } from '../../src/adaptive/testQueries.ts'

// phase8-04-test-queries
{
  function result(overrides: Record<string, unknown> = {}) {
    return {
      version: 'admin-test-query-result-v1',
      database_id: 'tq-flat',
      mapping_version: 2,
      query_id: 'offerings',
      columns: ['Course.code', 'CourseOffering.section'],
      rows: [
        ['CSE101', '01'],
        ['CSE201', null],
      ],
      row_count: 2,
      row_count_capped: false,
      result_sha256: 'd'.repeat(64),
      ...overrides,
    };
  }
  
  describe('phase8-04 result parser', () => {
    it('accepts a consistent result', () => {
      const parsed = parseTestQueryResult(result());
      assert.equal(parsed?.rows[1][1], null);
    });
  
    it('rejects inconsistent or unsafe results', () => {
      const manyRows = Array.from({ length: 6 }, () => ['X', '01']);
      const cases: Record<string, unknown>[] = [
        { extra: 1 },
        { sql: 'SELECT 1' },
        { version: 'admin-test-query-result-v0' },
        { mapping_version: 0 },
        { query_id: 'Offerings' },
        { columns: ['course_offerings.course_code', 'CourseOffering.section'] },
        { columns: ['Course.course_code', 'CourseOffering.section'] },
        { columns: ['Course.code', 'Course.code'] },
        { rows: [['CSE101']] },
        { rows: [['CSE101', { nested: true }]] },
        { rows: [['CSE101', Number.NaN]] },
        { rows: manyRows, row_count: 6 },
        { row_count: 7 },
        { row_count: 1 },
        { row_count_capped: true },
        { result_sha256: 'x' },
      ];
      for (const overrides of cases) {
        assert.equal(parseTestQueryResult(result(overrides)), null, JSON.stringify(overrides).slice(0, 80));
      }
      const capped = result({ rows: Array.from({ length: 5 }, () => ['X', '01']), row_count: 200, row_count_capped: true });
      assert.ok(parseTestQueryResult(capped));
    });
  
    it('parses the fixed list strictly', () => {
      const list = {
        version: 'admin-test-queries-v1',
        queries: [{ query_id: 'offerings', title: '개설강좌 목록', entity: 'CourseOffering', select: ['Course.code'] }],
      };
      assert.equal(parseTestQueryList(list)?.length, 1);
      assert.equal(parseTestQueryList({ ...list, queries: [] }), null);
      assert.equal(parseTestQueryList({ ...list, queries: [{ ...list.queries[0], sql: 'x' }] }), null);
      assert.equal(parseTestQueryList({ ...list, queries: [list.queries[0], list.queries[0]] }), null);
    });
  });
  
  describe('phase8-04 client', () => {
    it('requests exact URLs and checks ids', async () => {
      const urls: string[] = [];
      const fetchFn: TestQueryFetchLike = async (url) => {
        urls.push(url);
        if (url === TEST_QUERIES_PATH) {
          return {
            ok: true,
            status: 200,
            json: async () => ({
              version: 'admin-test-queries-v1',
              queries: [{ query_id: 'offerings', title: 't', entity: 'CourseOffering', select: ['Course.code'] }],
            }),
          };
        }
        return { ok: true, status: 200, json: async () => result() };
      };
      await fetchTestQueryList(fetchFn);
      const offerings = {
        query_id: 'offerings',
        title: 't',
        entity: 'CourseOffering',
        select: ['Course.code', 'CourseOffering.section'],
      };
      const parsed = await runTestQuery(fetchFn, 'tq-flat', offerings);
      assert.equal(parsed.mapping_version, 2);
      assert.deepEqual(urls, [TEST_QUERIES_PATH, `${TEST_QUERY_PATH}?database_id=tq-flat&query_id=offerings`]);
      await assert.rejects(runTestQuery(fetchFn, 'other-db', offerings), TestQueryClientError);
      await assert.rejects(runTestQuery(fetchFn, 'tq-flat', { ...offerings, query_id: 'seats' }), TestQueryClientError);
      // The answer must carry exactly the listed projection.
      await assert.rejects(
        runTestQuery(fetchFn, 'tq-flat', { ...offerings, select: ['Course.code', 'Course.name'] }),
        TestQueryClientError,
      );
    });
  
    it('refuses invalid ids without calling the server and maps failures', async () => {
      let calls = 0;
      const failing: TestQueryFetchLike = async () => {
        calls += 1;
        return {
          ok: false,
          status: 422,
          json: async () => ({ error: { code: 'ADMIN_TEST_QUERY_FAILED', failure_code: 'SQL_COMPILE_ERROR' } }),
        };
      };
      const definition = { query_id: 'offerings', title: 't', entity: 'CourseOffering', select: ['Course.code'] };
      await assert.rejects(runTestQuery(failing, 'Bad Id', definition), TestQueryClientError);
      await assert.rejects(runTestQuery(failing, 'tq-flat', { ...definition, query_id: 'DROP TABLE' }), TestQueryClientError);
      assert.equal(calls, 0);
      try {
        await runTestQuery(failing, 'tq-flat', definition);
        assert.fail('expected rejection');
      } catch (error) {
        assert.ok(toTestQueryMessage(error).includes('SQL_COMPILE_ERROR'));
      }
    });
  });
  
  
  it('accepts omitted canonical fields and validates the remaining projection', async () => {
    const payload = result({ columns: ['Course.code'], rows: [['CSE101'], ['CSE201']], omitted_fields: ['CourseOffering.section'] });
    assert.deepEqual(parseTestQueryResult(payload)?.omitted_fields, ['CourseOffering.section']);
    const caller: TestQueryFetchLike = async () => ({ ok: true, status: 200, json: async () => payload });
    assert.equal((await runTestQuery(caller, 'tq-flat', { query_id: 'offerings', title: 'test', entity: 'CourseOffering', select: ['Course.code', 'CourseOffering.section'] })).columns.length, 1);
    for (const omitted_fields of [['Course.code'], ['private.column'], ['CourseOffering.section', 'CourseOffering.section']]) {
      assert.equal(parseTestQueryResult({ ...payload, omitted_fields }), null);
    }
    await assert.rejects(runTestQuery(caller, 'tq-flat', { query_id: 'offerings', title: 'test', entity: 'CourseOffering', select: ['Course.code', 'Instructor.name'] }), TestQueryClientError);
  });
  
  it('explains unsupported queries and the PostgreSQL current-version limit', () => {
    assert.match(toTestQueryMessage(new TestQueryClientError('server', 'ADMIN_TEST_QUERY_FAILED', 'TEST_QUERY_UNSUPPORTED')), /이 DB에서는 실행할 수 없음/);
    assert.match(toTestQueryMessage(new TestQueryClientError('server', 'ADMIN_TEST_QUERY_FAILED', 'POSTGRES_CURRENT_VERSION_ONLY')), /현재 게시 버전/);
  });
}

// phase8-05-regression
{
  function entry(queryId: string, overrides: Record<string, unknown> = {}) {
    return {
      query_id: queryId,
      mapping_version: 1,
      result_sha256: 'a'.repeat(64),
      row_count: 6,
      verified_by: 'admin',
      verified_at: '2026-09-29T01:02:03Z',
      ...overrides,
    };
  }
  
  function regression(overrides: Record<string, unknown> = {}) {
    return {
      version: 'verified-query-regression-v1',
      database_id: 'vq-db',
      mapping_version: 2,
      active_mapping_version: 2,
      summary: { match: 1, changed: 1, failed: 1, unverified: 2 },
      results: [
        { query_id: 'offerings', status: 'CHANGED', baseline_mapping_version: 1, baseline_row_count: 6, row_count: 6, failure_code: null },
        { query_id: 'courses', status: 'UNVERIFIED', baseline_mapping_version: null, baseline_row_count: null, row_count: 6, failure_code: null },
        { query_id: 'departments', status: 'UNVERIFIED', baseline_mapping_version: null, baseline_row_count: null, row_count: 6, failure_code: null },
        { query_id: 'schedules', status: 'FAILED', baseline_mapping_version: 1, baseline_row_count: 6, row_count: null, failure_code: 'SQL_COMPILE_ERROR' },
        { query_id: 'seats', status: 'MATCH', baseline_mapping_version: 1, baseline_row_count: 6, row_count: 6, failure_code: null },
      ],
      ...overrides,
    };
  }
  
  describe('phase8-05 parsers', () => {
    it('accepts consistent baselines and regression', () => {
      const list = { version: 'verified-queries-v1', database_id: 'vq-db', entries: [entry('offerings'), entry('seats')] };
      assert.equal(parseVerifiedList(list, 'vq-db')?.length, 2);
      assert.equal(parseRegression(regression())?.summary.changed, 1);
    });
  
    it('rejects inconsistent baselines', () => {
      const base = { version: 'verified-queries-v1', database_id: 'vq-db' };
      const cases = [
        { ...base, entries: [entry('seats'), entry('offerings')] },
        { ...base, entries: [entry('seats'), entry('seats')] },
        { ...base, entries: [entry('seats', { rows: [['x']] })] },
        { ...base, entries: [entry('seats', { result_sha256: 'x' })] },
        { ...base, entries: [entry('seats', { row_count: 201 })] },
        { ...base, database_id: 'other-db', entries: [] },
      ];
      for (const value of cases) {
        assert.equal(parseVerifiedList(value, 'vq-db'), null, JSON.stringify(value).slice(0, 80));
      }
    });
  
    it('rejects inconsistent regression results', () => {
      // Replace one entry of the full fixed set and keep the summary consistent,
      // so each case fails only on the rule it names.
      function withEntry(index: number, patch: Record<string, unknown>) {
        const results: Record<string, unknown>[] = regression().results.map((item) => ({ ...item }));
        results[index] = { ...results[index], ...patch };
        const summary: Record<string, number> = { match: 0, changed: 0, failed: 0, unverified: 0 };
        for (const item of results) {
          const key = String(item['status']).toLowerCase();
          if (key in summary) summary[key] += 1;
        }
        return { results, summary };
      }
      const cases: Record<string, unknown>[] = [
        { summary: { match: 2, changed: 1, failed: 1, unverified: 2 } },
        { results: [] },
        withEntry(1, { query_id: 'offerings' }),
        withEntry(4, { baseline_mapping_version: null, baseline_row_count: null }),
        withEntry(1, { baseline_mapping_version: 1, baseline_row_count: 6 }),
        withEntry(3, { row_count: 6 }),
        withEntry(3, { failure_code: null }),
        withEntry(4, { failure_code: 'X' }),
        withEntry(4, { status: 'SKIPPED' }),
        withEntry(4, { result_sha256: 'a'.repeat(64) }),
        // Every fixed query must be reported, in server order.
        { results: [regression().results[4]], summary: { match: 1, changed: 0, failed: 0, unverified: 0 } },
        { results: [...regression().results].reverse() },
        // MATCH means an identical result, so the row counts must agree.
        withEntry(4, { row_count: 5 }),
        { mapping_version: 0 },
        { extra: true },
      ];
      for (const overrides of cases) {
        assert.equal(parseRegression(regression(overrides)), null, JSON.stringify(overrides).slice(0, 100));
      }
    });
  });
  
  describe('phase8-05 clients', () => {
    it('verify sends only ids and the reviewer', async () => {
      const bodies: unknown[] = [];
      const postFn: TestQueryPostLike = async (url, init) => {
        assert.equal(url, VERIFIED_QUERIES_PATH);
        bodies.push(JSON.parse(init.body));
        return {
          ok: true,
          status: 200,
          json: async () => ({ version: 'verified-queries-v1', database_id: 'vq-db', entries: [entry('seats')] }),
        };
      };
      const entries = await verifyTestQuery(postFn, 'vq-db', 'seats', 'admin');
      assert.equal(entries[0].query_id, 'seats');
      assert.deepEqual(bodies, [{ action: 'VERIFY', database_id: 'vq-db', query_id: 'seats', verified_by: 'admin' }]);
      await assert.rejects(verifyTestQuery(postFn, 'vq-db', 'offerings', 'admin'), TestQueryClientError);
      await assert.rejects(verifyTestQuery(postFn, 'vq-db', 'seats', ''), TestQueryClientError);
      assert.equal(bodies.length, 2);
    });
  
    it('regression requests the exact version', async () => {
      const urls: string[] = [];
      const fetchFn: TestQueryFetchLike = async (url) => {
        urls.push(url);
        return { ok: true, status: 200, json: async () => regression() };
      };
      const parsed = await fetchRegression(fetchFn, 'vq-db', 2);
      assert.equal(parsed.mapping_version, 2);
      assert.deepEqual(urls, [`${REGRESSION_PATH}?database_id=vq-db&mapping_version=2`]);
      await assert.rejects(fetchRegression(fetchFn, 'vq-db', 1), TestQueryClientError);
      await assert.rejects(fetchRegression(fetchFn, 'vq-db', 0), TestQueryClientError);
      assert.equal(urls.length, 2);
    });
  });
}
