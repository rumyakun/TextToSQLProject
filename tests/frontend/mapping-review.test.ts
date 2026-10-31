import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { MAPPING_REVIEW_PATH, MAPPING_REVIEW_VERSION, approveReviewField, canStartPublish, canStartReviewAction, fetchReviewDetail, getReviewErrorMessage, initReview, isValidReason, isValidReviewer, parseReviewDetail, parseReviewError, parseReviewPublish, publishReview, remapReviewField, toReviewMessage, ACTIVATION_BLOCKED_MESSAGE, MappingReviewClientError, activateMappingVersion, toActivationMessage, carryOverReview, PUBLISH_RUNTIME_BLOCKED_MESSAGE } from '../../src/adaptive/mappingReview.ts'
import type { ReviewFetchLike, ReviewFetchInit, ReviewField, ReviewDetail } from '../../src/adaptive/mappingReview.ts'
import { parseDatabaseRegistry } from '../../src/adaptive/databases.ts'
import { isStaleResponse, nextOperation } from '../../src/adaptive/state.ts'
import { SCHEMA_COMPARE_PATH, SchemaCompareClientError, fetchSchemaComparison, parseSchemaComparison, toCompareMessage } from '../../src/adaptive/schemaCompare.ts'
import type { CompareFetchLike } from '../../src/adaptive/schemaCompare.ts'
import { canSaveRowEdit, startRowEdit } from '../../src/adaptive/mappingRowEditor.ts'
import { bulkApprovalConfirmation, runBulkApproval, selectBulkApproval } from '../../src/adaptive/bulkApprove.ts'
import { reviewFieldName } from '../../src/adaptive/reviewFieldName.ts'
import ts from 'typescript'
import { disabledTitle } from '../../src/adaptive/adminStyles.ts'
import { focusReviewer, readReviewer, REVIEWER_STORAGE_KEY, writeReviewer } from '../../src/adaptive/reviewerMemory.ts'
import { reviewRefreshAction, reviewToolsRevision } from '../../src/adaptive/reviewToolsRefresh.ts'

// phase6-19-mapping-review
{
  function detailPayload() {
    return {
      version: 'private-mapping-review-v2',
      database_id: 'review-db-one',
      label: 'Review DB',
      status: 'PENDING_REVIEW',
      schema_fingerprint: 'b'.repeat(64),
      ontology_version: 1,
      review: {
        field_count: 2,
        fields: [
          {
            semantic_field: 'Course.name',
            decision: 'PENDING',
            table: null,
            column: null,
            candidates: [
              {
                index: 0,
                table: 'course_offerings',
                column: 'course_name',
                confidence: 0.9,
                review_status: 'NEEDS_REVIEW',
                source: 'AI',
              },
            ],
          },
          {
            semantic_field: 'Course.code',
            decision: 'APPROVED',
            table: 'course_offerings',
            column: 'course_code',
            candidates: [
              {
                index: 0,
                table: 'course_offerings',
                column: 'course_code',
                confidence: 1,
                review_status: 'AUTO_APPROVE_CANDIDATE',
                source: 'RULES',
              },
            ],
          },
        ],
        allowed_tables: [{ name: 'course_offerings', columns: ['course_code', 'course_name'] }],
      },
    };
  }
  
  describe('phase6-19 review parser and request contract', () => {
    it('accepts strict detail and rejects extra keys/version/status/fingerprint', () => {
      const parsed = parseReviewDetail(detailPayload());
      assert.ok(parsed !== null);
      assert.equal(parsed?.fields.length, 2);
      assert.equal(parsed?.mapping_version, null);
      assert.equal(parseReviewDetail({ ...detailPayload(), extra: 1 }), null);
      assert.equal(parseReviewDetail({ ...detailPayload(), version: 'vX' }), null);
      assert.equal(parseReviewDetail({ ...detailPayload(), status: 'DRAFT' }), null);
      assert.equal(
        parseReviewDetail({ ...detailPayload(), schema_fingerprint: 'zz' }),
        null,
      );
      const published = {
        ...detailPayload(),
        status: 'PUBLISHED',
        mapping_version: 1,
        active_mapping_version: 1,
        mapping_versions: [
          {
            version: 1,
            published_by: 'tester',
            published_at: '2026-09-29T00:00:00Z',
            active: true,
            latest: true,
            activatable: true,
          },
        ],
      };
      assert.equal(parseReviewDetail(published)?.mapping_version, 1);
      assert.equal(parseReviewDetail({ ...detailPayload(), status: 'PUBLISHED' }), null);
    });
  
    it('phase8-12: requires the candidate source and keeps it', () => {
      const parsed = parseReviewDetail(detailPayload());
      assert.equal(parsed?.fields[0]?.candidates[0]?.source, 'AI');
      assert.equal(parsed?.fields[1]?.candidates[0]?.source, 'RULES');
      for (const source of [undefined, 'MODEL', null]) {
        const payload = structuredClone(detailPayload()) as unknown as {
          review: { fields: Array<{ candidates: Array<Record<string, unknown>> }> };
        };
        const candidate = payload.review.fields[0]!.candidates[0]!;
        if (source === undefined) delete candidate['source'];
        else candidate['source'] = source;
        assert.equal(parseReviewDetail(payload), null);
      }
      for (const status of ['BOGUS', 'UNMAPPED', '']) {
        const payload = structuredClone(detailPayload());
        payload.review.fields[0]!.candidates[0]!.review_status = status;
        assert.equal(parseReviewDetail(payload), null);
      }
    });
  
    it('accepts strict publish and redacts unknown shapes', () => {
      const payload = {
        version: 'private-mapping-review-v2',
        database_id: 'review-db-one',
        status: 'PUBLISHED',
        mapping_version: 2,
        registry_eligible: true,
      };
      assert.deepEqual(parseReviewPublish(payload), {
        database_id: 'review-db-one',
        mapping_version: 2,
      });
      assert.equal(parseReviewPublish({ ...payload, registry_eligible: false }), null);
      assert.equal(parseReviewPublish({ ...payload, status: 'PENDING_REVIEW' }), null);
      assert.equal(parseReviewPublish({ ...payload, extra: 1 }), null);
    });
  
    it('parses fixed error codes and optional safe field lists only', () => {
      assert.deepEqual(parseReviewError({ error: { code: 'MAPPING_REVIEW_CONFLICT' } }), {
        code: 'MAPPING_REVIEW_CONFLICT',
        fields: undefined,
      });
      assert.deepEqual(
        parseReviewError({ error: { code: 'MAPPING_REVIEW_PUBLISH_BLOCKED', fields: ['Course.name'] } }),
        { code: 'MAPPING_REVIEW_PUBLISH_BLOCKED', fields: ['Course.name'] },
      );
      assert.equal(parseReviewError({ error: { code: 'x', detail: 'leak' } }), null);
      assert.equal(parseReviewError({ code: 'MAPPING_REVIEW_CONFLICT' }), null);
      assert.equal(getReviewErrorMessage('MAPPING_REVIEW_PUBLISH_BLOCKED').includes('승인'), true);
      assert.equal(toReviewMessage(new Error('boom')), '처리 중 오류가 발생했습니다. 다시 시도해 주세요.');
    });
  
    it('validates reviewer and reason without control characters', () => {
      assert.equal(isValidReviewer('local-reviewer'), true);
      assert.equal(isValidReviewer('  '), false);
      assert.equal(isValidReviewer(' lead'), false);
      assert.equal(isValidReviewer('x'.repeat(65)), false);
      assert.equal(isValidReviewer('a\u0000b'), false);
      assert.equal(isValidReason('관리자 확인'), true);
      assert.equal(isValidReason(''), false);
      assert.equal(isValidReason('x'.repeat(501)), false);
    });
  
    it('blocks duplicate actions while busy and allows retry after error', () => {
      assert.equal(canStartReviewAction('ready'), true);
      assert.equal(canStartReviewAction('error'), true);
      assert.equal(canStartReviewAction('acting'), false);
      assert.equal(canStartReviewAction('publishing'), false);
      assert.equal(canStartReviewAction('loading'), false);
      assert.equal(canStartReviewAction('published'), false);
      assert.equal(canStartPublish('ready'), true);
      assert.equal(canStartPublish('acting'), false);
    });
  
    it('sends strict action bodies and rejects database mismatch', async () => {
      const seen: Array<{ url: string; body: unknown }> = [];
      const fetchFn: ReviewFetchLike = async (url, init) => {
        seen.push({ url, body: init.body !== undefined ? JSON.parse(init.body) : null });
        return { ok: true, status: 200, json: async () => detailPayload() };
      };
      await initReview(fetchFn, 'review-db-one', 'tester');
      await approveReviewField(fetchFn, 'review-db-one', 'Course.name', 0, 'tester');
      await remapReviewField(
        fetchFn,
        'review-db-one',
        'Course.name',
        'course_offerings',
        'course_name',
        'tester',
        '관리자 확인',
      );
      assert.deepEqual(seen[0], {
        url: MAPPING_REVIEW_PATH,
        body: { action: 'INIT', database_id: 'review-db-one', reviewer: 'tester' },
      });
      assert.deepEqual(seen[1].body, {
        action: 'APPROVE',
        database_id: 'review-db-one',
        semantic_field: 'Course.name',
        candidate_index: 0,
        reviewer: 'tester',
      });
      assert.deepEqual(seen[2].body, {
        action: 'REMAP',
        database_id: 'review-db-one',
        semantic_field: 'Course.name',
        table: 'course_offerings',
        column: 'course_name',
        reviewer: 'tester',
        reason: '관리자 확인',
      });
      await assert.rejects(() => initReview(fetchFn, 'BAD ID', 'tester'));
      await assert.rejects(() => publishReview(fetchFn, 'review-db-one', '  '));
      const mismatch: ReviewFetchLike = async () => ({
        ok: true,
        status: 200,
        json: async () => ({ ...detailPayload(), database_id: 'other-db' }),
      });
      await assert.rejects(() => fetchReviewDetail(mismatch, 'review-db-one'));
    });
  
    it('maps transport and server failures to fixed messages', async () => {
      const failing: ReviewFetchLike = async () => {
        throw new Error('down');
      };
      await assert.rejects(() => fetchReviewDetail(failing, 'review-db-one'));
      try {
        await fetchReviewDetail(failing, 'review-db-one');
      } catch (error) {
        assert.equal(toReviewMessage(error), '연결에 실패했습니다. 다시 시도해 주세요.');
      }
      const server: ReviewFetchLike = async () => ({
        ok: false,
        status: 422,
        json: async () => ({ error: { code: 'MAPPING_REVIEW_PUBLISH_BLOCKED', fields: ['Course.name'] } }),
      });
      try {
        await publishReview(server, 'review-db-one', 'tester');
        assert.fail('must throw');
      } catch (error) {
        assert.equal(
          toReviewMessage(error),
          getReviewErrorMessage('MAPPING_REVIEW_PUBLISH_BLOCKED'),
        );
      }
    });
  
    it('supports abort and stale operation guards', () => {
      const first = nextOperation(0);
      const second = nextOperation(first);
      assert.equal(isStaleResponse(second, first), true);
      assert.equal(isStaleResponse(second, second), false);
      assert.equal(MAPPING_REVIEW_VERSION, 'private-mapping-review-v2');
    });
  });
  
  describe('phase6-19 registry private variant', () => {
    function registryPayload() {
      return {
        version: 'synthetic-database-registry-v1',
        databases: [
          { database_id: 'synth-clarification-prototype', label: 'Flat', schema: 'flat' },
          { database_id: 'review-db-one', label: '검토용 강의 DB', schema: 'private' },
        ],
      };
    }
  
    it('accepts schema private and rejects unknown variants', () => {
      assert.equal(parseDatabaseRegistry(registryPayload())?.length, 2);
      for (const invalid of [
        { database_id: 'review-db-one', label: 'x', schema: 'weird' },
        { database_id: 'review-db-one', label: '   ', schema: 'private' },
      ]) {
        const valid = registryPayload().databases[0]!;
        const warnings: Array<string | null> = [];
        assert.deepEqual(parseDatabaseRegistry({
          version: 'synthetic-database-registry-v1',
          databases: [invalid, valid],
        }, (message) => warnings.push(message)), [valid]);
        assert.deepEqual(warnings, ['형식이 올바르지 않은 DB 1개를 목록에서 뺐습니다.']);
      }
    });
  });
  
  describe('phase6-19 review panel wiring', () => {
    it('starts review below upload summary without touching active selection', () => {
      const panel = readFileSync('src/adaptive/MappingReviewPanel.tsx', 'utf-8');
      assert.ok(panel.includes('mapping 검토 시작'));
      assert.ok(panel.includes('operationRef'));
      assert.ok(panel.includes('abortRef'));
      assert.ok(panel.includes('isStaleResponse'));
      assert.ok(!panel.includes('sessionStorage'));
      assert.ok(!panel.includes('setActiveDbId'));
      const publishSuccess = panel.indexOf("setStatus('published')");
      const detailRefresh = panel.indexOf('const refreshed = await fetchReviewDetail', publishSuccess);
      assert.ok(publishSuccess >= 0);
      assert.ok(detailRefresh > publishSuccess);
      assert.ok(panel.indexOf('onPublished?.(databaseId)', publishSuccess) < detailRefresh);
      const main = readFileSync('src/pages/MainPage.tsx', 'utf-8');
      assert.ok(main.includes('<MappingReviewPanel'));
      assert.ok(main.includes('onPublished={() => setRegistryRetry((value) => value + 1)}'));
      assert.ok(main.includes('mapping 검토 시작') === false);
      assert.ok(main.includes('{registry.map((entry) => ('));
    });
  });
}

// phase8-01-mapping-versions
{
  function version(n: number, overrides: Record<string, unknown> = {}) {
    return {
      version: n,
      published_by: 'tester',
      published_at: '2026-09-29T01:02:03Z',
      active: false,
      latest: false,
      activatable: true,
      ...overrides,
    };
  }
  
  function publishedDetail(overrides: Record<string, unknown> = {}) {
    return {
      version: 'private-mapping-review-v2',
      database_id: 'review-db-one',
      label: 'Review DB',
      status: 'PUBLISHED',
      schema_fingerprint: 'b'.repeat(64),
      ontology_version: 1,
      review: { field_count: 0, fields: [], allowed_tables: [] },
      mapping_version: 2,
      active_mapping_version: 1,
      mapping_versions: [version(1, { active: true }), version(2, { latest: true })],
      ...overrides,
    };
  }
  
  describe('phase8-01 version history parser', () => {
    it('accepts a consistent history', () => {
      const parsed = parseReviewDetail(publishedDetail());
      assert.ok(parsed !== null);
      assert.equal(parsed?.active_mapping_version, 1);
      assert.deepEqual(
        parsed?.mapping_versions.map((entry) => [entry.version, entry.active, entry.latest]),
        [
          [1, true, false],
          [2, false, true],
        ],
      );
    });
  
    it('accepts a null active version only when no entry is active', () => {
      const history = [version(1), version(2, { latest: true })];
      assert.ok(parseReviewDetail(publishedDetail({ active_mapping_version: null, mapping_versions: history })));
      assert.equal(parseReviewDetail(publishedDetail({ active_mapping_version: null })), null);
    });
  
    it('accepts an active version without reactivation support', () => {
      const history = [version(1, { active: true, activatable: false }), version(2, { latest: true })];
      const parsed = parseReviewDetail(publishedDetail({ mapping_versions: history }));
      assert.ok(parsed);
      assert.deepEqual(parsed.mapping_versions, history);
    });
  
    it('rejects inconsistent or malformed histories', () => {
      const cases: Record<string, unknown>[] = [
        { active_mapping_version: 2 },
        { mapping_versions: [version(1, { active: true }), version(2, { active: true, latest: true })] },
        { mapping_versions: [version(2, { latest: true }), version(1, { active: true })] },
        { mapping_versions: [version(1, { active: true }), version(1, { latest: true })] },
        { mapping_versions: [version(1, { active: true, latest: true })] },
        { mapping_versions: [version(1, { active: true }), version(3, { latest: true })] },
        { mapping_versions: [version(1, { active: true }), version(2)] },
        { mapping_versions: [version(1, { active: true, extra: 1 }), version(2, { latest: true })] },
        { mapping_versions: [version(1, { active: true, published_at: '2026-09-29' }), version(2, { latest: true })] },
        { mapping_versions: [version(1, { active: true, published_by: ' x' }), version(2, { latest: true })] },
        { mapping_versions: [version(1, { active: 'yes' }), version(2, { latest: true })] },
        { mapping_versions: 'none' },
        { mapping_versions: Array.from({ length: 51 }, (_, index) => version(index + 1)) },
        { mapping_version: 0 },
        { active_mapping_version: '1' },
      ];
      for (const overrides of cases) {
        assert.equal(parseReviewDetail(publishedDetail(overrides)), null, JSON.stringify(overrides));
      }
    });
  
    it('requires all history keys when published and none when pending', () => {
      const omit = (value: Record<string, unknown>, keys: string[]) =>
        Object.fromEntries(Object.entries(value).filter(([key]) => !keys.includes(key)));
      assert.equal(parseReviewDetail(omit(publishedDetail(), ['mapping_versions'])), null);
      const pending = { ...publishedDetail(), status: 'PENDING_REVIEW' };
      assert.equal(parseReviewDetail(pending), null);
      const history = ['mapping_version', 'active_mapping_version', 'mapping_versions'];
      assert.ok(parseReviewDetail(omit(pending, history)));
    });
  });
  
  describe('phase8-01 activation client', () => {
    function recorder(status: number, payload: unknown): { fetchFn: ReviewFetchLike; calls: [string, ReviewFetchInit][] } {
      const calls: [string, ReviewFetchInit][] = [];
      const fetchFn: ReviewFetchLike = async (url, init) => {
        calls.push([url, init]);
        return { ok: status >= 200 && status < 300, status, json: async () => payload };
      };
      return { fetchFn, calls };
    }
  
    const success = {
      version: 'private-mapping-review-v2',
      database_id: 'review-db-one',
      status: 'PUBLISHED',
      mapping_version: 1,
      registry_eligible: true,
    };
  
    it('posts the exact ACTIVATE body and checks the answered version', async () => {
      const { fetchFn, calls } = recorder(200, success);
      const result = await activateMappingVersion(fetchFn, 'review-db-one', 1, 'admin');
      assert.deepEqual(result, { database_id: 'review-db-one', mapping_version: 1 });
      assert.equal(calls.length, 1);
      assert.equal(calls[0][0], MAPPING_REVIEW_PATH);
      assert.equal(calls[0][1].method, 'POST');
      assert.deepEqual(JSON.parse(calls[0][1].body ?? ''), {
        action: 'ACTIVATE',
        database_id: 'review-db-one',
        mapping_version: 1,
        activated_by: 'admin',
      });
    });
  
    it('rejects a response for a different version', async () => {
      const { fetchFn } = recorder(200, { ...success, mapping_version: 2 });
      await assert.rejects(activateMappingVersion(fetchFn, 'review-db-one', 1, 'admin'), MappingReviewClientError);
    });
  
    it('refuses invalid inputs without calling the server', async () => {
      const { fetchFn, calls } = recorder(200, success);
      for (const [id, versionNumber, by] of [
        ['Bad Id', 1, 'admin'],
        ['review-db-one', 0, 'admin'],
        ['review-db-one', 1.5, 'admin'],
        ['review-db-one', 10000, 'admin'],
        ['review-db-one', 1, ''],
        ['review-db-one', 1, 'admin\n'],
      ] as [string, number, string][]) {
        await assert.rejects(activateMappingVersion(fetchFn, id, versionNumber, by), MappingReviewClientError);
      }
      assert.equal(calls.length, 0);
    });
  
    it('maps a blocked activation to a fixed message', async () => {
      const { fetchFn } = recorder(422, { error: { code: 'MAPPING_REVIEW_PUBLISH_BLOCKED' } });
      try {
        await activateMappingVersion(fetchFn, 'review-db-one', 1, 'admin');
        assert.fail('expected rejection');
      } catch (error) {
        assert.equal(toActivationMessage(error), ACTIVATION_BLOCKED_MESSAGE);
      }
      assert.equal(toActivationMessage(new Error('leak /private/path')).includes('/private'), false);
    });
  });
}

// phase8-02-schema-compare
{
  const shape = { declared_type: 'TEXT', affinity: 'TEXT', nullable: false, primary_key_position: 0 };
  
  function comparison(overrides: Record<string, unknown> = {}) {
    return {
      version: 'registration-compare-v1',
      base_database_id: 'base-db',
      candidate_database_id: 'next-db',
      base_mapping_version: 2,
      base_fingerprint: 'a'.repeat(64),
      candidate_fingerprint: 'b'.repeat(64),
      identical: false,
      ontology_version_match: true,
      carry_over_executable: false,
      truncated: false,
      tables_added: ['extra'],
      tables_removed: [],
      column_changes: [
        { table: 'course_offerings', column: 'course_name', change: 'REMOVED', base: shape, candidate: null },
        { table: 'course_offerings', column: 'title', change: 'ADDED', base: null, candidate: shape },
        {
          table: 'course_offerings',
          column: 'credits',
          change: 'TYPE_CHANGED',
          base: shape,
          candidate: { ...shape, declared_type: 'REAL', affinity: 'REAL' },
        },
      ],
      foreign_key_changes: [
        { table: 'sections', column: 'code', referenced_table: 'subjects', referenced_column: 'code', change: 'REMOVED' },
      ],
      field_impacts: [
        { semantic_field: 'Course.name', table: 'course_offerings', column: 'course_name', status: 'COLUMN_MISSING' },
        { semantic_field: 'Course.code', table: 'course_offerings', column: 'course_code', status: 'UNCHANGED' },
      ],
      ...overrides,
    };
  }
  
  describe('phase8-02 schema comparison parser', () => {
    it('accepts a consistent comparison', () => {
      const parsed = parseSchemaComparison(comparison());
      assert.ok(parsed !== null);
      assert.equal(parsed?.column_changes.length, 3);
      assert.equal(parsed?.field_impacts[0].status, 'COLUMN_MISSING');
    });
  
    it('accepts an identical schema that carries over', () => {
      const identical = comparison({
        candidate_fingerprint: 'a'.repeat(64),
        identical: true,
        carry_over_executable: true,
        tables_added: [],
        column_changes: [],
        foreign_key_changes: [],
        field_impacts: [
          { semantic_field: 'Course.name', table: 'course_offerings', column: 'course_name', status: 'UNCHANGED' },
        ],
      });
      assert.ok(parseSchemaComparison(identical));
    });
  
    it('rejects malformed or inconsistent comparisons', () => {
      const cases: Record<string, unknown>[] = [
        { extra: 1 },
        { version: 'registration-compare-v0' },
        { candidate_database_id: 'base-db' },
        { base_mapping_version: 0 },
        { base_fingerprint: 'x' },
        { identical: true },
        { carry_over_executable: true },
        { truncated: 'no' },
        { tables_added: [''] },
        { column_changes: [{ table: 't', column: 'c', change: 'ADDED', base: shape, candidate: shape }] },
        { column_changes: [{ table: 't', column: 'c', change: 'REMOVED', base: shape, candidate: shape }] },
        { column_changes: [{ table: 't', column: 'c', change: 'RENAMED', base: shape, candidate: shape }] },
        { column_changes: [{ table: 't', column: 'c', change: 'TYPE_CHANGED', base: { ...shape, sample_values: ['x'] }, candidate: shape }] },
        { foreign_key_changes: [{ table: 't', column: 'c', referenced_table: 'r', referenced_column: null, change: 'MOVED' }] },
        { column_changes: [{ table: 't', column: 'c', change: 'NULLABILITY_CHANGED', base: shape, candidate: shape }] },
        { column_changes: [{ table: 't', column: 'c', change: 'TYPE_CHANGED', base: shape, candidate: { ...shape, nullable: true } }] },
        { column_changes: [{ table: 't', column: 'c', change: 'PRIMARY_KEY_CHANGED', base: shape, candidate: shape }] },
        {
          field_impacts: [
            { semantic_field: 'Course.name', table: 't', column: 'c', status: 'UNCHANGED' },
            { semantic_field: 'Course.name', table: 't', column: 'd', status: 'UNCHANGED' },
          ],
        },
        { field_impacts: [{ semantic_field: 'Course.name', table: 't', column: 'c', status: 'GONE' }] },
        { tables_added: Array.from({ length: 501 }, (_, index) => `t${index}`) },
      ];
      for (const overrides of cases) {
        assert.equal(parseSchemaComparison(comparison(overrides)), null, JSON.stringify(overrides).slice(0, 120));
      }
    });
  });
  
  describe('phase8-02 schema comparison client', () => {
    it('requests the exact URL and checks the answered ids', async () => {
      const urls: string[] = [];
      const fetchFn: CompareFetchLike = async (url) => {
        urls.push(url);
        return { ok: true, status: 200, json: async () => comparison() };
      };
      const result = await fetchSchemaComparison(fetchFn, 'base-db', 'next-db');
      assert.equal(result.base_database_id, 'base-db');
      assert.deepEqual(urls, [`${SCHEMA_COMPARE_PATH}?base_database_id=base-db&candidate_database_id=next-db`]);
      await assert.rejects(fetchSchemaComparison(fetchFn, 'base-db', 'other-db'), SchemaCompareClientError);
    });
  
    it('refuses invalid ids without calling the server', async () => {
      let calls = 0;
      const fetchFn: CompareFetchLike = async () => {
        calls += 1;
        return { ok: true, status: 200, json: async () => comparison() };
      };
      await assert.rejects(fetchSchemaComparison(fetchFn, 'same-db', 'same-db'), SchemaCompareClientError);
      await assert.rejects(fetchSchemaComparison(fetchFn, 'Bad Id', 'next-db'), SchemaCompareClientError);
      assert.equal(calls, 0);
    });
  
    it('maps fixed server codes and hides unknown text', async () => {
      const fetchFn: CompareFetchLike = async () => ({
        ok: false,
        status: 404,
        json: async () => ({ error: { code: 'SCHEMA_COMPARE_NOT_FOUND' } }),
      });
      try {
        await fetchSchemaComparison(fetchFn, 'base-db', 'next-db');
        assert.fail('expected rejection');
      } catch (error) {
        assert.ok(toCompareMessage(error).includes('기준 DB'));
      }
      assert.equal(toCompareMessage(new Error('C:/private/path')).includes('private'), false);
    });
  });
}

// phase8-03-carry-over
{
  function field(name: string, decision = 'PENDING') {
    return {
      semantic_field: name,
      decision,
      table: decision === 'PENDING' ? null : 'course_offerings',
      column: decision === 'PENDING' ? null : 'c',
      candidates: [],
    };
  }
  
  function pendingDetail(overrides: Record<string, unknown> = {}) {
    return {
      version: 'private-mapping-review-v2',
      database_id: 'next-db',
      label: 'Next DB',
      status: 'PENDING_REVIEW',
      schema_fingerprint: 'c'.repeat(64),
      ontology_version: 1,
      review: {
        field_count: 3,
        fields: [field('Course.code', 'REMAPPED'), field('Course.name'), field('Instructor.name', 'REMAPPED')],
        allowed_tables: [],
      },
      succession: {
        base_database_id: 'base-db',
        base_mapping_version: 2,
        carried_fields: ['Course.code', 'Instructor.name'],
        stale_fields: [{ semantic_field: 'Course.name', status: 'COLUMN_MISSING' }],
      },
      ...overrides,
    };
  }
  
  describe('phase8-03 succession parser', () => {
    it('accepts a consistent record and a detail without one', () => {
      const parsed = parseReviewDetail(pendingDetail());
      assert.equal(parsed?.succession?.stale_fields[0].status, 'COLUMN_MISSING');
      const withoutRecord = Object.fromEntries(Object.entries(pendingDetail()).filter(([key]) => key !== 'succession'));
      assert.equal(parseReviewDetail(withoutRecord)?.succession, null);
    });
  
    it('rejects inconsistent records', () => {
      const base = pendingDetail().succession;
      const cases: Record<string, unknown>[] = [
        { ...base, base_database_id: 'next-db' },
        { ...base, base_database_id: 'Bad Id' },
        { ...base, base_mapping_version: 0 },
        { ...base, carried_fields: ['Course.code', 'Course.name'] },
        { ...base, carried_fields: ['Unknown.field'] },
        { ...base, stale_fields: [{ semantic_field: 'Course.name', status: 'RENAMED' }] },
        { ...base, stale_fields: [{ semantic_field: 'Course.name', status: 'COLUMN_MISSING', extra: 1 }] },
        { ...base, extra: 1 },
        { ...base, carried_fields: [] },
        { ...base, stale_fields: [] },
        { base_database_id: 'base-db', base_mapping_version: 2, carried_fields: [] },
        null,
      ];
      for (const succession of cases) {
        assert.equal(parseReviewDetail(pendingDetail({ succession })), null, JSON.stringify(succession));
      }
    });
  });
  
  describe('phase8-03 carry-over client', () => {
    function recorder(payload: unknown): { fetchFn: ReviewFetchLike; bodies: unknown[] } {
      const bodies: unknown[] = [];
      const fetchFn: ReviewFetchLike = async (_url: string, init: ReviewFetchInit) => {
        bodies.push(JSON.parse(init.body ?? 'null'));
        return { ok: true, status: 200, json: async () => payload };
      };
      return { fetchFn, bodies };
    }
  
    it('posts the exact body and requires the matching record', async () => {
      const { fetchFn, bodies } = recorder(pendingDetail());
      const detail = await carryOverReview(fetchFn, 'next-db', 'base-db', 'admin');
      assert.equal(detail.succession?.base_database_id, 'base-db');
      assert.deepEqual(bodies, [
        { action: 'CARRY_OVER', database_id: 'next-db', base_database_id: 'base-db', reviewer: 'admin' },
      ]);
      const other = recorder(pendingDetail({ succession: { ...pendingDetail().succession, base_database_id: 'else-db' } }));
      await assert.rejects(carryOverReview(other.fetchFn, 'next-db', 'base-db', 'admin'), MappingReviewClientError);
      const missing = recorder(Object.fromEntries(Object.entries(pendingDetail()).filter(([key]) => key !== 'succession')));
      await assert.rejects(carryOverReview(missing.fetchFn, 'next-db', 'base-db', 'admin'), MappingReviewClientError);
    });
  
    it('refuses invalid inputs without calling the server', async () => {
      const { fetchFn, bodies } = recorder(pendingDetail());
      await assert.rejects(carryOverReview(fetchFn, 'next-db', 'next-db', 'admin'), MappingReviewClientError);
      await assert.rejects(carryOverReview(fetchFn, 'next-db', 'base-db', ''), MappingReviewClientError);
      await assert.rejects(carryOverReview(fetchFn, 'Bad Id', 'base-db', 'admin'), MappingReviewClientError);
      assert.equal(bodies.length, 0);
    });
  });
}

// phase9-07-publish-block-reasons
{
  const BLOCKED = 'MAPPING_REVIEW_PUBLISH_BLOCKED';
  const reasons = [
    { field: 'Course.credits', label: '학점', reason: 'NUMBER_FORMAT' },
    { field: 'Schedule.end_time', label: '종료 시간', reason: 'TIME_FORMAT' },
    { field: 'Schedule.start_time', label: '시작 시간', reason: 'COLUMN_SHARED', other: 'Schedule.end_time', other_label: '종료 시간' },
    { field: 'CourseOffering.year', label: '개설 연도', reason: 'COLUMN_SHARED', other: 'CourseOffering.semester', other_label: '학기' },
  ];
  const fields = reasons.map((item) => item.field);
  
  describe('phase9-07 publish block reasons', () => {
    it('parses the reasons beside the fields', () => {
      const parsed = parseReviewError({ error: { code: BLOCKED, fields, reasons } });
      assert.deepEqual(parsed?.reasons, reasons);
      assert.deepEqual(parseReviewError({ error: { code: BLOCKED, fields } }), { code: BLOCKED, fields });
    });
  
    it('refuses malformed reasons', () => {
      for (const error of [
        { code: BLOCKED, reasons },  // reasons without fields
        { code: BLOCKED, fields, reasons: [{ ...reasons[0], extra: 'x' }] },
        { code: BLOCKED, fields, reasons: [{ ...reasons[0], field: 'Course.name' }] },  // not a blocked field
        { code: BLOCKED, fields, reasons: [{ field: 'Course.credits', label: '학점', reason: 'NUMBER_FORMAT', other: 'Course.name' }] },
        { code: BLOCKED, fields, reasons: [{ ...reasons[0], reason: 'bad reason' }] },
        { code: BLOCKED, fields, reasons: [{ ...reasons[0], label: '' }] },
        { code: BLOCKED, fields, reasons: 'NUMBER_FORMAT' },
        { code: BLOCKED, fields, other: 1 },
      ]) {
        assert.equal(parseReviewError({ error }), null, JSON.stringify(error));
      }
    });
  
    it('explains every blocked field in one message', () => {
      const message = toReviewMessage(new MappingReviewClientError('server', BLOCKED, fields, reasons));
      const lines = message.split('\n');
      assert.equal(lines.length, 1 + reasons.length);
      assert.match(lines[1], /^- 학점: 숫자 칸이 아닙니다/);
      assert.match(lines[2], /^- 종료 시간: 시간 값이 09:00 꼴이 아닙니다/);
      assert.match(lines[3], /^- 시작 시간: 종료 시간과 같은 칸을 씁니다/);
      assert.match(lines[4], /^- 개설 연도: 학기와 같은 칸을 씁니다/);
    });
  
    it('keeps the older messages when no reasons come', () => {
      assert.equal(toReviewMessage(new MappingReviewClientError('server', BLOCKED, [])), PUBLISH_RUNTIME_BLOCKED_MESSAGE);
      assert.equal(
        toReviewMessage(new MappingReviewClientError('server', BLOCKED, ['Course.name'])),
        getReviewErrorMessage(BLOCKED),
      );
    });
  
    it('the panels show line breaks in their error area', () => {
      for (const file of ['MappingReviewPanel.tsx', 'ExtensionFieldPanel.tsx', 'ValueDictionaryPanel.tsx']) {
        const source = readFileSync(new URL(`../../src/adaptive/${file}`, import.meta.url), 'utf8');
        assert.match(source, /whitespace-pre-line[^>]*role="alert"|role="alert"[^>]*whitespace-pre-line/, file);
      }
    });
  });
  
  describe('phase9-07 review fixes', () => {
    it('refuses an unknown reason code and lists that leave out or repeat a field', () => {
      const one = { field: 'Course.credits', label: '학점', reason: 'NUMBER_FORMAT' };
      for (const error of [
        { code: BLOCKED, fields: ['Course.credits'], reasons: [{ ...one, reason: 'FAKE_REASON' }] },
        { code: BLOCKED, fields: ['Course.name', 'Course.credits'], reasons: [one] },
        { code: BLOCKED, fields: ['Course.credits'], reasons: [one, one] },
      ]) {
        assert.equal(parseReviewError({ error }), null, JSON.stringify(error));
      }
    });
  });
  
  
  describe('phase9-30 publish block reasons', () => {
    it('parses and explains text-on-number and ambiguous-join blockers', () => {
      const nextReasons = [
        { field: 'Schedule.day', label: '강의 요일', reason: 'TEXT_ON_NUMBER' },
        { field: 'Instructor.name', label: '교원명', reason: 'JOIN_AMBIGUOUS' },
      ];
      const parsed = parseReviewError({ error: { code: BLOCKED, fields: nextReasons.map((item) => item.field), reasons: nextReasons } });
      assert.deepEqual(parsed?.reasons, nextReasons);
      const message = toReviewMessage(new MappingReviewClientError('server', BLOCKED, nextReasons.map((item) => item.field), nextReasons));
      assert.match(message, /- 강의 요일: 숫자 형식 칸에 글자 항목을 연결했습니다/);
      assert.match(message, /- 교원명: 이 항목의 table로 가는 연결 경로가 둘 이상입니다/);
    });
  });
}

// phase9-11-published-review-contract
{
  // PostgreSQL review GET and INIT share this published-detail response shape.
  function publishedPostgresDetail() {
    return {
      version: 'private-mapping-review-v2',
      database_id: 'campus-pg',
      label: 'Campus',
      status: 'PUBLISHED',
      schema_fingerprint: 'b'.repeat(64),
      ontology_version: 1,
      mapping_version: 2,
      active_mapping_version: 2,
      mapping_versions: [{
        version: 2,
        published_by: 'admin',
        published_at: '2026-10-12T12:39:49Z',
        active: true,
        latest: true,
        activatable: false,
      }],
      review: {
        field_count: 1,
        fields: [{
          semantic_field: 'Course.name',
          decision: 'APPROVED',
          table: 'course_offerings',
          column: 'course_name',
          candidates: [{
            index: 0,
            table: 'course_offerings',
            column: 'course_name',
            confidence: 1,
            review_status: 'AUTO_APPROVE_CANDIDATE',
            source: 'RULES',
          }],
        }],
        allowed_tables: [{ name: 'course_offerings', columns: ['course_name'] }],
      },
    };
  }
  
  describe('phase9-11 published PostgreSQL review contract', () => {
    it('accepts the server shape through both GET and INIT clients', async () => {
      const payload = publishedPostgresDetail();
      const parsed = parseReviewDetail(payload);
      assert.ok(parsed);
      assert.equal(parsed.active_mapping_version, 2);
      assert.deepEqual(parsed.mapping_versions, payload.mapping_versions);
      const caller: ReviewFetchLike = async () => ({
        ok: true, status: 200, json: async () => structuredClone(payload),
      });
      assert.deepEqual(await fetchReviewDetail(caller, 'campus-pg'), parsed);
      assert.deepEqual(await initReview(caller, 'campus-pg', 'admin'), parsed);
    });
  
    it('still requires seconds-only timestamps regardless of activation support', () => {
      for (const changes of [
        { published_at: '2026-10-12T12:39:49.389950Z' },
        { published_at: '2026-10-12T12:39:49.389950Z', activatable: true },
      ]) {
        const payload = publishedPostgresDetail();
        Object.assign(payload.mapping_versions[0], changes);
        assert.equal(parseReviewDetail(payload), null);
      }
    });
  
    it('hides unsupported actions and the unavailable label for an active PostgreSQL version', () => {
      const panel = readFileSync('src/adaptive/MappingReviewPanel.tsx', 'utf-8');
      assert.ok(panel.includes("{entry.active ? ' · 실행 중' : ''}"));
      assert.ok(panel.includes("{!entry.active && !entry.activatable ? ' · 현재 스키마에서 실행 불가' : ''}"));
      assert.match(panel, /\{entry\.activatable \? \([\s\S]*?회귀 확인[\s\S]*?\) : null\}/);
      assert.ok(panel.includes('!entry.active && entry.activatable && pendingVersion !== entry.version'));
      const parsed = parseReviewDetail(publishedPostgresDetail());
      assert.ok(parsed);
      assert.equal(parsed.mapping_versions.filter((entry) => entry.activatable).length, 0);
      assert.equal(parsed.mapping_versions.filter((entry) => !entry.active && !entry.activatable).length, 0);
      assert.equal(parsed.mapping_versions.filter((entry) => !entry.active && entry.activatable).length, 0);
    });
  });
}

// phase9-14-review-row-cleanup
{
  const field: ReviewField = {
    semantic_field: 'CourseOffering.enrolled', decision: 'REMAPPED',
    table: 'lecture', column: 'enrolled',
    candidates: [{ index: 4, table: 'lecture', column: 'capacity', confidence: 0.2, review_status: 'NEEDS_REVIEW', source: 'AI' }],
  };
  const tables = [{ name: 'dept', columns: ['college'] }, { name: 'lecture', columns: ['enrolled', 'capacity'] }];
  
  describe('phase9-14 review row defaults and save guards', () => {
    it('keeps the current connection rather than the first table or recommendation', () => {
      const edit = startRowEdit(field, 'remap');
      assert.equal(edit.table, 'lecture');
      assert.equal(edit.column, 'enrolled');
      assert.equal(canSaveRowEdit({ ...edit, reason: '확인' }, tables, 'reviewer'), true);
    });
    it('uses the recommendation when no connection exists, preserving candidate indices', () => {
      const edit = startRowEdit({ ...field, table: null, column: null, decision: 'PENDING' }, 'remap');
      assert.equal(edit.table, 'lecture');
      assert.equal(edit.column, 'capacity');
      assert.equal(edit.candidateIndex, 4);
    });
    it('leaves selection empty when neither connection nor recommendation exists', () => {
      const edit = startRowEdit({ ...field, table: null, column: null, candidates: [] }, 'remap');
      assert.equal(edit.table, '');
      assert.equal(edit.column, '');
      assert.equal(canSaveRowEdit({ ...edit, reason: '확인' }, tables, 'reviewer'), false);
    });
    it('blocks missing table, column, reason, reviewer and unavailable connections', () => {
      const edit = { ...startRowEdit(field, 'remap'), reason: '확인' };
      for (const invalid of [{ table: '' }, { column: '' }, { reason: '  ' }, { table: 'unknown' }, { column: 'college' }]) {
        assert.equal(canSaveRowEdit({ ...edit, ...invalid }, tables, 'reviewer'), false);
      }
      assert.equal(canSaveRowEdit(edit, tables, '  '), false);
    });
    it('requires a reason for absence without requiring a connection', () => {
      const edit = startRowEdit({ ...field, table: null, column: null, candidates: [] }, 'ignore');
      assert.equal(canSaveRowEdit(edit, tables, 'reviewer'), false);
      assert.equal(canSaveRowEdit({ ...edit, reason: '정보 없음' }, tables, 'reviewer'), true);
    });
    it('starts a fresh draft for each row and each edit mode', () => {
      const first = { ...startRowEdit(field, 'remap'), reason: 'old reason' };
      const second = startRowEdit({ ...field, semantic_field: 'Course.name' }, 'ignore');
      assert.equal(second.semanticField, 'Course.name');
      assert.equal(second.reason, '');
      assert.equal(first.reason, 'old reason');
    });
    it('shares the reviewer with all tools and owns one expanded row in the parent', () => {
      const panel = readFileSync('src/adaptive/MappingReviewPanel.tsx', 'utf8');
      const registration = readFileSync('src/adaptive/PostgresRegistrationPanel.tsx', 'utf8');
      assert.ok(panel.includes('const [rowEdit, setRowEdit]'));
      assert.ok(panel.includes('onEdit={setRowEdit}'));
      assert.ok(panel.includes('reviewTools(reviewer, reviewToolsRevision(detail)'));
      assert.equal((panel.match(/value=\{reviewer\}/g) ?? []).length, 1);
      assert.ok(panel.includes('reviewer={reviewer}'));
      assert.equal((registration.match(/reviewer=\{reviewer\}/g) ?? []).length, 2);
      for (const name of ['ValueDictionaryPanel', 'ExtensionFieldPanel']) {
        const tool = readFileSync(`src/adaptive/${name}.tsx`, 'utf8');
        assert.ok(tool.includes('const reviewer = sharedReviewer ?? localReviewer'));
        assert.ok(tool.includes('sharedReviewer === undefined ?'));
      }
    });
  });
}

// phase9-15-bulk-approve
{
  function field(name: string, decision = 'PENDING', index = 4): ReviewField {
    return {
      semantic_field: name, decision, table: null, column: null,
      candidates: [{ index, table: 'courses', column: 'name', confidence: 0.9, review_status: 'NEEDS_REVIEW', source: 'AI' }],
    };
  }
  
  function payload(fields: ReviewField[]) {
    return {
      version: 'private-mapping-review-v2', database_id: 'bulk-db', label: '검토 DB',
      status: 'PENDING_REVIEW', schema_fingerprint: 'b'.repeat(64), ontology_version: 1,
      review: { field_count: fields.length, fields, allowed_tables: [{ name: 'courses', columns: ['name'] }] },
    };
  }
  
  const fields = [field('Course.name'), field('Course.code', 'PENDING', 7), field('Course.credits', 'PENDING', 2)];
  const options = { fieldName: reviewFieldName, onProgress: () => {} };
  
  describe('phase9-15 bulk approval', () => {
    it('selects only pending fields with AI/rule candidates, using the first actual index', () => {
      const rule = field('Course.code', 'PENDING', 7);
      rule.candidates[0].source = 'RULES';
      const first = field('Course.name');
      first.candidates.push({ ...first.candidates[0], index: 0 });
      const without = { ...field('Schedule.day'), candidates: [] };
      const input = [first, rule, field('Course.credits', 'APPROVED'), field('Department.name', 'REMAPPED'), field('Instructor.name', 'IGNORED'), without];
      const before = structuredClone(input);
      assert.deepEqual(selectBulkApproval(input), {
        targets: [{ semanticField: 'Course.name', candidateIndex: 4 }, { semanticField: 'Course.code', candidateIndex: 7 }],
        withoutCandidates: ['Schedule.day'],
      });
      assert.deepEqual(input, before);
      assert.deepEqual(selectBulkApproval([]), { targets: [], withoutCandidates: [] });
    });
  
    it('confirms target/skipped counts and explains later corrections', () => {
      const selection = selectBulkApproval([...fields, { ...field('Schedule.day'), candidates: [] }]);
      assert.equal(bulkApprovalConfirmation(selection), '대기 중인 3개 항목을 AI 추천대로 승인합니다. 후보가 없는 1개는 그대로 둡니다. 승인 뒤에도 항목마다 [다시 정하기]로 고칠 수 있습니다.');
    });
  
    it('awaits each APPROVE response before sending the next and returns latest table/progress', async () => {
      const calls: unknown[] = [];
      const progress: Array<[number, number]> = [];
      let latest: ReviewDetail | undefined;
      let release!: () => void;
      const gate = new Promise<void>((resolve) => { release = resolve; });
      const current = structuredClone(fields);
      const fetchFn: ReviewFetchLike = async (_url, init) => {
        const body = JSON.parse(init.body!);
        calls.push(body);
        if (calls.length === 1) await gate;
        current.find((entry) => entry.semantic_field === body.semantic_field)!.decision = 'APPROVED';
        return { ok: true, status: 200, json: async () => payload(structuredClone(current)) };
      };
      const running = runBulkApproval(fetchFn, 'bulk-db', 'reviewer', selectBulkApproval(fields), {
        fieldName: reviewFieldName,
        onProgress: (approved, total, detail) => { progress.push([approved, total]); if (detail) latest = detail; },
      });
      await new Promise<void>((resolve) => setImmediate(resolve));
      assert.equal(calls.length, 1);
      release();
      assert.deepEqual(await running, { failed: false, message: '3개 승인 완료. 후보 없는 항목: 없음' });
      assert.deepEqual(calls, fields.map((entry) => ({ action: 'APPROVE', database_id: 'bulk-db', semantic_field: entry.semantic_field, candidate_index: entry.candidates[0].index, reviewer: 'reviewer' })));
      assert.deepEqual(progress, [[0, 3], [1, 3], [2, 3], [3, 3]]);
      assert.ok(latest?.fields.every((entry) => entry.decision === 'APPROVED'));
    });
  
    it('stops on a middle failure, retaining successes and the existing error message', async () => {
      let calls = 0;
      let latest: ReviewDetail | undefined;
      const fetchFn: ReviewFetchLike = async () => {
        calls += 1;
        if (calls === 2) return { ok: false, status: 409, json: async () => ({ error: { code: 'MAPPING_REVIEW_CONFLICT' } }) };
        return { ok: true, status: 200, json: async () => payload([{ ...fields[0], decision: 'APPROVED' }, ...fields.slice(1)]) };
      };
      const result = await runBulkApproval(fetchFn, 'bulk-db', 'reviewer', selectBulkApproval(fields), {
        ...options, onProgress: (_approved, _total, detail) => { if (detail) latest = detail; },
      });
      assert.equal(calls, 2);
      assert.deepEqual(result, { failed: true, message: `1개 승인, 과목 코드 항목에서 실패: ${getReviewErrorMessage('MAPPING_REVIEW_CONFLICT')}` });
      assert.equal(latest?.fields[0].decision, 'APPROVED');
      assert.equal(latest?.fields[1].decision, 'PENDING');
    });
  
    it('stops on transport failure without exposing the raw error', async () => {
      let calls = 0;
      const fetchFn: ReviewFetchLike = async () => { calls += 1; throw new Error('private transport detail'); };
      assert.deepEqual(await runBulkApproval(fetchFn, 'bulk-db', 'reviewer', selectBulkApproval(fields), options), {
        failed: true, message: '0개 승인, 과목명 항목에서 실패: 연결에 실패했습니다. 다시 시도해 주세요.',
      });
      assert.equal(calls, 1);
    });
  
    it('lists skipped fields by name without sending any request for them', async () => {
      const selection = selectBulkApproval([{ ...field('Schedule.day'), candidates: [] }, { ...field('Instructor.name'), candidates: [] }]);
      const fetchFn: ReviewFetchLike = async () => { assert.fail('no targets'); };
      assert.deepEqual(await runBulkApproval(fetchFn, 'bulk-db', 'reviewer', selection, options), {
        failed: false, message: '0개 승인 완료. 후보 없는 항목: 요일, 교수명',
      });
    });
  
    it('stops before the next request when the panel aborts after a success', async () => {
      let calls = 0;
      const controller = new AbortController();
      const fetchFn: ReviewFetchLike = async () => {
        calls += 1;
        return { ok: true, status: 200, json: async () => payload(fields) };
      };
      await assert.rejects(runBulkApproval(fetchFn, 'bulk-db', 'reviewer', selectBulkApproval(fields), {
        ...options, signal: controller.signal,
        onProgress: (approved) => { if (approved === 1) controller.abort(); },
      }), { name: 'AbortError' });
      assert.equal(calls, 1);
    });
  });
}

// phase9-22-admin-style
{
  function withStorage(descriptor: PropertyDescriptor, run: () => void) {
    const original = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
    Object.defineProperty(globalThis, 'localStorage', { configurable: true, ...descriptor });
    try {
      run();
    } finally {
      if (original) Object.defineProperty(globalThis, 'localStorage', original);
      else Reflect.deleteProperty(globalThis, 'localStorage');
    }
  }
  
  it('remembers edits, clearing and reopening across DBs with one application key', () => {
    const values = new Map<string, string>();
    withStorage({ value: {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => values.set(key, value),
    } }, () => {
      assert.equal(readReviewer(), '');
      writeReviewer('검토자 이름');
      assert.equal(readReviewer(), '검토자 이름');
      assert.deepEqual([...values.keys()], [REVIEWER_STORAGE_KEY]);
      writeReviewer('');
      assert.equal(readReviewer(), '');
    });
  });
  
  it('continues without storage when access, reads or writes are blocked', () => {
    const blocked = () => { throw new Error('blocked'); };
    for (const descriptor of [{ get: blocked }, { value: { getItem: blocked, setItem: blocked } }]) {
      withStorage(descriptor, () => {
        assert.equal(readReviewer(), '');
        assert.doesNotThrow(() => writeReviewer('in-memory reviewer'));
      });
    }
    withStorage({ value: undefined }, () => {
      assert.equal(readReviewer(), '');
      assert.doesNotThrow(() => writeReviewer('name'));
    });
  });
  
  it('scrolls to and focuses the reviewer without a second focus scroll', () => {
    const calls: unknown[] = [];
    const input = {
      scrollIntoView: (options: unknown) => calls.push(['scroll', options]),
      focus: (options: unknown) => calls.push(['focus', options]),
    } as unknown as HTMLInputElement;
    focusReviewer(input);
    assert.deepEqual(calls, [
      ['scroll', { block: 'center', behavior: 'smooth' }],
      ['focus', { preventScroll: true }],
    ]);
    assert.doesNotThrow(() => focusReviewer(null));
  });
  
  it('shows only a current blocking reason and follows condition priority', () => {
    assert.equal(disabledTitle([false, 'busy'], [false, 'reviewer']), undefined);
    assert.equal(disabledTitle([false, 'busy'], [true, 'reviewer']), 'reviewer');
    assert.equal(disabledTitle([true, 'busy'], [true, 'reviewer']), 'busy');
  });
  
  it('wires shared reviewer hints and scoped targets for all review tools', () => {
    for (const name of ['MappingReviewPanel', 'ValueDictionaryPanel', 'ExtensionFieldPanel', 'TestQueryPanel']) {
      const source = readFileSync(`src/adaptive/${name}.tsx`, 'utf8');
      assert.match(source, /reviewer\.trim\(\) === '' \? <ReviewerHint \/>/);
      assert.doesNotMatch(source, /먼저 검토자 이름을 넣어 주세요/);
      if (name !== 'TestQueryPanel') {
        assert.match(source, /useState\(readReviewer\)/);
        assert.match(source, /writeReviewer\(/);
        assert.match(source, /data-reviewer-scope/);
        assert.match(source, /data-reviewer-input value=\{reviewer\}/);
      }
    }
    const hint = readFileSync('src/adaptive/ReviewerHint.tsx', 'utf8');
    assert.match(hint, /type="button"/);
    assert.match(hint, /closest\('\[data-reviewer-scope\]'\)/);
    assert.match(hint, /맨 위 ‘검토자’ 칸에 이름을 넣어 주세요/);
  });
  
  it('uses shared styles and explains every explicitly disabled admin button', () => {
    const files = ['PostgresRegistrationPanel', 'MappingReviewPanel', 'MappingReviewRow', 'ValueDictionaryPanel',
      'CodeTableImport', 'ExtensionFieldPanel', 'TestQueryPanel', 'SchemaComparePanel'];
    for (const name of files) {
      const source = readFileSync(`src/adaptive/${name}.tsx`, 'utf8');
      const ast = ts.createSourceFile(name, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
      function visit(node: ts.Node) {
        if (ts.isJsxOpeningElement(node) && node.tagName.getText(ast) === 'button') {
          const attrs = node.attributes.properties;
          const has = (key: string) => attrs.some(a => ts.isJsxAttribute(a) && a.name.getText(ast) === key);
          assert.ok(has('className'), `${name}: button style`);
          if (has('disabled')) assert.ok(has('title'), `${name}: disabled reason`);
        }
        ts.forEachChild(node, visit);
      }
      visit(ast);
    }
  });
}

// phase9-26-dictionary-refresh
{
  it('reads once per changed revision, including initial load, without reading on unrelated renders', () => {
    let previous: string | undefined;
    let calls = 0;
    const read = () => { calls++; };
    for (const revision of ['initial', 'initial', 'approved', 'approved', 'published', 'published']) {
      const action = reviewRefreshAction(previous, revision, false, false);
      if (action === 'load') { previous = revision; read(); }
    }
    assert.equal(calls, 3);
  });
  
  it('dirty revision changes show a notice and preserve the draft until explicit refresh', () => {
    let previous = 'old';
    const draft = { label: 'unsaved label', aliases: ['unsaved alias'] };
    const before = structuredClone(draft);
    let calls = 0;
    const action = reviewRefreshAction(previous, 'new', true, false);
    if (action === 'load') calls++;
    if (action === 'notice') previous = 'new';
    assert.equal(action, 'notice');
    assert.equal(calls, 0);
    assert.deepEqual(draft, before);
    assert.equal(reviewRefreshAction(previous, 'new', false, false), 'none');
    assert.equal(reviewRefreshAction('new', 'newer', true, true), 'notice');
  });
  
  it('defers reads during saving and consumes the revision after saving ends', () => {
    assert.equal(reviewRefreshAction('old', 'new', false, true), 'wait');
    assert.equal(reviewRefreshAction('old', 'new', false, false), 'load');
  });
  
  it('revision detects decisions, remaps, allowed tables and published/active versions', () => {
    const detail: ReviewDetail = {
      database_id: 'synthetic', label: 'Synthetic', status: 'PENDING_REVIEW',
      schema_fingerprint: 'synthetic', ontology_version: 3, mapping_version: null,
      active_mapping_version: null, mapping_versions: [], succession: null,
      allowed_tables: [{ name: 'course', columns: ['name', 'title'] }],
      fields: [{ semantic_field: 'Course.name', decision: 'PENDING', table: null, column: null, candidates: [] }],
    };
    const revision = reviewToolsRevision(detail);
    assert.equal(revision, reviewToolsRevision(structuredClone(detail)));
    assert.notEqual(revision, reviewToolsRevision(null));
    for (const change of [
      { fields: [{ ...detail.fields[0]!, decision: 'APPROVED' as const, table: 'course', column: 'name' }] },
      { fields: [{ ...detail.fields[0]!, decision: 'REMAPPED' as const, table: 'course', column: 'title' }] },
      { fields: [{ ...detail.fields[0]!, decision: 'IGNORED' as const }] },
      { mapping_version: 1 }, { active_mapping_version: 2 }, { allowed_tables: [] },
    ]) assert.notEqual(revision, reviewToolsRevision({ ...detail, ...change }));
  });
  
  it('both panels use the refresh guard, cancellable reads and notices; tool publishing refreshes mapping detail', () => {
    for (const name of ['ValueDictionaryPanel', 'ExtensionFieldPanel']) {
      const source = readFileSync(`src/adaptive/${name}.tsx`, 'utf8');
      assert.ok(source.includes('useReviewToolsRefresh(reviewRevision, dirty, busy, load)'));
      assert.ok(source.includes('if (signal.aborted) return;'));
      assert.ok(source.includes('검토 내용이 바뀌었습니다.'));
      assert.ok(source.includes('onClick={() => void refresh()}'));
    }
    const parent = readFileSync('src/adaptive/PostgresRegistrationPanel.tsx', 'utf8');
    assert.equal((parent.match(/reviewRevision=\{revision\} onPublished=\{onToolsPublished\}/g) ?? []).length, 2);
    const review = readFileSync('src/adaptive/MappingReviewPanel.tsx', 'utf8');
    assert.ok(review.includes('setToolsPublishEpoch((value) => value + 1)'));
    assert.ok(review.includes('isBusy(status) || toolsPublishEpoch === refreshedToolsPublish.current'));
    assert.ok(review.includes('void fetchReviewDetail(caller, databaseId, { signal: controller.signal }).then'));
    assert.ok(review.includes('setPublishedVersion(next.mapping_version)'));
  });
}
