import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { LEGACY_SELECTION_ID, mainScreenDatabases, STORAGE_KEY, canonicalToCourse, catalogToCourses, databaseLabel, isDatabaseId, parseCatalogResponse, parseDatabaseRegistry, parsePrototypeDatabaseParam, readStoredDatabaseId, resolveDatabaseSelection, shouldResetCourseUi, writeStoredDatabaseId, REGISTRY_VERSION, RESERVED_DATABASE_ID_MESSAGE, DATABASE_ID_INPUT_MESSAGE, registrationDatabaseIdMessage } from '../../src/adaptive/databases.ts'
import { getCatalog, listDatabases } from '../../src/adaptive/catalogClient.ts'
import type { CatalogFetchLike } from '../../src/adaptive/catalogClient.ts'
import { startClarification } from '../../src/adaptive/client.ts'
import { readFileSync } from 'node:fs'
import { SQLITE_UPLOAD_CONTENT_TYPE, SQLITE_UPLOAD_PATH, SQLITE_UPLOAD_VERSION, UPLOAD_PENDING_NOTICE, canStartUpload, getUploadErrorMessage, isValidUploadLabel, parseUploadError, parseUploadResponse, toUploadMessage, uploadSqliteDatabase, SqliteUploadClientError, UPLOAD_CONTRACT_MESSAGE } from '../../src/adaptive/sqliteUpload.ts'
import type { UploadFetchLike } from '../../src/adaptive/sqliteUpload.ts'
import { isStaleResponse, nextOperation } from '../../src/adaptive/state.ts'
import { POSTGRES_REGISTER_PATH, PostgresRegistrationClientError, fetchPostgresSchemas, parseSchemas, parseSources, registerPostgresSchema, toPostgresMessage, parseRegistrations, canDeleteRegistration, deletePostgresRegistration, fetchRegistrations, POSTGRES_REGISTER_VERSION } from '../../src/adaptive/postgresRegistration.ts'
import type { PgFetchLike } from '../../src/adaptive/postgresRegistration.ts'
import { ignoreReviewField, PUBLISH_RUNTIME_BLOCKED_MESSAGE, publishReview, toReviewMessage } from '../../src/adaptive/mappingReview.ts'
import type { ReviewFetchLike } from '../../src/adaptive/mappingReview.ts'

// phase6-16-databases
{
  const FLAT_ID = 'synth-clarification-prototype';
  const LEGACY_ID = 'synth-clarification-legacy';
  
  function registryPayload() {
    return {
      version: 'synthetic-database-registry-v1',
      databases: [
        { database_id: FLAT_ID, label: 'Synthetic flat (2026 seed)', schema: 'flat' },
        { database_id: LEGACY_ID, label: 'Synthetic legacy (2026 seed)', schema: 'legacy' },
      ],
    };
  }
  
  function catalogPayload() {
    return {
      version: 'synthetic-course-catalog-v1',
      database_id: FLAT_ID,
      items: [
        {
          course_code: 'CSE101',
          course_name: '프로그래밍기초',
          credits: 3,
          category: '전공',
          department_code: 'CSE',
          department_name: '컴퓨터공학과',
          instructor_name: '교원A',
          year: 2026,
          semester: '1',
          section: '01',
          capacity: 40,
          enrolled: 32,
          day: 'MON',
          start_time: '09:00',
          end_time: '10:30',
        },
      ],
      total: 1,
    };
  }
  
  describe('phase6-16 databases parser', () => {
    it('Phase 9-37: hides development databases from main-screen selection', () => {
      const entries = [
        { database_id: FLAT_ID, label: 'Synthetic flat (2026 seed)', schema: 'flat' as const },
        { database_id: LEGACY_ID, label: 'Synthetic legacy (2026 seed)', schema: 'legacy' as const },
        { database_id: 'legacy-base', label: 'Legacy DB tables (QueryIR)', schema: 'private' as const },
        { database_id: 'legacy-view', label: 'Legacy DB (QueryIR)', schema: 'postgres' as const },
        { database_id: 'campus-pg', label: 'Campus PostgreSQL', schema: 'postgres' as const },
        { database_id: 'uploaded-sqlite', label: 'Uploaded SQLite', schema: 'private' as const },
      ];
      const visible = mainScreenDatabases(entries);
      assert.deepEqual(visible.map((entry) => entry.database_id), ['legacy-view', 'campus-pg', 'uploaded-sqlite']);
      const storage = new Map([[STORAGE_KEY, FLAT_ID]]);
      assert.equal(readStoredDatabaseId({ getItem: (key) => storage.get(key) ?? null, setItem: () => {} }, visible.map((entry) => entry.database_id)), LEGACY_SELECTION_ID);
    });
  
    it('accepts strict top-level registry and skips invalid entries', () => {
      assert.equal(parseDatabaseRegistry(registryPayload())?.length, 2);
      assert.equal(parseDatabaseRegistry({ version: 'synthetic-database-registry-v1', databases: [], extra: 1 }), null);
      assert.deepEqual(
        parseDatabaseRegistry({
          version: 'synthetic-database-registry-v1',
          databases: [{ database_id: FLAT_ID, label: 'x', schema: 'weird' }],
        }),
        [],
      );
      assert.deepEqual(
        parseDatabaseRegistry({
          version: 'synthetic-database-registry-v1',
          databases: [{ database_id: 'legacy', label: 'Collision', schema: 'flat' }],
        }),
        [],
      );
      assert.equal(
        parseDatabaseRegistry({
          version: 'synthetic-database-registry-v1',
          databases: [
            { database_id: FLAT_ID, label: 'Flat', schema: 'flat' },
            { database_id: FLAT_ID, label: 'Duplicate', schema: 'legacy' },
          ],
        }),
        null,
      );
      assert.equal(isDatabaseId('BAD ID'), false);
      assert.equal(isDatabaseId(FLAT_ID), true);
    });
  
    it('validates catalog rows, total, and key uniqueness', () => {
      assert.equal(parseCatalogResponse(catalogPayload())?.total, 1);
      assert.equal(parseCatalogResponse({ ...catalogPayload(), total: 2 }), null);
      assert.equal(parseCatalogResponse({ ...catalogPayload(), version: 'x' }), null);
      assert.equal(parseCatalogResponse({ ...catalogPayload(), injected: 'secret' }), null);
      const bad = catalogPayload() as unknown as { items: Array<Record<string, unknown>>; total: number; version: string; database_id: string };
      assert.equal(parseCatalogResponse({ ...bad, items: [{ ...bad.items[0], enrolled: 99999 }] }), null);
      assert.equal(parseCatalogResponse({ ...catalogPayload(), items: [{ ...catalogPayload().items[0], enrolled: 41 }] })?.total, 1);
    });
  
    it('Phase 9-36: accepts the expanded response cap and rejects above it', () => {
      const item = catalogPayload().items[0]!;
      const items = Array.from({ length: 1000 }, (_, index) => ({
        ...item,
        course_code: `C${String(index).padStart(5, '0')}`,
      }));
      assert.equal(parseCatalogResponse({ ...catalogPayload(), items, total: items.length })?.total, 1000);
      assert.equal(parseCatalogResponse({ ...catalogPayload(), items: [...items, { ...item, course_code: 'OVER' }], total: 1001 }), null);
    });
  
    it('accepts null only for optional fields and omits courses without time from timetable slots', () => {
      const base = catalogPayload().items[0]!;
      const optionalNulls = { ...base, instructor_name: null, day: null, start_time: null, end_time: null, credits: null, capacity: null, enrolled: null };
      const parsed = parseCatalogResponse({ ...catalogPayload(), items: [optionalNulls] });
      assert.ok(parsed);
      const course = catalogToCourses(parsed.items)[0]!;
      assert.equal(course.professor, '-');
      assert.equal(course.timeText, '-');
      assert.equal(course.credits, 0);
      assert.equal(course.capacity, undefined);
      assert.equal(course.enrolled, undefined);
      assert.deepEqual(course.slots, []);
      assert.equal(parseCatalogResponse({ ...catalogPayload(), items: [{ ...base, course_code: null }] }), null);
      assert.equal(parseCatalogResponse({ ...catalogPayload(), items: [{ ...base, course_name: null }] }), null);
    });
  
    it('accepts the retired year as null while preserving the catalog contract', () => {
      const payload = catalogPayload();
      const parsed = parseCatalogResponse({ ...payload, items: [{ ...payload.items[0], year: null }] });
      assert.ok(parsed);
      assert.equal(parsed.items[0]!.year, null);
      assert.ok(canonicalToCourse(parsed.items[0]!).id);
    });
  
    it('maps canonical DTO to legacy Course without leaking internals', () => {
      const parsed = parseCatalogResponse(catalogPayload());
      assert.ok(parsed !== null);
      const course = canonicalToCourse(parsed.items[0]!);
      assert.equal(course.id, '2026-1-CSE101-01');
      assert.equal(course.name, '프로그래밍기초');
      assert.equal(course.professor, '교원A');
      assert.equal(course.timeText, 'MON 09:00-10:30');
    });
  
    it('resolves storage fail-closed to legacy', () => {
      assert.equal(resolveDatabaseSelection('corrupt id!!', [FLAT_ID]), LEGACY_SELECTION_ID);
      assert.equal(resolveDatabaseSelection('unknown-db', [FLAT_ID]), LEGACY_SELECTION_ID);
      assert.equal(resolveDatabaseSelection(FLAT_ID, [FLAT_ID]), FLAT_ID);
      const store = new Map<string, string>();
      const storage = {
        getItem: (key: string) => store.get(key) ?? null,
        setItem: (key: string, value: string) => void store.set(key, value),
      };
      writeStoredDatabaseId(storage, FLAT_ID, [FLAT_ID]);
      assert.equal(store.get(STORAGE_KEY), FLAT_ID);
      assert.equal(readStoredDatabaseId(storage, [FLAT_ID]), FLAT_ID);
      assert.equal(readStoredDatabaseId(storage, [LEGACY_ID]), LEGACY_SELECTION_ID);
      assert.equal(
        readStoredDatabaseId(
          {
            getItem: () => {
              throw new Error('storage denied');
            },
            setItem: () => {
              throw new Error('storage denied');
            },
          },
          [FLAT_ID],
        ),
        LEGACY_SELECTION_ID,
      );
      store.set(STORAGE_KEY, 'broken!!');
      assert.equal(readStoredDatabaseId(storage, [FLAT_ID]), LEGACY_SELECTION_ID);
      assert.equal(shouldResetCourseUi(FLAT_ID, LEGACY_ID), true);
      assert.equal(shouldResetCourseUi(FLAT_ID, FLAT_ID), false);
      assert.equal(databaseLabel(registryPayload().databases, LEGACY_SELECTION_ID), '\uae30\ubcf8 DB');
      assert.equal(databaseLabel([{ database_id: 'legacy-view', label: 'Legacy DB (QueryIR)', schema: 'postgres' }], 'legacy-view'), '\uae30\ubcf8 DB (\uc801\uc751\ud615 \ucc57\ubd07)');
      assert.equal(parsePrototypeDatabaseParam(`?database=${FLAT_ID}`), FLAT_ID);
      assert.equal(parsePrototypeDatabaseParam('?database=BAD!!'), null);
      assert.equal(parsePrototypeDatabaseParam(`?database=${FLAT_ID}&database=${LEGACY_ID}`), null);
      assert.equal(parsePrototypeDatabaseParam(`?database=${FLAT_ID}&redirect=/`), null);
    });
  });
  
  describe('phase6-16 catalog client', () => {
    it('lists databases with GET and no auth header', async () => {
      const calls: string[] = [];
      const fetchFn: CatalogFetchLike = async (url, init) => {
        calls.push(url);
        assert.equal(init.method, 'GET');
        assert.deepEqual(init.headers, { Accept: 'application/json' });
        return { ok: true, status: 200, json: async () => registryPayload() };
      };
      const out = await listDatabases(fetchFn);
      assert.equal(out.length, 2);
      assert.deepEqual(calls, ['/api/adaptive/v1/databases']);
    });
  
    it('requests catalog with database_id and keyword, checks echoed id', async () => {
      const seen: string[] = [];
      const fetchFn: CatalogFetchLike = async (url) => {
        seen.push(url);
        return { ok: true, status: 200, json: async () => catalogPayload() };
      };
      const out = await getCatalog(fetchFn, FLAT_ID, { keyword: '프로그래밍' });
      assert.equal(out.database_id, FLAT_ID);
      assert.ok(seen[0]?.includes('database_id=synth-clarification-prototype'));
      assert.ok(seen[0]?.includes('keyword='));
      await assert.rejects(() => getCatalog(fetchFn, 'BAD ID', {}));
    });
  
    it('does not reflect raw server error bodies or unexpected error fields', async () => {
      const secret = 'SECRET_PATH_sqlite_SELECT';
      const fetchFn: CatalogFetchLike = async () => ({
        ok: false,
        status: 500,
        json: async () => ({ error: { code: 'INTERNAL_ERROR', detail: secret } }),
      });
      await assert.rejects(
        () => getCatalog(fetchFn, FLAT_ID),
        (error: unknown) => {
          assert.ok(error instanceof Error);
          assert.equal(error.message.includes(secret), false);
          assert.equal(String(error).includes(secret), false);
          return true;
        },
      );
    });
  
    it('sends dynamic database_id in clarification start', async () => {
      let body: Record<string, unknown> = {};
      const fetchFn = async (_url: string, init: { body: string }) => {
        body = JSON.parse(init.body) as Record<string, unknown>;
        return { ok: true, status: 200, json: async () => ({ version: 'clarification-http-v1', status: 'NOT_REQUIRED' }) };
      };
      await startClarification(fetchFn, '강의 보여줘', { requestId: 'req-616', databaseId: LEGACY_ID });
      assert.equal(body['database_id'], LEGACY_ID);
      await assert.rejects(
        () => startClarification(fetchFn, '강의 보여줘', { requestId: 'req-bad', databaseId: 'bad id' }),
      );
    });
  });
}

// phase6-18-sqlite-upload
{
  function successPayload() {
    return {
      version: 'private-sqlite-upload-v2',
      database_id: 'review-db-one',
      label: 'Review DB',
      status: 'PENDING_REVIEW',
      schema_fingerprint: 'a'.repeat(64),
      schema: {
        table_count: 1,
        column_count: 2,
        foreign_key_count: 0,
        tables: [
          {
            name: 'course_offerings',
            column_count: 2,
            columns: [
              {
                name: 'course_code',
                declared_type: 'TEXT',
                affinity: 'TEXT',
                nullable: false,
                primary_key_position: 1,
              },
              {
                name: 'course_name',
                declared_type: 'TEXT',
                affinity: 'TEXT',
                nullable: false,
                primary_key_position: 0,
              },
            ],
            primary_key: ['course_code'],
            foreign_key_count: 0,
          },
        ],
      },
      mapping: {
        ontology_version: 1,
        proposer: 'AI',
        field_count: 1,
        unmapped_count: 0,
        unmapped_fields: [],
        fields: [
          {
            semantic_field: 'Course.name',
            review_status: 'NEEDS_REVIEW',
            top_candidate: {
              table: 'course_offerings',
              column: 'course_name',
              confidence: 0.8,
              reasons: [
                { signal: 'COLUMN_NAME', score: 0.8, detail: 'name overlap' },
                { signal: 'AI_PROPOSAL', score: 1, detail: 'local model chose this column (ai-mapping-v2)' },
              ],
            },
          },
        ],
      },
    };
  }
  
  describe('phase6-18 upload parser and request contract', () => {
    it('accepts strict success and rejects extra keys/type/version/status', () => {
      const parsed = parseUploadResponse(successPayload());
      assert.ok(parsed !== null);
      assert.equal(parsed.status, 'PENDING_REVIEW');
      assert.equal(parsed.schema.table_count, 1);
      assert.equal(parsed.mapping.fields[0]?.semantic_field, 'Course.name');
      assert.equal(parseUploadResponse({ ...successPayload(), version: 'x' }), null);
      assert.equal(parseUploadResponse({ ...successPayload(), status: 'PUBLISHED' }), null);
      assert.equal(parseUploadResponse({ ...successPayload(), extra: 1 }), null);
      assert.equal(parseUploadResponse({ ...successPayload(), database_id: 'BAD ID' }), null);
      assert.equal(
        parseUploadResponse({ ...successPayload(), schema_fingerprint: 'short' }),
        null,
      );
      assert.equal(
        parseUploadResponse({
          ...successPayload(),
          schema: { ...successPayload().schema, column_count: -1 },
        }),
        null,
      );
      const invalidStatus = structuredClone(successPayload());
      invalidStatus.mapping.fields[0]!.review_status = 'PUBLISHED';
      assert.equal(parseUploadResponse(invalidStatus), null);
      const invalidConfidence = structuredClone(successPayload());
      invalidConfidence.mapping.fields[0]!.top_candidate!.confidence = 1.5;
      assert.equal(parseUploadResponse(invalidConfidence), null);
      const withSamples = structuredClone(successPayload()) as unknown as {
        schema: { tables: Array<{ columns: unknown[] }> };
      };
      withSamples.schema.tables[0]!.columns.push({
        name: 'injected',
        declared_type: 'TEXT',
        affinity: 'TEXT',
        nullable: true,
        primary_key_position: 0,
        sample_values: ['row-value'],
      });
      assert.equal(parseUploadResponse(withSamples), null);
    });
  
    it('phase8-12: requires a known proposer and accepts the AI reason', () => {
      const parsed = parseUploadResponse(successPayload());
      assert.equal(parsed?.mapping.proposer, 'AI');
      assert.equal(parsed?.mapping.fields[0]?.top_candidate?.reasons[1]?.signal, 'AI_PROPOSAL');
      for (const proposer of ['RULES', 'AI_UNAVAILABLE']) {
        const payload = structuredClone(successPayload());
        payload.mapping.proposer = proposer;
        assert.equal(parseUploadResponse(payload)?.mapping.proposer, proposer);
      }
      for (const proposer of [undefined, 'MODEL', 1]) {
        const payload = structuredClone(successPayload()) as unknown as { mapping: Record<string, unknown> };
        if (proposer === undefined) delete payload.mapping['proposer'];
        else payload.mapping['proposer'] = proposer;
        assert.equal(parseUploadResponse(payload), null);
      }
      const unknownSignal = structuredClone(successPayload());
      unknownSignal.mapping.fields[0]!.top_candidate!.reasons[1]!.signal = 'MODEL_GUESS';
      assert.equal(parseUploadResponse(unknownSignal), null);
    });
  
    it('parses fixed error shape and rejects detail reflection', () => {
      assert.deepEqual(parseUploadError({ error: { code: 'SQLITE_UPLOAD_CONFLICT' } }), {
        code: 'SQLITE_UPLOAD_CONFLICT',
      });
      assert.equal(
        parseUploadError({ error: { code: 'SQLITE_UPLOAD_CONFLICT', detail: 'secret' } }),
        null,
      );
      assert.equal(parseUploadError({ error: { code: 'lowercase' } }), null);
      assert.equal(isValidUploadLabel('   '), false);
      assert.equal(isValidUploadLabel('ok label'), true);
    });
  
    it('sends raw Blob body with fixed content type and strict query', async () => {
      const seen: Array<{ url: string; method: string; headers: Record<string, string>; body: unknown }> = [];
      const file = new Blob(['SQLite format 3\x00data'], { type: SQLITE_UPLOAD_CONTENT_TYPE });
      const fetchFn: UploadFetchLike = async (url, init) => {
        seen.push({ url, method: init.method, headers: init.headers, body: init.body });
        return { ok: true, status: 200, json: async () => successPayload() };
      };
      const out = await uploadSqliteDatabase(fetchFn, 'review-db-one', 'Review DB', file);
      assert.equal(out.database_id, 'review-db-one');
      assert.equal(seen.length, 1);
      assert.equal(seen[0]?.method, 'POST');
      assert.equal(seen[0]?.headers['Content-Type'], SQLITE_UPLOAD_CONTENT_TYPE);
      assert.ok(seen[0]?.url.startsWith(`${SQLITE_UPLOAD_PATH}?`));
      assert.ok(seen[0]?.url.includes('database_id=review-db-one'));
      assert.ok(seen[0]?.url.includes('label=Review+DB'));
      assert.ok(seen[0]?.body instanceof Blob);
      await assert.rejects(() => uploadSqliteDatabase(fetchFn, 'BAD ID', 'Review DB', file));
      await assert.rejects(() => uploadSqliteDatabase(fetchFn, 'review-db-one', '   ', file));
    });
  
    it('never reflects server bodies or exception strings', async () => {
      const secret = 'SECRET_PATH_sqlite_SELECT_traceback';
      const fetchFn: UploadFetchLike = async () => ({
        ok: false,
        status: 500,
        json: async () => ({ error: { code: 'SQLITE_UPLOAD_INTERNAL_ERROR', detail: secret } }),
      });
      const file = new Blob(['x'], { type: SQLITE_UPLOAD_CONTENT_TYPE });
      await assert.rejects(
        () => uploadSqliteDatabase(fetchFn, 'review-db-one', 'Review DB', file),
        (error: unknown) => {
          assert.ok(error instanceof Error);
          assert.equal(error.message.includes(secret), false);
          assert.equal(String(error).includes(secret), false);
          assert.equal(toUploadMessage(error).includes(secret), false);
          return true;
        },
      );
      assert.equal(getUploadErrorMessage('UNKNOWN_CODE_XYZ'), '처리 중 오류가 발생했습니다. 다시 시도해 주세요.');
      assert.ok(UPLOAD_PENDING_NOTICE.includes('검토 전에는 조회·질의에 사용할 수 없음'));
      assert.equal(SQLITE_UPLOAD_VERSION, 'private-sqlite-upload-v2');
    });
  });
  
  describe('phase6-18 upload state transitions', () => {
    it('blocks duplicate submit while uploading and allows retry', () => {
      assert.equal(canStartUpload('idle'), true);
      assert.equal(canStartUpload('uploading'), false);
      assert.equal(canStartUpload('success'), true);
      assert.equal(canStartUpload('error'), true);
    });
  
    it('propagates abort signal and maps abort to transport without stale overwrite', async () => {
      const controller = new AbortController();
      let observedSignal: AbortSignal | undefined;
      const fetchFn: UploadFetchLike = async (_url, init) => {
        observedSignal = init.signal;
        controller.abort();
        throw new DOMException('aborted', 'AbortError');
      };
      const file = new Blob(['x'], { type: SQLITE_UPLOAD_CONTENT_TYPE });
      await assert.rejects(
        () => uploadSqliteDatabase(fetchFn, 'review-db-one', 'Review DB', file, { signal: controller.signal }),
        (error: unknown) => {
          assert.equal(toUploadMessage(error), '연결에 실패했습니다. 다시 시도해 주세요.');
          return true;
        },
      );
      assert.equal(observedSignal, controller.signal);
      // stale responses must not overwrite the latest UI operation
      const current = nextOperation(0);
      const stale = 0;
      assert.equal(isStaleResponse(current, stale), true);
      assert.equal(isStaleResponse(current, current), false);
    });
  });
  
  describe('phase6-18 upload isolation in MainPage', () => {
    const source = readFileSync('src/pages/MainPage.tsx', 'utf-8').replace(/\r\n/g, '\n');
  
    it('resets an aborted upload when the database modal closes', () => {
      const start = source.indexOf('function closeDbModal');
      const end = source.indexOf('\n  }\n', start);
      const handler = source.slice(start, end);
      assert.ok(handler.includes('uploadAbortRef.current?.abort()'));
      assert.ok(handler.includes("setUploadStatus('idle')"));
      assert.ok(handler.includes('setUploadResult(null)'));
      assert.ok(handler.includes('setUploadError(null)'));
    });
  
    it('keeps selection, storage, and course UI untouched on upload success', () => {
      const start = source.indexOf('async function submitSqliteUpload');
      const end = source.indexOf('\n  }\n', start);
      const handler = source.slice(start, end);
      assert.ok(handler.includes('setUploadResult(result)'));
      for (const forbidden of [
        'setActiveDbId(',
        'setPendingDbId(',
        'setRegistry(',
        'writeStoredDatabaseId(',
        'setAllCourses(',
        'setSelectedCourses(',
      ]) {
        assert.equal(handler.includes(forbidden), false, forbidden);
      }
      assert.ok(source.includes('{UPLOAD_PENDING_NOTICE}'));
      assert.ok(source.includes('PENDING_REVIEW'));
    });
  
    it('does not add the registered DB to selection radios or clarification link', () => {
      const start = source.indexOf('SQLite 등록(검토 대기)');
      const end = source.indexOf('</fieldset>', start);
      const uploadSection = source.slice(start, end);
      assert.ok(uploadSection.includes('setUploadResult') || source.includes('setUploadResult(result)'));
      // radios render strictly from the server-owned registry snapshot
      assert.ok(source.includes('{registry.map((entry) => ('));
      assert.equal(uploadSection.includes('setRegistry('), false);
      assert.equal(uploadSection.includes('/adaptive-prototype?database='), false);
    });
  });
}

// phase7-06-postgres-selection
{
  const PG_ID = 'synth-pg-main';
  const FLAT_ID = 'synth-clarification-prototype';
  const PRIVATE_ID = 'review-db-one';
  
  function pgRegistry() {
    return {
      version: 'synthetic-database-registry-v1',
      databases: [
        { database_id: FLAT_ID, label: 'Synthetic flat (2026 seed)', schema: 'flat' },
        { database_id: PG_ID, label: 'Synthetic pg (2026 seed)', schema: 'postgres' },
      ],
    };
  }
  
  describe('phase7-06 postgres registry parser', () => {
    it('accepts a strict postgres entry', () => {
      const parsed = parseDatabaseRegistry(pgRegistry());
      assert.equal(parsed?.length, 2);
      assert.equal(parsed?.[1]?.schema, 'postgres');
      assert.equal(parsed?.[1]?.database_id, PG_ID);
    });
  
    it('rejects extra keys, unknown schema, invalid id, duplicate id', () => {
      for (const invalid of [
        { database_id: PG_ID, label: 'Pg', schema: 'postgres', extra: 1 },
        { database_id: PG_ID, label: 'Pg', schema: 'postgresql' },
        { database_id: 'BAD ID', label: 'Pg', schema: 'postgres' },
      ]) {
        const valid = pgRegistry().databases[0]!;
        const warnings: Array<string | null> = [];
        assert.deepEqual(parseDatabaseRegistry({
          version: 'synthetic-database-registry-v1',
          databases: [invalid, valid],
        }, (message) => warnings.push(message)), [valid]);
        assert.deepEqual(warnings, ['형식이 올바르지 않은 DB 1개를 목록에서 뺐습니다.']);
      }
      assert.equal(
        parseDatabaseRegistry({
          version: 'synthetic-database-registry-v1',
          databases: [
            { database_id: PG_ID, label: 'Pg one', schema: 'postgres' },
            { database_id: PG_ID, label: 'Pg two', schema: 'postgres' },
          ],
        }),
        null,
      );
    });
  
    it('rejects unsafe, unicode, control, blank, and too-long postgres labels', () => {
      for (const label of [
        '',
        '   ',
        '한글라벨',
        'Pg\nlabel',
        'Pg\x00label',
        'Pg; DROP TABLE',
        'Pg"quote',
        "Pg'quote",
        'Pg/path',
        'Pg\\path',
        'Pg$label',
        'a'.repeat(65),
      ]) {
        const valid = pgRegistry().databases[0]!;
        const warnings: Array<string | null> = [];
        assert.deepEqual(
          parseDatabaseRegistry({
            version: 'synthetic-database-registry-v1',
            databases: [{ database_id: PG_ID, label, schema: 'postgres' }, valid],
          }, (message) => warnings.push(message)),
          [valid],
          `label should be rejected: ${JSON.stringify(label)}`,
        );
        assert.deepEqual(warnings, ['형식이 올바르지 않은 DB 1개를 목록에서 뺐습니다.']);
      }
    });
  
    it('does not extend the private Korean-label exception to postgres', () => {
      assert.ok(
        parseDatabaseRegistry({
          version: 'synthetic-database-registry-v1',
          databases: [{ database_id: PRIVATE_ID, label: '검토용 강의 DB', schema: 'private' }],
        }) !== null,
      );
      const valid = { database_id: PRIVATE_ID, label: '검토용 강의 DB', schema: 'private' };
      const warnings: Array<string | null> = [];
      assert.deepEqual(
        parseDatabaseRegistry({
          version: 'synthetic-database-registry-v1',
          databases: [{ database_id: PG_ID, label: '검토용 강의 DB', schema: 'postgres' }, valid],
        }, (message) => warnings.push(message)),
        [valid],
      );
      assert.deepEqual(warnings, ['형식이 올바르지 않은 DB 1개를 목록에서 뺐습니다.']);
    });
  
    it('keeps flat and legacy contracts unchanged', () => {
      assert.ok(
        parseDatabaseRegistry({
          version: 'synthetic-database-registry-v1',
          databases: [{ database_id: FLAT_ID, label: 'Flat label (v1)', schema: 'flat' }],
        }) !== null,
      );
      const valid = pgRegistry().databases[1]!;
      const warnings: Array<string | null> = [];
      assert.deepEqual(
        parseDatabaseRegistry({
          version: 'synthetic-database-registry-v1',
          databases: [{ database_id: FLAT_ID, label: '한글', schema: 'flat' }, valid],
        }, (message) => warnings.push(message)),
        [valid],
      );
      assert.deepEqual(warnings, ['형식이 올바르지 않은 DB 1개를 목록에서 뺐습니다.']);
    });
  });
  
  describe('phase7-06 postgres selection storage', () => {
    it('restores a stored postgres id only while it remains in the registry', () => {
      assert.equal(resolveDatabaseSelection(PG_ID, [FLAT_ID, PG_ID]), PG_ID);
      assert.equal(resolveDatabaseSelection(PG_ID, [FLAT_ID]), LEGACY_SELECTION_ID);
      assert.equal(resolveDatabaseSelection('corrupt id!!', [PG_ID]), LEGACY_SELECTION_ID);
      assert.equal(databaseLabel(pgRegistry().databases, PG_ID), 'Synthetic pg (2026 seed)');
    });
  
    it('persists only validated postgres ids in sessionStorage', () => {
      const store = new Map<string, string>();
      const storage = {
        getItem: (key: string) => store.get(key) ?? null,
        setItem: (key: string, value: string) => void store.set(key, value),
      };
      writeStoredDatabaseId(storage, PG_ID, [FLAT_ID, PG_ID]);
      assert.equal(store.get(STORAGE_KEY), PG_ID);
      assert.equal(readStoredDatabaseId(storage, [FLAT_ID, PG_ID]), PG_ID);
      assert.equal(readStoredDatabaseId(storage, [FLAT_ID]), LEGACY_SELECTION_ID);
      store.set(STORAGE_KEY, 'pg; DROP');
      assert.equal(readStoredDatabaseId(storage, [FLAT_ID, PG_ID]), LEGACY_SELECTION_ID);
      writeStoredDatabaseId(storage, 'BAD ID', [FLAT_ID, PG_ID]);
      assert.notEqual(store.get(STORAGE_KEY), 'BAD ID');
    });
  });
  
  describe('phase7-06 postgres catalog and clarification wiring', () => {
    it('sends the postgres id as a query parameter and rejects echoed mismatches', async () => {
      const seen: string[] = [];
      const fetchFn: CatalogFetchLike = async (url) => {
        seen.push(url);
        return {
          ok: true,
          status: 200,
          json: async () => ({ version: 'synthetic-course-catalog-v1', database_id: PG_ID, items: [], total: 0 }),
        };
      };
      const out = await getCatalog(fetchFn, PG_ID);
      assert.equal(out.database_id, PG_ID);
      assert.ok(seen[0]?.includes(`database_id=${PG_ID}`));
      const mismatch: CatalogFetchLike = async () => ({
        ok: true,
        status: 200,
        json: async () => ({ version: 'synthetic-course-catalog-v1', database_id: FLAT_ID, items: [], total: 0 }),
      });
      await assert.rejects(() => getCatalog(mismatch, PG_ID));
    });
  
    it('sends the same postgres id in the clarification start body', async () => {
      let body: Record<string, unknown> = {};
      const fetchFn = async (_url: string, init: { body: string }) => {
        body = JSON.parse(init.body) as Record<string, unknown>;
        return { ok: true, status: 200, json: async () => ({ version: 'clarification-http-v1', status: 'NOT_REQUIRED' }) };
      };
      await startClarification(fetchFn, '강의 보여줘', { requestId: 'req-pg-706', databaseId: PG_ID });
      assert.equal(body['database_id'], PG_ID);
    });
  
    it('keeps prototype URL parsing fail-closed for invalid and unknown ids', () => {
      assert.equal(parsePrototypeDatabaseParam(`?database=${PG_ID}`), PG_ID);
      assert.equal(parsePrototypeDatabaseParam('?database=BAD!!'), null);
      assert.equal(parsePrototypeDatabaseParam(`?database=${PG_ID}&database=${FLAT_ID}`), null);
      assert.equal(parsePrototypeDatabaseParam(`?database=${PG_ID}&redirect=/`), null);
      const registryIds = (pgRegistry().databases as Array<{ database_id: string }>).map((entry) => entry.database_id);
      assert.ok(registryIds.includes(PG_ID));
      assert.ok(!registryIds.includes('unknown-pg-db'));
    });
  });
  
  describe('phase7-06 MainPage and prototype selection regression', () => {
    it('renders server-owned registry radios with explicit apply, reset, and same-id link', () => {
      const main = readFileSync('src/pages/MainPage.tsx', 'utf-8');
      assert.ok(main.includes('{registry.map((entry) => ('));
      assert.ok(main.includes('setPendingDbId(entry.database_id)'));
      assert.ok(main.includes('적용'));
      assert.ok(main.includes('applyDatabase(pendingDbId)'));
      assert.ok(main.includes('shouldResetCourseUi'));
      assert.ok(main.includes('setSelectedCourses([])'));
      assert.ok(main.includes('abortRef.current?.abort()'));
      assert.ok(main.includes('writeStoredDatabaseId(window.sessionStorage, resolved, known)'));
      assert.ok(main.includes('readStoredDatabaseId(window.sessionStorage, known)'));
      // Phase 9-37: the main screen filters development-only registry entries at bootstrap.
      assert.ok(main.includes('mainScreenDatabases(await listDatabases(adaptiveFetch'));
      assert.ok(!main.includes('synth-pg-main'));
      assert.ok(!main.includes('useState<RegistryDatabase[]>([{'));
      // Phase 9-34: the main page no longer links to the clarification prototype.
      assert.ok(!main.includes('/adaptive-prototype?database='));
      assert.ok(main.includes('getCatalog(adaptiveFetch, activeDbId'));
      assert.ok(!main.includes('DSN'));
      assert.ok(!main.includes('dsn'));
      assert.ok(!main.includes('password'));
    });
  
    it('prototype page revalidates the URL id against strict registry membership', () => {
      const page = readFileSync('src/adaptive/ClarificationDemoPage.tsx', 'utf-8');
      assert.ok(page.includes('parsePrototypeDatabaseParam(window.location.search)'));
      assert.ok(page.includes('await listDatabases(registryFetch'));
      assert.ok(page.includes('registry.find((entry) => entry.database_id === wanted)'));
      assert.ok(page.includes('databaseId: activeDatabase'));
      assert.ok(!page.includes('DSN'));
      assert.ok(!page.includes('dsn'));
      assert.ok(!page.includes('password'));
    });
  });
}

// phase8-15-postgres-registration
{
  function summary(overrides: Record<string, unknown> = {}) {
    return {
      version: 'private-postgres-registration-v1',
      database_id: 'campus-pg',
      label: 'Campus',
      status: 'PENDING_REVIEW',
      schema_fingerprint: 'c'.repeat(64),
      schema: { table_count: 0, column_count: 0, foreign_key_count: 0, tables: [] },
      mapping: { ontology_version: 1, proposer: 'RULES', field_count: 0, unmapped_count: 0, unmapped_fields: [], fields: [] },
      ...overrides,
    };
  }
  
  describe('phase8-15 parsers', () => {
    it('accepts connection ids and labels only', () => {
      const ok = { version: 'postgres-sources-v1', sources: [{ source_id: 'default', label: '기본 서버' }] };
      assert.deepEqual(parseSources(ok), [{ source_id: 'default', label: '기본 서버' }]);
      assert.equal(parseSources({ ...ok, sources: [{ source_id: 'default', label: 'x', dsn: 'postgresql://u:p@h/db' }] }), null);
      assert.equal(parseSources({ ...ok, sources: [{ source_id: 'Bad Id', label: 'x' }] }), null);
      assert.equal(parseSources({ ...ok, extra: 1 }), null);
    });
  
    it('accepts safe schema names for the asked connection', () => {
      const ok = { version: 'postgres-sources-v1', source_id: 'default', schemas: ['public', 'campus_2026'] };
      assert.deepEqual(parseSchemas(ok, 'default'), ['public', 'campus_2026']);
      assert.equal(parseSchemas(ok, 'other'), null);
      assert.equal(parseSchemas({ ...ok, schemas: ['a;drop'] }, 'default'), null);
    });
  });
  
  describe('phase8-15 client', () => {
    it('registers with ids only and parses the summary', async () => {
      const calls: { url: string; body: unknown }[] = [];
      const fetchFn: PgFetchLike = async (url, init) => {
        calls.push({ url, body: init.body === undefined ? undefined : JSON.parse(init.body) });
        return { ok: true, status: 200, json: async () => summary() };
      };
      const result = await registerPostgresSchema(fetchFn, {
        sourceId: 'default', schema: 'campus', databaseId: 'campus-pg', label: 'Campus',
      });
      assert.equal(result.database_id, 'campus-pg');
      assert.deepEqual(calls, [{
        url: POSTGRES_REGISTER_PATH,
        body: { source_id: 'default', schema: 'campus', database_id: 'campus-pg', label: 'Campus' },
      }]);
      const wrongVersion: PgFetchLike = async () => ({ ok: true, status: 200, json: async () => summary({ version: 'private-sqlite-upload-v2' }) });
      await assert.rejects(
        registerPostgresSchema(wrongVersion, { sourceId: 'default', schema: 'campus', databaseId: 'campus-pg', label: 'Campus' }),
        PostgresRegistrationClientError,
      );
    });
  
    it('maps fixed server codes to messages', async () => {
      const conflict: PgFetchLike = async () => ({
        ok: false, status: 409, json: async () => ({ error: { code: 'POSTGRES_REGISTRATION_REGISTRATION_CONFLICT' } }),
      });
      try {
        await fetchPostgresSchemas(conflict, 'default');
        assert.fail('should throw');
      } catch (error) {
        assert.equal(toPostgresMessage(error), '이미 사용 중인 Database ID입니다.');
      }
    });
  
    it('sends IGNORE with a reason', async () => {
      const bodies: unknown[] = [];
      const fetchFn = async (_url: string, init: { body?: string }) => {
        bodies.push(JSON.parse(init.body ?? '{}'));
        return { ok: false, status: 400, json: async () => ({ error: { code: 'MAPPING_REVIEW_INVALID_REQUEST' } }) };
      };
      await assert.rejects(ignoreReviewField(fetchFn as never, 'campus-pg', 'Department.code', 'admin', '없음'));
      assert.deepEqual(bodies, [{ action: 'IGNORE', database_id: 'campus-pg', semantic_field: 'Department.code', reviewer: 'admin', reason: '없음' }]);
    });
  });
  
  describe('phase8-15 label rule', () => {
    it('refuses labels the PostgreSQL runtime would refuse, before any request', async () => {
      let called = false;
      const fetchFn: PgFetchLike = async () => {
        called = true;
        return { ok: true, status: 200, json: async () => summary() };
      };
      try {
        await registerPostgresSchema(fetchFn, { sourceId: 'default', schema: 'public', databaseId: 'campus-pg', label: '레거시 DB' });
        assert.fail('should throw');
      } catch (error) {
        assert.match(toPostgresMessage(error), /영문/);
      }
      assert.equal(called, false);
    });
  });
}

// phase8-16-registrations
{
  const item = {
    database_id: 'campus-pg', label: 'Campus', source_id: 'default', schema: 'public',
    status: 'PUBLISHED', mapping_version: 2,
  };
  
  describe('phase8-16 registration list', () => {
    it('accepts published and pending registrations', () => {
      const parsed = parseRegistrations({
        version: 'postgres-registrations-v1',
        registrations: [item, { ...item, database_id: 'next-pg', status: 'PENDING_REVIEW', mapping_version: null }],
      });
      assert.equal(parsed?.length, 2);
      assert.equal(parsed?.[1]?.mapping_version, null);
    });
  
    it('rejects inconsistent status/version, extra keys and secrets', () => {
      const wrap = (value: Record<string, unknown>) => ({ version: 'postgres-registrations-v1', registrations: [value] });
      assert.equal(parseRegistrations(wrap({ ...item, mapping_version: null })), null);
      assert.equal(parseRegistrations(wrap({ ...item, status: 'PENDING_REVIEW' })), null);
      assert.equal(parseRegistrations(wrap({ ...item, dsn: 'postgresql://u:p@h/db' })), null);
      assert.equal(parseRegistrations(wrap({ ...item, schema: 'a;drop' })), null);
      assert.equal(parseRegistrations({ version: 'x', registrations: [] }), null);
    });
  });
  
  describe('phase8-16 label rule in the list', () => {
    it('rejects labels the PostgreSQL runtime refuses', () => {
      assert.equal(parseRegistrations({ version: 'postgres-registrations-v1', registrations: [{ ...item, label: '캠퍼스' }] }), null);
    });
  });
}

// phase8-17-other-school
{
  function blocked(fields: string[]): ReviewFetchLike {
    return async () => ({
      ok: false,
      status: 422,
      json: async () => ({ error: { code: 'MAPPING_REVIEW_PUBLISH_BLOCKED', fields } }),
    });
  }
  
  async function messageFor(fields: string[]): Promise<string> {
    try {
      await publishReview(blocked(fields), 'campus-pg', 'admin');
    } catch (error) {
      return toReviewMessage(error);
    }
    throw new Error('publish should fail');
  }
  
  describe('phase8-17 publish-blocked message', () => {
    it('names a refused query path when every field was reviewed', async () => {
      assert.equal(await messageFor([]), PUBLISH_RUNTIME_BLOCKED_MESSAGE);
    });
  
    it('keeps the unfinished-review message when fields are listed', async () => {
      const message = await messageFor(['Course.name']);
      assert.notEqual(message, PUBLISH_RUNTIME_BLOCKED_MESSAGE);
      assert.match(message, /검토가 끝나지 않은/);
    });
  });
  
  describe('phase8-17 alias length message', () => {
    it('does not call a short alias a collision', () => {
      const length = toPostgresMessage(new PostgresRegistrationClientError('server', 'POSTGRES_REGISTRATION_ALIAS_LENGTH'));
      const conflict = toPostgresMessage(new PostgresRegistrationClientError('server', 'POSTGRES_REGISTRATION_ALIAS_CONFLICT'));
      assert.match(length, /2~20자/);
      assert.notEqual(length, conflict);
    });
  });
  
  const offering = {
    course_code: 'SW1001', course_name: '프로그래밍기초', credits: 3, category: '전공필수',
    department_code: 'SW', department_name: '소프트웨어학부', instructor_name: '교원A',
    year: 2026, semester: '1학기', section: '001', capacity: 40, enrolled: 32,
    day: '월요일', start_time: '09:00', end_time: '10:15',
  };
  
  function catalog(items: Record<string, unknown>[]) {
    return { version: 'synthetic-course-catalog-v1', database_id: 'other-school', items, total: items.length };
  }
  
  describe('phase8-17 offerings that meet twice a week', () => {
    it('accepts one row per meeting and shows one course with two slots', () => {
      const parsed = parseCatalogResponse(catalog([offering, { ...offering, day: '수요일' }]));
      assert.ok(parsed);
      const courses = catalogToCourses(parsed.items);
      assert.equal(courses.length, 1);
      assert.deepEqual(courses[0]?.slots.map((slot) => slot.day), ['Mon', 'Wed']);
      assert.equal(courses[0]?.timeText, '월요일 09:00-10:15, 수요일 09:00-10:15');
    });
  
    it('still rejects rows that disagree or repeat', () => {
      assert.equal(parseCatalogResponse(catalog([offering, { ...offering, day: '수요일', capacity: 41 }])), null);
      assert.equal(parseCatalogResponse(catalog([offering, { ...offering }])), null);
    });
  
    it('places Korean weekdays on the timetable', () => {
      const [course] = catalogToCourses([{ ...offering, day: '금' }] as never);
      assert.deepEqual(course?.slots, [{ day: 'Fri', startHour: 9, endHour: 10.25 }]);
    });
  });
}

// phase9-17-reserved-database-id
{
  const good = { database_id: 'rehearsal-school', label: 'Rehearsal', schema: 'postgres' };
  const collision = { database_id: 'legacy', label: 'Legacy', schema: 'postgres' };
  const payload = (databases: unknown[]) => ({ version: REGISTRY_VERSION, databases });
  
  it('keeps valid databases and reports reserved and malformed entries without reflecting unsafe IDs', async () => {
    let warning: string | null = null;
    const databases = await listDatabases(async () => ({ ok: true, status: 200,
      json: async () => payload([good, collision, { database_id: '<secret>', schema: 'postgres' }, null]),
    }), { onWarning: (message) => { warning = message; } });
    assert.deepEqual(databases, [good]);
    assert.match(warning!, /legacy.*예약어/);
    assert.match(warning!, /형식.*2개/);
    assert.ok(!warning!.includes('<secret>'));
    parseDatabaseRegistry(payload([good]), (message) => { warning = message; });
    assert.equal(warning, null);
  });
  
  it('keeps top-level and duplicate ID checks strict, including IDs of omitted entries', () => {
    for (const value of [null, [], { ...payload([good]), version: 'wrong' },
      { ...payload([good]), extra: true }, payload([]), payload(Array(9).fill(good)),
      { version: REGISTRY_VERSION, databases: {} }, payload([good, good]),
      payload([good, { ...good, schema: 'bad' }]), payload([collision, collision])]) {
      assert.equal(parseDatabaseRegistry(value), null);
    }
    assert.deepEqual(parseDatabaseRegistry(payload([collision])), []);
    assert.deepEqual(parseDatabaseRegistry(payload([{ ...good, unexpected: true }, good])), null);
  });
  
  it('both registration form clients reject legacy before fetch with a system-name message', async () => {
    let calls = 0;
    const fetchFn = async () => { calls += 1; throw new Error('must not fetch'); };
    await assert.rejects(() => registerPostgresSchema(fetchFn, {
      sourceId: 'default', schema: 'public', databaseId: 'legacy', label: 'Legacy',
    }), (error: unknown) => toPostgresMessage(error) === RESERVED_DATABASE_ID_MESSAGE);
    await assert.rejects(() => uploadSqliteDatabase(fetchFn, 'legacy', 'Legacy', new Blob(['sqlite'])),
      (error: unknown) => toUploadMessage(error) === RESERVED_DATABASE_ID_MESSAGE);
    assert.equal(calls, 0);
  });
  
  it('main page displays the partial registry warning in the page and DB chooser', () => {
    const source = readFileSync(new URL('../../src/pages/MainPage.tsx', import.meta.url), 'utf8');
    assert.match(source, /onWarning:/);
    assert.match(source, /registryError \?\? registryWarning/);
    assert.match(source, /role="status">\{registryWarning\}/);
  });
}

// phase9-18-delete-registration
{
  it('enables deletion only for an exact ID while idle, including saved legacy', () => {
    assert.equal(canDeleteRegistration('campus-pg', 'campus-pg', false), true);
    for (const confirmation of ['', 'campus', 'Campus-pg', ' campus-pg', 'campus-pg ']) {
      assert.equal(canDeleteRegistration('campus-pg', confirmation, false), false);
    }
    assert.equal(canDeleteRegistration('campus-pg', 'campus-pg', true), false);
    assert.equal(canDeleteRegistration('../outside', '../outside', false), false);
    assert.equal(canDeleteRegistration('legacy', 'legacy', false), true);
  });
  
  it('deletes through the fixed API and refreshes the registration list with fake fetch', async () => {
    const calls: [string, string][] = [];
    const fetchFn: PgFetchLike = async (url, init) => {
      calls.push([url, init.method]);
      return { ok: true, status: 200, json: async () => init.method === 'DELETE'
        ? { database_id: 'legacy', deleted: true }
        : { version: 'postgres-registrations-v1', registrations: [] } };
    };
    await deletePostgresRegistration(fetchFn, 'legacy');
    assert.deepEqual(await fetchRegistrations(fetchFn), []);
    assert.deepEqual(calls, [
      ['/api/adaptive/v1/postgres/registrations?database_id=legacy', 'DELETE'],
      ['/api/adaptive/v1/postgres/registrations', 'GET'],
    ]);
  });
  
  it('refuses malformed success and fixed server errors', async () => {
    for (const payload of [{ database_id: 'other', deleted: true },
      { database_id: 'campus-pg', deleted: false },
      { database_id: 'campus-pg', deleted: true, extra: 'private' }, null]) {
      await assert.rejects(deletePostgresRegistration(async () => ({ ok: true, status: 200, json: async () => payload }), 'campus-pg'));
    }
    await assert.rejects(deletePostgresRegistration(async () => ({ ok: false, status: 404,
      json: async () => ({ error: { code: 'POSTGRES_REGISTRATION_NOT_FOUND' } }),
    }), 'campus-pg'), { serverCode: 'POSTGRES_REGISTRATION_NOT_FOUND' });
  });
  
  it('UI wires exact confirmation, refresh, closing review and selected DB fallback', () => {
    const panel = readFileSync(new URL('../../src/adaptive/PostgresRegistrationPanel.tsx', import.meta.url), 'utf8');
    assert.match(panel, /disabled=\{!canDeleteRegistration\(deleting, confirmation, busy\)\}/);
    assert.match(panel, /await deletePostgresRegistration\(fetchFn, id\)/);
    assert.match(panel, /if \(opened === id\) setOpened\(null\)/);
    assert.match(panel, /onDeleted\?\.\(id\);\s+reloadRegistrations\(\)/);
    const page = readFileSync(new URL('../../src/pages/MainPage.tsx', import.meta.url), 'utf8');
    assert.match(page, /if \(activeDbIdRef.current === databaseId\) applyDatabase\(LEGACY_SELECTION_ID\)/);
    assert.match(page, /else if \(pendingDbId === databaseId\) setPendingDbId\(LEGACY_SELECTION_ID\)/);
    assert.match(page, /onDeleted=\{[\s\S]*?setRegistryRetry\(\(value\) => value \+ 1\)/);
  });
}

// phase9-19-database-id-message
{
  const idMessage = 'ID는 영문 소문자·숫자·하이픈(-)만 쓸 수 있고, 첫 글자는 영문 소문자나 숫자여야 합니다(64자 이내). 예: legacy-pg';
  const file = new Blob(['fake sqlite']);
  const pgInput = { sourceId: 'default', schema: 'public', databaseId: 'legacy-pg', label: 'School PG' };
  
  for (const databaseId of ['legacy_test', 'Legacy-pg', '', 'a'.repeat(65), '-school']) {
    it(`rejects invalid ID ${JSON.stringify(databaseId)} as input in both clients without fetching`, async () => {
      let calls = 0;
      const fetchFn = async () => { calls += 1; throw new Error('must not fetch'); };
      await assert.rejects(registerPostgresSchema(fetchFn, { ...pgInput, databaseId }), (error: unknown) => {
        assert.ok(error instanceof PostgresRegistrationClientError);
        assert.equal(error.kind, 'input-id');
        assert.equal(toPostgresMessage(error), idMessage);
        return true;
      });
      await assert.rejects(uploadSqliteDatabase(fetchFn, databaseId, '학교 DB', file), (error: unknown) => {
        assert.ok(error instanceof SqliteUploadClientError);
        assert.equal(error.kind, 'input-id');
        assert.equal(toUploadMessage(error), idMessage);
        return true;
      });
      assert.equal(registrationDatabaseIdMessage(databaseId), idMessage);
      assert.equal(calls, 0);
    });
  }
  
  function summary(version: string, databaseId: string, label: string) {
    return {
      version, database_id: databaseId, label, status: 'PENDING_REVIEW', schema_fingerprint: 'a'.repeat(64),
      schema: { table_count: 0, column_count: 0, foreign_key_count: 0, tables: [] },
      mapping: { ontology_version: 3, proposer: 'RULES', field_count: 0, unmapped_count: 0, unmapped_fields: [], fields: [] },
    };
  }
  
  it('sends valid IDs including the 1 and 64 character boundaries through both clients', async () => {
    for (const databaseId of ['a', '1', 'legacy-pg', 'a'.repeat(64)]) {
      let calls = 0;
      const pg = await registerPostgresSchema(async (_url, init) => {
        calls += 1;
        assert.equal(JSON.parse(init.body!).database_id, databaseId);
        return { ok: true, status: 200, json: async () => summary(POSTGRES_REGISTER_VERSION, databaseId, 'School PG') };
      }, { ...pgInput, databaseId });
      const sqlite = await uploadSqliteDatabase(async (url) => {
        calls += 1;
        assert.equal(new URL(url, 'http://fixture.invalid').searchParams.get('database_id'), databaseId);
        return { ok: true, status: 200, json: async () => summary(SQLITE_UPLOAD_VERSION, databaseId, '학교 DB') };
      }, databaseId, '학교 DB', file);
      assert.equal(pg.database_id, databaseId);
      assert.equal(sqlite.database_id, databaseId);
      assert.equal(registrationDatabaseIdMessage(databaseId), null);
      assert.equal(calls, 2);
    }
  });
  
  it('reports label rules as input errors without fetching', async () => {
    let calls = 0;
    const fetchFn = async () => { calls += 1; throw new Error('must not fetch'); };
    for (const label of ['', '   ', 'a'.repeat(65), 'name\n', '학교 DB', '_School', 'School!']) {
      await assert.rejects(registerPostgresSchema(fetchFn, { ...pgInput, label }), (error: unknown) => {
        assert.ok(error instanceof PostgresRegistrationClientError);
        assert.equal(error.kind, 'input-label');
        assert.match(toPostgresMessage(error), /영문·숫자·공백·\( \) _ \. -.*첫 글자.*1~64자/);
        return true;
      });
    }
    for (const label of ['', '   ', 'a'.repeat(65), 'name\n', 'name\u007f']) {
      await assert.rejects(uploadSqliteDatabase(fetchFn, 'school', label, file), (error: unknown) => {
        assert.ok(error instanceof SqliteUploadClientError);
        assert.equal(error.kind, 'input-label');
        assert.match(toUploadMessage(error), /1~64자.*한글·영문·숫자·기호.*줄바꿈·제어 문자/);
        return true;
      });
    }
    assert.equal(calls, 0);
  });
  
  it('keeps malformed server responses and unreadable JSON as contract errors with the old messages', async () => {
    for (const json of [async () => ({}), async () => { throw new SyntaxError('bad JSON'); }]) {
      const fetchFn = async () => ({ ok: true, status: 200, json });
      await assert.rejects(registerPostgresSchema(fetchFn, pgInput), (error: unknown) => {
        assert.ok(error instanceof PostgresRegistrationClientError);
        assert.equal(error.kind, 'contract');
        assert.equal(toPostgresMessage(error), '서버 응답 형식이 올바르지 않습니다.');
        return true;
      });
      await assert.rejects(uploadSqliteDatabase(fetchFn, 'school', '학교 DB', file), (error: unknown) => {
        assert.ok(error instanceof SqliteUploadClientError);
        assert.equal(error.kind, 'contract');
        assert.equal(toUploadMessage(error), UPLOAD_CONTRACT_MESSAGE);
        assert.equal(toUploadMessage(error), '응답 형식이 올바르지 않습니다. 다시 시도해 주세요.');
        return true;
      });
    }
  });
  
  it('both forms always show ID guidance, report invalid input inline, and disable registration', () => {
    assert.equal(DATABASE_ID_INPUT_MESSAGE, idMessage);
    assert.equal(registrationDatabaseIdMessage('legacy'), RESERVED_DATABASE_ID_MESSAGE);
    const pg = readFileSync(new URL('../../src/adaptive/PostgresRegistrationPanel.tsx', import.meta.url), 'utf8');
    const main = readFileSync(new URL('../../src/pages/MainPage.tsx', import.meta.url), 'utf8');
    assert.match(pg, /aria-describedby="postgres-id-guidance"/);
    assert.match(pg, /\{DATABASE_ID_INPUT_MESSAGE\}/);
    assert.match(pg, /databaseIdError !== DATABASE_ID_INPUT_MESSAGE/);
    assert.match(pg, /disabled=\{busy \|\| schema === '' \|\| databaseIdError !== null\}/);
    assert.match(main, /aria-describedby="sqlite-id-guidance"/);
    assert.match(main, /\{DATABASE_ID_INPUT_MESSAGE\}/);
    assert.match(main, /uploadDbIdError !== DATABASE_ID_INPUT_MESSAGE/);
    assert.match(main, /disabled=\{uploadStatus === 'uploading' \|\| uploadDbIdError !== null\}/);
    assert.ok(!main.includes("file === null || databaseId === '' || label === ''"));
  });
}
