import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { POSTGRES_DICTIONARIES_PATH, parseCandidates, parseDictionaryDetail, setDictionaryEntry, suggestAliases, parseExtensionDetail, PostgresRegistrationClientError, describeImport, importDictionaryFromTable, toPostgresMessage } from '../../src/adaptive/postgresRegistration.ts'
import type { PgFetchLike } from '../../src/adaptive/postgresRegistration.ts'
import { readFileSync } from 'node:fs'
import { copyDictionaries, describeDictionaryCopy, fetchDictionarySources, parseDictionarySources } from '../../src/adaptive/dictionaryCopy.ts'

// phase8-07-value-dictionary
{
  const DB = 'campus-pg';
  
  function detail(overrides: Record<string, unknown> = {}) {
    return {
      version: 'postgres-dictionaries-v1',
      database_id: DB,
      fields: [
        {
          field: 'Schedule.day',
          label: '강의 요일',
          values: ['MON', 'TUE'],
          entries: [{ stored: 'MON', label: '월', aliases: ['월요일'] }],
        },
        { field: 'Department.name', label: '학과명', values: null, entries: [] },
      ],
      ...overrides,
    };
  }
  
  describe('phase8-07 dictionary parsers', () => {
    it('accepts fields, entries and optional warnings', () => {
      const parsed = parseDictionaryDetail(detail(), DB);
      assert.equal(parsed?.fields[0]?.entries[0]?.label, '월');
      assert.equal(parsed?.fields[1]?.values, null);
      assert.deepEqual(parsed?.warnings, []);
      const warned = parseDictionaryDetail(detail({ warnings: [{ alias: '전기', warning: '다른 값 속에 들어 있음' }] }), DB);
      assert.equal(warned?.warnings[0]?.alias, '전기');
    });
  
    it('rejects other databases, extra keys and malformed entries', () => {
      assert.equal(parseDictionaryDetail(detail(), 'other-db'), null);
      assert.equal(parseDictionaryDetail(detail({ extra: 1 }), DB), null);
      const bad = detail();
      (bad.fields[0]!.entries[0] as Record<string, unknown>)['aliases'] = '월요일';
      assert.equal(parseDictionaryDetail(bad, DB), null);
      const badField = detail();
      (badField.fields[0] as Record<string, unknown>)['field'] = 'v_course_info.day';
      assert.equal(parseDictionaryDetail(badField, DB), null);
    });
  
    it('parses candidates with source, block and warning', () => {
      const payload = {
        version: 'postgres-dictionaries-v1', database_id: DB, field: 'Course.category', ai_used: true,
        candidates: [
          { stored: '전공(핵심)', alias: '전핵', source: 'RULE', blocked: null, warning: null },
          { stored: '교양', alias: '일반선택', source: 'AI', blocked: '다른 값과 겹침: 일반선택', warning: null },
        ],
      };
      assert.equal(parseCandidates(payload, DB, 'Course.category')?.length, 2);
      assert.equal(parseCandidates({ ...payload, field: 'Schedule.day' }, DB, 'Course.category'), null);
      assert.equal(
        parseCandidates({ ...payload, candidates: [{ ...payload.candidates[0], source: 'MODEL' }] }, DB, 'Course.category'),
        null,
      );
    });
  });
  
  describe('phase8-07 dictionary client', () => {
    it('sends SET and SUGGEST bodies', async () => {
      const bodies: unknown[] = [];
      const fetchFn: PgFetchLike = async (url, init) => {
        assert.equal(url, POSTGRES_DICTIONARIES_PATH);
        const body = JSON.parse(init.body ?? '{}') as Record<string, unknown>;
        bodies.push(body);
        if (body['action'] === 'SUGGEST') {
          return { ok: true, status: 200, json: async () => ({ version: 'postgres-dictionaries-v1', database_id: DB, field: 'Schedule.day', ai_used: false, candidates: [] }) };
        }
        return { ok: true, status: 200, json: async () => ({ ...detail(), warnings: [] }) };
      };
      await setDictionaryEntry(fetchFn, { databaseId: DB, field: 'Schedule.day', stored: 'MON', label: '월', aliases: ['월요일'], reviewer: 'admin' });
      await suggestAliases(fetchFn, { databaseId: DB, field: 'Schedule.day', ai: false });
      assert.deepEqual(bodies, [
        { action: 'SET', database_id: DB, field: 'Schedule.day', stored: 'MON', label: '월', aliases: ['월요일'], reviewer: 'admin' },
        { action: 'SUGGEST', database_id: DB, field: 'Schedule.day', ai: false },
      ]);
    });
  });
}

// phase8-19-extensions
{
  const detail = {
    version: 'postgres-extensions-v1',
    database_id: 'other-school',
    fields: [{ key: 'classroom', label: '강의실', entity: 'Schedule', table: 'siganpyo', column: 'gangui_sil', data_type: 'string' }],
    tables: [{ name: 'siganpyo', columns: [{ name: 'gangui_sil', type: 'character varying(20)' }] }],
  };
  
  describe('phase8-19 extension detail', () => {
    it('parses fields and tables', () => {
      const parsed = parseExtensionDetail(detail, 'other-school');
      assert.equal(parsed?.fields[0]?.label, '강의실');
      assert.equal(parsed?.tables[0]?.columns[0]?.name, 'gangui_sil');
    });
  
    it('rejects unknown entities, types, keys, extra keys and another database', () => {
      const field = detail.fields[0]!;
      const wrap = (value: Record<string, unknown>) => ({ ...detail, fields: [{ ...field, ...value }] });
      assert.equal(parseExtensionDetail(wrap({ entity: 'Room' }), 'other-school'), null);
      assert.equal(parseExtensionDetail(wrap({ data_type: 'date' }), 'other-school'), null);
      assert.equal(parseExtensionDetail(wrap({ key: 'Class Room' }), 'other-school'), null);
      assert.equal(parseExtensionDetail(wrap({ dsn: 'postgresql://u:p@h/db' }), 'other-school'), null);
      assert.equal(parseExtensionDetail(detail, 'campus-pg'), null);
    });
  });
}

// phase8-24-dictionary-import
{
  const DB = 'code-school';
  const detail = {
    version: 'postgres-dictionaries-v1',
    database_id: DB,
    fields: [
      {
        field: 'Course.category',
        label: '이수 구분',
        values: ['11', '21'],
        entries: [{ stored: '21', label: '교양필수', aliases: [] }],
      },
    ],
  };
  
  describe('phase8-24 code-table import', () => {
    it('parses the import counts and keeps plain details unchanged', () => {
      const parsed = parseDictionaryDetail({ ...detail, imported: { added: 2, updated: 0, skipped: 1 } }, DB);
      assert.deepEqual(parsed?.imported, { added: 2, updated: 0, skipped: 1 });
      assert.equal(parseDictionaryDetail(detail, DB)?.imported, undefined);
      for (const imported of [
        { added: -1, updated: 0, skipped: 0 },
        { added: 1, updated: 0 },
        { added: 1.5, updated: 0, skipped: 0 },
        { added: 65, updated: 0, skipped: 0 }, // Luna 8-24 review: one import reads at most 64 pairs
        { added: 30, updated: 30, skipped: 30 },
      ]) {
        assert.equal(parseDictionaryDetail({ ...detail, imported }, DB), null);
      }
    });
  
    it('sends the choice and returns the updated dictionary', async () => {
      const sent: unknown[] = [];
      const fetchFn: PgFetchLike = async (_url, init) => {
        sent.push(JSON.parse(init.body ?? '{}'));
        return { ok: true, status: 200, json: async () => ({ ...detail, imported: { added: 1, updated: 0, skipped: 0 } }) };
      };
      const result = await importDictionaryFromTable(fetchFn, {
        databaseId: DB, field: 'Course.category', table: 'gongtong_code', codeColumn: 'code', nameColumn: 'code_nm',
        group: { column: 'code_group', value: 'ISU' }, reviewer: 'admin',
      });
      assert.equal(result.imported?.added, 1);
      assert.deepEqual(sent[0], {
        action: 'IMPORT', database_id: DB, field: 'Course.category', table: 'gongtong_code', code_column: 'code',
        name_column: 'code_nm', group_column: 'code_group', group_value: 'ISU', reviewer: 'admin',
      });
    });
  
    it('refuses an answer without counts', async () => {
      const fetchFn: PgFetchLike = async () => ({ ok: true, status: 200, json: async () => detail });
      await assert.rejects(
        importDictionaryFromTable(fetchFn, {
          databaseId: DB, field: 'Course.category', table: 't', codeColumn: 'c', nameColumn: 'n', group: null, reviewer: 'a',
        }),
        PostgresRegistrationClientError,
      );
    });
  
    it('explains the result and the refusals', () => {
      assert.match(describeImport({ added: 5, updated: 1, skipped: 2 }), /새 항목 5개.*이름 바뀐 항목 1개.*없는 코드 2개/);
      assert.doesNotMatch(describeImport({ added: 5, updated: 0, skipped: 0 }), /없는 코드/);
      for (const code of ['IMPORT_CONFLICT', 'TOO_MANY_CODES']) {
        const message = toPostgresMessage(new PostgresRegistrationClientError('server', `POSTGRES_REGISTRATION_${code}`));
        assert.ok(message.length > 10 && !message.includes(code));
      }
    });
  
    it('accepts catalog names PostgreSQL allows, such as Korean or spaces', () => {
      const extensions = {
        version: 'postgres-extensions-v1',
        database_id: DB,
        fields: [],
        tables: [{ name: '코드 표', columns: [{ name: '코드', type: 'text' }, { name: 'code_nm', type: 'text' }] }],
      };
      assert.equal(parseExtensionDetail(extensions, DB)?.tables[0]?.name, '코드 표');
      const control = { ...extensions, tables: [{ name: 'a' + String.fromCharCode(10) + 'b', columns: [] }] };
      assert.equal(parseExtensionDetail(control, DB), null);
    });
  
    it('the dictionary panel offers the import', () => {
      const panel = readFileSync(new URL('../../src/adaptive/ValueDictionaryPanel.tsx', import.meta.url), 'utf8');
      assert.match(panel, /<CodeTableImport/);
    });
  });
}

// phase9-20-copy-dictionaries
{
  const version = 'postgres-dictionary-copy-v1';
  const sources = [{ database_id: 'legacy-view', label: 'Legacy', fields: { 'Department.name': 1 } },
    { database_id: 'target', label: 'Target', fields: {} }];
  const fields = [{ field: 'Department.name', label: '학과명', imported: 1, skipped: 1, reasons: { VALUE_NOT_PRESENT: 1 } }];
  
  it('lists only source metadata and excludes the target', async () => {
    const fake: PgFetchLike = async (url, init) => {
      assert.equal(url, '/api/adaptive/v1/postgres/dictionaries/sources');
      assert.equal(init.method, 'GET');
      return { ok: true, status: 200, json: async () => ({ version, sources }) };
    };
    assert.deepEqual(await fetchDictionarySources(fake, 'target'), [sources[0]]);
    assert.equal(parseDictionarySources({ version, sources: [sources[0], sources[0]] }), null);
    assert.equal(parseDictionarySources({ version, sources: [{ ...sources[0], fields: { x: -1 } }] }), null);
    assert.equal(parseDictionarySources({ version, sources: [{ ...sources[0], entries: [] }] }), null);
  });
  
  it('sends the source, target and shared reviewer and describes field counts', async () => {
    const fake: PgFetchLike = async (_url, init) => {
      assert.deepEqual(JSON.parse(init.body ?? '{}'), { action: 'COPY', database_id: 'target', source_database_id: 'legacy-view', reviewer: 'admin' });
      return { ok: true, status: 200, json: async () => ({ version, database_id: 'target', source_database_id: 'legacy-view', fields }) };
    };
    const result = await copyDictionaries(fake, 'target', 'legacy-view', 'admin');
    assert.equal(describeDictionaryCopy(result), '학과명 1개 가져옴. 건너뜀 1개. 게시해야 챗봇에 적용됩니다.');
  });
  
  it('rejects missing reviewer and self copy before fetching', async () => {
    const fake: PgFetchLike = async () => { assert.fail('must not fetch'); };
    await assert.rejects(copyDictionaries(fake, 'target', 'target', 'admin'), PostgresRegistrationClientError);
    await assert.rejects(copyDictionaries(fake, 'target', 'legacy-view', ' '), PostgresRegistrationClientError);
  });
  
  it('rejects malformed counts, reason codes and mismatched databases', async () => {
    for (const override of [
      { database_id: 'wrong' }, { fields: [{ ...fields[0], skipped: 2 }] },
      { fields: [{ ...fields[0], reasons: { UNKNOWN: 1 } }] },
      { fields: [{ ...fields[0], imported: -1 }] }, { fields: [fields[0], fields[0]] },
    ]) {
      const fake: PgFetchLike = async () => ({ ok: true, status: 200, json: async () => ({ version, database_id: 'target', source_database_id: 'legacy-view', fields, ...override }) });
      await assert.rejects(copyDictionaries(fake, 'target', 'legacy-view', 'admin'), PostgresRegistrationClientError);
    }
  });
  
  it('panel offers a source selector, requires reviewer and refreshes the dictionary after copying', () => {
    const panel = readFileSync(new URL('../../src/adaptive/ValueDictionaryPanel.tsx', import.meta.url), 'utf8');
    assert.match(panel, /다른 DB 사전 가져오기/);
    assert.match(panel, /sources\.map/);
    assert.match(panel, /sourceId === '' \|\| reviewer\.trim\(\) === ''/);
    assert.match(panel, /copyDictionaries\(fetchFn, databaseId, sourceId, reviewer\)/);
    assert.match(panel, /setMessage\(describeDictionaryCopy\(result\)\)/);
    assert.match(panel, /setDetail\(await fetchDictionaries\(fetchFn, databaseId\)\)/);
  });
}
