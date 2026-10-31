from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from backend.adaptive.harness import (
    CompiledQuery,
    ConservativeSqlPolicy,
    DataType,
    ExecutionHarness,
    FakeModel,
    FieldMapping,
    HarnessErrorCode,
    HarnessStage,
    HarnessStatus,
    MappingProfile,
    MappingStatus,
    Ontology,
    OntologyField,
    Operator,
    QueryIR,
    SingleTableSQLiteCompiler,
    SQLiteReadOnlyExecutor,
)
from backend.adaptive.harness.errors import HarnessFailure
from backend.adaptive.harness.validator import validate_query_ir
from datetime import datetime, timezone
from backend.adaptive.catalog import DatabaseCatalog
from backend.adaptive.compiler import SchemaAdaptiveSQLiteCompiler
from backend.adaptive.fixtures import FixtureVariant, build_fixture_bundle
from backend.adaptive.harness import (
    DataType,
    ExecutionHarness,
    FakeModel,
    HarnessErrorCode,
    HarnessStatus,
    Ontology,
    OntologyField,
    Operator,
    SQLiteReadOnlyExecutor,
)
from backend.adaptive.mapping import (
    PublishedFieldMapping,
    PublishedMappingProfile,
    to_harness_mapping,
)
from backend.adaptive.policy import SchemaAdaptiveAstPolicy
from backend.adaptive.relations import build_physical_relation_graph


# From test_execution_harness.py

class ExecutionHarnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "courses.sqlite3"
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute(
                "CREATE TABLE courses (subject_name TEXT, credit_hours INTEGER, category TEXT)"
            )
            connection.executemany(
                "INSERT INTO courses VALUES (?, ?, ?)",
                [
                    ("데이터베이스", 3, "전공"),
                    ("운영체제", 3, "전공"),
                    ("글쓰기", 2, "교양"),
                ],
            )
            connection.commit()
        finally:
            connection.close()

        self.ontology = Ontology(
            entities={"Course"},
            fields={
                "Course.name": OntologyField(
                    name="Course.name",
                    data_type=DataType.STRING,
                    allowed_operators={Operator.EQUALS, Operator.CONTAINS},
                ),
                "Course.credits": OntologyField(
                    name="Course.credits",
                    data_type=DataType.NUMBER,
                    allowed_operators={
                        Operator.EQUALS,
                        Operator.GREATER_THAN,
                        Operator.GREATER_THAN_OR_EQUAL,
                    },
                ),
                "Course.category": OntologyField(
                    name="Course.category",
                    data_type=DataType.STRING,
                    allowed_operators={Operator.EQUALS, Operator.CONTAINS},
                ),
            },
        )
        self.mapping = MappingProfile(
            database_id="test-db",
            version=1,
            status=MappingStatus.PUBLISHED,
            fields={
                "Course.name": FieldMapping(table="courses", column="subject_name"),
                "Course.credits": FieldMapping(table="courses", column="credit_hours"),
                "Course.category": FieldMapping(table="courses", column="category"),
            },
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_harness(self, response: object) -> ExecutionHarness:
        return ExecutionHarness(
            model=FakeModel({"질문": response}),
            compiler=SingleTableSQLiteCompiler(),
            sql_policy=ConservativeSqlPolicy(),
            executor=SQLiteReadOnlyExecutor({"test-db": self.database_path}),
        )

    def test_successful_query_returns_semantic_columns_and_trace(self) -> None:
        harness = self.make_harness(
            {
                "entity": "Course",
                "select": ["Course.name", "Course.credits"],
                "filters": [
                    {"field": "Course.credits", "operator": "equals", "value": 3}
                ],
                "limit": 50,
            }
        )

        result = harness.run(
            {"database_id": "test-db", "question": "질문"}, self.ontology, self.mapping
        )

        self.assertEqual(result.status, HarnessStatus.SUCCEEDED)
        self.assertEqual(result.columns, ["Course.name", "Course.credits"])
        self.assertEqual(
            result.rows,
            [
                {"Course.name": "데이터베이스", "Course.credits": 3},
                {"Course.name": "운영체제", "Course.credits": 3},
            ],
        )
        self.assertIn("?", result.sql or "")
        self.assertNotIn("3학점", result.sql or "")
        self.assertEqual(
            [event.stage for event in result.trace],
            [
                HarnessStage.REQUEST_VALIDATED,
                HarnessStage.CONTEXT_READY,
                HarnessStage.IR_GENERATED,
                HarnessStage.IR_VALIDATED,
                HarnessStage.SQL_COMPILED,
                HarnessStage.SQL_VALIDATED,
                HarnessStage.EXECUTED,
                HarnessStage.RESULT_VALIDATED,
            ],
        )

    def test_unpublished_mapping_stops_before_model_generation(self) -> None:
        mapping = self.mapping.model_copy(update={"status": MappingStatus.DRAFT})
        result = self.make_harness({}).run(
            {"database_id": "test-db", "question": "질문"}, self.ontology, mapping
        )

        self.assertEqual(result.status, HarnessStatus.FAILED)
        self.assertEqual(result.error.code, HarnessErrorCode.MAPPING_NOT_PUBLISHED)
        self.assertEqual(result.sql, None)
        self.assertEqual(result.trace[-1].stage, HarnessStage.CONTEXT_READY)

    def test_unknown_field_is_rejected(self) -> None:
        result = self.make_harness(
            {"entity": "Course", "select": ["Course.professor"], "filters": []}
        ).run({"database_id": "test-db", "question": "질문"}, self.ontology, self.mapping)

        self.assertEqual(result.status, HarnessStatus.FAILED)
        self.assertEqual(result.error.code, HarnessErrorCode.UNKNOWN_FIELD)
        self.assertIsNone(result.sql)

    def test_operator_incompatible_with_field_type_is_rejected(self) -> None:
        result = self.make_harness(
            {
                "entity": "Course",
                "select": ["Course.name"],
                "filters": [
                    {"field": "Course.credits", "operator": "contains", "value": 3}
                ],
            }
        ).run({"database_id": "test-db", "question": "질문"}, self.ontology, self.mapping)

        self.assertEqual(result.status, HarnessStatus.FAILED)
        self.assertEqual(result.error.code, HarnessErrorCode.INVALID_OPERATOR)

    def test_malformed_model_output_is_rejected(self) -> None:
        result = self.make_harness({}).run(
            {"database_id": "test-db", "question": "질문"}, self.ontology, self.mapping
        )

        self.assertEqual(result.status, HarnessStatus.FAILED)
        self.assertEqual(result.error.code, HarnessErrorCode.MODEL_INVALID_OUTPUT)

    def test_mapping_for_another_database_is_rejected(self) -> None:
        mapping = self.mapping.model_copy(update={"database_id": "other-db"})
        result = self.make_harness({}).run(
            {"database_id": "test-db", "question": "질문"}, self.ontology, mapping
        )

        self.assertEqual(result.status, HarnessStatus.FAILED)
        self.assertEqual(result.error.code, HarnessErrorCode.MAPPING_MISSING)

    def test_multi_table_query_is_outside_mvp_scope(self) -> None:
        fields = dict(self.mapping.fields)
        fields["Course.name"] = FieldMapping(table="course_names", column="name")
        mapping = self.mapping.model_copy(update={"fields": fields})
        result = self.make_harness(
            {"entity": "Course", "select": ["Course.name", "Course.credits"], "filters": []}
        ).run({"database_id": "test-db", "question": "질문"}, self.ontology, mapping)

        self.assertEqual(result.status, HarnessStatus.FAILED)
        self.assertEqual(result.error.code, HarnessErrorCode.MULTI_TABLE_NOT_SUPPORTED)

    def test_missing_database_file_is_reported(self) -> None:
        harness = ExecutionHarness(
            model=FakeModel(
                {"질문": {"entity": "Course", "select": ["Course.name"], "filters": []}}
            ),
            compiler=SingleTableSQLiteCompiler(),
            sql_policy=ConservativeSqlPolicy(),
            executor=SQLiteReadOnlyExecutor({"test-db": Path(self.temp_dir.name) / "missing.db"}),
        )

        result = harness.run(
            {"database_id": "test-db", "question": "질문"}, self.ontology, self.mapping
        )

        self.assertEqual(result.status, HarnessStatus.FAILED)
        self.assertEqual(result.error.code, HarnessErrorCode.DATABASE_NOT_FOUND)

    def test_sql_policy_rejects_write_statements(self) -> None:
        policy = ConservativeSqlPolicy()

        with self.assertRaises(HarnessFailure) as raised:
            policy.validate(
                CompiledQuery(
                    sql="DELETE FROM courses WHERE credit_hours = ? LIMIT ?",
                    parameters=[3, 10],
                    expected_columns=[],
                )
            )

        self.assertEqual(raised.exception.code, HarnessErrorCode.UNSAFE_SQL)

    def test_extended_ontology_value_types_are_validated(self) -> None:
        ontology = Ontology(
            entities={"CourseOffering", "Schedule"},
            fields={
                "CourseOffering.year": OntologyField(
                    name="CourseOffering.year",
                    data_type=DataType.INTEGER,
                    allowed_operators={Operator.EQUALS},
                ),
                "Schedule.start_time": OntologyField(
                    name="Schedule.start_time",
                    data_type=DataType.TIME,
                    allowed_operators={Operator.EQUALS},
                ),
            },
        )
        mapping = MappingProfile(
            database_id="test-db",
            version=1,
            status=MappingStatus.PUBLISHED,
            fields={
                "CourseOffering.year": FieldMapping(table="courses", column="credit_hours"),
                "Schedule.start_time": FieldMapping(table="courses", column="subject_name"),
            },
        )
        valid_ir = QueryIR(
            entity="CourseOffering",
            select=["CourseOffering.year"],
            filters=[
                {"field": "CourseOffering.year", "operator": "equals", "value": 2026}
            ],
        )
        validate_query_ir(valid_ir, ontology, mapping)

        invalid_ir = QueryIR(
            entity="Schedule",
            select=["Schedule.start_time"],
            filters=[
                {"field": "Schedule.start_time", "operator": "equals", "value": 900}
            ],
        )
        with self.assertRaises(HarnessFailure) as raised:
            validate_query_ir(invalid_ir, ontology, mapping)
        self.assertEqual(raised.exception.code, HarnessErrorCode.INVALID_VALUE_TYPE)


# From test_schema_adaptive_harness.py

class SchemaAdaptiveHarnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        bundle = build_fixture_bundle(Path(self.temp_dir.name) / "fixtures")
        self.artifact = bundle.artifacts[FixtureVariant.NORMALIZED]
        catalog = DatabaseCatalog.model_validate_json(
            self.artifact.catalog_path.read_text(encoding="utf-8")
        )
        self.graph = build_physical_relation_graph(catalog)
        self.published = PublishedMappingProfile(
            database_id=catalog.database_id,
            version=1,
            schema_fingerprint=catalog.schema_fingerprint,
            ontology_version=1,
            published_by="test-admin",
            published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            fields={
                "Course.name": PublishedFieldMapping(
                    table="courses",
                    column="course_name",
                    reviewer="test-admin",
                ),
                "CourseOffering.year": PublishedFieldMapping(
                    table="course_offerings",
                    column="academic_year",
                    reviewer="test-admin",
                ),
                "Instructor.name": PublishedFieldMapping(
                    table="instructors",
                    column="instructor_name",
                    reviewer="test-admin",
                ),
                "Schedule.day": PublishedFieldMapping(
                    table="schedules",
                    column="day_of_week",
                    reviewer="test-admin",
                ),
            },
        )
        self.runtime_mapping = to_harness_mapping(self.published)
        self.ontology = Ontology(
            entities={"Course"},
            fields={
                "Course.name": OntologyField(
                    name="Course.name",
                    data_type=DataType.STRING,
                    allowed_operators={Operator.EQUALS},
                ),
                "CourseOffering.year": OntologyField(
                    name="CourseOffering.year",
                    data_type=DataType.INTEGER,
                    allowed_operators={Operator.EQUALS},
                ),
                "Instructor.name": OntologyField(
                    name="Instructor.name",
                    data_type=DataType.STRING,
                    allowed_operators={Operator.EQUALS},
                ),
                "Schedule.day": OntologyField(
                    name="Schedule.day",
                    data_type=DataType.STRING,
                    allowed_operators={Operator.EQUALS},
                ),
            },
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_existing_harness_executes_resolved_multi_table_query(self) -> None:
        response = {
            "entity": "Course",
            "select": ["Course.name", "Instructor.name", "Schedule.day"],
            "filters": [
                {"field": "Course.name", "operator": "equals", "value": "데이터베이스"},
                {"field": "CourseOffering.year", "operator": "equals", "value": 2026},
            ],
            "limit": 10,
        }
        harness = ExecutionHarness(
            model=FakeModel({"질문": response}),
            compiler=SchemaAdaptiveSQLiteCompiler(self.graph, self.published),
            sql_policy=SchemaAdaptiveAstPolicy(self.graph, self.published),
            executor=SQLiteReadOnlyExecutor(
                {self.artifact.database_id: self.artifact.database_path}
            ),
        )

        result = harness.run(
            {"database_id": self.artifact.database_id, "question": "질문"},
            self.ontology,
            self.runtime_mapping,
        )

        self.assertEqual(result.status, HarnessStatus.SUCCEEDED)
        self.assertEqual((result.sql or "").count(" JOIN "), 3)
        self.assertEqual(
            result.rows,
            [
                {
                    "Course.name": "데이터베이스",
                    "Instructor.name": "교원A",
                    "Schedule.day": "THU",
                }
            ],
        )

    def test_runtime_mapping_drift_is_rejected_before_execution(self) -> None:
        fields = dict(self.runtime_mapping.fields)
        fields["Course.name"] = fields["Course.name"].model_copy(
            update={"column": "unexpected_column"}
        )
        drifted_mapping = self.runtime_mapping.model_copy(update={"fields": fields})
        harness = ExecutionHarness(
            model=FakeModel(
                {
                    "질문": {
                        "entity": "Course",
                        "select": ["Course.name"],
                        "filters": [],
                    }
                }
            ),
            compiler=SchemaAdaptiveSQLiteCompiler(self.graph, self.published),
            executor=SQLiteReadOnlyExecutor(
                {self.artifact.database_id: self.artifact.database_path}
            ),
        )

        result = harness.run(
            {"database_id": self.artifact.database_id, "question": "질문"},
            self.ontology,
            drifted_mapping,
        )

        self.assertEqual(result.status.value, "FAILED")
        self.assertEqual(result.error.code, HarnessErrorCode.INVALID_JOIN_PLAN)
        self.assertIsNone(result.sql)
