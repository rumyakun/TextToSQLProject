from __future__ import annotations

import unittest
from backend.adaptive.harness import (
    FieldMapping,
    HarnessRequest,
    MappingProfile,
    MappingStatus,
    Ontology,
    QueryIR,
    RecordedFailure,
    RecordedFailureCode,
    RecordedModel,
    RecordedSuccess,
    RecordingCaptureError,
    RecordingCaptureErrorCode,
    RecordingCaptureSession,
    mapping_fingerprint,
    question_fingerprint,
)
import copy
from typing import Any
from backend.adaptive.harness import (
    CaptureTarget,
    FieldMapping,
    HarnessRequest,
    MappingProfile,
    MappingStatus,
    Ontology,
    QueryIR,
    RecordedFailure,
    RecordedFailureCode,
    RecordedModel,
    RecordedSuccess,
    RecordingOrchestrationError,
    RecordingOrchestrationErrorCode,
    RecordingProviderError,
    capture_recording,
)


# From test_recording_capture.py

class _PrivateObject:
    def __deepcopy__(self, memo):
        del memo
        raise RuntimeError("PRIVATE_RESPONSE_LITERAL")


class RecordingCaptureSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping = MappingProfile(
            database_id="fixture-flat-v1",
            version=1,
            status=MappingStatus.PUBLISHED,
            fields={"Course.name": FieldMapping(table="courses", column="name")},
        )

    @staticmethod
    def _request(index: int, question: str | None = None) -> HarnessRequest:
        return HarnessRequest(
            request_id=f"capture-run:P2-{index:03d}:flat",
            database_id="fixture-flat-v1",
            question=question or f"합성 질문 {index}",
        )

    def test_success_values_preserve_json_types_order_and_identity(self) -> None:
        values = [
            {"nested": [1, "값", None]},
            "invalid QueryIR string",
            17,
            1.25,
            True,
            None,
            [False, 2, "목록"],
        ]
        session = RecordingCaptureSession("capture-run-v1")
        requests = []
        for index, value in enumerate(values, start=1):
            request = self._request(index)
            requests.append(request)
            session.record_success(request, self.mapping, value)

        recording = session.finalize()
        self.assertEqual(recording.model_run_id, "capture-run-v1")
        self.assertEqual(
            [record.request_id for record in recording.records],
            [request.request_id for request in requests],
        )
        for record, request, expected in zip(
            recording.records, requests, values, strict=True
        ):
            with self.subTest(request_id=request.request_id):
                self.assertIsInstance(record, RecordedSuccess)
                self.assertEqual(record.response, expected)
                self.assertIs(type(record.response), type(expected))
                self.assertEqual(
                    record.question_sha256, question_fingerprint(request.question)
                )
                self.assertEqual(
                    record.mapping_sha256, mapping_fingerprint(self.mapping)
                )
                self.assertEqual(record.database_id, request.database_id)

    def test_response_is_defensively_copied_and_inputs_are_unchanged(self) -> None:
        request = self._request(1)
        request_before = request.model_dump(mode="json")
        mapping_before = self.mapping.model_dump(mode="json")
        response = {"nested": {"count": 1}}
        session = RecordingCaptureSession("capture-run-v1")
        session.record_success(request, self.mapping, response)
        response["nested"]["count"] = 99

        first = session.finalize()
        self.assertEqual(first.records[0].response["nested"]["count"], 1)
        first.records.clear()
        second = session.finalize()
        self.assertEqual(len(second.records), 1)
        self.assertEqual(second.records[0].response["nested"]["count"], 1)
        self.assertEqual(request.model_dump(mode="json"), request_before)
        self.assertEqual(self.mapping.model_dump(mode="json"), mapping_before)

    def test_each_explicit_failure_code_is_recorded(self) -> None:
        session = RecordingCaptureSession("capture-run-v1")
        for index, failure_code in enumerate(RecordedFailureCode, start=1):
            session.record_failure(self._request(index), self.mapping, failure_code)
        recording = session.finalize()
        self.assertEqual(
            [record.failure_code for record in recording.records],
            list(RecordedFailureCode),
        )
        self.assertTrue(
            all(isinstance(record, RecordedFailure) for record in recording.records)
        )

    def test_duplicate_ids_never_overwrite_existing_success_or_failure(self) -> None:
        scenarios = ("success-first", "failure-first")
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                request = self._request(1)
                session = RecordingCaptureSession("capture-run-v1")
                if scenario == "success-first":
                    session.record_success(request, self.mapping, {"kept": True})
                    expected_type = RecordedSuccess
                else:
                    session.record_failure(
                        request, self.mapping, RecordedFailureCode.TIMEOUT
                    )
                    expected_type = RecordedFailure
                with self.assertRaises(RecordingCaptureError) as caught:
                    if scenario == "success-first":
                        session.record_failure(
                            request, self.mapping, RecordedFailureCode.TIMEOUT
                        )
                    else:
                        session.record_success(
                            request, self.mapping, {"replacement": True}
                        )
                self.assertIs(
                    caught.exception.code,
                    RecordingCaptureErrorCode.DUPLICATE_REQUEST_ID,
                )
                recording = session.finalize()
                self.assertEqual(len(recording.records), 1)
                self.assertIsInstance(recording.records[0], expected_type)

    def test_invalid_values_are_rejected_without_consuming_request_id(self) -> None:
        invalid_values = (
            float("nan"),
            float("inf"),
            float("-inf"),
            {"nested": float("nan")},
            b"bytes",
            object(),
            _PrivateObject(),
        )
        for value in invalid_values:
            with self.subTest(value_type=type(value).__name__):
                request = self._request(1, "PRIVATE_QUESTION_LITERAL")
                session = RecordingCaptureSession("capture-run-v1")
                with self.assertRaises(RecordingCaptureError) as caught:
                    session.record_success(request, self.mapping, value)
                self.assertIs(
                    caught.exception.code, RecordingCaptureErrorCode.INVALID_RESPONSE
                )
                self.assertEqual(str(caught.exception), "INVALID_RESPONSE")
                self.assertNotIn("PRIVATE", str(caught.exception))
                session.record_success(request, self.mapping, "safe-json-string")
                self.assertEqual(len(session.finalize().records), 1)

    def test_invalid_failure_code_and_model_run_id_are_safe(self) -> None:
        private = "PRIVATE_FAILURE_OR_RUN_ID"
        with self.assertRaises(RecordingCaptureError) as caught:
            RecordingCaptureSession(private + "!")
        self.assertIs(
            caught.exception.code, RecordingCaptureErrorCode.INVALID_MODEL_RUN_ID
        )
        self.assertNotIn(private, str(caught.exception))

        session = RecordingCaptureSession("capture-run-v1")
        with self.assertRaises(RecordingCaptureError) as caught:
            session.record_failure(
                self._request(1, private),
                self.mapping,
                "TIMEOUT",  # type: ignore[arg-type]
            )
        self.assertIs(
            caught.exception.code, RecordingCaptureErrorCode.INVALID_FAILURE_CODE
        )
        self.assertNotIn(private, str(caught.exception))

    def test_empty_finalize_and_capture_after_finalize_are_blocked(self) -> None:
        empty = RecordingCaptureSession("capture-run-v1")
        with self.assertRaises(RecordingCaptureError) as caught:
            empty.finalize()
        self.assertIs(caught.exception.code, RecordingCaptureErrorCode.EMPTY_SESSION)

        session = RecordingCaptureSession("capture-run-v1")
        session.record_success(self._request(1), self.mapping, {})
        session.finalize()
        with self.assertRaises(RecordingCaptureError) as caught:
            session.record_success(self._request(2), self.mapping, {})
        self.assertIs(
            caught.exception.code, RecordingCaptureErrorCode.SESSION_FINALIZED
        )

    def test_same_inputs_are_deterministic_and_replayable(self) -> None:
        request = self._request(1)
        response = {
            "entity": "Course",
            "select": ["Course.name"],
            "filters": [],
            "limit": 10,
        }
        recordings = []
        for _ in range(2):
            session = RecordingCaptureSession("capture-run-v1")
            session.record_success(request, self.mapping, response)
            recordings.append(session.finalize())
        self.assertEqual(
            recordings[0].model_dump_json(), recordings[1].model_dump_json()
        )
        replayed = RecordedModel(recordings[0]).generate_ir(
            request, Ontology(entities=set(), fields={}), self.mapping
        )
        self.assertEqual(
            QueryIR.model_validate(replayed), QueryIR.model_validate(response)
        )


# From test_recording_orchestrator.py

class _SequenceProvider:
    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = outcomes
        self.calls: list[str] = []
        self.responses: list[Any] = []

    def generate_ir(self, request, ontology, mapping):
        del ontology, mapping
        self.calls.append(request.request_id)
        outcome = self._outcomes[len(self.calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        self.responses.append(outcome)
        return outcome


class RecordingOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ontology = Ontology(entities={"Course"}, fields={})
        self.mapping = MappingProfile(
            database_id="fixture-flat-v1",
            version=1,
            status=MappingStatus.PUBLISHED,
            fields={"Course.name": FieldMapping(table="courses", column="name")},
        )

    def _target(self, index: int, *, database_id: str | None = None) -> CaptureTarget:
        return CaptureTarget(
            request=HarnessRequest(
                request_id=f"capture:P2-{index:03d}:flat",
                database_id=database_id or self.mapping.database_id,
                question=f"합성 질문 {index}",
            ),
            ontology=self.ontology,
            mapping=self.mapping,
        )

    def test_success_json_types_order_and_single_invocation_are_preserved(self) -> None:
        values = [
            {"nested": [1, "값", None]},
            "invalid QueryIR string",
            17,
            1.25,
            True,
            None,
            [False, 2, "목록"],
        ]
        targets = [self._target(index) for index in range(1, len(values) + 1)]
        provider = _SequenceProvider(values)

        recording = capture_recording("provider-run-v1", provider, targets)

        self.assertEqual(provider.calls, [item.request.request_id for item in targets])
        self.assertEqual(len(recording.records), len(values))
        for record, expected in zip(recording.records, values, strict=True):
            self.assertIsInstance(record, RecordedSuccess)
            self.assertEqual(record.response, expected)
            self.assertIs(type(record.response), type(expected))

    def test_all_typed_failures_are_recorded_and_later_targets_continue(self) -> None:
        outcomes = [RecordingProviderError(code) for code in RecordedFailureCode]
        outcomes.append({"kept": True})
        targets = [self._target(index) for index in range(1, len(outcomes) + 1)]
        provider = _SequenceProvider(outcomes)

        recording = capture_recording("provider-run-v1", provider, targets)

        self.assertEqual(len(provider.calls), len(targets))
        self.assertEqual(
            [record.failure_code for record in recording.records[:-1]],
            list(RecordedFailureCode),
        )
        self.assertTrue(
            all(isinstance(record, RecordedFailure) for record in recording.records[:-1])
        )
        self.assertIsInstance(recording.records[-1], RecordedSuccess)

        all_failed = capture_recording(
            "provider-run-all-failed",
            _SequenceProvider(
                [RecordingProviderError(code) for code in RecordedFailureCode]
            ),
            targets[:-1],
        )
        self.assertEqual(len(all_failed.records), len(RecordedFailureCode))
        self.assertTrue(
            all(isinstance(record, RecordedFailure) for record in all_failed.records)
        )

    def test_unexpected_exception_and_invalid_json_become_safe_provider_failures(self) -> None:
        private = "PRIVATE_PROVIDER_EXCEPTION_OR_RESPONSE"
        invalid_values = [
            RuntimeError(private),
            float("nan"),
            float("inf"),
            b"bytes",
            object(),
            {"nested": float("nan")},
            {"safe": True},
        ]
        targets = [self._target(index) for index in range(1, len(invalid_values) + 1)]
        provider = _SequenceProvider(invalid_values)

        recording = capture_recording("provider-run-v1", provider, targets)

        self.assertEqual(len(provider.calls), len(targets))
        for record in recording.records[:-1]:
            self.assertIsInstance(record, RecordedFailure)
            self.assertIs(record.failure_code, RecordedFailureCode.PROVIDER_ERROR)
            self.assertNotIn(private, record.model_dump_json())
        self.assertIsInstance(recording.records[-1], RecordedSuccess)

    def test_preflight_rejects_bad_batch_before_any_provider_call(self) -> None:
        class ExplodingProvider:
            @property
            def generate_ir(self):
                raise RuntimeError("PRIVATE_PROVIDER_PROPERTY")

        scenarios = [
            (
                RecordingOrchestrationErrorCode.EMPTY_TARGETS,
                _SequenceProvider([]),
                [],
            ),
            (
                RecordingOrchestrationErrorCode.INVALID_PROVIDER,
                object(),
                [self._target(1)],
            ),
            (
                RecordingOrchestrationErrorCode.INVALID_PROVIDER,
                ExplodingProvider(),
                [self._target(1)],
            ),
            (
                RecordingOrchestrationErrorCode.DUPLICATE_REQUEST_ID,
                _SequenceProvider([{}, {}]),
                [self._target(1), self._target(1)],
            ),
            (
                RecordingOrchestrationErrorCode.DATABASE_MISMATCH,
                _SequenceProvider([{}]),
                [self._target(1, database_id="different-db")],
            ),
            (
                RecordingOrchestrationErrorCode.INVALID_TARGET,
                _SequenceProvider([{}]),
                [
                    CaptureTarget.model_construct(
                        request="invalid",
                        ontology=self.ontology,
                        mapping=self.mapping,
                    )
                ],
            ),
        ]

        for expected_code, provider, targets in scenarios:
            with self.subTest(code=expected_code):
                calls_before = list(getattr(provider, "calls", []))
                with self.assertRaises(RecordingOrchestrationError) as caught:
                    capture_recording(
                        "provider-run-v1",
                        provider,  # type: ignore[arg-type]
                        targets,
                    )
                self.assertIs(caught.exception.code, expected_code)
                self.assertEqual(str(caught.exception), expected_code.value)
                self.assertEqual(getattr(provider, "calls", []), calls_before)

    def test_provider_cannot_mutate_caller_context_or_recorded_response(self) -> None:
        target = self._target(1)
        request_before = target.request.model_dump(mode="json")
        ontology_before = target.ontology.model_dump(mode="json")
        mapping_before = target.mapping.model_dump(mode="json")
        response = {"nested": {"count": 1}}

        class MutatingProvider:
            def generate_ir(self, request, ontology, mapping):
                request.question = "changed"
                ontology.entities.clear()
                mapping.fields.clear()
                return response

        recording = capture_recording(
            "provider-run-v1", MutatingProvider(), [target]
        )
        response["nested"]["count"] = 99

        self.assertEqual(target.request.model_dump(mode="json"), request_before)
        self.assertEqual(target.ontology.model_dump(mode="json"), ontology_before)
        self.assertEqual(target.mapping.model_dump(mode="json"), mapping_before)
        self.assertEqual(recording.records[0].response["nested"]["count"], 1)

    def test_same_inputs_are_deterministic_and_replayable(self) -> None:
        response = {
            "entity": "Course",
            "select": ["Course.name"],
            "filters": [],
            "limit": 10,
        }
        target = self._target(1)
        recordings = []
        for _ in range(2):
            recordings.append(
                capture_recording(
                    "provider-run-v1",
                    _SequenceProvider([copy.deepcopy(response)]),
                    [target],
                )
            )

        self.assertEqual(
            recordings[0].model_dump_json(), recordings[1].model_dump_json()
        )
        replayed = RecordedModel(recordings[0]).generate_ir(
            target.request,
            target.ontology,
            target.mapping,
        )
        self.assertEqual(
            QueryIR.model_validate(replayed), QueryIR.model_validate(response)
        )

    def test_provider_failure_contract_rejects_untyped_code_without_payload(self) -> None:
        private = "PRIVATE_FAILURE_CODE"
        with self.assertRaises(TypeError) as caught:
            RecordingProviderError(private)  # type: ignore[arg-type]
        self.assertEqual(str(caught.exception), "INVALID_PROVIDER_FAILURE_CODE")
        self.assertNotIn(private, str(caught.exception))
