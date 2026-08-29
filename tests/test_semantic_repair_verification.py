from __future__ import annotations

import unittest

from pipeline.semantic_repair_verification import (
    _audit_metrics,
    _compatible_batches,
    _issue_matches,
    _patch_was_independently_approved,
    _repair_source_batch_id,
)


class SemanticRepairVerificationTests(unittest.TestCase):
    def test_atomic_repair_batch_maps_back_to_original_audit_batch(self) -> None:
        self.assertEqual(
            _repair_source_batch_id({
                "batch_id": "chapter:semantic-audit-001::issue-002",
                "source_batch_id": "chapter:semantic-audit-001",
            }),
            "chapter:semantic-audit-001",
        )
        self.assertEqual(
            _repair_source_batch_id({"batch_id": "chapter:semantic-audit-001::issue-002"}),
            "chapter:semantic-audit-001",
        )

    def test_metrics_treat_every_reviewed_warning_as_blocking(self) -> None:
        report = {
            "passed": False,
            "chapter_count": 1,
            "batch_count": 1,
            "batches": [{"batch_id": "b1", "issues": [
                {"category": "coverage", "severity": "warning", "confidence": 0.9},
                {"category": "event_frame", "severity": "warning", "confidence": 0.5},
                {"category": "audit_contract", "severity": "error", "confidence": 1.0},
            ]}],
        }
        metrics = _audit_metrics(report)
        self.assertEqual(metrics["blocking_issue_count"], 3)
        self.assertEqual(metrics["audit_contract_failure_count"], 1)

    def test_coverage_match_uses_category_kind_and_evidence(self) -> None:
        original = {
            "category": "coverage",
            "object_ids": ["missing:event:chapter:p0001"],
            "evidence_unit_ids": ["chapter:p0001", "chapter:p0002"],
        }
        same = {
            "category": "coverage",
            "object_ids": ["missing:event:chapter:p0002"],
            "evidence_unit_ids": ["chapter:p0002"],
        }
        different_kind = {**same, "object_ids": ["missing:fact:chapter:p0002"]}
        self.assertTrue(_issue_matches(original, same, {}))
        self.assertFalse(_issue_matches(original, different_kind, {}))

    def test_identity_match_follows_entity_redirect(self) -> None:
        original = {
            "category": "entity_identity", "object_ids": ["entity-old"],
            "evidence_unit_ids": ["chapter:p0001"],
        }
        candidate = {
            "category": "entity_identity", "object_ids": ["entity-keep"],
            "evidence_unit_ids": ["chapter:p0001"],
        }
        self.assertTrue(_issue_matches(original, candidate, {"entity-old": "entity-keep"}))

    def test_compatibility_requires_same_windows_model_and_prompt(self) -> None:
        baseline = {
            "chapter_ids": ["chapter"], "model_id": "model", "prompt_version": "v1",
            "batches": [{"batch_id": "b1", "source_unit_ids": ["p1"]}],
        }
        self.assertTrue(_compatible_batches(baseline, dict(baseline))[0])
        changed = {**baseline, "model_id": "other"}
        compatible, reasons = _compatible_batches(baseline, changed)
        self.assertFalse(compatible)
        self.assertTrue(reasons)

    def test_compatibility_allows_entity_resolution_batch_count_to_shrink(self) -> None:
        baseline = {
            "chapter_ids": ["chapter"], "model_id": "model", "prompt_version": "v1",
            "batches": [
                {"batch_id": "chapter:semantic-audit-001", "source_unit_ids": ["p1"]},
                {"batch_id": "chapter:entity-resolution-001", "source_unit_ids": ["p1", "p2"]},
            ],
        }
        candidate = {
            **baseline,
            "batches": [
                {"batch_id": "chapter:semantic-audit-001", "source_unit_ids": ["p1"]},
            ],
        }
        compatible, reasons = _compatible_batches(baseline, candidate)
        self.assertTrue(compatible)
        self.assertEqual(reasons, [])

    def test_non_identity_resolution_requires_independent_patch_approval(self) -> None:
        self.assertTrue(_patch_was_independently_approved({
            "application": {"semantic_review": {"结论": "通过"}},
        }))
        self.assertFalse(_patch_was_independently_approved({
            "application": {"semantic_review": {"结论": "延期"}},
        }))
        self.assertFalse(_patch_was_independently_approved({"application": {}}))


if __name__ == "__main__":
    unittest.main()
