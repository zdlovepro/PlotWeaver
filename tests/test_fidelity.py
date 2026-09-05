from __future__ import annotations

import json
import unittest

from pipeline.fidelity import _annotation_structure, _hard_structure_pass


class FidelityTests(unittest.TestCase):
    def test_structure_input_removes_source_evidence_and_resolves_ids(self) -> None:
        annotation = {
            "chapter_id": "Author/work/0001",
            "entities": [{"entity_id": "entity-a", "canonical_name": "角色甲", "kind": "person", "aliases": ["甲"], "evidence": [{"quote": "原文"}]}],
            "facts": [{"fact_id": "fact-a", "subject_id": "entity-a", "predicate": "获得", "value": "信物", "object_id": "", "kind": "resource", "certainty": 1, "evidence": [{"quote": "原文"}]}],
            "events": [{"event_id": "event-a", "order": 0, "summary": "角色甲获得信物", "participant_ids": ["entity-a"], "action": "取得信物", "obstacle": "", "decision": "", "precondition_fact_ids": [], "outcome_fact_ids": ["fact-a"], "cost_fact_ids": [], "evidence": [{"quote": "原文"}]}],
            "scenes": [{"order": 0, "participant_ids": ["entity-a"], "event_ids": ["event-a"], "objective": "取得信物", "entry_fact_ids": [], "exit_fact_ids": ["fact-a"], "tension": 2, "location_ids": [], "time_anchor_ids": [], "evidence": [{"quote": "原文"}]}],
            "state_changes": [{"operation": "add", "event_id": "event-a", "before_fact_id": "", "after_fact_id": "fact-a", "evidence": [{"quote": "原文"}]}],
            "time_anchors": [{"anchor_id": "time-a", "label": "随后", "kind": "relative", "evidence": [{"quote": "原文"}]}],
            "temporal_relations": [], "spatial_relations": [],
        }
        structure = _annotation_structure(annotation)
        rendered = json.dumps(structure, ensure_ascii=False)
        self.assertNotIn("evidence", rendered)
        self.assertNotIn("原文", rendered)
        self.assertIn("角色甲 获得 信物", rendered)
        self.assertEqual(structure["events"][0]["participants"], ["角色甲"])
        self.assertEqual(structure["events"][0]["contract_id"], "target-event-001")
        self.assertEqual(structure["state_changes"][0]["contract_id"], "target-change-001")

    def test_explicit_missing_requirement_cannot_pass_by_score(self) -> None:
        judgment = {
            "scores": {"event_causality": 5, "state_coverage": 5, "temporal_spatial": 5, "scene_pacing": 5, "logic": 5},
            "missing_requirements": ["关键事件未实现"],
            "contradictions": [],
        }
        self.assertFalse(_hard_structure_pass(judgment))


if __name__ == "__main__":
    unittest.main()
