from __future__ import annotations

import unittest

from pipeline.annotate import _derive_scene_state_from_atoms, _require_atom_draft, _require_scene_state_draft


class AnnotationStagingTests(unittest.TestCase):
    def _atoms(self) -> dict[str, object]:
        return {
            "entities": [{"id": "e1"}],
            "facts": [{"id": f"f{index}"} for index in range(1, 7)],
            "events": [
                {"id": "v1", "order": 1, "participant_ids": ["e1"], "trigger_fact_ids": ["f1"], "precondition_fact_ids": [], "outcome_fact_ids": ["f2"], "action": "行动甲", "obstacle": "", "decision": "", "evidence": [{"unit_id": "u1", "quote": "甲"}]},
                {"id": "v2", "order": 2, "participant_ids": ["e1"], "trigger_fact_ids": [], "precondition_fact_ids": [], "outcome_fact_ids": ["f3"], "action": "行动乙", "obstacle": "阻力", "decision": "取舍", "evidence": [{"unit_id": "u2", "quote": "乙"}]},
                {"id": "v3", "order": 3, "participant_ids": ["e1"], "trigger_fact_ids": [], "precondition_fact_ids": [], "outcome_fact_ids": ["f4"], "action": "行动丙", "obstacle": "", "decision": "", "evidence": [{"unit_id": "u3", "quote": "丙"}]},
            ],
        }

    def test_scene_state_is_derived_from_ordered_atomic_events(self) -> None:
        result = _derive_scene_state_from_atoms(self._atoms())
        self.assertEqual([item["event_ids"] for item in result["scenes"]], [["v1"], ["v2"], ["v3"]])
        self.assertEqual(result["scenes"][1]["entry_fact_ids"], ["f2"])
        self.assertEqual(result["scenes"][1]["tension"], 3)
        self.assertEqual([item["after_fact_id"] for item in result["state_changes"]], ["f2", "f3", "f4"])
        self.assertEqual(_require_scene_state_draft(result), result)

    def test_atom_gate_rejects_empty_or_under_dense_payload(self) -> None:
        with self.assertRaises(ValueError):
            _require_atom_draft({}, ())
        with self.assertRaises(ValueError):
            _require_atom_draft({"entities": [], "facts": [], "events": []}, ())


if __name__ == "__main__":
    unittest.main()
