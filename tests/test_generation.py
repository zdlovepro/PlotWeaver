from __future__ import annotations

import unittest
from unittest.mock import patch

from pipeline.generation import _chapter_minimum_chars, _complete_draft, _fidelity_plan_error, _prefer_style_repair, _segment_plan, _segment_story_state
from pipeline.model import ModelSettings


class GenerationTests(unittest.TestCase):
    def test_empty_model_text_is_retried_before_style_measurement(self) -> None:
        with patch(
            "pipeline.generation.complete_text",
            side_effect=[RuntimeError("temporary API failure"), "", "x" * 140],
        ) as complete:
            draft = _complete_draft("system", "prompt", ModelSettings(api_key="test"))
        self.assertEqual(len(draft), 140)
        self.assertEqual(complete.call_count, 3)

    def test_near_complete_segment_is_not_retried_for_three_char_difference(self) -> None:
        with patch("pipeline.generation.complete_text", return_value="x" * 956) as complete:
            draft = _complete_draft("system", "prompt", ModelSettings(api_key="test"), minimum_chars=959)
        self.assertEqual(len(draft), 956)
        self.assertEqual(complete.call_count, 1)

    def test_fidelity_plan_requires_each_explicit_contract(self) -> None:
        brief = {
            "reconstruction_mode": "fidelity_test_only",
            "required_events": [{"contract_id": "target-event-001"}],
            "required_state_changes": [{"contract_id": "target-change-001"}],
        }
        incomplete = {"covered_required_event_ids": ["target-event-001"], "covered_required_state_change_ids": [], "chapter_end_contract": []}
        self.assertIn("target-change-001", _fidelity_plan_error(incomplete, brief))
        complete = {"covered_required_event_ids": ["target-event-001"], "covered_required_state_change_ids": ["target-change-001"], "chapter_end_contract": ["在事件结果后结束"]}
        self.assertEqual(_fidelity_plan_error(complete, brief), "")

    def test_long_fidelity_plan_can_be_split_by_scene(self) -> None:
        plan = {
            "style_budget": {"metrics": [{"metric_id": "chapter_char_count", "preferred_min": 4400}]},
            "scenes": [{"scene_no": 1}, {"scene_no": 2}],
            "chapter_state_changes": [{"cause_scene_no": 1}, {"cause_scene_no": 2}],
            "covered_required_event_ids": ["target-event-001"],
        }
        self.assertEqual(_chapter_minimum_chars(plan), 4400)
        segment = _segment_plan(plan, plan["scenes"][0], 1, 2, 1900, {"events": ["事件甲"]})
        self.assertEqual(segment["scenes"], [{"scene_no": 1}])
        self.assertEqual(segment["chapter_state_changes"], [{"cause_scene_no": 1}])
        self.assertNotIn("covered_required_event_ids", segment)
        self.assertEqual(segment["current_segment"]["minimum_chars"], 1900)
        self.assertEqual(segment["scene_contract"]["events"], ["事件甲"])

    def test_segment_state_excludes_whole_chapter_targets(self) -> None:
        state = {"reconstruction_mode": "fidelity_test_only", "chapter_id": "work/0001", "known_entities": [{"name": "角色甲"}], "known_facts": ["不应显示"], "current_chapter_target": {"events": ["不应显示"]}}
        segment_state = _segment_story_state(state)
        self.assertIn("known_entities", segment_state)
        self.assertNotIn("known_facts", segment_state)
        self.assertNotIn("current_chapter_target", segment_state)

    def test_style_repair_candidate_must_improve_measured_budget(self) -> None:
        current = {"passed": False, "high_priority_in_range": 1, "mean_relative_deviation": 0.5}
        worse = {"passed": False, "high_priority_in_range": 1, "mean_relative_deviation": 0.49}
        improved = {"passed": False, "high_priority_in_range": 2, "mean_relative_deviation": 0.4}
        self.assertFalse(_prefer_style_repair(current, worse))
        self.assertTrue(_prefer_style_repair(current, improved))


if __name__ == "__main__":
    unittest.main()
