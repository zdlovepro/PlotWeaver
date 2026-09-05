from __future__ import annotations

import hashlib
import unittest

from pipeline.contracts import AuthorStyleProfile, StyleBaseline, build_chapter_document
from pipeline.style_distillation import build_chapter_style_card
from pipeline.style_execution import assess_style_budget, budget_from_metrics, resolve_style_budget, style_execution_blueprint, style_execution_directives, style_execution_spec


class StyleExecutionTests(unittest.TestCase):
    def test_reference_budget_accepts_its_own_measured_text(self) -> None:
        text = "他看见远处的光。\n\n“现在走吗？”她问。\n\n随后，他缓缓向前。"
        document = build_chapter_document("Author/work/0001", hashlib.sha256(text.encode("utf-8")).hexdigest(), text, "Start")
        budget = budget_from_metrics(build_chapter_style_card(document).metrics)
        report = assess_style_budget(text, budget)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["repair_priorities"], [])

    def test_fidelity_budget_can_constrain_chapter_length(self) -> None:
        text = "甲在雨中等了片刻。\n乙随后赶来。"
        document = build_chapter_document("test/length", hashlib.sha256(text.encode("utf-8")).hexdigest(), text, "测试")
        budget = budget_from_metrics(build_chapter_style_card(document).metrics, chapter_char_count=2400)
        row = next(item for item in budget["metrics"] if item["metric_id"] == "chapter_char_count")
        self.assertEqual(row["target"], 2400.0)
        self.assertEqual(row["priority"], "high")

    def test_length_budget_builds_paragraph_and_sentence_blueprint(self) -> None:
        budget = {"metrics": [
            {"metric_id": "chapter_char_count", "unit": "count", "target": 2400, "preferred_min": 2000, "preferred_max": 2800, "priority": "high"},
            {"metric_id": "mean_paragraph_chars", "unit": "mean_chars", "target": 60, "preferred_min": 50, "preferred_max": 70, "priority": "high"},
            {"metric_id": "mean_sentence_chars", "unit": "mean_chars", "target": 40, "preferred_min": 35, "preferred_max": 45, "priority": "high"},
            {"metric_id": "dialogue_char_ratio", "unit": "ratio", "target": 0.2, "preferred_min": 0.15, "preferred_max": 0.25, "priority": "high"},
        ]}
        blueprint = style_execution_blueprint(budget)
        self.assertEqual(blueprint["chapter_char_count"]["minimum"], 2000)
        self.assertGreater(blueprint["paragraph_count"]["target"], 1)
        self.assertGreater(blueprint["sentence_count"]["target"], 1)

    def test_planner_budget_is_constrained_to_author_range(self) -> None:
        profile = AuthorStyleProfile(
            "Author", "work", ("Author/work/0001", "Author/work/0002"),
            (
                StyleBaseline("mean_sentence_chars", 40, 20, 60, "mean_chars"),
                StyleBaseline("mean_paragraph_chars", 60, 40, 80, "mean_chars"),
                StyleBaseline("short_paragraph_ratio", 0.2, 0.1, 0.3, "ratio"),
            ),
            (), ("使用抽象约束。",),
        )
        spec = style_execution_spec(profile)
        resolved = resolve_style_budget(spec, {"metrics": [{
            "metric_id": "mean_sentence_chars", "unit": "mean_chars", "target": 999,
            "preferred_min": 900, "preferred_max": 1000, "priority": "high",
        }]})
        sentence = next(item for item in resolved["metrics"] if item["metric_id"] == "mean_sentence_chars")
        self.assertLessEqual(sentence["target"], sentence["preferred_max"])
        self.assertLessEqual(sentence["target"], 50)
        self.assertGreaterEqual(sentence["target"], 30)

    def test_style_directives_turn_largest_deviation_into_an_action(self) -> None:
        budget = {"metrics": [{
            "metric_id": "perception_marker_ratio", "unit": "ratio", "target": 0.15,
            "preferred_min": 0.1, "preferred_max": 0.2, "priority": "high",
        }]}
        assessment = {"metrics": [{
            "metric_id": "perception_marker_ratio", "actual": 0.01, "relative_deviation": 0.93,
        }]}
        directives = style_execution_directives(budget, assessment)
        self.assertEqual(directives[0]["direction"], "increase")
        self.assertIn("看到", directives[0]["action"])


if __name__ == "__main__":
    unittest.main()
