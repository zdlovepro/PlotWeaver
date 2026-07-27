from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from pipeline.contracts import (
    AuthorStyleProfile,
    NarrativeTemplate,
    NarrativeTemplateLibrary,
    StyleBaseline,
    StyleConstraint,
    WorkContinuity,
)
from pipeline.skill_compiler import _assess_inputs, _draft_prompt, _plan_prompt, _scene_draft_prompt, _scene_repair_prompt, _scene_validate_prompt, _skill_markdown, _validate_prompt
from pipeline.skill_runtime import _execution_mode_instruction, _render, _selected_templates


class SkillCompilerTests(unittest.TestCase):
    def test_prompt_programs_are_chinese_and_define_strict_json_examples(self) -> None:
        plan = _plan_prompt()
        validation = _validate_prompt()
        self.assertIn("只输出一个合法 JSON object", plan)
        self.assertIn("selected_template_ids", plan)
        self.assertIn("只输出一个合法 JSON object", validation)
        self.assertIn("style_observations", validation)
        scene_draft = _scene_draft_prompt()
        self.assertIn("段落正文", scene_draft)
        self.assertIn("叙事职责", scene_draft)
        self.assertIn("最低对白字数", scene_draft)
        self.assertIn("NARRATIVE_SUBGRAPH_JSON", scene_draft)
        self.assertIn("唯一可读取的跨段事实来源", scene_draft)
        self.assertNotIn("STORY_STATE_JSON", scene_draft)
        self.assertIn("提前泄露事件ID", _scene_validate_prompt())
        self.assertIn("未授权断言段落ID", _scene_validate_prompt())
        scene_repair = _scene_repair_prompt()
        self.assertIn("修复段落", scene_repair)
        self.assertIn("职责标签", scene_repair)
        self.assertIn("只输出正文", _draft_prompt())
        skill = _skill_markdown()
        self.assertIn("唯一跨段事实底座", skill)
        self.assertIn("chapter_graph_patch.candidate.json", skill)

    def test_matched_abstract_artifacts_pass_compiler_input_checks(self) -> None:
        ids = tuple(f"Author/work/{index:04d}" for index in range(1, 4))
        constraints = tuple(
            StyleConstraint(
                f"style:{family}-001", family, "采用可执行的抽象写作策略并保持因果清晰。",
                "在当前原创场景的变化处落实该策略并记录结果。", "不要机械套用固定表达或忽略当前原创状态。",
                ids[:2], 2, 0.75, (metric,),
            )
            for family, metric in (
                ("sentence_rhythm", "mean_sentence_chars"),
                ("paragraph_pacing", "short_paragraph_ratio"),
                ("dialogue", "dialogue_turn_count"),
                ("transition", "transition_marker_ratio"),
                ("focalization", "perception_marker_ratio"),
            )
        )
        profile = AuthorStyleProfile("Author", "work", ids, (StyleBaseline("paragraph_count", 2, 1, 3, "count"),), constraints, ("只使用抽象约束。",))
        templates = tuple(
            NarrativeTemplate(
                f"library:{level}-001", level, "以目标、行动、阻力和结果组织可复用结构。",
                ("行动者", "压力来源"), ("建立目标", "采取行动", "形成结果"),
                ("状态变化",), ("推进",), ("阻力来源",), ids, 3, 0.8,
            )
            for level in ("event", "scene", "chapter")
        )
        library = NarrativeTemplateLibrary("Author", "work", ids, "1.0", templates)
        continuity = WorkContinuity("Author", "work", ids, "2.1", (), (), (), ())
        self.assertEqual(_assess_inputs(profile, library, continuity, ()), ())

    def test_runtime_renders_json_and_rejects_unknown_template(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "prompt.md"
            template.write_text("状态：{STATE_JSON}\n草稿：{DRAFT_TEXT}", encoding="utf-8")
            rendered = _render(template, {"STATE_JSON": {"goal": "原创目标"}, "DRAFT_TEXT": "原创正文"})
        self.assertIn('"goal": "原创目标"', rendered)
        self.assertIn("原创正文", rendered)
        library = {"templates": [{"template_id": "library:event-001"}]}
        self.assertEqual(_selected_templates(library, {"selected_template_ids": ["library:event-001"]})[0]["template_id"], "library:event-001")
        with self.assertRaises(ValueError):
            _selected_templates(library, {"selected_template_ids": ["unknown"]})

    def test_fidelity_mode_permits_only_explicit_structural_inputs(self) -> None:
        instruction = _execution_mode_instruction({"reconstruction_mode": "fidelity_test_only"})
        self.assertIn("可使用", instruction)
        self.assertIn("不得自行补入", instruction)
        self.assertIn("不得输入、复制或改写原文句子", instruction)


if __name__ == "__main__":
    unittest.main()
