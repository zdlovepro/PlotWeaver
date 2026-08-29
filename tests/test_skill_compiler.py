from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from pipeline.contracts import (
    AuthorSkillBundle,
    AuthorStyleProfile,
    DistilledAuthorTrait,
    NarrativeTemplate,
    NarrativeTemplateLibrary,
    SkillQualification,
    StyleBaseline,
    StyleConstraint,
    WorkContinuity,
)
from pipeline.skill_compiler import _assess_inputs, _draft_prompt, _draft_release_issues, _paragraph_repair_prompt, _plan_prompt, _scene_draft_prompt, _scene_repair_prompt, _scene_validate_prompt, _semantic_audit_evidence, _skill_markdown, _validate_prompt
from pipeline.skill_package import build_draft_references, build_openai_yaml, normalize_skill_name, select_style_context_script, self_check_script
from pipeline.skill_runtime import _execution_mode_instruction, _package_file, _render, _selected_templates


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
        self.assertIn("PARAGRAPH_WRITER_PACKET_JSON", scene_draft)
        self.assertIn("一个当前段落", scene_draft)
        self.assertIn("唯一可读取的跨段事实来源", scene_draft)
        self.assertIn("最低正文字符", scene_draft)
        self.assertIn("目标正文字符", scene_draft)
        self.assertIn("最后长度修复规则", _paragraph_repair_prompt())
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

    def test_release_contract_does_not_promote_single_work_evidence(self) -> None:
        qualification = SkillQualification(source_work_count=1, portable_self_check_passed=True)
        self.assertEqual(qualification.highest_release_status, "draft")
        bundle = AuthorSkillBundle(
            author_id="Author",
            skill_name="author-novel-author",
            release_status="draft",
            source_work_ids=("work-001",),
            chapter_ids=("Author/work-001/0001",),
            candidate_trait_count=1,
            validated_author_trait_count=0,
            narrative_policy_count=1,
            generated_files=("SKILL.md",),
            limitations=("只有单作品证据。",),
            qualification=qualification,
        )
        bundle.validate()
        with self.assertRaises(ValueError):
            AuthorSkillBundle(
                **{**bundle.__dict__, "release_status": "validated"},
            ).validate()

    def test_validated_trait_requires_cross_work_and_contrast(self) -> None:
        candidate = DistilledAuthorTrait(
            "trait-001", "dialogue", "规则", "应用", "避免",
            ("work-001",), ("Author/work-001/0001",), 1, 0.8,
        )
        candidate.validate()
        with self.assertRaises(ValueError):
            DistilledAuthorTrait(
                **{**candidate.__dict__, "validation_status": "validated"},
            ).validate()

    def test_standard_skill_metadata_and_scripts_are_valid_python(self) -> None:
        skill_name = normalize_skill_name("Example_Author")
        self.assertEqual(skill_name, "example-author-novel-author")
        skill = _skill_markdown(skill_name)
        self.assertIn(f"name: {skill_name}", skill)
        self.assertIn("references/author_core.json", skill)
        openai_yaml = build_openai_yaml(skill_name, "Example_Author")
        self.assertIn(f"${skill_name}", openai_yaml)
        compile(self_check_script(), "self_check.py", "exec")
        compile(select_style_context_script(), "select_style_context.py", "exec")

    def test_runtime_prefers_standard_layout_and_supports_legacy_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy.json"
            legacy.write_text("{}", encoding="utf-8")
            self.assertEqual(_package_file(root, "references/value.json", "legacy.json"), legacy)
            modern = root / "references" / "value.json"
            modern.parent.mkdir()
            modern.write_text("{}", encoding="utf-8")
            self.assertEqual(_package_file(root, "references/value.json", "legacy.json"), modern)

    def test_single_work_references_leave_author_core_empty(self) -> None:
        ids = tuple(f"Author/work/{index:04d}" for index in range(1, 4))
        constraint = StyleConstraint(
            "style:dialogue-001", "dialogue", "用对话推动信息变化。",
            "在人物交锋场景中执行。", "不要写成无结果寒暄。",
            ids[:2], 2, 0.75, ("dialogue_turn_count",),
        )
        profile = AuthorStyleProfile(
            "Author", "work", ids,
            (StyleBaseline("dialogue_turn_count", 3, 1, 5, "count"),),
            (constraint,), ("不复用来源正文。",),
        )
        template = NarrativeTemplate(
            "library:chapter-001", "chapter", "以压力推动目标变化。",
            ("行动者",), ("目标", "阻力", "结果"), ("目标变化",),
            ("推进",), ("阻力",), ids, 3, 0.8,
        )
        library = NarrativeTemplateLibrary("Author", "work", ids, "1.0", (template,))
        continuity = WorkContinuity("Author", "work", ids, "2.1", (), (), (), ())
        references = build_draft_references(profile, library, continuity, {"metrics": []})
        self.assertEqual(references["references/author_core.json"]["validated_traits"], [])
        traits = references["references/work_deltas.json"]["works"][0]["candidate_traits"]
        self.assertEqual(traits[0]["validation_status"], "single_work_candidate")

    def test_semantic_audit_summary_uses_largest_sample_and_release_warning_is_honest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "semantic_audit.sample-1.quality.json").write_text(
                '{"pass_scope":"model_content_semantic_audit","content_audit_chapter_count":1,"passed":false,"issue_count":2,"error_count":1}',
                encoding="utf-8",
            )
            (folder / "semantic_audit.sample-5.quality.json").write_text(
                '{"pass_scope":"model_content_semantic_audit","content_audit_chapter_count":5,"passed":true,"issue_count":0,"error_count":0,"report":"semantic_audit.sample-5.json"}',
                encoding="utf-8",
            )
            evidence = _semantic_audit_evidence(folder)
        self.assertEqual(evidence["content_audit_chapter_count"], 5)
        self.assertTrue(evidence["passed"])
        codes = {item.code for item in _draft_release_issues(evidence)}
        self.assertNotIn("semantic_audit_missing", codes)
        self.assertNotIn("semantic_audit_failed", codes)


if __name__ == "__main__":
    unittest.main()
    SkillQualification,
