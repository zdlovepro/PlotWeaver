from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

from pipeline.contracts import ChapterProgram, EntityBinding, EventProgram, FactContract, NarrativeBeat, ParagraphDraft, ParagraphProgram, SceneDraft, SceneProgram
from pipeline.contracts.program_llm import chapter_program_to_llm_dict
from pipeline.contracts.scene_output import paragraph_validation_from_llm, scene_affordance_validation_from_llm, scene_draft_from_llm, scene_draft_to_llm_dict, scene_prose_validation_from_llm, scene_prose_validation_to_llm_dict, scene_validation_from_llm, scene_validation_to_llm_dict
from pipeline.graph_runtime import paragraph_writer_packet
from pipeline.jsonio import read_json, write_json
from pipeline.model import ModelSettings
from pipeline.scene_generation import _batch_scene_payload, _paragraph_minimum, _paragraph_role_issues, _paragraph_style_budgets, _prefer_style_assessment, _replace_non_role_marker, _scene_blueprint, _style_targets, _unlicensed_entity_issues, generate_from_program


class SceneGenerationTests(unittest.TestCase):
    def _scene(self) -> SceneProgram:
        fact = FactContract("fact-001", "scene-001", "goal", "person-001", "决定", "离开", expression_mode="internal")
        event = EventProgram(
            "event-001", "scene-001", "人物决定离开", "做出离开决定", ("person-001",), (), ("fact-001",), ("fact-001",), "", "离开",
        )
        paragraph = ParagraphProgram("scene-001:paragraph-00", "scene-001", 0, "decision", ("fact-001",), ("event-001",), "person-001")
        return SceneProgram("scene-001", "chapter-001", 0, "做出离开决定", ("person-001",), (), (event,), (fact,), (paragraph,), ("fact-001",), ())

    def _beat_scene(self) -> SceneProgram:
        fact = FactContract("fact-001", "scene-001", "goal", "person-001", "决定", "离开", expression_mode="internal")
        event = EventProgram(
            "event-001", "scene-001", "人物在压力下离开", "转身离开", ("person-001",), (),
            ("fact-001",), ("fact-001",), "门外传来催促", "离开",
        )
        beat = NarrativeBeat(
            "scene-001:beat-00", "scene-001", 0, "choice", "person-001", "离开", "门外传来催促",
            "人物收起手中物件并转身迈步", "让人物听见催促后注意到掌心的变化", "人物以行动落实离开",
            ("fact-001",), ("event-001",), "",
        )
        paragraph = ParagraphProgram(
            "scene-001:paragraph-00", "scene-001", 0, "decision", ("fact-001",), ("event-001",), "person-001", "", ("scene-001:beat-00",),
        )
        return SceneProgram(
            "scene-001", "chapter-001", 0, "做出离开决定", ("person-001",), (), (event,), (fact,),
            (paragraph,), ("fact-001",), (), (beat,),
        )

    def _draft_payload(self) -> dict[str, object]:
        return {
            "场景ID": "scene-001",
            "段落正文": [{
                "段落ID": "scene-001:paragraph-00", "正文": "他沉默许久，终于收起犹豫，转身向外走去，决定离开熟悉的地方。" * 3,
                "自检兑现事实ID": ["fact-001"], "自检兑现事件ID": ["event-001"],
            }],
        }

    def _validation_payload(self) -> dict[str, object]:
        return {
            "场景ID": "scene-001", "通过": True,
            "已实现事实ID": ["fact-001"], "缺失事实ID": [],
            "已实现事件ID": ["event-001"], "缺失事件ID": [],
            "提前泄露事件ID": [], "未授权断言段落ID": [], "段落问题": [], "修复段落ID": [],
        }

    def _paragraph_validation_payload(self) -> dict[str, object]:
        return {
            "段落ID": "scene-001:paragraph-00", "通过": True,
            "已实现事实ID": ["fact-001"], "缺失事实ID": [],
            "已实现事件ID": ["event-001"], "缺失事件ID": [],
            "未授权断言": [], "问题": [],
        }

    def test_strict_chinese_scene_payloads_round_trip_through_contracts(self) -> None:
        scene = self._scene()
        draft = scene_draft_from_llm(self._draft_payload(), scene, minimum_chars=20)
        self.assertEqual(draft.paragraphs[0].paragraph_id, "scene-001:paragraph-00")
        validation = scene_validation_from_llm(self._validation_payload(), scene)
        self.assertTrue(validation.passed)
        self.assertEqual(set(scene_draft_to_llm_dict(draft)), {"场景ID", "段落正文"})
        self.assertIn("提前泄露事件ID", scene_validation_to_llm_dict(validation))

    def test_batch_payload_hides_later_paragraphs_and_their_beats(self) -> None:
        scene = self._beat_scene()
        paragraphs = list(scene.paragraphs)
        beats = list(scene.narrative_beats)
        for index in range(1, 8):
            beat_id = f"scene-001:beat-{index:02d}"
            beats.append(replace(
                beats[0], beat_id=beat_id, order=index, beat_type="reaction", required_fact_ids=(), event_ids=(),
                expansion_license="inference", support_fact_ids=("fact-001",),
            ))
            paragraphs.append(ParagraphProgram(
                f"scene-001:paragraph-{index:02d}", "scene-001", index, "perception_reaction", (), (), "person-001", "", (beat_id,), 60, 42,
            ))
        expanded_scene = replace(scene, paragraphs=tuple(paragraphs), narrative_beats=tuple(beats))
        program = ChapterProgram("chapter-001", "hash", (expanded_scene,), "3.0", (EntityBinding("person-001", "人物", "person"),))
        payload = chapter_program_to_llm_dict(program)["场景程序"][0]
        batch = _batch_scene_payload(
            payload,
            ("scene-001:paragraph-00", "scene-001:paragraph-01"),
            include_entry_state=True,
            include_exit_state=False,
        )
        self.assertEqual([item["段落ID"] for item in batch["段落程序"]], ["scene-001:paragraph-00", "scene-001:paragraph-01"])
        self.assertEqual({item["节拍ID"] for item in batch["戏剧节拍"]}, {"scene-001:beat-00", "scene-001:beat-01"})
        self.assertEqual(batch["出场状态事实ID"], [])

    def test_writer_packet_exposes_deep_mechanism_and_gate_accounts_for_it(self) -> None:
        scene = self._beat_scene()
        beat = replace(
            scene.narrative_beats[0],
            mechanism_ids=("mechanism-001",),
            narrative_function="让外部催促真正改变人物的取舍，而不是只交代离开结果",
            counterfactual_guard="删去催促造成的压力后，离开会变成没有触发因素的动作",
        )
        scene = replace(scene, narrative_beats=(beat,))
        program = ChapterProgram(
            "chapter-001", "source-hash", (scene,), schema_version="2.1",
            entities=(EntityBinding("person-001", "人物甲", "person"),),
        )

        packet = paragraph_writer_packet(
            None, {}, program, scene, scene.paragraphs[0],
        )
        packet_beat = packet["当前段落节拍"][0]
        self.assertEqual(packet_beat["叙事机制ID"], ["mechanism-001"])
        self.assertIn("改变人物的取舍", packet_beat["叙事任务"])
        self.assertIn("没有触发因素", packet_beat["反事实守卫"])

        legacy = self._paragraph_validation_payload()
        with self.assertRaisesRegex(ValueError, "全部叙事机制"):
            paragraph_validation_from_llm(
                legacy, scene,
                paragraph_id=scene.paragraphs[0].paragraph_id,
                expected_fact_ids=("fact-001",),
                expected_event_ids=("event-001",),
                expected_mechanism_ids=("mechanism-001",),
            )
        expanded = {
            **legacy,
            "已实现叙事机制ID": ["mechanism-001"],
            "缺失叙事机制ID": [],
        }
        validation = paragraph_validation_from_llm(
            expanded, scene,
            paragraph_id=scene.paragraphs[0].paragraph_id,
            expected_fact_ids=("fact-001",),
            expected_event_ids=("event-001",),
            expected_mechanism_ids=("mechanism-001",),
        )
        self.assertTrue(validation.passed)

    def test_scene_validation_rejects_incomplete_fact_accounting(self) -> None:
        payload = self._validation_payload()
        payload["已实现事实ID"] = []
        with self.assertRaises(ValueError):
            scene_validation_from_llm(payload, self._scene())

    def test_scene_affordance_preflight_requires_each_uncovered_obligation_to_be_explained(self) -> None:
        scene = self._beat_scene()
        payload = {
            "场景ID": "scene-001", "通过": False,
            "不可执行事件ID": ["event-001"], "不可执行节拍ID": [],
            "问题": [{
                "对象类型": "事件", "对象ID": "event-001",
                "问题": "缺少行动者", "修复方向": "补充参与实体",
            }],
        }
        validation = scene_affordance_validation_from_llm(payload, scene)
        self.assertFalse(validation.passed)
        payload["问题"] = []
        with self.assertRaises(ValueError):
            scene_affordance_validation_from_llm(payload, scene)

    def test_scene_validation_requires_an_unsupported_claim_to_be_repaired(self) -> None:
        payload = self._validation_payload()
        payload["通过"] = False
        payload["未授权断言段落ID"] = ["scene-001:paragraph-00"]
        with self.assertRaises(ValueError):
            scene_validation_from_llm(payload, self._scene())

    def test_known_entity_from_another_scene_is_not_licensed_by_name_alone(self) -> None:
        scene = self._scene()
        draft = SceneDraft("scene-001", (
            ParagraphDraft("scene-001:paragraph-00", "人物甲提起后来的对手，并说自己决定离开。" * 3, ("fact-001",), ("event-001",)),
        ))
        issues = _unlicensed_entity_issues(scene, draft, {"person-001": "人物甲", "person-002": "后来的对手"})
        self.assertIn("scene-001:paragraph-00", issues)

    def test_prose_validation_requires_every_dramatic_beat_to_be_accounted_for(self) -> None:
        scene = self._beat_scene()
        payload = {
            "场景ID": "scene-001", "正文性通过": True,
            "已戏剧化节拍ID": ["scene-001:beat-00"], "缺失节拍ID": [],
            "摘要化段落ID": [], "段落问题": [], "修复段落ID": [],
        }
        validation = scene_prose_validation_from_llm(payload, scene)
        self.assertTrue(validation.passed)
        self.assertIn("已戏剧化节拍ID", scene_prose_validation_to_llm_dict(validation))
        payload["已戏剧化节拍ID"] = []
        with self.assertRaises(ValueError):
            scene_prose_validation_from_llm(payload, scene)

    def test_scene_blueprint_exposes_paragraph_and_sentence_targets(self) -> None:
        budget = {"metrics": [
            {"metric_id": "mean_paragraph_chars", "target": 60, "preferred_min": 50, "preferred_max": 70},
            {"metric_id": "mean_sentence_chars", "target": 40, "preferred_min": 30, "preferred_max": 50},
            {"metric_id": "short_paragraph_ratio", "target": 0.2, "preferred_min": 0.1, "preferred_max": 0.3},
        ]}
        blueprint = _scene_blueprint(budget, 2, 3)
        self.assertEqual(blueprint["段落字符范围"]["目标"], 60)
        self.assertEqual(blueprint["句子字符范围"]["目标"], 40)
        self.assertGreaterEqual(_paragraph_minimum(blueprint, 3), 20)
        program = ChapterProgram("chapter-001", "source-hash", (self._scene(),), entities=(EntityBinding("person-001", "人物甲", "person"),))
        paragraph_budgets = _paragraph_style_budgets(program, budget)
        self.assertIn("scene-001:paragraph-00", paragraph_budgets)

    def test_paragraph_budgets_assign_style_roles_only_to_compatible_jobs(self) -> None:
        first_fact = FactContract("fact-001", "scene-001", "goal", "person-001", "确认", "离开")
        first_event = EventProgram(
            "event-001", "scene-001", "两人确认去向", "向同伴确认离开", ("person-001", "person-002"), (),
            ("fact-001",), ("fact-001",), "同伴质疑", "离开",
        )
        first_paragraph = ParagraphProgram(
            "scene-001:paragraph-00", "scene-001", 0, "dialogue_conflict", ("fact-001",),
            ("event-001",), "person-001", "确认去向",
        )
        second_fact = FactContract("fact-002", "scene-002", "emotion", "person-001", "心绪", "警觉", expression_mode="internal")
        second_event = EventProgram(
            "event-002", "scene-002", "人物因确认而警觉", "带着确认后的压力继续前行", ("person-001",), (),
            ("fact-002",), ("fact-002",), "前路未知", "继续前行",
        )
        second_paragraph = ParagraphProgram(
            "scene-002:paragraph-00", "scene-002", 0, "perception_reaction", ("fact-002",),
            ("event-002",), "person-001",
        )
        first_scene = SceneProgram(
            "scene-001", "chapter-001", 0, "确认去向", ("person-001", "person-002"), (),
            (first_event,), (first_fact,), (first_paragraph,), ("fact-001",), (),
        )
        second_scene = SceneProgram(
            "scene-002", "chapter-001", 1, "带着压力前行", ("person-001",), ("fact-001",),
            (second_event,), (second_fact,), (second_paragraph,), ("fact-002",), (),
        )
        program = ChapterProgram(
            "chapter-001", "source-hash", (first_scene, second_scene),
            entities=(EntityBinding("person-001", "人物甲", "person"), EntityBinding("person-002", "人物乙", "person")),
        )
        program.validate()
        budget = {"metrics": [
            {"metric_id": "mean_paragraph_chars", "target": 60, "preferred_min": 50, "preferred_max": 70},
            {"metric_id": "mean_sentence_chars", "target": 40, "preferred_min": 30, "preferred_max": 50},
            {"metric_id": "dialogue_char_ratio", "target": 0.4},
            {"metric_id": "transition_marker_ratio", "target": 0.1},
            {"metric_id": "perception_marker_ratio", "target": 0.1},
        ]}
        paragraph_budgets = _paragraph_style_budgets(program, budget)
        self.assertIn("对话", paragraph_budgets["scene-001:paragraph-00"]["职责标签"])
        self.assertNotIn("对话", paragraph_budgets["scene-002:paragraph-00"]["职责标签"])
        self.assertIn("转场", paragraph_budgets["scene-002:paragraph-00"]["职责标签"])
        self.assertIn("感知", paragraph_budgets["scene-002:paragraph-00"]["职责标签"])
        self.assertTrue(paragraph_budgets["scene-001:paragraph-00"]["叙事职责"])
        missing_roles = SceneDraft("scene-002", (
            ParagraphDraft("scene-002:paragraph-00", "人物继续前行，心绪仍未平静。"),
        ))
        # 转场和感知现在是语义职责；词汇标记只影响风格统计，不能作为
        # “看到/随后”式机械写法的硬性验收门槛。
        self.assertEqual(
            _paragraph_role_issues(missing_roles, {"scene-002:paragraph-00": paragraph_budgets["scene-002:paragraph-00"]}),
            {},
        )
        dialogue_draft = SceneDraft("scene-001", (
            ParagraphDraft(
                "scene-001:paragraph-00",
                "人物甲望向同伴，说道：“我已想清楚去向，此行纵然艰险，也要亲自走上一遭；你若担心，便把该说的话都说清，我不会再退。”",
            ),
        ))
        self.assertEqual(
            _paragraph_role_issues(dialogue_draft, {"scene-001:paragraph-00": paragraph_budgets["scene-001:paragraph-00"]}),
            {},
        )

    def test_style_repair_must_not_trade_measured_regression_for_semantic_safety(self) -> None:
        current = {"passed": False, "high_priority_in_range": 3, "mean_relative_deviation": 0.53}
        worse = {"passed": False, "high_priority_in_range": 0, "mean_relative_deviation": 0.62}
        better = {"passed": False, "high_priority_in_range": 3, "mean_relative_deviation": 0.45}
        self.assertFalse(_prefer_style_assessment(current, worse))
        self.assertTrue(_prefer_style_assessment(current, better))

    def test_style_repair_removes_extra_marker_from_non_role_paragraph(self) -> None:
        scene = self._scene()
        draft = SceneDraft("scene-001", (
            ParagraphDraft("scene-001:paragraph-00", "人物看到远处变化后，仍按既定决定离开。"),
        ))
        assessment = {"metrics": [{
            "metric_id": "perception_marker_ratio", "actual": 0.2,
            "preferred_min": 0.05, "preferred_max": 0.1,
            "in_preferred_range": False,
        }]}
        targets = _style_targets(
            scene, draft, assessment,
            {"scene-001:paragraph-00": {"职责标签": []}},
        )
        self.assertEqual(targets, ("scene-001:paragraph-00",))
        replacement = _replace_non_role_marker(
            draft, {"scene-001:paragraph-00": {"职责标签": []}}, "感知",
        )
        self.assertIsNotNone(replacement)
        repaired, record = replacement
        self.assertEqual(record["from"], "看到")
        self.assertNotIn("看到", repaired.paragraphs[0].text)

    def test_program_generation_runs_scene_by_scene_with_contract_outputs(self) -> None:
        scene = self._scene()
        program = ChapterProgram(
            "chapter-001", "source-hash", (scene,),
            entities=(EntityBinding("person-001", "人物甲", "person"),),
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            program_path = base / "program.json"
            write_json(program_path, program.to_dict())
            rendered_inputs: list[dict[str, object]] = []

            def capture_render(_template: Path, values: dict[str, object]) -> str:
                rendered_inputs.append(values)
                return "中文提示词"

            with patch("pipeline.scene_generation._skill_files", return_value=(base, {"author_id": "Example"}, {"templates": [{"template_id": "library:scene-001", "level": "scene"}]}, {}, {"metrics": []})), \
                 patch("pipeline.scene_generation._render", side_effect=capture_render), \
                 patch("pipeline.scene_generation.runs_dir", return_value=base / "runs"), \
                 patch("pipeline.scene_generation.ModelSettings.from_environment", return_value=ModelSettings(api_key="test", max_tokens=3000)), \
                 patch("pipeline.scene_generation.complete_json", side_effect=[self._draft_payload(), self._paragraph_validation_payload(), self._validation_payload()]) as complete, \
                 patch("pipeline.scene_generation.assess_style_budget", return_value={"passed": True}):
                result = generate_from_program("Example", program_path, run_id="scene-test")
            self.assertTrue(result["semantic_passed"])
            self.assertTrue(result["style_passed"])
            self.assertEqual(complete.call_count, 3)
            output = Path(str(result["output_dir"]))
            self.assertTrue((output / "generated_draft.txt").exists())
            self.assertTrue((output / "scene-01.result.json").exists())
            self.assertTrue(result["graph_patch_committed"])
            self.assertTrue((output / "chapter_graph_patch.candidate.json").exists())
            self.assertTrue((output / "chapter_graph_patch.json").exists())
            self.assertTrue((output / "runtime_graph_state.json").exists())
            writer_inputs = next(item for item in rendered_inputs if "AUTHOR_STYLE_PROFILE_JSON" in item)
            self.assertNotIn("STORY_STATE_JSON", writer_inputs)
            packet = writer_inputs["PARAGRAPH_WRITER_PACKET_JSON"]
            self.assertEqual(packet["当前段落程序"]["段落ID"], "scene-001:paragraph-00")
            self.assertIn("叙事事实子图", packet)
            self.assertNotIn("evidence", json.dumps(packet, ensure_ascii=False))
            self.assertNotIn("quote", json.dumps(packet, ensure_ascii=False))

    def test_controlled_expansion_refuses_a_non_graph_fallback(self) -> None:
        program = ChapterProgram(
            "chapter-001", "source-hash", (self._scene(),), generation_mode="controlled_expansion",
            target_char_min=4000, target_char_max=5000,
            entities=(EntityBinding("person-001", "人物甲", "person"),),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "program.json"
            write_json(path, program.to_dict())
            with self.assertRaisesRegex(ValueError, "narrative_graph"):
                generate_from_program("Example", path)

    def test_program_generation_requires_a_prose_audit_for_beat_programs(self) -> None:
        scene = self._beat_scene()
        program = ChapterProgram(
            "chapter-001", "source-hash", (scene,), schema_version="2.0",
            entities=(EntityBinding("person-001", "人物甲", "person"),),
        )
        prose_payload = {
            "场景ID": "scene-001", "正文性通过": True,
            "已戏剧化节拍ID": ["scene-001:beat-00"], "缺失节拍ID": [],
            "摘要化段落ID": [], "段落问题": [], "修复段落ID": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            program_path = base / "program.json"
            write_json(program_path, program.to_dict())
            with patch("pipeline.scene_generation._skill_files", return_value=(base, {"author_id": "Example"}, {"templates": [{"template_id": "library:scene-001", "level": "scene"}]}, {}, {"metrics": []})), \
                 patch("pipeline.scene_generation._render", return_value="中文提示词"), \
                 patch("pipeline.scene_generation.runs_dir", return_value=base / "runs"), \
                 patch("pipeline.scene_generation.ModelSettings.from_environment", return_value=ModelSettings(api_key="test", max_tokens=3000)), \
                 patch("pipeline.scene_generation.complete_json", side_effect=[self._draft_payload(), self._paragraph_validation_payload(), self._validation_payload(), prose_payload]) as complete, \
                 patch("pipeline.scene_generation.assess_style_budget", return_value={"passed": True}):
                result = generate_from_program("Example", program_path, run_id="prose-test")
            self.assertTrue(result["structural_passed"])
            self.assertTrue(result["prose_passed"])
            self.assertTrue(result["overall_passed"])
            self.assertEqual(complete.call_count, 4)

    def test_failed_prose_audit_keeps_only_candidate_draft(self) -> None:
        scene = self._beat_scene()
        program = ChapterProgram(
            "chapter-001", "source-hash", (scene,), schema_version="2.0",
            entities=(EntityBinding("person-001", "人物甲", "person"),),
        )
        failed_prose_payload = {
            "场景ID": "scene-001", "正文性通过": False,
            "已戏剧化节拍ID": [], "缺失节拍ID": ["scene-001:beat-00"],
            "摘要化段落ID": ["scene-001:paragraph-00"],
            "段落问题": [{
                "段落ID": "scene-001:paragraph-00",
                "问题": "正文只概述结果，未演出节拍过程",
                "修复要求": "补出触发、动作和反应",
            }],
            "修复段落ID": ["scene-001:paragraph-00"],
        }
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            program_path = base / "program.json"
            write_json(program_path, program.to_dict())
            with patch("pipeline.scene_generation._skill_files", return_value=(base, {"author_id": "Example"}, {"templates": [{"template_id": "library:scene-001", "level": "scene"}]}, {}, {"metrics": []})), \
                 patch("pipeline.scene_generation._render", return_value="中文提示词"), \
                 patch("pipeline.scene_generation.runs_dir", return_value=base / "runs"), \
                 patch("pipeline.scene_generation.ModelSettings.from_environment", return_value=ModelSettings(api_key="test", max_tokens=3000)), \
                 patch("pipeline.scene_generation.complete_json", side_effect=[self._draft_payload(), self._paragraph_validation_payload(), self._validation_payload(), failed_prose_payload]), \
                 patch("pipeline.scene_generation.assess_style_budget", return_value={"passed": True}):
                result = generate_from_program("Example", program_path, run_id="rejected-prose-test", max_repairs=0)
            output = Path(str(result["output_dir"]))
            self.assertFalse(result["overall_passed"])
            self.assertTrue((output / "candidate_draft.txt").exists())
            self.assertFalse((output / "generated_draft.txt").exists())
            self.assertFalse((output / "chapter_graph_patch.json").exists())
            self.assertEqual(read_json(output / "generation_status.json")["status"], "rejected")

    def test_program_generation_repairs_a_missing_paragraph_role_before_accepting_scene(self) -> None:
        fact = FactContract("fact-001", "scene-001", "goal", "person-001", "确认", "前往")
        event = EventProgram(
            "event-001", "scene-001", "两人确认去向", "向同伴确认前往", ("person-001", "person-002"), (),
            ("fact-001",), ("fact-001",), "同伴质疑", "前往",
            action_type="speech", actor_id="person-001", target_ids=("person-002",), basis_fact_ids=("fact-001",),
        )
        paragraph = ParagraphProgram(
            "scene-001:paragraph-00", "scene-001", 0, "action_progression", ("fact-001",),
            ("event-001",), "person-001",
        )
        scene = SceneProgram(
            "scene-001", "chapter-001", 0, "确认去向", ("person-001", "person-002"), (),
            (event,), (fact,), (paragraph,), ("fact-001",), (),
        )
        program = ChapterProgram(
            "chapter-001", "source-hash", (scene,),
            entities=(EntityBinding("person-001", "人物甲", "person"), EntityBinding("person-002", "人物乙", "person")),
        )
        draft = {
            "场景ID": "scene-001",
            "段落正文": [{
                "段落ID": "scene-001:paragraph-00", "正文": "人物甲向同伴确认了前往的决定，两人随后准备动身。" * 2,
                "自检兑现事实ID": ["fact-001"], "自检兑现事件ID": ["event-001"],
            }],
        }
        validation = {
            "场景ID": "scene-001", "通过": True,
            "已实现事实ID": ["fact-001"], "缺失事实ID": [],
            "已实现事件ID": ["event-001"], "缺失事件ID": [],
            "提前泄露事件ID": [], "未授权断言段落ID": [], "段落问题": [], "修复段落ID": [],
        }
        repair = {
            "场景ID": "scene-001",
            "修复段落": [{
                "段落ID": "scene-001:paragraph-00",
                "正文": "人物甲望着同伴，说道：“我已想清楚去向，此行虽有阻力，也该由我亲自走完，你不必再替我迟疑。”两人随即准备动身。",
                "自检兑现事实ID": ["fact-001"], "自检兑现事件ID": ["event-001"],
            }],
        }
        paragraph_repair = {"场景ID": repair["场景ID"], "段落正文": repair["修复段落"]}
        execution_spec = {"metrics": [
            {"metric_id": "dialogue_char_ratio", "target": 0.4, "preferred_min": 0.2, "preferred_max": 0.6, "priority": "high"},
        ]}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            program_path = base / "program.json"
            write_json(program_path, program.to_dict())
            with patch("pipeline.scene_generation._skill_files", return_value=(base, {"author_id": "Example"}, {"templates": [{"template_id": "library:scene-001", "level": "scene"}]}, {}, execution_spec)), \
                 patch("pipeline.scene_generation._render", return_value="中文提示词"), \
                 patch("pipeline.scene_generation.runs_dir", return_value=base / "runs"), \
                 patch("pipeline.scene_generation.ModelSettings.from_environment", return_value=ModelSettings(api_key="test", max_tokens=3000)), \
                 patch("pipeline.scene_generation.complete_json", side_effect=[draft, self._paragraph_validation_payload(), paragraph_repair, self._paragraph_validation_payload(), validation]) as complete, \
                 patch("pipeline.scene_generation.assess_style_budget", return_value={"passed": True}):
                result = generate_from_program("Example", program_path, run_id="role-test", max_repairs=1)
            self.assertTrue(result["semantic_passed"])
            self.assertTrue(result["overall_passed"])
            self.assertEqual(complete.call_count, 5)
            scene_result = read_json(Path(str(result["output_dir"])) / "scene-01.result.json")
            self.assertTrue(scene_result["role_passed"])
            self.assertEqual(scene_result["repairs_used"], 0)
