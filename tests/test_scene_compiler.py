from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace

from pipeline.contracts import ChapterAnnotation, Entity, EventAtom, Fact, NarrativeMechanism, SceneCard, SourceSpan, StateChange, build_chapter_document
from pipeline.contracts.program import ChapterProgram
from pipeline.contracts.program_llm import chapter_program_from_llm_dict, chapter_program_to_llm_dict
from pipeline.chapter_planner import assess_length_plan, iter_generation_batches, plan_controlled_expansion
from pipeline.scene_compiler import _require_expansion_quality, assess_chapter_program, compile_chapter_program
from pipeline.skill_compiler import _scene_affordance_validate_prompt, _scene_draft_prompt


class SceneCompilerTests(unittest.TestCase):
    def _annotation(self) -> ChapterAnnotation:
        chapter_id = "Author/work/0001"
        text = "沈舟到了山门。\n他得到令牌，决定入门。"
        document = build_chapter_document(chapter_id, hashlib.sha256(text.encode("utf-8")).hexdigest(), text, "第一章")
        first, second = document.units
        person_span = SourceSpan(chapter_id, 0, 2, "沈舟", first.unit_id)
        location_span = SourceSpan(chapter_id, 0, 5, "沈舟到了山门", first.unit_id)
        item_start = text.index("令牌")
        item_span = SourceSpan(chapter_id, item_start, item_start + 2, "令牌", second.unit_id)
        decision_start = text.index("决定入门")
        decision_span = SourceSpan(chapter_id, decision_start, decision_start + 4, "决定入门", second.unit_id)
        full_span = SourceSpan(chapter_id, 0, len(text), text, "")
        person = Entity("person-001", "沈舟", "person", (), (person_span,))
        location = Entity("location-001", "山门", "location", (), (location_span,))
        token = Entity("item-001", "令牌", "item", (), (item_span,))
        arrival = Fact("fact-001", chapter_id, "location", "person-001", "位于", "", "location-001", 1.0, (location_span,))
        gain = Fact("fact-002", chapter_id, "resource", "person-001", "获得", "", "item-001", 1.0, (item_span,))
        decision = Fact("fact-003", chapter_id, "goal", "person-001", "决定", "入门", "", 1.0, (decision_span,))
        event_one = EventAtom(
            "event-001", chapter_id, 0, "抵达山门并获得令牌", ("person-001",), ("fact-001",), (),
            "抵达并取得令牌", "", "", ("fact-002",), (), (full_span,),
        )
        event_two = EventAtom(
            "event-002", chapter_id, 1, "持令牌决定入门", ("person-001",), (), ("fact-002",),
            "做出入门决定", "", "入门", ("fact-003",), (), (full_span,),
        )
        scene_one = SceneCard(
            "scene-001", chapter_id, 0, ("person-001",), ("event-001",), "取得进入资格",
            ("fact-001",), ("fact-002",), 2, (full_span,), ("location-001",), (),
        )
        scene_two = SceneCard(
            "scene-002", chapter_id, 1, ("person-001",), ("event-002",), "决定入门",
            ("fact-002",), ("fact-003",), 3, (full_span,), ("location-001",), (),
        )
        change = StateChange("change-001", chapter_id, "event-001", "add", "", "fact-002")
        return ChapterAnnotation(
            chapter_id, document.source_hash, (person, location, token), (arrival, gain, decision),
            (event_one, event_two), (scene_one, scene_two), (change,),
        )

    def test_compiler_binds_every_fact_and_event_to_a_scene_paragraph(self) -> None:
        program = compile_chapter_program(self._annotation())
        program.validate()
        quality = assess_chapter_program(program)
        self.assertTrue(quality["passed"], quality)
        self.assertEqual(quality["counts"]["fact_contract_count"], 3)
        self.assertEqual(quality["counts"]["paragraph_bound_fact_count"], 3)
        self.assertEqual(quality["counts"]["event_program_count"], 2)
        self.assertEqual(quality["counts"]["paragraph_bound_event_count"], 2)
        self.assertEqual(len(program.scene_programs), 1)
        self.assertEqual(
            tuple(event.event_id for event in program.scene_programs[0].event_programs),
            ("event-001", "event-002"),
        )
        self.assertEqual(program.scene_programs[0].forbidden_event_ids, ())

    def test_compiler_preserves_explicit_multi_event_source_scenes(self) -> None:
        annotation = self._annotation()
        manually_grouped = replace(
            annotation.scenes[0],
            event_ids=("event-001", "event-002"),
            exit_fact_ids=("fact-003",),
        )
        program = compile_chapter_program(replace(annotation, scenes=(manually_grouped,)))
        self.assertEqual(program.scene_programs[0].scene_id, "scene-001")
        self.assertEqual(len(program.scene_programs[0].event_programs), 2)

    def test_compiler_expands_events_into_writeable_dramatic_beats(self) -> None:
        program = compile_chapter_program(self._annotation())
        self.assertEqual(program.schema_version, "2.0")
        first_scene = program.scene_programs[0]
        self.assertTrue(first_scene.narrative_beats)
        self.assertEqual(
            tuple(beat.order for beat in first_scene.narrative_beats),
            tuple(range(len(first_scene.narrative_beats))),
        )
        self.assertIn("action", {beat.beat_type for beat in first_scene.narrative_beats})
        self.assertIn("reaction", {beat.beat_type for beat in first_scene.narrative_beats})
        self.assertTrue(all(paragraph.beat_ids for paragraph in first_scene.paragraphs))
        self.assertTrue(all(len(paragraph.required_fact_ids) <= 2 for paragraph in first_scene.paragraphs))
        self.assertLess(len(first_scene.paragraphs), len(first_scene.narrative_beats))
        first_action = next(beat for beat in first_scene.narrative_beats if beat.beat_type == "action")
        self.assertIn("event-001", first_action.event_ids)
        self.assertIn("fact-001", first_action.required_fact_ids)

    def test_compiler_binds_every_deep_mechanism_to_a_writeable_beat(self) -> None:
        annotation = self._annotation()
        evidence = (annotation.facts[0].evidence[0], annotation.facts[1].evidence[0])
        mechanism = NarrativeMechanism(
            "mechanism-001", annotation.chapter_id, 0,
            "character_drive", "goal",
            "抵达后的资格变化把入门意愿转成下一步选择",
            "让人物先感受到资格变化，再以选择回应，而不是直接宣布结果",
            "删去资格变化与选择之间的连接后，入门决定会缺少现场触发",
            ("fact-001", "fact-002"), ("event-001",), ("person-001",),
            "structural", "not_applicable", 0.9, evidence,
        )
        annotation = replace(
            annotation,
            narrative_mechanisms=(mechanism,),
            schema_version="2.3",
        )
        program = compile_chapter_program(annotation)
        beats = [beat for scene in program.scene_programs for beat in scene.narrative_beats]
        bound = [beat for beat in beats if "mechanism-001" in beat.mechanism_ids]
        self.assertEqual(program.schema_version, "2.1")
        self.assertEqual(len(bound), 1)
        self.assertIn("选择回应", bound[0].narrative_function)
        self.assertIn("缺少现场触发", bound[0].counterfactual_guard)

    def test_multi_fact_event_response_is_split_without_losing_causality(self) -> None:
        annotation = self._annotation()
        span = annotation.events[0].evidence[0]
        extra_facts = tuple(
            Fact(
                f"fact-extra-{index}", annotation.chapter_id, "information", "person-001",
                f"已知条件{index}", f"内容{index}", "", 1.0, (span,),
            )
            for index in range(1, 6)
        )
        program = compile_chapter_program(replace(annotation, facts=(*annotation.facts, *extra_facts)))
        quality = assess_chapter_program(program)

        self.assertTrue(quality["passed"], quality)
        self.assertLessEqual(quality["counts"]["max_facts_per_paragraph"], 2)
        bound = {
            fact_id
            for scene in program.scene_programs
            for paragraph in scene.paragraphs
            for fact_id in paragraph.required_fact_ids
        }
        self.assertTrue({fact.fact_id for fact in extra_facts}.issubset(bound))

    def test_single_paragraph_prompt_forbids_orientation_from_staging_a_future_event(self) -> None:
        prompt = _scene_draft_prompt()
        self.assertIn("场景定位", prompt)
        self.assertIn("不得提前书写进入、返回、离开", prompt)
        self.assertIn("事实不是可以直接贴上的心理标签", prompt)
        self.assertIn("`叙事任务` 不是供参考的摘要", prompt)
        self.assertIn("`反事实守卫`", prompt)

    def test_affordance_prompt_validates_the_generic_event_frame(self) -> None:
        prompt = _scene_affordance_validate_prompt()
        self.assertIn("`行动类型`", prompt)
        self.assertIn("`actor_id`", prompt)
        self.assertIn("`target_ids`", prompt)
        self.assertIn("`basis_fact_ids`", prompt)
        self.assertIn("不得从“动作”摘要中搜索个别动词", prompt)
        self.assertIn("叙述性交代", prompt)

    def test_chinese_model_contract_round_trips_without_english_field_names(self) -> None:
        program = compile_chapter_program(self._annotation())
        model_payload = chapter_program_to_llm_dict(program)
        self.assertEqual(set(model_payload), {"契约版本", "章节ID", "源文本哈希", "生成模式", "章节目标字数下限", "章节目标字数上限", "实体字典", "场景程序"})
        restored = chapter_program_from_llm_dict(model_payload)
        self.assertEqual(restored.to_dict(), program.to_dict())
        model_payload["额外字段"] = True
        with self.assertRaises(ValueError):
            chapter_program_from_llm_dict(model_payload)

    def test_legacy_chinese_contract_remains_strictly_parseable(self) -> None:
        program = compile_chapter_program(self._annotation())
        payload = chapter_program_to_llm_dict(program)
        for key in ("生成模式", "章节目标字数下限", "章节目标字数上限"):
            payload.pop(key)
        for scene in payload["场景程序"]:
            for paragraph in scene["段落程序"]:
                paragraph.pop("目标字数")
                paragraph.pop("最低字数")
            for beat in scene["戏剧节拍"]:
                beat.pop("扩写许可")
                beat.pop("依据事实ID")
        restored = chapter_program_from_llm_dict(payload)
        self.assertEqual(restored.to_dict(), program.to_dict())
        payload["未知字段"] = True
        with self.assertRaises(ValueError):
            chapter_program_from_llm_dict(payload)

    def test_controlled_expansion_has_bounded_licensed_paragraph_plan(self) -> None:
        source_program = compile_chapter_program(self._annotation())
        program = plan_controlled_expansion(source_program, target_char_min=4_000, target_char_max=5_000)
        self.assertEqual(program.schema_version, "3.0")
        self.assertEqual(program.generation_mode, "controlled_expansion")
        quality = assess_length_plan(program)
        self.assertTrue(quality["passed"], quality)
        self.assertGreaterEqual(quality["planned_char_count"], 4_000)
        self.assertLessEqual(quality["planned_char_count"], 5_000)
        extension_beats = [
            beat for scene in program.scene_programs for beat in scene.narrative_beats
            if beat.expansion_license != "source"
        ]
        self.assertTrue(extension_beats)
        self.assertTrue(all(beat.support_fact_ids for beat in extension_beats))
        self.assertGreaterEqual(
            min(paragraph.target_chars for scene in program.scene_programs for paragraph in scene.paragraphs),
            90,
        )
        batches = iter_generation_batches(program, max_paragraphs=6)
        self.assertTrue(batches)
        self.assertTrue(all(1 <= len(paragraph_ids) <= 6 for _, paragraph_ids in batches))

    def test_controlled_expansion_rejects_a_coarse_long_chapter_annotation(self) -> None:
        annotation = self._annotation()
        text = "\n".join("沈舟观察山门的变化。" for _ in range(400))
        document = build_chapter_document(annotation.chapter_id, hashlib.sha256(text.encode("utf-8")).hexdigest(), text, "第一章")
        with self.assertRaises(ValueError):
            _require_expansion_quality(document, annotation)

    def test_duplicate_fact_ownership_fails_the_contract(self) -> None:
        program = compile_chapter_program(self._annotation())
        duplicate = ChapterProgram(program.chapter_id, program.source_hash, (program.scene_programs[0], program.scene_programs[0]))
        with self.assertRaises(ValueError):
            duplicate.validate()

    def test_unbound_fact_uses_evidence_position_not_first_matching_participant(self) -> None:
        annotation = self._annotation()
        chapter_id = annotation.chapter_id
        early = SourceSpan(chapter_id, 0, 10, "early-event", "")
        late = SourceSpan(chapter_id, 100, 110, "late-event", "")
        events = (
            replace(annotation.events[0], evidence=(early,)),
            replace(annotation.events[1], evidence=(late,)),
        )
        scenes = (
            replace(annotation.scenes[0], evidence=(early,)),
            replace(annotation.scenes[1], evidence=(late,)),
        )
        late_fact = Fact(
            "fact-late", chapter_id, "information", "person-001", "得知", "后续消息", "", 1.0, (late,),
        )
        program = compile_chapter_program(replace(annotation, facts=(*annotation.facts, late_fact), events=events, scenes=scenes))
        owners = {
            fact.fact_id: scene.scene_id
            for scene in program.scene_programs
            for fact in scene.fact_contracts
        }
        self.assertEqual(owners["fact-late"], program.scene_programs[0].scene_id)
