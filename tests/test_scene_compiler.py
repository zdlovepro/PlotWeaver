from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace

from pipeline.contracts import ChapterAnnotation, Entity, EventAtom, Fact, SceneCard, SourceSpan, StateChange, build_chapter_document
from pipeline.contracts.program import ChapterProgram
from pipeline.contracts.program_llm import chapter_program_from_llm_dict, chapter_program_to_llm_dict
from pipeline.chapter_planner import assess_length_plan, iter_generation_batches, plan_controlled_expansion
from pipeline.scene_compiler import _require_expansion_quality, assess_chapter_program, compile_chapter_program


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
        self.assertEqual(program.scene_programs[0].forbidden_event_ids, ("event-002",))
        self.assertEqual(program.scene_programs[1].entry_state_fact_ids, ("fact-002",))

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
        self.assertEqual(owners["fact-late"], "scene-002")
