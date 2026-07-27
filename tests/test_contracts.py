from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

from pipeline.contracts import (
    AuthorProfile,
    ChapterAnnotation,
    build_chapter_document,
    ChapterDocument,
    ChapterStyleCard,
    ContractValidationError,
    Entity,
    EventAtom,
    Fact,
    SceneCard,
    SourceSpan,
    StateChange,
    StyleMetric,
    StylePatternObservation,
    Template,
    PromptProgram,
    build_paragraph_units,
    validate_annotation,
    validate_document,
)
from pipeline.ingest import CHAPTER_RE, parse_chapters
from pipeline.annotate import AnnotationBuildError, annotate_document, annotation_from_draft, assess_annotation_quality, enrich_entity_evidence, sanitize_draft, split_annotation_units
from pipeline.spatiotemporal import enrich_spatiotemporal


class ContractTests(unittest.TestCase):
    def _annotation_fixture(self) -> tuple[ChapterDocument, ChapterAnnotation, ChapterStyleCard]:
        chapter_id = "Author/work/0001"
        text = "王林到村口。\n王林得到令牌。"
        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        document = ChapterDocument(chapter_id, source_hash, text, build_paragraph_units(chapter_id, text))
        person_span = SourceSpan(chapter_id, 0, 2, "王林", document.units[0].unit_id)
        location_span = SourceSpan(chapter_id, 0, 5, "王林到村口", document.units[0].unit_id)
        item_start = text.index("令牌")
        item_span = SourceSpan(chapter_id, item_start, item_start + 2, "令牌", document.units[1].unit_id)
        event_span = SourceSpan(chapter_id, 0, len(text), text, "")
        person = Entity("person-001", "王林", "person", ("铁柱",), (person_span,))
        location = Entity("location-001", "村口", "location", (), (location_span,))
        item = Entity("item-001", "令牌", "item", (), (item_span,))
        arrival = Fact("fact-001", chapter_id, "location", "person-001", "位于", "", "location-001", 1.0, (location_span,))
        gain = Fact("fact-002", chapter_id, "resource", "person-001", "获得", "", "item-001", 1.0, (item_span,))
        event = EventAtom(
            "event-001", chapter_id, 0, "人物到达村口并获得令牌", ("person-001",),
            ("fact-001",), (), "到达并取得物品", "", "", ("fact-002",), (), (event_span,),
        )
        scene = SceneCard(
            "scene-001", chapter_id, 0, ("person-001",), ("event-001",), "获得令牌",
            ("fact-001",), ("fact-002",), 3, (event_span,),
        )
        change = StateChange("change-001", chapter_id, "event-001", "add", "", "fact-002")
        annotation = ChapterAnnotation(chapter_id, source_hash, (person, location, item), (arrival, gain), (event,), (scene,), (change,))
        style = ChapterStyleCard(
            chapter_id,
            source_hash,
            (StyleMetric("paragraph_count", 2, "count"), StyleMetric("dialogue_ratio", 0, "ratio")),
            (StylePatternObservation("style-001", "paragraph", "transition", "动作后立即给出结果", 1, (event_span,)),),
        )
        return document, annotation, style

    def test_valid_annotation_round_trips_and_passes(self) -> None:
        document, annotation, style = self._annotation_fixture()
        self.assertTrue(validate_document(document).passed)
        report = validate_annotation(annotation, document, style)
        self.assertTrue(report.passed, report.to_dict())
        restored_document = ChapterDocument.from_dict(document.to_dict())
        restored_annotation = ChapterAnnotation.from_dict(annotation.to_dict())
        restored_style = ChapterStyleCard.from_dict(style.to_dict())
        self.assertEqual(restored_document.to_dict(), document.to_dict())
        self.assertEqual(restored_annotation.to_dict(), annotation.to_dict())
        self.assertEqual(restored_style.to_dict(), style.to_dict())
        self.assertTrue(validate_annotation(restored_annotation, restored_document, restored_style).passed)

    def test_annotation_chunk_checkpoint_reuses_a_valid_finished_fragment(self) -> None:
        document, annotation, _ = self._annotation_fixture()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "checkpoints"
            with patch("pipeline.annotate._annotate_chunk", return_value=annotation) as extract:
                first, _ = annotate_document(document, settings=object(), checkpoint_dir=checkpoint_dir)
            self.assertEqual(extract.call_count, 1)
            with patch("pipeline.annotate._annotate_chunk", side_effect=AssertionError("checkpoint was not reused")):
                second, _ = annotate_document(document, settings=object(), checkpoint_dir=checkpoint_dir)
            self.assertEqual(first.to_dict(), second.to_dict())

    def test_source_units_mark_boilerplate_and_preserve_quote_continuity(self) -> None:
        chapter_id = "Author/work/0001"
        text = "“这句话没有说完\n仍在同一句话里。”\n——\n求月票支持\n再加更一章\n（本章完）"
        units = build_paragraph_units(chapter_id, text)
        document = ChapterDocument(chapter_id, hashlib.sha256(text.encode("utf-8")).hexdigest(), text, units)
        self.assertTrue(validate_document(document).passed)
        self.assertEqual((units[0].quote_depth_before, units[0].quote_depth_after), (0, 1))
        self.assertEqual((units[1].quote_depth_before, units[1].quote_depth_after), (1, 0))
        self.assertEqual(units[2].kind, "separator")
        self.assertEqual(units[3].kind, "editorial_note")
        self.assertEqual(units[4].kind, "editorial_note")
        self.assertEqual(units[5].kind, "boilerplate")
        self.assertFalse(units[3].is_annotation_eligible)
        self.assertFalse(units[4].is_annotation_eligible)

    def test_document_marks_editorial_notice_and_publisher_tail(self) -> None:
        notice = build_chapter_document(
            "Author/work/0001",
            hashlib.sha256("任意笔名求月票\n（本章完）".encode("utf-8")).hexdigest(),
            "任意笔名求月票\n（本章完）",
            "有话说",
        )
        self.assertEqual(notice.content_kind, "editorial_notice")
        self.assertFalse(notice.is_narrative)
        publisher = build_paragraph_units("Author/work/0002", "正文。\n更多精彩小说，请访问：http://example.com\n（本章完）")
        self.assertEqual(publisher[1].kind, "publisher_note")
        self.assertFalse(publisher[1].is_annotation_eligible)

    def test_editorial_labelling_uses_publication_signals_not_a_writer_identity(self) -> None:
        for label in ("甲", "任意笔名"):
            text = f"{label}求月票支持\n（本章完）"
            document = build_chapter_document(
                f"Author/work/{label}",
                hashlib.sha256(text.encode("utf-8")).hexdigest(),
                text,
                "有话说",
            )
            self.assertEqual(document.units[0].kind, "editorial_note")
            self.assertEqual(document.content_kind, "editorial_notice")
            self.assertFalse(document.is_narrative)

        dialogue = build_paragraph_units("Author/work/dialogue", "角色说：‘感谢大家支持。’")
        self.assertEqual(dialogue[0].kind, "paragraph")

    def test_annotation_draft_resolves_exact_unit_evidence_and_quality(self) -> None:
        chapter_id = "Author/work/0003"
        text = "A reaches gate.\nA gains token."
        document = build_chapter_document(chapter_id, hashlib.sha256(text.encode("utf-8")).hexdigest(), text, "Start")
        first, second = document.units
        draft = {
            "entities": [
                {"id": "e1", "name": "A", "kind": "person", "aliases": [], "evidence": [{"unit_id": first.unit_id, "quote": "A"}]},
                {"id": "e2", "name": "gate", "kind": "location", "aliases": [], "evidence": [{"unit_id": first.unit_id, "quote": "gate"}]},
            ],
            "facts": [
                {"id": "f1", "kind": "location", "subject_id": "e1", "predicate": "reaches", "value": "gate", "object_id": "e2", "certainty": 1, "evidence": [{"unit_id": first.unit_id, "quote": "A reaches gate"}]},
                {"id": "f2", "kind": "resource", "subject_id": "e1", "predicate": "gains", "value": "token", "object_id": "", "certainty": 1, "evidence": [{"unit_id": second.unit_id, "quote": "A gains token"}]},
            ],
            "events": [{"id": "v1", "order": 1, "summary": "A reaches gate and gains token", "participant_ids": ["e1"], "trigger_fact_ids": ["f1"], "precondition_fact_ids": [], "action": "reaches gate and gains token", "obstacle": "", "decision": "", "outcome_fact_ids": ["f2"], "cost_fact_ids": [], "evidence": [{"unit_id": first.unit_id, "quote": "A reaches gate"}, {"unit_id": second.unit_id, "quote": "A gains token"}]}],
            "scenes": [{"id": "s1", "order": 1, "participant_ids": ["e1"], "event_ids": ["v1"], "objective": "gain token", "entry_fact_ids": ["f1"], "exit_fact_ids": ["f2"], "tension": 1, "evidence": [{"unit_id": first.unit_id, "quote": "A reaches gate"}, {"unit_id": second.unit_id, "quote": "A gains token"}]}],
            "state_changes": [{"id": "c1", "event_id": "v1", "operation": "add", "before_fact_id": "", "after_fact_id": "f2"}],
        }
        annotation = annotation_from_draft(document, draft)
        self.assertTrue(validate_annotation(annotation, document).passed)
        self.assertTrue(assess_annotation_quality(document, annotation).passed)
        self.assertEqual(annotation.events[0].evidence[0].quote, "A reaches gate")
        attribute_draft = {**draft, "facts": [{**draft["facts"][0], "kind": "属性"}, *draft["facts"][1:]]}
        self.assertEqual(annotation_from_draft(document, attribute_draft).facts[0].kind, "information")
        event_kind_draft = {**draft, "facts": [{**draft["facts"][0], "kind": "事件"}, *draft["facts"][1:]]}
        self.assertEqual(annotation_from_draft(document, event_kind_draft).facts[0].kind, "progression")
        expectation_draft = {**draft, "facts": [{**draft["facts"][0], "kind": "期望"}, *draft["facts"][1:]]}
        self.assertEqual(annotation_from_draft(document, expectation_draft).facts[0].kind, "goal")
        without_scenes = {key: list(value) for key, value in draft.items()}
        without_scenes["scenes"] = []
        derived = sanitize_draft(document, without_scenes, split_annotation_units(document)[0])
        self.assertEqual(len(derived["scenes"]), 1)
        self.assertTrue(validate_annotation(annotation_from_draft(document, derived), document).passed)
        indirect_location = replace(annotation.entities[1], evidence=(SourceSpan(chapter_id, 0, 1, "A", first.unit_id),))
        enriched = enrich_entity_evidence(document, replace(annotation, entities=(annotation.entities[0], indirect_location)))
        self.assertTrue(any(span.quote == "gate" for span in enriched.entities[1].evidence))
        draft["facts"][1]["evidence"][0]["unit_id"] = first.unit_id
        relocated = annotation_from_draft(document, draft)
        self.assertEqual(relocated.facts[1].evidence[0].unit_id, second.unit_id)
        draft["facts"].append({"id": "f-bad", "kind": "resource", "subject_id": "e1", "predicate": "gains", "value": "ghost", "object_id": "", "certainty": 1, "evidence": [{"unit_id": first.unit_id, "quote": "missing"}]})
        draft["facts"].append({"id": "f-empty", "kind": "resource", "subject_id": "e1", "predicate": "gains", "value": "", "object_id": "", "certainty": 1, "evidence": [{"unit_id": first.unit_id, "quote": "A"}]})
        draft["scenes"][0]["tension"] = 9
        cleaned = sanitize_draft(document, draft, split_annotation_units(document)[0])
        self.assertEqual(len(cleaned["facts"]), 2)
        self.assertEqual(cleaned["scenes"][0]["tension"], 5)

    def test_annotation_draft_rejects_evidence_not_found_in_declared_unit(self) -> None:
        chapter_id = "Author/work/0004"
        text = "A reaches gate.\nA gains token."
        document = build_chapter_document(chapter_id, hashlib.sha256(text.encode("utf-8")).hexdigest(), text, "Start")
        bad = {
            "entities": [{"id": "e1", "name": "A", "kind": "person", "aliases": [], "evidence": [{"unit_id": document.units[0].unit_id, "quote": "missing"}]}],
            "facts": [], "events": [], "scenes": [], "state_changes": [],
        }
        with self.assertRaises(AnnotationBuildError):
            annotation_from_draft(document, bad)

    def test_spatiotemporal_enrichment_keeps_evidence_and_scene_links(self) -> None:
        chapter_id = "Author/work/time-space"
        text = "甲清晨到达村口。\n随后甲进入客栈。"
        document = build_chapter_document(chapter_id, hashlib.sha256(text.encode("utf-8")).hexdigest(), text, "Start")
        first, second = document.units
        draft = {
            "entities": [
                {"id": "e1", "name": "甲", "kind": "person", "aliases": [], "evidence": [{"unit_id": first.unit_id, "quote": "甲"}]},
                {"id": "e2", "name": "村口", "kind": "location", "aliases": [], "evidence": [{"unit_id": first.unit_id, "quote": "村口"}]},
                {"id": "e3", "name": "客栈", "kind": "location", "aliases": [], "evidence": [{"unit_id": second.unit_id, "quote": "客栈"}]},
            ],
            "facts": [
                {"id": "f1", "kind": "location", "subject_id": "e1", "predicate": "到达", "value": "村口", "object_id": "e2", "certainty": 1, "evidence": [{"unit_id": first.unit_id, "quote": "甲清晨到达村口"}]},
                {"id": "f2", "kind": "location", "subject_id": "e1", "predicate": "进入", "value": "客栈", "object_id": "e3", "certainty": 1, "evidence": [{"unit_id": second.unit_id, "quote": "随后甲进入客栈"}]},
            ],
            "events": [
                {"id": "v1", "order": 1, "summary": "甲到达村口", "participant_ids": ["e1"], "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "到达村口", "obstacle": "", "decision": "", "outcome_fact_ids": ["f1"], "cost_fact_ids": [], "evidence": [{"unit_id": first.unit_id, "quote": "甲清晨到达村口"}]},
                {"id": "v2", "order": 2, "summary": "甲进入客栈", "participant_ids": ["e1"], "trigger_fact_ids": ["f1"], "precondition_fact_ids": [], "action": "进入客栈", "obstacle": "", "decision": "", "outcome_fact_ids": ["f2"], "cost_fact_ids": [], "evidence": [{"unit_id": second.unit_id, "quote": "随后甲进入客栈"}]},
            ],
            "scenes": [
                {"id": "s1", "order": 1, "participant_ids": ["e1"], "event_ids": ["v1"], "objective": "到达村口", "entry_fact_ids": [], "exit_fact_ids": ["f1"], "tension": 1, "evidence": [{"unit_id": first.unit_id, "quote": "甲清晨到达村口"}]},
                {"id": "s2", "order": 2, "participant_ids": ["e1"], "event_ids": ["v2"], "objective": "进入客栈", "entry_fact_ids": ["f1"], "exit_fact_ids": ["f2"], "tension": 1, "evidence": [{"unit_id": second.unit_id, "quote": "随后甲进入客栈"}]},
            ],
            "state_changes": [],
        }
        annotation = enrich_spatiotemporal(document, annotation_from_draft(document, draft))
        self.assertEqual(annotation.schema_version, "2.1")
        self.assertEqual(len(annotation.time_anchors), 2)
        self.assertEqual(len(annotation.temporal_relations), 1)
        self.assertEqual(annotation.temporal_relations[0].basis, "document_order")
        self.assertEqual([relation.kind for relation in annotation.spatial_relations], ["moves_to", "enters"])
        self.assertEqual(len(annotation.scenes[0].location_ids), 1)
        self.assertEqual(len(annotation.scenes[1].time_anchor_ids), 1)
        self.assertTrue(validate_annotation(annotation, document).passed)
        self.assertTrue(assess_annotation_quality(document, annotation).passed)

    def test_annotation_chunking_keeps_whole_units_with_bounded_overlap(self) -> None:
        chapter_id = "Author/work/0005"
        text = "\n".join(["A" * 600, "B" * 600, "C" * 600])
        document = build_chapter_document(chapter_id, hashlib.sha256(text.encode("utf-8")).hexdigest(), text, "Long")
        chunks = split_annotation_units(document, max_input_chars=1_000, overlap_units=1)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(sum(len(unit.text) for unit in chunk) <= 1_000 for chunk in chunks))
        self.assertEqual([chunk[0].unit_id for chunk in chunks], [unit.unit_id for unit in document.units])

        one_line = build_chapter_document(chapter_id, hashlib.sha256(("Z" * 2_100).encode("utf-8")).hexdigest(), "Z" * 2_100, "Long")
        sliced = split_annotation_units(one_line, max_input_chars=1_000, overlap_units=1)
        self.assertEqual(len(sliced), 3)
        self.assertTrue(all(sum(len(unit.text) for unit in chunk) <= 1_000 for chunk in sliced))
        self.assertTrue(all(unit.source_unit_id == one_line.units[0].unit_id for chunk in sliced for unit in chunk))

    def test_validator_rejects_false_quote_and_unknown_reference(self) -> None:
        document, annotation, style = self._annotation_fixture()
        bad_span = SourceSpan(annotation.chapter_id, 0, 2, "李慕婉", document.units[0].unit_id)
        bad_entity = replace(annotation.entities[0], evidence=(bad_span,))
        bad_fact = replace(annotation.facts[0], subject_id="person-missing")
        invalid = replace(annotation, entities=(bad_entity, *annotation.entities[1:]), facts=(bad_fact, *annotation.facts[1:]))
        report = validate_annotation(invalid, document, style)
        self.assertFalse(report.passed)
        codes = {issue.code for issue in report.issues}
        self.assertIn("invalid_contract", codes)
        self.assertIn("unknown_entity", codes)
        with self.assertRaises(ContractValidationError):
            validate_annotation(invalid, document, style, raise_on_error=True)

    def test_template_contract_rejects_false_support_count(self) -> None:
        template = Template(
            "scene-001", "scene", "受阻后转向", ("存在明确目标",), ("行动者",), ("目标", "阻力", "选择"),
            ("resource:add",), ("阻力来源",), ("无代价解决",), ("event-001", "event-002"), 3, 0.8,
        )
        with self.assertRaises(ValueError):
            template.validate()

    def test_skill_contracts_round_trip(self) -> None:
        template = Template(
            "scene-001", "scene", "受阻后转向", ("存在明确目标",), ("行动者",), ("目标", "阻力", "选择"),
            ("resource:add",), ("阻力来源",), ("无代价解决",), ("event-001", "event-002"), 2, 0.8,
        )
        profile = AuthorProfile("profile-001", "Author", ("work-001",), ("scene-001",), ("先落实目标",), ("对白比例",), 0.8)
        program = PromptProgram("program-001", "profile-001", "planning", ("story_state", "chapter_goal"), "ChapterPlan", ("先选模板",), ("logic_pass_rate",), "1.0")
        for contract in (template, profile, program):
            contract.validate()
            self.assertEqual(type(contract).from_dict(contract.to_dict()).to_dict(), contract.to_dict())

    def test_parsed_source_documents_preserve_exact_text_for_a_generic_fixture(self) -> None:
        text = """第1章 起始
人物甲来到村口。
“这句话还没说完
仍在同一句对话里。”
（本章完）
第2章 公告
任意笔名求月票支持
（本章完）
"""
        chapters = parse_chapters(text, "ContractCheck", "work-001")
        self.assertEqual([chapter.source_chapter_no for chapter in chapters], [1, 2])
        for chapter in chapters:
            document = build_chapter_document(chapter.chapter_id, chapter.source_hash, chapter.text, chapter.title)
            report = validate_document(document)
            self.assertTrue(report.passed, report.to_dict())
            self.assertIsNone(CHAPTER_RE.search(chapter.text), chapter.chapter_id)
            for unit in document.units:
                self.assertEqual(chapter.text[unit.start:unit.end], unit.text)
        self.assertTrue(build_chapter_document(chapters[0].chapter_id, chapters[0].source_hash, chapters[0].text, chapters[0].title).is_narrative)
        self.assertFalse(build_chapter_document(chapters[1].chapter_id, chapters[1].source_hash, chapters[1].text, chapters[1].title).is_narrative)


if __name__ == "__main__":
    unittest.main()
