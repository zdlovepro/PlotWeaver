from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipeline.ingest import parse_chapters
from pipeline.extract import _normalise_entity_ids, _scene_contract_issues, select_chapter_records, split_chapter_text
from pipeline.jsonio import read_jsonl, write_jsonl
from pipeline.model import _response_json_or_error, extract_json
from pipeline.original import _offline_preview
from pipeline.paths import validate_author_id
from pipeline.templates import select_templates
from pipeline.contracts import ChapterState, StoryState
from pipeline.state import _apply_scene_state, _register_entities, _resolve_entity, _scene_entry
from pipeline.compile_skill import _prompt_engineering


class V3Tests(unittest.TestCase):
    def test_author_id_validation(self) -> None:
        self.assertEqual(validate_author_id("Author_01-2"), "Author_01-2")
        with self.assertRaises(ValueError):
            validate_author_id("../unsafe")

    def test_duplicate_and_chinese_chapter_headings(self) -> None:
        text = """第1章 开始
甲。
第一章 开始
甲。
第二章 发展
乙。"""
        chapters = parse_chapters(text, "Author", "work")
        self.assertEqual(len(chapters), 2)
        self.assertEqual([chapter.source_chapter_no for chapter in chapters], [1, 2])
        self.assertEqual([chapter.chapter_no for chapter in chapters], [1, 2])
        self.assertNotIn("第一章 开始", chapters[0].text)

    def test_chapter_title_excludes_publication_note(self) -> None:
        chapters = parse_chapters("第1章 正文标题 （第一更！求月票！）\n正文。", "Author", "work")
        self.assertEqual(chapters[0].title, "正文标题")

    def test_duplicate_heading_chooses_plausible_title_and_removes_fragments(self) -> None:
        chapters = parse_chapters(
            "第1章 正常标题\n第1章 人\n正文甲。\n第2章 下章\n正文乙。",
            "Author",
            "work",
        )
        self.assertEqual([chapter.title for chapter in chapters], ["正常标题", "下章"])
        self.assertEqual(chapters[0].text, "正文甲。")

    def test_jsonl_accepts_string_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "rows.jsonl")
            write_jsonl(path, [{"id": 1}])
            self.assertEqual(read_jsonl(path), [{"id": 1}])

    def test_model_json_parser_requires_valid_json(self) -> None:
        self.assertEqual(extract_json('```json\n{"ok": true}\n```'), {"ok": True})
        with self.assertRaises(ValueError):
            extract_json('{"draft":"line one\nline two"}')
        with self.assertRaisesRegex(RuntimeError, "空内容"):
            _response_json_or_error(None)

    def test_long_chapter_is_split_with_bounded_overlap(self) -> None:
        text = "甲" * 1400 + "\n" + "乙" * 1400
        fragments = split_chapter_text(text, max_input_chars=1000, overlap_chars=100)
        self.assertGreaterEqual(len(fragments), 2)
        self.assertTrue(all(len(fragment) <= 1101 for fragment in fragments))
        self.assertIn("乙", fragments[-1])

    def test_stratified_selection_spans_the_book(self) -> None:
        records = parse_chapters("\n".join(f"第{i}章 标题{i}\n正文{i}" for i in range(1, 11)), "Author", "work")
        selected = select_chapter_records(records, 4, "stratified")
        self.assertEqual([row.chapter_no for row in selected], [1, 4, 7, 10])

    def test_original_preview_requires_new_brief_data(self) -> None:
        preview = _offline_preview({"chapter_goal": "Test goal", "world": {"name": "New World"}}, {"profile_version": "1.0"})
        self.assertIn("Test goal", preview)
        self.assertIn("New World", preview)

    def test_v3_has_no_v2_import_path(self) -> None:
        package = Path(__file__).resolve().parents[1] / "pipeline"
        source = "\n".join(path.read_text(encoding="utf-8") for path in package.rglob("*.py"))
        self.assertNotIn("pipeline.step", source)
        self.assertNotIn("pipeline.core", source)

    def test_template_selection_keeps_macro_level(self) -> None:
        library = {
            "macro_templates": [{"template_id": "macro-001", "selection_tags": ["pressure"], "support_count": 3}],
            "event_templates": [{"template_id": "event-001", "selection_tags": ["pressure"], "support_count": 2}],
            "micro_templates": [],
            "chapter_templates": [],
        }
        selected = select_templates(library, {"chapter_goal": "pressure test"})
        self.assertEqual(selected["macro_templates"][0]["template_id"], "macro-001")

    def test_entity_aliases_resolve_to_one_stable_id(self) -> None:
        state = StoryState(chapter_id="")
        _register_entities(state, [{"entity_id": "person-001", "canonical_name": "王林", "aliases": ["铁柱", "王林（铁柱）"]}])
        self.assertEqual(_resolve_entity("铁柱", state), "person-001")
        self.assertEqual(_resolve_entity("王林", state), "person-001")

    def test_repeated_entity_is_remapped_to_prior_stable_id(self) -> None:
        chapter = ChapterState.from_dict({
            "entities": [{"entity_id": "wanglin", "canonical_name": "王林", "aliases": ["铁柱"]}],
            "scene_beats": [{"scene_no": 1, "participants": ["wanglin"], "knowledge_updates": [{"entity": "wanglin", "learns": "新线索"}]}],
        }, "Author/work/0002")
        _normalise_entity_ids(chapter, {"person-001": {"entity_id": "person-001", "canonical_name": "王林", "aliases": ["铁柱"]}})
        self.assertEqual(chapter.entities[0]["entity_id"], "person-001")
        self.assertEqual(chapter.scene_beats[0]["participants"], ["person-001"])
        self.assertEqual(chapter.scene_beats[0]["knowledge_updates"][0]["entity"], "person-001")

    def test_empty_scene_contract_requires_repair(self) -> None:
        issues = _scene_contract_issues([], [])
        self.assertTrue(issues["empty_scene_contract"])

    def test_chapter_state_keeps_scene_transition_fields(self) -> None:
        state = ChapterState.from_dict({
            "scene_beats": [{"scene_no": 1, "preconditions": ["已到达地点"], "before_state": {"location": "甲"}, "after_state": {"location": "乙"}, "knowledge_updates": [{"entity": "person-001", "learns": "线索"}], "fact_provenance": [{"fact": "移动", "quote": "前往乙地"}]}],
            "entities": [{"entity_id": "person-001", "canonical_name": "角色甲"}],
        }, "Author/work/0001")
        self.assertEqual(state.entities[0]["entity_id"], "person-001")
        self.assertEqual(state.scene_beats[0]["after_state"]["location"], "乙")

    def test_scene_transition_applies_alias_location_and_knowledge(self) -> None:
        state = StoryState(chapter_id="Author/work/0001")
        _register_entities(state, [{"entity_id": "person-001", "canonical_name": "王林", "aliases": ["铁柱"]}])
        entry = _scene_entry(state.chapter_id, {
            "scene_no": 1,
            "location": "村口",
            "participants": ["铁柱"],
            "preconditions": ["已经到达村口"],
            "after_state": {"time": "傍晚", "location": "村口"},
            "knowledge_updates": [{"entity": "person-001", "learns": "四叔将到来"}],
        }, state)
        _apply_scene_state(state, entry)
        self.assertEqual(entry["participants"], ["person-001"])
        self.assertEqual(state.locations["person-001"], "村口")
        self.assertIn("四叔将到来", state.knowledge["person-001"])

    def test_prompt_engineering_contains_logic_contract(self) -> None:
        bundle = _prompt_engineering({"profile_version": "2.0", "prompt_engineering": {}}, {"library_version": "1.0"}, ["保留时间跨度"])
        self.assertIn("实体身份", bundle["stages"]["validation"]["checks"])
        self.assertIn("保留时间跨度", bundle["feedback_rules"])


if __name__ == "__main__":
    unittest.main()
