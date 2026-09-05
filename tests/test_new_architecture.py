from __future__ import annotations

import hashlib
import unittest

from pipeline.contracts import build_chapter_document
from pipeline.modules.module_00_ingestion.api import chinese_number, parse_chapters
from pipeline.orchestrator import ModuleLoadError, load_entrypoint


class NewArchitectureTests(unittest.TestCase):
    def test_chinese_chapter_number(self) -> None:
        self.assertEqual(chinese_number("十二", 0), 12)
        self.assertEqual(chinese_number("一百零三", 0), 103)
        self.assertEqual(chinese_number("１２８", 0), 128)

    def test_ingestion_parses_chinese_headings(self) -> None:
        records = parse_chapters(
            "第一章 起点\n正文甲。\n第二章 继续\n正文乙。",
            "Writer",
            "work-001",
        )
        self.assertEqual([item.source_chapter_no for item in records], [1, 2])
        self.assertEqual(records[0].title, "起点")

    def test_source_document_has_exact_spans(self) -> None:
        text = "第一段。\n“第二段。”"
        document = build_chapter_document(
            "Writer/work-001/0001",
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
            text,
        )
        document.validate()
        self.assertEqual(document.units[1].kind, "dialogue")
        self.assertEqual(document.text[document.units[1].start:document.units[1].end], "“第二段。”")

    def test_registry_reports_invalid_entrypoint(self) -> None:
        with self.assertRaises(ModuleLoadError):
            load_entrypoint("not-a-valid-entrypoint")


if __name__ == "__main__":
    unittest.main()

