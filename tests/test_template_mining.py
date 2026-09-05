from __future__ import annotations

import unittest

from pipeline.contracts import NarrativeTemplateLibrary
from pipeline.template_mining import _offline_draft, _templates_from_draft, assess_template_library


class TemplateMiningTests(unittest.TestCase):
    def test_offline_template_contract_requires_cross_phase_macro_support(self) -> None:
        chapter_ids = [f"Author/work/{index:04d}" for index in range(1, 11)]
        cards = [{"chapter_id": chapter_id} for chapter_id in chapter_ids]
        levels = ("micro", "event", "scene", "chapter", "macro")
        templates = _templates_from_draft(
            _offline_draft(cards, levels), levels, set(chapter_ids), "test",
        )
        library = NarrativeTemplateLibrary("Author", "work", tuple(chapter_ids), "1.0", templates)
        report = assess_template_library(library)
        self.assertTrue(report.passed, report.to_dict())
        self.assertEqual(report.templates_by_level["macro"], 1)

        short_ids = chapter_ids[:5]
        short_macro = _templates_from_draft(
            _offline_draft([{"chapter_id": chapter_id} for chapter_id in short_ids], ("macro",)),
            ("macro",), set(short_ids), "test",
        )
        self.assertEqual(short_macro, ())

    def test_shallow_macro_candidate_is_discarded_before_library_validation(self) -> None:
        chapter_ids = [f"Author/work/{index:04d}" for index in range(1, 11)]
        draft = _offline_draft([{"chapter_id": chapter_id} for chapter_id in chapter_ids], ("macro",))
        draft["macro_templates"][0]["beat_sequence"] = ["进入", "转折", "退出"]
        templates = _templates_from_draft(draft, ("macro",), set(chapter_ids), "test")
        self.assertEqual(templates, ())


if __name__ == "__main__":
    unittest.main()
