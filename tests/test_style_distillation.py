from __future__ import annotations

import hashlib
import unittest

from pipeline.contracts import AuthorStyleProfile, StyleBaseline, StyleConstraint, build_chapter_document
from pipeline.style_distillation import (
    _offline_constraints,
    _style_feature_cards,
    _constraint_rows,
    _neutralise_constraint_language,
    assess_style_profile,
    build_chapter_style_card,
)


class StyleDistillationTests(unittest.TestCase):
    def test_cards_measure_style_without_keeping_source_observations(self) -> None:
        text = "他看见远处的光。\n\n“快走！”她问：现在吗？\n\n随后，他缓缓向前。"
        document = build_chapter_document("Author/work/0001", hashlib.sha256(text.encode("utf-8")).hexdigest(), text, "Start")
        card = build_chapter_style_card(document)
        metrics = {item.metric_id: item for item in card.metrics}
        self.assertGreater(metrics["paragraph_count"].value, 0)
        self.assertGreater(metrics["dialogue_char_ratio"].value, 0)
        self.assertGreater(metrics["question_sentence_ratio"].value, 0)
        self.assertEqual(card.observations, ())

    def test_offline_profile_meets_coverage_and_family_contract(self) -> None:
        cards = []
        for index in range(1, 11):
            text = f"随后，他看见变化。\n\n“现在行动！”\n\n第{index}次判断带来结果。"
            document = build_chapter_document(
                f"Author/work/{index:04d}", hashlib.sha256(text.encode("utf-8")).hexdigest(), text, "Start",
            )
            cards.append(build_chapter_style_card(document))
        features = _style_feature_cards(cards)
        constraints = _constraint_rows({"constraints": _offline_constraints(features)}, {card.chapter_id for card in cards}, "test")
        profile = AuthorStyleProfile(
            "Author", "work", tuple(card.chapter_id for card in cards),
            (StyleBaseline("paragraph_count", 3, 3, 3, "count"),), constraints,
            ("使用抽象约束。",),
        )
        report = assess_style_profile(profile)
        self.assertTrue(report.passed, report.to_dict())
        self.assertGreaterEqual(len(report.families), 4)
        self.assertEqual(report.evidence_chapter_coverage, 10)

    def test_source_term_in_constraint_is_rejected(self) -> None:
        constraint = StyleConstraint(
            "test:sentence_rhythm-001", "sentence_rhythm", "围绕特定名字安排句式变化，形成完整节奏。",
            "在动作与判断之间安排完整的因果交代。", "不要固定使用同一种长度的句子。",
            ("Author/work/0001", "Author/work/0002"), 2, 0.8, ("mean_sentence_chars",),
        )
        profile = AuthorStyleProfile(
            "Author", "work", ("Author/work/0001", "Author/work/0002"),
            (StyleBaseline("paragraph_count", 1, 1, 1, "count"),), (constraint,), ("使用抽象约束。",),
        )
        report = assess_style_profile(profile, ("特定名字",))
        self.assertTrue(any(item.code == "source_or_copying_language" for item in report.issues))

    def test_neutraliser_removes_false_universal_quantifiers(self) -> None:
        row = StyleConstraint(
            "test:sentence_rhythm-001", "sentence_rhythm", "每章稳定在完整句与短句交替。",
            "每轮对话都在动作后落下判断。", "不要大量使用短句。",
            ("Author/work/0001", "Author/work/0002"), 2, 0.8, ("mean_sentence_chars",),
        )
        normalised = _neutralise_constraint_language((row,))[0]
        combined = "\n".join((normalised.rule, normalised.application, normalised.avoid))
        self.assertFalse(any(term in combined for term in ("每章", "稳定在", "每轮", "大量使用", "始终", "限定视角")))

    def test_neutraliser_rewrites_unsupported_viewpoint_claim(self) -> None:
        row = StyleConstraint(
            "test:focalization-001", "focalization", "叙述始终限定视角并偏重外部聚焦。",
            "只呈现人物感受到的信息。", "不要进入全知视角。",
            ("Author/work/0001", "Author/work/0002"), 2, 0.8,
            ("perception_marker_ratio", "interior_marker_ratio"),
        )
        normalised = _neutralise_constraint_language((row,))[0]
        combined = "\n".join((normalised.rule, normalised.application, normalised.avoid))
        self.assertNotIn("视角", combined)
        self.assertIn("反应层", combined)


if __name__ == "__main__":
    unittest.main()
