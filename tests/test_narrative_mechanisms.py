from __future__ import annotations

import hashlib
import unittest

from pipeline.annotate import (
    AnnotationBuildError,
    _chapter_mechanism_count_bounds,
    chapter_narrative_mechanism_prompt,
    narrative_mechanism_prompt,
    narrative_mechanisms_from_draft,
    split_annotation_units,
)
from pipeline.contracts import (
    ChapterAnnotation,
    Entity,
    EventAtom,
    Fact,
    NarrativeMechanism,
    SceneCard,
    SourceSpan,
    build_chapter_document,
)


class NarrativeMechanismTests(unittest.TestCase):
    def _fixture(self):
        chapter_id = "Author/work/0001"
        text = "陆遥想离开村庄，却因照顾病父而迟疑。\n村外的征召令给了他机会，他最终报名。"
        document = build_chapter_document(
            chapter_id, hashlib.sha256(text.encode("utf-8")).hexdigest(), text, "第一章",
        )
        first, second = document.units

        def span(quote: str, unit_id: str) -> SourceSpan:
            start = text.index(quote)
            return SourceSpan(chapter_id, start, start + len(quote), quote, unit_id)

        person_span = span("陆遥", first.unit_id)
        desire_span = span("想离开村庄", first.unit_id)
        constraint_span = span("照顾病父而迟疑", first.unit_id)
        result_span = span("最终报名", second.unit_id)
        person = Entity("person-001", "陆遥", "person", (), (person_span,))
        desire = Fact("fact-001", chapter_id, "goal", "person-001", "想", "离开村庄", "", 1.0, (desire_span,))
        constraint = Fact("fact-002", chapter_id, "rule", "person-001", "受限于", "照顾病父", "", 1.0, (constraint_span,))
        result = Fact("fact-003", chapter_id, "progression", "person-001", "已经", "报名", "", 1.0, (result_span,))
        event = EventAtom(
            "event-001", chapter_id, 0, "陆遥面对征召后报名", ("person-001",),
            ("fact-001",), ("fact-002",), "权衡后报名", "照顾病父形成牵制", "报名",
            ("fact-003",), (), (result_span,), "decision", "person-001", (),
            ("fact-001", "fact-002", "fact-003"),
        )
        scene = SceneCard(
            "scene-001", chapter_id, 0, ("person-001",), ("event-001",),
            "让愿望在现实牵制下形成选择", ("fact-001", "fact-002"), ("fact-003",),
            3, (desire_span, constraint_span, result_span),
        )
        annotation = ChapterAnnotation(
            chapter_id, document.source_hash, (person,), (desire, constraint, result),
            (event,), (scene,), (),
        )
        return document, annotation, desire_span, constraint_span, result_span

    def _valid_payload(self, desire_span: SourceSpan, constraint_span: SourceSpan) -> dict[str, object]:
        return {"mechanisms": [{
            "id": "m1", "order": 1, "kind": "限制压力", "logic_relation": "阻碍",
            "summary": "离乡愿望受到照顾亲人的现实责任牵制",
            "narrative_function": "让报名先经过责任压力，再成为有代价的主动选择",
            "counterfactual_loss": "删去照顾责任后，报名会变成没有权衡过程的顺势行动",
            "fact_ids": ["fact-001", "fact-002"], "event_ids": ["event-001"],
            "participant_ids": ["person-001"], "inference_level": "结构推断",
            "setup_status": "不适用", "confidence": 0.91,
            "evidence": [
                {"unit_id": desire_span.unit_id, "quote": desire_span.quote},
                {"unit_id": constraint_span.unit_id, "quote": constraint_span.quote},
            ],
        }]}

    def _units(self, document):
        return split_annotation_units(document, max_input_chars=10_000, overlap_units=0)[0]

    def test_deep_mechanism_parser_accepts_anchored_causal_pressure(self) -> None:
        document, annotation, desire_span, constraint_span, _ = self._fixture()
        mechanisms = narrative_mechanisms_from_draft(
            document,
            self._units(document),
            annotation,
            self._valid_payload(desire_span, constraint_span),
        )
        self.assertEqual(len(mechanisms), 1)
        self.assertEqual(mechanisms[0].kind, "constraint_pressure")
        self.assertEqual(mechanisms[0].logic_relation, "obstacle")
        self.assertIn("权衡", mechanisms[0].counterfactual_loss)

    def test_deep_mechanism_parser_rejects_generic_function_even_with_valid_ids(self) -> None:
        document, annotation, desire_span, constraint_span, _ = self._fixture()
        payload = self._valid_payload(desire_span, constraint_span)
        payload["mechanisms"][0]["narrative_function"] = "推动剧情"
        with self.assertRaisesRegex(AnnotationBuildError, "过于空泛"):
            narrative_mechanisms_from_draft(
                document, self._units(document), annotation, payload,
            )

    def test_structural_inference_requires_more_than_one_anchor(self) -> None:
        document, annotation, desire_span, constraint_span, _ = self._fixture()
        payload = self._valid_payload(desire_span, constraint_span)
        payload["mechanisms"][0]["fact_ids"] = ["fact-001"]
        payload["mechanisms"][0]["event_ids"] = []
        with self.assertRaisesRegex(AnnotationBuildError, "至少需要两个"):
            narrative_mechanisms_from_draft(
                document, self._units(document), annotation, payload,
            )

    def test_prompts_target_deep_logic_and_chapter_synthesis(self) -> None:
        document, annotation, desire_span, constraint_span, _ = self._fixture()
        mechanism = narrative_mechanisms_from_draft(
            document, self._units(document), annotation,
            self._valid_payload(desire_span, constraint_span),
        )[0]
        annotation = ChapterAnnotation(
            annotation.chapter_id, annotation.source_hash, annotation.entities,
            annotation.facts, annotation.events, annotation.scenes,
            annotation.state_changes, "2.3", (), (), (), (mechanism,),
        )
        local_prompt = narrative_mechanism_prompt(
            document, self._units(document), annotation,
        )
        chapter_prompt = chapter_narrative_mechanism_prompt(document, annotation)
        self.assertIn("信念/认知—欲望—限制或压力—选择—结果", local_prompt)
        self.assertIn("反事实删除", local_prompt)
        self.assertIn("不提供整章原文", chapter_prompt)
        self.assertIn("局部候选机制", chapter_prompt)
        self.assertIn("人物认知/信念—欲望—现实限制—选择—结果/代价", chapter_prompt)
        minimum, maximum = _chapter_mechanism_count_bounds(annotation)
        self.assertGreaterEqual(minimum, 3)
        self.assertGreaterEqual(maximum, minimum)

    def test_setup_cannot_masquerade_as_verified_payoff(self) -> None:
        document, annotation, desire_span, _, _ = self._fixture()
        mechanism = NarrativeMechanism(
            "mechanism-001", annotation.chapter_id, 0, "setup", "reveal",
            "征召令暂时只作为未来选择的开放入口",
            "建立后续选择可能性但不提前兑现",
            "删去后，后续选择缺少先行入口",
            ("fact-001",), (), ("person-001",), "explicit", "verified_payoff", 0.8,
            (desire_span,),
        )
        with self.assertRaisesRegex(ValueError, "remain open"):
            mechanism.validate()


if __name__ == "__main__":
    unittest.main()
