from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline.semantic_repair_cycle import _selected_chapter_numbers
from pipeline.annotate import annotate_work
from pipeline.semantic_repair import (
    _blocking_patch_quality_issues,
    _compound_coverage_defer_payload,
    _constrain_audience_only_patch,
    _constrain_future_arrangement_event_patch,
    _constrain_missing_fact_patch,
    _constrain_reaction_causal_patch,
    _deterministic_disjoint_event_outcome_patch,
    _deterministic_action_fact_object_patch,
    _deterministic_future_arrangement_speech_patch,
    _deterministic_observed_action_patch,
    _deterministic_relationship_direction_swap_patch,
    _deterministic_unsupported_relationship_removal_patch,
    _deterministic_ungrounded_speaker_removal_patch,
    _deterministic_safe_fact_removal_patch,
    _deterministic_missing_speech_patch,
    _deterministic_mixed_speech_evidence_patch,
    _deterministic_reporting_fact_kind_patch,
    _deterministic_reporting_listener_object_patch,
    _deterministic_unsupported_speech_shrink_patch,
    _normalize_repaired_annotation,
    _issue_fingerprint,
    _is_confirmed_repair_target,
    _parse_patch_review,
    _repair_issue_groups,
    _repair_output_paths,
    _repair_prompt,
    _repair_source_slices,
    _issue_superseded_by_prior_patch,
    _resumed_terminal_state,
)
from pipeline.annotate import AnnotationSourceSlice
from pipeline.contracts import ChapterAnnotation, build_chapter_document


class SemanticRepairCycleTests(unittest.TestCase):
    def test_listener_object_repair_preserves_proposition_and_clears_object(self) -> None:
        _, annotation = self._fixture_annotation_for_ungrounded_speaker()
        fact = replace(annotation.facts[0], kind="information", object_id=annotation.entities[0].entity_id)
        changed = replace(annotation, facts=(fact,))
        issue = {"issue_id": f"speech-listener-object-{fact.fact_id.rsplit(':', 1)[-1]}",
                 "category": "fact_semantics", "object_ids": [fact.fact_id]}

        payload = _deterministic_reporting_listener_object_patch(changed, [issue])

        self.assertIsNotNone(payload)
        row = payload["upsert_facts"][0]
        self.assertEqual(row["object_id"], "")
        self.assertEqual(row["value"], fact.value)

    def test_relationship_direction_repair_only_swaps_reviewed_endpoints(self) -> None:
        _, annotation = self._fixture_annotation_for_ungrounded_speaker()
        data = annotation.to_dict()
        span = data["facts"][0]["evidence"][0]
        data["entities"].append({
            "entity_id": "entity-c", "canonical_name": "丙", "kind": "person",
            "aliases": [], "evidence": [span],
        })
        data["facts"] = [{
            "fact_id": "fact-relation", "chapter_id": annotation.chapter_id,
            "kind": "relationship", "subject_id": "entity-b",
            "predicate": "是二叔家的", "value": "堂兄弟", "object_id": "entity-c",
            "certainty": 1.0, "evidence": [span],
        }]
        changed = ChapterAnnotation.from_dict(data)

        payload = _deterministic_relationship_direction_swap_patch(changed, [{
            "issue_id": "relation-direction", "category": "fact_semantics",
            "object_ids": ["fact-relation"], "diagnosis": "关系主客体方向错误。",
        }])

        self.assertIsNotNone(payload)
        row = payload["upsert_facts"][0]
        self.assertEqual((row["subject_id"], row["object_id"]), ("entity-c", "entity-b"))
        self.assertEqual((row["predicate"], row["value"]), ("是二叔家的", "堂兄弟"))
        self.assertEqual(row["evidence"], [{"unit_id": span["unit_id"], "quote": span["quote"]}])

    def test_action_object_repair_reuses_existing_event_target(self) -> None:
        _, annotation = self._fixture_annotation_for_ungrounded_speaker()
        data = annotation.to_dict()
        span = data["facts"][0]["evidence"][0]
        data["entities"].append({
            "entity_id": "entity-rescued", "canonical_name": "被救者", "kind": "person",
            "aliases": [], "evidence": [span],
        })
        data["facts"] = [{
            "fact_id": "fact-rescue", "chapter_id": annotation.chapter_id,
            "kind": "progression", "subject_id": "entity-b", "predicate": "救下",
            "value": "少年", "object_id": "", "certainty": 1.0, "evidence": [span],
        }]
        data["events"] = [{
            "event_id": "event-rescue", "chapter_id": annotation.chapter_id, "order": 0,
            "summary": "救下少年", "participant_ids": ["entity-b", "entity-rescued"],
            "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "救下少年",
            "obstacle": "", "decision": "", "outcome_fact_ids": ["fact-rescue"],
            "cost_fact_ids": [], "evidence": [span], "action_type": "other",
            "actor_id": "entity-b", "target_ids": ["entity-rescued"],
            "basis_fact_ids": ["fact-rescue"],
        }]
        changed = ChapterAnnotation.from_dict(data)

        payload = _deterministic_action_fact_object_patch(changed, [{
            "issue_id": "missing-object", "category": "fact_semantics",
            "object_ids": ["fact-rescue"],
            "diagnosis": "救下少年的 object_id 为空，被救对象未绑定。",
        }])

        self.assertIsNotNone(payload)
        self.assertEqual(payload["upsert_facts"][0]["object_id"], "entity-rescued")
        self.assertEqual(payload["upsert_events"], [])

    def test_observed_action_repair_materializes_actor_without_destination(self) -> None:
        _, annotation = self._fixture_annotation_for_ungrounded_speaker()
        data = annotation.to_dict()
        span = data["facts"][0]["evidence"][0]
        data["entities"].extend([
            {"entity_id": "entity-observer", "canonical_name": "观察者", "kind": "person",
             "aliases": [], "evidence": [span]},
            {"entity_id": "entity-walker", "canonical_name": "中年人", "kind": "person",
             "aliases": [], "evidence": [span]},
        ])
        data["facts"] = [{
            "fact_id": "fact-018", "chapter_id": annotation.chapter_id,
            "kind": "knowledge", "subject_id": "entity-observer", "predicate": "注意到",
            "value": "中年人从石阶落下并路过少年", "object_id": "entity-walker",
            "certainty": 1.0, "evidence": [span],
        }]
        data["events"] = []
        changed = ChapterAnnotation.from_dict(data)

        payload = _deterministic_observed_action_patch(changed, [{
            "issue_id": "missing-observed-movement", "category": "coverage",
            "object_ids": ["missing:event:Author/work-001/0001:p0001"],
            "diagnosis": "中年人下山路过少年这一 movement 事件缺失，观察者感知该事件的 fact-018 缺动作承载。",
        }])

        self.assertIsNotNone(payload)
        fact = payload["upsert_facts"][0]
        event = payload["upsert_events"][0]
        self.assertEqual(fact["subject_id"], "entity-walker")
        self.assertEqual(fact["kind"], "progression")
        self.assertEqual(fact["predicate"], "移动")
        self.assertEqual(fact["object_id"], "")
        self.assertEqual(event["action_type"], "other")
        self.assertEqual(event["actor_id"], "entity-walker")
        self.assertEqual(event["outcome_fact_ids"], [fact["id"]])

    def test_compound_coverage_is_deferred_for_atomic_reaudit(self) -> None:
        issue = {
            "issue_id": "mixed-action", "category": "coverage",
            "object_ids": ["missing:event:Author/work-001/0001:p0001"],
            "diagnosis": "弟子把少年带上山并统一喂药，缺少对应事件。",
        }

        payload = _compound_coverage_defer_payload([issue])

        self.assertIsNotNone(payload)
        self.assertEqual(payload["resolved_issue_ids"], [])
        self.assertEqual(payload["deferred_issues"][0]["issue_id"], "mixed-action")
        self.assertIn("movement", payload["deferred_issues"][0]["reason"])
        self.assertIn("treatment", payload["deferred_issues"][0]["reason"])
        self.assertEqual(payload["upsert_events"], [])

        atomic = {**issue, "diagnosis": "弟子把少年带上山，改变其位置，缺少对应事件。"}
        self.assertIsNone(_compound_coverage_defer_payload([atomic]))

    def test_audience_repair_freezes_event_content_and_evidence(self) -> None:
        _, annotation = self._fixture_annotation_for_ungrounded_speaker()
        data = annotation.to_dict()
        span = data["facts"][0]["evidence"][0]
        data["entities"].extend([
            {"entity_id": "entity-listener", "canonical_name": "听者", "kind": "person",
             "aliases": [], "evidence": [span]},
            {"entity_id": "entity-group", "canonical_name": "执行弟子", "kind": "group",
             "aliases": [], "evidence": [span]},
        ])
        data["events"] = [{
            "event_id": "event-command", "chapter_id": annotation.chapter_id, "order": 0,
            "summary": "甲向一人发布命令", "participant_ids": ["entity-b", "entity-listener"],
            "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "发布命令",
            "obstacle": "", "decision": "", "outcome_fact_ids": ["fact-a"],
            "cost_fact_ids": [], "evidence": [span], "action_type": "speech",
            "actor_id": "entity-b", "target_ids": ["entity-listener"],
            "basis_fact_ids": ["fact-a"],
        }]
        changed = ChapterAnnotation.from_dict(data)
        payload = {
            "resolved_issue_ids": ["collective-command-audience-event-command"],
            "deferred_issues": [], "merge_entities": [],
            "upsert_entities": [{"id": "new-entity-noise"}],
            "remove_fact_ids": ["fact-a"], "upsert_facts": [{"id": "new-fact-noise"}],
            "remove_event_ids": ["event-command"],
            "upsert_events": [{
                "id": "event-command", "summary": "被模型扩写", "action": "被模型改写",
                "target_ids": ["entity-group"],
                "evidence": [{"unit_id": "wrong-unit", "quote": "无关证据"}],
            }],
        }

        constrained = _constrain_audience_only_patch(payload, changed, [{
            "issue_id": "collective-command-audience-event-command",
            "category": "event_frame", "object_ids": ["event-command"],
        }])

        event = constrained["upsert_events"][0]
        self.assertEqual(event["target_ids"], ["entity-group"])
        self.assertEqual(event["participant_ids"], ["entity-b", "entity-group"])
        self.assertEqual(event["summary"], "甲向一人发布命令")
        self.assertEqual(event["action"], "发布命令")
        self.assertEqual(event["evidence"], [{"unit_id": span["unit_id"], "quote": span["quote"]}])
        self.assertEqual(constrained["upsert_facts"], [])
        self.assertEqual(constrained["upsert_entities"], [])

    def test_future_arrangement_constraint_keeps_only_existing_promise(self) -> None:
        text = "四叔说道：\u201c二哥，二嫂，我明天来接你们。\u201d"
        document = build_chapter_document(
            "Author/work-001/0001", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        unit = document.units[0]
        quote = "我明天来接你们"
        start = text.index(quote)
        span = {"chapter_id": document.chapter_id, "start": start, "end": start + len(quote),
                "quote": quote, "unit_id": unit.unit_id}
        entity_span = {"chapter_id": document.chapter_id, "start": 0, "end": len(text),
                       "quote": text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "entity-uncle", "canonical_name": "四叔", "kind": "person",
                 "aliases": [], "evidence": [entity_span]},
                {"entity_id": "entity-parents", "canonical_name": "父母", "kind": "group",
                 "aliases": ["二哥", "二嫂"], "evidence": [entity_span]},
                {"entity_id": "entity-clan", "canonical_name": "家族", "kind": "organization",
                 "aliases": [], "evidence": [entity_span]},
            ],
            "facts": [{
                "fact_id": "fact-promise", "chapter_id": document.chapter_id,
                "kind": "information", "subject_id": "entity-uncle", "predicate": "承诺",
                "value": quote, "object_id": "", "certainty": 1.0, "evidence": [span],
            }],
            "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })
        issue = {
            "issue_id": "future-arrangement-event-p0000", "category": "coverage",
            "object_ids": [f"missing:event:{unit.unit_id}"],
            "diagnosis": "未来承诺已有事实但缺少speech事件。",
        }
        proposed = {
            "resolved_issue_ids": [issue["issue_id"]], "deferred_issues": [],
            "merge_entities": [], "upsert_entities": [], "remove_fact_ids": [],
            "upsert_facts": [{"id": "fact-promise", "value": "被模型改写"}],
            "remove_event_ids": [],
            "upsert_events": [{
                "id": "new-event-promise", "target_ids": ["entity-parents"],
                "summary": "混入别的消息", "outcome_fact_ids": ["fact-promise", "fact-other"],
                "evidence": [{"unit_id": unit.unit_id, "quote": text}],
            }],
        }

        constrained = _constrain_future_arrangement_event_patch(
            proposed, document, annotation, [issue],
        )

        self.assertEqual(constrained["upsert_facts"], [])
        event = constrained["upsert_events"][0]
        self.assertEqual(event["target_ids"], ["entity-parents"])
        self.assertEqual(event["outcome_fact_ids"], ["fact-promise"])
        self.assertEqual(event["basis_fact_ids"], ["fact-promise"])
        self.assertEqual(event["evidence"], [{"unit_id": unit.unit_id, "quote": quote}])

        deterministic = _deterministic_future_arrangement_speech_patch(
            document, annotation, [issue],
        )
        self.assertIsNotNone(deterministic)
        self.assertEqual(deterministic["resolved_issue_ids"], [issue["issue_id"]])
        self.assertEqual(deterministic["upsert_events"][0]["outcome_fact_ids"], ["fact-promise"])

        invalid = dict(proposed)
        invalid["upsert_events"] = [{
            **proposed["upsert_events"][0], "target_ids": ["entity-clan"],
        }]
        self.assertIs(
            _constrain_future_arrangement_event_patch(invalid, document, annotation, [issue]),
            invalid,
        )

    def test_mixed_speech_repair_removes_only_offending_evidence(self) -> None:
        document, annotation = self._fixture_annotation_for_ungrounded_speaker()
        fact = annotation.facts[0]
        first = fact.evidence[0]
        extra = replace(first, start=max(0, first.start - 1), end=first.end, quote=document.text[max(0, first.start - 1):first.end], unit_id=first.unit_id)
        event = type(annotation).from_dict({
            **annotation.to_dict(),
            "events": [{"event_id": "event-speech", "chapter_id": annotation.chapter_id, "order": 0,
                        "summary": "甲说话", "participant_ids": [annotation.entities[0].entity_id],
                        "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "说话", "obstacle": "",
                        "decision": "", "outcome_fact_ids": [fact.fact_id], "cost_fact_ids": [],
                        "evidence": [first.to_dict(), extra.to_dict()], "action_type": "speech",
                        "actor_id": annotation.entities[0].entity_id, "target_ids": [],
                        "basis_fact_ids": [fact.fact_id]}],
        }).events[0]
        changed = replace(annotation, events=(event,))
        issue = {"issue_id": "mixed-speaker-evidence-event-speech", "category": "event_frame",
                 "object_ids": [event.event_id], "evidence_unit_ids": [extra.unit_id]}

        payload = _deterministic_mixed_speech_evidence_patch(changed, [issue])

        # Both spans share one unit in this synthetic fixture, so removing that
        # unit would erase the event; the safe deterministic path must decline.
        self.assertIsNone(payload)

    def test_unsupported_speech_event_shrinks_to_exact_physical_action(self) -> None:
        document, annotation = self._fixture_annotation_for_ungrounded_speaker()
        span = annotation.facts[0].evidence[0]
        fact = replace(
            annotation.facts[0], kind="progression", predicate="磕头", value="跪地磕头",
        )
        from pipeline.contracts import EventAtom
        event = EventAtom(
            "event-a", annotation.chapter_id, 0, "甲磕头并拍马屁", ("entity-b",),
            (), (), "磕头并拍马屁", "", "", (fact.fact_id,), (), (span,),
            "speech", "entity-b", (), (fact.fact_id,),
        )
        annotation = replace(annotation, facts=(fact,), events=(event,))
        issue = {"issue_id": "speech-evidence-event-a", "category": "event_frame",
                 "object_ids": [event.event_id], "diagnosis": "交流没有言语证据。"}

        repair = _deterministic_unsupported_speech_shrink_patch(annotation, [issue])

        self.assertIsNotNone(repair)
        self.assertEqual(repair["upsert_events"][0]["action_type"], "other")
        self.assertEqual(repair["upsert_events"][0]["action"], span.quote)

    def test_reporting_progression_repair_only_retypes_the_fact(self) -> None:
        document, annotation = self._fixture_annotation_for_ungrounded_speaker()
        fact = replace(
            annotation.facts[0], kind="progression", predicate="要求",
            value="乙一定要入选",
        )
        annotation = replace(annotation, facts=(fact,))
        issue = {
            "issue_id": "speech-fact-kind-fact-a", "category": "fact_semantics",
            "object_ids": [fact.fact_id], "diagnosis": "言语命题被存为progression。",
        }

        repair = _deterministic_reporting_fact_kind_patch(annotation, [issue])

        self.assertIsNotNone(repair)
        self.assertEqual(repair["upsert_facts"][0]["id"], fact.fact_id)
        self.assertEqual(repair["upsert_facts"][0]["kind"], "information")
        self.assertEqual(repair["upsert_events"], [])

    def test_ungrounded_speaker_fact_is_removed_without_guessing_replacement(self) -> None:
        _, annotation = self._fixture_annotation_for_ungrounded_speaker()
        issue = {"issue_id": "ungrounded-speaker-fact-a", "category": "fact_semantics",
                 "object_ids": ["fact-a"], "diagnosis": "独立引语没有说话者证据。"}

        repair = _deterministic_ungrounded_speaker_removal_patch(annotation, [issue])

        self.assertIsNotNone(repair)
        self.assertEqual(repair["remove_fact_ids"], ["fact-a"])
        self.assertEqual(repair["upsert_facts"], [])
        self.assertEqual(repair["upsert_entities"], [])

    @staticmethod
    def _fixture_annotation_for_ungrounded_speaker() -> tuple[object, ChapterAnnotation]:
        text = "“乙，你一定要入选。”"
        document = build_chapter_document(
            "Author/work-001/0001", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": unit.start, "end": unit.end,
                "quote": unit.text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "entity-b", "canonical_name": "乙", "kind": "person",
                          "aliases": [], "evidence": [span]}],
            "facts": [{"fact_id": "fact-a", "chapter_id": document.chapter_id, "kind": "goal",
                       "subject_id": "entity-b", "predicate": "希望", "value": "入选",
                       "object_id": "", "certainty": 1.0, "evidence": [span]}],
            "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })
        return document, annotation

    def test_disjoint_outcome_repair_detaches_edge_but_keeps_fact(self) -> None:
        text = "甲说：明天出发。\n\n" + ("路途漫长。" * 50) + "\n\n甲说：还有宴席。"
        document = build_chapter_document(
            "Author/work-001/0001", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        first, _, last = document.units
        first_span = {"chapter_id": document.chapter_id, "start": first.start, "end": first.end,
                      "quote": first.text, "unit_id": first.unit_id}
        last_span = {"chapter_id": document.chapter_id, "start": last.start, "end": last.end,
                     "quote": last.text, "unit_id": last.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "entity-a", "canonical_name": "甲", "kind": "person",
                          "aliases": [], "evidence": [first_span]}],
            "facts": [
                {"fact_id": "fact-near", "chapter_id": document.chapter_id, "kind": "information",
                 "subject_id": "entity-a", "predicate": "告知", "value": "明天出发", "object_id": "",
                 "certainty": 1.0, "evidence": [first_span]},
                {"fact_id": "fact-remote", "chapter_id": document.chapter_id, "kind": "information",
                 "subject_id": "entity-a", "predicate": "告知", "value": "还有宴席", "object_id": "",
                 "certainty": 1.0, "evidence": [last_span]},
            ],
            "events": [{"event_id": "event-a", "chapter_id": document.chapter_id, "order": 0,
                        "summary": "甲说明出发", "participant_ids": ["entity-a"],
                        "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "说明明天出发",
                        "obstacle": "", "decision": "", "outcome_fact_ids": ["fact-near", "fact-remote"],
                        "cost_fact_ids": [], "evidence": [first_span], "action_type": "speech",
                        "actor_id": "entity-a", "target_ids": [],
                        "basis_fact_ids": ["fact-near", "fact-remote"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })
        issue = {"issue_id": "bad-edge", "category": "event_frame", "object_ids": ["event-a"],
                 "diagnosis": "事件结果与事件证据相距过远，没有共享证据。"}

        repair = _deterministic_disjoint_event_outcome_patch(annotation, [issue])

        self.assertIsNotNone(repair)
        self.assertEqual(repair["remove_fact_ids"], [])
        self.assertEqual(repair["upsert_events"][0]["outcome_fact_ids"], ["fact-near"])
        self.assertEqual(repair["upsert_events"][0]["basis_fact_ids"], ["fact-near"])
        self.assertEqual(
            _blocking_patch_quality_issues(annotation, annotation, document, [{"object_ids": ["fact-near"]}]),
            [],
        )
        self.assertTrue(
            _blocking_patch_quality_issues(annotation, annotation, document, [{"object_ids": ["event-a"]}])
        )

    def test_near_departure_outcome_extends_event_evidence(self) -> None:
        text = "甲第一次离开村子。\n\n乙拉着甲上车，马车扬长而去。"
        document = build_chapter_document(
            "Author/work-001/0002", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        first, second = document.units
        first_span = {"chapter_id": document.chapter_id, "start": first.start, "end": first.end,
                      "quote": first.text, "unit_id": first.unit_id}
        second_span = {"chapter_id": document.chapter_id, "start": second.start, "end": second.end,
                       "quote": second.text, "unit_id": second.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "entity-a", "canonical_name": "甲", "kind": "person",
                 "aliases": [], "evidence": [first_span]},
                {"entity_id": "entity-b", "canonical_name": "乙", "kind": "person",
                 "aliases": [], "evidence": [second_span]},
            ],
            "facts": [
                {"fact_id": "fact-departure", "chapter_id": document.chapter_id,
                 "kind": "progression", "subject_id": "entity-a", "predicate": "首次离开",
                 "value": "村子", "object_id": "", "certainty": 1.0,
                 "evidence": [first_span]},
            ],
            "events": [
                {"event_id": "event-departure", "chapter_id": document.chapter_id,
                 "order": 0, "summary": "乙带甲乘车离开", "participant_ids": ["entity-a", "entity-b"],
                 "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "拉甲上车扬长而去",
                 "obstacle": "", "decision": "", "outcome_fact_ids": ["fact-departure"],
                 "cost_fact_ids": [], "evidence": [second_span], "action_type": "other",
                 "actor_id": "entity-b", "target_ids": ["entity-a"],
                 "basis_fact_ids": ["fact-departure"]},
            ],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })
        issue = {
            "issue_id": "outcome-evidence-event-departure-fact-departure",
            "category": "event_frame", "object_ids": ["event-departure"],
            "diagnosis": "事件结果与事件没有共享证据。",
        }

        repair = _deterministic_disjoint_event_outcome_patch(annotation, [issue])

        self.assertIsNotNone(repair)
        evidence = repair["upsert_events"][0]["evidence"]
        self.assertEqual({row["unit_id"] for row in evidence}, {first.unit_id, second.unit_id})

    def test_repair_source_is_issue_local_with_neighbouring_discourse_context(self) -> None:
        text = "甲说旧话。\n乙做别的事。\n甲说新话。"
        document = build_chapter_document(
            "Author/work-001/0010", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        first, _, third = document.units
        span1 = {"chapter_id": document.chapter_id, "start": first.start, "end": first.end, "quote": first.text, "unit_id": first.unit_id}
        span3 = {"chapter_id": document.chapter_id, "start": third.start, "end": third.end, "quote": third.text, "unit_id": third.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "e1", "canonical_name": "甲", "kind": "person", "aliases": [], "evidence": [span1]}],
            "facts": [{"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "information", "subject_id": "e1", "predicate": "说明", "value": "旧话", "object_id": "", "certainty": 1.0, "evidence": [span1]}],
            "events": [{"event_id": "v1", "chapter_id": document.chapter_id, "order": 0, "summary": "甲说新话", "participant_ids": ["e1"], "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "说新话", "obstacle": "", "decision": "", "outcome_fact_ids": ["f1"], "cost_fact_ids": [], "evidence": [span3], "action_type": "speech", "actor_id": "e1", "target_ids": [], "basis_fact_ids": ["f1"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })
        issue = {"object_ids": ["v1"], "evidence_unit_ids": [third.unit_id]}

        units = _repair_source_slices(document, annotation, [issue])

        self.assertEqual([item.unit_id for item in units], [item.unit_id for item in document.units])
        self.assertTrue(_issue_superseded_by_prior_patch(
            replace(annotation, events=(replace(annotation.events[0], evidence=annotation.facts[0].evidence),)), issue,
        ))

    def test_deleted_concrete_target_supersedes_duplicate_later_issue(self) -> None:
        _, annotation = self._fixture_annotation_for_ungrounded_speaker()
        without_fact = replace(annotation, facts=())

        self.assertTrue(_issue_superseded_by_prior_patch(
            without_fact,
            {"object_ids": ["fact-a"], "evidence_unit_ids": [annotation.facts[0].evidence[0].unit_id]},
        ))

    def test_derived_spatial_target_is_not_treated_as_deleted_patch_target(self) -> None:
        _, annotation = self._fixture_annotation_for_ungrounded_speaker()

        self.assertFalse(_issue_superseded_by_prior_patch(
            annotation,
            {"object_ids": [f"{annotation.chapter_id}:space-001"],
             "evidence_unit_ids": [annotation.facts[0].evidence[0].unit_id]},
        ))

    def test_missing_fact_patch_cannot_rewrite_existing_graph_objects(self) -> None:
        payload = {
            "resolved_issue_ids": ["missing-a"], "deferred_issues": [],
            "merge_entities": [{"keep_entity_id": "e1"}],
            "upsert_entities": [{"id": "e1"}, {"id": "new-entity-witness"}],
            "remove_fact_ids": ["fact-old"],
            "upsert_facts": [{"id": "fact-old"}, {"id": "new-fact-trigger"}],
            "remove_event_ids": ["event-old"], "upsert_events": [{"id": "event-old"}],
        }

        constrained = _constrain_missing_fact_patch(payload, [{
            "issue_id": "missing-a", "object_ids": ["missing:fact:chapter:p0001"],
        }])

        self.assertEqual(constrained["merge_entities"], [])
        self.assertEqual(constrained["upsert_entities"], [{"id": "new-entity-witness"}])
        self.assertEqual(constrained["remove_fact_ids"], [])
        self.assertEqual(constrained["upsert_facts"], [{"id": "new-fact-trigger"}])
        self.assertEqual(constrained["remove_event_ids"], [])
        self.assertEqual(constrained["upsert_events"], [])

    def test_reaction_causal_patch_freezes_downstream_event_except_new_trigger(self) -> None:
        _, annotation = self._fixture_annotation_for_ungrounded_speaker()
        fact = annotation.facts[0]
        event = {
            "event_id": "event-reaction", "chapter_id": annotation.chapter_id, "order": 0,
            "summary": "乙回应", "participant_ids": ["entity-b"], "trigger_fact_ids": [],
            "precondition_fact_ids": [], "action": "回应", "obstacle": "", "decision": "",
            "outcome_fact_ids": [fact.fact_id], "cost_fact_ids": [],
            "evidence": [fact.evidence[0].to_dict()], "action_type": "speech",
            "actor_id": "entity-b", "target_ids": [], "basis_fact_ids": [fact.fact_id],
        }
        annotation = replace(annotation, events=(type(annotation).from_dict({
            **annotation.to_dict(), "events": [event],
        }).events[0],))
        payload = {
            "resolved_issue_ids": ["reaction-a"], "deferred_issues": [],
            "merge_entities": [], "upsert_entities": [], "remove_fact_ids": [fact.fact_id],
            "upsert_facts": [{"id": "new-fact-predecessor"}], "remove_event_ids": ["event-reaction"],
            "upsert_events": [
                {"id": "new-event-predecessor"},
                {"id": "event-reaction", "summary": "被模型改坏", "trigger_fact_ids": ["new-fact-predecessor"]},
                {"id": "event-unrelated"},
            ],
        }
        issue = {
            "issue_id": "reaction-a", "category": "coverage",
            "object_ids": ["missing:event:chapter:p0001", "event-reaction"],
            "diagnosis": "反应缺少前置事件，须挂入 trigger_fact_ids。",
        }

        constrained = _constrain_reaction_causal_patch(payload, annotation, [issue])

        self.assertEqual(constrained["remove_fact_ids"], [])
        self.assertEqual(constrained["remove_event_ids"], [])
        self.assertEqual([row["id"] for row in constrained["upsert_events"]], [
            "new-event-predecessor", "event-reaction",
        ])
        frozen = constrained["upsert_events"][1]
        self.assertEqual(frozen["summary"], "乙回应")
        self.assertEqual(frozen["trigger_fact_ids"], ["new-fact-predecessor"])

    def test_checkpoint_does_not_normalize_untouched_later_chapters(self) -> None:
        # The concrete regression is exercised by repair_semantic_audit's
        # integration tests and real cycle. Keep a source assertion here so a
        # future refactor cannot silently move all-chapter normalization back
        # into every in-progress checkpoint.
        import inspect
        from pipeline.semantic_repair import repair_semantic_audit

        source = inspect.getsource(repair_semantic_audit)

        self.assertIn("if completed:", source)
        self.assertIn("untouched chapters must remain", source)

    def test_independently_reviewed_warning_is_repair_target_regardless_of_initial_confidence(self) -> None:
        self.assertTrue(_is_confirmed_repair_target({
            "category": "coverage", "severity": "warning", "confidence": 0.7,
        }))
        self.assertFalse(_is_confirmed_repair_target({
            "category": "coverage", "severity": "info", "confidence": 0.99,
        }))
        self.assertFalse(_is_confirmed_repair_target({
            "category": "audit_contract", "severity": "error", "confidence": 1.0,
        }))

    def test_repair_prompt_encodes_in_progress_movement_without_fake_arrival(self) -> None:
        text = "甲朝石阶走去。"
        document = build_chapter_document(
            "Author/work-001/0001", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        unit = document.units[0]
        prompt = _repair_prompt(
            document,
            (AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text),),
            {"entities": [], "facts": [], "events": []},
            [{
                "issue_id": "issue-move", "category": "event_frame", "severity": "warning",
                "object_ids": ["Author/work-001/0001:event-001"],
                "evidence_unit_ids": [unit.unit_id],
                "diagnosis": "人物朝石阶走去却被标为decision。", "proposed_action": "改为movement",
                "confidence": 0.8,
            }],
        )
        self.assertIn("人物—趋近—石阶（尚未表明到达）", prompt)
        self.assertIn("不得仍用 progression 事实凑 movement 契约", prompt)
        self.assertIn("不能因为全章已有“山峰”“院子”等 location 实体", prompt)
        self.assertIn("没有“石阶”实体时必须在同一补丁新建", prompt)
        self.assertIn("confidence` 仅保留初审检测器当时的分数", prompt)
        self.assertIn("绝不得以“置信度低于0.8”为理由延期", prompt)
        self.assertIn("missing:<kind>:<unit_id>` 只是缺失对象的锚点", prompt)
        self.assertIn("“如坠冰窟”“心如刀绞”“如遭雷击”", prompt)
        self.assertIn("不得外推为抑郁、崩溃等未出现的长期状态", prompt)
        self.assertIn("实际文字区间重叠", prompt)
        self.assertIn("1 至 3 项", prompt)
        self.assertIn("最早完成、已经足以解释下游惊讶/关注/态度变化", prompt)
        self.assertIn("other + progression", prompt)
        self.assertIn("不必为“五丈处、数步之后”等相对量强造 location 实体", prompt)

    def test_resume_drops_retryable_failures_and_recomputes_counts(self) -> None:
        rows, resolved, deferred = _resumed_terminal_state({"batches": [
            {"status": "candidate_applied", "application": {"resolved_issue_ids": ["i1"], "deferred_issues": []}},
            {"status": "service_blocked", "issue_ids": ["i2"]},
            {"status": "contract_failed", "issue_ids": ["i3"]},
            {"status": "deferred_by_model", "application": {"resolved_issue_ids": [], "deferred_issues": [{"issue_id": "i4"}]}},
        ]})
        self.assertEqual([item["status"] for item in rows], ["candidate_applied", "deferred_by_model"])
        self.assertEqual((resolved, deferred), (1, 1))

    def test_explicit_quote_materializes_event_from_existing_fact(self) -> None:
        text = "乙的父亲盯着乙，乙说道：“爹，我已经入选。”"
        document = build_chapter_document(
            "Author/work-001/0001", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        unit = document.units[0]
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id,
            "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "entity-son", "canonical_name": "乙", "kind": "person", "aliases": [],
                 "evidence": [{"chapter_id": document.chapter_id, "start": 0, "end": 1, "quote": "乙", "unit_id": unit.unit_id}]},
                {"entity_id": "entity-father", "canonical_name": "乙的父亲", "kind": "person", "aliases": [],
                 "evidence": [{"chapter_id": document.chapter_id, "start": 0, "end": 4, "quote": "乙的父亲", "unit_id": unit.unit_id}]},
            ],
            "facts": [{
                "fact_id": "fact-speech", "chapter_id": document.chapter_id, "kind": "information",
                "subject_id": "entity-son", "predicate": "声称", "value": "已经入选", "object_id": "", "certainty": 1.0,
                "evidence": [{"chapter_id": document.chapter_id, "start": text.index("我已经入选"),
                              "end": text.index("我已经入选") + len("我已经入选"), "quote": "我已经入选", "unit_id": unit.unit_id}],
            }],
            "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })
        patch = _deterministic_missing_speech_patch(document, annotation, [{
            "issue_id": "missing-speech", "category": "coverage",
            "object_ids": [f"missing:event:{unit.unit_id}"], "diagnosis": "明确对白speech事件缺失。",
        }])
        self.assertIsNotNone(patch)
        self.assertEqual(patch["upsert_facts"], [])
        self.assertEqual(patch["upsert_events"][0]["target_ids"], ["entity-father"])
        self.assertEqual(patch["upsert_events"][0]["outcome_fact_ids"], ["fact-speech"])

    def test_deterministic_duplicate_removal_keeps_referenced_sibling(self) -> None:
        text = "甲讥讽乙靠送礼取得资格。"
        document = build_chapter_document(
            "Author/work-001/0001", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        unit = document.units[0]
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id,
            "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "entity-a", "canonical_name": "甲", "kind": "person", "aliases": [],
                 "evidence": [{"chapter_id": document.chapter_id, "start": 0, "end": 1, "quote": "甲", "unit_id": unit.unit_id}]},
                {"entity_id": "entity-b", "canonical_name": "乙", "kind": "person", "aliases": [],
                 "evidence": [{"chapter_id": document.chapter_id, "start": 3, "end": 4, "quote": "乙", "unit_id": unit.unit_id}]},
            ],
            "facts": [
                {"fact_id": "fact-keep", "chapter_id": document.chapter_id, "kind": "information",
                 "subject_id": "entity-a", "predicate": "讥讽", "value": "乙送礼取得资格", "object_id": "entity-b", "certainty": 1.0,
                 "evidence": [{"chapter_id": document.chapter_id, "start": 0, "end": len(text), "quote": text, "unit_id": unit.unit_id}]},
                {"fact_id": "fact-drop", "chapter_id": document.chapter_id, "kind": "information",
                 "subject_id": "entity-a", "predicate": "讥讽", "value": "乙靠送礼取得资格", "object_id": "entity-b", "certainty": 1.0,
                 "evidence": [{"chapter_id": document.chapter_id, "start": 0, "end": len(text), "quote": text, "unit_id": unit.unit_id}]},
            ],
            "events": [{
                "event_id": "event-1", "chapter_id": document.chapter_id, "order": 0,
                "summary": "甲讥讽乙", "participant_ids": ["entity-a", "entity-b"],
                "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "讥讽乙",
                "obstacle": "", "decision": "", "outcome_fact_ids": ["fact-keep"], "cost_fact_ids": [],
                "evidence": [{"chapter_id": document.chapter_id, "start": 0, "end": len(text), "quote": text, "unit_id": unit.unit_id}],
                "action_type": "speech", "actor_id": "entity-a", "target_ids": ["entity-b"], "basis_fact_ids": ["fact-keep"],
            }],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })
        patch = _deterministic_safe_fact_removal_patch(annotation, [{
            "issue_id": "duplicate-fact-drop", "category": "fact_semantics",
            "object_ids": ["fact-drop"], "diagnosis": "两条事实语义重复。",
        }])
        self.assertIsNotNone(patch)
        self.assertEqual(patch["remove_fact_ids"], ["fact-drop"])
        self.assertEqual(patch["upsert_events"], [])

    def test_unsupported_relationship_removal_requires_grounded_inverse(self) -> None:
        text = "张三与众人同席。\n后来李四被明确称作张三的父亲。"
        document = build_chapter_document(
            "Author/work-001/0001", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        first, second = document.units
        span1 = {"chapter_id": document.chapter_id, "start": first.start, "end": first.end,
                 "quote": first.text, "unit_id": first.unit_id}
        span2 = {"chapter_id": document.chapter_id, "start": second.start, "end": second.end,
                 "quote": second.text, "unit_id": second.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "entity-a", "canonical_name": "张三", "kind": "person", "aliases": [], "evidence": [span1]},
                {"entity_id": "entity-b", "canonical_name": "李四", "kind": "person", "aliases": [], "evidence": [span2]},
            ],
            "facts": [
                {"fact_id": "fact-bad", "chapter_id": document.chapter_id, "kind": "relationship",
                 "subject_id": "entity-a", "predicate": "父亲", "value": "李四", "object_id": "entity-b",
                 "certainty": 1.0, "evidence": [span1]},
                {"fact_id": "fact-keep", "chapter_id": document.chapter_id, "kind": "relationship",
                 "subject_id": "entity-b", "predicate": "是", "value": "张三的父亲", "object_id": "entity-a",
                 "certainty": 1.0, "evidence": [span2]},
            ],
            "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        patch = _deterministic_unsupported_relationship_removal_patch(annotation, [{
            "issue_id": "unsupported", "category": "fact_semantics", "object_ids": ["fact-bad"],
            "diagnosis": "原证据未明确这项关系，证据不足。",
        }])

        self.assertIsNotNone(patch)
        self.assertEqual(patch["remove_fact_ids"], ["fact-bad"])
        self.assertEqual(patch["upsert_facts"], [])

    def test_deterministic_duplicate_removal_redirects_only_core_event_fact(self) -> None:
        text = "甲确认三人是推荐人选。"
        document = build_chapter_document(
            "Author/work-001/0002", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": 0, "end": len(text), "quote": text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "entity-a", "canonical_name": "甲", "kind": "person", "aliases": [], "evidence": [span]}],
            "facts": [
                {"fact_id": "fact-keep", "chapter_id": document.chapter_id, "kind": "knowledge", "subject_id": "entity-a", "predicate": "说明", "value": "三人是推荐人选", "object_id": "", "certainty": 1.0, "evidence": [span]},
                {"fact_id": "fact-drop", "chapter_id": document.chapter_id, "kind": "knowledge", "subject_id": "entity-a", "predicate": "确认", "value": "三人是推荐人选", "object_id": "", "certainty": 1.0, "evidence": [span]},
            ],
            "events": [{"event_id": "event-1", "chapter_id": document.chapter_id, "order": 0, "summary": "甲确认人选", "participant_ids": ["entity-a"], "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "确认三人是推荐人选", "obstacle": "", "decision": "", "outcome_fact_ids": ["fact-drop"], "cost_fact_ids": [], "evidence": [span], "action_type": "speech", "actor_id": "entity-a", "target_ids": [], "basis_fact_ids": ["fact-drop"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        patch = _deterministic_safe_fact_removal_patch(annotation, [{
            "issue_id": "duplicate-fact-drop", "category": "fact_semantics",
            "object_ids": ["fact-drop"], "diagnosis": "两条事实语义重复。",
        }])

        self.assertIsNotNone(patch)
        self.assertEqual(patch["upsert_events"][0]["basis_fact_ids"], ["fact-keep"])
        self.assertEqual(patch["upsert_events"][0]["outcome_fact_ids"], ["fact-keep"])

    def test_repair_normalization_collapses_duplicate_result_producers_before_gate(self) -> None:
        text = "甲确认三人是推荐人选。"
        document = build_chapter_document(
            "Author/work-001/0003", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": 0, "end": len(text), "quote": text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "entity-a", "canonical_name": "甲", "kind": "person", "aliases": [], "evidence": [span]}],
            "facts": [{"fact_id": "fact-keep", "chapter_id": document.chapter_id, "kind": "knowledge", "subject_id": "entity-a", "predicate": "说明", "value": "三人是推荐人选", "object_id": "", "certainty": 1.0, "evidence": [span]}],
            "events": [
                {"event_id": "event-1", "chapter_id": document.chapter_id, "order": 0, "summary": "甲说明人选", "participant_ids": ["entity-a"], "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "说明三人是推荐人选", "obstacle": "", "decision": "", "outcome_fact_ids": ["fact-keep"], "cost_fact_ids": [], "evidence": [span], "action_type": "speech", "actor_id": "entity-a", "target_ids": [], "basis_fact_ids": ["fact-keep"]},
                {"event_id": "event-2", "chapter_id": document.chapter_id, "order": 1, "summary": "甲确认人选", "participant_ids": ["entity-a"], "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "确认三人是推荐人选", "obstacle": "", "decision": "", "outcome_fact_ids": ["fact-keep"], "cost_fact_ids": [], "evidence": [span], "action_type": "speech", "actor_id": "entity-a", "target_ids": [], "basis_fact_ids": ["fact-keep"]},
            ],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        normalized = _normalize_repaired_annotation(document, annotation)

        self.assertEqual(len(normalized.events), 1)

    def test_patch_review_requires_exact_issue_coverage_and_no_pass_defects(self) -> None:
        payload = {
            "结论": "通过",
            "已核问题ID": ["i1"],
            "检查结果": {
                "原问题成立": True,
                "人物角色正确": True,
                "直接修复目标对象": True,
                "单一核心动作": True,
                "行动类型与结果一致": True,
                "无原文外新增": True,
            },
            "缺陷": [],
        }
        self.assertEqual(_parse_patch_review(payload, ("i1",))["结论"], "通过")
        with self.assertRaisesRegex(ValueError, "没有逐一覆盖"):
            _parse_patch_review(payload, ("i1", "i2"))
        with self.assertRaisesRegex(ValueError, "通过时缺陷必须为空"):
            _parse_patch_review({
                **payload,
                "缺陷": [{
                    "类型": "混合动作", "对象ID": ["event-1"],
                    "说明": "事件同时表达到达和询问。", "置信度": 0.9,
                }],
            }, ("i1",))

    def test_repair_groups_only_identity_resolution_transitively(self) -> None:
        identity = [
            {"issue_id": "i1", "category": "entity_identity"},
            {"issue_id": "i2", "category": "entity_identity"},
        ]
        self.assertEqual(_repair_issue_groups(identity), [identity])

        mixed = [
            {"issue_id": "i1", "category": "entity_identity"},
            {"issue_id": "e1", "category": "event_frame"},
            {"issue_id": "f1", "category": "fact_semantics"},
        ]
        self.assertEqual(_repair_issue_groups(mixed), [[mixed[0]], [mixed[1]], [mixed[2]]])

    def test_selected_chapters_preserve_baseline_audit_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.jsonl"
            path.write_text("\n".join(json.dumps({"chapter_id": item}) for item in ("c1", "c2", "c3")), encoding="utf-8")
            self.assertEqual(_selected_chapter_numbers(path, ["c3", "c1"]), (3, 1))
            with self.assertRaisesRegex(ValueError, "missing baseline-audited chapters"):
                _selected_chapter_numbers(path, ["c4"])

    def test_reusable_patch_fingerprint_changes_when_same_issue_id_changes_meaning(self) -> None:
        base = {
            "issue_id": "issue-001", "category": "fact_semantics", "severity": "error",
            "object_ids": ["fact-001"], "evidence_unit_ids": ["chapter:p0001"],
            "diagnosis": "主体错误", "proposed_action": "修正事实语义", "confidence": 0.95,
        }
        self.assertNotEqual(
            _issue_fingerprint([base]),
            _issue_fingerprint([{**base, "object_ids": ["fact-002"], "diagnosis": "客体错误"}]),
        )

    def test_repair_outputs_are_isolated_by_candidate_label(self) -> None:
        folder = Path("corpus")
        first = _repair_output_paths(folder, 5, "round-1")
        second = _repair_output_paths(folder, 5, "round-2")
        self.assertNotEqual(first, second)
        with self.assertRaisesRegex(ValueError, "output_label"):
            _repair_output_paths(folder, 5, "../unsafe")

    def test_annotation_output_label_is_validated_before_file_access(self) -> None:
        with self.assertRaisesRegex(ValueError, "output_label"):
            annotate_work("Author", "work-001", limit=1, output_label="../unsafe")

    def test_repair_prompt_teaches_minimal_identity_and_viewpoint_fixes(self) -> None:
        text = "甲讥讽道：乙只是药童。"
        document = build_chapter_document(
            "Author/work-001/0001", hashlib.sha256(text.encode("utf-8")).hexdigest(), text, "第一章",
        )
        unit = document.units[0]
        source = (AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text),)
        prompt = _repair_prompt(document, source, {"entities": [], "facts": [], "events": []}, [{
            "issue_id": "i1", "category": "fact_semantics", "object_ids": ["fact-1"],
        }])

        self.assertIn("remove_fact_ids`、`upsert_facts`、`remove_event_ids`、`upsert_events` 必须全部是空数组", prompt)
        self.assertIn("最小修复就是把目标事实ID放入 `remove_fact_ids`", prompt)
        self.assertIn("不能只在事实里引用一个未声明的新ID", prompt)
        self.assertIn("新建实体前必须先查", prompt)
        self.assertIn("禁止插入 `...`", prompt)
        self.assertIn("一条正确结果", prompt)
        self.assertIn("甲讥讽道", prompt)
        self.assertIn("不能写成被谈论者已经成立的 identity", prompt)
        self.assertIn("微动作”不是按字数判断", prompt)


if __name__ == "__main__":
    unittest.main()
