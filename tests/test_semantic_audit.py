from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace
from unittest.mock import patch

from pipeline.annotate import AnnotationSourceSlice, split_annotation_units
from pipeline.contracts import ChapterAnnotation, SemanticAuditBatch, SemanticAuditIssue, SourceSpan, build_chapter_document
from pipeline.semantic_audit import (
    _audit_output_paths,
    _attributed_speaker_ids,
    _audit_prompt,
    _annotation_packet,
    _checkpoint_path,
    _downgrade_external_alias_claims,
    _deterministic_duplicate_fact_batch,
    _deterministic_event_outcome_evidence_batch,
    _deterministic_event_frame_batch,
    _deterministic_compound_core_action_batch,
    _deterministic_strong_action_coverage_batch,
    _deterministic_fact_frame_batch,
    _deterministic_incomplete_action_fact_batch,
    _deterministic_explicit_future_arrangement_batch,
    _deterministic_explicit_goal_batch,
    _deterministic_speech_turn_boundary_batch,
    _deterministic_ungrounded_quote_subject_batch,
    _deterministic_public_rule_audience_batch,
    _deterministic_unlinked_reaction_batch,
    _entity_candidate_pairs,
    _entity_resolution_consensus,
    _entity_resolution_prompt,
    _parse_audit_payload,
    _parse_issue_review_payload,
    _issue_review_prompt,
    _parse_entity_resolution_payload,
    _promote_distinct_alias_collisions,
    _promote_third_party_alias_equivalences,
    _object_evidence_index,
    _require_auditable_annotations,
    _resolve_annotation_artifact,
    _select_chapters,
    _suppress_field_contradictory_findings,
    _suppress_deterministic_event_model_findings,
    _suppress_deterministic_fact_model_findings,
    _suppress_duplicate_model_findings,
    coverage_action_atoms,
)


class SemanticAuditTests(unittest.TestCase):
    def test_completed_treatment_is_not_hidden_by_a_movement_event(self) -> None:
        text = "弟子把众少年带到山上，统一喂食药物。"
        document = build_chapter_document(
            "Author/work-001/0001", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        unit = document.units[0]
        whole = SourceSpan(document.chapter_id, unit.start, unit.end, unit.text, unit.unit_id)
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "actor", "canonical_name": "弟子", "kind": "person", "aliases": [], "evidence": [whole.to_dict()]},
                {"entity_id": "target", "canonical_name": "众少年", "kind": "group", "aliases": [], "evidence": [whole.to_dict()]},
                {"entity_id": "place", "canonical_name": "山上", "kind": "location", "aliases": [], "evidence": [whole.to_dict()]},
            ],
            "facts": [{"fact_id": "location", "chapter_id": document.chapter_id, "kind": "location",
                       "subject_id": "actor", "predicate": "到达", "value": "山上", "object_id": "place",
                       "certainty": 1.0, "evidence": [whole.to_dict()]}],
            "events": [{"event_id": "carry", "chapter_id": document.chapter_id, "order": 0,
                        "summary": "弟子带少年上山", "participant_ids": ["actor", "target"],
                        "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "带到山上", "obstacle": "",
                        "decision": "", "outcome_fact_ids": ["location"], "cost_fact_ids": [],
                        "evidence": [whole.to_dict()], "action_type": "movement", "actor_id": "actor",
                        "target_ids": ["target"], "basis_fact_ids": ["location"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        batch = _deterministic_strong_action_coverage_batch(document, annotation)

        self.assertIsNotNone(batch)
        self.assertEqual(len(batch.issues), 1)
        self.assertIn("给药", batch.issues[0].diagnosis)

    def test_future_treatment_plan_is_not_a_completed_action(self) -> None:
        text = "弟子准备给少年喂药。"
        document = build_chapter_document(
            "Author/work-001/0001", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [], "facts": [], "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        self.assertIsNone(_deterministic_strong_action_coverage_batch(document, annotation))

    def test_delivered_document_is_not_person_transport(self) -> None:
        text = "请帖送到后，宾客陆续前来。"
        document = build_chapter_document(
            "Author/work-001/0001", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [], "facts": [], "events": [], "scenes": [], "state_changes": [],
            "schema_version": "2.2",
        })

        self.assertIsNone(_deterministic_strong_action_coverage_batch(document, annotation))
    def test_compound_movement_and_treatment_event_is_rejected(self) -> None:
        _, annotation = self._fixture()
        event = replace(
            annotation.events[0],
            summary="弟子带少年上山喂药",
            action="带少年上山",
            action_type="other",
        )

        batch = _deterministic_compound_core_action_batch(replace(
            annotation, events=(event,),
        ))

        self.assertIsNotNone(batch)
        self.assertEqual(batch.issues[0].object_ids, (event.event_id,))

    def test_single_family_movement_chain_remains_atomic(self) -> None:
        _, annotation = self._fixture()
        event = replace(
            annotation.events[0], summary="甲离开后到达村口", action="离开并到达村口",
        )

        self.assertIsNone(_deterministic_compound_core_action_batch(replace(
            annotation, events=(event,),
        )))

    def test_incomplete_action_fact_reports_completed_progression(self) -> None:
        text = "甲正要鞠躬，乙连忙拦住。"
        document = build_chapter_document(
            "Author/work-001/0001", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": unit.start, "end": unit.end,
                "quote": unit.text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "entity-a", "canonical_name": "甲", "kind": "person",
                          "aliases": [], "evidence": [span]}],
            "facts": [{"fact_id": "fact-a", "chapter_id": document.chapter_id, "kind": "progression",
                       "subject_id": "entity-a", "predicate": "鞠躬", "value": "向乙鞠躬",
                       "object_id": "", "certainty": 1.0, "evidence": [span]}],
            "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        batch = _deterministic_incomplete_action_fact_batch(document, annotation)

        self.assertIsNotNone(batch)
        self.assertEqual(batch.issues[0].object_ids, ("fact-a",))

    def test_speaker_resolution_rejects_listener_and_observed_person(self) -> None:
        text = "甲含笑望着乙，又对丙说道：“出发。”"
        document = build_chapter_document(
            "Author/work-001/0001", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        unit = document.units[0]
        def span(name: str):
            start = text.index(name)
            return {"chapter_id": document.chapter_id, "start": start, "end": start + len(name),
                    "quote": name, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "entity-a", "canonical_name": "甲", "kind": "person", "aliases": [], "evidence": [span("甲")]},
                {"entity_id": "entity-b", "canonical_name": "乙", "kind": "person", "aliases": [], "evidence": [span("乙")]},
                {"entity_id": "entity-c", "canonical_name": "丙", "kind": "person", "aliases": ["乙丙"], "evidence": [span("丙")]},
            ],
            "facts": [], "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        speakers = _attributed_speaker_ids(document, annotation, unit.unit_id)

        self.assertEqual(speakers, ("entity-a",))

    def test_self_talk_keeps_clause_subject_not_observed_person(self) -> None:
        text = "中年人注视少年双眼，手在少年头上一按，随后摇头自语道：“资质普通。”"
        document = build_chapter_document(
            "Author/work-001/0001", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        unit = document.units[0]
        def span(name: str):
            start = text.index(name)
            return {"chapter_id": document.chapter_id, "start": start, "end": start + len(name),
                    "quote": name, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "observer", "canonical_name": "中年人", "kind": "person", "aliases": [], "evidence": [span("中年人")]},
                {"entity_id": "observed", "canonical_name": "少年", "kind": "person", "aliases": [], "evidence": [span("少年")]},
            ],
            "facts": [], "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        self.assertEqual(_attributed_speaker_ids(document, annotation, unit.unit_id), ("observer",))

    def test_speech_turn_boundary_detects_evidence_from_another_speaker(self) -> None:
        text = "甲问道：“他们是谁？”\n\n乙答道：“他们是弟子。”"
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
                {"entity_id": "entity-a", "canonical_name": "甲", "kind": "person", "aliases": [], "evidence": [span1]},
                {"entity_id": "entity-b", "canonical_name": "乙", "kind": "person", "aliases": [], "evidence": [span2]},
            ],
            "facts": [{"fact_id": "fact-a", "chapter_id": document.chapter_id, "kind": "information",
                       "subject_id": "entity-a", "predicate": "询问", "value": "他们是谁",
                       "object_id": "", "certainty": 1.0, "evidence": [span1]}],
            "events": [{"event_id": "event-a", "chapter_id": document.chapter_id, "order": 0,
                        "summary": "甲询问身份", "participant_ids": ["entity-a", "entity-b"],
                        "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "询问他们是谁",
                        "obstacle": "", "decision": "", "outcome_fact_ids": ["fact-a"],
                        "cost_fact_ids": [], "evidence": [span1, span2], "action_type": "speech",
                        "actor_id": "entity-a", "target_ids": ["entity-b"], "basis_fact_ids": ["fact-a"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        batch = _deterministic_speech_turn_boundary_batch(document, annotation)

        self.assertIsNotNone(batch)
        self.assertEqual(batch.issues[0].evidence_unit_ids, (second.unit_id,))

    def test_explicit_future_arrangement_is_not_covered_by_background_fact(self) -> None:
        text = "甲说道：“宴席三天后举行，我明日来接你们。”"
        document = build_chapter_document(
            "Author/work-001/0001", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": unit.start, "end": unit.end,
                "quote": unit.text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "entity-a", "canonical_name": "甲", "kind": "person",
                          "aliases": [], "evidence": [span]}],
            "facts": [{"fact_id": "fact-a", "chapter_id": document.chapter_id, "kind": "information",
                       "subject_id": "entity-a", "predicate": "告知", "value": "宴席三天后举行",
                       "object_id": "", "certainty": 1.0, "evidence": [span]}],
            "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        batch = _deterministic_explicit_future_arrangement_batch(document, annotation)

        self.assertIsNotNone(batch)
        self.assertEqual(batch.issues[0].object_ids, (f"missing:fact:{unit.unit_id}",))

    def test_reporting_listener_object_is_reported_when_absent_from_proposition(self) -> None:
        _, annotation = self._fixture()
        speaker, listener = annotation.entities[:2]
        proposition_span = replace(annotation.facts[0].evidence[0], quote="令牌")
        fact = replace(
            annotation.facts[0], kind="information", subject_id=speaker.entity_id,
            predicate="回答", value="三人是本族弟子", object_id=listener.entity_id,
            evidence=(proposition_span,),
        )
        event = replace(
            annotation.events[0], action_type="speech", actor_id=listener.entity_id,
            participant_ids=(listener.entity_id, speaker.entity_id), target_ids=(speaker.entity_id,),
            basis_fact_ids=(fact.fact_id,), outcome_fact_ids=(fact.fact_id,),
            evidence=(proposition_span,),
        )
        changed = replace(annotation, facts=(fact,), events=(event,))

        batch = _deterministic_fact_frame_batch(changed)

        self.assertIsNotNone(batch)
        self.assertEqual(batch.issues[0].issue_id, f"speech-listener-object-{fact.fact_id.rsplit(':', 1)[-1]}")

    def test_reporting_listener_object_detects_same_speech_target(self) -> None:
        _, annotation = self._fixture()
        speaker, listener = annotation.entities[:2]
        proposition_span = replace(annotation.facts[0].evidence[0], quote="令牌")
        fact = replace(
            annotation.facts[0], kind="information", subject_id=speaker.entity_id,
            predicate="报告", value="共三人合格", object_id=listener.entity_id,
            evidence=(proposition_span,),
        )
        event = replace(
            annotation.events[0], action_type="speech", actor_id=speaker.entity_id,
            participant_ids=(speaker.entity_id, listener.entity_id), target_ids=(listener.entity_id,),
            basis_fact_ids=(fact.fact_id,), outcome_fact_ids=(fact.fact_id,), evidence=(proposition_span,),
        )

        batch = _deterministic_fact_frame_batch(replace(annotation, facts=(fact,), events=(event,)))

        self.assertIsNotNone(batch)
        self.assertTrue(batch.issues[0].issue_id.startswith("speech-listener-object-"))

    def test_requirement_object_is_the_affected_person_not_a_false_listener_edge(self) -> None:
        _, annotation = self._fixture()
        speaker, affected = annotation.entities[:2]
        proposition_span = replace(annotation.facts[0].evidence[0], quote="你要按时出发")
        fact = replace(
            annotation.facts[0], kind="information", subject_id=speaker.entity_id,
            predicate="要求", value="按时出发", object_id=affected.entity_id,
            evidence=(proposition_span,),
        )
        event = replace(
            annotation.events[0], action_type="speech", actor_id=speaker.entity_id,
            participant_ids=(speaker.entity_id, affected.entity_id), target_ids=(affected.entity_id,),
            basis_fact_ids=(fact.fact_id,), outcome_fact_ids=(fact.fact_id,), evidence=(proposition_span,),
        )

        batch = _deterministic_fact_frame_batch(replace(annotation, facts=(fact,), events=(event,)))

        self.assertIsNone(batch)

    def test_information_transfer_recipient_cannot_disappear_from_proposition(self) -> None:
        document, annotation = self._fixture()
        giver, recipient = annotation.entities[:2]
        unit = document.units[0]
        quote = f"{giver.canonical_name}把名额让给{recipient.canonical_name}"
        span = SourceSpan(document.chapter_id, unit.start, unit.start + len(quote), quote, unit.unit_id)
        fact = replace(
            annotation.facts[0], kind="knowledge", subject_id=giver.entity_id,
            predicate="得知", value=f"{giver.canonical_name}让名额", object_id="",
            evidence=(span,),
        )

        batch = _deterministic_fact_frame_batch(replace(annotation, facts=(fact,)))

        self.assertIsNotNone(batch)
        self.assertTrue(any(issue.issue_id.startswith("speech-missing-recipient-") for issue in batch.issues))

    def test_unresolved_who_question_cannot_bind_a_specific_object(self) -> None:
        document, annotation = self._fixture()
        speaker, candidate = annotation.entities[:2]
        unit = document.units[0]
        quote = "到时候看看谁会失败"
        span = SourceSpan(document.chapter_id, unit.start, unit.start + len(quote), quote, unit.unit_id)
        fact = replace(
            annotation.facts[0], kind="information", subject_id=speaker.entity_id,
            predicate="比较", value="到时候看谁会失败", object_id=candidate.entity_id,
            evidence=(span,),
        )

        batch = _deterministic_fact_frame_batch(replace(annotation, facts=(fact,)))

        self.assertIsNotNone(batch)
        self.assertTrue(any(issue.issue_id.startswith("unresolved-object-") for issue in batch.issues))

    def test_public_rule_to_crowd_cannot_name_only_one_listener(self) -> None:
        text = "甲和十名少年站在房前。\n\n青年说道：“能走进房间者合格。”"
        document = build_chapter_document(
            "Author/work-001/0001", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        first, second = document.units
        def span(unit):
            return {"chapter_id": document.chapter_id, "start": unit.start, "end": unit.end,
                    "quote": unit.text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "speaker", "canonical_name": "青年", "kind": "person", "aliases": [], "evidence": [span(second)]},
                {"entity_id": "one", "canonical_name": "甲", "kind": "person", "aliases": [], "evidence": [span(first)]},
                {"entity_id": "crowd", "canonical_name": "十名少年", "kind": "group", "aliases": [], "evidence": [span(first)]},
            ],
            "facts": [{"fact_id": "rule", "chapter_id": document.chapter_id, "kind": "rule",
                       "subject_id": "speaker", "predicate": "宣布", "value": "能走进房间者合格",
                       "object_id": "", "certainty": 1.0, "evidence": [span(second)]}],
            "events": [{"event_id": "event-rule", "chapter_id": document.chapter_id, "order": 0,
                        "summary": "宣布规则", "participant_ids": ["speaker", "one"],
                        "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "宣布规则",
                        "obstacle": "", "decision": "", "outcome_fact_ids": ["rule"], "cost_fact_ids": [],
                        "evidence": [span(second)], "action_type": "speech", "actor_id": "speaker",
                        "target_ids": ["one"], "basis_fact_ids": ["rule"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        batch = _deterministic_public_rule_audience_batch(document, annotation)

        self.assertIsNotNone(batch)
        self.assertEqual(batch.issues[0].object_ids, ("event-rule",))

    def test_public_rule_with_one_group_target_is_already_complete(self) -> None:
        text = "十名少年站在房前。\n\n青年说道：“能走进房间者合格。”"
        document = build_chapter_document(
            "Author/work-001/0002", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        first, second = document.units
        def span(unit):
            return {"chapter_id": document.chapter_id, "start": unit.start, "end": unit.end,
                    "quote": unit.text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "speaker", "canonical_name": "青年", "kind": "person", "aliases": [], "evidence": [span(second)]},
                {"entity_id": "crowd", "canonical_name": "十名少年", "kind": "group", "aliases": [], "evidence": [span(first)]},
            ],
            "facts": [{"fact_id": "rule", "chapter_id": document.chapter_id, "kind": "rule",
                       "subject_id": "speaker", "predicate": "宣布", "value": "能走进房间者合格",
                       "object_id": "", "certainty": 1.0, "evidence": [span(second)]}],
            "events": [{"event_id": "event-rule", "chapter_id": document.chapter_id, "order": 0,
                        "summary": "宣布规则", "participant_ids": ["speaker", "crowd"],
                        "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "宣布规则",
                        "obstacle": "", "decision": "", "outcome_fact_ids": ["rule"], "cost_fact_ids": [],
                        "evidence": [span(second)], "action_type": "speech", "actor_id": "speaker",
                        "target_ids": ["crowd"], "basis_fact_ids": ["rule"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        self.assertIsNone(_deterministic_public_rule_audience_batch(document, annotation))

    def test_explicit_reaction_requires_predecessor_fact_anchor(self) -> None:
        text = "甲轻松踏过五丈。\n\n青年咦了一声，露出感兴趣之色。"
        document = build_chapter_document(
            "Author/work-001/0001", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        first, second = document.units
        second_span = {"chapter_id": document.chapter_id, "start": second.start, "end": second.end,
                       "quote": second.text, "unit_id": second.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "speaker", "canonical_name": "青年", "kind": "person",
                          "aliases": [], "evidence": [second_span]}],
            "facts": [{"fact_id": "reaction", "chapter_id": document.chapter_id, "kind": "emotion",
                       "subject_id": "speaker", "predicate": "感到", "value": "感兴趣", "object_id": "",
                       "certainty": 1.0, "evidence": [second_span]}],
            "events": [{"event_id": "event-reaction", "chapter_id": document.chapter_id, "order": 0,
                        "summary": "青年感兴趣", "participant_ids": ["speaker"], "trigger_fact_ids": [],
                        "precondition_fact_ids": [], "action": "露出兴趣", "obstacle": "", "decision": "",
                        "outcome_fact_ids": ["reaction"], "cost_fact_ids": [], "evidence": [second_span],
                        "action_type": "state_change", "actor_id": "speaker", "target_ids": [],
                        "basis_fact_ids": ["reaction"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        batch = _deterministic_unlinked_reaction_batch(document, annotation)

        self.assertIsNotNone(batch)
        self.assertEqual(
            batch.issues[0].object_ids,
            (f"missing:event:{first.unit_id}", "event-reaction"),
        )

    def test_explicit_speech_cannot_be_typed_as_other(self) -> None:
        text = "门外有人喊道：开门。"
        document = build_chapter_document(
            "Author/work-001/0001", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": unit.start, "end": unit.end,
                "quote": unit.text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "caller", "canonical_name": "门外的人", "kind": "person",
                          "aliases": [], "evidence": [span]}],
            "facts": [{"fact_id": "location", "chapter_id": document.chapter_id, "kind": "progression",
                       "subject_id": "caller", "predicate": "出现", "value": "门外", "object_id": "",
                       "certainty": 1.0, "evidence": [span]}],
            "events": [{"event_id": "event-call", "chapter_id": document.chapter_id, "order": 0,
                        "summary": "门外的人呼喊开门", "participant_ids": ["caller"],
                        "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "呼喊开门",
                        "obstacle": "", "decision": "", "outcome_fact_ids": ["location"],
                        "cost_fact_ids": [], "evidence": [span], "action_type": "other",
                        "actor_id": "caller", "target_ids": [], "basis_fact_ids": ["location"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        batch = _deterministic_event_frame_batch(annotation)

        self.assertIsNotNone(batch)
        self.assertTrue(any(issue.issue_id == "speech-type-event-call" for issue in batch.issues))

    def test_reporting_progression_is_a_deterministic_fact_type_error(self) -> None:
        _, annotation = self._fixture()
        fact = replace(annotation.facts[0], kind="progression", predicate="要求", value="乙开门")

        batch = _deterministic_fact_frame_batch(replace(annotation, facts=(fact,)))

        self.assertIsNotNone(batch)
        self.assertEqual(batch.issues[0].object_ids, (fact.fact_id,))

    def test_perception_event_is_not_retyped_from_embedded_source_speech(self) -> None:
        document, annotation = self._fixture()
        event = replace(
            annotation.events[0],
            summary="甲听见门外众人问来问去",
            action="听见门外的声音",
            action_type="perception",
        )
        source_span = replace(
            event.evidence[0],
            quote="甲听见门外的声音，众人问来问去。",
        )
        event = replace(event, summary="甲听见门外动静", action="听见门外动静", evidence=(source_span,))

        self.assertIsNone(_deterministic_event_frame_batch(replace(annotation, events=(event,))))

    def test_perception_event_can_name_an_observed_announcement(self) -> None:
        _, annotation = self._fixture()
        event = replace(
            annotation.events[0],
            summary="甲听到对方宣布考核失败",
            action="听到失败宣告",
            action_type="perception",
        )

        self.assertIsNone(_deterministic_event_frame_batch(replace(annotation, events=(event,))))

    def test_standalone_quote_requires_a_grounded_speaker(self) -> None:
        text = "众人纷纷劝说。\n\n“乙，你无论如何也要入选。”"
        document = build_chapter_document(
            "Author/work-001/0001", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        quote_unit = document.units[1]
        span = SourceSpan(
            document.chapter_id, quote_unit.start, quote_unit.end,
            quote_unit.text, quote_unit.unit_id,
        )
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "entity-b", "canonical_name": "乙", "kind": "person",
                          "aliases": [], "evidence": [span.to_dict()]}],
            "facts": [{"fact_id": "fact-goal", "chapter_id": document.chapter_id,
                       "kind": "goal", "subject_id": "entity-b", "predicate": "希望",
                       "value": "入选", "object_id": "", "certainty": 1.0,
                       "evidence": [span.to_dict()]}],
            "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        batch = _deterministic_ungrounded_quote_subject_batch(document, annotation)

        self.assertIsNotNone(batch)
        self.assertEqual(batch.issues[0].object_ids, ("fact-goal",))
        self.assertIn("相邻语境", batch.issues[0].diagnosis)

    def test_standalone_quote_uses_previous_explicit_descriptive_referent(self) -> None:
        text = "主持者是一个身穿白衣的陌生青年。\n\n“能走进房间者合格。”青年言辞简短。"
        document = build_chapter_document(
            "Author/work-001/0002", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        previous, quote_unit = document.units
        entity_start = previous.start + previous.text.index("白衣的陌生青年")
        quote_start = quote_unit.start + quote_unit.text.index("能走进房间者合格")
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "speaker", "canonical_name": "白衣青年", "kind": "person",
                          "aliases": [], "evidence": [{"chapter_id": document.chapter_id,
                          "start": entity_start, "end": entity_start + len("白衣的陌生青年"),
                          "quote": "白衣的陌生青年", "unit_id": previous.unit_id}]}],
            "facts": [{"fact_id": "rule", "chapter_id": document.chapter_id, "kind": "rule",
                       "subject_id": "speaker", "predicate": "宣布", "value": "能走进房间者合格",
                       "object_id": "", "certainty": 1.0, "evidence": [{"chapter_id": document.chapter_id,
                       "start": quote_start, "end": quote_start + len("能走进房间者合格"),
                       "quote": "能走进房间者合格", "unit_id": quote_unit.unit_id}]}],
            "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        self.assertIsNone(_deterministic_ungrounded_quote_subject_batch(document, annotation))

    def test_field_filter_accepts_annotations_by_chapter_and_optional_knowledge_object(self) -> None:
        document, annotation = self._fixture()
        fact = replace(
            annotation.facts[0], kind="knowledge", object_id="",
            value="对方是仙人", predicate="意识到",
        )
        annotation = replace(annotation, facts=(fact,))
        issue = SemanticAuditIssue(
            "missing-object", "fact_semantics", "warning", (fact.fact_id,),
            (fact.evidence[0].unit_id,), "知识事实客体缺失，object_id应补人物。", "修正事实语义", 0.9,
        )
        batch = SemanticAuditBatch(
            "batch", annotation.chapter_id, (fact.evidence[0].unit_id,), (), (fact.fact_id,), (),
            "review", (issue,),
        )

        filtered = _suppress_field_contradictory_findings(
            {annotation.chapter_id: annotation}, (batch,),
        )

        self.assertEqual(filtered[0].issues, ())

    def test_prompts_bind_relationship_endpoints_not_speech_roles(self) -> None:
        text = "甲对乙说：恭喜你成为丙的弟子。"
        document = build_chapter_document(
            "Author/work-001/0001", hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        unit = document.units[0]
        units = (AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text),)
        _, annotation = self._fixture()
        annotation = replace(annotation, chapter_id=document.chapter_id, source_hash=document.source_hash)
        # The test checks instruction presence; use the source-local packet so
        # prompt construction remains valid independently of fixture wording.
        annotation = replace(annotation, entities=(), facts=(), events=(), scenes=(), state_changes=())
        packet, expected, _ = _annotation_packet(annotation, units)
        audit_prompt = _audit_prompt(document, units, packet, expected)
        issue = SemanticAuditIssue(
            "issue-relationship", "coverage", "error", (f"missing:relationship:{unit.unit_id}",),
            (unit.unit_id,), "师徒关系误把说话者当作师父。", "修正事实语义", 0.95,
        )
        batch = SemanticAuditBatch(
            "batch-relationship", document.chapter_id, (unit.unit_id,),
            tuple(expected["entities"]), tuple(expected["facts"]), tuple(expected["events"]),
            "review", (issue,),
        )
        review_prompt = _issue_review_prompt(document, units, packet, batch)

        self.assertIn("甲对乙说：恭喜你成为丙的弟子", audit_prompt)
        self.assertIn("师徒事实只能是 subject=乙、object=丙", audit_prompt)
        self.assertIn("甲只是 speech.actor", review_prompt)

    def test_prompts_distinguish_imperative_goal_and_existing_targets(self) -> None:
        document, annotation = self._fixture()
        units = split_annotation_units(document, max_input_chars=1000, overlap_units=0)[0]
        packet, expected, _ = _annotation_packet(annotation, units)
        audit_prompt = _audit_prompt(document, units, packet, expected)
        issue = SemanticAuditIssue(
            "issue-target", "event_frame", "error", (annotation.events[0].event_id,),
            (units[0].unit_id,), "受事未加入target_ids。", "修正事件语义框架", 0.9,
        )
        batch = SemanticAuditBatch(
            "batch-target", document.chapter_id, tuple(item.unit_id for item in units),
            tuple(expected["entities"]), tuple(expected["facts"]), tuple(expected["events"]),
            "review", (issue,),
        )
        review_prompt = _issue_review_prompt(document, units, packet, batch)

        self.assertIn("祈使句中的愿望承载者", audit_prompt)
        self.assertIn("逐项回译现有 target_ids", audit_prompt)
        self.assertIn("被携带者已在 target_ids", review_prompt)
        self.assertIn("knowledge 可以由公开告知", review_prompt)
        self.assertIn("直接面向该人物的称呼语+告知命题", review_prompt)
        self.assertIn("图谱孤岛", audit_prompt)
        self.assertIn("资源无法打开并导致后续鉴定", review_prompt)
        self.assertIn("不在同一叙事层", audit_prompt)
        self.assertIn("实际文字区间重叠", audit_prompt)

    def test_exact_duplicate_facts_are_reported_deterministically(self) -> None:
        _, annotation = self._fixture()
        original = annotation.facts[0]
        duplicate = replace(original, fact_id="fact-duplicate")
        batch = _deterministic_duplicate_fact_batch(
            replace(annotation, facts=(*annotation.facts, duplicate)),
        )
        self.assertIsNotNone(batch)
        self.assertEqual(batch.issues[0].category, "fact_semantics")
        self.assertIn("重复事实", batch.issues[0].diagnosis)

    def test_carried_people_and_premature_arrival_are_reported_deterministically(self) -> None:
        text = "甲抓起几个啼哭者离开。\n甲带着三名弟子踏上山峰。"
        document = build_chapter_document(
            "Author/work-001/0010", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        first, second = document.units
        span1 = {"chapter_id": document.chapter_id, "start": first.start, "end": first.end, "quote": first.text, "unit_id": first.unit_id}
        span2 = {"chapter_id": document.chapter_id, "start": second.start, "end": second.end, "quote": second.text, "unit_id": second.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "e1", "canonical_name": "甲", "kind": "person", "aliases": [], "evidence": [span1]},
                {"entity_id": "e2", "canonical_name": "三名弟子", "kind": "group", "aliases": [], "evidence": [span2]},
                {"entity_id": "loc", "canonical_name": "山峰", "kind": "location", "aliases": [], "evidence": [span2]},
            ],
            "facts": [
                {"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "progression", "subject_id": "e1", "predicate": "带走", "value": "几个啼哭者", "object_id": "", "certainty": 1.0, "evidence": [span1]},
                {"fact_id": "f2", "chapter_id": document.chapter_id, "kind": "location", "subject_id": "e1", "predicate": "到达", "value": "山峰", "object_id": "loc", "certainty": 1.0, "evidence": [span2]},
            ],
            "events": [
                {"event_id": "v1", "chapter_id": document.chapter_id, "order": 0, "summary": "甲带走啼哭者", "participant_ids": ["e1"], "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "抓起几个啼哭者离开", "obstacle": "", "decision": "", "outcome_fact_ids": ["f1"], "cost_fact_ids": [], "evidence": [span1], "action_type": "other", "actor_id": "e1", "target_ids": [], "basis_fact_ids": ["f1"]},
                {"event_id": "v2", "chapter_id": document.chapter_id, "order": 1, "summary": "甲带弟子到达山峰", "participant_ids": ["e1", "e2"], "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "带着三名弟子踏上山峰", "obstacle": "", "decision": "", "outcome_fact_ids": ["f2"], "cost_fact_ids": [], "evidence": [span2], "action_type": "movement", "actor_id": "e1", "target_ids": ["e2"], "basis_fact_ids": ["f2"]},
            ],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        batch = _deterministic_event_frame_batch(annotation)

        self.assertIsNotNone(batch)
        self.assertEqual({issue.issue_id for issue in batch.issues}, {"carried-target-v1", "motion-phase-v2"})

    def test_carried_target_and_explicit_arrival_do_not_trigger_deterministic_event_issue(self) -> None:
        document, annotation = self._fixture()
        event = replace(annotation.events[0], summary="角色甲带角色乙到达门口", action_type="movement", action="带角色乙到达门口")
        location = replace(annotation.facts[0], kind="location", predicate="到达")
        annotation = replace(annotation, facts=(location,), events=(replace(event, target_ids=("entity-002",)),))

        self.assertIsNone(_deterministic_event_frame_batch(annotation))

    def test_named_first_person_commitment_requires_goal_fact(self) -> None:
        text = "“我一定能被选中！”王林坚定地想到。"
        document = build_chapter_document(
            "Author/work-001/0011", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        span = SourceSpan(document.chapter_id, unit.start, unit.end, unit.text, unit.unit_id)
        _, fixture = self._fixture()
        entity = replace(
            fixture.entities[0], entity_id="e1", canonical_name="王林", aliases=(), evidence=(span,),
        )
        annotation = replace(
            fixture, chapter_id=document.chapter_id, source_hash=document.source_hash,
            entities=(entity,), facts=(), events=(), scenes=(), state_changes=(),
            time_anchors=(), temporal_relations=(), spatial_relations=(),
        )

        missing = _deterministic_explicit_goal_batch(document, annotation)
        goal = replace(
            fixture.facts[0], fact_id="f1", chapter_id=document.chapter_id, kind="goal",
            subject_id="e1", predicate="决心", value="被选中", object_id="", evidence=(span,),
        )

        self.assertIsNotNone(missing)
        self.assertEqual(missing.issues[0].object_ids, (f"missing:fact:{unit.unit_id}",))
        self.assertIsNone(_deterministic_explicit_goal_batch(document, replace(annotation, facts=(goal,))))

    def test_outcome_requires_direct_event_evidence_even_when_source_is_adjacent(self) -> None:
        document, annotation = self._fixture()
        remote_span = replace(
            annotation.facts[0].evidence[0], start=1000, end=1010, unit_id="remote:p0099",
        )
        remote = replace(annotation, facts=(replace(annotation.facts[0], evidence=(remote_span,)),))

        batch = _deterministic_event_outcome_evidence_batch(remote)

        self.assertIsNotNone(batch)
        self.assertEqual(batch.issues[0].object_ids, (annotation.events[0].event_id,))
        self.assertIsNotNone(_deterministic_event_outcome_evidence_batch(annotation))
        covered_event = replace(
            annotation.events[0],
            evidence=(*annotation.events[0].evidence, *annotation.facts[0].evidence),
        )
        self.assertIsNone(_deterministic_event_outcome_evidence_batch(
            replace(annotation, events=(covered_event,)),
        ))

    def test_near_duplicate_prefers_referenced_fact_and_suppresses_model_copy(self) -> None:
        _, annotation = self._fixture()
        original = annotation.facts[0]
        referenced = replace(original, value="王浩拍马送礼获得药童资格")
        duplicate = replace(
            original, fact_id="fact-duplicate", value="王浩拍马屁送礼得药童资格",
        )
        event = replace(
            annotation.events[0], basis_fact_ids=(referenced.fact_id,),
            outcome_fact_ids=(referenced.fact_id,),
        )
        changed = replace(annotation, facts=(referenced, duplicate), events=(event,))
        deterministic = _deterministic_duplicate_fact_batch(changed)
        self.assertIsNotNone(deterministic)
        self.assertEqual(deterministic.issues[0].object_ids, (duplicate.fact_id,))
        model_issue = replace(
            deterministic.issues[0], issue_id="model-duplicate",
            object_ids=(referenced.fact_id, duplicate.fact_id), diagnosis="两条事实语义重叠，应合并。",
        )
        model_batch = replace(
            deterministic, batch_id=f"{annotation.chapter_id}:semantic-audit-001",
            issues=(model_issue,),
        )
        filtered = _suppress_duplicate_model_findings((deterministic, model_batch))
        self.assertTrue(filtered[1].passed)

    def test_deterministic_event_checks_do_not_suppress_each_other(self) -> None:
        _, annotation = self._fixture()
        event_issue = SemanticAuditIssue(
            "speech-type", "event_frame", "error", (annotation.events[0].event_id,),
            (annotation.events[0].evidence[0].unit_id,), "事件类型错误。", "修正事件语义框架", 1.0,
        )
        outcome_issue = replace(event_issue, issue_id="outcome-evidence", diagnosis="结果证据错接。")
        event_batch = SemanticAuditBatch(
            f"{annotation.chapter_id}:deterministic-event-frames", annotation.chapter_id,
            (annotation.events[0].evidence[0].unit_id,), (), (), (annotation.events[0].event_id,),
            "review", (event_issue,),
        )
        outcome_batch = replace(
            event_batch, batch_id=f"{annotation.chapter_id}:deterministic-outcome-evidence",
            issues=(outcome_issue,),
        )

        filtered = _suppress_deterministic_event_model_findings((event_batch, outcome_batch))

        self.assertEqual(len(filtered[0].issues), 1)
        self.assertEqual(len(filtered[1].issues), 1)

    def test_deterministic_fact_check_suppresses_model_copy_only(self) -> None:
        _, annotation = self._fixture()
        fact_id = annotation.facts[0].fact_id
        deterministic_issue = SemanticAuditIssue(
            "fact-kind", "fact_semantics", "error", (fact_id,),
            (annotation.facts[0].evidence[0].unit_id,), "事实类型错误。", "修正事实类型", 1.0,
        )
        model_issue = replace(deterministic_issue, issue_id="model-fact-kind")
        deterministic = SemanticAuditBatch(
            f"{annotation.chapter_id}:deterministic-fact-frames", annotation.chapter_id,
            (annotation.facts[0].evidence[0].unit_id,), (), (fact_id,), (),
            "review", (deterministic_issue,),
        )
        model = replace(
            deterministic, batch_id=f"{annotation.chapter_id}:semantic-audit-001",
            issues=(model_issue,),
        )

        filtered = _suppress_deterministic_fact_model_findings((deterministic, model))

        self.assertEqual(len(filtered[0].issues), 1)
        self.assertTrue(filtered[1].passed)

    def test_listener_replacement_is_suppressed_when_current_object_is_in_proposition(self) -> None:
        document, annotation = self._fixture()
        subject, current_object = annotation.entities[:2]
        fact = replace(
            annotation.facts[0], kind="information", subject_id=subject.entity_id,
            predicate="告知", value=f"{current_object.canonical_name}明日接人",
            object_id=current_object.entity_id,
        )
        changed = replace(annotation, facts=(fact,))
        issue = SemanticAuditIssue(
            "listener-object", "fact_semantics", "error", (fact.fact_id,),
            (fact.evidence[0].unit_id,), "事实对象应为听话者，而非当前组织。", "修正事实语义", 0.9,
        )
        batch = SemanticAuditBatch(
            f"{annotation.chapter_id}:semantic-audit-001", annotation.chapter_id,
            (fact.evidence[0].unit_id,), (), (fact.fact_id,), (), "review", (issue,),
        )

        filtered = _suppress_field_contradictory_findings(changed, (batch,))

        self.assertTrue(filtered[0].passed)

    def test_local_speech_target_resolves_terse_information_object(self) -> None:
        _, annotation = self._fixture()
        actor, target = annotation.entities[:2]
        fact = replace(
            annotation.facts[0], kind="information", subject_id=actor.entity_id,
            predicate="宣布", value="不合格", object_id=target.entity_id,
        )
        event = replace(
            annotation.events[0], summary=f"裁判宣布{target.canonical_name}不合格",
            action="宣布不合格", actor_id=actor.entity_id,
            participant_ids=(actor.entity_id, target.entity_id), target_ids=(target.entity_id,),
            basis_fact_ids=(fact.fact_id,), outcome_fact_ids=(fact.fact_id,),
        )
        changed = replace(annotation, facts=(fact,), events=(event,))
        issue = SemanticAuditIssue(
            "ambiguous-object", "fact_semantics", "warning", (fact.fact_id,),
            (fact.evidence[0].unit_id,), "前文对别人说过同样的话，现有object_id指代歧义。", "修正事实语义", 0.85,
        )
        batch = SemanticAuditBatch(
            f"{annotation.chapter_id}:semantic-audit-001", annotation.chapter_id,
            (fact.evidence[0].unit_id,), (), (fact.fact_id,), (event.event_id,), "review", (issue,),
        )

        filtered = _suppress_field_contradictory_findings(changed, (batch,))

        self.assertTrue(filtered[0].passed)

    def test_event_claim_is_suppressed_when_all_outcomes_overlap_event_evidence(self) -> None:
        _, annotation = self._fixture()
        fact = annotation.facts[0]
        event = replace(
            annotation.events[0], evidence=fact.evidence,
            basis_fact_ids=(fact.fact_id,), outcome_fact_ids=(fact.fact_id,),
        )
        changed = replace(annotation, events=(event,))
        issue = SemanticAuditIssue(
            "missed-evidence", "event_frame", "warning", (event.event_id,),
            (fact.evidence[0].unit_id,), "事件证据仅支持前置动作，未直接体现结果。", "修正事件语义框架", 0.8,
        )
        batch = SemanticAuditBatch(
            f"{annotation.chapter_id}:semantic-audit-001", annotation.chapter_id,
            (fact.evidence[0].unit_id,), (), (fact.fact_id,), (event.event_id,), "review", (issue,),
        )

        filtered = _suppress_field_contradictory_findings(changed, (batch,))

        self.assertTrue(filtered[0].passed)

    def test_speculative_existing_object_diagnosis_is_not_actionable(self) -> None:
        _, annotation = self._fixture()
        fact = annotation.facts[0]
        issue = SemanticAuditIssue(
            "speculative-owner", "fact_semantics", "warning", (fact.fact_id,),
            (fact.evidence[0].unit_id,), "事实主体或应为在场群体。", "修正事实语义", 0.8,
        )
        batch = SemanticAuditBatch(
            f"{annotation.chapter_id}:semantic-audit-001", annotation.chapter_id,
            (fact.evidence[0].unit_id,), (), (fact.fact_id,), (), "review", (issue,),
        )

        filtered = _suppress_field_contradictory_findings(annotation, (batch,))

        self.assertTrue(filtered[0].passed)

    def test_issue_review_prompt_warns_against_nearest_noun_pronoun_resolution(self) -> None:
        document, annotation = self._fixture()
        units = split_annotation_units(document, max_input_chars=1000, overlap_units=0)[0]
        packet, _, _ = _annotation_packet(annotation, units)
        fact = annotation.facts[0]
        issue = SemanticAuditIssue(
            "pronoun-owner", "fact_semantics", "warning", (fact.fact_id,),
            (fact.evidence[0].unit_id,), "他们应指最近出现的人物。", "修正事实语义", 0.8,
        )
        batch = SemanticAuditBatch(
            f"{annotation.chapter_id}:semantic-audit-001", annotation.chapter_id,
            tuple(item.unit_id for item in units), (), (fact.fact_id,), (), "review", (issue,),
        )

        prompt = _issue_review_prompt(document, units, packet, batch)

        self.assertIn("代词解指不能机械选择最近名词", prompt)
        self.assertIn("只是时间从句", prompt)

    def test_audit_prompt_requires_atomic_object_limit_and_listener_boundary(self) -> None:
        document, annotation = self._fixture()
        units = split_annotation_units(document, max_input_chars=1000, overlap_units=0)[0]
        packet, expected, _ = _annotation_packet(annotation, units)

        prompt = _audit_prompt(document, units, packet, expected)

        self.assertIn("绝不能输出4个及以上对象ID", prompt)
        self.assertIn("缺事件只报一条 `coverage + missing:event`", prompt)
        self.assertIn("对白听者只在 speech.target_ids", prompt)

    def test_missing_target_claim_is_suppressed_when_carried_target_exists(self) -> None:
        document, annotation = self._fixture()
        carried = replace(
            annotation.entities[1], canonical_name="三名弟子", aliases=(),
        )
        event = replace(
            annotation.events[0], action_type="other", summary="甲带着三个弟子离开",
            action="带着三个弟子离开", target_ids=(carried.entity_id,),
        )
        annotation = replace(annotation, entities=(annotation.entities[0], carried), events=(event,))
        issue = SemanticAuditIssue(
            "issue-mixed", "event_frame", "error", (event.event_id,),
            (event.evidence[0].unit_id,),
            "事件阶段需复核，且移动受事未加入 target_ids。", "修正事件语义框架", 0.95,
        )
        batch = SemanticAuditBatch(
            "batch-mixed", annotation.chapter_id, (event.evidence[0].unit_id,),
            (), (), (event.event_id,), "review", (issue,),
        )

        filtered = _suppress_field_contradictory_findings(annotation, (batch,))

        self.assertTrue(filtered[0].passed)

    def test_one_off_praise_prediction_without_causal_consumer_is_not_coverage(self) -> None:
        _, annotation = self._fixture()
        unit_id = annotation.events[0].evidence[0].unit_id
        issue = SemanticAuditIssue(
            "praise", "coverage", "warning", (f"missing:fact:{unit_id}",), (unit_id,),
            "老者称赞并预测少年可能入选，缺少该态度命题。", "重标当前窗口", 0.9,
        )
        batch = SemanticAuditBatch(
            "batch-praise", annotation.chapter_id, (unit_id,), (), (), (), "review", (issue,),
        )

        filtered = _suppress_field_contradictory_findings(annotation, (batch,))

        self.assertTrue(filtered[0].passed)

    def test_invalid_review_vote_abstains_instead_of_failing_window(self) -> None:
        from pipeline.model import ModelSettings
        from pipeline.semantic_audit import _independent_issue_review

        document, annotation = self._fixture()
        units = split_annotation_units(document, max_input_chars=1000, overlap_units=0)[0]
        packet, _, _ = _annotation_packet(annotation, units)
        issue = SemanticAuditIssue(
            "issue-001", "fact_semantics", "error", (annotation.facts[0].fact_id,),
            (units[0].unit_id,), "事实类型错误。", "修正事实语义", 0.95,
        )
        batch = SemanticAuditBatch(
            "batch-001", document.chapter_id, (units[0].unit_id,), (),
            (annotation.facts[0].fact_id,), (), "review", (issue,),
        )
        contradictory = {"判定": [{
            "问题ID": "issue-001", "结论": "误报", "对象确实矛盾": False,
            "已有语义覆盖": True, "剧情承载必要性": False,
            "理由": "诊断正确，主体错误。", "置信度": 0.9,
        }]}
        with patch("pipeline.semantic_audit.complete_json", return_value=contradictory):
            reviewed, payload = _independent_issue_review(
                document, units, packet, batch, ModelSettings(model="test"),
            )
        self.assertTrue(reviewed.passed)
        self.assertEqual(payload["判定"][0]["结论"], "证据不足")

    def test_independent_review_keeps_only_confirmed_findings(self) -> None:
        issue = SemanticAuditIssue(
            "issue-001", "event_frame", "error", ("event-001",), ("chapter:p0001",),
            "事件行动者与原文不符。", "修正事件语义框架", 0.92,
        )
        batch = SemanticAuditBatch(
            "batch-001", "chapter", ("chapter:p0001",), (), (), ("event-001",),
            "review", (issue,),
        )
        confirmed = _parse_issue_review_payload({
            "判定": [{
                "问题ID": "issue-001", "结论": "成立",
                "对象确实矛盾": True, "已有语义覆盖": False, "剧情承载必要性": True,
                "理由": "原文明确由另一人物执行动作。", "置信度": 0.86,
            }],
        }, batch)
        self.assertEqual(len(confirmed.issues), 1)
        self.assertEqual(confirmed.issues[0].confidence, 0.86)

        rejected = _parse_issue_review_payload({
            "判定": [{
                "问题ID": "issue-001", "结论": "误报",
                "对象确实矛盾": False, "已有语义覆盖": False, "剧情承载必要性": False,
                "理由": "行动者与原文实际一致。", "置信度": 0.97,
            }],
        }, batch)
        self.assertTrue(rejected.passed)

    def test_model_cannot_emit_internal_audit_contract_category(self) -> None:
        document, annotation = self._fixture()
        units = split_annotation_units(document, max_input_chars=1000, overlap_units=0)[0]
        _, expected, known_ids = _annotation_packet(annotation, units)
        payload = {
            "结论": "需复核",
            "已核实体ID": list(expected["entities"]),
            "已核事实ID": list(expected["facts"]),
            "已核事件ID": list(expected["events"]),
            "问题": [{
                "问题ID": "issue-001", "类别": "audit_contract", "严重度": "warning",
                "对象ID": [annotation.events[0].event_id],
                "证据单元ID": [units[0].unit_id],
                "诊断": "普通引用语义问题。", "建议动作": "重新审计当前窗口", "置信度": 0.9,
            }],
        }
        with self.assertRaisesRegex(ValueError, "只能由程序"):
            _parse_audit_payload(
                payload, batch_id="batch", chapter_id=document.chapter_id,
                source_unit_ids=tuple(item.unit_id for item in units), expected=expected,
                known_ids=known_ids, object_evidence=_object_evidence_index(annotation),
            )

    def _fixture(self):
        chapter_id = "Author/work-001/0001"
        text = "角色甲把令牌交给角色乙。\n角色乙收下令牌。"
        document = build_chapter_document(chapter_id, hashlib.sha256(text.encode("utf-8")).hexdigest(), text)
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": chapter_id,
            "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "entity-001", "chapter_id": chapter_id, "kind": "人物", "canonical_name": "角色甲", "aliases": [], "evidence": [{"chapter_id": chapter_id, "start": 0, "end": 3, "quote": "角色甲", "unit_id": document.units[0].unit_id}]},
                {"entity_id": "entity-002", "chapter_id": chapter_id, "kind": "人物", "canonical_name": "角色乙", "aliases": [], "evidence": [{"chapter_id": chapter_id, "start": 8, "end": 11, "quote": "角色乙", "unit_id": document.units[0].unit_id}]},
            ],
            "facts": [
                {"fact_id": "fact-001", "chapter_id": chapter_id, "kind": "资源", "subject_id": "entity-002", "predicate": "获得", "value": "令牌", "object_id": "", "certainty": 1, "evidence": [{"chapter_id": chapter_id, "start": 13, "end": 21, "quote": "角色乙收下令牌", "unit_id": document.units[1].unit_id}]},
            ],
            "events": [
                {"event_id": "event-001", "chapter_id": chapter_id, "order": 0, "summary": "角色甲交付令牌", "participant_ids": ["entity-001", "entity-002"], "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "把令牌交给角色乙", "obstacle": "", "decision": "", "outcome_fact_ids": ["fact-001"], "cost_fact_ids": [], "evidence": [{"chapter_id": chapter_id, "start": 0, "end": 11, "quote": "角色甲把令牌交给角色乙", "unit_id": document.units[0].unit_id}], "action_type": "transfer", "actor_id": "entity-001", "target_ids": ["entity-002"], "basis_fact_ids": ["fact-001"]},
            ],
            "scenes": [
                {"scene_id": "scene-001", "chapter_id": chapter_id, "order": 0, "participant_ids": ["entity-001", "entity-002"], "event_ids": ["event-001"], "objective": "交付令牌", "entry_fact_ids": [], "exit_fact_ids": ["fact-001"], "tension": 1, "evidence": [{"chapter_id": chapter_id, "start": 0, "end": 11, "quote": "角色甲把令牌交给角色乙", "unit_id": document.units[0].unit_id}]},
            ],
            "state_changes": [
                {"change_id": "change-001", "chapter_id": chapter_id, "event_id": "event-001", "operation": "add", "before_fact_id": "", "after_fact_id": "fact-001"},
            ],
            "time_anchors": [], "temporal_relations": [], "spatial_relations": [],
            "schema_version": "2.2",
        })
        return document, annotation

    def test_packet_is_bounded_to_current_source_window(self) -> None:
        document, annotation = self._fixture()
        first_window = (split_annotation_units(document, max_input_chars=1000, overlap_units=0)[0][0],)
        packet, expected, known = _annotation_packet(annotation, first_window)
        self.assertEqual(expected["events"], ("event-001",))
        self.assertIn("fact-001", known)
        self.assertEqual(packet["只供引用解析的上下文"]["facts"][0]["fact_id"], "fact-001")

    def test_strict_payload_requires_complete_audited_ids(self) -> None:
        document, annotation = self._fixture()
        units = split_annotation_units(document, max_input_chars=1000, overlap_units=0)[0]
        _, expected, known = _annotation_packet(annotation, units)
        source_ids = tuple(item.unit_id for item in units)
        payload = {
            "结论": "通过",
            "已核实体ID": list(expected["entities"]),
            "已核事实ID": list(expected["facts"]),
            "已核事件ID": list(expected["events"]),
            "问题": [],
        }
        batch = _parse_audit_payload(
            payload, batch_id="batch-001", chapter_id=document.chapter_id,
            source_unit_ids=source_ids, expected=expected, known_ids=known,
        )
        self.assertTrue(batch.passed)
        with self.assertRaises(ValueError):
            _parse_audit_payload(
                {**payload, "已核事件ID": []}, batch_id="batch-001", chapter_id=document.chapter_id,
                source_unit_ids=source_ids, expected=expected, known_ids=known,
            )

    def test_issue_must_use_current_source_and_known_object(self) -> None:
        document, annotation = self._fixture()
        units = split_annotation_units(document, max_input_chars=1000, overlap_units=0)[0]
        _, expected, known = _annotation_packet(annotation, units)
        source_ids = tuple(item.unit_id for item in units)
        issue = {
            "问题ID": "issue-001", "类别": "event_frame", "严重度": "error",
            "对象ID": ["event-001"], "证据单元ID": [source_ids[0]],
            "诊断": "行动结果方向需要复核。", "建议动作": "修正事件语义框架", "置信度": 0.9,
        }
        payload = {
            "结论": "需复核", "已核实体ID": list(expected["entities"]),
            "已核事实ID": list(expected["facts"]), "已核事件ID": list(expected["events"]),
            "问题": [issue],
        }
        self.assertFalse(_parse_audit_payload(
            payload, batch_id="batch-001", chapter_id=document.chapter_id,
            source_unit_ids=source_ids, expected=expected, known_ids=known,
        ).passed)
        with self.assertRaises(ValueError):
            _parse_audit_payload(
                {**payload, "问题": [{**issue, "对象ID": ["unknown"]}]},
                batch_id="batch-001", chapter_id=document.chapter_id,
                source_unit_ids=source_ids, expected=expected, known_ids=known,
            )

    def test_ungrounded_issue_row_does_not_erase_grounded_sibling(self) -> None:
        document, annotation = self._fixture()
        # Audit only the fact's second-paragraph evidence. The event belongs
        # to the first paragraph, so its finding must be discarded alone.
        second = document.units[1]
        units = (AnnotationSourceSlice(second.unit_id, second.unit_id, second.start, second.end, second.text),)
        _, expected, known = _annotation_packet(annotation, units)
        source_ids = tuple(item.unit_id for item in units)
        payload = {
            "结论": "需复核", "已核实体ID": list(expected["entities"]),
            "已核事实ID": list(expected["facts"]), "已核事件ID": list(expected["events"]),
            "问题": [
                {
                    "问题ID": "grounded", "类别": "fact_semantics", "严重度": "error",
                    "对象ID": [annotation.facts[0].fact_id], "证据单元ID": [document.units[1].unit_id],
                    "诊断": "事实命题与原文矛盾。", "建议动作": "修正事实语义", "置信度": 0.9,
                },
                {
                    "问题ID": "ungrounded", "类别": "event_frame", "严重度": "warning",
                    "对象ID": [annotation.events[0].event_id], "证据单元ID": [document.units[1].unit_id],
                    "诊断": "用另一段证据指控当前事件。", "建议动作": "修正事件语义框架", "置信度": 0.8,
                },
            ],
        }

        batch = _parse_audit_payload(
            payload, batch_id="batch-isolated", chapter_id=document.chapter_id,
            source_unit_ids=source_ids, expected=expected, known_ids=known,
            object_evidence=_object_evidence_index(annotation),
        )

        self.assertEqual([issue.issue_id for issue in batch.issues], ["grounded"])

    def test_coverage_issue_requires_typed_missing_object(self) -> None:
        document, annotation = self._fixture()
        units = split_annotation_units(document, max_input_chars=1000, overlap_units=0)[0]
        _, expected, known = _annotation_packet(annotation, units)
        source_ids = tuple(item.unit_id for item in units)
        base = {
            "结论": "需复核", "已核实体ID": list(expected["entities"]),
            "已核事实ID": list(expected["facts"]), "已核事件ID": list(expected["events"]),
        }
        issue = {
            "问题ID": "issue-001", "类别": "coverage", "严重度": "warning",
            "对象ID": [f"missing:event:{source_ids[1]}"], "证据单元ID": [source_ids[1]],
            "诊断": "收取动作没有任何事件对应。", "建议动作": "重标当前窗口", "置信度": 0.9,
        }
        self.assertFalse(_parse_audit_payload(
            {**base, "问题": [issue]}, batch_id="batch-001", chapter_id=document.chapter_id,
            source_unit_ids=source_ids, expected=expected, known_ids=known,
        ).passed)
        compound = {**issue, "对象ID": [f"missing:event:{source_ids[0]}:{source_ids[1].rsplit(':', 1)[-1]}"]}
        normalized = _parse_audit_payload(
            {**base, "问题": [compound]}, batch_id="batch-001", chapter_id=document.chapter_id,
            source_unit_ids=source_ids, expected=expected, known_ids=known,
        )
        self.assertEqual(normalized.issues[0].object_ids, (f"missing:event:{source_ids[1]}",))
        with self.assertRaises(ValueError):
            _parse_audit_payload(
                {**base, "问题": [{**issue, "对象ID": ["event-001"]}]},
                batch_id="batch-001", chapter_id=document.chapter_id,
                source_unit_ids=source_ids, expected=expected, known_ids=known,
            )

    def test_compound_coverage_action_is_rejected_for_atomic_reaudit(self) -> None:
        document, annotation = self._fixture()
        units = split_annotation_units(document, max_input_chars=1000, overlap_units=0)[0]
        _, expected, known = _annotation_packet(annotation, units)
        source_ids = tuple(item.unit_id for item in units)
        payload = {
            "结论": "需复核",
            "已核实体ID": list(expected["entities"]),
            "已核事实ID": list(expected["facts"]),
            "已核事件ID": list(expected["events"]),
            "问题": [{
                "问题ID": "compound-action",
                "类别": "coverage",
                "严重度": "error",
                "对象ID": [f"missing:event:{source_ids[0]}"],
                "证据单元ID": [source_ids[0]],
                "诊断": "人物移动到新地点并随后接受治疗，直接改变其位置与身体状态。",
                "建议动作": "重标当前窗口",
                "置信度": 0.95,
            }],
        }
        with self.assertRaisesRegex(ValueError, "捆绑了多个动作类型"):
            _parse_audit_payload(
                payload, batch_id="batch-compound", chapter_id=document.chapter_id,
                source_unit_ids=source_ids, expected=expected, known_ids=known,
            )

    def test_unique_diagnosis_quote_relocates_neighbouring_missing_anchor(self) -> None:
        document, annotation = self._fixture()
        units = split_annotation_units(document, max_input_chars=1000, overlap_units=0)[0]
        _, expected, known = _annotation_packet(annotation, units)
        source_ids = tuple(item.unit_id for item in units)
        payload = {
            "结论": "需复核", "已核实体ID": list(expected["entities"]),
            "已核事实ID": list(expected["facts"]), "已核事件ID": list(expected["events"]),
            "问题": [{
                "问题ID": "mislocated", "类别": "coverage", "严重度": "warning",
                "对象ID": [f"missing:fact:{source_ids[0]}"], "证据单元ID": [source_ids[0]],
                "诊断": "原文‘角色乙收下令牌’表明资源已转移，现有事实未保存。",
                "建议动作": "重标当前窗口", "置信度": 0.9,
            }],
        }

        batch = _parse_audit_payload(
            payload, batch_id="batch-relocate", chapter_id=document.chapter_id,
            source_unit_ids=source_ids, expected=expected, known_ids=known,
            source_unit_texts={item.unit_id: item.text for item in document.units},
        )

        self.assertEqual(batch.issues[0].object_ids, (f"missing:fact:{source_ids[1]}",))
        self.assertEqual(batch.issues[0].evidence_unit_ids[0], source_ids[1])

    def test_category_and_action_must_match_object_semantics(self) -> None:
        document, annotation = self._fixture()
        units = split_annotation_units(document, max_input_chars=1000, overlap_units=0)[0]
        _, expected, known = _annotation_packet(annotation, units)
        source_ids = tuple(item.unit_id for item in units)
        issue = {
            "问题ID": "issue-001", "类别": "temporal", "严重度": "warning",
            "对象ID": ["event-001"], "证据单元ID": [source_ids[0]],
            "诊断": "时间顺序错误。", "建议动作": "修正时间关系", "置信度": 0.9,
        }
        payload = {
            "结论": "需复核", "已核实体ID": list(expected["entities"]),
            "已核事实ID": list(expected["facts"]), "已核事件ID": list(expected["events"]),
            "问题": [issue],
        }
        with self.assertRaisesRegex(ValueError, "对象类型"):
            _parse_audit_payload(
                payload, batch_id="batch-001", chapter_id=document.chapter_id,
                source_unit_ids=source_ids, expected=expected, known_ids=known,
            )
        with self.assertRaisesRegex(ValueError, "建议动作"):
            _parse_audit_payload(
                {**payload, "问题": [{**issue, "类别": "event_frame"}]},
                batch_id="batch-001", chapter_id=document.chapter_id,
                source_unit_ids=source_ids, expected=expected, known_ids=known,
            )

    def test_mixed_context_objects_are_narrowed_to_typed_repair_target(self) -> None:
        document, annotation = self._fixture()
        units = split_annotation_units(document, max_input_chars=1000, overlap_units=0)[0]
        _, expected, known = _annotation_packet(annotation, units)
        source_ids = tuple(item.unit_id for item in units)
        relation_id = "Author/work-001/0001:space-001"
        issue = {
            "问题ID": "issue-001", "类别": "spatial", "严重度": "warning",
            "对象ID": ["event-001", "fact-001", relation_id],
            "证据单元ID": [source_ids[0]],
            "诊断": "现有空间关系的地点方向错误。", "建议动作": "修正空间关系", "置信度": 0.9,
        }
        payload = {
            "结论": "需复核", "已核实体ID": list(expected["entities"]),
            "已核事实ID": list(expected["facts"]), "已核事件ID": list(expected["events"]),
            "问题": [issue],
        }
        batch = _parse_audit_payload(
            payload, batch_id="batch-001", chapter_id=document.chapter_id,
            source_unit_ids=source_ids, expected=expected, known_ids={*known, relation_id},
            object_evidence={relation_id: frozenset({source_ids[0]})},
        )
        self.assertEqual(batch.issues[0].object_ids, (relation_id,))

    def test_existing_object_issue_adds_own_evidence_to_later_contradiction(self) -> None:
        document, annotation = self._fixture()
        units = split_annotation_units(document, max_input_chars=1000, overlap_units=0)[0]
        _, expected, known = _annotation_packet(annotation, units)
        source_ids = tuple(item.unit_id for item in units)
        evidence = _object_evidence_index(annotation)
        base_issue = {
            "问题ID": "issue-001", "类别": "event_frame", "严重度": "warning",
            "对象ID": ["event-001"], "证据单元ID": [source_ids[1]],
            "诊断": "现有事件的行动语义错误。", "建议动作": "修正事件语义框架", "置信度": 0.9,
        }
        payload = {
            "结论": "需复核", "已核实体ID": list(expected["entities"]),
            "已核事实ID": list(expected["facts"]), "已核事件ID": list(expected["events"]),
            "问题": [base_issue],
        }
        grounded = _parse_audit_payload(
            payload, batch_id="batch-001", chapter_id=document.chapter_id,
            source_unit_ids=source_ids, expected=expected, known_ids=known, object_evidence=evidence,
        )
        self.assertEqual(
            grounded.issues[0].evidence_unit_ids,
            (source_ids[1], source_ids[0]),
        )
        self_negating = {
            **base_issue,
            "证据单元ID": [source_ids[0]],
            "诊断": "核心语义已经捕捉，不构成错误。",
        }
        parsed = _parse_audit_payload(
            {**payload, "问题": [self_negating]}, batch_id="batch-001", chapter_id=document.chapter_id,
            source_unit_ids=source_ids, expected=expected, known_ids=known, object_evidence=evidence,
        )
        self.assertEqual(parsed.verdict, "pass")
        self.assertEqual(parsed.issues, ())

        semantically_correct = {
            **base_issue,
            "证据单元ID": [source_ids[0]],
            "诊断": "现有事件语义正确，只是摘要还可更清楚。",
        }
        parsed = _parse_audit_payload(
            {**payload, "问题": [semantically_correct]}, batch_id="batch-001", chapter_id=document.chapter_id,
            source_unit_ids=source_ids, expected=expected, known_ids=known, object_evidence=evidence,
        )
        self.assertEqual(parsed.verdict, "pass")
        self.assertEqual(parsed.issues, ())

    def test_conjoined_actions_in_same_family_are_still_atomic(self) -> None:
        self.assertEqual(
            set(coverage_action_atoms("人物甲离开并由人物乙送别至村口。")),
            {"depart", "escort"},
        )
    def test_entity_candidates_use_known_alias_substitution_without_auto_merging(self) -> None:
        document, annotation = self._fixture()
        protagonist, parent = annotation.entities
        protagonist = replace(protagonist, canonical_name="人物甲", aliases=("小名甲",))
        parent = replace(parent, canonical_name="人物甲父亲", aliases=())
        duplicate_parent = replace(
            parent,
            entity_id="entity-003",
            canonical_name="小名甲父亲",
            evidence=(parent.evidence[0],),
        )
        annotation = replace(annotation, entities=(protagonist, parent, duplicate_parent))

        candidates = _entity_candidate_pairs(annotation)

        pairs = {tuple(item["entity_ids"]) for item in candidates}
        self.assertIn(("entity-002", "entity-003"), pairs)
        self.assertEqual(len(annotation.entities), 3)

    def test_entity_candidates_tolerate_possessive_particle_in_relational_name(self) -> None:
        _, annotation = self._fixture()
        protagonist, parent = annotation.entities
        parent = replace(parent, canonical_name="小名甲父亲", aliases=())
        duplicate_parent = replace(
            parent, entity_id="entity-003", canonical_name="小名甲的父亲",
        )
        annotation = replace(
            annotation, entities=(protagonist, parent, duplicate_parent),
        )

        pairs = {tuple(item["entity_ids"]) for item in _entity_candidate_pairs(annotation)}

        self.assertIn((parent.entity_id, duplicate_parent.entity_id), pairs)

    def test_owner_is_not_identity_candidate_with_their_relational_entity(self) -> None:
        _, annotation = self._fixture()
        protagonist, parent = annotation.entities
        protagonist = replace(protagonist, canonical_name="人物甲", aliases=("小名甲",))
        parent = replace(parent, canonical_name="人物甲父亲", aliases=())
        candidates = _entity_candidate_pairs(replace(
            annotation, entities=(protagonist, parent),
        ))

        self.assertEqual(candidates, ())

    def test_same_role_suffix_with_different_owner_does_not_create_candidate(self) -> None:
        _, annotation = self._fixture()
        left, right = annotation.entities
        left = replace(left, canonical_name="人物甲父亲", aliases=())
        right = replace(right, canonical_name="人物乙父亲", aliases=())

        self.assertEqual(_entity_candidate_pairs(replace(annotation, entities=(left, right))), ())

    def test_entity_resolution_requires_every_candidate_and_exact_pair_targets(self) -> None:
        document, annotation = self._fixture()
        left, right = annotation.entities
        left = replace(left, canonical_name="四叔", aliases=())
        right = replace(right, canonical_name="中年汉子", aliases=("四叔",))
        annotation = replace(annotation, entities=(left, right))
        candidates = _entity_candidate_pairs(annotation)
        self.assertEqual(len(candidates), 1)
        _, source_ids = _entity_resolution_prompt(document, annotation, candidates)
        payload = {
            "判定": [{
                "候选ID": candidates[0]["candidate_id"],
                "身份检查": "同一对象证据成立",
                "结论": "同一实体",
                "保留实体ID": left.entity_id,
                "错误别名所属实体ID": "",
                "错误别名": "",
                "证据单元ID": list(source_ids),
                "理由": "中年汉子的原文称谓随后明确回指四叔。",
                "置信度": 0.95,
            }],
        }
        batch = _parse_entity_resolution_payload(
            payload,
            chapter_id=document.chapter_id,
            candidates=candidates,
            annotation=annotation,
            source_unit_ids=source_ids,
            batch_index=1,
        )
        self.assertEqual(batch.issues[0].object_ids, (left.entity_id, right.entity_id))
        with self.assertRaisesRegex(ValueError, "候选ID"):
            _parse_entity_resolution_payload(
                {"判定": []},
                chapter_id=document.chapter_id,
                candidates=candidates,
                annotation=annotation,
                source_unit_ids=source_ids,
                batch_index=1,
            )

        contradictory = {
            "判定": [{**payload["判定"][0], "身份检查": "明确为两个对象"}],
        }
        with self.assertRaisesRegex(ValueError, "身份检查与结论矛盾"):
            _parse_entity_resolution_payload(
                contradictory,
                chapter_id=document.chapter_id,
                candidates=candidates,
                annotation=annotation,
                source_unit_ids=source_ids,
                batch_index=1,
            )

        right_with_unrelated_alias = replace(right, aliases=("四叔", "无关姓名"))
        unrelated_annotation = replace(annotation, entities=(left, right_with_unrelated_alias))
        unrelated_payload = {
            "判定": [{
                **payload["判定"][0],
                "身份检查": "明确为两个对象",
                "结论": "别名错误",
                "保留实体ID": "",
                "错误别名所属实体ID": right.entity_id,
                "错误别名": "无关姓名",
            }],
        }
        with self.assertRaisesRegex(ValueError, "并未与候选另一实体"):
            _parse_entity_resolution_payload(
                unrelated_payload,
                chapter_id=document.chapter_id,
                candidates=candidates,
                annotation=unrelated_annotation,
                source_unit_ids=source_ids,
                batch_index=1,
            )
        downgraded = _downgrade_external_alias_claims(
            unrelated_payload, candidates, unrelated_annotation,
        )
        safe_batch = _parse_entity_resolution_payload(
            downgraded,
            chapter_id=document.chapter_id,
            candidates=candidates,
            annotation=unrelated_annotation,
            source_unit_ids=source_ids,
            batch_index=1,
        )
        self.assertTrue(safe_batch.passed)

    def test_entity_resolution_consensus_rejects_disputed_mutation(self) -> None:
        first = {"判定": [{
            "候选ID": "pair-1", "身份检查": "同一对象证据成立", "结论": "同一实体",
            "保留实体ID": "e1", "错误别名所属实体ID": "", "错误别名": "",
            "证据单元ID": ["u1"], "理由": "初审认为同指", "置信度": 0.9,
        }]}
        second = {"判定": [{
            "候选ID": "pair-1", "身份检查": "明确为两个对象", "结论": "不同实体",
            "保留实体ID": "", "错误别名所属实体ID": "", "错误别名": "",
            "证据单元ID": ["u2"], "理由": "复核发现二人关系", "置信度": 0.95,
        }]}

        result = _entity_resolution_consensus(first, second)["判定"][0]

        self.assertEqual(result["身份检查"], "无法判断")
        self.assertEqual(result["结论"], "证据不足")
        self.assertEqual(result["保留实体ID"], "")
        self.assertEqual(result["证据单元ID"], ["u1", "u2"])

    def test_entity_resolution_consensus_accepts_alias_chain_against_only_insufficient_vote(self) -> None:
        _, annotation = self._fixture()
        first_entity, second_entity = annotation.entities
        second_entity = replace(second_entity, aliases=(first_entity.canonical_name,))
        annotation = replace(annotation, entities=(first_entity, second_entity))
        candidates = ({
            "candidate_id": "pair-1",
            "entity_ids": (first_entity.entity_id, second_entity.entity_id),
            "reason": "规范名或别名直接重合",
        },)
        first = {"判定": [{
            "候选ID": "pair-1", "身份检查": "同一对象证据成立", "结论": "同一实体",
            "保留实体ID": first_entity.entity_id, "错误别名所属实体ID": "", "错误别名": "",
            "证据单元ID": ["u1", "u2"], "理由": "连续动作链确认同指", "置信度": 0.95,
        }]}
        second = {"判定": [{
            "候选ID": "pair-1", "身份检查": "无法判断", "结论": "证据不足",
            "保留实体ID": "", "错误别名所属实体ID": "", "错误别名": "",
            "证据单元ID": ["u2"], "理由": "未找到反证", "置信度": 0.8,
        }]}
        result = _entity_resolution_consensus(
            first, second, candidates=candidates, annotation=annotation,
        )["判定"][0]
        self.assertEqual(result["结论"], "同一实体")
        self.assertEqual(result["保留实体ID"], first_entity.entity_id)

    def test_proven_distinct_entities_promote_canonical_alias_collision(self) -> None:
        _, annotation = self._fixture()
        text = "中年人叫角色甲过去。"
        document = build_chapter_document(
            annotation.chapter_id, hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        unit_id = document.units[0].unit_id
        left, right = annotation.entities
        left = replace(
            left, canonical_name="角色甲", aliases=(),
            evidence=(SourceSpan(annotation.chapter_id, 4, 7, "角色甲", unit_id),),
        )
        right = replace(
            right, canonical_name="中年人", aliases=("角色甲",),
            evidence=(SourceSpan(annotation.chapter_id, 0, 3, "中年人", unit_id),),
        )
        annotation = replace(annotation, entities=(left, right))
        candidates = _entity_candidate_pairs(annotation)
        payload = {"判定": [{
            "候选ID": candidates[0]["candidate_id"],
            "身份检查": "明确为两个对象",
            "结论": "不同实体",
            "保留实体ID": "",
            "错误别名所属实体ID": "",
            "错误别名": "",
            "证据单元ID": [left.evidence[0].unit_id],
            "理由": "原文明确是说话者和接收者。",
            "置信度": 0.95,
        }]}

        row = _promote_distinct_alias_collisions(payload, candidates, annotation, document)["判定"][0]

        self.assertEqual(row["结论"], "别名错误")
        self.assertEqual(row["错误别名所属实体ID"], right.entity_id)
        self.assertEqual(row["错误别名"], "角色甲")

    def test_third_party_alias_proves_equivalent_relational_names(self) -> None:
        _, annotation = self._fixture()
        base, left = annotation.entities
        base = replace(base, canonical_name="王林", aliases=("铁柱",))
        left = replace(left, canonical_name="王林父亲", aliases=())
        right = replace(left, entity_id="entity-003", canonical_name="铁柱父亲")
        annotation = replace(annotation, entities=(base, left, right))
        candidates = _entity_candidate_pairs(annotation)
        candidate = next(
            item for item in candidates
            if set(item["entity_ids"]) == {left.entity_id, right.entity_id}
        )
        payload = {"判定": [{
            "候选ID": candidate["candidate_id"],
            "身份检查": "无法判断",
            "结论": "证据不足",
            "保留实体ID": "",
            "错误别名所属实体ID": "",
            "错误别名": "",
            "证据单元ID": [left.evidence[0].unit_id],
            "理由": "模型未能确认。",
            "置信度": 0.5,
        }]}

        row = _promote_third_party_alias_equivalences(
            payload, (candidate,), annotation,
        )["判定"][0]

        self.assertEqual(row["结论"], "同一实体")
        self.assertEqual(row["保留实体ID"], left.entity_id)
        self.assertGreaterEqual(row["置信度"], 0.95)

    def test_old_annotation_schema_is_rejected(self) -> None:
        _, annotation = self._fixture()
        stale = ChapterAnnotation.from_dict({**annotation.to_dict(), "schema_version": "2.1"})
        with self.assertRaisesRegex(ValueError, "schema-2.2"):
            _require_auditable_annotations([stale], __import__("pathlib").Path("sample.jsonl"))

    def test_checkpoint_changes_when_annotations_change(self) -> None:
        document, annotation = self._fixture()
        units = split_annotation_units(document, max_input_chars=1000, overlap_units=0)[0]
        packet, _, _ = _annotation_packet(annotation, units)
        folder = __import__("pathlib").Path("corpus")
        first = _checkpoint_path(folder, document, units, "model", packet)
        changed = {**packet, "audit_revision": 2}
        second = _checkpoint_path(folder, document, units, "model", changed)
        self.assertNotEqual(first, second)

    def test_random_sample_is_seeded_and_explicit_chapters_win(self) -> None:
        ids = [f"chapter-{index}" for index in range(1, 21)]
        self.assertEqual(_select_chapters(ids, 5, 7), _select_chapters(ids, 5, 7))
        self.assertEqual(_select_chapters(ids, 5, 7, (2, 5)), ["chapter-2", "chapter-5"])

    def test_candidate_audit_requires_safe_distinct_output_label(self) -> None:
        from pathlib import Path
        with self.assertRaisesRegex(ValueError, "requires output_label"):
            _resolve_annotation_artifact(Path("corpus"), 1, Path("candidate.jsonl"), "")
        with self.assertRaisesRegex(ValueError, "output_label"):
            _resolve_annotation_artifact(Path("corpus"), 1, None, "../bad")
        report, quality = _audit_output_paths(Path("corpus"), 5, "repair-candidate")
        self.assertEqual(report.name, "semantic_audit.repair-candidate.sample-5.json")
        self.assertEqual(quality.name, "semantic_audit.repair-candidate.sample-5.quality.json")

    def test_issue_review_rejects_verdict_reason_contradiction(self) -> None:
        from pipeline.contracts import SemanticAuditBatch, SemanticAuditIssue
        from pipeline.semantic_audit import _parse_issue_review_payload
        issue = SemanticAuditIssue(
            "issue-001", "fact_semantics", "error", ("fact-001",),
            ("chapter:p0001",), "事实主体错误。", "修正事实语义", 0.95,
        )
        batch = SemanticAuditBatch(
            "batch-001", "chapter", ("chapter:p0001",), (), ("fact-001",), (),
            "review", (issue,),
        )
        with self.assertRaisesRegex(ValueError, "相矛盾"):
            _parse_issue_review_payload({"判定": [{
                "问题ID": "issue-001", "结论": "误报",
                "对象确实矛盾": False, "已有语义覆盖": False, "剧情承载必要性": False,
                "理由": "诊断正确，主体错误，所以结论应为成立。", "置信度": 0.98,
            }]}, batch)
        with self.assertRaisesRegex(ValueError, "相矛盾"):
            _parse_issue_review_payload({"判定": [{
                "问题ID": "issue-001", "结论": "成立",
                "对象确实矛盾": True, "已有语义覆盖": False, "剧情承载必要性": False,
                "理由": "现有主体和客体方向正确，诊断错误，应判误报。", "置信度": 0.98,
            }]}, batch)
        with self.assertRaisesRegex(ValueError, "相矛盾"):
            _parse_issue_review_payload({"判定": [{
                "问题ID": "issue-001", "结论": "成立",
                "对象确实矛盾": True, "已有语义覆盖": False, "剧情承载必要性": False,
                "理由": "现有关系读作甲是乙的父亲；诊断称方向颠倒，错误。", "置信度": 0.98,
            }]}, batch)

    def test_coverage_review_requires_uncovered_persistent_or_causal_impact(self) -> None:
        from pipeline.contracts import SemanticAuditBatch, SemanticAuditIssue
        from pipeline.semantic_audit import _parse_issue_review_payload
        issue = SemanticAuditIssue(
            "issue-001", "coverage", "warning", ("missing:event:chapter:p0001",),
            ("chapter:p0001",), "回头动作缺失。", "重标当前窗口", 0.9,
        )
        batch = SemanticAuditBatch(
            "batch-001", "chapter", ("chapter:p0001",), (), (), (), "review", (issue,),
        )
        with self.assertRaisesRegex(ValueError, "剧情承载必要性"):
            _parse_issue_review_payload({"判定": [{
                "问题ID": "issue-001", "结论": "成立", "对象确实矛盾": False,
                "已有语义覆盖": False, "剧情承载必要性": False,
                "理由": "只是回头看了一眼。", "置信度": 0.9,
            }]}, batch)
        resolved = _parse_issue_review_payload({"判定": [{
            "问题ID": "issue-001", "结论": "误报", "对象确实矛盾": False,
            "已有语义覆盖": False, "剧情承载必要性": False,
            "理由": "瞬时动作没有持久状态或后续因果。", "置信度": 0.95,
        }]}, batch)
        self.assertTrue(resolved.passed)

    def test_coverage_review_accepts_completed_story_bearing_action(self) -> None:
        issue = SemanticAuditIssue(
            "issue-treatment", "coverage", "warning", ("missing:event:chapter:p0001",),
            ("chapter:p0001",), "人物完成施治，当前行动链没有对应事件。", "重标当前窗口", 0.95,
        )
        batch = SemanticAuditBatch(
            "batch-treatment", "chapter", ("chapter:p0001",), (), (), (), "review", (issue,),
        )
        reviewed = _parse_issue_review_payload({"判定": [{
            "问题ID": "issue-treatment", "结论": "成立", "对象确实矛盾": False,
            "已有语义覆盖": False, "剧情承载必要性": True,
            "理由": "原文明示完成治疗并改变本场人物处置，属于核心行动链。", "置信度": 0.96,
        }]}, batch)
        self.assertEqual(len(reviewed.issues), 1)

    def test_completed_treatment_cannot_be_rejected_as_scene_craft(self) -> None:
        issue = SemanticAuditIssue(
            "issue-treatment", "coverage", "warning", ("missing:event:chapter:p0001",),
            ("chapter:p0001",), "人物完成喂药治疗，当前事件集合完全缺失。", "重标当前窗口", 0.95,
        )
        batch = SemanticAuditBatch(
            "batch-treatment", "chapter", ("chapter:p0001",), (), (), (), "review", (issue,),
        )
        with self.assertRaisesRegex(ValueError, "核心动作不能"):
            _parse_issue_review_payload({"判定": [{
                "问题ID": "issue-treatment", "结论": "误报", "对象确实矛盾": False,
                "已有语义覆盖": False, "剧情承载必要性": False,
                "理由": "喂药只是场景技法，没有长期状态。", "置信度": 0.9,
            }]}, batch)

    def test_issue_review_scopes_coverage_to_author_skill_story_truth(self) -> None:
        from pipeline.contracts import SemanticAuditBatch, SemanticAuditIssue
        from pipeline.semantic_audit import _annotation_packet, _issue_review_prompt

        document, annotation = self._fixture()
        units = split_annotation_units(document, max_input_chars=1000, overlap_units=0)[0]
        packet, _, _ = _annotation_packet(annotation, units)
        issue = SemanticAuditIssue(
            "issue-001", "coverage", "warning", (f"missing:event:{units[0].unit_id}",),
            (units[0].unit_id,), "村民道贺与帮忙招待未形成事件。", "重标当前窗口", 0.9,
        )
        batch = SemanticAuditBatch(
            "batch-001", document.chapter_id, (units[0].unit_id,), (), (), (), "review", (issue,),
        )
        prompt = _issue_review_prompt(document, units, packet, batch)
        self.assertIn("蒸馏可迁移的作者写作 Skill", prompt)
        self.assertIn("不逐句抄写原文", prompt)
        self.assertIn("章节近似还原", prompt)
        self.assertIn("剧情承载必要性", prompt)
        self.assertIn("村民围观/道贺/赞叹/帮忙招待", prompt)
        self.assertIn("后续 fact_id/event_id", prompt)
        self.assertIn("已存在则 missing:event 误报", prompt)

    def test_issue_review_does_not_force_question_propositions_to_progression(self) -> None:
        from pipeline.contracts import SemanticAuditBatch, SemanticAuditIssue
        from pipeline.semantic_audit import _annotation_packet, _issue_review_prompt

        document, annotation = self._fixture()
        units = split_annotation_units(document, max_input_chars=1000, overlap_units=0)[0]
        packet, _, _ = _annotation_packet(annotation, units)
        issue = SemanticAuditIssue(
            "issue-question", "fact_semantics", "warning", (annotation.facts[0].fact_id,),
            (units[0].unit_id,), "询问是动作，kind必须改成progression。", "修正事实类型", 0.9,
        )
        batch = SemanticAuditBatch(
            "batch-question", document.chapter_id, (units[0].unit_id,), (),
            (annotation.facts[0].fact_id,), (), "review", (issue,),
        )
        prompt = _issue_review_prompt(document, units, packet, batch)
        self.assertIn("谓词为“询问/追问”的事实可用 information 或 knowledge", prompt)
        self.assertIn("不能仅因“询问是一个动作”", prompt)
        self.assertIn("别处旧标注", prompt)

    def test_entity_resolution_prefers_context_that_mentions_the_peer(self) -> None:
        from pipeline.contracts import Entity

        chapter_id = "Author/work-001/0001"
        text = "四叔答应带来书。\n中年汉子到了门外。\n父亲说：快给你四叔拿凳子，中年汉子点了点头。"
        document = build_chapter_document(
            chapter_id, hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        first, second, joint = document.units
        uncle = Entity(
            f"{chapter_id}:entity-001", "四叔", "person", (),
            (
                SourceSpan(chapter_id, first.start, first.start + 2, "四叔", first.unit_id),
                SourceSpan(chapter_id, joint.start + 8, joint.start + 10, "四叔", joint.unit_id),
            ),
        )
        man = Entity(
            f"{chapter_id}:entity-002", "中年汉子", "person", (),
            (
                SourceSpan(chapter_id, second.start, second.start + 4, "中年汉子", second.unit_id),
                SourceSpan(chapter_id, joint.start + 14, joint.start + 18, "中年汉子", joint.unit_id),
            ),
        )
        annotation = replace(self._fixture()[1], chapter_id=chapter_id, source_hash=document.source_hash, entities=(uncle, man))
        prompt, source_ids = _entity_resolution_prompt(
            document,
            annotation,
            ({"candidate_id": "entity-pair-001", "entity_ids": (uncle.entity_id, man.entity_id), "reason": "称谓候选"},),
        )
        self.assertIn(joint.unit_id, source_ids)
        self.assertIn("快给你四叔拿凳子，中年汉子点了点头", prompt)

    def test_prompts_share_relationship_and_speech_object_semantics(self) -> None:
        from pipeline.semantic_audit import _audit_prompt, _issue_review_prompt
        from pipeline.semantic_repair import _repair_prompt

        document, annotation = self._fixture()
        units = split_annotation_units(document, max_input_chars=1000, overlap_units=0)[0]
        packet, expected, _ = _annotation_packet(annotation, units)
        audit = _audit_prompt(document, units, packet, expected)
        issue = SemanticAuditIssue(
            "issue-001", "fact_semantics", "error", (annotation.facts[0].fact_id,),
            (units[0].unit_id,), "主体错误。", "修正事实语义", 0.9,
        )
        batch = SemanticAuditBatch(
            "batch-001", document.chapter_id, (units[0].unit_id,), (),
            (annotation.facts[0].fact_id,), (), "review", (issue,),
        )
        review = _issue_review_prompt(document, units, packet, batch)
        repair = _repair_prompt(document, units, packet, [issue.to_dict()])
        for prompt in (audit, review, repair):
            self.assertIn("subject", prompt)
            self.assertIn("object", prompt)
            self.assertIn("听者", prompt)
            self.assertIn("certainty", prompt)
            self.assertIn("提取置信度", prompt)
        self.assertIn("subject 相对于 object", audit)
        self.assertIn("subject 相对于 object", repair)
        self.assertIn("这是你弟弟王林", review)
        self.assertIn("subject=王林", review)
        self.assertIn("knowledge", audit)
        self.assertIn("获知者", review)
        self.assertIn("全章事件索引", audit)


if __name__ == "__main__":
    unittest.main()
