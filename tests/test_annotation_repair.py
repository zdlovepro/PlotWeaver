from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace

from pipeline.annotation_repair import AnnotationPatchError, apply_semantic_patch
from pipeline.contracts import ChapterAnnotation, Entity, EventAtom, Fact, SceneCard, SourceSpan, StateChange, build_chapter_document
from pipeline.spatiotemporal import enrich_spatiotemporal


class AnnotationRepairTests(unittest.TestCase):
    def _fixture(self):
        chapter_id = "Author/work-001/0001"
        text = "角色甲又名甲师兄，把令牌交给角色乙。\n角色乙决定保管令牌。"
        document = build_chapter_document(chapter_id, hashlib.sha256(text.encode("utf-8")).hexdigest(), text)
        first, second = document.units

        def span(unit, quote):
            offset = unit.text.index(quote)
            return SourceSpan(chapter_id, unit.start + offset, unit.start + offset + len(quote), quote, unit.unit_id)

        e1 = Entity("entity-001", "角色甲", "person", (), (span(first, "角色甲"),))
        e2 = Entity("entity-002", "角色乙", "person", (), (span(first, "角色乙"),))
        e3 = Entity("entity-003", "甲师兄", "person", ("角色甲",), (span(first, "甲师兄"),))
        f1 = Fact("fact-001", chapter_id, "resource", "entity-002", "获得", "令牌", "", 1, (span(first, "把令牌交给角色乙"),))
        v1 = EventAtom(
            "event-001", chapter_id, 0, "角色甲交付令牌", ("entity-003", "entity-002"),
            (), (), "交付令牌", "", "", ("fact-001",), (), (span(first, "把令牌交给角色乙"),),
            "transfer", "entity-003", ("entity-002",), ("fact-001",),
        )
        scene = SceneCard(
            "scene-001", chapter_id, 0, ("entity-003", "entity-002"), ("event-001",),
            "交付令牌", (), ("fact-001",), 1, v1.evidence,
        )
        change = StateChange("change-001", chapter_id, "event-001", "add", "", "fact-001")
        annotation = enrich_spatiotemporal(document, ChapterAnnotation(
            chapter_id, document.source_hash, (e1, e2, e3), (f1,), (v1,), (scene,), (change,), "2.2",
        ))
        return document, annotation

    def _base_patch(self):
        return {
            "resolved_issue_ids": ["issue-identity", "issue-decision"],
            "deferred_issues": [],
            "merge_entities": [{
                "keep_entity_id": "entity-001",
                "merge_entity_ids": ["entity-003"],
                "evidence_unit_ids": ["Author/work-001/0001:p0000"],
            }],
            "upsert_entities": [],
            "remove_fact_ids": [],
            "upsert_facts": [{
                "id": "new-fact-decision",
                "kind": "goal",
                "subject_id": "entity-002",
                "predicate": "决定",
                "value": "保管令牌",
                "object_id": "",
                "certainty": 1.0,
                "evidence": [{"unit_id": "Author/work-001/0001:p0001", "quote": "决定保管令牌"}],
            }],
            "remove_event_ids": [],
            "upsert_events": [{
                "id": "new-event-decision",
                "summary": "角色乙决定保管令牌",
                "action_type": "decision",
                "actor_id": "entity-002",
                "target_ids": [],
                "participant_ids": ["entity-002"],
                "trigger_fact_ids": ["fact-001"],
                "precondition_fact_ids": [],
                "basis_fact_ids": ["new-fact-decision"],
                "action": "决定保管令牌",
                "obstacle": "",
                "decision": "保管令牌",
                "outcome_fact_ids": ["new-fact-decision"],
                "cost_fact_ids": [],
                "evidence": [{"unit_id": "Author/work-001/0001:p0001", "quote": "决定保管令牌"}],
            }],
        }

    def test_patch_merges_entities_allocates_ids_and_rebuilds_derived_relations(self) -> None:
        document, annotation = self._fixture()
        repaired, summary = apply_semantic_patch(
            document, annotation, self._base_patch(),
            expected_issue_ids=("issue-identity", "issue-decision"),
        )
        self.assertEqual(len(repaired.entities), 2)
        self.assertEqual(repaired.events[0].actor_id, "entity-001")
        self.assertIn("甲师兄", repaired.entities[0].aliases)
        self.assertEqual(len(repaired.facts), 2)
        self.assertEqual(len(repaired.events), 2)
        new_event = next(item for item in repaired.events if item.event_id != "event-001")
        self.assertEqual(new_event.action_type, "decision")
        self.assertTrue(any(item.event_id == new_event.event_id for item in repaired.state_changes))
        self.assertEqual(summary["entity_redirects"], {"entity-003": "entity-001"})

    def test_patch_rejects_non_source_quote(self) -> None:
        document, annotation = self._fixture()
        patch = self._base_patch()
        patch["upsert_facts"][0]["evidence"][0]["quote"] = "原文不存在"
        with self.assertRaisesRegex(AnnotationPatchError, "证据预检失败.*不是原文单元"):
            apply_semantic_patch(document, annotation, patch)

    def test_standalone_coverage_fact_may_use_adjacent_unit_in_bounded_repair_window(self) -> None:
        chapter_id = "Author/work-001/0002"
        text = "乙听到结果，如坠冰窟。\n他耳边轰鸣，站在原地。"
        document = build_chapter_document(
            chapter_id, hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )
        first, second = document.units
        start = first.start + first.text.index("乙")
        person = Entity(
            "entity-person", "乙", "person", (),
            (SourceSpan(chapter_id, start, start + 1, "乙", first.unit_id),),
        )
        seed_fact = Fact(
            "fact-seed", chapter_id, "progression", person.entity_id,
            "听到", "结果", "", 1.0,
            (SourceSpan(chapter_id, start, first.end, first.text, first.unit_id),),
        )
        seed_event = EventAtom(
            "event-seed", chapter_id, 0, "乙听到结果", (person.entity_id,),
            (), (), "听到结果", "", "", (seed_fact.fact_id,), (), seed_fact.evidence,
            "other", person.entity_id, (), (seed_fact.fact_id,),
        )
        seed_scene = SceneCard(
            "scene-seed", chapter_id, 0, (person.entity_id,), (seed_event.event_id,),
            "乙听到结果", (), (seed_fact.fact_id,), 1.0, seed_fact.evidence,
        )
        seed_change = StateChange(
            "change-seed", chapter_id, seed_event.event_id, "add", "", seed_fact.fact_id,
        )
        annotation = ChapterAnnotation(
            chapter_id, document.source_hash, (person,), (seed_fact,), (seed_event,),
            (seed_scene,), (seed_change,), "2.2",
        )
        patch = {
            "resolved_issue_ids": ["issue-emotion"], "deferred_issues": [],
            "merge_entities": [], "upsert_entities": [], "remove_fact_ids": [],
            "upsert_facts": [{
                "id": "new-fact-emotion", "kind": "emotion", "subject_id": "entity-person",
                "predicate": "感到", "value": "如坠冰窟", "object_id": "", "certainty": 1.0,
                "evidence": [{"unit_id": first.unit_id, "quote": "如坠冰窟"}],
            }],
            "remove_event_ids": [], "upsert_events": [],
        }
        descriptor = {
            "issue_id": "issue-emotion", "category": "coverage",
            "object_ids": [f"missing:fact:{second.unit_id}"],
            "evidence_unit_ids": [second.unit_id],
        }
        with self.assertRaisesRegex(AnnotationPatchError, "既未被本次事件引用"):
            apply_semantic_patch(
                document, annotation, patch,
                expected_issue_ids=("issue-emotion",), issue_descriptors=(descriptor,),
            )
        repaired, _ = apply_semantic_patch(
            document, annotation, patch,
            expected_issue_ids=("issue-emotion",), issue_descriptors=(descriptor,),
            allowed_evidence_unit_ids=(first.unit_id, second.unit_id),
        )
        self.assertTrue(any(item.value == "如坠冰窟" for item in repaired.facts))

    def test_patch_rejects_dangling_fact_removal(self) -> None:
        document, annotation = self._fixture()
        patch = self._base_patch()
        patch["remove_fact_ids"] = ["fact-001"]
        with self.assertRaisesRegex(AnnotationPatchError, "待删除事实|未知事实|引用悬空"):
            apply_semantic_patch(document, annotation, patch)

    def test_patch_cleans_deleted_context_fact_but_not_core_fact(self) -> None:
        document, annotation = self._fixture()
        trigger = Fact(
            "fact-trigger", document.chapter_id, "information", "entity-001",
            "准备", "交付", "", 1.0, annotation.events[0].evidence,
        )
        original_event = replace(annotation.events[0], trigger_fact_ids=(trigger.fact_id,))
        annotation = replace(
            annotation,
            facts=(*annotation.facts, trigger),
            events=(original_event,),
            scenes=(replace(annotation.scenes[0], entry_fact_ids=(trigger.fact_id,)),),
        )
        patch = self._base_patch()
        patch["remove_fact_ids"] = [trigger.fact_id]
        repaired, summary = apply_semantic_patch(document, annotation, patch)
        kept_event = next(item for item in repaired.events if item.event_id == "event-001")
        self.assertEqual(kept_event.trigger_fact_ids, ())
        self.assertEqual(len(summary["normalized_removed_fact_refs"]), 1)

    def test_every_issue_must_be_resolved_or_deferred_once(self) -> None:
        document, annotation = self._fixture()
        patch = self._base_patch()
        patch["resolved_issue_ids"] = ["issue-identity"]
        with self.assertRaisesRegex(AnnotationPatchError, "每个待处理问题"):
            apply_semantic_patch(
                document, annotation, patch,
                expected_issue_ids=("issue-identity", "issue-decision"),
            )

    def test_resolved_issue_cannot_use_empty_patch(self) -> None:
        document, annotation = self._fixture()
        patch = self._base_patch()
        for key in ("merge_entities", "upsert_entities", "remove_fact_ids", "upsert_facts", "remove_event_ids", "upsert_events"):
            patch[key] = []
        with self.assertRaisesRegex(AnnotationPatchError, "至少包含一个定点修改"):
            apply_semantic_patch(document, annotation, patch)

    def test_actor_and_targets_are_deterministically_included_as_participants(self) -> None:
        document, annotation = self._fixture()
        patch = self._base_patch()
        patch["upsert_events"][0]["target_ids"] = ["entity-001"]
        patch["upsert_events"][0]["participant_ids"] = ["entity-002"]
        repaired, _ = apply_semantic_patch(document, annotation, patch)
        new_event = next(item for item in repaired.events if item.event_id != "event-001")
        self.assertEqual(new_event.participant_ids, ("entity-002", "entity-001"))

    def test_existing_scene_inherits_participants_added_by_event_repair(self) -> None:
        document, annotation = self._fixture()
        patch = self._base_patch()
        patch["resolved_issue_ids"] = ["issue-event"]
        patch["merge_entities"] = []
        patch["upsert_facts"] = []
        patch["upsert_events"] = [{
            "id": "event-001",
            "summary": "角色甲把令牌交给角色乙",
            "action_type": "transfer",
            "actor_id": "entity-003",
            "target_ids": ["entity-002", "entity-001"],
            "participant_ids": ["entity-003", "entity-002"],
            "trigger_fact_ids": [],
            "precondition_fact_ids": [],
            "basis_fact_ids": ["fact-001"],
            "action": "交付令牌",
            "obstacle": "",
            "decision": "",
            "outcome_fact_ids": ["fact-001"],
            "cost_fact_ids": [],
            "evidence": [{
                "unit_id": "Author/work-001/0001:p0000",
                "quote": "把令牌交给角色乙",
            }],
        }]
        repaired, _ = apply_semantic_patch(
            document,
            annotation,
            patch,
            expected_issue_ids=("issue-event",),
            issue_descriptors=({
                "issue_id": "issue-event",
                "category": "event_frame",
                "object_ids": ["event-001"],
            },),
        )
        scene = next(item for item in repaired.scenes if "event-001" in item.event_ids)
        self.assertIn("entity-001", scene.participant_ids)

    def test_patch_rejects_location_fact_pointing_to_non_location_entity(self) -> None:
        document, annotation = self._fixture()
        patch = self._base_patch()
        patch["upsert_facts"][0].update({
            "kind": "location",
            "object_id": "entity-001",
            "value": "角色甲身边",
        })
        with self.assertRaisesRegex(AnnotationPatchError, "必须引用 location 实体"):
            apply_semantic_patch(document, annotation, patch)

    def test_patch_rejects_relationship_without_other_party(self) -> None:
        document, annotation = self._fixture()
        patch = self._base_patch()
        patch["upsert_facts"][0].update({
            "kind": "relationship",
            "object_id": "",
            "value": "同伴",
        })
        with self.assertRaisesRegex(AnnotationPatchError, "关系另一方"):
            apply_semantic_patch(document, annotation, patch)

    def test_patch_rejects_action_incompatible_basis_or_outcome_fact(self) -> None:
        document, annotation = self._fixture()
        patch = self._base_patch()
        patch["upsert_events"][0]["action_type"] = "movement"
        with self.assertRaisesRegex(AnnotationPatchError, "没有合法的直接结果"):
            apply_semantic_patch(document, annotation, patch)

    def test_patch_rejects_carried_participant_missing_from_targets(self) -> None:
        document, annotation = self._fixture()
        patch = self._base_patch()
        patch["upsert_events"][0].update({
            "summary": "角色甲带着角色乙离开",
            "action": "带着角色乙离开",
            "action_type": "other",
            "actor_id": "entity-001",
            "target_ids": [],
            "participant_ids": ["entity-001", "entity-002"],
        })
        with self.assertRaisesRegex(AnnotationPatchError, "未进入 target_ids"):
            apply_semantic_patch(document, annotation, patch)

    def test_patch_drops_only_extra_action_incompatible_outcome(self) -> None:
        document, annotation = self._fixture()
        patch = self._base_patch()
        patch["upsert_facts"].append({
            "id": "new-fact-identity",
            "kind": "identity",
            "subject_id": "entity-002",
            "predicate": "身份",
            "value": "保管者",
            "object_id": "",
            "certainty": 1.0,
            "evidence": [{"unit_id": "Author/work-001/0001:p0001", "quote": "角色乙决定保管令牌"}],
        })
        patch["upsert_events"][0]["basis_fact_ids"].append("new-fact-identity")
        patch["upsert_events"][0]["outcome_fact_ids"].append("new-fact-identity")
        patch["upsert_events"][0]["precondition_fact_ids"].append("new-fact-identity")
        repaired, summary = apply_semantic_patch(document, annotation, patch)
        new_event = next(item for item in repaired.events if item.event_id != "event-001")
        self.assertEqual(len(new_event.outcome_fact_ids), 1)
        self.assertEqual(len(summary["normalized_event_fact_refs"]), 1)

    def test_patch_drops_new_fact_made_orphan_by_type_normalization(self) -> None:
        document, annotation = self._fixture()
        patch = self._base_patch()
        patch["upsert_facts"].append({
            "id": "new-fact-identity",
            "kind": "identity",
            "subject_id": "entity-002",
            "predicate": "身份",
            "value": "保管者",
            "object_id": "",
            "certainty": 1.0,
            "evidence": [{"unit_id": "Author/work-001/0001:p0001", "quote": "角色乙决定保管令牌"}],
        })
        patch["upsert_events"][0]["basis_fact_ids"].append("new-fact-identity")
        patch["upsert_events"][0]["outcome_fact_ids"].append("new-fact-identity")
        repaired, summary = apply_semantic_patch(document, annotation, patch)
        dropped = summary["normalized_dropped_new_fact_ids"]
        self.assertEqual(len(dropped), 1)
        self.assertNotIn(dropped[0], {item.fact_id for item in repaired.facts})

    def test_patch_rejects_new_fact_unrelated_to_resolved_issue(self) -> None:
        document, annotation = self._fixture()
        patch = self._base_patch()
        patch["upsert_facts"].append({
            "id": "new-fact-comment",
            "kind": "information",
            "subject_id": "entity-002",
            "predicate": "评论",
            "value": "令牌",
            "object_id": "",
            "certainty": 1.0,
            "evidence": [{"unit_id": "Author/work-001/0001:p0001", "quote": "角色乙决定保管令牌"}],
        })
        with self.assertRaisesRegex(AnnotationPatchError, "不得顺手扩写"):
            apply_semantic_patch(
                document,
                annotation,
                patch,
                expected_issue_ids=("issue-identity", "issue-decision"),
                issue_descriptors=(
                    {"issue_id": "issue-identity", "category": "entity_identity", "object_ids": ["entity-003"], "evidence_unit_ids": ["Author/work-001/0001:p0000"]},
                    {"issue_id": "issue-decision", "category": "coverage", "object_ids": ["missing:event:Author/work-001/0001:p0001"], "evidence_unit_ids": ["Author/work-001/0001:p0001"]},
                ),
            )

    def test_new_fact_cannot_duplicate_an_existing_proposition_at_same_evidence(self) -> None:
        document, annotation = self._fixture()
        existing = Fact(
            "fact-existing-decision", document.chapter_id, "knowledge", "entity-002",
            "打算", "保管令牌", "", 1.0, annotation.events[0].evidence,
        )
        annotation = replace(annotation, facts=(*annotation.facts, existing))
        patch = self._base_patch()
        patch["upsert_facts"][0]["evidence"] = [{
            "unit_id": "Author/work-001/0001:p0000",
            "quote": "令牌",
        }]
        with self.assertRaisesRegex(AnnotationPatchError, "表达相同命题"):
            apply_semantic_patch(document, annotation, patch)

    def test_entity_identity_issue_does_not_authorize_standalone_fact(self) -> None:
        document, annotation = self._fixture()
        patch = self._base_patch()
        patch["resolved_issue_ids"] = ["issue-identity"]
        patch["deferred_issues"] = [{"issue_id": "issue-decision", "reason": "证据不足，延期处理"}]
        patch["upsert_facts"] = [{
            "id": "new-fact-alias",
            "kind": "identity",
            "subject_id": "entity-001",
            "predicate": "别名",
            "value": "甲师兄",
            "object_id": "",
            "certainty": 1.0,
            "evidence": [{"unit_id": "Author/work-001/0001:p0000", "quote": "甲师兄"}],
        }]
        patch["upsert_events"] = []
        with self.assertRaisesRegex(AnnotationPatchError, "entity_identity"):
            apply_semantic_patch(
                document,
                annotation,
                patch,
                expected_issue_ids=("issue-identity", "issue-decision"),
                issue_descriptors=(
                    {"issue_id": "issue-identity", "category": "entity_identity", "object_ids": ["entity-003"], "evidence_unit_ids": ["Author/work-001/0001:p0000"]},
                    {"issue_id": "issue-decision", "category": "coverage", "object_ids": ["missing:event:Author/work-001/0001:p0001"], "evidence_unit_ids": ["Author/work-001/0001:p0001"]},
                ),
            )

    def test_patch_rejects_identity_fact_that_disguises_entity_relation(self) -> None:
        document, annotation = self._fixture()
        patch = self._base_patch()
        patch["upsert_facts"][0].update({
            "kind": "identity",
            "object_id": "entity-001",
            "value": "角色甲的同伴",
        })
        with self.assertRaisesRegex(AnnotationPatchError, "身份事实不得"):
            apply_semantic_patch(document, annotation, patch)

    def test_patch_limits_new_events_to_resolved_missing_event_issues(self) -> None:
        document, annotation = self._fixture()
        patch = self._base_patch()
        patch["upsert_events"].append({
            **patch["upsert_events"][0],
            "id": "new-event-extra",
        })
        with self.assertRaisesRegex(AnnotationPatchError, "要求恰好净新增 1 个核心事件"):
            apply_semantic_patch(
                document,
                annotation,
                patch,
                expected_issue_ids=("issue-identity", "issue-decision"),
                issue_descriptors=(
                    {"issue_id": "issue-identity", "category": "entity_identity", "object_ids": ["entity-003"]},
                    {"issue_id": "issue-decision", "category": "coverage", "object_ids": ["missing:event:Author/work-001/0001:p0001"]},
                ),
            )

    def test_new_event_may_replace_related_old_event_without_counting_as_coverage(self) -> None:
        document, annotation = self._fixture()
        patch = self._base_patch()
        patch["resolved_issue_ids"] = ["issue-event"]
        patch["merge_entities"] = []
        patch["upsert_facts"] = []
        patch["remove_event_ids"] = ["event-001"]
        patch["upsert_events"] = [{
            "id": "new-event-transfer-replacement",
            "summary": "角色甲把令牌交给角色乙",
            "action_type": "transfer",
            "actor_id": "entity-003",
            "target_ids": ["entity-002"],
            "participant_ids": ["entity-003", "entity-002"],
            "trigger_fact_ids": [],
            "precondition_fact_ids": [],
            "basis_fact_ids": ["fact-001"],
            "action": "交付令牌",
            "obstacle": "",
            "decision": "",
            "outcome_fact_ids": ["fact-001"],
            "cost_fact_ids": [],
            "evidence": [{
                "unit_id": "Author/work-001/0001:p0000",
                "quote": "把令牌交给角色乙",
            }],
        }]
        repaired, _ = apply_semantic_patch(
            document,
            annotation,
            patch,
            expected_issue_ids=("issue-event",),
            issue_descriptors=({
                "issue_id": "issue-event",
                "category": "event_frame",
                "object_ids": ["event-001"],
                "evidence_unit_ids": ["Author/work-001/0001:p0000"],
            },),
        )
        self.assertEqual(len(repaired.events), 1)
        self.assertNotEqual(repaired.events[0].event_id, "event-001")

    def test_compound_event_issue_may_split_into_exactly_two_atomic_events(self) -> None:
        document, annotation = self._fixture()
        patch = self._base_patch()
        patch["resolved_issue_ids"] = ["event-001:compound-core-action"]
        patch["deferred_issues"] = []
        patch["merge_entities"] = []
        patch["upsert_facts"] = [patch["upsert_facts"][0]]
        patch["upsert_events"] = [
            {
                "id": "event-001", "summary": "角色甲交付令牌", "action_type": "transfer",
                "actor_id": "entity-003", "target_ids": ["entity-002"],
                "participant_ids": ["entity-003", "entity-002"], "trigger_fact_ids": [],
                "precondition_fact_ids": [], "basis_fact_ids": ["fact-001"],
                "action": "交付令牌", "obstacle": "", "decision": "",
                "outcome_fact_ids": ["fact-001"], "cost_fact_ids": [],
                "evidence": [{"unit_id": "Author/work-001/0001:p0000", "quote": "把令牌交给角色乙"}],
            },
            patch["upsert_events"][0],
        ]

        repaired, _ = apply_semantic_patch(
            document, annotation, patch,
            expected_issue_ids=("event-001:compound-core-action",),
            issue_descriptors=({
                "issue_id": "event-001:compound-core-action", "category": "event_frame",
                "object_ids": ["event-001"], "evidence_unit_ids": [
                    "Author/work-001/0001:p0000", "Author/work-001/0001:p0001",
                ],
                "diagnosis": "一个事件捆绑多个核心动作，必须拆成原子事件。",
            },),
        )

        self.assertEqual(len(repaired.events), 2)
        self.assertEqual({item.action_type for item in repaired.events}, {"transfer", "decision"})

    def test_missing_clue_coverage_may_add_a_standalone_fact(self) -> None:
        document, annotation = self._fixture()
        patch = self._base_patch()
        patch["resolved_issue_ids"] = ["issue-clue"]
        patch["merge_entities"] = []
        patch["upsert_events"] = []
        patch["upsert_facts"] = [{
            "id": "new-fact-promise",
            "kind": "information",
            "subject_id": "entity-002",
            "predicate": "获知",
            "value": "角色乙决定保管令牌",
            "object_id": "",
            "certainty": 1.0,
            "evidence": [{
                "unit_id": "Author/work-001/0001:p0001",
                "quote": "决定保管令牌",
            }],
        }]
        repaired, _ = apply_semantic_patch(
            document,
            annotation,
            patch,
            expected_issue_ids=("issue-clue",),
            issue_descriptors=({
                "issue_id": "issue-clue",
                "category": "coverage",
                "object_ids": ["missing:clue:Author/work-001/0001:p0001"],
                "evidence_unit_ids": ["Author/work-001/0001:p0001"],
            },),
        )
        self.assertTrue(any(item.value == "角色乙决定保管令牌" for item in repaired.facts))

    def test_movement_keeps_affected_person_separate_from_location_outcome(self) -> None:
        document, annotation = self._fixture()
        patch = self._base_patch()
        patch["upsert_entities"] = [{
            "id": "new-entity-place",
            "canonical_name": "令牌存放处",
            "kind": "location",
            "aliases": [],
            "evidence": [{"unit_id": "Author/work-001/0001:p0001", "quote": "保管令牌"}],
        }]
        patch["upsert_facts"][0].update({
            "kind": "location",
            "predicate": "到达",
            "value": "令牌存放处",
            "object_id": "new-entity-place",
        })
        patch["upsert_events"][0].update({
            "action_type": "movement",
            "target_ids": ["entity-001"],
            "participant_ids": ["entity-002"],
            "action": "到达令牌存放处",
            "decision": "",
        })
        repaired, summary = apply_semantic_patch(document, annotation, patch)
        new_event = next(item for item in repaired.events if item.event_id != "event-001")
        place_id = summary["allocated_entity_ids"]["new-entity-place"]
        self.assertEqual(new_event.target_ids, ("entity-001",))
        self.assertNotIn(place_id, new_event.target_ids)
        self.assertEqual(summary["normalized_event_targets"], [])

    def test_event_target_rejects_location_entity(self) -> None:
        document, annotation = self._fixture()
        patch = self._base_patch()
        patch["upsert_entities"] = [{
            "id": "new-entity-place", "canonical_name": "存放处", "kind": "location",
            "aliases": [], "evidence": [{
                "unit_id": "Author/work-001/0001:p0001", "quote": "保管令牌",
            }],
        }]
        patch["upsert_events"][0]["target_ids"] = ["new-entity-place"]
        with self.assertRaisesRegex(AnnotationPatchError, "不得把地点当受事者"):
            apply_semantic_patch(document, annotation, patch)

    def test_location_fact_rejects_same_type_but_wrong_place_binding(self) -> None:
        document, annotation = self._fixture()
        patch = self._base_patch()
        patch["upsert_entities"] = [{
            "id": "new-entity-wrong-place", "canonical_name": "山峰", "kind": "location",
            "aliases": [], "evidence": [{
                "unit_id": "Author/work-001/0001:p0000", "quote": "令牌交给角色乙",
            }],
        }]
        patch["upsert_facts"][0].update({
            "kind": "location", "predicate": "朝向", "value": "走向石阶",
            "object_id": "new-entity-wrong-place",
        })
        patch["upsert_events"][0].update({
            "action_type": "movement", "action": "走向石阶", "decision": "",
        })
        with self.assertRaisesRegex(AnnotationPatchError, "相邻地点错绑"):
            apply_semantic_patch(document, annotation, patch)

    def test_chinese_local_placeholder_is_allocated_to_stable_numeric_id(self) -> None:
        document, annotation = self._fixture()
        patch = self._base_patch()
        patch["upsert_facts"][0]["id"] = "new-fact-保管决定"
        for key in ("basis_fact_ids", "outcome_fact_ids"):
            patch["upsert_events"][0][key] = ["new-fact-保管决定"]
        patch["upsert_events"][0]["id"] = "new-event-决定保管"
        repaired, summary = apply_semantic_patch(document, annotation, patch)
        self.assertRegex(summary["allocated_fact_ids"]["new-fact-保管决定"], r":fact-\d{3}$")
        self.assertRegex(summary["allocated_event_ids"]["new-event-决定保管"], r":event-\d{3}$")
        allocated = summary["allocated_event_ids"]["new-event-决定保管"]
        self.assertTrue(any(item.event_id == allocated for item in repaired.events))


if __name__ == "__main__":
    unittest.main()
