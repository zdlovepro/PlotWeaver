from __future__ import annotations

import unittest
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from pipeline.annotate import ANNOTATION_CONTRACT_REVISION, AnnotationSourceSlice, FACT_KIND_MAP, apply_atomic_patch, apply_fact_inventory_patch, assess_annotation_quality, atomic_annotation_prompt, atomic_event_overlay_prompt, atomic_fact_inventory_patch_prompt, atomic_fact_inventory_prompt, atomic_fact_inventory_reconciliation_prompt, atomic_patch_prompt, atomic_semantic_patch_verification_prompt, atomic_semantic_reconciliation_prompt, _allowed_fact_evidence_gap, _collapse_duplicate_atoms, _derive_scene_state_from_atoms, _event_frame_issues, _explicit_local_speaker_ids, _is_overlapping_event_duplicate, _largest_evidence_gap, _load_chunk_checkpoint, _merge_event_overlay, _needs_atomic_semantic_patch, _normalize_action_result_fact_kinds, _normalize_reporting_event_fact_roles, _parse_atom_semantic_reconciliation, _parse_fact_inventory_reconciliation, _request_annotation_json, _require_atom_draft, _require_fact_inventory, _require_scene_state_draft, annotation_from_draft, enrich_entity_evidence, merge_annotations, normalize_carried_event_targets, normalize_fact_role_contracts, normalize_location_fact_objects, sanitize_draft
from pipeline.contracts import ChapterAnnotation, SourceSpan, build_chapter_document
from pipeline.model import ModelServiceError, ModelSettings


class AnnotationStagingTests(unittest.TestCase):
    def test_service_failure_does_not_enter_json_syntax_retry(self) -> None:
        settings = ModelSettings(api_key="placeholder")
        diagnostics: list[dict[str, object]] = []
        with patch(
            "pipeline.annotate.complete_json",
            side_effect=ModelServiceError("模型服务返回 HTTP 402: Insufficient Balance"),
        ) as completion, self.assertRaisesRegex(ModelServiceError, "402"):
            _request_annotation_json(
                "系统", "任务", settings, diagnostics=diagnostics,
            )
        self.assertEqual(completion.call_count, 1)
        self.assertEqual(len(diagnostics), 1)

    def test_sanitize_drops_only_unreferenced_ungrounded_location_fact(self) -> None:
        text = "角色甲看着眼前景象，怔在当场。"
        document = build_chapter_document(
            "Author/work/0028", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        units = (AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text),)
        payload = {
            "entities": [
                {"id": "person", "name": "角色甲", "kind": "人物", "aliases": [],
                 "evidence": [{"unit_id": unit.unit_id, "quote": "角色甲"}]},
                {"id": "place", "name": "试炼广场", "kind": "地点", "aliases": ["此处"],
                 "evidence": [{"unit_id": unit.unit_id, "quote": "当场"}]},
            ],
            "facts": [{"id": "invented-place", "kind": "地点", "subject_id": "person",
                       "predicate": "位于", "value": "", "object_id": "place", "certainty": "明确",
                       "evidence": [{"unit_id": unit.unit_id, "quote": "怔在当场"}]},
                      {"id": "perception", "kind": "认知", "subject_id": "person",
                       "predicate": "看到", "value": "眼前景象", "object_id": "", "certainty": "明确",
                       "evidence": [{"unit_id": unit.unit_id, "quote": "看着眼前景象"}]}],
            "events": [{"id": "look", "order": 1, "summary": "角色甲看到眼前景象", "action_type": "感知",
                        "actor_id": "person", "target_ids": [], "basis_fact_ids": ["perception"],
                        "participant_ids": ["person"], "trigger_fact_ids": [], "precondition_fact_ids": [],
                        "action": "看到眼前景象", "obstacle": "", "decision": "",
                        "outcome_fact_ids": ["perception"], "cost_fact_ids": [],
                        "evidence": [{"unit_id": unit.unit_id, "quote": "看着眼前景象"}]}],
            "scenes": [], "state_changes": [],
        }

        cleaned = sanitize_draft(document, payload, units)

        self.assertEqual([fact["id"] for fact in cleaned["facts"]], ["perception"])

    def test_sanitize_keeps_referenced_ungrounded_location_for_strict_repair(self) -> None:
        text = "角色甲看着眼前景象，怔在当场。"
        document = build_chapter_document(
            "Author/work/0029", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        units = (AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text),)
        payload = {
            "entities": [
                {"id": "person", "name": "角色甲", "kind": "人物", "aliases": [],
                 "evidence": [{"unit_id": unit.unit_id, "quote": "角色甲"}]},
                {"id": "place", "name": "试炼广场", "kind": "地点", "aliases": ["此处"],
                 "evidence": [{"unit_id": unit.unit_id, "quote": "当场"}]},
            ],
            "facts": [{"id": "invented-place", "kind": "地点", "subject_id": "person",
                       "predicate": "位于", "value": "", "object_id": "place", "certainty": "明确",
                       "evidence": [{"unit_id": unit.unit_id, "quote": "怔在当场"}]}],
            "events": [{"id": "move", "order": 1, "summary": "角色甲位于试炼广场", "action_type": "移动",
                        "actor_id": "person", "target_ids": [], "basis_fact_ids": ["invented-place"],
                        "participant_ids": ["person"], "trigger_fact_ids": [], "precondition_fact_ids": [],
                        "action": "位于试炼广场", "obstacle": "", "decision": "",
                        "outcome_fact_ids": ["invented-place"], "cost_fact_ids": [],
                        "evidence": [{"unit_id": unit.unit_id, "quote": "怔在当场"}]}],
            "scenes": [], "state_changes": [],
        }

        cleaned = sanitize_draft(document, payload, units)

        self.assertEqual([fact["id"] for fact in cleaned["facts"]], ["invented-place"])

    def test_checkpoint_requires_semantic_reconciliation_proof(self) -> None:
        text = "角色甲走到门前。\n远处也有一扇门。"
        document = build_chapter_document(
            "Author/work/0024", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        first, second = document.units
        chunk = (AnnotationSourceSlice(first.unit_id, first.unit_id, first.start, first.end, first.text),)
        draft = {
            "entities": [
                {"id": "e1", "name": "角色甲", "kind": "人物", "aliases": [],
                 "evidence": [{"unit_id": first.unit_id, "quote": "角色甲"}]},
                {"id": "e2", "name": "门", "kind": "地点", "aliases": [],
                 "evidence": [{"unit_id": first.unit_id, "quote": "门"},
                              {"unit_id": second.unit_id, "quote": "门"}]},
            ],
            "facts": [{"id": "f1", "kind": "地点", "subject_id": "e1", "predicate": "到达",
                       "value": "", "object_id": "e2", "certainty": "明确",
                       "evidence": [{"unit_id": first.unit_id, "quote": "走到门前"}]},
                      {"id": "f2", "kind": "信息", "subject_id": "e1", "predicate": "当前行动",
                       "value": "走到门前", "object_id": "", "certainty": "明确",
                       "evidence": [{"unit_id": first.unit_id, "quote": "走到门前"}]}],
            "events": [{"id": "v1", "order": 1, "summary": "角色甲走到门前", "action_type": "移动",
                        "actor_id": "e1", "target_ids": [], "basis_fact_ids": ["f1"],
                        "participant_ids": ["e1"], "trigger_fact_ids": [], "precondition_fact_ids": [],
                        "action": "走到门前", "obstacle": "", "decision": "", "outcome_fact_ids": ["f1"],
                        "cost_fact_ids": [], "evidence": [{"unit_id": first.unit_id, "quote": "走到门前"}]}],
            "scenes": [{"id": "s1", "order": 1, "participant_ids": ["e1"], "event_ids": ["v1"],
                        "objective": "走到门前", "entry_fact_ids": [], "exit_fact_ids": ["f1"], "tension": 1,
                        "evidence": [{"unit_id": first.unit_id, "quote": "走到门前"}]}],
            "state_changes": [{"id": "c1", "event_id": "v1", "operation": "add",
                               "before_fact_id": "", "after_fact_id": "f1"}],
        }
        annotation = annotation_from_draft(document, draft, evidence_units=None)
        # Migration quality is about evidence scope, not density.  Repeat a
        # source-grounded static proposition so this tiny synthetic chunk meets
        # the same two-fact floor as production chunks.
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.json"
            path.write_text(json.dumps({
                "schema_version": "1.0", "annotation_contract_revision": "3.14-reporting-role-contract",
                "source_hash": document.source_hash, "chapter_id": document.chapter_id,
                "chunk_index": 1, "annotation": annotation.to_dict(),
            }, ensure_ascii=False), encoding="utf-8")

            migrated = _load_chunk_checkpoint(path, document, chunk)

            self.assertIsNone(migrated)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["annotation_contract_revision"], "3.14-reporting-role-contract")

            scoped = annotation.to_dict()
            for entity in scoped["entities"]:
                entity["evidence"] = [
                    span for span in entity["evidence"] if span["unit_id"] == first.unit_id
                ]
            path.write_text(json.dumps({
                "schema_version": "1.0",
                "annotation_contract_revision": ANNOTATION_CONTRACT_REVISION,
                "semantic_reconciled": True,
                "source_hash": document.source_hash,
                "chapter_id": document.chapter_id,
                "chunk_index": 1,
                "annotation": scoped,
            }, ensure_ascii=False), encoding="utf-8")

            accepted = _load_chunk_checkpoint(path, document, chunk)

            self.assertIsNotNone(accepted)

    def test_unique_explicit_organization_alias_merges_generic_window_entity(self) -> None:
        text = "宗族获得名额。\n青河氏宗族派人来访。"
        document = build_chapter_document(
            "Author/work/0021", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        first_unit, second_unit = document.units
        first_span = {"chapter_id": document.chapter_id, "start": first_unit.start,
                      "end": first_unit.start + 2, "quote": "宗族", "unit_id": first_unit.unit_id}
        second_span = {"chapter_id": document.chapter_id, "start": second_unit.start,
                       "end": second_unit.start + 5, "quote": "青河氏宗族", "unit_id": second_unit.unit_id}
        first = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "generic", "canonical_name": "宗族", "kind": "organization",
                          "aliases": [], "evidence": [first_span]}],
            "facts": [], "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })
        second = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "specific", "canonical_name": "青河氏宗族", "kind": "organization",
                          "aliases": ["宗族"], "evidence": [second_span]}],
            "facts": [], "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        merged = merge_annotations(document, [first, second])

        self.assertEqual(len(merged.entities), 1)
        self.assertEqual(merged.entities[0].canonical_name, "青河氏宗族")
        self.assertIn("宗族", merged.entities[0].aliases)
        self.assertEqual(len(merged.entities[0].evidence), 2)

    def test_ambiguous_generic_organization_is_not_merged_by_suffix(self) -> None:
        text = "宗族议事。\n青河氏宗族到场。\n白石氏宗族到场。"
        document = build_chapter_document(
            "Author/work/0022", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        annotations = []
        for index, (name, aliases) in enumerate((("宗族", []), ("青河氏宗族", ["宗族"]), ("白石氏宗族", ["宗族"]))):
            unit = document.units[index]
            span = {"chapter_id": document.chapter_id, "start": unit.start,
                    "end": unit.start + len(name), "quote": name, "unit_id": unit.unit_id}
            annotations.append(ChapterAnnotation.from_dict({
                "chapter_id": document.chapter_id, "source_hash": document.source_hash,
                "entities": [{"entity_id": f"e{index}", "canonical_name": name,
                              "kind": "organization", "aliases": aliases, "evidence": [span]}],
                "facts": [], "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
            }))

        merged = merge_annotations(document, annotations)

        self.assertEqual(len(merged.entities), 3)

    def test_explicit_transfer_result_information_is_retyped_as_resource(self) -> None:
        payload = {
            "entities": [
                {"id": "giver", "name": "来客", "kind": "人物群体", "aliases": [], "evidence": []},
                {"id": "receiver", "name": "主人", "kind": "人物", "aliases": [], "evidence": []},
            ],
            "facts": [{"id": "gift", "kind": "信息", "subject_id": "giver", "predicate": "赠送",
                       "value": "礼物", "object_id": "receiver", "certainty": "明确",
                       "evidence": [{"unit_id": "u1", "quote": "来客送来礼物"}]}],
            "events": [{"id": "give", "order": 1, "summary": "来客送礼", "action_type": "交付",
                        "actor_id": "giver", "target_ids": ["receiver"], "basis_fact_ids": ["gift"],
                        "participant_ids": ["giver", "receiver"], "trigger_fact_ids": [],
                        "precondition_fact_ids": [], "action": "送来礼物", "obstacle": "", "decision": "",
                        "outcome_fact_ids": ["gift"], "cost_fact_ids": [],
                        "evidence": [{"unit_id": "u1", "quote": "来客送来礼物"}]}],
            "scenes": [], "state_changes": [],
        }

        normalized = _normalize_action_result_fact_kinds(payload)

        self.assertEqual(normalized["facts"][0]["kind"], "资源")

    def test_explicit_movement_result_pointing_to_place_is_retyped_as_location(self) -> None:
        payload = {
            "entities": [
                {"id": "person", "name": "角色甲", "kind": "人物", "aliases": [], "evidence": []},
                {"id": "home", "name": "家中", "kind": "地点", "aliases": [], "evidence": []},
            ],
            "facts": [{"id": "arrival", "kind": "运动", "subject_id": "person", "predicate": "走向",
                       "value": "家中", "object_id": "home", "certainty": "明确",
                       "evidence": [{"unit_id": "u1", "quote": "起身向家中走去"}]}],
            "events": [{"id": "move", "order": 1, "summary": "角色甲走向家中", "action_type": "移动",
                        "actor_id": "person", "target_ids": [], "basis_fact_ids": ["arrival"],
                        "participant_ids": ["person"], "trigger_fact_ids": [], "precondition_fact_ids": [],
                        "action": "走向家中", "obstacle": "", "decision": "",
                        "outcome_fact_ids": ["arrival"], "cost_fact_ids": [],
                        "evidence": [{"unit_id": "u1", "quote": "起身向家中走去"}]}],
            "scenes": [], "state_changes": [],
        }

        normalized = _normalize_action_result_fact_kinds(payload)

        self.assertEqual(normalized["facts"][0]["kind"], "地点")
        self.assertEqual(normalized["facts"][0]["object_id"], "home")

    def test_static_location_progression_is_retyped_without_inventing_event(self) -> None:
        payload = {
            "entities": [
                {"id": "person", "kind": "人物"},
                {"id": "outside", "kind": "地点"},
            ],
            "facts": [{"id": "f1", "kind": "进展", "subject_id": "person",
                       "predicate": "位于", "value": "门外", "object_id": "outside"}],
            "events": [],
        }

        normalized = _normalize_action_result_fact_kinds(payload)

        self.assertEqual(normalized["facts"][0]["kind"], "地点")
        self.assertEqual(normalized["events"], [])

    def test_movement_without_explicit_place_edge_remains_progression(self) -> None:
        payload = {
            "entities": [{"id": "person", "name": "角色甲", "kind": "人物", "aliases": [], "evidence": []}],
            "facts": [{"id": "travel", "kind": "运动", "subject_id": "person", "predicate": "赶路",
                       "value": "继续前行", "object_id": "", "certainty": "明确",
                       "evidence": [{"unit_id": "u1", "quote": "继续向前赶路"}]}],
            "events": [{"id": "move", "order": 1, "summary": "角色甲继续赶路", "action_type": "移动",
                        "actor_id": "person", "target_ids": [], "basis_fact_ids": ["travel"],
                        "participant_ids": ["person"], "trigger_fact_ids": [], "precondition_fact_ids": [],
                        "action": "继续赶路", "obstacle": "", "decision": "",
                        "outcome_fact_ids": ["travel"], "cost_fact_ids": [],
                        "evidence": [{"unit_id": "u1", "quote": "继续向前赶路"}]}],
            "scenes": [], "state_changes": [],
        }

        normalized = _normalize_action_result_fact_kinds(payload)

        self.assertEqual(FACT_KIND_MAP[normalized["facts"][0]["kind"]], "progression")

    def test_explicit_spoken_response_is_retyped_as_speech_without_model_repair(self) -> None:
        payload = {
            "entities": [
                {"id": "child", "name": "孩子", "kind": "人物"},
                {"id": "parent", "name": "父亲", "kind": "人物"},
            ],
            "facts": [{"id": "reply", "kind": "进展", "subject_id": "child",
                       "predicate": "应付询问", "value": "应付了几句", "object_id": "parent",
                       "evidence": [{"unit_id": "u1", "quote": "孩子应付了几句"}]}],
            "events": [{"id": "answer", "action_type": "其他", "actor_id": "child",
                        "target_ids": ["parent"], "summary": "孩子应付父亲的询问",
                        "action": "应付了几句", "outcome_fact_ids": ["reply"]}],
        }

        normalized = _normalize_action_result_fact_kinds(payload)

        self.assertEqual(normalized["events"][0]["action_type"], "交流")

    def test_nonverbal_response_is_not_retyped_as_speech(self) -> None:
        payload = {
            "entities": [{"id": "person", "name": "角色甲", "kind": "人物"}],
            "facts": [{"id": "nod", "kind": "进展", "subject_id": "person",
                       "predicate": "点头回应", "value": "", "object_id": "",
                       "evidence": [{"unit_id": "u1", "quote": "角色甲点头回应"}]}],
            "events": [{"id": "respond", "action_type": "其他", "actor_id": "person",
                        "target_ids": [], "summary": "角色甲点头回应", "action": "点头回应",
                        "outcome_fact_ids": ["nod"]}],
        }

        normalized = _normalize_action_result_fact_kinds(payload)

        self.assertEqual(normalized["events"][0]["action_type"], "其他")

    def test_transfer_summary_cannot_override_only_preparation_in_exact_quote(self) -> None:
        payload = {
            "entities": [
                {"id": "giver", "name": "来客", "kind": "人物群体", "aliases": [], "evidence": []},
                {"id": "receiver", "name": "主人", "kind": "人物", "aliases": [], "evidence": []},
            ],
            "facts": [{"id": "gift", "kind": "信息", "subject_id": "giver", "predicate": "道贺",
                       "value": "送来礼物", "object_id": "receiver", "certainty": "明确",
                       "evidence": [{"unit_id": "u1", "quote": "每位来客都会准备礼物"}]}],
            "events": [{"id": "give", "order": 1, "summary": "来客道贺送礼", "action_type": "交付",
                        "actor_id": "giver", "target_ids": ["receiver"], "basis_fact_ids": ["gift"],
                        "participant_ids": ["giver", "receiver"], "trigger_fact_ids": [],
                        "precondition_fact_ids": [], "action": "道贺送礼", "obstacle": "", "decision": "",
                        "outcome_fact_ids": ["gift"], "cost_fact_ids": [],
                        "evidence": [{"unit_id": "u1", "quote": "每位来客都会准备礼物"}]}],
            "scenes": [], "state_changes": [],
        }

        normalized = _normalize_action_result_fact_kinds(payload)

        self.assertEqual(normalized["facts"][0]["kind"], "信息")

    def test_future_transfer_statement_is_not_retyped(self) -> None:
        payload = {
            "entities": [{"id": "speaker", "name": "角色甲", "kind": "人物", "aliases": [], "evidence": []}],
            "facts": [{"id": "promise", "kind": "信息", "subject_id": "speaker", "predicate": "表示",
                       "value": "将要交付物品", "object_id": "", "certainty": "明确",
                       "evidence": [{"unit_id": "u1", "quote": "以后会交付物品"}]}],
            "events": [{"id": "say", "order": 1, "summary": "角色甲表示以后交付", "action_type": "交流",
                        "actor_id": "speaker", "target_ids": [], "basis_fact_ids": ["promise"],
                        "participant_ids": ["speaker"], "trigger_fact_ids": [], "precondition_fact_ids": [],
                        "action": "表示以后会交付", "obstacle": "", "decision": "",
                        "outcome_fact_ids": ["promise"], "cost_fact_ids": [],
                        "evidence": [{"unit_id": "u1", "quote": "以后会交付物品"}]}],
            "scenes": [], "state_changes": [],
        }

        normalized = _normalize_action_result_fact_kinds(payload)

        self.assertEqual(normalized["facts"][0]["kind"], "信息")

    def test_reporting_event_separates_speech_outcome_from_existing_relationship(self) -> None:
        payload = {
            "entities": [
                {"id": "speaker", "name": "监护人", "kind": "人物", "aliases": [], "evidence": []},
                {"id": "listener", "name": "访客", "kind": "人物", "aliases": [], "evidence": []},
                {"id": "child", "name": "角色乙", "kind": "人物", "aliases": [], "evidence": []},
            ],
            "facts": [{
                "id": "relation", "kind": "关系", "subject_id": "child",
                "predicate": "是", "value": "孩子", "object_id": "speaker",
                "certainty": "明确", "evidence": [{"unit_id": "u1", "quote": "这是我的孩子角色乙"}],
            }],
            "events": [{
                "id": "intro", "order": 1, "summary": "监护人介绍角色乙", "action_type": "交流",
                "actor_id": "speaker", "target_ids": ["listener"], "basis_fact_ids": ["relation"],
                "participant_ids": ["speaker", "listener"], "trigger_fact_ids": [], "precondition_fact_ids": [],
                "action": "介绍角色乙", "obstacle": "", "decision": "",
                "outcome_fact_ids": ["relation"], "cost_fact_ids": [],
                "evidence": [{"unit_id": "u1", "quote": "这是我的孩子角色乙"}],
            }],
            "scenes": [], "state_changes": [],
        }

        normalized = _normalize_reporting_event_fact_roles(payload)

        event = normalized["events"][0]
        self.assertEqual(event["precondition_fact_ids"], ["relation"])
        self.assertNotEqual(event["outcome_fact_ids"], ["relation"])
        report = next(row for row in normalized["facts"] if row["id"] == event["outcome_fact_ids"][0])
        self.assertEqual(report["kind"], "信息")
        self.assertEqual(report["subject_id"], "speaker")
        self.assertEqual(report["object_id"], "child")
        self.assertIn("角色乙", report["value"])
        self.assertNotIn("孩子角色乙", report["value"])

    def test_performative_speech_keeps_state_it_creates(self) -> None:
        payload = {
            "entities": [
                {"id": "speaker", "name": "主持者", "kind": "人物", "aliases": [], "evidence": []},
                {"id": "candidate", "name": "候选人", "kind": "人物", "aliases": [], "evidence": []},
            ],
            "facts": [{
                "id": "appointment", "kind": "身份", "subject_id": "candidate",
                "predicate": "成为", "value": "执事", "object_id": "", "certainty": "明确",
                "evidence": [{"unit_id": "u1", "quote": "宣布候选人成为执事"}],
            }],
            "events": [{
                "id": "announce", "order": 1, "summary": "主持者宣布任命", "action_type": "交流",
                "actor_id": "speaker", "target_ids": ["candidate"], "basis_fact_ids": ["appointment"],
                "participant_ids": ["speaker", "candidate"], "trigger_fact_ids": [], "precondition_fact_ids": [],
                "action": "宣布任命", "obstacle": "", "decision": "", "outcome_fact_ids": ["appointment"],
                "cost_fact_ids": [], "evidence": [{"unit_id": "u1", "quote": "宣布候选人成为执事"}],
            }],
            "scenes": [], "state_changes": [],
        }

        normalized = _normalize_reporting_event_fact_roles(payload)

        self.assertEqual(normalized["events"][0]["outcome_fact_ids"], ["appointment"])
        self.assertEqual(len(normalized["facts"]), 1)

    def test_sanitize_narrows_invalid_entity_description_to_exact_name(self) -> None:
        text = "地面上站立一白衣青年。角色乙，漫不经心地看了一眼。"
        document = build_chapter_document(
            "Author/work/0019", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        units = (AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text),)
        payload = {
            "entities": [
                {"id": "e1", "name": "白衣青年", "kind": "人物", "aliases": [],
                 "evidence": [{"unit_id": unit.unit_id, "quote": "地面站立一白衣青年"}]},
                {"id": "e2", "name": "角色乙", "kind": "人物", "aliases": [],
                 "evidence": [{"unit_id": unit.unit_id, "quote": "角色乙漫不经心地看了一眼"}]},
            ],
            "facts": [], "events": [], "scenes": [], "state_changes": [],
        }

        clean = sanitize_draft(document, payload, units)

        self.assertEqual([row["evidence"][0]["quote"] for row in clean["entities"]], ["白衣青年", "角色乙"])

    def test_cross_chunk_named_collective_prefers_organization_over_group(self) -> None:
        text = "青石商会的成员抵达。\n青石商会发布命令。"
        document = build_chapter_document(
            "Author/work/0018", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        first_unit, second_unit = document.units
        first = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{
                "entity_id": "collective-a", "canonical_name": "青石商会", "kind": "group",
                "aliases": ["商会成员"],
                "evidence": [{"chapter_id": document.chapter_id, "start": first_unit.start,
                              "end": first_unit.start + 4, "quote": "青石商会", "unit_id": first_unit.unit_id}],
            }],
            "facts": [], "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })
        second = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{
                "entity_id": "collective-b", "canonical_name": "青石商会", "kind": "organization",
                "aliases": [],
                "evidence": [{"chapter_id": document.chapter_id, "start": second_unit.start,
                              "end": second_unit.start + 4, "quote": "青石商会", "unit_id": second_unit.unit_id}],
            }],
            "facts": [], "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        merged = merge_annotations(document, [first, second])

        self.assertEqual(len(merged.entities), 1)
        self.assertEqual(merged.entities[0].kind, "organization")
        self.assertIn("商会成员", merged.entities[0].aliases)

    def test_cross_chunk_singular_and_collective_roles_remain_distinct(self) -> None:
        text = "一个守卫留在门前。\n几个守卫把伤者带到屋内。"
        document = build_chapter_document(
            "Author/work/0029", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        first_unit, second_unit = document.units

        def entity(entity_id: str, kind: str, unit, quote: str):
            local = unit.text.index(quote)
            return {
                "entity_id": entity_id, "canonical_name": "守卫", "kind": kind,
                "aliases": [], "evidence": [{
                    "chapter_id": document.chapter_id, "start": unit.start + local,
                    "end": unit.start + local + len(quote), "quote": quote,
                    "unit_id": unit.unit_id,
                }],
            }

        first = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [entity("single", "person", first_unit, "一个守卫")],
            "facts": [], "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })
        second = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [entity("collective", "group", second_unit, "几个守卫")],
            "facts": [], "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        merged = merge_annotations(document, [first, second])

        self.assertEqual(len(merged.entities), 2)
        self.assertEqual({item.canonical_name for item in merged.entities}, {"守卫", "几个守卫"})
        self.assertEqual({item.kind for item in merged.entities}, {"person", "group"})

    def test_source_atom_reconciliation_requires_full_unit_ledger(self) -> None:
        text = "角色甲开门。\n角色乙进入。"
        document = build_chapter_document(
            "Author/work/0014", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        units = tuple(
            AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text)
            for unit in document.units
        )
        payload = {
            "结论": "通过",
            "已核原文单元ID": [units[0].unit_id],
            "问题": [],
        }

        with self.assertRaisesRegex(ValueError, "按输入顺序完整等于"):
            _parse_atom_semantic_reconciliation(payload, units)

    def test_source_atom_reconciliation_rejects_empty_issue_list_for_repair(self) -> None:
        text = "角色甲开门。"
        document = build_chapter_document(
            "Author/work/0015", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        units = (AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text),)
        payload = {
            "结论": "需修复", "已核原文单元ID": [unit.unit_id], "问题": [],
        }

        with self.assertRaisesRegex(ValueError, "至少给出一个问题"):
            _parse_atom_semantic_reconciliation(payload, units)

    def test_source_atom_reconciliation_drops_noop_and_duplicate_issues(self) -> None:
        text = "角色甲来到院门。"
        document = build_chapter_document(
            "Author/work/0039", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        units = (AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text),)
        missing = f"missing:event:{unit.unit_id}"
        payload = {
            "结论": "需修复", "已核原文单元ID": [unit.unit_id], "问题": [
                {"问题ID": "q1", "类别": "核心覆盖", "对象ID": [missing],
                 "原文单元ID": [unit.unit_id], "诊断": "到达动作缺少事件",
                 "修复要求": "新增地点事实和移动事件"},
                {"问题ID": "q2", "类别": "事件原子性", "对象ID": [missing],
                 "原文单元ID": [unit.unit_id], "诊断": "同一个到达动作未原子化",
                 "修复要求": "新增移动事件"},
                {"问题ID": "q3", "类别": "重复对象", "对象ID": ["v1"],
                 "原文单元ID": [unit.unit_id], "诊断": "若已正确则忽略",
                 "修复要求": "无需修改"},
            ],
        }

        verdict, issues = _parse_atom_semantic_reconciliation(payload, units)

        self.assertEqual(verdict, "需修复")
        self.assertEqual([item["问题ID"] for item in issues], ["q1"])

    def test_source_atom_reconciliation_limits_each_patch_batch(self) -> None:
        text = "角色甲来到院门。"
        document = build_chapter_document(
            "Author/work/0040", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        units = (AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text),)
        payload = {
            "结论": "需修复", "已核原文单元ID": [unit.unit_id], "问题": [
                {"问题ID": f"q{index}", "类别": "核心覆盖",
                 "对象ID": [f"missing:event:{unit.unit_id}:{index}"],
                 "原文单元ID": [unit.unit_id], "诊断": f"缺少核心事件{index}",
                 "修复要求": f"新增事件{index}"}
                for index in range(1, 6)
            ],
        }

        with self.assertRaisesRegex(ValueError, "单轮问题最多 4 项"):
            _parse_atom_semantic_reconciliation(payload, units)

    def test_source_atom_reconciliation_adjudicates_no_conflict_as_pass(self) -> None:
        text = "角色甲回答长辈。"
        document = build_chapter_document(
            "Author/work/0045", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        units = (AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text),)
        payload = {
            "结论": "需修复", "已核原文单元ID": [unit.unit_id], "问题": [{
                "问题ID": "q1", "类别": "实体同指", "对象ID": ["e1"],
                "原文单元ID": [unit.unit_id],
                "诊断": "当前实体定义无冲突，称谓已经一致，仅证据数量可能影响展示。",
                "修复要求": "确认现有实体即可。",
            }],
        }

        verdict, issues = _parse_atom_semantic_reconciliation(payload, units)

        self.assertEqual(verdict, "通过")
        self.assertEqual(issues, [])

    def test_source_atom_reconciliation_adjudicates_daily_texture_as_pass(self) -> None:
        text = "角色甲坐下吃饭，亲人端菜招呼。"
        document = build_chapter_document(
            "Author/work/0046", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        units = (AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text),)
        payload = {
            "结论": "需修复", "已核原文单元ID": [unit.unit_id], "问题": [{
                "问题ID": "q1", "类别": "核心覆盖",
                "对象ID": [f"missing:event:{unit.unit_id}"],
                "原文单元ID": [unit.unit_id],
                "诊断": "坐下吃饭和端菜没有建立事件。",
                "修复要求": "新增吃饭事件和端菜事件。",
            }],
        }

        verdict, issues = _parse_atom_semantic_reconciliation(payload, units)

        self.assertEqual(verdict, "通过")
        self.assertEqual(issues, [])

    def test_source_atom_reconciliation_does_not_upgrade_meal_transfer_wording(self) -> None:
        text = "母亲把不多的几块肉给孩子夹过去。"
        document = build_chapter_document(
            "Author/work/0049", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        units = (AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text),)
        payload = {
            "结论": "需修复", "已核原文单元ID": [unit.unit_id], "问题": [{
                "问题ID": "q1", "类别": "核心覆盖",
                "对象ID": [f"missing:event:{unit.unit_id}"],
                "原文单元ID": [unit.unit_id],
                "诊断": "母亲把几块肉夹给孩子，当前没有建立资源转移事件。",
                "修复要求": "新增交付事件并写入孩子获得几块肉。",
            }],
        }

        verdict, issues = _parse_atom_semantic_reconciliation(payload, units)

        self.assertEqual(verdict, "通过")
        self.assertEqual(issues, [])

    def test_source_atom_reconciliation_rejects_ambiguous_repair_choice(self) -> None:
        text = "角色甲看到同伴起身。"
        document = build_chapter_document(
            "Author/work/0047", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        units = (AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text),)
        payload = {
            "结论": "需修复", "已核原文单元ID": [unit.unit_id], "问题": [{
                "问题ID": "q1", "类别": "事件角色", "对象ID": ["v1"],
                "原文单元ID": [unit.unit_id], "诊断": "行动者不清楚。",
                "修复要求": "建议改成角色甲或者另立同伴事件。",
            }],
        }

        with self.assertRaisesRegex(ValueError, "必须是唯一确定的修改"):
            _parse_atom_semantic_reconciliation(payload, units)

    def test_source_atom_reconciliation_prompt_is_generic_and_checks_story_truth(self) -> None:
        text = "角色甲把伤者带进屋内，随后给伤者服药。"
        document = build_chapter_document(
            "Author/work/0016", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        units = (AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text),)
        prompt = atomic_semantic_reconciliation_prompt(
            document, units, {"entities": [], "facts": [], "events": []},
        )

        self.assertIn("源文—原子标注", prompt)
        self.assertIn("原子事件”以一次连贯的因果变化为边界", prompt)
        self.assertIn("不得为了每个微动作虚构", prompt)
        self.assertIn("同一说话者未被打断的一轮发言是一个交流事件", prompt)
        self.assertIn("当前值已经满足修复要求", prompt)
        self.assertIn("普通寒暄、礼数、气氛动作", prompt)
        self.assertIn("端饭、吃饭、夹菜", prompt)
        self.assertIn("反事实判断", prompt)
        self.assertNotIn("耳根", prompt)

    def test_source_atom_reconciliation_prompt_carries_resolved_history(self) -> None:
        text = "角色甲进入山门。"
        document = build_chapter_document(
            "Author/work/0048", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        units = (AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text),)

        prompt = atomic_semantic_reconciliation_prompt(
            document, units, {"entities": [], "facts": [], "events": []},
            resolved_issues=["q1 已修复角色甲的地点阶段"],
        )

        self.assertIn("已完成定点修补的问题历史", prompt)
        self.assertIn("不得换类别、换措辞", prompt)

    def test_final_patch_verification_prompt_forbids_new_issues(self) -> None:
        text = "角色甲进入山门。"
        document = build_chapter_document(
            "Author/work/0049", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        units = (AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text),)
        issue = {"问题ID": "q1", "类别": "动作阶段", "对象ID": ["v1"],
                 "原文单元ID": [unit.unit_id], "诊断": "阶段写错",
                 "修复要求": "将朝向改为进入"}

        prompt = atomic_semantic_patch_verification_prompt(
            document, units, {"entities": [], "facts": [], "events": []}, [issue],
        )

        self.assertIn("不增加新问题", prompt)
        self.assertIn("不得新造问题ID", prompt)
        self.assertIn('"问题ID": "q1"', prompt)

    def test_atomic_prompt_requires_span_overlap_and_directional_transfer(self) -> None:
        text = "来客送礼，主人收下；来客离开时主人回礼。"
        document = build_chapter_document(
            "Author/work/0025", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        units = (AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text),)
        prompt = atomic_annotation_prompt(document, units, 1, 1)

        self.assertIn("实际文字区间重叠", prompt)
        self.assertIn("资源往返必须按方向拆分", prompt)
        self.assertIn("准备礼物/准备回礼", prompt)

    def test_first_module_quality_rejects_unmodelled_completed_transport(self) -> None:
        text = "照料者拉着伤者一直送到院门。"
        document = build_chapter_document(
            "Author/work/0026", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [], "facts": [], "events": [], "scenes": [], "state_changes": [],
            "schema_version": "2.2",
        })

        quality = assess_annotation_quality(document, annotation)

        issue = next(item for item in quality.issues if item.code == "strong_action_coverage")
        self.assertIn("送到", issue.message)
        self.assertEqual(issue.severity, "error")

    def test_first_module_quality_does_not_force_ceremonial_gift_acceptance(self) -> None:
        text = "来客备下礼物，主人推脱不掉也就收下。"
        document = build_chapter_document(
            "Author/work/0027", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [], "facts": [], "events": [], "scenes": [], "state_changes": [],
            "schema_version": "2.2",
        })

        quality = assess_annotation_quality(document, annotation)

        self.assertFalse(any(item.code == "strong_action_coverage" for item in quality.issues))

    def test_first_module_recognizes_feeding_medicine_as_treatment_event(self) -> None:
        text = "弟子把众少年带到山上，统一喂食药物。"
        document = build_chapter_document(
            "Author/work/0028", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        whole = SourceSpan(document.chapter_id, unit.start, unit.end, unit.text, unit.unit_id)
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "actor", "canonical_name": "弟子", "kind": "person", "aliases": [], "evidence": [whole.to_dict()]},
                {"entity_id": "target", "canonical_name": "众少年", "kind": "group", "aliases": [], "evidence": [whole.to_dict()]},
            ],
            "facts": [{"fact_id": "treated", "chapter_id": document.chapter_id, "kind": "progression",
                       "subject_id": "target", "predicate": "被喂食", "value": "药物", "object_id": "",
                       "certainty": 1.0, "evidence": [whole.to_dict()]}],
            "events": [{"event_id": "treat", "chapter_id": document.chapter_id, "order": 0,
                        "summary": "弟子统一喂食药物给众少年", "participant_ids": ["actor", "target"],
                        "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "统一喂食药物", "obstacle": "",
                        "decision": "", "outcome_fact_ids": ["treated"], "cost_fact_ids": [],
                        "evidence": [whole.to_dict()], "action_type": "other", "actor_id": "actor",
                        "target_ids": ["target"], "basis_fact_ids": ["treated"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        quality = assess_annotation_quality(document, annotation)

        self.assertFalse(any(
            item.code == "strong_action_coverage" and "给药" in item.message
            for item in quality.issues
        ))

    def test_reporting_fact_does_not_use_the_listener_as_proposition_object(self) -> None:
        text = "角色甲对角色乙说道：“明天出发。”"
        document = build_chapter_document(
            "Author/work/0030", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        whole = SourceSpan(document.chapter_id, unit.start, unit.end, unit.text, unit.unit_id)
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "speaker", "canonical_name": "角色甲", "kind": "person", "aliases": [], "evidence": [whole.to_dict()]},
                {"entity_id": "listener", "canonical_name": "角色乙", "kind": "person", "aliases": [], "evidence": [whole.to_dict()]},
            ],
            "facts": [{"fact_id": "message", "chapter_id": document.chapter_id, "kind": "information",
                       "subject_id": "speaker", "predicate": "告知", "value": "明天出发", "object_id": "listener",
                       "certainty": 1.0, "evidence": [whole.to_dict()]}],
            "events": [{"event_id": "speech", "chapter_id": document.chapter_id, "order": 0,
                        "summary": "角色甲告知明天出发", "participant_ids": ["speaker", "listener"],
                        "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "告知明天出发", "obstacle": "",
                        "decision": "", "outcome_fact_ids": ["message"], "cost_fact_ids": [],
                        "evidence": [whole.to_dict()], "action_type": "speech", "actor_id": "speaker",
                        "target_ids": ["listener"], "basis_fact_ids": ["message"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        normalized = normalize_fact_role_contracts(annotation)

        self.assertEqual(normalized.facts[0].object_id, "")

    def test_first_module_rejects_speech_event_spanning_a_different_speaker(self) -> None:
        text = "角色甲说道：“先走。”\n角色乙答道：“我留下。”"
        document = build_chapter_document(
            "Author/work/0031", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        first, second = document.units
        first_span = SourceSpan(document.chapter_id, first.start, first.end, first.text, first.unit_id)
        second_span = SourceSpan(document.chapter_id, second.start, second.end, second.text, second.unit_id)
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "first", "canonical_name": "角色甲", "kind": "person", "aliases": [], "evidence": [first_span.to_dict()]},
                {"entity_id": "second", "canonical_name": "角色乙", "kind": "person", "aliases": [], "evidence": [second_span.to_dict()]},
            ],
            "facts": [{"fact_id": "message", "chapter_id": document.chapter_id, "kind": "information",
                       "subject_id": "first", "predicate": "表示", "value": "先走并留下", "object_id": "",
                       "certainty": 1.0, "evidence": [first_span.to_dict()]}],
            "events": [{"event_id": "speech", "chapter_id": document.chapter_id, "order": 0,
                        "summary": "角色甲说先走并留下", "participant_ids": ["first"],
                        "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "说先走并留下", "obstacle": "",
                        "decision": "", "outcome_fact_ids": ["message"], "cost_fact_ids": [],
                        "evidence": [first_span.to_dict(), second_span.to_dict()], "action_type": "speech", "actor_id": "first",
                        "target_ids": [], "basis_fact_ids": ["message"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        quality = assess_annotation_quality(document, annotation)

        self.assertTrue(any(item.code == "event_mixed_speaker_evidence" for item in quality.issues))

    def test_first_module_requires_explicit_first_person_future_arrangement(self) -> None:
        text = "角色甲说道：“我明天来接你。”"
        document = build_chapter_document(
            "Author/work/0032", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        whole = SourceSpan(document.chapter_id, unit.start, unit.end, unit.text, unit.unit_id)
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "speaker", "canonical_name": "角色甲", "kind": "person",
                          "aliases": [], "evidence": [whole.to_dict()]}],
            "facts": [], "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        quality = assess_annotation_quality(document, annotation)

        self.assertTrue(any(item.code == "future_arrangement_coverage" for item in quality.issues))

    def test_first_module_rejects_proposal_as_completed_state_and_event(self) -> None:
        text = "角色甲问角色乙：“可愿意做我的弟子？”"
        document = build_chapter_document(
            "Author/work/0033", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        whole = SourceSpan(document.chapter_id, unit.start, unit.end, unit.text, unit.unit_id)
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "inviter", "canonical_name": "角色甲", "kind": "person", "aliases": [], "evidence": [whole.to_dict()]},
                {"entity_id": "invitee", "canonical_name": "角色乙", "kind": "person", "aliases": [], "evidence": [whole.to_dict()]},
            ],
            "facts": [{"fact_id": "apprentice", "chapter_id": document.chapter_id, "kind": "relationship",
                       "subject_id": "invitee", "predicate": "成为弟子", "value": "", "object_id": "inviter",
                       "certainty": 1.0, "evidence": [whole.to_dict()]}],
            "events": [{"event_id": "join", "chapter_id": document.chapter_id, "order": 0,
                        "summary": "角色乙成为弟子", "participant_ids": ["inviter", "invitee"],
                        "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "拜师成为弟子", "obstacle": "",
                        "decision": "", "outcome_fact_ids": ["apprentice"], "cost_fact_ids": [],
                        "evidence": [whole.to_dict()], "action_type": "other", "actor_id": "invitee",
                        "target_ids": ["inviter"], "basis_fact_ids": ["apprentice"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        quality = assess_annotation_quality(document, annotation)
        codes = {item.code for item in quality.issues}

        self.assertIn("fact_proposal_modality_lost", codes)
        self.assertIn("event_proposal_modality_lost", codes)

    def test_first_module_accepts_proposed_action_only_after_explicit_acceptance(self) -> None:
        text = "角色甲问角色乙：“可愿意同行？”角色乙点头答应。"
        document = build_chapter_document(
            "Author/work/0034", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        whole = SourceSpan(document.chapter_id, unit.start, unit.end, unit.text, unit.unit_id)
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "inviter", "canonical_name": "角色甲", "kind": "person", "aliases": [], "evidence": [whole.to_dict()]},
                {"entity_id": "invitee", "canonical_name": "角色乙", "kind": "person", "aliases": [], "evidence": [whole.to_dict()]},
            ],
            "facts": [{"fact_id": "accepted", "chapter_id": document.chapter_id, "kind": "progression",
                       "subject_id": "invitee", "predicate": "答应", "value": "同行", "object_id": "inviter",
                       "certainty": 1.0, "evidence": [whole.to_dict()]}],
            "events": [{"event_id": "accept", "chapter_id": document.chapter_id, "order": 0,
                        "summary": "角色乙答应同行", "participant_ids": ["inviter", "invitee"],
                        "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "点头答应同行", "obstacle": "",
                        "decision": "答应", "outcome_fact_ids": ["accepted"], "cost_fact_ids": [],
                        "evidence": [whole.to_dict()], "action_type": "decision", "actor_id": "invitee",
                        "target_ids": ["inviter"], "basis_fact_ids": ["accepted"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        codes = {item.code for item in assess_annotation_quality(document, annotation).issues}

        self.assertNotIn("fact_proposal_modality_lost", codes)
        self.assertNotIn("event_proposal_modality_lost", codes)

    def test_first_module_requires_named_thinkers_explicit_self_goal(self) -> None:
        text = "“我一定要通过试炼。”角色甲暗想。"
        document = build_chapter_document(
            "Author/work/0035", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        whole = SourceSpan(document.chapter_id, unit.start, unit.end, unit.text, unit.unit_id)
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "thinker", "canonical_name": "角色甲", "kind": "person", "aliases": [], "evidence": [whole.to_dict()]},
                {"entity_id": "other", "canonical_name": "角色乙", "kind": "person", "aliases": [], "evidence": [whole.to_dict()]},
            ],
            "facts": [{"fact_id": "wrong-goal", "chapter_id": document.chapter_id, "kind": "goal",
                       "subject_id": "other", "predicate": "决心", "value": "通过试炼", "object_id": "",
                       "certainty": 1.0, "evidence": [whole.to_dict()]}],
            "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        quality = assess_annotation_quality(document, annotation)

        self.assertTrue(any(item.code == "explicit_self_goal_coverage" for item in quality.issues))

    def test_first_module_rejects_standalone_quote_with_ungrounded_subject(self) -> None:
        text = "“这次必须通过试炼。”"
        document = build_chapter_document(
            "Author/work/0036", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        whole = SourceSpan(document.chapter_id, unit.start, unit.end, unit.text, unit.unit_id)
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "guessed", "canonical_name": "角色甲", "kind": "person",
                          "aliases": [], "evidence": [whole.to_dict()]}],
            "facts": [{"fact_id": "quote", "chapter_id": document.chapter_id, "kind": "information",
                       "subject_id": "guessed", "predicate": "表示", "value": "必须通过试炼", "object_id": "",
                       "certainty": 1.0, "evidence": [whole.to_dict()]}],
            "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        quality = assess_annotation_quality(document, annotation)

        self.assertTrue(any(item.code == "fact_quote_subject_ungrounded" for item in quality.issues))

    def test_first_module_accepts_standalone_quote_bound_by_speech_event(self) -> None:
        text = "角色甲开口说道：\n“这次必须通过试炼。”"
        document = build_chapter_document(
            "Author/work/0037", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        first, second = document.units
        first_span = SourceSpan(document.chapter_id, first.start, first.end, first.text, first.unit_id)
        second_span = SourceSpan(document.chapter_id, second.start, second.end, second.text, second.unit_id)
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "speaker", "canonical_name": "角色甲", "kind": "person",
                          "aliases": [], "evidence": [first_span.to_dict()]}],
            "facts": [{"fact_id": "quote", "chapter_id": document.chapter_id, "kind": "information",
                       "subject_id": "speaker", "predicate": "表示", "value": "必须通过试炼", "object_id": "",
                       "certainty": 1.0, "evidence": [second_span.to_dict()]}],
            "events": [{"event_id": "speech", "chapter_id": document.chapter_id, "order": 0,
                        "summary": "角色甲表示必须通过", "participant_ids": ["speaker"],
                        "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "表示必须通过试炼", "obstacle": "",
                        "decision": "", "outcome_fact_ids": ["quote"], "cost_fact_ids": [],
                        "evidence": [second_span.to_dict()], "action_type": "speech", "actor_id": "speaker",
                        "target_ids": [], "basis_fact_ids": ["quote"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        codes = {item.code for item in assess_annotation_quality(document, annotation).issues}

        self.assertNotIn("fact_quote_subject_ungrounded", codes)

    def test_entity_enrichment_prefers_exact_canonical_mention_over_existing_alias(self) -> None:
        text = "中年人先开口。\n黑衣中年人随后出现。"
        document = build_chapter_document(
            "Author/work/0038", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        first, second = document.units
        alias_span = SourceSpan(document.chapter_id, first.start, first.start + 3, "中年人", first.unit_id)
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "person", "canonical_name": "黑衣中年人", "kind": "person",
                          "aliases": ["中年人"], "evidence": [alias_span.to_dict()]}],
            "facts": [], "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        enriched = enrich_entity_evidence(document, annotation)

        self.assertTrue(any(span.quote == "黑衣中年人" and span.unit_id == second.unit_id
                            for span in enriched.entities[0].evidence))

    def test_source_atom_reconciliation_does_not_promote_daily_texture(self) -> None:
        text = "角色甲落座吃饭，亲人给他夹菜。"
        document = build_chapter_document(
            "Author/work/0017", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        units = (AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text),)
        prompt = atomic_semantic_reconciliation_prompt(
            document, units, {"entities": [], "facts": [], "events": []},
        )

        self.assertIn("删掉该动作后", prompt)
        self.assertIn("不是缺失事件", prompt)
        self.assertIn("不能用“家庭互动”", prompt)

    def test_cross_chunk_relational_names_use_confirmed_owner_aliases(self) -> None:
        text = "角色甲又叫阿甲。角色甲父亲回家。阿甲的父亲落泪。"
        document = build_chapter_document(
            "Author/work/0013", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        def span(quote: str):
            start = text.index(quote)
            return {"chapter_id": document.chapter_id, "start": start, "end": start + len(quote),
                    "quote": quote, "unit_id": unit.unit_id}
        first = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "lead-a", "canonical_name": "角色甲", "kind": "person", "aliases": ["阿甲"], "evidence": [span("角色甲又叫阿甲")]},
                {"entity_id": "father-a", "canonical_name": "角色甲父亲", "kind": "person", "aliases": [], "evidence": [span("角色甲父亲")]},
            ], "facts": [], "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })
        second = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "lead-b", "canonical_name": "阿甲", "kind": "person", "aliases": ["角色甲"], "evidence": [span("阿甲")]},
                {"entity_id": "father-b", "canonical_name": "阿甲的父亲", "kind": "person", "aliases": [], "evidence": [span("阿甲的父亲")]},
            ], "facts": [], "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        merged = merge_annotations(document, [first, second])

        self.assertEqual(len(merged.entities), 2)
        father = next(item for item in merged.entities if "父亲" in item.canonical_name)
        self.assertIn("阿甲的父亲", father.aliases)

    def test_cross_chunk_relational_alias_merges_generic_canonical_role(self) -> None:
        text = "角色甲在院里。角色甲的父亲回家。\n父亲落泪，众人称他为角色甲父亲。"
        document = build_chapter_document(
            "Author/work/0023", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        first_unit, second_unit = document.units

        def span(unit, quote: str):
            local = unit.text.index(quote)
            return {"chapter_id": document.chapter_id, "start": unit.start + local,
                    "end": unit.start + local + len(quote), "quote": quote, "unit_id": unit.unit_id}

        first = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "owner", "canonical_name": "角色甲", "kind": "person", "aliases": [],
                 "evidence": [span(first_unit, "角色甲")]},
                {"entity_id": "father-a", "canonical_name": "角色甲的父亲", "kind": "person", "aliases": [],
                 "evidence": [span(first_unit, "角色甲的父亲")]},
            ],
            "facts": [], "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })
        second = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "father-b", "canonical_name": "父亲", "kind": "person",
                          "aliases": ["角色甲父亲"], "evidence": [span(second_unit, "父亲")]}],
            "facts": [], "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        merged = merge_annotations(document, [first, second])

        self.assertEqual(len(merged.entities), 2)
        father = next(item for item in merged.entities if item.canonical_name != "角色甲")
        self.assertEqual(len(father.evidence), 2)
        self.assertIn("父亲", father.aliases)

    def test_adjacent_overlap_snippets_of_same_event_are_duplicates(self) -> None:
        text = "庞大的力道忽然出现，涌向甲，甲被迫飞快退后十多丈。"
        document = build_chapter_document(
            "Author/work/0011", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        base = {
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "person", "canonical_name": "甲", "kind": "person",
                          "aliases": [], "evidence": [{"chapter_id": document.chapter_id,
                          "start": text.index("甲"), "end": text.index("甲") + 1,
                          "quote": "甲", "unit_id": unit.unit_id}]}],
            "facts": [
                {"fact_id": "fact-a", "chapter_id": document.chapter_id, "kind": "progression",
                 "subject_id": "person", "predicate": "被逼退", "value": "退后十多丈", "object_id": "",
                 "certainty": 1.0, "evidence": [{"chapter_id": document.chapter_id,
                 "start": 0, "end": 9, "quote": text[:9], "unit_id": unit.unit_id}]},
                {"fact_id": "fact-b", "chapter_id": document.chapter_id, "kind": "progression",
                 "subject_id": "person", "predicate": "被迫退后", "value": "十多丈", "object_id": "",
                 "certainty": 1.0, "evidence": [{"chapter_id": document.chapter_id,
                 "start": text.index("飞快退后"), "end": text.index("飞快退后") + 4,
                 "quote": "飞快退后", "unit_id": unit.unit_id}]},
            ],
            "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        }
        annotation = ChapterAnnotation.from_dict(base)
        first = annotation.events
        from pipeline.contracts import EventAtom
        left = EventAtom("a", document.chapter_id, 0, "被力道推退十多丈", ("person",), (), (),
                         "被逼退十多丈", "", "", ("fact-a",), (), annotation.facts[0].evidence,
                         "other", "person", (), ("fact-a",))
        right = EventAtom("b", document.chapter_id, 1, "甲被力道逼退十余丈", ("person",), (), (),
                          "被力道涌向后退", "", "", ("fact-b",), (), annotation.facts[1].evidence,
                          "state_change", "person", (), ("fact-b",))

        self.assertTrue(_is_overlapping_event_duplicate(left, right))

    def test_compound_core_actions_fail_first_module_quality(self) -> None:
        text = "弟子把少年带到山上，随后给他喂药。"
        document = build_chapter_document(
            "Author/work/0012", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        whole = {"chapter_id": document.chapter_id, "start": 0, "end": len(text),
                 "quote": text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "actor", "canonical_name": "弟子", "kind": "person", "aliases": [], "evidence": [whole]},
                {"entity_id": "target", "canonical_name": "少年", "kind": "person", "aliases": [], "evidence": [whole]},
            ],
            "facts": [{"fact_id": "progress", "chapter_id": document.chapter_id, "kind": "progression",
                       "subject_id": "actor", "predicate": "处置", "value": "带上山并给药", "object_id": "target",
                       "certainty": 1.0, "evidence": [whole]}],
            "events": [{"event_id": "compound", "chapter_id": document.chapter_id, "order": 0,
                        "summary": "弟子带少年上山随后给药", "participant_ids": ["actor", "target"],
                        "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "带上山并给药",
                        "obstacle": "", "decision": "", "outcome_fact_ids": ["progress"], "cost_fact_ids": [],
                        "evidence": [whole], "action_type": "other", "actor_id": "actor",
                        "target_ids": ["target"], "basis_fact_ids": ["progress"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        report = assess_annotation_quality(document, annotation)

        self.assertTrue(any(issue.code == "event_compound_core_action" for issue in report.issues))

    def test_reported_movement_inside_speech_is_not_a_compound_action(self) -> None:
        text = "四叔说：到地方了，这是我家，你先休息。"
        document = build_chapter_document(
            "Author/work/0041", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        whole = {"chapter_id": document.chapter_id, "start": 0, "end": len(text),
                 "quote": text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "speaker", "canonical_name": "四叔", "kind": "person",
                 "aliases": [], "evidence": [whole]},
                {"entity_id": "listener", "canonical_name": "晚辈", "kind": "person",
                 "aliases": [], "evidence": [whole]},
            ],
            "facts": [{"fact_id": "message", "chapter_id": document.chapter_id,
                       "kind": "information", "subject_id": "speaker", "predicate": "告知并安排",
                       "value": "已经到地方并先休息", "object_id": "listener", "certainty": 1.0,
                       "evidence": [whole]}],
            "events": [{"event_id": "speech", "chapter_id": document.chapter_id, "order": 0,
                        "summary": "四叔告知到达并安排休息", "participant_ids": ["speaker", "listener"],
                        "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "告知到达并安排休息",
                        "obstacle": "", "decision": "", "outcome_fact_ids": ["message"],
                        "cost_fact_ids": [], "evidence": [whole], "action_type": "speech",
                        "actor_id": "speaker", "target_ids": ["listener"],
                        "basis_fact_ids": ["message"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        report = assess_annotation_quality(document, annotation)

        self.assertFalse(any(issue.code == "event_compound_core_action" for issue in report.issues))

    def test_nonverbal_response_is_not_forced_into_speech(self) -> None:
        text = "角色甲轻笑，点头回应。"
        document = build_chapter_document(
            "Author/work/0042", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": 0, "end": len(text),
                "quote": text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "actor", "canonical_name": "角色甲", "kind": "person",
                          "aliases": [], "evidence": [span]}],
            "facts": [{"fact_id": "response", "chapter_id": document.chapter_id,
                       "kind": "progression", "subject_id": "actor", "predicate": "点头回应",
                       "value": "", "object_id": "", "certainty": 1.0, "evidence": [span]}],
            "events": [{"event_id": "nod", "chapter_id": document.chapter_id, "order": 0,
                        "summary": "角色甲点头回应", "participant_ids": ["actor"],
                        "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "轻笑点头回应",
                        "obstacle": "", "decision": "", "outcome_fact_ids": ["response"],
                        "cost_fact_ids": [], "evidence": [span], "action_type": "other",
                        "actor_id": "actor", "target_ids": [], "basis_fact_ids": ["response"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        issues = _event_frame_issues(annotation.events[0], {"response": annotation.facts[0]})

        self.assertFalse(any(code == "event_explicit_speech_type_mismatch" for code, _ in issues))

    def test_speaker_resolver_does_not_choose_physical_action_target(self) -> None:
        text = "四叔拍了拍铁柱肩膀，说道：“先休息。”"
        document = build_chapter_document(
            "Author/work/0043", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        whole = {"chapter_id": document.chapter_id, "start": 0, "end": len(text),
                 "quote": text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "uncle", "canonical_name": "四叔", "kind": "person",
                 "aliases": [], "evidence": [whole]},
                {"entity_id": "listener", "canonical_name": "铁柱", "kind": "person",
                 "aliases": [], "evidence": [whole]},
            ],
            "facts": [], "events": [], "scenes": [], "state_changes": [],
            "schema_version": "2.2",
        })

        speakers = _explicit_local_speaker_ids(document, annotation, unit.unit_id)

        self.assertEqual(speakers, ("uncle",))

    def test_speaker_resolver_does_not_choose_observed_person(self) -> None:
        text = "老者看到铁柱后略点头，说道：“跟着王卓学。”"
        document = build_chapter_document(
            "Author/work/0044", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        whole = {"chapter_id": document.chapter_id, "start": 0, "end": len(text),
                 "quote": text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "elder", "canonical_name": "老者", "kind": "person",
                 "aliases": [], "evidence": [whole]},
                {"entity_id": "listener", "canonical_name": "铁柱", "kind": "person",
                 "aliases": [], "evidence": [whole]},
            ],
            "facts": [], "events": [], "scenes": [], "state_changes": [],
            "schema_version": "2.2",
        })

        speakers = _explicit_local_speaker_ids(document, annotation, unit.unit_id)

        self.assertEqual(speakers, ("elder",))

    def test_named_character_cannot_absorb_another_person_behind_them(self) -> None:
        text = "甲咬牙坚持，就在这时，在他身后一个少年忽然踩空。"
        document = build_chapter_document(
            "Author/work/0010", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": 0, "end": len(text),
                "quote": "甲咬牙坚持，在他身后一个少年忽然踩空", "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "e1", "canonical_name": "甲", "kind": "person",
                          "aliases": ["少年"], "evidence": [span]}],
            "facts": [{"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "progression",
                       "subject_id": "e1", "predicate": "踩空", "value": "跌落", "object_id": "",
                       "certainty": 1.0, "evidence": [span]}],
            "events": [{"event_id": "v1", "chapter_id": document.chapter_id, "order": 0,
                        "summary": "少年踩空", "participant_ids": ["e1"], "trigger_fact_ids": [],
                        "precondition_fact_ids": [], "action": "踩空", "obstacle": "", "decision": "",
                        "outcome_fact_ids": ["f1"], "cost_fact_ids": [], "evidence": [span],
                        "action_type": "state_change", "actor_id": "e1", "target_ids": [],
                        "basis_fact_ids": ["f1"]}],
            "scenes": [{"scene_id": "s1", "chapter_id": document.chapter_id, "order": 0,
                        "participant_ids": ["e1"], "event_ids": ["v1"], "objective": "踩空",
                        "entry_fact_ids": [], "exit_fact_ids": ["f1"], "tension": 2, "evidence": [span]}],
            "state_changes": [{"change_id": "c1", "chapter_id": document.chapter_id,
                               "event_id": "v1", "operation": "add", "before_fact_id": "",
                               "after_fact_id": "f1"}], "schema_version": "2.2",
        })

        quality = assess_annotation_quality(document, annotation)

        self.assertTrue(any(item.code == "entity_distinct_referent_merged" for item in quality.issues))

    def test_canonical_name_cannot_be_another_entities_alias(self) -> None:
        text = "甲向乙交代事务，乙点头。"
        document = build_chapter_document(
            "Author/work/0011", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": 0, "end": len(text),
                "quote": text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "e1", "canonical_name": "甲", "kind": "person",
                 "aliases": ["乙"], "evidence": [span]},
                {"entity_id": "e2", "canonical_name": "乙", "kind": "person",
                 "aliases": [], "evidence": [span]},
            ],
            "facts": [{"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "progression",
                       "subject_id": "e1", "predicate": "交代", "value": "事务", "object_id": "e2",
                       "certainty": 1.0, "evidence": [span]}],
            "events": [{"event_id": "v1", "chapter_id": document.chapter_id, "order": 0,
                        "summary": "甲交代事务", "participant_ids": ["e1", "e2"], "trigger_fact_ids": [],
                        "precondition_fact_ids": [], "action": "交代事务", "obstacle": "", "decision": "",
                        "outcome_fact_ids": ["f1"], "cost_fact_ids": [], "evidence": [span],
                        "action_type": "speech", "actor_id": "e1", "target_ids": ["e2"],
                        "basis_fact_ids": ["f1"]}],
            "scenes": [{"scene_id": "s1", "chapter_id": document.chapter_id, "order": 0,
                        "participant_ids": ["e1", "e2"], "event_ids": ["v1"], "objective": "交代事务",
                        "entry_fact_ids": [], "exit_fact_ids": ["f1"], "tension": 1, "evidence": [span]}],
            "state_changes": [{"change_id": "c1", "chapter_id": document.chapter_id,
                               "event_id": "v1", "operation": "add", "before_fact_id": "",
                               "after_fact_id": "f1"}], "schema_version": "2.2",
        })

        quality = assess_annotation_quality(document, annotation)

        self.assertTrue(any(item.code == "entity_surface_collision" for item in quality.issues))

    def test_reporting_knowledge_fact_is_normalized_to_information(self) -> None:
        text = "甲宣布第二项测试。"
        document = build_chapter_document(
            "Author/work/0008", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": 0, "end": len(text), "quote": text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "e1", "canonical_name": "甲", "kind": "person", "aliases": [], "evidence": [span]}],
            "facts": [{"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "knowledge", "subject_id": "e1", "predicate": "宣布", "value": "第二项测试", "object_id": "", "certainty": 1.0, "evidence": [span]}],
            "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        normalized = normalize_fact_role_contracts(annotation)

        self.assertEqual(normalized.facts[0].kind, "information")

    def test_event_outcome_must_share_an_evidence_unit(self) -> None:
        text = "甲说第一件事。\n" + ("旁白。" * 80) + "\n甲说第二件事。"
        document = build_chapter_document(
            "Author/work/0009", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        first, second = document.units[0], document.units[-1]
        first_span = {"chapter_id": document.chapter_id, "start": first.start, "end": first.end, "quote": first.text, "unit_id": first.unit_id}
        second_span = {"chapter_id": document.chapter_id, "start": second.start, "end": second.end, "quote": second.text, "unit_id": second.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "e1", "canonical_name": "甲", "kind": "person", "aliases": [], "evidence": [first_span]}],
            "facts": [{"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "information", "subject_id": "e1", "predicate": "说明", "value": "第二件事", "object_id": "", "certainty": 1.0, "evidence": [second_span]}],
            "events": [{"event_id": "v1", "chapter_id": document.chapter_id, "order": 0, "summary": "甲说第一件事", "participant_ids": ["e1"], "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "说第一件事", "obstacle": "", "decision": "", "outcome_fact_ids": ["f1"], "cost_fact_ids": [], "evidence": [first_span], "action_type": "speech", "actor_id": "e1", "target_ids": [], "basis_fact_ids": ["f1"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        issues = _event_frame_issues(annotation.events[0], {"f1": annotation.facts[0]})

        self.assertIn("event_outcome_evidence_disjoint", {code for code, _ in issues})

    def test_adjacent_outcome_requires_its_own_event_evidence_span(self) -> None:
        text = "甲递出令牌。\n乙接过令牌。"
        document = build_chapter_document(
            "Author/work/0012", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        first, second = document.units
        span1 = {"chapter_id": document.chapter_id, "start": first.start, "end": first.end, "quote": first.text, "unit_id": first.unit_id}
        span2 = {"chapter_id": document.chapter_id, "start": second.start, "end": second.end, "quote": second.text, "unit_id": second.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "e1", "canonical_name": "甲", "kind": "person", "aliases": [], "evidence": [span1]},
                {"entity_id": "e2", "canonical_name": "乙", "kind": "person", "aliases": [], "evidence": [span2]},
            ],
            "facts": [{"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "resource", "subject_id": "e2", "predicate": "获得", "value": "令牌", "object_id": "", "certainty": 1.0, "evidence": [span2]}],
            "events": [{"event_id": "v1", "chapter_id": document.chapter_id, "order": 0, "summary": "甲交付令牌", "participant_ids": ["e1", "e2"], "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "递出令牌", "obstacle": "", "decision": "", "outcome_fact_ids": ["f1"], "cost_fact_ids": [], "evidence": [span1, span2], "action_type": "transfer", "actor_id": "e1", "target_ids": ["e2"], "basis_fact_ids": ["f1"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        issues = _event_frame_issues(annotation.events[0], {"f1": annotation.facts[0]})

        self.assertNotIn("event_outcome_evidence_disjoint", {code for code, _ in issues})

    def test_explicit_aspiration_is_goal_instead_of_knowledge(self) -> None:
        text = "甲内心对于修行，更加向往。"
        document = build_chapter_document(
            "Author/work/0003", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": 0, "end": len(text), "quote": text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "e1", "canonical_name": "甲", "kind": "person", "aliases": [], "evidence": [span]}],
            "facts": [{"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "knowledge", "subject_id": "e1", "predicate": "更加向往", "value": "修行", "object_id": "", "certainty": 1.0, "evidence": [span]}],
            "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        normalized = normalize_fact_role_contracts(annotation)

        self.assertEqual(normalized.facts[0].kind, "goal")

    def test_same_cognition_proposition_deduplicates_viewpoint_predicates(self) -> None:
        text = "甲觉得乙很机灵。"
        document = build_chapter_document(
            "Author/work/0006", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": 0, "end": len(text), "quote": text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "e1", "canonical_name": "甲", "kind": "person", "aliases": [], "evidence": [span]},
                {"entity_id": "e2", "canonical_name": "乙", "kind": "person", "aliases": [], "evidence": [span]},
            ],
            "facts": [
                {"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "knowledge", "subject_id": "e1", "predicate": "觉得", "value": "乙最为机灵", "object_id": "e2", "certainty": 1.0, "evidence": [span]},
                {"fact_id": "f2", "chapter_id": document.chapter_id, "kind": "knowledge", "subject_id": "e1", "predicate": "认为", "value": "乙机灵", "object_id": "e2", "certainty": 1.0, "evidence": [span]},
            ],
            "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        normalized = normalize_fact_role_contracts(annotation)

        self.assertEqual(len(normalized.facts), 1)

    def test_fact_redirect_collapses_now_identical_events(self) -> None:
        text = "甲说明三人是推荐人选。"
        document = build_chapter_document(
            "Author/work/0009", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": 0, "end": len(text), "quote": text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "e1", "canonical_name": "甲", "kind": "person", "aliases": [], "evidence": [span]}],
            "facts": [
                {"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "knowledge", "subject_id": "e1", "predicate": "确认", "value": "三人是推荐人选", "object_id": "", "certainty": 1.0, "evidence": [span]},
                {"fact_id": "f2", "chapter_id": document.chapter_id, "kind": "knowledge", "subject_id": "e1", "predicate": "说明", "value": "三人是推荐人选", "object_id": "", "certainty": 1.0, "evidence": [span]},
            ],
            "events": [
                {"event_id": "v1", "chapter_id": document.chapter_id, "order": 0, "summary": "甲确认人选", "participant_ids": ["e1"], "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "确认三人是推荐人选", "obstacle": "", "decision": "", "outcome_fact_ids": ["f1"], "cost_fact_ids": [], "evidence": [span], "action_type": "speech", "actor_id": "e1", "target_ids": [], "basis_fact_ids": ["f1"]},
                {"event_id": "v2", "chapter_id": document.chapter_id, "order": 1, "summary": "甲说明推荐人选", "participant_ids": ["e1"], "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "说明三人是推荐人选", "obstacle": "", "decision": "", "outcome_fact_ids": ["f2"], "cost_fact_ids": [], "evidence": [span], "action_type": "speech", "actor_id": "e1", "target_ids": [], "basis_fact_ids": ["f2"]},
            ],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        normalized = normalize_fact_role_contracts(annotation)

        self.assertEqual(len(normalized.facts), 1)
        self.assertEqual(len(normalized.events), 1)

    def test_resource_stage_predicates_are_not_deduplicated(self) -> None:
        text = "甲从怀里拿出玉盒，双手交上。"
        document = build_chapter_document(
            "Author/work/0007", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": 0, "end": len(text), "quote": text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "e1", "canonical_name": "甲", "kind": "person", "aliases": [], "evidence": [span]},
                {"entity_id": "e2", "canonical_name": "玉盒", "kind": "item", "aliases": [], "evidence": [span]},
            ],
            "facts": [
                {"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "resource", "subject_id": "e1", "predicate": "拥有", "value": "玉盒", "object_id": "e2", "certainty": 1.0, "evidence": [span]},
                {"fact_id": "f2", "chapter_id": document.chapter_id, "kind": "resource", "subject_id": "e1", "predicate": "献出", "value": "玉盒", "object_id": "e2", "certainty": 1.0, "evidence": [span]},
            ],
            "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        normalized = normalize_fact_role_contracts(annotation)

        self.assertEqual(len(normalized.facts), 2)

    def test_short_past_fact_is_not_collapsed_into_larger_multi_stage_statement(self) -> None:
        text = "他当初被逼走，如今对方又来道喜。"
        document = build_chapter_document(
            "Author/work/0008", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": 0, "end": len(text), "quote": text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "e1", "canonical_name": "他", "kind": "person", "aliases": [], "evidence": [span]},
                {"entity_id": "e2", "canonical_name": "对方", "kind": "person", "aliases": [], "evidence": [span]},
            ],
            "facts": [
                {"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "knowledge", "subject_id": "e1", "predicate": "解释", "value": "当初被对方逼走", "object_id": "e2", "certainty": 1.0, "evidence": [span]},
                {"fact_id": "f2", "chapter_id": document.chapter_id, "kind": "knowledge", "subject_id": "e1", "predicate": "抱怨", "value": "当初被对方逼走如今对方又来道喜", "object_id": "e2", "certainty": 1.0, "evidence": [span]},
            ],
            "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        normalized = normalize_fact_role_contracts(annotation)

        self.assertEqual(len(normalized.facts), 2)

    def test_cognition_with_aspiration_in_value_is_not_reclassified(self) -> None:
        text = "甲知道乙向往修行。"
        document = build_chapter_document(
            "Author/work/0004", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": 0, "end": len(text), "quote": text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "e1", "canonical_name": "甲", "kind": "person", "aliases": [], "evidence": [span]}],
            "facts": [{"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "knowledge", "subject_id": "e1", "predicate": "知道", "value": "乙向往修行", "object_id": "", "certainty": 1.0, "evidence": [span]}],
            "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        normalized = normalize_fact_role_contracts(annotation)

        self.assertEqual(normalized.facts[0].kind, "knowledge")

    def test_relationship_duplicate_redirects_every_fact_reference(self) -> None:
        text = "甲是乙的父亲。"
        document = build_chapter_document(
            "Author/work/0001", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": 0, "end": len(text), "quote": text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "e1", "canonical_name": "甲", "kind": "person", "aliases": [], "evidence": [span]},
                {"entity_id": "e2", "canonical_name": "乙", "kind": "person", "aliases": [], "evidence": [span]},
            ],
            "facts": [
                {"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "relationship", "subject_id": "e1", "predicate": "是", "value": "父亲", "object_id": "e2", "certainty": 1.0, "evidence": [span]},
                {"fact_id": "f2", "chapter_id": document.chapter_id, "kind": "relationship", "subject_id": "e1", "predicate": "是父亲", "value": "乙的父亲", "object_id": "e2", "certainty": 1.0, "evidence": [span]},
            ],
            "events": [{"event_id": "v1", "chapter_id": document.chapter_id, "order": 0, "summary": "确认关系", "participant_ids": ["e1", "e2"], "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "确认父子关系", "obstacle": "", "decision": "", "outcome_fact_ids": ["f2"], "cost_fact_ids": [], "evidence": [span], "action_type": "state_change", "actor_id": "e1", "target_ids": ["e2"], "basis_fact_ids": ["f2"]}],
            "scenes": [{"scene_id": "s1", "chapter_id": document.chapter_id, "order": 0, "participant_ids": ["e1", "e2"], "event_ids": ["v1"], "objective": "确认关系", "entry_fact_ids": ["f1"], "exit_fact_ids": ["f2"], "tension": 0, "evidence": [span]}],
            "state_changes": [{"change_id": "c1", "chapter_id": document.chapter_id, "event_id": "v1", "operation": "update", "before_fact_id": "f1", "after_fact_id": "f2"}],
            "schema_version": "2.2",
        })

        normalized = normalize_fact_role_contracts(annotation)

        self.assertEqual(len(normalized.facts), 1)
        kept_id = normalized.facts[0].fact_id
        self.assertEqual(normalized.events[0].basis_fact_ids, (kept_id,))
        self.assertEqual(normalized.events[0].outcome_fact_ids, (kept_id,))
        self.assertEqual(normalized.scenes[0].entry_fact_ids, (kept_id,))
        self.assertEqual(normalized.scenes[0].exit_fact_ids, (kept_id,))
        self.assertEqual(normalized.state_changes, ())

    def test_relationship_aspect_wrappers_do_not_duplicate_same_relation(self) -> None:
        text = "甲成为乙的弟子。"
        document = build_chapter_document(
            "Author/work/0002", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": 0, "end": len(text), "quote": text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "e1", "canonical_name": "甲", "kind": "person", "aliases": [], "evidence": [span]},
                {"entity_id": "e2", "canonical_name": "乙", "kind": "person", "aliases": [], "evidence": [span]},
            ],
            "facts": [
                {"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "relationship", "subject_id": "e1", "predicate": "成为弟子", "value": "师徒关系", "object_id": "e2", "certainty": 1.0, "evidence": [span]},
                {"fact_id": "f2", "chapter_id": document.chapter_id, "kind": "relationship", "subject_id": "e1", "predicate": "是弟子", "value": "师徒关系", "object_id": "e2", "certainty": 1.0, "evidence": [span]},
            ],
            "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        normalized = normalize_fact_role_contracts(annotation)

        self.assertEqual(len(normalized.facts), 1)
        self.assertIn(normalized.facts[0].predicate, {"成为弟子", "是弟子"})

    def test_location_fact_materializes_only_verbatim_location_entity(self) -> None:
        text = "甲到达四叔家。"
        document = build_chapter_document(
            "Author/work/0001", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": 0, "end": len(text), "quote": text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": f"{document.chapter_id}:entity-001", "canonical_name": "甲", "kind": "person", "aliases": [], "evidence": [span]}],
            "facts": [{"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "location", "subject_id": f"{document.chapter_id}:entity-001", "predicate": "到达", "value": "四叔家", "object_id": "", "certainty": 1.0, "evidence": [span]}],
            "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        normalized = normalize_location_fact_objects(annotation)

        self.assertEqual(len(normalized.entities), 2)
        self.assertEqual(normalized.entities[-1].canonical_name, "四叔家")
        self.assertEqual(normalized.entities[-1].kind, "location")
        self.assertEqual(normalized.facts[0].object_id, normalized.entities[-1].entity_id)
        self.assertEqual(normalized.entities[-1].evidence[0].quote, "四叔家")

    def test_location_fact_rebinds_a_well_typed_but_wrong_place(self) -> None:
        text = "少年被救下，轻轻落在山脚。"
        document = build_chapter_document(
            "Author/work/0002", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        whole = {"chapter_id": document.chapter_id, "start": 0, "end": len(text),
                 "quote": text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "person", "canonical_name": "少年", "kind": "person", "aliases": [], "evidence": [whole]},
                {"entity_id": "wrong-place", "canonical_name": "石阶", "kind": "location", "aliases": [], "evidence": [whole]},
            ],
            "facts": [{"fact_id": "arrival", "chapter_id": document.chapter_id,
                       "kind": "location", "subject_id": "person", "predicate": "到达",
                       "value": "山脚", "object_id": "wrong-place", "certainty": 1.0,
                       "evidence": [whole]}],
            "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        normalized = normalize_location_fact_objects(annotation)
        destination = next(item for item in normalized.entities if item.entity_id == normalized.facts[0].object_id)

        self.assertNotEqual(normalized.facts[0].object_id, "wrong-place")
        self.assertEqual(destination.canonical_name, "山脚")
        self.assertEqual(destination.kind, "location")
        self.assertFalse(any(
            issue.code == "location_object_ungrounded"
            for issue in assess_annotation_quality(document, normalized).issues
        ))

    def test_organization_site_is_split_from_organization_and_binds_movement(self) -> None:
        text = "青年落下，到达青云门的山门。"
        document = build_chapter_document(
            "Author/work/0001", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": 0, "end": len(text),
                "quote": text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "e1", "canonical_name": "青年", "kind": "person", "aliases": [], "evidence": [span]},
                {"entity_id": "e2", "canonical_name": "青云门", "kind": "organization", "aliases": [], "evidence": [span]},
            ],
            "facts": [{"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "progression",
                       "subject_id": "e1", "predicate": "抵达", "value": "青云门山门", "object_id": "e2",
                       "certainty": 1.0, "evidence": [span]}],
            "events": [], "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        normalized = normalize_location_fact_objects(annotation)

        self.assertEqual(normalized.facts[0].kind, "location")
        self.assertNotEqual(normalized.facts[0].object_id, "e2")
        site = next(item for item in normalized.entities if item.entity_id == normalized.facts[0].object_id)
        self.assertEqual(site.kind, "location")
        self.assertEqual(site.canonical_name, "青云门的山门")

    def test_carried_participant_is_promoted_only_when_name_matches_action(self) -> None:
        document = build_chapter_document(
            "Author/work/0001", hashlib.sha256("甲带着三名弟子登山。".encode()).hexdigest(),
            "甲带着三名弟子登山。",
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": 0, "end": len(unit.text), "quote": unit.text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "e1", "canonical_name": "甲", "kind": "person", "aliases": [], "evidence": [span]},
                {"entity_id": "e2", "canonical_name": "三名弟子", "kind": "group", "aliases": [], "evidence": [span]},
                {"entity_id": "e3", "canonical_name": "旁观者", "kind": "person", "aliases": [], "evidence": [span]},
            ],
            "facts": [{"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "progression", "subject_id": "e1", "predicate": "带领", "value": "三名弟子登山", "object_id": "", "certainty": 1.0, "evidence": [span]}],
            "events": [{"event_id": "v1", "chapter_id": document.chapter_id, "order": 0, "summary": "甲带着三名弟子登山", "participant_ids": ["e1", "e2", "e3"], "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "带着三名弟子登山", "obstacle": "", "decision": "", "outcome_fact_ids": ["f1"], "cost_fact_ids": [], "evidence": [span], "action_type": "other", "actor_id": "e1", "target_ids": [], "basis_fact_ids": ["f1"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })
        normalized = normalize_carried_event_targets(annotation)
        self.assertEqual(normalized.events[0].target_ids, ("e2",))

    def test_self_targeted_carrier_materializes_explicit_affected_group(self) -> None:
        text = "几名弟子抓起几个哭泣者，迅速离开。"
        document = build_chapter_document(
            "Author/work/0005", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        payload = {
            "entities": [{"id": "e1", "name": "几名弟子", "kind": "人物群体", "aliases": [],
                          "evidence": [{"unit_id": unit.unit_id, "quote": "几名弟子"}]}],
            "facts": [{"id": "f1", "kind": "进展", "subject_id": "e1", "predicate": "抓起并带离",
                       "value": "几个哭泣者", "object_id": "", "certainty": "明确",
                       "evidence": [{"unit_id": unit.unit_id, "quote": "抓起几个哭泣者"}]}],
            "events": [{"id": "v1", "order": 1, "summary": "弟子带走哭泣者", "action_type": "其他",
                        "actor_id": "e1", "target_ids": ["e1"], "basis_fact_ids": ["f1"],
                        "participant_ids": ["e1"], "trigger_fact_ids": [], "precondition_fact_ids": [],
                        "action": "抓起几个哭泣者离开", "obstacle": "", "decision": "",
                        "outcome_fact_ids": ["f1"], "cost_fact_ids": [],
                        "evidence": [{"unit_id": unit.unit_id, "quote": "抓起几个哭泣者"}]}],
            "scenes": [], "state_changes": [],
        }
        units = (AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text),)

        cleaned = sanitize_draft(document, payload, units)

        target_id = cleaned["events"][0]["target_ids"][0]
        self.assertNotEqual(target_id, "e1")
        target = next(item for item in cleaned["entities"] if item["id"] == target_id)
        self.assertEqual(target["name"], "几个哭泣者")
        self.assertEqual(target["kind"], "人物群体")

    def test_carried_participant_matches_equivalent_chinese_counters(self) -> None:
        text = "甲带着三个天之骄子登山。"
        document = build_chapter_document(
            "Author/work/0005", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": 0, "end": len(text), "quote": text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "e1", "canonical_name": "甲", "kind": "person", "aliases": [], "evidence": [span]},
                {"entity_id": "e2", "canonical_name": "三名天之骄子", "kind": "group", "aliases": [], "evidence": [span]},
            ],
            "facts": [{"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "progression", "subject_id": "e1", "predicate": "带领", "value": "三个天之骄子登山", "object_id": "", "certainty": 1.0, "evidence": [span]}],
            "events": [{"event_id": "v1", "chapter_id": document.chapter_id, "order": 0, "summary": "甲带着三个天之骄子登山", "participant_ids": ["e1", "e2"], "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "带着三个天之骄子登山", "obstacle": "", "decision": "", "outcome_fact_ids": ["f1"], "cost_fact_ids": [], "evidence": [span], "action_type": "other", "actor_id": "e1", "target_ids": [], "basis_fact_ids": ["f1"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        normalized = normalize_carried_event_targets(annotation)

        self.assertEqual(normalized.events[0].target_ids, ("e2",))

    def test_fact_gap_limit_scales_with_window_and_remains_bounded(self) -> None:
        self.assertEqual(_allowed_fact_evidence_gap(500), 320)
        self.assertEqual(_allowed_fact_evidence_gap(965), 338)
        self.assertEqual(_allowed_fact_evidence_gap(5_000), 480)

    def test_largest_gap_reports_exact_intersecting_source_units(self) -> None:
        units = (
            AnnotationSourceSlice("u1", "u1", 0, 50, "甲" * 50),
            AnnotationSourceSlice("u2", "u2", 50, 150, "乙" * 100),
            AnnotationSourceSlice("u3", "u3", 150, 220, "丙" * 70),
        )
        self.assertEqual(_largest_evidence_gap((0, 40, 180, 220), units), (40, 180, 140, ("u1", "u2", "u3")))

    def test_atomic_patch_prompt_has_concrete_fact_example_and_gap_instruction(self) -> None:
        text = "人物甲完成原文明确进展。"
        document = build_chapter_document("Author/work/0001", hashlib.sha256(text.encode()).hexdigest(), text)
        units = (AnnotationSourceSlice(document.units[0].unit_id, document.units[0].unit_id, 0, len(text), text),)
        prompt = atomic_patch_prompt(
            document, units, {"entities": [], "facts": [], "events": []},
            ("facts: 空档偏移 2-8，涉及原文单元 ['u1']",), "Author/work/0001:chunk-01",
        )
        self.assertIn('"upsert_facts": [', prompt)
        self.assertIn('"id": "f2"', prompt)
        self.assertIn("只在这些单元中补缺失对象", prompt)
        self.assertIn("一个 fact id 最多只能出现在一个事件", prompt)
        self.assertIn("新增事件必须同时新增一条", prompt)

    def test_atomic_patch_prompt_distinguishes_spoken_proposition_from_emotion(self) -> None:
        text = "人物甲称赞人物乙以后不用再干苦活了。"
        document = build_chapter_document("Author/work/0001", hashlib.sha256(text.encode()).hexdigest(), text)
        units = (AnnotationSourceSlice(document.units[0].unit_id, document.units[0].unit_id, 0, len(text), text),)
        prompt = atomic_patch_prompt(
            document, units, {"entities": [], "facts": [], "events": []},
            ("交流类型没有匹配的行动依据事实",), "Author/work/0001:chunk-01",
        )
        self.assertIn("人物内心情绪", prompt)
        self.assertIn("人物说出的命题", prompt)
        self.assertIn("该事实应标为 `信息`", prompt)
        self.assertIn("不得为了过检把有明确说话动作的事件改成“其他”", prompt)

    def test_perception_fact_label_is_normalized_to_knowledge(self) -> None:
        self.assertEqual(FACT_KIND_MAP["perception"], "knowledge")
        self.assertEqual(FACT_KIND_MAP["感知"], "knowledge")

    def test_atomic_patch_prompt_localizes_zero_based_object_path(self) -> None:
        text = "人物甲看见山门。"
        document = build_chapter_document("Author/work/0001", hashlib.sha256(text.encode()).hexdigest(), text)
        units = (AnnotationSourceSlice(document.units[0].unit_id, document.units[0].unit_id, 0, len(text), text),)
        prompt = atomic_patch_prompt(
            document,
            units,
            {"entities": [], "facts": [{"id": "f-visible"}], "events": []},
            ("facts[0].kind 不受支持",),
            "Author/work/0001:chunk-01",
        )
        self.assertIn("facts[0]（本地ID：f-visible）.kind", prompt)

    def test_collection_level_event_fault_uses_compact_atomic_patch(self) -> None:
        self.assertTrue(_needs_atomic_semantic_patch(["events: 同一结果事实被重复生产"]))
        self.assertTrue(_needs_atomic_semantic_patch(["events[event-1]: 行动依据不匹配"]))
        self.assertFalse(_needs_atomic_semantic_patch(["scenes[scene-1]: 场景引用错误"]))

    def test_atomic_patch_preserves_unmentioned_valid_rows(self) -> None:
        current = {
            "entities": [{"id": "e1", "name": "人物甲"}],
            "facts": [{"id": "f1", "kind": "进展"}],
            "events": [{"id": "v1", "summary": "原事件"}],
        }
        patch = {
            "upsert_entities": [], "upsert_facts": [],
            "upsert_events": [{"id": "v2", "summary": "新事件"}],
            "remove_entity_ids": [], "remove_fact_ids": [], "remove_event_ids": [],
        }
        merged = apply_atomic_patch(current, patch)
        self.assertEqual([item["id"] for item in merged["events"]], ["v1", "v2"])
        self.assertEqual(merged["facts"], current["facts"])

    def test_atomic_patch_can_derive_basis_from_an_explicit_event_outcome(self) -> None:
        current = {"entities": [], "facts": [], "events": []}
        event = {"id": "v1", "outcome_fact_ids": ["f1"], "basis_fact_ids": []}
        patch = {
            "upsert_entities": [], "upsert_facts": [], "upsert_events": [event],
            "remove_entity_ids": [], "remove_fact_ids": [], "remove_event_ids": [],
        }
        merged = apply_atomic_patch(current, patch)
        self.assertEqual(merged["events"][0]["basis_fact_ids"], ["f1"])

    def test_structural_duplicate_collapse_prefers_richer_fact_and_event(self) -> None:
        evidence = [{"unit_id": "u1", "quote": "人物离开并到了门外"}]
        payload = {
            "entities": [{"id": "e1"}, {"id": "place"}],
            "facts": [
                {"id": "f1", "kind": "进展", "subject_id": "e1", "predicate": "离开", "value": "门外", "object_id": "", "evidence": evidence},
                {"id": "f2", "kind": "进展", "subject_id": "e1", "predicate": "离开", "value": "门外", "object_id": "place", "evidence": evidence},
            ],
            "events": [
                {"id": "v1", "action_type": "移动", "actor_id": "e1", "action": "离开", "outcome_fact_ids": ["f1"], "basis_fact_ids": ["f1"], "participant_ids": ["e1"], "evidence": evidence},
                {"id": "v2", "action_type": "移动", "actor_id": "e1", "action": "离开并到了门外", "outcome_fact_ids": ["f2"], "basis_fact_ids": ["f2"], "participant_ids": ["e1"], "evidence": evidence},
            ],
            "scenes": [{"id": "s1", "event_ids": ["v1", "v2"], "entry_fact_ids": [], "exit_fact_ids": ["f1", "f2"]}],
            "state_changes": [{"id": "c1", "event_id": "v1", "before_fact_id": "", "after_fact_id": "f1"}],
        }
        result = _collapse_duplicate_atoms(payload)

        self.assertEqual([item["id"] for item in result["facts"]], ["f2"])
        self.assertEqual([item["id"] for item in result["events"]], ["v2"])
        self.assertEqual(result["scenes"][0]["event_ids"], ["v2"])
        self.assertEqual(result["scenes"][0]["exit_fact_ids"], ["f2"])
        self.assertEqual(result["state_changes"][0]["event_id"], "v2")
        self.assertEqual(result["state_changes"][0]["after_fact_id"], "f2")

    def test_exact_evidence_collapses_paraphrased_duplicate_fact_and_rewrites_event(self) -> None:
        evidence = [{"unit_id": "u1", "quote": "上仙可收你为徒？"}]
        payload = {
            "entities": [{"id": "speaker"}, {"id": "listener"}],
            "facts": [
                {"id": "f1", "kind": "信息", "subject_id": "speaker", "predicate": "询问", "value": "上仙是否收徒", "object_id": "listener", "evidence": evidence},
                {"id": "f2", "kind": "信息", "subject_id": "speaker", "predicate": "询问", "value": "上仙收徒情况", "object_id": "listener", "evidence": evidence},
            ],
            "events": [{
                "id": "v1", "action_type": "交流", "actor_id": "speaker",
                "basis_fact_ids": ["f1"], "outcome_fact_ids": ["f2"],
                "participant_ids": ["speaker", "listener"], "evidence": evidence,
            }],
            "scenes": [], "state_changes": [],
        }

        result = _collapse_duplicate_atoms(payload)

        self.assertEqual(len(result["facts"]), 1)
        kept_id = result["facts"][0]["id"]
        self.assertEqual(result["events"][0]["basis_fact_ids"], [kept_id])
        self.assertEqual(result["events"][0]["outcome_fact_ids"], [kept_id])

    def test_transfer_predicate_synonyms_collapse_to_one_fact(self) -> None:
        evidence = [{"unit_id": "u1", "quote": "把几块肉给他夹过去"}]
        payload = {
            "entities": [{"id": "mother"}, {"id": "son"}],
            "facts": [
                {"id": "f1", "kind": "资源", "subject_id": "mother", "predicate": "夹给",
                 "value": "几块肉", "object_id": "son", "evidence": evidence},
                {"id": "f2", "kind": "资源", "subject_id": "mother", "predicate": "给予",
                 "value": "几块肉", "object_id": "son", "evidence": evidence},
            ],
            "events": [{
                "id": "v1", "action_type": "交付", "actor_id": "mother", "action": "夹肉给儿子",
                "outcome_fact_ids": ["f2"], "basis_fact_ids": ["f2"],
                "participant_ids": ["mother", "son"], "evidence": evidence,
            }],
            "scenes": [], "state_changes": [],
        }

        result = _collapse_duplicate_atoms(payload)

        self.assertEqual(len(result["facts"]), 1)
        kept_id = result["facts"][0]["id"]
        self.assertEqual(result["events"][0]["outcome_fact_ids"], [kept_id])

    def test_two_stage_prompts_separate_fact_inventory_from_event_binding(self) -> None:
        text = "人物甲本名王甲，住在山村。人物甲随后决定离开。"
        document = build_chapter_document(
            "Author/work/0050", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        units = (AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text),)
        inventory = {"entities": [{"id": "e1"}], "facts": [{"id": "f1"}, {"id": "f2"}]}

        fact_prompt = atomic_fact_inventory_prompt(document, units, 1, 1)
        event_prompt = atomic_event_overlay_prompt(document, units, 1, 1, inventory)

        self.assertIn("本轮只做实体与事实库存，不抽事件", fact_prompt)
        self.assertIn("逐个 source_unit 清点", fact_prompt)
        self.assertIn("value 绝对不得为空", fact_prompt)
        self.assertIn("错误_把整条命题塞进谓词", fact_prompt)
        self.assertIn("不能单独证明此人就是父亲", fact_prompt)
        self.assertIn("事实库存已经锁定", event_prompt)
        self.assertIn("additional_facts", event_prompt)
        self.assertIn("父子关系事实ID", event_prompt)
        self.assertIn("不是询问造成的结果", event_prompt)
        self.assertIn(json.dumps(inventory, ensure_ascii=False), event_prompt)

    def test_fact_inventory_audit_uses_reconstruction_not_row_count(self) -> None:
        text = "人物甲出身寒微，却一心想去县城求学。"
        document = build_chapter_document(
            "Author/work/0051", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        units = (AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text),)
        inventory = {"entities": [{"id": "e1"}], "facts": [{"id": "f1"}]}

        prompt = atomic_fact_inventory_reconciliation_prompt(document, units, inventory)

        self.assertIn("遮住原文", prompt)
        self.assertIn("事实数量不是", atomic_fact_inventory_patch_prompt(
            document, units, inventory, [{"问题ID": "q1"}],
        ))
        self.assertIn(json.dumps(inventory, ensure_ascii=False), prompt)

    def test_fact_inventory_audit_rejects_event_level_category(self) -> None:
        units = (AnnotationSourceSlice("u1", "u1", 0, 6, "人物甲离开。"),)
        payload = {
            "结论": "需修复", "已核原文单元ID": ["u1"],
            "问题": [{
                "问题ID": "q1", "类别": "事件原子性", "对象ID": ["missing:event:u1"],
                "原文单元ID": ["u1"], "诊断": "缺少事件", "修复要求": "新增事件。",
            }],
        }

        with self.assertRaisesRegex(ValueError, "不得讨论事件"):
            _parse_fact_inventory_reconciliation(payload, units)

    def test_fact_inventory_patch_cannot_modify_events(self) -> None:
        inventory = {
            "entities": [{"id": "e1", "name": "人物甲"}],
            "facts": [{"id": "f1", "predicate": "旧事实"}],
        }
        patch = {
            "upsert_entities": [],
            "upsert_facts": [{"id": "f1", "predicate": "修正事实"}],
            "remove_entity_ids": [], "remove_fact_ids": [],
        }

        result = apply_fact_inventory_patch(inventory, patch, 1)

        self.assertEqual(result["facts"], [{"id": "f1", "predicate": "修正事实"}])
        self.assertNotIn("events", result)

    def test_event_overlay_preserves_locked_inventory(self) -> None:
        units = (AnnotationSourceSlice("u1", "u1", 0, 8, "人物甲决定离开。"),)
        inventory = {
            "entities": [{"id": "e1"}],
            "facts": [{"id": "f1"}, {"id": "f2"}],
        }
        overlay = {
            "additional_entities": [],
            "additional_facts": [{"id": "f3"}],
            "events": [{"id": "v1"}],
        }

        merged = _merge_event_overlay(inventory, overlay, units)

        self.assertEqual([item["id"] for item in merged["entities"]], ["e1"])
        self.assertEqual([item["id"] for item in merged["facts"]], ["f1", "f2", "f3"])
        self.assertEqual([item["id"] for item in merged["events"]], ["v1"])

    def test_orphan_progression_fact_does_not_materialise_an_event(self) -> None:
        evidence = [{"unit_id": "u1", "quote": "人物甲带人物乙离开"}]
        result = _collapse_duplicate_atoms({
            "entities": [
                {"id": "e1", "kind": "人物"},
                {"id": "e2", "kind": "人物"},
            ],
            "facts": [{
                "id": "f1", "kind": "进展", "subject_id": "e1", "predicate": "带走",
                "value": "离开", "object_id": "e2", "evidence": evidence,
            }],
            "events": [], "scenes": [], "state_changes": [],
        })

        self.assertEqual(result["events"], [])
        self.assertEqual([item["id"] for item in result["facts"]], ["f1"])

    def _atoms(self) -> dict[str, object]:
        return {
            "entities": [{"id": "e1"}],
            "facts": [{"id": f"f{index}"} for index in range(1, 7)],
            "events": [
                {"id": "v1", "order": 1, "participant_ids": ["e1"], "trigger_fact_ids": ["f1"], "precondition_fact_ids": [], "outcome_fact_ids": ["f2"], "action": "行动甲", "obstacle": "", "decision": "", "evidence": [{"unit_id": "u1", "quote": "甲"}]},
                {"id": "v2", "order": 2, "participant_ids": ["e1"], "trigger_fact_ids": [], "precondition_fact_ids": [], "outcome_fact_ids": ["f3"], "action": "行动乙", "obstacle": "阻力", "decision": "取舍", "evidence": [{"unit_id": "u2", "quote": "乙"}]},
                {"id": "v3", "order": 3, "participant_ids": ["e1"], "trigger_fact_ids": [], "precondition_fact_ids": [], "outcome_fact_ids": ["f4"], "action": "行动丙", "obstacle": "", "decision": "", "evidence": [{"unit_id": "u3", "quote": "丙"}]},
            ],
        }

    def test_scene_state_is_derived_from_ordered_atomic_events(self) -> None:
        result = _derive_scene_state_from_atoms(self._atoms())
        self.assertEqual([item["event_ids"] for item in result["scenes"]], [["v1", "v2", "v3"]])
        self.assertEqual(result["scenes"][0]["entry_fact_ids"], ["f1"])
        self.assertEqual(result["scenes"][0]["tension"], 3)
        self.assertEqual([item["after_fact_id"] for item in result["state_changes"]], ["f2", "f3", "f4"])
        self.assertEqual(_require_scene_state_draft(result), result)

    def test_atom_gate_rejects_empty_or_under_dense_payload(self) -> None:
        with self.assertRaises(ValueError):
            _require_atom_draft({}, ())
        with self.assertRaises(ValueError):
            _require_atom_draft({"entities": [], "facts": [], "events": []}, ())

    def test_atom_gate_fills_redundant_basis_from_valid_outcome(self) -> None:
        units = (AnnotationSourceSlice("u1", "u1", 0, 10, "甲完成行动。"),)
        payload = {
            "entities": [{"id": "e1"}],
            "facts": [{"id": "f1"}, {"id": "f2"}],
            "events": [{
                "id": "v1", "action_type": "其他", "actor_id": "e1",
                "target_ids": [], "basis_fact_ids": [], "outcome_fact_ids": ["f1"],
            }],
        }
        result = _require_atom_draft(payload, units)
        self.assertEqual(result["events"][0]["basis_fact_ids"], ["f1"])

    def test_atom_gate_keeps_causal_basis_and_adds_its_own_outcome(self) -> None:
        units = (AnnotationSourceSlice("u1", "u1", 0, 10, "甲观察后说出评价。"),)
        payload = {
            "entities": [{"id": "e1"}],
            "facts": [{"id": "f1"}, {"id": "f2"}],
            "events": [{
                "id": "v1", "action_type": "交流", "actor_id": "e1",
                "target_ids": [], "basis_fact_ids": ["f1"],
                "outcome_fact_ids": ["f2"],
            }],
        }

        result = _require_atom_draft(payload, units)

        self.assertEqual(result["events"][0]["basis_fact_ids"], ["f1", "f2"])

    def test_quality_gate_rejects_explicit_speech_typed_as_other(self) -> None:
        text = "甲呼喊乙开门。"
        document = build_chapter_document(
            "Author/work/0009", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": unit.start, "end": unit.end,
                "quote": unit.text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "e1", "canonical_name": "甲", "kind": "person", "aliases": [], "evidence": [span]},
                {"entity_id": "e2", "canonical_name": "乙", "kind": "person", "aliases": [], "evidence": [span]},
            ],
            "facts": [{"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "progression",
                       "subject_id": "e1", "predicate": "呼喊", "value": "乙开门", "object_id": "e2",
                       "certainty": 1.0, "evidence": [span]}],
            "events": [{"event_id": "v1", "chapter_id": document.chapter_id, "order": 0,
                        "summary": "甲呼喊乙开门", "participant_ids": ["e1", "e2"],
                        "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "呼喊乙开门",
                        "obstacle": "", "decision": "", "outcome_fact_ids": ["f1"], "cost_fact_ids": [],
                        "evidence": [span], "action_type": "other", "actor_id": "e1",
                        "target_ids": ["e2"], "basis_fact_ids": ["f1"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        issues = _event_frame_issues(annotation.events[0], {"f1": annotation.facts[0]})

        self.assertTrue(any(code == "event_explicit_speech_type_mismatch" for code, _ in issues))

    def test_quality_gate_keeps_observed_announcement_as_perception(self) -> None:
        text = "甲听到门外传来失败宣告。"
        document = build_chapter_document(
            "Author/work/0011", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": unit.start, "end": unit.end,
                "quote": unit.text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "e1", "canonical_name": "甲", "kind": "person",
                          "aliases": [], "evidence": [span]}],
            "facts": [{"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "knowledge",
                       "subject_id": "e1", "predicate": "得知", "value": "考核失败", "object_id": "",
                       "certainty": 1.0, "evidence": [span]}],
            "events": [{"event_id": "v1", "chapter_id": document.chapter_id, "order": 0,
                        "summary": "甲听到失败宣告", "participant_ids": ["e1"],
                        "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "听到失败宣告",
                        "obstacle": "", "decision": "", "outcome_fact_ids": ["f1"], "cost_fact_ids": [],
                        "evidence": [span], "action_type": "perception", "actor_id": "e1",
                        "target_ids": [], "basis_fact_ids": ["f1"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        issues = _event_frame_issues(annotation.events[0], {"f1": annotation.facts[0]})

        self.assertFalse(any(code == "event_explicit_speech_type_mismatch" for code, _ in issues))

    def test_static_location_snapshot_is_not_an_event(self) -> None:
        text = "四叔站在门外。"
        document = build_chapter_document(
            "Author/work/0048", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": unit.start, "end": unit.end,
                "quote": unit.text, "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [
                {"entity_id": "uncle", "canonical_name": "四叔", "kind": "person",
                 "aliases": [], "evidence": [span]},
                {"entity_id": "outside", "canonical_name": "门外", "kind": "location",
                 "aliases": [], "evidence": [span]},
            ],
            "facts": [{"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "location",
                       "subject_id": "uncle", "predicate": "位于", "value": "门外",
                       "object_id": "outside", "certainty": 1.0, "evidence": [span]}],
            "events": [{"event_id": "v1", "chapter_id": document.chapter_id, "order": 0,
                        "summary": "四叔位于门外", "participant_ids": ["uncle"],
                        "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "位于门外",
                        "obstacle": "", "decision": "", "outcome_fact_ids": ["f1"],
                        "cost_fact_ids": [], "evidence": [span], "action_type": "other",
                        "actor_id": "uncle", "target_ids": [], "basis_fact_ids": ["f1"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        issues = _event_frame_issues(annotation.events[0], {"f1": annotation.facts[0]})

        self.assertIn("event_stative_snapshot_not_change", {code for code, _ in issues})

    def test_speech_event_requires_its_own_communication_evidence(self) -> None:
        text = "甲跪地磕头，随后才开口奉承。"
        document = build_chapter_document(
            "Author/work/0012", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        unit = document.units[0]
        span = {"chapter_id": document.chapter_id, "start": unit.start, "end": unit.start + 5,
                "quote": "甲跪地磕头", "unit_id": unit.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "e1", "canonical_name": "甲", "kind": "person",
                          "aliases": [], "evidence": [span]}],
            "facts": [{"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "progression",
                       "subject_id": "e1", "predicate": "磕头", "value": "跪地磕头", "object_id": "",
                       "certainty": 1.0, "evidence": [span]}],
            "events": [{"event_id": "v1", "chapter_id": document.chapter_id, "order": 0,
                        "summary": "甲拍马屁", "participant_ids": ["e1"], "trigger_fact_ids": [],
                        "precondition_fact_ids": [], "action": "跪地磕头并拍马屁", "obstacle": "",
                        "decision": "", "outcome_fact_ids": ["f1"], "cost_fact_ids": [],
                        "evidence": [span], "action_type": "speech", "actor_id": "e1",
                        "target_ids": [], "basis_fact_ids": ["f1"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        issues = _event_frame_issues(annotation.events[0], {"f1": annotation.facts[0]})

        self.assertTrue(any(code == "event_speech_evidence_missing" for code, _ in issues))

    def test_outcome_fact_must_share_event_source_unit(self) -> None:
        text = "甲决定离开。\n甲随后登车。"
        document = build_chapter_document(
            "Author/work/0010", hashlib.sha256(text.encode()).hexdigest(), text,
        )
        first, second = document.units
        span1 = {"chapter_id": document.chapter_id, "start": first.start, "end": first.end,
                 "quote": first.text, "unit_id": first.unit_id}
        span2 = {"chapter_id": document.chapter_id, "start": second.start, "end": second.end,
                 "quote": second.text, "unit_id": second.unit_id}
        annotation = ChapterAnnotation.from_dict({
            "chapter_id": document.chapter_id, "source_hash": document.source_hash,
            "entities": [{"entity_id": "e1", "canonical_name": "甲", "kind": "person", "aliases": [], "evidence": [span1]}],
            "facts": [{"fact_id": "f1", "chapter_id": document.chapter_id, "kind": "progression",
                       "subject_id": "e1", "predicate": "决定", "value": "离开", "object_id": "",
                       "certainty": 1.0, "evidence": [span1]}],
            "events": [{"event_id": "v1", "chapter_id": document.chapter_id, "order": 0,
                        "summary": "甲登车", "participant_ids": ["e1"], "trigger_fact_ids": [],
                        "precondition_fact_ids": [], "action": "登车", "obstacle": "", "decision": "",
                        "outcome_fact_ids": ["f1"], "cost_fact_ids": [], "evidence": [span2],
                        "action_type": "other", "actor_id": "e1", "target_ids": [], "basis_fact_ids": ["f1"]}],
            "scenes": [], "state_changes": [], "schema_version": "2.2",
        })

        issues = _event_frame_issues(annotation.events[0], {"f1": annotation.facts[0]})

        self.assertTrue(any(code == "event_outcome_evidence_disjoint" for code, _ in issues))

    def test_atom_gate_rejects_missing_non_movement_outcome(self) -> None:
        units = (AnnotationSourceSlice("u1", "u1", 0, 12, "中年人宣布第二项测试。"),)
        evidence = [{"unit_id": "u1", "quote": "宣布第二项测试"}]
        payload = {
            "entities": [{"id": "e1"}],
            "facts": [{"id": "f1"}, {"id": "f2"}],
            "events": [{
                "id": "v1", "action_type": "交流", "actor_id": "e1",
                "target_ids": [], "basis_fact_ids": [], "outcome_fact_ids": None,
                "action": "宣布第二项测试", "summary": "中年人宣布第二项测试",
                "evidence": evidence,
            }],
        }

        with self.assertRaisesRegex(ValueError, "outcome_fact_ids"):
            _require_atom_draft(payload, units)

    def test_atom_gate_does_not_invent_a_movement_destination(self) -> None:
        units = (AnnotationSourceSlice("u1", "u1", 0, 8, "人物被带走。"),)
        payload = {
            "entities": [{"id": "e1"}],
            "facts": [{"id": "f1"}, {"id": "f2"}],
            "events": [{
                "id": "v1", "action_type": "移动", "actor_id": "e1",
                "target_ids": [], "basis_fact_ids": [], "outcome_fact_ids": None,
                "action": "被带走", "summary": "人物被带走",
                "evidence": [{"unit_id": "u1", "quote": "人物被带走"}],
            }],
        }

        with self.assertRaises(ValueError):
            _require_atom_draft(payload, units)


if __name__ == "__main__":
    unittest.main()
