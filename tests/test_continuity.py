from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.annotate import annotation_from_draft
from pipeline.continuity import assess_continuity, build_work_continuity
from pipeline.contracts import NarrativeGraph, SourceSpan, StateLedgerEntry, TimelineEvent, WorkContinuity, build_chapter_document
from pipeline.jsonio import write_json, write_jsonl
from pipeline.narrative_graph import (
    assess_narrative_graph,
    build_narrative_graph,
    build_narrative_graph_work,
    _cross_chapter_state_issues,
    load_narrative_graph,
)
from pipeline.spatiotemporal import enrich_spatiotemporal


def _annotation(chapter_id: str, person: str, predicate: str, location: str):
    text = f"{person}{predicate}{location}。"
    document = build_chapter_document(chapter_id, hashlib.sha256(text.encode("utf-8")).hexdigest(), text, "Start")
    unit = document.units[0]
    quote = text[:-1]
    draft = {
        "entities": [
            {"id": "e1", "name": person, "kind": "person", "aliases": [], "evidence": [{"unit_id": unit.unit_id, "quote": person}]},
            {"id": "e2", "name": location, "kind": "location", "aliases": [], "evidence": [{"unit_id": unit.unit_id, "quote": location}]},
        ],
        "facts": [
            {"id": "f1", "kind": "location", "subject_id": "e1", "predicate": predicate, "value": location, "object_id": "e2", "certainty": 1, "evidence": [{"unit_id": unit.unit_id, "quote": quote}]},
        ],
        "events": [
            {"id": "v1", "order": 1, "summary": quote, "participant_ids": ["e1"], "trigger_fact_ids": [], "precondition_fact_ids": [], "action": quote, "obstacle": "", "decision": "", "outcome_fact_ids": ["f1"], "cost_fact_ids": [], "evidence": [{"unit_id": unit.unit_id, "quote": quote}]},
        ],
        "scenes": [
            {"id": "s1", "order": 1, "participant_ids": ["e1"], "event_ids": ["v1"], "objective": quote, "entry_fact_ids": [], "exit_fact_ids": ["f1"], "tension": 1, "evidence": [{"unit_id": unit.unit_id, "quote": quote}]},
        ],
        "state_changes": [{"id": "c1", "event_id": "v1", "operation": "add", "before_fact_id": "", "after_fact_id": "f1"}],
    }
    return document, enrich_spatiotemporal(document, annotation_from_draft(document, draft))


def _relationship_and_foreshadow_annotation():
    text = "王林与李慕婉结为同伴，留下未解疑团。王林循着疑团揭开真相。"
    chapter_id = "Author/work/0001"
    document = build_chapter_document(chapter_id, hashlib.sha256(text.encode("utf-8")).hexdigest(), text, "Start")
    unit = document.units[0]
    draft = {
        "entities": [
            {"id": "e1", "name": "王林", "kind": "person", "aliases": [], "evidence": [{"unit_id": unit.unit_id, "quote": "王林"}]},
            {"id": "e2", "name": "李慕婉", "kind": "person", "aliases": [], "evidence": [{"unit_id": unit.unit_id, "quote": "李慕婉"}]},
        ],
        "facts": [
            {"id": "f1", "kind": "relationship", "subject_id": "e1", "predicate": "同伴", "value": "同伴", "object_id": "e2", "certainty": 1, "evidence": [{"unit_id": unit.unit_id, "quote": "王林与李慕婉结为同伴"}]},
            {"id": "f2", "kind": "information", "subject_id": "e1", "predicate": "线索", "value": "未解疑团", "object_id": "", "certainty": 1, "evidence": [{"unit_id": unit.unit_id, "quote": "留下未解疑团"}]},
            {"id": "f3", "kind": "knowledge", "subject_id": "e1", "predicate": "得知", "value": "真相", "object_id": "", "certainty": 1, "evidence": [{"unit_id": unit.unit_id, "quote": "揭开真相"}]},
        ],
        "events": [
            {"id": "v1", "order": 1, "summary": "留下未解疑团", "participant_ids": ["e1", "e2"], "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "留下未解疑团", "obstacle": "", "decision": "", "outcome_fact_ids": ["f1", "f2"], "cost_fact_ids": [], "evidence": [{"unit_id": unit.unit_id, "quote": "留下未解疑团"}]},
            {"id": "v2", "order": 2, "summary": "揭开真相", "participant_ids": ["e1"], "trigger_fact_ids": [], "precondition_fact_ids": ["f2"], "action": "揭开真相", "obstacle": "", "decision": "", "outcome_fact_ids": ["f3"], "cost_fact_ids": [], "evidence": [{"unit_id": unit.unit_id, "quote": "揭开真相"}]},
        ],
        "scenes": [
            {"id": "s1", "order": 1, "participant_ids": ["e1", "e2"], "event_ids": ["v1"], "objective": "留下疑团", "entry_fact_ids": [], "exit_fact_ids": ["f1", "f2"], "tension": 1, "evidence": [{"unit_id": unit.unit_id, "quote": "留下未解疑团"}]},
            {"id": "s2", "order": 2, "participant_ids": ["e1"], "event_ids": ["v2"], "objective": "揭开真相", "entry_fact_ids": ["f2"], "exit_fact_ids": ["f3"], "tension": 1, "evidence": [{"unit_id": unit.unit_id, "quote": "揭开真相"}]},
        ],
        "state_changes": [],
    }
    return document, enrich_spatiotemporal(document, annotation_from_draft(document, draft))


class ContinuityTests(unittest.TestCase):
    def test_exact_aliases_merge_and_movements_form_a_cross_chapter_chain(self) -> None:
        _, first = _annotation("Author/work/0001", "王林", "到达", "村口")
        _, second = _annotation("Author/work/0002", "铁柱（王林）", "进入", "客栈")
        continuity = build_work_continuity("Author", "work", [first, second])
        report = assess_continuity(continuity, [first, second])
        self.assertTrue(report.passed, report.to_dict())
        protagonist = next(entity for entity in continuity.global_entities if entity.canonical_name == "王林")
        self.assertEqual(len(protagonist.member_entity_ids), 2)
        self.assertEqual(len(continuity.timeline_events), 2)
        self.assertEqual(len(continuity.location_transitions), 2)
        self.assertEqual(
            continuity.location_transitions[1].from_location_global_id,
            continuity.location_transitions[0].to_location_global_id,
        )

    def test_generic_role_labels_are_not_merged_across_chapters(self) -> None:
        _, first = _annotation("Author/work/0001", "师兄", "到达", "山门")
        _, second = _annotation("Author/work/0002", "师兄", "进入", "大殿")
        continuity = build_work_continuity("Author", "work", [first, second])
        role_entities = [entity for entity in continuity.global_entities if entity.canonical_name == "师兄"]
        self.assertEqual(len(role_entities), 2)
        self.assertTrue(any(issue.code == "generic_alias_not_merged" for issue in continuity.issues))

    def test_annotations_and_continuity_become_an_evidence_complete_fact_graph(self) -> None:
        _, first = _annotation("Author/work/0001", "王林", "到达", "村口")
        _, second = _annotation("Author/work/0002", "铁柱（王林）", "进入", "客栈")
        continuity = build_work_continuity("Author", "work", [first, second])
        graph = build_narrative_graph("Author", "work", [first, second], continuity)
        report = assess_narrative_graph(graph, [first, second], continuity)
        self.assertTrue(report.passed, report.to_dict())
        self.assertEqual(report.node_counts["entity"], len(continuity.global_entities))
        self.assertEqual(report.node_counts["event"], 2)
        self.assertEqual(report.node_counts["scene"], 2)
        self.assertEqual(report.edge_counts["location_transition"], 2)
        recovered = NarrativeGraph.from_dict(graph.to_dict())
        recovered.validate()
        self.assertEqual(recovered.chapter_ids, ("Author/work/0001", "Author/work/0002"))

    def test_graph_records_relationships_causal_links_and_foreshadow_lifecycles(self) -> None:
        _, annotation = _relationship_and_foreshadow_annotation()
        continuity = build_work_continuity("Author", "work", [annotation])
        graph = build_narrative_graph("Author", "work", [annotation], continuity)
        report = assess_narrative_graph(graph, [annotation], continuity)

        self.assertTrue(report.passed, report.to_dict())
        self.assertEqual(sum(edge.kind == "relationship" for edge in graph.edges), 1)
        self.assertEqual(sum(edge.kind == "causes" for edge in graph.edges), 1)
        foreshadows = [node for node in graph.nodes if node.kind == "foreshadow"]
        self.assertEqual(len(foreshadows), 1)
        self.assertEqual(foreshadows[0].attributes["lifecycle"], "resolved")
        self.assertTrue(any(edge.kind == "foreshadow_resolved" for edge in graph.edges))

    def test_cross_chapter_state_check_uses_timeline_order_not_event_id_lexical_order(self) -> None:
        span = SourceSpan("chapter-001", 0, 1, "甲", "chapter-001:p0000")
        continuity = WorkContinuity(
            "Author", "work", ("chapter-001",), "2.1", (),
            (
                TimelineEvent("timeline-1", "chapter-001", "event-z", 0, 0, ("global:person:0001",), evidence=(span,)),
                TimelineEvent("timeline-2", "chapter-001", "event-a", 0, 1, ("global:person:0001",), evidence=(span,)),
            ),
            # Deliberately reverse the stored entries.  The timeline still says
            # add occurs before update, so this must not be reported as an
            # update-without-state error.
            (
                StateLedgerEntry("state-2", "chapter-001", "event-a", 0, "global:person:0001", "fact-2", "update", "goal:意图", "前往", evidence=(span,)),
                StateLedgerEntry("state-1", "chapter-001", "event-z", 0, "global:person:0001", "fact-1", "add", "goal:意图", "离开", evidence=(span,)),
            ),
            (),
        )
        self.assertEqual(_cross_chapter_state_issues(continuity), [])

    def test_work_builder_publishes_the_fixed_local_graph_artifact(self) -> None:
        first_document, first = _annotation("Author/work/0001", "Hero", "arrives", "village")
        second_document, second = _annotation("Author/work/0002", "Hero (alias)", "enters", "inn")
        continuity = build_work_continuity("Author", "work", [first, second])
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            write_jsonl(folder / "chapter_annotations.sample-2.jsonl", [first.to_dict(), second.to_dict()])
            write_jsonl(folder / "chapter_documents.jsonl", [first_document.to_dict(), second_document.to_dict()])
            write_json(folder / "work_continuity.sample-2.json", continuity.to_dict())
            with patch("pipeline.narrative_graph.corpus_dir", return_value=folder):
                target, quality = build_narrative_graph_work("Author", "work", limit=2)
                loaded = load_narrative_graph("Author", "work")
            self.assertEqual(target.name, "narrative_graph.json")
            self.assertEqual(quality.name, "narrative_graph.quality.json")
            self.assertTrue(target.exists())
            self.assertTrue(quality.exists())
            self.assertEqual(loaded.work_id, "work")


if __name__ == "__main__":
    unittest.main()
