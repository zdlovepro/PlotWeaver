from __future__ import annotations

import json
import unittest

from pipeline.contracts import (
    ChapterProgram,
    EntityBinding,
    EventProgram,
    FactContract,
    NarrativeGraph,
    NarrativeGraphEdge,
    NarrativeGraphNode,
    ParagraphProgram,
    SceneProgram,
    SourceSpan,
)
from pipeline.graph_runtime import (
    build_chapter_graph_patch,
    commit_graph_patch,
    initialize_runtime_graph,
    narrative_subgraph_for_paragraph,
    validate_graph_patch,
)


def _program() -> ChapterProgram:
    fact = FactContract("fact-001", "scene-001", "goal", "person-001", "确认", "离开")
    event = EventProgram(
        "event-001", "scene-001", "人物确认去向", "收起行囊准备离开", ("person-001",), (),
        ("fact-001",), ("fact-001",), "同伴催促", "离开",
    )
    paragraph = ParagraphProgram(
        "scene-001:paragraph-00", "scene-001", 0, "decision", ("fact-001",),
        ("event-001",), "person-001",
    )
    scene = SceneProgram(
        "scene-001", "chapter-001", 0, "确认离开", ("person-001",), (),
        (event,), (fact,), (paragraph,), ("fact-001",), (),
    )
    program = ChapterProgram(
        "chapter-001", "source-hash", (scene,), schema_version="3.0",
        entities=(EntityBinding("person-001", "人物甲", "person"),),
    )
    program.validate()
    return program


def _graph_with_a_future_foreshadow_resolution() -> NarrativeGraph:
    first = SourceSpan("chapter-001", 0, 1, "甲", "chapter-001:p0000")
    second = SourceSpan("chapter-002", 0, 1, "乙", "chapter-002:p0000")
    entity = NarrativeGraphNode(
        "entity:global:person:0001", "entity", "人物甲",
        ("entity:chapter-001:person-001", "entity:chapter-002:person-001"), (first, second),
        attributes={"entity_kind": "person"},
    )
    fact = NarrativeGraphNode(
        "fact:chapter-001:fact-001", "fact", "确认离开", ("fact:chapter-001:fact-001",), (first,),
        "chapter-001", 0, attributes={"fact_kind": "goal", "predicate": "确认", "value": "离开", "subject_local_id": "person-001"},
    )
    opening_event = NarrativeGraphNode(
        "event:chapter-001:event-001", "event", "留下线索", ("event:chapter-001:event-001",), (first,),
        "chapter-001", 0, 0, {"action": "留下线索"},
    )
    foreshadow = NarrativeGraphNode(
        "foreshadow:chapter-001:fact-fact-001", "foreshadow", "未解叙事线索", ("fact:chapter-001:fact-001",), (first,),
        "chapter-001", 0, attributes={"lifecycle": "resolved", "resolved_by_event_id": "event-002"},
    )
    future_event = NarrativeGraphNode(
        "event:chapter-002:event-002", "event", "揭开真相", ("event:chapter-002:event-002",), (second,),
        "chapter-002", 1, 0, {"action": "揭开真相"},
    )
    edges = (
        NarrativeGraphEdge("edge-001", "asserts", entity.node_id, fact.node_id, ("fact:chapter-001:fact-001",), (first,), "chapter-001", 0),
        NarrativeGraphEdge("edge-002", "event_participant", entity.node_id, opening_event.node_id, ("event:chapter-001:event-001",), (first,), "chapter-001", 0, 0),
        NarrativeGraphEdge("edge-003", "foreshadow_open", fact.node_id, foreshadow.node_id, ("fact:chapter-001:fact-001",), (first,), "chapter-001", 0),
        NarrativeGraphEdge("edge-004", "foreshadow_resolved", future_event.node_id, foreshadow.node_id, ("event:chapter-002:event-002",), (second,), "chapter-002", 1, 0),
    )
    graph = NarrativeGraph(
        "Example", "work", ("chapter-001", "chapter-002"),
        {"chapter-001": "source-hash", "chapter-002": "source-hash-2"},
        (entity, fact, opening_event, foreshadow, future_event), edges,
    )
    graph.validate()
    return graph


class GraphRuntimeTests(unittest.TestCase):
    def test_paragraph_fallback_subgraph_is_structural_and_has_no_prose(self) -> None:
        program = _program()
        runtime = initialize_runtime_graph(None, program=program)
        scene = program.scene_programs[0]
        subgraph = narrative_subgraph_for_paragraph(None, runtime, program, scene, scene.paragraphs[0])

        self.assertEqual(subgraph["段落ID"], "scene-001:paragraph-00")
        self.assertEqual([item["事实ID"] for item in subgraph["允许事实"]], ["fact-001"])
        rendered = json.dumps(subgraph, ensure_ascii=False)
        self.assertNotIn("原文", rendered)
        self.assertNotIn("正文", rendered)
        self.assertNotIn("quote", rendered)
        self.assertNotIn("evidence", rendered)

    def test_patch_commits_only_after_facts_events_and_state_are_validated(self) -> None:
        program = _program()
        runtime = initialize_runtime_graph(None, program=program)
        scene_result = {
            "scene_id": "scene-001",
            "passed": True,
            "prose_passed": True,
            "role_passed": True,
            "validations": [{"realized_fact_ids": ["fact-001"], "realized_event_ids": ["event-001"]}],
        }
        patch = build_chapter_graph_patch(
            None, runtime, program, [scene_result], prose_passed=True, style_passed=True, length_passed=True,
        )
        validation = validate_graph_patch(None, runtime, patch)
        self.assertTrue(validation["passed"], validation)

        committed = commit_graph_patch(runtime, patch, validation)
        self.assertEqual(committed["committed_patches"][0]["status"], "committed")
        self.assertEqual(len(committed["effective_state"]), 1)

    def test_patch_with_an_unrealised_fact_cannot_commit(self) -> None:
        program = _program()
        runtime = initialize_runtime_graph(None, program=program)
        scene_result = {
            "scene_id": "scene-001",
            "passed": True,
            "prose_passed": True,
            "role_passed": True,
            "validations": [{"realized_fact_ids": [], "realized_event_ids": ["event-001"]}],
        }
        patch = build_chapter_graph_patch(
            None, runtime, program, [scene_result], prose_passed=True, style_passed=True, length_passed=True,
        )
        validation = validate_graph_patch(None, runtime, patch)
        self.assertFalse(validation["passed"])
        self.assertIn("generation_validation_not_passed", validation["issues"])
        with self.assertRaises(ValueError):
            commit_graph_patch(runtime, patch, validation)

    def test_future_foreshadow_resolution_is_hidden_until_its_patch_is_committed(self) -> None:
        program = _program()
        graph = _graph_with_a_future_foreshadow_resolution()
        runtime = initialize_runtime_graph(graph, program=program)
        scene = program.scene_programs[0]
        subgraph = narrative_subgraph_for_paragraph(graph, runtime, program, scene, scene.paragraphs[0])

        hint = next(item for item in subgraph["节点"] if item["类型"] == "foreshadow")
        self.assertEqual(hint["生命周期"], "open")
        rendered = json.dumps(subgraph, ensure_ascii=False)
        self.assertNotIn("event:chapter-002:event-002", rendered)

        scene_result = {
            "scene_id": "scene-001", "passed": True, "prose_passed": True, "role_passed": True,
            "validations": [{"realized_fact_ids": ["fact-001"], "realized_event_ids": ["event-001"]}],
        }
        patch = build_chapter_graph_patch(
            graph, runtime, program, [scene_result], prose_passed=True, style_passed=True, length_passed=True,
        )
        validation = validate_graph_patch(graph, runtime, patch)
        self.assertTrue(validation["passed"], validation)
        committed = commit_graph_patch(runtime, patch, validation)
        self.assertEqual(committed["effective_foreshadows"][hint["节点ID"]]["lifecycle"], "open")


if __name__ == "__main__":
    unittest.main()
