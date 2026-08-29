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
    paragraph_writer_packet,
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


def _relationship_program(chapter_id: str, source_hash: str, parent_id: str) -> ChapterProgram:
    fact = FactContract("fact-relation", "scene-001", "relationship", "child", "亲属", "父亲", parent_id)
    event = EventProgram(
        "event-relation", "scene-001", "确认血缘关系", "当众确认父子身份", ("child", parent_id), (),
        ("fact-relation",), ("fact-relation",), "旁人质疑", "承认关系",
    )
    paragraph = ParagraphProgram(
        "scene-001:paragraph-00", "scene-001", 0, "decision", ("fact-relation",),
        ("event-relation",), "child",
    )
    scene = SceneProgram(
        "scene-001", chapter_id, 0, "确认亲缘", ("child", parent_id), (),
        (event,), (fact,), (paragraph,), ("fact-relation",), (),
    )
    program = ChapterProgram(
        chapter_id, source_hash, (scene,), schema_version="3.0",
        entities=(EntityBinding("child", "人物甲", "person"), EntityBinding(parent_id, parent_id, "person")),
    )
    program.validate()
    return program


def _scene_result(scene_id: str, fact_id: str, event_id: str) -> dict[str, object]:
    return {
        "scene_id": scene_id,
        "passed": True,
        "prose_passed": True,
        "role_passed": True,
        "validations": [{"realized_fact_ids": [fact_id], "realized_event_ids": [event_id]}],
    }


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

    def test_writer_packet_has_only_one_paragraph_neighbourhood_and_generated_handoff(self) -> None:
        program = _program()
        runtime = initialize_runtime_graph(None, program=program)
        scene = program.scene_programs[0]
        packet = paragraph_writer_packet(
            None, runtime, program, scene, scene.paragraphs[0],
            prior_paragraph={
                "paragraph_id": "previous:paragraph-00",
                "text": "这是上一次生成的正文结尾。",
                "claimed_fact_ids": ["old-fact"],
                "claimed_event_ids": ["old-event"],
            },
        )

        self.assertEqual(packet["当前段落程序"]["段落ID"], "scene-001:paragraph-00")
        self.assertEqual(packet["当前段落程序"]["段落功能"], "做出决定")
        self.assertEqual(packet["生成模式"], "忠实复现")
        self.assertEqual(packet["前文交接"]["上一段落ID"], "previous:paragraph-00")
        self.assertIn("叙事事实子图", packet)
        self.assertEqual(packet["许可实体"][0]["图谱实体ID"], "person-001")
        rendered = json.dumps(packet, ensure_ascii=False)
        self.assertNotIn("源文本哈希", rendered)
        self.assertNotIn("evidence", rendered)
        self.assertNotIn("quote", rendered)

    def test_writer_packet_excludes_later_scene_participants(self) -> None:
        base = _program()
        source_scene = base.scene_programs[0]
        scene = SceneProgram(
            source_scene.scene_id,
            source_scene.chapter_id,
            source_scene.order,
            "当前人物离开，之后另一人物才会说话",
            ("person-001", "later-person"),
            source_scene.entry_state_fact_ids,
            source_scene.event_programs,
            source_scene.fact_contracts,
            source_scene.paragraphs,
            source_scene.exit_state_fact_ids,
            source_scene.forbidden_event_ids,
            source_scene.narrative_beats,
            source_scene.context_fact_contracts,
        )
        program = ChapterProgram(
            base.chapter_id,
            base.source_hash,
            (scene,),
            schema_version=base.schema_version,
            entities=(*base.entities, EntityBinding("later-person", "后续人物", "person")),
        )
        program.validate()
        runtime = initialize_runtime_graph(None, program=program)

        packet = paragraph_writer_packet(None, runtime, program, scene, scene.paragraphs[0])

        rendered = json.dumps(packet, ensure_ascii=False)
        self.assertNotIn("later-person", rendered)
        self.assertNotIn("后续人物", rendered)
        self.assertNotIn("之后另一人物", rendered)

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

    def test_unlicensed_relationship_adjacent_to_current_actor_is_not_exposed(self) -> None:
        program = _program()
        graph = _graph_with_a_future_foreshadow_resolution()
        span = SourceSpan("chapter-001", 2, 3, "乙", "chapter-001:p0001")
        future_person = NarrativeGraphNode(
            "entity:global:person:future", "entity", "不应暴露的人物",
            ("entity:chapter-001:future",), (span,), attributes={"entity_kind": "person"},
        )
        future_relationship = NarrativeGraphEdge(
            "edge-future-relationship", "relationship", "entity:global:person:0001", future_person.node_id,
            ("fact:chapter-001:unlicensed",), (span,), "chapter-001", 0,
            attributes={"predicate": "认识", "value": "未来关系"},
        )
        graph = NarrativeGraph(
            graph.author_id, graph.work_id, graph.chapter_ids, graph.source_hashes,
            (*graph.nodes, future_person), (*graph.edges, future_relationship),
        )
        graph.validate()
        runtime = initialize_runtime_graph(graph, program=program)
        scene = program.scene_programs[0]

        subgraph = narrative_subgraph_for_paragraph(graph, runtime, program, scene, scene.paragraphs[0])

        rendered = json.dumps(subgraph, ensure_ascii=False)
        self.assertNotIn("不应暴露的人物", rendered)
        self.assertNotIn("未来关系", rendered)

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

    def test_unconfirmed_cue_is_not_exposed_as_a_foreshadow(self) -> None:
        program = _program()
        graph = _graph_with_a_future_foreshadow_resolution()
        graph = NarrativeGraph(
            graph.author_id, graph.work_id, graph.chapter_ids, graph.source_hashes,
            graph.nodes, tuple(edge for edge in graph.edges if edge.kind != "foreshadow_resolved"),
        )
        graph.validate()
        runtime = initialize_runtime_graph(graph, program=program)
        scene = program.scene_programs[0]
        subgraph = narrative_subgraph_for_paragraph(graph, runtime, program, scene, scene.paragraphs[0])

        self.assertFalse(any(item["类型"] == "foreshadow" for item in subgraph["节点"]))

    def test_parent_identity_cannot_silently_change_between_committed_chapters(self) -> None:
        first = _relationship_program("chapter-001", "hash-001", "father-a")
        runtime = initialize_runtime_graph(None, program=first)
        first_patch = build_chapter_graph_patch(
            None, runtime, first, [_scene_result("scene-001", "fact-relation", "event-relation")],
            prose_passed=True, style_passed=True, length_passed=True,
        )
        first_validation = validate_graph_patch(None, runtime, first_patch, program=first)
        self.assertTrue(first_validation["passed"], first_validation)
        runtime = commit_graph_patch(runtime, first_patch, first_validation)

        second = _relationship_program("chapter-002", "hash-002", "father-b")
        second_patch = build_chapter_graph_patch(
            None, runtime, second, [_scene_result("scene-001", "fact-relation", "event-relation")],
            prose_passed=True, style_passed=True, length_passed=True,
        )
        second_validation = validate_graph_patch(None, runtime, second_patch, program=second)

        self.assertFalse(second_validation["passed"])
        self.assertTrue(any(item.startswith("immutable_relationship_conflict") for item in second_validation["issues"]))

    def test_event_precondition_cannot_be_scheduled_after_the_event(self) -> None:
        fact = FactContract("fact-001", "scene-001", "goal", "person-001", "确认", "离开")
        event = EventProgram(
            "event-001", "scene-001", "人物确认去向", "收起行囊准备离开", ("person-001",), ("fact-001",),
            ("fact-001",), ("fact-001",), "同伴催促", "离开",
        )
        first = ParagraphProgram("scene-001:paragraph-00", "scene-001", 0, "action_progression", (), ("event-001",), "person-001")
        second = ParagraphProgram("scene-001:paragraph-01", "scene-001", 1, "decision", ("fact-001",), (), "person-001")
        scene = SceneProgram("scene-001", "chapter-001", 0, "确认离开", ("person-001",), (), (event,), (fact,), (first, second), ("fact-001",), ())
        program = ChapterProgram("chapter-001", "source-hash", (scene,), schema_version="3.0", entities=(EntityBinding("person-001", "人物甲", "person"),))
        program.validate()
        runtime = initialize_runtime_graph(None, program=program)
        patch = build_chapter_graph_patch(
            None, runtime, program, [_scene_result("scene-001", "fact-001", "event-001")],
            prose_passed=True, style_passed=True, length_passed=True,
        )
        validation = validate_graph_patch(None, runtime, patch, program=program)

        self.assertFalse(validation["passed"])
        self.assertIn("generation_validation_not_passed", validation["issues"])
        self.assertIn("precondition_after_event:event-001:fact-001", patch["checks"]["issues"])


if __name__ == "__main__":
    unittest.main()
