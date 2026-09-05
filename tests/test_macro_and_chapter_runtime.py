from __future__ import annotations

import json
import unittest

from pipeline.chapter_runtime import (
    commit_paragraph_patch,
    complete_chapter_working_ledger,
    initialize_chapter_working_ledger,
    paragraph_obligations,
    working_ledger_context,
)
from pipeline.contracts import (
    ChapterProgram,
    EntityBinding,
    EventProgram,
    FactContract,
    NarrativeGraph,
    NarrativeGraphEdge,
    NarrativeGraphNode,
    ParagraphDraft,
    ParagraphProgram,
    ParagraphValidation,
    SceneProgram,
    SourceSpan,
)
from pipeline.macro_planner import (
    build_chapter_contracts,
    build_event_graph,
    build_work_story_plan,
    chapter_contract_writer_context,
    validate_program_chapter_contract,
)


def _graph() -> NarrativeGraph:
    first = SourceSpan("chapter-001", 0, 1, "甲", "chapter-001:p0000")
    second = SourceSpan("chapter-002", 0, 1, "乙", "chapter-002:p0000")
    entity = NarrativeGraphNode(
        "entity:global:person:0001", "entity", "人物甲",
        ("entity:chapter-001:person-001", "entity:chapter-002:person-001"), (first, second),
        attributes={"entity_kind": "person"},
    )
    first_event = NarrativeGraphNode(
        "event:chapter-001:event-001", "event", "取得线索", ("event:chapter-001:event-001",), (first,),
        "chapter-001", 0, 0, {"action": "取得线索"},
    )
    second_event = NarrativeGraphNode(
        "event:chapter-002:event-002", "event", "根据线索行动", ("event:chapter-002:event-002",), (second,),
        "chapter-002", 1, 0, {"action": "根据线索行动"},
    )
    edges = (
        NarrativeGraphEdge("edge-001", "event_participant", entity.node_id, first_event.node_id, ("event:chapter-001:event-001",), (first,), "chapter-001", 0),
        NarrativeGraphEdge("edge-002", "event_participant", entity.node_id, second_event.node_id, ("event:chapter-002:event-002",), (second,), "chapter-002", 1),
        NarrativeGraphEdge("edge-003", "causes", first_event.node_id, second_event.node_id, ("event:chapter-002:event-002",), (second,), "chapter-002", 1),
    )
    graph = NarrativeGraph(
        "Example", "work", ("chapter-001", "chapter-002"),
        {"chapter-001": "hash-001", "chapter-002": "hash-002"},
        (entity, first_event, second_event), edges,
    )
    graph.validate()
    return graph


def _two_paragraph_program() -> ChapterProgram:
    first_fact = FactContract("fact-001", "scene-001", "goal", "person-001", "目标", "离开")
    second_fact = FactContract("fact-002", "scene-001", "location", "person-001", "位置", "山门")
    first_event = EventProgram("event-001", "scene-001", "决定离开", "收拾行囊", ("person-001",), (), ("fact-001",), ("fact-001",))
    second_event = EventProgram("event-002", "scene-001", "抵达山门", "走到山门", ("person-001",), ("fact-001",), ("fact-002",), ("fact-002",))
    first_paragraph = ParagraphProgram("scene-001:paragraph-00", "scene-001", 0, "decision", ("fact-001",), ("event-001",), "person-001")
    second_paragraph = ParagraphProgram("scene-001:paragraph-01", "scene-001", 1, "transition", ("fact-002",), ("event-002",), "person-001")
    scene = SceneProgram(
        "scene-001", "chapter-001", 0, "离开住处", ("person-001",), (),
        (first_event, second_event), (first_fact, second_fact), (first_paragraph, second_paragraph), ("fact-002",), (),
    )
    program = ChapterProgram("chapter-001", "hash-001", (scene,), entities=(EntityBinding("person-001", "人物甲", "person"),))
    program.validate()
    return program


def _chapter_one_program() -> ChapterProgram:
    fact = FactContract("fact-001", "scene-001", "goal", "person-001", "目标", "取得线索")
    event = EventProgram("event-001", "scene-001", "取得线索", "取得线索", ("person-001",), (), ("fact-001",), ("fact-001",))
    paragraph = ParagraphProgram("scene-001:paragraph-00", "scene-001", 0, "action_progression", ("fact-001",), ("event-001",), "person-001")
    scene = SceneProgram("scene-001", "chapter-001", 0, "取得线索", ("person-001",), (), (event,), (fact,), (paragraph,), ("fact-001",), ())
    program = ChapterProgram("chapter-001", "hash-001", (scene,), entities=(EntityBinding("person-001", "人物甲", "person"),))
    program.validate()
    return program


class MacroAndChapterRuntimeTests(unittest.TestCase):
    def test_macro_contract_locks_future_events_without_serializing_evidence(self) -> None:
        graph = _graph()
        event_graph = build_event_graph(graph)
        plan = build_work_story_plan(event_graph, stage_chapters=1)
        contracts = build_chapter_contracts(graph, event_graph, plan)

        self.assertEqual(contracts[0].forbidden_event_ids, ("event:chapter-002:event-002",))
        self.assertEqual(contracts[1].required_prior_event_ids, ("event:chapter-001:event-001",))
        self.assertTrue(plan.stages[0].summary)
        rendered = json.dumps(event_graph.to_dict(), ensure_ascii=False)
        self.assertNotIn("evidence", rendered)
        self.assertNotIn("甲", rendered)

    def test_contract_window_only_exposes_the_current_event(self) -> None:
        graph = _graph()
        event_graph = build_event_graph(graph)
        contract = build_chapter_contracts(graph, event_graph, build_work_story_plan(event_graph, stage_chapters=1))[0]
        program = _chapter_one_program()
        validate_program_chapter_contract(program, contract, event_graph)
        context = chapter_contract_writer_context(event_graph, contract, program, paragraph_event_ids=("event-001",))

        self.assertEqual([item["事件ID"] for item in context["当前可推进事件"]], ["event-001"])
        self.assertEqual(context["未来锁定事件数"], 1)
        self.assertNotIn("event:chapter-002:event-002", json.dumps(context, ensure_ascii=False))

    def test_paragraph_ledger_requires_order_and_only_hands_off_audited_state(self) -> None:
        program = _two_paragraph_program()
        scene = program.scene_programs[0]
        first, second = scene.paragraphs
        ledger = initialize_chapter_working_ledger(program)
        first_facts, first_events = paragraph_obligations(scene, first)
        first_validation = ParagraphValidation(first.paragraph_id, True, first_facts, (), first_events, ())
        first_draft = ParagraphDraft(first.paragraph_id, "人物甲收起行囊，决定离开此地。", first_facts, first_events)
        second_facts, second_events = paragraph_obligations(scene, second)
        second_validation = ParagraphValidation(second.paragraph_id, True, second_facts, (), second_events, ())
        second_draft = ParagraphDraft(second.paragraph_id, "他沿着山路前行，终于站到山门之前。", second_facts, second_events)

        with self.assertRaises(ValueError):
            commit_paragraph_patch(ledger, program, scene, second, second_draft, second_validation)
        ledger = commit_paragraph_patch(ledger, program, scene, first, first_draft, first_validation)
        context = working_ledger_context(ledger, scene, second)
        self.assertEqual(context["上一段交接"]["段落ID"], first.paragraph_id)
        self.assertIn("fact-001", context["已兑现事实ID"])
        self.assertEqual(len(context["本章暂存状态"]), 1)
        ledger = commit_paragraph_patch(ledger, program, scene, second, second_draft, second_validation)
        self.assertTrue(complete_chapter_working_ledger(ledger, program)["completed"])


if __name__ == "__main__":
    unittest.main()
