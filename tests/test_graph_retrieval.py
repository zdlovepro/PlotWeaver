from __future__ import annotations

import unittest

from pipeline.contracts import NarrativeGraph, NarrativeGraphEdge, NarrativeGraphNode, SourceSpan
from pipeline.graph_retrieval import retrieve_graph_neighbourhood


def _span(chapter_id: str, text: str) -> SourceSpan:
    return SourceSpan(chapter_id, 0, len(text), text, f"{chapter_id}:p0000")


def _graph() -> NarrativeGraph:
    first = _span("chapter-001", "前章")
    current = _span("chapter-002", "本章")
    future = _span("chapter-003", "后章")
    protagonist = NarrativeGraphNode(
        "entity:global:person:hero", "entity", "主角",
        ("entity:chapter-001:hero", "entity:chapter-002:hero", "entity:chapter-003:hero"),
        (first, current, future), attributes={"entity_kind": "person"},
    )
    outsider = NarrativeGraphNode(
        "entity:global:person:outsider", "entity", "无关人物",
        ("entity:chapter-001:outsider",), (first,), attributes={"entity_kind": "person"},
    )
    prior_fact = NarrativeGraphNode(
        "fact:chapter-001:location", "fact", "位于旧地",
        ("fact:chapter-001:location",), (first,), "chapter-001", 0,
        attributes={"fact_kind": "location", "predicate": "位于", "value": "旧地"},
    )
    current_fact = NarrativeGraphNode(
        "fact:chapter-002:location", "fact", "位于新地",
        ("fact:chapter-002:location",), (current,), "chapter-002", 1,
        attributes={"fact_kind": "location", "predicate": "位于", "value": "新地"},
    )
    prior_event = NarrativeGraphNode(
        "event:chapter-001:arrival", "event", "抵达旧地",
        ("event:chapter-001:arrival",), (first,), "chapter-001", 0, 0,
        {"action": "主角抵达旧地", "action_type": "movement"},
    )
    irrelevant_event = NarrativeGraphNode(
        "event:chapter-001:irrelevant", "event", "旁人处理别事",
        ("event:chapter-001:irrelevant",), (first,), "chapter-001", 0, 1,
        {"action": "旁人离开"},
    )
    committed_same_chapter_event = NarrativeGraphNode(
        "event:chapter-002:earlier", "event", "本章前段已经完成的行动",
        ("event:chapter-002:earlier",), (current,), "chapter-002", 1, 0,
        {"action": "主角完成前段行动"},
    )
    current_event = NarrativeGraphNode(
        "event:chapter-002:current", "event", "继续行动",
        ("event:chapter-002:current",), (current,), "chapter-002", 1, 1,
        {"action": "主角继续行动"},
    )
    future_event = NarrativeGraphNode(
        "event:chapter-003:future", "event", "未来真相",
        ("event:chapter-003:future",), (future,), "chapter-003", 2, 0,
        {"action": "主角发现未来真相"},
    )
    nodes = (
        protagonist, outsider, prior_fact, current_fact, prior_event, irrelevant_event,
        committed_same_chapter_event, current_event, future_event,
    )

    def edge(edge_id: str, kind: str, source: str, target: str, chapter_id: str, chapter_index: int, source_id: str) -> NarrativeGraphEdge:
        span = {"chapter-001": first, "chapter-002": current, "chapter-003": future}[chapter_id]
        return NarrativeGraphEdge(edge_id, kind, source, target, (source_id,), (span,), chapter_id, chapter_index)

    edges = (
        edge("e01", "asserts", protagonist.node_id, prior_fact.node_id, "chapter-001", 0, "fact:chapter-001:location"),
        edge("e02", "asserts", protagonist.node_id, current_fact.node_id, "chapter-002", 1, "fact:chapter-002:location"),
        edge("e03", "event_participant", protagonist.node_id, prior_event.node_id, "chapter-001", 0, "event:chapter-001:arrival"),
        edge("e04", "event_produces", prior_event.node_id, prior_fact.node_id, "chapter-001", 0, "event:chapter-001:arrival"),
        edge("e05", "event_participant", outsider.node_id, irrelevant_event.node_id, "chapter-001", 0, "event:chapter-001:irrelevant"),
        edge("e06", "event_participant", protagonist.node_id, committed_same_chapter_event.node_id, "chapter-002", 1, "event:chapter-002:earlier"),
        edge("e07", "event_participant", protagonist.node_id, current_event.node_id, "chapter-002", 1, "event:chapter-002:current"),
        edge("e08", "event_requires", current_event.node_id, current_fact.node_id, "chapter-002", 1, "event:chapter-002:current"),
        edge("e09", "event_participant", protagonist.node_id, future_event.node_id, "chapter-003", 2, "event:chapter-003:future"),
        edge("e10", "timeline_precedes", prior_event.node_id, current_event.node_id, "chapter-002", 1, "event:chapter-002:current"),
        edge("e11", "timeline_precedes", current_event.node_id, future_event.node_id, "chapter-003", 2, "event:chapter-003:future"),
    )
    graph = NarrativeGraph(
        "Example", "work", ("chapter-001", "chapter-002", "chapter-003"),
        {"chapter-001": "h1", "chapter-002": "h2", "chapter-003": "h3"}, nodes, edges,
    )
    graph.validate()
    return graph


class GraphRetrievalTests(unittest.TestCase):
    def test_retrieves_causal_parent_and_state_but_never_future(self) -> None:
        graph = _graph()
        result = retrieve_graph_neighbourhood(
            graph,
            chapter_id="chapter-002",
            seed_node_ids=(
                "entity:global:person:hero",
                "fact:chapter-002:location",
                "event:chapter-002:current",
            ),
            licensed_source_ids=("fact:chapter-002:location", "event:chapter-002:current"),
            required_prior_event_ids=("event:chapter-001:arrival",),
        )

        self.assertIn("event:chapter-001:arrival", result.node_ids)
        self.assertIn("fact:chapter-001:location", result.node_ids)
        self.assertIn("同一状态槽的近期记录", result.reasons["fact:chapter-001:location"])
        self.assertNotIn("event:chapter-001:irrelevant", result.node_ids)
        self.assertNotIn("event:chapter-003:future", result.node_ids)

    def test_same_chapter_history_requires_a_committed_source_id(self) -> None:
        graph = _graph()
        common = {
            "chapter_id": "chapter-002",
            "seed_node_ids": ("entity:global:person:hero", "event:chapter-002:current"),
            "licensed_source_ids": ("event:chapter-002:current",),
        }
        hidden = retrieve_graph_neighbourhood(graph, **common)
        visible = retrieve_graph_neighbourhood(
            graph,
            **common,
            committed_source_ids=("event:chapter-002:earlier",),
        )

        self.assertNotIn("event:chapter-002:earlier", hidden.node_ids)
        self.assertIn("event:chapter-002:earlier", visible.node_ids)

    def test_node_budget_keeps_all_direct_seeds(self) -> None:
        graph = _graph()
        seeds = (
            "entity:global:person:hero",
            "fact:chapter-002:location",
            "event:chapter-002:current",
        )
        result = retrieve_graph_neighbourhood(
            graph,
            chapter_id="chapter-002",
            seed_node_ids=seeds,
            licensed_source_ids=("fact:chapter-002:location", "event:chapter-002:current"),
            node_budget=1,
        )
        self.assertTrue(set(seeds).issubset(result.node_ids))


if __name__ == "__main__":
    unittest.main()
