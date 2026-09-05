"""Time-safe structural retrieval over the narrative knowledge graph.

The retriever is deliberately domain agnostic.  It does not look for named
characters, cultivation vocabulary, kinship words, or particular actions.
Instead it ranks graph nodes by their structural role around the paragraph's
licensed facts, events and entities.  Source evidence is used to build the
graph, but is never returned to prose generation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Any, Iterable

from .contracts import NarrativeGraph


RETRIEVAL_EDGE_KINDS = frozenset({
    "asserts", "fact_object", "event_participant", "event_trigger", "event_requires",
    "event_produces", "event_cost", "relationship", "causes", "spatial_relation",
    "location_transition", "state_change", "state_observed", "foreshadow_open",
    "foreshadow_resolved", "temporal_before", "temporal_simultaneous", "timeline_precedes",
})
STATE_FACT_KINDS = frozenset({
    "identity", "goal", "relationship", "location", "resource", "knowledge",
    "rule", "progression", "information",
})
DEFAULT_GRAPH_RAG_MAX_HOPS = 2
DEFAULT_GRAPH_RAG_NODE_BUDGET = 32
DEFAULT_GRAPH_RAG_EDGE_BUDGET = 48
_EVENT_FACT_EDGES = frozenset({"event_trigger", "event_requires", "event_produces", "event_cost", "state_change", "state_observed"})
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class GraphRetrievalResult:
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    reasons: dict[str, tuple[str, ...]]
    scores: dict[str, float]
    seed_count: int
    historical_count: int
    committed_same_chapter_count: int
    max_hops: int
    node_budget: int
    edge_budget: int


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item) for item in values if str(item).strip()))


def _node_text(node: Any) -> str:
    attributes = node.attributes if isinstance(node.attributes, dict) else {}
    return " ".join(str(item) for item in (
        node.label,
        attributes.get("action", ""),
        attributes.get("predicate", ""),
        attributes.get("value", ""),
    ) if str(item).strip())


def _semantic_units(text: str, *, excluded_terms: Iterable[str] = ()) -> set[str]:
    """Return generic word and CJK-bigram units without a domain lexicon."""

    raw = "".join(str(text).lower().split())
    for term in sorted((str(item).lower().strip() for item in excluded_terms if str(item).strip()), key=len, reverse=True):
        raw = raw.replace("".join(term.split()), "")
    tokens = {item for item in _TOKEN_PATTERN.findall(raw) if item.strip()}
    chinese = "".join(item for item in raw if "\u3400" <= item <= "\u9fff")
    tokens.update(chinese[index:index + 2] for index in range(max(0, len(chinese) - 1)))
    tokens.update(chinese[index:index + 3] for index in range(max(0, len(chinese) - 2)))
    return tokens


def _similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _fact_subjects(graph: NarrativeGraph) -> dict[str, tuple[str, ...]]:
    values: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        if edge.kind == "asserts":
            values[edge.target_node_id].append(edge.source_node_id)
    return {key: _unique(items) for key, items in values.items()}


def _event_participants(graph: NarrativeGraph) -> dict[str, tuple[str, ...]]:
    values: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        if edge.kind == "event_participant":
            values[edge.target_node_id].append(edge.source_node_id)
    return {key: _unique(items) for key, items in values.items()}


def _chapter_index(graph: NarrativeGraph, chapter_id: str) -> int:
    try:
        return graph.chapter_ids.index(chapter_id)
    except ValueError as exc:
        raise ValueError(f"chapter {chapter_id} does not belong to the narrative graph") from exc


def retrieve_graph_neighbourhood(
    graph: NarrativeGraph,
    *,
    chapter_id: str,
    seed_node_ids: Iterable[str],
    licensed_source_ids: Iterable[str],
    committed_source_ids: Iterable[str] = (),
    required_prior_event_ids: Iterable[str] = (),
    max_hops: int = DEFAULT_GRAPH_RAG_MAX_HOPS,
    node_budget: int = DEFAULT_GRAPH_RAG_NODE_BUDGET,
    edge_budget: int = DEFAULT_GRAPH_RAG_EDGE_BUDGET,
) -> GraphRetrievalResult:
    """Retrieve a compact, causally useful graph window for one paragraph.

    Visibility is temporal, not lexical: direct paragraph seeds are visible;
    earlier chapters are visible; same-chapter material is visible only after
    it has passed the paragraph gate; later material is always hidden.
    Candidate ranking combines graph role, shared state slot, participant
    overlap, recency and generic textual similarity.
    """

    graph.validate()
    current_index = _chapter_index(graph, chapter_id)
    nodes = {node.node_id: node for node in graph.nodes}
    seeds = {item for item in _unique(seed_node_ids) if item in nodes}
    direct_licensed = set(_unique(licensed_source_ids))
    committed = set(_unique(committed_source_ids))
    licensed = direct_licensed | committed
    required_prior = {
        item for item in _unique(required_prior_event_ids)
        if item in nodes and nodes[item].kind == "event" and nodes[item].chapter_index < current_index
    }

    def node_visible(node: Any) -> bool:
        if node.kind == "entity" or node.node_id in seeds or node.node_id in required_prior:
            return True
        if node.chapter_index < current_index:
            return True
        if node.chapter_index > current_index:
            return False
        return bool(set(node.source_ids) & licensed)

    visible = {node_id for node_id, node in nodes.items() if node_visible(node)}
    scores: dict[str, float] = {}
    reasons: dict[str, list[str]] = defaultdict(list)

    def offer(node_id: str, score: float, reason: str) -> None:
        if node_id not in visible:
            return
        if score > scores.get(node_id, float("-inf")):
            scores[node_id] = score
        if reason not in reasons[node_id]:
            reasons[node_id].append(reason)

    for node_id in seeds:
        offer(node_id, 1000.0, "当前段落直接义务")
    for node_id in required_prior:
        offer(node_id, 920.0, "跨章因果前置")

    fact_subjects = _fact_subjects(graph)
    event_participants = _event_participants(graph)
    seed_entities = {node_id for node_id in seeds if nodes[node_id].kind == "entity"}
    for node_id in seeds:
        seed_entities.update(fact_subjects.get(node_id, ()))
        seed_entities.update(event_participants.get(node_id, ()))
    seed_entity_names = tuple(nodes[node_id].label for node_id in seed_entities if node_id in nodes)
    query_units: set[str] = set()
    for node_id in seeds:
        if nodes[node_id].kind in {"fact", "event"}:
            query_units.update(_semantic_units(_node_text(nodes[node_id]), excluded_terms=seed_entity_names))

    # Incoming causal parents are more valuable than generic historical
    # neighbours, and may themselves have a parent.  The depth is bounded so
    # a long event chain cannot consume the entire prompt.
    frontier = {node_id for node_id in seeds | required_prior if nodes[node_id].kind == "event"}
    causal_edges = [edge for edge in graph.edges if edge.kind == "causes"]
    for hop in range(1, max(1, int(max_hops)) + 1):
        parents = {
            edge.source_node_id
            for edge in causal_edges
            if edge.target_node_id in frontier and edge.source_node_id in visible
        }
        for node_id in parents:
            offer(node_id, 900.0 - 35.0 * hop, f"因果父链第{hop}跳")
        frontier = parents
        if not frontier:
            break

    # Retrieve the latest compatible state records for facts used now.  State
    # identity comes from graph type + subject + predicate, never from a list
    # of story-specific words.
    seed_fact_slots: set[tuple[str, str, str]] = set()
    for node_id in seeds:
        node = nodes[node_id]
        if node.kind != "fact":
            continue
        attributes = node.attributes if isinstance(node.attributes, dict) else {}
        kind = str(attributes.get("fact_kind", ""))
        predicate = str(attributes.get("predicate", ""))
        if kind not in STATE_FACT_KINDS:
            continue
        for subject_id in fact_subjects.get(node_id, ()):
            seed_fact_slots.add((subject_id, kind, predicate))

    state_candidates: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for node_id in visible:
        node = nodes[node_id]
        if node.kind != "fact" or node.chapter_index > current_index:
            continue
        if node.chapter_index == current_index and not set(node.source_ids) & committed:
            continue
        attributes = node.attributes if isinstance(node.attributes, dict) else {}
        kind = str(attributes.get("fact_kind", ""))
        predicate = str(attributes.get("predicate", ""))
        for subject_id in fact_subjects.get(node_id, ()):
            slot = (subject_id, kind, predicate)
            if slot in seed_fact_slots:
                state_candidates[slot].append(node)
    for candidates in state_candidates.values():
        for rank, node in enumerate(sorted(candidates, key=lambda item: (item.chapter_index, item.order, item.node_id), reverse=True)[:2]):
            offer(node.node_id, 820.0 - 20.0 * rank, "同一状态槽的近期记录")

    source_nodes: dict[str, list[str]] = defaultdict(list)
    for node in graph.nodes:
        for source_id in node.source_ids:
            source_nodes[source_id].append(node.node_id)

    def edge_time_visible(edge: Any) -> bool:
        if edge.chapter_index < current_index:
            return True
        if edge.chapter_index > current_index:
            return False
        return bool(set(edge.source_ids) & licensed)

    # Relationship edges and past participant events restore character
    # context.  Generic semantic overlap keeps a protagonist's entire history
    # from crowding out the action currently being written.
    for edge in graph.edges:
        if edge.kind != "relationship" or edge.chapter_index >= current_index:
            continue
        endpoints = {edge.source_node_id, edge.target_node_id}
        if not endpoints & seed_entities:
            continue
        for endpoint in endpoints:
            offer(endpoint, 700.0, "相关人物关系")
        for source_id in edge.source_ids:
            for node_id in source_nodes.get(source_id, ()):
                offer(node_id, 710.0, "人物关系依据")

    current_distance = max(1, current_index)
    for event_id, participants in event_participants.items():
        if event_id not in visible or nodes[event_id].chapter_index > current_index:
            continue
        same_chapter_committed = (
            nodes[event_id].chapter_index == current_index
            and bool(set(nodes[event_id].source_ids) & committed)
        )
        if nodes[event_id].chapter_index == current_index and not same_chapter_committed:
            continue
        overlap = len(set(participants) & seed_entities)
        if not overlap:
            continue
        participant_names = tuple(nodes[node_id].label for node_id in participants if node_id in nodes)
        semantic = _similarity(
            query_units,
            _semantic_units(_node_text(nodes[event_id]), excluded_terms=(*seed_entity_names, *participant_names)),
        )
        if overlap < 2 and semantic <= 0.0:
            continue
        recency = 1.0 if same_chapter_committed else max(0.0, 1.0 - (current_index - nodes[event_id].chapter_index - 1) / current_distance)
        score = 560.0 if same_chapter_committed else 430.0
        score += min(2, overlap) * 60.0 + semantic * 180.0 + recency * 45.0
        offer(event_id, score, "已验证的同章事件" if same_chapter_committed else "相关人物的历史事件")
        if semantic > 0:
            reasons[event_id].append("与当前行动语义相关")

    # Expand high-value events/facts to the structural nodes that explain
    # them.  This is the graph analogue of retrieving a small evidence chunk,
    # but it contains structured facts only.
    structural_edges = [
        edge for edge in graph.edges
        if edge.kind in RETRIEVAL_EDGE_KINDS
        and edge.source_node_id in visible and edge.target_node_id in visible
        and edge_time_visible(edge)
    ]
    anchor_ids = set(scores)
    for edge in structural_edges:
        if edge.source_node_id not in anchor_ids and edge.target_node_id not in anchor_ids:
            continue
        anchor_id = edge.source_node_id if edge.source_node_id in anchor_ids else edge.target_node_id
        other_id = edge.target_node_id if anchor_id == edge.source_node_id else edge.source_node_id
        base = scores.get(anchor_id, 0.0)
        if edge.kind in _EVENT_FACT_EDGES:
            offer(other_id, base - 45.0, "事件的前提或结果")
        elif edge.kind == "asserts" and edge.target_node_id in anchor_ids:
            offer(edge.source_node_id, base - 70.0, "事实主体")
        elif edge.kind == "fact_object" and edge.source_node_id in anchor_ids:
            offer(edge.target_node_id, base - 70.0, "事实客体")
        elif edge.kind == "event_participant" and edge.target_node_id in anchor_ids:
            offer(edge.source_node_id, base - 70.0, "事件参与者")
        elif edge.kind in {"causes", "foreshadow_open", "foreshadow_resolved"}:
            offer(other_id, base - 55.0, "逻辑关系连接节点")

    budget = max(len(seeds), int(node_budget))
    ordered_ids = sorted(
        scores,
        key=lambda node_id: (
            node_id not in seeds,
            -scores[node_id],
            -nodes[node_id].chapter_index,
            -nodes[node_id].order,
            node_id,
        ),
    )[:budget]
    selected = set(ordered_ids)

    def edge_visible(edge: Any) -> bool:
        if edge.source_node_id not in selected or edge.target_node_id not in selected:
            return False
        if edge.kind not in RETRIEVAL_EDGE_KINDS:
            return False
        # A future payoff edge must never reveal the resolving event.  A past
        # or currently licensed payoff is safe because both endpoints have
        # already passed the time gate.
        if edge.kind == "foreshadow_resolved" and edge.source_node_id not in visible:
            return False
        return edge_time_visible(edge)

    selected_edges = [edge for edge in graph.edges if edge_visible(edge)]
    selected_edges.sort(key=lambda edge: (
        -(scores.get(edge.source_node_id, 0.0) + scores.get(edge.target_node_id, 0.0)),
        edge.chapter_index,
        edge.order,
        edge.edge_id,
    ))
    selected_edges = selected_edges[:max(0, int(edge_budget))]
    historical_count = sum(1 for node_id in ordered_ids if 0 <= nodes[node_id].chapter_index < current_index)
    committed_same_chapter_count = sum(
        1 for node_id in ordered_ids
        if nodes[node_id].chapter_index == current_index and bool(set(nodes[node_id].source_ids) & committed)
    )
    return GraphRetrievalResult(
        node_ids=tuple(ordered_ids),
        edge_ids=tuple(edge.edge_id for edge in selected_edges),
        reasons={node_id: tuple(reasons[node_id]) for node_id in ordered_ids},
        scores={node_id: scores[node_id] for node_id in ordered_ids},
        seed_count=len(seeds),
        historical_count=historical_count,
        committed_same_chapter_count=committed_same_chapter_count,
        max_hops=max(1, int(max_hops)),
        node_budget=budget,
        edge_budget=max(0, int(edge_budget)),
    )
