"""Typed, evidence-first contracts for the local narrative fact graph.

The graph is a lossless structural projection of accepted chapter annotations.
It deliberately stores only source-backed facts and the deterministic links
between them; it is not a model-invented world-setting graph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .source import SourceSpan


GRAPH_NODE_KINDS = frozenset({"entity", "fact", "event", "scene", "time_anchor", "foreshadow"})
GRAPH_EDGE_KINDS = frozenset({
    "asserts", "fact_object", "event_participant", "event_trigger", "event_requires",
    "event_produces", "event_cost", "scene_contains", "scene_participant", "scene_entry_state",
    "scene_exit_state", "scene_at", "scene_time", "temporal_before", "temporal_simultaneous", "timeline_precedes",
    "state_change", "state_observed", "spatial_relation", "location_transition", "relationship",
    "causes", "foreshadow_open", "foreshadow_resolved",
})


def _required(value: str, field_name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


def _unique(values: tuple[str, ...], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")


def _validate_evidence(spans: tuple[SourceSpan, ...], chapter_id: str, field_name: str) -> None:
    if not spans:
        raise ValueError(f"{field_name} must have source evidence")
    for span in spans:
        span.validate()
        if chapter_id and span.chapter_id != chapter_id:
            raise ValueError(f"{field_name} evidence chapter does not match")


@dataclass(frozen=True)
class NarrativeGraphNode:
    """A source-backed narrative object, with deterministic graph identity."""

    node_id: str
    kind: str
    label: str
    source_ids: tuple[str, ...]
    evidence: tuple[SourceSpan, ...]
    chapter_id: str = ""
    chapter_index: int = -1
    order: int = -1
    attributes: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _required(self.node_id, "graph node_id")
        if self.kind not in GRAPH_NODE_KINDS:
            raise ValueError(f"unsupported graph node kind: {self.kind}")
        _required(self.label, "graph node label")
        if not self.source_ids:
            raise ValueError("graph node needs source ids")
        _unique(self.source_ids, "graph node source ids")
        if self.chapter_index < -1 or self.order < -1:
            raise ValueError("graph node indexes must be >= -1")
        if not isinstance(self.attributes, dict):
            raise ValueError("graph node attributes must be an object")
        _validate_evidence(self.evidence, self.chapter_id, "graph node")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "source_ids": list(self.source_ids),
            "evidence": [span.to_dict() for span in self.evidence],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NarrativeGraphNode":
        attributes = payload.get("attributes", {})
        return cls(
            node_id=str(payload.get("node_id", "")),
            kind=str(payload.get("kind", "")),
            label=str(payload.get("label", "")),
            source_ids=tuple(str(item) for item in payload.get("source_ids", []) if str(item).strip()),
            evidence=tuple(SourceSpan.from_dict(item) for item in payload.get("evidence", []) if isinstance(item, dict)),
            chapter_id=str(payload.get("chapter_id", "")),
            chapter_index=int(payload.get("chapter_index", -1)),
            order=int(payload.get("order", -1)),
            attributes=dict(attributes) if isinstance(attributes, dict) else {},
        )


@dataclass(frozen=True)
class NarrativeGraphEdge:
    """A typed, evidence-backed relation between two graph nodes."""

    edge_id: str
    kind: str
    source_node_id: str
    target_node_id: str
    source_ids: tuple[str, ...]
    evidence: tuple[SourceSpan, ...]
    chapter_id: str = ""
    chapter_index: int = -1
    order: int = -1
    attributes: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _required(self.edge_id, "graph edge_id")
        if self.kind not in GRAPH_EDGE_KINDS:
            raise ValueError(f"unsupported graph edge kind: {self.kind}")
        _required(self.source_node_id, "graph edge source")
        _required(self.target_node_id, "graph edge target")
        if self.source_node_id == self.target_node_id:
            raise ValueError("graph edge cannot connect a node to itself")
        if not self.source_ids:
            raise ValueError("graph edge needs source ids")
        _unique(self.source_ids, "graph edge source ids")
        if self.chapter_index < -1 or self.order < -1:
            raise ValueError("graph edge indexes must be >= -1")
        if not isinstance(self.attributes, dict):
            raise ValueError("graph edge attributes must be an object")
        _validate_evidence(self.evidence, self.chapter_id, "graph edge")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "source_ids": list(self.source_ids),
            "evidence": [span.to_dict() for span in self.evidence],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NarrativeGraphEdge":
        attributes = payload.get("attributes", {})
        return cls(
            edge_id=str(payload.get("edge_id", "")),
            kind=str(payload.get("kind", "")),
            source_node_id=str(payload.get("source_node_id", "")),
            target_node_id=str(payload.get("target_node_id", "")),
            source_ids=tuple(str(item) for item in payload.get("source_ids", []) if str(item).strip()),
            evidence=tuple(SourceSpan.from_dict(item) for item in payload.get("evidence", []) if isinstance(item, dict)),
            chapter_id=str(payload.get("chapter_id", "")),
            chapter_index=int(payload.get("chapter_index", -1)),
            order=int(payload.get("order", -1)),
            attributes=dict(attributes) if isinstance(attributes, dict) else {},
        )


@dataclass(frozen=True)
class NarrativeGraph:
    """The local, evidence-first fact base for one imported novel work."""

    author_id: str
    work_id: str
    chapter_ids: tuple[str, ...]
    source_hashes: dict[str, str]
    nodes: tuple[NarrativeGraphNode, ...]
    edges: tuple[NarrativeGraphEdge, ...]
    schema_version: str = "1.0"

    def validate(self) -> None:
        _required(self.author_id, "graph author_id")
        _required(self.work_id, "graph work_id")
        if not self.chapter_ids:
            raise ValueError("graph needs chapters")
        _unique(self.chapter_ids, "graph chapter ids")
        if set(self.source_hashes) != set(self.chapter_ids):
            raise ValueError("graph source hashes must cover exactly its chapters")
        node_ids = [item.node_id for item in self.nodes]
        edge_ids = [item.edge_id for item in self.edges]
        _unique(tuple(node_ids), "graph node ids")
        _unique(tuple(edge_ids), "graph edge ids")
        node_id_set = set(node_ids)
        for node in self.nodes:
            node.validate()
        for edge in self.edges:
            edge.validate()
            if edge.source_node_id not in node_id_set or edge.target_node_id not in node_id_set:
                raise ValueError(f"graph edge {edge.edge_id} references an unknown node")

    def to_dict(self) -> dict[str, Any]:
        return {
            "author_id": self.author_id,
            "work_id": self.work_id,
            "chapter_ids": list(self.chapter_ids),
            "source_hashes": self.source_hashes,
            "nodes": [item.to_dict() for item in self.nodes],
            "edges": [item.to_dict() for item in self.edges],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NarrativeGraph":
        hashes = payload.get("source_hashes", {})
        return cls(
            author_id=str(payload.get("author_id", "")),
            work_id=str(payload.get("work_id", "")),
            chapter_ids=tuple(str(item) for item in payload.get("chapter_ids", []) if str(item).strip()),
            source_hashes={str(key): str(value) for key, value in hashes.items()} if isinstance(hashes, dict) else {},
            nodes=tuple(NarrativeGraphNode.from_dict(item) for item in payload.get("nodes", []) if isinstance(item, dict)),
            edges=tuple(NarrativeGraphEdge.from_dict(item) for item in payload.get("edges", []) if isinstance(item, dict)),
            schema_version=str(payload.get("schema_version", "1.0")),
        )
