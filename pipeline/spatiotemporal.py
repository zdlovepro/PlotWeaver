"""Deterministic, evidence-preserving time and space enrichment.

The model extracts atomic facts and events.  This module makes their temporal
ordering and fact-backed locations explicit without asking a model to invent a
map, a clock, or an unstated movement.  Every created relation retains either
the underlying fact evidence or the two ordered event evidences.
"""

from __future__ import annotations

import re
from typing import Iterable

from .contracts import (
    ChapterAnnotation,
    ChapterDocument,
    SceneCard,
    SourceSpan,
    SpatialRelation,
    TemporalRelation,
    TimeAnchor,
)


# Match only textual time cues.  The chronology relation itself is separately
# marked as ``document_order`` so consumers never mistake layout order for a
# literal statement such as "three days later".
_TIME_CUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("relative", re.compile(r"(?:翌日|次日|当日|此时|随后|不久后|片刻后|半晌后|数日后|数月后|数年后|一炷香后)")),
    ("relative", re.compile(r"第[一二三四五六七八九十百千0-9]+[天日]")),
    ("relative", re.compile(r"[一二两三四五六七八九十百千0-9]+(?:天|日|月|年|时辰|刻钟|炷香)(?:后|前)")),
    ("explicit", re.compile(r"(?:清晨|清早|黎明|午时|正午|傍晚|黄昏|深夜|夜里)")),
)


def _dedupe_spans(spans: Iterable[SourceSpan]) -> tuple[SourceSpan, ...]:
    seen: set[tuple[int, int, str, str]] = set()
    result: list[SourceSpan] = []
    for span in spans:
        key = (span.start, span.end, span.quote, span.unit_id)
        if key not in seen:
            seen.add(key)
            result.append(span)
    return tuple(result)


def _first_start(item: object) -> int:
    evidence = getattr(item, "evidence", ())
    return min((span.start for span in evidence), default=10**12)


def _time_anchors(document: ChapterDocument) -> tuple[TimeAnchor, ...]:
    found: list[tuple[int, int, str, str, SourceSpan]] = []
    for unit in document.annotation_units:
        for kind, pattern in _TIME_CUE_PATTERNS:
            for match in pattern.finditer(unit.text):
                label = match.group(0)
                start = unit.start + match.start()
                span = SourceSpan(document.chapter_id, start, start + len(label), label, unit.unit_id)
                found.append((start, span.end, label, kind, span))
    unique: list[tuple[int, int, str, str, SourceSpan]] = []
    seen: set[tuple[int, int, str]] = set()
    for item in sorted(found, key=lambda row: (row[0], row[1], row[2], row[3])):
        key = (item[0], item[1], item[2])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return tuple(
        TimeAnchor(f"{document.chapter_id}:time-{index:03d}", document.chapter_id, label, kind, (span,))
        for index, (_, _, label, kind, span) in enumerate(unique, start=1)
    )


def _spatial_kind(predicate: str) -> str:
    normalized = predicate.replace(" ", "")
    if any(token in normalized for token in ("进入", "踏入", "走进")):
        return "enters"
    if any(token in normalized for token in ("离开", "出去", "走出", "下山", "逐出", "赶出")):
        return "leaves"
    if any(token in normalized for token in ("到达", "来到", "抵达", "前往", "回到", "返回", "赶往", "上山", "定居", "带到")):
        return "moves_to"
    return "at"


def _overlaps(a: tuple[SourceSpan, ...], b: tuple[SourceSpan, ...]) -> bool:
    if not a or not b:
        return False
    a_start, a_end = min(span.start for span in a), max(span.end for span in a)
    b_start, b_end = min(span.start for span in b), max(span.end for span in b)
    return a_start <= b_end and b_start <= a_end


def enrich_spatiotemporal(document: ChapterDocument, annotation: ChapterAnnotation) -> ChapterAnnotation:
    """Return a schema-2.1 annotation with conservative time/space relations.

    The operation is idempotent: it recomputes only derived fields from the
    current atomic annotation, rather than trusting potentially stale derived
    IDs from a previous schema version.
    """

    location_ids = {entity.entity_id for entity in annotation.entities if entity.kind == "location"}
    spatial_relations = tuple(
        SpatialRelation(
            f"{document.chapter_id}:space-{index:03d}",
            document.chapter_id,
            fact.subject_id,
            fact.object_id,
            _spatial_kind(fact.predicate),
            fact.fact_id,
            fact.evidence,
        )
        for index, fact in enumerate(
            (fact for fact in annotation.facts if fact.kind == "location" and fact.object_id in location_ids),
            start=1,
        )
    )

    ordered_events = tuple(sorted(annotation.events, key=lambda event: (_first_start(event), event.order, event.event_id)))
    temporal_relations = tuple(
        TemporalRelation(
            f"{document.chapter_id}:time-relation-{index:03d}",
            document.chapter_id,
            previous.event_id,
            current.event_id,
            "before",
            "document_order",
            _dedupe_spans((*previous.evidence, *current.evidence)),
        )
        for index, (previous, current) in enumerate(zip(ordered_events, ordered_events[1:]), start=1)
    )
    anchors = _time_anchors(document)

    event_facts = {
        event.event_id: {
            *event.trigger_fact_ids,
            *event.precondition_fact_ids,
            *event.outcome_fact_ids,
            *event.cost_fact_ids,
        }
        for event in annotation.events
    }
    locations_by_fact: dict[str, list[str]] = {}
    for relation in spatial_relations:
        locations_by_fact.setdefault(relation.fact_id, []).append(relation.location_id)

    scenes: list[SceneCard] = []
    for scene in annotation.scenes:
        facts = {fact_id for event_id in scene.event_ids for fact_id in event_facts.get(event_id, set())}
        locations = tuple(dict.fromkeys(location for fact_id in facts for location in locations_by_fact.get(fact_id, [])))
        scene_anchors = tuple(anchor.anchor_id for anchor in anchors if _overlaps(scene.evidence, anchor.evidence))
        scenes.append(SceneCard(
            scene.scene_id, scene.chapter_id, scene.order, scene.participant_ids, scene.event_ids,
            scene.objective, scene.entry_fact_ids, scene.exit_fact_ids, scene.tension, scene.evidence,
            locations, scene_anchors,
        ))

    return ChapterAnnotation(
        annotation.chapter_id,
        annotation.source_hash,
        annotation.entities,
        annotation.facts,
        annotation.events,
        tuple(scenes),
        annotation.state_changes,
        "2.1",
        anchors,
        temporal_relations,
        spatial_relations,
    )
