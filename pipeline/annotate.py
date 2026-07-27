"""Evidence-grounded chapter annotation for the new distillation pipeline.

The legacy ``extract`` module remains available while later stages are migrated.
This module is intentionally independent: it consumes immutable ``ChapterDocument``
objects and produces only typed, source-verifiable narrative annotations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
import json
from pathlib import Path
import time
from typing import Any, Iterable

from .contracts import (
    ChapterAnnotation,
    ChapterDocument,
    Entity,
    EventAtom,
    Fact,
    SceneCard,
    SourceSpan,
    SourceUnit,
    StateChange,
    build_chapter_document,
    validate_annotation,
    validate_document,
)
from .contracts.validation import ValidationReport
from .jsonio import read_jsonl, write_json, write_jsonl
from .model import ModelSettings, complete_json
from .paths import corpus_dir
from .spatiotemporal import enrich_spatiotemporal


# Keep one model call to a continuous local window.  At chapter scale a model
# tends to compress many causal changes into a plot synopsis.
DEFAULT_ANNOTATION_INPUT_CHARS = 1_200
DEFAULT_ANNOTATION_OVERLAP_UNITS = 1
DEFAULT_ANNOTATION_LIMIT = 20
DEFAULT_ANNOTATION_OUTPUT_TOKENS = 3_500
DEFAULT_ANNOTATION_REPAIR_TOKENS = 2_400
MAX_CHAPTER_QUALITY_REPAIRS = 1
# A long-form planner cannot recover process that was collapsed into a few
# chapter-level bullets.  Keep the extraction density explicit and shared by
# prompts and deterministic quality gates.
DENSE_EVENT_CHARS = 500

ENTITY_KIND_MAP = {
    "person": "person", "人物": "person", "角色": "person", "人": "person",
    "organization": "organization", "组织": "organization", "势力": "organization", "门派": "organization", "群体": "organization", "团体": "organization", "人群": "organization",
    "location": "location", "地点": "location", "场所": "location", "地域": "location",
    "item": "item", "物品": "item", "法宝": "item", "资源": "item",
    "concept": "concept", "概念": "concept", "功法": "concept", "规则": "concept",
    "creature": "creature", "生物": "creature", "妖兽": "creature", "灵兽": "creature",
}
FACT_KIND_MAP = {
    "identity": "identity", "身份": "identity",
    "goal": "goal", "目标": "goal", "意图": "goal", "期望": "goal", "愿望": "goal",
    "emotion": "emotion", "情绪": "emotion",
    "relationship": "relationship", "关系": "relationship",
    "location": "location", "位置": "location", "地点": "location",
    "resource": "resource", "资源": "resource", "物品": "resource", "拥有": "resource", "获得": "resource", "失去": "resource",
    "knowledge": "knowledge", "认知": "knowledge", "信息": "knowledge",
    "rule": "rule", "规则": "rule", "限制": "rule",
    "progression": "progression", "进展": "progression", "事件": "progression", "境界": "progression",
    "information": "information", "伏笔": "information", "属性": "information", "特征": "information", "状态": "information",
}
CERTAINTY_MAP = {"明确": 1.0, "直接": 1.0, "可确认": 1.0, "较明确": 0.8, "不确定": 0.6}


ANNOTATION_DRAFT_SCHEMA = {
    "entities": [{"id": "e1", "name": "", "kind": "人物", "aliases": [], "evidence": [{"unit_id": "", "quote": ""}]}],
    "facts": [{"id": "f1", "kind": "地点", "subject_id": "e1", "predicate": "", "value": "", "object_id": "", "certainty": "明确", "evidence": [{"unit_id": "", "quote": ""}]}],
    "events": [{"id": "v1", "order": 1, "summary": "", "participant_ids": ["e1"], "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "", "obstacle": "", "decision": "", "outcome_fact_ids": ["f1"], "cost_fact_ids": [], "evidence": [{"unit_id": "", "quote": ""}]}],
    "scenes": [{"id": "s1", "order": 1, "participant_ids": ["e1"], "event_ids": ["v1"], "objective": "", "entry_fact_ids": [], "exit_fact_ids": ["f1"], "tension": 0, "evidence": [{"unit_id": "", "quote": ""}]}],
    "state_changes": [{"id": "c1", "event_id": "v1", "operation": "add", "before_fact_id": "", "after_fact_id": "f1"}],
}

ANNOTATION_DRAFT_EXAMPLE = {
    "entities": [
        {"id": "e1", "name": "角色甲", "kind": "人物", "aliases": [], "evidence": [{"unit_id": "章节:p0000", "quote": "角色甲"}]},
        {"id": "e2", "name": "村口", "kind": "地点", "aliases": [], "evidence": [{"unit_id": "章节:p0000", "quote": "村口"}]},
    ],
    "facts": [
        {"id": "f1", "kind": "地点", "subject_id": "e1", "predicate": "到达", "value": "村口", "object_id": "e2", "certainty": "明确", "evidence": [{"unit_id": "章节:p0000", "quote": "角色甲来到村口"}]},
        {"id": "f2", "kind": "资源", "subject_id": "e1", "predicate": "获得", "value": "令牌", "object_id": "", "certainty": "明确", "evidence": [{"unit_id": "章节:p0001", "quote": "得到一枚令牌"}]},
    ],
    "events": [{"id": "v1", "order": 1, "summary": "角色甲到达村口后获得令牌", "participant_ids": ["e1"], "trigger_fact_ids": ["f1"], "precondition_fact_ids": [], "action": "前往村口并取得令牌", "obstacle": "", "decision": "", "outcome_fact_ids": ["f2"], "cost_fact_ids": [], "evidence": [{"unit_id": "章节:p0000", "quote": "角色甲来到村口"}, {"unit_id": "章节:p0001", "quote": "得到一枚令牌"}]}],
    "scenes": [{"id": "s1", "order": 1, "participant_ids": ["e1"], "event_ids": ["v1"], "objective": "取得令牌", "entry_fact_ids": ["f1"], "exit_fact_ids": ["f2"], "tension": 2, "evidence": [{"unit_id": "章节:p0000", "quote": "角色甲来到村口"}, {"unit_id": "章节:p0001", "quote": "得到一枚令牌"}]}],
    "state_changes": [{"id": "c1", "event_id": "v1", "operation": "add", "before_fact_id": "", "after_fact_id": "f2"}],
}


@dataclass(frozen=True)
class AnnotationQualityIssue:
    code: str
    path: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class AnnotationQualityReport:
    chapter_id: str
    issues: tuple[AnnotationQualityIssue, ...] = field(default_factory=tuple)
    event_count: int = 0
    scene_count: int = 0
    time_anchor_count: int = 0
    temporal_relation_count: int = 0
    spatial_relation_count: int = 0
    evidence_unit_count: int = 0
    eligible_unit_count: int = 0
    fact_count: int = 0
    anchored_event_count: int = 0

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter_id": self.chapter_id,
            "passed": self.passed,
            "event_count": self.event_count,
            "scene_count": self.scene_count,
            "time_anchor_count": self.time_anchor_count,
            "temporal_relation_count": self.temporal_relation_count,
            "spatial_relation_count": self.spatial_relation_count,
            "evidence_unit_count": self.evidence_unit_count,
            "eligible_unit_count": self.eligible_unit_count,
            "fact_count": self.fact_count,
            "anchored_event_count": self.anchored_event_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }


class AnnotationBuildError(ValueError):
    """A model draft is syntactically valid JSON but fails evidence contracts."""

    def __init__(self, messages: Iterable[str]) -> None:
        self.messages = tuple(str(message) for message in messages if str(message).strip())
        super().__init__("; ".join(self.messages) or "annotation draft is invalid")


@dataclass(frozen=True)
class AnnotationSourceSlice:
    """One prompt-sized excerpt that still maps to exact original offsets."""

    unit_id: str
    source_unit_id: str
    start: int
    end: int
    text: str


def _required_text(value: Any, path: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise AnnotationBuildError([f"{path}: 不能为空"])
    return result


def _rows(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key, [])
    if not isinstance(value, list):
        raise AnnotationBuildError([f"{key}: 必须是数组"])
    if not all(isinstance(item, dict) for item in value):
        raise AnnotationBuildError([f"{key}: 数组元素必须是对象"])
    return value


def _string_ids(value: Any, path: str, *, required: bool = False) -> tuple[str, ...]:
    if value in (None, ""):
        result: tuple[str, ...] = ()
    elif not isinstance(value, list):
        raise AnnotationBuildError([f"{path}: 必须是字符串数组"])
    else:
        result = tuple(str(item).strip() for item in value if str(item).strip())
    if required and not result:
        raise AnnotationBuildError([f"{path}: 至少需要一个引用"])
    if len(result) != len(set(result)):
        raise AnnotationBuildError([f"{path}: 不得重复引用同一 id"])
    return result


def _mapped_kind(value: Any, mapping: dict[str, str], path: str) -> str:
    raw = _required_text(value, path)
    result = mapping.get(raw.lower(), mapping.get(raw, ""))
    if not result:
        allowed = "、".join(sorted(set(mapping.values())))
        raise AnnotationBuildError([f"{path}: 不支持“{raw}”，应映射为 {allowed}"])
    return result


def _certainty(value: Any, path: str) -> float:
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        raw = str(value or "明确").strip()
        try:
            result = float(raw)
        except ValueError:
            result = CERTAINTY_MAP.get(raw, 0.0)
    if not 0.0 < result <= 1.0:
        raise AnnotationBuildError([f"{path}: 必须是 0 到 1 的数值或明确/较明确/不确定"])
    return result


def _dedupe_spans(spans: Iterable[SourceSpan]) -> tuple[SourceSpan, ...]:
    seen: set[tuple[int, int, str, str]] = set()
    result: list[SourceSpan] = []
    for span in spans:
        key = (span.start, span.end, span.quote, span.unit_id)
        if key not in seen:
            seen.add(key)
            result.append(span)
    return tuple(result)


def _spans(
    raw: Any,
    document: ChapterDocument,
    path: str,
    evidence_units: tuple[AnnotationSourceSlice, ...] | None = None,
) -> tuple[SourceSpan, ...]:
    if not isinstance(raw, list) or not raw:
        raise AnnotationBuildError([f"{path}: 必须提供至少一条原文证据"])
    if evidence_units is None:
        evidence_units = tuple(
            AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text)
            for unit in document.units
        )
    unit_by_id = {unit.unit_id: unit for unit in evidence_units}
    spans: list[SourceSpan] = []
    errors: list[str] = []
    for index, item in enumerate(raw):
        item_path = f"{path}[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{item_path}: 必须是对象")
            continue
        unit_id = str(item.get("unit_id", "")).strip()
        quote = str(item.get("quote", "")).strip()
        unit = unit_by_id.get(unit_id)
        if unit is None:
            errors.append(f"{item_path}.unit_id: 未知单元 {unit_id!r}")
            continue
        if not quote:
            errors.append(f"{item_path}.quote: 不能为空")
            continue
        if "\n" in quote or "\r" in quote:
            errors.append(f"{item_path}.quote: 不得跨行")
            continue
        relative_start = unit.text.find(quote)
        if relative_start < 0:
            # Models occasionally copy an exact quote but attach the neighbouring
            # paragraph id.  We may repair only an unambiguous locator mismatch;
            # paraphrases still fail because they cannot be found verbatim.
            candidates = [(candidate, candidate.text.find(quote)) for candidate in evidence_units if quote in candidate.text]
            if len(candidates) == 1:
                unit, relative_start = candidates[0]
            else:
                declared_index = next((position for position, candidate in enumerate(evidence_units) if candidate.unit_id == unit_id), -1)
                distances = [(abs(position - declared_index), position, candidate, start) for position, (candidate, start) in enumerate(candidates)] if declared_index >= 0 else []
                distances.sort(key=lambda item: item[:2])
                if not distances or distances[0][0] > 3 or (len(distances) > 1 and distances[0][0] == distances[1][0]):
                    errors.append(f"{item_path}.quote: 不是 unit_id 对应单元中的原文逐字片段，且无法在当前片段唯一定位")
                    continue
                _, _, unit, relative_start = distances[0]
        start = unit.start + relative_start
        spans.append(SourceSpan(document.chapter_id, start, start + len(quote), quote, unit.source_unit_id))
    if errors:
        raise AnnotationBuildError(errors)
    return _dedupe_spans(spans)


def annotation_from_draft(
    document: ChapterDocument,
    payload: dict[str, Any],
    id_prefix: str = "",
    evidence_units: tuple[AnnotationSourceSlice, ...] | None = None,
) -> ChapterAnnotation:
    """Resolve a model's local IDs and quoted unit evidence into typed contracts."""

    if not isinstance(payload, dict):
        raise AnnotationBuildError(["根对象必须是 JSON object"])
    prefix = id_prefix or document.chapter_id
    entity_rows = _rows(payload, "entities")
    fact_rows = _rows(payload, "facts")
    event_rows = _rows(payload, "events")
    scene_rows = _rows(payload, "scenes")
    change_rows = _rows(payload, "state_changes")
    errors: list[str] = []

    def unique_local_ids(rows: list[dict[str, Any]], field: str) -> list[str]:
        values = [str(row.get("id", "")).strip() for row in rows]
        missing = [index for index, value in enumerate(values) if not value]
        duplicated = sorted({value for value in values if value and values.count(value) > 1})
        if missing:
            errors.append(f"{field}: 第 {','.join(str(index + 1) for index in missing)} 项缺少 id")
        if duplicated:
            errors.append(f"{field}: 本地 id 重复：{','.join(duplicated)}")
        return values

    entity_local = unique_local_ids(entity_rows, "entities")
    fact_local = unique_local_ids(fact_rows, "facts")
    event_local = unique_local_ids(event_rows, "events")
    scene_local = unique_local_ids(scene_rows, "scenes")
    change_local = unique_local_ids(change_rows, "state_changes")
    if errors:
        raise AnnotationBuildError(errors)
    entity_ids = {local: f"{prefix}:entity-{index:03d}" for index, local in enumerate(entity_local, start=1)}
    fact_ids = {local: f"{prefix}:fact-{index:03d}" for index, local in enumerate(fact_local, start=1)}
    event_ids = {local: f"{prefix}:event-{index:03d}" for index, local in enumerate(event_local, start=1)}
    scene_ids = {local: f"{prefix}:scene-{index:03d}" for index, local in enumerate(scene_local, start=1)}

    def references(values: tuple[str, ...], lookup: dict[str, str], path: str) -> tuple[str, ...]:
        unknown = [value for value in values if value not in lookup]
        if unknown:
            raise AnnotationBuildError([f"{path}: 未声明的引用 {','.join(unknown)}"])
        return tuple(lookup[value] for value in values)

    entities: list[Entity] = []
    for index, row in enumerate(entity_rows):
        path = f"entities[{index}]"
        aliases = _string_ids(row.get("aliases", []), f"{path}.aliases")
        name = _required_text(row.get("name"), f"{path}.name")
        aliases = tuple(alias for alias in aliases if alias != name)
        entities.append(Entity(
            entity_ids[entity_local[index]], name, _mapped_kind(row.get("kind"), ENTITY_KIND_MAP, f"{path}.kind"), aliases,
            _spans(row.get("evidence"), document, f"{path}.evidence", evidence_units),
        ))
    facts: list[Fact] = []
    for index, row in enumerate(fact_rows):
        path = f"facts[{index}]"
        subject = _required_text(row.get("subject_id"), f"{path}.subject_id")
        object_value = str(row.get("object_id", "")).strip()
        facts.append(Fact(
            fact_ids[fact_local[index]], document.chapter_id, _mapped_kind(row.get("kind"), FACT_KIND_MAP, f"{path}.kind"),
            references((subject,), entity_ids, f"{path}.subject_id")[0], _required_text(row.get("predicate"), f"{path}.predicate"),
            str(row.get("value", "")).strip(), references((object_value,), entity_ids, f"{path}.object_id")[0] if object_value else "",
            _certainty(row.get("certainty", "明确"), f"{path}.certainty"), _spans(row.get("evidence"), document, f"{path}.evidence", evidence_units),
        ))
    events: list[EventAtom] = []
    for index, row in enumerate(event_rows):
        path = f"events[{index}]"
        events.append(EventAtom(
            event_ids[event_local[index]], document.chapter_id, int(row.get("order", index + 1)) - 1,
            _required_text(row.get("summary"), f"{path}.summary"),
            references(_string_ids(row.get("participant_ids"), f"{path}.participant_ids", required=True), entity_ids, f"{path}.participant_ids"),
            references(_string_ids(row.get("trigger_fact_ids", []), f"{path}.trigger_fact_ids"), fact_ids, f"{path}.trigger_fact_ids"),
            references(_string_ids(row.get("precondition_fact_ids", []), f"{path}.precondition_fact_ids"), fact_ids, f"{path}.precondition_fact_ids"),
            _required_text(row.get("action"), f"{path}.action"), str(row.get("obstacle", "")).strip(), str(row.get("decision", "")).strip(),
            references(_string_ids(row.get("outcome_fact_ids"), f"{path}.outcome_fact_ids", required=True), fact_ids, f"{path}.outcome_fact_ids"),
            references(_string_ids(row.get("cost_fact_ids", []), f"{path}.cost_fact_ids"), fact_ids, f"{path}.cost_fact_ids"),
            _spans(row.get("evidence"), document, f"{path}.evidence", evidence_units),
        ))
    scenes: list[SceneCard] = []
    for index, row in enumerate(scene_rows):
        path = f"scenes[{index}]"
        scenes.append(SceneCard(
            scene_ids[scene_local[index]], document.chapter_id, int(row.get("order", index + 1)) - 1,
            references(_string_ids(row.get("participant_ids"), f"{path}.participant_ids", required=True), entity_ids, f"{path}.participant_ids"),
            references(_string_ids(row.get("event_ids"), f"{path}.event_ids", required=True), event_ids, f"{path}.event_ids"),
            _required_text(row.get("objective"), f"{path}.objective"),
            references(_string_ids(row.get("entry_fact_ids", []), f"{path}.entry_fact_ids"), fact_ids, f"{path}.entry_fact_ids"),
            references(_string_ids(row.get("exit_fact_ids"), f"{path}.exit_fact_ids", required=True), fact_ids, f"{path}.exit_fact_ids"),
            int(row.get("tension", 0)), _spans(row.get("evidence"), document, f"{path}.evidence", evidence_units),
        ))
    changes: list[StateChange] = []
    for index, row in enumerate(change_rows):
        path = f"state_changes[{index}]"
        event_local_id = _required_text(row.get("event_id"), f"{path}.event_id")
        before = str(row.get("before_fact_id", "")).strip()
        after = str(row.get("after_fact_id", "")).strip()
        changes.append(StateChange(
            f"{prefix}:change-{index + 1:03d}", document.chapter_id,
            references((event_local_id,), event_ids, f"{path}.event_id")[0], _required_text(row.get("operation"), f"{path}.operation"),
            references((before,), fact_ids, f"{path}.before_fact_id")[0] if before else "",
            references((after,), fact_ids, f"{path}.after_fact_id")[0] if after else "",
        ))
    # Chunk-level drafts predate derived time/space fields.  They become schema
    # 2.1 only after merged events have a stable source order.
    return ChapterAnnotation(document.chapter_id, document.source_hash, tuple(entities), tuple(facts), tuple(events), tuple(scenes), tuple(changes), "2.0")


def _all_spans(annotation: ChapterAnnotation) -> tuple[SourceSpan, ...]:
    return tuple(
        span
        for item in (*annotation.entities, *annotation.facts, *annotation.events, *annotation.scenes)
        for span in item.evidence
    )


def enrich_entity_evidence(document: ChapterDocument, annotation: ChapterAnnotation) -> ChapterAnnotation:
    """Add an exact name mention when the model cited only a pronoun or relation."""

    enriched: list[Entity] = []
    for entity in annotation.entities:
        names = (entity.canonical_name, *entity.aliases)
        if any(name and name in span.quote for name in names for span in entity.evidence):
            enriched.append(entity)
            continue
        direct_span: SourceSpan | None = None
        for name in names:
            if not name:
                continue
            for unit in document.annotation_units:
                position = unit.text.find(name)
                if position >= 0:
                    direct_span = SourceSpan(document.chapter_id, unit.start + position, unit.start + position + len(name), name, unit.unit_id)
                    break
            if direct_span is not None:
                break
        enriched.append(
            Entity(entity.entity_id, entity.canonical_name, entity.kind, entity.aliases, _dedupe_spans((*entity.evidence, *( (direct_span,) if direct_span else ()))))
        )
    return ChapterAnnotation(
        annotation.chapter_id, annotation.source_hash, tuple(enriched), annotation.facts,
        annotation.events, annotation.scenes, annotation.state_changes, annotation.schema_version,
    )


def order_annotation_by_evidence(annotation: ChapterAnnotation) -> ChapterAnnotation:
    """Use source offsets, rather than model numbering, as the canonical chronology."""

    events = tuple(
        EventAtom(
            event.event_id, event.chapter_id, index, event.summary, event.participant_ids,
            event.trigger_fact_ids, event.precondition_fact_ids, event.action, event.obstacle,
            event.decision, event.outcome_fact_ids, event.cost_fact_ids, event.evidence,
        )
        for index, event in enumerate(sorted(annotation.events, key=lambda item: (_min_evidence_start(item), item.order, item.event_id)))
    )
    scenes = tuple(
        SceneCard(
            scene.scene_id, scene.chapter_id, index, scene.participant_ids, scene.event_ids,
            scene.objective, scene.entry_fact_ids, scene.exit_fact_ids, scene.tension, scene.evidence,
        )
        for index, scene in enumerate(sorted(annotation.scenes, key=lambda item: (_min_evidence_start(item), item.order, item.scene_id)))
    )
    return ChapterAnnotation(
        annotation.chapter_id, annotation.source_hash, annotation.entities, annotation.facts,
        events, scenes, annotation.state_changes, annotation.schema_version,
    )


def assess_annotation_quality(
    document: ChapterDocument,
    annotation: ChapterAnnotation,
    coverage_units: Iterable[AnnotationSourceSlice] | None = None,
) -> AnnotationQualityReport:
    """Check semantic coverage only after exact-evidence contract validation."""

    issues: list[AnnotationQualityIssue] = []
    eligible = tuple(coverage_units) if coverage_units is not None else document.annotation_units
    spans = _all_spans(annotation)
    cited_units = {span.unit_id for span in spans if span.unit_id}
    eligible_chars = sum(len(unit.text) for unit in eligible)
    # Each event is an atomic causal change that later prose may need to enact.
    # This prevents a 5--6k-character chapter becoming three plot bullets.
    min_events = max(1, math.ceil(eligible_chars / DENSE_EVENT_CHARS))
    min_cited_units = min(len(eligible), max(2, math.ceil(len(eligible) * 0.08)))
    if len(annotation.events) < min_events:
        issues.append(AnnotationQualityIssue("event_coverage", "events", f"正文约 {eligible_chars} 字，至少应抽取 {min_events} 个因果事件"))
    min_facts = max(2, min_events * 2)
    if len(annotation.facts) < min_facts:
        issues.append(AnnotationQualityIssue("fact_density", "facts", f"正文至少需要 {min_facts} 条带证据的事实，当前为 {len(annotation.facts)}"))
    min_scenes = max(1, math.ceil(eligible_chars / 1_600))
    if len(annotation.scenes) < min_scenes:
        issues.append(AnnotationQualityIssue("scene_coverage", "scenes", f"正文约 {eligible_chars} 字，至少应划分 {min_scenes} 个可追踪场景"))
    if len(cited_units) < min_cited_units:
        issues.append(AnnotationQualityIssue("evidence_coverage", "evidence", f"至少应覆盖 {min_cited_units} 个正文单元，当前为 {len(cited_units)}"))
    if eligible_chars >= 1_800:
        source_start, source_end = eligible[0].start, eligible[-1].end
        event_starts = [span.start for event in annotation.events for span in event.evidence]
        event_ends = [span.end for event in annotation.events for span in event.evidence]
        source_span = max(1, source_end - source_start)
        if not event_starts or min(event_starts) > source_start + source_span * 0.35:
            issues.append(AnnotationQualityIssue("opening_event_coverage", "events", "事件证据没有覆盖正文开端的关键变化，建议复核", "warning"))
        if not event_ends or max(event_ends) < source_start + source_span * 0.65:
            issues.append(AnnotationQualityIssue("ending_event_coverage", "events", "事件证据没有覆盖正文结尾的关键变化，建议复核", "warning"))
        # Opening/end checks alone miss a large unmodelled middle section.
        coverage_points = sorted({source_start, source_end, *event_starts, *event_ends})
        largest_gap = max((right - left for left, right in zip(coverage_points, coverage_points[1:])), default=source_span)
        allowed_gap = max(1_200, math.ceil(source_span * 0.28))
        if largest_gap > allowed_gap:
            issues.append(AnnotationQualityIssue(
                "event_coverage_gap",
                "events",
                f"事件证据之间最大空档为 {largest_gap} 字，超过允许的 {allowed_gap} 字；请补足中段的原子事件",
            ))
    if not annotation.state_changes:
        issues.append(AnnotationQualityIssue("missing_state_change", "state_changes", "章节应至少记录一项由事件造成的状态变化", "warning"))
    for index, entity in enumerate(annotation.entities):
        names = (entity.canonical_name, *entity.aliases)
        if not any(name and name in span.quote for name in names for span in entity.evidence):
            issues.append(AnnotationQualityIssue("entity_evidence_weak", f"entities[{index}]", "实体证据未直接出现规范名或别名，需在后续人工抽检中关注", "warning"))
    for index, scene in enumerate(annotation.scenes):
        if not scene.entry_fact_ids and not scene.exit_fact_ids:
            issues.append(AnnotationQualityIssue("scene_transition_missing", f"scenes[{index}]", "场景没有可追踪的状态事实"))
    if annotation.schema_version >= "2.1":
        expected_relations = max(0, len(annotation.events) - 1)
        if len(annotation.temporal_relations) < expected_relations:
            issues.append(AnnotationQualityIssue(
                "temporal_order_missing", "temporal_relations",
                f"{len(annotation.events)} 个事件至少需要 {expected_relations} 条可追踪的先后关系",
            ))
        location_entity_ids = {entity.entity_id for entity in annotation.entities if entity.kind == "location"}
        location_fact_ids = {
            fact.fact_id for fact in annotation.facts
            if fact.kind == "location" and fact.object_id in location_entity_ids
        }
        spatial_fact_ids = {relation.fact_id for relation in annotation.spatial_relations}
        missing_spatial = location_fact_ids - spatial_fact_ids
        if missing_spatial:
            issues.append(AnnotationQualityIssue(
                "spatial_relation_missing", "spatial_relations",
                "地点事实未全部映射为可追踪的空间关系",
                "warning",
            ))
    return AnnotationQualityReport(
        document.chapter_id,
        tuple(issues),
        len(annotation.events),
        len(annotation.scenes),
        len(annotation.time_anchors),
        len(annotation.temporal_relations),
        len(annotation.spatial_relations),
        len(cited_units),
        len(eligible),
        len(annotation.facts),
        sum(bool(event.evidence) for event in annotation.events),
    )


def _validation_messages(
    document: ChapterDocument,
    annotation: ChapterAnnotation,
    coverage_units: Iterable[AnnotationSourceSlice] | None = None,
) -> list[str]:
    report: ValidationReport = validate_annotation(annotation, document)
    messages = [f"{issue.path}: {issue.message}" for issue in report.issues]
    quality = assess_annotation_quality(document, annotation, coverage_units)
    messages.extend(f"{issue.path}: {issue.message}" for issue in quality.issues if issue.severity == "error")
    return messages


def _assert_acceptable(
    document: ChapterDocument,
    annotation: ChapterAnnotation,
    coverage_units: Iterable[AnnotationSourceSlice] | None = None,
) -> AnnotationQualityReport:
    messages = _validation_messages(document, annotation, coverage_units)
    if messages:
        raise AnnotationBuildError(messages)
    quality = assess_annotation_quality(document, annotation, coverage_units)
    if not quality.passed:
        raise AnnotationBuildError(
            f"{issue.code}: {issue.message}"
            for issue in quality.issues if issue.severity == "error"
        )
    return quality


def _slice_unit(unit: SourceUnit, max_input_chars: int) -> list[AnnotationSourceSlice]:
    """Split an oversized physical line while retaining original source offsets."""

    if len(unit.text) <= max_input_chars:
        return [AnnotationSourceSlice(unit.unit_id, unit.unit_id, unit.start, unit.end, unit.text)]
    pieces: list[AnnotationSourceSlice] = []
    offset = 0
    punctuation = "。！？；.!?;"
    while offset < len(unit.text):
        tentative_end = min(len(unit.text), offset + max_input_chars)
        if tentative_end < len(unit.text):
            candidates = [position for position in range(max(offset + max_input_chars - 240, offset), tentative_end) if unit.text[position] in punctuation]
            if candidates:
                tentative_end = candidates[-1] + 1
        text = unit.text[offset:tentative_end]
        pieces.append(AnnotationSourceSlice(
            f"{unit.unit_id}@{len(pieces) + 1}", unit.unit_id,
            unit.start + offset, unit.start + tentative_end, text,
        ))
        offset = tentative_end
    return pieces


def split_annotation_units(document: ChapterDocument, max_input_chars: int = DEFAULT_ANNOTATION_INPUT_CHARS, overlap_units: int = DEFAULT_ANNOTATION_OVERLAP_UNITS) -> list[tuple[AnnotationSourceSlice, ...]]:
    """Batch long chapters on paragraph boundaries without changing evidence offsets."""

    max_input_chars = max(1_000, int(max_input_chars))
    overlap_units = max(0, int(overlap_units))
    eligible = [slice_ for unit in document.annotation_units for slice_ in _slice_unit(unit, max_input_chars)]
    if not eligible:
        return []
    chunks: list[tuple[Any, ...]] = []
    current: list[AnnotationSourceSlice] = []
    current_chars = 0
    for unit in eligible:
        if current and current_chars + len(unit.text) > max_input_chars:
            chunks.append(tuple(current))
            current = current[-overlap_units:] if overlap_units else []
            current_chars = sum(len(item.text) for item in current)
            if current and current_chars + len(unit.text) > max_input_chars:
                current = []
                current_chars = 0
        current.append(unit)
        current_chars += len(unit.text)
    if current:
        chunks.append(tuple(current))
    return chunks


def _source_payload(document: ChapterDocument, units: tuple[AnnotationSourceSlice, ...]) -> dict[str, Any]:
    return {
        "chapter_id": document.chapter_id,
        "source_hash": document.source_hash,
        "source_units": [{"unit_id": unit.unit_id, "text": unit.text} for unit in units],
    }


def annotation_prompt(document: ChapterDocument, units: tuple[AnnotationSourceSlice, ...], fragment_index: int, fragment_count: int) -> str:
    fragment_note = "整章" if fragment_count == 1 else f"第 {fragment_index}/{fragment_count} 个连续片段"
    source_chars = sum(len(unit.text) for unit in units)
    minimum_events = max(1, math.ceil(source_chars / DENSE_EVENT_CHARS))
    maximum_events = min(6, minimum_events + 2)
    maximum_facts = min(16, max(8, minimum_events * 3 + 3))
    anchors = []
    for unit in units:
        quote = unit.text.strip().replace("\n", "")[:20]
        if len(quote) >= 2:
            anchors.append({"unit_id": unit.unit_id, "quote": quote})
        if len(anchors) == 3:
            break
    return f"""任务：从下面的小说正文中抽取可核验的叙事结构。这是{fragment_note}，不是续写、点评或改写。

只输出一个合法 JSON object；不得输出 Markdown、说明、前后缀或额外顶层字段。顶层字段必须且只能为：entities、facts、events、scenes、state_changes。所有 JSON 字符串中的双引号、反斜杠和换行必须按 JSON 标准转义；quote 不得含换行。

JSON 结构如下：
{json.dumps(ANNOTATION_DRAFT_SCHEMA, ensure_ascii=False, indent=2)}

具体格式示例（示例中的角色甲、村口、令牌只用于说明结构，绝不可复制到当前结果）：
{json.dumps(ANNOTATION_DRAFT_EXAMPLE, ensure_ascii=False, indent=2)}

强制原文锚点（均逐字来自本片段）：{json.dumps(anchors, ensure_ascii=False)}。输出不得为空；至少一个实体、一个事实和一个事件的 evidence 必须使用这些锚点中的原文短引文，或同一 source_units 中更精确的逐字短引文。

提交前逐项自检：不得输出空对象 `{{}}`；`entities`、`facts`、`events`、`scenes`、`state_changes` 五个数组都必须非空。若任一数组为空，说明你没有完成任务，必须从 source_units 补出带逐字证据的记录后再输出。不要用空 JSON 表示“无法判断”。

硬性规则：
1. 全部自然语言字段必须使用中文。id 使用本次输出内唯一的 e1、f1、v1、s1、c1 形式；引用只能使用本次输出中已声明的 id。
2. entity.kind 只能是：人物、组织、地点、物品、概念、生物。fact.kind 只能是：身份、目标、情绪、关系、地点、资源、认知、规则、进展、信息。
3. evidence 的 unit_id 必须逐字复制自给定 source_units；quote 必须是该单元 text 中连续、逐字一致、不跨行的原文短引文，长度控制在 2 到 32 字。没有原文证据就不要填写该实体、事实、事件或场景。
4. 每个事实必须有主体、谓词、取值或客体。每个事件必须有参与者、明确行动、结果事实；每个场景必须引用至少一个事件和一个离场事实。不要把同一章总述伪装成一个事件。
5. 按正文顺序提取事件和场景。事件 order 从 1 开始连续编号，场景 order 从 1 开始连续编号；tension 是 0 到 5 的整数。
6. 只记录文本明确支持的内容，不得猜测动机、境界、关系、时间、地点、因果或未发生的结果。确实没有的信息填空字符串或空数组；不要凭常识补写。
7. 本片段正文约 {source_chars} 字，绝不是空文本：entities、facts、events、scenes、state_changes 五个顶层数组都不得为空，必须输出至少 {minimum_events} 个可区分的因果事件。请按时间、地点、目标、资源、关系、认知或局势的变化拆分；不能只抽取结尾事件。每个片段输出 3 到 8 个实体、4 到 {maximum_facts} 个事实、{minimum_events} 到 {maximum_events} 个事件、1 到 3 个场景、1 到 4 项状态变化；若原文确实不足，只能在每个事件都有明确证据时减少。每个实体、事实、事件、场景最多给 2 条 evidence。

密度补充规则：每个事件只能覆盖一个可见的因果变化，禁止将连续动作、对话、得知信息和决定合并成一句章节总述。优先补足片段中尚无事件证据的连续区间；本片段至少需要 {minimum_events} 个事件。

待分析文本：
{json.dumps(_source_payload(document, units), ensure_ascii=False)}"""


def repair_prompt(document: ChapterDocument, units: tuple[AnnotationSourceSlice, ...], draft: dict[str, Any], errors: Iterable[str]) -> str:
    errors_text = "\n".join(f"- {message}" for message in errors)
    anchors = []
    for unit in units:
        quote = unit.text.strip().replace("\n", "")[:20]
        if len(quote) >= 2:
            anchors.append({"unit_id": unit.unit_id, "quote": quote})
        if len(anchors) == 3:
            break
    return f"""任务：修复下面的章节叙事标注 JSON。只能依据给定 source_units，不能加入原文未支持的内容。绝对禁止输出空对象 `{{}}`，也禁止让 entities、facts、events、scenes、state_changes 任一数组为空；必须补齐每项的逐字原文证据。

上一次结果未通过以下硬性质量检查：
{errors_text}

请只输出修复后的一个合法 JSON object。顶层字段必须且只能为 entities、facts、events、scenes、state_changes，结构必须符合：
{json.dumps(ANNOTATION_DRAFT_SCHEMA, ensure_ascii=False, indent=2)}

修复规则：每条 evidence 的 unit_id 必须存在，quote 必须逐字出现在该单元；所有引用 id 必须存在；每个事件必须有结果事实；每个事件必须被场景引用；给定片段不是空文本，entities、facts、events、scenes、state_changes 五个数组都不得为空；不要使用空证据填充数量。优先从这些逐字原文锚点建出非空结果：{json.dumps(anchors, ensure_ascii=False)}。

原始草稿：
{json.dumps(draft, ensure_ascii=False)}

可用原文：
{json.dumps(_source_payload(document, units), ensure_ascii=False)}"""


def _is_empty_annotation_draft(payload: dict[str, Any]) -> bool:
    return all(not payload.get(key) for key in ("entities", "facts", "events", "scenes", "state_changes"))


def nonempty_annotation_prompt(document: ChapterDocument, units: tuple[AnnotationSourceSlice, ...]) -> str:
    """A compact fallback for a model that returned ``{}`` to a valid fragment."""

    anchors = []
    for unit in units:
        quote = unit.text.strip().replace("\n", "")[:24]
        if len(quote) >= 2:
            anchors.append({"unit_id": unit.unit_id, "quote": quote})
        if len(anchors) == 3:
            break
    example = {
        "entities": [{"id": "e1", "name": "原文人物", "kind": "人物", "aliases": [], "evidence": [{"unit_id": "原文单元ID", "quote": "原文短引文"}]}],
        "facts": [{"id": "f1", "kind": "信息", "subject_id": "e1", "predicate": "发生", "value": "原文明确内容", "object_id": "", "certainty": "明确", "evidence": [{"unit_id": "原文单元ID", "quote": "原文短引文"}]}],
        "events": [{"id": "v1", "order": 1, "summary": "原文中的一个变化", "participant_ids": ["e1"], "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "原文动作", "obstacle": "", "decision": "", "outcome_fact_ids": ["f1"], "cost_fact_ids": [], "evidence": [{"unit_id": "原文单元ID", "quote": "原文短引文"}]}],
        "scenes": [{"id": "s1", "order": 1, "participant_ids": ["e1"], "event_ids": ["v1"], "objective": "当前变化", "entry_fact_ids": [], "exit_fact_ids": ["f1"], "tension": 1, "evidence": [{"unit_id": "原文单元ID", "quote": "原文短引文"}]}],
        "state_changes": [{"id": "c1", "event_id": "v1", "operation": "add", "before_fact_id": "", "after_fact_id": "f1"}],
    }
    return f"""你刚才错误地返回了空 JSON，但给定文本不是空文本。现在只做最小可用标注：输出 2 个由原文支持的原子事件及其事实和场景。

必须只输出一个合法 JSON object；不得返回 {{}}；entities、facts、events、scenes、state_changes 五个数组都必须非空。每条 evidence 的 unit_id 和 quote 必须逐字来自 source_units。事实 kind 只能为身份、目标、情绪、关系、地点、资源、认知、规则、进展、信息；quote 不得跨行。事件必须引用其结果事实，场景必须引用事件和离场事实。不得猜测或续写。

字段模板（只说明结构，所有内容必须替换成当前原文）：
{json.dumps(example, ensure_ascii=False)}

强制使用的逐字原文锚点：{json.dumps(anchors, ensure_ascii=False)}

source_units：
{json.dumps(_source_payload(document, units), ensure_ascii=False)}"""


_ATOM_TOP_LEVEL_KEYS = {"entities", "facts", "events"}
_SCENE_STATE_TOP_LEVEL_KEYS = {"scenes", "state_changes"}


def _fragment_minima(units: tuple[AnnotationSourceSlice, ...]) -> tuple[int, int]:
    chars = sum(len(unit.text) for unit in units)
    event_count = max(1, math.ceil(chars / DENSE_EVENT_CHARS))
    return event_count, max(2, event_count * 2)


def atomic_annotation_prompt(document: ChapterDocument, units: tuple[AnnotationSourceSlice, ...], fragment_index: int, fragment_count: int) -> str:
    """First pass: extract only evidence-backed entities, facts and event atoms."""

    minimum_events, minimum_facts = _fragment_minima(units)
    anchors = [
        {"unit_id": unit.unit_id, "quote": unit.text.strip().replace("\n", "")[:20]}
        for unit in units if len(unit.text.strip()) >= 2
    ][:3]
    example = {
        "entities": [{"id": "e1", "name": "人物甲", "kind": "人物", "aliases": [], "evidence": [{"unit_id": "章节:p0000", "quote": "人物甲"}]}],
        "facts": [{"id": "f1", "kind": "进展", "subject_id": "e1", "predicate": "得知", "value": "消息", "object_id": "", "certainty": "明确", "evidence": [{"unit_id": "章节:p0000", "quote": "得知消息"}]}],
        "events": [{"id": "v1", "order": 1, "summary": "人物甲得知消息", "participant_ids": ["e1"], "trigger_fact_ids": [], "precondition_fact_ids": [], "action": "得知消息", "obstacle": "", "decision": "", "outcome_fact_ids": ["f1"], "cost_fact_ids": [], "evidence": [{"unit_id": "章节:p0000", "quote": "得知消息"}]}],
    }
    return f"""你是中文小说原子事件抽取器。当前处理第 {fragment_index}/{fragment_count} 个连续原文片段，只抽取“实体、事实、原子事件”；不要输出场景、状态变化、续写、解释或 Markdown。

只输出一个合法 JSON object，顶层字段必须且只能是 `entities`、`facts`、`events`。绝对禁止输出空对象 `{{}}`，三个数组均不得为空。必须输出至少 {minimum_events} 个按原文顺序编号的原子事件，以及至少 {minimum_facts} 条有逐字证据的事实；一个事件只覆盖一个可见的因果变化，不能把连续行动、对话、得知和决定压缩成章节梗概。

JSON 示例（只示范字段结构，内容必须替换为当前原文）：
{json.dumps(example, ensure_ascii=False, indent=2)}

规则：
1. 所有自然语言字段使用中文；实体类型只能是人物、组织、地点、物品、概念、生物；事实类型只能是身份、目标、情绪、关系、地点、资源、认知、规则、进展、信息。
2. evidence 的 unit_id 必须逐字复制 source_units；quote 必须是对应 text 内 2 到 32 字的连续原文，不得跨行。事件、事实和实体都必须带 evidence。
3. 事件必须有参与者、行动和至少一个结果事实；所有 id 只能引用本次 JSON 已声明的 id。
4. 不得猜测动机、境界、关系、时间、地点或未发生结果。无法确认时继续寻找片段中其他明确变化，不能返回空数组。
5. 提交前自检：顶层仅三字段、三个数组非空、事件数不少于 {minimum_events}、事实数不少于 {minimum_facts}、每条 quote 能在 source_units 找到。

强制原文锚点：{json.dumps(anchors, ensure_ascii=False)}
source_units：
{json.dumps(_source_payload(document, units), ensure_ascii=False)}"""


def scene_state_prompt(document: ChapterDocument, units: tuple[AnnotationSourceSlice, ...], atom_draft: dict[str, Any]) -> str:
    """Second pass: derive scenes and state changes without altering atom data."""

    example = {
        "scenes": [{"id": "s1", "order": 1, "participant_ids": ["e1"], "event_ids": ["v1"], "objective": "应对当前变化", "entry_fact_ids": [], "exit_fact_ids": ["f1"], "tension": 2, "evidence": [{"unit_id": "章节:p0000", "quote": "原文短引文"}]}],
        "state_changes": [{"id": "c1", "event_id": "v1", "operation": "add", "before_fact_id": "", "after_fact_id": "f1"}],
    }
    return f"""你是中文小说场景与状态编排器。给定的实体、事实、原子事件已经通过原文证据抽取；不得修改、删除或新增其中任何对象，只需把已有事件编成可追踪场景，并为明确结果建立状态变化。

只输出一个合法 JSON object，顶层字段必须且只能是 `scenes`、`state_changes`；两个数组均不得为空，禁止输出 `{{}}`、解释或 Markdown。场景必须引用给定事件和离场事实；state_changes 必须引用给定事件和事实。所有场景 evidence 的 unit_id/quote 必须逐字来自 source_units。

JSON 示例：
{json.dumps(example, ensure_ascii=False, indent=2)}

已锁定的原子结构：
{json.dumps(atom_draft, ensure_ascii=False)}
source_units：
{json.dumps(_source_payload(document, units), ensure_ascii=False)}"""


def _require_atom_draft(payload: dict[str, Any], units: tuple[AnnotationSourceSlice, ...]) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != _ATOM_TOP_LEVEL_KEYS:
        raise AnnotationBuildError(["原子抽取 JSON 顶层字段必须且只能是 entities、facts、events"])
    if any(not isinstance(payload[key], list) for key in _ATOM_TOP_LEVEL_KEYS):
        raise AnnotationBuildError(["原子抽取的 entities、facts、events 必须是数组"])
    minimum_events, minimum_facts = _fragment_minima(units)
    issues: list[str] = []
    if not payload["entities"]:
        issues.append("原子抽取 entities 不得为空")
    if len(payload["facts"]) < minimum_facts:
        issues.append(f"原子抽取 facts 至少需要 {minimum_facts} 条")
    if len(payload["events"]) < minimum_events:
        issues.append(f"原子抽取 events 至少需要 {minimum_events} 条")
    if any(not isinstance(item, dict) for key in _ATOM_TOP_LEVEL_KEYS for item in payload[key]):
        issues.append("原子抽取数组元素必须是对象")
    if issues:
        raise AnnotationBuildError(issues)
    return payload


def _require_scene_state_draft(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != _SCENE_STATE_TOP_LEVEL_KEYS:
        raise AnnotationBuildError(["场景编排 JSON 顶层字段必须且只能是 scenes、state_changes"])
    if any(not isinstance(payload[key], list) or not payload[key] for key in _SCENE_STATE_TOP_LEVEL_KEYS):
        raise AnnotationBuildError(["场景编排的 scenes、state_changes 均不得为空"])
    if any(not isinstance(item, dict) for key in _SCENE_STATE_TOP_LEVEL_KEYS for item in payload[key]):
        raise AnnotationBuildError(["场景编排数组元素必须是对象"])
    return payload


def _derive_scene_state_from_atoms(atom_draft: dict[str, Any]) -> dict[str, Any]:
    """Build conservative scene/state scaffolding without another model call.

    Event atoms already carry chronological order, participants, outcomes and
    exact evidence.  Turning one atomic change into one local scene is safer
    than asking the model to restate them, and avoids a second long request
    becoming a single point of failure for every source fragment.
    """

    events = sorted(
        (item for item in atom_draft["events"] if isinstance(item, dict)),
        key=lambda item: (int(item.get("order", 0) or 0), str(item.get("id", ""))),
    )
    scenes: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    prior_exit_fact_ids: list[str] = []
    for index, event in enumerate(events, start=1):
        event_id = str(event.get("id", "")).strip()
        outcome_fact_ids = [str(item).strip() for item in event.get("outcome_fact_ids", []) if str(item).strip()]
        precondition_fact_ids = [str(item).strip() for item in event.get("precondition_fact_ids", []) if str(item).strip()]
        trigger_fact_ids = [str(item).strip() for item in event.get("trigger_fact_ids", []) if str(item).strip()]
        entry_fact_ids = precondition_fact_ids or trigger_fact_ids or prior_exit_fact_ids[:1]
        evidence = event.get("evidence", []) if isinstance(event.get("evidence"), list) else []
        action = str(event.get("action", "")).strip()
        summary = str(event.get("summary", "")).strip()
        scenes.append({
            "id": f"s{index}",
            "order": index,
            "participant_ids": list(event.get("participant_ids", [])),
            "event_ids": [event_id],
            "objective": action or summary,
            "entry_fact_ids": entry_fact_ids,
            "exit_fact_ids": outcome_fact_ids,
            "tension": min(5, 1 + int(bool(str(event.get("obstacle", "")).strip())) + int(bool(str(event.get("decision", "")).strip()))),
            "evidence": evidence,
        })
        if outcome_fact_ids:
            changes.append({
                "id": f"c{index}",
                "event_id": event_id,
                "operation": "add",
                "before_fact_id": "",
                "after_fact_id": outcome_fact_ids[0],
            })
        prior_exit_fact_ids = outcome_fact_ids
    return {"scenes": scenes, "state_changes": changes}


def _has_resolvable_evidence(row: dict[str, Any], document: ChapterDocument, path: str, units: tuple[AnnotationSourceSlice, ...]) -> bool:
    try:
        _spans(row.get("evidence"), document, path, units)
        return True
    except AnnotationBuildError:
        return False


def _reference_list(row: dict[str, Any], key: str) -> list[str] | None:
    value = row.get(key, [])
    if not isinstance(value, list):
        return None
    return [str(item).strip() for item in value if str(item).strip()]


def sanitize_draft(document: ChapterDocument, payload: dict[str, Any], units: tuple[AnnotationSourceSlice, ...]) -> dict[str, Any]:
    """Discard only evidence-invalid rows, then remove their dangling references.

    This is deliberately subtractive: it never invents a quote, a relation, or a
    replacement fact.  The usual validation and coverage checks still decide
    whether the remaining annotation is sufficient to accept.
    """

    if not isinstance(payload, dict):
        return payload
    clean = {key: list(value) if isinstance(value, list) else value for key, value in payload.items()}
    entities = [
        row for index, row in enumerate(clean.get("entities", []))
        if isinstance(row, dict) and str(row.get("id", "")).strip() and _has_resolvable_evidence(row, document, f"entities[{index}].evidence", units)
    ]
    entity_ids = {str(row["id"]).strip() for row in entities}
    facts = []
    for index, row in enumerate(clean.get("facts", [])):
        if not isinstance(row, dict) or not str(row.get("id", "")).strip():
            continue
        subject = str(row.get("subject_id", "")).strip()
        object_id = str(row.get("object_id", "")).strip()
        value = str(row.get("value", "")).strip()
        if subject in entity_ids and (not object_id or object_id in entity_ids) and (value or object_id) and _has_resolvable_evidence(row, document, f"facts[{index}].evidence", units):
            facts.append(row)
    fact_ids = {str(row["id"]).strip() for row in facts}
    events = []
    for index, row in enumerate(clean.get("events", [])):
        if not isinstance(row, dict) or not str(row.get("id", "")).strip():
            continue
        participants = _reference_list(row, "participant_ids")
        trigger = _reference_list(row, "trigger_fact_ids")
        preconditions = _reference_list(row, "precondition_fact_ids")
        outcomes = _reference_list(row, "outcome_fact_ids")
        costs = _reference_list(row, "cost_fact_ids")
        if (
            participants and outcomes and all(value in entity_ids for value in participants)
            and all(value in fact_ids for value in [*(trigger or []), *(preconditions or []), *outcomes, *(costs or [])])
            and _has_resolvable_evidence(row, document, f"events[{index}].evidence", units)
        ):
            events.append(row)
    event_ids = {str(row["id"]).strip() for row in events}
    scenes = []
    for index, row in enumerate(clean.get("scenes", [])):
        if not isinstance(row, dict) or not str(row.get("id", "")).strip():
            continue
        participants = _reference_list(row, "participant_ids")
        event_refs = _reference_list(row, "event_ids")
        entry = _reference_list(row, "entry_fact_ids")
        exits = _reference_list(row, "exit_fact_ids")
        try:
            tension = min(5, max(0, int(row.get("tension", 0))))
        except (TypeError, ValueError):
            continue
        if (
            participants and event_refs and exits and all(value in entity_ids for value in participants)
            and all(value in event_ids for value in event_refs)
            and all(value in fact_ids for value in [*(entry or []), *exits])
            and _has_resolvable_evidence(row, document, f"scenes[{index}].evidence", units)
        ):
            scenes.append({**row, "tension": tension})
    event_ids = {str(row["id"]).strip() for row in events}
    scenes = [row for row in scenes if all(str(value).strip() in event_ids for value in row.get("event_ids", []))]
    covered_events = {str(event_id).strip() for row in scenes for event_id in row.get("event_ids", [])}
    for event in events:
        event_id = str(event["id"]).strip()
        if event_id in covered_events:
            continue
        outcomes = _reference_list(event, "outcome_fact_ids") or []
        participants = _reference_list(event, "participant_ids") or []
        entry = [*(_reference_list(event, "trigger_fact_ids") or []), *(_reference_list(event, "precondition_fact_ids") or [])]
        scenes.append({
            "id": f"derived-scene-{len(scenes) + 1}", "order": len(scenes) + 1,
            "participant_ids": participants, "event_ids": [event_id],
            "objective": str(event.get("summary", "")).strip() or str(event.get("action", "")).strip(),
            "entry_fact_ids": entry, "exit_fact_ids": outcomes,
            "tension": 2 if str(event.get("obstacle", "")).strip() else 1,
            "evidence": event.get("evidence", []),
        })
    events = [{**row, "order": index + 1} for index, row in enumerate(events)]
    scenes = [{**row, "order": index + 1} for index, row in enumerate(scenes)]
    changes = []
    for row in clean.get("state_changes", []):
        if not isinstance(row, dict) or not str(row.get("id", "")).strip():
            continue
        event_id = str(row.get("event_id", "")).strip()
        operation = str(row.get("operation", "")).strip()
        before = str(row.get("before_fact_id", "")).strip()
        after = str(row.get("after_fact_id", "")).strip()
        valid_operation = (
            operation == "add" and after in fact_ids
            or operation == "remove" and before in fact_ids
            or operation == "update" and before in fact_ids and after in fact_ids
        )
        if event_id in event_ids and valid_operation:
            changes.append(row)
    return {"entities": entities, "facts": facts, "events": events, "scenes": scenes, "state_changes": changes}


def _validated_atom_draft(
    document: ChapterDocument,
    payload: dict[str, Any],
    units: tuple[AnnotationSourceSlice, ...],
) -> dict[str, Any]:
    """Reject an atom response if its quoted rows cannot survive local cleanup.

    Counting rows before evidence resolution was a recovery trap: a failed
    checkpoint could appear dense enough, then lose most rows in
    ``sanitize_draft`` and trigger an unnecessarily broad full-contract model
    repair.  Validate the exact atom payload that will be used downstream.
    """

    atoms = _require_atom_draft(payload, units)
    cleaned = sanitize_draft(document, {
        **atoms,
        "scenes": [],
        "state_changes": [],
    }, units)
    return _require_atom_draft({key: cleaned[key] for key in _ATOM_TOP_LEVEL_KEYS}, units)


def _is_transport_error(error: RuntimeError) -> bool:
    detail = str(error).lower()
    return any(token in detail for token in (
        "apiconnectionerror", "apitimeouterror", "connection error", "readtimeout", "request timed out", "ssl:",
        "模型返回空内容",
    ))


def _request_annotation_json(
    system: str,
    prompt: str,
    settings: ModelSettings,
    *,
    max_tokens: int = DEFAULT_ANNOTATION_OUTPUT_TOKENS,
    diagnostics: list[dict[str, Any]] | None = None,
    phase: str = "annotation",
) -> dict[str, Any]:
    """Issue a strict request with separate transport and JSON recovery paths."""

    for syntax_round in range(2):
        suffix = "\n输出前自行检查 JSON 括号闭合、字符串转义和顶层字段；不合格时也不要输出解释。"
        if syntax_round:
            suffix += "上一轮不是可解析 JSON；本轮优先输出闭合、可解析的 JSON object，不得截断或夹带解释。"
        last_transport_error: RuntimeError | None = None
        for transport_round in range(3):
            started = time.monotonic()
            try:
                # A compact source window never needs a chapter-scale completion.
                # This bound also discourages redundant evidence rows.
                payload = complete_json(
                    system + suffix,
                    prompt,
                    settings,
                    attempts=1,
                    max_tokens=min(settings.max_tokens, max_tokens),
                )
                if diagnostics is not None:
                    diagnostics.append({
                        "phase": phase,
                        "syntax_round": syntax_round + 1,
                        "transport_round": transport_round + 1,
                        "status": "returned",
                        "duration_ms": round((time.monotonic() - started) * 1000),
                        "top_level_keys": sorted(payload) if isinstance(payload, dict) else [],
                        "json_char_count": len(json.dumps(payload, ensure_ascii=False)) if isinstance(payload, dict) else 0,
                    })
                return payload
            except RuntimeError as error:
                if diagnostics is not None:
                    diagnostics.append({
                        "phase": phase,
                        "syntax_round": syntax_round + 1,
                        "transport_round": transport_round + 1,
                        "status": "error",
                        "duration_ms": round((time.monotonic() - started) * 1000),
                        "error_type": type(error).__name__,
                        "error": str(error)[:500],
                    })
                if not _is_transport_error(error):
                    if syntax_round:
                        raise
                    break
                last_transport_error = error
                if transport_round == 2:
                    raise
                delay = 2 ** transport_round
                print(f"[annotate] 模型连接失败，{delay} 秒后重试传输 {transport_round + 1}/2", flush=True)
                time.sleep(delay)
        if last_transport_error is not None:
            raise last_transport_error
    raise RuntimeError("unreachable")


def _annotate_chunk(
    document: ChapterDocument,
    units: tuple[AnnotationSourceSlice, ...],
    chunk_index: int,
    chunk_count: int,
    settings: ModelSettings,
    failure_path: Path | None = None,
    initial_draft: dict[str, Any] | None = None,
) -> ChapterAnnotation:
    diagnostics: list[dict[str, Any]] = [{
        "phase": "chunk",
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "source_char_count": sum(len(unit.text) for unit in units),
        "source_unit_count": len(units),
        "minimum_events": _fragment_minima(units)[0],
        "minimum_facts": _fragment_minima(units)[1],
    }]
    last_model_draft: dict[str, Any] = {}

    def diagnostics_path() -> Path | None:
        if failure_path is None:
            return None
        name = failure_path.name.removesuffix(".failed.json")
        return failure_path.with_name(f"{name}.diagnostics.json")

    def write_diagnostics(status: str, error: Exception | None = None) -> None:
        path = diagnostics_path()
        if path is None:
            return
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "chapter_id": document.chapter_id,
            "source_hash": document.source_hash,
            "chunk_index": chunk_index,
            "status": status,
            "attempts": diagnostics,
        }
        if error is not None:
            payload["error"] = f"{type(error).__name__}: {error}"
        write_json(path, payload)

    def write_failure(error: Exception | AnnotationBuildError) -> None:
        if failure_path is None:
            return
        messages = list(error.messages) if isinstance(error, AnnotationBuildError) else [f"{type(error).__name__}: {error}"]
        write_json(failure_path, {
            "chapter_id": document.chapter_id,
            "source_hash": document.source_hash,
            "chunk_index": chunk_index,
            "errors": messages,
            "last_model_draft": last_model_draft,
            "request_diagnostics": diagnostics,
        })
        write_diagnostics("failed", error)

    if initial_draft and set(initial_draft) == _ATOM_TOP_LEVEL_KEYS:
        try:
            atoms = _validated_atom_draft(document, initial_draft, units)
            last_model_draft = atoms
            scene_state = _derive_scene_state_from_atoms(atoms)
            draft = {**atoms, **scene_state}
            diagnostics.append({"phase": "recovered_atomic_draft", "status": "reused", "top_level_keys": sorted(atoms)})
            diagnostics.append({"phase": "deterministic_scene_state", "status": "derived", "scene_count": len(scene_state["scenes"]), "state_change_count": len(scene_state["state_changes"])})
        except AnnotationBuildError as atom_error:
            # Keep the original checkpoint for diagnostics, but ask only for a
            # compact atom re-extraction.  A full annotation repair is much
            # larger and is exactly the request most likely to time out here.
            last_model_draft = initial_draft
            diagnostics.append({"phase": "recovered_atomic_draft", "status": "contract_rejected", "errors": list(atom_error.messages)})
            try:
                atoms = _request_annotation_json(
                    "你是中文小说原子事件抽取修复器。旧检查点的证据或引用在本地校验中失效；只输出完整、非空的严格 JSON。",
                    atomic_annotation_prompt(document, units, chunk_index, chunk_count) + "\n旧检查点必须修复的问题：\n" + "\n".join(atom_error.messages),
                    settings,
                    max_tokens=DEFAULT_ANNOTATION_REPAIR_TOKENS,
                    diagnostics=diagnostics,
                    phase="recovered_atomic_repair",
                )
                atoms = _validated_atom_draft(document, atoms, units)
                last_model_draft = atoms
                scene_state = _derive_scene_state_from_atoms(atoms)
                draft = {**atoms, **scene_state}
                diagnostics.append({"phase": "deterministic_scene_state", "status": "derived", "scene_count": len(scene_state["scenes"]), "state_change_count": len(scene_state["state_changes"])})
            except (RuntimeError, AnnotationBuildError) as error:
                write_failure(error)
                raise
    elif initial_draft and not _is_empty_annotation_draft(initial_draft):
        draft = initial_draft
        last_model_draft = initial_draft
        diagnostics.append({"phase": "recovered_draft", "status": "reused", "top_level_keys": sorted(draft)})
    else:
        try:
            atoms = _request_annotation_json(
                "你是中文小说原子事件抽取器。只输出严格 JSON；片段绝非空文本，禁止返回 {} 或空数组。",
                atomic_annotation_prompt(document, units, chunk_index, chunk_count), settings,
                diagnostics=diagnostics, phase="atomic_events",
            )
            try:
                atoms = _validated_atom_draft(document, atoms, units)
            except AnnotationBuildError as atom_error:
                diagnostics.append({"phase": "atomic_events", "status": "contract_rejected", "errors": list(atom_error.messages)})
                atoms = _request_annotation_json(
                    "你是中文小说原子事件抽取修复器。上一轮数量或 JSON 契约不合格；只输出完整、非空的严格 JSON。",
                    atomic_annotation_prompt(document, units, chunk_index, chunk_count) + "\n上一轮必须修复的问题：\n" + "\n".join(atom_error.messages),
                    settings,
                    max_tokens=DEFAULT_ANNOTATION_REPAIR_TOKENS,
                    diagnostics=diagnostics,
                    phase="atomic_events_repair",
                )
                atoms = _validated_atom_draft(document, atoms, units)
            last_model_draft = atoms
            scene_state = _derive_scene_state_from_atoms(atoms)
            scene_state = _require_scene_state_draft(scene_state)
            diagnostics.append({
                "phase": "deterministic_scene_state",
                "status": "derived",
                "scene_count": len(scene_state["scenes"]),
                "state_change_count": len(scene_state["state_changes"]),
            })
            draft = {**atoms, **scene_state}
        except (RuntimeError, AnnotationBuildError) as error:
            write_failure(error)
            raise

    last_error: AnnotationBuildError | None = None
    for repair_round in range(2):
        draft = sanitize_draft(document, draft, units)
        try:
            annotation = annotation_from_draft(document, draft, f"{document.chapter_id}:chunk-{chunk_index:02d}", units)
            annotation = order_annotation_by_evidence(enrich_entity_evidence(document, annotation))
            _assert_acceptable(document, annotation, units)
            write_diagnostics("accepted")
            return annotation
        except AnnotationBuildError as error:
            last_error = error
            if repair_round == 1:
                break
            try:
                draft = _request_annotation_json(
                    "你是小说叙事标注修复器。只输出严格 JSON；修复引用、证据和因果结构，不得补写原文没有的事实。",
                    repair_prompt(document, units, draft, error.messages),
                    settings,
                    max_tokens=DEFAULT_ANNOTATION_REPAIR_TOKENS,
                    diagnostics=diagnostics,
                    phase="full_contract_repair",
                )
            except RuntimeError as request_error:
                write_failure(request_error)
                raise
            last_model_draft = draft
    if last_error is not None:
        write_failure(last_error)
    raise last_error or AnnotationBuildError(["模型标注修复失败"])


def _min_evidence_start(item: Any) -> int:
    evidence = getattr(item, "evidence", ())
    return min((span.start for span in evidence), default=10**12)


def _merge_spans(items: Iterable[Any]) -> tuple[SourceSpan, ...]:
    return _dedupe_spans(span for item in items for span in getattr(item, "evidence", ()))


def merge_annotations(document: ChapterDocument, annotations: list[ChapterAnnotation]) -> ChapterAnnotation:
    """Merge overlapping fragment annotations and deterministically remap all IDs."""

    if len(annotations) == 1:
        return annotations[0]
    entity_rows: list[Entity] = []
    entity_map: dict[str, str] = {}
    by_entity_key: dict[tuple[str, str], Entity] = {}
    for annotation in annotations:
        for entity in annotation.entities:
            key = (entity.kind, entity.canonical_name)
            current = by_entity_key.get(key)
            if current is None:
                new = Entity(f"{document.chapter_id}:entity-{len(entity_rows) + 1:03d}", entity.canonical_name, entity.kind, entity.aliases, entity.evidence)
                by_entity_key[key] = new
                entity_rows.append(new)
                current = new
            else:
                aliases = tuple(dict.fromkeys((*current.aliases, *entity.aliases)))
                replacement = Entity(current.entity_id, current.canonical_name, current.kind, aliases, _dedupe_spans((*current.evidence, *entity.evidence)))
                entity_rows[entity_rows.index(current)] = replacement
                by_entity_key[key] = replacement
                current = replacement
            entity_map[entity.entity_id] = current.entity_id

    fact_rows: list[Fact] = []
    fact_map: dict[str, str] = {}
    by_fact_key: dict[tuple[str, str, str, str, str], Fact] = {}
    for annotation in annotations:
        for fact in annotation.facts:
            subject = entity_map[fact.subject_id]
            object_id = entity_map.get(fact.object_id, "")
            key = (fact.kind, subject, fact.predicate, fact.value, object_id)
            current = by_fact_key.get(key)
            if current is None:
                new = Fact(f"{document.chapter_id}:fact-{len(fact_rows) + 1:03d}", document.chapter_id, fact.kind, subject, fact.predicate, fact.value, object_id, fact.certainty, fact.evidence)
                by_fact_key[key] = new
                fact_rows.append(new)
                current = new
            else:
                replacement = Fact(current.fact_id, current.chapter_id, current.kind, current.subject_id, current.predicate, current.value, current.object_id, max(current.certainty, fact.certainty), _dedupe_spans((*current.evidence, *fact.evidence)))
                fact_rows[fact_rows.index(current)] = replacement
                by_fact_key[key] = replacement
                current = replacement
            fact_map[fact.fact_id] = current.fact_id

    original_events = sorted((event for annotation in annotations for event in annotation.events), key=lambda event: (event.order, _min_evidence_start(event)))
    event_rows: list[EventAtom] = []
    event_map: dict[str, str] = {}
    event_keys: dict[tuple[str, tuple[str, ...], tuple[str, ...]], EventAtom] = {}
    for event in original_events:
        participants = tuple(entity_map[item] for item in event.participant_ids)
        outcomes = tuple(fact_map[item] for item in event.outcome_fact_ids)
        key = (event.summary, participants, outcomes)
        current = event_keys.get(key)
        if current is None:
            new = EventAtom(
                f"{document.chapter_id}:event-{len(event_rows) + 1:03d}", document.chapter_id, len(event_rows), event.summary, participants,
                tuple(fact_map[item] for item in event.trigger_fact_ids), tuple(fact_map[item] for item in event.precondition_fact_ids),
                event.action, event.obstacle, event.decision, outcomes, tuple(fact_map[item] for item in event.cost_fact_ids), event.evidence,
            )
            event_keys[key] = new
            event_rows.append(new)
            current = new
        else:
            replacement = EventAtom(current.event_id, current.chapter_id, current.order, current.summary, current.participant_ids, current.trigger_fact_ids, current.precondition_fact_ids, current.action, current.obstacle, current.decision, current.outcome_fact_ids, current.cost_fact_ids, _dedupe_spans((*current.evidence, *event.evidence)))
            event_rows[event_rows.index(current)] = replacement
            event_keys[key] = replacement
            current = replacement
        event_map[event.event_id] = current.event_id

    original_scenes = sorted((scene for annotation in annotations for scene in annotation.scenes), key=lambda scene: (scene.order, _min_evidence_start(scene)))
    scene_rows: list[SceneCard] = []
    scene_keys: dict[tuple[tuple[str, ...], tuple[str, ...], str], SceneCard] = {}
    for scene in original_scenes:
        participants = tuple(entity_map[item] for item in scene.participant_ids)
        events = tuple(event_map[item] for item in scene.event_ids)
        exits = tuple(fact_map[item] for item in scene.exit_fact_ids)
        key = (events, exits, scene.objective)
        current = scene_keys.get(key)
        if current is None:
            new = SceneCard(
                f"{document.chapter_id}:scene-{len(scene_rows) + 1:03d}", document.chapter_id, len(scene_rows), participants, events,
                scene.objective, tuple(fact_map[item] for item in scene.entry_fact_ids), exits, scene.tension, scene.evidence,
            )
            scene_rows.append(new)
            scene_keys[key] = new
        else:
            replacement = SceneCard(current.scene_id, current.chapter_id, current.order, current.participant_ids, current.event_ids, current.objective, current.entry_fact_ids, current.exit_fact_ids, current.tension, _dedupe_spans((*current.evidence, *scene.evidence)))
            scene_rows[scene_rows.index(current)] = replacement
            scene_keys[key] = replacement

    changes: list[StateChange] = []
    seen_changes: set[tuple[str, str, str, str]] = set()
    for annotation in annotations:
        for change in annotation.state_changes:
            event_id = event_map.get(change.event_id)
            before = fact_map.get(change.before_fact_id, "")
            after = fact_map.get(change.after_fact_id, "")
            if not event_id or (change.operation == "add" and not after) or (change.operation == "remove" and not before) or (change.operation == "update" and (not before or not after)):
                continue
            key = (event_id, change.operation, before, after)
            if key not in seen_changes:
                seen_changes.add(key)
                changes.append(StateChange(f"{document.chapter_id}:change-{len(changes) + 1:03d}", document.chapter_id, event_id, change.operation, before, after))
    return ChapterAnnotation(document.chapter_id, document.source_hash, tuple(entity_rows), tuple(fact_rows), tuple(event_rows), tuple(scene_rows), tuple(changes))


def _chunk_checkpoint_path(
    checkpoint_dir: Path,
    document: ChapterDocument,
    chunk: tuple[AnnotationSourceSlice, ...],
    chunk_index: int,
) -> Path:
    fingerprint = "|".join(
        (document.chapter_id, document.source_hash, str(chunk_index), *(f"{item.start}:{item.end}" for item in chunk))
    )
    return checkpoint_dir / f"{hashlib.sha256(fingerprint.encode('utf-8')).hexdigest()[:20]}.chunk.json"


def _load_chunk_checkpoint(
    path: Path,
    document: ChapterDocument,
    chunk: tuple[AnnotationSourceSlice, ...],
) -> ChapterAnnotation | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("source_hash") != document.source_hash:
            return None
        raw_annotation = payload.get("annotation")
        if not isinstance(raw_annotation, dict):
            return None
        annotation = ChapterAnnotation.from_dict(raw_annotation)
        if _validation_messages(document, annotation, chunk):
            return None
        if not assess_annotation_quality(document, annotation, chunk).passed:
            return None
        return annotation
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _load_chunk_failure_draft(path: Path, document: ChapterDocument) -> dict[str, Any] | None:
    """Reuse a model response that failed only a now-fixed local contract.

    The file is keyed by the source-hash-derived checkpoint path.  We still
    verify the explicit source hash when it is available, so a stale diagnostic
    cannot be applied to changed source text.
    """

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("chapter_id") != document.chapter_id:
            return None
        source_hash = str(payload.get("source_hash", ""))
        if source_hash and source_hash != document.source_hash:
            return None
        draft = payload.get("last_model_draft")
        return draft if isinstance(draft, dict) else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _chapter_quality_repair_targets(
    document: ChapterDocument,
    chunks: list[tuple[AnnotationSourceSlice, ...]],
    annotations: list[ChapterAnnotation],
    quality: AnnotationQualityReport,
) -> tuple[int, ...]:
    """Choose a small set of source-local chunks for one targeted re-extraction."""

    error_codes = {item.code for item in quality.issues if item.severity == "error"}
    if not error_codes:
        return ()
    selected: set[int] = set()
    if "event_coverage_gap" in error_codes:
        eligible = document.annotation_units
        start = eligible[0].start if eligible else 0
        end = eligible[-1].end if eligible else 0
        points = sorted({start, end, *(span.start for event in annotations for span in event.evidence), *(span.end for event in annotations for span in event.evidence)})
        if len(points) >= 2:
            gap_start, gap_end = max(zip(points, points[1:]), key=lambda pair: pair[1] - pair[0])
            selected.update(
                index for index, chunk in enumerate(chunks, start=1)
                if chunk and chunk[-1].end >= gap_start and chunk[0].start <= gap_end
            )
    density_codes = {"event_coverage", "fact_density", "scene_coverage", "evidence_coverage"}
    if error_codes & density_codes:
        deficits: list[tuple[float, int]] = []
        for index, (chunk, annotation) in enumerate(zip(chunks, annotations), start=1):
            minimum_events, minimum_facts = _fragment_minima(chunk)
            score = min(
                len(annotation.events) / max(minimum_events, 1),
                len(annotation.facts) / max(minimum_facts, 1),
                len(annotation.scenes),
            )
            deficits.append((score, index))
        selected.update(index for _, index in sorted(deficits)[:2])
    return tuple(sorted(selected))[:2]


def annotate_document(
    document: ChapterDocument,
    max_input_chars: int = DEFAULT_ANNOTATION_INPUT_CHARS,
    overlap_units: int = DEFAULT_ANNOTATION_OVERLAP_UNITS,
    settings: ModelSettings | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[ChapterAnnotation, AnnotationQualityReport]:
    document_report = validate_document(document)
    if not document_report.passed:
        raise AnnotationBuildError(f"source document invalid: {issue.message}" for issue in document_report.issues)
    chunks = split_annotation_units(document, max_input_chars, overlap_units)
    if not chunks:
        raise AnnotationBuildError(["章节没有可标注的正文单元"])
    active_settings = settings or ModelSettings.from_environment()
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    annotations = []
    for index, chunk in enumerate(chunks, start=1):
        checkpoint_path = _chunk_checkpoint_path(checkpoint_dir, document, chunk, index) if checkpoint_dir is not None else None
        cached = _load_chunk_checkpoint(checkpoint_path, document, chunk) if checkpoint_path is not None else None
        if cached is not None:
            print(f"[annotate] {document.chapter_id}: 已复用分片 {index}/{len(chunks)}", flush=True)
            annotations.append(cached)
            continue
        print(f"[annotate] {document.chapter_id}: 正在提取分片 {index}/{len(chunks)}", flush=True)
        failure_path = checkpoint_path.with_name(f"{checkpoint_path.stem}.failed.json") if checkpoint_path is not None else None
        recovered_draft = _load_chunk_failure_draft(failure_path, document) if failure_path is not None else None
        if recovered_draft is not None:
            print(f"[annotate] {document.chapter_id}: 正在复验先前失败分片 {index}/{len(chunks)}", flush=True)
        annotation = _annotate_chunk(document, chunk, index, len(chunks), active_settings, failure_path, recovered_draft)
        annotations.append(annotation)
        if checkpoint_path is not None:
            write_json(checkpoint_path, {
                "schema_version": "1.0",
                "source_hash": document.source_hash,
                "chapter_id": document.chapter_id,
                "chunk_index": index,
                "annotation": annotation.to_dict(),
            })
    annotation = enrich_spatiotemporal(document, order_annotation_by_evidence(merge_annotations(document, annotations)))
    try:
        return annotation, _assert_acceptable(document, annotation)
    except AnnotationBuildError:
        quality = assess_annotation_quality(document, annotation)
        targets = _chapter_quality_repair_targets(document, chunks, annotations, quality)
        if checkpoint_dir is None or not targets:
            raise
        for repair_round in range(MAX_CHAPTER_QUALITY_REPAIRS):
            print(f"[annotate] {document.chapter_id}: 章节质量未通过，定点补标分片 {list(targets)}（第 {repair_round + 1}/{MAX_CHAPTER_QUALITY_REPAIRS} 轮）", flush=True)
            for index in targets:
                chunk = chunks[index - 1]
                checkpoint_path = _chunk_checkpoint_path(checkpoint_dir, document, chunk, index)
                failure_path = checkpoint_path.with_name(f"{checkpoint_path.stem}.failed.json")
                replacement = _annotate_chunk(document, chunk, index, len(chunks), active_settings, failure_path)
                annotations[index - 1] = replacement
                write_json(checkpoint_path, {
                    "schema_version": "1.1",
                    "source_hash": document.source_hash,
                    "chapter_id": document.chapter_id,
                    "chunk_index": index,
                    "annotation": replacement.to_dict(),
                    "quality_repair_round": repair_round + 1,
                })
            annotation = enrich_spatiotemporal(document, order_annotation_by_evidence(merge_annotations(document, annotations)))
            try:
                return annotation, _assert_acceptable(document, annotation)
            except AnnotationBuildError:
                quality = assess_annotation_quality(document, annotation)
                targets = _chapter_quality_repair_targets(document, chunks, annotations, quality)
                if not targets:
                    raise
        raise AnnotationBuildError(["章节定点补标后仍未满足高密度质量门槛"])


def _documents_for_work(author_id: str, work_id: str) -> list[ChapterDocument]:
    folder = corpus_dir(author_id, work_id)
    records = read_jsonl(folder / "chapters.jsonl")
    stored = {str(row.get("chapter_id", "")): row for row in read_jsonl(folder / "chapter_documents.jsonl")}
    documents: list[ChapterDocument] = []
    for row in records:
        chapter_id = str(row.get("chapter_id", ""))
        payload = stored.get(chapter_id)
        if payload:
            document = ChapterDocument.from_dict(payload)
        else:
            document = build_chapter_document(chapter_id, str(row.get("source_hash", "")), str(row.get("text", "")), str(row.get("title", "")))
        report = validate_document(document)
        if not report.passed:
            raise AnnotationBuildError(f"{chapter_id}: {issue.message}" for issue in report.issues)
        documents.append(document)
    return documents


def _quality_file_payload(author_id: str, work_id: str, chapter_limit: int, reports: list[dict[str, Any]]) -> dict[str, Any]:
    warning_counts: dict[str, int] = {}
    error_count = 0
    for report in reports:
        for issue in report.get("issues", []) if isinstance(report.get("issues", []), list) else []:
            if not isinstance(issue, dict):
                continue
            if issue.get("severity") == "error":
                error_count += 1
            elif issue.get("severity") == "warning":
                code = str(issue.get("code", "unknown_warning"))
                warning_counts[code] = warning_counts.get(code, 0) + 1
    averages = {
        key: round(sum(float(report.get(key, 0) or 0) for report in reports) / len(reports), 2)
        for key in (
            "event_count", "scene_count", "time_anchor_count", "temporal_relation_count",
            "spatial_relation_count", "evidence_unit_count", "eligible_unit_count",
        )
    } if reports else {}
    return {
        "author_id": author_id,
        "work_id": work_id,
        "chapter_limit": chapter_limit,
        "accepted_count": len(reports),
        "all_hard_checks_passed": len(reports) == chapter_limit and error_count == 0,
        "summary": {"hard_issue_count": error_count, "warning_counts": warning_counts, "averages": averages},
        "reports": reports,
    }


def annotate_work(author_id: str, work_id: str, *, limit: int = DEFAULT_ANNOTATION_LIMIT, max_input_chars: int = DEFAULT_ANNOTATION_INPUT_CHARS, overlap_units: int = DEFAULT_ANNOTATION_OVERLAP_UNITS, resume: bool = True) -> tuple[Path, Path]:
    """Annotate a bounded chapter sample and write only accepted annotations."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    folder = corpus_dir(author_id, work_id)
    documents = _documents_for_work(author_id, work_id)[:limit]
    if not documents:
        raise FileNotFoundError(f"no chapters found for {author_id}/{work_id}")
    target = folder / f"chapter_annotations.sample-{len(documents)}.jsonl"
    quality_target = folder / f"chapter_annotations.sample-{len(documents)}.quality.json"
    completed = {str(row.get("chapter_id", "")): row for row in read_jsonl(target)} if resume else {}
    if resume:
        # A smaller accepted sample is a valid cache for a later, larger sample.
        # Each row is revalidated against the current contracts before reuse.
        for prior_target in sorted(folder.glob("chapter_annotations.sample-*.jsonl")):
            if prior_target == target:
                continue
            for row in read_jsonl(prior_target):
                chapter_id = str(row.get("chapter_id", ""))
                if chapter_id and chapter_id not in completed:
                    completed[chapter_id] = row
    accepted: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    settings = ModelSettings.from_environment()
    if not settings.api_key:
        raise RuntimeError("模型密钥不可用：请在 config.yaml 的 deepseek.api_key 或环境变量中配置后重试")
    for index, document in enumerate(documents, start=1):
        existing = completed.get(document.chapter_id)
        if existing:
            annotation = enrich_spatiotemporal(document, order_annotation_by_evidence(enrich_entity_evidence(document, ChapterAnnotation.from_dict(existing))))
            messages = _validation_messages(document, annotation)
            if not messages:
                quality = assess_annotation_quality(document, annotation)
                accepted.append(annotation.to_dict())
                reports.append({"chapter_id": document.chapter_id, "status": "reused", **quality.to_dict()})
                print(f"[annotate] {work_id}: {index}/{len(documents)} 已复用合格章节", flush=True)
                continue
        annotation, quality = annotate_document(
            document,
            max_input_chars,
            overlap_units,
            settings,
            checkpoint_dir=folder / "annotation_checkpoints",
        )
        accepted.append(annotation.to_dict())
        reports.append({"chapter_id": document.chapter_id, "status": "accepted", **quality.to_dict()})
        write_jsonl(target, accepted)
        write_json(quality_target, _quality_file_payload(author_id, work_id, len(documents), reports))
        print(f"[annotate] {work_id}: {index}/{len(documents)} 已通过证据与覆盖率校验", flush=True)
    write_jsonl(target, accepted)
    write_json(quality_target, _quality_file_payload(author_id, work_id, len(documents), reports))
    return target, quality_target
