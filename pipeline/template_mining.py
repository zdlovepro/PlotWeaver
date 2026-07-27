"""Stage 5: batch-safe mining of reusable structural narrative templates."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

from .contracts import ChapterAnnotation, NarrativeTemplate, NarrativeTemplateLibrary, WorkContinuity
from .jsonio import read_jsonl, write_json
from .model import ModelSettings, complete_json
from .paths import corpus_dir


DEFAULT_TEMPLATE_LIMIT = 20
DEFAULT_TEMPLATE_BATCH_CHAPTERS = 5
MACRO_WINDOW_CHAPTERS = 5
_LEVELS = ("micro", "event", "scene", "chapter", "macro")
# A macro template must recur across more than one five-chapter phase.  Five
# chapters describe one arc instance; ten or more are the minimum evidence for
# treating that shape as reusable author-level structure.
_MIN_SUPPORT = {"micro": 2, "event": 2, "scene": 2, "chapter": 2, "macro": 10}


@dataclass(frozen=True)
class TemplateQualityIssue:
    code: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class TemplateQualityReport:
    author_id: str
    work_id: str
    chapter_count: int
    templates_by_level: dict[str, int]
    evidence_chapter_coverage: int
    issues: tuple[TemplateQualityIssue, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "author_id": self.author_id,
            "work_id": self.work_id,
            "chapter_count": self.chapter_count,
            "templates_by_level": self.templates_by_level,
            "evidence_chapter_coverage": self.evidence_chapter_coverage,
            "passed": self.passed,
            "issues": [item.to_dict() for item in self.issues],
        }


def _chunks(values: list[Any], size: int) -> list[list[Any]]:
    size = max(2, int(size))
    return [values[index:index + size] for index in range(0, len(values), size)]


def _event_fact_ids(event: Any) -> set[str]:
    return {
        *event.trigger_fact_ids,
        *event.precondition_fact_ids,
        *event.outcome_fact_ids,
        *event.cost_fact_ids,
    }


def _event_frame(annotation: ChapterAnnotation, event: Any) -> dict[str, Any]:
    facts = {fact.fact_id: fact for fact in annotation.facts}
    entities = {entity.entity_id: entity for entity in annotation.entities}
    changes = [change.operation for change in annotation.state_changes if change.event_id == event.event_id]
    scenes = [scene for scene in annotation.scenes if event.event_id in scene.event_ids]
    relation_ids = {
        relation.relation_id for relation in annotation.spatial_relations
        if relation.fact_id in _event_fact_ids(event)
    }
    return {
        "participants": sorted(Counter(entities[item].kind for item in event.participant_ids).items()),
        "trigger_fact_kinds": sorted({facts[item].kind for item in event.trigger_fact_ids}),
        "precondition_fact_kinds": sorted({facts[item].kind for item in event.precondition_fact_ids}),
        "outcome_fact_kinds": sorted({facts[item].kind for item in event.outcome_fact_ids}),
        "cost_fact_kinds": sorted({facts[item].kind for item in event.cost_fact_ids}),
        "has_obstacle": bool(event.obstacle),
        "has_decision": bool(event.decision),
        "state_operations": sorted(changes),
        "has_spatial_change": bool(relation_ids),
        "scene_tensions": sorted(scene.tension for scene in scenes),
    }


def chapter_feature_cards(annotations: list[ChapterAnnotation], continuity: WorkContinuity) -> list[dict[str, Any]]:
    """Build author-neutral, name-free cards suitable for model batching."""

    transitions_by_chapter: dict[str, list[Any]] = defaultdict(list)
    for transition in continuity.location_transitions:
        transitions_by_chapter[transition.chapter_id].append(transition)
    ledger_by_chapter: dict[str, list[Any]] = defaultdict(list)
    for entry in continuity.state_ledger:
        ledger_by_chapter[entry.chapter_id].append(entry)
    cards: list[dict[str, Any]] = []
    for annotation in annotations:
        event_frames = [_event_frame(annotation, event) for event in sorted(annotation.events, key=lambda item: item.order)]
        tension_curve = [scene.tension for scene in sorted(annotation.scenes, key=lambda item: item.order)]
        state_operations = Counter(entry.operation for entry in ledger_by_chapter[annotation.chapter_id])
        cards.append({
            "chapter_id": annotation.chapter_id,
            "event_frames": event_frames,
            "scene_frames": [{
                "event_count": len(scene.event_ids),
                "tension": scene.tension,
                "has_location": bool(scene.location_ids),
                "has_time_anchor": bool(scene.time_anchor_ids),
                "has_entry_state": bool(scene.entry_fact_ids),
                "has_exit_state": bool(scene.exit_fact_ids),
            } for scene in sorted(annotation.scenes, key=lambda item: item.order)],
            "chapter_metrics": {
                "event_count": len(annotation.events),
                "scene_count": len(annotation.scenes),
                "tension_curve": tension_curve,
                "time_anchor_count": len(annotation.time_anchors),
                "spatial_relation_kinds": sorted(Counter(item.kind for item in annotation.spatial_relations).items()),
                "state_operations": sorted(state_operations.items()),
                "location_transition_count": len(transitions_by_chapter[annotation.chapter_id]),
                "has_chapter_end_event": bool(annotation.events),
            },
        })
    return cards


def _schema(levels: Iterable[str]) -> dict[str, Any]:
    return {
        f"{level}_templates": [{
            "purpose": "",
            "role_slots": ["行动者"],
            "beat_sequence": ["起始", "推进", "结果"],
            "state_effects": ["至少一项状态变化"],
            "selection_tags": ["推进"],
            "variation_axes": ["阻力来源"],
            "evidence_chapter_ids": ["work/0001", "work/0002"],
        }]
        for level in levels
    }


def _template_prompt(cards: list[dict[str, Any]], levels: tuple[str, ...], minimum_support: int | None = None) -> str:
    schema = _schema(levels)
    ids = [card["chapter_id"] for card in cards]
    example = {
        "event_templates": [{
            "purpose": "以目标受阻迫使行动者调整策略，并形成可追踪结果",
            "role_slots": ["行动者", "阻力来源"],
            "beat_sequence": ["目标出现", "采取行动", "阻力加压", "作出选择", "状态改变"],
            "state_effects": ["目标、资源、关系、认知或位置至少一项改变"],
            "selection_tags": ["目标推进", "压力", "选择"],
            "variation_axes": ["阻力类别", "付出代价", "破局方式"],
            "evidence_chapter_ids": ids[:2],
        }],
    }
    support = minimum_support if minimum_support is not None else max(_MIN_SUPPORT[level] for level in levels)
    return f"""任务：根据下列无专名的章节结构特征，归纳可复用的叙事模板。你不是续写，也不能复述任何来源作品。

只输出一个合法 JSON object；顶层字段必须且只能是：{'、'.join(schema)}。
所有自然语言字段必须使用中文。模板只能描述抽象的叙事功能、角色槽位、节拍、状态效果与可替换变量；不得出现任何角色名、地名、物品名、门派名、具体设定、原文句子或唯一情节。

输出结构：
{json.dumps(schema, ensure_ascii=False, indent=2)}

具体 JSON 示例（仅说明字段格式，不可照抄为结论）：
{json.dumps(example, ensure_ascii=False, indent=2)}

硬性规则：
1. `evidence_chapter_ids` 只能从 {json.dumps(ids, ensure_ascii=False)} 中选择，且不得重复。
2. 每个模板必须由至少 {support} 个不同章节支持；证据不足不要输出。
3. 每个模板的 `beat_sequence` 至少 3 项，`selection_tags` 至少 1 项。
4. 只归纳跨章节重复或可由多章结构共同支持的模式；不要把单章摘要伪装成模板。
5. `macro_templates` 必须描述连续多章的进入状态、压力升级、转折、阶段回报和退出状态；最终宏观模板至少引用 10 章，以证明它跨越至少两个阶段而非单一剧情段。

待归纳的章节特征：
{json.dumps(cards, ensure_ascii=False)}"""


def _offline_draft(cards: list[dict[str, Any]], levels: tuple[str, ...]) -> dict[str, Any]:
    ids = [card["chapter_id"] for card in cards]
    common = {
        "role_slots": ["行动者", "压力来源"],
        "beat_sequence": ["建立目标", "采取行动", "压力变化", "形成结果"],
        "state_effects": ["资源、认知、关系、位置或进展至少一项变化"],
        "selection_tags": ["目标推进", "状态变化"],
        "variation_axes": ["压力来源", "行动代价", "结果类型"],
        "evidence_chapter_ids": ids,
    }
    result: dict[str, Any] = {}
    if "micro" in levels:
        result["micro_templates"] = [{**common, "purpose": "以局部压力推动角色采取行动并产生可见后果"}]
    if "event" in levels:
        result["event_templates"] = [{**common, "purpose": "围绕目标、行动、阻力与结果构成可执行的因果事件"}]
    if "scene" in levels:
        result["scene_templates"] = [{**common, "purpose": "以进入状态、行动事件和离开状态完成一段场景推进"}]
    if "chapter" in levels:
        result["chapter_templates"] = [{**common, "purpose": "通过多场景递进完成阶段推进并保留后续压力"}]
    if "macro" in levels:
        result["macro_templates"] = [{
            **common,
            "purpose": "以连续章节完成目标建立、压力升级、阶段转折、回报结算与新状态进入",
            "beat_sequence": ["建立阶段目标", "压力逐步升级", "形成关键转折", "结算阶段回报或代价", "带着新状态进入下一阶段"],
            "selection_tags": ["阶段推进", "压力升级", "阶段转折"],
        }]
    return result


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _templates_from_draft(
    draft: dict[str, Any],
    levels: tuple[str, ...],
    allowed_chapter_ids: set[str],
    prefix: str,
) -> tuple[NarrativeTemplate, ...]:
    templates: list[NarrativeTemplate] = []
    for level in levels:
        rows = draft.get(f"{level}_templates", []) if isinstance(draft, dict) else []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            evidence = tuple(item for item in _strings(row.get("evidence_chapter_ids")) if item in allowed_chapter_ids)
            if len(evidence) < _MIN_SUPPORT[level]:
                continue
            purpose = str(row.get("purpose", "")).strip()
            beats = _strings(row.get("beat_sequence"))
            tags = _strings(row.get("selection_tags"))
            if not purpose or len(beats) < 3 or not tags:
                continue
            confidence = min(0.95, 0.55 + len(evidence) * 0.04)
            templates.append(NarrativeTemplate(
                f"{prefix}:{level}-{len(templates) + 1:03d}",
                level,
                purpose,
                _strings(row.get("role_slots")),
                beats,
                _strings(row.get("state_effects")),
                tags,
                _strings(row.get("variation_axes")),
                evidence,
                len(evidence),
                confidence,
            ))
    return tuple(templates)


def _observations(
    cards: list[dict[str, Any]],
    settings: ModelSettings | None,
    offline: bool,
    batch_size: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, batch in enumerate(_chunks(cards, batch_size), start=1):
        levels = ("micro", "event", "scene", "chapter")
        payload = _offline_draft(batch, levels) if offline else complete_json(
            "你是小说叙事结构模板提炼器。所有结论必须保持抽象、可复用、无来源专名，并严格遵守用户给出的 JSON 契约。",
            _template_prompt(batch, levels),
            settings or ModelSettings.from_environment(),
            attempts=2,
            max_tokens=min((settings or ModelSettings.from_environment()).max_tokens, 12_000),
        )
        result.append({"batch_index": index, "chapter_ids": [card["chapter_id"] for card in batch], "draft": payload})
    return result


def _synthesis_prompt(cards: list[dict[str, Any]], observations: list[dict[str, Any]], levels: tuple[str, ...]) -> str:
    # The final call sees compact structural cards plus prior candidates, not
    # source prose.  This both fits context limits and prevents source phrasing
    # from leaking into a reusable skill.
    return _template_prompt(cards, levels) + "\n\n分批候选（需要跨批去重、合并并重新选择证据章节）：\n" + json.dumps(observations, ensure_ascii=False)


def _synthesise(
    cards: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    settings: ModelSettings | None,
    offline: bool,
) -> tuple[NarrativeTemplate, ...]:
    levels = ("micro", "event", "scene", "chapter")
    payload = _offline_draft(cards, levels) if offline else complete_json(
        "你是小说叙事模板总编。你只保留多章证据支持的抽象模式，删除重复、专名和单章结论，并输出严格 JSON。",
        _synthesis_prompt(cards, observations, levels),
        settings or ModelSettings.from_environment(),
        attempts=2,
        max_tokens=min((settings or ModelSettings.from_environment()).max_tokens, 12_000),
    )
    return _templates_from_draft(payload, levels, {card["chapter_id"] for card in cards}, "library")


def _macro_templates(
    cards: list[dict[str, Any]],
    settings: ModelSettings | None,
    offline: bool,
) -> tuple[NarrativeTemplate, ...]:
    if len(cards) < MACRO_WINDOW_CHAPTERS:
        return ()
    # Sliding windows retain phase boundaries while still being bounded.  The
    # final synthesis receives only anonymous structural cards.
    windows = [cards[index:index + MACRO_WINDOW_CHAPTERS] for index in range(0, len(cards) - MACRO_WINDOW_CHAPTERS + 1, MACRO_WINDOW_CHAPTERS)]
    observations = []
    for index, window in enumerate(windows, start=1):
        payload = _offline_draft(window, ("macro",)) if offline else complete_json(
            "你是小说多章节宏观结构提炼器。只总结抽象的阶段推进，不得使用来源专名或具体剧情，并严格输出 JSON。",
            _template_prompt(window, ("macro",), minimum_support=MACRO_WINDOW_CHAPTERS),
            settings or ModelSettings.from_environment(),
            attempts=2,
            max_tokens=min((settings or ModelSettings.from_environment()).max_tokens, 12_000),
        )
        observations.append({"window_index": index, "chapter_ids": [card["chapter_id"] for card in window], "draft": payload})
    payload = _offline_draft(cards, ("macro",)) if offline else complete_json(
        "你是小说宏观模板总编。只保留至少五章证据支持、可用于规划连续章节的抽象结构；不得写入来源专名或具体情节。",
        _synthesis_prompt(cards, observations, ("macro",)),
        settings or ModelSettings.from_environment(),
        attempts=2,
        max_tokens=min((settings or ModelSettings.from_environment()).max_tokens, 12_000),
    )
    return _templates_from_draft(payload, ("macro",), {card["chapter_id"] for card in cards}, "library")


def assess_template_library(library: NarrativeTemplateLibrary) -> TemplateQualityReport:
    issues: list[TemplateQualityIssue] = []
    try:
        library.validate()
    except (TypeError, ValueError) as exc:
        issues.append(TemplateQualityIssue("invalid_contract", str(exc)))
    by_level = Counter(item.level for item in library.templates)
    for level in ("event", "scene", "chapter"):
        if not by_level[level]:
            issues.append(TemplateQualityIssue("missing_template_level", f"缺少 {level} 级模板"))
    if len(library.chapter_ids) >= 10 and not by_level["macro"]:
        issues.append(TemplateQualityIssue("missing_macro_template", "样本达到 10 章但没有宏观模板"))
    covered = {chapter_id for item in library.templates for chapter_id in item.evidence_chapter_ids}
    required_coverage = max(2, int(len(library.chapter_ids) * 0.6 + 0.999))
    if len(covered) < required_coverage:
        issues.append(TemplateQualityIssue("insufficient_evidence_coverage", f"模板证据仅覆盖 {len(covered)} 章，至少需要 {required_coverage} 章"))
    for item in library.templates:
        if item.support_count < _MIN_SUPPORT[item.level]:
            issues.append(TemplateQualityIssue("insufficient_template_support", f"{item.template_id} 支持章节不足"))
        if item.level == "macro" and len(item.beat_sequence) < 5:
            issues.append(TemplateQualityIssue("macro_structure_too_shallow", f"{item.template_id} 缺少完整的阶段节拍"))
    return TemplateQualityReport(
        library.author_id,
        library.work_id,
        len(library.chapter_ids),
        {level: by_level[level] for level in _LEVELS},
        len(covered),
        tuple(issues),
    )


def _annotation_path(folder: Path, limit: int) -> Path:
    target = folder / f"chapter_annotations.sample-{limit}.jsonl"
    if target.exists():
        return target
    candidates = sorted(folder.glob("chapter_annotations.sample-*.jsonl"))
    if not candidates:
        raise FileNotFoundError("no evidence-first chapter annotations found")
    return candidates[-1]


def mine_template_library(
    author_id: str,
    work_id: str,
    *,
    limit: int = DEFAULT_TEMPLATE_LIMIT,
    batch_size: int = DEFAULT_TEMPLATE_BATCH_CHAPTERS,
    offline: bool = False,
) -> tuple[Path, Path]:
    """Mine a validated template library from stage-3/4 artifacts."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    folder = corpus_dir(author_id, work_id)
    annotations = [ChapterAnnotation.from_dict(row) for row in read_jsonl(_annotation_path(folder, limit))][:limit]
    continuity_path = folder / f"work_continuity.sample-{len(annotations)}.json"
    if not continuity_path.exists():
        raise FileNotFoundError("stage 4 continuity artifact is required before template mining")
    continuity = WorkContinuity.from_dict(json.loads(continuity_path.read_text(encoding="utf-8")))
    if tuple(annotation.chapter_id for annotation in annotations) != continuity.chapter_ids:
        raise ValueError("continuity chapters do not match annotation sample")
    if not offline:
        settings = ModelSettings.from_environment()
        if not settings.api_key:
            raise RuntimeError("模型密钥不可用：请在 config.yaml 或环境变量中配置后重试")
    else:
        settings = None
    cards = chapter_feature_cards(annotations, continuity)
    observations = _observations(cards, settings, offline, batch_size)
    templates = (*_synthesise(cards, observations, settings, offline), *_macro_templates(cards, settings, offline))
    library = NarrativeTemplateLibrary(author_id, work_id, tuple(card["chapter_id"] for card in cards), continuity.schema_version, tuple(templates))
    quality = assess_template_library(library)
    if not quality.passed:
        messages = "; ".join(f"{item.code}: {item.message}" for item in quality.issues)
        raise ValueError(f"template library quality checks failed: {messages}")
    target = folder / f"narrative_templates.sample-{len(cards)}.json"
    quality_target = folder / f"narrative_templates.sample-{len(cards)}.quality.json"
    write_json(target, library.to_dict())
    write_json(quality_target, {
        **quality.to_dict(),
        "batch_count": len(observations),
        "batch_chapter_ids": [row["chapter_ids"] for row in observations],
    })
    return target, quality_target
