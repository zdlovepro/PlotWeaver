"""Compile evidence-backed chapter annotations into deterministic prose programs.

This stage is deliberately model-free.  It makes fact and event coverage
auditable before a later model is allowed to turn a scene program into prose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .annotate import assess_annotation_quality
from .chapter_planner import assess_length_plan, iter_generation_batches, plan_controlled_expansion
from .contracts import ChapterAnnotation, ChapterDocument, ChapterProgram, EventProgram, FactContract, NarrativeBeat, ParagraphProgram, SceneProgram
from .contracts.program import EntityBinding
from .contracts.program_llm import chapter_program_to_llm_dict
from .jsonio import read_jsonl, write_json
from .narrative_graph import load_narrative_graph
from .paths import corpus_dir, runs_dir, validate_author_id


# A prose paragraph may carry one factual change, occasionally two tightly
# coupled facts.  More than that consistently produces event recaps instead of
# dramatised prose.
MAX_FACTS_PER_PARAGRAPH = 2


def _unique_ids(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _expression_mode(fact_kind: str) -> str:
    return {
        "identity": "narration",
        "knowledge": "narration",
        "rule": "narration",
        "information": "narration",
        "relationship": "narration",
        "goal": "internal",
        "emotion": "internal",
        "location": "action",
        "resource": "action",
        "progression": "action",
    }.get(fact_kind, "narration")


def _annotation_path(folder: Path, limit: int) -> Path:
    exact = folder / f"chapter_annotations.sample-{limit}.jsonl"
    if exact.exists():
        return exact
    candidates = sorted(folder.glob("chapter_annotations.sample-*.jsonl"))
    if not candidates:
        raise FileNotFoundError("缺少章节标注产物：请先运行 annotate")
    return candidates[-1]


def _ensure_known_ids(annotation: ChapterAnnotation) -> None:
    fact_ids = {item.fact_id for item in annotation.facts}
    event_ids = {item.event_id for item in annotation.events}
    for scene in annotation.scenes:
        unknown = set(scene.event_ids) - event_ids
        unknown |= set(scene.entry_fact_ids) - fact_ids
        unknown |= set(scene.exit_fact_ids) - fact_ids
        if unknown:
            raise ValueError(f"场景 {scene.scene_id} 引用了不存在的契约 ID：{sorted(unknown)}")
    for event in annotation.events:
        event_fact_ids = set(event.trigger_fact_ids) | set(event.precondition_fact_ids)
        event_fact_ids |= set(event.outcome_fact_ids) | set(event.cost_fact_ids)
        unknown = event_fact_ids - fact_ids
        if unknown:
            raise ValueError(f"事件 {event.event_id} 引用了不存在的事实 ID：{sorted(unknown)}")
    for change in annotation.state_changes:
        unknown = {item for item in (change.before_fact_id, change.after_fact_id) if item} - fact_ids
        if change.event_id not in event_ids or unknown:
            raise ValueError(f"状态变化 {change.change_id} 引用了不存在的事件或事实")


def compile_chapter_program(annotation: ChapterAnnotation) -> ChapterProgram:
    """Bind every extracted fact to exactly one scene and paragraph program."""

    if not annotation.scenes:
        raise ValueError("章节标注不含场景，无法编译章节程序")
    _ensure_known_ids(annotation)
    scenes = sorted(annotation.scenes, key=lambda item: (item.order, item.scene_id))
    events = sorted(annotation.events, key=lambda item: (item.order, item.event_id))
    facts = list(annotation.facts)
    facts_by_id = {item.fact_id: item for item in facts}
    events_by_id = {item.event_id: item for item in events}

    event_scene: dict[str, str] = {}
    for scene in scenes:
        for event_id in scene.event_ids:
            existing = event_scene.get(event_id)
            if existing and existing != scene.scene_id:
                raise ValueError(f"事件 {event_id} 同时属于多个场景")
            event_scene[event_id] = scene.scene_id
    for position, event in enumerate(events):
        event_scene.setdefault(event.event_id, scenes[min(position, len(scenes) - 1)].scene_id)

    def evidence_interval(items: Iterable[Any]) -> tuple[int, int] | None:
        spans = [span for item in items for span in getattr(item, "evidence", ())]
        if not spans:
            return None
        return min(span.start for span in spans), max(span.end for span in spans)

    # A fact about the protagonist is not necessarily an opening-scene fact.
    # Preserve source chronology; participant overlap is only a tie-breaker.
    scene_intervals: dict[str, tuple[int, int] | None] = {}
    for scene in scenes:
        related_items: list[Any] = [scene]
        related_items.extend(event for event in events if event_scene[event.event_id] == scene.scene_id)
        scene_intervals[scene.scene_id] = evidence_interval(related_items)

    fact_scene: dict[str, str] = {}

    def bind_fact(fact_id: str, scene_id: str) -> None:
        if fact_id in facts_by_id:
            fact_scene.setdefault(fact_id, scene_id)

    # Results and explicit costs belong to the event that produces them.  A
    # precondition may originate in an earlier scene, so it is only assigned
    # here if it has not already received a more direct state/scene binding.
    for event in events:
        scene_id = event_scene[event.event_id]
        for fact_id in (*event.cost_fact_ids, *event.outcome_fact_ids):
            bind_fact(fact_id, scene_id)
    for change in annotation.state_changes:
        if change.after_fact_id:
            bind_fact(change.after_fact_id, event_scene[change.event_id])
    for scene in scenes:
        for fact_id in (*scene.entry_fact_ids, *scene.exit_fact_ids):
            bind_fact(fact_id, scene.scene_id)
    for event in events:
        for fact_id in (*event.trigger_fact_ids, *event.precondition_fact_ids):
            bind_fact(fact_id, event_scene[event.event_id])
    for fact in facts:
        if fact.fact_id in fact_scene:
            continue
        fact_interval = evidence_interval((fact,))
        if fact_interval is None:
            candidates = [scene for scene in scenes if fact.subject_id in scene.participant_ids or fact.object_id in scene.participant_ids]
            fact_scene[fact.fact_id] = (candidates[0] if candidates else scenes[0]).scene_id
            continue
        fact_position = (fact_interval[0] + fact_interval[1]) / 2

        def scene_distance(scene: Any) -> tuple[float, int, int, str]:
            interval = scene_intervals[scene.scene_id]
            if interval is None:
                distance = float("inf")
            elif interval[0] <= fact_position <= interval[1]:
                distance = 0.0
            else:
                distance = min(abs(fact_position - interval[0]), abs(fact_position - interval[1]))
            participant_mismatch = int(fact.subject_id not in scene.participant_ids and fact.object_id not in scene.participant_ids)
            return distance, participant_mismatch, scene.order, scene.scene_id

        fact_scene[fact.fact_id] = min(scenes, key=scene_distance).scene_id

    events_for_scene: dict[str, list[Any]] = {scene.scene_id: [] for scene in scenes}
    for event in events:
        events_for_scene[event_scene[event.event_id]].append(event)
    facts_for_scene: dict[str, list[Any]] = {scene.scene_id: [] for scene in scenes}
    for fact in facts:
        facts_for_scene[fact_scene[fact.fact_id]].append(fact)

    all_event_ids = tuple(item.event_id for item in events)
    programs: list[SceneProgram] = []
    for output_order, scene in enumerate(scenes):
        local_events = events_for_scene[scene.scene_id]
        local_facts = sorted(
            facts_for_scene[scene.scene_id],
            key=lambda fact: ((evidence_interval((fact,)) or (float("inf"), float("inf")))[0], fact.fact_id),
        )
        event_programs = tuple(
            EventProgram(
                event_id=event.event_id,
                scene_id=scene.scene_id,
                summary=event.summary,
                action=event.action,
                participant_ids=event.participant_ids,
                precondition_fact_ids=_unique_ids((*event.trigger_fact_ids, *event.precondition_fact_ids)),
                required_fact_ids=_unique_ids((*event.cost_fact_ids, *event.outcome_fact_ids)),
                outcome_fact_ids=_unique_ids(event.outcome_fact_ids),
                obstacle=event.obstacle,
                decision=event.decision,
            )
            for event in local_events
        )
        fact_contracts = tuple(
            FactContract(
                fact_id=fact.fact_id,
                scene_id=scene.scene_id,
                fact_kind=fact.kind,
                subject_id=fact.subject_id,
                predicate=fact.predicate,
                value=fact.value,
                object_id=fact.object_id,
                expression_mode=_expression_mode(fact.kind),
                must_realize=True,
            )
            for fact in local_facts
        )
        paragraphs: list[ParagraphProgram] = []
        narrative_beats: list[NarrativeBeat] = []

        def add_dramatic_paragraph(
            function: str,
            beat_type: str,
            *,
            actor_id: str,
            objective: str,
            pressure: str,
            observable_action: str,
            focal_detail: str,
            state_change: str,
            required_fact_ids: Iterable[str] = (),
            event_ids: Iterable[str] = (),
            dialogue_pressure: str = "",
        ) -> None:
            """Emit one small, writeable dramatic unit.

            The compiler deliberately creates a beat even for setup and
            reaction paragraphs.  This prevents a model from treating the
            event's result as an acceptable substitute for its visible process.
            """

            facts_to_write = _unique_ids(required_fact_ids)
            events_to_write = _unique_ids(event_ids)
            viewpoint = actor_id or default_viewpoint
            if not viewpoint:
                raise ValueError(f"scene {scene.scene_id} has no usable dramatic viewpoint")
            beat_id = f"{scene.scene_id}:beat-{len(narrative_beats):02d}"
            narrative_beats.append(NarrativeBeat(
                beat_id=beat_id,
                scene_id=scene.scene_id,
                order=len(narrative_beats),
                beat_type=beat_type,
                actor_id=viewpoint,
                objective=objective or scene.objective,
                pressure=pressure,
                observable_action=observable_action,
                focal_detail=focal_detail,
                state_change=state_change,
                required_fact_ids=facts_to_write,
                event_ids=events_to_write,
                dialogue_pressure=dialogue_pressure,
            ))
            paragraphs.append(ParagraphProgram(
                paragraph_id=f"{scene.scene_id}:paragraph-{len(paragraphs):02d}",
                scene_id=scene.scene_id,
                order=len(paragraphs),
                function=function,
                required_fact_ids=facts_to_write,
                event_ids=events_to_write,
                viewpoint_entity_id=viewpoint,
                dialogue_act=dialogue_pressure,
                beat_ids=(beat_id,),
            ))

        def add_fact_beats(
            function: str,
            beat_type: str,
            fact_ids: Iterable[str],
            *,
            viewpoint: str,
            objective: str,
            observable_action: str,
            focal_detail: str,
            state_change: str,
        ) -> None:
            """Separate facts into observable moments instead of a fact list."""

            values = _unique_ids(fact_ids)
            for start in range(0, len(values), MAX_FACTS_PER_PARAGRAPH):
                group = values[start:start + MAX_FACTS_PER_PARAGRAPH]
                add_dramatic_paragraph(
                    function,
                    beat_type,
                    actor_id=viewpoint,
                    objective=objective,
                    pressure="",
                    observable_action=observable_action,
                    focal_detail=focal_detail,
                    state_change=state_change,
                    required_fact_ids=group,
                )

        local_fact_ids = {item.fact_id for item in local_facts}
        default_viewpoint = scene.participant_ids[0] if scene.participant_ids else ""
        event_required_fact_ids = {fact_id for event in event_programs for fact_id in event.required_fact_ids}
        covered_fact_ids: set[str] = set()

        def take(kinds: set[str], include_entry: bool = False) -> list[str]:
            selected: list[str] = []
            entry_ids = set(scene.entry_fact_ids) if include_entry else set()
            for fact in local_facts:
                if fact.fact_id in covered_fact_ids or fact.fact_id in event_required_fact_ids:
                    continue
                if fact.kind in kinds or fact.fact_id in entry_ids:
                    selected.append(fact.fact_id)
                    covered_fact_ids.add(fact.fact_id)
            return selected

        # Establish durable state before the first active beat.  These facts
        # still need a visible focal action; they are not permission to insert
        # a detached setting summary.
        add_fact_beats(
            "orientation", "setup", take({"identity", "relationship", "location", "rule"}, include_entry=True),
            viewpoint=default_viewpoint, objective=scene.objective,
            observable_action="以人物在场内的动作或停顿落定当前处境",
            focal_detail="让视角人物注意到与当前处境直接相关的可观察细节",
            state_change="读者明确当前人物、位置或关系所构成的起点",
        )
        for event, event_program in zip(local_events, event_programs):
            actor_id = event.participant_ids[0] if event.participant_ids else default_viewpoint
            fact_ids = list(event_program.required_fact_ids)
            # First show an attempt.  The event ID is bound here only, so the
            # semantic audit can distinguish an enacted event from a later
            # reaction to its outcome.
            add_dramatic_paragraph(
                "dialogue_conflict" if event.obstacle and len(event.participant_ids) >= 2 else "action_progression",
                "action",
                actor_id=actor_id,
                objective=event.action,
                pressure=event.obstacle,
                observable_action=event.action,
                focal_detail="让视角人物先捕捉到行动引起的具体变化",
                state_change="行动已经开始，并对场内其他人或局势施加影响",
                required_fact_ids=fact_ids[:1],
                event_ids=(event.event_id,),
                dialogue_pressure="用对白提出、质疑、拒绝或确认当前行动" if event.obstacle and len(event.participant_ids) >= 2 else "",
            )
            if event.obstacle:
                add_dramatic_paragraph(
                    "obstacle",
                    "pressure",
                    actor_id=actor_id,
                    objective=event.action,
                    pressure=event.obstacle,
                    observable_action="让阻力通过对方的动作、话语或局面变化显形",
                    focal_detail="让视角人物注意到阻力造成的具体不适、迟滞或威胁",
                    state_change="原先的顺利推进被阻断，人物必须回应",
                    dialogue_pressure="对白必须形成施压与回应，且回应影响下一步行动",
                )
            add_dramatic_paragraph(
                "decision" if event.decision else "perception_reaction",
                "choice" if event.decision else "reaction",
                actor_id=actor_id,
                objective=event.decision or event.action,
                pressure=event.obstacle,
                observable_action="写出人物在压力后的具体回应、取舍或下一步动作",
                focal_detail="让视角人物的感知或身体反应改变其判断",
                state_change=event.decision or "人物对局势作出可见回应，结果开始成立",
                required_fact_ids=fact_ids[1:],
            )
            covered_fact_ids.update(event_program.required_fact_ids)
        add_fact_beats(
            "perception_reaction", "reaction", take({"knowledge", "emotion"}),
            viewpoint=default_viewpoint, objective=scene.objective,
            observable_action="以人物对已发生变化的可见反应推进叙事",
            focal_detail="写出影响判断的感知、身体反应或注意力变化",
            state_change="人物的认识或情绪获得可观察的落点",
        )
        add_fact_beats(
            "decision", "choice", take({"goal"}),
            viewpoint=default_viewpoint, objective=scene.objective,
            observable_action="让人物用动作、对白或明确取舍落实下一步选择",
            focal_detail="让视角人物从当前局势中捕捉促成选择的细节",
            state_change="人物形成并显露下一步目标",
        )
        add_fact_beats(
            "aftermath", "aftermath", take({"resource", "progression"}),
            viewpoint=default_viewpoint, objective=scene.objective,
            observable_action="写出行动结束后的收束动作或局面余波",
            focal_detail="让视角人物注意到结果留下的具体痕迹",
            state_change="场景结果稳定下来，并为下一场留下压力",
        )
        missing_fact_ids = [item.fact_id for item in local_facts if item.fact_id not in covered_fact_ids]
        local_exit_ids = {fact_id for fact_id in scene.exit_fact_ids if fact_id in local_fact_ids}
        if missing_fact_ids:
            function = "transition" if local_exit_ids.intersection(missing_fact_ids) else "aftermath"
            add_fact_beats(
                function, "transition" if function == "transition" else "aftermath", missing_fact_ids,
                viewpoint=default_viewpoint, objective=scene.objective,
                observable_action="用可观察的移动、收束或注意力切换承接现有状态",
                focal_detail="让视角人物注意到状态变化带来的具体线索",
                state_change="当前场景的剩余状态被落定",
            )

        if not paragraphs:
            raise ValueError(f"场景 {scene.scene_id} 无法编译出段落程序")
        last_event_order = max((event.order for event in local_events), default=-1)
        forbidden = tuple(event_id for event_id in all_event_ids if events_by_id[event_id].order > last_event_order)
        programs.append(SceneProgram(
            scene_id=scene.scene_id,
            chapter_id=annotation.chapter_id,
            order=output_order,
            objective=scene.objective,
            participant_ids=scene.participant_ids,
            entry_state_fact_ids=_unique_ids(scene.entry_fact_ids),
            event_programs=event_programs,
            fact_contracts=fact_contracts,
            paragraphs=tuple(paragraphs),
            exit_state_fact_ids=_unique_ids(scene.exit_fact_ids),
            forbidden_event_ids=forbidden,
            narrative_beats=tuple(narrative_beats),
        ))
    program = ChapterProgram(
        annotation.chapter_id,
        annotation.source_hash,
        tuple(programs),
        schema_version="2.0",
        entities=tuple(EntityBinding(item.entity_id, item.canonical_name, item.kind) for item in annotation.entities),
    )
    program.validate()
    return program


def assess_chapter_program(program: ChapterProgram) -> dict[str, Any]:
    """Return a machine-readable quality gate for the deterministic compiler."""

    issues: list[dict[str, str]] = []
    try:
        program.validate()
    except ValueError as exc:
        issues.append({"code": "PROGRAM_INVALID", "message": str(exc)})
    scenes = list(program.scene_programs)
    facts = [fact for scene in scenes for fact in scene.fact_contracts]
    events = [event for scene in scenes for event in scene.event_programs]
    paragraphs = [paragraph for scene in scenes for paragraph in scene.paragraphs]
    beats = [beat for scene in scenes for beat in scene.narrative_beats]
    covered_facts = {fact_id for paragraph in paragraphs for fact_id in paragraph.required_fact_ids}
    covered_events = {event_id for paragraph in paragraphs for event_id in paragraph.event_ids}
    must_fact_ids = {fact.fact_id for fact in facts if fact.must_realize}
    if must_fact_ids - covered_facts:
        issues.append({"code": "UNBOUND_FACTS", "message": "存在未绑定到段落的必须表达事实"})
    event_ids = {event.event_id for event in events}
    if event_ids - covered_events:
        issues.append({"code": "UNBOUND_EVENTS", "message": "存在未绑定到段落的必现事件"})
    overloaded = [paragraph.paragraph_id for paragraph in paragraphs if len(paragraph.required_fact_ids) > MAX_FACTS_PER_PARAGRAPH]
    if overloaded:
        issues.append({"code": "FACT_DUMP", "message": "存在承载过多事实的段落程序"})
    if program.schema_version.startswith(("2", "3")):
        unbound_beats = {beat.beat_id for beat in beats} - {beat_id for paragraph in paragraphs for beat_id in paragraph.beat_ids}
        if unbound_beats:
            issues.append({"code": "UNBOUND_DRAMATIC_BEATS", "message": "存在未绑定到段落的戏剧节拍"})
        if not beats:
            issues.append({"code": "MISSING_DRAMATIC_BEATS", "message": "新版章节程序缺少戏剧节拍"})
    if program.generation_mode == "controlled_expansion":
        planned_chars = sum(paragraph.target_chars for paragraph in paragraphs)
        if not (program.target_char_min <= planned_chars <= program.target_char_max):
            issues.append({"code": "CHAR_BUDGET_OUT_OF_RANGE", "message": "受控扩写的段落字数计划未落入章节目标区间"})
        missing_budget = [paragraph.paragraph_id for paragraph in paragraphs if paragraph.target_chars <= 0 or paragraph.minimum_chars <= 0]
        if missing_budget:
            issues.append({"code": "PARAGRAPH_BUDGET_MISSING", "message": "受控扩写存在未分配字数预算的段落"})
        unlicensed = [beat.beat_id for beat in beats if beat.expansion_license != "source" and not beat.support_fact_ids]
        if unlicensed:
            issues.append({"code": "EXPANSION_LICENSE_MISSING", "message": "扩写节拍缺少依据事实，不能生成"})
    return {
        "schema_version": "1.0",
        "chapter_id": program.chapter_id,
        "passed": not issues,
        "counts": {
            "scene_count": len(scenes),
            "fact_contract_count": len(facts),
            "must_realize_fact_count": len(must_fact_ids),
            "paragraph_bound_fact_count": len(covered_facts),
            "event_program_count": len(events),
            "paragraph_bound_event_count": len(covered_events),
            "paragraph_program_count": len(paragraphs),
            "narrative_beat_count": len(beats),
            "licensed_expansion_beat_count": sum(beat.expansion_license != "source" for beat in beats),
            "planned_char_count": sum(paragraph.target_chars for paragraph in paragraphs),
            "max_facts_per_paragraph": max((len(paragraph.required_fact_ids) for paragraph in paragraphs), default=0),
            "future_event_barrier_count": sum(len(scene.forbidden_event_ids) for scene in scenes),
        },
        "issues": issues,
    }


def _chapter_number(chapter_id: str) -> int | None:
    try:
        return int(str(chapter_id).replace("\\", "/").rsplit("/", 1)[-1])
    except ValueError:
        return None


def _require_expansion_quality(document: ChapterDocument, annotation: ChapterAnnotation) -> dict[str, Any]:
    """Block long-form planning when the source was only summarised coarsely."""

    report = assess_annotation_quality(document, annotation)
    if not report.passed:
        hard_issues = [f"{item.code}: {item.message}" for item in report.issues if item.severity == "error"]
        raise ValueError("章节标注密度不足，不能进入受控扩写；请先重跑高密度标注。" + "；".join(hard_issues))
    return report.to_dict()


def _documents_by_id(folder: Path) -> dict[str, ChapterDocument]:
    return {
        str(row.get("chapter_id", "")): ChapterDocument.from_dict(row)
        for row in read_jsonl(folder / "chapter_documents.jsonl")
    }


def compile_chapter_program_work(
    author_id: str,
    work_id: str,
    chapter_number: int,
    *,
    limit: int = 20,
    run_id: str = "chapter-program",
) -> dict[str, Path]:
    """Compile one stored annotation and keep all runtime artifacts under ``runs``."""

    if chapter_number < 1:
        raise ValueError("chapter_number must be positive")
    author_id = validate_author_id(author_id)
    rows = read_jsonl(_annotation_path(corpus_dir(author_id, work_id), limit))
    selected = next((row for row in rows if _chapter_number(str(row.get("chapter_id", ""))) == chapter_number), None)
    if selected is None:
        raise ValueError(f"在当前标注样本中找不到第 {chapter_number} 章")
    program = compile_chapter_program(ChapterAnnotation.from_dict(selected))
    quality = assess_chapter_program(program)
    target = runs_dir(author_id, run_id) / "chapter_programs"
    base = target / f"chapter-{chapter_number:04d}"
    program_path = base.with_suffix(".program.json")
    quality_path = base.with_suffix(".quality.json")
    llm_path = base.with_suffix(".model-contract.zh.json")
    write_json(program_path, program.to_dict())
    write_json(quality_path, quality)
    write_json(llm_path, chapter_program_to_llm_dict(program))
    return {"program": program_path, "quality": quality_path, "model_contract": llm_path}


def plan_expansion_program_work(
    author_id: str,
    work_id: str,
    chapter_number: int,
    *,
    target_char_min: int,
    target_char_max: int,
    limit: int = 20,
    run_id: str = "chapter-plan",
) -> dict[str, Path]:
    """Compile an annotation into a target-length, evidence-bounded program."""

    if chapter_number < 1:
        raise ValueError("chapter_number must be positive")
    author_id = validate_author_id(author_id)
    graph = load_narrative_graph(author_id, work_id)
    rows = read_jsonl(_annotation_path(corpus_dir(author_id, work_id), limit))
    selected = next((row for row in rows if _chapter_number(str(row.get("chapter_id", ""))) == chapter_number), None)
    if selected is None:
        raise ValueError(f"在当前标注样本中找不到第 {chapter_number} 章")
    annotation = ChapterAnnotation.from_dict(selected)
    if annotation.chapter_id not in graph.chapter_ids or graph.source_hashes.get(annotation.chapter_id) != annotation.source_hash:
        raise ValueError("当前章节与 narrative_graph.json 不一致；请先重新构建剧情事实图谱")
    document = _documents_by_id(corpus_dir(author_id, work_id)).get(annotation.chapter_id)
    if document is None:
        raise ValueError(f"缺少章节原始文档：{annotation.chapter_id}")
    annotation_quality = _require_expansion_quality(document, annotation)
    source_program = compile_chapter_program(annotation)
    program = plan_controlled_expansion(
        source_program,
        target_char_min=target_char_min,
        target_char_max=target_char_max,
    )
    structural_quality = assess_chapter_program(program)
    length_quality = assess_length_plan(program)
    target = runs_dir(author_id, run_id) / "chapter_programs"
    base = target / f"chapter-{chapter_number:04d}"
    program_path = base.with_suffix(".program.json")
    quality_path = base.with_suffix(".quality.json")
    llm_path = base.with_suffix(".model-contract.zh.json")
    batches_path = base.with_suffix(".batches.json")
    write_json(program_path, program.to_dict())
    write_json(quality_path, {"annotation": annotation_quality, "structural": structural_quality, "length_plan": length_quality})
    write_json(llm_path, chapter_program_to_llm_dict(program))
    write_json(batches_path, {
        "schema_version": "1.0",
        "batch_paragraph_limit": 6,
        "batches": [
            {"scene_id": scene_id, "paragraph_ids": list(paragraph_ids)}
            for scene_id, paragraph_ids in iter_generation_batches(program)
        ],
    })
    return {"program": program_path, "quality": quality_path, "model_contract": llm_path, "batches": batches_path}


def plan_expansion_programs_work(
    author_id: str,
    work_id: str,
    *,
    chapter_limit: int,
    target_char_min: int,
    target_char_max: int,
    annotation_limit: int = 20,
    run_id: str = "chapter-plan-work",
) -> Path:
    """Plan a sequential sample without invoking the model.

    One index records all program, quality and batch artifacts, so the next
    stage can resume a chapter range rather than hand-selecting JSON files.
    """

    if chapter_limit < 1:
        raise ValueError("chapter_limit must be positive")
    author_id = validate_author_id(author_id)
    graph = load_narrative_graph(author_id, work_id)
    rows = read_jsonl(_annotation_path(corpus_dir(author_id, work_id), annotation_limit))[:chapter_limit]
    if len(rows) < chapter_limit:
        raise ValueError(f"当前标注样本只有 {len(rows)} 章，少于请求的 {chapter_limit} 章")
    target = runs_dir(author_id, run_id) / "chapter_programs"
    documents = _documents_by_id(corpus_dir(author_id, work_id))
    index_rows: list[dict[str, str | int | bool]] = []
    for number, row in enumerate(rows, start=1):
        annotation = ChapterAnnotation.from_dict(row)
        if annotation.chapter_id not in graph.chapter_ids or graph.source_hashes.get(annotation.chapter_id) != annotation.source_hash:
            raise ValueError("章节标注与 narrative_graph.json 不一致；请先重新构建剧情事实图谱")
        document = documents.get(annotation.chapter_id)
        if document is None:
            raise ValueError(f"缺少章节原始文档：{annotation.chapter_id}")
        annotation_quality = _require_expansion_quality(document, annotation)
        source_program = compile_chapter_program(annotation)
        program = plan_controlled_expansion(
            source_program,
            target_char_min=target_char_min,
            target_char_max=target_char_max,
        )
        structural_quality = assess_chapter_program(program)
        length_quality = assess_length_plan(program)
        base = target / f"chapter-{number:04d}"
        program_path = base.with_suffix(".program.json")
        quality_path = base.with_suffix(".quality.json")
        contract_path = base.with_suffix(".model-contract.zh.json")
        batches_path = base.with_suffix(".batches.json")
        write_json(program_path, program.to_dict())
        write_json(quality_path, {"annotation": annotation_quality, "structural": structural_quality, "length_plan": length_quality})
        write_json(contract_path, chapter_program_to_llm_dict(program))
        batches = [
            {"scene_id": scene_id, "paragraph_ids": list(paragraph_ids)}
            for scene_id, paragraph_ids in iter_generation_batches(program)
        ]
        write_json(batches_path, {"schema_version": "1.0", "batch_paragraph_limit": 6, "batches": batches})
        index_rows.append({
            "chapter_no": number,
            "chapter_id": program.chapter_id,
            "program": str(program_path),
            "quality": str(quality_path),
            "batches": str(batches_path),
            "passed": bool(structural_quality["passed"] and length_quality["passed"]),
        })
    index_path = runs_dir(author_id, run_id) / "expansion_plan_index.json"
    write_json(index_path, {
        "schema_version": "1.0",
        "author_id": author_id,
        "work_id": work_id,
        "generation_mode": "controlled_expansion",
        "target_char_min": target_char_min,
        "target_char_max": target_char_max,
        "chapter_limit": chapter_limit,
        "narrative_graph": str(corpus_dir(author_id, work_id) / "narrative_graph.json"),
        "chapters": index_rows,
        "passed": all(bool(item["passed"]) for item in index_rows),
    })
    return index_path
