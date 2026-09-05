"""Compile evidence-backed chapter annotations into deterministic prose programs.

This stage is deliberately model-free.  It makes fact and event coverage
auditable before a later model is allowed to turn a scene program into prose.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .annotate import assess_annotation_quality
from .chapter_planner import assess_length_plan, iter_generation_batches, plan_controlled_expansion
from .contracts import ChapterAnnotation, ChapterDocument, ChapterProgram, EventGraph, EventProgram, FactContract, NarrativeBeat, ParagraphProgram, SceneCard, SceneProgram
from .contracts.program import EntityBinding
from .contracts.program_llm import chapter_program_to_llm_dict
from .jsonio import read_json, read_jsonl, write_json
from .macro_planner import build_macro_plan_work, load_chapter_contracts, validate_program_chapter_contract
from .narrative_graph import load_narrative_graph
from .paths import corpus_dir, runs_dir, validate_author_id


# A prose paragraph may carry one factual change, occasionally two tightly
# coupled facts.  More than that consistently produces event recaps instead of
# dramatised prose.
MAX_FACTS_PER_PARAGRAPH = 2

# The atom extractor intentionally keeps every factual change separate.  That
# is the right granularity for audit and repair, but it is too fine to hand to
# a prose writer: a 5--8k-character chapter made of twenty atom-scenes forces
# every dramatic unit to be only a few hundred characters long.  The compiler
# therefore makes a second, deterministic pass that groups adjacent atoms into
# writeable dramatic scenes without merging or discarding their facts/events.
MAX_EVENTS_PER_DRAMATIC_SCENE = 4
MAX_EVENT_GAP_CHARS = 650


def _unique_ids(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _evidence_interval(items: Iterable[Any]) -> tuple[int, int] | None:
    """Return the source interval covering ``items``, when evidence exists."""

    spans = [span for item in items for span in getattr(item, "evidence", ())]
    if not spans:
        return None
    return min(span.start for span in spans), max(span.end for span in spans)


def _atomic_scenes_to_dramatic_scenes(annotation: ChapterAnnotation) -> tuple[SceneCard, ...]:
    """Aggregate a run of one-event source scenes into modest prose scenes.

    Annotations authored as genuine multi-event scenes are left untouched.
    Aggregation is only for the mechanically-derived one-event cards used by
    the high-density atom extractor.  A group remains bounded by both source
    distance and event count, so a chapter's entire setup can never become one
    indiscriminate prompt.
    """

    source_scenes = tuple(sorted(annotation.scenes, key=lambda item: (item.order, item.scene_id)))
    if not source_scenes or any(len(scene.event_ids) != 1 for scene in source_scenes):
        return source_scenes

    events_by_id = {event.event_id: event for event in annotation.events}
    if any(event_id not in events_by_id for scene in source_scenes for event_id in scene.event_ids):
        # _ensure_known_ids emits the user-facing error later.  Do not make an
        # incomplete scene here and hide the actual annotation defect.
        return source_scenes

    grouped: list[list[SceneCard]] = []
    current: list[SceneCard] = []
    current_end: int | None = None
    for source_scene in source_scenes:
        event = events_by_id[source_scene.event_ids[0]]
        interval = _evidence_interval((event, source_scene))
        event_start = interval[0] if interval is not None else None
        may_join = bool(current) and len(current) < MAX_EVENTS_PER_DRAMATIC_SCENE
        if may_join and event_start is not None and current_end is not None:
            may_join = event_start - current_end <= MAX_EVENT_GAP_CHARS
        elif may_join:
            # Missing evidence means chronology is not sufficiently grounded
            # to make an aggregation decision.
            may_join = False
        if not may_join:
            if current:
                grouped.append(current)
            current = [source_scene]
        else:
            current.append(source_scene)
        if interval is not None:
            current_end = interval[1]
        elif not current[:-1]:
            current_end = None
    if current:
        grouped.append(current)

    # Keep original source IDs when no real merge occurred.  This makes the
    # transformation backward-compatible for sparse, manually-curated input.
    if all(len(group) == 1 for group in grouped):
        return source_scenes

    dramatic_scenes: list[SceneCard] = []
    for order, group in enumerate(grouped):
        first, last = group[0], group[-1]
        objectives = _unique_ids(scene.objective for scene in group)
        objective = objectives[0] if len(objectives) == 1 else f"{objectives[0]}，并推进至{objectives[-1]}"
        dramatic_scenes.append(SceneCard(
            scene_id=f"{annotation.chapter_id}:dramatic-scene-{order:03d}",
            chapter_id=annotation.chapter_id,
            order=order,
            participant_ids=_unique_ids(participant for scene in group for participant in scene.participant_ids),
            event_ids=tuple(event_id for scene in group for event_id in scene.event_ids),
            objective=objective,
            entry_fact_ids=_unique_ids(first.entry_fact_ids),
            exit_fact_ids=_unique_ids(last.exit_fact_ids),
            tension=max(scene.tension for scene in group),
            evidence=tuple(span for scene in group for span in scene.evidence),
            location_ids=_unique_ids(location for scene in group for location in scene.location_ids),
            time_anchor_ids=_unique_ids(anchor for scene in group for anchor in scene.time_anchor_ids),
        ))
    return tuple(dramatic_scenes)


def _consolidate_event_action_response_paragraphs(
    paragraphs: Iterable[ParagraphProgram],
) -> list[ParagraphProgram]:
    """Keep an event's attempt and immediate response in one readable paragraph.

    The atom contract keeps an action beat and its response distinct for
    validation.  They need not become two visual paragraph breaks, however.
    Combining only an adjacent response/choice with its immediately preceding
    event beat preserves causal order and the two-fact ceiling while avoiding
    a staccato sequence of 50--70-character fragments.
    """

    consolidated: list[ParagraphProgram] = []
    response_functions = {"perception_reaction", "decision"}
    action_functions = {"action_progression", "dialogue_conflict"}
    for paragraph in paragraphs:
        previous = consolidated[-1] if consolidated else None
        merged_fact_ids = _unique_ids((*(previous.required_fact_ids if previous else ()), *paragraph.required_fact_ids))
        can_merge = (
            previous is not None
            and previous.function in action_functions
            and bool(previous.event_ids)
            and paragraph.function in response_functions
            and not paragraph.event_ids
            and previous.viewpoint_entity_id == paragraph.viewpoint_entity_id
            and len(merged_fact_ids) <= MAX_FACTS_PER_PARAGRAPH
        )
        if not can_merge:
            consolidated.append(paragraph)
            continue
        consolidated[-1] = ParagraphProgram(
            paragraph_id=previous.paragraph_id,
            scene_id=previous.scene_id,
            order=previous.order,
            function=previous.function,
            required_fact_ids=merged_fact_ids,
            event_ids=previous.event_ids,
            viewpoint_entity_id=previous.viewpoint_entity_id,
            dialogue_act=previous.dialogue_act or paragraph.dialogue_act,
            beat_ids=_unique_ids((*previous.beat_ids, *paragraph.beat_ids)),
            target_chars=previous.target_chars,
            minimum_chars=previous.minimum_chars,
        )
    return [replace(paragraph, order=index) for index, paragraph in enumerate(consolidated)]


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
        event_fact_ids |= set(event.outcome_fact_ids) | set(event.cost_fact_ids) | set(event.basis_fact_ids)
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
    scenes = list(_atomic_scenes_to_dramatic_scenes(annotation))
    events = sorted(annotation.events, key=lambda item: (item.order, item.event_id))
    facts = list(annotation.facts)
    facts_by_id = {item.fact_id: item for item in facts}
    events_by_id = {item.event_id: item for item in events}
    mechanisms = sorted(
        annotation.narrative_mechanisms,
        key=lambda item: (item.order, item.mechanism_id),
    )

    event_scene: dict[str, str] = {}
    for scene in scenes:
        for event_id in scene.event_ids:
            existing = event_scene.get(event_id)
            if existing and existing != scene.scene_id:
                raise ValueError(f"事件 {event_id} 同时属于多个场景")
            event_scene[event_id] = scene.scene_id
    for position, event in enumerate(events):
        event_scene.setdefault(event.event_id, scenes[min(position, len(scenes) - 1)].scene_id)

    # A fact about the protagonist is not necessarily an opening-scene fact.
    # Preserve source chronology; participant overlap is only a tie-breaker.
    scene_intervals: dict[str, tuple[int, int] | None] = {}
    for scene in scenes:
        related_items: list[Any] = [scene]
        related_items.extend(event for event in events if event_scene[event.event_id] == scene.scene_id)
        scene_intervals[scene.scene_id] = _evidence_interval(related_items)

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
        fact_interval = _evidence_interval((fact,))
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
        # A trigger/precondition can be produced in an earlier scene.  Keep it
        # in the current writer packet as read-only context rather than either
        # dropping it or forcing prose to repeat it as a fresh development.
        context_fact_ids = {
            fact_id
            for event in local_events
            for fact_id in (*event.trigger_fact_ids, *event.precondition_fact_ids, *event.basis_fact_ids)
            if fact_scene.get(fact_id) != scene.scene_id
        }
        local_facts = sorted(
            [*facts_for_scene[scene.scene_id], *(facts_by_id[item] for item in sorted(context_fact_ids) if item in facts_by_id)],
            key=lambda fact: ((_evidence_interval((fact,)) or (float("inf"), float("inf")))[0], fact.fact_id),
        )
        event_programs = tuple(
            EventProgram(
                event_id=event.event_id,
                scene_id=scene.scene_id,
                summary=event.summary,
                action=event.action,
                participant_ids=event.participant_ids,
                precondition_fact_ids=_unique_ids((
                    *event.trigger_fact_ids,
                    *event.precondition_fact_ids,
                    *(fact_id for fact_id in event.basis_fact_ids if fact_id not in event.outcome_fact_ids and fact_id not in event.cost_fact_ids),
                )),
                required_fact_ids=_unique_ids((*event.cost_fact_ids, *event.outcome_fact_ids)),
                outcome_fact_ids=_unique_ids(event.outcome_fact_ids),
                obstacle=event.obstacle,
                decision=event.decision,
                action_type=event.action_type,
                actor_id=event.actor_id,
                target_ids=event.target_ids,
                basis_fact_ids=event.basis_fact_ids,
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
                must_realize=fact.fact_id not in context_fact_ids,
            )
            for fact in local_facts
            if fact.fact_id not in context_fact_ids
        )
        context_fact_contracts = tuple(
            FactContract(
                fact_id=fact.fact_id,
                scene_id=scene.scene_id,
                fact_kind=fact.kind,
                subject_id=fact.subject_id,
                predicate=fact.predicate,
                value=fact.value,
                object_id=fact.object_id,
                expression_mode=_expression_mode(fact.kind),
                must_realize=False,
            )
            for fact in local_facts
            if fact.fact_id in context_fact_ids
        )
        paragraphs: list[ParagraphProgram] = []
        narrative_beats: list[NarrativeBeat] = []
        assigned_mechanism_ids: set[str] = set()

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
            relevant_mechanisms = [
                mechanism for mechanism in mechanisms
                if mechanism.mechanism_id not in assigned_mechanism_ids
                and (
                    set(mechanism.fact_ids).intersection(facts_to_write)
                    or set(mechanism.event_ids).intersection(events_to_write)
                )
            ][:3]
            mechanism_ids = tuple(item.mechanism_id for item in relevant_mechanisms)
            assigned_mechanism_ids.update(mechanism_ids)
            functions = _unique_ids(item.narrative_function for item in relevant_mechanisms)
            losses = _unique_ids(item.counterfactual_loss for item in relevant_mechanisms)
            inferred_pressure = next((
                item.summary for item in relevant_mechanisms
                if item.kind in {"constraint_pressure", "contrast"}
            ), "")
            inferred_objective = next((
                item.summary for item in relevant_mechanisms
                if item.kind in {"character_drive", "setup"}
            ), "")
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
                objective=inferred_objective or objective or scene.objective,
                pressure=pressure or inferred_pressure,
                observable_action=observable_action,
                focal_detail=focal_detail,
                state_change=state_change,
                required_fact_ids=facts_to_write,
                event_ids=events_to_write,
                dialogue_pressure=dialogue_pressure,
                mechanism_ids=mechanism_ids,
                narrative_function="；".join(functions[:2]),
                counterfactual_guard="；".join(losses[:2]),
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
                # A remaining fact may describe a reaction by a scene
                # participant other than the default viewpoint.  Attribute
                # its beat to that factual subject instead of forcing (for
                # example) a mother's farewell reaction onto the father.
                fact_viewpoint = next(
                    (
                        facts_by_id[fact_id].subject_id
                        for fact_id in group
                        if fact_id in facts_by_id and facts_by_id[fact_id].subject_id in scene.participant_ids
                    ),
                    viewpoint,
                )
                add_dramatic_paragraph(
                    function,
                    beat_type,
                    actor_id=fact_viewpoint,
                    objective=objective,
                    pressure="",
                    observable_action=observable_action,
                    focal_detail=focal_detail,
                    state_change=state_change,
                    required_fact_ids=group,
                )

        local_fact_ids = {item.fact_id for item in fact_contracts}
        default_viewpoint = scene.participant_ids[0] if scene.participant_ids else ""
        event_required_fact_ids = {fact_id for event in event_programs for fact_id in event.required_fact_ids}
        event_bound_fact_ids = {
            fact_id
            for event in event_programs
            for fact_id in (*event.precondition_fact_ids, *event.required_fact_ids)
        }
        covered_fact_ids: set[str] = set()

        def take(kinds: set[str], include_entry: bool = False) -> list[str]:
            selected: list[str] = []
            entry_ids = set(scene.entry_fact_ids) if include_entry else set()
            for fact in local_facts:
                if fact.fact_id in context_fact_ids:
                    continue
                if fact.fact_id in covered_fact_ids or fact.fact_id in event_required_fact_ids:
                    continue
                if fact.kind in kinds or fact.fact_id in entry_ids:
                    selected.append(fact.fact_id)
                    covered_fact_ids.add(fact.fact_id)
            return selected

        def opening_exposition() -> list[str]:
            """Keep evidence-rich setup that occurs before the first event.

            The previous compiler eliminated every standalone setup paragraph
            after a sparse annotation caused static padding.  That also erased
            legitimate opening exposition once the annotation became richer.
            Here setup is admitted only when it contains at least two distinct
            source facts, and never steals a fact needed to stage the imminent
            event.  It therefore remains a factual narration job rather than
            a licence for invented atmosphere or premature movement.
            """

            if not local_events:
                return []
            first_interval = _evidence_interval((local_events[0],))
            if first_interval is None:
                return []
            selected = [
                fact.fact_id
                for fact in local_facts
                if fact.fact_id not in context_fact_ids
                and fact.fact_id not in covered_fact_ids
                and fact.fact_id not in event_bound_fact_ids
                and (fact_interval := _evidence_interval((fact,))) is not None
                and fact_interval[1] <= first_interval[0]
            ]
            return selected if len(selected) >= 2 else []

        outcome_fact_ids = {
            fact_id
            for event in event_programs
            for fact_id in event.outcome_fact_ids
        }

        def pre_event_exposition(event: Any) -> list[str]:
            """Return earlier durable facts that the current action can enact.

            A source often introduces a person immediately before that person
            acts.  Splitting those facts into a standalone orientation block
            leaves a model with no legitimate process to write and encourages
            either padding or a premature movement.  We keep outcome facts
            with their producing event, but weave compatible preceding facts
            into the next event whose participant they describe.
            """

            interval = _evidence_interval((event,))
            if interval is None:
                return []
            participants = set(event.participant_ids)
            selected: list[str] = []
            for fact in local_facts:
                if fact.fact_id in context_fact_ids or fact.fact_id in covered_fact_ids:
                    continue
                if fact.fact_id in outcome_fact_ids:
                    continue
                fact_interval = _evidence_interval((fact,))
                if fact_interval is None or fact_interval[0] > interval[1]:
                    continue
                if fact.subject_id not in participants and fact.object_id not in participants:
                    continue
                selected.append(fact.fact_id)
            return selected

        # Evidence-rich opening exposition is a real part of many chapters.
        # It receives its own fact-bound narration paragraphs; the writer
        # packet explicitly forbids it from inventing a current movement or
        # pulling later scene participants forward.
        opening_fact_ids = opening_exposition()
        add_fact_beats(
            "orientation", "setup", opening_fact_ids,
            viewpoint=default_viewpoint, objective=scene.objective,
            observable_action="用叙述交代当前人物、关系、处境或已在场状态；不得书写进门、出门、归来、离开、跨越门槛、收拾行李等尚未绑定事件的推进动作",
            focal_detail="每个描述都必须兑现当前事实，不得补写天气、器物、环境或人物动作",
            state_change="读者明确当前人物、位置或关系所构成的起点",
        )
        covered_fact_ids.update(opening_fact_ids)
        for event, event_program in zip(local_events, event_programs):
            # Event participants are ordered by mention, not by agency.  When
            # an event's asserted result has a participant as its subject,
            # that subject is the safest structural actor for the action
            # beat (for example, a father who gives advice to the viewpoint
            # character).
            actor_id = event_program.actor_id or next(
                (
                    facts_by_id[fact_id].subject_id
                    for fact_id in (*event_program.basis_fact_ids, *event_program.required_fact_ids)
                    if fact_id in facts_by_id and facts_by_id[fact_id].subject_id in event.participant_ids
                ),
                event.participant_ids[0] if event.participant_ids else default_viewpoint,
            )
            exposition_fact_ids = pre_event_exposition(event)
            event_fact_ids = list(event_program.required_fact_ids)
            action_fact_ids = list(exposition_fact_ids[:MAX_FACTS_PER_PARAGRAPH])
            if len(action_fact_ids) < MAX_FACTS_PER_PARAGRAPH:
                action_fact_ids.extend(event_fact_ids[:MAX_FACTS_PER_PARAGRAPH - len(action_fact_ids)])
            response_fact_ids = list(exposition_fact_ids[MAX_FACTS_PER_PARAGRAPH:])
            response_fact_ids.extend(fact_id for fact_id in event_fact_ids if fact_id not in action_fact_ids)
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
                required_fact_ids=action_fact_ids,
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
            # An event may legitimately produce several independent facts.
            # They remain one causal response sequence, but no visual
            # paragraph is allowed to become a fact list merely because all
            # of them share the same event.
            response_groups = [
                response_fact_ids[start:start + MAX_FACTS_PER_PARAGRAPH]
                for start in range(0, len(response_fact_ids), MAX_FACTS_PER_PARAGRAPH)
            ] or [[]]
            for response_index, response_group in enumerate(response_groups):
                response_actor_id = next(
                    (
                        facts_by_id[fact_id].subject_id
                        for fact_id in response_group
                        if fact_id in facts_by_id and facts_by_id[fact_id].subject_id in scene.participant_ids
                    ),
                    actor_id,
                )
                add_dramatic_paragraph(
                    "decision" if event.decision and response_index == 0 else "perception_reaction",
                    "choice" if event.decision and response_index == 0 else "reaction",
                    actor_id=response_actor_id,
                    objective=event.decision or event.action,
                    pressure=event.obstacle if response_index == 0 else "",
                    observable_action=(
                        "写出人物在压力后的具体回应、取舍或下一步动作"
                        if response_index == 0
                        else "延续同一事件的结果，让尚未落地的状态通过可见反应成立"
                    ),
                    focal_detail="让视角人物的感知或身体反应改变其判断",
                    state_change=event.decision or "人物对局势作出可见回应，结果开始成立",
                    required_fact_ids=response_group,
                )
            covered_fact_ids.update(exposition_fact_ids)
            covered_fact_ids.update(event_fact_ids)
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
        missing_fact_ids = [item.fact_id for item in local_facts if item.fact_id not in context_fact_ids and item.fact_id not in covered_fact_ids]
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

        paragraphs = _consolidate_event_action_response_paragraphs(paragraphs)
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
            context_fact_contracts=context_fact_contracts,
        ))
    compiled_mechanism_ids = {
        mechanism_id
        for scene_program in programs
        for beat in scene_program.narrative_beats
        for mechanism_id in beat.mechanism_ids
    }
    missing_mechanisms = {
        item.mechanism_id for item in annotation.narrative_mechanisms
    } - compiled_mechanism_ids
    if missing_mechanisms:
        raise ValueError(
            "以下叙事机制没有进入任何写作节拍：" + "、".join(sorted(missing_mechanisms))
        )
    program = ChapterProgram(
        annotation.chapter_id,
        annotation.source_hash,
        tuple(programs),
        schema_version="2.1" if annotation.narrative_mechanisms else "2.0",
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
    macro_paths = build_macro_plan_work(author_id, work_id)
    event_payload = read_json(macro_paths["event_graph"])
    if not isinstance(event_payload, dict):
        raise ValueError("event graph must be a JSON object")
    event_graph = EventGraph.from_dict(event_payload)
    contracts = load_chapter_contracts(macro_paths["chapter_contracts"], event_graph=event_graph)
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
    contract = contracts.get(program.chapter_id)
    if contract is None:
        raise ValueError("章节缺少宏观章节合同")
    validate_program_chapter_contract(program, contract, event_graph)
    structural_quality = assess_chapter_program(program)
    length_quality = assess_length_plan(program)
    target = runs_dir(author_id, run_id) / "chapter_programs"
    base = target / f"chapter-{chapter_number:04d}"
    program_path = base.with_suffix(".program.json")
    quality_path = base.with_suffix(".quality.json")
    llm_path = base.with_suffix(".model-contract.zh.json")
    batches_path = base.with_suffix(".batches.json")
    macro_contract_path = base.with_suffix(".chapter-contract.json")
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
    write_json(macro_contract_path, contract.to_dict())
    return {"program": program_path, "quality": quality_path, "model_contract": llm_path, "batches": batches_path, "chapter_contract": macro_contract_path}


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
    macro_paths = build_macro_plan_work(author_id, work_id)
    event_payload = read_json(macro_paths["event_graph"])
    if not isinstance(event_payload, dict):
        raise ValueError("event graph must be a JSON object")
    event_graph = EventGraph.from_dict(event_payload)
    contracts = load_chapter_contracts(macro_paths["chapter_contracts"], event_graph=event_graph)
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
        macro_contract = contracts.get(program.chapter_id)
        if macro_contract is None:
            raise ValueError("章节缺少宏观章节合同")
        validate_program_chapter_contract(program, macro_contract, event_graph)
        structural_quality = assess_chapter_program(program)
        length_quality = assess_length_plan(program)
        base = target / f"chapter-{number:04d}"
        program_path = base.with_suffix(".program.json")
        quality_path = base.with_suffix(".quality.json")
        contract_path = base.with_suffix(".model-contract.zh.json")
        batches_path = base.with_suffix(".batches.json")
        macro_contract_path = base.with_suffix(".chapter-contract.json")
        write_json(program_path, program.to_dict())
        write_json(quality_path, {"annotation": annotation_quality, "structural": structural_quality, "length_plan": length_quality})
        write_json(contract_path, chapter_program_to_llm_dict(program))
        batches = [
            {"scene_id": scene_id, "paragraph_ids": list(paragraph_ids)}
            for scene_id, paragraph_ids in iter_generation_batches(program)
        ]
        write_json(batches_path, {"schema_version": "1.0", "batch_paragraph_limit": 6, "batches": batches})
        write_json(macro_contract_path, macro_contract.to_dict())
        index_rows.append({
            "chapter_no": number,
            "chapter_id": program.chapter_id,
            "program": str(program_path),
            "quality": str(quality_path),
            "batches": str(batches_path),
            "chapter_contract": str(macro_contract_path),
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
        "event_graph": str(macro_paths["event_graph"]),
        "work_story_plan": str(macro_paths["story_plan"]),
        "chapter_contracts": str(macro_paths["chapter_contracts"]),
        "chapters": index_rows,
        "passed": all(bool(item["passed"]) for item in index_rows),
    })
    return index_path
