"""Length-driven, evidence-bounded chapter planning.

The deterministic scene compiler answers *what must happen*.  This module
answers *how much writable work is required* without inventing new plot facts.
It expands only the observable process around already licensed scene facts.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

from .contracts import ChapterProgram, NarrativeBeat, ParagraphProgram, SceneProgram


# A Chinese web-fiction paragraph normally carries a complete micro-movement
# rather than a single sentence.  The old 64-character default inflated a
# 6k chapter into almost one hundred requests and made the assembled prose
# visibly choppy.  Contracts still retain sub-paragraph beats; the writer now
# realises several compatible beats inside a readable 140--180-character
# paragraph.
DEFAULT_PARAGRAPH_TARGET_CHARS = 160
MIN_PARAGRAPH_TARGET_CHARS = 90
DEFAULT_BATCH_PARAGRAPHS = 6


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _scene_weight(scene: SceneProgram) -> int:
    """Give concrete events more room than static facts."""

    return max(1, len(scene.event_programs) * 3 + len(scene.fact_contracts))


def _allocate_counts(scenes: tuple[SceneProgram, ...], target_count: int) -> list[int]:
    """Allocate paragraph slots while never deleting a source-bound paragraph."""

    counts = [len(scene.paragraphs) for scene in scenes]
    target_count = max(target_count, sum(counts))
    weights = [_scene_weight(scene) for scene in scenes]
    while sum(counts) < target_count:
        total_weight = sum(weights)
        # Prefer the scene currently furthest below its proportional share.
        index = max(
            range(len(scenes)),
            key=lambda item: (weights[item] / total_weight * target_count - counts[item], -item),
        )
        counts[index] += 1
    return counts


def _expansion_shape(index: int) -> tuple[str, str, str, str]:
    """Return function, beat type, licence and a bounded prose instruction."""

    shapes = (
        ("perception_reaction", "reaction", "inference", "让视角人物对既有动作作出可观察的感知或身体反应，不新增事实。"),
        ("action_progression", "action", "inference", "把已知行动拆成一个可见的准备、停顿或回应，不改变事件结果。"),
        ("aftermath", "aftermath", "inference", "写出既有结果留下的局面余波，只承接已存在的状态。"),
        ("transition", "transition", "atmosphere", "用环境、距离或注意力变化承接既有状态，不新增人物、设定或事件。"),
    )
    return shapes[index % len(shapes)]


def _anchor_fact(scene: SceneProgram, beat: NarrativeBeat | None) -> str:
    if beat:
        supported = _unique((*beat.required_fact_ids, *beat.support_fact_ids))
        if supported:
            return supported[0]
    if scene.exit_state_fact_ids:
        return scene.exit_state_fact_ids[0]
    if scene.entry_state_fact_ids:
        return scene.entry_state_fact_ids[0]
    return scene.fact_contracts[0].fact_id


def _beat_by_id(scene: SceneProgram) -> dict[str, NarrativeBeat]:
    return {beat.beat_id: beat for beat in scene.narrative_beats}


def _expand_scene(scene: SceneProgram, desired_count: int) -> SceneProgram:
    """Add inference/atmosphere beats, each attached to a local hard fact."""

    original = list(sorted(scene.paragraphs, key=lambda item: (item.order, item.paragraph_id)))
    extra = max(0, desired_count - len(original))
    if not extra:
        return scene
    beat_by_id = _beat_by_id(scene)
    distribution = [0] * len(original)
    for item in range(extra):
        distribution[item % len(original)] += 1

    paragraphs: list[ParagraphProgram] = []
    added_beats: list[NarrativeBeat] = []
    for source_index, paragraph in enumerate(original):
        paragraphs.append(paragraph)
        anchor = next((beat_by_id.get(beat_id) for beat_id in paragraph.beat_ids if beat_id in beat_by_id), None)
        actor_id = paragraph.viewpoint_entity_id or (anchor.actor_id if anchor else scene.participant_ids[0])
        fact_id = _anchor_fact(scene, anchor)
        for ordinal in range(distribution[source_index]):
            function, beat_type, license_level, action = _expansion_shape(len(added_beats))
            beat_id = f"{scene.scene_id}:expansion-{len(added_beats) + 1:03d}"
            added_beats.append(NarrativeBeat(
                beat_id=beat_id,
                scene_id=scene.scene_id,
                order=0,
                beat_type=beat_type,
                actor_id=actor_id,
                objective=scene.objective,
                pressure=anchor.pressure if anchor else "维持当前已知局势压力",
                observable_action=action,
                focal_detail="只展开当前场景已经可见的人物动作、感知、环境或对白反应。",
                state_change="不新增硬事实；使既有状态在正文中获得可感知的过程。",
                expansion_license=license_level,
                support_fact_ids=(fact_id,),
            ))
            paragraphs.append(ParagraphProgram(
                paragraph_id=f"{scene.scene_id}:expansion-paragraph-{len(added_beats):03d}",
                scene_id=scene.scene_id,
                order=0,
                function=function,
                viewpoint_entity_id=actor_id,
                beat_ids=(beat_id,),
            ))

    all_beats = list(scene.narrative_beats) + added_beats
    beat_order = {beat.beat_id: index for index, beat in enumerate(all_beats)}
    # Order beats by the paragraph that realises them.  This is vital when a
    # model receives the contract in batches: it must not see a later reaction
    # before the action it expands.
    paragraph_beat_ids = [beat_id for paragraph in paragraphs for beat_id in paragraph.beat_ids]
    ordered_ids = list(dict.fromkeys(paragraph_beat_ids + [beat.beat_id for beat in all_beats]))
    ordered_beats = [next(beat for beat in all_beats if beat.beat_id == beat_id) for beat_id in ordered_ids]
    ordered_beats = [replace(beat, order=index) for index, beat in enumerate(ordered_beats)]
    del beat_order
    return replace(
        scene,
        paragraphs=tuple(replace(paragraph, order=index) for index, paragraph in enumerate(paragraphs)),
        narrative_beats=tuple(ordered_beats),
    )


def _apply_character_budgets(program: ChapterProgram, target_chars: int) -> ChapterProgram:
    paragraphs = [paragraph for scene in program.scene_programs for paragraph in scene.paragraphs]
    if not paragraphs:
        raise ValueError("章节程序缺少可分配字数的段落")
    base, remainder = divmod(target_chars, len(paragraphs))
    targets = [base + (1 if index < remainder else 0) for index in range(len(paragraphs))]
    offset = 0
    scenes: list[SceneProgram] = []
    for scene in program.scene_programs:
        revised: list[ParagraphProgram] = []
        for paragraph in scene.paragraphs:
            target = max(MIN_PARAGRAPH_TARGET_CHARS, targets[offset])
            revised.append(replace(paragraph, target_chars=target, minimum_chars=max(70, round(target * 0.7))))
            offset += 1
        scenes.append(replace(scene, paragraphs=tuple(revised)))
    return replace(program, scene_programs=tuple(scenes))


def plan_controlled_expansion(
    program: ChapterProgram,
    *,
    target_char_min: int,
    target_char_max: int,
    paragraph_target_chars: int = DEFAULT_PARAGRAPH_TARGET_CHARS,
) -> ChapterProgram:
    """Return a schema-3 program that can fill a bounded long-form chapter.

    The new beats never add a plot result.  They only licence observable
    process around a scene-local fact, and are therefore safe for later
    deterministic auditing.
    """

    program.validate()
    target_char_min, target_char_max = int(target_char_min), int(target_char_max)
    if target_char_min < 1 or target_char_min > target_char_max:
        raise ValueError("受控扩写要求有效的目标字数区间")
    paragraph_target_chars = max(MIN_PARAGRAPH_TARGET_CHARS, int(paragraph_target_chars))
    target_midpoint = round((target_char_min + target_char_max) / 2)
    desired_paragraphs = max(
        sum(len(scene.paragraphs) for scene in program.scene_programs),
        (target_midpoint + paragraph_target_chars - 1) // paragraph_target_chars,
    )
    counts = _allocate_counts(program.scene_programs, desired_paragraphs)
    scenes = tuple(_expand_scene(scene, count) for scene, count in zip(program.scene_programs, counts))
    planned = ChapterProgram(
        chapter_id=program.chapter_id,
        source_hash=program.source_hash,
        scene_programs=scenes,
        schema_version="3.0",
        entities=program.entities,
        generation_mode="controlled_expansion",
        target_char_min=target_char_min,
        target_char_max=target_char_max,
    )
    planned = _apply_character_budgets(planned, target_midpoint)
    planned.validate()
    return planned


def assess_length_plan(program: ChapterProgram) -> dict[str, Any]:
    """Return a deterministic quality gate for a chapter length plan."""

    program.validate()
    paragraphs = [paragraph for scene in program.scene_programs for paragraph in scene.paragraphs]
    beats = [beat for scene in program.scene_programs for beat in scene.narrative_beats]
    planned_chars = sum(paragraph.target_chars for paragraph in paragraphs)
    issues: list[dict[str, str]] = []
    if program.generation_mode == "controlled_expansion":
        if not (program.target_char_min <= planned_chars <= program.target_char_max):
            issues.append({"code": "CHAR_BUDGET_OUT_OF_RANGE", "message": "段落目标字数之和未落在章节目标区间内"})
        missing_budget = [paragraph.paragraph_id for paragraph in paragraphs if paragraph.target_chars <= 0 or paragraph.minimum_chars <= 0]
        if missing_budget:
            issues.append({"code": "PARAGRAPH_BUDGET_MISSING", "message": "存在未设置目标字数或最低字数的段落"})
        unlicensed = [beat.beat_id for beat in beats if beat.expansion_license != "source" and not beat.support_fact_ids]
        if unlicensed:
            issues.append({"code": "EXPANSION_LICENSE_MISSING", "message": "存在没有依据事实的扩写节拍"})
    return {
        "schema_version": "1.0",
        "chapter_id": program.chapter_id,
        "generation_mode": program.generation_mode,
        "planned_char_count": planned_chars,
        "target_char_min": program.target_char_min,
        "target_char_max": program.target_char_max,
        "paragraph_count": len(paragraphs),
        "dramatic_beat_count": len(beats),
        "expanded_beat_count": sum(beat.expansion_license != "source" for beat in beats),
        "passed": not issues,
        "issues": issues,
    }


def iter_generation_batches(program: ChapterProgram, max_paragraphs: int = DEFAULT_BATCH_PARAGRAPHS) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Split only at paragraph boundaries; a batch never crosses scenes."""

    program.validate()
    max_paragraphs = max(1, int(max_paragraphs))
    batches: list[tuple[str, tuple[str, ...]]] = []
    for scene in program.scene_programs:
        paragraphs = tuple(item.paragraph_id for item in scene.paragraphs)
        for start in range(0, len(paragraphs), max_paragraphs):
            batches.append((scene.scene_id, paragraphs[start:start + max_paragraphs]))
    return tuple(batches)
