"""Generate prose from a chapter program one scene and one paragraph at a time."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Callable, TypeVar

from .chapter_runtime import (
    commit_paragraph_patch,
    complete_chapter_working_ledger,
    load_or_initialize_chapter_working_ledger,
    paragraph_obligations,
    write_chapter_working_ledger,
)
from .contracts import ChapterContract, ChapterProgram, EventGraph, NarrativeGraph, ParagraphDraft, ParagraphValidation, SceneAffordanceValidation, SceneDraft, SceneProseValidation, SceneValidation
from .contracts.program_llm import chapter_program_to_llm_dict
from .contracts.scene_output import (
    paragraph_validation_from_llm,
    paragraph_validation_to_llm_dict,
    scene_affordance_validation_from_llm,
    scene_affordance_validation_to_llm_dict,
    scene_draft_from_llm,
    scene_draft_to_llm_dict,
    scene_prose_validation_from_llm,
    scene_prose_validation_to_llm_dict,
    scene_repair_from_llm,
    scene_validation_from_llm,
    scene_validation_to_llm_dict,
)
from .jsonio import read_json, write_json
from .graph_runtime import (
    build_chapter_graph_patch,
    commit_graph_patch,
    load_or_initialize_runtime_graph,
    narrative_subgraph_for_paragraph,
    paragraph_writer_packet,
    validate_graph_patch,
    write_runtime_graph,
)
from .model import ModelSettings, complete_json
from .paths import runs_dir, validate_author_id
from .macro_planner import build_chapter_contracts, build_event_graph, build_work_story_plan, validate_program_chapter_contract
from .skill_runtime import _json_object, _render, _run_id, _selected_templates, _skill_files
from .style_execution import assess_style_budget, resolve_style_budget, style_execution_blueprint, style_execution_directives


DEFAULT_SCENE_MAX_REPAIRS = 2
DEFAULT_STYLE_REPAIR_ROUNDS = 2
# A writing request is deliberately one paragraph wide.  Planning and
# annotation may batch data, but a prose model must receive only the matching
# paragraph's factual neighbourhood, never a whole-source chapter or a union
# of later paragraph subgraphs.
DEFAULT_GENERATION_BATCH_PARAGRAPHS = 1
DEFAULT_PARAGRAPH_MAX_REPAIRS = 3
_Result = TypeVar("_Result")


class _ContractParseError(ValueError):
    """Keep invalid model JSON available to the owning runtime artifact."""

    def __init__(self, message: str, payloads: list[dict[str, Any]]):
        super().__init__(message)
        self.payloads = payloads
_DIALOGUE_SPAN = re.compile(r"[“\"‘「『](.*?)[”\"’」』]", flags=re.DOTALL)
_ROLE_MARKERS = {
    "转场": ("随后", "此时", "片刻后", "不久", "旋即", "继而", "当即"),
    "感知": ("看到", "看见", "望着", "听到", "感到", "察觉"),
}
_MARKER_REPLACEMENTS = {
    "转场": (("随后", "接着"), ("此时", "这一刻"), ("片刻后", "过了一会"), ("不久", "没过多久"), ("旋即", "转眼"), ("继而", "接下来"), ("当即", "立刻")),
    # These replacements preserve the local referent while avoiding the exact
    # marker set used by the style measurement.  Less direct verbs such as
    # “听到” and “感到” are deliberately left to the model because a safe
    # substitution depends on their surrounding syntax.
    "感知": (("看到", "目光落到"), ("看见", "视线落到"), ("望着", "目光落在"), ("察觉", "意识到")),
}
_ACTION_EVIDENCE = frozenset("走跑停站坐转抬低抬伸收握攥松推拉按敲退进避闪看听望扫落贴靠拔掀" )
_SENSORY_EVIDENCE = ("看", "听", "目光", "视线", "声", "气息", "手指", "掌心", "背脊", "心口", "呼吸", "脚步")
_SUMMARY_PAIRS = (
    ("得知", "决定"), ("发现", "决定"), ("知道", "决定"), ("说出", "于是"),
    ("告诉", "于是"), ("明白", "决定"),
)


def _program_context(program: ChapterProgram) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = chapter_program_to_llm_dict(program)
    return (
        {
            "契约版本": payload["契约版本"],
            "章节ID": payload["章节ID"],
            "生成模式": payload["生成模式"],
            "章节目标字数下限": payload["章节目标字数下限"],
            "章节目标字数上限": payload["章节目标字数上限"],
            "实体字典": payload["实体字典"],
        },
        list(payload["场景程序"]),
    )


def _default_templates(library: dict[str, Any]) -> list[dict[str, Any]]:
    order = {"micro": 0, "event": 1, "scene": 2, "chapter": 3, "macro": 4}
    candidates = [item for item in library.get("templates", []) if isinstance(item, dict)]
    candidates.sort(key=lambda item: (order.get(str(item.get("level", "")), 99), str(item.get("template_id", ""))))
    return candidates[:4]


def _paragraph_style_budgets(program: ChapterProgram, budget: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Turn chapter-level style targets into semantically safe paragraph jobs.

    A chapter-level dialogue or transition ratio is not useful to a generator by
    itself: it needs to know which existing paragraph may carry that function.
    The allocations below only use a scene's known participants, events and
    viewpoint, so a style target cannot require an invented speaker or fact.
    """

    metrics = {str(item.get("metric_id", "")): item for item in budget.get("metrics", []) if isinstance(item, dict)}
    paragraph_metric = metrics.get("mean_paragraph_chars", {})
    sentence_metric = metrics.get("mean_sentence_chars", {})
    short_metric = metrics.get("short_paragraph_ratio", {})
    paragraph_target = max(35, round(float(paragraph_metric.get("target", 60))))
    paragraph_maximum = max(paragraph_target, round(float(paragraph_metric.get("preferred_max", paragraph_target)) * 1.15))
    sentence_target = max(16, round(float(sentence_metric.get("target", 36))))
    all_paragraphs = [item for scene in program.scene_programs for item in scene.paragraphs]
    scene_by_id = {scene.scene_id: scene for scene in program.scene_programs}
    dramatic_paragraph_ids = {
        paragraph.paragraph_id
        for scene in program.scene_programs if scene.narrative_beats
        for paragraph in scene.paragraphs if paragraph.beat_ids
    }
    event_by_id = {
        event.event_id: event
        for scene in program.scene_programs
        for event in scene.event_programs
    }
    fact_by_id = {
        fact.fact_id: fact
        for scene in program.scene_programs
        for fact in scene.fact_contracts
    }

    # A participant count is not a dialogue licence.  Use the explicit event
    # frame emitted upstream, so new wording, genres and languages do not need
    # another verb list here.

    def event_fact_ids(event: Any) -> tuple[str, ...]:
        return tuple(dict.fromkeys((
            *event.precondition_fact_ids,
            *event.required_fact_ids,
            *event.outcome_fact_ids,
        )))

    def dialogue_capacity(paragraph: Any) -> int:
        speech_events = [
            event_by_id[event_id]
            for event_id in paragraph.event_ids
            if event_id in event_by_id
            and event_by_id[event_id].action_type == "speech"
            and len(event_by_id[event_id].participant_ids) >= 2
        ]
        explicit_dialogue = paragraph.function == "dialogue_conflict" or bool(paragraph.dialogue_act)
        if explicit_dialogue:
            speech_events.extend(
                event_by_id[event_id]
                for event_id in paragraph.event_ids
                if event_id in event_by_id and event_by_id[event_id] not in speech_events
            )
        fact_ids = tuple(dict.fromkeys((
            *paragraph.required_fact_ids,
            *(fact_id for event in speech_events for fact_id in event_fact_ids(event)),
        )))
        usable = [fact_by_id[fact_id] for fact_id in fact_ids if fact_id in fact_by_id]
        if not speech_events:
            return 0
        # Each preserved proposition earns enough direct-speech space to be
        # expressed naturally, but never turns a short source utterance into a
        # fabricated monologue just to meet an author-level ratio.
        return min(96, max(30, 30 * len(usable)))

    def target_count(metric_id: str) -> int:
        metric = metrics.get(metric_id, {})
        try:
            ratio = float(metric.get("target", 0))
        except (TypeError, ValueError):
            return 0
        if ratio <= 0 or not all_paragraphs:
            return 0
        return max(1, min(len(all_paragraphs), round(len(all_paragraphs) * ratio)))

    def take(candidates: list[Any], amount: int, used: set[str]) -> set[str]:
        if amount <= 0:
            return set()
        picked: list[str] = []
        for paragraph in candidates:
            if paragraph.paragraph_id not in used and paragraph.paragraph_id not in picked:
                picked.append(paragraph.paragraph_id)
            if len(picked) >= amount:
                break
        # A role may legitimately share a paragraph if the chapter program has
        # too few semantically suitable jobs.  This is preferable to assigning
        # it to a paragraph that cannot safely express the role.
        if len(picked) < amount:
            for paragraph in candidates:
                if paragraph.paragraph_id not in picked:
                    picked.append(paragraph.paragraph_id)
                if len(picked) >= amount:
                    break
        return set(picked)

    dialogue_candidates = sorted(
        (
            paragraph
            for paragraph in all_paragraphs
            if (
                dialogue_capacity(paragraph) > 0
            )
            and len(scene_by_id[paragraph.scene_id].participant_ids) >= 2
        ),
        key=lambda paragraph: (
            paragraph.function != "dialogue_conflict",
            not bool(paragraph.dialogue_act),
            paragraph.order,
            paragraph.paragraph_id,
        ),
    )
    ordered_scenes = sorted(program.scene_programs, key=lambda scene: (scene.order, scene.scene_id))
    transition_candidates = [
        min(scene.paragraphs, key=lambda paragraph: (paragraph.order, paragraph.paragraph_id))
        for scene in ordered_scenes[1:]
        if scene.paragraphs
    ]
    transition_candidates.extend(
        paragraph
        for paragraph in all_paragraphs
        if paragraph.function in {"transition", "aftermath", "action_progression"}
        and paragraph not in transition_candidates
    )
    perception_candidates = sorted(
        (paragraph for paragraph in all_paragraphs if paragraph.viewpoint_entity_id),
        key=lambda paragraph: (
            paragraph.function not in {"perception_reaction", "decision"},
            paragraph.function != "perception_reaction",
            paragraph.order,
            paragraph.paragraph_id,
        ),
    )
    short_target = max(0, round(float(short_metric.get("target", 0)) * len(all_paragraphs)))
    priority = {"turn": 0, "decision": 1, "transition": 2, "aftermath": 3, "perception_reaction": 4, "action_progression": 5, "orientation": 6, "dialogue_conflict": 7, "obstacle": 8}
    short_ids = {
        item.paragraph_id
        for item in sorted(all_paragraphs, key=lambda item: (priority.get(item.function, 99), item.paragraph_id))[:short_target]
    }
    role_ids: dict[str, set[str]] = {"对话": set(), "转场": set(), "感知": set()}
    occupied: set[str] = set()
    role_ids["对话"] = take(dialogue_candidates, target_count("dialogue_char_ratio"), occupied)
    occupied.update(role_ids["对话"])
    role_ids["转场"] = take(transition_candidates, target_count("transition_marker_ratio"), occupied)
    occupied.update(role_ids["转场"])
    role_ids["感知"] = take(perception_candidates, target_count("perception_marker_ratio"), occupied)
    def prose_target(paragraph: Any, is_short: bool) -> int:
        if int(getattr(paragraph, "target_chars", 0) or 0) > 0:
            return int(paragraph.target_chars)
        if is_short:
            return 32
        # A beat needs enough room for action, pressure/reaction and a change.
        # This is a prose floor, not a replacement for the author's measured
        # paragraph rhythm.
        return max(paragraph_target, 105) if paragraph.paragraph_id in dramatic_paragraph_ids else paragraph_target

    role_instructions = {
        "对话": "用一至两轮直接对话承担试探、确认、施压或决定；对白必须改变行动或判断，不得只复述事实。",
        "转场": "在段首或动作结果处用一次自然承接，如“随后”“此时”或“片刻后”，把上一状态带入当前压力。",
        "感知": "在动作或信息变化后写出视角人物可感知的反应，并让该反应影响下一步判断。",
    }
    result: dict[str, dict[str, Any]] = {}
    for paragraph in all_paragraphs:
        # A controlled-expansion program has already allocated each paragraph
        # to make the chapter budget exact.  Do not silently shrink it to a
        # style-driven short paragraph here.
        is_short = paragraph.paragraph_id in short_ids and not int(getattr(paragraph, "target_chars", 0) or 0)
        target = prose_target(paragraph, is_short)
        maximum = 35 if is_short else max(paragraph_maximum, round(target * 1.5))
        labels = [label for label, ids in role_ids.items() if paragraph.paragraph_id in ids]
        instructions = [role_instructions[label] for label in labels]
        if "对话" in labels:
            dialogue_minimum = dialogue_capacity(paragraph)
            instructions[labels.index("对话")] += f" 对白总字数不得少于{dialogue_minimum}字，且每个信息点必须对应当前写作包中的事实；不得为凑字数补写背景、时间、动机或听者。"
        result[paragraph.paragraph_id] = {
            "段落ID": paragraph.paragraph_id,
            "目标字数": target,
            "最大字数": maximum,
            "建议句数": 1 if is_short else max(2, round(target / sentence_target)),
            "短段": is_short,
            "职责标签": labels,
            "叙事职责": instructions,
        }
        planned_minimum = int(getattr(paragraph, "minimum_chars", 0) or 0)
        if planned_minimum:
            result[paragraph.paragraph_id]["最低字数"] = planned_minimum
        elif paragraph.paragraph_id in dramatic_paragraph_ids and not is_short:
            result[paragraph.paragraph_id]["最低字数"] = max(70, round(target * 0.72))
        if "对话" in labels:
            result[paragraph.paragraph_id]["最低对白字数"] = dialogue_capacity(paragraph)
    return result


def _paragraph_local_requirements(paragraph_id: str, rule: dict[str, Any]) -> dict[str, Any]:
    """Expose the current paragraph's measurable pre-commit duties only."""

    def integer(key: str, default: int = 0) -> int:
        try:
            return int(rule.get(key, default))
        except (TypeError, ValueError):
            return default

    labels = list(rule.get("职责标签", [])) if isinstance(rule.get("职责标签"), list) else []
    result: dict[str, Any] = {
        "段落ID": paragraph_id,
        # Supply the planned length as well as the hard gate.  A bare floor
        # encourages early stopping rather than deliberate scene expansion.
        "目标正文字符": integer("目标字数"),
        "最低正文字符": integer("最低字数"),
        "职责标签": labels,
    }
    if bool(rule.get("短段")):
        result["最高正文字符"] = integer("最大字数", 35)
    if "对话" in labels:
        result["最低直接对白字符"] = integer("最低对白字数", 16)
    return result


def _scene_blueprint(
    budget: dict[str, Any],
    scene_count: int,
    paragraph_count: int,
    paragraph_style_budgets: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Scale chapter-level soft targets without changing scene paragraph IDs."""

    result = json.loads(json.dumps(style_execution_blueprint(budget), ensure_ascii=False))
    divisor = max(scene_count, 1)
    for key, value in list(result.items()):
        if key == "paragraph_count" or not isinstance(value, dict):
            continue
        if "char_count" in key or key == "sentence_count":
            for field in ("target", "minimum", "maximum"):
                if isinstance(value.get(field), (int, float)):
                    value[field] = max(1, round(value[field] / divisor))
    result["paragraph_count"] = {"target": paragraph_count, "minimum": paragraph_count, "maximum": paragraph_count}
    metric_rows = {
        str(item.get("metric_id", "")): item
        for item in budget.get("metrics", []) if isinstance(item, dict)
    }
    paragraph_metric = metric_rows.get("mean_paragraph_chars", {})
    sentence_metric = metric_rows.get("mean_sentence_chars", {})
    short_metric = metric_rows.get("short_paragraph_ratio", {})
    if paragraph_metric:
        result["段落字符范围"] = {
            "目标": round(float(paragraph_metric.get("target", 0))),
            "下限": max(25, round(float(paragraph_metric.get("preferred_min", 0)) * 0.65)),
            "上限": max(35, round(float(paragraph_metric.get("preferred_max", 0)) * 1.15)),
        }
    if sentence_metric:
        result["句子字符范围"] = {
            "目标": round(float(sentence_metric.get("target", 0))),
            "下限": max(16, round(float(sentence_metric.get("preferred_min", 0)) * 0.7)),
            "上限": max(24, round(float(sentence_metric.get("preferred_max", 0)) * 1.1)),
        }
    if short_metric:
        result["短段落提醒"] = {
            "章节目标比例": round(float(short_metric.get("target", 0)), 3),
            "说明": "仅在转折、决断或情绪落点自然出现时使用短段；不得为凑数拆散因果。",
        }
    if paragraph_style_budgets:
        result["段落写作预算"] = list(paragraph_style_budgets.values())
    result["instruction"] = "只写当前场景的指定段落；事实、因果与段落 ID 不得为凑指标而改变。"
    return result


def _paragraph_minimum(blueprint: dict[str, Any], paragraph_count: int) -> int:
    # This is a validity floor, not a style target: short paragraphs may be
    # intentionally assigned by the paragraph-level budget.
    if isinstance(blueprint.get("段落写作预算"), list):
        return 20
    paragraph_range = blueprint.get("段落字符范围", {})
    if isinstance(paragraph_range, dict) and isinstance(paragraph_range.get("下限"), (int, float)):
        return max(20, min(35, int(paragraph_range["下限"])))
    length = blueprint.get("chapter_char_count", {})
    if not isinstance(length, dict):
        return 80
    target = length.get("minimum", 0)
    try:
        return max(20, min(160, round(float(target) / max(paragraph_count, 1) * 0.35)))
    except (TypeError, ValueError):
        return 80


def _complete_contract(
    system: str,
    prompt: str,
    settings: ModelSettings,
    parser: Callable[[dict[str, Any]], _Result],
    *,
    max_tokens: int,
    contract_attempts: int = 2,
    model_attempts: int = 1,
) -> _Result:
    """Use a strict Chinese prompt plus contract parsing before accepting model JSON."""

    error = ""
    invalid_payloads: list[dict[str, Any]] = []
    for _ in range(max(1, contract_attempts)):
        request = prompt if not error else prompt + f"\n\n上次 JSON 未通过契约校验：{error}。请只修复格式和 ID 对齐问题，重新输出完整 JSON。"
        payload = complete_json(system, request, settings, attempts=max(1, model_attempts), max_tokens=max_tokens)
        try:
            return parser(payload)
        except ValueError as exc:
            invalid_payloads.append(payload)
            error = str(exc)
    raise _ContractParseError(f"模型 JSON 连续不符合场景契约：{error}", invalid_payloads)


def _validation_targets(scene: Any, validation: SceneValidation) -> tuple[str, ...]:
    targets = set(validation.repair_paragraph_ids)
    for issue in validation.paragraph_issues:
        targets.add(issue.paragraph_id)
    for paragraph in scene.paragraphs:
        if set(paragraph.required_fact_ids) & set(validation.missing_fact_ids):
            targets.add(paragraph.paragraph_id)
        if set(paragraph.event_ids) & set(validation.missing_event_ids):
            targets.add(paragraph.paragraph_id)
    if validation.premature_event_ids and not targets:
        targets.update(item.paragraph_id for item in scene.paragraphs)
    return tuple(item.paragraph_id for item in scene.paragraphs if item.paragraph_id in targets)


def _replace_paragraphs(draft: SceneDraft, replacements: SceneDraft) -> SceneDraft:
    by_id = {item.paragraph_id: item for item in replacements.paragraphs}
    return SceneDraft(draft.scene_id, tuple(by_id.get(item.paragraph_id, item) for item in draft.paragraphs))


def _scene_text(draft: SceneDraft) -> str:
    return "\n\n".join(item.text for item in draft.paragraphs)


def _chapter_length_assessment(program: ChapterProgram, text: str) -> dict[str, Any]:
    """Apply the length range as a hard gate for controlled expansion only."""

    actual = len(text)
    required = program.generation_mode == "controlled_expansion"
    passed = (not required) or (program.target_char_min <= actual <= program.target_char_max)
    return {
        "generation_mode": program.generation_mode,
        "required": required,
        "target_char_min": program.target_char_min,
        "target_char_max": program.target_char_max,
        "actual_char_count": actual,
        "passed": passed,
    }


def _advance_generation_state(
    prior_state: dict[str, Any],
    scene: Any,
    draft: SceneDraft,
    entity_names: dict[str, str],
) -> dict[str, Any]:
    """Return a compact orchestration receipt, never inherited prose content.

    The receipt is saved with the chapter exit but is not rendered into a prose
    prompt.  Narrative continuity is carried by the committed graph patch.
    """

    state: dict[str, Any] = {}
    facts = {item.fact_id: item for item in scene.fact_contracts}
    exits: list[dict[str, str]] = []
    for fact_id in scene.exit_state_fact_ids:
        fact = facts.get(fact_id)
        if fact is None:
            continue
        subject = entity_names.get(fact.subject_id, fact.subject_id)
        object_name = entity_names.get(fact.object_id, fact.object_id) if fact.object_id else fact.value
        exits.append({"事实ID": fact_id, "主体": subject, "谓词": fact.predicate, "结果": object_name})
    finished = prior_state.get("已完成场景", []) if isinstance(prior_state, dict) else []
    if not isinstance(finished, list):
        finished = []
    if scene.scene_id not in finished:
        finished.append(scene.scene_id)
    state["已完成场景"] = finished
    state["上一场景出口状态"] = exits
    state["上一场景ID"] = scene.scene_id
    return state


def _batch_scene_payload(
    scene_payload: dict[str, Any],
    paragraph_ids: tuple[str, ...],
    *,
    include_entry_state: bool,
    include_exit_state: bool,
) -> dict[str, Any]:
    """Limit one request to its local jobs and the state it may legitimately use.

    Intermediate batches may inherit the generated tail of the prior batch, but
    must not be told the final scene state before they earn it in prose.
    """

    payload = json.loads(json.dumps(scene_payload, ensure_ascii=False))
    wanted = set(paragraph_ids)
    paragraphs = [item for item in payload.get("段落程序", []) if item.get("段落ID") in wanted]
    beat_ids = {beat_id for paragraph in paragraphs for beat_id in paragraph.get("节拍ID列表", [])}
    beats = [item for item in payload.get("戏剧节拍", []) if item.get("节拍ID") in beat_ids]
    fact_ids = {
        fact_id
        for paragraph in paragraphs for fact_id in paragraph.get("必须表达的事实ID", [])
    }
    fact_ids.update(fact_id for beat in beats for fact_id in beat.get("必须表达的事实ID", []))
    fact_ids.update(fact_id for beat in beats for fact_id in beat.get("依据事实ID", []))
    event_ids = {
        event_id
        for paragraph in paragraphs for event_id in paragraph.get("事件ID列表", [])
    }
    event_ids.update(event_id for beat in beats for event_id in beat.get("事件ID列表", []))
    payload["段落程序"] = paragraphs
    payload["戏剧节拍"] = beats
    payload["事实表达契约"] = [item for item in payload.get("事实表达契约", []) if item.get("事实ID") in fact_ids]
    payload["必现事件"] = [item for item in payload.get("必现事件", []) if item.get("事件ID") in event_ids]
    if not include_entry_state:
        payload["入场状态事实ID"] = []
    if not include_exit_state:
        payload["出场状态事实ID"] = []
    return payload


def _prose_contract_active(scene: Any) -> bool:
    """Return whether a scene was compiled by the dramatic-beat pipeline."""

    return bool(getattr(scene, "narrative_beats", ()))


def _local_prose_issues(scene: Any, draft: SceneDraft) -> dict[str, list[str]]:
    """Catch obvious synopsis-shaped paragraphs without prescribing keywords.

    This is intentionally a small independent guard, not a substitute for the
    model-based prose audit.  It only rejects a paragraph when its assigned
    beat has no visible action/reaction evidence *and* the text collapses a
    causal change into a summary-like construction.
    """

    if not _prose_contract_active(scene):
        return {}
    beat_by_id = {beat.beat_id: beat for beat in scene.narrative_beats}
    paragraph_by_id = {paragraph.paragraph_id: paragraph for paragraph in scene.paragraphs}
    issues: dict[str, list[str]] = {}
    for paragraph in draft.paragraphs:
        program = paragraph_by_id.get(paragraph.paragraph_id)
        if program is None:
            continue
        text = paragraph.text.strip()
        action_count = sum(character in _ACTION_EVIDENCE for character in text)
        has_sensory = any(marker in text for marker in _SENSORY_EVIDENCE)
        has_dialogue = bool(_DIALOGUE_SPAN.search(text))
        summary_pair = any(first in text and second in text for first, second in _SUMMARY_PAIRS)
        paragraph_issues: list[str] = []
        for beat_id in program.beat_ids:
            beat = beat_by_id.get(beat_id)
            if beat is None:
                continue
            if beat.dialogue_pressure and not has_dialogue:
                paragraph_issues.append(f"节拍 {beat_id} 要求对白交锋，但正文没有直接对白")
            if beat.beat_type in {"action", "pressure", "choice", "consequence"} and action_count == 0:
                paragraph_issues.append(f"节拍 {beat_id} 没有可观察动作，不能只交代结果")
            if beat.beat_type in {"reaction", "choice", "pressure"} and not has_sensory and action_count < 2:
                paragraph_issues.append(f"节拍 {beat_id} 缺少会影响判断的感知或身体反应")
            if summary_pair and action_count < 2 and not has_dialogue:
                paragraph_issues.append(f"节拍 {beat_id} 把因果压缩成“得知/决定/于是”式摘要，需演出中间过程")
        if paragraph_issues:
            issues[paragraph.paragraph_id] = list(dict.fromkeys(paragraph_issues))
    return issues


def _unlicensed_entity_issues(
    scene: Any,
    draft: SceneDraft,
    entity_names: dict[str, str],
) -> dict[str, list[str]]:
    """Reject a known chapter entity when the current scene never licenses it.

    This deterministic guard catches cross-scene leakage such as mentioning a
    later adversary in an opening domestic scene.  It deliberately does not
    police generic nouns; the independent model audit handles newly invented
    relationships, motives, times, resources, and names not in the dictionary.
    """

    allowed_ids = set(scene.participant_ids)
    for fact in scene.fact_contracts:
        allowed_ids.add(fact.subject_id)
        if fact.object_id:
            allowed_ids.add(fact.object_id)
    for event in scene.event_programs:
        allowed_ids.update(event.participant_ids)
    forbidden = {
        entity_id: name.strip()
        for entity_id, name in entity_names.items()
        if entity_id not in allowed_ids and len(name.strip()) >= 2
    }
    issues: dict[str, list[str]] = {}
    for paragraph in draft.paragraphs:
        leaked = [name for name in forbidden.values() if name in paragraph.text]
        if leaked:
            issues[paragraph.paragraph_id] = [
                f"出现当前场景未授权的章节实体：{'、'.join(dict.fromkeys(leaked))}"
            ]
    return issues


def _prose_repair_directives(local_issues: dict[str, list[str]]) -> list[str]:
    """Translate deterministic prose failures into paragraph-local repairs."""

    return [
        f"{paragraph_id}：{'；'.join(issues)}。保留既有事实和结果，补出动作、压力、反应或选择的过程。"
        for paragraph_id, issues in local_issues.items()
    ]


def _prose_validation_targets(scene: Any, validation: SceneProseValidation) -> tuple[str, ...]:
    targets = set(validation.repair_paragraph_ids) | set(validation.summary_paragraph_ids)
    for issue in validation.paragraph_issues:
        targets.add(issue.paragraph_id)
    for paragraph in scene.paragraphs:
        if set(paragraph.beat_ids) & set(validation.missing_beat_ids):
            targets.add(paragraph.paragraph_id)
    return tuple(paragraph.paragraph_id for paragraph in scene.paragraphs if paragraph.paragraph_id in targets)


def _paragraph_role_issues(
    draft: SceneDraft,
    paragraph_style_budgets: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    """Check only locally assigned, measurable style responsibilities.

    These checks are deliberately narrow.  They verify the visible evidence
    that the global metric measures, while the separate scene validator keeps
    ownership of facts, events and causality.
    """

    issues: dict[str, list[str]] = {}
    for paragraph in draft.paragraphs:
        rule = paragraph_style_budgets.get(paragraph.paragraph_id, {})
        labels = rule.get("职责标签", [])
        if not isinstance(labels, list):
            labels = []
        paragraph_issues: list[str] = []
        try:
            minimum = int(rule.get("最低字数", 0))
        except (TypeError, ValueError):
            minimum = 0
        if minimum and len(paragraph.text.strip()) < minimum:
            paragraph_issues.append(f"戏剧段落至少需要 {minimum} 字来展开过程，当前只有 {len(paragraph.text.strip())} 字")
        if bool(rule.get("短段")):
            try:
                maximum = int(rule.get("最大字数", 35))
            except (TypeError, ValueError):
                maximum = 35
            if len(paragraph.text) > maximum:
                paragraph_issues.append(f"短段职责要求不超过{maximum}字，当前为{len(paragraph.text)}字")
        if "对话" in labels:
            dialogue_chars = sum(len(item) for item in _DIALOGUE_SPAN.findall(paragraph.text))
            try:
                minimum = int(rule.get("最低对白字数", 16))
            except (TypeError, ValueError):
                minimum = 16
            if dialogue_chars < minimum:
                paragraph_issues.append(f"对话职责要求中文引号内对白不少于{minimum}字，当前为{dialogue_chars}字")
        if paragraph_issues:
            issues[paragraph.paragraph_id] = paragraph_issues
    return issues


def _role_repair_directives(role_issues: dict[str, list[str]]) -> list[str]:
    """Render failed local role checks as targeted Chinese repair instructions."""

    return [
        f"{paragraph_id}：{'；'.join(issues)}。这些是本轮必须满足的可检测职责。"
        for paragraph_id, issues in role_issues.items()
    ]


def _replace_non_role_marker(
    draft: SceneDraft,
    paragraph_style_budgets: dict[str, dict[str, Any]],
    label: str,
) -> tuple[SceneDraft, dict[str, str]] | None:
    """Make one audited lexical substitution outside a required style role."""

    for paragraph in draft.paragraphs:
        labels = paragraph_style_budgets.get(paragraph.paragraph_id, {}).get("职责标签", [])
        if isinstance(labels, list) and label in labels:
            continue
        for original, replacement in _MARKER_REPLACEMENTS.get(label, ()):
            if original not in paragraph.text:
                continue
            updated = ParagraphDraft(
                paragraph.paragraph_id,
                paragraph.text.replace(original, replacement, 1),
                paragraph.claimed_fact_ids,
                paragraph.claimed_event_ids,
            )
            return (
                _replace_paragraphs(draft, SceneDraft(draft.scene_id, (updated,))),
                {"paragraph_id": paragraph.paragraph_id, "label": label, "from": original, "to": replacement},
            )
    return None


def _metric_rows(assessment: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("metric_id", "")): item
        for item in assessment.get("metrics", []) if isinstance(item, dict)
    }


def _metric_direction(metric: dict[str, Any]) -> str:
    """Return the required correction direction for one assessed metric."""

    try:
        actual = float(metric.get("actual"))
        minimum = float(metric.get("preferred_min"))
        maximum = float(metric.get("preferred_max"))
    except (TypeError, ValueError):
        return "maintain"
    if actual < minimum:
        return "increase"
    if actual > maximum:
        return "decrease"
    return "maintain"


def _sentence_mean(text: str) -> float:
    sentences = [item.strip() for item in re.split(r"[。！？!?]+", text) if item.strip()]
    return sum(len(item) for item in sentences) / len(sentences) if sentences else 0.0


def _style_targets(
    scene: Any,
    draft: SceneDraft,
    assessment: dict[str, Any],
    paragraph_style_budgets: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, ...]:
    """Select only paragraphs whose local form can improve a failed style metric."""

    metrics = _metric_rows(assessment)
    targets: set[str] = set()
    mean_paragraph = metrics.get("mean_paragraph_chars", {})
    if mean_paragraph and not mean_paragraph.get("in_preferred_range", True):
        maximum = float(mean_paragraph.get("preferred_max", 10**9))
        minimum = float(mean_paragraph.get("preferred_min", 0))
        targets.update(item.paragraph_id for item in draft.paragraphs if len(item.text) > maximum or len(item.text) < minimum)
    mean_sentence = metrics.get("mean_sentence_chars", {})
    if mean_sentence and not mean_sentence.get("in_preferred_range", True):
        lower = float(mean_sentence.get("preferred_min", 0))
        targets.update(item.paragraph_id for item in draft.paragraphs if _sentence_mean(item.text) < lower)
    function_by_id = {item.paragraph_id: item.function for item in scene.paragraphs}

    def role_targets(label: str) -> set[str]:
        if not paragraph_style_budgets:
            return set()
        return {
            item.paragraph_id
            for item in draft.paragraphs
            if label in paragraph_style_budgets.get(item.paragraph_id, {}).get("职责标签", [])
        }

    dialogue_metric = metrics.get("dialogue_char_ratio", {})
    if dialogue_metric and not dialogue_metric.get("in_preferred_range", True):
        dialogue_roles = role_targets("对话")
        if _metric_direction(dialogue_metric) == "increase":
            targets.update(dialogue_roles)
        else:
            targets.update(item.paragraph_id for item in draft.paragraphs if item.paragraph_id in dialogue_roles and _DIALOGUE_SPAN.search(item.text))
        if not dialogue_roles:
            targets.update(
                item.paragraph_id
                for item in draft.paragraphs
                if function_by_id.get(item.paragraph_id) == "dialogue_conflict"
            )
    transition_metric = metrics.get("transition_marker_ratio", {})
    if transition_metric and not transition_metric.get("in_preferred_range", True):
        transition_roles = role_targets("转场")
        if _metric_direction(transition_metric) == "increase":
            targets.update(transition_roles)
        else:
            targets.update(
                item.paragraph_id
                for item in draft.paragraphs
                if item.paragraph_id not in transition_roles and any(marker in item.text for marker in _ROLE_MARKERS["转场"])
            )
        if not transition_roles and _metric_direction(transition_metric) == "increase":
            targets.update(item.paragraph_id for item in draft.paragraphs if function_by_id.get(item.paragraph_id) in {"transition", "action_progression", "aftermath"})
    perception_metric = metrics.get("perception_marker_ratio", {})
    if perception_metric and not perception_metric.get("in_preferred_range", True):
        perception_roles = role_targets("感知")
        if _metric_direction(perception_metric) == "increase":
            targets.update(perception_roles)
        else:
            targets.update(
                item.paragraph_id
                for item in draft.paragraphs
                if item.paragraph_id not in perception_roles and any(marker in item.text for marker in _ROLE_MARKERS["感知"])
            )
        if not perception_roles and _metric_direction(perception_metric) == "increase":
            targets.update(item.paragraph_id for item in draft.paragraphs if function_by_id.get(item.paragraph_id) in {"perception_reaction", "decision"})
    short_metric = metrics.get("short_paragraph_ratio", {})
    if short_metric and not short_metric.get("in_preferred_range", True):
        if _metric_direction(short_metric) == "decrease":
            targets.update(item.paragraph_id for item in draft.paragraphs if len(item.text) <= 35)
        else:
            candidates = [item.paragraph_id for item in scene.paragraphs if item.function in {"turn", "decision", "transition", "aftermath"}]
            if candidates:
                targets.add(candidates[-1])
    return tuple(item.paragraph_id for item in scene.paragraphs if item.paragraph_id in targets)


def _local_style_directives(budget: dict[str, Any], assessment: dict[str, Any]) -> list[str]:
    # Reserve room for concrete role instructions below; generic aggregate
    # directives alone cannot tell the model where a missing dialogue belongs.
    directives = list(style_execution_directives(budget, assessment, limit=3))
    metrics = _metric_rows(assessment)
    if not metrics.get("mean_sentence_chars", {}).get("in_preferred_range", True):
        directives.append("把同一因果单元合并为一到两个完整长句，避免用连续短句逐项解释。")
    if not metrics.get("short_paragraph_ratio", {}).get("in_preferred_range", True):
        if _metric_direction(metrics["short_paragraph_ratio"]) == "increase":
            directives.append("仅在本场景的决断、转折或余波落点，将一个段落收束为不超过三十五字的短段；不得删失事实。")
        else:
            directives.append("把非必要的短段扩展为完整因果单元，保留已分配短段职责的段落不动。")
    if not metrics.get("transition_marker_ratio", {}).get("in_preferred_range", True):
        if _metric_direction(metrics["transition_marker_ratio"]) == "increase":
            directives.append("在合适的行动结果或注意力切换处使用一次自然的承接标记，例如“随后”“此时”或“片刻后”。")
        else:
            directives.append("删除非职责段中重复的承接标记，保留已分配转场职责的段落及其因果衔接。")
    if not metrics.get("dialogue_char_ratio", {}).get("in_preferred_range", True):
        if _metric_direction(metrics["dialogue_char_ratio"]) == "increase":
            directives.append("只在已列出的参与者之间保留一轮能改变行动或判断的直接对白；不得为增加对白引入新人物、新信息或新事件。")
        else:
            directives.append("压缩不改变行动或判断的重复对白，但不得低于已分配对话职责的最低对白字数。")
    if not metrics.get("perception_marker_ratio", {}).get("in_preferred_range", True):
        if _metric_direction(metrics["perception_marker_ratio"]) == "increase":
            directives.append("在已有动作或信息变化后补出视角人物的可感知反应，并使该反应服务于已给定的下一步判断。")
        else:
            directives.append("删除非职责段中重复的感知标记，保留已分配感知职责的段落及其必要反应。")
    return directives[:8]


def _prefer_style_assessment(current: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Keep semantic-safe paragraph repairs only when their measured style improves."""

    if bool(candidate.get("passed")) and not bool(current.get("passed")):
        return True
    current_high = int(current.get("high_priority_in_range", 0))
    candidate_high = int(candidate.get("high_priority_in_range", 0))
    current_deviation = float(current.get("mean_relative_deviation", 1.0))
    candidate_deviation = float(candidate.get("mean_relative_deviation", 1.0))
    return candidate_high >= current_high and candidate_deviation + 0.02 < current_deviation


def _draft_from_dict(payload: dict[str, Any]) -> SceneDraft:
    paragraphs = payload.get("paragraphs", [])
    if not isinstance(paragraphs, list):
        raise ValueError("stored scene draft paragraphs must be a list")
    return SceneDraft(
        scene_id=str(payload.get("scene_id", "")),
        paragraphs=tuple(
            ParagraphDraft(
                paragraph_id=str(item.get("paragraph_id", "")),
                text=str(item.get("text", "")),
                claimed_fact_ids=tuple(str(value) for value in item.get("claimed_fact_ids", []) if str(value).strip()),
                claimed_event_ids=tuple(str(value) for value in item.get("claimed_event_ids", []) if str(value).strip()),
            )
            for item in paragraphs if isinstance(item, dict)
        ),
    )


def _stored_validation_to_llm_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep persisted runtime records out of the Chinese model-facing boundary."""

    issues = payload.get("paragraph_issues", []) if isinstance(payload.get("paragraph_issues", []), list) else []
    return {
        "场景ID": str(payload.get("scene_id", "")),
        "通过": bool(payload.get("reported_passed", False)),
        "已实现事实ID": list(payload.get("realized_fact_ids", [])),
        "缺失事实ID": list(payload.get("missing_fact_ids", [])),
        "已实现事件ID": list(payload.get("realized_event_ids", [])),
        "缺失事件ID": list(payload.get("missing_event_ids", [])),
        "提前泄露事件ID": list(payload.get("premature_event_ids", [])),
        "未授权断言段落ID": list(payload.get("unsupported_claim_paragraph_ids", [])),
        "段落问题": [
            {"段落ID": str(item.get("paragraph_id", "")), "问题": str(item.get("problem", "")), "修复要求": str(item.get("repair", ""))}
            for item in issues if isinstance(item, dict)
        ],
        "修复段落ID": list(payload.get("repair_paragraph_ids", [])),
    }


def _stored_prose_validation_to_llm_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate a persisted prose audit back to its model-facing boundary."""

    issues = payload.get("paragraph_issues", []) if isinstance(payload.get("paragraph_issues", []), list) else []
    return {
        "场景ID": str(payload.get("scene_id", "")),
        "正文性通过": bool(payload.get("reported_passed", False)),
        "已戏剧化节拍ID": list(payload.get("dramatized_beat_ids", [])),
        "缺失节拍ID": list(payload.get("missing_beat_ids", [])),
        "摘要化段落ID": list(payload.get("summary_paragraph_ids", [])),
        "段落问题": [
            {"段落ID": str(item.get("paragraph_id", "")), "问题": str(item.get("problem", "")), "修复要求": str(item.get("repair", ""))}
            for item in issues if isinstance(item, dict)
        ],
        "修复段落ID": list(payload.get("repair_paragraph_ids", [])),
    }


def _completed_scene(
    path: Path,
    scene: Any,
    minimum_chars: int,
    paragraph_style_budgets: dict[str, dict[str, Any]],
    entity_names: dict[str, str],
    committed_paragraph_ids: set[str],
) -> tuple[SceneDraft, dict[str, Any]] | None:
    """Resume only a scene whose durable result still satisfies the current contract."""

    payload = read_json(path)
    if not isinstance(payload, dict) or not bool(payload.get("passed")):
        return None
    draft_payload = payload.get("draft")
    if not isinstance(draft_payload, dict):
        return None
    try:
        draft = _draft_from_dict(draft_payload)
        draft.validate_for(scene, minimum_chars=minimum_chars)
    except ValueError:
        return None
    if _paragraph_role_issues(draft, paragraph_style_budgets):
        return None
    if _unlicensed_entity_issues(scene, draft, entity_names):
        return None
    if _prose_contract_active(scene) and (not bool(payload.get("prose_passed")) or _local_prose_issues(scene, draft)):
        return None
    if not {item.paragraph_id for item in draft.paragraphs}.issubset(committed_paragraph_ids):
        return None
    return draft, payload


def _load_chapter_contract(
    graph: NarrativeGraph | None,
    program: ChapterProgram,
    path: Path | None,
) -> tuple[EventGraph | None, ChapterContract | None]:
    """Load a persisted macro contract, or deterministically derive one.

    Direct one-chapter calls remain usable, while work-level planning persists
    the same contract next to every program.  In both cases the contract is
    validated against the exact graph and program hash before writing begins.
    """

    if graph is None:
        if path is not None:
            raise ValueError("章节合同需要 narrative_graph.json 作为验证底座")
        return None, None
    event_graph = build_event_graph(graph)
    if path is None:
        story_plan = build_work_story_plan(event_graph)
        contracts = build_chapter_contracts(graph, event_graph, story_plan)
        contract = next((item for item in contracts if item.chapter_id == program.chapter_id), None)
    else:
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ValueError("chapter contract must be a JSON object")
        if isinstance(payload.get("contracts"), list):
            candidates = [
                ChapterContract.from_dict(item)
                for item in payload["contracts"]
                if isinstance(item, dict) and str(item.get("chapter_id", "")) == program.chapter_id
            ]
            if len(candidates) != 1:
                raise ValueError("chapter contracts file does not contain exactly one matching chapter")
            contract = candidates[0]
        else:
            contract = ChapterContract.from_dict(payload)
    if contract is None:
        raise ValueError("当前章节没有可执行的章节合同")
    validate_program_chapter_contract(program, contract, event_graph)
    return event_graph, contract


def generate_from_program(
    author_id: str,
    program_path: Path,
    *,
    story_state_path: Path | None = None,
    chapter_plan_path: Path | None = None,
    style_budget_path: Path | None = None,
    graph_path: Path | None = None,
    chapter_contract_path: Path | None = None,
    runtime_graph_path: Path | None = None,
    run_id: str = "program-generated",
    max_repairs: int = DEFAULT_SCENE_MAX_REPAIRS,
) -> dict[str, str | int | bool]:
    """Generate and semantically repair a chapter program without whole-chapter rewrites."""

    if max_repairs < 0:
        raise ValueError("max_repairs must be non-negative")
    author_id = validate_author_id(author_id)
    program_payload = read_json(program_path)
    if not isinstance(program_payload, dict):
        raise ValueError("chapter program JSON must be an object")
    program = ChapterProgram.from_dict(program_payload)
    program.validate()
    if graph_path is None and program.generation_mode == "controlled_expansion":
        raise ValueError("受控扩写必须传入 narrative_graph.json；章节程序回退模式不能替代唯一事实底座")
    entity_names = {item.entity_id: item.canonical_name for item in program.entities}
    folder, profile, library, _, execution_spec = _skill_files(author_id)
    story_state = _json_object(story_state_path) if story_state_path else {}
    chapter_plan = _json_object(chapter_plan_path) if chapter_plan_path else {}
    selected = _selected_templates(library, chapter_plan) if chapter_plan else _default_templates(library)
    if not selected:
        raise ValueError("compiled skill has no templates available for scene generation")
    override = _json_object(style_budget_path) if style_budget_path else None
    budget = resolve_style_budget(execution_spec, chapter_plan.get("style_budget") if chapter_plan else None, override)
    paragraph_style_budgets = _paragraph_style_budgets(program, budget)
    settings = ModelSettings.from_environment()
    if not settings.api_key:
        raise RuntimeError("模型密钥不可用：请在 config.yaml 或环境变量中配置后重试")

    destination = runs_dir(author_id, _run_id(run_id)) / "program_generation"
    destination.mkdir(parents=True, exist_ok=True)
    graph: NarrativeGraph | None = None
    if graph_path is not None:
        graph_payload = read_json(graph_path)
        if not isinstance(graph_payload, dict):
            raise ValueError("narrative graph must be a JSON object")
        graph = NarrativeGraph.from_dict(graph_payload)
        graph.validate()
        graph_hash = graph.source_hashes.get(program.chapter_id, "")
        if graph_hash and graph_hash != program.source_hash:
            raise ValueError("chapter program source hash does not match the narrative graph")
    event_graph, chapter_contract = _load_chapter_contract(graph, program, chapter_contract_path)
    active_runtime_graph_path = runtime_graph_path or (destination / "runtime_graph_state.json")
    runtime_graph = load_or_initialize_runtime_graph(active_runtime_graph_path, graph, program)
    write_runtime_graph(active_runtime_graph_path, runtime_graph)
    working_ledger_path = destination / "chapter_working_ledger.json"
    working_ledger = load_or_initialize_chapter_working_ledger(
        working_ledger_path,
        program,
        contract=chapter_contract,
    )
    write_chapter_working_ledger(working_ledger_path, working_ledger)
    write_json(destination / "chapter_program.json", program.to_dict())
    if chapter_contract is not None:
        write_json(destination / "chapter_contract.json", chapter_contract.to_dict())
    write_json(destination / "style_budget.json", budget)
    context, scene_payloads = _program_context(program)
    if len(scene_payloads) != len(program.scene_programs):
        raise ValueError("章节程序的场景序列不完整")
    affordance_validations: list[SceneAffordanceValidation] = []
    # A macro-controlled generation must not ask the prose model to invent a
    # missing actor, relationship, object or prerequisite merely because an
    # event summary happened to mention one.  This preflight consumes only the
    # program's typed structure, never source prose.
    if graph is not None and chapter_contract is not None:
        for scene, scene_payload in zip(program.scene_programs, scene_payloads):
            affordance_prompt = _render(folder / "prompt_templates" / "scene_affordance_validate.md", {
                "CHAPTER_PROGRAM_CONTEXT_JSON": context,
                "SCENE_PROGRAM_JSON": scene_payload,
            })
            validation = _complete_contract(
                "你是中文小说场景结构预检器。只输出符合指定中文 JSON 契约的对象。",
                affordance_prompt,
                settings,
                lambda payload, target_scene=scene: scene_affordance_validation_from_llm(payload, target_scene),
                max_tokens=min(settings.max_tokens, 2_500),
            )
            affordance_validations.append(validation)
        affordance_path = destination / "scene_affordance_validation.json"
        write_json(affordance_path, {"passed": all(item.passed for item in affordance_validations), "scenes": [item.to_dict() for item in affordance_validations]})
        if not all(item.passed for item in affordance_validations):
            feedback_path = destination / "annotation_repair_feedback.json"
            write_json(feedback_path, {
                "schema_version": "1.0",
                "chapter_id": program.chapter_id,
                "source_hash": program.source_hash,
                "purpose": "补齐无法由当前事实图谱写出的实体、参与关系、状态或节拍依据；不得以生成正文反推原文。",
                "scenes": [item.to_dict() for item in affordance_validations if not item.passed],
            })
            write_json(destination / "generation_status.json", {
                "stage": "scene_affordance_validation", "status": "rejected",
                "reason": "章节程序存在无法仅凭事实图谱写出的事件或节拍",
            })
            chapter_exit_path = destination / "chapter_exit_state.json"
            write_json(chapter_exit_path, {
                "schema_version": "1.0", "chapter_id": program.chapter_id, "source_hash": program.source_hash,
                "generation_mode": program.generation_mode, "overall_passed": False,
                "graph_patch_passed": False, "graph_patch_committed": False,
                "runtime_graph_state": str(active_runtime_graph_path), "state": {},
                "reason": "scene_affordance_validation_failed",
                "annotation_repair_feedback": str(feedback_path),
            })
            write_json(destination / "generation_manifest.json", {
                "author_id": author_id,
                "program_path": str(program_path),
                "chapter_contract_path": str(chapter_contract_path) if chapter_contract_path else "",
                "stage": "scene_affordance_validation",
                "overall_passed": False,
                "annotation_repair_feedback": str(feedback_path),
            })
            return {
                "output_dir": str(destination), "scene_count": 0, "repairs_used": 0,
                "style_repairs_used": 0, "semantic_passed": False, "structural_passed": False,
                "prose_passed": False, "role_passed": False, "style_passed": False,
                "length_passed": False, "graph_patch_passed": False,
                "graph_patch_committed": False, "overall_passed": False,
                "chapter_exit_state": str(chapter_exit_path),
            }
    write_json(destination / "generation_status.json", {"stage": "scene_drafting", "status": "in_progress"})

    scene_results: list[dict[str, Any]] = []
    final_drafts: list[SceneDraft] = []
    generation_states: list[dict[str, Any]] = []
    for index, (scene, scene_payload) in enumerate(zip(program.scene_programs, scene_payloads), start=1):
        scene_paragraph_budgets = {
            item.paragraph_id: paragraph_style_budgets[item.paragraph_id]
            for item in scene.paragraphs if item.paragraph_id in paragraph_style_budgets
        }
        blueprint = _scene_blueprint(budget, len(program.scene_programs), len(scene.paragraphs), scene_paragraph_budgets)
        minimum_chars = _paragraph_minimum(blueprint, len(scene.paragraphs))
        batch_groups = tuple(
            tuple(item.paragraph_id for item in scene.paragraphs[start:start + DEFAULT_GENERATION_BATCH_PARAGRAPHS])
            for start in range(0, len(scene.paragraphs), DEFAULT_GENERATION_BATCH_PARAGRAPHS)
        )
        result_path = destination / f"scene-{index:02d}.result.json"
        completed = _completed_scene(
            result_path,
            scene,
            minimum_chars,
            scene_paragraph_budgets,
            entity_names,
            set(str(item) for item in working_ledger.get("committed_paragraph_ids", [])),
        )
        if completed is not None:
            draft, result = completed
            result = {**result, "resumed": True}
            scene_results.append(result)
            final_drafts.append(draft)
            story_state = _advance_generation_state(story_state, scene, draft, entity_names)
            generation_states.append({"scene_no": index, "scene_id": scene.scene_id, "state": story_state})
            continue
        write_json(destination / "generation_status.json", {
            "stage": "scene_drafting", "status": "in_progress", "scene_no": index, "scene_id": scene.scene_id,
        })
        partial_path = destination / f"scene-{index:02d}.draft.partial.json"
        partial_payload = read_json(partial_path)
        try:
            draft = _draft_from_dict(partial_payload) if isinstance(partial_payload, dict) else None
            if draft is not None:
                draft.validate_for(scene, minimum_chars=minimum_chars)
                committed = set(str(item) for item in working_ledger.get("committed_paragraph_ids", []))
                if not {item.paragraph_id for item in draft.paragraphs}.issubset(committed):
                    draft = None
        except ValueError:
            draft = None
        paragraph_validations: list[dict[str, Any]] = []
        if draft is None:
            batch_drafts: list[SceneDraft] = []
            for batch_no, paragraph_ids in enumerate(batch_groups, start=1):
                batch_path = destination / f"scene-{index:02d}.batch-{batch_no:02d}.draft.partial.json"
                audit_path = destination / f"scene-{index:02d}.batch-{batch_no:02d}.paragraph-validation.json"
                stored_batch = read_json(batch_path)
                paragraphs_by_id = {item.paragraph_id: item for item in scene.paragraphs}
                paragraph = paragraphs_by_id[paragraph_ids[0]]
                already_committed = paragraph.paragraph_id in set(str(item) for item in working_ledger.get("committed_paragraph_ids", []))
                try:
                    batch_draft = _draft_from_dict(stored_batch) if isinstance(stored_batch, dict) else None
                    if batch_draft is not None:
                        batch_draft.validate_for(scene, expected_paragraph_ids=paragraph_ids, minimum_chars=minimum_chars)
                except ValueError:
                    batch_draft = None
                if already_committed and batch_draft is None:
                    raise ValueError("已提交段落缺少对应正文，不能在不一致状态下继续生成")
                batch_budgets = {paragraph_id: scene_paragraph_budgets[paragraph_id] for paragraph_id in paragraph_ids}
                batch_blueprint = _scene_blueprint(budget, len(batch_groups), len(paragraph_ids), batch_budgets)
                local_requirements = _paragraph_local_requirements(
                    paragraph.paragraph_id,
                    scene_paragraph_budgets[paragraph.paragraph_id],
                )
                writer_packet = paragraph_writer_packet(
                    graph,
                    runtime_graph,
                    program,
                    scene,
                    paragraph,
                    chapter_contract=chapter_contract,
                    event_graph=event_graph,
                    working_ledger=working_ledger,
                )
                if batch_draft is None:
                    if len(paragraph_ids) != 1:
                        raise RuntimeError("正文生成必须保持单段写作包；请勿增大 DEFAULT_GENERATION_BATCH_PARAGRAPHS")
                    draft_prompt = _render(folder / "prompt_templates" / "scene_draft.md", {
                        "AUTHOR_STYLE_PROFILE_JSON": profile,
                        "SELECTED_TEMPLATES_JSON": selected,
                        "STYLE_EXECUTION_BLUEPRINT_JSON": batch_blueprint,
                        "PARAGRAPH_LOCAL_REQUIREMENTS_JSON": local_requirements,
                        "PARAGRAPH_WRITER_PACKET_JSON": writer_packet,
                    })
                    try:
                        batch_draft = _complete_contract(
                            "你是中文小说场景写作者。只输出符合指定中文 JSON 契约的对象。",
                            draft_prompt,
                            settings,
                            lambda payload: scene_draft_from_llm(
                                payload, scene, minimum_chars=minimum_chars, expected_paragraph_ids=paragraph_ids,
                            ),
                            max_tokens=min(settings.max_tokens, max(3_000, 1_200 * len(paragraph_ids))),
                        )
                    except Exception as exc:
                        write_json(destination / "generation_status.json", {
                            "stage": "scene_drafting", "status": "failed", "scene_no": index, "scene_id": scene.scene_id,
                            "batch_no": batch_no, "error": f"{type(exc).__name__}: {exc}",
                        })
                        raise
                    write_json(batch_path, batch_draft.to_dict())
                    if not already_committed:
                        expected_fact_ids, expected_event_ids = paragraph_obligations(scene, paragraph)
                        beat_by_id = {item.beat_id: item for item in scene.narrative_beats}
                        expected_mechanism_ids = tuple(dict.fromkeys(
                            mechanism_id
                            for beat_id in paragraph.beat_ids
                            for mechanism_id in (
                                beat_by_id[beat_id].mechanism_ids if beat_id in beat_by_id else ()
                            )
                        ))
                        paragraph_validation: ParagraphValidation | None = None
                    paragraph_attempts: list[dict[str, Any]] = []
                    for audit_attempt in range(DEFAULT_PARAGRAPH_MAX_REPAIRS + 1):
                        audit_prompt = _render(folder / "prompt_templates" / "paragraph_validate.md", {
                            "PARAGRAPH_WRITER_PACKET_JSON": writer_packet,
                            "PARAGRAPH_DRAFT_JSON": scene_draft_to_llm_dict(batch_draft),
                            "PARAGRAPH_OBLIGATION_IDS_JSON": {
                                "必须核验事实ID": list(expected_fact_ids),
                                "必须核验事件ID": list(expected_event_ids),
                                "必须核验叙事机制ID": list(expected_mechanism_ids),
                            },
                        })
                        try:
                            paragraph_validation = _complete_contract(
                                "你是中文小说段落语义校验器。只输出符合指定中文 JSON 契约的对象。",
                                audit_prompt,
                                settings,
                                lambda payload: paragraph_validation_from_llm(
                                    payload,
                                    scene,
                                    paragraph_id=paragraph.paragraph_id,
                                    expected_fact_ids=expected_fact_ids,
                                    expected_event_ids=expected_event_ids,
                                    expected_mechanism_ids=expected_mechanism_ids,
                                ),
                                max_tokens=min(settings.max_tokens, 2_500),
                            )
                        except _ContractParseError as exc:
                            write_json(audit_path, {
                                "passed": False,
                                "attempts": paragraph_attempts,
                                "invalid_model_outputs": exc.payloads,
                                "error": str(exc),
                            })
                            raise
                        single_paragraph_draft = SceneDraft(scene.scene_id, (batch_draft.paragraphs[0],))
                        role_issues_now = _paragraph_role_issues(
                            single_paragraph_draft,
                            {paragraph.paragraph_id: scene_paragraph_budgets[paragraph.paragraph_id]},
                        )
                        prose_issues_now = _local_prose_issues(scene, single_paragraph_draft)
                        entity_issues_now = _unlicensed_entity_issues(scene, single_paragraph_draft, entity_names)
                        local_gate_issues = [
                            *role_issues_now.get(paragraph.paragraph_id, []),
                            *prose_issues_now.get(paragraph.paragraph_id, []),
                            *entity_issues_now.get(paragraph.paragraph_id, []),
                        ]
                        if paragraph_validation.passed and local_gate_issues:
                            # The semantic auditor accounts for IDs, while
                            # this deterministic local gate accounts for
                            # measurable paragraph duties before the text is
                            # allowed to become a hand-off.
                            paragraph_validation = ParagraphValidation(
                                paragraph_id=paragraph_validation.paragraph_id,
                                reported_passed=False,
                                realized_fact_ids=paragraph_validation.realized_fact_ids,
                                missing_fact_ids=paragraph_validation.missing_fact_ids,
                                realized_event_ids=paragraph_validation.realized_event_ids,
                                missing_event_ids=paragraph_validation.missing_event_ids,
                                unsupported_assertions=paragraph_validation.unsupported_assertions,
                                issues=tuple(f"局部正文门槛：{item}" for item in local_gate_issues),
                                realized_mechanism_ids=paragraph_validation.realized_mechanism_ids,
                                missing_mechanism_ids=paragraph_validation.missing_mechanism_ids,
                            )
                            paragraph_validation.validate_for(
                                scene,
                                expected_fact_ids=expected_fact_ids,
                                expected_event_ids=expected_event_ids,
                                expected_mechanism_ids=expected_mechanism_ids,
                            )
                        attempt_record = {"attempt": audit_attempt, **paragraph_validation.to_dict()}
                        paragraph_attempts.append(attempt_record)
                        paragraph_validations.append(attempt_record)
                        if paragraph_validation.passed:
                            break
                        if audit_attempt >= DEFAULT_PARAGRAPH_MAX_REPAIRS:
                            break
                        repair_prompt = _render(folder / "prompt_templates" / "paragraph_repair.md", {
                            "PARAGRAPH_WRITER_PACKET_JSON": writer_packet,
                            "PARAGRAPH_DRAFT_JSON": scene_draft_to_llm_dict(batch_draft),
                            "PARAGRAPH_VALIDATION_JSON": paragraph_validation_to_llm_dict(paragraph_validation),
                            "STYLE_EXECUTION_BLUEPRINT_JSON": batch_blueprint,
                            "PARAGRAPH_LOCAL_REQUIREMENTS_JSON": local_requirements,
                        })
                        batch_draft = _complete_contract(
                            "你是中文小说单段定点修复器。只输出符合指定中文 JSON 契约的对象。",
                            repair_prompt,
                            settings,
                            lambda payload: scene_draft_from_llm(
                                payload, scene, minimum_chars=minimum_chars, expected_paragraph_ids=paragraph_ids,
                            ),
                            max_tokens=min(settings.max_tokens, 3_000),
                        )
                        write_json(batch_path, batch_draft.to_dict())
                    if paragraph_validation is None or not paragraph_validation.passed:
                        write_json(audit_path, {"passed": False, "attempts": paragraph_attempts})
                        write_json(destination / "generation_status.json", {
                            "stage": "paragraph_validation", "status": "failed", "scene_no": index,
                            "scene_id": scene.scene_id, "batch_no": batch_no,
                        })
                        raise RuntimeError("当前段落未通过语义校验，禁止把候选状态传给下一段")
                    working_ledger = commit_paragraph_patch(
                        working_ledger,
                        program,
                        scene,
                        paragraph,
                        batch_draft.paragraphs[0],
                        paragraph_validation,
                    )
                    write_chapter_working_ledger(working_ledger_path, working_ledger)
                    write_json(audit_path, {"passed": True, "attempts": paragraph_attempts})
                batch_drafts.append(batch_draft)
            draft = SceneDraft(scene.scene_id, tuple(paragraph for batch in batch_drafts for paragraph in batch.paragraphs))
            draft.validate_for(scene, minimum_chars=minimum_chars)
            write_json(partial_path, draft.to_dict())
        validations: list[dict[str, Any]] = []
        prose_validations: list[dict[str, Any]] = []
        role_checks: list[dict[str, Any]] = []
        validation: SceneValidation | None = None
        prose_validation: SceneProseValidation | None = None
        role_issues: dict[str, list[str]] = {}
        prose_issues: dict[str, list[str]] = {}
        unlicensed_entity_issues: dict[str, list[str]] = {}
        repairs_used = 0
        for attempt in range(max_repairs + 1):
            scene_subgraph = {
                "段落子图": [
                    narrative_subgraph_for_paragraph(graph, runtime_graph, program, scene, paragraph)
                    for paragraph in scene.paragraphs
                ]
            }
            validate_prompt = _render(folder / "prompt_templates" / "scene_validate.md", {
                "CHAPTER_PROGRAM_CONTEXT_JSON": context,
                "SCENE_PROGRAM_JSON": scene_payload,
                "SCENE_DRAFT_JSON": scene_draft_to_llm_dict(draft),
                "NARRATIVE_SUBGRAPH_JSON": scene_subgraph,
            })
            try:
                validation = _complete_contract(
                    "你是中文小说场景语义校验器。只输出符合指定中文 JSON 契约的对象。",
                    validate_prompt,
                    settings,
                    lambda payload: scene_validation_from_llm(payload, scene),
                    max_tokens=min(settings.max_tokens, 4_000),
                )
            except Exception as exc:
                write_json(destination / f"scene-{index:02d}.draft.partial.json", draft.to_dict())
                write_json(destination / "generation_status.json", {
                    "stage": "scene_validation", "status": "failed", "scene_no": index, "scene_id": scene.scene_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                raise
            validations.append(validation.to_dict())
            if _prose_contract_active(scene):
                prose_prompt = _render(folder / "prompt_templates" / "scene_prose_validate.md", {
                    "CHAPTER_PROGRAM_CONTEXT_JSON": context,
                    "SCENE_PROGRAM_JSON": scene_payload,
                    "SCENE_DRAFT_JSON": scene_draft_to_llm_dict(draft),
                    "NARRATIVE_SUBGRAPH_JSON": scene_subgraph,
                })
                try:
                    prose_validation = _complete_contract(
                        "你是中文小说正文性校验器。只输出符合指定中文 JSON 契约的对象。",
                        prose_prompt,
                        settings,
                        lambda payload: scene_prose_validation_from_llm(payload, scene),
                        max_tokens=min(settings.max_tokens, 4_000),
                    )
                except Exception as exc:
                    write_json(destination / f"scene-{index:02d}.draft.partial.json", draft.to_dict())
                    write_json(destination / "generation_status.json", {
                        "stage": "scene_prose_validation", "status": "failed", "scene_no": index, "scene_id": scene.scene_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    raise
                prose_validations.append(prose_validation.to_dict())
            role_issues = _paragraph_role_issues(draft, scene_paragraph_budgets)
            prose_issues = _local_prose_issues(scene, draft)
            unlicensed_entity_issues = _unlicensed_entity_issues(scene, draft, entity_names)
            role_checks.append({
                "attempt": attempt,
                "passed": not role_issues,
                "paragraph_issues": role_issues,
            })
            prose_passed = prose_validation.passed if prose_validation is not None else True
            if validation.passed and prose_passed and not role_issues and not prose_issues and not unlicensed_entity_issues:
                break
            if attempt >= max_repairs:
                break
            target_set = set(_validation_targets(scene, validation))
            if prose_validation is not None:
                target_set.update(_prose_validation_targets(scene, prose_validation))
            target_set.update(role_issues)
            target_set.update(prose_issues)
            target_set.update(unlicensed_entity_issues)
            target_ids = tuple(item.paragraph_id for item in scene.paragraphs if item.paragraph_id in target_set)
            if not target_ids:
                break
            # Paragraphs are already handed off to later prose only after
            # their own audit.  Replacing one here would silently invalidate
            # those hand-offs.  A scene-level failure is therefore a hard
            # rejection: it must be caught by the paragraph gate and repaired
            # before commit, never papered over by a late whole-scene rewrite.
            if set(target_ids).issubset(set(working_ledger.get("committed_paragraph_ids", []))):
                break
            repair_prompt = _render(folder / "prompt_templates" / "scene_repair.md", {
                "CHAPTER_PROGRAM_CONTEXT_JSON": context,
                "SCENE_PROGRAM_JSON": scene_payload,
                "SCENE_DRAFT_JSON": scene_draft_to_llm_dict(draft),
                "SCENE_VALIDATION_JSON": scene_validation_to_llm_dict(validation),
                "SCENE_PROSE_VALIDATION_JSON": scene_prose_validation_to_llm_dict(prose_validation) if prose_validation else {},
                "NARRATIVE_SUBGRAPH_JSON": scene_subgraph,
                "TARGET_PARAGRAPH_IDS_JSON": list(target_ids),
                "STYLE_EXECUTION_BLUEPRINT_JSON": blueprint,
                "STYLE_REPAIR_DIRECTIVES_JSON": _role_repair_directives(role_issues) + _prose_repair_directives(prose_issues) + _prose_repair_directives(unlicensed_entity_issues),
            })
            try:
                replacements = _complete_contract(
                    "你是中文小说段落定点修复器。只输出符合指定中文 JSON 契约的对象。",
                    repair_prompt,
                    settings,
                    lambda payload: scene_repair_from_llm(payload, scene, target_ids, minimum_chars=minimum_chars),
                    max_tokens=min(settings.max_tokens, max(3_000, 900 * len(target_ids))),
                )
            except Exception as exc:
                write_json(destination / f"scene-{index:02d}.draft.partial.json", draft.to_dict())
                write_json(destination / "generation_status.json", {
                    "stage": "paragraph_repair", "status": "failed", "scene_no": index, "scene_id": scene.scene_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                raise
            draft = _replace_paragraphs(draft, replacements)
            draft.validate_for(scene, minimum_chars=minimum_chars)
            repairs_used += 1
        if validation is None:
            raise RuntimeError("场景校验未产生结果")
        prose_passed = prose_validation.passed if prose_validation is not None else True
        scene_result = {
            "scene_no": index,
            "scene_id": scene.scene_id,
            "batch_count": len(batch_groups),
            "batch_paragraph_limit": DEFAULT_GENERATION_BATCH_PARAGRAPHS,
            "minimum_paragraph_chars": minimum_chars,
            "repairs_used": repairs_used,
            "passed": validation.passed and not unlicensed_entity_issues,
            "prose_passed": prose_passed and not prose_issues,
            "role_passed": not role_issues,
            "licensing_passed": not unlicensed_entity_issues,
            "draft": draft.to_dict(),
            "paragraph_validations": paragraph_validations,
            "validations": validations,
            "prose_validations": prose_validations,
            "prose_checks": {"passed": not prose_issues, "paragraph_issues": prose_issues},
            "role_checks": role_checks,
            "licensing_checks": {"passed": not unlicensed_entity_issues, "paragraph_issues": unlicensed_entity_issues},
        }
        write_json(result_path, scene_result)
        scene_results.append(scene_result)
        final_drafts.append(draft)
        story_state = _advance_generation_state(story_state, scene, draft, entity_names)
        generation_states.append({"scene_no": index, "scene_id": scene.scene_id, "state": story_state})

    # This does not publish state to the next chapter.  It only records that
    # every local hand-off in the candidate was individually audited.
    working_ledger = complete_chapter_working_ledger(working_ledger, program)
    write_chapter_working_ledger(working_ledger_path, working_ledger)
    draft_text = "\n\n".join(_scene_text(item) for item in final_drafts)
    # This remains a candidate until every independent quality gate passes.
    (destination / "candidate_draft.txt").write_text(draft_text, encoding="utf-8")
    style_assessment = assess_style_budget(draft_text, budget)
    structural_passed = all(bool(item["passed"]) for item in scene_results)
    prose_passed = all(bool(item.get("prose_passed", True)) for item in scene_results)
    style_repairs_used = 0
    style_directives: list[str] = []
    style_rounds: list[dict[str, Any]] = []
    # A global rewrite after paragraph-level commits would invalidate the
    # exact prose tail that later paragraphs consumed.  Chapter-wide style is
    # still measured as a hard gate; improvements must happen in the writer or
    # its dedicated paragraph repair before a local patch is committed.
    allow_global_style_rewrite = not bool(working_ledger.get("committed_paragraph_ids"))
    for style_round in range(1, DEFAULT_STYLE_REPAIR_ROUNDS + 1):
        if not allow_global_style_rewrite:
            break
        if not structural_passed or not prose_passed or bool(style_assessment.get("passed")):
            break
        style_directives = _local_style_directives(budget, style_assessment)
        accepted_this_round = 0
        deterministic_repairs: list[dict[str, str]] = []
        for label, metric_id in (("感知", "perception_marker_ratio"), ("转场", "transition_marker_ratio")):
            metric = _metric_rows(style_assessment).get(metric_id, {})
            if _metric_direction(metric) != "decrease":
                continue
            for index, (scene, draft, result) in enumerate(zip(program.scene_programs, final_drafts, scene_results), start=1):
                scene_paragraph_budgets = {
                    item.paragraph_id: paragraph_style_budgets[item.paragraph_id]
                    for item in scene.paragraphs if item.paragraph_id in paragraph_style_budgets
                }
                replacement = _replace_non_role_marker(draft, scene_paragraph_budgets, label)
                if replacement is None:
                    continue
                candidate, repair_record = replacement
                minimum_chars = int(result["minimum_paragraph_chars"])
                try:
                    candidate.validate_for(scene, minimum_chars=minimum_chars)
                except ValueError:
                    continue
                if _paragraph_role_issues(candidate, scene_paragraph_budgets):
                    continue
                candidate_drafts = list(final_drafts)
                candidate_drafts[index - 1] = candidate
                candidate_text = "\n\n".join(_scene_text(item) for item in candidate_drafts)
                candidate_assessment = assess_style_budget(candidate_text, budget)
                if not _prefer_style_assessment(style_assessment, candidate_assessment):
                    continue
                final_drafts[index - 1] = candidate
                style_assessment = candidate_assessment
                result["draft"] = candidate.to_dict()
                result.setdefault("deterministic_style_repairs", []).append(repair_record)
                write_json(destination / f"scene-{index:02d}.result.json", result)
                deterministic_repairs.append(repair_record)
                style_repairs_used += 1
                accepted_this_round += 1
                break
        if bool(style_assessment.get("passed")):
            draft_text = "\n\n".join(_scene_text(item) for item in final_drafts)
            (destination / "candidate_draft.txt").write_text(draft_text, encoding="utf-8")
            style_rounds.append({
                "round": style_round,
                "directives": style_directives,
                "deterministic_repairs": deterministic_repairs,
                "repairs_used": accepted_this_round,
                "style_assessment": style_assessment,
            })
            break
        style_directives = _local_style_directives(budget, style_assessment)
        for index, (scene, scene_payload, draft, result) in enumerate(zip(program.scene_programs, scene_payloads, final_drafts, scene_results), start=1):
            target_ids = _style_targets(scene, draft, style_assessment, paragraph_style_budgets)
            if not target_ids:
                continue
            scene_paragraph_budgets = {
                item.paragraph_id: paragraph_style_budgets[item.paragraph_id]
                for item in scene.paragraphs if item.paragraph_id in paragraph_style_budgets
            }
            blueprint = _scene_blueprint(budget, len(program.scene_programs), len(scene.paragraphs), scene_paragraph_budgets)
            minimum_chars = int(result["minimum_paragraph_chars"])
            latest_validation = result["validations"][-1] if result.get("validations") else {}
            latest_prose_validation = result.get("prose_validations", [])[-1] if result.get("prose_validations") else {}
            style_scene_subgraph = {
                "段落子图": [
                    narrative_subgraph_for_paragraph(graph, runtime_graph, program, scene, paragraph)
                    for paragraph in scene.paragraphs
                ]
            }
            repair_prompt = _render(folder / "prompt_templates" / "scene_repair.md", {
                "CHAPTER_PROGRAM_CONTEXT_JSON": context,
                "SCENE_PROGRAM_JSON": scene_payload,
                "SCENE_DRAFT_JSON": scene_draft_to_llm_dict(draft),
                "SCENE_VALIDATION_JSON": _stored_validation_to_llm_dict(latest_validation),
                "SCENE_PROSE_VALIDATION_JSON": _stored_prose_validation_to_llm_dict(latest_prose_validation),
                "NARRATIVE_SUBGRAPH_JSON": style_scene_subgraph,
                "TARGET_PARAGRAPH_IDS_JSON": list(target_ids),
                "STYLE_EXECUTION_BLUEPRINT_JSON": blueprint,
                "STYLE_REPAIR_DIRECTIVES_JSON": style_directives,
            })
            try:
                replacements = _complete_contract(
                    "你是中文小说局部风格修复器。只输出符合指定中文 JSON 契约的对象。",
                    repair_prompt,
                    settings,
                    lambda payload: scene_repair_from_llm(payload, scene, target_ids, minimum_chars=minimum_chars),
                    max_tokens=min(settings.max_tokens, max(3_000, 900 * len(target_ids))),
                )
                candidate = _replace_paragraphs(draft, replacements)
                candidate.validate_for(scene, minimum_chars=minimum_chars)
                validate_prompt = _render(folder / "prompt_templates" / "scene_validate.md", {
                    "CHAPTER_PROGRAM_CONTEXT_JSON": context,
                    "SCENE_PROGRAM_JSON": scene_payload,
                    "SCENE_DRAFT_JSON": scene_draft_to_llm_dict(candidate),
                    "NARRATIVE_SUBGRAPH_JSON": style_scene_subgraph,
                })
                candidate_validation = _complete_contract(
                    "你是中文小说场景语义校验器。只输出符合指定中文 JSON 契约的对象。",
                    validate_prompt,
                    settings,
                    lambda payload: scene_validation_from_llm(payload, scene),
                    max_tokens=min(settings.max_tokens, 4_000),
                )
                candidate_prose_validation: SceneProseValidation | None = None
                if _prose_contract_active(scene):
                    prose_prompt = _render(folder / "prompt_templates" / "scene_prose_validate.md", {
                        "CHAPTER_PROGRAM_CONTEXT_JSON": context,
                        "SCENE_PROGRAM_JSON": scene_payload,
                        "SCENE_DRAFT_JSON": scene_draft_to_llm_dict(candidate),
                        "NARRATIVE_SUBGRAPH_JSON": style_scene_subgraph,
                    })
                    candidate_prose_validation = _complete_contract(
                        "你是中文小说正文性校验器。只输出符合指定中文 JSON 契约的对象。",
                        prose_prompt,
                        settings,
                        lambda payload: scene_prose_validation_from_llm(payload, scene),
                        max_tokens=min(settings.max_tokens, 4_000),
                    )
            except (RuntimeError, ValueError) as exc:
                result.setdefault("style_repair_errors", []).append(f"{type(exc).__name__}: {exc}")
                write_json(destination / f"scene-{index:02d}.result.json", result)
                continue
            if not candidate_validation.passed:
                result.setdefault("style_repair_rejections", []).append(candidate_validation.to_dict())
                write_json(destination / f"scene-{index:02d}.result.json", result)
                continue
            if (candidate_prose_validation is not None and not candidate_prose_validation.passed) or _local_prose_issues(scene, candidate):
                result.setdefault("style_repair_rejections", []).append({
                    "reason": "局部风格修复破坏了正文性",
                    "prose_validation": candidate_prose_validation.to_dict() if candidate_prose_validation else {},
                    "local_prose_issues": _local_prose_issues(scene, candidate),
                })
                write_json(destination / f"scene-{index:02d}.result.json", result)
                continue
            if _paragraph_role_issues(candidate, scene_paragraph_budgets):
                result.setdefault("style_repair_rejections", []).append({
                    "reason": "局部风格修复破坏了段落职责或最低正文长度",
                    "role_issues": _paragraph_role_issues(candidate, scene_paragraph_budgets),
                })
                write_json(destination / f"scene-{index:02d}.result.json", result)
                continue
            candidate_drafts = list(final_drafts)
            candidate_drafts[index - 1] = candidate
            candidate_text = "\n\n".join(_scene_text(item) for item in candidate_drafts)
            candidate_assessment = assess_style_budget(candidate_text, budget)
            if not _prefer_style_assessment(style_assessment, candidate_assessment):
                result.setdefault("style_repair_rejections", []).append({
                    "reason": "局部修复未带来可测量的风格改进",
                    "style_assessment": candidate_assessment,
                })
                write_json(destination / f"scene-{index:02d}.result.json", result)
                continue
            final_drafts[index - 1] = candidate
            style_assessment = candidate_assessment
            result["draft"] = candidate.to_dict()
            result.setdefault("style_validations", []).append(candidate_validation.to_dict())
            if candidate_prose_validation is not None:
                result.setdefault("prose_validations", []).append(candidate_prose_validation.to_dict())
            result["style_repairs_used"] = int(result.get("style_repairs_used", 0)) + 1
            write_json(destination / f"scene-{index:02d}.result.json", result)
            style_repairs_used += 1
            accepted_this_round += 1
        draft_text = "\n\n".join(_scene_text(item) for item in final_drafts)
        (destination / "candidate_draft.txt").write_text(draft_text, encoding="utf-8")
        style_assessment = assess_style_budget(draft_text, budget)
        structural_passed = all(bool(item["passed"]) for item in scene_results)
        prose_passed = all(bool(item.get("prose_passed", True)) for item in scene_results)
        style_rounds.append({
            "round": style_round,
            "directives": style_directives,
            "deterministic_repairs": deterministic_repairs,
            "repairs_used": accepted_this_round,
            "style_assessment": style_assessment,
        })
        if not accepted_this_round:
            break
    # Semantic success is not enough: short or role-incomplete prose is not a
    # publishable chapter and must remain visibly separate from the final file.
    role_passed = all(bool(item.get("role_passed", True)) for item in scene_results)
    draft_text = "\n\n".join(_scene_text(item) for item in final_drafts)
    length_assessment = _chapter_length_assessment(program, draft_text)
    base_overall_passed = structural_passed and prose_passed and role_passed and bool(style_assessment.get("passed")) and bool(length_assessment["passed"])
    graph_patch = build_chapter_graph_patch(
        graph, runtime_graph, program, scene_results,
        prose_passed=prose_passed,
        style_passed=bool(style_assessment.get("passed")),
        length_passed=bool(length_assessment["passed"]),
    )
    candidate_graph_patch_path = destination / "chapter_graph_patch.candidate.json"
    write_json(candidate_graph_patch_path, graph_patch)
    graph_patch_validation = validate_graph_patch(graph, runtime_graph, graph_patch, program=program)
    graph_patch_passed = bool(graph_patch_validation.get("passed"))
    overall_passed = base_overall_passed and graph_patch_passed
    candidate_path = destination / "candidate_draft.txt"
    candidate_path.write_text(draft_text, encoding="utf-8")
    generated_path = destination / "generated_draft.txt"
    committed_graph_patch_path = destination / "chapter_graph_patch.json"
    if overall_passed:
        runtime_graph = commit_graph_patch(runtime_graph, graph_patch, graph_patch_validation)
        write_runtime_graph(active_runtime_graph_path, runtime_graph)
        committed_patch = {**graph_patch, "status": "committed", "commit_validation": graph_patch_validation}
        write_json(committed_graph_patch_path, committed_patch)
        generated_path.write_text(draft_text, encoding="utf-8")
    else:
        # A reused destination must not keep an obsolete final-looking draft or
        # a stale committed state from a previous candidate.
        if generated_path.exists():
            generated_path.unlink()
        if committed_graph_patch_path.exists():
            committed_graph_patch_path.unlink()

    write_json(destination / "style_assessment.json", style_assessment)
    write_json(destination / "length_assessment.json", length_assessment)
    write_json(destination / "scene_continuity_states.json", generation_states)
    chapter_exit_path = destination / "chapter_exit_state.json"
    write_json(chapter_exit_path, {
        "schema_version": "1.0",
        "chapter_id": program.chapter_id,
        "source_hash": program.source_hash,
        "generation_mode": program.generation_mode,
        "overall_passed": overall_passed,
        "graph_patch_passed": graph_patch_passed,
        "graph_patch_committed": overall_passed,
        "graph_patch": str(committed_graph_patch_path) if overall_passed else "",
        "graph_patch_candidate": str(candidate_graph_patch_path),
        "runtime_graph_state": str(active_runtime_graph_path),
        "chapter_working_ledger": str(working_ledger_path),
        "chapter_contract": str(destination / "chapter_contract.json") if chapter_contract is not None else "",
        "final_scene_id": program.scene_programs[-1].scene_id if program.scene_programs else "",
        "state": story_state,
    })
    write_json(destination / "style_repair_plan.json", {
        "attempted": bool(style_directives), "directives": style_directives,
        "repairs_used": style_repairs_used, "rounds": style_rounds,
        "style_passed": bool(style_assessment.get("passed")),
    })
    write_json(destination / "generation_manifest.json", {
        "author_id": author_id,
        "program_path": str(program_path),
        "story_state_path": str(story_state_path) if story_state_path else "",
        "chapter_plan_path": str(chapter_plan_path) if chapter_plan_path else "",
        "selected_template_ids": [str(item.get("template_id", "")) for item in selected],
        "scene_count": len(scene_results),
        "repairs_used": sum(int(item["repairs_used"]) for item in scene_results),
        "style_repairs_used": style_repairs_used,
        "semantic_passed": structural_passed,
        "structural_passed": structural_passed,
        "prose_passed": prose_passed,
        "role_passed": role_passed,
        "style_passed": bool(style_assessment.get("passed")),
        "length_passed": bool(length_assessment["passed"]),
        "graph_patch_passed": graph_patch_passed,
        "length_assessment": length_assessment,
        "overall_passed": overall_passed,
        "candidate_draft": candidate_path.name,
        "published_draft": generated_path.name if overall_passed else "",
        "chapter_exit_state": chapter_exit_path.name,
        "graph_patch_candidate": candidate_graph_patch_path.name,
        "graph_patch": committed_graph_patch_path.name if overall_passed else "",
        "runtime_graph_state": str(active_runtime_graph_path),
        "chapter_working_ledger": str(working_ledger_path),
        "chapter_contract_path": str(chapter_contract_path) if chapter_contract_path else "",
        "chapter_contract_fallback": chapter_contract is not None and chapter_contract_path is None,
    })
    write_json(destination / "generation_status.json", {
        "stage": "complete" if overall_passed else "validation_failed", "status": "complete" if overall_passed else "rejected", "semantic_passed": structural_passed,
        "structural_passed": structural_passed, "prose_passed": prose_passed,
        "role_passed": role_passed,
        "style_passed": bool(style_assessment.get("passed")),
        "length_passed": bool(length_assessment["passed"]),
        "graph_patch_passed": graph_patch_passed,
        "overall_passed": overall_passed,
    })
    return {
        "output_dir": str(destination),
        "scene_count": len(scene_results),
        "repairs_used": sum(int(item["repairs_used"]) for item in scene_results),
        "style_repairs_used": style_repairs_used,
        "semantic_passed": structural_passed,
        "structural_passed": structural_passed,
        "prose_passed": prose_passed,
        "role_passed": role_passed,
        "style_passed": bool(style_assessment.get("passed")),
        "length_passed": bool(length_assessment["passed"]),
        "graph_patch_passed": graph_patch_passed,
        "graph_patch_committed": overall_passed,
        "overall_passed": overall_passed,
        "chapter_exit_state": str(chapter_exit_path),
    }
