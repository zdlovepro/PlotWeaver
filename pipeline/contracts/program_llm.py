"""Chinese JSON boundary for chapter programs sent to a Chinese-language model."""

from __future__ import annotations

from typing import Any

from .program import ChapterProgram, EntityBinding, EventProgram, FactContract, NarrativeBeat, ParagraphProgram, SceneProgram


EXPRESSION_TO_ZH = {
    "narration": "叙述交代", "action": "行动呈现", "dialogue": "对话呈现", "internal": "心理或感知", "implicit": "隐性表现",
}
EXPRESSION_FROM_ZH = {value: key for key, value in EXPRESSION_TO_ZH.items()}
FUNCTION_TO_ZH = {
    "orientation": "场景定位", "action_progression": "行动推进", "obstacle": "阻力显现", "perception_reaction": "感知反应",
    "dialogue_conflict": "对话交锋", "turn": "局势转折", "decision": "做出决定", "transition": "场景转场", "aftermath": "余波收束",
}
FUNCTION_FROM_ZH = {value: key for key, value in FUNCTION_TO_ZH.items()}
BEAT_TO_ZH = {
    "setup": "起势", "action": "行动", "pressure": "压力", "reaction": "反应",
    "choice": "选择", "consequence": "结果", "transition": "转场", "aftermath": "余波",
}
BEAT_FROM_ZH = {value: key for key, value in BEAT_TO_ZH.items()}
ACTION_TYPE_TO_ZH = {
    "speech": "交流", "movement": "移动", "transfer": "交付",
    "perception": "感知", "decision": "决定", "confrontation": "对抗",
    "state_change": "状态变化", "other": "其他",
}
ACTION_TYPE_FROM_ZH = {value: key for key, value in ACTION_TYPE_TO_ZH.items()}


def _require_exact_keys(payload: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise ValueError(f"{context} 的字段必须严格匹配；缺少={sorted(expected - actual)}，多出={sorted(actual - expected)}")


def _string_list(payload: dict[str, Any], key: str, context: str) -> tuple[str, ...]:
    value = payload[key]
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{context}.{key} 必须是字符串数组")
    return tuple(value)


def _string(payload: dict[str, Any], key: str, context: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise ValueError(f"{context}.{key} 必须是字符串")
    return value


def _fact_to_llm(item: FactContract) -> dict[str, Any]:
    return {
        "事实ID": item.fact_id, "场景ID": item.scene_id, "事实类型": item.fact_kind, "主体ID": item.subject_id,
        "谓词": item.predicate, "值": item.value, "客体ID": item.object_id, "表达方式": EXPRESSION_TO_ZH[item.expression_mode],
        "必须表达": item.must_realize, "禁止相反含义": list(item.forbidden_inverse_meanings),
    }


def _fact_from_llm(payload: dict[str, Any]) -> FactContract:
    context = "事实表达契约"
    _require_exact_keys(payload, {"事实ID", "场景ID", "事实类型", "主体ID", "谓词", "值", "客体ID", "表达方式", "必须表达", "禁止相反含义"}, context)
    expression = _string(payload, "表达方式", context)
    if expression not in EXPRESSION_FROM_ZH:
        raise ValueError(f"{context}.表达方式 不受支持：{expression}")
    if not isinstance(payload["必须表达"], bool):
        raise ValueError(f"{context}.必须表达 必须是布尔值")
    return FactContract(
        fact_id=_string(payload, "事实ID", context), scene_id=_string(payload, "场景ID", context),
        fact_kind=_string(payload, "事实类型", context), subject_id=_string(payload, "主体ID", context),
        predicate=_string(payload, "谓词", context), value=_string(payload, "值", context), object_id=_string(payload, "客体ID", context),
        expression_mode=EXPRESSION_FROM_ZH[expression], must_realize=payload["必须表达"],
        forbidden_inverse_meanings=_string_list(payload, "禁止相反含义", context),
    )


def _event_to_llm(item: EventProgram) -> dict[str, Any]:
    return {
        "事件ID": item.event_id, "场景ID": item.scene_id, "事件概述": item.summary, "行动": item.action,
        "参与实体ID": list(item.participant_ids), "阻力": item.obstacle, "决断": item.decision,
        "前置事实ID": list(item.precondition_fact_ids),
        "必须表达的事实ID": list(item.required_fact_ids), "结果事实ID": list(item.outcome_fact_ids),
        "行动类型": ACTION_TYPE_TO_ZH[item.action_type], "行动者ID": item.actor_id,
        "受事者ID": list(item.target_ids), "行动依据事实ID": list(item.basis_fact_ids),
    }


def _event_from_llm(payload: dict[str, Any]) -> EventProgram:
    context = "必现事件"
    legacy_keys = {"事件ID", "场景ID", "事件概述", "行动", "参与实体ID", "阻力", "决断", "前置事实ID", "必须表达的事实ID", "结果事实ID"}
    framed_keys = legacy_keys | {"行动类型", "行动者ID", "受事者ID", "行动依据事实ID"}
    if set(payload) not in (legacy_keys, framed_keys):
        raise ValueError(f"{context} 的字段必须严格匹配旧版或语义框架版契约；实际={sorted(payload)}")
    action_type_zh = _string(payload, "行动类型", context) if "行动类型" in payload else "其他"
    if action_type_zh not in ACTION_TYPE_FROM_ZH:
        raise ValueError(f"{context}.行动类型 不受支持：{action_type_zh}")
    return EventProgram(
        event_id=_string(payload, "事件ID", context), scene_id=_string(payload, "场景ID", context),
        summary=_string(payload, "事件概述", context), action=_string(payload, "行动", context),
        participant_ids=_string_list(payload, "参与实体ID", context), obstacle=_string(payload, "阻力", context), decision=_string(payload, "决断", context),
        precondition_fact_ids=_string_list(payload, "前置事实ID", context),
        required_fact_ids=_string_list(payload, "必须表达的事实ID", context), outcome_fact_ids=_string_list(payload, "结果事实ID", context),
        action_type=ACTION_TYPE_FROM_ZH[action_type_zh],
        actor_id=_string(payload, "行动者ID", context) if "行动者ID" in payload else "",
        target_ids=_string_list(payload, "受事者ID", context) if "受事者ID" in payload else (),
        basis_fact_ids=_string_list(payload, "行动依据事实ID", context) if "行动依据事实ID" in payload else (),
    )


def _paragraph_to_llm(item: ParagraphProgram) -> dict[str, Any]:
    return {
        "段落ID": item.paragraph_id, "场景ID": item.scene_id, "段落序号": item.order, "段落功能": FUNCTION_TO_ZH[item.function],
        "必须表达的事实ID": list(item.required_fact_ids), "事件ID列表": list(item.event_ids),
        "视角人物ID": item.viewpoint_entity_id, "对话行为": item.dialogue_act, "节拍ID列表": list(item.beat_ids),
        "目标字数": item.target_chars, "最低字数": item.minimum_chars,
    }


def _paragraph_from_llm(payload: dict[str, Any]) -> ParagraphProgram:
    context = "段落程序"
    legacy_keys = {"段落ID", "场景ID", "段落序号", "段落功能", "必须表达的事实ID", "事件ID列表", "视角人物ID", "对话行为", "节拍ID列表"}
    expanded_keys = legacy_keys | {"目标字数", "最低字数"}
    if set(payload) not in (legacy_keys, expanded_keys):
        raise ValueError(f"{context} 的字段必须严格匹配旧版或扩写版契约；实际={sorted(payload)}")
    function = _string(payload, "段落功能", context)
    if function not in FUNCTION_FROM_ZH:
        raise ValueError(f"{context}.段落功能 不受支持：{function}")
    if not isinstance(payload["段落序号"], int):
        raise ValueError(f"{context}.段落序号 必须是整数")
    target_chars = payload.get("目标字数", 0)
    minimum_chars = payload.get("最低字数", 0)
    if not isinstance(target_chars, int) or not isinstance(minimum_chars, int):
        raise ValueError(f"{context}.目标字数 和 最低字数 必须是整数")
    return ParagraphProgram(
        paragraph_id=_string(payload, "段落ID", context), scene_id=_string(payload, "场景ID", context), order=payload["段落序号"],
        function=FUNCTION_FROM_ZH[function], required_fact_ids=_string_list(payload, "必须表达的事实ID", context),
        event_ids=_string_list(payload, "事件ID列表", context), viewpoint_entity_id=_string(payload, "视角人物ID", context),
        dialogue_act=_string(payload, "对话行为", context), beat_ids=_string_list(payload, "节拍ID列表", context),
        target_chars=target_chars, minimum_chars=minimum_chars,
    )


def _beat_to_llm(item: NarrativeBeat) -> dict[str, Any]:
    return {
        "节拍ID": item.beat_id, "场景ID": item.scene_id, "节拍序号": item.order,
        "节拍类型": BEAT_TO_ZH[item.beat_type], "行动者ID": item.actor_id,
        "人物目标": item.objective, "压力或阻力": item.pressure,
        "可见动作": item.observable_action, "视角细节": item.focal_detail,
        "状态变化": item.state_change, "必须表达的事实ID": list(item.required_fact_ids),
        "事件ID列表": list(item.event_ids), "对白压力": item.dialogue_pressure,
        "扩写许可": {"source": "原文复现", "inference": "合理推导", "atmosphere": "氛围扩写"}[item.expansion_license],
        "依据事实ID": list(item.support_fact_ids),
        "叙事机制ID": list(item.mechanism_ids),
        "叙事任务": item.narrative_function,
        "反事实守卫": item.counterfactual_guard,
    }


def _beat_from_llm(payload: dict[str, Any]) -> NarrativeBeat:
    context = "戏剧节拍"
    legacy_keys = {"节拍ID", "场景ID", "节拍序号", "节拍类型", "行动者ID", "人物目标", "压力或阻力", "可见动作", "视角细节", "状态变化", "必须表达的事实ID", "事件ID列表", "对白压力"}
    expanded_keys = legacy_keys | {"扩写许可", "依据事实ID"}
    mechanism_fields = {"叙事机制ID", "叙事任务", "反事实守卫"}
    legacy_mechanism_keys = legacy_keys | mechanism_fields
    expanded_mechanism_keys = expanded_keys | mechanism_fields
    if set(payload) not in (legacy_keys, expanded_keys, legacy_mechanism_keys, expanded_mechanism_keys):
        raise ValueError(f"{context} 的字段必须严格匹配旧版、扩写版或叙事机制版契约；实际={sorted(payload)}")
    beat_type = _string(payload, "节拍类型", context)
    if beat_type not in BEAT_FROM_ZH:
        raise ValueError(f"{context}.节拍类型 不受支持：{beat_type}")
    if not isinstance(payload["节拍序号"], int):
        raise ValueError(f"{context}.节拍序号 必须是整数")
    license_value = _string(payload, "扩写许可", context) if "扩写许可" in payload else "原文复现"
    license_from_zh = {"原文复现": "source", "合理推导": "inference", "氛围扩写": "atmosphere"}
    if license_value not in license_from_zh:
        raise ValueError(f"{context}.扩写许可 不受支持：{license_value}")
    return NarrativeBeat(
        beat_id=_string(payload, "节拍ID", context), scene_id=_string(payload, "场景ID", context),
        order=payload["节拍序号"], beat_type=BEAT_FROM_ZH[beat_type], actor_id=_string(payload, "行动者ID", context),
        objective=_string(payload, "人物目标", context), pressure=_string(payload, "压力或阻力", context),
        observable_action=_string(payload, "可见动作", context), focal_detail=_string(payload, "视角细节", context),
        state_change=_string(payload, "状态变化", context),
        required_fact_ids=_string_list(payload, "必须表达的事实ID", context),
        event_ids=_string_list(payload, "事件ID列表", context), dialogue_pressure=_string(payload, "对白压力", context),
        expansion_license=license_from_zh[license_value],
        support_fact_ids=_string_list(payload, "依据事实ID", context) if "依据事实ID" in payload else (),
        mechanism_ids=_string_list(payload, "叙事机制ID", context) if "叙事机制ID" in payload else (),
        narrative_function=_string(payload, "叙事任务", context) if "叙事任务" in payload else "",
        counterfactual_guard=_string(payload, "反事实守卫", context) if "反事实守卫" in payload else "",
    )


def chapter_program_to_llm_dict(program: ChapterProgram) -> dict[str, Any]:
    """Render the exact Chinese-key JSON contract a model receives."""

    program.validate()
    return {
        "契约版本": program.schema_version, "章节ID": program.chapter_id, "源文本哈希": program.source_hash,
        "生成模式": {"faithful": "忠实复现", "controlled_expansion": "受控扩写"}[program.generation_mode],
        "章节目标字数下限": program.target_char_min, "章节目标字数上限": program.target_char_max,
        "实体字典": [{"实体ID": item.entity_id, "名称": item.canonical_name, "类型": item.kind} for item in program.entities],
        "场景程序": [{
            "场景ID": scene.scene_id, "章节ID": scene.chapter_id, "场景序号": scene.order,
            "场景目标": scene.objective, "参与实体ID": list(scene.participant_ids),
            "入场状态事实ID": list(scene.entry_state_fact_ids), "必现事件": [_event_to_llm(item) for item in scene.event_programs],
            "事实表达契约": [_fact_to_llm(item) for item in scene.fact_contracts],
            "只读上下文事实": [_fact_to_llm(item) for item in scene.context_fact_contracts],
            "戏剧节拍": [_beat_to_llm(item) for item in scene.narrative_beats],
            "段落程序": [_paragraph_to_llm(item) for item in scene.paragraphs],
            "出场状态事实ID": list(scene.exit_state_fact_ids), "禁止提前事件ID": list(scene.forbidden_event_ids),
        } for scene in program.scene_programs],
    }


def chapter_program_from_llm_dict(payload: dict[str, Any]) -> ChapterProgram:
    """Parse a model JSON response only when it exactly follows the Chinese schema."""

    context = "章节程序"
    legacy_keys = {"契约版本", "章节ID", "源文本哈希", "实体字典", "场景程序"}
    expanded_keys = legacy_keys | {"生成模式", "章节目标字数下限", "章节目标字数上限"}
    if set(payload) not in (legacy_keys, expanded_keys):
        raise ValueError(f"{context} 的字段必须严格匹配旧版或扩写版契约；实际={sorted(payload)}")
    if not isinstance(payload["场景程序"], list):
        raise ValueError(f"{context}.场景程序 必须是数组")
    if not isinstance(payload["实体字典"], list) or any(not isinstance(item, dict) for item in payload["实体字典"]):
        raise ValueError(f"{context}.实体字典 必须是对象数组")
    entities: list[EntityBinding] = []
    for item in payload["实体字典"]:
        _require_exact_keys(item, {"实体ID", "名称", "类型"}, "实体字典")
        entities.append(EntityBinding(_string(item, "实体ID", "实体字典"), _string(item, "名称", "实体字典"), _string(item, "类型", "实体字典")))
    scenes: list[SceneProgram] = []
    for item in payload["场景程序"]:
        if not isinstance(item, dict):
            raise ValueError(f"{context}.场景程序 的每项必须是对象")
        scene_context = "场景程序"
        scene_keys = {"场景ID", "章节ID", "场景序号", "场景目标", "参与实体ID", "入场状态事实ID", "必现事件", "事实表达契约", "只读上下文事实", "戏剧节拍", "段落程序", "出场状态事实ID", "禁止提前事件ID"}
        if set(item) not in (scene_keys, scene_keys - {"只读上下文事实"}):
            raise ValueError(f"{scene_context} 的字段必须严格匹配新版或旧版契约；实际={sorted(item)}")
        if not isinstance(item["场景序号"], int):
            raise ValueError(f"{scene_context}.场景序号 必须是整数")
        for key in ("必现事件", "事实表达契约", "戏剧节拍", "段落程序"):
            if not isinstance(item[key], list) or any(not isinstance(child, dict) for child in item[key]):
                raise ValueError(f"{scene_context}.{key} 必须是对象数组")
        if "只读上下文事实" in item and (not isinstance(item["只读上下文事实"], list) or any(not isinstance(child, dict) for child in item["只读上下文事实"])):
            raise ValueError(f"{scene_context}.只读上下文事实 必须是对象数组")
        scenes.append(SceneProgram(
            scene_id=_string(item, "场景ID", scene_context), chapter_id=_string(item, "章节ID", scene_context), order=item["场景序号"],
            objective=_string(item, "场景目标", scene_context), participant_ids=_string_list(item, "参与实体ID", scene_context),
            entry_state_fact_ids=_string_list(item, "入场状态事实ID", scene_context),
            event_programs=tuple(_event_from_llm(child) for child in item["必现事件"]),
            fact_contracts=tuple(_fact_from_llm(child) for child in item["事实表达契约"]),
            context_fact_contracts=tuple(_fact_from_llm(child) for child in item.get("只读上下文事实", [])),
            paragraphs=tuple(_paragraph_from_llm(child) for child in item["段落程序"]),
            exit_state_fact_ids=_string_list(item, "出场状态事实ID", scene_context),
            forbidden_event_ids=_string_list(item, "禁止提前事件ID", scene_context),
            narrative_beats=tuple(_beat_from_llm(child) for child in item["戏剧节拍"]),
        ))
    mode = _string(payload, "生成模式", context) if "生成模式" in payload else "忠实复现"
    mode_from_zh = {"忠实复现": "faithful", "受控扩写": "controlled_expansion"}
    if mode not in mode_from_zh:
        raise ValueError(f"{context}.生成模式 不受支持：{mode}")
    target_char_min = payload.get("章节目标字数下限", 0)
    target_char_max = payload.get("章节目标字数上限", 0)
    if not isinstance(target_char_min, int) or not isinstance(target_char_max, int):
        raise ValueError(f"{context}.章节目标字数下限 和 章节目标字数上限 必须是整数")
    program = ChapterProgram(
        chapter_id=_string(payload, "章节ID", context), source_hash=_string(payload, "源文本哈希", context),
        schema_version=_string(payload, "契约版本", context), scene_programs=tuple(scenes), entities=tuple(entities),
        generation_mode=mode_from_zh[mode], target_char_min=target_char_min, target_char_max=target_char_max,
    )
    program.validate()
    return program
