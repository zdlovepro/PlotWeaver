"""Runtime graph reads and commit-gated patches for chapter generation.

The corpus graph remains evidence-only and immutable.  A generation run keeps
its own compact state ledger: candidates create a patch, validators decide
whether it is safe, and only then is that patch committed for the next chapter.
No helper in this module returns source quotes or complete source chapters.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .chapter_runtime import working_ledger_context
from .contracts import ChapterContract, ChapterProgram, EventGraph, NarrativeGraph, ParagraphProgram, SceneProgram
from .graph_retrieval import (
    DEFAULT_GRAPH_RAG_EDGE_BUDGET,
    DEFAULT_GRAPH_RAG_MAX_HOPS,
    DEFAULT_GRAPH_RAG_NODE_BUDGET,
    retrieve_graph_neighbourhood,
)
from .jsonio import read_json, write_json
from .macro_planner import chapter_contract_writer_context


STATE_BEARING_FACT_KINDS = frozenset({
    "identity", "goal", "relationship", "location", "resource", "knowledge", "rule", "progression", "information",
})

# These relations define a person's stable identity inside an ongoing story.
# They are intentionally narrow: ordinary alliances and attitudes may change,
# but a character cannot silently acquire a different parent, child, sibling or
# blood-relative in a later generated chapter.
_IMMUTABLE_RELATIONSHIP_CUES = (
    "父", "母", "爹", "娘", "子", "女", "兄", "弟", "姐", "妹", "祖", "孙", "叔", "伯", "舅", "姑", "姨", "血亲",
)
_IMMUTABLE_IDENTITY_CUES = ("姓名", "名字", "性别", "出身", "血脉", "身份")
_FAILURE_LOOP_CUES = ("失败", "落败", "未通过", "淘汰", "受挫", "羞辱", "嘲笑", "不甘", "屈辱")


def graph_fingerprint(graph: NarrativeGraph) -> str:
    body = json.dumps(graph.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _source_ref(chapter_id: str, kind: str, local_id: str) -> str:
    return f"{kind}:{chapter_id}:{local_id}"


def _node_by_source(graph: NarrativeGraph) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for node in graph.nodes:
        for source_id in node.source_ids:
            result.setdefault(source_id, node)
    return result


def _compact_node(
    node: Any,
    *,
    foreshadow_lifecycle: str | None = None,
    resolved_by_event_id: str = "",
    retrieval_reasons: Iterable[str] = (),
) -> dict[str, Any]:
    """Return structured meaning only; evidence quotes never enter a prompt."""

    attributes = node.attributes if isinstance(node.attributes, dict) else {}
    item: dict[str, Any] = {"节点ID": node.node_id, "类型": node.kind}
    if node.kind == "entity":
        item.update({"名称": node.label, "实体类型": attributes.get("entity_kind", "")})
    elif node.kind == "fact":
        item.update({
            "事实类型": attributes.get("fact_kind", ""), "谓词": attributes.get("predicate", ""),
            "值": attributes.get("value", ""), "主体局部ID": attributes.get("subject_local_id", ""),
            "客体局部ID": attributes.get("object_local_id", ""),
        })
    elif node.kind == "event":
        item.update({
            "动作": attributes.get("action", ""),
            "行动类型": attributes.get("action_type", ""),
            "行动者实体ID": attributes.get("actor_entity_id", ""),
            "受事者实体ID": attributes.get("target_entity_ids", []),
            "阻力": attributes.get("obstacle", ""),
            "选择": attributes.get("decision", ""),
        })
    elif node.kind == "scene":
        item.update({"紧张度": attributes.get("tension", 0)})
    elif node.kind == "time_anchor":
        item.update({"时间锚类型": attributes.get("anchor_kind", "")})
    elif node.kind == "foreshadow":
        # The immutable corpus graph can know a later resolution.  A writing
        # request must instead see only what has been committed before this
        # paragraph (plus a resolution event licensed in this paragraph).
        item.update({
            "生命周期": foreshadow_lifecycle or "open",
            "兑现事件": resolved_by_event_id if foreshadow_lifecycle == "resolved" else "",
        })
    reasons = tuple(dict.fromkeys(str(value) for value in retrieval_reasons if str(value).strip()))
    if reasons:
        item["检索用途"] = list(reasons)
    return item


def _edge_compact(edge: Any) -> dict[str, Any]:
    return {
        "关系ID": edge.edge_id,
        "关系": edge.kind,
        "起点": edge.source_node_id,
        "终点": edge.target_node_id,
        "属性": edge.attributes if isinstance(edge.attributes, dict) else {},
    }


def initialize_runtime_graph(graph: NarrativeGraph | None, *, program: ChapterProgram | None = None) -> dict[str, Any]:
    """Create a mutable ledger separate from the immutable source graph."""

    return {
        "schema_version": "1.1",
        "mode": "narrative_graph" if graph is not None else "program_contract_fallback",
        "author_id": graph.author_id if graph is not None else "",
        "work_id": graph.work_id if graph is not None else "",
        "base_graph_fingerprint": graph_fingerprint(graph) if graph is not None else "",
        "program_source_hash": program.source_hash if program is not None else "",
        "committed_patches": [],
        "effective_state": {},
        "effective_foreshadows": {},
        "committed_chapter_ids": [],
    }


def load_or_initialize_runtime_graph(
    path: Path | None,
    graph: NarrativeGraph | None,
    program: ChapterProgram,
) -> dict[str, Any]:
    if path is None or not path.exists():
        return initialize_runtime_graph(graph, program=program)
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("runtime graph state must be a JSON object")
    expected = graph_fingerprint(graph) if graph is not None else ""
    if str(payload.get("base_graph_fingerprint", "")) != expected:
        raise ValueError("runtime graph state belongs to a different base graph")
    if not isinstance(payload.get("committed_patches"), list) or not isinstance(payload.get("effective_state"), dict):
        raise ValueError("runtime graph state is incomplete")
    foreshadows = payload.get("effective_foreshadows", {})
    if not isinstance(foreshadows, dict):
        raise ValueError("runtime graph foreshadow state is incomplete")
    # Runtime graph files are durable between chapters.  Keep the first
    # version readable while upgrading it in memory, rather than forcing the
    # user to delete an otherwise valid generation run.
    committed_chapter_ids = payload.get("committed_chapter_ids")
    if not isinstance(committed_chapter_ids, list):
        committed_chapter_ids = [
            str(item.get("chapter_id", ""))
            for item in payload["committed_patches"]
            if isinstance(item, dict) and str(item.get("chapter_id", "")).strip()
        ]
    return {
        **payload,
        "schema_version": "1.1",
        "committed_chapter_ids": list(dict.fromkeys(str(item) for item in committed_chapter_ids if str(item).strip())),
    }


def _event_and_fact_ids(scene: SceneProgram, paragraph: ParagraphProgram) -> tuple[set[str], set[str]]:
    beats = {item.beat_id: item for item in scene.narrative_beats}
    fact_ids = set(paragraph.required_fact_ids)
    event_ids = set(paragraph.event_ids)
    for beat_id in paragraph.beat_ids:
        beat = beats.get(beat_id)
        if beat is not None:
            fact_ids.update(beat.required_fact_ids)
            fact_ids.update(beat.support_fact_ids)
            event_ids.update(beat.event_ids)
    events = {item.event_id: item for item in scene.event_programs}
    for event_id in tuple(event_ids):
        event = events.get(event_id)
        if event is not None:
            fact_ids.update(event.precondition_fact_ids)
            fact_ids.update(event.required_fact_ids)
    return fact_ids, event_ids


def _scene_facts(scene: SceneProgram) -> dict[str, Any]:
    """Expose local facts plus read-only preceding-state context."""

    return {item.fact_id: item for item in (*scene.fact_contracts, *scene.context_fact_contracts)}


def _paragraph_entity_ids(scene: SceneProgram, paragraph: ParagraphProgram) -> set[str]:
    """Return only entities directly writeable in one paragraph window.

    A compiled scene may contain several sequential events.  Its full
    participant list is therefore not safe writer context: it can reveal the
    speaker or location of a later paragraph and invites premature dialogue.
    """

    fact_ids, event_ids = _event_and_fact_ids(scene, paragraph)
    facts = _scene_facts(scene)
    events = {item.event_id: item for item in scene.event_programs}
    beat_by_id = {item.beat_id: item for item in scene.narrative_beats}
    entity_ids = {paragraph.viewpoint_entity_id} if paragraph.viewpoint_entity_id else set()
    for fact_id in fact_ids:
        fact = facts.get(fact_id)
        if fact is not None:
            entity_ids.add(fact.subject_id)
            if fact.object_id:
                entity_ids.add(fact.object_id)
    for event_id in event_ids:
        event = events.get(event_id)
        if event is not None:
            entity_ids.update(event.participant_ids)
    for beat_id in paragraph.beat_ids:
        beat = beat_by_id.get(beat_id)
        if beat is not None and beat.actor_id:
            entity_ids.add(beat.actor_id)
    return entity_ids


def _program_contract_subgraph(
    runtime: dict[str, Any],
    program: ChapterProgram,
    scene: SceneProgram,
    paragraph: ParagraphProgram,
) -> dict[str, Any]:
    fact_ids, event_ids = _event_and_fact_ids(scene, paragraph)
    facts = _scene_facts(scene)
    events = {item.event_id: item for item in scene.event_programs}
    names = {item.entity_id: item.canonical_name for item in program.entities}
    entity_ids = _paragraph_entity_ids(scene, paragraph)
    return {
        "模式": "程序契约子图",
        "章节ID": program.chapter_id,
        "场景ID": scene.scene_id,
        "段落ID": paragraph.paragraph_id,
        "实体": [{"实体ID": item, "名称": names.get(item, item)} for item in sorted(entity_ids)],
        "允许事实": [{
            "事实ID": item.fact_id, "类型": item.fact_kind, "主体": item.subject_id,
            "谓词": item.predicate, "值": item.value, "客体": item.object_id,
        } for item in (*scene.fact_contracts, *scene.context_fact_contracts) if item.fact_id in fact_ids],
        "允许事件": [{
            "事件ID": item.event_id, "动作": item.action, "参与者": list(item.participant_ids),
            "前提事实ID": list(item.precondition_fact_ids), "结果事实ID": list(item.outcome_fact_ids),
            "行动类型": item.action_type, "行动者ID": item.actor_id,
            "受事者ID": list(item.target_ids), "行动依据事实ID": list(item.basis_fact_ids),
        } for item in scene.event_programs if item.event_id in event_ids],
        "关系与因果": [],
        "已提交状态": _state_for_entities(runtime, entity_ids),
        "硬约束": ["仅可使用本子图列出的实体、事实、事件与已提交状态", "不得添加未列出的关系、结果或后续事件"],
    }


def _state_for_entities(runtime: dict[str, Any], entity_ids: Iterable[str]) -> list[dict[str, Any]]:
    wanted = set(entity_ids)
    result: list[dict[str, Any]] = []
    for key, value in runtime.get("effective_state", {}).items():
        if not isinstance(value, dict):
            continue
        if str(value.get("subject_id", "")) in wanted:
            result.append({"状态键": key, **value})
    return result


def _foreshadow_participants(graph: NarrativeGraph, foreshadow_id: str) -> tuple[str, ...]:
    """Return only graph entity IDs structurally attached to a hint."""

    opened_by = {
        edge.source_node_id
        for edge in graph.edges
        if edge.kind == "foreshadow_open" and edge.target_node_id == foreshadow_id
    }
    participants = {
        edge.source_node_id
        for edge in graph.edges
        if edge.target_node_id in opened_by and edge.kind in {"asserts", "event_participant"}
    }
    return tuple(sorted(participants))


def _runtime_foreshadows_for_entities(runtime: dict[str, Any], entity_ids: Iterable[str]) -> list[dict[str, Any]]:
    wanted = set(entity_ids)
    result: list[dict[str, Any]] = []
    for item in runtime.get("effective_foreshadows", {}).values():
        if not isinstance(item, dict) or str(item.get("lifecycle", "")) != "open":
            continue
        participants = {str(value) for value in item.get("participant_ids", []) if str(value).strip()}
        if participants & wanted:
            result.append({
                "节点ID": str(item.get("foreshadow_id", "")),
                "类型": "foreshadow",
                "生命周期": "open",
                "来源": "已提交图谱补丁",
            })
    return result


def _has_confirmed_foreshadow_resolution(graph: NarrativeGraph, foreshadow_id: str) -> bool:
    """A cue is writable as a foreshadow only when the corpus proves a payoff.

    The corpus graph can contain unresolved mystery-like cues.  Treating every
    one as a foreshadow teaches the generator to dig arbitrary holes.  A node
    is therefore exposed to prose generation only when a later source-backed
    resolving event is present in the graph; the identity of that event remains
    hidden until its own patch is committed.
    """

    return any(
        edge.kind == "foreshadow_resolved" and edge.target_node_id == foreshadow_id
        for edge in graph.edges
    )


def narrative_subgraph_for_paragraph(
    graph: NarrativeGraph | None,
    runtime: dict[str, Any],
    program: ChapterProgram,
    scene: SceneProgram,
    paragraph: ParagraphProgram,
    *,
    chapter_contract: ChapterContract | None = None,
    working_ledger: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retrieve the time-safe graph neighbourhood for one prose paragraph."""

    if graph is None:
        return _program_contract_subgraph(runtime, program, scene, paragraph)
    sources = _node_by_source(graph)
    fact_ids, event_ids = _event_and_fact_ids(scene, paragraph)
    fact_nodes = {
        sources[_source_ref(program.chapter_id, "fact", item)].node_id
        for item in fact_ids if _source_ref(program.chapter_id, "fact", item) in sources
    }
    event_nodes = {
        sources[_source_ref(program.chapter_id, "event", item)].node_id
        for item in event_ids if _source_ref(program.chapter_id, "event", item) in sources
    }
    entity_local_ids = _paragraph_entity_ids(scene, paragraph)
    entity_nodes = {
        sources[_source_ref(program.chapter_id, "entity", item)].node_id
        for item in entity_local_ids if _source_ref(program.chapter_id, "entity", item) in sources
    }
    licensed_source_ids = {
        *(_source_ref(program.chapter_id, "fact", item) for item in fact_ids),
        *(_source_ref(program.chapter_id, "event", item) for item in event_ids),
    }
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    committed_source_ids: set[str] = set()
    if isinstance(working_ledger, dict):
        committed_source_ids.update(
            _source_ref(program.chapter_id, "fact", str(item))
            for item in working_ledger.get("realized_fact_ids", []) if str(item).strip()
        )
        committed_source_ids.update(
            _source_ref(program.chapter_id, "event", str(item))
            for item in working_ledger.get("realized_event_ids", []) if str(item).strip()
        )
    required_prior_event_ids = chapter_contract.required_prior_event_ids if chapter_contract is not None else ()
    retrieval = retrieve_graph_neighbourhood(
        graph,
        chapter_id=program.chapter_id,
        seed_node_ids=(*fact_nodes, *event_nodes, *entity_nodes),
        licensed_source_ids=licensed_source_ids,
        committed_source_ids=committed_source_ids,
        required_prior_event_ids=required_prior_event_ids,
        max_hops=DEFAULT_GRAPH_RAG_MAX_HOPS,
        node_budget=DEFAULT_GRAPH_RAG_NODE_BUDGET,
        edge_budget=DEFAULT_GRAPH_RAG_EDGE_BUDGET,
    )
    selected_nodes = [
        nodes_by_id[node_id]
        for node_id in retrieval.node_ids
        if node_id in nodes_by_id
        and (nodes_by_id[node_id].kind != "foreshadow" or _has_confirmed_foreshadow_resolution(graph, node_id))
    ]
    selected_node_ids = {node.node_id for node in selected_nodes}
    selected_edge_ids = set(retrieval.edge_ids)
    selected_edges = [
        edge for edge in graph.edges
        if edge.edge_id in selected_edge_ids
        and edge.source_node_id in selected_node_ids and edge.target_node_id in selected_node_ids
    ]
    visible_foreshadows: dict[str, tuple[str, str]] = {}
    resolving_events = {edge.target_node_id: edge.source_node_id for edge in selected_edges if edge.kind == "foreshadow_resolved"}
    for node in selected_nodes:
        if node.kind != "foreshadow":
            continue
        resolved_by = resolving_events.get(node.node_id, "")
        visible_foreshadows[node.node_id] = ("resolved" if resolved_by else "open", resolved_by)
    runtime_foreshadows = _runtime_foreshadows_for_entities(runtime, entity_nodes)
    selected_node_payloads = [
        _compact_node(
            node,
            foreshadow_lifecycle=visible_foreshadows.get(node.node_id, ("", ""))[0],
            resolved_by_event_id=visible_foreshadows.get(node.node_id, ("", ""))[1],
            retrieval_reasons=retrieval.reasons.get(node.node_id, ()),
        )
        for node in selected_nodes
    ]
    known_node_ids = {item["节点ID"] for item in selected_node_payloads}
    selected_node_payloads.extend(item for item in runtime_foreshadows if item["节点ID"] and item["节点ID"] not in known_node_ids)
    return {
        "模式": "时间安全的结构化Graph-RAG子图",
        "图谱指纹": graph_fingerprint(graph),
        "章节ID": program.chapter_id,
        "场景ID": scene.scene_id,
        "段落ID": paragraph.paragraph_id,
        "节点": selected_node_payloads,
        "关系与因果": [_edge_compact(edge) for edge in selected_edges],
        "检索摘要": {
            "策略": "直接义务种子 + 因果父链 + 状态槽 + 人物关系 + 相关历史事件",
            "种子节点数": retrieval.seed_count,
            "历史节点数": retrieval.historical_count,
            "已验证同章节点数": retrieval.committed_same_chapter_count,
            "最大扩展跳数": retrieval.max_hops,
            "节点预算": retrieval.node_budget,
            "关系预算": retrieval.edge_budget,
        },
        "已提交状态": _state_for_entities(runtime, entity_nodes | entity_local_ids),
        "硬约束": ["本子图不含原文引文；不得索取或补写整章原文", "仅可使用节点、关系、因果和已提交状态中的信息", "历史检索结果只用于保持连续性，不得重复演出已经发生的事件", "未出现的伏笔仍为未解状态，不得提前兑现"],
    }


def _paragraph_program_payload(paragraph: ParagraphProgram) -> dict[str, Any]:
    function_names = {
        "orientation": "场景定位", "action_progression": "行动推进", "obstacle": "阻力显现",
        "perception_reaction": "感知反应", "dialogue_conflict": "对话交锋", "turn": "局势转折",
        "decision": "做出决定", "transition": "场景转场", "aftermath": "余波收束",
    }
    return {
        "段落ID": paragraph.paragraph_id,
        "段落序号": paragraph.order,
        "段落功能": function_names.get(paragraph.function, paragraph.function),
        "必须表达的事实ID": list(paragraph.required_fact_ids),
        "事件ID列表": list(paragraph.event_ids),
        "视角人物ID": paragraph.viewpoint_entity_id,
        "对话行为": paragraph.dialogue_act,
        "节拍ID列表": list(paragraph.beat_ids),
        "目标字数": paragraph.target_chars,
        "最低字数": paragraph.minimum_chars,
    }


def paragraph_writer_packet(
    graph: NarrativeGraph | None,
    runtime: dict[str, Any],
    program: ChapterProgram,
    scene: SceneProgram,
    paragraph: ParagraphProgram,
    *,
    prior_paragraph: dict[str, Any] | None = None,
    chapter_contract: ChapterContract | None = None,
    event_graph: EventGraph | None = None,
    working_ledger: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile the sole model input for one new paragraph.

    This package deliberately excludes the complete chapter program, other
    paragraphs and every source span.  A short tail of *already generated*
    prose is permitted only as a hand-off for local cohesion; it is never
    corpus prose and cannot reveal a future event.
    """

    fact_ids, event_ids = _event_and_fact_ids(scene, paragraph)
    entities = {item.entity_id: item for item in program.entities}
    licensed_entity_ids = _paragraph_entity_ids(scene, paragraph)
    facts = _scene_facts(scene)
    events = {item.event_id: item for item in scene.event_programs}
    beat_ids = set(paragraph.beat_ids)
    beat_names = {
        "setup": "起势", "action": "行动", "pressure": "压力", "reaction": "反应",
        "choice": "选择", "consequence": "结果", "transition": "转场", "aftermath": "余波",
    }
    license_names = {"source": "原文复现", "inference": "合理推导", "atmosphere": "氛围扩写"}
    beats = [
        {
            "节拍ID": item.beat_id,
            "节拍类型": beat_names.get(item.beat_type, item.beat_type),
            "行动者ID": item.actor_id,
            "人物目标": item.objective,
            "压力或阻力": item.pressure,
            "可见动作": item.observable_action,
            "视角细节": item.focal_detail,
            "状态变化": item.state_change,
            "对白压力": item.dialogue_pressure,
            "扩写许可": license_names.get(item.expansion_license, item.expansion_license),
            "依据事实ID": list(item.support_fact_ids),
            "叙事机制ID": list(item.mechanism_ids),
            "叙事任务": item.narrative_function,
            "反事实守卫": item.counterfactual_guard,
        }
        for item in scene.narrative_beats if item.beat_id in beat_ids
    ]
    window_goal = "；".join(dict.fromkeys(
        str(item.get("人物目标", "")).strip()
        for item in beats if str(item.get("人物目标", "")).strip()
    ))
    if not window_goal:
        window_goal = "；".join(events[event_id].action for event_id in event_ids if event_id in events)
    handoff: dict[str, Any] = {"类型": "章节起始", "上一段落ID": "", "已兑现事实ID": [], "已兑现事件ID": [], "结尾": ""}
    if isinstance(prior_paragraph, dict):
        text = str(prior_paragraph.get("text", "")).strip()
        handoff = {
            "类型": "已生成正文交接",
            "上一段落ID": str(prior_paragraph.get("paragraph_id", "")),
            "已兑现事实ID": [str(item) for item in prior_paragraph.get("claimed_fact_ids", []) if str(item).strip()],
            "已兑现事件ID": [str(item) for item in prior_paragraph.get("claimed_event_ids", []) if str(item).strip()],
            "结尾": text[-220:],
        }
    local_ledger = working_ledger_context(working_ledger, scene, paragraph) if working_ledger is not None else {}
    if local_ledger:
        handoff = dict(local_ledger.get("上一段交接", handoff))
        handoff["类型"] = "已验证正文交接" if handoff.get("段落ID") else "章节起始"
        handoff["上一段落ID"] = handoff.pop("段落ID", "")
    macro_context = (
        chapter_contract_writer_context(
            event_graph, chapter_contract, program, paragraph_event_ids=event_ids,
        )
        if chapter_contract is not None and event_graph is not None
        else {}
    )
    return {
        "写作包版本": "1.2",
        "章节ID": program.chapter_id,
        "生成模式": {"faithful": "忠实复现", "controlled_expansion": "受控扩写"}[program.generation_mode],
        "当前场景": {"场景ID": scene.scene_id, "场景目标": window_goal},
        "当前段落程序": _paragraph_program_payload(paragraph),
        "许可实体": [
            {
                "实体ID": entity_id,
                "图谱实体ID": _entity_graph_id(graph, program.chapter_id, entity_id),
                "名称": entities[entity_id].canonical_name,
                "类型": entities[entity_id].kind,
            }
            for entity_id in sorted(licensed_entity_ids) if entity_id in entities
        ],
        "当前段落节拍": beats,
        "前文交接": handoff,
        "章节合同窗口": macro_context,
        "本章段落状态簿": local_ledger,
        "叙事事实子图": narrative_subgraph_for_paragraph(
            graph,
            runtime,
            program,
            scene,
            paragraph,
            chapter_contract=chapter_contract,
            working_ledger=working_ledger,
        ),
        "硬约束": [
            "只写当前段落；不得读取、索取或补写整章来源正文。",
            "只能兑现当前段落程序和事实子图许可的事实、事件、关系、时间、空间与状态。",
            "许可实体中图谱实体ID相同的局部ID指向同一实体，不得写成两个人或两件物。",
            "前文交接仅来自本次已生成正文，不能据此新增未获许可的事实或提前兑现后续事件。",
            "章节合同窗口只开放当前段落可推进的事件；未来事件即使存在于底层图谱中也不可写入。",
            "本章段落状态簿只记录已经通过校验的前段结果；当前段落必须通过校验后才能更新它。",
        ],
    }


def _entity_graph_id(graph: NarrativeGraph | None, chapter_id: str, local_id: str) -> str:
    if graph is None:
        return local_id
    node = _node_by_source(graph).get(_source_ref(chapter_id, "entity", local_id))
    return node.node_id if node is not None else local_id


def _state_slot(fact: Any, object_id: str) -> str:
    if fact.fact_kind == "location":
        return "location:current"
    if fact.fact_kind in {"identity", "goal"}:
        return f"{fact.fact_kind}:{fact.predicate}"
    if fact.fact_kind == "relationship":
        # Relationship facts often encode the role in ``value`` (for example
        # “王林 — 亲属 — 父亲 — 王天水”), while other extractors put it in the
        # predicate.  Key stable kinship by the role, not by the other person,
        # so a later father/uncle swap becomes a detectable conflict.
        role = _relationship_role(fact)
        return f"relationship:{role}"
    if fact.fact_kind == "progression":
        level_cues = ("境界", "修为", "修炼", "突破", "层", "阶", "级")
        if any(cue in f"{fact.predicate} {fact.value}" for cue in level_cues):
            return "progression:level"
    return f"{fact.fact_kind}:{fact.predicate}:{object_id or fact.value}"


def _relationship_role(fact: Any) -> str:
    predicate = str(getattr(fact, "predicate", "")).strip()
    value = str(getattr(fact, "value", "")).strip()
    candidates = (value, predicate)
    for candidate in candidates:
        if any(cue in candidate for cue in _IMMUTABLE_RELATIONSHIP_CUES):
            return candidate
    return f"{predicate}:{value}".strip(":") or "unspecified"


def _is_immutable_relationship(fact: Any) -> bool:
    return any(cue in _relationship_role(fact) for cue in _IMMUTABLE_RELATIONSHIP_CUES)


def _is_immutable_identity(fact: Any) -> bool:
    text = f"{getattr(fact, 'predicate', '')} {getattr(fact, 'value', '')}"
    return any(cue in text for cue in _IMMUTABLE_IDENTITY_CUES)


def _scene_validation_snapshot(scene_results: list[dict[str, Any]]) -> tuple[set[str], set[str], list[str]]:
    facts: set[str] = set()
    events: set[str] = set()
    issues: list[str] = []
    for result in scene_results:
        validations = result.get("validations", []) if isinstance(result, dict) else []
        latest = validations[-1] if isinstance(validations, list) and validations and isinstance(validations[-1], dict) else {}
        if not bool(result.get("passed")) or not bool(result.get("prose_passed", True)) or not bool(result.get("role_passed", True)):
            issues.append(f"scene:{result.get('scene_id', '')}:validation_failed")
        facts.update(str(item) for item in latest.get("realized_fact_ids", []) if str(item).strip())
        events.update(str(item) for item in latest.get("realized_event_ids", []) if str(item).strip())
    return facts, events, issues


def _program_fact_and_event_maps(program: ChapterProgram) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {item.fact_id: item for scene in program.scene_programs for item in scene.fact_contracts},
        {item.event_id: item for scene in program.scene_programs for item in scene.event_programs},
    )


def _paragraph_binding_positions(program: ChapterProgram) -> tuple[dict[str, int], dict[str, int]]:
    """Return the first *explicitly scheduled* fact and event positions.

    ``_event_and_fact_ids`` intentionally enriches a writer packet with an
    event's prerequisites.  It must not be used here: otherwise a prerequisite
    scheduled after an event appears to be available simply because the event
    referenced it, hiding exactly the causal inversion this validator exists
    to catch.
    """

    fact_positions: dict[str, int] = {}
    event_positions: dict[str, int] = {}
    position = 0
    for scene in program.scene_programs:
        beats = {item.beat_id: item for item in scene.narrative_beats}
        for paragraph in scene.paragraphs:
            fact_ids = set(paragraph.required_fact_ids)
            event_ids = set(paragraph.event_ids)
            for beat_id in paragraph.beat_ids:
                beat = beats.get(beat_id)
                if beat is not None:
                    fact_ids.update(beat.required_fact_ids)
                    fact_ids.update(beat.support_fact_ids)
                    event_ids.update(beat.event_ids)
            for fact_id in fact_ids:
                fact_positions.setdefault(fact_id, position)
            for event_id in event_ids:
                event_positions.setdefault(event_id, position)
            position += 1
    return fact_positions, event_positions


def _has_directed_cycle(edges: Iterable[tuple[str, str]]) -> bool:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for source, target in edges:
        if source != target:
            adjacency[source].add(target)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(next_node) for next_node in adjacency.get(node, ())):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in tuple(adjacency) if node not in visited)


def _causal_links_and_issues(program: ChapterProgram, fact_by_id: dict[str, Any], event_by_id: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Verify that event conditions occur before their effects in prose order."""

    fact_positions, event_positions = _paragraph_binding_positions(program)
    producer_by_fact = {
        fact_id: event.event_id
        for event in event_by_id.values()
        for fact_id in event.outcome_fact_ids
    }
    links: list[dict[str, Any]] = []
    issues: list[str] = []
    graph_edges: list[tuple[str, str]] = []
    for event in event_by_id.values():
        event_position = event_positions.get(event.event_id)
        if event_position is None:
            issues.append(f"event_not_bound_to_paragraph:{event.event_id}")
            continue
        for fact_id in event.precondition_fact_ids:
            fact_position = fact_positions.get(fact_id)
            if fact_id not in fact_by_id:
                issues.append(f"unknown_causal_precondition:{event.event_id}:{fact_id}")
                continue
            if fact_position is None:
                issues.append(f"precondition_not_bound_to_paragraph:{event.event_id}:{fact_id}")
            elif fact_position > event_position:
                issues.append(f"precondition_after_event:{event.event_id}:{fact_id}")
            producer = producer_by_fact.get(fact_id)
            if producer and producer != event.event_id:
                links.append({"起因事件ID": producer, "结果事件ID": event.event_id, "依据事实ID": fact_id})
                graph_edges.append((producer, event.event_id))
        for fact_id in event.outcome_fact_ids:
            fact_position = fact_positions.get(fact_id)
            if fact_position is None:
                issues.append(f"outcome_not_bound_to_paragraph:{event.event_id}:{fact_id}")
            elif fact_position < event_position:
                issues.append(f"outcome_before_event:{event.event_id}:{fact_id}")
    if _has_directed_cycle(graph_edges):
        issues.append("causal_cycle")
    return links, issues


def _repetitive_failure_loop_issues(program: ChapterProgram, scene_results: list[dict[str, Any]]) -> list[str]:
    """Block a third unearned failure-humiliation beat within one chapter.

    This is intentionally a high-precision guard rather than a general style
    score.  Repeating one setback can be a deliberate escalation; repeating
    the same failure, humiliation and unwillingness cluster three times with
    no new event is the exact flat-loop failure observed in generated samples.
    """

    paragraph_to_event_ids = {
        paragraph.paragraph_id: set(paragraph.event_ids)
        for scene in program.scene_programs for paragraph in scene.paragraphs
    }
    loop_paragraphs: list[tuple[str, set[str]]] = []
    for result in scene_results:
        draft = result.get("draft", {}) if isinstance(result, dict) else {}
        paragraphs = draft.get("paragraphs", []) if isinstance(draft, dict) else []
        for paragraph in paragraphs:
            if not isinstance(paragraph, dict):
                continue
            text = str(paragraph.get("text", ""))
            cue_count = sum(cue in text for cue in _FAILURE_LOOP_CUES)
            if cue_count >= 2:
                paragraph_id = str(paragraph.get("paragraph_id", ""))
                claimed_events = {str(item) for item in paragraph.get("claimed_event_ids", []) if str(item).strip()}
                loop_paragraphs.append((paragraph_id, claimed_events | paragraph_to_event_ids.get(paragraph_id, set())))
    if len(loop_paragraphs) < 3:
        return []
    third_paragraph, third_events = loop_paragraphs[2]
    previous_events = set().union(*(events for _, events in loop_paragraphs[:2]))
    if third_events - previous_events:
        return []
    return [f"repetitive_failure_humiliation_loop:{third_paragraph}"]


def _state_update_for_fact(
    graph: NarrativeGraph | None,
    program: ChapterProgram,
    scene: SceneProgram,
    fact: Any,
    effective: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    subject_id = _entity_graph_id(graph, program.chapter_id, fact.subject_id)
    object_id = _entity_graph_id(graph, program.chapter_id, fact.object_id) if fact.object_id else ""
    slot = _state_slot(fact, object_id)
    state_key = f"{subject_id}|{slot}"
    value = object_id or fact.value
    previous = effective.get(state_key)
    operation = "add" if previous is None else ("observe" if str(previous.get("value", "")) == value else "update")
    update = {
        "state_key": state_key,
        "subject_id": subject_id,
        "slot": slot,
        "value": value,
        "object_id": object_id,
        "operation": operation,
        "fact_id": fact.fact_id,
        "scene_id": scene.scene_id,
        "fact_kind": fact.fact_kind,
        "predicate": fact.predicate,
        "relationship_role": _relationship_role(fact) if fact.fact_kind == "relationship" else "",
        "immutable": _is_immutable_relationship(fact) or (fact.fact_kind == "identity" and _is_immutable_identity(fact)),
    }
    effective[state_key] = update
    return update


def _foreshadow_patch_updates(
    graph: NarrativeGraph | None,
    program: ChapterProgram,
    realized_fact_ids: set[str],
    realized_event_ids: set[str],
) -> list[dict[str, Any]]:
    """Record only hints opened or resolved by this accepted chapter.

    The source graph may include future chapters, therefore its final
    lifecycle value is never copied into runtime state.  A resolution becomes
    visible only if the resolving event has been validated in this patch.
    """

    if graph is None:
        return []
    realized_sources = {
        *(_source_ref(program.chapter_id, "fact", item) for item in realized_fact_ids),
        *(_source_ref(program.chapter_id, "event", item) for item in realized_event_ids),
    }
    updates: list[dict[str, Any]] = []
    for node in graph.nodes:
        if node.kind != "foreshadow" or not _has_confirmed_foreshadow_resolution(graph, node.node_id):
            continue
        resolution_edges = [
            edge for edge in graph.edges
            if edge.kind == "foreshadow_resolved" and edge.target_node_id == node.node_id
            and set(edge.source_ids) & realized_sources
        ]
        opened_here = node.chapter_id == program.chapter_id and bool(set(node.source_ids) & realized_sources)
        if not opened_here and not resolution_edges:
            continue
        resolved_by_event_id = ""
        if resolution_edges:
            source_ids = set(resolution_edges[0].source_ids) & realized_sources
            for source_id in source_ids:
                prefix = f"event:{program.chapter_id}:"
                if source_id.startswith(prefix):
                    resolved_by_event_id = source_id[len(prefix):]
                    break
        updates.append({
            "foreshadow_id": node.node_id,
            "lifecycle": "resolved" if resolution_edges else "open",
            "participant_ids": list(_foreshadow_participants(graph, node.node_id)),
            "opened_in_chapter": node.chapter_id,
            "resolved_by_event_id": resolved_by_event_id,
        })
    return updates


def build_chapter_graph_patch(
    graph: NarrativeGraph | None,
    runtime: dict[str, Any],
    program: ChapterProgram,
    scene_results: list[dict[str, Any]],
    *,
    prose_passed: bool,
    style_passed: bool,
    length_passed: bool,
) -> dict[str, Any]:
    """Build a candidate patch from already-validated program claims only."""

    realized_facts, realized_events, issues = _scene_validation_snapshot(scene_results)
    # Exit-state facts are mandatory even when the compiler classified their
    # prose expression as optional.  A state patch may never claim an
    # un-realised change merely because its fact was not a headline beat.
    all_facts = {
        item.fact_id
        for scene in program.scene_programs
        for item in scene.fact_contracts
        if item.must_realize
    }
    all_facts.update(
        fact_id
        for scene in program.scene_programs
        for fact_id in scene.exit_state_fact_ids
    )
    all_events = {item.event_id for scene in program.scene_programs for item in scene.event_programs}
    missing_facts = sorted(all_facts - realized_facts)
    missing_events = sorted(all_events - realized_events)
    if missing_facts:
        issues.append("missing_realized_facts:" + ",".join(missing_facts))
    if missing_events:
        issues.append("missing_realized_events:" + ",".join(missing_events))
    expected_scene_ids = {scene.scene_id for scene in program.scene_programs}
    reported_scene_ids = [str(item.get("scene_id", "")) for item in scene_results if isinstance(item, dict)]
    if set(reported_scene_ids) != expected_scene_ids or len(reported_scene_ids) != len(set(reported_scene_ids)):
        issues.append("scene_results_do_not_cover_program")
    fact_by_id, event_by_id = _program_fact_and_event_maps(program)
    updates: list[dict[str, Any]] = []
    effective = {str(key): dict(value) for key, value in runtime.get("effective_state", {}).items() if isinstance(value, dict)}
    causal_links, causal_issues = _causal_links_and_issues(program, fact_by_id, event_by_id)
    issues.extend(causal_issues)
    for scene in program.scene_programs:
        for fact in scene.fact_contracts:
            if fact.fact_id not in realized_facts or fact.fact_kind not in STATE_BEARING_FACT_KINDS:
                continue
            updates.append(_state_update_for_fact(graph, program, scene, fact, effective))
    foreshadow_updates = _foreshadow_patch_updates(graph, program, realized_facts, realized_events)
    repetition_issues = _repetitive_failure_loop_issues(program, scene_results)
    issues.extend(repetition_issues)
    checks = {
        "scene_validation_passed": not any(item.startswith("scene:") for item in issues),
        "prose_passed": prose_passed,
        "style_passed": style_passed,
        "length_passed": length_passed,
        "missing_fact_ids": missing_facts,
        "missing_event_ids": missing_events,
        "causal_link_count": len(causal_links),
        "foreshadow_update_count": len(foreshadow_updates),
        "repetition_issue_count": len(repetition_issues),
        "issues": issues,
    }
    patch_id = hashlib.sha256((program.chapter_id + program.source_hash + json.dumps({"state": updates, "foreshadows": foreshadow_updates}, ensure_ascii=False, sort_keys=True)).encode("utf-8")).hexdigest()[:20]
    return {
        "schema_version": "1.0", "patch_id": f"patch:{patch_id}", "status": "candidate",
        "base_graph_fingerprint": graph_fingerprint(graph) if graph is not None else "",
        "chapter_id": program.chapter_id, "program_source_hash": program.source_hash,
        "realized_fact_ids": sorted(realized_facts), "realized_event_ids": sorted(realized_events),
        "state_updates": updates, "causal_links": causal_links, "foreshadow_updates": foreshadow_updates, "checks": checks,
    }


def validate_graph_patch(
    graph: NarrativeGraph | None,
    runtime: dict[str, Any],
    patch: dict[str, Any],
    *,
    program: ChapterProgram | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    expected = graph_fingerprint(graph) if graph is not None else ""
    if str(patch.get("status", "")) != "candidate":
        issues.append("patch_not_candidate")
    if str(patch.get("base_graph_fingerprint", "")) != expected:
        issues.append("base_graph_mismatch")
    if program is not None:
        if str(patch.get("chapter_id", "")) != program.chapter_id or str(patch.get("program_source_hash", "")) != program.source_hash:
            issues.append("patch_program_mismatch")
        known_facts, known_events = _program_fact_and_event_maps(program)
        if not set(str(item) for item in patch.get("realized_fact_ids", [])).issubset(known_facts):
            issues.append("patch_claims_unknown_fact")
        if not set(str(item) for item in patch.get("realized_event_ids", [])).issubset(known_events):
            issues.append("patch_claims_unknown_event")
    committed_chapters = {str(item) for item in runtime.get("committed_chapter_ids", []) if str(item).strip()}
    if str(patch.get("chapter_id", "")) in committed_chapters:
        issues.append("chapter_already_committed")
    checks = patch.get("checks", {}) if isinstance(patch.get("checks"), dict) else {}
    if checks.get("issues") or not all(bool(checks.get(item)) for item in ("scene_validation_passed", "prose_passed", "style_passed", "length_passed")):
        issues.append("generation_validation_not_passed")
    active = {str(key): dict(value) for key, value in runtime.get("effective_state", {}).items() if isinstance(value, dict)}
    for update in patch.get("state_updates", []):
        if not isinstance(update, dict):
            issues.append("invalid_state_update")
            continue
        key = str(update.get("state_key", ""))
        operation = str(update.get("operation", ""))
        if not key or operation not in {"add", "update", "observe"}:
            issues.append("invalid_state_operation")
            continue
        previous = active.get(key)
        if operation == "add" and previous is not None:
            issues.append(f"state_add_conflict:{key}")
        if operation == "update" and previous is None:
            issues.append(f"state_update_without_prior:{key}")
        if str(update.get("fact_kind", "")) == "relationship" and previous is not None:
            old_value = str(previous.get("value", ""))
            new_value = str(update.get("value", ""))
            immutable = bool(update.get("immutable")) or bool(previous.get("immutable"))
            if immutable and old_value and new_value and old_value != new_value:
                issues.append(f"immutable_relationship_conflict:{key}")
        if str(update.get("fact_kind", "")) == "identity" and previous is not None:
            old_value = str(previous.get("value", ""))
            new_value = str(update.get("value", ""))
            immutable = bool(update.get("immutable")) or bool(previous.get("immutable"))
            if immutable and old_value and new_value and old_value != new_value:
                issues.append(f"immutable_identity_conflict:{key}")
        if program is not None:
            fact = known_facts.get(str(update.get("fact_id", "")))
            if fact is None:
                issues.append(f"state_update_unknown_fact:{update.get('fact_id', '')}")
            else:
                expected_subject = _entity_graph_id(graph, program.chapter_id, fact.subject_id)
                expected_object = _entity_graph_id(graph, program.chapter_id, fact.object_id) if fact.object_id else ""
                expected_slot = _state_slot(fact, expected_object)
                if (
                    str(update.get("subject_id", "")) != expected_subject
                    or str(update.get("object_id", "")) != expected_object
                    or str(update.get("slot", "")) != expected_slot
                    or str(update.get("value", "")) != (expected_object or fact.value)
                    or str(update.get("fact_kind", "")) != fact.fact_kind
                    or str(update.get("predicate", "")) != fact.predicate
                ):
                    issues.append(f"state_update_contract_mismatch:{fact.fact_id}")
        active[key] = update
    foreshadow_updates = patch.get("foreshadow_updates", [])
    if not isinstance(foreshadow_updates, list):
        issues.append("invalid_foreshadow_updates")
        foreshadow_updates = []
    known_foreshadows = {
        node.node_id: node
        for node in graph.nodes
        if node.kind == "foreshadow"
    } if graph is not None else {}
    realized_sources = {
        *(_source_ref(str(patch.get("chapter_id", "")), "fact", str(item)) for item in patch.get("realized_fact_ids", []) if str(item).strip()),
        *(_source_ref(str(patch.get("chapter_id", "")), "event", str(item)) for item in patch.get("realized_event_ids", []) if str(item).strip()),
    }
    seen_foreshadows: set[str] = set()
    for update in foreshadow_updates:
        if not isinstance(update, dict):
            issues.append("invalid_foreshadow_update")
            continue
        foreshadow_id = str(update.get("foreshadow_id", ""))
        lifecycle = str(update.get("lifecycle", ""))
        if not foreshadow_id or lifecycle not in {"open", "resolved"}:
            issues.append("invalid_foreshadow_update")
            continue
        if foreshadow_id in seen_foreshadows:
            issues.append(f"duplicate_foreshadow_update:{foreshadow_id}")
            continue
        seen_foreshadows.add(foreshadow_id)
        if graph is None:
            issues.append(f"foreshadow_update_without_graph:{foreshadow_id}")
            continue
        node = known_foreshadows.get(foreshadow_id)
        if node is None:
            issues.append(f"unknown_foreshadow:{foreshadow_id}")
            continue
        if not _has_confirmed_foreshadow_resolution(graph, foreshadow_id):
            issues.append(f"unconfirmed_foreshadow:{foreshadow_id}")
        if lifecycle == "resolved":
            resolved = any(
                edge.kind == "foreshadow_resolved"
                and edge.target_node_id == foreshadow_id
                and set(edge.source_ids) & realized_sources
                for edge in graph.edges
            )
            if not resolved:
                issues.append(f"unrealised_foreshadow_resolution:{foreshadow_id}")
            prior = runtime.get("effective_foreshadows", {}).get(foreshadow_id)
            opened_here = node.chapter_id == str(patch.get("chapter_id", "")) and bool(set(node.source_ids) & realized_sources)
            if not isinstance(prior, dict) and not opened_here:
                issues.append(f"resolves_unopened_foreshadow:{foreshadow_id}")
        elif node.chapter_id != str(patch.get("chapter_id", "")) or not set(node.source_ids) & realized_sources:
            issues.append(f"unrealised_foreshadow_opening:{foreshadow_id}")
    return {
        "passed": not issues,
        "issues": issues,
        "state_update_count": len(patch.get("state_updates", [])),
        "foreshadow_update_count": len(foreshadow_updates),
    }


def commit_graph_patch(runtime: dict[str, Any], patch: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    if not bool(validation.get("passed")):
        raise ValueError("cannot commit a graph patch that failed validation")
    committed = json.loads(json.dumps(patch, ensure_ascii=False))
    committed["status"] = "committed"
    committed["commit_validation"] = validation
    existing = runtime.get("committed_patches", [])
    if any(isinstance(item, dict) and item.get("patch_id") == committed["patch_id"] for item in existing):
        return runtime
    state = {str(key): dict(value) for key, value in runtime.get("effective_state", {}).items() if isinstance(value, dict)}
    for update in committed.get("state_updates", []):
        if isinstance(update, dict):
            state[str(update["state_key"])] = update
    foreshadows = {
        str(key): dict(value)
        for key, value in runtime.get("effective_foreshadows", {}).items()
        if isinstance(value, dict)
    }
    for update in committed.get("foreshadow_updates", []):
        if isinstance(update, dict) and str(update.get("foreshadow_id", "")).strip():
            foreshadows[str(update["foreshadow_id"])] = update
    chapter_ids = [str(item) for item in runtime.get("committed_chapter_ids", []) if str(item).strip()]
    chapter_id = str(committed.get("chapter_id", "")).strip()
    if chapter_id and chapter_id not in chapter_ids:
        chapter_ids.append(chapter_id)
    return {
        **runtime,
        "committed_patches": [*existing, committed],
        "effective_state": state,
        "effective_foreshadows": foreshadows,
        "committed_chapter_ids": chapter_ids,
    }


def write_runtime_graph(path: Path, runtime: dict[str, Any]) -> Path:
    write_json(path, runtime)
    return path
