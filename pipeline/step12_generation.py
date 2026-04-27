from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

import config
from pipeline.core.common_json import read_json_file, safe_json_load, write_json_file
from pipeline.core.story_models import CharacterSheet
from pipeline.core.utils import chat_completion_json, get_deepseek_client
from pipeline.core.world_building_core import FusedWorld
from pipeline.step11_reassembly import ReassembledEvent


_STEP12_FILENAME = "step12_volume_outlines.json"
_GENERIC_VOLUME_MOTIFS = ("设局", "识破", "逆袭", "试炼", "拍卖会", "秘境", "暗中相助", "宝珠")


def _quality_chat_completion_json(client, **kwargs) -> str:
    return chat_completion_json(client, model_name=config.DEEPSEEK_LAST_STEPS_MODEL, **kwargs)


@dataclass
class VolumeOutline:
    volume_number: int
    volume_title: str
    realm_range: str
    chapter_summaries: List[str] = field(default_factory=list)
    tension_curve: str = ""
    ending_state: str = ""
    raw_text: str = ""


def generate_volumes(reassembled_events: List[ReassembledEvent], fused_world: FusedWorld, character_sheet: CharacterSheet) -> List[VolumeOutline]:
    client = get_deepseek_client()
    volume_groups = _group_events_into_volumes(reassembled_events)
    total_volumes = len(volume_groups)
    print(f"[Step 12] Generating {total_volumes} volumes. Expanding events into chapters...")

    volumes: List[VolumeOutline] = []
    previous_tail_chapters: List[str] = []
    previous_ending_state = ""

    for volume_number, (arc_name, events) in enumerate(tqdm(volume_groups.items(), desc="[Step 12] Generating volumes", unit="vol"), start=1):
        tqdm.write(f"[Step 12] Generating volume {volume_number}: {arc_name} (events: {len(events)})...")
        stage_network = _get_stage_network_for_volume(character_sheet, volume_number, total_volumes)
        system_context = _build_system_context(fused_world, character_sheet, stage_network)
        outline = _generate_single_volume(
            client=client,
            volume_number=volume_number,
            arc_name=arc_name,
            events=events,
            system_context=system_context,
            previous_ending_state=previous_ending_state,
            previous_tail_chapters=previous_tail_chapters,
        )
        volumes.append(outline)
        previous_ending_state = outline.ending_state
        previous_tail_chapters = outline.chapter_summaries[-5:] if len(outline.chapter_summaries) >= 5 else outline.chapter_summaries

    tqdm.write(f"[Step 12] All {len(volumes)} volumes generated.")
    save_step12_output(volumes)
    return volumes


def save_step12_output(volumes: List[VolumeOutline]) -> Path:
    out_dir = Path(config.INTERMEDIATE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / _STEP12_FILENAME
    write_json_file(path, [asdict(volume) for volume in volumes])
    print(f"[Step 12] Intermediate output saved -> {path.name}")
    return path


def load_step12_output(intermediate_dir: str | Path | None = None) -> List[VolumeOutline]:
    inter_dir = Path(intermediate_dir or config.INTERMEDIATE_DIR)
    path = inter_dir / _STEP12_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Step 12 file not found: {path}")
    data = read_json_file(path)
    return [VolumeOutline(**item) for item in data]


def _group_events_into_volumes(events: List[ReassembledEvent]) -> Dict[str, List[ReassembledEvent]]:
    groups: Dict[str, List[ReassembledEvent]] = {}
    for event in events:
        groups.setdefault(event.arc_name, []).append(event)
    ordered: Dict[str, List[ReassembledEvent]] = {}
    for arc_name in sorted(groups.keys(), key=_arc_sort_key):
        ordered[arc_name] = sorted(groups[arc_name], key=_event_sort_key)
    return ordered


def _get_stage_network_for_volume(char_sheet: CharacterSheet, volume_number: int, total_volumes: int) -> str:
    if not char_sheet.relationship_networks:
        return "暂无关系网络。"
    ratio = (volume_number - 1) / total_volumes if total_volumes > 1 else 0
    stage_index = int(ratio * len(char_sheet.relationship_networks))
    active = char_sheet.relationship_networks[min(stage_index, len(char_sheet.relationship_networks) - 1)]
    return f"【本卷阶段】{active.stage}\n活跃角色：{', '.join(active.active_characters)}\n关系局势：{active.relationship_status}"


def _build_system_context(fused_world: FusedWorld, character_sheet: CharacterSheet, current_stage_network: str) -> str:
    protagonist = character_sheet.protagonist
    realms_text = "\n".join(
        f"{realm.name}(第{realm.level}境): {realm.breakthrough_condition}"
        for realm in fused_world.cultivation_realms
    )
    supporting_text = "\n".join(
        f"- [{item.role}] {item.name} | 执念: {item.dao_heart} | 背景: {item.background}"
        for item in character_sheet.supporting[:20]
    )
    return (
        "===== 世界设定 =====\n"
        f"世界名: {fused_world.world_name}\n"
        f"全局主题: {fused_world.global_theme}\n"
        f"力量来源: {fused_world.power_source}\n"
        f"世界背景: {fused_world.world_background}\n\n"
        f"修炼体系:\n{realms_text}\n\n"
        "===== 角色图鉴 =====\n"
        f"主角: {protagonist.name}\n"
        f"  执念: {protagonist.dao_heart}\n"
        f"  战斗: {protagonist.combat_style}\n"
        f"  缺陷: {protagonist.personality_flaw}\n"
        f"配角:\n{supporting_text}\n\n"
        "===== 当前关系局势 =====\n"
        f"{current_stage_network}\n"
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _generate_single_volume(client, volume_number: int, arc_name: str, events: List[ReassembledEvent], system_context: str, previous_ending_state: str, previous_tail_chapters: List[str]) -> VolumeOutline:
    max_realm_level = max([event.realm_level for event in events] + [1])
    events_lines: List[str] = []
    for idx, event in enumerate(events, start=1):
        plan = event.event_plan or {}
        target_chapter_count = int(plan.get("target_chapter_count") or max(4, len(plan.get("chapter_blueprint", [])) or 4))
        events_lines.append(
            f"核心事件{idx}. [{event.pacing_role}] {event.adapted_summary}\n"
            f"  - 建议章数: {target_chapter_count}\n"
            f"  - 触发: {plan.get('trigger', '')}\n"
            f"  - 目标: {plan.get('goal', '')}\n"
            f"  - 阻碍: {plan.get('obstacle', '')}\n"
            f"  - 转折: {plan.get('reversal', '')}\n"
            f"  - 代价: {plan.get('cost', '')}\n"
            f"  - 活跃角色: {', '.join(event.active_characters) or '待生成'}\n"
            f"  - 章节蓝图: {json.dumps(plan.get('chapter_blueprint', []), ensure_ascii=False)}\n"
        )

    transition_parts: List[str] = []
    if previous_tail_chapters:
        transition_parts.append("【上卷末尾剧情】\n" + "\n".join(previous_tail_chapters))
    if previous_ending_state:
        transition_parts.append(f"【上卷结尾状态】\n{previous_ending_state}")
    transition_text = "\n\n".join(transition_parts)

    user_prompt = (
        f"请为第{volume_number}卷（{arc_name}）生成详细中文章节大纲。\n\n"
        f"{transition_text}\n\n"
        "【本卷必须推进的核心事件骨架】\n"
        f"{chr(10).join(events_lines)}\n\n"
        "【硬性要求】\n"
        "1. 每个核心事件按建议章数展开，允许章节功能重叠。\n"
        f"2. 战力与冲突强度不得明显超出第{max_realm_level}阶段的上限。\n"
        "3. 每章摘要必须是自然中文，不要输出英文说明，不要输出 Python 字典字面量。\n"
        "4. 不要写“按原文”“时间线矛盾”“注：”这类元注释，也不要用回忆插叙打乱当前卷时间线。\n"
        "5. 每章摘要控制在约 80 到 120 个中文字符，前后章必须有因果承接。\n"
        f"{_build_volume_diversity_brief(events)}\n"
        "6. 必须尊重事件计划与状态变化，不能丢失后果、人物关系变化和资源代价。\n"
        "请只返回 JSON，字段包含 volume_title, realm_range, expanded_events, tension_curve, ending_state。"
    )
    system_prompt = (
        "你是一名中文网文大纲策划师。"
        "请写出简洁但连贯的中文章节摘要，保持章节因果连续、人物状态连续、时间线连续。\n\n"
        f"{system_context}"
    )
    raw = _quality_chat_completion_json(client, system=system_prompt, user=user_prompt, json_mode=True)
    data = _safe_json_load(raw)
    expanded_events = _normalize_expanded_events_payload(data.get("expanded_events", []), events)
    expanded_events = _validate_or_repair_volume_payload(client, arc_name, events, expanded_events)
    if not _is_volume_payload_usable(expanded_events, events):
        expanded_events = _build_fallback_expanded_events(events)

    flattened_chapters: List[str] = []
    chapter_index = 1
    for block in expanded_events:
        event_name = str(block.get("event_name") or block.get("event_title") or "主线推进").strip()
        flattened_chapters.append(f"#### 剧情点：{event_name}")
        for chapter in block.get("chapters", []):
            clean_text = _normalize_chapter_summary(chapter)
            flattened_chapters.append(f"**第{chapter_index}章**：{clean_text}")
            chapter_index += 1
        flattened_chapters.append("")

    return VolumeOutline(
        volume_number=volume_number,
        volume_title=data.get("volume_title", f"第{volume_number}卷"),
        realm_range=data.get("realm_range", arc_name),
        chapter_summaries=flattened_chapters,
        tension_curve=_normalize_tension_curve(data.get("tension_curve")),
        ending_state=data.get("ending_state", _build_fallback_ending_state(events)),
        raw_text=raw,
    )


def _build_volume_diversity_brief(events: List[ReassembledEvent]) -> str:
    motif_hits = []
    for motif in _GENERIC_VOLUME_MOTIFS:
        count = sum(1 for event in events if motif in event.adapted_summary)
        if count >= 2:
            motif_hits.append(f"{motif} x{count}")
    lines = ["【本卷反重复提醒】"]
    lines.append("每个事件都必须使用明显不同的戏剧引擎，不能默认回到同一套设局、识破、逆袭。")
    if motif_hits:
        lines.append("骨架里已经重复出现的母题有：" + "、".join(motif_hits) + "。后续不要再把它们当默认解法。")
    else:
        lines.append("当前骨架未检测到特别明显的重复母题，但仍需主动拉开事件质感。")
    return "\n".join(lines)


def _is_volume_payload_usable(expanded_events: Any, source_events: List[ReassembledEvent]) -> bool:
    if not isinstance(expanded_events, list) or not expanded_events:
        return False
    if len(expanded_events) < max(1, len(source_events) - 1):
        return False
    valid_blocks = 0
    for index, block in enumerate(expanded_events):
        chapters = block.get("chapters", []) if isinstance(block, dict) else []
        if isinstance(chapters, list) and any(_normalize_chapter_summary(chapter) for chapter in chapters):
            if index < len(source_events):
                target_count = int((source_events[index].event_plan or {}).get("target_chapter_count") or max(4, len((source_events[index].event_plan or {}).get("chapter_blueprint", [])) or 4))
                if abs(len(chapters) - target_count) > 3:
                    continue
            valid_blocks += 1
    return valid_blocks >= max(1, len(source_events) // 2)


def _build_fallback_expanded_events(events: List[ReassembledEvent]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for event in events:
        plan = event.event_plan or {}
        event_name = _build_fallback_event_name(event)
        blueprint = plan.get("chapter_blueprint", [])
        chapters: List[Dict[str, Any]] = []
        local_chapter_index = 1
        if isinstance(blueprint, list) and blueprint:
            for beat in blueprint:
                primary = beat.get("primary_function", "推进")
                secondary = beat.get("secondary_functions", [])
                secondary_text = f"，并兼顾{'/'.join(secondary)}" if isinstance(secondary, list) and secondary else ""
                purpose = beat.get("purpose", "") or event.adapted_summary
                span = beat.get("chapter_span", [])
                if isinstance(span, list) and len(span) == 2:
                    start, end = int(span[0]), int(span[1])
                else:
                    start, end = 1, 1
                for _chapter_no in range(start, end + 1):
                    chapters.append(
                        {
                            "chapter_number": local_chapter_index,
                            "summary": f"{primary}{secondary_text}：{purpose}",
                        }
                    )
                    local_chapter_index += 1
        if not chapters:
            chapters = [
                {"chapter_number": 1, "summary": f"铺垫目标与阻碍：{event.adapted_summary}"},
                {"chapter_number": 2, "summary": f"冲突升级并迫使角色调整策略：围绕{plan.get('obstacle', event.pacing_role)}展开新的选择与对抗。"},
                {"chapter_number": 3, "summary": f"阶段性收束但留下后续压力：代价落在{plan.get('cost', '资源、人情或伤势')}上，并把主线推向下一步。"},
            ]
        results.append({"event_name": event_name, "chapters": chapters})
    return results


def _build_fallback_event_name(event: ReassembledEvent) -> str:
    summary = (event.adapted_summary or "").strip()
    if not summary:
        return "主线推进"
    return summary.replace("，", " ").replace("。", " ").split(" ", 1)[0][:18] or "主线推进"


def _build_fallback_ending_state(events: List[ReassembledEvent]) -> str:
    if not events:
        return "本卷完成阶段性推进。"
    return f"本卷收束于：{events[-1].adapted_summary}"


def _strip_leading_chapter_label(text: str) -> str:
    if not text:
        return "主线推进。"
    return re.sub(r"^\s*第[0-9一二三四五六七八九十百千]+章[：:]\s*", "", text).strip()


def _arc_sort_key(arc_name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", str(arc_name or ""))
    return (int(match.group(1)) if match else 9999, str(arc_name or ""))


def _event_sort_key(event: ReassembledEvent) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(event.event_id or ""))
    return (int(match.group(1)) if match else 9999, str(event.event_id or ""))


def _normalize_expanded_events_payload(expanded_events: Any, source_events: List[ReassembledEvent]) -> List[Dict[str, Any]]:
    if not isinstance(expanded_events, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for index, block in enumerate(expanded_events):
        if not isinstance(block, dict):
            continue
        event_name = str(block.get("event_name") or block.get("event_title") or f"主线事件{index + 1}").strip()
        raw_chapters = block.get("chapters", [])
        if not isinstance(raw_chapters, list):
            continue
        chapters: List[Dict[str, Any]] = []
        for chapter_index, chapter in enumerate(raw_chapters, start=1):
            summary = _normalize_chapter_summary(chapter)
            if not summary:
                continue
            chapter_number = chapter_index
            if isinstance(chapter, dict):
                try:
                    chapter_number = int(chapter.get("chapter_number", chapter_index))
                except (TypeError, ValueError):
                    chapter_number = chapter_index
            chapters.append({"chapter_number": chapter_number, "summary": summary})
        if chapters:
            normalized.append({"event_name": event_name, "chapters": chapters})
    if not normalized and source_events:
        return _build_fallback_expanded_events(source_events)
    return normalized


def _normalize_chapter_summary(chapter: Any) -> str:
    if isinstance(chapter, str):
        text = chapter.strip()
        if text.startswith("{") and text.endswith("}"):
            parsed = _safe_json_load(text)
            if isinstance(parsed, dict):
                return _normalize_chapter_summary(parsed)
    if isinstance(chapter, dict):
        for key in ("summary", "content", "chapter_summary", "outline"):
            value = chapter.get(key)
            if value:
                return _strip_leading_chapter_label(str(value).strip())
        return ""
    return _strip_leading_chapter_label(str(chapter or "").strip())


def _normalize_tension_curve(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        return text or "本卷张力持续推进。"
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if cleaned:
            return "章节张力大致依次为：" + " → ".join(cleaned)
    return "本卷张力持续推进。"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _validate_or_repair_volume_payload(client, arc_name: str, source_events: List[ReassembledEvent], expanded_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not expanded_events:
        return expanded_events
    event_specs = []
    for index, event in enumerate(source_events, start=1):
        plan = event.event_plan or {}
        blueprint = plan.get("chapter_blueprint", [])
        required_coverage = _required_coverage_from_blueprint(blueprint)
        event_specs.append(
            {
                "event_index": index,
                "event_summary": event.adapted_summary,
                "target_chapter_count": int(plan.get("target_chapter_count") or max(4, len(blueprint) or 4)),
                "required_coverage": required_coverage,
                "chapter_blueprint": blueprint,
            }
        )

    prompt = (
        f"你是卷级章节排布审校器。请检查第{arc_name}卷的章节展开是否真正遵守事件模板，而不是只写出一组松散章节。\n"
        f"事件规格: {json.dumps(event_specs, ensure_ascii=False)}\n"
        f"当前展开结果: {json.dumps(expanded_events, ensure_ascii=False)}\n"
        "检查重点：\n"
        "1. 每个事件的章节数是否大致符合 target_chapter_count；\n"
        "2. 章节排布是否覆盖 required_coverage；\n"
        "3. 事件内部是否有起承转合，而不是只列几个并排短句；\n"
        "4. 相邻事件之间是否有因果承接。\n"
        "只返回 JSON，字段包含 is_valid, issues, repaired_expanded_events。若当前结果可用，repaired_expanded_events 原样返回即可。"
    )
    raw = _quality_chat_completion_json(client, system="只返回合法 JSON。", user=prompt, json_mode=True, temperature=0.2)
    data = _safe_json_load(raw)
    if not isinstance(data, dict):
        return expanded_events
    repaired = _normalize_expanded_events_payload(data.get("repaired_expanded_events", expanded_events), source_events)
    return repaired or expanded_events


def _required_coverage_from_blueprint(blueprint: Any) -> List[str]:
    if not isinstance(blueprint, list):
        return []
    tags: List[str] = []
    for beat in blueprint:
        if not isinstance(beat, dict):
            continue
        primary = str(beat.get("primary_function", "")).strip()
        if primary:
            tags.append(primary)
        for item in beat.get("secondary_functions", []):
            text = str(item).strip()
            if text:
                tags.append(text)
    deduped = []
    for tag in tags:
        if tag not in deduped:
            deduped.append(tag)
    return deduped


_safe_json_load = safe_json_load
