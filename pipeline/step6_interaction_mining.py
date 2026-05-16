"""
step6_interaction_mining.py

Dynamic interaction and pattern mining built on:
candidate mining -> clustering -> consolidation -> ranking -> coverage report.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import config
from pipeline.core.artifacts import interaction_patterns_from_fused_world, save_json_artifact
from pipeline.core.common_text import coerce_text, dedupe_text_values, flatten_text_values, normalize_text
from pipeline.core.world_building_core import (
    FusedWorld,
    KnowledgeBase,
    _character_mentions,
    load_world_snapshot,
    save_world_snapshot,
)
from pipeline.step2_extraction import PlotAtom


_STEP6_FILENAME = "step6_interaction_mining.json"
_STEP6_PATTERNS_FILENAME = "step6_interaction_patterns.json"

_STEP6_MAX_PAIR_CANDIDATES = int(getattr(config, "STEP6_MAX_PAIR_CANDIDATES", 200) or 200)
_STEP6_MAX_FINAL_MICRO = int(getattr(config, "STEP6_MAX_FINAL_MICRO", 80) or 80)
_STEP6_MAX_FINAL_MACRO = int(getattr(config, "STEP6_MAX_FINAL_MACRO", 60) or 60)
_STEP6_MAX_FINAL_THREADS = int(getattr(config, "STEP6_MAX_FINAL_THREADS", 80) or 80)

_MICRO_TYPE_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("压制反制", ("压制", "打压", "羞辱", "反制", "打脸", "翻盘")),
    ("试探博弈", ("试探", "探底", "套话", "博弈", "误导")),
    ("交易争利", ("交易", "交换", "议价", "拍卖", "竞价")),
    ("互救结盟", ("援手", "互救", "结盟", "联手", "合作")),
    ("误会修复", ("误会", "和解", "缓和", "解释", "修复")),
    ("背叛利用", ("背叛", "利用", "借刀", "出卖", "反咬")),
    ("追杀围猎", ("追杀", "围杀", "截杀", "追捕", "堵截")),
    ("传承试炼", ("拜师", "传承", "考核", "试炼", "收徒")),
]

_MACRO_STAGE_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("低位受压", ("被压", "羞辱", "受限", "困境", "质疑")),
    ("资源积累", ("积累", "筹备", "修炼", "布局", "收集")),
    ("局部反击", ("反击", "试探", "回击", "夺回", "破局")),
    ("公开爆发", ("公开", "揭露", "爆发", "对决", "翻盘")),
    ("余波升级", ("余波", "追击", "后果", "升级", "报复")),
    ("关系重估", ("和解", "误会", "结盟", "背叛", "决裂")),
]

_THREAD_TYPE_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("敌对升级线", ("压制", "羞辱", "追杀", "复仇", "清算")),
    ("利益合作线", ("合作", "交易", "联手", "互助", "共赢")),
    ("误会修复线", ("误会", "解释", "和解", "信任", "修复")),
    ("师徒传承线", ("拜师", "传承", "考核", "授法", "护道")),
    ("权力审判线", ("裁决", "规矩", "执法", "审问", "审查")),
]

_CONFLICT_ENGINE_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("规则压迫", ("规则", "规矩", "执法", "考核", "资格")),
    ("资源竞争", ("资源", "宝物", "拍卖", "竞价", "利益")),
    ("信息差", ("秘密", "线索", "真相", "身份", "误导")),
    ("围追堵截", ("追杀", "围杀", "堵截", "追捕", "围攻")),
    ("旧怨升级", ("旧怨", "复仇", "宿敌", "清算", "报复")),
    ("传承筛选", ("传承", "试炼", "考核", "拜师", "择徒")),
]

_RELATION_DELTA_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("恶化", ("敌意", "羞辱", "压制", "追杀", "决裂")),
    ("缓和", ("和解", "解释", "理解", "修复")),
    ("绑定", ("结盟", "互救", "合作", "护送", "共谋")),
    ("反转", ("打脸", "翻盘", "揭露", "看走眼")),
]

_BYSTANDER_KEYWORDS = ("围观", "众人", "长老", "执法", "旁观", "舆论", "全场", "商会")

_RARE_CONFLICT_ENGINES = {"信息差", "传承筛选"}


def mine_story_patterns(
    all_atoms: Dict[str, List[PlotAtom]],
    kb: KnowledgeBase,
    fused_world: FusedWorld,
) -> FusedWorld:
    atoms = [atom for bucket in all_atoms.values() for atom in bucket if getattr(atom, "summary", "") or getattr(atom, "core_action", "")]

    fused_world.micro_interaction_candidates = _mine_micro_interaction_candidates(None, atoms)
    fused_world.macro_trope_candidates = _mine_macro_trope_candidates(None, atoms)
    fused_world.plot_thread_candidates = _mine_plot_thread_candidates(None, atoms)

    fused_world.micro_interaction_clusters = _cluster_micro_interaction_candidates(fused_world.micro_interaction_candidates)
    fused_world.macro_trope_clusters = _cluster_macro_trope_candidates(fused_world.macro_trope_candidates)
    fused_world.plot_thread_clusters = _cluster_plot_thread_candidates(fused_world.plot_thread_candidates)

    fused_world.micro_interactions = _consolidate_micro_clusters(None, fused_world.micro_interaction_clusters)
    fused_world.macro_tropes = _consolidate_macro_clusters(None, fused_world.macro_trope_clusters)
    fused_world.plot_threads = _consolidate_thread_clusters(None, fused_world.plot_thread_clusters)
    fused_world.interaction_pattern_coverage_report = _build_step6_coverage_report(
        atoms=atoms,
        micro_candidates=fused_world.micro_interaction_candidates,
        macro_candidates=fused_world.macro_trope_candidates,
        thread_candidates=fused_world.plot_thread_candidates,
        micro_clusters=fused_world.micro_interaction_clusters,
        macro_clusters=fused_world.macro_trope_clusters,
        thread_clusters=fused_world.plot_thread_clusters,
        final_micro=fused_world.micro_interactions,
        final_macro=fused_world.macro_tropes,
        final_threads=fused_world.plot_threads,
    )

    print(f"[Step 6] Atoms: {len(atoms)}")
    print(f"[Step 6] Micro candidates: {len(fused_world.micro_interaction_candidates)}")
    print(f"[Step 6] Macro candidates: {len(fused_world.macro_trope_candidates)}")
    print(f"[Step 6] Thread candidates: {len(fused_world.plot_thread_candidates)}")
    print(f"[Step 6] Micro clusters: {len(fused_world.micro_interaction_clusters)}")
    print(f"[Step 6] Macro clusters: {len(fused_world.macro_trope_clusters)}")
    print(f"[Step 6] Thread clusters: {len(fused_world.plot_thread_clusters)}")
    print(f"[Step 6] Final micro interactions: {len(fused_world.micro_interactions)}")
    print(f"[Step 6] Final macro tropes: {len(fused_world.macro_tropes)}")
    print(f"[Step 6] Final plot threads: {len(fused_world.plot_threads)}")

    kb.add_micro_interactions(fused_world.micro_interactions)
    return fused_world


def save_step6_output(fused_world: FusedWorld):
    path = save_world_snapshot(_STEP6_FILENAME, fused_world)
    artifact_path = save_json_artifact(
        Path(config.INTERMEDIATE_DIR) / _STEP6_PATTERNS_FILENAME,
        interaction_patterns_from_fused_world(fused_world),
    )
    print(f"[Step 6] Intermediate output saved -> {path.name}; interaction patterns -> {artifact_path.name}")
    return path


def load_step6_output() -> FusedWorld:
    return load_world_snapshot(_STEP6_FILENAME)


def _mine_micro_interaction_candidates(client, atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    del client
    candidates: List[Dict[str, Any]] = []
    for atom in atoms:
        mentions = _character_mentions(atom)
        if len(mentions) < 2:
            continue
        interaction_type = _match_rule(_atom_text(atom), _MICRO_TYPE_RULES, default="")
        if not interaction_type:
            continue
        lead_pair = mentions[:2]
        phase_sequence = _infer_phase_sequence(atom)
        conflict_engine = _infer_conflict_engine(atom)
        relationship_delta = _infer_relationship_delta(atom)
        candidate = {
            "candidate_id": f"micro_candidate_{len(candidates) + 1:04d}",
            "source_refs": [atom.atom_id],
            "novel_sources": [atom.novel_source],
            "arc_names": [atom.arc_name],
            "interaction_name": f"{interaction_type}互动",
            "interaction_type": interaction_type,
            "role_A": lead_pair[0]["display"],
            "role_B": lead_pair[1]["display"],
            "phase_sequence": phase_sequence,
            "conflict_engine": conflict_engine,
            "relationship_delta": relationship_delta,
            "bystander_effect": "公开场域放大" if _has_keywords(_atom_text(atom), _BYSTANDER_KEYWORDS) else "局部场景反馈",
            "emotional_turn": coerce_text(getattr(atom, "emotion", "")) or "情绪压强提升",
            "summary": _clip_text(atom.summary or atom.core_action or atom.raw_summary, 96),
            "applicable_scene": coerce_text([atom.location, atom.raw_location, atom.conflict_type]) or "关系冲突场景",
            "support_hint": 1,
        }
        candidates.append(candidate)
    return candidates


def _mine_macro_trope_candidates(client, atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    del client
    grouped: Dict[Tuple[str, str], List[PlotAtom]] = defaultdict(list)
    for atom in atoms:
        grouped[(atom.novel_source or "", atom.arc_name or "Unnamed Arc")].append(atom)

    candidates: List[Dict[str, Any]] = []
    for (novel_source, arc_name), bucket in grouped.items():
        ordered = sorted(bucket, key=lambda item: (item.chapter_start, item.chapter_end, item.atom_id))
        windows = _sliding_windows(ordered, min_size=5, max_size=8, stride=3)
        if not windows and ordered:
            windows = [ordered]
        for window in windows:
            dominant_conflicts = _top_values([atom.conflict_type for atom in window], limit=3)
            dominant_functions = _top_values([atom.narrative_function for atom in window], limit=3)
            stage_sequence = _dedupe_stage_sequence(_infer_macro_stage(atom) for atom in window)
            candidate = {
                "candidate_id": f"macro_candidate_{len(candidates) + 1:04d}",
                "source_refs": [atom.atom_id for atom in window],
                "novel_sources": [novel_source],
                "arc_name": arc_name,
                "name": _build_macro_name(dominant_conflicts, stage_sequence),
                "abstract_arc": _build_macro_summary(stage_sequence, dominant_conflicts, dominant_functions),
                "stage_sequence": stage_sequence,
                "dominant_conflicts": dominant_conflicts,
                "dominant_functions": dominant_functions,
                "state_progression": _merge_state_progression(window),
                "summary": _build_macro_summary(stage_sequence, dominant_conflicts, dominant_functions),
                "support_hint": max(1, len(window)),
            }
            candidates.append(candidate)
    return candidates


def _mine_plot_thread_candidates(client, atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    del client
    pair_buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for atom in atoms:
        mentions = _character_mentions(atom)
        if len(mentions) < 2:
            continue
        trimmed = mentions[:4]
        for first_index in range(len(trimmed) - 1):
            for second_index in range(first_index + 1, len(trimmed)):
                first = trimmed[first_index]
                second = trimmed[second_index]
                pair_key = tuple(sorted((first["key"], second["key"])))
                bucket = pair_buckets.setdefault(
                    pair_key,
                    {
                        "character_keys": list(pair_key),
                        "display_characters": [first["display"], second["display"]],
                        "atoms": [],
                    },
                )
                bucket["atoms"].append(atom)

    ranked_pairs = sorted(
        pair_buckets.values(),
        key=lambda item: (
            len(item["atoms"]),
            _chapter_span(item["atoms"]),
            len(_dedupe_stage_sequence(_infer_relationship_delta(atom) for atom in item["atoms"])),
        ),
        reverse=True,
    )[:_STEP6_MAX_PAIR_CANDIDATES]

    candidates: List[Dict[str, Any]] = []
    for pair in ranked_pairs:
        atoms_for_pair: List[PlotAtom] = sorted(
            pair["atoms"],
            key=lambda atom: (atom.chapter_start, atom.chapter_end, atom.atom_id),
        )
        if len(atoms_for_pair) < 2:
            continue
        relationship_delta_sequence = [_infer_relationship_delta(atom) for atom in atoms_for_pair]
        stage_sequence = _dedupe_stage_sequence(_infer_thread_stage(atom) for atom in atoms_for_pair)
        thread_type = _match_rule(
            " ".join(
                flatten_text_values(
                    [stage_sequence, relationship_delta_sequence, [atom.summary for atom in atoms_for_pair]]
                )
            ),
            _THREAD_TYPE_RULES,
            default="关系推进线",
        )
        candidates.append(
            {
                "candidate_id": f"thread_candidate_{len(candidates) + 1:04d}",
                "source_refs": [atom.atom_id for atom in atoms_for_pair],
                "novel_sources": _dedupe_text_values([atom.novel_source for atom in atoms_for_pair]),
                "character_keys": pair["character_keys"],
                "display_characters": pair["display_characters"],
                "thread_type": thread_type,
                "name": f"{thread_type}:{'/'.join(pair['display_characters'])}",
                "stage_sequence": stage_sequence,
                "relationship_delta_sequence": relationship_delta_sequence,
                "support_count": len(atoms_for_pair),
                "summary": _clip_text(" -> ".join(atom.summary or atom.core_action for atom in atoms_for_pair[:4]), 180),
            }
        )
    return candidates


def _cluster_micro_interaction_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _cluster_candidates(
        candidates=candidates,
        prefix="micro_cluster",
        similarity_fn=_micro_similarity,
        key_builder=lambda candidate: "|".join(
            [
                normalize_text(candidate.get("interaction_type", "")),
                normalize_text(candidate.get("conflict_engine", "")),
                "-".join(_normalize_sequence(candidate.get("phase_sequence", []))),
                normalize_text(candidate.get("relationship_delta", "")),
            ]
        ),
    )


def _cluster_macro_trope_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _cluster_candidates(
        candidates=candidates,
        prefix="macro_cluster",
        similarity_fn=_macro_similarity,
        key_builder=lambda candidate: "|".join(
            [
                "-".join(_normalize_sequence(candidate.get("dominant_conflicts", []))),
                "-".join(_normalize_sequence(candidate.get("dominant_functions", []))),
                "-".join(_normalize_sequence(candidate.get("stage_sequence", []))),
            ]
        ),
    )


def _cluster_plot_thread_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _cluster_candidates(
        candidates=candidates,
        prefix="thread_cluster",
        similarity_fn=_thread_similarity,
        key_builder=lambda candidate: "|".join(
            [
                normalize_text(candidate.get("thread_type", "")),
                "-".join(_normalize_sequence(candidate.get("stage_sequence", []))),
                "-".join(_normalize_sequence(candidate.get("relationship_delta_sequence", []))),
            ]
        ),
    )


def _consolidate_micro_clusters(client, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    del client
    selected = _select_clusters(
        clusters,
        max_items=_STEP6_MAX_FINAL_MICRO,
        rare_predicate=lambda cluster: cluster.get("support_count", 0) == 1
        and cluster.get("representative_candidate", {}).get("conflict_engine") in _RARE_CONFLICT_ENGINES,
    )
    finals: List[Dict[str, Any]] = []
    for cluster in selected:
        rep = cluster.get("representative_candidate", {})
        phases = rep.get("phase_sequence", [])
        dance = {
            f"phase_{index + 1}": phase
            for index, phase in enumerate(phases[:5])
        }
        finals.append(
            {
                "interaction_name": rep.get("interaction_name", rep.get("interaction_type", "互动模式")),
                "interaction_type": rep.get("interaction_type", ""),
                "applicable_scene": rep.get("applicable_scene", ""),
                "role_A": rep.get("role_A", ""),
                "role_B": rep.get("role_B", ""),
                "the_dance_of_interaction": dance,
                "phase_sequence": phases,
                "conflict_engine": rep.get("conflict_engine", ""),
                "relationship_delta": rep.get("relationship_delta", ""),
                "bystander_effect": rep.get("bystander_effect", ""),
                "emotional_turn": rep.get("emotional_turn", ""),
                "summary": rep.get("summary", ""),
                "support_count": cluster.get("support_count", 0),
                "source_refs": cluster.get("source_refs", []),
                "candidate_ids": cluster.get("candidate_ids", []),
                "novel_sources": cluster.get("novel_sources", []),
                "rare_pattern": bool(cluster.get("is_rare_pattern", False)),
            }
        )
    return finals


def _consolidate_macro_clusters(client, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    del client
    selected = _select_clusters(
        clusters,
        max_items=_STEP6_MAX_FINAL_MACRO,
        rare_predicate=lambda cluster: cluster.get("support_count", 0) == 1
        and len(cluster.get("representative_candidate", {}).get("stage_sequence", [])) >= 4,
    )
    finals: List[Dict[str, Any]] = []
    for cluster in selected:
        rep = cluster.get("representative_candidate", {})
        finals.append(
            {
                "name": rep.get("name", "宏观套路"),
                "description": rep.get("abstract_arc", rep.get("summary", "")),
                "stages": rep.get("stage_sequence", []),
                "stage_sequence": rep.get("stage_sequence", []),
                "dominant_conflicts": rep.get("dominant_conflicts", []),
                "dominant_functions": rep.get("dominant_functions", []),
                "state_progression": rep.get("state_progression", {}),
                "summary": rep.get("summary", ""),
                "support_count": cluster.get("support_count", 0),
                "source_refs": cluster.get("source_refs", []),
                "candidate_ids": cluster.get("candidate_ids", []),
                "novel_sources": cluster.get("novel_sources", []),
                "rare_pattern": bool(cluster.get("is_rare_pattern", False)),
            }
        )
    return finals


def _consolidate_thread_clusters(client, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    del client
    selected = _select_clusters(
        clusters,
        max_items=_STEP6_MAX_FINAL_THREADS,
        rare_predicate=lambda cluster: cluster.get("support_count", 0) == 1
        and len(cluster.get("representative_candidate", {}).get("relationship_delta_sequence", [])) >= 3,
    )
    finals: List[Dict[str, Any]] = []
    for cluster in selected:
        rep = cluster.get("representative_candidate", {})
        finals.append(
            {
                "name": rep.get("name", "关系长线"),
                "thread_type": rep.get("thread_type", ""),
                "stages": rep.get("stage_sequence", []),
                "stage_sequence": rep.get("stage_sequence", []),
                "relationship_delta_sequence": rep.get("relationship_delta_sequence", []),
                "display_characters": rep.get("display_characters", []),
                "character_keys": rep.get("character_keys", []),
                "support_count": cluster.get("support_count", 0),
                "summary": rep.get("summary", ""),
                "source_refs": cluster.get("source_refs", []),
                "candidate_ids": cluster.get("candidate_ids", []),
                "novel_sources": cluster.get("novel_sources", []),
                "rare_pattern": bool(cluster.get("is_rare_pattern", False)),
            }
        )
    return finals


def _build_step6_coverage_report(
    atoms: List[PlotAtom],
    micro_candidates: List[Dict[str, Any]],
    macro_candidates: List[Dict[str, Any]],
    thread_candidates: List[Dict[str, Any]],
    micro_clusters: List[Dict[str, Any]],
    macro_clusters: List[Dict[str, Any]],
    thread_clusters: List[Dict[str, Any]],
    final_micro: List[Dict[str, Any]],
    final_macro: List[Dict[str, Any]],
    final_threads: List[Dict[str, Any]],
) -> Dict[str, Any]:
    atom_ids = {atom.atom_id for atom in atoms}
    covered_ids = set()
    for collection in (micro_candidates, macro_candidates, thread_candidates):
        for item in collection:
            covered_ids.update(str(ref) for ref in item.get("source_refs", []))
    notes: List[str] = []
    if len(final_micro) < len(micro_clusters):
        notes.append("Micro interactions were trimmed by dynamic ranking and rare-pattern retention rules.")
    if len(final_macro) < len(macro_clusters):
        notes.append("Macro tropes were consolidated from overlapping sliding windows.")
    if len(final_threads) < len(thread_clusters):
        notes.append("Plot threads were merged across similar relationship progressions.")
    return {
        "atom_count": len(atoms),
        "micro_candidate_count": len(micro_candidates),
        "macro_candidate_count": len(macro_candidates),
        "thread_candidate_count": len(thread_candidates),
        "micro_cluster_count": len(micro_clusters),
        "macro_cluster_count": len(macro_clusters),
        "thread_cluster_count": len(thread_clusters),
        "final_micro_count": len(final_micro),
        "final_macro_count": len(final_macro),
        "final_thread_count": len(final_threads),
        "covered_atom_ratio": round((len(covered_ids & atom_ids) / len(atom_ids)) if atom_ids else 0.0, 4),
        "notes": notes,
    }


def _cluster_candidates(
    candidates: List[Dict[str, Any]],
    prefix: str,
    similarity_fn,
    key_builder,
) -> List[Dict[str, Any]]:
    clusters: List[Dict[str, Any]] = []
    for candidate in candidates:
        best_index = -1
        best_score = 0.0
        for index, cluster in enumerate(clusters):
            score = similarity_fn(candidate, cluster["representative_candidate"])
            if score > best_score:
                best_score = score
                best_index = index
        if best_index >= 0 and best_score >= 0.65:
            clusters[best_index]["members"].append(candidate)
        else:
            clusters.append(
                {
                    "cluster_id": f"{prefix}_{len(clusters) + 1:04d}",
                    "cluster_key": key_builder(candidate),
                    "members": [candidate],
                }
            )
    normalized: List[Dict[str, Any]] = []
    for cluster in clusters:
        members = cluster.pop("members")
        representative = _pick_representative_candidate(members)
        normalized.append(
            {
                "cluster_id": cluster["cluster_id"],
                "cluster_key": cluster["cluster_key"],
                "candidate_ids": [item.get("candidate_id", "") for item in members],
                "source_refs": _dedupe_text_values(
                    [ref for item in members for ref in flatten_text_values([item.get("source_refs", [])])]
                ),
                "support_count": len(members),
                "novel_sources": _dedupe_text_values(
                    [source for item in members for source in flatten_text_values([item.get("novel_sources", [])])]
                ),
                "representative_candidate": representative,
                "summary": representative.get("summary", ""),
            }
        )
    return normalized


def _select_clusters(
    clusters: List[Dict[str, Any]],
    max_items: int,
    rare_predicate,
) -> List[Dict[str, Any]]:
    if not clusters:
        return []
    scored = sorted(clusters, key=_cluster_rank_key, reverse=True)
    mainline = [cluster for cluster in scored if cluster.get("support_count", 0) >= 2]
    rare = []
    for cluster in scored:
        if cluster in mainline:
            cluster["is_rare_pattern"] = False
            continue
        keep_rare = bool(rare_predicate(cluster))
        cluster["is_rare_pattern"] = keep_rare
        if keep_rare:
            rare.append(cluster)
    selected = mainline[:max_items]
    if len(selected) < max_items:
        rare_budget = max(3, max_items // 6)
        selected.extend(rare[: max(0, min(rare_budget, max_items - len(selected)))])
    return selected[:max_items]


def _cluster_rank_key(cluster: Dict[str, Any]) -> Tuple[int, int, int, int]:
    representative = cluster.get("representative_candidate", {})
    return (
        int(cluster.get("support_count", 0)),
        len(cluster.get("novel_sources", [])),
        len(flatten_text_values([representative.get("stage_sequence", representative.get("phase_sequence", []))])),
        len(cluster.get("source_refs", [])),
    )


def _micro_similarity(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    type_score = 1.0 if normalize_text(left.get("interaction_type")) == normalize_text(right.get("interaction_type")) else 0.0
    engine_score = _token_overlap(left.get("conflict_engine", ""), right.get("conflict_engine", ""))
    phase_score = _sequence_overlap(left.get("phase_sequence", []), right.get("phase_sequence", []))
    relation_score = 1.0 if normalize_text(left.get("relationship_delta")) == normalize_text(right.get("relationship_delta")) else 0.0
    return round(type_score * 0.35 + engine_score * 0.2 + phase_score * 0.3 + relation_score * 0.15, 4)


def _macro_similarity(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    conflict_score = _sequence_overlap(left.get("dominant_conflicts", []), right.get("dominant_conflicts", []))
    function_score = _sequence_overlap(left.get("dominant_functions", []), right.get("dominant_functions", []))
    stage_score = _sequence_overlap(left.get("stage_sequence", []), right.get("stage_sequence", []))
    return round(conflict_score * 0.3 + function_score * 0.25 + stage_score * 0.45, 4)


def _thread_similarity(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    type_score = 1.0 if normalize_text(left.get("thread_type")) == normalize_text(right.get("thread_type")) else 0.0
    stage_score = _sequence_overlap(left.get("stage_sequence", []), right.get("stage_sequence", []))
    relation_score = _sequence_overlap(
        left.get("relationship_delta_sequence", []),
        right.get("relationship_delta_sequence", []),
    )
    return round(type_score * 0.35 + stage_score * 0.35 + relation_score * 0.3, 4)


def _pick_representative_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    return max(
        candidates,
        key=lambda item: (
            int(item.get("support_count", item.get("support_hint", 1)) or 1),
            len(item.get("source_refs", [])),
            len(flatten_text_values([item.get("stage_sequence", item.get("phase_sequence", []))])),
            len(str(item.get("summary", ""))),
        ),
    )


def _sliding_windows(items: Sequence[PlotAtom], min_size: int, max_size: int, stride: int) -> List[List[PlotAtom]]:
    if len(items) < min_size:
        return []
    windows: List[List[PlotAtom]] = []
    size = max(min_size, min(max_size, max(min_size, len(items) // 2 if len(items) <= 10 else 6)))
    for start in range(0, len(items), max(1, stride)):
        window = list(items[start : start + size])
        if len(window) < min_size:
            break
        windows.append(window)
    if windows and windows[-1][-1] is not items[-1]:
        tail = list(items[max(0, len(items) - size) :])
        if len(tail) >= min_size:
            windows.append(tail)
    return windows


def _top_values(values: Iterable[Any], limit: int = 3) -> List[str]:
    counter = Counter(value for value in flatten_text_values(list(values)) if value)
    return [item for item, _count in counter.most_common(limit)]


def _atom_text(atom: PlotAtom) -> str:
    return " ".join(
        flatten_text_values(
            [
                atom.summary,
                atom.raw_summary,
                atom.core_action,
                atom.raw_core_action,
                atom.conflict_type,
                atom.narrative_function,
                atom.location,
                atom.raw_location,
                atom.cultivation_elements,
                atom.state_delta,
            ]
        )
    )


def _infer_phase_sequence(atom: PlotAtom) -> List[str]:
    beats = []
    text = _atom_text(atom)
    if _has_keywords(text, ("试探", "接触", "引导", "开场")):
        beats.append("接触试探")
    if _has_keywords(text, ("压制", "冲突", "争夺", "追杀", "误会")):
        beats.append("冲突升级")
    if _has_keywords(text, ("布局", "交易", "联手", "应对", "出手")):
        beats.append("策略动作")
    if _has_keywords(text, ("翻盘", "反制", "揭露", "破局", "得手", "受挫")):
        beats.append("结果落地")
    if _has_keywords(text, ("后果", "余波", "关注", "伏笔", "钩子")):
        beats.append("后续扩散")
    if not beats:
        beats = ["接触试探", "冲突升级", "结果落地"]
    return beats


def _infer_conflict_engine(atom: PlotAtom) -> str:
    return _match_rule(_atom_text(atom), _CONFLICT_ENGINE_RULES, default=coerce_text(atom.conflict_type) or "局势施压")


def _infer_relationship_delta(atom: PlotAtom) -> str:
    return _match_rule(_atom_text(atom), _RELATION_DELTA_RULES, default="变化未定")


def _infer_macro_stage(atom: PlotAtom) -> str:
    return _match_rule(_atom_text(atom), _MACRO_STAGE_RULES, default=coerce_text(atom.narrative_function) or "主线推进")


def _infer_thread_stage(atom: PlotAtom) -> str:
    if _has_keywords(_atom_text(atom), ("结盟", "合作", "并肩", "互救")):
        return "协作绑定"
    if _has_keywords(_atom_text(atom), ("误会", "解释", "试探", "对立")):
        return "立场错位"
    if _has_keywords(_atom_text(atom), ("翻盘", "清算", "决裂", "背叛")):
        return "关系转折"
    if _has_keywords(_atom_text(atom), ("追杀", "报复", "压制")):
        return "对抗升级"
    return "关系推进"


def _build_macro_name(dominant_conflicts: List[str], stage_sequence: List[str]) -> str:
    conflict = dominant_conflicts[0] if dominant_conflicts else "主线推进"
    if not stage_sequence:
        return f"{conflict}结构"
    return f"{conflict}:{stage_sequence[0]}至{stage_sequence[-1]}"


def _build_macro_summary(stage_sequence: List[str], dominant_conflicts: List[str], dominant_functions: List[str]) -> str:
    stages = " -> ".join(stage_sequence[:5]) or "主线推进"
    conflicts = "、".join(dominant_conflicts[:3]) or "复合冲突"
    functions = "、".join(dominant_functions[:3]) or "推进"
    return f"围绕{conflicts}展开，通常经历{stages}，叙事功能集中在{functions}。"


def _merge_state_progression(atoms: Sequence[PlotAtom]) -> Dict[str, Any]:
    progression: Dict[str, List[str]] = defaultdict(list)
    for atom in atoms:
        state_delta = getattr(atom, "state_delta", {}) or {}
        if not isinstance(state_delta, dict):
            continue
        for key, value in state_delta.items():
            clean_key = str(key or "").strip()
            clean_value = str(value or "").strip()
            if clean_key and clean_value and clean_value not in progression[clean_key]:
                progression[clean_key].append(clean_value)
    return dict(progression)


def _dedupe_stage_sequence(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        clean = str(value or "").strip()
        if clean and clean not in result:
            result.append(clean)
    return result


def _normalize_sequence(values: Sequence[Any]) -> List[str]:
    return [normalize_text(value) for value in flatten_text_values(list(values))]


def _sequence_overlap(left: Sequence[Any], right: Sequence[Any]) -> float:
    left_set = set(_normalize_sequence(left))
    right_set = set(_normalize_sequence(right))
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _token_overlap(left: Any, right: Any) -> float:
    left_tokens = set(_normalize_sequence([left]))
    right_tokens = set(_normalize_sequence([right]))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _match_rule(text: str, rules: List[Tuple[str, Tuple[str, ...]]], default: str) -> str:
    clean_text = str(text or "")
    for label, keywords in rules:
        if any(keyword in clean_text for keyword in keywords):
            return label
    return default


def _has_keywords(text: str, keywords: Iterable[str]) -> bool:
    haystack = str(text or "")
    return any(keyword in haystack for keyword in keywords)


def _chapter_span(atoms: Sequence[PlotAtom]) -> int:
    if not atoms:
        return 0
    start = min(int(getattr(atom, "chapter_start", 0) or 0) for atom in atoms)
    end = max(int(getattr(atom, "chapter_end", 0) or 0) for atom in atoms)
    return max(1, end - start + 1)


def _clip_text(text: Any, limit: int) -> str:
    clean = str(text or "").strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."
