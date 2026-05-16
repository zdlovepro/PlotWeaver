from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

from pipeline.core.common_json import read_json_file, write_json_file


@dataclass
class ExecutableTemplate:
    template_id: str = ""
    template_name: str = ""
    level: str = "event"
    abstract_function: str = ""
    conflict_engine: str = ""
    required_preconditions: List[str] = field(default_factory=list)
    role_slots: Dict[str, str] = field(default_factory=dict)
    beat_sequence: List[Dict[str, Any]] = field(default_factory=list)
    state_delta: Dict[str, Any] = field(default_factory=dict)
    variation_axes: Dict[str, List[str]] = field(default_factory=dict)
    forbidden_source_details: List[str] = field(default_factory=list)
    source_refs: List[str] = field(default_factory=list)
    source_pattern_summary: str = ""


@dataclass
class WorldBaseArtifact:
    world_name: str = ""
    global_theme: str = ""
    world_background: str = ""
    power_source: str = ""
    major_factions: List[str] = field(default_factory=list)
    cultivation_realms: List[Dict[str, Any]] = field(default_factory=list)
    raw_system_text: str = ""


@dataclass
class InteractionPatternsArtifact:
    macro_tropes: List[Dict[str, Any]] = field(default_factory=list)
    plot_threads: List[Dict[str, Any]] = field(default_factory=list)
    micro_interactions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TemplateMiningArtifact:
    role_slot_templates: List[Dict[str, Any]] = field(default_factory=list)
    event_templates: List[Dict[str, Any]] = field(default_factory=list)
    volume_templates: List[Dict[str, Any]] = field(default_factory=list)
    event_flow_templates: List[Dict[str, Any]] = field(default_factory=list)
    executable_templates: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PipelineStateSnapshot:
    world_base: WorldBaseArtifact = field(default_factory=WorldBaseArtifact)
    interaction_patterns: InteractionPatternsArtifact = field(default_factory=InteractionPatternsArtifact)
    template_mining: TemplateMiningArtifact = field(default_factory=TemplateMiningArtifact)
    metadata: Dict[str, Any] = field(default_factory=dict)


def save_json_artifact(path: str | Path, payload: Any) -> Path:
    return write_json_file(path, _to_serializable(payload))


def load_json_artifact(path: str | Path) -> Any:
    return read_json_file(path)


def world_base_from_fused_world(fused_world: Any) -> WorldBaseArtifact:
    return WorldBaseArtifact(
        world_name=str(getattr(fused_world, "world_name", "") or ""),
        global_theme=str(getattr(fused_world, "global_theme", "") or ""),
        world_background=str(getattr(fused_world, "world_background", "") or ""),
        power_source=str(getattr(fused_world, "power_source", "") or ""),
        major_factions=_as_list(getattr(fused_world, "major_factions", [])),
        cultivation_realms=_normalize_dict_list(getattr(fused_world, "cultivation_realms", [])),
        raw_system_text=str(getattr(fused_world, "raw_system_text", "") or ""),
    )


def interaction_patterns_from_fused_world(fused_world: Any) -> InteractionPatternsArtifact:
    return InteractionPatternsArtifact(
        macro_tropes=_normalize_dict_list(getattr(fused_world, "macro_tropes", [])),
        plot_threads=_normalize_dict_list(getattr(fused_world, "plot_threads", [])),
        micro_interactions=_normalize_dict_list(getattr(fused_world, "micro_interactions", [])),
    )


def template_mining_from_fused_world(fused_world: Any) -> TemplateMiningArtifact:
    return TemplateMiningArtifact(
        role_slot_templates=_normalize_dict_list(getattr(fused_world, "role_slot_templates", [])),
        event_templates=_normalize_dict_list(getattr(fused_world, "event_templates", [])),
        volume_templates=_normalize_dict_list(getattr(fused_world, "volume_templates", [])),
        event_flow_templates=_normalize_dict_list(getattr(fused_world, "event_flow_templates", [])),
        executable_templates=_normalize_dict_list(getattr(fused_world, "executable_templates", [])),
    )


def pipeline_state_snapshot_from_fused_world(fused_world: Any, metadata: Dict[str, Any] | None = None) -> PipelineStateSnapshot:
    return PipelineStateSnapshot(
        world_base=world_base_from_fused_world(fused_world),
        interaction_patterns=interaction_patterns_from_fused_world(fused_world),
        template_mining=template_mining_from_fused_world(fused_world),
        metadata=dict(metadata or {}),
    )


def _normalize_dict_list(values: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for value in _as_list(values):
        if isinstance(value, dict):
            normalized.append(dict(value))
            continue
        if is_dataclass(value):
            normalized.append(asdict(value))
            continue
        if hasattr(value, "__dict__"):
            normalized.append(dict(vars(value)))
    return normalized


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return list(value)
    if value is None:
        return []
    return [value]


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    return value

