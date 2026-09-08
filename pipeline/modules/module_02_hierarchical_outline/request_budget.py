"""Bound and report model requests made by the hierarchical outline pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


REQUEST_INPUT_CHAR_LIMIT = 140_000


class OutlineRequestBudgetExceeded(RuntimeError):
    def __init__(self, stage: str, actual: int, limit: int):
        self.code = "OUTLINE_REQUEST_BUDGET_EXCEEDED"
        self.failed_stage = stage
        super().__init__(f"{stage} request size {actual} exceeds limit {limit}")


@dataclass
class OutlineRequestLedger:
    source_chars: int
    logical_request_count: int = 0
    actual_model_call_count: int = 0
    cache_hit_count: int = 0
    logical_input_character_count: int = 0
    actual_input_character_count: int = 0
    max_single_request_chars: int = 0
    stage_logical_calls: dict[str, int] = field(default_factory=dict)
    stage_actual_calls: dict[str, int] = field(default_factory=dict)
    stage_input_chars: dict[str, int] = field(default_factory=dict)
    stage_call_limits: dict[str, int] = field(default_factory=lambda: {
        "boundary-detection": 1,
        "boundary-review": 2,
        "top-down-review": 1,
    })
    max_logical_requests: int | None = None

    def _refresh_total_limit(self) -> None:
        self.max_logical_requests = sum(self.stage_call_limits.values())

    def configure_arc_count(self, arc_count: int) -> None:
        if arc_count < 1:
            raise ValueError("story arc flow requires at least one arc")
        self.stage_call_limits.update({
            "node-synthesis": arc_count,
            "node-fidelity-review": 2 * arc_count,
            "node-coverage-review": 2 * arc_count,
            "node-repair": arc_count,
        })
        self._refresh_total_limit()

    def enable_volume_boundaries(self) -> None:
        self.stage_call_limits.update({
            "volume-boundary-detection": 1,
            "volume-boundary-review": 2,
        })
        self._refresh_total_limit()

    def configure_parent_count(self, level: str, node_count: int) -> None:
        if level not in {"volume", "book"}:
            raise ValueError(f"unsupported parent level: {level}")
        if node_count < 1:
            raise ValueError(f"{level} flow requires at least one node")
        self.stage_call_limits.update({
            f"{level}-node-synthesis": node_count,
            f"{level}-fidelity-review": 2 * node_count,
            f"{level}-coverage-review": 2 * node_count,
            f"{level}-node-repair": node_count,
            f"{level}-hierarchy-review": 1,
        })
        self._refresh_total_limit()

    def reserve(self, stage: str, input_chars: int) -> None:
        if input_chars > REQUEST_INPUT_CHAR_LIMIT:
            raise OutlineRequestBudgetExceeded(
                stage, input_chars, REQUEST_INPUT_CHAR_LIMIT
            )
        next_stage_count = self.stage_logical_calls.get(stage, 0) + 1
        stage_limit = self.stage_call_limits.get(stage)
        if stage_limit is None or next_stage_count > stage_limit:
            raise OutlineRequestBudgetExceeded(stage, next_stage_count, stage_limit or 0)
        next_total = self.logical_request_count + 1
        if self.max_logical_requests is not None and next_total > self.max_logical_requests:
            raise OutlineRequestBudgetExceeded(
                stage, next_total, self.max_logical_requests
            )
        self.logical_request_count = next_total
        self.logical_input_character_count += input_chars
        self.max_single_request_chars = max(self.max_single_request_chars, input_chars)
        self.stage_logical_calls[stage] = next_stage_count
        self.stage_input_chars[stage] = self.stage_input_chars.get(stage, 0) + input_chars

    def mark_cache_hit(self) -> None:
        self.cache_hit_count += 1

    def mark_actual_call(self, stage: str, input_chars: int) -> None:
        self.actual_model_call_count += 1
        self.actual_input_character_count += input_chars
        self.stage_actual_calls[stage] = self.stage_actual_calls.get(stage, 0) + 1

    def to_dict(self) -> dict[str, object]:
        ratio = self.logical_input_character_count / max(1, self.source_chars)
        return {
            "source_synopsis_character_count": self.source_chars,
            "logical_request_count": self.logical_request_count,
            "actual_model_call_count": self.actual_model_call_count,
            "cache_hit_count": self.cache_hit_count,
            "logical_input_character_count": self.logical_input_character_count,
            "actual_input_character_count": self.actual_input_character_count,
            "max_single_request_input_chars": self.max_single_request_chars,
            "input_amplification_ratio": round(ratio, 6),
            "stage_logical_request_count": dict(sorted(self.stage_logical_calls.items())),
            "stage_actual_model_call_count": dict(sorted(self.stage_actual_calls.items())),
            "stage_input_character_count": dict(sorted(self.stage_input_chars.items())),
        }
