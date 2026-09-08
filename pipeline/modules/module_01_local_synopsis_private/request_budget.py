from __future__ import annotations

from dataclasses import dataclass, field

SEGMENTATION_REQUEST_CHAR_LIMIT = 130_000
# 技术请求容量只保护模型接口，不参与语义分区。完整分区绝不为满足该值而切开。
LOCAL_REQUEST_CHAR_LIMIT = 140_000
CHAPTER_REQUEST_CHAR_LIMIT = 60_000


class RequestBudgetExceeded(RuntimeError):
    def __init__(self, code: str, stage: str, actual: int, limit: int):
        self.code, self.stage, self.actual, self.limit = code, stage, actual, limit
        super().__init__(f"{code}: {stage} 请求量 {actual} 超过限制 {limit}")


@dataclass
class RequestLedger:
    """记录固定流程的逻辑请求与真实模型调用，不按累计字符中止章节。"""

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
    stage_records: list[dict[str, object]] = field(default_factory=list)
    stage_call_limits: dict[str, int] = field(default_factory=lambda: {
        "chapter-segmentation": 1,
        "chapter-segmentation-review": 1,
        "chapter-segmentation-repair": 1,
        "chapter-segmentation-recheck": 1,
    })
    max_logical_requests: int | None = None
    planned_local_batch_count: int = 0
    planned_local_block_count: int = 0

    def configure_fixed_flow(self, batch_count: int, block_count: int | None = None) -> None:
        if batch_count < 1:
            raise ValueError("fixed flow requires at least one local batch")
        self.max_logical_requests = 14 + 5 * batch_count
        self.planned_local_batch_count = batch_count
        self.planned_local_block_count = int(block_count or 0)
        self.stage_call_limits.update({
            "local-summary-draft": batch_count,
            "local-summary-risk-review": batch_count,
            "local-summary-major-adjudication": batch_count,
            "local-summary-repair": batch_count,
            "local-summary-repair-verification": batch_count,
            "chapter-opening-extract": 1, "chapter-opening-review": 1,
            "chapter-ending-extract": 1, "chapter-ending-review": 1,
            "chapter-summary": 1,
            "chapter-summary-fidelity-review": 2,
            "chapter-summary-coverage-review": 2,
            "chapter-summary-repair": 1,
        })

    def reserve_logical(self, *, stage: str, system: str, prompt: str, per_request_limit: int) -> int:
        size = len(system) + len(prompt)
        if size > per_request_limit:
            raise RequestBudgetExceeded("REQUEST_INPUT_LIMIT_EXCEEDED", stage, size, per_request_limit)
        stage_count = self.stage_logical_calls.get(stage, 0) + 1
        limit = self.stage_call_limits.get(stage)
        if limit is None:
            raise RequestBudgetExceeded("UNDECLARED_STAGE", stage, stage_count, 0)
        if stage_count > limit:
            raise RequestBudgetExceeded("STAGE_CALL_LIMIT_EXCEEDED", stage, stage_count, limit)
        next_total = self.logical_request_count + 1
        if self.max_logical_requests is not None and next_total > self.max_logical_requests:
            raise RequestBudgetExceeded("FLOW_CALL_LIMIT_EXCEEDED", stage, next_total, self.max_logical_requests)
        self.logical_request_count = next_total
        self.logical_input_character_count += size
        self.max_single_request_chars = max(self.max_single_request_chars, size)
        self.stage_logical_calls[stage] = stage_count
        self.stage_input_chars[stage] = self.stage_input_chars.get(stage, 0) + size
        return size

    def mark_cache_hit(self) -> None:
        self.cache_hit_count += 1

    def mark_actual_call(self, *, stage: str, input_chars: int) -> None:
        self.actual_model_call_count += 1
        self.actual_input_character_count += input_chars
        self.stage_actual_calls[stage] = self.stage_actual_calls.get(stage, 0) + 1

    def mark_response(self, *, stage: str, identity: str, cached: bool) -> None:
        self.stage_records.append({"stage": stage, "identity": identity,
                                   "response_received": True, "cached": cached,
                                   "parse_succeeded": None})

    def mark_parse(self, *, stage: str, succeeded: bool) -> None:
        for record in reversed(self.stage_records):
            if record.get("stage") == stage and record.get("parse_succeeded") is None:
                record["parse_succeeded"] = succeeded
                return

    def to_dict(self) -> dict[str, object]:
        ratio = self.logical_input_character_count / max(1, self.source_chars)
        warnings = []
        if ratio > 12:
            warnings.append({"code": "HIGH_INPUT_AMPLIFICATION", "ratio": round(ratio, 6)})
        return {
            "source_character_count": self.source_chars,
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
            "warnings": warnings,
            "planned_local_batch_count": self.planned_local_batch_count,
            "planned_local_block_count": self.planned_local_block_count,
        }
