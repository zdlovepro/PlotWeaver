from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...contracts import ChapterSynopsisBundle


@dataclass(frozen=True)
class ChapterExecutionResult:
    """一次章节执行的显式结果；产物层不再反向扫描检查点猜测状态。"""

    chapter_id: str
    source_hash: str
    bundle: ChapterSynopsisBundle | None = None
    local_candidates: tuple[dict[str, Any], ...] = ()
    chapter_candidate: dict[str, Any] | None = None
    decisions: tuple[dict[str, Any], ...] = ()
    failure: dict[str, Any] | None = None
    request_metrics: dict[str, Any] = field(default_factory=dict)
    stage_records: tuple[dict[str, Any], ...] = ()

    @property
    def accepted(self) -> bool:
        return self.bundle is not None and self.failure is None
