from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageContext:
    """传给一个独立流水线模块的最小运行上下文。"""

    author_id: str
    work_id: str
    run_id: str
    profile: str
    workspace_root: Path
    input_paths: dict[str, Path] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageResult:
    """一个模块完成后交还调度器的标准结果。"""

    module: str
    module_version: str
    schema_version: str
    accepted: bool
    output_paths: dict[str, Path] = field(default_factory=dict)
    quality_report: Path | None = None
    diagnostics: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_paths"] = {key: str(value) for key, value in self.output_paths.items()}
        payload["quality_report"] = str(self.quality_report) if self.quality_report else ""
        payload["diagnostics"] = list(self.diagnostics)
        return payload

