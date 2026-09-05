from ...orchestrator import MigrationPendingError, StageContext, StageResult


def run(context: StageContext) -> StageResult:
    raise MigrationPendingError("模块06旧实现已归位，尚未接入新版场景化风格契约")

