from ...orchestrator import MigrationPendingError, StageContext, StageResult


def run(context: StageContext) -> StageResult:
    raise MigrationPendingError("模块09旧评测实现已归位，尚未接入新版结构指标")

