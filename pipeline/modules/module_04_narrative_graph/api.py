from ...orchestrator import MigrationPendingError, StageContext, StageResult


def run(context: StageContext) -> StageResult:
    raise MigrationPendingError("模块04旧实现已归位，尚未迁移到新版必要事实契约")

