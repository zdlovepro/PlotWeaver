from ...orchestrator import MigrationPendingError, StageContext, StageResult


def run(context: StageContext) -> StageResult:
    raise MigrationPendingError("模块05旧的多重模板实现已保留，等待新版大纲和图谱契约")

