from ...orchestrator import MigrationPendingError, StageContext, StageResult


def run(context: StageContext) -> StageResult:
    raise MigrationPendingError("模块07旧实现已归位，等待稳定作者模板和风格画像")

