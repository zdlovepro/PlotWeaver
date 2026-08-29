from ...orchestrator import MigrationPendingError, StageContext, StageResult


def run(context: StageContext) -> StageResult:
    raise MigrationPendingError(
        "第三模块等待第二模块的HierarchicalOutlineBundle；禁止回退到事实优先提取"
    )

