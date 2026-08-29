from ...orchestrator import MigrationPendingError, StageContext, StageResult


def run(context: StageContext) -> StageResult:
    raise MigrationPendingError(
        "第二模块等待第一模块的ChapterSynopsisBundle，且不得从叙事图谱反向生成原作大纲"
    )
