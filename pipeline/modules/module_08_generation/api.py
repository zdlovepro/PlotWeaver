from ...orchestrator import MigrationPendingError, StageContext, StageResult


def run(context: StageContext) -> StageResult:
    raise MigrationPendingError("模块08旧生成实现已归位，尚未接入新版Skill与图谱接口")

