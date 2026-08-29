class MigrationPendingError(RuntimeError):
    """目标模块已经归位，但尚未迁移到新版文件契约。"""

