"""公共流水线调度接口。业务模块不得把调度器当作数据存储。"""

from .context import StageContext, StageResult
from .registry import ModuleLoadError, load_entrypoint
from .errors import MigrationPendingError

__all__ = ["MigrationPendingError", "ModuleLoadError", "StageContext", "StageResult", "load_entrypoint"]
