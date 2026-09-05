from __future__ import annotations

from importlib import import_module
from typing import Callable

from .context import StageContext, StageResult


class ModuleLoadError(RuntimeError):
    """配置的流水线模块不存在，或者没有公开正确入口。"""


StageEntrypoint = Callable[[StageContext], StageResult]


def load_entrypoint(spec: str) -> StageEntrypoint:
    """加载 ``package.module:function`` 形式的模块入口。"""

    module_name, separator, function_name = str(spec or "").partition(":")
    if not separator or not module_name.strip() or not function_name.strip():
        raise ModuleLoadError("模块入口必须使用 package.module:function 格式")
    try:
        module = import_module(module_name.strip())
    except ModuleNotFoundError as exc:
        if "module_01_local_synopsis_private" in module_name:
            raise ModuleLoadError(
                "未安装私有梗概提取模块：请部署 "
                "pipeline/modules/module_01_local_synopsis_private"
            ) from exc
        raise ModuleLoadError(f"无法导入流水线模块：{module_name}") from exc
    entrypoint = getattr(module, function_name.strip(), None)
    if not callable(entrypoint):
        raise ModuleLoadError(f"模块入口不可调用：{spec}")
    return entrypoint
