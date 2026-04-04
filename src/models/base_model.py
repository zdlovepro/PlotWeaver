from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseLanguageModel(ABC):
    """基础语言模型接口"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        raise NotImplementedError

    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """对话补全"""
        raise NotImplementedError


class ModelFactory:
    """模型工厂类"""

    @staticmethod
    def create_model(model_type: str = "deepseek", **kwargs) -> BaseLanguageModel:
        """创建模型实例"""
        if model_type == "deepseek":
            from .deepseek_model import DeepSeekModel
            return DeepSeekModel(**kwargs)
        elif model_type == "local":
            # 预留本地模型接口
            from .local_model import LocalModel  # noqa: F401
            return LocalModel(**kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")