import requests
import json
import time
from typing import Dict, Any, List
from .base_model import BaseLanguageModel
from src.config.setting import Config


class DeepSeekModel(BaseLanguageModel):
    """DeepSeek API模型封装"""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or Config.DEEPSEEK_API_KEY
        self.base_url = base_url or Config.DEEPSEEK_API_BASE
        self.model = model or Config.DEEPSEEK_MODEL
        self.max_retries = 3  # 最大重试次数
        self.timeout = 120  # 超时时间（秒）

    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat_completion(messages, **kwargs)

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """对话补全 - 带重试机制"""
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", Config.MAX_TOKENS),
            "temperature": kwargs.get("temperature", Config.TEMPERATURE),
            "stream": False
        }

        # 重试机制
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()

                result = response.json()
                return result["choices"][0]["message"]["content"]

            except requests.exceptions.Timeout:
                print(f"API请求超时，第 {attempt + 1} 次重试...")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    return "API请求超时，请检查网络连接或稍后重试"

            except requests.exceptions.RequestException as e:
                print(f"API请求错误 (尝试 {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return f"API请求失败: {e}"

            except KeyError as e:
                print(f"API响应解析错误: {e}")
                return "API响应解析失败"

        return "API请求失败，已达到最大重试次数"