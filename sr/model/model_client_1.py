import abc
import asyncio
import os
from typing import List, Dict, Optional
import httpx

DEFAULT_TIMEOUT = 600
DEFAULT_MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 0.8
PRINT_MAX_CHARS = 10000

DEEPSEEK_API_KEY = "sk-46c68c49c96243aa84fd8b89f4f24ae9"


class BaseAsyncModelClient(abc.ABC):
    @abc.abstractmethod
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        ...


class DeepSeekAsyncClient(BaseAsyncModelClient):
        ...
    async def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        return_usage: bool = False,
    ) -> str | tuple[str, Optional[Dict[str, int]]]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        def _print_debug():
            print("\n=== DeepSeek Request ===")
            print(f"model: {payload['model']}, temperature: {payload['temperature']}")
            for i, m in enumerate(messages, 1):
                content = m["content"]
                if len(content) > PRINT_MAX_CHARS:
                    content = content[:PRINT_MAX_CHARS] + "... [truncated]"
                print(f"[{i}] {m['role']}:\n{content}\n")

        _print_debug()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = await client.post(self.base_url, headers=headers, json=payload)

                    resp_text = resp.text
                    if len(resp_text) > PRINT_MAX_CHARS:
                        resp_text = resp_text[:PRINT_MAX_CHARS] + "... [truncated]"
                    print("=== DeepSeek Response ===")
                    print(f"status: {resp.status_code}")
                    print(f"body: {resp_text}\n")

                    if resp.status_code == 402:
                        raise RuntimeError(f"402 Payment Required: {resp.text}")
                    resp.raise_for_status()
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage")
                    # 兼容老接口：默认仅返回文本，同时在实例上保留 usage
                    self.last_usage = usage
                    if return_usage:
                        return content, usage
                    return content
                except Exception as e:
                    if attempt == self.max_retries or (
                        isinstance(e, httpx.HTTPStatusError)
                        and 400 <= e.response.status_code < 500
                    ):
                        raise
                    await asyncio.sleep(2 * attempt)


class LocalAsyncClient(BaseAsyncModelClient):
    def __init__(
        self,
        endpoint: str = "http://localhost:8000/generate",
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.endpoint = endpoint
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.last_usage = None  # 本地接口无 usage

    async def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        return_usage: bool = False,
    ) -> str | tuple[str, Optional[Dict[str, int]]]:
        prompt = "\n\n".join([f"[{m['role']}]\n{m['content']}" for m in messages])
        payload = {
            "prompt": prompt,
            "max_tokens": 64000,
            "temperature": self.temperature,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = await client.post(self.endpoint, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    text = data.get("text") or data.get("content") or ""
                    self.last_usage = None
                    if return_usage:
                        return text, None
                    return text
                except Exception as e:
                    if attempt == self.max_retries or (
                        isinstance(e, httpx.HTTPStatusError)
                        and 400 <= e.response.status_code < 500
                    ):
                        raise
                    await asyncio.sleep(2 * attempt)