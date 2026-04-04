# -*- coding: utf-8 -*-
"""
逐维度调用 DeepSeek 进行大纲连贯性审阅（异步版，自动保存）
"""
import argparse
import asyncio
import os
import sys
from typing import Dict, List

# 兼容直接运行与包运行
if __name__ == "__main__" and __package__ is None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    __package__ = "src.models"

from sr.model.model_client import DeepSeekAsyncClient  # 你的异步 DeepSeek 客户端

DEFAULT_PATH = r"D:\_study\小说大纲\第三轮尝试\洗稿1\washed.txt"
MAX_CHARS = 15000

ASPECT_PROMPTS: Dict[str, str] = {
    "time": """你是一名资深编辑。只关注“时间线”连贯性，审阅下列大纲。
找出年份/年龄/事件先后矛盾、未标注回忆导致的时间倒流等。
输出 Markdown 列表：问题类型、位置（章节或原句片段）、描述、修正建议。
大纲：
{outline}""",
    "causality": """只关注“因果/伏笔”。找出先用后得、无中生有、伏笔未回收、动机缺失。
输出 Markdown 列表：问题���型、位置、描述、修正建议。
大纲：
{outline}""",
    "character": """只关注“角色一致性”。找出角色长时间消失又突现、称谓/名字不一致、身份前后矛盾。
输出 Markdown 列表：问题类型、位置、描述、修正建议。
大纲：
{outline}""",
    "setting": """只关注“设定冲突”（世界规则/能力等级/地理关系/时间背景等前后矛盾）。
输出 Markdown 列表：问题类型、位置、描述、修正建议。
大纲：
{outline}""",
    "transition": """只关注“场景/章节衔接”。找出视角或场景突跳、缺少过渡、关键行动未交代。
输出 Markdown 列表：问题类型、位置、描述、修正建议。
大纲：
{outline}""",
    "suspense": """只关注“悬念回收”。找出提出的谜团/问题未在后文解答或呼应不足。
输出 Markdown 列表：问题类型、位置、描述、修正建议。
大纲：
{outline}""",
}


def read_file(path: str) -> str:
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "gb18030", "big5", "latin-1"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    raise ValueError(f"无法读取文件: {path}")


async def review_aspect(client: DeepSeekAsyncClient, aspect: str, outline: str) -> str:
    prompt = ASPECT_PROMPTS[aspect].format(outline=outline[:MAX_CHARS])
    messages = [{"role": "user", "content": prompt}]
    return (await client.generate(messages)).strip()


async def main():
    parser = argparse.ArgumentParser(description="逐维度 LLM 审阅大纲（自动保存）")
    parser.add_argument("path", nargs="?", default=DEFAULT_PATH, help="大纲文件路径")
    parser.add_argument(
        "--aspects",
        nargs="+",
        choices=list(ASPECT_PROMPTS.keys()),
        default=list(ASPECT_PROMPTS.keys()),
        help="要检查的维度，默认全部",
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"❌ 文件不存在: {args.path}")
        sys.exit(1)

    outline = read_file(args.path)
    if not outline.strip():
        print("❌ 文件为空")
        sys.exit(1)

    client = DeepSeekAsyncClient()

    print(f"📄 文件: {args.path}")
    print(f"🔎 将按顺序检查维度: {', '.join(args.aspects)}\n")

    base_prefix = os.path.splitext(os.path.abspath(args.path))[0]
    base_dir = os.path.dirname(base_prefix)
    os.makedirs(base_dir, exist_ok=True)  # 确保目录存在

    for aspect in args.aspects:
        print(f"=== [{aspect}] 开始 ===")
        try:
            res = await review_aspect(client, aspect, outline)
            print(res)
            out_path = f"{base_prefix}_llm审阅_{aspect}.md"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"## {aspect}\n{res}\n")
            print(f"💾 已自动保存到: {out_path}", flush=True)
        except Exception as e:
            print(f"[{aspect}] 调用失败: {e}")
        print(f"=== [{aspect}] 结束 ===\n")


if __name__ == "__main__":
    asyncio.run(main())