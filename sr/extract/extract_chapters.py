import argparse
import asyncio
import json
import os
from typing import List, Dict, Any
from sr.model.model_client import DeepSeekAsyncClient, LocalAsyncClient

# 提示词：按章节提取详细梗概（便于洗稿与伏笔追踪）
CHAPTER_PROMPT = """你是资深小说编辑，负责为单章生成详细梗概，便于后续洗稿和伏笔追踪。
请直接输出详细梗概文本（无需 JSON 结构），包括：
- 起因-冲突-转折-高潮-收束的因果链条
- 主要人物动机与关系演变
- 关键事件节奏点
- 伏笔/悬念及可能的呼应线索
要求内容具体、连贯，便于后续改写保持逻辑。
"""

DEFAULT_PATH = r"D:\_study\小说大纲\第三轮尝试\一念永恒"
DEFAULT_OUTPUT_ROOT = os.path.dirname(DEFAULT_PATH)


def build_prompt(chapter: str, text: str) -> List[Dict[str, str]]:
    user = (
        CHAPTER_PROMPT
        + "\n\n本章编号:\n"
        + chapter
        + "\n\n本章全文:\n"
        + text
    )
    return [
        {"role": "system", "content": "你是资深小说编辑，只需返回梗概正文。"},
        {"role": "user", "content": user},
    ]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_chapter_file(output_root: str, novel_name: str, chapter_no: str, content: str):
    """
    将章节梗概写入 {output_root}/{novel_name}_章/{chapter_no}.txt
    """
    ch_dir = os.path.join(output_root, f"{novel_name}_章")
    ensure_dir(ch_dir)
    fname = os.path.join(ch_dir, f"{chapter_no}.txt")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[saved] {fname}")


async def process_chapter(client, chapter_no: str, text: str, output_root: str, novel_name: str, results: List[Dict[str, Any]]):
    prompt = build_prompt(chapter_no, text)
    resp = await client.generate(prompt)
    # 不做 JSON 解析，直接落盘
    write_chapter_file(output_root, novel_name, chapter_no, resp)
    results.append({"chapter": chapter_no, "summary": resp})


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--chapters_dir",
        required=False,
        default=DEFAULT_PATH,
        help="章节原文目录（默认指向仙逆，可覆盖）",
    )
    ap.add_argument(
        "--model",
        choices=["deepseek", "local"],
        default="deepseek",
        help="默认使用 deepseek，如需本地模型选择 local",
    )
    ap.add_argument("--deepseek_key", help="可选；未提供则从 config_local.py/环境变量/默认常量读取")
    ap.add_argument("--local_endpoint", default="http://localhost:8000/generate")
    ap.add_argument(
        "--output_dir",
        required=False,
        default=DEFAULT_OUTPUT_ROOT,
        help="输出根目录（默认为输入目录的上一级）",
    )
    args = ap.parse_args()

    client = (
        DeepSeekAsyncClient(api_key=args.deepseek_key)
        if args.model == "deepseek"
        else LocalAsyncClient(endpoint=args.local_endpoint)
    )

    novel_name = os.path.basename(os.path.abspath(args.chapters_dir))
    # 收集章节文件
    chapter_files = [fn for fn in os.listdir(args.chapters_dir) if fn.lower().endswith(".txt")]
    chapter_files.sort()

    results: List[Dict[str, Any]] = []
    for fn in chapter_files:
        chapter_no = os.path.splitext(fn)[0]
        fp = os.path.join(args.chapters_dir, fn)
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue
        await process_chapter(client, chapter_no, text, args.output_dir, novel_name, results)

    # 汇总 JSON（便于后续检查），存储章节号与梗概文本
    merged_json = os.path.join(args.output_dir, f"{novel_name}_chapter_summaries.json")
    ensure_dir(os.path.dirname(merged_json))
    with open(merged_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[saved] {merged_json}")


if __name__ == "__main__":
    asyncio.run(main())