import argparse
import asyncio
import glob
import os
from typing import Optional, Dict, Any, List, Tuple

from sr.model.model_client import DeepSeekAsyncClient, LocalAsyncClient

# 大纲目录（按章节命名 001.txt, 002.txt ...）
DEFAULT_OUTLINE_DIR = r"D:\_study\小说大纲\第三轮尝试\洗稿1"
# 输出目录
DEFAULT_OUTPUT_DIR = r"D:\_study\小说大纲\第三轮尝试\小说"

SYSTEM_PROMPT = (
    "你是资深小说作者。给定全书分章大纲，每次只写一章正文，保持文风统一、情节连贯。"
    "每章目标字数约 2000 字，可上下浮动。禁止输出任何附加说明，只输出本章正文。"
    "请严格按时间线，不要提前引入未在当前或以往大纲中出现的设定/物品。"
)

def load_outlines(outline_dir: str) -> List[Tuple[int, str]]:
    files = sorted(glob.glob(os.path.join(outline_dir, "*.txt")))
    outlines: List[Tuple[int, str]] = []
    for fp in files:
        base = os.path.basename(fp)
        try:
            num = int(os.path.splitext(base)[0])
        except ValueError:
            continue
        with open(fp, "r", encoding="utf-8") as f:
            outlines.append((num, f.read().strip()))
    outlines.sort(key=lambda x: x[0])
    return outlines

def get_window(outlines: List[Tuple[int, str]], chapter_num: int, window: int = 3) -> List[Tuple[int, str]]:
    return [(n, txt) for n, txt in outlines if chapter_num - window <= n <= chapter_num + window]

def build_messages(
    window_outlines: List[Tuple[int, str]],
    chapter_num: int,
    prev_chapter: Optional[str],
) -> List[Dict[str, str]]:
    win_text = "\n\n".join([f"【大纲 第{n:03d}章】\n{txt}" for n, txt in window_outlines])
    user_parts = [
        "【分章大纲窗口（含当前章前3/后3章，如果存在）】",
        win_text,
        f"【写作任务】请创作第 {chapter_num:03d} 章正文（仅本章）。",
    ]
    if prev_chapter:
        user_parts.append("【上一章正文，供衔接参考】")
        user_parts.append(prev_chapter)
    user_parts.append("请按照大纲与前文衔接，输出本章完整正文。")
    user_content = "\n\n".join(user_parts)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

async def generate_chapter(
    client,
    window_outlines: List[Tuple[int, str]],
    chapter_num: int,
    prev_chapter: Optional[str],
    max_tokens: Optional[int],
) -> tuple[str, Optional[Dict[str, Any]]]:
    msgs = build_messages(window_outlines, chapter_num, prev_chapter)
    resp = await client.generate(msgs, max_tokens=max_tokens, return_usage=True)  # type: ignore
    if isinstance(resp, tuple):
        content, usage = resp
    else:
        content, usage = resp, None
    return content, usage

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outline_dir", default=DEFAULT_OUTLINE_DIR, help="分章大纲目录（001.txt, 002.txt ...）")
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="小说输出目录")
    ap.add_argument("--model", choices=["deepseek", "local"], default="deepseek")
    ap.add_argument("--deepseek_key")
    ap.add_argument("--local_endpoint", default="http://localhost:8000/generate")
    ap.add_argument("--max_tokens", type=int, default=12000, help="单章生成上限 tokens（受接口限制）")
    ap.add_argument("--start_chapter", type=int, default=1, help="从第几章开始写（1 表示 001）")
    ap.add_argument("--limit", type=int, default=0, help="写多少章，0 表示直到大纲结束")
    ap.add_argument("--window", type=int, default=3, help="滚动窗口半径（前后各 window 章大纲）")
    args = ap.parse_args()

    client = (
        DeepSeekAsyncClient(api_key=args.deepseek_key, max_tokens=args.max_tokens)
        if args.model == "deepseek"
        else LocalAsyncClient(endpoint=args.local_endpoint)
    )

    outlines = load_outlines(args.outline_dir)
    if not outlines:
        raise RuntimeError("未找到任何大纲文件。")
    max_chapter = outlines[-1][0]
    start = args.start_chapter
    end = max_chapter if args.limit == 0 else min(max_chapter, start + args.limit - 1)

    os.makedirs(args.output_dir, exist_ok=True)

    prev_chapter = None
    for chap in range(start, end + 1):
        chap_no = f"{chap:03d}"
        print(f"[info] 生成章节 {chap_no}...")
        window_outlines = get_window(outlines, chap, window=args.window)
        chapter_text, usage = await generate_chapter(
            client, window_outlines, chap, prev_chapter, args.max_tokens
        )
        out_path = os.path.join(args.output_dir, f"{chap_no}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(chapter_text)
        prev_chapter = chapter_text
        if usage:
            print(f"[usage] chapter {chap_no}: {usage}")
        print(f"[saved] {out_path}")

if __name__ == "__main__":
    asyncio.run(main())