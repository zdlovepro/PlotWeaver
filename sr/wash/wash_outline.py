import argparse
import asyncio
import glob
import os

from sr.model.model_client import DeepSeekAsyncClient, LocalAsyncClient
from sr.prompt.prompt_templates import WASH_PROMPT

# 默认输入目录（四个）
DEFAULT_DIRS = [
    r"D:\_study\小说大纲\第三轮尝试\仙逆_卷",
    r"D:\_study\小说大纲\第三轮尝试\仙逆_章_情节",
    r"D:\_study\小说大纲\第三轮尝试\一念永恒_卷",
    r"D:\_study\小说大纲\第三轮尝试\一念永恒_章_情节",
]
# 默认输出文件
DEFAULT_OUTPUT = r"D:\_study\小说大纲\第三轮尝试\洗稿1\washed.txt"


def load_all_txt(dirs):
    parts = []
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "*.txt")))
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            parts.append(f"## Source: {fp}\n{txt}")
    return "\n\n".join(parts)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS, help="输入目录列表，包含 .txt 文件")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="输出文件路径")
    ap.add_argument("--model", choices=["deepseek", "local"], default="deepseek")
    ap.add_argument("--deepseek_key")
    ap.add_argument("--local_endpoint", default="http://localhost:8000/generate")
    ap.add_argument("--max_tokens", type=int, default=20000, help="生成上限 tokens（受接口上限约束）")
    args = ap.parse_args()

    client = (
        DeepSeekAsyncClient(api_key=args.deepseek_key, max_tokens=args.max_tokens)
        if args.model == "deepseek"
        else LocalAsyncClient(endpoint=args.local_endpoint)
    )

    corpus = load_all_txt(args.dirs)
    prompt = (
        WASH_PROMPT
        + "\n\n### 全部素材\n"
        + corpus
        + "\n\n### 任务\n请基于以上素材生成洗稿大纲，直接输出可读文本。"
    )

    resp = await client.generate(
        [
            {"role": "system", "content": "你是资深剧情策划，负责洗稿。输出可读大纲文本。"},
            {"role": "user", "content": prompt},
        ]
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(resp)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    asyncio.run(main())