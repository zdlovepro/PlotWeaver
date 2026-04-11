import argparse
import asyncio
import glob
import os
from typing import List, Dict
from sr.model.model_client import DeepSeekAsyncClient, LocalAsyncClient

# 提示词：从一批章节中提炼“大事件/卷级梗概”，并在输出中注明章节序号范围
VOLUME_PROMPT = """你是资深小说编辑。给定一批连续章节文本（当前批次 {batch_start}-{batch_end}），请识别其中的大事件/卷级剧情。
要求：
- 每个大事件通常覆盖几章，不要超过20章，务必在输出中写明涉及的章节范围（例如：章节范围：001-050）。
- 说明起因-冲突-转折-高潮-阶段性收束，
- 交代主要人物动机、关系
- 每个大事件的伏笔/悬念。
- 若本批次结尾仍有未收束矛盾，可在末尾说明“可能续接后续章节：...”
- 直接输出梗概正文（非 JSON），可以分点或分段，便于后续洗稿。"""

DEFAULT_INPUT_PATH = r"D:\_study\小说大纲\第三轮尝试\一念永恒_章_情节"
# 固定卷梗概输出目录
DEFAULT_OUTPUT_DIR = r"D:\_study\小说大纲\第三轮尝试\一念永恒_卷"

def load_chapters(batch_files: List[str]) -> List[Dict[str, str]]:
    contents = []
    for fp in batch_files:
        with open(fp, "r", encoding="utf-8") as f:
            contents.append({"chapter": os.path.basename(fp).split(".")[0], "text": f.read()})
    return contents

def build_prompt(batch_start: int, batch_end: int, chapters: List[Dict[str, str]]) -> List[Dict[str, str]]:
    chapter_texts = "\n\n".join([f"## {c['chapter']}\n{c['text']}" for c in chapters])
    user = VOLUME_PROMPT.format(batch_start=batch_start, batch_end=batch_end) + "\n\n" + chapter_texts
    return [
        {"role": "system", "content": "你是资深小说编辑，负责提炼大事件/卷级梗概。直接输出梗概正文。"},
        {"role": "user", "content": user},
    ]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_volume_files(output_root: str, batches: List[Dict[str, str]]):
    """
    将每个批次的大事件梗概写入 {output_root}/001.txt ...
    文件名按批次顺序编号，文件内写明章节范围。
    """
    ensure_dir(output_root)
    for idx, item in enumerate(batches, 1):
        fname = os.path.join(output_root, f"{idx:03d}.txt")
        header = [
            f"章节范围：{item['batch_start']:03d}-{item['batch_end']:03d}",
            "",
        ]
        with open(fname, "w", encoding="utf-8") as f:
            f.write("\n".join(header) + item["summary"])
        print(f"[saved] {fname}")

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir",
        required=False,
        default=DEFAULT_INPUT_PATH,
        help="含 001.txt ... 的目录（默认指向一念永恒_章_情节，可覆盖）",
    )
    ap.add_argument("--batch_size", type=int, default=50, help="批次大小（默认 50 章）")
    ap.add_argument(
        "--model",
        choices=["deepseek", "local"],
        default="deepseek",
    )
    ap.add_argument("--deepseek_key", help="可选；未提供则从 config_local.py 或环境变量读取")
    ap.add_argument("--local_endpoint", default="http://localhost:8000/generate")
    ap.add_argument(
        "--output_dir",
        required=False,
        default=DEFAULT_OUTPUT_DIR,
        help="卷梗概输出目录（默认固定为一念永恒_卷）",
    )
    args = ap.parse_args()

    client = (
        DeepSeekAsyncClient(api_key=args.deepseek_key)
        if args.model == "deepseek"
        else LocalAsyncClient(endpoint=args.local_endpoint)
    )

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.txt")))
    if not files:
        raise RuntimeError("未找到任何章节文件。")

    batches = []
    for i in range(0, len(files), args.batch_size):
        batch_files = files[i: i + args.batch_size]
        if not batch_files:
            continue
        batch_start = int(os.path.basename(batch_files[0]).split(".")[0])
        batch_end = int(os.path.basename(batch_files[-1]).split(".")[0])
        chapters = load_chapters(batch_files)
        prompt = build_prompt(batch_start, batch_end, chapters)
        summary = await client.generate(prompt)
        batches.append({"batch_start": batch_start, "batch_end": batch_end, "summary": summary})

    write_volume_files(args.output_dir, batches)

if __name__ == "__main__":
    asyncio.run(main())