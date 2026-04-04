import argparse
import asyncio
import glob
import os
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sr.model.model_client import DeepSeekAsyncClient, LocalAsyncClient
from sr.prompt.prompt_templates import WASH_PROMPT

DEFAULT_OUTPUT = r"D:\_study\小说大纲\第三轮尝试\洗稿\washed.txt"
DEFAULT_DIRS = [
    r"D:\_study\小说大纲\第三轮尝试\仙逆_卷",
    r"D:\_study\小说大纲\第三轮尝试\仙逆_章_情节",
    r"D:\_study\小说大纲\第三轮尝试\一念永恒_卷",
    r"D:\_study\小说大纲\第三轮尝试\一念永恒_章_情节",
]

def chunk_text_gen(text: str, max_chars: int = 1600, overlap: int = 200):
    """生成器方式切块，避免一次性构建大列表。"""
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        yield text[start:end]
        # 防止重叠导致死循环
        start = end - overlap
        if start <= 0:
            start = end
        if start <= 0 or start >= n:
            break

def load_and_chunk(dirs: List[str], max_chars: int, overlap: int, max_file_chars: int, max_chunks: int) -> Tuple[List[str], List[str]]:
    """
    返回 chunks 文本列表和对应的源标签列表（用于提示来源）。
    增加截断与总块数上限，避免 OOM。
    """
    texts = []
    tags = []
    total = 0
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "*.txt")))
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                raw = f.read()
            if max_file_chars and len(raw) > max_file_chars:
                raw = raw[:max_file_chars]
            for idx, ch in enumerate(chunk_text_gen(raw, max_chars=max_chars, overlap=overlap)):
                texts.append(ch)
                tags.append(f"{fp}#chunk{idx+1}")
                total += 1
                if max_chunks and total >= max_chunks:
                    print(f"[warn] 达到 max_chunks={max_chunks}，后续文件被跳过。")
                    return texts, tags
    return texts, tags

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS, help="输入目录列表，包含 .txt 文件")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="输出文件路径")
    ap.add_argument("--model", choices=["deepseek", "local"], default="deepseek")
    ap.add_argument("--deepseek_key")
    ap.add_argument("--local_endpoint", default="http://localhost:8000/generate")
    ap.add_argument("--chunk_chars", type=int, default=1600, help="切块大小（字符）")
    ap.add_argument("--chunk_overlap", type=int, default=200, help="切块重叠（字符）")
    ap.add_argument("--top_k", type=int, default=30, help="选取的上下文块数")
    ap.add_argument("--max_file_chars", type=int, default=200000, help="单文件截断字符数，0 表示不截断")
    ap.add_argument("--max_chunks", type=int, default=5000, help="最大切块总数，0 表示不限制")
    args = ap.parse_args()

    client = (
        DeepSeekAsyncClient(api_key=args.deepseek_key)
        if args.model == "deepseek"
        else LocalAsyncClient(endpoint=args.local_endpoint)
    )

    # 1) 载入并切块（带截断和总数上限）
    chunks, tags = load_and_chunk(
        args.dirs,
        max_chars=args.chunk_chars,
        overlap=args.chunk_overlap,
        max_file_chars=args.max_file_chars,
        max_chunks=args.max_chunks,
    )
    if not chunks:
        raise RuntimeError("未找到任何文本块，请检查目录或文件。")
    print(f"[info] 总块数: {len(chunks)}")

    # 2) 构建检索上下文
    vec = TfidfVectorizer(max_features=50000)
    mat = vec.fit_transform(chunks + [WASH_PROMPT])
    query_vec = mat[-1]
    chunk_mat = mat[:-1]
    scores = cosine_similarity(query_vec, chunk_mat).flatten()
    top_idx = scores.argsort()[::-1][:args.top_k]
    context = "\n\n".join([f"## Source: {tags[i]}\n{chunks[i]}" for i in top_idx])

    prompt = (
        WASH_PROMPT
        + "\n\n### 相关素材（Top-K 检索）\n"
        + context
        + "\n\n### 任务\n请基于以上素材生成洗稿大纲，直接输出可读文本（无需 JSON）。"
    )

    resp = await client.generate([
        {"role": "system", "content": "你是资深剧情策划，负责洗稿。输出可读大纲文本。"},
        {"role": "user", "content": prompt},
    ])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(resp)
    print(f"[saved] {args.output}")

if __name__ == "__main__":
    asyncio.run(main())