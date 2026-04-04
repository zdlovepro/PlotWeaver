CHAPTER_SEPARATOR = "\n===CHAPTER_BREAK===\n"

def build_prompt(volume_summary: str, chapter_outline: str) -> List[Dict[str, str]]:
    system = (
        "你是一名小说作家。每章目标约 2000 字。"
        "严格依照大纲展开情节，保持与卷概要的连贯性。"
        "注重场景描写与对话，避免直白叙述。"
        "保持人名、世界规则、语气一致。"
        f"如果一个文件包含多个章节大纲，请逐章输出，并用分隔符 {CHAPTER_SEPARATOR!r} 分隔章节。"
        "不要输出额外说明。"
    )
    user = (
        "卷概要：\n"
        f"{volume_summary}\n\n"
        "本文件包含一个或多个章节大纲：\n"
        f"{chapter_outline}\n\n"
        f"请逐章生成，每章约 2000 字，并用分隔符 {CHAPTER_SEPARATOR!r} 分隔。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def save_chapters_from_response(chapter_text: str, output_dir: str, src_name: str):
    os.makedirs(output_dir, exist_ok=True)
    parts = [p.strip() for p in chapter_text.split(CHAPTER_SEPARATOR) if p.strip()]
    for i, part in enumerate(parts, 1):
        out_name = f"chapter_{src_name}_part{i:02d}.txt"
        with open(os.path.join(output_dir, out_name), "w", encoding="utf-8") as f:
            f.write(part)