# -*- coding: utf-8 -*-
"""
逐维度调用 LLM 审阅小说大纲。
运行：
    python llm_aspect_review.py                      # 默认跑所有维度
    python llm_aspect_review.py --aspects time causality  # 仅跑时间线与因果
"""

import argparse
import os
import sys
from typing import List, Dict
# -*- coding: utf-8 -*-
"""
使用 DeepSeek 直接对小说大纲进行连贯性审阅。
"""
import os
import sys
from typing import List, Dict


if __name__ == "__main__" and __package__ is None:
    # 将项目根目录加入 sys.path（当前文件在 src/models/ 下）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    __package__ = "src.models"


from .base_model import ModelFactory
from src.config.setting import Config  # 确保已配置 API Key/Base/Model


DEFAULT_PATH = r"D:\_study\小说大纲\第三轮尝试\洗稿1\washed.txt"

# 定义各维度的提示词
ASPECT_PROMPTS: Dict[str, str] = {
    "time": """你是一名资深编辑。只关注“时间线”连贯性，审阅下列大纲。
请列出所有时间问题：年份/年龄/先后顺序矛盾、回忆未标注导致倒流等。
输出 Markdown 列表，包含：问题类型、位置（章节或原句片段）、描述、修正建议。
大纲：
{outline}""",
    "causality": """你是一名资深编辑。只关注“因果/伏笔”连贯性，审阅下列大纲。
找出：先用后得、无中生有、伏笔未回收、铺垫缺失、动机不足。
输出 Markdown 列表，包含：问题类型、位置、描述、修正建议。
大纲：
{outline}""",
    "character": """只关注“角色一致性”。找出：角色长时间消失又突现、称谓/名字不一致、身份前后矛盾。
输出 Markdown 列表（问题类型、位置、描述、修正建议）。
大纲：
{outline}""",
    "setting": """只关注“设定冲突”。找出同一设定（世界规则、能力等级、地理关系、时间背景等）前后矛盾。
输出 Markdown 列表（问题类型、位置、描述、修正建议）。
大纲：
{outline}""",
    "transition": """只关注“场景与章节衔接”。找出视角/场景突跳、缺少过渡、关键行动缺补充交代。
输出 Markdown 列表（问题类型、位置、描述、修正建议）。
大纲：
{outline}""",
    "suspense": """只关注“悬念回收”。找出提出的问题/谜团未在后文解答或呼应不足。
输出 Markdown 列表（问题类型、位置、描述、修正建议）。
大纲：
{outline}""",
}

# 读取文件（多编码尝试）
def read_file(path: str) -> str:
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'big5', 'latin-1']
    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    raise ValueError(f"无法读取文件: {path}")

def call_llm(aspect: str, outline: str) -> str:
    prompt = ASPECT_PROMPTS[aspect].format(outline=outline[:15000])  # 防止过长
    model = ModelFactory.create_model(
        model_type="deepseek",
        api_key=Config.DEEPSEEK_API_KEY,
        base_url=Config.DEEPSEEK_API_BASE,
        model=Config.DEEPSEEK_MODEL,
    )
    messages = [{"role": "user", "content": prompt}]
    return model.chat_completion(messages, max_tokens=Config.MAX_TOKENS, temperature=Config.TEMPERATURE)

def main():
    parser = argparse.ArgumentParser(description="逐维度 LLM 审阅大纲")
    parser.add_argument("path", nargs="?", default=DEFAULT_PATH, help="大纲文件路径")
    parser.add_argument("--aspects", nargs="+", choices=list(ASPECT_PROMPTS.keys()), default=list(ASPECT_PROMPTS.keys()),
                        help="要检查的维度，默认全部")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"❌ 文件不存在: {args.path}")
        sys.exit(1)

    outline = read_file(args.path)
    if not outline.strip():
        print("❌ 文件为空")
        sys.exit(1)

    print(f"📄 文件: {args.path}")
    print(f"🔎 将按顺序检查维度: {', '.join(args.aspects)}\n")

    out_file = os.path.splitext(args.path)[0] + "_llm审阅.md"

    for aspect in args.aspects:
        print(f"=== [{aspect}] 开始 ===")
        result = call_llm(aspect, outline)
        print(result.strip())
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(f"## {aspect}\n")
        f.write(result.strip() + "\n\n")
        print(f"=== [{aspect}] 结束 ===\n")

if __name__ == "__main__":
    main()