import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """项目配置类"""

    # 项目根目录
    PROJECT_ROOT = Path(__file__).parent.parent.parent

    # 数据目录配置
    DATA_DIR = PROJECT_ROOT / "data"
    INPUTS_DIR = DATA_DIR / "inputs"
    EXAMPLES_DIR = DATA_DIR / "examples"

    # 输出目录配置
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    LOGS_DIR = OUTPUTS_DIR / "logs"
    RESULTS_DIR = OUTPUTS_DIR / "results"

    # 确保目录存在
    for dir_path in [INPUTS_DIR, EXAMPLES_DIR, LOGS_DIR, RESULTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # API配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-90ac868504e94e5aaec7016bb4a1b4c6")
    DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL = "deepseek-chat"

    # 如果要切换为其他API，可以添加如下配置：
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE = "https://api.openai.com/v1"
    OPENAI_MODEL = "gpt-3.5-turbo"

    # 本地模型配置
    LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "")
    LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "")

    # 当前使用的模型类型 - 添加这个缺失的属性
    CURRENT_MODEL_TYPE = os.getenv("CURRENT_MODEL_TYPE", "deepseek")  # deepseek, openai, local

    # 生成参数
    MAX_TOKENS = 5000
    TEMPERATURE = 0.7

    # CoT配置
    COT_MAX_STEPS = 3
    COT_ENABLED = True

    # 逻辑检查配置
    LOGIC_CHECK_ENABLED = True
    MAX_REVISION_ATTEMPTS = 3

    # 文件配置
    DEFAULT_INPUT_FILES = {
        "outline_a": INPUTS_DIR / "outline_a.txt",
        "outline_b": INPUTS_DIR / "outline_b.txt"
    }

    # 提示词模板
    PROMPT_TEMPLATES = {
        "fusion_generation": """
作为专业的小说创作助手，请将以下两个小说大纲进行融合：

大纲A（主体）：
{outline_a}

大纲B（插入情节）：
{outline_b}

融合要求：
1. 保持大纲A的主要结构和人物设定
2. 将大纲B的情节有机地插入或替换到大纲A中
3. 确保情节过渡自然，逻辑连贯

请逐步思考融合策略：
""",

        "cot_reasoning": """
基于以上分析，请按照以下步骤推理：

步骤1：识别大纲A的关键情节节点和人物关系
步骤2：分析大纲B的核心情节元素和冲突
步骤3：确定最佳的融合位置和方式
步骤4：评估融合后的逻辑一致性

请完成推理并给出最终的大纲：
""",

        "logic_check": """
请检查以下融合后的大纲的逻辑性：

融合后的大纲：
{fused_outline}

请分析：
1. 情节发展是否合理
2. 人物行为是否一致
3. 时间线是否连贯
4. 冲突解决是否自然

发现问题及修改建议：
""",

        "revision": """
根据以下问题和建议，请修改大纲：

原始大纲：
{original_outline}

发现的问题：
{issues}

修改建议：
{suggestions}

请给出修改后的大纲：
"""
    }