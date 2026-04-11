from typing import Dict, Tuple, List,Any
from src.models.base_model import ModelFactory
from .cot_reasoning import CoTReasoning
from src.config.setting import Config


class CGANFusionEngine:
    """基于CGAN思想的小说大纲融合引擎"""

    def __init__(self, model_type: str = "deepseek"):
        self.generator = ModelFactory.create_model(model_type)  # 生成器
        self.cot_engine = CoTReasoning(model_type)

    def generate_fused_outline(self, outline_a: str, outline_b: str,
                               use_cot: bool = True) -> Dict[str, any]:
        """生成融合后的大纲"""

        if use_cot and Config.COT_ENABLED:
            # 使用CoT技术进行推理
            fused_outline, reasoning_steps = self.cot_engine.full_cot_reasoning(
                outline_a, outline_b
            )
        else:
            # 直接生成
            prompt = Config.PROMPT_TEMPLATES["fusion_generation"].format(
                outline_a=outline_a,
                outline_b=outline_b
            )
            fused_outline = self.generator.generate(prompt)
            reasoning_steps = {"direct_generation": fused_outline}

        return {
            "fused_outline": fused_outline,
            "reasoning_steps": reasoning_steps,
            "metadata": {
                "outline_a_length": len(outline_a),
                "outline_b_length": len(outline_b),
                "used_cot": use_cot
            }
        }

    def conditional_generation(self, condition: str, outline_a: str,
                               outline_b: str) -> str:
        """条件生成 - 根据特定条件调整生成结果"""

        condition_prompt = f"""
在融合以下两个大纲时，请特别考虑以下条件：
{condition}

大纲A：
{outline_a}

大纲B：
{outline_b}

请根据上述条件进行有针对性的融合：
"""
        return self.generator.generate(condition_prompt)