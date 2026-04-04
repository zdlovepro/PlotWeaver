from typing import Dict, List, Tuple, Any
from src.models.base_model import ModelFactory
from src.config.setting import Config  # 修复导入路径


class CoTReasoning:
    """Chain of Thought 推理引擎"""

    def __init__(self, model_type: str = "deepseek"):
        self.model = ModelFactory.create_model(model_type)

    def analyze_outlines(self, outline_a: str, outline_b: str) -> Dict[str, Any]:
        """分析两个大纲的结构和内容"""

        analysis_prompt = f"""
请分析以下两个小说大纲：

大纲A：
{outline_a}

大纲B：
{outline_b}

请从以下角度进行分析：
1. 主要人物和关系
2. 核心情节结构
3. 关键冲突点
4. 主题和风格
5. 可能的融合点

分析结果：
"""
        analysis = self.model.generate(analysis_prompt)
        return {"analysis": analysis}

    def generate_fusion_strategy(self, outline_a: str, outline_b: str, analysis: str) -> Dict[str, Any]:
        """生成融合策略"""

        strategy_prompt = f"""
基于以下分析：
{analysis}

请制定具体的融合策略：

大纲A（主体）：
{outline_a}

大纲B（插入情节）：
{outline_b}

融合策略需要考虑：
1. 插入位置选择（开头、中间、结尾）
2. 替换哪些原有情节
3. 人物如何整合
4. 时间线如何调整
5. 冲突如何重新组织

请给出详细的融合计划：
"""
        strategy = self.model.generate(strategy_prompt)
        return {"strategy": strategy}

    # 修改 src/core/cot_reasoning.py 中的 execute_fusion 方法

    def execute_fusion(self, outline_a: str, outline_b: str, strategy: str) -> str:
        """执行融合并生成新大纲"""

        fusion_prompt = f"""
    根据以下融合策略，**直接生成**融合后的小说大纲：

    **融合策略**：
    {strategy}

    **大纲A（主体）**：
    {outline_a}

    **大纲B（插入情节）**：
    {outline_b}

    **重要要求**：
    1. 严格执行上述融合策略，生成完整、连贯、可直接使用的大纲文本
    2. 不得输出任何推理过程、分析说明或修改建议
    3. 不得输出"你应该"、"建议"等指导性语言
    4. 仅输出最终融合完成的完整大纲内容

    融合后的大纲：
    """

        fused_outline = self.model.generate(fusion_prompt)
        return fused_outline

    def full_cot_reasoning(self, outline_a: str, outline_b: str) -> Tuple[str, Dict[str, str]]:
        """完整的CoT推理流程"""

        analysis_result = self.analyze_outlines(outline_a, outline_b)
        strategy_result = self.generate_fusion_strategy(
            outline_a, outline_b, analysis_result["analysis"]
        )
        fused_outline = self.execute_fusion(
            outline_a, outline_b, strategy_result["strategy"]
        )

        reasoning_steps = {
            "analysis": analysis_result["analysis"],
            "strategy": strategy_result["strategy"],
            "final_outline": fused_outline
        }

        return fused_outline, reasoning_steps