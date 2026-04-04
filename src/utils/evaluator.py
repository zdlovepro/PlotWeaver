import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from src.models.base_model import ModelFactory
from src.config.setting import Config


@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    coherence_score: float  # 连贯性评分 (0-10)
    logic_score: float  # 逻辑性评分 (0-10)
    creativity_score: float  # 创意性评分 (0-10)
    fusion_quality_score: float  # 融合质量评分 (0-10)
    readability_score: float  # 可读性评分 (0-10)
    overall_score: float  # 综合评分 (0-10)

    issues: List[str]  # 发现的问题
    strengths: List[str]  # 优点
    suggestions: List[str]  # 改进建议


class OutlineEvaluator:
    """大纲评估器"""

    def __init__(self, model_type: str = "deepseek"):
        self.model = ModelFactory.create_model(model_type)

    def evaluate_fusion_quality(self, outline_a: str, outline_b: str,
                                fused_outline: str) -> EvaluationMetrics:
        """评估融合质量"""

        evaluation_prompt = f"""
请从专业角度评估以下大纲融合的质量：

原始大纲A：
{outline_a}

原始大纲B：
{outline_b}

融合后的大纲：
{fused_outline}

请从以下维度进行评估（每个维度0-10分）：

1. 连贯性 - 情节发展是否自然流畅，过渡是否合理
2. 逻辑性 - 人物行为、时间线、因果关系是否合理
3. 创意性 - 融合是否具有创新性，是否产生新的戏剧冲突
4. 融合质量 - 两个大纲元素的结合是否有机和谐
5. 可读性 - 大纲结构是否清晰，表达是否准确

请按以下格式回复：
连贯性: [分数] - [评语]
逻辑性: [分数] - [评语]
创意性: [分数] - [评语]
融合质量: [分数] - [评语]
可读性: [分数] - [评语]

主要优点：
- [优点1]
- [优点2]

主要问题：
- [问题1]
- [问题2]

改进建议：
- [建议1]
- [建议2]
"""

        evaluation_text = self.model.generate(evaluation_prompt)
        return self._parse_evaluation_result(evaluation_text)

    def _parse_evaluation_result(self, evaluation_text: str) -> EvaluationMetrics:
        """解析评估结果文本"""

        # 初始化默认值
        scores = {
            "coherence": 5.0,
            "logic": 5.0,
            "creativity": 5.0,
            "fusion_quality": 5.0,
            "readability": 5.0
        }

        issues = []
        strengths = []
        suggestions = []

        lines = evaluation_text.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测分数行
            score_match = re.match(r'^(连贯性|逻辑性|创意性|融合质量|可读性):\s*(\d+(?:\.\d+)?)', line)
            if score_match:
                dimension = score_match.group(1)
                score = float(score_match.group(2))

                dimension_map = {
                    "连贯性": "coherence",
                    "逻辑性": "logic",
                    "创意性": "creativity",
                    "融合质量": "fusion_quality",
                    "可读性": "readability"
                }

                if dimension in dimension_map:
                    scores[dimension_map[dimension]] = min(max(score, 0), 10)
                continue

            # 检测章节标题
            if "主要优点" in line:
                current_section = "strengths"
                continue
            elif "主要问题" in line:
                current_section = "issues"
                continue
            elif "改进建议" in line:
                current_section = "suggestions"
                continue
            elif line.startswith('-') or line.startswith('•'):
                # 提取列表项内容
                content = re.sub(r'^[-\•]\s*', '', line)
                if current_section == "strengths" and content:
                    strengths.append(content)
                elif current_section == "issues" and content:
                    issues.append(content)
                elif current_section == "suggestions" and content:
                    suggestions.append(content)

        # 计算综合评分（加权平均）
        weights = {
            "coherence": 0.25,
            "logic": 0.25,
            "creativity": 0.15,
            "fusion_quality": 0.20,
            "readability": 0.15
        }

        overall_score = sum(scores[dim] * weight for dim, weight in weights.items())

        return EvaluationMetrics(
            coherence_score=scores["coherence"],
            logic_score=scores["logic"],
            creativity_score=scores["creativity"],
            fusion_quality_score=scores["fusion_quality"],
            readability_score=scores["readability"],
            overall_score=overall_score,
            issues=issues[:5],  # 限制数量
            strengths=strengths[:5],
            suggestions=suggestions[:5]
        )

    def compare_with_baselines(self, fused_outline: str,
                               baseline_outlines: List[str]) -> Dict[str, Any]:
        """与基线大纲进行比较评估"""

        baseline_text = "\n\n".join([f"基线{i + 1}:\n{outline}" for i, outline in enumerate(baseline_outlines)])

        comparison_prompt = f"""
请比较以下融合后的大纲与几个基线大纲：

融合后的大纲：
{fused_outline}

{baseline_text}

请分析：
1. 融合大纲相比基线有哪些改进
2. 在哪些方面还有不足
3. 整体质量排名

评估结果：
"""

        comparison_analysis = self.model.generate(comparison_prompt)

        # 简单提取排名信息
        rank_match = re.search(r'排名[：:]\s*(\d+)', comparison_analysis)
        rank = int(rank_match.group(1)) if rank_match else len(baseline_outlines) + 1

        return {
            "analysis": comparison_analysis,
            "estimated_rank": rank,
            "total_baselines": len(baseline_outlines)
        }

    def calculate_improvement_metrics(self, original_outline: str,
                                      revised_outline: str) -> Dict[str, Any]:
        """计算修订前后的改进指标"""

        improvement_prompt = f"""
请分析以下大纲修订前后的改进情况：

修订前：
{original_outline}

修订后：
{revised_outline}

请评估：
1. 逻辑性改进程度（0-10分）
2. 连贯性改进程度（0-10分） 
3. 整体质量提升程度（0-10分）
4. 主要解决了哪些问题
5. 是否引入了新的问题

评估结果：
"""

        improvement_analysis = self.model.generate(improvement_prompt)

        # 提取改进分数
        logic_improvement = self._extract_score(improvement_analysis, "逻辑性改进")
        coherence_improvement = self._extract_score(improvement_analysis, "连贯性改进")
        overall_improvement = self._extract_score(improvement_analysis, "整体质量提升")

        return {
            "analysis": improvement_analysis,
            "logic_improvement": logic_improvement,
            "coherence_improvement": coherence_improvement,
            "overall_improvement": overall_improvement,
            "has_new_issues": "新问题" in improvement_analysis or "引入了问题" in improvement_analysis
        }

    def _extract_score(self, text: str, dimension: str) -> float:
        """从文本中提取分数"""
        pattern = rf"{dimension}.*?(\d+(?:\.\d+)?)"
        match = re.search(pattern, text)
        return float(match.group(1)) if match else 5.0

    def generate_comprehensive_report(self, outline_a: str, outline_b: str,
                                      fused_outline: str, revision_history: List[Dict] = None) -> Dict[str, Any]:
        """生成综合评估报告"""

        # 基础质量评估
        metrics = self.evaluate_fusion_quality(outline_a, outline_b, fused_outline)

        report = {
            "evaluation_metrics": {
                "coherence": metrics.coherence_score,
                "logic": metrics.logic_score,
                "creativity": metrics.creativity_score,
                "fusion_quality": metrics.fusion_quality_score,
                "readability": metrics.readability_score,
                "overall": metrics.overall_score
            },
            "qualitative_analysis": {
                "strengths": metrics.strengths,
                "issues": metrics.issues,
                "suggestions": metrics.suggestions
            },
            "quality_level": self._determine_quality_level(metrics.overall_score),
            "timestamp": self._get_current_timestamp()
        }

        # 如果有修订历史，分析改进轨迹
        if revision_history and len(revision_history) > 1:
            improvement_analysis = self.analyze_revision_progress(revision_history)
            report["revision_progress"] = improvement_analysis

        return report

    def _determine_quality_level(self, score: float) -> str:
        """根据分数确定质量等级"""
        if score >= 9:
            return "优秀"
        elif score >= 7:
            return "良好"
        elif score >= 5:
            return "一般"
        elif score >= 3:
            return "需要改进"
        else:
            return "较差"

    def _get_current_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

    def analyze_revision_progress(self, revision_history: List[Dict]) -> Dict[str, Any]:
        """分析修订过程中的进步情况"""

        if len(revision_history) < 2:
            return {"analysis": "修订历史不足，无法分析进步情况"}

        progress_text = "修订历史分析：\n"
        for i, revision in enumerate(revision_history):
            progress_text += f"修订轮次 {i + 1}:\n"
            progress_text += f"主要问题: {revision.get('issues', ['无'])[0] if revision.get('issues') else '无'}\n\n"

        progress_prompt = f"""
基于以下修订历史，请分析改进的轨迹：

{progress_text}

请评估：
1. 每次修订是否有效解决了前一轮的问题
2. 整体质量是否在逐步提升
3. 修订过程是否高效

分析结果：
"""

        progress_analysis = self.model.generate(progress_prompt)

        return {
            "analysis": progress_analysis,
            "total_revisions": len(revision_history),
            "has_continuous_improvement": "提升" in progress_analysis or "改进" in progress_analysis
        }