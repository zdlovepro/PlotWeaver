from typing import Dict, List, Tuple
from src.models.base_model import ModelFactory
from src.config.setting import Config


class LogicChecker:
    """逻辑一致性检查器"""

    def __init__(self, model_type: str = "deepseek"):
        self.model = ModelFactory.create_model(model_type)

    def check_logic_consistency(self, outline: str) -> Dict[str, any]:
        """检查大纲的逻辑一致性"""

        check_prompt = Config.PROMPT_TEMPLATES["logic_check"].format(
            fused_outline=outline
        )

        analysis = self.model.generate(check_prompt)

        # 解析分析结果
        issues = self._extract_issues(analysis)
        suggestions = self._extract_suggestions(analysis)

        return {
            "analysis": analysis,
            "issues": issues,
            "suggestions": suggestions,
            "has_critical_issues": len(issues) > 0
        }

    def revise_outline(self, original_outline: str, issues: List[str],
                       suggestions: List[str]) -> str:
        """根据检查结果修订大纲"""

        revision_prompt = Config.PROMPT_TEMPLATES["revision"].format(
            original_outline=original_outline,
            issues="\n".join(issues),
            suggestions="\n".join(suggestions)
        )

        return self.model.generate(revision_prompt)

    def _extract_issues(self, analysis: str) -> List[str]:
        """从分析中提取具体问题"""
        issues = []
        lines = analysis.split('\n')

        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['问题', '不合理', '矛盾', '不一致', '冲突']):
                if line and not line.startswith('#'):
                    issues.append(line)

        return issues[:5]  # 返回前5个问题

    def _extract_suggestions(self, analysis: str) -> List[str]:
        """从分析中提取建议"""
        suggestions = []
        lines = analysis.split('\n')

        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['建议', '应该', '可以', '推荐', '改进']):
                if line and not line.startswith('#'):
                    suggestions.append(line)

        return suggestions[:5]  # 返回前5个建议

    def full_logic_check_and_revise(self, outline: str, max_attempts: int = None) -> Dict[str, any]:
        """完整的逻辑检查和修订流程"""

        if max_attempts is None:
            max_attempts = Config.MAX_REVISION_ATTEMPTS

        revisions = []
        current_outline = outline

        for attempt in range(max_attempts):
            # 检查逻辑
            check_result = self.check_logic_consistency(current_outline)

            # 如果没有关键问题，停止修订
            if not check_result["has_critical_issues"]:
                return {
                    "final_outline": current_outline,
                    "revisions": revisions,
                    "checked_and_revised": True,
                    "revision_attempts": attempt
                }

            # 修订大纲
            revised_outline = self.revise_outline(
                current_outline,
                check_result["issues"],
                check_result["suggestions"]
            )

            revisions.append({
                "attempt": attempt + 1,
                "original": current_outline,
                "revised": revised_outline,
                "issues": check_result["issues"],
                "suggestions": check_result["suggestions"]
            })

            current_outline = revised_outline

        return {
            "final_outline": current_outline,
            "revisions": revisions,
            "checked_and_revised": True,
            "revision_attempts": max_attempts
        }