import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any  # 添加类型导入

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 现在可以正确导入src模块
from src.core.cgan_fusion import CGANFusionEngine
from src.core.logic_checker import LogicChecker
from src.utils.preprocessor import OutlinePreprocessor
from src.utils.evaluator import OutlineEvaluator
from src.utils.file_utils import FileManager
from src.config.setting import Config


class NovelOutlineFusionApp:
    """小说大纲融合应用"""

    def __init__(self, model_type: str = None):
        if model_type is None:
            model_type = Config.CURRENT_MODEL_TYPE

        self.fusion_engine = CGANFusionEngine(model_type=model_type)
        self.logic_checker = LogicChecker(model_type=model_type)
        self.preprocessor = OutlinePreprocessor()
        self.evaluator = OutlineEvaluator(model_type=model_type)
        self.file_manager = FileManager()
        self.config = Config()
        self.model_type = model_type

    def run_fusion_pipeline(self, outline_a: str = None, outline_b: str = None,
                            input_file_a: Path = None, input_file_b: Path = None,
                            enable_logic_check: bool = True,
                            enable_evaluation: bool = True) -> Dict[str, Any]:  # 修复类型注解
        """运行完整的大纲融合管道"""

        print("开始处理大纲融合...")

        # 获取输入数据
        if outline_a is None or outline_b is None:
            input_data = self.file_manager.read_input_files(input_file_a, input_file_b)
            outline_a = input_data["outline_a"]
            outline_b = input_data["outline_b"]

        # 预处理
        outline_a_clean = self.preprocessor.clean_outline(outline_a)
        outline_b_clean = self.preprocessor.clean_outline(outline_b)

        print("步骤1: 使用CoT技术进行大纲融合...")
        fusion_result = self.fusion_engine.generate_fused_outline(
            outline_a_clean, outline_b_clean, use_cot=True
        )

        result = {
            "timestamp": datetime.now().isoformat(),
            "original_outlines": {
                "outline_a": outline_a_clean,
                "outline_b": outline_b_clean
            },
            "fusion_result": fusion_result
        }

        if enable_logic_check and Config.LOGIC_CHECK_ENABLED:
            print("步骤2: 进行逻辑一致性检查...")
            logic_result = self.logic_checker.full_logic_check_and_revise(
                fusion_result["fused_outline"]
            )
            result["logic_check_result"] = logic_result
            final_outline = logic_result["final_outline"]
        else:
            result["logic_check_result"] = {
                "final_outline": fusion_result["fused_outline"],
                "checked_and_revised": False
            }
            final_outline = fusion_result["fused_outline"]

        if enable_evaluation:
            print("步骤3: 进行融合质量评估...")
            evaluation_report = self.evaluator.generate_comprehensive_report(
                outline_a_clean, outline_b_clean, final_outline,
                revision_history=result.get("logic_check_result", {}).get("revisions", [])
            )
            result["evaluation_report"] = evaluation_report

        print("处理完成!")
        return result

    def display_results(self, result: Dict[str, Any]):  # 修复类型注解
        """显示处理结果"""
        print("\n" + "=" * 60)
        print("大纲融合结果")
        print("=" * 60)

        final_outline = result["logic_check_result"]["final_outline"]

        print("\n融合后的大纲:")
        print("-" * 40)
        print(final_outline)

        if "revisions" in result["logic_check_result"]:
            revisions = result["logic_check_result"]["revisions"]
            print(f"\n修订次数: {len(revisions)}")

            for i, revision in enumerate(revisions, 1):
                print(f"\n修订 {i}:")
                print(f"问题: {revision['issues'][0] if revision['issues'] else '无'}")

        if "reasoning_steps" in result["fusion_result"]:
            print("\nCoT推理摘要:")
            print("分析完成 → 策略制定 → 融合执行 → 逻辑检查")

        if "evaluation_report" in result:
            self._display_evaluation_results(result["evaluation_report"])

    def _display_evaluation_results(self, evaluation_report: Dict[str, Any]):  # 修复类型注解
        """显示评估结果"""
        print("\n" + "=" * 60)
        print("融合质量评估报告")
        print("=" * 60)

        metrics = evaluation_report["evaluation_metrics"]
        qualitative = evaluation_report["qualitative_analysis"]

        print(f"\n综合评分: {metrics['overall']:.1f}/10.0 ({evaluation_report['quality_level']})")
        print("\n详细评分:")
        print(f"  连贯性: {metrics['coherence']:.1f}/10.0")
        print(f"  逻辑性: {metrics['logic']:.1f}/10.0")
        print(f"  创意性: {metrics['creativity']:.1f}/10.0")
        print(f"  融合质量: {metrics['fusion_quality']:.1f}/10.0")
        print(f"  可读性: {metrics['readability']:.1f}/10.0")

        print("\n主要优点:")
        for strength in qualitative['strengths'][:3]:
            print(f"  ✓ {strength}")

        print("\n待改进问题:")
        for issue in qualitative['issues'][:3]:
            print(f"  ✗ {issue}")

        print("\n改进建议:")
        for suggestion in qualitative['suggestions'][:3]:
            print(f"  💡 {suggestion}")

    def save_all_results(self, result: Dict[str, Any], base_filename: str = None) -> Dict[str, Path]:  # 修复类型注解
        """保存所有结果"""
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"fusion_result_{timestamp}"

        json_file = self.file_manager.save_result(result, f"{base_filename}.json")
        final_outline = result["logic_check_result"]["final_outline"]
        txt_file = self.file_manager.save_outline_text(final_outline, f"{base_filename}.txt")

        print(f"\n所有结果已保存:")
        print(f"  JSON文件: {json_file}")
        print(f"  文本文件: {txt_file}")

        return {
            "json_file": json_file,
            "txt_file": txt_file
        }


def main():
    """主函数"""
    app = NovelOutlineFusionApp()

    print("小说大纲融合AI系统")
    print("=" * 40)
    print(f"输入目录: {app.config.INPUTS_DIR}")
    print(f"输出目录: {app.config.RESULTS_DIR}")
    print("=" * 40)

    input_files = app.config.DEFAULT_INPUT_FILES
    if input_files["outline_a"].exists() and input_files["outline_b"].exists():
        print("检测到输入文件，使用文件输入模式")
        use_file_input = True
    else:
        print("未找到输入文件，使用控制台输入模式")
        use_file_input = False

    if use_file_input:
        result = app.run_fusion_pipeline()
    else:
        outline_a = input("请输入大纲A（主体）: ") or """
        主角李明是一个普通的程序员，某天发现自己能看见别人的情绪颜色。
        他利用这个能力帮助同事解决心理问题，但逐渐发现这个能力有副作用。
        最终他学会了控制这个能力，并找到了真正的自我。
        """

        outline_b = input("请输入大纲B（插入情节）: ") or """
        一个神秘组织"色彩猎手"在追捕有特殊能力的人。
        他们试图控制李明，利用他的能力进行犯罪。
        李明在对抗中发现组织的惊天秘密。
        """

        result = app.run_fusion_pipeline(outline_a, outline_b)

    app.display_results(result)

    saved_files = app.save_all_results(result)
    print(f"\n您可以在以下位置查看结果:")
    print(f"  完整结果: {saved_files['json_file']}")
    print(f"  纯文本大纲: {saved_files['txt_file']}")


if __name__ == "__main__":
    main()