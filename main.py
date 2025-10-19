import json
import torch
from config import Config
from models.generator import OutlineGenerator
from models.discriminator import OutlineDiscriminator
from training.trainer import OutlineGANTrainer
from training.validator import OutlineValidator
from utils.prompt_utils import PromptCreator
from data.data import get_sample_data, get_real_outlines


class OutlineGANController:
    def __init__(self):
        self.device = Config.DEVICE
        print(f"使用设备: {self.device}")

        # 初始化组件
        self.generator = OutlineGenerator()
        self.discriminator = OutlineDiscriminator().to(self.device)
        self.prompt_creator = PromptCreator()
        self.validator = OutlineValidator(self.discriminator, self.generator.tokenizer)
        self.trainer = OutlineGANTrainer(self.generator, self.discriminator, self.generator.tokenizer)

        print("模型初始化完成")

    def generate_outline(self, original_outline, insert_outline, insertion_points):
        """生成融合大纲"""
        prompt = self.prompt_creator.create_prompt(original_outline, insert_outline, insertion_points)
        generated_text = self.generator.generate(prompt)
        generated_outline = self.prompt_creator.extract_generated_outline(generated_text)
        return generated_outline

    def run_pipeline(self, original_outline, insert_outline, insertion_points, real_outlines=None):
        """运行完整流程"""
        print("=== 开始生成融合大纲 ===")

        # 首先生成一个初始版本
        initial_outline = self.generate_outline(original_outline, insert_outline, insertion_points)
        print("初始生成的大纲:")
        print(initial_outline)

        # 验证初始质量
        validation_result = self.validator.validate(initial_outline)
        print(f"\n初始大纲验证结果: {validation_result}")

        # 如果验证分数较低且有真实数据，进行对抗训练
        if (validation_result["validity_score"] < Config.VALIDITY_THRESHOLD and
                real_outlines and len(real_outlines) >= 2):
            print("\n开始对抗训练优化...")
            self.trainer.adversarial_training(
                self.prompt_creator,
                original_outline,
                insert_outline,
                insertion_points,
                real_outlines,
                epochs=Config.ADVERSARIAL_EPOCHS
            )

            # 重新生成优化后的大纲
            print("\n=== 生成优化后的大纲 ===")
            final_outline = self.generate_outline(original_outline, insert_outline, insertion_points)
            print("优化后的大纲:")
            print(final_outline)

            final_validation = self.validator.validate(final_outline)
            print(f"\n最终大纲验证结果: {final_validation}")
            used_outline = final_outline
            used_validation = final_validation
        else:
            print("\n初始大纲质量已达标，无需进一步训练")
            used_outline = initial_outline
            used_validation = validation_result

        # 保存结果
        self.save_results(original_outline, insert_outline, insertion_points,
                          used_outline, used_validation)

        return used_outline, used_validation

    def save_results(self, original_outline, insert_outline, insertion_points,
                     final_outline, validation_score):
        """保存结果"""
        result = {
            "original_outline": original_outline,
            "insert_outline": insert_outline,
            "insertion_points": insertion_points,
            "final_outline": final_outline,
            "validation_score": validation_score
        }

        with open(Config.OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存到 {Config.OUTPUT_FILE}")


def main():
    # 初始化控制器
    controller = OutlineGANController()

    # 获取数据
    original_outline, insert_outline, insertion_points = get_sample_data()
    real_outlines = get_real_outlines()

    # 运行流程
    final_outline, validation_score = controller.run_pipeline(
        original_outline,
        insert_outline,
        insertion_points,
        real_outlines
    )

    print("\n=== 流程完成 ===")
    print(f"最终大纲长度: {len(final_outline)}")
    print(f"最终验证分数: {validation_score['validity_score']:.4f}")


if __name__ == "__main__":
    main()