import torch
from config import Config


class OutlineValidator:
    def __init__(self, discriminator, tokenizer):
        self.discriminator = discriminator
        self.tokenizer = tokenizer
        self.device = Config.DEVICE

    def validate(self, outline):
        """验证生成大纲的质量"""
        validation_input = self.tokenizer(
            outline,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            validity_score = self.discriminator(validation_input['input_ids'],
                                                validation_input['attention_mask'])

        return {
            "validity_score": validity_score[0][1].item(),  # 真实概率
            "length": len(outline),
            "coherence_check": self.check_coherence(outline)
        }

    def check_coherence(self, outline):
        """简单的一致性检查"""
        required_elements = ["主线", "情节", "人物"]
        return all(element in outline for element in required_elements)