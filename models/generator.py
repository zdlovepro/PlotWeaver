import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import Config


class OutlineGenerator:
    def __init__(self, model_name=Config.MODEL_NAME):
        self.device = Config.DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt, max_length=Config.MAX_LENGTH, temperature=Config.TEMPERATURE):
        """生成文本"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                top_p=Config.TOP_P,
                repetition_penalty=Config.REPETITION_PENALTY
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def train_mode(self):
        """切换到训练模式"""
        self.model.train()

    def eval_mode(self):
        """切换到评估模式"""
        self.model.eval()