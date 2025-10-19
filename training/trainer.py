import torch
import torch.optim as optim
from config import Config


class OutlineGANTrainer:
    def __init__(self, generator, discriminator, tokenizer):
        self.generator = generator
        self.discriminator = discriminator
        self.tokenizer = tokenizer
        self.device = Config.DEVICE

        # 优化器
        self.g_optimizer = optim.AdamW(self.generator.model.parameters(), lr=Config.LEARNING_RATE)
        self.d_optimizer = optim.AdamW(self.discriminator.parameters(), lr=Config.LEARNING_RATE)

        # 损失函数
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_discriminator(self, real_outlines, fake_outlines, epochs=Config.D_EPOCHS):
        """训练判别器"""
        self.discriminator.train()

        all_outlines = real_outlines + fake_outlines
        labels = [1] * len(real_outlines) + [0] * len(fake_outlines)

        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(all_outlines), Config.BATCH_SIZE):
                batch_texts = all_outlines[i:i + Config.BATCH_SIZE]
                batch_labels = labels[i:i + Config.BATCH_SIZE]

                if len(batch_texts) == 0:
                    continue

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                labels_tensor = torch.tensor(batch_labels).to(self.device)

                # 前向传播
                self.d_optimizer.zero_grad()
                outputs = self.discriminator(inputs['input_ids'], inputs['attention_mask'])
                loss = self.criterion(outputs, labels_tensor)

                # 反向传播
                loss.backward()
                self.d_optimizer.step()

                total_loss += loss.item()

            print(f"判别器训练 Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    def train_generator(self, prompt_creator, original_outline, insert_outline,
                        insertion_points, num_samples=5):
        """训练生成器（对抗训练）"""
        self.generator.train_mode()

        total_g_loss = 0
        for i in range(num_samples):
            # 生成大纲
            prompt = prompt_creator.create_prompt(original_outline, insert_outline, insertion_points)
            generated_outline = self.generator.generate(prompt)

            # 提取纯大纲内容
            if "生成的新大纲:" in generated_outline:
                generated_outline = generated_outline.split("生成的新大纲:")[-1].strip()

            # 判别器评估
            inputs = self.tokenizer(
                generated_outline,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                d_output = self.discriminator(inputs['input_ids'], inputs['attention_mask'])
                fake_prob = d_output[:, 1]  # 获取"真实"的概率

            # 生成器损失：希望判别器认为生成的是真实的
            g_loss = -torch.log(fake_prob + 1e-8).mean()

            # 反向传播（只更新生成器）
            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()

            total_g_loss += g_loss.item()

            print(f"生成器训练 Sample {i + 1}/{num_samples}, Loss: {g_loss.item():.4f}")

        return total_g_loss / num_samples

    def adversarial_training(self, prompt_creator, original_outline, insert_outline,
                             insertion_points, real_outlines, epochs=Config.ADVERSARIAL_EPOCHS):
        """对抗训练循环"""
        print("开始对抗训练...")

        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch + 1}/{epochs} ===")

            # 生成虚假大纲用于判别器训练
            fake_outlines = []
            for _ in range(len(real_outlines)):
                prompt = prompt_creator.create_prompt(original_outline, insert_outline, insertion_points)
                fake_outline = self.generator.generate(prompt)
                if "生成的新大纲:" in fake_outline:
                    fake_outline = fake_outline.split("生成的新大纲:")[-1].strip()
                fake_outlines.append(fake_outline)

            # 训练判别器
            self.train_discriminator(real_outlines, fake_outlines, epochs=1)

            # 训练生成器
            g_loss = self.train_generator(prompt_creator, original_outline, insert_outline,
                                          insertion_points, num_samples=len(real_outlines))
            print(f"生成器平均损失: {g_loss:.4f}")