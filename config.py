# 配置文件
import torch


class Config:
    # 模型配置
    MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"
    DISCRIMINATOR_HIDDEN_SIZE = 768
    NUM_CLASSES = 2

    # 训练配置
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 4
    MAX_LENGTH = 1024
    TEMPERATURE = 0.8
    TOP_P = 0.9
    REPETITION_PENALTY = 1.1

    # 训练轮次
    D_EPOCHS = 3
    G_EPOCHS = 5
    ADVERSARIAL_EPOCHS = 3

    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 文件路径
    OUTPUT_FILE = "outline_fusion_result.json"

    # 验证阈值
    VALIDITY_THRESHOLD = 0.7