import torch.nn as nn
from transformers import AutoModel
from config import Config


class OutlineDiscriminator(nn.Module):
    """大纲判别器 - 判断生成的大纲是否合理"""

    def __init__(self, hidden_size=Config.DISCRIMINATOR_HIDDEN_SIZE,
                 num_classes=Config.NUM_CLASSES):
        super(OutlineDiscriminator, self).__init__()
        self.bert = AutoModel.from_pretrained("bert-base-chinese")
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # 取[CLS] token
        return self.classifier(pooled_output)