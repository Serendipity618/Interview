import torch.nn as nn
from transformers import AutoModel

from transformers import logging

logging.set_verbosity_error()


class SentenceTransformerMTL(nn.Module):
    """
    Task 1: Encodes input sentences using a pre-trained transformer and mean pooling.
    Task 2: Supports multi-task learning with two classifier heads:
      - Task A: Sentence Classification
      - Task B: Sentiment Analysis
    """

    def __init__(self, model_name, num_classes_task_a=3, num_classes_task_b=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, ignore_mismatched_sizes=True)
        self.hidden_size = self.encoder.config.hidden_size

        # Mean pooling strategy to get sentence embeddings
        self.pooling = lambda x, m: (x * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True)

        # Task-specific classifier heads
        self.classifier_a = nn.Linear(self.hidden_size, num_classes_task_a)
        self.classifier_b = nn.Linear(self.hidden_size, num_classes_task_b)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        pooled = self.pooling(last_hidden, attention_mask)

        logits_a = self.classifier_a(pooled)
        logits_b = self.classifier_b(pooled)
        return logits_a, logits_b
