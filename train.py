import torch
from app.model import SentenceTransformerMTL

MODEL_NAME = 'distilbert-base-uncased'
model = SentenceTransformerMTL(MODEL_NAME)

# Freeze transformer encoder for feature-based transfer learning
for param in model.encoder.parameters():
    param.requires_grad = False

# Optimizer only updates the task-specific heads
optimizer = torch.optim.Adam(
    list(model.classifier_a.parameters()) + list(model.classifier_b.parameters()), lr=1e-3
)

# Define task-specific loss functions
criterion_a = torch.nn.CrossEntropyLoss()
criterion_b = torch.nn.CrossEntropyLoss()

# Optionally unfreeze encoder for full fine-tuning:
# for param in model.encoder.parameters():
#     param.requires_grad = True
# optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
