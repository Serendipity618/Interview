import torch
from transformers import AutoTokenizer
from app.model import SentenceTransformerMTL
from app.utils import training_step

MODEL_NAME = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = SentenceTransformerMTL(MODEL_NAME)

# Example sentences and labels for Task A and Task B
sentences = [
    "The movie was fantastic!",
    "I didn’t like the food at all.",
    "This book is worth reading."
]
labels_a = torch.tensor([0, 1, 2])  # Task A: arbitrary class labels
labels_b = torch.tensor([1, 0, 1])  # Task B: sentiment labels (1=positive, 0=negative)

# Task 1: Tokenization of inputs
encoding = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

# Task 3: Apply feature-based transfer learning by freezing encoder
for param in model.encoder.parameters():
    param.requires_grad = False

# Optimizer and loss functions
optimizer = torch.optim.Adam(
    list(model.classifier_a.parameters()) + list(model.classifier_b.parameters()), lr=1e-3
)
criterion_a = torch.nn.CrossEntropyLoss()
criterion_b = torch.nn.CrossEntropyLoss()

# Task 4: Run one training step with dummy batch
batch = (input_ids, attention_mask, labels_a, labels_b)
loss = training_step(model, batch, optimizer, criterion_a, criterion_b)
print(f"Training loss: {loss:.4f}")

# Task 1 & 2: Run inference to validate predictions
model.eval()
with torch.no_grad():
    logits_a, logits_b = model(input_ids, attention_mask)
    preds_a = torch.argmax(logits_a, dim=1)
    preds_b = torch.argmax(logits_b, dim=1)
    print("Logits Task A (Classification):", logits_a)
    print("Predicted Labels Task A:", preds_a)
    print("Logits Task B (Sentiment):", logits_b)
    print("Predicted Labels Task B:", preds_b)
