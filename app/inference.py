from transformers import AutoTokenizer
from model import SentenceTransformerMTL

MODEL_NAME = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = SentenceTransformerMTL(MODEL_NAME)

sentences = [
    "The movie was fantastic!",
    "I didn’t like the food at all.",
    "This book is worth reading."
]

encoding = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
logits_a, logits_b = model(**encoding)
print("Logits Task A (Classification):", logits_a)
print("Logits Task B (Sentiment):", logits_b)
