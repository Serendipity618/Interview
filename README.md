# ML Apprentice Take-Home Project

This project implements a multi-task sentence transformer using BERT (`bert-base-uncased`), following the ML Apprentice take-home exercise requirements. It is structured for clarity and ease of reproducibility.

---

## 🧠 Tasks Implemented

### ✅ Task 1: Sentence Transformer
- Uses a pre-trained `bert-base-uncased` model.
- Applies mean pooling to get fixed-length sentence embeddings.

### ✅ Task 2: Multi-Task Learning
- Two task-specific heads:
  - **Task A**: Sentence classification (e.g., topic)
  - **Task B**: Sentiment analysis

### ✅ Task 3: Training Considerations
- Freezing of the transformer encoder is supported for transfer learning.
- Optionally allows fine-tuning the encoder.

### ✅ Task 4: Training Loop
- One-step training loop with support for multiple epochs.
- Supports joint optimization of multi-task losses.

---

## 📂 Project Structure

```
Interview/
├── app/
│   ├── model.py         # Multi-task sentence transformer (Task 1 & 2)
│   ├── utils.py         # Training loop with epoch support (Task 4)
│   └── inference.py     # Sample inference script with predicted labels
├── train.py             # Transfer learning setup (Task 3)
├── main.py              # End-to-end execution of all tasks
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 🚀 Usage

### 📌 Install dependencies
```bash
pip install -r requirements.txt
```

### ▶️ Run all tasks
```bash
python main.py
```

### 🧪 Run inference only
```bash
python -m app.inference
```

---

## 🔧 Model
- Model: `bert-base-uncased`
- Framework: PyTorch + HuggingFace Transformers

---

## 📝 Notes
- Predictions are printed alongside logits for interpretability.
- Dummy labels are used for demonstration — replace with your dataset for training.
