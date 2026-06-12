import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

model_path = r"models\finetuned ModernBERT"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path
)
id2label = {
    0: "All-or-nothing thinking",
    1: "Emotional Reasoning",
    2: "Fortune-Telling",
    3: "Labeling",
    4: "Magnification",
    5: "Mental filter",
    6: "Mind Reading",
    7: "No Distortion",
    8: "Overgeneralization",
    9: "Personalization",
    10: "Should statements"
}
text = "i hate my face and i am sure others think the same"
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=256
)

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

probs = torch.sigmoid(logits)[0]

results = [
    (id2label[i], probs[i].item())
    for i in range(len(probs))
]

results = sorted(
    results,
    key=lambda x: x[1],
    reverse=True
)

for distortion, prob in results:
    print(f"{distortion}: {prob:.4f}")