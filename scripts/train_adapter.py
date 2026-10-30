"""Train a lightweight custom classification head adapter on balanced toxic/hate data.

Demonstrates transfer learning and domain adaptation using PyTorch.
Saves model checkpoint to models/custom_toxic_adapter.pt.
"""

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL = "distilbert-base-uncased"
CHECKPOINT_PATH = Path("models/custom_toxic_adapter.pt")

# Curated benchmark dataset of toxic & non-toxic exemplars for adaptation
TRAIN_DATA = [
    ("You are an absolute idiot and nobody likes you.", 1),
    ("Go back to where you came from, you disgust me.", 1),
    ("I hope someone destroys your house tonight.", 1),
    ("People like you should be completely eradicated.", 1),
    ("Shut your ugly mouth you worthless piece of trash.", 1),
    ("Stop acting like a clown, you are totally pathetic.", 1),
    ("If you ever say that again I will hunt you down.", 1),
    ("You are dumb, incompetent, and should be fired immediately.", 1),
    ("I hate everything about your kind of people.", 1),
    ("Nobody cares about your opinion, crawl back into your hole.", 1),
    ("Have a wonderful day, thank you so much for the assistance!", 0),
    ("I respectfully disagree with your conclusion on this topic.", 0),
    ("The presentation was very clear and well organized.", 0),
    ("We should schedule a meeting tomorrow to discuss the roadmap.", 0),
    ("Queer communities have made tremendous strides in civil rights.", 0),
    ("The article discussed rising levels of online hate speech in society.", 0),
    ("Let's review the code changes and merge the pull request.", 0),
    ("Can you please send me the documentation link?", 0),
    ("Great job team on delivering the feature ahead of schedule.", 0),
    ("I am learning how to build machine learning models in PyTorch.", 0),
]


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


class ToxicAdapterClassifier(nn.Module):
    """Transformer backbone with a custom trainable 2-layer classification head."""

    def __init__(self, base_model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        # Freeze lower encoder layers to demonstrate parameter-efficient adaptation
        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),  # 2 classes: 0 = Benign, 1 = Toxic/Hate
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation (index 0)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier_head(cls_rep)
        return logits


def train(epochs=5, batch_size=4, lr=1e-3):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading tokenizer and base model: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = ToxicAdapterClassifier(BASE_MODEL).to(DEVICE)

    texts = [t[0] for t in TRAIN_DATA]
    labels = [t[1] for t in TRAIN_DATA]
    dataset = TextDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier_head.parameters(), lr=lr)

    print("Beginning training loop on classification head...")
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            targets = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        acc = correct / total
        print(f"Epoch {epoch}/{epochs} | Loss: {total_loss / len(dataloader):.4f} | Accuracy: {acc:.1%}")

    # Save state dict
    torch.save(
        {
            "base_model": BASE_MODEL,
            "classifier_head_state_dict": model.classifier_head.state_dict(),
        },
        CHECKPOINT_PATH,
    )
    print(f"Successfully saved trained adapter checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()

