"""Toxicity classification with toxic-bert and an optional hate-speech model."""

from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


@lru_cache(maxsize=2)
def _load(model_name: str):
    """Load and cache a tokenizer/model pair in eval mode."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def preload(use_secondary: bool = False) -> None:
    """Load the model(s) now so the first request does not pay the load cost."""
    _load(config.PRIMARY_MODEL)
    if use_secondary:
        _load(config.SECONDARY_MODEL)


def _logits(model_name: str, texts: list[str]) -> torch.Tensor:
    """Return logits for all texts, processed in batches."""
    tokenizer, model = _load(model_name)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), config.CLASSIFIER_BATCH_SIZE):
            batch = texts[start : start + config.CLASSIFIER_BATCH_SIZE]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=config.CLASSIFIER_MAX_TOKENS,
                return_tensors="pt",
            )
            chunks.append(model(**enc).logits)
    return torch.cat(chunks)


def score(texts: list[str]) -> list[dict[str, float]]:
    """Score each text on the six toxic-bert labels (independent probabilities)."""
    if not texts:
        return []
    probs = torch.sigmoid(_logits(config.PRIMARY_MODEL, texts))
    return [dict(zip(LABELS, (round(p, 4) for p in row.tolist()))) for row in probs]


def score_secondary(texts: list[str]) -> list[float]:
    """Return the probability that each text is hate speech (secondary model)."""
    if not texts:
        return []
    _, model = _load(config.SECONDARY_MODEL)
    hate_idx = next(
        (i for i, name in model.config.id2label.items() if name.lower() == "hate"), 1
    )
    probs = torch.softmax(_logits(config.SECONDARY_MODEL, texts), dim=-1)
    return [round(p, 4) for p in probs[:, hate_idx].tolist()]
