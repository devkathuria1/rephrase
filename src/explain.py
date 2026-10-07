"""Leave-one-out occlusion explanations for toxicity scores."""

from __future__ import annotations

import config
from src import classifier


def explain(text: str, base_scores: dict[str, float], label: str | None = None) -> dict:
    """Score how much each word contributes to one toxicity label.

    Each word (or n-gram window, see ``config.OCCLUSION_NGRAM``) is removed in
    turn and the drop in the label's probability is measured. ``label``
    defaults to the highest-scoring label. All variants
    are scored in batches. Input is capped at ``config.MAX_OCCLUSION_WORDS``.

    Returns:
        ``{"label": str, "words": [{"word": str, "importance": float}, ...],
        "truncated": bool}``. Importance is the score drop (can be negative);
        words beyond the cap are not scored.
    """
    words = text.split()
    truncated = len(words) > config.MAX_OCCLUSION_WORDS
    words = words[: config.MAX_OCCLUSION_WORDS]
    if label not in base_scores:
        label = max(base_scores, key=base_scores.get)
    if not words:
        return {"label": label, "words": [], "truncated": False}

    n = min(config.OCCLUSION_NGRAM, len(words))
    spans = [(i, i + n) for i in range(len(words) - n + 1)]
    # Removing the only words leaves an empty string; "." is a neutral stand-in.
    variants = [" ".join(words[:s] + words[e:]) or "." for s, e in spans]
    results = classifier.score(variants)

    base = base_scores[label]
    importance: list[float | None] = [None] * len(words)
    for (s, e), res in zip(spans, results):
        drop = base - res[label]
        for i in range(s, e):
            importance[i] = drop if importance[i] is None else max(importance[i], drop)
    return {
        "label": label,
        "words": [
            {"word": w, "importance": round(imp or 0.0, 4)}
            for w, imp in zip(words, importance)
        ],
        "truncated": truncated,
    }
