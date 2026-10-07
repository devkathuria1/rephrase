"""Map classifier scores to a severity tier."""

from __future__ import annotations

from dataclasses import dataclass, field

import config

HATE = "Hate speech"
HARASSMENT = "Harassment"
MILD = "Mildly offensive"
NONE = "None"


@dataclass
class Severity:
    """A severity tier and the labels that triggered it."""

    tier: str
    triggered: list[str] = field(default_factory=list)


def assess(scores: dict[str, float], secondary_hate: float | None = None) -> Severity:
    """Return the severity tier for a set of label scores.

    Tiers are checked from most to least severe; the first match wins.

    Args:
        scores: toxic-bert label probabilities.
        secondary_hate: optional hate probability from the secondary model.
    """
    hate = [
        label
        for label, limit in (
            ("identity_hate", config.HATE_THRESHOLD),
            ("severe_toxic", config.SEVERE_TOXIC_THRESHOLD),
        )
        if scores.get(label, 0.0) >= limit
    ]
    if secondary_hate is not None and secondary_hate >= config.SECONDARY_HATE_THRESHOLD:
        hate.append("secondary_hate")
    if hate:
        return Severity(HATE, hate)

    harassment = [
        label
        for label, limit in (
            ("threat", config.THREAT_THRESHOLD),
            ("insult", config.INSULT_THRESHOLD),
        )
        if scores.get(label, 0.0) >= limit
    ]
    if harassment:
        return Severity(HARASSMENT, harassment)

    mild = [
        label
        for label in ("toxic", "obscene")
        if scores.get(label, 0.0) >= config.MILD_TOXIC_THRESHOLD
    ]
    if mild:
        return Severity(MILD, mild)
    return Severity(NONE)
