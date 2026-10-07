"""Presentation helpers for the Streamlit app (no Streamlit import, so testable)."""

from __future__ import annotations

import html
import json
from pathlib import Path

import config
from src import llm, severity

FAILURE_EXAMPLES_PATH = Path(__file__).resolve().parent.parent / "data" / "failure_examples.json"
EXAMPLE_FIELDS = ("id", "category", "text", "expected", "why_it_fails", "who_is_harmed")
EXPECTED_VALUES = ("should flag", "should not flag")

TIER_COLORS = {
    severity.NONE: "#2e9e5b",
    severity.MILD: "#d4a017",
    severity.HARASSMENT: "#e8751a",
    severity.HATE: "#d32f2f",
}
# Words below this importance (drop in label probability) are not highlighted:
# smaller changes are within noise.
MIN_IMPORTANCE = 0.005
MAX_HIGHLIGHT_ALPHA = 0.75
_HIGHLIGHT_RGB = "255, 99, 71"


SECONDARY_NOTE = (
    "On HateCheck, this model alone wrongly flagged far fewer harmless cases "
    "(false positive rate 0.08 vs 0.61 for toxic-bert). Caveat: its score may be "
    "inflated if HateCheck-like data was in its training. Here it adds to "
    "toxic-bert's flags (it can raise a text to Hate speech but never lowers a "
    "tier), and toxic-bert still provides the per-label breakdown."
)
NO_SAFE_REPHRASING ="No safe rephrasing available"


def no_rephrasing_card(explanation: str) -> str:
    """Markdown for the card shown when the model declines to rephrase."""
    return (
        f"**{NO_SAFE_REPHRASING}.** The statement targets a group, and any polite "
        f"version would keep the harmful claim.\n\n{explanation}"
    )


def badge_html(tier: str) -> str:
    """Return a coloured pill for a severity tier."""
    color = TIER_COLORS.get(tier, "#777777")
    return (
        f'<span style="background:{color};color:#fff;padding:4px 14px;'
        f'border-radius:14px;font-weight:600;">{html.escape(tier)}</span>'
    )


def intensity(importance: float, max_importance: float) -> float:
    """Map a word's importance to 0..1 relative to the strongest word.

    Words at or below ``MIN_IMPORTANCE`` (or in texts where nothing reaches it)
    get 0.
    """
    if importance < MIN_IMPORTANCE or max_importance < MIN_IMPORTANCE:
        return 0.0
    return min(1.0, importance / max_importance)


def highlight_html(text: str, word_importance: dict | None) -> str:
    """Render text as HTML with each scored word shaded by occlusion importance.

    All user text is HTML-escaped. Words beyond the explainer's word cap (and
    all words when there is no explanation) are rendered unshaded.
    """
    words = text.split()
    scored = (word_importance or {}).get("words", [])
    max_imp = max((w["importance"] for w in scored), default=0.0)
    parts = []
    for i, word in enumerate(words):
        safe = html.escape(word)
        alpha = 0.0
        if i < len(scored):
            alpha = intensity(scored[i]["importance"], max_imp) * MAX_HIGHLIGHT_ALPHA
        if alpha > 0:
            parts.append(
                f'<span style="background:rgba({_HIGHLIGHT_RGB},{alpha:.2f});'
                f'padding:1px 3px;border-radius:3px;">{safe}</span>'
            )
        else:
            parts.append(safe)
    return " ".join(parts)


def provider_status() -> tuple[str, bool, str | None]:
    """Return (provider, rephrasing enabled, reason if not). Never exposes keys."""
    return config.llm_provider(), llm.is_enabled(), llm.disabled_reason()


def is_flagged(tier: str) -> bool:
    """Return True if the tier is anything other than None."""
    return tier != severity.NONE


def top_scores(scores: dict[str, float], secondary: float | None, n: int = 3) -> str:
    """Summarise the n highest label scores (plus the secondary hate score)."""
    items = dict(scores)
    if secondary is not None:
        items["secondary_hate"] = secondary
    ranked = sorted(items.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return ", ".join(f"{k} {v:.2f}" for k, v in ranked)


def load_failure_examples(path: Path = FAILURE_EXAMPLES_PATH) -> list[dict]:
    """Load and validate the failure examples file.

    Raises:
        ValueError: if the file is not a non-empty list of well-formed entries
            with unique ids.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("failure examples must be a non-empty list")
    seen = set()
    for entry in data:
        missing = [f for f in EXAMPLE_FIELDS if not entry.get(f)]
        if missing:
            raise ValueError(f"example {entry.get('id')!r} is missing {missing}")
        if entry["expected"] not in EXPECTED_VALUES:
            raise ValueError(f"example {entry['id']}: bad 'expected' {entry['expected']!r}")
        if entry["id"] in seen:
            raise ValueError(f"duplicate example id {entry['id']}")
        seen.add(entry["id"])
    return data
