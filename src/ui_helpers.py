"""Presentation helpers for the Streamlit app (no Streamlit import, so testable).

Every HTML builder escapes user-derived text. HTML is kept on one line because
Markdown treats blank or indented lines inside HTML as code blocks.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import config
from src import llm, severity

FAILURE_EXAMPLES_PATH = Path(__file__).resolve().parent.parent / "data" / "failure_examples.json"
EXAMPLE_FIELDS = ("id", "category", "text", "expected", "why_it_fails", "who_is_harmed")
EXPECTED_VALUES = ("should flag", "should not flag")

# Mid-tone fills with white text (contrast >= 4.5:1 in light and dark themes).
TIER_COLORS = {
    severity.NONE: "#1b7f4b",
    severity.MILD: "#946200",
    severity.HARASSMENT: "#b34700",
    severity.HATE: "#c62828",
}
# Icon and one-line meaning per tier, so colour is never the only signal.
TIER_ICONS = {
    severity.NONE: "✅",
    severity.MILD: "🟡",
    severity.HARASSMENT: "🟠",
    severity.HATE: "🔴",
}
TIER_MEANINGS = {
    severity.NONE: "Nothing was flagged. This is not a guarantee the text is harmless.",
    severity.MILD: "Rude or crude wording, but no clear target or threat.",
    severity.HARASSMENT: "Insults or threats aimed at a person.",
    severity.HATE: "Attacks a person or group over who they are. Highest severity.",
}
# Words below this importance (drop in label probability) are not highlighted:
# smaller changes are within noise.
MIN_IMPORTANCE = 0.005
MAX_HIGHLIGHT_ALPHA = 0.75
# Words at or above this alpha are also underlined, so shading is not the only cue.
HIGHLIGHT_STRONG = 0.45
_HIGHLIGHT_RGB = "255, 99, 71"


SECONDARY_NOTE = (
    "On HateCheck, this model alone wrongly flagged far fewer harmless cases "
    "(false positive rate 0.08 vs 0.61 for toxic-bert). Caveat: its score may be "
    "inflated if HateCheck-like data was in its training. Here it adds to "
    "toxic-bert's flags (it can raise a text to Hate speech but never lowers a "
    "tier), and toxic-bert still provides the per-label breakdown."
)
NO_SAFE_REPHRASING = "No safe rephrasing available"
NO_REPHRASING_REASON = (
    "The statement targets a group, and any polite version would keep the harmful claim."
)

TAGLINE = "Check a message, understand why it may hurt, and find a respectful way to say it."
STEPS = (
    ("Input", "Paste text or a screenshot"),
    ("Detect", "Score toxicity and hate"),
    ("Explain", "See which words matter"),
    ("Rephrase", "Get a kinder wording"),
)
# Short, mild, slur-free sentences for the "Try an example" buttons.
EXAMPLES = (
    ("😒 Mild insult", "That was a really dumb idea and you should feel bad about it."),
    ("⚠️ Threat", "If you post that again, you will regret it."),
    ("🚫 Group-targeted", "People from that country are all lazy and cannot be trusted."),
    ("🙂 Harmless", "Thanks for the help today, I really appreciate it."),
)
FOOTER = "Rephrase is an assistive aid, not a moderator. No text is stored by this app."


def no_rephrasing_card(explanation: str) -> str:
    """Markdown for the card shown when the model declines to rephrase."""
    return f"**{NO_SAFE_REPHRASING}.** {NO_REPHRASING_REASON}\n\n{explanation}"


def badge_html(tier: str) -> str:
    """Return a small coloured pill (with icon) for a severity tier."""
    color = TIER_COLORS.get(tier, "#555555")
    icon = TIER_ICONS.get(tier, "")
    return (
        f'<span class="rp-badge-small" style="background:{color};">'
        f"{icon} {html.escape(tier)}</span>"
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
            underline = "text-decoration:underline;" if alpha >= HIGHLIGHT_STRONG else ""
            parts.append(
                f'<span style="background:rgba({_HIGHLIGHT_RGB},{alpha:.2f});{underline}'
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


# ---------------------------------------------------------------- HTML builders


def hero_html() -> str:
    """Centered title, tagline and the 4-step strip."""
    steps = "".join(
        f'<div class="rp-step"><div class="rp-step-num">{i}</div>'
        f"<div><b>{html.escape(name)}</b><span>{html.escape(desc)}</span></div></div>"
        for i, (name, desc) in enumerate(STEPS, 1)
    )
    return (
        f'<div class="rp-hero"><h1>💬 Rephrase</h1><p>{html.escape(TAGLINE)}</p>'
        f'<div class="rp-steps">{steps}</div></div>'
    )


def footer_html() -> str:
    """One-line footer shown on every page."""
    return f'<div class="rp-footer">{html.escape(FOOTER)}</div>'


def pill_html(label: str, on: bool) -> str:
    """Small status pill with an icon (state is not conveyed by colour alone)."""
    icon, cls = ("●", "rp-on") if on else ("○", "rp-off")
    return f'<span class="rp-pill {cls}">{icon} {html.escape(label)}</span>'


def provider_pill_html(provider: str, enabled: bool) -> str:
    """Pill for the LLM provider status."""
    return pill_html(f"{provider}: {'ready' if enabled else 'off'}", enabled)


def severity_card_html(tier: str) -> str:
    """Large coloured badge plus a one-line meaning for the tier."""
    color = TIER_COLORS.get(tier, "#555555")
    icon = TIER_ICONS.get(tier, "")
    meaning = TIER_MEANINGS.get(tier, "")
    return (
        f'<div class="rp-severity"><span class="rp-badge" style="background:{color};">'
        f"{icon} {html.escape(tier)}</span>"
        f'<span class="rp-meaning">{html.escape(meaning)}</span></div>'
    )


def score_bars_html(scores: dict[str, float], hot_at: float = 0.5) -> str:
    """Horizontal bars (one per label, in the given order) with the value as text."""
    rows = []
    for label, value in scores.items():
        pct = max(0.0, min(1.0, float(value))) * 100
        hot = " rp-hot" if value >= hot_at else ""
        name = html.escape(str(label))
        rows.append(
            f'<div class="rp-bar-row"><span>{name}</span>'
            f'<div class="rp-bar-track" role="img" aria-label="{name} {pct:.0f} percent">'
            f'<div class="rp-bar-fill{hot}" style="width:{pct:.1f}%;"></div></div>'
            f'<span class="rp-bar-val">{pct:.0f}%</span></div>'
        )
    return "".join(rows)


def compare_cards_html(original: str, rephrased: str, explanation: str = "") -> str:
    """Two equal cards: Original and Rephrased (or a distinct 'no safe rephrasing' card)."""
    left = f'<div class="rp-card"><h4>📝 Original</h4><p>{html.escape(original)}</p></div>'
    if rephrased:
        right = (
            f'<div class="rp-card rp-good"><h4>✅ Rephrased</h4>'
            f"<p>{html.escape(rephrased)}</p></div>"
        )
    else:
        extra = f"<br><br>{html.escape(explanation)}" if explanation else ""
        right = (
            f'<div class="rp-card rp-none"><h4>⚠️ {NO_SAFE_REPHRASING}</h4>'
            f"<p>{html.escape(NO_REPHRASING_REASON)}{extra}</p></div>"
        )
    return f'<div class="rp-compare">{left}{right}</div>'


def text_card_html(title: str, body: str, icon: str = "") -> str:
    """A single titled card with escaped body text."""
    return (
        f'<div class="rp-card"><h4>{icon} {html.escape(title)}</h4>'
        f"<p>{html.escape(body)}</p></div>"
    )


def legend_html() -> str:
    """Legend for the word highlights (slight, strong, none)."""

    def swatch(alpha: float, underline: bool = False) -> str:
        deco = "text-decoration:underline;" if underline else ""
        return (
            f'<span class="rp-swatch" style="background:rgba({_HIGHLIGHT_RGB},{alpha:.2f});'
            f'{deco}"></span>'
        )

    return (
        '<div class="rp-legend">'
        f"<span>{swatch(0.2)}Slight influence</span>"
        f"<span>{swatch(MAX_HIGHLIGHT_ALPHA, True)}Strong influence (underlined)</span>"
        "<span>No shading: no effect on the score</span></div>"
    )


def classify_warnings(warnings: list[str]) -> dict[str, list[str]]:
    """Group pipeline warnings into 'info' (expected, e.g. rephrasing off) and 'warning'."""
    groups: dict[str, list[str]] = {"warning": [], "info": []}
    for message in warnings:
        kind = "info" if message.startswith("Rephrasing disabled") else "warning"
        groups[kind].append(message)
    return groups


def example_card_html(row: dict, correct: bool) -> str:
    """Header, quoted text and expected/actual line for one limitations example."""
    mark = (
        '<span class="rp-mark rp-ok">✅ Correct</span>'
        if correct
        else '<span class="rp-mark rp-bad">❌ Wrong</span>'
    )
    return (
        f'<div class="rp-example-head"><b>{html.escape(str(row["id"]))}. '
        f'{html.escape(row["category"])}</b>{mark}</div>'
        f'<div class="rp-quote">{html.escape(row["text"])}</div>'
        '<div class="rp-meta">'
        f'<div><small>Expected</small>{html.escape(row["expected"])}</div>'
        f'<div><small>Actual</small>{badge_html(row["tier"])}</div>'
        f'<div><small>Top scores</small>{html.escape(row["top_scores"])}</div></div>'
    )
