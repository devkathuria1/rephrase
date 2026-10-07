"""Export the worst functionalities with masked example errors.

Usage: python -m evaluation.examples
Slurs and explicit words are masked; raw text appears only in the predictions CSV.
"""

from __future__ import annotations

import re
import sys

import pandas as pd

from evaluation import common, metrics
from evaluation.run_hatecheck import detect_columns

TOP_FUNCTIONALITIES = 10
EXAMPLES_PER_FUNCTIONALITY = 3
CONTENT_WARNING = (
    "> **Content warning.** This file lists test sentences from HateCheck, a "
    "hate speech test suite, and may be upsetting. Slurs and explicit words are "
    "masked as [slur] / [explicit], but the sentences still express hateful "
    "views about groups.\n"
)
# Slur stems (the terms HateCheck uses; derived forms are caught by \w*).
_SLUR = re.compile(
    r"\b(?:bitch|dyke|fag|faggot|hoe|nigg\w*|queer|slut|whore|cunt|retard|tranny)\w*", re.I
)
# Explicit words, including obfuscated spellings (leet, spacing, swaps, deletions).
_EXPLICIT = re.compile(
    r"\w*(?:fuck|fuk|f\*+k|shit|sh1t|siht|asshole|4ssh|4ss)\w*"
    r"|\b(?:f\W?u\W?c\W?k|s\W?h\W?[i1]\W?t|a\W?s\W?s\W?holes?|dumbass|dick\w*|bastard|prick|"
    r"piss\w*|cock(?!roach)|twat|wank\w*|crap\w*|damn\w*)\w*",
    re.I,
)
_IDENTITY_PLACEHOLDER = re.compile(r"\[IDENTITY_[A-Z]\]")
_SLUR_PLACEHOLDER = re.compile(r"\[SLUR_[A-Z](?:_[A-Z]+)*\]")


def mask_text(
    text: str,
    case_templ: str | None = None,
    target: str | None = None,
    focus_words: str | None = None,
) -> str:
    """Mask slurs and explicit words in a test sentence.

    If the template has a [SLUR_*] placeholder, the sentence is rebuilt from the
    template so obfuscated slur spellings are never shown. The row's own focus
    words are also masked when they look explicit or slur-like.
    """
    if isinstance(case_templ, str) and "[SLUR" in case_templ:
        text = _SLUR_PLACEHOLDER.sub("[slur]", case_templ)
        text = _IDENTITY_PLACEHOLDER.sub(target if isinstance(target, str) else "[group]", text)
    text = _SLUR.sub("[slur]", text)
    text = _EXPLICIT.sub("[explicit]", text)
    if isinstance(focus_words, str) and (_SLUR.search(focus_words) or _EXPLICIT.search(focus_words)):
        text = re.sub(re.escape(focus_words.strip()), "[explicit]", text, flags=re.I)
    return text


def worst_cases(df: pd.DataFrame, pred: pd.Series, func: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return up to 3 misclassified rows (seeded sample) for each of the 10 worst functionalities."""
    out = {}
    for name in func.head(TOP_FUNCTIONALITIES)["functionality"]:
        wrong = df[(df["functionality"] == name) & (df["label_gold"] != pred)]
        out[name] = wrong.sample(n=min(EXAMPLES_PER_FUNCTIONALITY, len(wrong)), random_state=0)
    return out


def render(
    df: pd.DataFrame, pred: pd.Series, func: pd.DataFrame, templates: pd.DataFrame | None = None
) -> str:
    """Build the masked markdown report.

    ``templates`` (optional) has columns case_templ and focus_words, indexed like ``df``.
    """
    lines = [
        CONTENT_WARNING,
        "# HateCheck: worst functionalities (toxic-bert, binary mapping)\n",
        "Error rate = share of the functionality's cases the model got wrong. "
        "These are examples, not statistics.\n",
    ]
    stats = func.set_index("functionality")
    for name, rows in worst_cases(df, pred, func).items():
        s = stats.loc[name]
        lines.append(f"## {name} (gold: {s['gold']}, n={int(s['n'])}, error rate {s['error_rate']:.0%})\n")
        for idx, row in rows.iterrows():
            templ = focus = None
            if templates is not None:
                templ, focus = templates.loc[idx, "case_templ"], templates.loc[idx, "focus_words"]
            masked = mask_text(row["text"], templ, row["target_ident"], focus)
            predicted = "hateful" if pred.loc[idx] else "non-hateful"
            gold = "hateful" if row["label_gold"] else "non-hateful"
            top = max(metrics.LABELS, key=lambda k: row[k])
            tier = row["tier"] if pd.notna(row["tier"]) else "None"  # pandas reads "None" as NaN
            lines.append(
                f'- "{masked}"  \n  gold: {gold}, predicted: {predicted} '
                f"(tier {tier}, top label {top} {row[top]:.2f})"
            )
        lines.append("")
    return "\n".join(lines)


def load_templates(df: pd.DataFrame) -> pd.DataFrame | None:
    """Look up case_templ and focus_words for each prediction row by sentence text."""
    if not common.DATA_PATH.exists():
        return None
    raw = pd.read_csv(common.DATA_PATH)
    if not {"case_templ", "focus_words"} <= set(raw.columns):
        return None
    raw["text"] = raw[detect_columns(list(raw.columns))["text"]].astype(str).str.strip()
    lookup = raw.drop_duplicates("text")[["text", "case_templ", "focus_words"]]
    merged = df[["text"]].reset_index().merge(lookup, on="text", how="left").set_index("index")
    return merged[["case_templ", "focus_words"]]


def main() -> int:
    if not common.PREDICTIONS_PATH.exists():
        print(f"{common.PREDICTIONS_PATH} not found; run `python -m evaluation.run_hatecheck` first.")
        return 1
    df = pd.read_csv(common.PREDICTIONS_PATH)
    pred = metrics.predict_primary(df)
    func = metrics.by_functionality(df, pred)
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    common.WORST_CASES_PATH.write_text(render(df, pred, func, load_templates(df)), encoding="utf-8")
    print("Wrote", common.WORST_CASES_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
