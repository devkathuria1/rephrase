"""Run the classifier(s) over HateCheck and save per-row predictions.

Usage: python -m evaluation.run_hatecheck [--limit N] [--secondary]
No LLM is ever called.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config
from evaluation import common
from src import classifier, severity

CANDIDATES = {
    "text": ("test_case", "text", "sentence"),
    "label_gold": ("label_gold", "label", "gold_label"),
    "functionality": ("functionality", "functionality_name"),
    "target_ident": ("target_ident", "target", "target_group"),
}
REQUIRED = ("text", "label_gold", "functionality")
CHUNK = 128
DOWNLOAD_HELP = """\
HateCheck file not found: {path}

Download it (Rottger et al., 2021) and save it as data/hatecheck.csv:
  https://github.com/paul-rottger/hatecheck-data  ->  test_suite_cases.csv
  e.g. curl -L -o data/hatecheck.csv https://raw.githubusercontent.com/paul-rottger/hatecheck-data/main/test_suite_cases.csv
"""


def detect_columns(columns: list[str]) -> dict[str, str | None]:
    """Map standard names (text, label_gold, functionality, target_ident) to CSV columns.

    Matching is case-insensitive. Target is optional (None if absent).

    Raises:
        ValueError: if a required column cannot be found.
    """
    lowered = {c.lower(): c for c in columns}
    found = {
        std: next((lowered[c] for c in names if c in lowered), None)
        for std, names in CANDIDATES.items()
    }
    missing = [std for std in REQUIRED if found[std] is None]
    if missing:
        raise ValueError(f"could not detect columns for {missing} in {list(columns)}")
    return found


def load_hatecheck(path: Path, limit: int | None = None) -> pd.DataFrame:
    """Load HateCheck into a frame with columns text, label_gold (bool), functionality, target_ident.

    ``label_gold`` is True for hateful. With ``limit``, a reproducible random
    sample is taken (seed 0) so the subset spans functionalities.
    """
    raw = pd.read_csv(path)
    cols = detect_columns(list(raw.columns))
    df = pd.DataFrame(
        {
            "text": raw[cols["text"]].astype(str).str.strip(),
            "label_gold": raw[cols["label_gold"]].astype(str).str.strip().str.lower(),
            "functionality": raw[cols["functionality"]],
            "target_ident": raw[cols["target_ident"]] if cols["target_ident"] else None,
        }
    )
    unknown = set(df["label_gold"]) - {"hateful", "non-hateful"}
    if unknown:
        raise ValueError(f"unexpected gold labels: {sorted(unknown)}")
    df["label_gold"] = df["label_gold"] == "hateful"
    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=0).sort_index()
    return df.reset_index(drop=True)


def run(df: pd.DataFrame, secondary: bool) -> pd.DataFrame:
    """Score every text in batches and add label scores, tier and predictions."""
    texts = df["text"].tolist()
    scores: list[dict[str, float]] = []
    for start in tqdm(range(0, len(texts), CHUNK), desc="toxic-bert", unit="batch"):
        scores.extend(classifier.score(texts[start : start + CHUNK]))
    out = pd.concat([df, pd.DataFrame(scores)], axis=1)
    out["tier"] = [severity.assess(s).tier for s in scores]
    out["pred_hateful"] = out["tier"] != severity.NONE
    if secondary:
        hate: list[float] = []
        for start in tqdm(range(0, len(texts), CHUNK), desc="secondary", unit="batch"):
            hate.extend(classifier.score_secondary(texts[start : start + CHUNK]))
        out["secondary_hate"] = hate
        out["pred_hateful_secondary"] = out["secondary_hate"] >= config.SECONDARY_HATE_THRESHOLD
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=common.DATA_PATH)
    parser.add_argument("--out", type=Path, default=common.PREDICTIONS_PATH)
    parser.add_argument("--limit", type=int, help="random sample of N rows for quick tests")
    parser.add_argument("--secondary", action="store_true", help="also run the secondary classifier (reported separately)")
    args = parser.parse_args(argv)

    if not args.data.exists():
        print(DOWNLOAD_HELP.format(path=args.data))
        return 1
    df = load_hatecheck(args.data, args.limit)
    predictions = run(df, args.secondary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out, index=False)
    print(f"Wrote {len(predictions)} predictions to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
