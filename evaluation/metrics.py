"""Metrics for the HateCheck predictions (positive class = hateful).

Usage: python -m evaluation.metrics
"""

from __future__ import annotations

import json
import math
import sys
from unittest import mock

import pandas as pd

import config
from evaluation import common
from src import classifier, severity

LABELS = classifier.LABELS


def _div(num: float, den: float) -> float:
    """Divide, returning NaN when the denominator is zero."""
    return num / den if den else math.nan


def confusion(y_true: pd.Series, y_pred: pd.Series) -> dict[str, int]:
    """Return tp/fp/tn/fn counts for boolean series."""
    t, p = y_true.astype(bool), y_pred.astype(bool)
    return {
        "tp": int((t & p).sum()),
        "fp": int((~t & p).sum()),
        "tn": int((~t & ~p).sum()),
        "fn": int((t & ~p).sum()),
    }


def rates(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Return n, counts, accuracy, precision, recall, F1, FPR and FNR (NaN if undefined)."""
    c = confusion(y_true, y_pred)
    n = sum(c.values())
    precision = _div(c["tp"], c["tp"] + c["fp"])
    recall = _div(c["tp"], c["tp"] + c["fn"])
    if math.isnan(precision) or math.isnan(recall):
        f1 = math.nan
    else:
        f1 = _div(2 * precision * recall, precision + recall)
    return {
        "n": n,
        **c,
        "accuracy": _div(c["tp"] + c["tn"], n),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": _div(c["fp"], c["fp"] + c["tn"]),
        "fnr": _div(c["fn"], c["fn"] + c["tp"]),
    }


def by_functionality(df: pd.DataFrame, pred: pd.Series) -> pd.DataFrame:
    """Per-functionality n, accuracy, error rate, FPR and FNR, worst first.

    Each HateCheck functionality has a single gold class, so FPR (non-hateful
    functionalities) or FNR (hateful ones) is the relevant one; the other is NaN.
    """
    rows = []
    for name, idx in df.groupby("functionality").groups.items():
        gold = df.loc[idx, "label_gold"]
        r = rates(gold, pred.loc[idx])
        if gold.all():
            gold_name = "hateful"
        elif not gold.any():
            gold_name = "non-hateful"
        else:
            gold_name = "mixed"
        rows.append(
            {
                "functionality": name,
                "gold": gold_name,
                "error_rate": 1 - r["accuracy"],
                **{k: r[k] for k in ("n", "accuracy", "fpr", "fnr")},
            }
        )
    return pd.DataFrame(rows).sort_values("error_rate", ascending=False).reset_index(drop=True)


def flag_rate_by_target(df: pd.DataFrame, pred: pd.Series) -> pd.DataFrame:
    """Flag rate (share predicted hateful) per target group, with FPR and FNR.

    Rows without a target group are excluded; empty if there is no target column.
    """
    columns = ["target_ident", "n", "flag_rate", "fpr", "fnr"]
    if "target_ident" not in df or df["target_ident"].isna().all():
        return pd.DataFrame(columns=columns)
    rows = []
    with_target = df.dropna(subset=["target_ident"])
    for name, idx in with_target.groupby("target_ident").groups.items():
        r = rates(df.loc[idx, "label_gold"], pred.loc[idx])
        rows.append(
            {
                "target_ident": name,
                "n": r["n"],
                "flag_rate": pred.loc[idx].mean(),
                "fpr": r["fpr"],
                "fnr": r["fnr"],
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("flag_rate", ascending=False).reset_index(drop=True)


def predict_primary(df: pd.DataFrame, mild_threshold: float | None = None) -> pd.Series:
    """Recompute the primary model's binary prediction from stored label scores.

    Uses the current thresholds in ``config``; ``mild_threshold`` temporarily
    overrides ``MILD_TOXIC_THRESHOLD`` (other tiers are unchanged).
    """
    value = config.MILD_TOXIC_THRESHOLD if mild_threshold is None else mild_threshold
    with mock.patch.object(config, "MILD_TOXIC_THRESHOLD", value):
        flagged = [
            severity.assess(row).tier != severity.NONE for row in df[LABELS].to_dict("records")
        ]
    return pd.Series(flagged, index=df.index)


def threshold_sensitivity(df: pd.DataFrame, grid=common.THRESHOLD_GRID) -> pd.DataFrame:
    """Overall rates for the primary model at each mild threshold in ``grid``."""
    rows = [{"mild_threshold": t, **rates(df["label_gold"], predict_primary(df, t))} for t in grid]
    return pd.DataFrame(rows)


def model_predictions(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Return binary predictions per model; the secondary model is kept separate."""
    preds = {common.PRIMARY: predict_primary(df)}
    if "pred_hateful_secondary" in df:
        preds[common.SECONDARY] = df["pred_hateful_secondary"].astype(bool)
    return preds


def _clean(value):
    """Make a value JSON-safe (NaN -> None, numpy scalars -> Python)."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def compute_all(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Return (metrics dict, per-functionality frame, threshold-sensitivity frame)."""
    metrics: dict = {
        "n": len(df),
        "n_hateful": int(df["label_gold"].sum()),
        "mapping": "binary: predicted hateful = severity tier above 'None'",
        "thresholds": {
            "mild_toxic": config.MILD_TOXIC_THRESHOLD,
            "insult": config.INSULT_THRESHOLD,
            "threat": config.THREAT_THRESHOLD,
            "hate": config.HATE_THRESHOLD,
            "severe_toxic": config.SEVERE_TOXIC_THRESHOLD,
            "secondary_hate": config.SECONDARY_HATE_THRESHOLD,
        },
        "models": {},
    }
    func_frames = []
    for model, pred in model_predictions(df).items():
        func = by_functionality(df, pred)
        func.insert(0, "model", model)
        func_frames.append(func)
        target = flag_rate_by_target(df, pred)
        metrics["models"][model] = {
            "overall": _clean(rates(df["label_gold"], pred)),
            "flag_rate_by_target": _clean(target.set_index("target_ident").to_dict("index"))
            if not target.empty
            else {},
        }
    return metrics, pd.concat(func_frames, ignore_index=True), threshold_sensitivity(df)


def main() -> int:
    if not common.PREDICTIONS_PATH.exists():
        print(f"{common.PREDICTIONS_PATH} not found; run `python -m evaluation.run_hatecheck` first.")
        return 1
    df = pd.read_csv(common.PREDICTIONS_PATH)
    metrics, func, thresholds = compute_all(df)
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    common.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    func.to_csv(common.BY_FUNCTIONALITY_PATH, index=False)
    thresholds.to_csv(common.THRESHOLD_PATH, index=False)
    for model, m in metrics["models"].items():
        o = m["overall"]
        print(
            f"{model}: n={o['n']} acc={o['accuracy']:.3f} P={o['precision']:.3f} "
            f"R={o['recall']:.3f} F1={o['f1']:.3f} FPR={o['fpr']:.3f} FNR={o['fnr']:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
