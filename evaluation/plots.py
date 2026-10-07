"""PNG charts for the HateCheck evaluation.

Usage: python -m evaluation.plots
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from evaluation import common, metrics  # noqa: E402


def plot_error_by_functionality(func: pd.DataFrame, path: Path) -> None:
    """Horizontal bars of error rate per functionality, worst at the top."""
    func = func.sort_values("error_rate")
    colors = ["#d32f2f" if g == "hateful" else "#1976d2" for g in func["gold"]]
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(func) + 1.5))
    ax.barh(func["functionality"], func["error_rate"], color=colors)
    ax.set_xlabel("error rate")
    ax.set_xlim(0, 1)
    ax.set_title(
        "Error rate per functionality\n"
        "(red: hateful gold, missed; blue: non-hateful gold, wrongly flagged)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(df: pd.DataFrame, pred: pd.Series, path: Path) -> None:
    """Confusion matrix with counts (rows gold, columns predicted)."""
    c = metrics.confusion(df["label_gold"], pred)
    grid = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(grid, cmap="Blues")
    for (i, j), v in np.ndenumerate(grid):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > grid.max() * 0.6 else "black")
    ax.set_xticks([0, 1], ["not flagged", "flagged"])
    ax.set_yticks([0, 1], ["non-hateful", "hateful"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_target_flag_rate(target: pd.DataFrame, path: Path) -> None:
    """Grouped bars: overall flag rate and false positive rate per target group."""
    x = np.arange(len(target))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, target["flag_rate"], 0.4, label="flag rate (all cases)", color="#455a64")
    ax.bar(x + 0.2, target["fpr"].fillna(0), 0.4,
           label="false positive rate (non-hateful cases)", color="#e8751a")
    ax.set_xticks(x, target["target_ident"], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate")
    ax.set_title("Flag rate per target group")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_all(df: pd.DataFrame, out_dir: Path = common.RESULTS_DIR) -> list[Path]:
    """Write all charts for every model present; return the file paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for model, pred in metrics.model_predictions(df).items():
        suffix = "" if model == common.PRIMARY else f"_{model}"
        func = metrics.by_functionality(df, pred)
        path = out_dir / f"error_by_functionality{suffix}.png"
        plot_error_by_functionality(func, path)
        written.append(path)
        path = out_dir / f"confusion_matrix{suffix}.png"
        plot_confusion(df, pred, path)
        written.append(path)
        target = metrics.flag_rate_by_target(df, pred)
        if not target.empty:
            path = out_dir / f"flag_rate_by_target{suffix}.png"
            plot_target_flag_rate(target, path)
            written.append(path)
    return written


def main() -> int:
    if not common.PREDICTIONS_PATH.exists():
        print(f"{common.PREDICTIONS_PATH} not found; run `python -m evaluation.run_hatecheck` first.")
        return 1
    for path in make_all(pd.read_csv(common.PREDICTIONS_PATH)):
        print("Wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
