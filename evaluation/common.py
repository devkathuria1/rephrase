"""Shared paths and constants for the HateCheck evaluation."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "hatecheck.csv"
RESULTS_DIR = ROOT / "results"
PREDICTIONS_PATH = RESULTS_DIR / "hatecheck_predictions.csv"
METRICS_PATH = RESULTS_DIR / "hatecheck_metrics.json"
BY_FUNCTIONALITY_PATH = RESULTS_DIR / "hatecheck_by_functionality.csv"
THRESHOLD_PATH = RESULTS_DIR / "threshold_sensitivity.csv"
WORST_CASES_PATH = RESULTS_DIR / "worst_cases.md"

PRIMARY = "toxic_bert"
SECONDARY = "secondary"
THRESHOLD_GRID = (0.25, 0.4, 0.55)
