"""Tests for the HateCheck evaluation workflow."""

import math
import re

import pandas as pd
import pytest

import config
from evaluation import common, examples, metrics, plots, run_hatecheck
from src import classifier


def table() -> pd.DataFrame:
    """Hand-made predictions: 4 hateful (3 caught), 4 non-hateful (1 wrongly flagged)."""
    zero = {label: 0.0 for label in classifier.LABELS}
    rows = [
        # functionality, gold hateful, toxic score, target
        ("derog", True, 0.9, "women"),
        ("derog", True, 0.9, "women"),
        ("derog", True, 0.9, "gay people"),
        ("derog", True, 0.1, "gay people"),  # missed
        ("ident", False, 0.1, "women"),
        ("ident", False, 0.1, "gay people"),
        ("ident", False, 0.1, "gay people"),
        ("ident", False, 0.9, "women"),  # wrongly flagged
    ]
    return pd.DataFrame(
        [
            {"text": f"t{i}", "functionality": f, "label_gold": g, "target_ident": t,
             **{**zero, "toxic": s}, "tier": "x"}
            for i, (f, g, s, t) in enumerate(rows)
        ]
    )


def test_confusion_and_rates():
    df = table()
    pred = metrics.predict_primary(df)
    assert metrics.confusion(df["label_gold"], pred) == {"tp": 3, "fp": 1, "tn": 3, "fn": 1}
    r = metrics.rates(df["label_gold"], pred)
    assert r["accuracy"] == pytest.approx(0.75)
    assert r["precision"] == pytest.approx(0.75)
    assert r["recall"] == pytest.approx(0.75)
    assert r["f1"] == pytest.approx(0.75)
    assert r["fpr"] == pytest.approx(0.25)
    assert r["fnr"] == pytest.approx(0.25)


def test_rates_undefined_values_are_nan():
    r = metrics.rates(pd.Series([False, False]), pd.Series([False, False]))
    assert math.isnan(r["precision"]) and math.isnan(r["recall"]) and math.isnan(r["f1"])
    assert r["fpr"] == 0.0 and math.isnan(r["fnr"])


def test_by_functionality_and_target():
    df = table()
    pred = metrics.predict_primary(df)
    func = metrics.by_functionality(df, pred).set_index("functionality")
    assert func.loc["derog", "error_rate"] == pytest.approx(0.25)
    assert func.loc["derog", "gold"] == "hateful" and func.loc["ident", "gold"] == "non-hateful"
    assert func.loc["ident", "fpr"] == pytest.approx(0.25)
    target = metrics.flag_rate_by_target(df, pred).set_index("target_ident")
    assert target.loc["women", "flag_rate"] == pytest.approx(0.75)
    assert target.loc["gay people", "flag_rate"] == pytest.approx(0.25)
    assert metrics.flag_rate_by_target(df.drop(columns="target_ident"), pred).empty


def test_threshold_sensitivity_trades_fp_for_fn():
    df = table()
    df.loc[df["text"] == "t3", "toxic"] = 0.3  # hateful case between 0.25 and 0.4
    df.loc[df["text"] == "t4", "toxic"] = 0.3  # non-hateful case between 0.25 and 0.4
    sens = metrics.threshold_sensitivity(df).set_index("mild_threshold")
    assert sens.loc[0.25, "fn"] < sens.loc[0.4, "fn"]
    assert sens.loc[0.25, "fp"] > sens.loc[0.4, "fp"]
    assert config.MILD_TOXIC_THRESHOLD == 0.4  # temporary override was restored


def test_main_primary_matches_threshold_row():
    df = table()
    all_metrics, func, sens = metrics.compute_all(df)
    row = sens.set_index("mild_threshold").loc[config.MILD_TOXIC_THRESHOLD]
    assert all_metrics["models"]["toxic_bert"]["overall"]["accuracy"] == pytest.approx(row["accuracy"])
    assert "secondary" not in all_metrics["models"]


def test_secondary_reported_separately():
    df = table()
    df["pred_hateful_secondary"] = df["label_gold"]  # perfect secondary
    all_metrics, func, _ = metrics.compute_all(df)
    assert all_metrics["models"]["secondary"]["overall"]["accuracy"] == 1.0
    assert all_metrics["models"]["toxic_bert"]["overall"]["accuracy"] == pytest.approx(0.75)
    assert set(func["model"]) == {"toxic_bert", "secondary"}


def test_column_detection():
    cols = ["Functionality", "case_id", "Test_Case", "label_gold", "target_ident"]
    assert run_hatecheck.detect_columns(cols) == {
        "text": "Test_Case", "label_gold": "label_gold",
        "functionality": "Functionality", "target_ident": "target_ident",
    }
    assert run_hatecheck.detect_columns(["text", "label", "functionality"])["target_ident"] is None
    with pytest.raises(ValueError):
        run_hatecheck.detect_columns(["text", "functionality"])


def write_csv(path, n=40):
    rows = [
        {"functionality": "f_h" if i % 2 else "f_nh", "case_id": i,
         "test_case": "you are a stupid idiot" if i % 2 else "have a nice day friend",
         "label_gold": "hateful" if i % 2 else "non-hateful",
         "target_ident": "women" if i % 4 < 2 else "gay people",
         "focus_words": "", "case_templ": "x"}
        for i in range(n)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_missing_file_prints_instructions(tmp_path, capsys):
    code = run_hatecheck.main(["--data", str(tmp_path / "nope.csv")])
    assert code == 1 and "test_suite_cases.csv" in capsys.readouterr().out


def test_smoke_limit_20(tmp_path, monkeypatch):
    data, out = tmp_path / "hc.csv", tmp_path / "pred.csv"
    write_csv(data)
    assert run_hatecheck.main(["--data", str(data), "--out", str(out), "--limit", "20"]) == 0
    df = pd.read_csv(out)
    assert len(df) == 20 and {"toxic", "tier", "pred_hateful"} <= set(df.columns)
    assert "secondary_hate" not in df  # no --secondary
    all_metrics, func, sens = metrics.compute_all(df)
    assert all_metrics["n"] == 20 and len(sens) == 3
    assert len(plots.make_all(df, tmp_path)) == 3
    assert all((tmp_path / name).exists() for name in
               ("error_by_functionality.png", "confusion_matrix.png", "flag_rate_by_target.png"))


def test_mask_text():
    assert examples.mask_text("That is such bullshit.") == "That is such [explicit]."
    assert examples.mask_text("What is it with you fucking women?") == "What is it with you [explicit] women?"
    assert examples.mask_text("x", "That's for a [SLUR_S] like you, [IDENTITY_P].", "women") == (
        "That's for a [slur] like you, women."
    )
    assert examples.mask_text("Us queer folk are people.") == "Us [slur] folk are people."
    assert examples.mask_text("Plain sentence.") == "Plain sentence."


@pytest.mark.skipif(not common.DATA_PATH.exists(), reason="HateCheck not downloaded")
def test_masking_covers_every_hatecheck_row():
    raw = pd.read_csv(common.DATA_PATH)
    terms = set()
    for prefix in ("slur", "profanity"):
        for words in raw[raw["functionality"].str.startswith(prefix)]["focus_words"].dropna():
            terms.update(w for w in re.findall(r"[a-z]+", re.sub(r"\[[^\]]*\]", "", words.lower())) if len(w) > 2)
    terms -= {"the", "you", "get", "out", "off", "him", "her", "them", "his", "myself", "himself",
              "herself", "themselves", "scumbag", "scumbags", "pissing", "anyone", "got", "shut", "shutting", "piss"}
    leaks = re.compile(r"\b(?:" + "|".join(sorted(terms)) + r")\b", re.I)
    for _, row in raw.iterrows():
        masked = examples.mask_text(str(row["test_case"]), row["case_templ"], row["target_ident"], row["focus_words"])
        assert not leaks.search(masked), masked
