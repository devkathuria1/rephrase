"""Evaluation and Audit Dashboard: Displays empirical performance on the HateCheck benchmark."""

import json
from pathlib import Path
import pandas as pd
import streamlit as st
from src import styles, ui_helpers

st.set_page_config(page_title="Model Evaluation & Audit", page_icon="📊", layout="wide")
styles.inject()

st.title("📊 Model Evaluation & Empirical Audit")
st.markdown(
    "To move beyond qualitative testing, we benchmarked the models on the academic **HateCheck** "
    "test suite (Röttger et al., 2021, ACL). This page reports quantitative metrics, confusion matrices, "
    "and false-positive rates across demographic identities."
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

if not (RESULTS_DIR / "hatecheck_metrics.json").exists():
    st.warning("Benchmark results not found. Please run the evaluation pipeline first.")
    st.stop()

with open(RESULTS_DIR / "hatecheck_metrics.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)

# Top summary KPIs - access metrics["models"]["toxic_bert"]["overall"]
models_data = metrics.get("models", {})
primary_m = models_data.get("toxic_bert", {}).get("overall", {})
sec_m = models_data.get("secondary", {}).get("overall", {})

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "toxic-bert Accuracy",
        f"{primary_m.get('accuracy', 0.0):.1%}",
        help="Overall accuracy on HateCheck test cases",
    )
with col2:
    st.metric(
        "toxic-bert FPR",
        f"{primary_m.get('fpr', 0.0):.1%}",
        delta=f"-{primary_m.get('fpr', 0.0):.1%}",
        delta_color="inverse",
        help="False Positive Rate (benign sentences falsely flagged)",
    )
with col3:
    st.metric(
        "Secondary Model Accuracy",
        f"{sec_m.get('accuracy', 0.0):.1%}",
        delta=f"+{sec_m.get('accuracy', 0.0) - primary_m.get('accuracy', 0.0):.1%}",
        help="RoBERTa Dynabench accuracy",
    )
with col4:
    st.metric(
        "Secondary Model FPR",
        f"{sec_m.get('fpr', 0.0):.1%}",
        delta=f"-{sec_m.get('fpr', 0.0):.1%}",
        delta_color="normal",
        help="False Positive Rate on harmless sentences",
    )

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Confusion Matrices",
    "🎯 Demographic Bias Audit",
    "⚙️ Threshold Sensitivity",
    "🔍 Error Breakdown by Functionality",
])

with tab1:
    st.subheader("Model Confusion Matrices")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Primary Model: `unitary/toxic-bert`**")
        if (RESULTS_DIR / "confusion_matrix.png").exists():
            st.image(str(RESULTS_DIR / "confusion_matrix.png"), use_container_width=True)
        st.caption(
            f"Precision: {primary_m.get('precision', 0):.3f} | Recall: {primary_m.get('recall', 0):.3f} | F1: {primary_m.get('f1', 0):.3f}"
        )
    with c2:
        st.markdown("**Secondary Model: `facebook/roberta-hate-speech-dynabench-r4-target`**")
        if (RESULTS_DIR / "confusion_matrix_secondary.png").exists():
            st.image(str(RESULTS_DIR / "confusion_matrix_secondary.png"), use_container_width=True)
        st.caption(
            f"Precision: {sec_m.get('precision', 0):.3f} | Recall: {sec_m.get('recall', 0):.3f} | F1: {sec_m.get('f1', 0):.3f}"
        )

with tab2:
    st.subheader("Demographic & Identity Flagging Rates")
    st.markdown(
        "A critical ethical vulnerability in toxic comment classifiers is **demographic bias**: "
        "sentences containing identity terms (e.g. *gay*, *Muslim*, *black*, *transgender*) "
        "often trigger false positives even when used in positive or reclaimed contexts."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**toxic-bert Flag Rate by Target Group**")
        if (RESULTS_DIR / "flag_rate_by_target.png").exists():
            st.image(str(RESULTS_DIR / "flag_rate_by_target.png"), use_container_width=True)
    with col_b:
        st.markdown("**Secondary Model Flag Rate by Target Group**")
        if (RESULTS_DIR / "flag_rate_by_target_secondary.png").exists():
            st.image(str(RESULTS_DIR / "flag_rate_by_target_secondary.png"), use_container_width=True)

with tab3:
    st.subheader("Threshold Sensitivity Trade-Off")
    st.markdown(
        "Lowering the decision threshold catches more abuse (higher Recall) but falsely flags "
        "more innocent speakers (higher False Positive Rate)."
    )
    if (RESULTS_DIR / "threshold_sensitivity.csv").exists():
        sens_df = pd.read_csv(RESULTS_DIR / "threshold_sensitivity.csv")
        st.dataframe(sens_df, use_container_width=True)

with tab4:
    st.subheader("Detailed Errors by HateCheck Functionality")
    if (RESULTS_DIR / "hatecheck_by_functionality.csv").exists():
        func_df = pd.read_csv(RESULTS_DIR / "hatecheck_by_functionality.csv")
        st.dataframe(func_df, use_container_width=True)

st.markdown("---")
st.markdown(ui_helpers.footer_html(), unsafe_allow_html=True)

