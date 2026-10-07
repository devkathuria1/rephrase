"""Limitations page: live failure examples for the classifiers (no LLM, no user data)."""

import streamlit as st

from src import classifier, severity, styles, ui_helpers

st.set_page_config(page_title="Where this model fails", page_icon="⚠️", layout="centered")


@st.cache_data(show_spinner="Running the classifiers on each example...")
def evaluate_examples() -> list[dict]:
    """Run every example through both classifiers; results contain no user data."""
    examples = ui_helpers.load_failure_examples()
    texts = [e["text"] for e in examples]
    primary = classifier.score(texts)
    secondary = classifier.score_secondary(texts)
    rows = []
    for example, scores, hate in zip(examples, primary, secondary):
        sev = severity.assess(scores, hate)
        flagged = ui_helpers.is_flagged(sev.tier)
        rows.append(
            {
                **example,
                "tier": sev.tier,
                "top_scores": ui_helpers.top_scores(scores, hate),
                "correct": flagged == (example["expected"] == "should flag"),
            }
        )
    return rows


styles.inject()
st.title("⚠️ Where this model fails")
st.info(
    "Content note: these examples mention insults, identity terms and hate speech "
    "in mild, non-graphic form. Slurs are replaced by placeholders such as [slur].",
    icon="📝",
)
st.write(
    "A **false positive** is harmless text that gets flagged; a **false negative** is "
    "harmful text that is missed. Both hurt people: false positives silence "
    "speakers (often those the system is meant to protect), and false negatives let "
    "abuse through. Lowering the threshold trades one for the other; it cannot "
    "remove both."
)
st.caption(
    "These 8 sentences are hand-picked illustrations, **not statistics**. Some are "
    "handled correctly, and that is shown too. A larger, systematic evaluation on "
    "HateCheck (Röttger et al., 2021) comes later. Tiers below use toxic-bert plus "
    "the secondary classifier, with the app's default thresholds (untuned)."
)

rows = evaluate_examples()
wrong = sum(not r["correct"] for r in rows)
st.markdown(f"##### In this run the system was wrong on {wrong} of {len(rows)} examples.")

for row in rows:
    with st.container(border=True):
        st.markdown(ui_helpers.example_card_html(row, row["correct"]), unsafe_allow_html=True)
        with st.expander("Why it can fail, and who is harmed"):
            st.markdown(f"**Why it can fail.** {row['why_it_fails']}")
            st.markdown(f"**Who is harmed.** {row['who_is_harmed']}")

st.markdown(ui_helpers.footer_html(), unsafe_allow_html=True)
