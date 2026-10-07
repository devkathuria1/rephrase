"""Limitations page: live failure examples for the classifiers (no LLM, no user data)."""

import streamlit as st

from src import classifier, severity, ui_helpers

st.set_page_config(page_title="Where this model fails", page_icon="⚠️", layout="wide")


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


st.title("Where this model fails")
st.caption(
    "Content note: these examples mention insults, identity terms and hate speech "
    "in mild, non-graphic form. Slurs are replaced by placeholders such as [slur]."
)
st.write(
    "A **false positive** is harmless text that gets flagged; a **false negative** is "
    "harmful text that is missed. Both hurt people: false positives silence "
    "speakers (often those the system is meant to protect), and false negatives let "
    "abuse through. Lowering the threshold trades one for the other; it cannot "
    "remove both."
)
st.info(
    "These 8 sentences are hand-picked illustrations, **not statistics**. Some are "
    "handled correctly, and that is shown too. A larger, systematic evaluation on "
    "HateCheck (Röttger et al., 2021) comes later. Tiers below use toxic-bert plus "
    "the secondary classifier, with the app's default thresholds (untuned)."
)

rows = evaluate_examples()
wrong = sum(not r["correct"] for r in rows)
st.write(f"**In this run the system was wrong on {wrong} of {len(rows)} examples.**")

for row in rows:
    with st.container(border=True):
        st.subheader(f"{row['id']}. {row['category']}")
        st.markdown(f"> {row['text']}")
        cols = st.columns([1, 1, 1, 1])
        cols[0].markdown("**Tier**")
        cols[0].markdown(ui_helpers.badge_html(row["tier"]), unsafe_allow_html=True)
        cols[1].markdown("**Top scores**")
        cols[1].write(row["top_scores"])
        cols[2].markdown("**Expected**")
        cols[2].write(row["expected"])
        cols[3].markdown("**Result**")
        if row["correct"]:
            cols[3].success("✅ correct")
        else:
            cols[3].error("❌ wrong")
        st.markdown(f"**Why it can fail.** {row['why_it_fails']}")
        st.markdown(f"**Who is harmed.** {row['who_is_harmed']}")
