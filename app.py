"""Rephrase: Streamlit front end. Nothing the user enters is stored or logged."""

import os

import pandas as pd
import streamlit as st

from src import classifier, ocr, pipeline, severity, ui_helpers

st.set_page_config(page_title="Rephrase", page_icon="💬", layout="wide")


def _load_streamlit_secrets() -> None:
    """Expose Streamlit Cloud secrets as env vars (local runs use .env instead)."""
    try:
        for key in ("LLM_PROVIDER", "LLM_MODEL", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "OLLAMA_URL"):
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = str(st.secrets[key])
    except Exception:  # no secrets file locally
        pass


@st.cache_resource(show_spinner="Loading models (first run only)...")
def load_models(use_secondary: bool) -> bool:
    """Load classifiers and the OCR reader once per server process."""
    classifier.preload(use_secondary)
    ocr.preload()
    return True


@st.cache_resource(show_spinner=False)
def _init_env() -> bool:
    _load_streamlit_secrets()
    return True


def sidebar() -> bool:
    """Render the sidebar and return whether the secondary classifier is on."""
    with st.sidebar:
        st.header("Settings")
        use_secondary = st.toggle(
            "Secondary hate-speech classifier",
            help="Adds facebook/roberta-hate-speech-dynabench-r4-target.",
        )
        provider, enabled, reason = ui_helpers.provider_status()
        if enabled:
            st.success(f"LLM provider: {provider} (rephrasing enabled)")
        else:
            st.warning(f"LLM provider: {provider} (rephrasing disabled: {reason})")
        st.caption("Accuracy drops for non-English text; the models are English-only.")
        st.divider()
        st.subheader("Please read")
        st.info(
            "Rephrase is an **assistive aid, not a moderator**. The models have "
            "known biases and make mistakes, so human judgment is required.\n\n"
            "This app stores and logs no user text. If rephrasing is enabled, "
            "your text is sent to a third-party LLM provider, whose own privacy "
            "policy applies."
        )
    return use_secondary


def input_section() -> tuple[str, str]:
    """Collect text (pasted, or OCR'd and editable) and optional context."""
    mode = st.radio("Input", ["Paste text", "Upload screenshot"], horizontal=True)
    if mode == "Paste text":
        text = st.text_area("Text to check", height=150, key="pasted_text")
    else:
        upload = st.file_uploader("Screenshot (PNG or JPG)", type=["png", "jpg", "jpeg"])
        if upload is None:
            st.session_state.pop("ocr_file_id", None)
            st.session_state.pop("ocr_text", None)
        elif st.session_state.get("ocr_file_id") != upload.file_id:
            with st.spinner("Reading text from the image..."):
                st.session_state["ocr_text"] = ocr.extract_text(upload.getvalue())
            st.session_state["ocr_file_id"] = upload.file_id
        text = ""
        if upload is not None:
            if not st.session_state.get("ocr_text"):
                st.warning("No text was found in the image. You can type the text below instead.")
            text = st.text_area(
                "Extracted text (edit if the OCR made mistakes)", height=150, key="ocr_text"
            )
    context = st.text_input(
        "Context (optional)", placeholder='e.g. "this was a joke between friends"'
    )
    return text, context


def render_result(result: dict) -> None:
    """Render the analysis result in two columns."""
    for warning in result["warnings"]:
        st.warning(warning)
    tier = result["severity"]
    left, right = st.columns(2)

    with left:
        st.markdown(ui_helpers.badge_html(tier), unsafe_allow_html=True)
        if not result["text"]:
            return
        if tier == severity.NONE:
            st.success("Nothing was flagged.")
            st.caption(
                "This is not a guarantee that the text is harmless: the models miss "
                "sarcasm, implicit hate and context. See the Limitations page."
            )
        st.markdown("**Text with word-level highlights**")
        st.markdown(
            f'<div style="line-height:2;">{ui_helpers.highlight_html(result["text"], result["word_importance"])}</div>',
            unsafe_allow_html=True,
        )
        if result["word_importance"]:
            st.caption(
                "Shading shows how much the "
                f'"{result["word_importance"]["label"]}" score drops when a word is '
                "removed, relative to the strongest word in this text. Faint or no "
                "shading means no single word dominates."
            )
        with st.expander("Raw scores"):
            table = pd.DataFrame(result["scores"], index=["probability"]).T
            st.dataframe(table)
            if result["secondary_hate"] is not None:
                st.write(f"Secondary classifier, hate probability: {result['secondary_hate']:.4f}")

    with right:
        llm_out = result["llm"]
        if llm_out:
            st.markdown("**Before / after**")
            before, after = st.columns(2)
            before.markdown("*Original*")
            before.write(result["text"])
            after.markdown("*Rephrased*")
            after.success(llm_out["rephrased"])
            st.markdown("**Why this may be harmful**")
            st.write(llm_out["explanation"])
            st.markdown("**What you seem to mean**")
            st.write(llm_out["intent_note"])
            if llm_out["confidence_caveat"]:
                st.markdown("**Caveat**")
                st.write(llm_out["confidence_caveat"])
            st.caption(
                "AI-generated suggestions can soften wording while keeping a hurtful "
                "claim, or change your meaning. Check before using."
            )
        elif tier != severity.NONE:
            st.info("No rephrasing is available for this text (see the warnings above).")


def main() -> None:
    _init_env()
    st.title("Rephrase")
    st.write(
        "Check text or a screenshot for toxicity, see why it was flagged, "
        "and get a respectful way to say the same thing."
    )
    use_secondary = sidebar()
    load_models(use_secondary)
    text, context = input_section()
    if st.button("Analyze", type="primary"):
        if not text.strip():
            st.session_state.pop("result", None)
            st.error("Please enter some text (or upload a screenshot that contains text).")
        else:
            with st.spinner("Analysing..."):
                st.session_state["result"] = pipeline.analyze(
                    text=text, context=context, use_secondary=use_secondary
                )
    if "result" in st.session_state:
        render_result(st.session_state["result"])


main()
