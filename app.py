"""Rephrase: Streamlit front end. Nothing the user enters is stored or logged."""

import os

import pandas as pd
import streamlit as st

from src import classifier, ocr, pipeline, severity, styles, ui_helpers

st.set_page_config(page_title="Rephrase", page_icon="💬", layout="centered")

PASTE, SCREENSHOT = "Paste text", "Upload screenshot"


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


def sidebar() -> tuple[bool, bool]:
    """Render the sidebar and return settings."""
    with st.sidebar:
        st.header("⚙️ Settings")
        use_secondary = st.toggle(
            "Secondary hate-speech classifier",
            value=True,
            help="Adds facebook/roberta-hate-speech-dynabench-r4-target.",
        )
        always_rephrase = st.toggle(
            "Always rephrase (even if harmless)",
            value=True,
            help="Calls the LLM to suggest polished phrasing even when no toxicity is flagged.",
        )
        provider, enabled, reason = ui_helpers.provider_status()
        st.markdown(
            "LLM provider " + ui_helpers.provider_pill_html(provider, enabled),
            unsafe_allow_html=True,
        )
        if not enabled:
            st.caption(f"Rephrasing disabled: {reason}")
        with st.expander("About the classifiers"):
            st.caption(ui_helpers.SECONDARY_NOTE)
            st.caption("The models are English-only; accuracy drops for other languages.")
        with st.expander("ℹ️ About"):
            st.caption(
                "Rephrase checks text for toxicity, shows which words drove the result, "
                "and suggests a respectful rewording. Built for an AI & Society course."
            )
        with st.expander("🔒 Ethics & privacy", expanded=True):
            st.caption(
                "Rephrase is an **assistive aid, not a moderator**. The models have "
                "known biases and make mistakes, so human judgment is required."
            )
            st.caption(
                "This app stores and logs no user text. If rephrasing is enabled, "
                "your text is sent to a third-party LLM provider, whose own privacy "
                "policy applies."
            )
    return use_secondary, always_rephrase


def _use_example(text: str) -> None:
    """Button callback: switch to the paste view and fill the text box."""
    st.session_state["input_mode"] = PASTE
    st.session_state["pasted_text"] = text


def input_section() -> tuple[str, str]:
    """Collect text (pasted, or OCR'd and editable) and optional context."""
    options = [PASTE, SCREENSHOT]
    if hasattr(st, "segmented_control"):
        mode = st.segmented_control(
            "Input", options, default=PASTE, key="input_mode", label_visibility="collapsed"
        )
    else:  # older Streamlit
        mode = st.radio("Input", options, horizontal=True, key="input_mode")
    if mode == SCREENSHOT:
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
    else:
        st.caption("Try an example")
        cols = st.columns(len(ui_helpers.EXAMPLES))
        for col, (label, example) in zip(cols, ui_helpers.EXAMPLES):
            col.button(
                label,
                key=f"ex_{label}",
                on_click=_use_example,
                args=(example,),
                use_container_width=True,
            )
        text = st.text_area("Text to check", height=150, key="pasted_text")
    context = st.text_input(
        "Context (optional)", placeholder='e.g. "this was a joke between friends"'
    )
    return text, context


def render_warnings(warnings: list[str]) -> None:
    """Compact, grouped warnings: real problems first, then expected notices."""
    groups = ui_helpers.classify_warnings(warnings)
    for message in groups["warning"]:
        st.warning(message, icon="⚠️")
    for message in groups["info"]:
        st.info(message, icon="ℹ️")


def _result_tab(result: dict) -> None:
    tier = result["severity"]
    st.markdown(ui_helpers.severity_card_html(tier), unsafe_allow_html=True)
    if result["scores"]:
        st.markdown("##### Label scores")
        st.markdown(ui_helpers.score_bars_html(result["scores"]), unsafe_allow_html=True)
    llm_out = result["llm"]
    if tier == severity.NONE:
        st.success("Nothing was flagged.", icon="✅")
        st.caption(
            "This is not a guarantee that the text is harmless: the models miss "
            "sarcasm, implicit hate and context. See the Limitations page."
        )

    if llm_out:
        st.markdown("##### Before / after")
        st.markdown(
            ui_helpers.compare_cards_html(
                result["text"], llm_out["rephrased"], llm_out["explanation"]
            ),
            unsafe_allow_html=True,
        )
        left, right = st.columns(2)
        if llm_out["rephrased"]:
            left.markdown(
                ui_helpers.text_card_html("Why this may be harmful / Tone Note", llm_out["explanation"], "💡"),
                unsafe_allow_html=True,
            )
        else:
            left.markdown(
                ui_helpers.text_card_html(
                    "Why no rephrasing", ui_helpers.NO_REPHRASING_REASON, "💡"
                ),
                unsafe_allow_html=True,
            )
        right.markdown(
            ui_helpers.text_card_html("What you seem to mean", llm_out["intent_note"], "🧭"),
            unsafe_allow_html=True,
        )
        if llm_out["confidence_caveat"]:
            st.caption(f"Caveat: {llm_out['confidence_caveat']}")
        st.caption(
            "AI-generated suggestions can soften wording while keeping a hurtful "
            "claim, or change your meaning. Check before using."
        )
    elif tier != severity.NONE:
        st.info("No rephrasing is available for this text (see the notes above).", icon="ℹ️")


def _highlights_tab(result: dict) -> None:
    st.markdown(ui_helpers.legend_html(), unsafe_allow_html=True)
    st.markdown(
        '<div class="rp-card rp-text">'
        f'{ui_helpers.highlight_html(result["text"], result["word_importance"])}</div>',
        unsafe_allow_html=True,
    )
    if result["word_importance"]:
        st.caption(
            "Shading shows how much the "
            f'"{result["word_importance"]["label"]}" score drops when a word is '
            "removed, relative to the strongest word in this text. Faint or no "
            "shading means no single word dominates."
        )
    else:
        st.caption("No word-level explanation: nothing was flagged.")


def _raw_scores_tab(result: dict) -> None:
    primary, secondary = st.columns(2)
    with primary:
        st.markdown("**Primary: toxic-bert**")
        st.dataframe(
            pd.DataFrame(result["scores"], index=["probability"]).T, use_container_width=True
        )
    with secondary:
        st.markdown("**Secondary: hate-speech model**")
        if result["secondary_hate"] is None:
            st.caption("Not run (turned off in the sidebar).")
        else:
            st.dataframe(
                pd.DataFrame({"probability": {"hate": result["secondary_hate"]}}),
                use_container_width=True,
            )


def render_result(result: dict) -> None:
    """Render the analysis result in Result / Highlights / Raw scores tabs."""
    render_warnings(result["warnings"])
    if not result["text"]:
        return
    result_tab, highlights_tab, raw_tab = st.tabs(["Result", "Highlights", "Raw scores"])
    with result_tab:
        _result_tab(result)
    with highlights_tab:
        _highlights_tab(result)
    with raw_tab:
        _raw_scores_tab(result)


def main() -> None:
    _init_env()
    styles.inject()
    st.markdown(ui_helpers.hero_html(), unsafe_allow_html=True)
    use_secondary, always_rephrase = sidebar()
    load_models(use_secondary)
    text, context = input_section()
    if st.button("Analyze", type="primary", use_container_width=True):
        if not text.strip():
            st.session_state.pop("result", None)
            st.error("Please enter some text (or upload a screenshot that contains text).")
        else:
            with st.spinner("Analysing..."):
                st.session_state["result"] = pipeline.analyze(
                    text=text, context=context, use_secondary=use_secondary, force_rephrase=always_rephrase
                )
    if "result" in st.session_state:
        render_result(st.session_state["result"])
    st.markdown(ui_helpers.footer_html(), unsafe_allow_html=True)


main()
