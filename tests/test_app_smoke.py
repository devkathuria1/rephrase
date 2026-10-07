"""Headless smoke tests of the Streamlit pages with models and LLM stubbed out."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import classifier, ocr, pipeline, severity, ui_helpers

ROOT = Path(__file__).resolve().parent.parent
SECRET = "sk-test-secret-123"


def fake_result(tier=severity.HARASSMENT, rephrased="Please say this kindly."):
    return {
        "text": "You are <b>dumb</b>",
        "severity": tier,
        "triggered": ["insult"],
        "scores": {"toxic": 0.9, "insult": 0.8, "threat": 0.01},
        "secondary_hate": 0.2,
        "word_importance": {
            "label": "insult",
            "words": [{"word": "You", "importance": 0.0}, {"word": "are", "importance": 0.0},
                      {"word": "<b>dumb</b>", "importance": 0.3}],
        },
        "llm_called": True,
        "rephrasing_enabled": True,
        "llm": {"rephrased": rephrased, "explanation": "It insults.",
                "intent_note": "You seem frustrated.", "confidence_caveat": ""},
        "warnings": ["Input truncated to 5 characters.", "Rephrasing disabled: no key"],
    }


@pytest.fixture(autouse=True)
def stub_models(monkeypatch):
    monkeypatch.setattr(classifier, "preload", lambda *a, **k: None)
    monkeypatch.setattr(ocr, "preload", lambda *a, **k: None)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", SECRET)


def run_app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    return at


def markdown_text(at):
    return " ".join(m.value for m in at.markdown)


def test_landing_page_has_hero_examples_footer_and_no_secret():
    at = run_app()
    page = markdown_text(at)
    assert "Rephrase" in page and "rp-step" in page and "not a moderator" in page
    assert len([b for b in at.button if b.key and b.key.startswith("ex_")]) == 4
    assert SECRET not in page + " ".join(str(w.value) for w in at.sidebar.markdown)
    assert any("groq" in m.value for m in at.sidebar.markdown)


def test_example_button_fills_text_box():
    at = run_app()
    at.button(key=f"ex_{ui_helpers.EXAMPLES[1][0]}").click().run()
    assert at.text_area(key="pasted_text").value == ui_helpers.EXAMPLES[1][1]


def test_empty_analyze_shows_error():
    at = run_app()
    at.button[-1].click().run()
    assert at.error and not at.exception


@pytest.mark.parametrize("rephrased,expected", [
    ("Please say this kindly.", "Rephrased"),
    ("", ui_helpers.NO_SAFE_REPHRASING),
])
def test_results_render_in_tabs_and_escape_user_text(monkeypatch, rephrased, expected):
    monkeypatch.setattr(pipeline, "analyze", lambda **kw: fake_result(rephrased=rephrased))
    at = run_app()
    at.text_area(key="pasted_text").set_value("You are <b>dumb</b>")
    at.button[-1].click().run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Result", "Highlights", "Raw scores"]
    page = markdown_text(at)
    assert expected in page and "Dumb" not in page
    assert "<b>dumb</b>" not in page and "&lt;b&gt;dumb&lt;/b&gt;" in page
    assert len(at.warning) == 1 and len(at.info) == 1  # grouped warnings


def test_limitations_page_renders_cards(monkeypatch):
    monkeypatch.setattr(classifier, "score", lambda texts: [{"toxic": 0.9, "insult": 0.1}] * len(texts))
    monkeypatch.setattr(classifier, "score_secondary", lambda texts: [0.1] * len(texts))
    at = AppTest.from_file(str(ROOT / "pages" / "1_Limitations.py"), default_timeout=30)
    at.run()
    assert not at.exception
    page = markdown_text(at)
    assert page.count('class="rp-example-head"') == 8
    assert "not a moderator" in page
    assert len(at.expander) == 8
