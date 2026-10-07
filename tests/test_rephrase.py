"""Unit tests; the classifier and LLM are mocked so no models or network are needed."""

import pytest

import config
from src import classifier, llm, pipeline, severity

ZERO = {label: 0.0 for label in classifier.LABELS}
GOOD_JSON = (
    '{"explanation": "e", "rephrased": "r", "intent_note": "i", "confidence_caveat": ""}'
)


def scores(**kw: float) -> dict[str, float]:
    return {**ZERO, **kw}


@pytest.mark.parametrize(
    "given, tier",
    [
        (scores(identity_hate=0.9), severity.HATE),
        (scores(severe_toxic=0.7, insult=0.9), severity.HATE),
        (scores(threat=0.8), severity.HARASSMENT),
        (scores(insult=0.9), severity.HARASSMENT),
        (scores(toxic=0.5), severity.MILD),
        (scores(obscene=0.45), severity.MILD),
        (scores(toxic=0.1, insult=0.2), severity.NONE),
    ],
)
def test_severity_mapping(given, tier):
    assert severity.assess(given).tier == tier


def test_secondary_hate_raises_tier():
    assert severity.assess(ZERO, secondary_hate=0.9).tier == severity.HATE
    assert severity.assess(ZERO, secondary_hate=0.1).tier == severity.NONE


def test_parse_valid_and_fenced_json():
    expected = {
        "explanation": "e",
        "rephrased": "r",
        "intent_note": "i",
        "confidence_caveat": "",
    }
    assert llm.parse_response(GOOD_JSON) == expected
    assert llm.parse_response(f"```json\n{GOOD_JSON}\n```") == expected


@pytest.mark.parametrize(
    "raw",
    ["not json", "{broken", "[1, 2]", '{"explanation": "e"}', GOOD_JSON.replace('"r"', '""')],
)
def test_parse_invalid_json_raises(raw):
    with pytest.raises(ValueError):
        llm.parse_response(raw)


@pytest.fixture
def fake_classifier(monkeypatch):
    """Score text as toxic in proportion to occurrences of the word 'idiot'."""

    def fake_score(texts):
        return [scores(insult=min(1.0, 0.9 * t.lower().split().count("idiot"))) for t in texts]

    monkeypatch.setattr(classifier, "score", fake_score)
    monkeypatch.setattr(classifier, "score_secondary", lambda texts: [0.0] * len(texts))


def test_pipeline_with_mocked_llm(fake_classifier, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    calls = []

    def fake_rephrase(text, context, tier, sc, top_words):
        calls.append((tier, top_words))
        return {"ok": True, "explanation": "e", "rephrased": "r", "intent_note": "i",
                "confidence_caveat": "c"}

    monkeypatch.setattr(llm, "rephrase", fake_rephrase)
    result = pipeline.analyze(text="you are an idiot")
    assert result["severity"] == severity.HARASSMENT
    assert result["llm_called"] and result["llm"]["rephrased"] == "r"
    assert calls == [(severity.HARASSMENT, ["idiot"])]
    top = max(result["word_importance"]["words"], key=lambda w: w["importance"])
    assert top["word"] == "idiot"


def test_llm_failure_falls_back(fake_classifier, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(llm, "rephrase", lambda *a: {"ok": False, "error": "boom"})
    result = pipeline.analyze(text="you are an idiot")
    assert result["llm"] is None and "boom" in result["warnings"]
    assert result["severity"] == severity.HARASSMENT


def test_no_key_disables_rephrasing(fake_classifier, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = pipeline.analyze(text="you are an idiot")
    assert not result["rephrasing_enabled"] and not result["llm_called"]
    assert result["word_importance"] is not None
    assert any("disabled" in w for w in result["warnings"])


def test_nothing_flagged_skips_llm(fake_classifier, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(llm, "rephrase", lambda *a: pytest.fail("LLM must not be called"))
    result = pipeline.analyze(text="have a lovely day, my friend")
    assert result["severity"] == severity.NONE and not result["llm_called"]


@pytest.mark.parametrize("empty", [None, "", "   \n"])
def test_empty_input(empty):
    result = pipeline.analyze(text=empty)
    assert result["severity"] == severity.NONE
    assert any("Empty input" in w for w in result["warnings"])


def test_image_without_text(monkeypatch):
    monkeypatch.setattr(pipeline.ocr, "extract_text", lambda image: "")
    result = pipeline.analyze(image=object())
    assert any("No text was found" in w for w in result["warnings"])


def test_long_input_truncated(fake_classifier, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = pipeline.analyze(text="the cat sat on the mat " * 500)
    assert len(result["text"]) == config.MAX_INPUT_CHARS
    assert any("truncated" in w for w in result["warnings"])


def test_non_english_warning(fake_classifier):
    result = pipeline.analyze(text="これは日本語の文章です")
    assert any("not be English" in w for w in result["warnings"])
