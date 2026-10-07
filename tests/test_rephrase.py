"""Unit tests; the classifier and LLM are mocked so no models or network are needed."""

import pytest

import config
from src import classifier, llm, pipeline, severity

ZERO = {label: 0.0 for label in classifier.LABELS}
GOOD_JSON = (
    '{"explanation": "e", "rephrased": "r", "intent_note": "i", "confidence_caveat": ""}'
)


def enable_llm(monkeypatch, provider="groq"):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def disable_llm(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


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
    enable_llm(monkeypatch)
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
    enable_llm(monkeypatch)
    monkeypatch.setattr(llm, "rephrase", lambda *a: {"ok": False, "error": "boom"})
    result = pipeline.analyze(text="you are an idiot")
    assert result["llm"] is None and "boom" in result["warnings"]
    assert result["severity"] == severity.HARASSMENT


def test_no_key_disables_rephrasing(fake_classifier, monkeypatch):
    disable_llm(monkeypatch)
    result = pipeline.analyze(text="you are an idiot")
    assert not result["rephrasing_enabled"] and not result["llm_called"]
    assert result["word_importance"] is not None
    assert any("disabled" in w for w in result["warnings"])


def test_nothing_flagged_skips_llm(fake_classifier, monkeypatch):
    enable_llm(monkeypatch)
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
    disable_llm(monkeypatch)
    result = pipeline.analyze(text="the cat sat on the mat " * 500)
    assert len(result["text"]) == config.MAX_INPUT_CHARS
    assert any("truncated" in w for w in result["warnings"])


def test_non_english_warning(fake_classifier):
    result = pipeline.analyze(text="これは日本語の文章です")
    assert any("not be English" in w for w in result["warnings"])


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def call_llm():
    return llm.rephrase("you idiot", "", severity.HARASSMENT, {"insult": 0.9}, ["idiot"])


def test_groq_success_sends_json_mode(monkeypatch):
    enable_llm(monkeypatch, "groq")
    sent = {}

    def fake_post(url, json, headers, timeout):
        sent.update(url=url, json=json, headers=headers)
        return FakeResponse(body={"choices": [{"message": {"content": GOOD_JSON}}]})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    reply = call_llm()
    assert reply["ok"] and reply["rephrased"] == "r"
    assert sent["url"] == config.GROQ_URL
    assert sent["json"]["response_format"] == {"type": "json_object"}
    assert sent["headers"]["Authorization"] == "Bearer test-key"


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(429, {}),
        FakeResponse(500, {}),
        FakeResponse(200, None),
        FakeResponse(200, {"choices": []}),
        FakeResponse(200, {"choices": [{"message": {"content": "not json"}}]}),
    ],
)
def test_groq_failures_fall_back_without_leaking_key(monkeypatch, response):
    enable_llm(monkeypatch, "groq")
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: response)
    reply = call_llm()
    assert not reply["ok"] and "disabled" in reply["error"]
    assert "test-key" not in reply["error"]


def test_groq_network_error_falls_back(monkeypatch):
    enable_llm(monkeypatch, "groq")

    def boom(*a, **k):
        raise llm.requests.ConnectionError("secret-detail")

    monkeypatch.setattr(llm.requests, "post", boom)
    reply = call_llm()
    assert not reply["ok"] and "network error" in reply["error"]
    assert "secret-detail" not in reply["error"]


def test_ollama_success_and_unreachable(monkeypatch):
    enable_llm(monkeypatch, "ollama")
    monkeypatch.delenv("GROQ_API_KEY")
    sent = {}

    def fake_post(url, json, headers, timeout):
        sent.update(url=url, json=json)
        return FakeResponse(body={"message": {"content": GOOD_JSON}})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    assert call_llm()["ok"]
    assert sent["url"].endswith("/api/chat") and sent["json"]["format"] == "json"

    def refuse(*a, **k):
        raise llm.requests.ConnectionError()

    monkeypatch.setattr(llm.requests, "post", refuse)
    assert not call_llm()["ok"]


def test_anthropic_provider(monkeypatch):
    import anthropic

    enable_llm(monkeypatch, "anthropic")

    class Block:
        type, text = "text", GOOD_JSON

    class FakeClient:
        def __init__(self, **kw):
            self.messages = self

        def create(self, **kw):
            assert kw["system"] and kw["model"]
            return type("R", (), {"content": [Block()]})()

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    assert call_llm()["ok"]


def test_none_and_unknown_provider_are_disabled(monkeypatch):
    for value in ("none", "bogus"):
        monkeypatch.setenv("LLM_PROVIDER", value)
        assert not llm.is_enabled()
        assert not call_llm()["ok"]


def test_default_models():
    assert config.DEFAULT_MODELS["groq"] == "openai/gpt-oss-120b"
