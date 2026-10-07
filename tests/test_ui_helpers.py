"""Tests for the UI helpers and the failure examples file."""

import json

import pytest

from src import severity, ui_helpers


def test_failure_examples_load_and_validate():
    examples = ui_helpers.load_failure_examples()
    assert len(examples) == 8
    assert [e["id"] for e in examples] == list(range(1, 9))
    assert any("vermin" in e["text"] for e in examples)


def write(tmp_path, data):
    path = tmp_path / "ex.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


GOOD = {
    "id": 1, "category": "c", "text": "t", "expected": "should flag",
    "why_it_fails": "w", "who_is_harmed": "h",
}


def test_validation_accepts_good_entry(tmp_path):
    assert ui_helpers.load_failure_examples(write(tmp_path, [GOOD])) == [GOOD]


@pytest.mark.parametrize(
    "bad",
    [
        [],
        {"not": "a list"},
        [{**GOOD, "text": ""}],
        [{**GOOD, "expected": "maybe"}],
        [GOOD, GOOD],
    ],
)
def test_validation_rejects_bad_files(tmp_path, bad):
    with pytest.raises(ValueError):
        ui_helpers.load_failure_examples(write(tmp_path, bad))


def test_highlight_escapes_html():
    words = {"words": [{"word": "<script>alert(1)</script>", "importance": 0.5}]}
    out = ui_helpers.highlight_html("<script>alert(1)</script>", words)
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_highlight_escapes_without_explanation():
    out = ui_helpers.highlight_html('a & "b" <i>', None)
    assert out == "a &amp; &quot;b&quot; &lt;i&gt;"


def test_intensity_mapping():
    assert ui_helpers.intensity(0.5, 0.5) == 1.0
    assert ui_helpers.intensity(0.25, 0.5) == 0.5
    assert ui_helpers.intensity(0.001, 0.5) == 0.0  # below noise floor
    assert ui_helpers.intensity(-0.2, 0.5) == 0.0
    assert ui_helpers.intensity(0.003, 0.003) == 0.0  # nothing dominates


def test_highlight_shades_proportionally_and_skips_uncovered_words():
    words = {"words": [
        {"word": "a", "importance": 0.0},
        {"word": "b", "importance": 0.1},
        {"word": "c", "importance": 0.2},
    ]}
    out = ui_helpers.highlight_html("a b c d", words)
    assert out.startswith("a <span")
    assert f"{ui_helpers.MAX_HIGHLIGHT_ALPHA / 2:.2f}" in out
    assert f"{ui_helpers.MAX_HIGHLIGHT_ALPHA:.2f}" in out
    assert out.endswith("c</span> d")  # 4th word was beyond the scored cap


def test_badge_colors_cover_all_tiers_and_escape():
    for tier in (severity.NONE, severity.MILD, severity.HARASSMENT, severity.HATE):
        assert ui_helpers.TIER_COLORS[tier] in ui_helpers.badge_html(tier)
    assert "&lt;" in ui_helpers.badge_html("<b>")


def test_is_flagged_and_top_scores():
    assert not ui_helpers.is_flagged(severity.NONE)
    assert ui_helpers.is_flagged(severity.MILD)
    summary = ui_helpers.top_scores({"toxic": 0.9, "insult": 0.5, "threat": 0.1}, 0.95, n=2)
    assert summary == "secondary_hate 0.95, toxic 0.90"


def test_provider_status_never_exposes_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "super-secret")
    provider, enabled, reason = ui_helpers.provider_status()
    assert (provider, enabled, reason) == ("groq", True, None)
    monkeypatch.delenv("GROQ_API_KEY")
    provider, enabled, reason = ui_helpers.provider_status()
    assert not enabled and "GROQ_API_KEY" in reason and "super-secret" not in reason


def test_no_rephrasing_card_includes_explanation():
    card = ui_helpers.no_rephrasing_card("Targets a group.")
    assert ui_helpers.NO_SAFE_REPHRASING in card and "Targets a group." in card


def test_secondary_note_states_benefit_and_caveat():
    note = ui_helpers.SECONDARY_NOTE
    assert "0.08" in note and "0.61" in note and "inflated" in note and "training" in note
