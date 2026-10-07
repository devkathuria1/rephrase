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


def test_highlight_underlines_only_strong_words():
    words = {"words": [{"word": "a", "importance": 0.1}, {"word": "b", "importance": 0.4}]}
    out = ui_helpers.highlight_html("a b", words)
    assert out.count("text-decoration:underline") == 1  # b is strong, a is faint (0.19)


def test_severity_card_has_icon_text_and_meaning_for_every_tier():
    for tier in (severity.NONE, severity.MILD, severity.HARASSMENT, severity.HATE):
        card = ui_helpers.severity_card_html(tier)
        assert ui_helpers.TIER_ICONS[tier] in card
        assert ui_helpers.TIER_COLORS[tier] in card
        assert ui_helpers.TIER_MEANINGS[tier] in card
    assert "&lt;b&gt;" in ui_helpers.severity_card_html("<b>")


def test_score_bars_clamp_and_escape():
    out = ui_helpers.score_bars_html({"<x>": 1.7, "toxic": -0.2, "insult": 0.5})
    assert "&lt;x&gt;" in out and "<x>" not in out
    assert "width:100.0%" in out and "width:0.0%" in out and "50%" in out
    assert out.count("rp-hot") == 2  # 1.7 and 0.5 reach the hot threshold


def test_compare_cards_with_rephrasing():
    out = ui_helpers.compare_cards_html("<i>bad</i>", "kind <b>words</b>")
    assert "Original" in out and "Rephrased" in out
    assert "<i>" not in out and "&lt;i&gt;bad" in out and "&lt;b&gt;words" in out
    assert "rp-none" not in out


def test_compare_cards_without_rephrasing_is_distinct_and_never_blank():
    out = ui_helpers.compare_cards_html("original", "", "Because <script>")
    assert ui_helpers.NO_SAFE_REPHRASING in out and "rp-none" in out
    assert "Because &lt;script&gt;" in out and "<script>" not in out
    assert "rp-good" not in out


def test_text_card_and_pills_escape():
    assert "&lt;b&gt;" in ui_helpers.text_card_html("t", "<b>x</b>")
    assert "&lt;t&gt;" in ui_helpers.text_card_html("<t>", "x")
    assert "&lt;p&gt;" in ui_helpers.pill_html("<p>", True)
    assert "rp-on" in ui_helpers.provider_pill_html("groq", True)
    assert "off" in ui_helpers.provider_pill_html("groq", False)


def test_classify_warnings_groups():
    groups = ui_helpers.classify_warnings(
        ["Rephrasing disabled: no key", "Input truncated to 5 characters."]
    )
    assert groups == {
        "warning": ["Input truncated to 5 characters."],
        "info": ["Rephrasing disabled: no key"],
    }
    assert ui_helpers.classify_warnings([]) == {"warning": [], "info": []}


def test_example_card_escapes_and_marks_correctness():
    row = {"id": 3, "category": "<c>", "text": "<script>", "expected": "should flag",
           "tier": severity.HATE, "top_scores": "toxic 0.90"}
    good, bad = ui_helpers.example_card_html(row, True), ui_helpers.example_card_html(row, False)
    assert "Correct" in good and "rp-ok" in good and "Wrong" not in good
    assert "Wrong" in bad and "rp-bad" in bad
    assert "<script>" not in good and "&lt;script&gt;" in good and "&lt;c&gt;" in good


def test_hero_footer_and_legend():
    hero = ui_helpers.hero_html()
    for name, _ in ui_helpers.STEPS:
        assert name in hero
    assert "Rephrase" in hero and hero.count("rp-step-num") == 4
    footer = ui_helpers.footer_html()
    assert "not a moderator" in footer and "No text is stored" in footer
    assert "underlined" in ui_helpers.legend_html()


def test_builders_have_no_blank_lines():
    # Blank lines would make Markdown render the HTML as a code block.
    for out in (
        ui_helpers.hero_html(),
        ui_helpers.compare_cards_html("a", "", "e"),
        ui_helpers.score_bars_html({"a": 0.2}),
        ui_helpers.legend_html(),
    ):
        assert "\n" not in out


def test_try_examples_are_four_short_and_nonempty():
    assert len(ui_helpers.EXAMPLES) == 4
    for label, text in ui_helpers.EXAMPLES:
        assert label and 0 < len(text) < 100
