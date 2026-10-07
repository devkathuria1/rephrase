"""End-to-end pipeline: OCR -> classify -> severity -> explain -> rephrase."""

from __future__ import annotations

import re

import config
from src import classifier, explain, llm, ocr, severity

_ENGLISH_STOPWORDS = frozenset(
    "the a an and or but is are was were be to of in on at for with that this it "
    "you i he she we they not do have has will my your his her our their what who "
    "so if as by from just can would should".split()
)
TOP_WORDS = 5


def _looks_non_english(text: str) -> bool:
    """Cheap heuristic: mostly non-ASCII letters, or few English stopwords."""
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isascii() for c in letters) / len(letters) < 0.8:
        return True
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < 4:
        return False
    hits = sum(w in _ENGLISH_STOPWORDS for w in words)
    return hits / len(words) < config.MIN_ENGLISH_STOPWORD_RATIO


def analyze(
    text: str | None = None,
    image=None,
    context: str = "",
    use_secondary: bool = False,
) -> dict:
    """Analyse text and/or a screenshot for toxicity and suggest a rephrasing.

    Args:
        text: Text to analyse.
        image: Screenshot (path, bytes, array or PIL image); OCR text is used,
            and appended to ``text`` if both are given.
        context: Optional extra context passed to the LLM only.
        use_secondary: Also run the secondary hate-speech classifier.

    Returns:
        A dict with ``text``, ``severity``, ``triggered``, ``scores``,
        ``secondary_hate``, ``word_importance``, ``llm_called``,
        ``rephrasing_enabled``, ``llm`` (LLM fields, or None) and ``warnings``.
        Nothing is stored; all data lives only in the returned dict.
    """
    warnings: list[str] = []
    parts = [text.strip()] if text and text.strip() else []
    if image is not None:
        ocr_text = ocr.extract_text(image)
        if ocr_text:
            parts.append(ocr_text)
        else:
            warnings.append("No text was found in the image.")
    combined = " ".join(parts)

    result: dict = {
        "text": combined,
        "severity": severity.NONE,
        "triggered": [],
        "scores": {},
        "secondary_hate": None,
        "word_importance": None,
        "llm_called": False,
        "rephrasing_enabled": llm.is_enabled(),
        "llm": None,
        "warnings": warnings,
    }
    if not combined:
        warnings.append("Empty input: nothing to analyse.")
        return result

    if len(combined) > config.MAX_INPUT_CHARS:
        combined = combined[: config.MAX_INPUT_CHARS]
        result["text"] = combined
        warnings.append(f"Input truncated to {config.MAX_INPUT_CHARS} characters.")
    if _looks_non_english(combined):
        warnings.append(
            "Text may not be English; the classifiers are English-only and "
            "results may be unreliable."
        )

    scores = classifier.score([combined])[0]
    result["scores"] = scores
    if use_secondary:
        result["secondary_hate"] = classifier.score_secondary([combined])[0]
    sev = severity.assess(scores, result["secondary_hate"])
    result["severity"], result["triggered"] = sev.tier, sev.triggered
    if sev.tier == severity.NONE:
        return result

    result["word_importance"] = explain.explain(combined, scores, sev.triggered[0])
    if not result["rephrasing_enabled"]:
        warnings.append("Rephrasing disabled: ANTHROPIC_API_KEY is not set.")
        return result

    ranked = sorted(
        result["word_importance"]["words"], key=lambda w: w["importance"], reverse=True
    )
    top_words = [w["word"] for w in ranked[:TOP_WORDS] if w["importance"] > 0]
    result["llm_called"] = True
    reply = llm.rephrase(combined, context, sev.tier, scores, top_words)
    if reply["ok"]:
        result["llm"] = {k: reply[k] for k in llm.KEYS}
    else:
        warnings.append(reply["error"])
    return result
