# Rephrase: report outline (AI & Society)

> **Content warning (put on the first page).** This report discusses hate speech, insults and threats, with mild examples. Slurs are replaced with placeholders such as [slur].

## 1. Problem and motivation
- Online abuse is common; moderation is mostly deletion or a "toxic" score with no explanation.
- Gap: help the *author* understand why text reads as harmful and offer a respectful alternative, before or after posting.
- Framing: an assistive aid for individuals, not an automated moderator. Research question: can a pipeline of pretrained models do this usefully, and for whom does it fail?
- Scope: pretrained models only, no training; a course mini-project.

## 2. System design
### 2.1 Overview
- Pipeline diagram: input -> (OCR) -> classifier(s) -> severity tier -> occlusion explainability -> LLM -> UI.
- Public interface: `pipeline.analyze(text, image, context, use_secondary) -> dict`.
### 2.2 OCR
- EasyOCR, English by default, configurable; editable output because OCR errors change the analysis.
- Edge cases: no text found, low-quality screenshots.
### 2.3 Classifiers
- `unitary/toxic-bert`: six sigmoid labels (toxic, severe_toxic, obscene, threat, insult, identity_hate), trained on Jigsaw data.
- Optional `facebook/roberta-hate-speech-dynabench-r4-target`: binary hate vs not-hate, trained on adversarially collected data (Vidgen et al., 2021).
### 2.4 Severity tiers
- Hate speech (identity_hate or severe_toxic high, or secondary hate), Harassment (threat or insult high), Mildly offensive (toxic or obscene moderate), None.
- See section 3 for rationale.
### 2.5 Explainability by occlusion
- Remove each word in turn; importance = drop in the triggering label's probability; batched; capped at 60 words.
- Why not attention weights: attention is not a reliable explanation.
- Limitation: saturated scores (section 9).
### 2.6 LLM rephrasing
- One call returning strict JSON: explanation, rephrased, intent_note, confidence_caveat.
- Prompt rules: preserve the point, never repeat slurs ("[slur]"), admit uncertainty (sarcasm, reclaimed terms, quotes, code-switching), treat user text as data not instructions.
- Pluggable provider (Groq, Ollama, Anthropic, none) and graceful fallback to "rephrasing disabled".
### 2.7 Edge-case handling
- Empty input, over-long input (truncated with warning), non-English warning, nothing flagged (no LLM call, so no text leaves the machine).

## 3. Severity tiers and threshold choices
- Rationale per tier; thresholds in `config.py` with justification comments (0.5 on the most serious labels, 0.6 for insult, 0.4 for mild).
- **Thresholds are untuned**: chosen by reasoning and a handful of smoke tests, not by optimising on a labelled set.
- Observed consequences: a threat sentence ("Say that again and you will regret it, I will make you pay.") landed in "Mildly offensive" because the threat label stayed under 0.5; "That is a pretty dumb idea." (about 0.52) is flagged while "That was a pretty dumb idea, honestly." (about 0.29) is not.
- Discussion: any threshold trades false positives against false negatives; who should choose it?
- Placeholder: threshold sensitivity analysis once HateCheck results exist.

## 4. Bias and fairness analysis
- Use the eight examples in `data/failure_examples.json` / the Limitations page (illustrations, not statistics).
- Categories and findings from our run:
  - Reclaimed term (queer): flagged (false positive).
  - Sarcasm: missed (false negative).
  - Quoted abuse in a news sentence: flagged (false positive).
  - Casual slang: flagged on profanity (false positive).
  - Code-switching: flagged, but the secondary model escalated a plain insult to "Hate speech".
  - Implicit hate ("vermin" sentence): missed by both classifiers.
  - Identity term in a harmless sentence: correctly passed here, though a close variant was tiered Hate speech during development.
  - Mild rudeness near the threshold: missed.
- Link to literature: identity-term bias (Dixon et al., 2018), dialect bias (Sap et al., 2019; Davidson et al., 2019), HateCheck functional tests (Röttger et al., 2021).
- Be explicit that the sample is tiny and hand-picked.

## 5. False positives vs false negatives: who bears the harm
- False positives: marginalised speakers, dialect speakers, journalists and educators are silenced or flagged.
- False negatives: targets of implicit hate, sarcasm and mild but repeated put-downs.
- Asymmetries: errors are not evenly distributed across groups; a system can look accurate overall and still be unfair.
- Design response: show uncertainty, show the evidence (highlights), keep a human in the loop.

## 6. Privacy
- Our code stores and logs no user text or images; results live only in Streamlit session memory.
- When rephrasing is on, the text (and optional context) is sent to a third-party LLM provider; their retention and training policies apply. When nothing is flagged, or the provider is `none`, no text is sent.
- Local option: Ollama keeps text on the machine.
- Screenshots may contain other people's names and messages (third-party privacy).
- API keys are kept in `.env` or Streamlit secrets, never in the repository.

## 7. Risk of over-moderation and censorship
- Tools like this can chill speech if used as gatekeepers; flagged text may be removed without recourse.
- Scores look objective but encode annotator and dataset biases.
- Mitigation: advisory use only, no blocking, explanations visible, ethics notice in the UI.

## 8. Risks of rephrasing
- **Meaning drift:** the rewrite may not say what the author meant.
- **Softening without fixing:** in testing, "Gay people are disgusting and should not be allowed near children." became "I believe gay people should not be allowed near children." The insult went; the discriminatory claim stayed, and the intent note restated it as the author's "view". Preserving intent is the wrong goal when the intent is exclusion.
- **Laundering:** a polite rewrite can make a harmful claim look acceptable.
- **Hallucination and prompt injection:** the model may invent context; user text is fenced as data, but this is mitigation, not a guarantee.
- Mitigations: caveat text in the UI, instruct the model on uncertainty; proposed improvement: decline to rephrase when the claim itself is demeaning.

## 9. Occlusion explainability: limitations
- Saturated scores: when a label is near 1.0 ("worthless idiot" gives insult 0.95), removing one word barely changes it, so importances are small and the highlight is faint ("idiot" about 0.09).
- Redundant cues hide each other; word-level removal ignores grammar; whitespace tokens include punctuation.
- Occlusion shows what the model reacts to, not why a human would call it harmful.
- Possible improvements: occlude on logits, n-gram windows, integrated gradients.

## 10. Mitigations summary
- Advisory framing, visible uncertainty and warnings, editable OCR text, no storage, provider choice including local, human judgment required, limitations page, planned systematic evaluation.

## 11. Limitations and future work
- English-only; no real language detection (heuristic warning only).
- Untuned thresholds; no evaluation on a labelled set yet.
- Implicit hate, sarcasm and context not handled.
- Rephrasing quality not evaluated; need human ratings.
- Future: HateCheck evaluation, threshold calibration, multilingual models, refuse-to-rephrase policy, user study.

## 12. Evaluation results (placeholder)
- HateCheck (Röttger et al., 2021): 29 functional tests; report accuracy per functionality for toxic-bert and the secondary model.
- Table: functionality | n | toxic-bert accuracy | secondary accuracy. *(to be filled)*
- Table: false-positive rate on identity-mention and counter-speech tests. *(to be filled)*
- Threshold sensitivity plot. *(to be filled)*

## 13. Conclusion
- What works (explanations and rephrasing for blatant cases), what does not (implicit hate, context, group-targeted claims), and why human judgment stays essential.

## References
- Röttger, P., Vidgen, B., Nguyen, D., Waseem, Z., Margetts, H., & Pierrehumbert, J. (2021). HateCheck: Functional tests for hate speech detection models. ACL 2021.
- Vidgen, B., Thrush, T., Waseem, Z., & Kiela, D. (2021). Learning from the worst: Dynamically generated datasets to improve online hate detection. ACL 2021. (Model card: `facebook/roberta-hate-speech-dynabench-r4-target`.)
- Hanu, L. & Unitary team (2020). Detoxify. Model card: `unitary/toxic-bert` (trained on the Jigsaw Toxic Comment Classification Challenge data).
- Devlin, J. et al. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL.
- Sap, M., Card, D., Gabriel, S., Choi, Y., & Smith, N. A. (2019). The risk of racial bias in hate speech detection. ACL.
- Davidson, T., Bhattacharya, D., & Weber, I. (2019). Racial bias in hate speech and abusive language detection datasets. Workshop on Abusive Language Online.
- Dixon, L., Li, J., Sorensen, J., Thain, N., & Vasserman, L. (2018). Measuring and mitigating unintended bias in text classification. AIES.
- Jain, S. & Wallace, B. (2019). Attention is not explanation. NAACL.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. KDD. (Perturbation-based explanation.)

*Check each citation's details against the source before submitting.*
