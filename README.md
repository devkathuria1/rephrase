# Rephrase

Detects toxicity / hate speech in text or screenshots, explains why, and suggests a
respectful rephrasing. A mini-project for an "AI & Society" course. It is an
**assistive aid, not a moderator**: the models are biased and make mistakes, so human
judgment is required.

Pipeline: OCR (EasyOCR) -> `unitary/toxic-bert` (+ optional
`facebook/roberta-hate-speech-dynabench-r4-target`) -> severity tier -> leave-one-out
word importance -> one LLM call (explanation, rephrasing, intent note, caveat).
No training or fine-tuning.

> Content note: the app, the Limitations page and the report materials discuss insults and hate speech in mild form. Slurs are replaced by placeholders.

## Setup (Windows)

Run these in order, one at a time, in the foreground:

```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --progress-bar off
copy .env.example .env
```

Edit `.env` (it is git-ignored). The first run downloads the models (a few hundred MB).

## Environment variables

| Variable | Meaning |
|---|---|
| `LLM_PROVIDER` | `groq` (default), `ollama`, `anthropic` or `none` |
| `GROQ_API_KEY` | key for Groq (free tier) |
| `ANTHROPIC_API_KEY` | key for the optional Anthropic provider |
| `OLLAMA_URL` | local Ollama server, default `http://localhost:11434` |
| `LLM_MODEL` | model name; blank uses the provider default (groq `openai/gpt-oss-120b`, ollama `llama3.1`, anthropic `claude-sonnet-5-5`) |

If the provider is missing a key, rate limited, unreachable or returns bad JSON, the
pipeline does not crash: rephrasing is disabled and the reason appears in `warnings`.
Keys are never logged, returned or displayed. If you get an HTTP 404 from Groq, the
model was probably retired; list current ones with
`GET https://api.groq.com/openai/v1/models`.

## Run

```
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Pages: the main app, and "Limitations" (`pages/1_Limitations.py`), which runs 8
illustrative failure cases live through the classifiers (no LLM call).

The secondary hate-speech classifier is **on by default in the app's sidebar**: on
HateCheck it wrongly flagged far fewer harmless cases than toxic-bert (false positive
rate 0.08 vs 0.61; see the evaluation section), but that score may be inflated if
HateCheck-like data was in its training. In the app it *adds* to toxic-bert's flags (it
can raise a text to "Hate speech" but never lowers a tier), and toxic-bert still
provides the per-label breakdown (insult, threat, ...). `pipeline.analyze` itself still
defaults to `use_secondary=False`; the app passes the toggle's value.

Use the pipeline directly:

```python
from src import pipeline
result = pipeline.analyze(text="...", context="", use_secondary=False)
```

## Tests

```
.\.venv\Scripts\python.exe -m pytest
```

Classifiers and every LLM provider are mocked; no models or network are needed.

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub (`.env`, `.venv/` and user data are git-ignored; commit `data/failure_examples.json`, which is whitelisted).
2. Create the app at share.streamlit.io, main file `app.py`.
3. In the app's **Settings -> Secrets**, add (never commit these):
   ```
   LLM_PROVIDER = "groq"
   GROQ_API_KEY = "your-key"
   LLM_MODEL = ""
   ```
   The app copies these into the environment at startup.
4. Notes: the free tier has limited memory, so the torch + EasyOCR models may be tight; the first load is slow. Ollama is not available there, so use `groq`, `anthropic` or `none`. `requirements.txt` installs CPU-capable torch only if you add the CPU index line yourself (add `--extra-index-url https://download.pytorch.org/whl/cpu` at the top of `requirements.txt` for the cloud build).

## GPU option

Everything runs on CPU by default (CPU torch, EasyOCR with `gpu=False`). For a GPU,
install a CUDA build of torch from https://pytorch.org instead of the CPU wheel, then
move the models and inputs to CUDA in `src/classifier.py` and set `gpu=True` in
`src/ocr.py`. This is not wired up or tested.

## Project structure

```
app.py                      Streamlit app
pages/1_Limitations.py      Failure examples, run live
config.py                   Thresholds (with justifications), models, provider settings
prompts.py                  LLM system prompt and user-message builder
src/ocr.py                  EasyOCR text extraction
src/classifier.py           toxic-bert and secondary classifier
src/severity.py             Scores -> severity tier
src/explain.py              Leave-one-out occlusion
src/llm.py                  Provider-pluggable LLM call, JSON validation, fallback
src/pipeline.py             analyze(): the public entry point
src/ui_helpers.py           HTML highlighting, badges, example loading (no Streamlit)
data/failure_examples.json  The 8 illustrative failure cases
evaluation/                 HateCheck evaluation: run_hatecheck, metrics, plots, examples
report/                     report_outline.md, demo_script.md
tests/                      pytest suite
```

## Privacy

Our code stores no user text or images and logs none; results exist only in the
Streamlit session's memory for the current visit. If rephrasing is enabled and
something is flagged, the text (and optional context) is sent to the configured
third-party LLM provider, whose own retention and privacy policy applies. Nothing is
sent if nothing is flagged or the provider is `none`; Ollama keeps text on your machine.
Screenshots may contain other people's messages, so use them with care.

## Evaluation on HateCheck

No training, no LLM calls: the pretrained classifiers are run over the public HateCheck
test suite (Rottger et al., 2021). **If you use these results, cite: Paul Rottger,
Bertie Vidgen, Dong Nguyen, Zeerak Waseem, Helen Margetts and Janet Pierrehumbert (2021),
"HateCheck: Functional Tests for Hate Speech Detection Models", ACL 2021.**

**1. Download the data** (not committed; `data/` is git-ignored except the failure examples):

```
curl -L -o data/hatecheck.csv https://raw.githubusercontent.com/paul-rottger/hatecheck-data/main/test_suite_cases.csv
```

Expected columns include `test_case`, `label_gold`, `functionality`, `target_ident`
(column names are detected automatically). If the file is missing the script prints
these instructions and stops.

**2. Run** (from the project root, in order):

```
.\.venv\Scripts\python.exe -m evaluation.run_hatecheck --secondary
.\.venv\Scripts\python.exe -m evaluation.metrics
.\.venv\Scripts\python.exe -m evaluation.plots
.\.venv\Scripts\python.exe -m evaluation.examples
```

`--limit N` evaluates a reproducible random sample of N rows for a quick test;
`--secondary` also runs the secondary classifier, which is always reported separately
from toxic-bert. Full run: about 3 minutes on a laptop CPU for both models (3,728 cases),
plus the one-time model download.

**Mapping:** a case is predicted hateful if the severity tier is anything above "None"
(binary mapping of the 4 tiers). The secondary model alone is predicted hateful at
probability >= 0.5. Positive class = hateful.

**Outputs** (all in `results/`, git-ignored):

| File | Meaning |
|---|---|
| `hatecheck_predictions.csv` | One row per case: text, gold label, functionality, target group, all six toxic-bert scores, tier, prediction (and secondary score and prediction). Contains the raw, unmasked test sentences. |
| `hatecheck_metrics.json` | Per model: accuracy, precision, recall, F1, FPR, FNR, confusion counts, and flag rate / FPR / FNR per target group; plus the thresholds used. |
| `hatecheck_by_functionality.csv` | The same per HateCheck functionality (worst first). Each functionality has one gold class, so FPR (non-hateful) or FNR (hateful) is the relevant error. |
| `threshold_sensitivity.csv` | toxic-bert metrics with the mild threshold at 0.25, 0.4 (current) and 0.55: the false positive / false negative trade-off. |
| `error_by_functionality.png`, `confusion_matrix.png`, `flag_rate_by_target.png` | Charts (secondary model: same names with `_secondary`). |
| `worst_cases.md` | The 10 worst functionalities with 3 example errors each, with slurs and explicit words masked as [slur] / [explicit], behind a content warning. |

Raw explicit text appears only in the predictions CSV. Do not publish that file.

## Known limitations

English-only; thresholds are untuned; implicit hate, sarcasm and context are missed;
occlusion highlights are faint when scores are saturated; rephrasing can keep a hurtful
claim in softer words. See the Limitations page and `report/report_outline.md`.
