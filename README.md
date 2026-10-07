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

## HateCheck evaluation (placeholder)

A systematic evaluation on HateCheck (Röttger et al., 2021) is planned. Results to add:

| Functionality group | n | toxic-bert accuracy | secondary accuracy |
|---|---|---|---|
| *(to be filled)* | | | |

## Known limitations

English-only; thresholds are untuned; implicit hate, sarcasm and context are missed;
occlusion highlights are faint when scores are saturated; rephrasing can keep a hurtful
claim in softer words. See the Limitations page and `report/report_outline.md`.
