# Rephrase (backend)

Detects toxicity / hate speech in text or screenshots, explains why, and suggests a
non-toxic rephrasing. Backend only; no UI. No user data is stored.

Pipeline: OCR (EasyOCR) -> `unitary/toxic-bert` (+ optional
`facebook/roberta-hate-speech-dynabench-r4-target`) -> severity tier -> leave-one-out
word importance -> one LLM call (explanation, rephrasing, intent note, caveat).

## Setup (Windows)

```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --progress-bar off
copy .env.example .env
```

## LLM providers

Set `LLM_PROVIDER` in `.env`:

| Value | Needs | Default model |
|---|---|---|
| `groq` | `GROQ_API_KEY` (free tier) | `openai/gpt-oss-120b` |
| `ollama` | local server at `OLLAMA_URL` (default `http://localhost:11434`) | `llama3.1` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` |
| `none` | nothing | rephrasing disabled |

Override the model with `LLM_MODEL`. Groq retires models over time; list current ones
with `GET https://api.groq.com/openai/v1/models` if you get an HTTP 404.

If the provider is missing a key, rate limited, unreachable, or returns bad JSON, the
pipeline does not crash: `llm` is `None`, `rephrasing_enabled` is `False`, and a
`"Rephrasing disabled: ..."` entry is added to `warnings`. API keys are never logged or
returned.

## Usage

```python
from src import pipeline
result = pipeline.analyze(text="...", context="", use_secondary=False)
result = pipeline.analyze(image="screenshot.png")
```

Result keys: `text`, `severity` (Hate speech / Harassment / Mildly offensive / None),
`triggered`, `scores`, `secondary_hate`, `word_importance`, `llm_called`,
`rephrasing_enabled`, `llm`, `warnings`. Thresholds live in `config.py`.

## Tests

```
.\.venv\Scripts\python.exe -m pytest
```

The classifier and every LLM provider are mocked; no models or network are needed.

## Limitations

The classifiers miss implicit or dehumanising language that names no group, and are
English-only. Thresholds are not tuned on a dataset.
