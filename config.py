"""Central configuration: thresholds, limits and model names."""

import os

from dotenv import load_dotenv

load_dotenv()

# --- Models ---------------------------------------------------------------
PRIMARY_MODEL = "unitary/toxic-bert"
SECONDARY_MODEL = "facebook/roberta-hate-speech-dynabench-r4-target"
LLM_MAX_TOKENS = 1024
LLM_TIMEOUT_SECONDS = 30.0

# --- LLM provider ----------------------------------------------------------
PROVIDERS = ("groq", "ollama", "anthropic", "none")
DEFAULT_PROVIDER = "groq"  # free tier, so the project works without paid APIs
DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "ollama": "llama3.1",
    "anthropic": "claude-sonnet-5-5",
}
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def llm_provider() -> str:
    """Return the provider from LLM_PROVIDER (lower-cased), or the default."""
    return (os.environ.get("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()


def llm_model(provider: str) -> str:
    """Return the model from LLM_MODEL, or the provider's default."""
    return os.environ.get("LLM_MODEL") or DEFAULT_MODELS[provider]


def ollama_url() -> str:
    """Return the Ollama base URL from OLLAMA_URL, or the local default."""
    return (os.environ.get("OLLAMA_URL") or DEFAULT_OLLAMA_URL).rstrip("/")


# --- Severity thresholds (probabilities from toxic-bert's sigmoid heads) ---
# 0.5 is the natural decision boundary of each independent sigmoid head, so we
# use it for the labels that define the most serious tiers: a model that is
# "more likely than not" calling something hate or a threat should be believed.
HATE_THRESHOLD = 0.5  # identity_hate -> Hate speech
SEVERE_TOXIC_THRESHOLD = 0.5  # severe_toxic -> Hate speech
THREAT_THRESHOLD = 0.5  # threat -> Harassment
# toxic-bert's insult head fires readily on ordinary rudeness; a higher bar
# keeps "Harassment" for clearly personal attacks.
INSULT_THRESHOLD = 0.6
# Lower bar for the mild tier: we would rather over-flag mild offensiveness
# (the cost is a gentle suggestion) than miss it.
MILD_TOXIC_THRESHOLD = 0.4  # toxic / obscene -> Mildly offensive
# The dynabench R4 model is trained on adversarial hate examples and is
# reasonably calibrated; 0.5 is its decision boundary.
SECONDARY_HATE_THRESHOLD = 0.5

# --- Input limits ----------------------------------------------------------
# Rephrasing/explanation of long documents is not the goal; capping bounds
# classifier latency and LLM cost. 512 tokens is the BERT context limit, and
# ~1500 characters is comfortably below it for English.
MAX_INPUT_CHARS = 1500
CLASSIFIER_MAX_TOKENS = 512
CLASSIFIER_BATCH_SIZE = 16

# --- Explainability --------------------------------------------------------
# Occlusion needs one forward pass per unit; capping words bounds that cost.
MAX_OCCLUSION_WORDS = 60
# Window size for occlusion: 1 = single words. 2 would also catch phrases
# whose toxicity only appears jointly, at roughly the same cost.
OCCLUSION_NGRAM = 1
# Minimum fraction of English stopwords (among words of >= 4-word inputs)
# below which we warn the text may not be English.
MIN_ENGLISH_STOPWORD_RATIO = 0.1

# --- OCR -------------------------------------------------------------------
OCR_LANGUAGES = ["en"]
