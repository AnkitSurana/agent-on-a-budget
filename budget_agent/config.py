"""All the settings for the project live here, in one place.

Change a number here and the whole project uses the new value.
"""

import os
import sys
from datetime import date
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

# ---------- Folders and files ----------
ROOT = Path(__file__).resolve().parent.parent


def other_python():
    """Is this a different Python from the project's own .venv (for example Anaconda's)? Then return its path.

    Commands that start with `uv run` always use the project's .venv. Typed without it, `python` or `streamlit`
    can be another Python, with other (often older) packages.
    """
    venv = ROOT / ".venv"
    if venv.exists() and Path(sys.prefix).resolve() != venv.resolve():
        return sys.prefix
    return None

# Read the API keys and settings from THIS project's .env file only (never from a parent folder).
ENV_FILE = ROOT / ".env"
load_dotenv(ENV_FILE)
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

MESSAGES_CSV = DATA_DIR / "messages.csv"          # customer messages (from the Bitext dataset)
MESSY_CSV = DATA_DIR / "messy_messages.csv"       # hand-written messy messages (typos, slang, anger)
ORDERS_CSV = DATA_DIR / "orders.csv"              # fake NovaMart orders
POLICY_MD = DATA_DIR / "policy.md"                # NovaMart refund / shipping policy
LLM_LABELS_CSV = DATA_DIR / "llm_labels.csv"      # labels written by the LLM (made by `uv run python run.py label`)
CLASSIFIER_PATH = MODELS_DIR / "intent_classifier.joblib"
BENCHMARK_CSV = DATA_DIR / "benchmark_results.csv"  # expert only vs the whole office (made by `run.py benchmark`)

# Saved answers (see replay.py): the same request is answered from this file, free and offline.
REPLAY_LOG = DATA_DIR / "saved_llm_answers.jsonl"

# The "today" of our fake shop. Fixed, so results are the same every time.
TODAY = date(2026, 10, 5)

# ---------- Models ----------
# Inside Docker, "localhost" is the container itself. Ollama runs on your computer, which Docker calls host.docker.internal.
IN_DOCKER = Path("/.dockerenv").exists()


def _read_settings():
    """Read the settings that come from .env. Runs at the start, and again whenever .env changes."""
    global REPLAY, PROVIDERS, LOCAL_MODEL, OLLAMA_URL
    REPLAY = os.getenv("REPLAY", "on").lower() != "off"     # REPLAY=off in .env = always ask the LLM fresh

    # Small model for layer 2 of the router. Runs on your laptop with Ollama. Free, private, works offline.
    LOCAL_MODEL = os.getenv("LOCAL_MODEL", "llama3.2:3b")
    OLLAMA_URL = os.getenv("OLLAMA_URL") or ("http://host.docker.internal:11434" if IN_DOCKER else "http://localhost:11434")

    # Which LLM is the senior expert, set in .env with just three lines, the same for every provider:
    #   LLM_PROVIDER=openai      (or anthropic, gemini, ollama)
    #   LLM_API_KEY=...          (the key for that provider)
    #   LLM_MODEL=...            (optional: otherwise the default model below)
    PROVIDERS = {
        "openai":    {"model": "gpt-5.4-mini"},
        "anthropic": {"model": "claude-opus-5-5"},
        "gemini":    {"model": "gemini-3.8-flash", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"},
        # No API key? Ollama runs a model on your laptop for free (slower and less smart, but it works).
        "ollama":    {"model": "llama3.2:3b", "base_url": OLLAMA_URL + "/v1"},
    }
    chosen = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if chosen in PROVIDERS and os.getenv("LLM_MODEL"):
        PROVIDERS[chosen]["model"] = os.getenv("LLM_MODEL")


def api_key():
    """The key for the chosen provider. One name for every provider: LLM_API_KEY."""
    return os.getenv("LLM_API_KEY") or ""


_read_settings()
_env_seen = {k: v for k, v in dotenv_values(ENV_FILE).items() if v is not None} if ENV_FILE.exists() else {}
_env_time = ENV_FILE.stat().st_mtime if ENV_FILE.exists() else None


def refresh():
    """Edited .env while the app or a notebook is running? Use the new values now, no restart needed.

    Only lines you changed in .env are applied, so a value set in the terminal still wins otherwise.
    """
    global _env_seen, _env_time
    time_now = ENV_FILE.stat().st_mtime if ENV_FILE.exists() else None
    if time_now == _env_time:
        return False
    values = {k: v for k, v in dotenv_values(ENV_FILE).items() if v is not None} if time_now else {}
    for name in _env_seen.keys() - values.keys():          # line deleted from .env
        os.environ.pop(name, None)
    for name, value in values.items():                     # line added or changed
        if _env_seen.get(name) != value:
            os.environ[name] = value
    _env_seen, _env_time = values, time_now
    _read_settings()
    return True


# Price in US dollars per 1 million tokens: (input, output). Checked October 2026.
# Prices change, so check the provider's pricing page. Models not listed here count as free.
PRICES_PER_MILLION = {
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5-mini": (0.25, 2.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gemini-3.8-flash": (0.75, 3.75),   # promo price until 31 Dec 2026 (then 1.50 / 7.50); free tier available
    "gemini-3.1-pro-preview": (2.00, 12.00),
}

# ---------- Router settings (step 6) ----------
# If the ML model is at least this sure, we answer with no LLM at all.
ML_CONFIDENCE = 0.80
# If the ML model is at least this sure (but below ML_CONFIDENCE), the small local LLM answers.
LOCAL_LLM_CONFIDENCE = 0.40

# Intents that change something (money, orders, accounts) or need a human touch.
# These ALWAYS go to the big agent with tools, no matter how sure the ML model is.
ACTION_INTENTS = {
    "cancel_order",
    "change_order",
    "change_shipping_address",
    "complaint",
    "contact_human_agent",
    "delete_account",
    "get_refund",
    "payment_issue",
}

# All 27 intents in the Bitext dataset.
INTENTS = [
    "cancel_order", "change_order", "change_shipping_address", "check_cancellation_fee",
    "check_invoice", "check_payment_methods", "check_refund_policy", "complaint",
    "contact_customer_service", "contact_human_agent", "create_account", "delete_account",
    "delivery_options", "delivery_period", "edit_account", "get_invoice",
    "get_refund", "newsletter_subscription", "payment_issue", "place_order",
    "recover_password", "registration_problems", "review", "set_up_shipping_address",
    "switch_account", "track_order", "track_refund",
]

# ---------- The "support office" story, used the same way everywhere ----------
# Each router layer is a member of a support office. The code says "ml", "local_llm", "cloud_llm";
# the README, notebooks, app and slides use these names.
HELPERS = {
    "ml": "Receptionist (ML model)",
    "local_llm": "Junior assistant (local LLM)",
    "cloud_llm": "Senior expert (cloud LLM agent)",
    "human": "Manager (human)",
}

# One example message per helper, checked against data/orders.csv. Used in every demo.
DEMO_MESSAGES = {
    "ml": "track my order NM10009",                                  # shipped running shoes
    "local_llm": "cash on delivery hai kya?",                        # payment question, in Hinglish
    "cloud_llm": "my order NM10002 arrived broken, I want a refund",  # ₹5,999 watch: refund allowed
    "human": "refund NM10018, I don't like the chair",               # ₹12,999 chair: needs a human
}
