"""All the settings for the project live here, in one place.

Change a number here and the whole project uses the new value.
"""

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

# Read ANTHROPIC_API_KEY (and any other settings) from a .env file if there is one.
load_dotenv()

# ---------- Folders and files ----------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

MESSAGES_CSV = DATA_DIR / "messages.csv"          # customer messages (from the Bitext dataset)
MESSY_CSV = DATA_DIR / "messy_messages.csv"       # hand-written messy messages (typos, slang, anger)
ORDERS_CSV = DATA_DIR / "orders.csv"              # fake NovaMart orders
POLICY_MD = DATA_DIR / "policy.md"                # NovaMart refund / shipping policy
LLM_LABELS_CSV = DATA_DIR / "llm_labels.csv"      # labels written by the LLM (made by `python run.py label`)
CLASSIFIER_PATH = MODELS_DIR / "intent_classifier.joblib"

# The "today" of our fake shop. Fixed, so results are the same every time.
TODAY = date(2026, 10, 5)

# ---------- Models ----------
# Which LLM provider to use for the agent and for labelling.
# Leave LLM_PROVIDER empty and the project uses the first API key it finds in .env.
PROVIDERS = {
    "openai":    {"key": "OPENAI_API_KEY",    "model": os.getenv("OPENAI_MODEL", "gpt-6.1-sol")},
    "anthropic": {"key": "ANTHROPIC_API_KEY", "model": os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")},
    "gemini":    {"key": "GEMINI_API_KEY",    "model": os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
                  "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"},
    # No API key? Ollama runs a model on your laptop for free (slower and less smart, but it works).
    "ollama":    {"key": "OLLAMA_API_KEY",    "model": os.getenv("OLLAMA_AGENT_MODEL", "llama3.2:3b"),
                  "base_url": "http://localhost:11434/v1"},
}

# Small model for layer 2 of the router. Runs on your laptop with Ollama. Free, private, works offline.
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "llama3.2:3b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

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

# ---------- Router settings (version 4) ----------
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
