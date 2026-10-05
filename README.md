# 🛒 Agent on a Budget

**An AI support agent that gets cheaper and faster by teaching a small Machine Learning model to do most of its work.**

Most AI agents send *every* request to a big, expensive LLM. This project starts there and then improves it step by step:
an LLM labels data, a small ML model learns from those labels, and a router sends each message to the **cheapest model that can handle it**.

> Built for the workshop **"Build Your First AI Agent: From Machine Learning & LLMs to Real-World Workflows"**.

---

## The idea in one picture

```
Customer message
      │
      ▼
┌──────────────────┐  very sure + simple question
│ 1. ML model      │ ─────────────────────────────▶ ready-made answer      ~0.05 ms · $0
└────────┬─────────┘
         │ fairly sure + simple question
         ▼
┌──────────────────┐
│ 2. Local LLM     │ ─────────────────────────────▶ short answer           ~1 s · $0
│    (Ollama)      │
└────────┬─────────┘
         │ action, angry customer, or unsure
         ▼
┌──────────────────┐
│ 3. Cloud LLM     │ ─────────────────────────────▶ looks up orders,       a few s · paid
│    agent + tools │                                 refunds, escalates…
└──────────────────┘
```

## The story: four versions of one agent

| Version | What it is | What you learn |
|---|---|---|
| **V1** | A cloud LLM answers customers. No tools. | What an LLM is, prompts, tokens, cost |
| **V2** | The LLM **with tools**: look up orders, refund, cancel, change address, hand over to a human | What makes an *agent*: the tool loop, guardrails |
| **V3** | The LLM **labels** messages, a small **ML model** learns from them | ML basics: features, training, accuracy, confidence, *distillation* |
| **V4** | A **router**: ML model → local LLM → cloud LLM | *Inference engineering*: cost, speed, and picking the right model |

## The shop: NovaMart

NovaMart is a pretend online shop in India. The agent works with:

| File | What's inside |
|---|---|
| `data/messages.csv` | 1,620 customer messages (27 intents × 60) from the [Bitext customer support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), split into train and test |
| `data/messy_messages.csv` | 50 hand-written hard messages: typos, Hinglish, anger, two requests in one |
| `data/orders.csv` | 300 fake orders (made by `python run.py data`) |
| `data/policy.md` | The shop's refund, delivery and payment rules |

## Results so far

The ML model (TF-IDF on pieces of words + Logistic Regression), trained on 1,080 messages:

| | Accuracy |
|---|---|
| Clean test messages | **98.3%** |
| Messy messages | 60.0% |
| Messages where it is ≥ 80% sure (79% of a mixed inbox) | **100%** |

It takes about **0.05 milliseconds** per message and costs nothing.
When it's unsure, the router sends the message to a smarter model instead.

> These numbers use the dataset's own labels. After you run `python run.py label`, the model learns from the **LLM's** labels instead.
> Cost and speed for V2 vs V4 come from `python run.py benchmark`.

---

## Setup

You need a Mac or Linux computer and Python 3.10+.

**Use whichever LLM you have.** Put ONE key in `.env` and the project uses it automatically:

| Provider | Default model | Get a key |
|---|---|---|
| OpenAI | `gpt-6.1-sol` | [platform.openai.com](https://platform.openai.com/api-keys) |
| Anthropic | `claude-opus-5-5` | [console.anthropic.com](https://console.anthropic.com/) |
| Gemini | `gemini-3.8-flash` | [aistudio.google.com](https://aistudio.google.com/apikey) (has a **free tier**) |
| No key | `llama3.2:3b` on your laptop via Ollama | free, see below |

Have more than one key? Set `LLM_PROVIDER=openai` (or `anthropic`, `gemini`, `ollama`) in `.env`.
You can change any model name in `.env` too.

```bash
git clone git@github.com:AnkitSurana/agent-on-a-budget.git
cd agent-on-a-budget

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then open .env and paste ONE API key
```

**Optional: Ollama, the free local LLM** (layer 2 of the router, or everything if you have no key):

```bash
brew install ollama
ollama serve                 # leave this running in its own terminal
ollama pull llama3.2:3b      # about 2 GB
```

Without Ollama the project still works. The router simply skips layer 2.

## How to run it

```bash
python run.py data                  # 1. make the data files
python run.py chat --version 1      #    talk to V1 (no tools)
python run.py chat --version 2      #    talk to V2 (with tools)
python run.py label                 # 2. the LLM labels the training messages (a few minutes, under $1)
python run.py train                 # 3. train the ML model
python run.py route                 #    see which layer answers each messy message (free)
python run.py chat --version 4      #    talk to V4 (the router)
python run.py benchmark --n 30      # 4. compare V2 and V4 on 30 messages (costs a little)
streamlit run app.py                #    the demo app: chat + charts
```

## Notebooks: the step-by-step version

The notebooks show every step as a beginner would try it, mistakes included.

| Notebook | What happens | Needs |
|---|---|---|
| `01_look_at_the_data` | Explore the messages and orders | nothing |
| `02_first_chat_with_an_llm` | First API call, cost per message, V1 → V2 | API key (or Ollama) |
| `03_llm_labels_ml_learns` | The LLM labels, 3 ML models compared, picking a confidence threshold | nothing (API key for the LLM's labels) |
| `04_hello_ollama` | Run a free LLM on your laptop, quantization | Ollama |
| `05_the_router` | V4 router, V2 vs V4 comparison | API key or Ollama |

## Project layout

```
budget_agent/
  config.py       all settings: models, prices, thresholds
  data.py         make and load the data
  tools.py        the shop "database" and the agent's tools
  llm.py          talk to OpenAI / Claude / Gemini / Ollama, measure time and cost
  agent.py        V1 and V2: the agent loop
  labeler.py      V3: the LLM labels messages
  classifier.py   V3: the small ML model
  router.py       V4: ML → local LLM → cloud LLM
run.py            command line for everything
app.py            Streamlit demo app
notebooks/        step-by-step notebooks
tests/            tests (no internet or API key needed)
```

## Tests

```bash
pytest
```

The tests use fake LLMs, so they're free and run offline. They check the tools and their rules,
the agent loop, labelling, the ML model, and the router's decisions.

## Safety rules built in

- The LLM can only *ask* for tools. Our Python code decides what really happens.
- Refunds above ₹10,000 always go to a human, no matter what the LLM says.
- Orders can only be cancelled or changed while they are "Processing".
- Anything that changes money or orders always goes to the big cloud LLM, never to the small models.

## What I'd add next

- Retrain the ML model regularly on new messages the LLM has labelled
- Cache the LLM's answers to repeated questions
- Track answer quality, not only cost and speed
