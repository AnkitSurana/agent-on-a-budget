# 🛒 Agent on a Budget

**An AI support agent that gets cheaper and faster by teaching a small Machine Learning model to do most of its work.**

Most AI agents send *every* message to a big, expensive LLM. That's like sending every question in a support office to your most expensive expert.
This project builds the agent, lets the LLM **teach a small ML model** to handle the easy questions, and then routes each message to the **cheapest helper that can answer it**.

> Built for the workshop **"Build Your First AI Agent: From Machine Learning & LLMs to Real-World Workflows"**.

---

## The story in three chapters

The workshop follows its title, one chapter per part:

| Title | Chapter | What we build | The one idea |
|---|---|---|---|
| **Build Your First AI Agent** | 1. Give an LLM hands | An LLM that can only talk, then the same LLM with tools: look up orders, refund, cancel, hand over to a human | An agent = an LLM + tools + a loop |
| **From Machine Learning & LLMs** | 2. The LLM teaches a small ML model | The LLM labels 1,080 messages; a small ML model learns from those labels | The big model teaches, the small model does the easy work |
| **To Real-World Workflows** | 3. The whole support office | Each message goes to the cheapest helper that can answer it | Real systems use the right tool for each job |

## The support office

Every message goes to the cheapest helper that can handle it:

| Helper | What it is | Takes | Example message | Cost |
|---|---|---|---|---|
| 🟢 **Receptionist** | The small ML model | Simple questions it is at least 80% sure about | "track my order NM10009" → running shoes, shipped, arriving 8 Oct | $0, about 0.05 ms |
| 🟡 **Junior assistant** | A small LLM on the laptop (Ollama) | Policy questions it is 40% to 80% sure about | "kitne din me delivery hoti hai" | $0 |
| 🔵 **Senior expert** | The cloud LLM agent with tools | Refunds, cancels, address changes, complaints, anything unclear | "my order NM10002 arrived broken, I want a refund" → ₹5,999 refund started | Paid per token |
| 🔴 **Manager** | A human | Refunds above ₹10,000 and very upset customers, passed on by the expert | "refund NM10018, I don't like the chair" → ₹12,999, so a human must approve | Staff time |

```
                          Customer message
                                 │
                                 ▼
                 ┌──────────────────────────────┐
                 │ Receptionist (ML model)       │  at least 80% sure + simple → answers itself
                 └──────────────┬───────────────┘
                                │ less sure, policy question
                                ▼
                 ┌──────────────────────────────┐
                 │ Junior assistant (local LLM)  │  answers from the policy
                 └──────────────┬───────────────┘
                                │ action, complaint, or unsure
                                ▼
                 ┌──────────────────────────────┐
                 │ Senior expert (cloud LLM)     │  uses tools: orders, refunds, cancels
                 └──────────────┬───────────────┘
                                │ big refund or very upset customer
                                ▼
                         Manager (a human)
```

**New to these ideas?** Read the [theory guide](docs/theory-guide.md): every concept in plain English, a glossary, and a free reading list for the prerequisites.

**Full architecture diagram:** open [`docs/architecture/agent-on-a-budget.html`](docs/architecture/agent-on-a-budget.html) in a browser (interactive, light/dark, export to PNG). Made with [Archify](https://github.com/tt-a1i/archify) from `docs/architecture/architecture.json`.

### Chapters and code versions

The code calls the steps "versions". This is how they line up with the chapters:

| Chapter | Code | Command |
|---|---|---|
| 1. Give an LLM hands | Version 1 (LLM only), Version 2 (LLM + tools = the senior expert) | `python run.py chat --version 1` / `--version 2` |
| 2. The LLM teaches ML | Version 3 (label, then train the receptionist) | `python run.py label`, then `python run.py train` |
| 3. The whole office | Version 4 (the router) | `python run.py chat --version 4` |

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
When the receptionist is unsure, the message goes to a smarter helper instead.

> These numbers use the dataset's own labels. After you run `python run.py label`, the model learns from the **LLM's** labels instead.
> Cost and speed for the expert alone vs the whole office come from `python run.py benchmark`.

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

**Optional: Ollama, the free local LLM** (the junior assistant, or every helper if you have no key):

```bash
brew install ollama
ollama serve                 # leave this running in its own terminal
ollama pull llama3.2:3b      # about 2 GB
```

Without Ollama the project still works. The office simply has no junior assistant.

## How to run it

```bash
python run.py data                  # make the data files

# Chapter 1: build your first AI agent
python run.py chat --version 1      # the LLM alone: it can only talk
python run.py chat --version 2      # the LLM with tools: the senior expert

# Chapter 2: the LLM teaches a small ML model
python run.py label                 # the LLM labels the training messages (a few minutes, under $1)
python run.py train                 # the receptionist (ML model) learns from those labels

# Chapter 3: the whole support office
python run.py route                 # who would answer each messy message (free)
python run.py chat --version 4      # talk to the whole office
python run.py benchmark --n 30      # expert only vs the whole office, 30 messages (costs a little)
streamlit run app.py                # the demo app: one button per helper, plus charts
```

## Notebooks: the step-by-step version

The notebooks show every step as a beginner would try it, mistakes included. They use the same code as `budget_agent/`, written out by hand first where it matters, and the same example messages as the app.

| Notebook | Chapter | What happens | Needs |
|---|---|---|---|
| `01_look_at_the_data` | Setup | Explore the messages and orders | nothing |
| `02_first_chat_with_an_llm` | 1 | First API call, cost per message, then the agent loop written by hand (same as `agent.py`) | API key (or Ollama) |
| `03_llm_labels_ml_learns` | 2 | The LLM labels, 3 ML models compared, picking the receptionist's 80% rule | nothing (API key for the LLM's labels) |
| `04_hello_ollama` | 3 | Meet the junior assistant: a free LLM on your laptop | Ollama |
| `05_the_router` | 3 | The routing rule written by hand (checked against `router.py`), then expert only vs the whole office | API key or Ollama |

## Project layout

```
budget_agent/
  config.py       all settings: models, prices, thresholds, helper names, demo messages
  data.py         make and load the data
  tools.py        the shop "database" and the agent's tools
  llm.py          talk to OpenAI / Claude / Gemini / Ollama, measure time and cost
  agent.py        chapter 1: the agent loop (versions 1 and 2)
  labeler.py      chapter 2: the LLM labels messages
  classifier.py   chapter 2: the small ML model (the receptionist)
  router.py       chapter 3: the whole office (version 4)
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
- Anything that changes money or orders always goes to the senior expert, never to the receptionist or the junior.
- The junior assistant never answers order questions, because it can't look orders up.

## What I'd add next

- Retrain the ML model regularly on new messages the LLM has labelled
- Cache the LLM's answers to repeated questions
- Track answer quality, not only cost and speed
