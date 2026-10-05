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
| 1. Give an LLM hands | Version 1 (LLM only), Version 2 (LLM + tools = the senior expert) | `uv run python run.py chat --version 1` / `--version 2` |
| 2. The LLM teaches ML | Version 3 (label, then train the receptionist) | `uv run python run.py label`, then `uv run python run.py train` |
| 3. The whole office | Version 4 (the router) | `uv run python run.py chat --version 4` |

## The shop: NovaMart

NovaMart is a pretend online shop in India. The agent works with:

| File | What's inside |
|---|---|
| `data/messages.csv` | 1,620 customer messages (27 intents × 60) from the [Bitext customer support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), split into train and test |
| `data/messy_messages.csv` | 50 hand-written hard messages: typos, Hinglish, anger, two requests in one |
| `data/orders.csv` | 300 fake orders (made by `uv run python run.py data`) |
| `data/policy.md` | The shop's refund, delivery and payment rules |

## Results so far

Real runs with OpenAI `gpt-5.4-mini`, October 2026. Your numbers will vary a little.

**Chapter 2: the LLM teaches the ML model**

| | Result |
|---|---|
| Cost to label 1,080 messages with the LLM | **$0.07** |
| LLM labels that match the dataset's true labels | **91.5%** (most "mistakes" are close pairs like `check_invoice` vs `get_invoice`) |
| Receptionist accuracy on clean test messages | **92.4%** (98.3% if trained on perfect labels: the student can't beat its teacher) |
| Receptionist accuracy on messy messages | 58.0% |
| When it is at least 80% sure (72% of a mixed inbox) | **98.3%** right |
| Time per message | about **0.05 milliseconds**, for $0 |

**Chapter 3: expert only vs the whole office** (60 messages, a third of them messy; all four helpers running)

| | Expert only | Whole office |
|---|---|---|
| Cost for 60 messages | $0.059 | **$0.031** |
| Cost per 1,000 messages | $0.99 | **$0.52** |
| Average time per message | 1.70 s | **1.04 s** |
| Answered for $0 | 0% | **55%** (receptionist 43%, junior assistant 12%) |

The whole office was **1.9× cheaper and 1.6× faster**.
The receptionist's answers were as good as the expert's. The junior assistant (a 3B model on the laptop, about 55 tokens a second) was right on simple policy questions but sometimes made up details, which is why it only gets low-risk questions, and why `LOCAL_LLM_CONFIDENCE` is a dial you can turn up.
The saving grows as more messages are simple: refunds, cancels and complaints always go to the expert, on purpose.

**Every number, and every message with both answers, is in [docs/results.md](docs/results.md).** Rebuild that page any time, for free: `uv run python run.py report`.

## Saved answers: run it again for free

Every real LLM answer is saved in `data/saved_llm_answers.jsonl` (128 answers so far, which cost $0.07 to make).
When the app, a notebook, the chat or the benchmark sends **exactly the same request** again, the saved answer is replayed:
instantly, for $0, even without internet. Only new questions are sent to the LLM. You'll see `♻️ saved answer, $0 this time`.

- Re-running the whole 60-message benchmark from saved answers takes about 3 seconds and costs $0.
- The four demo messages, in the app and in every notebook, are all saved. A demo can't fail because of slow Wi-Fi.
- Want fresh answers? Put `REPLAY=off` in `.env`.
- Saved answers are matched by provider and model, so they're used when you run with the same LLM (OpenAI `gpt-5.4-mini` here).

The LLM's labels (`data/llm_labels.csv`) and the benchmark results (`data/benchmark_results.csv`) are included too. To redo everything from scratch, delete those files and run `uv run python run.py label`, `uv run python run.py train` and `uv run python run.py benchmark --n 60`.

---|---|
| Clean test messages | **98.3%** |
| Messy messages | 60.0% |
| Messages where it is ≥ 80% sure (79% of a mixed inbox) | **100%** |

It takes about **0.05 milliseconds** per message and costs nothing.
When the receptionist is unsure, the message goes to a smarter helper instead.

> These numbers use the dataset's own labels. After you run `uv run python run.py label`, the model learns from the **LLM's** labels instead.
> Cost and speed for the expert alone vs the whole office come from `uv run python run.py benchmark`.

---

## Setup

Works on **Mac, Windows and Linux**. You don't even need Python installed: [uv](https://docs.astral.sh/uv/) installs the right version for you.

📘 **New to this? Follow the [step-by-step install guide](docs/install-guide.md)** (about 10 minutes, every command for every system).

Quick version, if you've done this before:

```bash
git clone https://github.com/AnkitSurana/agent-on-a-budget.git
cd agent-on-a-budget
uv sync                     # installs Python 3.12 and every package
uv run pytest               # check: "35 passed"
```

Then copy `.env.example` to `.env` and paste **one** API key, whichever you have:

| Provider | Default model | Get a key |
|---|---|---|
| OpenAI | `gpt-5.4-mini` | [platform.openai.com](https://platform.openai.com/api-keys) |
| Anthropic | `claude-opus-5-5` | [console.anthropic.com](https://console.anthropic.com/) |
| Gemini | `gemini-3.8-flash` | [aistudio.google.com](https://aistudio.google.com/apikey) (has a **free tier**) |
| No key | `llama3.2:3b` on your laptop via [Ollama](https://ollama.com/download) | free |

Have more than one key? Set `LLM_PROVIDER=openai` (or `anthropic`, `gemini`, `ollama`) in `.env`.
You can change any model name in `.env` too.

Without Ollama the project still works. The office simply has no junior assistant.

## How to run it

```bash
uv run python run.py data                  # make the data files

# Chapter 1: build your first AI agent
uv run python run.py chat --version 1      # the LLM alone: it can only talk
uv run python run.py chat --version 2      # the LLM with tools: the senior expert

# Chapter 2: the LLM teaches a small ML model
uv run python run.py label                 # the LLM labels the training messages (a few minutes, under $1)
uv run python run.py train                 # the receptionist (ML model) learns from those labels

# Chapter 3: the whole support office
uv run python run.py route                 # who would answer each messy message (free)
uv run python run.py chat --version 4      # talk to the whole office
uv run python run.py benchmark --n 30      # expert only vs the whole office, 30 messages (costs a little)
uv run streamlit run app.py                # the demo app: one button per helper, plus charts
uv run python run.py report         # write docs/results.md from the saved results (free)
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
  replay.py       saved answers: replay an LLM answer instead of paying again
  report.py       writes docs/results.md from the saved results
run.py            command line for everything
app.py            Streamlit demo app
notebooks/        step-by-step notebooks
tests/            tests (no internet or API key needed)
docs/             install guide, theory guide, results, architecture diagram
data/             messages, orders, policy, the LLM's labels, benchmark, saved answers
pyproject.toml    the project's packages (used by uv)
uv.lock           exact package versions, the same on every computer
requirements.txt  the same packages, for people who prefer plain pip
```

## Tests

```bash
uv run pytest
```

The tests use fake LLMs, so they're free and run offline. They check the tools and their rules,
the agent loop, labelling, the ML model, and the router's decisions.

## If something goes wrong

The project explains common setup problems in one line instead of a long error:

| Message starts with | What to do |
|---|---|
| "Your ... API key was not accepted" | Check the key in `.env` (no spaces or quotes) |
| "Could not reach Ollama" | No key was found, so it tried the free local model. Open the Ollama app (Linux: `ollama serve`), or add a key to `.env` |
| "The model ... is not available" or "can't use tools" | Pick another model in `.env`, for example `OPENAI_MODEL=gpt-5.4-mini` |
| "too many requests, or no credit left" | Wait a minute, or add credit on your provider's billing page |

Only the project's own `.env` file is read, never one from a parent folder.
Install problems (uv, Windows, Ollama)? See the troubleshooting table in the [install guide](docs/install-guide.md#if-something-goes-wrong).

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
