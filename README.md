# 🛒 Agent on a Budget

**An AI support agent that gets cheaper and faster by teaching a small Machine Learning model to do most of its work.**

Most AI agents (AIs that can act, like looking up an order or starting a refund, not just write text) send *every* customer message to a big, expensive AI model (an LLM, a large language model like the one behind ChatGPT). That is like a support office where the most expensive expert also answers every phone call.
This project builds an AI agent, lets the big model **teach a small ML model** to take the easy questions, and then sends each message to the **cheapest helper that can answer it**.

> Built for the workshop **"Build Your First AI Agent: From Machine Learning & LLMs to Real-World Workflows"**.

## Contents

1. [What this project is](#1-what-this-project-is)
2. [What you need](#2-what-you-need)
3. [Setup, step by step](#3-setup-step-by-step)
   - [Step 8: start the project, every time](#step-8-start-the-project-every-time) (environment on, Ollama running)
4. [Run the notebooks, in order](#4-run-the-notebooks-in-order)
5. [Run the app](#5-run-the-app)
6. [Run it from the command line](#6-run-it-from-the-command-line)
7. [How a message is answered](#7-how-a-message-is-answered)
8. [What is saved, stored and replayed](#8-what-is-saved-stored-and-replayed)
9. [Settings you can change](#9-settings-you-can-change)
10. [No laptop setup? Use Google Colab](#10-no-laptop-setup-use-google-colab)
11. [Troubleshooting](#11-troubleshooting)
12. [Project files](#12-project-files)

---

## 1. What this project is

NovaMart is a pretend online shop in India. Its customers send messages like "where is my order", "I want a refund" or "cash on delivery hai kya?". Four helpers answer them, like people in a support office:

| Helper | What it really is | Where it runs | What it answers | Example message | Cost |
|---|---|---|---|---|---|
| 🟢 **Receptionist** | A small ML model, trained on labels made by the big LLM | Your laptop | Simple questions it is at least 80% sure about | "track my order NM10009" | $0, about 0.05 ms |
| 🟡 **Junior assistant** | A small LLM (Llama 3.2: 3 billion learned numbers, called parameters) running in **Ollama** | Your laptop | Policy questions it is 40% to 80% sure about | "cash on delivery hai kya?" | $0, about 1 second |
| 🔵 **Senior expert** | A big LLM (OpenAI, Claude or Gemini) that can use **tools**: look up orders, start refunds, cancel orders | The cloud | Refunds, cancellations, complaints, and anything unclear | "my order NM10002 arrived broken, I want a refund" | A fraction of a cent |
| 🔴 **Manager** | A human | | Refunds above ₹10,000 and very upset customers, handed over by the senior expert | "refund NM10018, I don't like the chair" | Staff time |

The workshop builds this in three chapters: first you build a working agent, then two steps put it on a budget.

1. **Build your first agent.** First an LLM that can only talk. Then the same LLM with tools, so it can look things up and take actions. This becomes the senior expert.
2. **The LLM teaches a small ML model: budget step 1.** The LLM labels 1,080 customer messages once (for about 7 cents). A small, classic ML model learns from those labels. This becomes the receptionist.
3. **The whole support office: budget step 2.** The receptionist, the junior assistant, the senior expert and the manager work together. Each message stops at the cheapest helper that can answer it.

The three chapters are built in 7 steps, one notebook at a time: see [section 4](#4-run-the-notebooks-in-order).

**The result:** on 60 test messages, the whole office was **1.9 times cheaper and 1.6 times faster** than sending everything to the senior expert ([details](#7-how-a-message-is-answered)).

---

## 2. What you need

| | What | Why |
|---|---|---|
| A laptop | Mac, Windows 10/11 or Linux, with 8 GB of memory (16 GB is better) | Everything runs on it |
| Disk space | About 3 GB free | 0.7 GB for the project, 2 GB for the Ollama model |
| Internet | For the setup | Downloads, and live answers from the senior expert |
| An API key (optional) | A password for an AI service: OpenAI, Anthropic or Google (Gemini has a free tier) | So the senior expert can answer your **own** questions. See step 6 |
| Time | About 20 minutes for the setup | |

You do **not** need Python or Anaconda installed: the setup installs its own Python, only for this project.

> **No API key? Don't worry.** Every example (all notebooks and all app buttons) is replayed for free from answers a real LLM gave once. Only your own new questions need a key. See [section 8](#8-what-is-saved-stored-and-replayed).

---

## 3. Setup, step by step

Do these steps in order. Each step says how to check that it worked.

### Step 1: Open a terminal

A terminal is a window where you type commands.

- **Mac:** press `Cmd + Space`, type **Terminal**, press Enter.
- **Windows:** press the Windows key, type **PowerShell**, press Enter. (Use PowerShell, not "Command Prompt".)
- **Linux:** press `Ctrl + Alt + T`.

To paste into the terminal: `Cmd + V` on Mac, `Ctrl + V` on Windows, `Ctrl + Shift + V` on Linux.

### Step 2: Install uv

[uv](https://docs.astral.sh/uv/) is a free tool that installs Python and every package this project needs, inside the project folder only.

Mac and Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Windows (PowerShell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Now close the terminal and open a new one.** Then check:
```bash
uv --version
```
You should see a version number, such as `uv 0.10.5`.

### Step 3: Get the project

With git:
```bash
git clone https://github.com/AnkitSurana/agent-on-a-budget.git
cd agent-on-a-budget
```

Without git: open https://github.com/AnkitSurana/agent-on-a-budget, click the green **Code** button, then **Download ZIP**. Unzip it, then in the terminal go into the folder, for example `cd ~/Downloads/agent-on-a-budget-main` (Windows: `cd $HOME\Downloads\agent-on-a-budget-main`).

**Check:** type `ls` (Windows: `dir`). You should see `README.md`, `app.py`, `run.py` and the folders `notebooks`, `budget_agent` and `data`.

**From now on, always run commands from inside this folder.**

### Step 4: Install the project

```bash
uv sync
```

This takes 1 to 3 minutes the first time and downloads about 500 MB. It creates a folder called `.venv` inside the project: a private Python environment named **`agent-on-a-budget`**. It does **not** change anything else on your computer: your other Python installations (Anaconda, the system Python, other projects) are never touched.

**Check:** the last line says something like `Installed 57 packages`.

**Important:** start every command with **`uv run`**, as this README does. `uv run` always uses the project's own environment. If you type `python` or `jupyter` without it, your computer may use another Python (for example Anaconda's), and the project will warn: "This is not the project's Python".

To turn the project's environment on in your terminal, see step 8.

### Step 5: Install and run Ollama (the junior assistant)

[Ollama](https://ollama.com) is a free app that runs AI models on your own laptop. The junior assistant runs in it.

**5a. Install Ollama**

- **Mac:** download [Ollama.dmg](https://ollama.com/download/Ollama.dmg), open it, and drag Ollama into Applications.
- **Windows:** download [OllamaSetup.exe](https://ollama.com/download/OllamaSetup.exe) and run it.
- **Linux:** in the terminal: `curl -fsSL https://ollama.com/install.sh | sh`

**5b. Start Ollama**

Ollama must be **running** whenever you use the junior assistant.

- **Mac:** open **Ollama** from Applications. A llama icon appears in the menu bar at the top of the screen. It keeps running in the background.
- **Windows:** open **Ollama** from the Start menu. A llama icon appears in the taskbar, near the clock.
- **Linux:** open a **second** terminal and run `ollama serve`. Keep that terminal open.

**Check:** open http://localhost:11434 in your browser. It should say **"Ollama is running"**.

**5c. Download the model** (once, about 2 GB, a few minutes):
```bash
ollama pull llama3.2:3b
```

**5d. Test it:**
```bash
ollama list
```
You should see `llama3.2:3b` and `2.0 GB`. Then ask it something:
```bash
ollama run llama3.2:3b "Say hello in one short sentence."
```
After a few seconds it answers, for example "Hello." The first answer is slower because the model is loading into memory.

**Stopping Ollama:** Mac: click the llama icon in the menu bar, then **Quit Ollama**. Windows: right-click the llama icon in the taskbar, then **Quit**. Linux: press `Ctrl + C` in the terminal running `ollama serve`. Remember to start it again before the next session.

### Step 6: Get an API key and add it

The senior expert is a big LLM in the cloud. To use it, you need an **API key**: a secret password that lets this project use your account. Pick **one** provider:

| Provider | Cost | Get a key |
|---|---|---|
| **Gemini** (Google) | **Free tier**: about 5 requests a minute, enough to chat | [aistudio.google.com/apikey](https://aistudio.google.com/apikey): sign in with Google, click **Create API key**, copy it |
| **OpenAI** | Pay as you go: add $5 of credit, which lasts a very long time here | [platform.openai.com/api-keys](https://platform.openai.com/api-keys): sign in, add credit under **Settings → Billing**, then **Create new secret key**, copy it |
| **Anthropic** (Claude) | Pay as you go: add $5 of credit | [console.anthropic.com](https://console.anthropic.com/): sign in, add credit under **Billing**, then **API Keys → Create Key**, copy it |

A key is shown only once, so paste it somewhere safe straight away. Treat it like a password: never share it, post it, or show it on screen.

**6a. Create your settings file** by copying the template:
```bash
cp .env.example .env          # Windows: copy .env.example .env
```

**6b. Open it:**
- Mac: `open -e .env`
- Windows: `notepad .env`
- Linux: `nano .env` (save with `Ctrl + O`, Enter; close with `Ctrl + X`)

**6c. Fill in two lines**, with no spaces and no quotes, then save the file. The names are the same for every provider:
```
LLM_PROVIDER=gemini           # or openai, anthropic
LLM_API_KEY=AIza...           # your key
```
Keys start with `AIza` (Gemini), `sk-proj-` (OpenAI) or `sk-ant-` (Anthropic).

`.env` is your private file. It is never uploaded: git is set up to ignore it. (On a Mac, files starting with a dot are hidden in Finder; press `Cmd + Shift + .` to see them.)

**6d. Test the key with a live question:**
```bash
uv run python run.py chat --version 1
```
Type a question nobody asked before, for example `what is your return policy for gifts?`, and press Enter. You get an answer and its cost, **without** the `♻️ saved answer` sign: that means it came live from your provider. Type `quit` to stop.

No key yet? Skip this step: every example still works. You can add the key later, even while the app is running.

### Step 7: Train the receptionist, then check everything

**7a. Train the receptionist** (once, a few seconds). It is the small ML model you meet in chapter 2; git does not download trained models, so every new copy trains its own:
```bash
uv run python run.py train
```
You should see `Accuracy on clean test messages: 92.4%` and `Saved to models/intent_classifier.joblib`.

**7b. Check everything:**
```bash
uv run python run.py check
```

| Line | Should be | If not |
|---|---|---|
| Python and every package are installed | ✅ | Run `uv sync` again (step 4) |
| 🟢 Receptionist (ML model) is trained | ✅ | Run step 7a. (Notebook 03 and the chapter 3 commands also train it the first time) |
| 🟡 Junior assistant: Ollama is running | ✅ | Start Ollama (step 5b) and download the model (step 5c) |
| 🔵 Senior expert | ✅ with your provider's name, or ⚠️ "LLM_API_KEY is empty" if you have no key | No key is fine: the examples replay for $0. With a key, check your `.env` (step 6) |
| Saved answers ON | ♻️ | Questions asked before come from the store. `REPLAY=off` in `.env` makes every question live |

### Step 8: Start the project (every time)

Setup is done once. These steps you do **every time** you open a new terminal to work on the project.

**8a. Go to the project folder:**
```bash
cd agent-on-a-budget
```

**8b. Turn on the project's Python environment:**
```bash
source .venv/bin/activate          # Mac and Linux
.venv\Scripts\activate             # Windows (PowerShell)
```
Your prompt now starts with **`(agent-on-a-budget)`**: you are using the project's own Python. If it shows `(base)` too, run `conda deactivate` first (that is Anaconda). To turn the environment off later: `deactivate`.

The commands in this README start with `uv run`, which always uses the project's environment, so they work even if you skip 8b.

**8c. Start Ollama** (the junior assistant), if it is not running yet:
- **Mac / Windows:** open the **Ollama** app. A llama icon appears in the menu bar or taskbar.
- **Linux:** in a **second** terminal: `ollama serve` (keep it open).

**8d. Check Ollama is running:** open http://localhost:11434 in your browser. It says **"Ollama is running"**. Then ask the model one thing (the first answer takes a few seconds while it loads):
```bash
ollama run llama3.2:3b "Say hello"
```

**8e. Check everything:** `uv run python run.py check`. Now open the notebooks (section 4) or the app (section 5).

---

## 4. Run the notebooks, in order

The notebooks are where the whole story is told, step by step: the code, what it prints, and why.

### Start Jupyter

Jupyter shows the notebooks in your browser: pages where you run code one small step at a time and see each result right under it.

Do [step 8](#step-8-start-the-project-every-time) first (project folder, environment on, Ollama running), then:
```bash
uv run --group notebooks jupyter lab
```
The first time, this adds the notebook tools (about 200 MB, a minute or two). Then Jupyter opens in your browser at http://localhost:8888. The terminal stays busy while Jupyter runs: that is normal. Leave it open.

### Open and run a notebook

1. In the file list on the left, double-click the **`notebooks`** folder.
2. Double-click a notebook, for example **`01_look_at_the_data.ipynb`**.
3. Run it all with **Run → Run All Cells**. Or run one cell at a time with `Shift + Enter`, reading each result before going on (best for learning).
4. To start a notebook again from the top: **Kernel → Restart Kernel and Run All Cells**.

Each notebook runs in under 15 seconds. When you are done, close the browser tab and press `Ctrl + C` in the terminal (press `y` if it asks).

### The notebooks

Go through them in this order: it is the order of the live workshop. Every notebook works without an API key, because its LLM answers are replayed from the store.

| Step | Chapter | Open | Then |
|---|---|---|---|
| 1 · Look at the data | 1 · Build your first agent | `01_look_at_the_data` | |
| 2 · Make an LLM talk, and see what it costs | 1 | `02_first_chat_with_an_llm` | |
| 3 · Give it tools, a loop and guardrails | 1 | `02_first_chat_with_an_llm` | `06_try_it_yourself`, chapter 1 |
| 4 · Train a small ML model from the LLM's labels | 2 · The LLM teaches ML | `03_llm_labels_ml_learns` | `06_try_it_yourself`, chapter 2 |
| 5 · Add a free LLM on your laptop | 3 · The whole support office | `04_hello_ollama` | |
| 6 · Build the router: the cheapest helper first | 3 | `05_the_router` | `06_try_it_yourself`, chapter 3 |
| 7 · Put it in a web app | 3 | `uv run python run.py app` ([section 5](#5-run-the-app)) | |

**1. `01_look_at_the_data`: chapter 1, step 1, meet the shop.** The 1,620 customer messages and their 27 intents (what the customer wants, such as `track_order` or `get_refund`), split into 1,080 to learn from and 540 to test with; the 50 messy messages with typos and Hinglish; and the 300 made-up orders. Look for: the orders used in every demo (NM10009, NM10002, NM10018).

**2. `02_first_chat_with_an_llm`: chapter 1, steps 2 and 3, build your first agent.** Your first LLM call, its tokens and its cost; the LLM alone, which can't see orders; the same LLM with tools; the agent loop written by hand in about 15 lines; a refund; and two ways to reach a human: an angry customer, and the safety rule that sends a ₹12,999 refund to a person. Look for: the list of tools the agent called for each message, and the cost of each answer.

**3. `03_llm_labels_ml_learns`: chapter 2, step 4, the LLM teaches a small ML model.** The LLM's labels (91.5% match the true ones); each model trained on the 1,080 and tested on the clean and the messy messages; three tries compared; the settings that make the final model sure when it is right; and the confidence table that decides the 80% rule. Look for: "if the model is X% sure, how often is it right", and the speed: about 0.05 milliseconds per message.

**4. `04_hello_ollama`: chapter 3, step 5, meet the junior assistant.** Llama 3.2 running on your laptop through Ollama: which models you have, why a 3-billion-number model fits in 2 GB (quantization: each number stored in fewer bits), how fast it answers, and what it should not do (it can't look up orders). **This one needs Ollama running**, because it talks to Ollama live.

**5. `05_the_router`: chapter 3, step 6, the whole support office.** Who would answer each message, the routing rule written by hand and checked against the project's code, one example per helper, and the expert alone versus the whole office on 60 messages. Look for: the 1.9 times cheaper result, and real answers side by side.

**6. `06_try_it_yourself`: every hands-on exercise.** Do each part right after its notebook: chapter 1 after 02, chapter 2 after 03, chapter 3 after 05. Call the agent's tools yourself, ask the receptionist your own messages, turn the confidence dials, and talk to the whole office. It also runs in Google Colab: see [section 10](#10-no-laptop-setup-use-google-colab).

**After the workshop: `00_the_whole_story`.** The complete flow in one notebook, from the data through all three chapters to the final savings chart, in about 10 minutes of reading. Use it to recap, or to catch up if you missed a part.

### Notebooks in VS Code instead of Jupyter

Open the project folder in VS Code, open a notebook, click **Select Kernel** (top right), then **Python Environments**, then the one in `.venv`. To see it by name (**agent-on-a-budget**) in any Jupyter, register it once:
```bash
uv run --group notebooks python -m ipykernel install --user --name agent-on-a-budget --display-name "agent-on-a-budget"
```

---

## 5. Run the app

The app shows the whole support office with buttons. Do [step 8](#step-8-start-the-project-every-time) first (project folder, environment on, Ollama running). Then, in a terminal inside the project folder (a second one if Jupyter is running in the first):
```bash
uv run python run.py app
```
After a few seconds your browser opens http://localhost:8501. The terminal stays busy while the app runs: that is normal. The very first time, the terminal says *"training it now"* and then *"Receptionist trained in … s"*: that is the ML model being built on your laptop, once.

### The 💬 Chat tab

At the top, pick a chapter:

| Chapter | What answers | What to notice |
|---|---|---|
| **Chapter 1a: LLM only (can talk)** | The cloud LLM, with no tools | It can't see orders, so it apologises or guesses |
| **Chapter 1b: LLM + tools (the expert)** | The same LLM, with tools | It looks up the order, reads the policy and starts the refund |
| **Chapter 3: the whole support office** | Whichever helper is cheapest | Each message stops at a different helper |

Then click the four example buttons, one per helper (🟢 🟡 🔵 🔴), or type your own message at the bottom. Under each answer you see:

- **who answered**, for example `🟢 Receptionist (ML model)` or `🔵 Senior expert (cloud LLM agent) → 🔴 Manager (human)`
- **the intent and how sure** the receptionist was, for example `track_order (93%)`
- **the time and the cost** of the answer
- **`♻️ saved answer, $0 this time`** if the answer was replayed from the store (the cost shown is what it cost when it was first made)

You can click the buttons in any order and as often as you like: each message starts from the original orders.

### The 📊 Expert only vs the whole office tab

The same 60 messages sent to the senior expert alone, then through the whole office: the cost, the time, and who answered each one.

**Stop the app:** press `Ctrl + C` in its terminal.

---

## 6. Run it from the command line

The same three chapters, in the terminal. In a chat, type your message and press Enter; type `quit` to stop.

| Command | What it does | Needs |
|---|---|---|
| `uv run python run.py check` | Shows what works on this laptop and what to fix | Nothing |
| `uv run python run.py chat --version 1` | Chapter 1: chat with the LLM alone (no tools) | A key for new questions |
| `uv run python run.py chat --version 2` | Chapter 1: chat with the LLM with tools (the senior expert) | A key for new questions |
| `uv run python run.py label` | Chapter 2: the LLM labels the 1,080 training messages again (a few minutes, under $1) | A key |
| `uv run python run.py train` | Chapter 2: train the receptionist again from the labels | Nothing |
| `uv run python run.py route` | Chapter 3: show which helper would answer each messy message (no LLM is called) | Nothing |
| `uv run python run.py chat --version 4` | Chapter 3: chat with the whole office | Ollama; a key for new questions |
| `uv run python run.py benchmark --n 30` | Chapter 3: the expert alone versus the whole office on 30 messages | A key (it rewrites the results the app shows) |
| `uv run python run.py app` | Start the app | Ollama |
| `uv run --group data python run.py data` | Rebuild the data files from the internet (they are already included) | Internet |
| `uv run pytest` | Run the 55 tests (free and offline, with fake LLMs) | Nothing |

Try this in `chat --version 4`: `track my order NM10005`, then `how do i reset my pasword`, then `refund NM10018, I don't like the chair`. Watch which helper answers each one.

---

## 7. How a message is answered

1. **The receptionist reads it first.** The small ML model guesses the intent (one of 27, such as `track_order`) and how sure it is (0% to 100%). This takes about 0.05 ms and no LLM is called.
2. **Then the router decides who answers:**
   - At least **80%** sure and a simple question: the **receptionist** answers with a ready-made reply (looking up the order if needed).
   - **40% to 80%** sure and a **policy** question, and Ollama is running: the **junior assistant** answers from the shop policy.
   - Everything else: the **senior expert**.
   - Two rules beat everything: anything that **changes money or orders** (refunds, cancellations, address changes, complaints) **always** goes to the senior expert, however sure the receptionist is. And **order questions never go to the junior**, because it can't look orders up.
3. **The senior expert works in a loop:** it reads the message, asks for a tool (for example `lookup_order`), our Python code runs the tool and sends back the result, and this repeats until the expert writes its answer. It stops after 8 steps at most.
4. **Safety rules live in the Python code, not in the AI's instructions,** so the AI can't talk its way past them:
   - Refunds only for delivered orders, within 30 days of delivery.
   - Refunds above **₹10,000 go to a human** (the manager). The refund tool refuses and opens a ticket.
   - Orders can only be cancelled or changed while they are still "Processing".

**Measured results** (OpenAI `gpt-5.4-mini`, October 2026):

| | Result |
|---|---|
| Cost to label 1,080 messages with the LLM | $0.07 |
| LLM labels that match the true labels | 91.5% |
| Receptionist accuracy on clean / messy messages | 92.4% / 58% |
| When the receptionist is at least 80% sure | it answers 72% of messages, and 98.3% of those are right |
| Expert alone vs whole office, per 1,000 messages | $0.99 vs $0.52 (1.9 times cheaper) |
| Expert alone vs whole office, average wait | 1.70 s vs 1.04 s (1.6 times faster) |
| Messages answered for $0 by the whole office | 55% (receptionist 43%, junior 12%) |

---

## 8. What is saved, stored and replayed

Three different things are kept in this project. They are easy to mix up:

| What | What it is | Where it is stored | How it is made |
|---|---|---|---|
| **The receptionist** (ML model) | The trained intent classifier | `models/intent_classifier.joblib` on your laptop. Not in git: each laptop makes its own | Trained on your laptop the first time it is needed (a few seconds), from `data/messages.csv` and `data/llm_labels.csv`. Then it **really runs on every message**: it is instant and free, so it is never replayed |
| **The LLM's labels** | The cloud LLM's intent for each of the 1,080 training messages | `data/llm_labels.csv` (in git) | Made once by the cloud LLM, for $0.07 (`run.py label`) |
| **Stored LLM answers** | Answers from the cloud LLM (the senior expert) **and** from Ollama (the junior assistant) | `data/saved_llm_answers.jsonl` (in git), 135 answers | Saved automatically every time an LLM really answers |

**How replay works:** before calling an LLM, the project checks whether this **exact** request (same provider, model, instructions and conversation) was answered before.

- **Yes:** it reuses the stored answer. It is instant, costs $0, works without internet, and you see `♻️ saved answer, $0 this time`.
- **No:** it asks the LLM live (the cloud needs your key; the junior needs Ollama running), and stores the new answer for next time.

So every **example** works even without a key. **Your own new questions** go to the cloud LLM if you have a key. With no key, a plain question goes to the free local model in Ollama (slower and less smart, but free), and a new request for the senior expert, like a refund, is handed to a human with a message that it needs a key. To always ask live, put `REPLAY=off` in `.env`. Reusing answers like this is called **caching**, one of the main ways to make AI cheaper and faster.

**Ollama's model** (Llama 3.2, 2 GB) is separate again: Ollama keeps it in its own folder (on a Mac: `~/.ollama/models`), downloaded by `ollama pull`. It is not part of this project.

---

## 9. Settings you can change

**In `.env`** (your private settings file, step 6). Changes apply on the next question, even while the app or a notebook is running:

| Setting | What it does | Example |
|---|---|---|
| `LLM_PROVIDER` | Which cloud LLM is the senior expert | `openai`, `anthropic`, `gemini` (or `ollama` to use the laptop model for everything) |
| `LLM_API_KEY` | Your key for that provider | `sk-proj-...` |
| `LLM_MODEL` | A different model from the same provider | `gpt-5.4-mini` (the default for OpenAI) |
| `REPLAY` | `on`: reuse stored answers (the default). `off`: always ask live | `on` |
| `LOCAL_MODEL` | The junior assistant's model in Ollama | `llama3.2:3b` |
| `OLLAMA_URL` | Where Ollama runs (found automatically, also from Docker) | `http://localhost:11434` |

The stored answers were made with OpenAI `gpt-5.4-mini`. With another provider or model, each example is asked live the first time (a fraction of a cent), then stored.

**In `budget_agent/config.py`** (the code's settings):

| Setting | Default | What it does |
|---|---|---|
| `ML_CONFIDENCE` | `0.80` | How sure the receptionist must be to answer. Higher is safer but sends more to the paid expert |
| `LOCAL_LLM_CONFIDENCE` | `0.40` | The lowest confidence the junior may take |
| `DEMO_MESSAGES` | the four examples | The messages on the app's buttons |

---

## 10. No laptop setup? Use Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AnkitSurana/agent-on-a-budget/blob/main/notebooks/06_try_it_yourself.ipynb)

Notebook 06 (every hands-on exercise) also runs in your browser, on Google's computers:

1. Click **Open in Colab** above, and sign in with a Google account if asked.
2. Click the first code cell and press `Shift + Enter`. If Colab warns that the notebook was not written by Google, click **Run anyway**.
3. After about 20 seconds you see "Ready: the project is loaded". Run the other cells one by one.

Nothing is installed on your laptop and no key is needed. Two differences: Colab can't run Ollama, so the junior's questions go to the senior expert instead; and Colab forgets everything after a while idle, so if a cell fails, run the first cell again.

---

## 11. Troubleshooting

**Setup**

| Problem | Fix |
|---|---|
| `uv: command not found` or `'uv' is not recognized` | Close the terminal and open a new one. Still failing? Repeat step 2 |
| Windows: "running scripts is disabled on this system" | Use the exact command in step 2: it includes `-ExecutionPolicy ByPass` |
| Windows with an ARM (Snapdragon) chip: an error about `httptools` | Run `uv sync --python cpython-3.12-windows-x86_64-none` |
| `No such file or directory: pyproject.toml` | You are not inside the project folder. `cd agent-on-a-budget` first |
| "This is not the project's Python", or the app crashes with `unexpected keyword argument 'width'` | The command ran without `uv run`, on another Python (often Anaconda's). Run it again starting with `uv run` |
| The prompt shows `(base)` | That is Anaconda. It does no harm if you use `uv run`. See step 8b to switch the terminal |
| The terminal shows `(.venv)` or the wrong Python | An old `.venv` made by another tool. Delete the `.venv` folder and run `uv sync` again |

**Ollama**

| Problem | Fix |
|---|---|
| "Could not reach Ollama", or the junior's messages go to the senior expert | Ollama isn't running. Start it (step 5b) and check http://localhost:11434 |
| `ollama: command not found` | Ollama isn't installed, or the terminal was opened before installing it. Install it (step 5a) and open a new terminal |
| "Ollama doesn't have the model" | Run `ollama pull llama3.2:3b` (step 5c) |
| The first junior answer is slow | Normal: the model loads into memory the first time (a few seconds) |

**API key**

| Problem | Fix |
|---|---|
| "Your ... API key was not accepted" | Check `LLM_API_KEY` in `.env`: the full key, no spaces, no quotes. Check that `LLM_PROVIDER` matches the key |
| `check` says the key "looks like a key for openai, but LLM_PROVIDER is gemini" | `LLM_PROVIDER` and the key don't match. Fix one of them |
| "too many requests, or no credit left" | Wait a minute (Gemini's free tier allows about 5 requests a minute), or add credit on your provider's billing page |
| Your own questions get "Could not reach Ollama" | There is no key, so new questions went to Ollama. Add a key (step 6) or start Ollama |

**Notebooks and app**

| Problem | Fix |
|---|---|
| `jupyter: command not found`, or `No module named jupyterlab` | Start it with `uv run --group notebooks jupyter lab` (section 4), not `jupyter lab` |
| In a notebook: `No module named 'budget_agent'` or `'sklearn'` | Jupyter was started without `uv run`, or from outside the project. Close it and start it as in section 4 |
| Notebook 04 fails | Ollama isn't running. Start it (step 5b) and run the notebook again |
| "Port 8501 (or 8888) is already in use" | The app (or Jupyter) is already running in another terminal. Use that one, or stop it with `Ctrl + C` |
| The app's terminal shows "training it now" and nothing happens | It finished: the next line says "Receptionist trained in … s". The terminal stays busy while the app runs. Look at the browser |

---

## 12. Project files

```
notebooks/          the step-by-step notebooks: 01 to 06 in order, 00 is the recap
  images/             two diagrams used by notebook 00
budget_agent/       the agent's code
  config.py           all settings: models, prices, thresholds, helper names, example messages
  data.py             makes and loads the data
  tools.py            the shop "database" and the agent's tools, with the safety rules
  llm.py              talks to OpenAI, Claude, Gemini and Ollama, and measures time and cost
  agent.py            chapter 1: the agent loop (the senior expert)
  labeler.py          chapter 2: the LLM labels messages
  classifier.py       chapter 2: the small ML model (the receptionist)
  router.py           chapter 3: the whole office
  replay.py           stored answers: reuse an LLM answer instead of paying again
data/
  messages.csv              1,620 customer messages with 27 intents (from the public Bitext dataset)
  messy_messages.csv        50 hand-written hard messages: typos, Hinglish, anger, two requests in one
  orders.csv                300 made-up orders
  policy.md                 the shop's refund, delivery and payment rules
  llm_labels.csv            the LLM's labels for the 1,080 training messages
  saved_llm_answers.jsonl   stored LLM answers (cloud and Ollama), replayed for free
  benchmark_results.csv     the expert alone versus the whole office, 60 messages
models/             the trained receptionist, made on your laptop on first use (not in git)
app.py              the Streamlit app
run.py              the command line
tests/              55 tests with fake LLMs: free, no internet needed
pyproject.toml      the packages (plus the optional notebook and data tools) and the Python version
uv.lock             the exact package versions, the same on every computer
.env.example        the settings template: copy it to .env (step 6)
.gitignore          keeps .env, .venv and models/ out of git
```
