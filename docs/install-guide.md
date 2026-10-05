# Install guide: Mac, Windows and Linux

This takes about **10 minutes**. You copy and paste every command, and you don't need to install Python yourself.

We use **[uv](https://docs.astral.sh/uv/)**, a free tool that installs the right Python version and every package the project needs, the same way on every computer.

| Step | What | Needed? |
|---|---|---|
| 1 | Open a terminal | Yes |
| 2 | Install uv | Yes |
| 3 | Get the project | Yes |
| 4 | Install the project | Yes |
| 5 | Check it works | Yes |
| 6 | Add an API key | Recommended |
| 7 | Install Ollama, a free AI model on your laptop | Optional |
| 8 | Run it | Yes |

---

## Step 1: Open a terminal

A terminal is a window where you type commands.

- **Mac:** press `Cmd + Space`, type **Terminal**, press Enter.
- **Windows:** press the Windows key, type **PowerShell**, press Enter. (Use PowerShell, not "Command Prompt".)
- **Linux:** press `Ctrl + Alt + T`, or open **Terminal** from your apps.

> Tip: paste with `Cmd + V` on Mac, `Ctrl + V` on Windows, and `Ctrl + Shift + V` on Linux.

---

## Step 2: Install uv

Copy the command for your computer, paste it into the terminal and press Enter.

**Mac and Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Now close the terminal and open a new one** (this is important). Then check that it worked:

```bash
uv --version
```

You should see something like `uv 0.12.23`. If you see "command not found" or "not recognized", close and reopen the terminal once more.

> Already use Homebrew on Mac? `brew install uv` works too. On Windows, `winget install --id=astral-sh.uv -e` also works.

---

## Step 3: Get the project

Pick **one** option.

### Option A: Download a ZIP (easiest, no extra tools)

1. Open https://github.com/AnkitSurana/agent-on-a-budget
2. Click the green **Code** button, then **Download ZIP**.
3. Unzip it. You get a folder called `agent-on-a-budget-main`. Move it somewhere easy to find, like your Documents folder.
4. In the terminal, go into that folder:

**Mac and Linux:**

```bash
cd ~/Documents/agent-on-a-budget-main
```

**Windows:**

```powershell
cd $HOME\Documents\agent-on-a-budget-main
```

> Tip: type `cd ` (with a space) and drag the folder onto the terminal window. It fills in the path for you.

### Option B: Use git (if you have it, or want it)

Check if you have git: `git --version`. If not, install it:

- **Mac:** run `xcode-select --install` and click Install.
- **Windows:** run `winget install --id Git.Git -e`, then close and reopen PowerShell.
- **Linux (Ubuntu/Debian):** run `sudo apt install git`

Then download the project and go into its folder:

```bash
cd ~/Documents
git clone https://github.com/AnkitSurana/agent-on-a-budget.git
cd agent-on-a-budget
```

(On Windows, the first line is `cd $HOME\Documents`.)

---

## Step 4: Install the project

Make sure your terminal is **inside the project folder** (Step 3), then run:

```bash
uv sync
```

The first time takes **1 to 3 minutes**. uv downloads Python 3.12 if you don't have it, and then every package. When it finishes you'll see a line like `Installed 151 packages`.

> **Windows laptop with a Snapdragon (ARM) chip?** Some newer Windows laptops ("Copilot+ PCs") use ARM chips. If `uv sync` fails with an error about `httptools`, run this instead:
>
> ```powershell
> uv sync --python cpython-3.12-windows-x86_64-none
> ```

---

## Step 5: Check it works

```bash
uv run pytest
```

You should see **`35 passed`** at the end. Then train the small ML model (the "receptionist"):

```bash
uv run python run.py train
```

You should see `Accuracy on clean test messages: 98.3%`.

Try it, with no API key needed yet:

```bash
uv run python run.py chat --version 4
```

Type `track my order NM10009` and press Enter. The receptionist answers instantly, for free. Type `quit` to stop.

---

## Step 6: Add an API key (recommended)

The "senior expert" needs a big cloud LLM. You need **one** key, from any one of these:

| Provider | Where to get a key | Cost |
|---|---|---|
| Gemini (Google) | https://aistudio.google.com/apikey | **Free tier available** (about 5 requests a minute: fine for chatting, slow for labelling and the benchmark) |
| OpenAI | https://platform.openai.com/api-keys | Paid, a few dollars is plenty |
| Anthropic (Claude) | https://console.anthropic.com/ | Paid, a few dollars is plenty |

No key at all? Skip to Step 7. The project can use a free model on your laptop instead.

**1. Make your settings file** by copying the example:

**Mac and Linux:**

```bash
cp .env.example .env
```

**Windows:**

```powershell
copy .env.example .env
```

**2. Open it:**

- **Mac:** `open -e .env`
- **Windows:** `notepad .env`
- **Linux:** `nano .env` (save with `Ctrl + O`, Enter, then exit with `Ctrl + X`)

**3. Paste your key** after the `=` on the matching line, with no spaces or quotes. For example:

```
GEMINI_API_KEY=AIzaSy...your-key...
```

Save and close the file.

> **Keep your key secret.** Never share `.env` or post your key online. The project is set up so git never uploads it.

---

## Step 7: Install Ollama (optional, free)

Ollama runs a small AI model on your own laptop. It's the "junior assistant", and it's also the free fallback if you have no API key. It needs about **2 GB** of disk space and works best with 8 GB of RAM or more.

**Mac:** download and open https://ollama.com/download/Ollama.dmg, then drag Ollama into Applications and open it.

**Windows:** download and run https://ollama.com/download/OllamaSetup.exe.

**Linux:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Then download the model once (all systems):

```bash
ollama pull llama3.2:3b
```

On Mac and Windows, Ollama runs in the background by itself (look for the llama icon in the menu bar or taskbar). On Linux, if it isn't running, start it with `ollama serve` in a separate terminal.

---

## Step 8: Run it

Always run commands from inside the project folder.

| What | Command |
|---|---|
| The demo app (opens in your browser) | `uv run streamlit run app.py` |
| The notebooks (open in your browser) | `uv run jupyter lab` |
| Chapter 1: the LLM alone | `uv run python run.py chat --version 1` |
| Chapter 1: the LLM with tools (the senior expert) | `uv run python run.py chat --version 2` |
| Chapter 2: the LLM labels the data | `uv run python run.py label` |
| Chapter 2: train the receptionist | `uv run python run.py train` |
| Chapter 3: the whole support office | `uv run python run.py chat --version 4` |
| Chapter 3: compare expert-only with the whole office | `uv run python run.py benchmark --n 30` |

To stop the app or Jupyter, click the terminal and press `Ctrl + C`.

> Using **VS Code**? Open the project folder, open a notebook, click **Select Kernel** (top right) and pick the `.venv` one.

---

## If something goes wrong

| Problem | Fix |
|---|---|
| `uv: command not found` or `'uv' is not recognized` | Close the terminal and open a new one. Still failing? Repeat Step 2. |
| Windows: "running scripts is disabled on this system" | Use the exact Step 2 command; it includes `-ExecutionPolicy ByPass` for this reason. |
| `No such file or directory: pyproject.toml` | You're not inside the project folder. Go back to Step 3 and use `cd`. |
| Windows on ARM: error about `httptools` | Use the ARM command in Step 4. |
| "Your ... API key was not accepted" | Check the key in `.env`: the right line, no spaces, no quotes. |
| "Could not reach Ollama" | No API key was found, so it tried the free local model. Open the Ollama app (Linux: `ollama serve`), or add a key in Step 6. |
| "The model ... is not available" or "can't use tools" | Pick another model in `.env`, for example `OPENAI_MODEL=gpt-5.4-mini`. |
| "too many requests, or no credit left" | Wait a minute, or add credit on your provider's billing page. Gemini's free tier allows about 5 requests a minute. |
| "... is overloaded right now" | The provider is busy (not your fault). Try again in a minute. |
| The app says "port 8501 is already in use" | It's already running in another terminal. Use that one, or run `uv run streamlit run app.py --server.port 8502`. |
| Office or university network blocks downloads | Try a phone hotspot for the install (Steps 2, 4 and 7). |

## Don't want uv?

If you already manage Python yourself (version 3.10 or newer), plain pip works too:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then use `python run.py ...` instead of `uv run python run.py ...`.
