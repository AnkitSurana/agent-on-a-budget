"""One command line for the whole project. Run `uv run python run.py --help` to see everything.

    uv run python run.py check                # what works on this laptop, and what to fix
    uv run python run.py app                  # start the app in your browser
    uv run python run.py data                 # make the data files

    Chapter 1: build your first agent
    uv run python run.py chat --version 1     # the LLM alone: it can only talk
    uv run python run.py chat --version 2     # the LLM with tools: the senior expert

    Chapter 2: the LLM teaches a small ML model
    uv run python run.py label                # the LLM labels the training messages
    uv run python run.py train                # the ML model (the receptionist) learns from those labels

    Chapter 3: the whole support office
    uv run python run.py route                # who would answer each messy message (free, no LLM)
    uv run python run.py chat --version 4     # talk to the whole office
    uv run python run.py benchmark --n 30     # expert only vs the whole office (needs a key)
"""

import argparse
import sys

import pandas as pd

from budget_agent import classifier, config, data
from budget_agent.llm import LLMSetupError


def cmd_check(args):
    """What works on this laptop, and what to do about anything that doesn't. Free, no LLM called."""
    import os
    from budget_agent.llm import LocalLLM, pick_provider
    from budget_agent.replay import _load

    def line(ok, text, fix=""):
        print(f"{'✅' if ok else '⚠️ '} {text}" + (f"\n     → {fix}" if fix and not ok else ""))

    line(not config.other_python(), f"Python and every package are installed (Python {sys.version.split()[0]})"
         if not config.other_python() else f"This is not the project's Python: {config.other_python()}",
         "Start the command with uv run: uv run python run.py check")
    trained = config.CLASSIFIER_PATH.exists()
    line(trained, "🟢 Receptionist (ML model) is trained" if trained else "🟢 Receptionist is not trained yet",
         "Train it once, in a few seconds: uv run python run.py train")
    local = LocalLLM()
    ready = local.is_ready()
    line(ready, f"🟡 Junior assistant: Ollama is running with {local.model}" if ready
         else f"🟡 Junior assistant: Ollama isn't running, or {local.model} isn't downloaded yet",
         f"Install Ollama (README, setup step 5: https://ollama.com/download), open it, then run: ollama pull {local.model}")
    provider = pick_provider()
    model = config.PROVIDERS[provider]["model"]
    if provider == "ollama" and not os.getenv("LLM_PROVIDER"):
        line(False, "🔵 Senior expert: no API key, so saved answers are replayed (the demo messages work, for $0)",
             "Add your API key to .env (README, setup step 6) to ask your own questions. Without one, the saved examples still work.")
    elif provider == "ollama":
        line(local.is_ready(), f"🔵 Senior expert: Ollama ({model}), free but slower and less smart",
             "Open Ollama, or put an API key in the .env file")
    else:
        from budget_agent.llm import provider_from_key
        key, value = "LLM_API_KEY", config.api_key()
        placeholder = len(value) < 20 or any(word in value.lower() for word in ["your", "here", "xxx", "paste"])
        if not value:
            ok, fix = False, (f"LLM_PROVIDER is {provider} but LLM_API_KEY is empty in .env: saved answers replay for $0, new questions "
                             "go to the free local model in Ollama, and new actions like refunds go to a human")
        elif placeholder:
            ok, fix = False, f"{key} in .env looks like an example ({value[:7]}...{value[-4:]}), not a real key: paste your real one"
        elif provider != "ollama" and provider_from_key(value) != provider:
            ok, fix = False, (f"{key} looks like a key for {provider_from_key(value)}, but LLM_PROVIDER is {provider}: "
                              "change one of them so they match")
        else:
            ok, fix = True, ""
        line(ok, f"🔵 Senior expert: {provider} ({model})", fix)
    if config.REPLAY:
        print(f"\n♻️  Saved answers ON: {len(_load())} answers replay for $0. (REPLAY=off in .env = always ask fresh)")
    else:
        print("\n🔴 Saved answers OFF (REPLAY=off in .env): every question goes to the LLM, live and paid.")
    print("Start the app with: uv run python run.py app   (edit .env any time: no restart needed)")


def cmd_data(args):
    data.build_all()


SAVED = "  ♻️ saved answer: $0 this time"   # shown when replay.py answered instead of the LLM


def cmd_chat(args):
    if args.version == 4:
        from budget_agent.router import Router
        bot = Router()
    else:
        from budget_agent.agent import SupportAgent
        bot = SupportAgent(use_tools=(args.version == 2))
        print(f"Using {bot.llm.provider} ({bot.llm.model})")

    print(f"NovaMart support, version {args.version}. Type 'quit' to stop.\n")
    while (text := input("You: ").strip()).lower() not in {"quit", "exit", "q"}:
        try:
            if args.version == 4:
                r = bot.handle(text)
                who = config.HELPERS[r.layer] + (f" → {config.HELPERS['human']}" if r.escalated else "")
                print(f"Bot: {r.reply}\n     [{who}  intent={r.intent} ({r.confidence:.0%})  "
                      f"{r.seconds:.2f}s  ${r.cost_usd:.4f}{SAVED if r.replayed else ''}]\n")
            else:
                r = bot.run(text)
                tools = ", ".join(name for name, _, _ in r.tool_calls) or "none"
                print(f"Bot: {r.reply}\n     [tools={tools}  {r.seconds:.2f}s  ${r.cost_usd:.4f}"
                      f"{SAVED if r.replayed else ''}]\n")
        except LLMSetupError as problem:
            print(f"⚠️  {problem}\n")


def cmd_label(args):
    from budget_agent.labeler import label_messages
    train = data.load_messages().query("split == 'train'")
    label_messages(train.head(args.n))


def cmd_train(args):
    texts, labels, source = data.training_data()
    model = classifier.train(texts, labels)
    classifier.save(model)

    test = data.load_messages().query("split == 'test'")
    messy = data.load_messy()
    print(f"Trained on {len(texts)} messages using {source}.")
    print(f"Accuracy on clean test messages: {classifier.accuracy(model, test.text, test.true_intent):.1%}")
    print(f"Accuracy on messy messages:      {classifier.accuracy(model, messy.text, messy.true_intent):.1%}")
    print(f"Saved to {config.CLASSIFIER_PATH.relative_to(config.ROOT)}")


def cmd_route(args):
    from budget_agent.router import Router
    router = Router()
    messy = data.load_messy()
    rows = [(text, *router.choose_layer(text)) for text in messy.text]
    table = pd.DataFrame(rows, columns=["message", "layer", "intent", "confidence"])
    table["layer"] = table["layer"].map(config.HELPERS)
    pd.set_option("display.width", 160, "display.max_colwidth", 60)
    print(table.to_string(index=False, formatters={"confidence": "{:.0%}".format}))
    print("\n" + table.layer.value_counts(normalize=True).map("{:.0%}".format).to_string())


def demo_messages(n, seed=7):
    """A mix of clean test messages and messy hand-written ones, like a real inbox."""
    test = data.load_messages().query("split == 'test'")[["text", "true_intent"]]
    messy = data.load_messy()
    n_messy = min(len(messy), n // 3)
    mix = pd.concat([test.sample(n - n_messy, random_state=seed), messy.sample(n_messy, random_state=seed)])
    return mix.sample(frac=1, random_state=seed).reset_index(drop=True)


def cmd_benchmark(args):
    from budget_agent.agent import SupportAgent
    from budget_agent.router import Router

    messages = demo_messages(args.n)
    agent, router = SupportAgent(use_tools=True), Router()
    print(f"Cloud LLM: {agent.llm.provider} ({agent.llm.model})")
    rows, spent_now = [], 0.0
    for i, msg in enumerate(messages.itertuples(), 1):
        v2 = agent.run(msg.text)
        v4 = router.handle(msg.text)
        spent_now += sum(c.cost_usd for c in v2.calls if not c.replayed) + (0 if v4.replayed else v4.cost_usd)
        rows.append({"message": msg.text, "true_intent": msg.true_intent,
                     "v2_seconds": v2.seconds, "v2_cost": v2.cost_usd,
                     "v4_layer": v4.layer, "v4_seconds": v4.seconds, "v4_cost": v4.cost_usd,
                     "v2_reply": v2.reply, "v4_reply": v4.reply})
        print(f"{i}/{len(messages)}  v2 {v2.seconds:5.2f}s ${v2.cost_usd:.4f}  |  "
              f"v4 [{v4.layer:9}] {v4.seconds:5.2f}s ${v4.cost_usd:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv(config.BENCHMARK_CSV, index=False)
    print_summary(results)
    print(f"\nMoney actually spent on this run: ${spent_now:.3f} (saved answers are free)")


def print_summary(results):
    n = len(results)
    print(f"\n--- {n} messages ---")
    print(f"Expert only (LLM for everything): ${results.v2_cost.sum():.3f} total, "
          f"{results.v2_seconds.mean():.2f}s average")
    print(f"Whole office (router):             ${results.v4_cost.sum():.3f} total, "
          f"{results.v4_seconds.mean():.2f}s average")
    if results.v4_cost.sum() > 0:
        print(f"The whole office is {results.v2_cost.sum() / results.v4_cost.sum():.1f}x cheaper "
              f"and {results.v2_seconds.mean() / results.v4_seconds.mean():.1f}x faster on average.")
    print("Who answered:\n" + results.v4_layer.map(config.HELPERS).value_counts().to_string())


def cmd_app(args):
    """Start the Streamlit app. Same as `streamlit run app.py`, minus its first-run email question and usage stats."""
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(config.ROOT / "app.py"),
                              "--server.showEmailPrompt", "false", "--browser.gatherUsageStats", "false",
                              *args.extra]))   # for example: run.py app --server.port 8502


def main():
    # Windows terminals sometimes can't print ₹ or emoji. UTF-8 output fixes that on every system.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if config.other_python():
        print(f"⚠️  This is not the project's Python ({config.other_python()}), so other package versions may be used.\n"
              f"   Start every command with uv run, for example: uv run python {' '.join(sys.argv)}\n", file=sys.stderr)
    parser = argparse.ArgumentParser(description="agent-on-a-budget")
    sub = parser.add_subparsers(required=True)
    sub.add_parser("check", help="what works on this laptop, and what to fix (free)").set_defaults(func=cmd_check)
    sub.add_parser("data", help="make the data files").set_defaults(func=cmd_data)
    chat = sub.add_parser("chat", help="talk to the agent: 1 = LLM only, 2 = LLM + tools, 4 = whole office")
    chat.add_argument("--version", type=int, choices=[1, 2, 4], default=2)
    chat.set_defaults(func=cmd_chat)
    label = sub.add_parser("label", help="the LLM labels the training messages")
    label.add_argument("--n", type=int, default=1080)
    label.set_defaults(func=cmd_label)
    sub.add_parser("train", help="train the ML model").set_defaults(func=cmd_train)
    sub.add_parser("route", help="show which helper answers each messy message").set_defaults(func=cmd_route)
    bench = sub.add_parser("benchmark", help="expert only (chat version 2) vs the whole office (version 4); needs a key")
    bench.add_argument("--n", type=int, default=30)
    bench.set_defaults(func=cmd_benchmark)
    sub.add_parser("app", help="start the app in your browser").set_defaults(func=cmd_app)
    args, extra = parser.parse_known_args()
    if extra and args.func is not cmd_app:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    args.extra = extra
    try:
        args.func(args)
    except LLMSetupError as problem:   # a setup problem: show a short message, not a long error
        sys.exit(f"⚠️  {problem}")


if __name__ == "__main__":
    main()
