"""One command line for the whole project. Run `python run.py --help` to see everything.

    python run.py data                 # make the data files

    Chapter 1 - Build your first AI agent
    python run.py chat --version 1     # the LLM alone: it can only talk
    python run.py chat --version 2     # the LLM with tools: the senior expert

    Chapter 2 - From Machine Learning & LLMs: the LLM teaches a small ML model
    python run.py label                # the LLM labels the training messages
    python run.py train                # the ML model (the receptionist) learns from those labels

    Chapter 3 - To real-world workflows: the whole support office
    python run.py route                # who would answer each messy message (free, no LLM)
    python run.py chat --version 4     # talk to the whole office
    python run.py benchmark --n 30     # expert only vs the whole office (costs a little money)
"""

import argparse

import pandas as pd

from budget_agent import classifier, config, data


def cmd_data(args):
    data.build_all()


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
        if args.version == 4:
            r = bot.handle(text)
            who = config.HELPERS[r.layer] + (f" → {config.HELPERS['human']}" if r.escalated else "")
            print(f"Bot: {r.reply}\n     [{who}  intent={r.intent} ({r.confidence:.0%})  "
                  f"{r.seconds:.2f}s  ${r.cost_usd:.4f}]\n")
        else:
            r = bot.run(text)
            tools = ", ".join(name for name, _, _ in r.tool_calls) or "none"
            print(f"Bot: {r.reply}\n     [tools={tools}  {r.seconds:.2f}s  ${r.cost_usd:.4f}]\n")


def cmd_label(args):
    from budget_agent.labeler import label_messages
    train = data.load_messages().query("split == 'train'")
    label_messages(train.head(args.n))


def training_data():
    """Use the LLM's labels if we have them. If not, fall back to the true labels (and say so)."""
    train = data.load_messages().query("split == 'train'")
    if config.LLM_LABELS_CSV.exists():
        labels = pd.read_csv(config.LLM_LABELS_CSV)
        train = train.merge(labels, on="id")
        return train["text"], train["llm_intent"], "the LLM's labels"
    return train["text"], train["true_intent"], "the TRUE labels (run `python run.py label` to use the LLM's)"


def cmd_train(args):
    texts, labels, source = training_data()
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
    rows = []
    for i, msg in enumerate(messages.itertuples(), 1):
        v2 = agent.run(msg.text)
        v4 = router.handle(msg.text)
        rows.append({"message": msg.text, "true_intent": msg.true_intent,
                     "v2_seconds": v2.seconds, "v2_cost": v2.cost_usd,
                     "v4_layer": v4.layer, "v4_seconds": v4.seconds, "v4_cost": v4.cost_usd,
                     "v2_reply": v2.reply, "v4_reply": v4.reply})
        print(f"{i}/{len(messages)}  v2 {v2.seconds:5.2f}s ${v2.cost_usd:.4f}  |  "
              f"v4 [{v4.layer:9}] {v4.seconds:5.2f}s ${v4.cost_usd:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv(config.DATA_DIR / "benchmark_results.csv", index=False)
    print_summary(results)


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


def main():
    parser = argparse.ArgumentParser(description="agent-on-a-budget")
    sub = parser.add_subparsers(required=True)
    sub.add_parser("data", help="make the data files").set_defaults(func=cmd_data)
    chat = sub.add_parser("chat", help="talk to the agent: 1 = LLM only, 2 = LLM + tools, 4 = whole office")
    chat.add_argument("--version", type=int, choices=[1, 2, 4], default=2)
    chat.set_defaults(func=cmd_chat)
    label = sub.add_parser("label", help="the LLM labels the training messages")
    label.add_argument("--n", type=int, default=1080)
    label.set_defaults(func=cmd_label)
    sub.add_parser("train", help="train the ML model").set_defaults(func=cmd_train)
    sub.add_parser("route", help="show which helper answers each messy message").set_defaults(func=cmd_route)
    bench = sub.add_parser("benchmark", help="compare expert only (version 2) with the whole office (version 4)")
    bench.add_argument("--n", type=int, default=30)
    bench.set_defaults(func=cmd_benchmark)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
