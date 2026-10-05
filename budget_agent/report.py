"""Write docs/results.md: every result of the project in one page.

It only READS saved files (labels, benchmark, saved answers) and the trained model.
No LLM is called, so it's free. Run it with:  uv run python run.py report
"""

import json
from datetime import date

import pandas as pd

from budget_agent import classifier, config, data


def _table(df):
    """A small pandas DataFrame as a markdown table (no extra packages needed)."""
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, *rows])


def _saved_answers():
    if not config.REPLAY_LOG.exists():
        return pd.DataFrame(columns=["text", "model", "cost"])
    entries = [json.loads(line) for line in config.REPLAY_LOG.read_text(encoding="utf-8").splitlines() if line]
    return pd.DataFrame({"text": [e["text"] for e in entries],
                         "model": [e["stats"]["model"] for e in entries],
                         "cost": [e["stats"]["cost_usd"] for e in entries]})


def build():
    messages, messy = data.load_messages(), data.load_messy()
    test = messages[messages.split == "test"]
    saved = _saved_answers()
    out = [f"# Results\n\nWritten by `uv run python run.py report` on {date.today():%d %B %Y}, "
           "from the saved files in `data/`. Making this page called no LLM and cost nothing.\n"]

    # ---------- Chapter 2 ----------
    out.append("## Chapter 2: the LLM teaches the ML model\n")
    if config.LLM_LABELS_CSV.exists():
        labels = messages.merge(pd.read_csv(config.LLM_LABELS_CSV), on="id")
        agree = (labels.llm_intent == labels.true_intent).mean()
        label_cost = saved[saved.text.str.startswith('{"labels"')].cost.sum()
        out.append(f"- The LLM labelled **{len(labels):,}** messages. Its labels match the dataset's true labels "
                   f"**{agree:.1%}** of the time.")
        if label_cost:
            out.append(f"- Labelling cost **${label_cost:.2f}** in total.")
        wrong = labels[labels.llm_intent != labels.true_intent]
        pairs = wrong.groupby(["true_intent", "llm_intent"]).size().sort_values(ascending=False).head(6)
        out.append("\nWhere the LLM disagreed most (often a close pair, not a real mistake):\n")
        out.append(_table(pairs.reset_index().rename(columns={
            "true_intent": "Dataset says", "llm_intent": "LLM said", 0: "Messages"})))
    if config.CLASSIFIER_PATH.exists():
        model = classifier.load()
        inbox = pd.concat([test[["text", "true_intent"]], messy])
        probs = model.predict_proba(inbox.text)
        sure, right = probs.max(axis=1), model.classes_[probs.argmax(axis=1)] == inbox.true_intent.values
        out.append(f"\nThe receptionist (the ML model) on {len(test)} clean test messages: "
                   f"**{classifier.accuracy(model, test.text, test.true_intent):.1%}** right. "
                   f"On {len(messy)} messy messages: **{classifier.accuracy(model, messy.text, messy.true_intent):.1%}**.\n")
        rows = [{"Answers only when at least": f"{t:.0%} sure",
                 "Share of messages it answers": f"{(sure >= t).mean():.0%}",
                 "Right on those": f"{right[sure >= t].mean():.1%}"} for t in [0.4, 0.6, 0.8, 0.9]]
        out.append(_table(pd.DataFrame(rows)))

    # ---------- Chapter 3 ----------
    if config.BENCHMARK_CSV.exists():
        r = pd.read_csv(config.BENCHMARK_CSV)
        n = len(r)
        out.append(f"\n## Chapter 3: expert only vs the whole office\n\n{n} messages "
                   "(a mix of normal and messy ones), each sent to the senior expert alone and to the whole office.\n")
        summary = pd.DataFrame({
            "": ["Total cost", "Cost per 1,000 messages", "Average time per message", "Answered for $0"],
            "Expert only": [f"${r.v2_cost.sum():.3f}", f"${r.v2_cost.mean() * 1000:.2f}",
                            f"{r.v2_seconds.mean():.2f} s", "0%"],
            "Whole office": [f"${r.v4_cost.sum():.3f}", f"${r.v4_cost.mean() * 1000:.2f}",
                             f"{r.v4_seconds.mean():.2f} s", f"{(r.v4_layer != 'cloud_llm').mean():.0%}"]})
        out.append(_table(summary))
        out.append(f"\n**The whole office was {r.v2_cost.sum() / r.v4_cost.sum():.1f}× cheaper and "
                   f"{r.v2_seconds.mean() / r.v4_seconds.mean():.1f}× faster.**\n")
        helpers = r.groupby("v4_layer").agg(messages=("message", "size"), seconds=("v4_seconds", "mean"),
                                            cost=("v4_cost", "mean")).reset_index()
        helpers["v4_layer"] = helpers.v4_layer.map(config.HELPERS)
        helpers["seconds"] = helpers.seconds.map("{:.2f} s".format)
        helpers["cost"] = helpers.cost.map("${:.4f}".format)
        out.append(_table(helpers.rename(columns={"v4_layer": "Helper", "messages": "Messages",
                                                  "seconds": "Average time", "cost": "Average cost"})))
        out.append("\n### Every message, both answers\n")
        every = pd.DataFrame({"Message": r.message, "Who answered": r.v4_layer.map(config.HELPERS),
                              "Expert only": r.v2_reply.str.slice(0, 160),
                              "Whole office": r.v4_reply.str.slice(0, 160)})
        out.append(_table(every))

    # ---------- Saved answers ----------
    if len(saved):
        models = ", ".join(f"`{m}`" for m in sorted(saved.model.unique()))
        out.append(f"\n## Saved answers\n\n`{config.REPLAY_LOG.relative_to(config.ROOT)}` holds **{len(saved)}** real "
                   f"LLM answers (models: {models}), which cost **${saved.cost.sum():.2f}** when they were made. "
                   "The app, notebooks, chat and benchmark replay them for free whenever the same request comes again.\n")

    config.ROOT.joinpath("docs").mkdir(exist_ok=True)
    path = config.ROOT / "docs" / "results.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path
