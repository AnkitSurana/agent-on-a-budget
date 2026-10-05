"""Version 3, step 1: the LLM labels our data.

A real company has thousands of customer messages but no labels.
Labelling by hand is slow. So we ask an LLM to do it, then train a small ML model on its labels.
"""

import json

import pandas as pd

from budget_agent import config
from budget_agent.llm import make_llm

LABEL_PROMPT = (
    "You label customer support messages for an online shop. "
    "For each message, pick the ONE intent that fits best from the allowed list. "
    "If a message has two requests, pick the main one."
)

# We ask for JSON in a fixed shape, so the answer is always easy to read in Python.
LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "intent": {"type": "string", "enum": config.INTENTS},
                },
                "required": ["id", "intent"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}


def label_batch(llm, batch):
    """Label a small group of messages in ONE request (cheaper than one request per message).

    batch is a DataFrame with 'id' and 'text'. Returns ({id: intent}, stats).
    """
    lines = "\n".join(f"{row.id}: {row.text}" for row in batch.itertuples())
    reply = llm.chat(
        system=LABEL_PROMPT,
        history=[{"role": "user", "content": f"Label these messages:\n{lines}"}],
        effort="low",          # labelling is easy, so the model doesn't need to think hard
        max_tokens=8000,
        json_schema=LABEL_SCHEMA,
    )
    labels = json.loads(reply.text)["labels"]
    return {item["id"]: item["intent"] for item in labels}, reply.stats


def label_messages(messages, llm=None, batch_size=25, save_to=config.LLM_LABELS_CSV):
    """Label every message. Saves progress after each batch, so you can stop and continue later."""
    llm = llm or make_llm()
    print(f"Labelling with {llm.provider} ({llm.model})")
    done = pd.read_csv(save_to) if save_to and save_to.exists() else pd.DataFrame(columns=["id", "llm_intent"])
    todo = messages[~messages["id"].isin(done["id"])]
    total_cost = 0.0

    for start in range(0, len(todo), batch_size):
        batch = todo.iloc[start:start + batch_size]
        labels, stats = label_batch(llm, batch)
        new = pd.DataFrame({"id": list(labels), "llm_intent": list(labels.values())})
        done = pd.concat([done, new], ignore_index=True)
        if save_to:
            done.to_csv(save_to, index=False)
        total_cost += stats.cost_usd
        print(f"Labelled {min(start + batch_size, len(todo))}/{len(todo)}  "
              f"({stats.seconds:.1f}s, ${stats.cost_usd:.4f})")

    print(f"Done. Cost of this run: ${total_cost:.4f}")
    skipped = len(set(messages["id"]) - set(done["id"]))
    if skipped:
        print(f"The LLM skipped {skipped} message(s). Run the same command again to label just those.")
    return done
