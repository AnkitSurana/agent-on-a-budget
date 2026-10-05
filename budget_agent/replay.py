"""Saved answers: replay an LLM answer instead of paying for it again.

Every real LLM answer is written to data/saved_llm_answers.jsonl (one line per answer).
If the exact same request comes again (same model, same prompt, same conversation),
the saved answer is used: instant, free, and it works without internet.
Only new questions are sent to the LLM.

Turn it off with REPLAY=off in the .env file, to always ask the LLM fresh.
"""

import hashlib
import json

from budget_agent import config

_saved = None   # loaded from the file the first time it's needed


def _plain(obj):
    """Turn SDK objects into plain dicts and lists, so they can be saved as JSON."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    if isinstance(obj, (list, tuple)):
        return [_plain(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _plain(value) for key, value in obj.items()}
    return obj


def key_for(provider, model, system, history, tools=None, json_schema=None):
    """A fingerprint of the request. Same request -> same key."""
    conversation = []
    for turn in history:
        if turn["role"] == "assistant":
            conversation.append({"role": "assistant", "raw": _plain(turn["reply"].raw)})
        else:
            conversation.append(_plain(turn))
    request = {"provider": provider, "model": model, "system": system, "conversation": conversation,
               "tools": [t["name"] for t in tools or []], "json_schema": json_schema}
    return hashlib.sha256(json.dumps(request, sort_keys=True, default=str).encode()).hexdigest()


def _load():
    global _saved
    if _saved is None:
        _saved = {}
        if config.REPLAY_LOG.exists():
            for line in config.REPLAY_LOG.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    _saved[entry["key"]] = entry
    return _saved


def find(key):
    """The saved answer for this request, or None."""
    return _load().get(key) if config.REPLAY else None


def save(key, **answer):
    """Remember a new answer, in memory and in the file."""
    if not config.REPLAY:
        return
    entry = {"key": key, **_plain(answer)}
    _load()[key] = entry
    config.REPLAY_LOG.parent.mkdir(exist_ok=True)
    with open(config.REPLAY_LOG, "a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")
