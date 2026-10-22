"""Talking to LLMs. Works with whichever key you have: OpenAI, Anthropic (Claude), or Gemini.
No key at all? Saved answers are replayed for free, and new questions go to the free local model in Ollama
(new requests for the senior expert, like refunds, go to a human instead: see agent.py).

Every provider is used the same way:

    llm = make_llm()                       # picks a provider from your .env
    reply = llm.chat(system, history, tools=TOOL_DEFINITIONS)
    reply.text, reply.tool_calls, reply.stats

`stats` tells you how many tokens, seconds and dollars the call took.
Measuring this is the whole point of the project.
"""

import json
import os
import time
from dataclasses import dataclass, field

import requests

from budget_agent import config, replay


@dataclass
class CallStats:
    model: str
    input_tokens: int
    output_tokens: int
    seconds: float
    cost_usd: float
    replayed: bool = False        # True = a saved answer was used: it cost nothing this time


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class Reply:
    text: str
    stop: str                     # "done", "tool_use" or "refused"
    stats: CallStats
    tool_calls: list = field(default_factory=list)
    raw: object = None            # the provider's own message, sent back unchanged in the next request


class LLMSetupError(Exception):
    """A setup problem (missing key, wrong model, Ollama not running), explained in plain English."""


def explain(provider, model, sdk, error):
    """Turn the most common API errors into a short message a beginner can act on."""
    bad_key = isinstance(error, sdk.BadRequestError) and "api key" in str(error).lower()   # how Gemini says it
    if bad_key or isinstance(error, (sdk.AuthenticationError, sdk.PermissionDeniedError)):
        return LLMSetupError(f"Your {provider} API key was not accepted. Check LLM_API_KEY in the .env file.")
    if isinstance(error, sdk.BadRequestError) and "tools" in str(error).lower() and "not supported" in str(error).lower():
        return LLMSetupError(f"The model '{model}' can't use tools in this project. "
                             "Choose another one with LLM_MODEL=... in the .env file "
                             "(for OpenAI, gpt-5.4-mini works well).")
    if isinstance(error, sdk.NotFoundError):
        if provider == "ollama":
            return LLMSetupError(f"Ollama doesn't have the model '{model}' yet. Run: ollama pull {model}")
        return LLMSetupError(f"The model '{model}' is not available on your {provider} account. "
                             "Choose another one with LLM_MODEL=... in the .env file.")
    if isinstance(error, sdk.InternalServerError):
        return LLMSetupError(f"{provider} is overloaded right now (this is on their side, not yours). "
                             "Try again in a minute, or switch provider with LLM_PROVIDER in the .env file.")
    if isinstance(error, sdk.RateLimitError):
        return LLMSetupError(f"{provider} says: too many requests, or no credit left. "
                             "Wait a minute, or check your account's billing page.")
    if isinstance(error, sdk.APIConnectionError):
        if provider == "ollama":
            return LLMSetupError("Could not reach Ollama, the free local model. Open the Ollama app "
                                 "(on Linux: ollama serve), and once run: ollama pull llama3.2:3b. "
                                 "Or put an API key in the .env file.")
        return LLMSetupError(f"Could not reach {provider}. Check your internet connection.")
    return None


def cost_usd(model, input_tokens, output_tokens):
    """Dollars for one call. Unknown and local models count as free."""
    if model not in config.PRICES_PER_MILLION:
        return 0.0
    price_in, price_out = config.PRICES_PER_MILLION[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


# A conversation ("history") is a list of these, the same for every provider:
#   {"role": "user", "content": "where is my order?"}
#   {"role": "assistant", "reply": Reply(...)}
#   {"role": "tool_results", "results": [(tool_call_id, output_text), ...]}


class LLM:
    """What every provider shares: a saved answer is replayed instead of paid for again (see replay.py)."""

    def chat(self, system, history, tools=None, effort="medium", max_tokens=8000, json_schema=None):
        key = replay.key_for(self.provider, self.model, system, history, tools, json_schema)
        saved = replay.find(key)
        if saved:
            stats = CallStats(**{**saved["stats"], "replayed": True})
            return Reply(saved["text"], saved["stop"], stats,
                         [ToolCall(**call) for call in saved["tool_calls"]], saved["raw"])
        reply = self._ask(system, history, tools, effort, max_tokens, json_schema)
        replay.save(key, text=reply.text, stop=reply.stop, stats=vars(reply.stats),
                    tool_calls=[vars(call) for call in reply.tool_calls], raw=reply.raw)
        return reply


class OpenAIChat(LLM):
    """OpenAI, and anything that speaks the same language: Gemini and Ollama do too.

    Same code, just a different web address (base_url) and key.
    """

    def __init__(self, provider="openai", model=None, client=None):
        settings = config.PROVIDERS[provider]
        self.provider = provider
        self.model = model or settings["model"]
        if client is None:
            from openai import OpenAI
            # max_retries: if the provider is busy, wait and try again a few times before giving up.
            # Not for Ollama: if it isn't running on your laptop, trying again won't help, so say so at once.
            client = OpenAI(api_key=config.api_key() or "no-key", base_url=settings.get("base_url"),
                            max_retries=0 if provider == "ollama" else 5)
        self.client = client

    def _ask(self, system, history, tools, effort, max_tokens, json_schema):
        messages = [{"role": "system", "content": system}]
        for turn in history:
            if turn["role"] == "user":
                messages.append({"role": "user", "content": turn["content"]})
            elif turn["role"] == "assistant":
                messages.append(turn["reply"].raw)
            else:
                for call_id, output in turn["results"]:
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": output})

        request = {"model": self.model, "messages": messages, "max_completion_tokens": max_tokens}
        if tools:
            request["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
                for t in tools]
        if json_schema:
            request["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": "answer", "schema": json_schema, "strict": True}}

        import openai
        start = time.perf_counter()
        try:
            try:
                response = self.client.chat.completions.create(**request)
            except openai.BadRequestError as error:
                # Some OpenAI models only use tools here with reasoning switched off. Try once more that way.
                if not (tools and "reasoning_effort to 'none'" in str(error)):
                    raise
                response = self.client.chat.completions.create(**request, reasoning_effort="none")
        except openai.APIError as error:
            raise (explain(self.provider, self.model, openai, error) or error) from error
        seconds = time.perf_counter() - start

        choice = response.choices[0]
        message = choice.message
        tool_calls = [ToolCall(c.id, c.function.name, json.loads(c.function.arguments or "{}"))
                      for c in (message.tool_calls or [])]
        if tool_calls:
            stop = "tool_use"
        elif choice.finish_reason == "content_filter" or getattr(message, "refusal", None):
            stop = "refused"
        else:
            stop = "done"

        usage = response.usage
        stats = CallStats(self.model, usage.prompt_tokens, usage.completion_tokens, seconds,
                          cost_usd(self.model, usage.prompt_tokens, usage.completion_tokens))
        # model_dump keeps every field the provider sent (Gemini hides "thought signatures" in there)
        raw = message.model_dump(exclude_none=True)
        return Reply((message.content or "").strip(), stop, stats, tool_calls, raw)


class Claude(LLM):
    """Anthropic's Claude."""

    provider = "anthropic"

    def __init__(self, model=None, client=None):
        self.model = model or config.PROVIDERS["anthropic"]["model"]
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=config.api_key() or "no-key", max_retries=5)
        self.client = client

    def _ask(self, system, history, tools, effort, max_tokens, json_schema):
        messages = []
        for turn in history:
            if turn["role"] == "user":
                messages.append({"role": "user", "content": turn["content"]})
            elif turn["role"] == "assistant":
                messages.append({"role": "assistant", "content": turn["reply"].raw})
            else:
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": call_id, "content": output}
                    for call_id, output in turn["results"]]})

        output_config = {"effort": effort}
        if json_schema:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        extra = {"tools": tools} if tools else {}

        import anthropic
        start = time.perf_counter()
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                output_config=output_config,
                # If Claude declines a request for safety reasons, the API retries it on a fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                **extra,
            )
        except anthropic.APIError as error:
            raise (explain(self.provider, self.model, anthropic, error) or error) from error
        seconds = time.perf_counter() - start

        tool_calls = [ToolCall(b.id, b.name, b.input) for b in response.content if b.type == "tool_use"]
        stop = {"tool_use": "tool_use", "refusal": "refused"}.get(response.stop_reason, "done")
        used_model = response.model if response.model in config.PRICES_PER_MILLION else self.model
        usage = response.usage
        stats = CallStats(used_model, usage.input_tokens, usage.output_tokens, seconds,
                          cost_usd(used_model, usage.input_tokens, usage.output_tokens))
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return Reply(text, stop, stats, tool_calls, raw=response.content)


def provider_from_key(key):
    """Guess the provider from how the key starts. Used when LLM_PROVIDER is left empty."""
    if key.startswith("sk-ant-"):
        return "anthropic"
    if key.startswith("AIza"):
        return "gemini"
    return "openai"


def pick_provider():
    """LLM_PROVIDER from .env. Left empty? Guess it from LLM_API_KEY. No key at all? The free local model."""
    chosen = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if chosen:
        if chosen not in config.PROVIDERS:
            raise LLMSetupError(f"LLM_PROVIDER={chosen} in .env is not one we know. "
                                f"Use one of: {', '.join(config.PROVIDERS)}.")
        return chosen
    if config.api_key():
        return provider_from_key(config.api_key())
    return "ollama"


class SavedOrLocal(LLM):
    """No API key at all: replay the saved answers (made with OpenAI), so every demo works for free.

    A question that was never asked before goes to the free local model in Ollama. A new request that needs
    tools (the senior expert) does not: the agent hands it to a human instead, see can_use_tools.
    """

    def __init__(self, saved=None, local=None):
        self.saved = saved or OpenAIChat("openai", client="never called: only saved answers are used")
        self.local = local or OpenAIChat("ollama")
        # It reports the model the saved answers came from, so prices and printouts match the examples it replays
        self.provider, self.model = self.saved.provider, self.saved.model

    def has_saved(self, system, history, tools=None, json_schema=None):
        """Is there a saved answer for this exact request?"""
        return bool(replay.find(replay.key_for(self.saved.provider, self.saved.model, system, history, tools, json_schema)))

    def can_use_tools(self, system, history, tools):
        """Llama is too small to run tools like refunds reliably, so only saved conversations get tools."""
        return self.has_saved(system, history, tools)

    def chat(self, system, history, tools=None, effort="medium", max_tokens=8000, json_schema=None):
        llm = self.saved if self.has_saved(system, history, tools, json_schema) else self.local
        return llm.chat(system, history, tools, effort, max_tokens, json_schema)


class FollowsEnv:
    """The LLM your .env asks for, checked again before every question.

    Put a real key in .env, or change LLM_PROVIDER or REPLAY, while the app or a notebook is running:
    the next question uses it. No restart needed.
    """

    def __init__(self):
        self._llm, self._made_for = None, None

    @property
    def llm(self):
        config.refresh()
        settings = (pick_provider(), repr(config.PROVIDERS), config.api_key())
        if settings != self._made_for:                 # first question, or .env changed: connect again
            self._llm, self._made_for = _connect(), settings
        return self._llm

    provider = property(lambda self: self.llm.provider)
    model = property(lambda self: self.llm.model)

    def chat(self, *args, **kwargs):
        return self.llm.chat(*args, **kwargs)

    def can_use_tools(self, system, history, tools):
        return getattr(self.llm, "can_use_tools", lambda *a: True)(system, history, tools)


def _connect(provider=None, model=None):
    if provider is None and model is None:
        chosen = pick_provider()
        no_key = chosen != "ollama" and not config.api_key()     # for example LLM_PROVIDER=openai, copied from .env.example, key still empty
        if no_key or (chosen == "ollama" and not os.getenv("LLM_PROVIDER")):
            return SavedOrLocal()
    provider = provider or pick_provider()
    if provider == "anthropic":
        return Claude(model=model)
    return OpenAIChat(provider, model=model)


def make_llm(provider=None, model=None):
    """make_llm() follows your .env, even if you edit it later. make_llm("openai") always uses that one."""
    if provider is None and model is None:
        return FollowsEnv()
    return _connect(provider, model)


class LocalLLM:
    """The small model for layer 2 of the router, running on your laptop with Ollama. Free and private."""

    def __init__(self, model=None, url=None):
        self._model, self._url = model, url

    @property
    def model(self):
        return self._model or config.LOCAL_MODEL

    @property
    def url(self):
        return self._url or config.OLLAMA_URL

    def is_ready(self):
        """True if Ollama is running AND the model has been downloaded."""
        config.refresh()
        try:
            tags = requests.get(f"{self.url}/api/tags", timeout=2).json()
        except requests.RequestException:
            return False
        names = {m["name"] for m in tags.get("models", [])}
        return self.model in names or f"{self.model}:latest" in names

    def chat(self, system, user, max_tokens=300):
        """Ask the local model one question. Returns (text, stats)."""
        config.refresh()
        key = replay.key_for("local", self.model, system, [{"role": "user", "content": user}])
        saved = replay.find(key)
        if saved:
            return saved["text"], CallStats(**{**saved["stats"], "replayed": True})
        start = time.perf_counter()
        reply = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": max_tokens, "seed": 42},   # same question, same answer
            },
            timeout=120,
        ).json()
        seconds = time.perf_counter() - start
        stats = CallStats(self.model, reply.get("prompt_eval_count", 0), reply.get("eval_count", 0), seconds, 0.0)
        text = reply["message"]["content"].strip()
        replay.save(key, text=text, stats=vars(stats))
        return text, stats
