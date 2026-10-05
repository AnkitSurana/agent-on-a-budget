"""Talking to LLMs. Works with whichever key you have: OpenAI, Anthropic (Claude), or Gemini.
No key at all? Use the free local model in Ollama.

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

from budget_agent import config


@dataclass
class CallStats:
    model: str
    input_tokens: int
    output_tokens: int
    seconds: float
    cost_usd: float


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


MODEL_SETTING = {"openai": "OPENAI_MODEL", "anthropic": "ANTHROPIC_MODEL",
                 "gemini": "GEMINI_MODEL", "ollama": "OLLAMA_AGENT_MODEL"}


def explain(provider, model, sdk, error):
    """Turn the most common API errors into a short message a beginner can act on."""
    key = config.PROVIDERS[provider]["key"]
    bad_key = isinstance(error, sdk.BadRequestError) and "api key" in str(error).lower()   # how Gemini says it
    if bad_key or isinstance(error, (sdk.AuthenticationError, sdk.PermissionDeniedError)):
        return LLMSetupError(f"Your {provider} API key was not accepted. Check {key} in the .env file.")
    if isinstance(error, sdk.BadRequestError) and "tools" in str(error).lower() and "not supported" in str(error).lower():
        return LLMSetupError(f"The model '{model}' can't use tools in this project. "
                             f"Choose another one with {MODEL_SETTING[provider]}=... in the .env file "
                             "(for OpenAI, gpt-5.4-mini works well).")
    if isinstance(error, sdk.NotFoundError):
        if provider == "ollama":
            return LLMSetupError(f"Ollama doesn't have the model '{model}' yet. Run: ollama pull {model}")
        return LLMSetupError(f"The model '{model}' is not available on your {provider} account. "
                             f"Choose another one with {MODEL_SETTING[provider]}=... in the .env file.")
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


class OpenAIChat:
    """OpenAI, and anything that speaks the same language: Gemini and Ollama do too.

    Same code, just a different web address (base_url) and key.
    """

    def __init__(self, provider="openai", model=None, client=None):
        settings = config.PROVIDERS[provider]
        self.provider = provider
        self.model = model or settings["model"]
        if client is None:
            from openai import OpenAI
            # max_retries: if the provider is busy, wait and try again a few times before giving up
            client = OpenAI(api_key=os.getenv(settings["key"], "ollama"), base_url=settings.get("base_url"),
                            max_retries=5)
        self.client = client

    def chat(self, system, history, tools=None, effort="medium", max_tokens=8000, json_schema=None):
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


class Claude:
    """Anthropic's Claude."""

    provider = "anthropic"

    def __init__(self, model=None, client=None):
        self.model = model or config.PROVIDERS["anthropic"]["model"]
        if client is None:
            import anthropic
            client = anthropic.Anthropic(max_retries=5)  # reads ANTHROPIC_API_KEY from the environment
        self.client = client

    def chat(self, system, history, tools=None, effort="medium", max_tokens=8000, json_schema=None):
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


def pick_provider():
    """Use LLM_PROVIDER from .env if set. Otherwise use the first API key we can find."""
    chosen = os.getenv("LLM_PROVIDER")
    if chosen:
        return chosen
    for provider in ["openai", "anthropic", "gemini"]:
        if os.getenv(config.PROVIDERS[provider]["key"]):
            return provider
    return "ollama"   # no keys at all: use the free local model


def make_llm(provider=None, model=None):
    provider = provider or pick_provider()
    if provider == "anthropic":
        return Claude(model=model)
    return OpenAIChat(provider, model=model)


class LocalLLM:
    """The small model for layer 2 of the router, running on your laptop with Ollama. Free and private."""

    def __init__(self, model=None, url=None):
        self.model = model or config.LOCAL_MODEL
        self.url = url or config.OLLAMA_URL

    def is_ready(self):
        """True if Ollama is running AND the model has been downloaded."""
        try:
            tags = requests.get(f"{self.url}/api/tags", timeout=2).json()
        except requests.RequestException:
            return False
        names = {m["name"] for m in tags.get("models", [])}
        return self.model in names or f"{self.model}:latest" in names

    def chat(self, system, user, max_tokens=300):
        """Ask the local model one question. Returns (text, stats)."""
        start = time.perf_counter()
        reply = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": max_tokens},
            },
            timeout=120,
        ).json()
        seconds = time.perf_counter() - start
        stats = CallStats(self.model, reply.get("prompt_eval_count", 0), reply.get("eval_count", 0), seconds, 0.0)
        return reply["message"]["content"].strip(), stats
