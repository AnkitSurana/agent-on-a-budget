"""Shared test helpers. The tests never call a real LLM, so they are free, fast and work offline."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from openai.types.chat import ChatCompletionMessage

from budget_agent.tools import Shop


@pytest.fixture
def orders():
    return pd.DataFrame([
        # order_id, status, amount, delivery_date   (config.TODAY is 2026-10-05)
        {"order_id": "NM10001", "customer_name": "Priya Sharma", "city": "Pune", "product": "Smart Watch",
         "quantity": 1, "amount_inr": 5999, "order_date": "2026-09-20", "status": "Delivered",
         "delivery_date": "2026-09-25"},
        {"order_id": "NM10002", "customer_name": "Rahul Rao", "city": "Delhi", "product": "Backpack",
         "quantity": 1, "amount_inr": 1799, "order_date": "2026-10-04", "status": "Processing",
         "delivery_date": "2026-10-08"},
        {"order_id": "NM10003", "customer_name": "Isha Nair", "city": "Mumbai", "product": "Office Chair",
         "quantity": 1, "amount_inr": 12999, "order_date": "2026-09-25", "status": "Delivered",
         "delivery_date": "2026-09-30"},
        {"order_id": "NM10004", "customer_name": "Kabir Das", "city": "Jaipur", "product": "Yoga Mat",
         "quantity": 1, "amount_inr": 899, "order_date": "2026-08-01", "status": "Delivered",
         "delivery_date": "2026-08-05"},
        {"order_id": "NM10005", "customer_name": "Meera Iyer", "city": "Chennai", "product": "Desk Lamp",
         "quantity": 1, "amount_inr": 1299, "order_date": "2026-10-01", "status": "Shipped",
         "delivery_date": "2026-10-07"},
    ])


@pytest.fixture
def shop(orders):
    return Shop(orders=orders, policy="Refunds within 30 days of delivery.")


# ---------- A fake Claude, so we can test the agent loop without the internet ----------

def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(name, tool_input, block_id="tool_1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def fake_response(blocks, stop_reason, model="claude-opus-5-5"):
    return SimpleNamespace(content=blocks, stop_reason=stop_reason, model=model,
                           usage=SimpleNamespace(input_tokens=1000, output_tokens=100))


class FakeAnthropicClient:
    """Pretends to be anthropic.Anthropic(). Gives back the responses we prepared, in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        return self.responses.pop(0)


# ---------- A fake OpenAI (also how Gemini and Ollama talk) ----------

def openai_text(text, model="gpt-5.4-mini", finish_reason="stop"):
    message = ChatCompletionMessage(role="assistant", content=text)
    return _openai_response(message, finish_reason, model)


def openai_tools(calls, model="gpt-5.4-mini"):
    """calls: [(id, name, arguments_dict), ...]"""
    message = ChatCompletionMessage.model_validate({"role": "assistant", "content": None, "tool_calls": [
        {"id": i, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for i, n, a in calls]})
    return _openai_response(message, "tool_calls", model)


def _openai_response(message, finish_reason, model):
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
                           usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=100), model=model)


class FakeOpenAIClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        return self.responses.pop(0)
