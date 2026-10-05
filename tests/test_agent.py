"""The agent loop, tested with fake OpenAI and fake Claude clients (no internet, no API key)."""

import pytest
from conftest import (FakeAnthropicClient, FakeOpenAIClient, fake_response, openai_text, openai_tools,
                      text_block, tool_block)

from budget_agent.agent import SupportAgent
from budget_agent.llm import Claude, OpenAIChat, cost_usd, pick_provider


def openai_agent(responses, shop, use_tools=True):
    client = FakeOpenAIClient(responses)
    return SupportAgent(use_tools=use_tools, llm=OpenAIChat("openai", client=client), shop=shop), client


def claude_agent(responses, shop, use_tools=True):
    client = FakeAnthropicClient(responses)
    return SupportAgent(use_tools=use_tools, llm=Claude(client=client), shop=shop), client


# ---------- cost and provider choice ----------

def test_cost_calculation():
    # 1M input tokens at $2 + 1M output tokens at $10
    assert cost_usd("gpt-6.1-sol", 1_000_000, 1_000_000) == pytest.approx(12.0)
    assert cost_usd("llama3.2:3b", 5000, 5000) == 0.0       # local models are free


def test_provider_follows_your_keys(monkeypatch):
    for name in ["LLM_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]:
        monkeypatch.delenv(name, raising=False)
    assert pick_provider() == "ollama"                       # no keys -> free local model
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert pick_provider() == "gemini"
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert pick_provider() == "openai"
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")          # an explicit choice always wins
    assert pick_provider() == "anthropic"


# ---------- OpenAI-style providers (OpenAI, Gemini, Ollama) ----------

def test_v1_just_replies(shop):
    agent, client = openai_agent([openai_text("Hi! How can I help?")], shop, use_tools=False)
    assert agent.run("hello").reply == "Hi! How can I help?"
    assert "tools" not in client.requests[0]


def test_v2_runs_a_tool_then_answers(shop):
    agent, client = openai_agent([
        openai_tools([("call_1", "lookup_order", {"order_id": "NM10005"})]),
        openai_text("Your desk lamp has shipped!"),
    ], shop)

    result = agent.run("where is NM10005")

    assert result.reply == "Your desk lamp has shipped!"
    assert result.tool_calls[0][0] == "lookup_order"
    assert '"Shipped"' in result.tool_calls[0][2]
    # The tool result went back to the LLM in the second request
    last = client.requests[1]["messages"][-1]
    assert last["role"] == "tool" and last["tool_call_id"] == "call_1"
    assert result.cost_usd == pytest.approx(2 * cost_usd("gpt-6.1-sol", 1000, 100))


def test_v2_sends_every_tool_result(shop):
    agent, client = openai_agent([
        openai_tools([("a", "lookup_order", {"order_id": "NM10001"}), ("b", "lookup_order", {"order_id": "NM10002"})]),
        openai_text("Done"),
    ], shop)
    agent.run("check both orders")
    tool_messages = [m for m in client.requests[1]["messages"] if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["a", "b"]


def test_agent_stops_after_too_many_steps(shop):
    agent, _ = openai_agent([openai_tools([(str(i), "get_policy", {})]) for i in range(3)], shop)
    agent.max_steps = 3
    assert "human" in agent.run("loop forever").reply


def test_refusal_is_handled(shop):
    agent, _ = openai_agent([openai_text("", finish_reason="content_filter")], shop)
    assert "human" in agent.run("something unsafe").reply


# ---------- Claude ----------

def test_claude_runs_a_tool_then_answers(shop):
    agent, client = claude_agent([
        fake_response([tool_block("lookup_order", {"order_id": "NM10005"})], "tool_use"),
        fake_response([text_block("Your desk lamp has shipped!")], "end_turn"),
    ], shop)

    result = agent.run("where is NM10005")

    assert result.reply == "Your desk lamp has shipped!"
    last = client.requests[1]["messages"][-1]
    assert last["content"][0]["type"] == "tool_result"
    assert last["content"][0]["tool_use_id"] == "tool_1"


def test_claude_refusal_is_handled(shop):
    agent, _ = claude_agent([fake_response([], "refusal")], shop)
    assert "human" in agent.run("something unsafe").reply
