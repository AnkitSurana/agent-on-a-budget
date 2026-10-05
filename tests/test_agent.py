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
    assert result.cost_usd == pytest.approx(2 * cost_usd("gpt-5.4-mini", 1000, 100))


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


def test_big_refund_is_marked_as_escalated(shop):
    # NM10003 in the test shop is a ₹12,999 chair: the refund tool itself hands it to a human
    agent, _ = openai_agent([
        openai_tools([("call_1", "start_refund", {"order_id": "NM10003", "reason": "broke"})]),
        openai_text("A manager will approve this refund."),
    ], shop)
    assert agent.run("refund NM10003").escalated is True


# ---------- setup problems become short, plain-English messages ----------

def failing_client(error):
    def create(**request):
        raise error
    from types import SimpleNamespace
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_wrong_key_gives_a_plain_message():
    import httpx2
    import openai
    from budget_agent.llm import LLMSetupError

    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    error = openai.AuthenticationError("bad key", response=httpx2.Response(401, request=request), body=None)
    llm = OpenAIChat("openai", client=failing_client(error))
    with pytest.raises(LLMSetupError, match="OPENAI_API_KEY"):
        llm.chat("system", [{"role": "user", "content": "hi"}])


def test_ollama_not_running_gives_a_plain_message():
    import httpx2
    import openai
    from budget_agent.llm import LLMSetupError

    error = openai.APIConnectionError(request=httpx2.Request("POST", "http://localhost:11434/v1/chat/completions"))
    llm = OpenAIChat("ollama", client=failing_client(error))
    with pytest.raises(LLMSetupError, match="ollama serve"):
        llm.chat("system", [{"role": "user", "content": "hi"}])


def test_models_that_need_reasoning_off_get_a_second_try():
    import httpx2
    import openai

    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    needs_off = openai.BadRequestError(
        "Function tools with reasoning_effort are not supported for this model. "
        "To use function tools, use /v1/responses or set reasoning_effort to 'none'.",
        response=httpx2.Response(400, request=request), body=None)
    answers = [needs_off, openai_text("It shipped!")]
    requests = []

    def create(**req):
        requests.append(dict(req))
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    from types import SimpleNamespace
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    tools = [{"name": "lookup_order", "description": "d", "input_schema": {"type": "object", "properties": {}}}]
    reply = OpenAIChat("openai", client=client).chat("system", [{"role": "user", "content": "hi"}], tools=tools)
    assert reply.text == "It shipped!"
    assert "reasoning_effort" not in requests[0] and requests[1]["reasoning_effort"] == "none"
