"""Saved answers: the same request is answered from the file, without calling the LLM again."""

import json

import pytest
from conftest import FakeOpenAIClient, openai_text, openai_tools

from budget_agent import replay
from budget_agent.agent import SupportAgent
from budget_agent.llm import OpenAIChat, SavedOrLocal, make_llm


@pytest.fixture
def saved_answers(tmp_path, monkeypatch):
    log = tmp_path / "saved.jsonl"
    monkeypatch.setattr("budget_agent.config.REPLAY", True)
    monkeypatch.setattr("budget_agent.config.REPLAY_LOG", log)
    monkeypatch.setattr(replay, "_saved", None)     # start with an empty memory
    return log


def test_second_identical_request_is_free(saved_answers):
    client = FakeOpenAIClient([openai_text("Hello!")])   # only ONE real answer available
    llm = OpenAIChat("openai", client=client)
    history = [{"role": "user", "content": "hi"}]

    first = llm.chat("system", history)
    second = llm.chat("system", history)                 # would crash if it called the client again

    assert first.text == second.text == "Hello!"
    assert first.stats.replayed is False and second.stats.replayed is True
    assert second.stats.cost_usd == first.stats.cost_usd  # the original cost is kept, for fair benchmarks
    assert len(client.requests) == 1
    assert len(saved_answers.read_text().splitlines()) == 1


def test_a_different_question_still_asks_the_llm(saved_answers):
    client = FakeOpenAIClient([openai_text("A"), openai_text("B")])
    llm = OpenAIChat("openai", client=client)
    assert llm.chat("system", [{"role": "user", "content": "one"}]).text == "A"
    assert llm.chat("system", [{"role": "user", "content": "two"}]).text == "B"
    assert len(client.requests) == 2


def test_whole_agent_conversation_replays(saved_answers, shop):
    def agent_with(responses):
        client = FakeOpenAIClient(responses)
        return SupportAgent(llm=OpenAIChat("openai", client=client), shop=shop), client

    agent, _ = agent_with([openai_tools([("c1", "lookup_order", {"order_id": "NM10005"})]),
                           openai_text("Your desk lamp has shipped!")])
    first = agent.run("where is NM10005")

    agent, client = agent_with([])                       # no answers left: everything must come from the file
    again = agent.run("where is NM10005")
    assert again.reply == first.reply and again.replayed
    assert client.requests == []


def test_saved_answers_survive_a_restart(saved_answers):
    OpenAIChat("openai", client=FakeOpenAIClient([openai_text("Saved")])).chat("s", [{"role": "user", "content": "q"}])
    replay._saved = None                                 # like starting Python again
    reply = OpenAIChat("openai", client=FakeOpenAIClient([])).chat("s", [{"role": "user", "content": "q"}])
    assert reply.text == "Saved" and reply.stats.replayed
    assert json.loads(saved_answers.read_text())["text"] == "Saved"


def test_replay_can_be_switched_off(saved_answers, monkeypatch):
    monkeypatch.setattr("budget_agent.config.REPLAY", False)
    client = FakeOpenAIClient([openai_text("1"), openai_text("2")])
    llm = OpenAIChat("openai", client=client)
    llm.chat("s", [{"role": "user", "content": "q"}])
    assert llm.chat("s", [{"role": "user", "content": "q"}]).text == "2"
    assert not saved_answers.exists()


def test_no_key_replays_saved_answers_and_asks_ollama_only_for_new_ones(saved_answers):
    OpenAIChat("openai", client=FakeOpenAIClient([openai_text("Saved!")])).chat("system", [{"role": "user", "content": "old"}])
    ollama = FakeOpenAIClient([openai_text("Fresh from Ollama", model="llama3.2:3b")])
    llm = SavedOrLocal(saved=OpenAIChat("openai", client=FakeOpenAIClient([])),   # would crash if really called
                       local=OpenAIChat("ollama", client=ollama))

    assert llm.chat("system", [{"role": "user", "content": "old"}]).text == "Saved!"
    assert llm.chat("system", [{"role": "user", "content": "new"}]).text == "Fresh from Ollama"
    assert len(ollama.requests) == 1


def test_make_llm_with_no_keys_uses_saved_answers(monkeypatch):
    assert isinstance(make_llm().llm, SavedOrLocal)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")                  # asked for Ollama on purpose: just Ollama
    assert not isinstance(make_llm().llm, SavedOrLocal)


def test_a_provider_with_no_key_still_uses_saved_answers(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")                  # as copied from .env.example, key left empty
    monkeypatch.setenv("LLM_API_KEY", "")
    llm = make_llm()
    assert isinstance(llm.llm, SavedOrLocal)
    assert (llm.provider, llm.model) == ("openai", "gpt-5.4-mini")   # the saved answers' model, so prices match them


def test_no_key_new_question_for_the_expert_goes_to_a_human(saved_answers):
    ollama = FakeOpenAIClient([])                                   # would crash if Llama were asked
    llm = SavedOrLocal(saved=OpenAIChat("openai", client=FakeOpenAIClient([])), local=OpenAIChat("ollama", client=ollama))
    agent = SupportAgent(use_tools=True, llm=llm)

    result = agent.run("refund my order NM10002 please, the strap broke")

    assert result.escalated and "API key" in result.reply and "T0001" in result.reply
    assert agent.shop.actions[0]["action"] == "escalate_to_human"
    assert not ollama.requests
