"""Editing .env while the app or a notebook is running: the next question uses the new values."""

import os

import pytest

from budget_agent import config
from budget_agent.llm import make_llm

NAMES = ["LLM_PROVIDER", "LLM_API_KEY", "LLM_MODEL", "REPLAY"]


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)
    for setting in ["REPLAY", "PROVIDERS", "LOCAL_MODEL", "OLLAMA_URL", "_env_seen", "_env_time"]:
        monkeypatch.setattr(config, setting, getattr(config, setting))     # put everything back afterwards
    path = tmp_path / ".env"
    monkeypatch.setattr(config, "ENV_FILE", path)
    monkeypatch.setattr(config, "_env_seen", {})
    monkeypatch.setattr(config, "_env_time", None)

    def write(text):
        path.write_text(text)
        stamp = (config._env_time or 0) + 10                # a newer time, as a real save would give
        os.utime(path, (stamp, stamp))
    return write


def test_a_new_key_is_used_without_restarting(env_file):
    env_file("LLM_PROVIDER=openai\nLLM_API_KEY=sk-your-key-here\n")
    llm = make_llm()
    assert llm.llm.client.api_key == "sk-your-key-here"

    env_file("LLM_PROVIDER=openai\nLLM_API_KEY=sk-proj-the-real-one\n")       # you paste the real key and save
    assert llm.llm.client.api_key == "sk-proj-the-real-one"


def test_switching_provider_without_restarting(env_file):
    env_file("LLM_PROVIDER=openai\nLLM_API_KEY=sk-proj-one\n")
    llm = make_llm()
    assert llm.provider == "openai"
    env_file("LLM_PROVIDER=anthropic\nLLM_API_KEY=sk-ant-two\n")              # same key name, other provider
    assert llm.provider == "anthropic" and llm.llm.client.api_key == "sk-ant-two"


def test_replay_can_be_switched_off_and_on_again(env_file):
    env_file("REPLAY=off\n")
    config.refresh()
    assert config.REPLAY is False
    env_file("")                                             # line deleted: back to the default
    config.refresh()
    assert config.REPLAY is True and "REPLAY" not in os.environ


def test_a_value_set_in_the_terminal_still_wins(env_file, monkeypatch):
    env_file("LLM_MODEL=gpt-5.4-mini\n")
    config.refresh()
    monkeypatch.setenv("LLM_MODEL", "gpt-6.1-sol")              # for example: LLM_MODEL=... uv run ...
    env_file("LLM_MODEL=gpt-5.4-mini\nREPLAY=off\n")             # a different line changed
    config.refresh()
    assert os.environ["LLM_MODEL"] == "gpt-6.1-sol"


def test_nothing_changed_means_nothing_to_do(env_file):
    env_file("REPLAY=off\n")
    assert config.refresh() is True
    assert config.refresh() is False
