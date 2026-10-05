import json
import random

import pandas as pd
import pytest
from conftest import FakeOpenAIClient, openai_text

from budget_agent import classifier
from budget_agent.agent import AgentResult
from budget_agent.data import fill_placeholders, make_messages, make_orders
from budget_agent.labeler import label_batch
from budget_agent.llm import CallStats, OpenAIChat
from budget_agent.router import Router

TRAIN = {
    "track_order": ["where is my order", "track my order", "order status please", "when will my order arrive",
                    "has my order shipped", "where is my package"],
    "check_payment_methods": ["what payment methods do you accept", "can i pay with upi", "do you take cards",
                              "payment options please", "which payment methods are there", "can i pay by card"],
    "get_refund": ["i want a refund", "give me my money back", "refund my order", "i need a refund please",
                   "how do i get my money back", "refund please"],
}


@pytest.fixture(scope="module")
def model():
    texts = [t for ts in TRAIN.values() for t in ts]
    labels = [intent for intent, ts in TRAIN.items() for _ in ts]
    return classifier.train(texts, labels)


# ---------- data ----------

def test_orders_are_the_same_every_time():
    assert make_orders(n=20).equals(make_orders(n=20))
    assert make_orders(n=20)["order_id"].is_unique


def test_placeholders_are_filled():
    text = fill_placeholders("cancel {{Order Number}} for {{Person Name}}", ["NM10001"], random.Random(0))
    assert text == "cancel NM10001 for Rahul"


def test_messages_are_balanced_and_split():
    raw = pd.DataFrame({"instruction": [f"msg {i}" for i in range(60)], "intent": ["a", "b", "c"] * 20})
    df = make_messages(raw, ["NM10001"], train_per_intent=4, test_per_intent=2)
    assert len(df) == 18
    assert (df.split == "train").sum() == 12
    assert df.groupby("true_intent").size().tolist() == [6, 6, 6]


# ---------- ML model ----------

def test_classifier_predicts_with_confidence(model):
    intent, confidence = classifier.predict(model, "where is my order")
    assert intent == "track_order"
    assert 0 < confidence <= 1


def test_classifier_save_and_load(model, tmp_path):
    classifier.save(model, tmp_path / "m.joblib")
    loaded = classifier.load(tmp_path / "m.joblib")
    assert classifier.predict(loaded, "refund please")[0] == "get_refund"


# ---------- labelling ----------

def test_label_batch_reads_json():
    answer = json.dumps({"labels": [{"id": 1, "intent": "track_order"}, {"id": 2, "intent": "get_refund"}]})
    client = FakeOpenAIClient([openai_text(answer)])
    batch = pd.DataFrame({"id": [1, 2], "text": ["where is it", "money back"]})
    labels, stats = label_batch(OpenAIChat("openai", client=client), batch)
    assert labels == {1: "track_order", 2: "get_refund"}
    assert client.requests[0]["response_format"]["type"] == "json_schema"


# ---------- router ----------

class FakeLocalLLM:
    def __init__(self, ready=True):
        self.ready = ready

    def is_ready(self):
        return self.ready

    def chat(self, system, user):
        return "local answer", CallStats("local", 10, 10, 0.5, 0.0)


class FakeAgent:
    def run(self, text):
        return AgentResult(reply="cloud answer", calls=[CallStats("gpt-6.1-sol", 1000, 100, 2.0, 0.006)])


def make_router(model, shop, local_ready=True):
    return Router(model=model, agent=FakeAgent(), local_llm=FakeLocalLLM(local_ready), shop=shop)


def test_sure_and_simple_goes_to_ml(model, shop, monkeypatch):
    monkeypatch.setattr("budget_agent.config.ML_CONFIDENCE", 0.0)
    result = make_router(model, shop).handle("what payment methods do you accept")
    assert result.layer == "ml"
    assert result.cost_usd == 0
    assert "UPI" in result.reply


def test_ml_layer_looks_up_the_order(model, shop, monkeypatch):
    monkeypatch.setattr("budget_agent.config.ML_CONFIDENCE", 0.0)
    result = make_router(model, shop).handle("where is my order NM10005")
    assert result.layer == "ml"
    assert "Shipped" in result.reply


def test_medium_confidence_goes_to_local_llm(model, shop, monkeypatch):
    monkeypatch.setattr("budget_agent.config.ML_CONFIDENCE", 1.01)        # ML is never "sure enough"
    monkeypatch.setattr("budget_agent.config.LOCAL_LLM_CONFIDENCE", 0.0)
    assert make_router(model, shop).handle("can i pay with upi").layer == "local_llm"


def test_no_local_llm_means_cloud_llm(model, shop, monkeypatch):
    monkeypatch.setattr("budget_agent.config.ML_CONFIDENCE", 1.01)
    monkeypatch.setattr("budget_agent.config.LOCAL_LLM_CONFIDENCE", 0.0)
    assert make_router(model, shop, local_ready=False).handle("can i pay with upi").layer == "cloud_llm"


def test_actions_always_go_to_cloud_llm(model, shop, monkeypatch):
    monkeypatch.setattr("budget_agent.config.ML_CONFIDENCE", 0.0)       # even when ML is "sure"
    result = make_router(model, shop).handle("i want a refund")
    assert result.layer == "cloud_llm"
    assert result.cost_usd == pytest.approx(0.006)
