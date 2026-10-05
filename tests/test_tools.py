import json

from budget_agent.tools import TOOL_DEFINITIONS, find_order_id


def test_find_order_id():
    assert find_order_id("where is nm10234 please") == "NM10234"
    assert find_order_id("my order hasn't come") is None


def test_lookup_order(shop):
    assert shop.lookup_order("nm10001")["product"] == "Smart Watch"
    assert "error" in shop.lookup_order("NM99999")


def test_refund_works_for_recent_delivered_order(shop):
    result = shop.start_refund("NM10001", "broken")
    assert result["ok"] is True
    assert shop.orders["NM10001"]["status"] == "Refund requested"


def test_refund_refused_when_not_delivered(shop):
    assert shop.start_refund("NM10002", "changed my mind")["ok"] is False


def test_refund_refused_after_30_days(shop):
    assert "30 days" in shop.start_refund("NM10004", "too late")["why"]


def test_big_refund_goes_to_a_human(shop):
    result = shop.start_refund("NM10003", "chair broke")
    assert result["ok"] is False
    assert shop.actions[-1]["action"] == "escalate_to_human"
    assert shop.orders["NM10003"]["status"] == "Delivered"   # nothing was refunded


def test_cancel_only_while_processing(shop):
    assert shop.cancel_order("NM10002")["ok"] is True
    assert shop.cancel_order("NM10005")["ok"] is False      # already shipped


def test_update_address_only_while_processing(shop):
    assert shop.update_address("NM10002", "12 MG Road")["ok"] is True
    assert shop.update_address("NM10001", "12 MG Road")["ok"] is False


def test_run_tool_returns_json_even_for_mistakes(shop):
    assert json.loads(shop.run_tool("lookup_order", {"order_id": "NM10001"}))["city"] == "Pune"
    assert "error" in json.loads(shop.run_tool("fly_to_moon", {}))
    assert "error" in json.loads(shop.run_tool("lookup_order", {"wrong_name": "x"}))


def test_every_tool_definition_has_a_function(shop):
    for tool in TOOL_DEFINITIONS:
        assert hasattr(shop, tool["name"]), tool["name"]
