"""The tools our agent can use. A tool is just a normal Python function.

The LLM cannot touch the database itself. It can only *ask* us to run one of
these functions. We run it and send back the result. That keeps us in control.
"""

import json
import re
from datetime import date

from budget_agent import config
from budget_agent.data import load_orders, load_policy

HUMAN_APPROVAL_LIMIT_INR = 10_000   # refunds above this need a human (a "guardrail")
REFUND_WINDOW_DAYS = 30


def find_order_id(text):
    """Find an order number like NM10234 in a message. Returns None if there is none."""
    match = re.search(r"\bNM\d{5}\b", text.upper())
    return match.group(0) if match else None


class Shop:
    """A tiny in-memory copy of the NovaMart database.

    Every change is also written to `self.actions`, so we can see what the agent did.
    """

    def __init__(self, orders=None, policy=None):
        orders = load_orders() if orders is None else orders
        self.orders = {row["order_id"]: dict(row) for row in orders.to_dict("records")}
        self.policy = load_policy() if policy is None else policy
        self.actions = []

    # ---------- tools ----------
    def lookup_order(self, order_id):
        order = self.orders.get(order_id.strip().upper())
        if order is None:
            return {"error": f"No order found with id {order_id}."}
        return order

    def get_policy(self):
        return {"policy": self.policy}

    def start_refund(self, order_id, reason):
        order = self.orders.get(order_id.strip().upper())
        if order is None:
            return {"error": f"No order found with id {order_id}."}
        if order["status"] != "Delivered":
            return {"ok": False, "why": f"Order is '{order['status']}'. Only delivered orders can be refunded."}
        days = (config.TODAY - date.fromisoformat(order["delivery_date"])).days
        if days > REFUND_WINDOW_DAYS:
            return {"ok": False, "why": f"Delivered {days} days ago. The refund window is {REFUND_WINDOW_DAYS} days."}
        if order["amount_inr"] > HUMAN_APPROVAL_LIMIT_INR:
            self.escalate_to_human(f"Refund over ₹{HUMAN_APPROVAL_LIMIT_INR:,} for {order_id}: {reason}", "high")
            return {"ok": False, "why": "Refunds above ₹10,000 need a human. A human agent has been asked to approve it."}
        order["status"] = "Refund requested"
        self.actions.append({"action": "start_refund", "order_id": order["order_id"], "reason": reason})
        return {"ok": True, "message": f"Refund of ₹{order['amount_inr']:,} started. It arrives in 5-7 working days."}

    def cancel_order(self, order_id):
        order = self.orders.get(order_id.strip().upper())
        if order is None:
            return {"error": f"No order found with id {order_id}."}
        if order["status"] != "Processing":
            return {"ok": False, "why": f"Order is '{order['status']}'. Only 'Processing' orders can be cancelled."}
        order["status"] = "Cancelled"
        self.actions.append({"action": "cancel_order", "order_id": order["order_id"]})
        return {"ok": True, "message": "Order cancelled. No fee was charged."}

    def update_address(self, order_id, new_address):
        order = self.orders.get(order_id.strip().upper())
        if order is None:
            return {"error": f"No order found with id {order_id}."}
        if order["status"] != "Processing":
            return {"ok": False, "why": f"Order is '{order['status']}'. The address can only change while 'Processing'."}
        order["address"] = new_address
        self.actions.append({"action": "update_address", "order_id": order["order_id"], "new_address": new_address})
        return {"ok": True, "message": f"Address updated to: {new_address}"}

    def escalate_to_human(self, reason, priority="normal"):
        ticket = f"T{len(self.actions) + 1:04d}"
        self.actions.append({"action": "escalate_to_human", "ticket": ticket, "reason": reason, "priority": priority})
        return {"ok": True, "ticket": ticket, "message": "A human agent will reply within 2 hours."}

    # ---------- run a tool by name (the agent calls this) ----------
    def run_tool(self, name, tool_input):
        tools = {
            "lookup_order": self.lookup_order,
            "get_policy": self.get_policy,
            "start_refund": self.start_refund,
            "cancel_order": self.cancel_order,
            "update_address": self.update_address,
            "escalate_to_human": self.escalate_to_human,
        }
        if name not in tools:
            return json.dumps({"error": f"Unknown tool {name}"})
        try:
            return json.dumps(tools[name](**tool_input), default=str, ensure_ascii=False)
        except TypeError as e:  # the LLM sent the wrong arguments
            return json.dumps({"error": f"Bad arguments for {name}: {e}"})


def _tool(name, description, properties=None, required=None):
    """Small helper so each tool description below fits on a few lines."""
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties or {},
            "required": required or [],
            "additionalProperties": False,
        },
    }


ORDER_ID = {"type": "string", "description": "Order number, like NM10234"}

# This is how we describe the tools to the LLM. It reads the descriptions to decide what to call.
TOOL_DEFINITIONS = [
    _tool("lookup_order", "Get the details and status of an order. Always do this before acting on an order.",
          {"order_id": ORDER_ID}, ["order_id"]),
    _tool("get_policy", "Read the NovaMart policy on refunds, cancellations, delivery, payments and accounts."),
    _tool("start_refund", "Start a refund for a delivered order.",
          {"order_id": ORDER_ID, "reason": {"type": "string"}}, ["order_id", "reason"]),
    _tool("cancel_order", "Cancel an order that is still Processing.", {"order_id": ORDER_ID}, ["order_id"]),
    _tool("update_address", "Change the delivery address of an order that is still Processing.",
          {"order_id": ORDER_ID, "new_address": {"type": "string"}}, ["order_id", "new_address"]),
    _tool("escalate_to_human",
          "Hand the conversation to a human agent. Use when the customer is very upset, asks for a human, "
          "has a payment problem, wants to delete their account, or when you cannot solve the problem.",
          {"reason": {"type": "string"}, "priority": {"type": "string", "enum": ["normal", "high"]}},
          ["reason", "priority"]),
]
