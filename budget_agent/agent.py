"""Steps 2 and 3: the LLM alone, then the LLM with tools (the agent).

use_tools=False (chat --version 1): the LLM alone. It can talk, but it cannot DO anything.
use_tools=True  (chat --version 2): the LLM can call tools (look up orders, start refunds, ...).
                                    Now it is a real agent: the "senior expert" of the support office.

The agent loop is the heart of every AI agent:
    1. Send the conversation to the LLM.
    2. If the LLM asks for a tool, run the tool and send back the result.
    3. Repeat until the LLM gives a final answer.
"""

from dataclasses import dataclass, field

from budget_agent import config
from budget_agent.llm import make_llm
from budget_agent.tools import TOOL_DEFINITIONS, Shop

SYSTEM_PROMPT = f"""You are the customer support assistant for NovaMart, an online shop in India.
Today is {config.TODAY:%d %B %Y}.

How to work:
- Be friendly and short: 2-4 sentences.
- Never make up order details. Look the order up first.
- If the customer did not give an order number and you need one, ask for it.
- Follow the NovaMart policy. Read it with get_policy when you are not sure.
- Hand over to a human (escalate_to_human) when the customer is very upset, asks for a human,
  has a payment problem, wants to delete their account, or when you cannot help."""

NO_TOOLS_PROMPT = SYSTEM_PROMPT.split("How to work:")[0] + (
    "Be friendly and short: 2-4 sentences. You have no access to orders or systems."
)


@dataclass
class AgentResult:
    reply: str
    tool_calls: list = field(default_factory=list)   # [(tool name, input, result), ...]
    calls: list = field(default_factory=list)        # one CallStats per LLM request
    escalated: bool = False                          # True if a human (the manager) was asked to take over

    @property
    def seconds(self):
        return sum(c.seconds for c in self.calls)

    @property
    def cost_usd(self):
        return sum(c.cost_usd for c in self.calls)

    @property
    def replayed(self):
        """True if every LLM answer came from the saved answers (so it cost nothing this time)."""
        return bool(self.calls) and all(c.replayed for c in self.calls)


class SupportAgent:
    def __init__(self, use_tools=True, llm=None, shop=None, max_steps=8):
        self.use_tools = use_tools
        self.llm = llm or make_llm()      # OpenAI, Claude, Gemini or Ollama, from your .env
        self.shop = shop or Shop()
        self.max_steps = max_steps

    def fresh_shop(self):
        """Start again from the original orders, so the same question always gets the same answer."""
        self.shop = Shop()

    def run(self, user_message):
        system = SYSTEM_PROMPT if self.use_tools else NO_TOOLS_PROMPT
        tools = TOOL_DEFINITIONS if self.use_tools else None
        history = [{"role": "user", "content": user_message}]
        result = AgentResult(reply="")
        actions_before = len(self.shop.actions)

        if tools and not getattr(self.llm, "can_use_tools", lambda *a: True)(system, history, tools):
            return self._hand_over_without_key(user_message)

        for _ in range(self.max_steps):
            reply = self.llm.chat(system, history, tools=tools)
            result.calls.append(reply.stats)

            if reply.stop == "refused":
                result.reply = "Sorry, I can't help with that here. Let me connect you to a human agent."
                return result

            if reply.stop != "tool_use":       # the LLM gave its final answer
                result.reply = reply.text
                return result

            # The LLM asked for one or more tools. Run every one, then send ALL results back together.
            history.append({"role": "assistant", "reply": reply})
            outputs = []
            for call in reply.tool_calls:
                output = self.shop.run_tool(call.name, call.input)
                result.tool_calls.append((call.name, call.input, output))
                outputs.append((call.id, output))
            history.append({"role": "tool_results", "results": outputs})
            result.escalated = any(a["action"] == "escalate_to_human" for a in self.shop.actions[actions_before:])

        result.reply = "Sorry, this is taking too long. A human agent will follow up with you."
        return result

    def _hand_over_without_key(self, user_message):
        """No cloud key and no saved answer: don't let the small local model try refunds. A human takes it."""
        reason = f"No cloud LLM key, so the senior expert could not handle: {user_message[:150]}"
        output = self.shop.run_tool("escalate_to_human", {"reason": reason, "priority": "normal"})
        ticket = self.shop.actions[-1]["ticket"]
        reply = (f"I can't handle this one myself yet: requests like refunds need the cloud AI, and no API key is set up in .env. "
                 f"So I've passed it to a human: ticket {ticket}, and someone will reply within 2 hours.")
        return AgentResult(reply=reply, tool_calls=[("escalate_to_human", {"reason": reason, "priority": "normal"}, output)],
                           escalated=True)
