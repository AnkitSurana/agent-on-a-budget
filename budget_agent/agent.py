"""Version 1 and Version 2 of our agent.

V1 (use_tools=False): the LLM just reads the message and writes a reply. It can talk, but it cannot DO anything.
V2 (use_tools=True):  the LLM can call tools (look up orders, start refunds, ...). Now it is a real agent.

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
    layer: str = "llm"
    tool_calls: list = field(default_factory=list)   # [(tool name, input, result), ...]
    calls: list = field(default_factory=list)        # one CallStats per LLM request

    @property
    def seconds(self):
        return sum(c.seconds for c in self.calls)

    @property
    def cost_usd(self):
        return sum(c.cost_usd for c in self.calls)


class SupportAgent:
    def __init__(self, use_tools=True, llm=None, shop=None, max_steps=8):
        self.use_tools = use_tools
        self.llm = llm or make_llm()      # OpenAI, Claude, Gemini or Ollama, from your .env
        self.shop = shop or Shop()
        self.max_steps = max_steps

    def run(self, user_message):
        system = SYSTEM_PROMPT if self.use_tools else NO_TOOLS_PROMPT
        tools = TOOL_DEFINITIONS if self.use_tools else None
        history = [{"role": "user", "content": user_message}]
        result = AgentResult(reply="")

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

        result.reply = "Sorry, this is taking too long. A human agent will follow up with you."
        return result
