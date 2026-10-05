"""Version 4: the router. Send every message to the CHEAPEST model that can handle it.

    Layer 1 - ML model      : very sure + simple question  -> ready-made answer   (~1 ms, free)
    Layer 2 - local LLM     : fairly sure + simple question -> small model writes (~1 s, free)
    Layer 3 - cloud LLM agent: actions, angry customers, or unsure -> full agent with tools (paid)

Picking the right model for each request is a big part of "inference engineering".
"""

import time
from dataclasses import dataclass, field

from budget_agent import classifier, config
from budget_agent.agent import SupportAgent
from budget_agent.llm import LocalLLM
from budget_agent.tools import Shop, find_order_id

# Ready-made answers for simple questions. No LLM needed.
TEMPLATES = {
    "check_cancellation_fee": "There is no cancellation fee. Orders can be cancelled for free while they are Processing.",
    "check_invoice": "Invoices are emailed when your order ships. You can also download them from My Orders.",
    "get_invoice": "You can download your invoice from My Orders on the website. It is also emailed when the order ships.",
    "check_payment_methods": "We accept UPI, credit and debit cards, net banking, and cash on delivery for orders up to ₹5,000.",
    "check_refund_policy": "You can ask for a refund within 30 days of delivery. The money is back in 5-7 working days.",
    "contact_customer_service": "You can chat with us here 24/7 or email support@novamart.example.",
    "create_account": "Click Sign in, then Create account. It takes less than a minute.",
    "delivery_options": "Standard delivery takes 3-5 working days (free above ₹499). Express takes 1-2 days for ₹99.",
    "delivery_period": "Standard delivery takes 3-5 working days. Express delivery takes 1-2 working days.",
    "edit_account": "You can edit your details under Account → Settings.",
    "newsletter_subscription": "You can subscribe or unsubscribe under Account → Settings → Emails.",
    "place_order": "Add items to your cart and click Checkout. Need help with a product? Just ask!",
    "recover_password": "Click Sign in, then 'Forgot password'. We will email you a reset link.",
    "registration_problems": "Sorry about that! Please try again on the Sign in page, or tell me the error you see.",
    "review": "Thank you! You can leave a review on the product page under My Orders.",
    "set_up_shipping_address": "You can add a shipping address under Account → Addresses, or at checkout.",
    "switch_account": "Sign out from the menu, then sign in with your other account.",
}
ORDER_INTENTS = {"track_order", "track_refund"}   # simple, but need an order lookup


@dataclass
class RouteResult:
    reply: str
    layer: str                 # "ml", "local_llm" or "cloud_llm"
    intent: str
    confidence: float
    seconds: float
    cost_usd: float
    tool_calls: list = field(default_factory=list)


class Router:
    def __init__(self, model=None, agent=None, local_llm=None, shop=None):
        self.model = model or classifier.load()
        self.shop = shop or Shop()
        self.local_llm = local_llm or LocalLLM()
        self._agent = agent            # created only when first needed (so no API key is needed for layers 1-2)
        self._local_ready = None

    @property
    def agent(self):
        if self._agent is None:
            self._agent = SupportAgent(use_tools=True, shop=self.shop)
        return self._agent

    def local_ready(self):
        if self._local_ready is None:
            self._local_ready = self.local_llm.is_ready()
        return self._local_ready

    def choose_layer(self, text):
        """Decide which layer should answer. No LLM is called here, so this is instant."""
        intent, confidence = classifier.predict(self.model, text)
        simple = intent not in config.ACTION_INTENTS
        if simple and confidence >= config.ML_CONFIDENCE:
            return "ml", intent, confidence
        if simple and confidence >= config.LOCAL_LLM_CONFIDENCE and self.local_ready():
            return "local_llm", intent, confidence
        return "cloud_llm", intent, confidence

    def handle(self, text):
        start = time.perf_counter()
        layer, intent, confidence = self.choose_layer(text)

        if layer == "ml":
            reply = self._template_reply(intent, text)
            return RouteResult(reply, "ml", intent, confidence, time.perf_counter() - start, 0.0)

        if layer == "local_llm":
            system = ("You are NovaMart's support assistant. Answer in 1-3 short sentences, "
                      "using ONLY this policy:\n\n" + self.shop.policy)
            reply, stats = self.local_llm.chat(system, text)
            return RouteResult(reply, "local_llm", intent, confidence, time.perf_counter() - start, 0.0)

        result = self.agent.run(text)
        return RouteResult(result.reply, "cloud_llm", intent, confidence,
                           time.perf_counter() - start, result.cost_usd, result.tool_calls)

    def _template_reply(self, intent, text):
        if intent in ORDER_INTENTS:
            order_id = find_order_id(text)
            if order_id is None:
                return "Sure! What is your order number? It looks like NM10234."
            order = self.shop.lookup_order(order_id)
            if "error" in order:
                return f"I couldn't find order {order_id}. Could you check the number?"
            if intent == "track_refund":
                if order["status"] == "Refund requested":
                    return f"Your refund for {order_id} is on its way. It takes 5-7 working days."
                return f"I don't see a refund for {order_id} yet. Its status is: {order['status']}."
            return (f"Your order {order_id} ({order['product']}) is {order['status']}. "
                    f"Delivery date: {order['delivery_date']}.")
        return TEMPLATES.get(intent, "Thanks for your message! A support agent will help you shortly.")

