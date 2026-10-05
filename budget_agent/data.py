"""Make and load the data for NovaMart, our pretend online shop.

Three things are created here:
1. messages.csv - real-looking customer messages from the Bitext dataset
2. orders.csv   - fake orders, so the agent has something to look up
3. (policy.md and messy_messages.csv are written by hand and live in data/)
"""

import random
import re
from datetime import timedelta

import pandas as pd

from budget_agent import config

PRODUCTS = [
    ("Wireless Earbuds", 2499), ("Smart Watch", 5999), ("Running Shoes", 3299),
    ("Coffee Maker", 4499), ("Backpack", 1799), ("Bluetooth Speaker", 2999),
    ("Yoga Mat", 899), ("Desk Lamp", 1299), ("Mechanical Keyboard", 6499),
    ("Air Fryer", 7999), ("Phone Case", 499), ("Gaming Mouse", 2199),
    ("Office Chair", 12999), ("Water Bottle", 599), ("Laptop Stand", 1599),
]
FIRST_NAMES = ["Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Sneha", "Arjun", "Meera",
               "Kabir", "Isha", "Rahul", "Diya", "Aditya", "Neha", "Karan", "Pooja"]
LAST_NAMES = ["Sharma", "Patel", "Iyer", "Reddy", "Gupta", "Singh", "Nair", "Mehta", "Rao", "Das"]
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Hyderabad", "Chennai", "Kolkata", "Jaipur"]

# The Bitext messages contain blanks like {{Order Number}}. We fill them with real-looking values.
PLACEHOLDER_VALUES = {
    "Person Name": "Rahul",
    "Account Type": "premium",
    "Account Category": "premium",
    "Delivery City": "Pune",
    "Delivery Country": "India",
    "Invoice Number": "INV-2041",
    "Refund Amount": "₹1,499",
    "Currency Symbol": "₹",
    "Client First Name": "Priya",
    "Client Last Name": "Sharma",
}


def make_orders(n=300, seed=42):
    """Create n fake orders. Same seed -> same orders every time."""
    rng = random.Random(seed)
    statuses = ["Processing", "Shipped", "Delivered", "Delivered", "Delivered", "Delayed", "Cancelled"]
    rows = []
    for i in range(n):
        product, price = rng.choice(PRODUCTS)
        quantity = rng.choice([1, 1, 1, 2])
        order_date = config.TODAY - timedelta(days=rng.randint(1, 60))
        status = rng.choice(statuses)
        if status == "Delivered":
            eta = order_date + timedelta(days=rng.randint(2, 6))
        elif status == "Delayed":
            eta = config.TODAY + timedelta(days=rng.randint(3, 10))
        else:
            eta = config.TODAY + timedelta(days=rng.randint(1, 5))
        rows.append({
            "order_id": f"NM{10000 + i}",
            "customer_name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
            "city": rng.choice(CITIES),
            "product": product,
            "quantity": quantity,
            "amount_inr": price * quantity,
            "order_date": order_date.isoformat(),
            "status": status,
            "delivery_date": eta.isoformat(),
        })
    return pd.DataFrame(rows)


def fill_placeholders(text, order_ids, rng):
    """Replace {{Something}} blanks in a message with a realistic value."""
    def replace(match):
        name = match.group(1).strip()
        if name == "Order Number":
            return rng.choice(order_ids)
        return PLACEHOLDER_VALUES.get(name, name.lower())
    return re.sub(r"\{\{(.*?)\}\}", replace, text)


def make_messages(raw, order_ids, train_per_intent=40, test_per_intent=20, seed=42):
    """Pick a small, balanced sample of messages from the big Bitext table.

    raw needs the columns 'instruction' (the message) and 'intent' (the true answer).
    We keep the true intent so we can CHECK the LLM and the ML model later.
    """
    rng = random.Random(seed)
    parts = []
    for intent, group in raw.groupby("intent"):
        sample = group.sample(n=train_per_intent + test_per_intent, random_state=seed)
        sample = sample.assign(split=["train"] * train_per_intent + ["test"] * test_per_intent)
        parts.append(sample)
    df = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    df["text"] = [fill_placeholders(t, order_ids, rng) for t in df["instruction"]]
    df["id"] = range(len(df))
    df = df.rename(columns={"intent": "true_intent"})
    return df[["id", "text", "true_intent", "split"]]


def build_all():
    """Download the dataset and write orders.csv + messages.csv into data/."""
    from datasets import load_dataset  # imported here because it is slow to import

    config.DATA_DIR.mkdir(exist_ok=True)
    orders = make_orders()
    orders.to_csv(config.ORDERS_CSV, index=False)
    print(f"Wrote {len(orders)} orders to {config.ORDERS_CSV.name}")

    raw = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")["train"].to_pandas()
    messages = make_messages(raw, orders["order_id"].tolist())
    messages.to_csv(config.MESSAGES_CSV, index=False)
    print(f"Wrote {len(messages)} messages to {config.MESSAGES_CSV.name} "
          f"({(messages.split == 'train').sum()} train, {(messages.split == 'test').sum()} test)")


def load_messages():
    return pd.read_csv(config.MESSAGES_CSV)


def load_messy():
    return pd.read_csv(config.MESSY_CSV)


def load_orders():
    return pd.read_csv(config.ORDERS_CSV, dtype={"order_id": str})


def load_policy():
    return config.POLICY_MD.read_text()
