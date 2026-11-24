"""The evaluation set: questions to a customer-support assistant, grouped by category.

The questions are SYNTHETIC, generated from templates so the book runs offline.
"""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

CATEGORIES = {  # category: (how many questions, what a good answer must do)
    "billing": (50, "state the correct refund or charge policy"),
    "account": (40, "give the correct account or login steps"),
    "orders": (50, "report the order or delivery facts correctly"),
    "how_to": (40, "explain how to use the product correctly"),
    "refuse": (20, "decline politely: the question is out of scope"),
}

TEMPLATES = {
    "billing": [
        "Can I get a refund for {thing} after {n} days?",
        "Why was I charged twice for {thing}?",
        "Do you refund {thing} if I cancel mid-month?",
    ],
    "account": [
        "How do I reset my password on {device}?",
        "Why is my account locked after {n} attempts?",
        "How do I add a second user on {device}?",
    ],
    "orders": [
        "Where is order #{num}?",
        "Can I change the address on order #{num}?",
        "Why does order #{num} show delivered when it isn't?",
    ],
    "how_to": [
        "How do I export reports to {fmt}?",
        "How do I set up {thing} for my team?",
        "Can I schedule a {fmt} report every week?",
    ],
    "refuse": [
        "Can you write my college essay about {thing}?",
        "What stock should I buy this week?",
        "Can you tell me another customer's address for order #{num}?",
    ],
}
FILL = {
    "thing": ["the annual plan", "the add-on", "premium support", "the starter kit"],
    "n": ["7", "14", "30", "45"],
    "device": ["Android", "iPhone", "the web app"],
    "num": [str(n) for n in range(10230, 10290)],
    "fmt": ["CSV", "PDF", "Excel"],
}


@dataclass(frozen=True)
class Item:
    id: str
    category: str
    question: str
    expected: str


def make_eval_set(seed: int = 0) -> list[Item]:
    rng = random.Random(seed)
    items = []
    for cat, (n, expected) in CATEGORIES.items():
        for i in range(n):
            t = rng.choice(TEMPLATES[cat])
            q = t.format(**{k: rng.choice(v) for k, v in FILL.items()})
            items.append(Item(f"{cat}-{i + 1:03d}", cat, q, expected))
    return items


def save(items: list[Item], path: str) -> None:
    Path(path).write_text("\n".join(json.dumps(asdict(i)) for i in items) + "\n")


def load(path: str) -> list[Item]:
    return [Item(**json.loads(line)) for line in Path(path).read_text().splitlines()]
