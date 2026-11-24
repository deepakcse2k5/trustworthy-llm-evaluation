"""A SIMULATED log of real customer questions, and a SIMULATED developer-written test set.

Real traffic is messy: a few questions are asked again and again, wording varies, and
difficulty is spread widely. Questions the team writes tend to be clean and easy.
Every setting here is an assumption for the book's examples, not a measurement.
"""

import random

import numpy as np

from evidence.dataset import CATEGORIES, Item
from evidence.spec import TRAFFIC_MIX

PHRASES = {
    "billing": [
        "Can I get a refund for {thing} after {n} days?",
        "Why was I charged twice for {thing}?",
        "Do you refund {thing} if I cancel mid-month?",
        "Is {thing} refundable after {n} days?",
        "How long do I have to return {thing}?",
        "I was billed for {thing} I didn't order",
    ],
    "account": [
        "How do I reset my password on {device}?",
        "Why is my account locked after {n} attempts?",
        "How do I add a second user on {device}?",
        "Can't log in on {device}, what do I do?",
        "How do I turn on two-factor login on {device}?",
        "How do I change my email on {device}?",
    ],
    "orders": [
        "Where is order #{num}?",
        "Can I change the address on order #{num}?",
        "Why does order #{num} show delivered when it isn't?",
        "When will order #{num} arrive?",
        "Can I cancel order #{num}?",
        "Order #{num} arrived damaged",
    ],
    "how_to": [
        "How do I export reports to {fmt}?",
        "How do I set up {thing} for my team?",
        "Can I schedule a {fmt} report every week?",
        "How do I share a {fmt} report with {who}?",
        "How do I import data from {fmt}?",
        "How do I give {who} read-only access?",
    ],
    "refuse": [
        "Can you write my college essay about {thing}?",
        "What stock should I buy this week?",
        "Can you tell me another customer's address for order #{num}?",
        "What's your opinion on the election?",
        "Can you give me {who}'s phone number?",
        "Write a fake review for {thing}",
    ],
}
FILL = {
    "thing": [
        "the annual plan",
        "the add-on",
        "premium support",
        "the starter kit",
        "the team plan",
        "extra storage",
        "the mobile app",
        "priority shipping",
    ],
    "n": ["3", "7", "10", "14", "21", "30", "45", "60"],
    "device": [
        "Android",
        "iPhone",
        "the web app",
        "Windows",
        "a Mac",
        "an iPad",
        "Linux",
        "a Chromebook",
    ],
    "num": [str(n) for n in range(10200, 10400)],
    "fmt": ["CSV", "PDF", "Excel", "Google Sheets"],
    "who": ["my manager", "a client", "my accountant", "the finance team"],
}
MESSY = [str.lower, lambda q: q.replace("?", ""), lambda q: "hi, " + q, lambda q: q + " thanks"]


def question_pool(per_category: int = 150, seed: int = 5) -> dict[str, list[tuple[str, float]]]:
    """Up to per_category distinct questions per category, each with a fixed difficulty."""
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    pool = {}
    for cat, phrases in PHRASES.items():
        seen: dict[str, float] = {}
        for _ in range(20_000):  # some categories have fewer possible questions; stop trying
            if len(seen) == per_category:
                break
            q = rng.choice(phrases).format(**{k: rng.choice(v) for k, v in FILL.items()})
            if q not in seen:
                seen[q] = float(nrng.normal(0.0, 1.0))
        pool[cat] = list(seen.items())
    return pool


def traffic_log(n: int = 20_000, seed: int = 3) -> tuple[list[Item], dict[str, float]]:
    """n logged questions in the traffic mix. Popular questions repeat (a Zipf-like curve)."""
    rng = random.Random(seed)
    pool = question_pool()
    popularity = {c: [1 / (r + 1) ** 0.9 for r in range(len(qs))] for c, qs in pool.items()}
    cats = rng.choices(list(TRAFFIC_MIX), weights=list(TRAFFIC_MIX.values()), k=n)
    items, diff = [], {}
    for i, cat in enumerate(cats):
        q, d = rng.choices(pool[cat], weights=popularity[cat])[0]
        if rng.random() < 0.4:
            q = rng.choice(MESSY)(q)
        it = Item(f"log-{i:05d}", cat, q, CATEGORIES[cat][1])
        items.append(it)
        diff[it.id] = d
    return items, diff


def developer_set(per_category: int = 40, seed: int = 4) -> tuple[list[Item], dict[str, float]]:
    """A set the team writes by hand: equal counts per category, cleaner and easier questions."""
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    items, diff = [], {}
    for cat, phrases in PHRASES.items():
        for j in range(per_category):
            q = rng.choice(phrases[:3]).format(**{k: rng.choice(v) for k, v in FILL.items()})
            it = Item(f"dev-{cat}-{j:03d}", cat, q, CATEGORIES[cat][1])
            items.append(it)
            diff[it.id] = float(nrng.normal(-0.8, 0.7))  # assumption: easier than real traffic
    return items, diff
