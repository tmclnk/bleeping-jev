#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["typesafe-sdk", "numpy"]
# ///
"""Text-only smoke test of the Jev judgment path — no microphone needed."""

import bleep
from typesafe_sdk import TypeSafeClient

client = TypeSafeClient()
features = {"language": True, "vulgarity": True, "pii": True, "shade": True}

cases = [
    "I have a fucking great feeling about this demo",
    "Das ist doch totale Scheiße, aber es funktioniert",
    "Esta mierda funciona muy bien, no puedo creerlo",
    "Call me at 555-867-5309 or email tom@example.com",
    "The weather in Boston is lovely today",
    "Honestly Dave from accounting is completely useless at his job",
    "My manager has no clue what she's doing, it's embarrassing",
    "Dave from accounting sent over the Q3 numbers this morning",
    "This vendor's API is garbage and their docs are worse",
]

for text in cases:
    language, vulgar, pii, shade, latency, tokens = bleep.judge_segment(
        client, text, features, "jev-latest")
    print(bleep.render(text, language, vulgar, pii, shade))
    print(bleep.status(latency, tokens, 0.0))
    print()
