#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["typesafe-sdk", "numpy"]
# ///
"""Text-only smoke test of the Jev judgment path — no microphone needed."""

import argparse

import bleep
from typesafe_sdk import TypeSafeClient

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--show-request", action="store_true",
                help="print each outgoing Jev request body")
ap.add_argument("--show-response", action="store_true",
                help="print each incoming Jev response body")
ap.add_argument("--debug", action="store_true",
                help="shorthand for --show-request --show-response")
args = ap.parse_args()

client = TypeSafeClient()
features = {"language": True, "vulgarity": True, "pii": True, "shade": True}

cases = [
    "I have a fucking great feeling about this demo",
    "Das ist doch totale Scheiße, aber es funktioniert",
    "Esta mierda funciona muy bien, no puedo creerlo",
    "Call me at 555-867-5309 or email tom@example.com",
    "The weather in Boston is lovely today",
    "Son of a bitch, I stubbed my toe on the table again",
    "Honestly Dave from accounting is completely useless at his job",
    "My manager has no clue what she's doing, it's embarrassing",
    "My brother-in-law is a total idiot and everyone in the family knows it",
    "Dave from accounting sent over the Q3 numbers this morning",
    "This vendor's API is garbage and their docs are worse",
]

for text in cases:
    language, vulgar, pii, shade, latency, tokens = bleep.judge_segment(
        client, text, features, "jev-latest",
        show_request=args.debug or args.show_request,
        show_response=args.debug or args.show_response)
    print(bleep.render(text, language, vulgar, pii, shade))
    print(bleep.status(latency, tokens, 0.0))
    print()
