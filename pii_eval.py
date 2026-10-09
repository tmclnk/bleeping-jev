#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["typesafe-sdk", "numpy"]
# ///
"""Score the PII questions against pii_corpus.txt (live API, no mic).

Prints each utterance with misses (FN) and false alarms (FP) marked, then
per-word precision/recall. --variants FILE --variant NAME evaluates an
alternative pii_questions(words, text) -> (state, {i: Noul}) from FILE
instead of bleep.pii_questions."""

import argparse
import importlib.util
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import bleep
from typesafe_sdk import TypeSafeClient

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--corpus", default=str(Path(__file__).with_name("pii_corpus.txt")))
ap.add_argument("--split", choices=["dev", "test", "test2", "all"], default="dev")
ap.add_argument("--variants", help="python file defining variant functions")
ap.add_argument("--variant", help="function name in --variants")
ap.add_argument("--threshold", type=float, default=0.5)
ap.add_argument("--quiet", action="store_true", help="only print errors + score")
args = ap.parse_args()

if args.variants:
    spec = importlib.util.spec_from_file_location("variants", args.variants)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    build = getattr(mod, args.variant)
else:
    build = bleep.pii_questions


def load(path: str, split: str):
    cases, cur = [], None
    for line in Path(path).read_text().splitlines():
        if line.startswith("## "):
            cur = line[3:].strip()
            continue
        if not line.strip() or line.startswith("#"):
            continue
        if split != "all" and cur != split:
            continue
        words, labels = [], []
        for tok in line.split():
            words.append(tok.replace("[[", "").replace("]]", ""))
            labels.append("[[" in tok)
        cases.append((" ".join(words), words, labels))
    return cases


client = TypeSafeClient()


def run(case):
    text, words, labels = case
    state, questions = build(words, text)
    qs = {f"pii_{i}": q for i, q in questions.items()}
    t0 = time.perf_counter()
    result = client.system_one(state, qs, model="jev-latest")
    latency = time.perf_counter() - t0
    probs = [result.nouls[f"pii_{i}"].noul for i in range(len(words))]
    usage = result.raw_http_response.json().get("usage") or {}
    return text, words, labels, probs, latency, usage.get("input_tokens", 0)


cases = load(args.corpus, args.split)
with ThreadPoolExecutor(8) as pool:
    results = list(pool.map(run, cases))

tp = fp = fn = 0
for text, words, labels, probs, _, _ in results:
    marks, bad = [], False
    for w, gold, p in zip(words, labels, probs):
        pred = p >= args.threshold
        if pred and gold:
            tp += 1
            marks.append(f"{bleep.YELLOW}{w}{bleep.FG_RESET}")
        elif pred:
            fp += 1
            bad = True
            marks.append(f"{bleep.RED}{w}[FP {p:.2f}]{bleep.FG_RESET}")
        elif gold:
            fn += 1
            bad = True
            marks.append(f"{bleep.MAGENTA}{w}[FN {p:.2f}]{bleep.FG_RESET}")
        else:
            marks.append(w)
    if bad or not args.quiet:
        print(" ".join(marks))

prec = tp / (tp + fp) if tp + fp else 1.0
rec = tp / (tp + fn) if tp + fn else 1.0
f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
print(f"\n{args.variant or 'bleep'} [{args.split}]  TP {tp}  FP {fp}  FN {fn}  "
      f"precision {prec:.3f}  recall {rec:.3f}  F1 {f1:.3f}  "
      f"avg {sum(r[5] for r in results) / len(results):.0f} tok, "
      f"{sum(r[4] for r in results) / len(results) * 1000:.0f}ms")
