#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "typesafe-sdk",
#     "faster-whisper>=1.0",
#     "sounddevice>=0.4",
#     "numpy",
# ]
# ///
"""bleep — live broadcast-delay demo powered by Jev (typesafe.ai).

Listens to the microphone, transcribes speech locally with Whisper, then asks
Jev one batched System One request per utterance:

  * language   — Choice: which language is the segment in?
  * vulgarity  — one Noul per word: is this word a profanity here?
  * pii        — one Noul per word: does this word reveal PII here?
  * shade      — one Noul for the whole utterance: is the speaker talking
                 shit about a coworker? If so the entire line is elided.

Flagged words are redacted with block characters before the text ever hits
the screen. Each line prints with Jev's wall-clock latency and token cost,
because that's the point of the demo.

Keys while running:  l = toggle language tags   v = toggle vulgarity bleep
                     p = toggle PII redaction   s = toggle coworker shade
                     q = quit
"""

from __future__ import annotations

import argparse
import json
import queue
import random
import re
import sys
import termios
import threading
import time
import tty

import numpy as np

SAMPLE_RATE = 16_000
BLOCK_MS = 30
BLOCK_SAMPLES = SAMPLE_RATE * BLOCK_MS // 1000

JEV_PRICE_PER_MTOK = 0.042  # USD, input tokens only (docs.typesafe.ai/models)

DIM = "\033[2m"
BOLD = "\033[1m"
RED = "\033[31m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

VULGAR_GLYPHS = "@#$%&*!?¤§¶†‡∞≠☠☹✖⚡"
PII_BLOCK = "▒"
SHADE_BLOCK = "▓"
SHADE_THRESHOLD = 0.5


# ---------------------------------------------------------------- audio ----

class UtteranceCapture:
    """Microphone capture with a simple energy gate: yields utterances of
    float32 PCM once the speaker pauses."""

    def __init__(self, device: int | None, gate_db: float, silence_ms: int,
                 max_utterance_s: float):
        self.device = device
        self.gate_db = gate_db
        self.silence_blocks = max(1, silence_ms // BLOCK_MS)
        self.max_blocks = int(max_utterance_s * 1000 / BLOCK_MS)
        self.blocks: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"{DIM}[audio] {status}{RESET}", file=sys.stderr)
        self.blocks.put(indata[:, 0].copy())

    def utterances(self):
        import sounddevice as sd

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            blocksize=BLOCK_SAMPLES, dtype="float32",
                            device=self.device, callback=self._callback):
            current: list[np.ndarray] = []
            quiet_run = 0
            while True:
                block = self.blocks.get()
                rms = float(np.sqrt(np.mean(block**2)))
                loud_db = 20 * np.log10(max(rms, 1e-10))
                speaking = loud_db > self.gate_db

                if speaking:
                    current.append(block)
                    quiet_run = 0
                elif current:
                    current.append(block)
                    quiet_run += 1
                    ended = quiet_run >= self.silence_blocks
                    if ended or len(current) >= self.max_blocks:
                        audio = np.concatenate(current)
                        current, quiet_run = [], 0
                        # ignore blips shorter than ~300ms of speech
                        if len(audio) > SAMPLE_RATE * 0.3:
                            yield audio


# ------------------------------------------------------------------ jev ----

WORD_RE = re.compile(r"\S+")


def dump_http(label: str, body: bytes | str, head: str) -> None:
    """Print one HTTP body for --show-request/--show-response; objects one
    entry per line so ~70-question requests stay readable."""
    print(f"{CYAN}{label} {head}{RESET}")
    try:
        obj = json.loads(body)
    except ValueError:
        print(f"{CYAN}{body!r}{RESET}")
        return
    for k, v in obj.items():
        if isinstance(v, dict):
            print(f"{CYAN}  {k}:{RESET}")
            for kk, vv in v.items():
                print(f"{DIM}    {kk}: {json.dumps(vv, ensure_ascii=False)}{RESET}")
        else:
            print(f"{DIM}  {k}: {json.dumps(v, ensure_ascii=False)}{RESET}")


def judge_segment(client, text: str, features: dict[str, bool], model: str,
                  show_request: bool = False, show_response: bool = False):
    """One System One request: language Choice + per-word vulgarity/PII Nouls
    + one segment-level coworker-shade Noul.
    Returns (language, vulgar_flags, pii_flags, shade, latency_s, tokens)."""
    from typesafe_sdk import Choice, Noul, NoulCriteria

    words = WORD_RE.findall(text)
    state = json.dumps({"segment": text, "words": words})

    questions: dict = {}
    if features["language"]:
        questions["language"] = Choice(
            instructions="Which language is `segment` spoken in?",
            criteria={
                "english": None,
                "spanish": None,
                "german": None,
                "other": "Any language other than English, Spanish, or German",
            },
        )
    if features["vulgarity"]:
        for i in range(len(words)):
            questions[f"vulgar_{i}"] = Noul(
                instructions=(
                    f"Is `words[{i}]`, as used in `segment`, a profanity, "
                    "vulgarity, or curse word that a broadcaster would bleep? "
                    "Judge the word in whatever language it is spoken."
                ),
            )
    if features["pii"]:
        for i in range(len(words)):
            questions[f"pii_{i}"] = Noul(
                instructions=(
                    f"Does `words[{i}]`, as used in `segment`, reveal "
                    "personally identifiable information about a specific "
                    "person — part of a personal name, phone number, street "
                    "address, email address, or ID/account number?"
                ),
            )
    if features["shade"]:
        questions["shade"] = Noul(
            instructions=(
                "In `segment`, is the speaker talking shit about a coworker, "
                "colleague, boss, teammate, or someone they work with — "
                "insulting them, mocking them, complaining about them, or "
                "gossiping about them behind their back?"
            ),
            criteria=NoulCriteria(
                true=("Disparages, mocks, or vents about a specific person "
                      "the speaker works with, e.g. 'Dave in accounting is "
                      "useless', 'my manager has no idea what she's doing'"),
                false=("No coworker is mentioned, or a coworker is mentioned "
                       "neutrally or positively, or the complaint is about a "
                       "product, company, or situation rather than a person"),
            ),
        )

    if not questions:
        return None, [False] * len(words), [False] * len(words), False, 0.0, 0

    t0 = time.perf_counter()
    result = client.system_one(state, questions, model=model)
    latency = time.perf_counter() - t0

    if show_request or show_response:
        http = result.raw_http_response
        if show_request:
            req = http.request
            dump_http("→ request", req.content, f"{req.method} {req.url}")
        if show_response:
            dump_http("← response", http.text, f"{http.status_code}")

    language = None
    if features["language"]:
        language = result.choices["language"].choice

    def flags(prefix: str, enabled: bool, threshold: float = 0.5):
        if not enabled:
            return [False] * len(words)
        return [result.nouls[f"{prefix}_{i}"].noul >= threshold
                for i in range(len(words))]

    vulgar = flags("vulgar", features["vulgarity"])
    pii = flags("pii", features["pii"])
    shade = (features["shade"]
             and result.nouls["shade"].noul >= SHADE_THRESHOLD)

    tokens = 0
    try:
        usage = result.raw_http_response.json().get("usage") or {}
        tokens = usage.get("input_tokens") or usage.get("total_tokens") or 0
    except Exception:
        pass
    if not tokens:  # rough fallback: ~4 chars/token over state + questions
        tokens = (len(state) + sum(len(str(q.instructions))
                                   for q in questions.values())) // 4

    return language, vulgar, pii, shade, latency, tokens


# ------------------------------------------------------------- rendering ----

def redact(word: str, block: str) -> str:
    return block * len(word)


def scramble(word: str) -> str:
    """Cartoon-swearing: random glyphs, same length as the word."""
    return "".join(random.choice(VULGAR_GLYPHS) for _ in word)


def render(text: str, language: str | None, vulgar: list[bool],
           pii: list[bool], shade: bool = False) -> str:
    out: list[str] = []
    for i, m in enumerate(WORD_RE.finditer(text)):
        w = m.group()
        if shade:  # whole utterance elided: nothing survives, lengths do
            out.append(MAGENTA + redact(w, SHADE_BLOCK) + RESET)
        elif vulgar[i]:
            out.append(RED + scramble(w) + RESET)
        elif pii[i]:
            out.append(YELLOW + redact(w, PII_BLOCK) + RESET)
        else:
            out.append(w)
    line = " ".join(out)
    if language:
        line = f"<{language}>{line}</{language}>"
    return BOLD + line + RESET


def status(latency: float, tokens: int, whisper_s: float) -> str:
    cost = tokens / 1_000_000 * JEV_PRICE_PER_MTOK
    return (f"{DIM}  jev {latency*1000:.0f}ms · {tokens} tok · "
            f"${cost:.6f} · whisper {whisper_s:.1f}s{RESET}")


# ------------------------------------------------------------------ main ----

def key_listener(features: dict[str, bool], stop: threading.Event):
    """Raw-mode stdin reader so features can be toggled mid-demo."""
    try:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
    except (termios.error, ValueError, OSError):
        return  # no tty (piped/background run): toggles unavailable
    names = {"l": "language", "v": "vulgarity", "p": "pii", "s": "shade"}
    try:
        tty.setcbreak(fd)
        while not stop.is_set():
            ch = sys.stdin.read(1).lower()
            if ch == "q":
                stop.set()
            elif ch in names:
                features[names[ch]] = not features[names[ch]]
                on = [k for k, v in features.items() if v]
                print(f"{DIM}[features] {', '.join(on) or 'none'}{RESET}")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-language", action="store_true")
    ap.add_argument("--no-vulgarity", action="store_true")
    ap.add_argument("--no-pii", action="store_true")
    ap.add_argument("--no-shade", action="store_true",
                    help="don't elide utterances that trash-talk a coworker")
    ap.add_argument("--jev-model", default="jev-latest")
    ap.add_argument("--whisper-model", default="small",
                    help="faster-whisper model (small handles de/es well)")
    ap.add_argument("--device", type=int, default=None,
                    help="input device index (see --list-devices)")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--gate-db", type=float, default=-40.0,
                    help="energy gate in dBFS; raise if room noise triggers it")
    ap.add_argument("--silence-ms", type=int, default=700,
                    help="pause length that ends an utterance")
    ap.add_argument("--show-request", action="store_true",
                    help="print each outgoing Jev request body")
    ap.add_argument("--show-response", action="store_true",
                    help="print each incoming Jev response body")
    ap.add_argument("--debug", action="store_true",
                    help="shorthand for --show-request --show-response")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    features = {
        "language": not args.no_language,
        "vulgarity": not args.no_vulgarity,
        "pii": not args.no_pii,
        "shade": not args.no_shade,
    }

    from typesafe_sdk import TypeSafeClient
    client = TypeSafeClient()

    print(f"{DIM}loading whisper '{args.whisper_model}'…{RESET}")
    from faster_whisper import WhisperModel
    whisper = WhisperModel(args.whisper_model, compute_type="int8")

    stop = threading.Event()
    threading.Thread(target=key_listener, args=(features, stop),
                     daemon=True).start()

    on = [k for k, v in features.items() if v]
    print(f"{BOLD}bleep{RESET} {DIM}listening · features: {', '.join(on)}"
          f" · keys: l/v/p/s toggle, q quit{RESET}\n")

    capture = UtteranceCapture(args.device, args.gate_db, args.silence_ms,
                               max_utterance_s=12)
    for audio in capture.utterances():
        if stop.is_set():
            break
        t0 = time.perf_counter()
        segments, _info = whisper.transcribe(
            audio, beam_size=1, vad_filter=True,
            # bias whisper away from self-censoring transcripts
            initial_prompt="Okay, so, damn, I swear, shit happens, fuck it.",
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        whisper_s = time.perf_counter() - t0
        if not text:
            continue

        snapshot = dict(features)  # freeze toggles for this utterance
        try:
            language, vulgar, pii, shade, latency, tokens = judge_segment(
                client, text, snapshot, args.jev_model,
                show_request=args.debug or args.show_request,
                show_response=args.debug or args.show_response)
        except Exception as e:
            print(f"{RED}[jev error]{RESET} {e}\n{text}")
            continue

        print(render(text, language, vulgar, pii, shade))
        print(status(latency, tokens, whisper_s) + "\n")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(0)
