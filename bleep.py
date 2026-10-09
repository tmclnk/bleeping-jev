#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "typesafe-sdk",
#     "pywhispercpp>=1.3",
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
                     p = toggle PII redaction   s = toggle shade
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

    def utterances(self, partial_ms: int | None = None):
        """Yields (audio, final). final=True once the speaker pauses. With
        partial_ms, also yields the utterance-so-far (final=False) every
        partial_ms of new audio, but only once the consumer has caught up
        with the mic, so a slow consumer skips stale partials."""
        import sounddevice as sd

        partial_blocks = partial_ms // BLOCK_MS if partial_ms else 0
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            blocksize=BLOCK_SAMPLES, dtype="float32",
                            device=self.device, callback=self._callback):
            current: list[np.ndarray] = []
            quiet_run = 0
            last_partial = 0
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
                        current, quiet_run, last_partial = [], 0, 0
                        # ignore blips shorter than ~300ms of speech
                        if len(audio) > SAMPLE_RATE * 0.3:
                            yield audio, True
                        continue

                if (partial_blocks and self.blocks.empty()
                        and len(current) - last_partial >= partial_blocks
                        and len(current) * BLOCK_MS >= 500):
                    last_partial = len(current)
                    yield np.concatenate(current), False


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
                "In `segment`, is the speaker talking shit about another "
                "person — insulting them, mocking them, complaining about "
                "them, or gossiping about them behind their back?"
            ),
            criteria=NoulCriteria(
                true=("Disparages, mocks, or vents about a specific person "
                      "(anyone, not just a coworker), e.g. 'Dave is useless', "
                      "'my sister has no idea what she's doing'"),
                false=("No person is mentioned, or a person is mentioned "
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


def status(latency: float, tokens: int, whisper_s: float,
           partials: int = 0, partial_tokens: int = 0) -> str:
    cost = tokens / 1_000_000 * JEV_PRICE_PER_MTOK
    line = (f"{DIM}  jev {latency*1000:.0f}ms · {tokens} tok · "
            f"${cost:.6f} · whisper {whisper_s:.1f}s")
    if partials:  # streaming: be honest about what the live preview cost
        total = (tokens + partial_tokens) / 1_000_000 * JEV_PRICE_PER_MTOK
        line += (f" · +{partials} live ({partial_tokens} tok) "
                 f"= ${total:.6f}")
    return line + RESET


ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def screen_rows(line: str) -> int:
    """Terminal rows a printed line occupies, so a live line can be redrawn
    in place even after it wraps."""
    import shutil
    import unicodedata
    cols = shutil.get_terminal_size().columns
    width = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1
                for c in ANSI_RE.sub("", line))
    return max(1, -(-width // cols))


def clear_rows(rows: int) -> None:
    """Erase the `rows` rows ending at the cursor and return to their start."""
    if rows:
        up = f"\033[{rows - 1}A" if rows > 1 else ""
        sys.stdout.write(up + "\r\033[J")


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


# Bare profanity list: enough style cue to stop Whisper writing `f***`, with
# little sentence structure for it to echo back on silence/noise.
WHISPER_PROMPT = "damn, shit, fuck."
_PROMPT_WORDS = set(re.findall(r"[a-z']+", WHISPER_PROMPT.lower()))


def is_prompt_echo(text: str) -> bool:
    """True if Whisper just regurgitated the initial prompt (hallucination)."""
    words = re.findall(r"[a-z']+", text.lower())
    return len(words) >= 3 and all(w in _PROMPT_WORDS for w in words)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-language", action="store_true")
    ap.add_argument("--no-vulgarity", action="store_true")
    ap.add_argument("--no-pii", action="store_true")
    ap.add_argument("--no-shade", action="store_true",
                    help="don't elide utterances that trash-talk a person")
    ap.add_argument("--jev-model", default="jev-latest")
    ap.add_argument("--whisper-model", default="small",
                    help="whisper.cpp model (small handles de/es well)")
    ap.add_argument("--device", type=int, default=None,
                    help="input device index (see --list-devices)")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--gate-db", type=float, default=-40.0,
                    help="energy gate in dBFS; raise if room noise triggers it")
    ap.add_argument("--silence-ms", type=int, default=700,
                    help="pause length that ends an utterance")
    ap.add_argument("--stream", action="store_true",
                    help="show a live, already-redacted line while speaking "
                         "(one extra Jev request per update)")
    ap.add_argument("--partial-ms", type=int, default=1000,
                    help="--stream: new audio between live updates")
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
    from pywhispercpp.model import Model
    whisper = Model(
        args.whisper_model, n_threads=4, print_realtime=False,
        print_progress=False,
        # bias whisper away from self-censoring transcripts
        initial_prompt=WHISPER_PROMPT)

    stop = threading.Event()
    threading.Thread(target=key_listener, args=(features, stop),
                     daemon=True).start()

    on = [k for k, v in features.items() if v]
    print(f"{BOLD}bleep{RESET} {DIM}listening · features: {', '.join(on)}"
          f" · keys: l/v/p/s toggle, q quit{RESET}\n")

    show_request = args.debug or args.show_request
    show_response = args.debug or args.show_response
    # HTTP dumps would land under the live line and break the in-place
    # redraw, so with them on each live update gets its own line instead.
    redraw = not (show_request or show_response)

    capture = UtteranceCapture(args.device, args.gate_db, args.silence_ms,
                               max_utterance_s=12)
    live_rows = 0  # rows taken by the live line currently on screen
    partials = partial_tokens = 0
    for audio, final in capture.utterances(
            args.partial_ms if args.stream else None):
        if stop.is_set():
            break
        t0 = time.perf_counter()
        text = " ".join(s.text.strip() for s in whisper.transcribe(audio)).strip()
        whisper_s = time.perf_counter() - t0
        if not text or is_prompt_echo(text):
            if final:
                clear_rows(live_rows)
                sys.stdout.flush()
                live_rows = partials = partial_tokens = 0
            continue

        snapshot = dict(features)  # freeze toggles for this utterance
        try:
            language, vulgar, pii, shade, latency, tokens = judge_segment(
                client, text, snapshot, args.jev_model,
                show_request=show_request, show_response=show_response)
        except Exception as e:
            if final:
                clear_rows(live_rows)
                print(f"{RED}[jev error]{RESET} {e}\n{text}")
                live_rows = partials = partial_tokens = 0
            continue  # a failed live update just keeps the last one up

        line = render(text, language, vulgar, pii, shade)
        if redraw:
            clear_rows(live_rows)
        if final:
            print(line)
            print(status(latency, tokens, whisper_s,
                         partials, partial_tokens) + "\n")
            live_rows = partials = partial_tokens = 0
        else:
            # Never shows raw text: a live line is judged before it's drawn.
            live = f"{line} {DIM}…{RESET}"
            sys.stdout.write(live if redraw else live + "\n")
            sys.stdout.flush()
            live_rows = screen_rows(live) if redraw else 0
            partials += 1
            partial_tokens += tokens

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(0)
