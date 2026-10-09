# bleep

A live broadcast-delay demo powered by [Jev](https://typesafe.ai), TypeSafe's
System One model.

Broadcasters run a 7-second delay so someone can hit the "dump" button before
a curse word reaches air. This demo replaces the button with a model: you talk
into a microphone, and each sentence prints to the screen with its language
tagged, vulgarities bleeped, personal information redacted, and any sentence
where you're talking shit about a coworker elided entirely — judged by Jev in
roughly 150 milliseconds for about five thousandths of a cent.

```
<english>I have a @#$%&!† great feeling about this demo</english>
  jev 212ms · 1248 tok · $0.000052 · whisper 1.9s

<german>Das ist doch totale ☠§%&*!?¤ aber es funktioniert</german>
  jev 107ms · 1160 tok · $0.000049 · whisper 2.1s

<english>Call me at ▒▒▒▒▒▒▒▒▒▒▒▒ or email ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒</english>
  jev 136ms · 1070 tok · $0.000045 · whisper 1.7s

<english>▓▓▓▓▓▓▓▓ ▓▓▓▓ ▓▓▓▓ ▓▓▓▓▓▓▓▓▓▓ ▓▓ ▓▓▓▓▓▓▓▓▓▓ ▓▓▓▓▓▓▓ ▓▓ ▓▓▓ ▓▓▓</english>
  jev 118ms · 1495 tok · $0.000063 · whisper 2.0s
```

Red `@#$%&!` is a vulgarity replaced with random cartoon-swear glyphs; yellow `▒▒▒▒` is redacted PII; a whole line
of magenta `▓▓▓▓` means you were talking shit about a coworker. The status
line under each sentence is the pitch: Jev's wall-clock latency, the tokens
sent, and what they cost.

## Quick start

```sh
make setup   # check uv + API key, resolve deps, pre-download the Whisper model
make test    # smoke-test the Jev path with canned sentences — no mic needed
make run     # live demo
```

`make` alone lists every target. Requirements: macOS, [uv](https://docs.astral.sh/uv/),
a `TYPESAFE_API_KEY` from [console.typesafe.ai/keys](https://console.typesafe.ai/keys)
(the repo's `.envrc` exports it for direnv). Python dependencies are declared
inline in each script (PEP 723), so `uv run bleep.py` works without a venv.

## Controls

| Key | Action |
| --- | --- |
| `l` | toggle language tags |
| `v` | toggle vulgarity bleeping |
| `p` | toggle PII redaction |
| `s` | toggle coworker-shade elision |
| `q` | quit |

Toggling live is a good demo beat: say the same sentence with `v` off and on.

Flags for `bleep.py` (or `make run WHISPER_MODEL=base DEVICE=4`):

| Flag | Default | Purpose |
| --- | --- | --- |
| `--no-language` / `--no-vulgarity` / `--no-pii` / `--no-shade` | all on | start with a feature disabled |
| `--whisper-model` | `small` | faster-whisper size; `base` is faster, `small` handles de/es better |
| `--device N` | system default | input device index; `--list-devices` shows them |
| `--gate-db` | `-40` | energy gate in dBFS; raise toward `-30` in a noisy room |
| `--silence-ms` | `700` | pause length that ends an utterance |
| `--jev-model` | `jev-latest` | pin e.g. `jev-1.13.0` if you tune thresholds |

## How it works

```
mic ──▶ energy gate ──▶ Whisper (local) ──▶ Jev (1 request) ──▶ render ──▶ stdout
        30ms blocks     faster-whisper      ~36 questions       glyphs / ▒ / ▓ / <lang>
        pause = cut     small, int8         in parallel
```

1. **Capture.** `sounddevice` streams 16 kHz mono audio in 30 ms blocks. A
   simple RMS gate marks speech; ~700 ms of quiet ends the utterance.
2. **Transcribe.** `faster-whisper` runs the multilingual `small` model on CPU.
   An `initial_prompt` containing mild profanity nudges Whisper away from its
   habit of self-censoring to `f***`.
3. **Judge.** The utterance becomes one Jev `system_one` request. State is
   `{"segment": text, "words": [...]}` and the questions are:
   - `language` — a **Choice** among `english`, `spanish`, `german`, `other`.
   - `vulgar_{i}` — one **Noul** per word: *"Is `words[i]`, as used in
     `segment`, a profanity a broadcaster would bleep?"*
   - `pii_{i}` — one **Noul** per word: *"Does `words[i]` reveal a personal
     name, phone, address, email, or ID number?"*
   - `shade` — one **Noul** for the whole segment: *"Is the speaker talking
     shit about a coworker, colleague, boss, or teammate?"* with `criteria`
     spelling out the boundary: venting about a specific person you work with
     is a yes; a neutral mention, praise, or a complaint about a product or
     company is a no.

   All questions run in parallel inside a single request (TypeSafe's
   [speculative fan-out](https://docs.typesafe.ai/patterns/fan-out.md)
   pattern). A 15-word sentence is ~31 questions and ~1,100 tokens.
4. **Render.** Code — not the model — does the redaction: any word whose Noul
   probability is ≥ 0.5 is replaced with block characters of the same length,
   and the line is wrapped in the language tag. If `shade` is ≥ 0.5 the
   entire utterance is elided word-by-word (lengths preserved, naturally). Jev never generates text; it
   returns typed answers that the program consumes directly.

### Why Jev fits here

- **Latency.** Judgments return in 100–200 ms, well inside a broadcast delay.
- **Cost.** Pricing is $0.042 per million input tokens with free output, so an
  utterance costs ~$0.00005. An hour of continuous talk is a few cents.
- **Typed answers.** Probabilities and choices, not prose to parse. The
  threshold, the redaction style, and the feature toggles all live in code
  and can change without touching the model.
- **Cross-lingual.** The same per-word question catches `Scheiße` and
  `mierda` without a word list. TypeSafe notes Jev is strongest in English
  and handles other languages less reliably; German and Spanish both worked
  cleanly in testing.

## Files

| File | Purpose |
| --- | --- |
| `bleep.py` | the demo: capture, transcribe, judge, render |
| `smoke_test.py` | runs `judge_segment` on nine canned sentences (en/de/es, vulgar, PII, clean, coworker shade + near-misses) |
| `Makefile` | self-documenting setup/test/run targets |
| `.envrc` | exports `TYPESAFE_API_KEY` (gitignored `.env` also supported by direnv) |

## Demo-day checklist

- Run `make setup` and then `make run` once from the terminal you'll present
  with. macOS prompts for microphone permission on the first live run.
- Fullscreen the terminal and raise the font (`Cmd +`) until one sentence
  fills the projector.
- Whisper on CPU adds 1–3 s per utterance; Jev is the fast part. Pause about
  a second between sentences so the gate cuts cleanly.
- One language per sentence: Whisper detects language per chunk, and Jev tags
  the whole segment.
- Rehearse the exact non-English lines once. Whisper occasionally mishears
  short foreign phrases, and a mangled transcript gives Jev nothing to judge.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Nothing prints while talking | `make devices`, pass `--device N`; or lower `--gate-db` to `-45` |
| Captures constantly in a loud room | raise `--gate-db` to `-30` |
| `✗ TYPESAFE_API_KEY not set` | `direnv allow`, or `export TYPESAFE_API_KEY=...` |
| Whisper outputs `f***` | the `initial_prompt` usually prevents this; try `--whisper-model small` if on `base` |
| Long lag after each sentence | `make run-fast` (Whisper `base`) |
| Jev error in output | check the key and network; the raw transcript still prints so the demo continues |
