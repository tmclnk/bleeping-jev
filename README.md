# bleep

A live broadcast-delay demo powered by [Jev](https://typesafe.ai), TypeSafe's
System One model.

Broadcasters run a 7-second delay so someone can hit the "dump" button before
something regrettable reaches air. This demo replaces the button with a model:
you talk into a microphone, and each sentence prints to the screen with its
language tagged, offensive words swapped for 🙊, personal information
redacted, and any sentence
where you're talking shit about someone elided entirely — judged by Jev in
roughly 150 milliseconds for about five thousandths of a cent.

```
<english>I have a 🙊 great feeling about this demo</english>
  jev 212ms · 1248 tok · $0.000052 · whisper 1.9s

<german>Das ist doch totale 🙊 aber es funktioniert</german>
  jev 107ms · 1160 tok · $0.000049 · whisper 2.1s

<english>Call me at ▒▒▒▒▒▒▒▒▒▒▒▒ or email ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒</english>
  jev 136ms · 1070 tok · $0.000045 · whisper 1.7s

<english>▓▓▓▓▓▓▓▓ ▓▓▓▓ ▓▓▓▓ ▓▓▓▓▓▓▓▓▓▓ ▓▓ ▓▓▓▓▓▓▓▓▓▓ ▓▓▓▓▓▓▓ ▓▓ ▓▓▓ ▓▓▓</english>
  jev 118ms · 1495 tok · $0.000063 · whisper 2.0s
```

🙊 replaces each word of an offensive word or phrase (*son of a bitch* is
`🙊 🙊 🙊 🙊`); yellow `▒▒▒▒` is redacted PII; a whole line
of magenta `▓▓▓▓` means you were talking shit about someone. The status
line under each sentence is the pitch: Jev's wall-clock latency, the tokens
sent, and what they cost.

## Quick start

```sh
mise run setup   # check uv + API key, resolve deps, pre-download the Whisper model
mise run test    # smoke-test the Jev path with canned sentences — no mic needed
mise run start     # live demo (add --debug to dump Jev requests/responses)
```

`mise tasks` lists every task, `mise run <task> --help` its flags. Requirements: macOS, [mise](https://mise.jdx.dev/) (installs `uv` per `mise.toml`),
a `TYPESAFE_API_KEY` from [console.typesafe.ai/keys](https://console.typesafe.ai/keys)
(the repo's `.envrc` exports it for direnv). Python dependencies are declared
inline in each script (PEP 723), so `uv run bleep.py` works without a venv.

## Controls

| Key | Action |
| --- | --- |
| `l` | toggle language tags |
| `v` | toggle vulgarity filter |
| `p` | toggle PII redaction |
| `s` | toggle shade elision |
| `Ctrl-D` | quit |

Toggling live is a good demo beat: say the same sentence with `v` off and on.

Flags for `bleep.py` (or `mise run start --whisper-model base --device 4`; extra flags after `--`):

| Flag | Default | Purpose |
| --- | --- | --- |
| `--no-language` / `--no-vulgarity` / `--no-pii` / `--no-shade` | all on | start with a feature disabled |
| `--whisper-model` | `medium` | whisper.cpp model size; `small`/`base` are faster, `medium` is more accurate |
| `--device N` | system default | input device index; `--list-devices` shows them |
| `--gate-db` | `-40` | energy gate in dBFS; raise toward `-30` in a noisy room |
| `--silence-ms` | `700` | pause length that ends an utterance |
| `--jev-model` | `jev-latest` | pin e.g. `jev-1.13.0` if you tune thresholds |
| `--stream` | off | live line while speaking (`mise run start-stream`) |
| `--partial-ms` | `1000` | `--stream`: new audio between live updates |

### Streaming mode

`--stream` re-transcribes the utterance-so-far every `--partial-ms` of new
audio, judges it with Jev, and redraws one live line in place; the pause
still produces the final line and status. Every live line is judged before
it's drawn, so unredacted PII never hits the screen. Updates are skipped when
Whisper + Jev fall behind the mic. The cost is one extra Jev request per
update, roughly 5–8× the tokens per utterance, and the status line shows it:
`+7 live (9927 tok) = $0.000484`.

## How it works

```
mic ──▶ energy gate ──▶ Whisper (local) ──▶ Jev (1 request) ──▶ render ──▶ stdout
        30ms blocks     whisper.cpp          ~20 questions       ▒ / ▓ / <lang>
        pause = cut     medium, Metal      in parallel
```

1. **Capture.** `sounddevice` streams 16 kHz mono audio in 30 ms blocks. A
   simple RMS gate marks speech; ~700 ms of quiet ends the utterance.
2. **Transcribe.** whisper.cpp (via `pywhispercpp`) runs the multilingual `medium` model, on the GPU via Metal on Apple Silicon.
3. **Judge.** The utterance becomes one Jev `system_one` request. State is
   `{"segment": text, "words": [...]}` and the questions are:
   - `language` — a **Choice** among `english`, `spanish`, `german`, `other`.
   - `vulgar_{i}` — one **Noul** per word: *"Is `words[i]` part of a word or
     phrase that would commonly be considered offensive?"* Multi-word
     expressions count every word, so a phrase is flagged as a run.
   - `pii_{i}` — one **Noul** per word: *"Does `words[i]` reveal a personal
     name, phone, address, email, or ID number?"*
   - `shade` — one **Noul** for the whole segment: *"Is the speaker talking
     shit about another person?"* with `criteria`
     spelling out the boundary: venting about a specific person you work with
     is a yes; a neutral mention, praise, or a complaint about a product or
     company is a no.

   All questions run in parallel inside a single request (TypeSafe's
   [speculative fan-out](https://docs.typesafe.ai/patterns/fan-out.md)
   pattern). A 15-word sentence is ~32 questions.
4. **Render.** Code — not the model — does the redaction: any word whose Noul
   probability is ≥ 0.5 is replaced — PII with block characters of the same
   length, each offensive word with one 🙊 — and the line is wrapped in the language tag. If `shade` is ≥ 0.5 the
   entire utterance is elided word-by-word (lengths preserved, naturally). Jev never generates text; it
   returns typed answers that the program consumes directly.

### Why Jev fits here

- **Latency.** Judgments return in 100–200 ms, well inside a broadcast delay.
- **Cost.** Pricing is $0.042 per million input tokens with free output, so an
  utterance costs ~$0.00005. An hour of continuous talk is a few cents.
- **Typed answers.** Probabilities and choices, not prose to parse. The
  threshold, the redaction style, and the feature toggles all live in code
  and can change without touching the model.
- **Cross-lingual.** The same questions work on German and Spanish without
  per-language rules. TypeSafe notes Jev is strongest in English
  and handles other languages less reliably
  cleanly in testing.

## Files

| File | Purpose |
| --- | --- |
| `bleep.py` | the demo: capture, transcribe, judge, render |
| `smoke_test.py` | runs `judge_segment` on eleven canned sentences (en/de/es, profanity incl. a multi-word phrase, PII, clean, shade incl. non-coworker + near-misses) |
| `Makefile` | self-documenting setup/test/run targets |
| `.envrc` | exports `TYPESAFE_API_KEY` (gitignored `.env` also supported by direnv) |

## Demo-day checklist

- Run `mise run setup` and then `mise run start` once from the terminal you'll present
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
| Nothing prints while talking | `mise run devices`, pass `--device N`; or lower `--gate-db` to `-45` |
| Captures constantly in a loud room | raise `--gate-db` to `-30` |
| `✗ TYPESAFE_API_KEY not set` | `direnv allow`, or `export TYPESAFE_API_KEY=...` |
| Long lag after each sentence | `mise run start-fast` (Whisper `base`) |
| Jev error in output | check the key and network; the raw transcript still prints so the demo continues |
