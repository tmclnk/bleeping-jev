# bleep

Live broadcast-delay demo powered by [Jev](https://typesafe.ai) (TypeSafe System One).

Speak into the mic → Whisper transcribes locally → one Jev request judges the
utterance → the line prints to stdout with:

- `<english>` / `<spanish>` / `<german>` language tags (Jev `Choice`)
- vulgarities bleeped as red `████` (one Jev `Noul` per word)
- PII redacted as yellow `▒▒▒▒` (one Jev `Noul` per word)
- a status line with Jev's wall-clock latency and token cost — the whole
  judgment runs in ~100–200ms for ~$0.00005 per utterance

## Run

```sh
make setup   # resolve deps, pre-download the Whisper model
make test    # smoke-test the Jev path with canned sentences (no mic)
make run     # live demo
```

`make` alone lists all targets. The API key is read from `TYPESAFE_API_KEY`
(set via `.envrc`). Dependencies are inline (PEP 723), so `uv run bleep.py`
also works directly.

## Controls

| Key | Action |
| --- | --- |
| `l` | toggle language tags |
| `v` | toggle vulgarity bleeping |
| `p` | toggle PII redaction |
| `q` | quit |

Flags: `--no-language --no-vulgarity --no-pii` to start with a feature off,
`--whisper-model base` for faster (less accurate) transcription,
`--gate-db -35` if room noise keeps triggering capture, `--list-devices` /
`--device N` to pick a mic.

## How it works

Each utterance (speech followed by a ~700ms pause) becomes one Jev request.
State is `{"segment": text, "words": [...]}`; the questions are a language
`Choice` plus two `Noul`s per word (profanity? PII?), all batched — Jev runs
them in parallel, per the [speculative fan-out pattern](https://docs.typesafe.ai/patterns/fan-out.md).
Flagged words are replaced with block characters before printing. Jev never
generates text; code owns the output, Jev supplies the judgments.

## Demo-day checklist

- Run once beforehand: first run downloads the Whisper `small` model (~460MB)
  and macOS will prompt for microphone permission for your terminal.
- Fullscreen the terminal and crank the font size (`Cmd +`) for the projector.
- Whisper on CPU adds ~1–3s per utterance; Jev itself is ~150ms. Pause briefly
  between sentences so utterances segment cleanly.
- Speak one language per sentence — the chunker splits on pauses, and Whisper
  detects language per chunk.
- TypeSafe docs note Jev is strongest in English; Spanish/German judgments
  worked well in testing but rehearse your exact lines once.
