# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A conference demo: live microphone speech → local Whisper transcription → one
[Jev](https://typesafe.ai) request per utterance → stdout with `<english>`/
`<spanish>`/`<german>` tags, PII as yellow `▒`
blocks, whole utterances that trash-talk a person as magenta `▓` blocks,
and a latency/cost status line. The point of the demo is Jev's speed
and cost, so keep the status line and the one-request-per-utterance shape.

## Commands

```sh
mise tasks                # list tasks (defined in mise.toml, which also pins uv)
mise run setup            # check TYPESAFE_API_KEY, resolve deps, cache Whisper model
mise run test             # smoke test: judge_segment on 10 canned sentences (live API, no mic)
mise run start              # live demo; --whisper-model base, --device N, extra bleep.py flags after --
mise run devices          # list audio input devices
# --debug / --show-request / --show-response work on both `run` and `test`
uv run bleep.py --help
uv run bleep.py --debug   # also --show-request / --show-response: dump Jev HTTP bodies
```

There is no lint or unit-test suite; `smoke_test.py` is the whole test. It
requires `TYPESAFE_API_KEY` (exported by `.envrc` via direnv).

Dependencies are PEP 723 inline metadata in each script, not a pyproject. A
bare `uv run python -c ...` has none of them; use `uv run --with <pkg> python`
or run the script itself. The app feeds Whisper numpy arrays, so pass audio
as arrays in any test harness.

## Working with Jev / TypeSafe

Use the `typesafe:typesafe-ai` skill and the live docs
(https://docs.typesafe.ai/llms.txt) as the source of truth; Jev is not in
training data. Facts that shaped this code:

- Jev is **text-only** and **never generates text** — it returns typed
  judgments (`Choice`, `Noul`, `Score`). Redaction is done in code.
- Questions in one `system_one` call run in parallel and can't see each
  other; batching ~35 questions per request is the intended pattern
  (speculative fan-out), not something to "optimize" into fewer calls.
- State is passed as a JSON string; questions reference it with backticked
  paths like `` `words[3]` ``. Question IDs are never shown to the model, so
  the full meaning must be in `instructions`.
- `Choice` leans toward the first option; `Noul` ≈ 0.5 means uncertain, not
  "medium". Redaction threshold is 0.5 in `judge_segment`.
- Pricing: $0.042/Mtok input, output free (`JEV_PRICE_PER_MTOK` in `bleep.py`).
  Token count comes from `raw_http_response.json()["usage"]` with a
  chars/4 fallback.

## Architecture (`bleep.py`, single file)

Four stages, each a top-level section in the file:

1. `UtteranceCapture` — `sounddevice` stream, 16 kHz mono, 30 ms blocks.
   An RMS energy gate (`--gate-db`) starts an utterance; `--silence-ms` of
   quiet or 12 s max ends it. Yields float32 numpy arrays.
2. Whisper — whisper.cpp via `pywhispercpp` (`small`, Metal on Apple
   Silicon; ~3x faster than the old `faster-whisper` CPU path). No `initial_prompt`.
3. `judge_segment` — builds the request: `language` Choice (english /
   spanish / german / other) plus a `pii_{i}` Noul for every
   whitespace-split word, plus one segment-level `shade` Noul (is the speaker
   disparaging any person?) with `NoulCriteria` true/false descriptions. Only
   enabled features add questions. Returns
   `(language, pii_flags, shade, latency_s, tokens)`.
4. `render`/`status` — ANSI output. Words are redacted whole-token, same
   length as the original (that's the joke). `shade` elides every word in the
   utterance the same way, and takes precedence over per-word flags.

`--stream`: `utterances(partial_ms)` also yields `(audio_so_far, False)`
every `partial_ms` of new audio, only when the block queue is empty (the
consumer has caught up). Main re-runs Whisper + `judge_segment` on each and
redraws one live line in place (`screen_rows`/`clear_rows`, wrap-aware); the
final `(audio, True)` prints the permanent line and a status that includes
live-update token cost. This is the one deliberate exception to
one-request-per-utterance; batch mode is still the default. With
`--show-request/--show-response`, live lines print on their own lines
instead of redrawing.

`features` is a dict mutated live by `key_listener` (raw-mode stdin thread:
`l`/`p`/`s`; Ctrl-D quits); the main loop snapshots it per utterance. The listener
silently exits when stdin isn't a TTY so the script runs headless.

`smoke_test.py` imports `bleep` and calls `judge_segment` + `render`
directly — the way to test judgment changes without a microphone.
