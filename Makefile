# bleep — live broadcast-delay demo powered by Jev (typesafe.ai)
# Run `make` with no arguments to see this help.

.DEFAULT_GOAL := help
WHISPER_MODEL ?= small

.PHONY: help check setup devices test run run-fast clean

help: ## Show available targets
	@echo "bleep — Jev broadcast-delay demo"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Variables: WHISPER_MODEL=$(WHISPER_MODEL) (e.g. make run WHISPER_MODEL=base)"

check: ## Verify uv is installed and TYPESAFE_API_KEY is set
	@command -v uv >/dev/null || { echo "✗ uv not found — install: brew install uv"; exit 1; }
	@echo "✓ uv $$(uv --version | cut -d' ' -f2)"
	@test -n "$$TYPESAFE_API_KEY" || { echo "✗ TYPESAFE_API_KEY not set — see .envrc (direnv allow)"; exit 1; }
	@echo "✓ TYPESAFE_API_KEY set"

setup: check ## Resolve dependencies and pre-download the Whisper model
	@echo "Resolving Python dependencies…"
	@uv run bleep.py --list-devices >/dev/null
	@echo "Downloading/caching Whisper '$(WHISPER_MODEL)' model…"
	@uv run --with faster-whisper python -c \
		"from faster_whisper import WhisperModel; WhisperModel('$(WHISPER_MODEL)', compute_type='int8')"
	@echo "✓ setup complete — run 'make test', then 'make run'"
	@echo "  (first 'make run' will prompt for microphone permission)"

devices: ## List audio input devices (pick one with: make run DEVICE=N)
	@uv run bleep.py --list-devices

test: check ## Smoke-test the Jev judgment path with canned sentences (no mic)
	@uv run smoke_test.py

run: check ## Run the live demo (keys: l/v/p/s toggle features, q quits)
	@uv run bleep.py --whisper-model $(WHISPER_MODEL) $(if $(DEVICE),--device $(DEVICE))

run-fast: check ## Run with the faster, less accurate 'base' Whisper model
	@$(MAKE) run WHISPER_MODEL=base

clean: ## Remove uv's cached environments for these scripts
	@uv cache clean >/dev/null 2>&1 || true
	@echo "✓ uv cache cleaned (Whisper models in ~/.cache/huggingface are kept)"
