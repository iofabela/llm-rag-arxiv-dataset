#!/usr/bin/env bash
# Ensure `.env` exists and contains a usable OPENAI_API_KEY.
#
# Usage: bash scripts/ensure_env.sh
#   - If ".env" is missing it is created from ".env.example".
#   - If OPENAI_API_KEY is already set (non-empty), the script exits clean.
#   - Otherwise it prompts for the key, writes it to ".env", or aborts.
set -euo pipefail

ENV_FILE=".env"
ENV_EXAMPLE=".env.example"
KEY_PATTERN='^OPENAI_API_KEY='

log() { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[setup]\033[0m %s\n' "$*" >&2; exit 1; }

if [[ ! -f "$ENV_FILE" ]]; then
  [[ -f "$ENV_EXAMPLE" ]] || die "Missing $ENV_EXAMPLE — run this from the repository root."
  cp "$ENV_EXAMPLE" "$ENV_FILE"
  log "Created $ENV_FILE from $ENV_EXAMPLE."
fi

if grep -Eq "^OPENAI_API_KEY=.+$" "$ENV_FILE"; then
  log "OPENAI_API_KEY is already set in $ENV_FILE."
  exit 0
fi

read -r -p "[setup] OPENAI_API_KEY is empty. Paste your OpenAI API key (press Enter to abort): " KEY
if [[ -z "$KEY" ]]; then
  die "Aborted. Add OPENAI_API_KEY=... to $ENV_FILE and run 'make setup' again."
fi

if grep -Eq "$KEY_PATTERN" "$ENV_FILE"; then
  # Portable in-place replace (works on both GNU sed and BSD/macOS sed).
  perl -i -pe "s/^OPENAI_API_KEY=.*/OPENAI_API_KEY=$KEY/" "$ENV_FILE"
else
  printf 'OPENAI_API_KEY=%s\n' "$KEY" >> "$ENV_FILE"
fi

log "Saved OPENAI_API_KEY to $ENV_FILE."