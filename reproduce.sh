#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PHYTOMNI_SAVE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if command -v uv >/dev/null 2>&1 && [[ -x "$ROOT/.venv/bin/python" ]]; then
  exec uv run python -m scripts.reproduce_cli "$@"
fi
exec python3 -m scripts.reproduce_cli "$@"
