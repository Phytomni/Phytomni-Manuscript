#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if ! command -v python >/dev/null 2>&1 || ! python -c 'import os, sys; sys.exit(not (sys.version_info >= (3, 12) and (sys.prefix != sys.base_prefix or os.environ.get("CONDA_PREFIX") == sys.prefix)))'; then
  echo "Activate a prepared Python >=3.12 environment first: uv sync --frozen; source .venv/bin/activate" >&2
  exit 1
fi
exec python -m scripts.reproduce_cli "$@"
