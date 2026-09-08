# DataAgent Evaluation

This is an evaluation workflow, not a public figure-rendering target. Commands
below use the repository root unless stated otherwise. See the
[repository guide](../README.md) for figure reproduction.

`DataAgent Evaluation/` is a rewrite → NL2SQL pipeline for the Data agent. Scripts read `../data`, `../output`, `../result` by bare relative paths — run them from inside `DataAgent Evaluation/src/`: `exp_rewrite.py` rewrites questions from `../data/PhytoBench-Data.xlsx` into `../output/`, then `exp_nl2sql.py` runs those through an NL2SQL service into `../result/`. Beyond the figure environment it requires:

- **`langchain`, `openai`, `httpx`, `tqdm`, `pyyaml`, `requests`** — the base figure environment is not a complete runtime for these services; install and validate evaluator dependencies separately. `pyyaml` is also a base plotting dependency.
- **A `prompt_template.yaml` in `src/`** — not shipped in the repo; `rewrite.py` loads it at import time, so the scripts fail immediately without it. Supply your own.
- **Private backends** — LLM endpoint, IAM token service, NL2SQL service URL, and RAG knowledge-search URL are `Change_to_your_*` / `change_to_your_*` placeholders you must fill in (`src/model/config.py`, `src/utils.py`, `src/exp_nl2sql.py`, `src/rag/rag_run.py`, etc.).

The `output/` and `result/` directories it creates are gitignored.
