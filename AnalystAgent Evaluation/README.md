# AnalystAgent Evaluation

This is an evaluation workflow, not a public figure-rendering target. Commands
below use the repository root unless stated otherwise. See the
[repository guide](../README.md) for figure reproduction.

`AnalystAgent Evaluation/evaluation_scripts.ipynb` benchmarks the Phytomni analyst agent over 10 bioinformatics tasks and writes JSON run-logs to `submit_log/` (no PDF/PNG). Beyond the figure environment it requires:

- **`mcp_server_phytomni`** — install from the public
  [Phytomni-Bot repository](https://github.com/Phytomni/Phytomni-Bot) by
  following its Quick Start instructions. The package is separate from this
  figure repository and its manifests.
- **`huggingface_hub`** — `pip install huggingface_hub`. On first run the notebook auto-downloads benchmark data from <https://huggingface.co/datasets/Phytomni/PhytoBench-Analysis> into `AnalystAgent Evaluation/PhytoBench-Analysis/`.
- **A configured Phytomni-Bot backend** to which the notebook submits tasks
  through `retrieve_plan_submit`.

Downloaded data and `submit_log/` are gitignored.
