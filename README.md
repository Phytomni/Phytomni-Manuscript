# Phytomni Manuscript Code

Phytomni: Reproducibility code and notebooks for "An agentic AI for scientific discovery and design in plant research"

## Overview

This repository provides the code, scripts, and notebooks to reproduce figures, tables, and quantitative results reported in the manuscript "Phytomni: An agentic AI for scientific discovery and design in plant research." Phytomni is a domain-specific, LLM-powered multi-agent system built on the Model Context Protocol (MCP) that integrates a plant-focused full-text knowledge base (~4.0M publications plus abstracts and patents), multi-omics data spanning 65 species, and 125 bioinformatics tools. The platform orchestrates hierarchically coordinated agents—Knowledge, Data, Analyst—and composite agents (e.g., In Silico Research Agent, Deep Genome Agent, Gene Network Agent, Digital Design Agent) to automate literature-grounded reasoning, data retrieval, and end-to-end bioinformatic analyses.

Code in this repository is organized by figure directory and has been tested with Python 3.12+ and R 4.0+. Each top-level directory corresponds to one figure or panel set in the paper and is self-contained.

## Environment setup

### Recommended setup

The repo ships a [`pyproject.toml`](pyproject.toml) (for [`uv`](https://docs.astral.sh/uv/)) and an [`environment.yml`](environment.yml) (for `conda`). Pick one — each installs all top-level dependencies in a single command.

**Option A — `uv` (Python only, fastest):**
```bash
uv sync          # creates .venv and installs every Python dep from pyproject.toml
source .venv/bin/activate
```
For the R script and R-language notebooks, additionally run the R install line under [R Dependencies](#r-dependencies).

**Option B — `conda` (Python + R in one env):**
```bash
conda env create -f environment.yml
conda activate phytomni-fig
```

> **Note:** Both manifests pin `plotly==6.0.1` and `kaleido==0.2.1` (conda: `python-kaleido=0.2.1`). The notebooks use `pio.kaleido.scope.default_format`, which kaleido v1.x removed, so these pins are mandatory.

The remainder of this section describes the individual dependencies in case you prefer to manage them manually.

### Python Dependencies
```bash
pip install "pandas>=2.1" matplotlib seaborn numpy openpyxl \
    "plotly==6.0.1" "kaleido==0.2.1" "nbformat>=4.2.0" ipywidgets \
    colorlover scipy
```

> Python ≥ 3.12 is required (matches the pinned manifests). The `kaleido==0.2.1` and `plotly==6.0.1` pins are required because the notebooks use the kaleido v0 `pio.kaleido.scope` API; v1.x removed it. `pandas>=2.1` is the true minimum (the notebooks use `DataFrame.map`, added in 2.1); the environment is verified on pandas 3.0.2.

### R Dependencies
```bash
R -e "install.packages(c('tidyverse', 'scales', 'treemapify', 'viridis', 'readxl', 'RColorBrewer', 'rmarkdown'))"
# ggradar is GitHub-only (not on CRAN), required by extended_data_fig. 6abc.Rmd:
R -e "install.packages('remotes'); remotes::install_github('ricardo-bion/ggradar')"
```

### Jupyter Notebook Support
```bash
pip install jupyterlab
# R kernel in Jupyter (required for the R notebooks; installs a kernel named "ir")
R -e "install.packages(c('IRkernel'))"
R -e "IRkernel::installspec()"
```

### R version locking (renv)

`environment.yml` installs R packages via conda-forge. `renv.lock` records the exact versions used when figures were validated (including GitHub `ggradar`). After creating the conda (or system R) environment:

```r
renv::restore()
```

Under conda, `renv::restore()` may conflict with the conda R library path — treat it as best-effort (it may fail). When restore is awkward, rely on `environment.yml` for installation; **`renv.lock` remains the authoritative R package fingerprint for reviewers.** The lockfile was snapshotted on R 4.3.3 while `environment.yml` keeps `r-base>=4.0`, so minor R-version drift is possible.

This repo does not ship a Docker image.

> **Conda vs `uv.lock`:** `environment.yml` pins the same major Python package versions as `uv.lock` (pandas 3.0.2, numpy 2.4.4, etc.). Conda may resolve different build strings per platform/Python minor; use `uv sync --frozen` when you need bit-for-bit parity with the committed Python lockfile.

## Reproducing the figures

> **Authoritative target list:** [`reproduce.manifest.yaml`](reproduce.manifest.yaml). The table below is generated; CI fails if it drifts.

Each figure directory is self-contained and reads its data via relative paths, so notebooks run from a fresh clone. **Run everything from the repository root.** (This applies to the figure directories; `AnalystAgent Evaluation/` is an agent-evaluation harness, not a figure — see [Agent evaluation](#agent-evaluation-not-a-figure) below.)

The headless run command is identical for every notebook; define it once:
```bash
NBX='jupyter nbconvert --to notebook --execute --inplace'
```
(Interactive alternative: launch `jupyter lab`, open the file named in the table, and run all cells.)

### One command

```bash
PHYTOMNI_SAVE=1 ./reproduce.sh     # run every figure; artifacts land in each figure's output/
./reproduce.sh --check             # smoke: run all, print ✓/✘/⊘ summary, non-zero exit on failure
```
The ED Fig. 6 radar target has an explicit data/toolchain pre-check, so it is reported `⊘` (with the reason) and never fails the run when `PhytoBench-RAG-for_plot.csv`, `ggradar`, or R is missing. ED Fig. 5b shares one notebook with 5a: when `Phytomni-DocType-for_plot.csv` is absent the notebook's internal guard skips only the 5b panel (5a still renders), so the bundled `Ext. Data 5a,b` target still reports `✓` — check for `extended_data_fig.5b.pdf` in `output/` to confirm 5b actually emitted. Every push is exercised end-to-end by GitHub Actions (see `.github/workflows/reproduce.yml`).

### Reproduction matrix

<!-- BEGIN:REPRODUCE_MATRIX -->
Legend: ✓ = data ships in the repo · ⚠ = data pending (you must supply it) · *inline* = data is hardcoded in the notebook. Artifacts are written only when `PHYTOMNI_SAVE=1` is set; a default run renders inline and writes nothing.

| Figure | File | Kernel | Input data | Status | Emits (into output/ when PHYTOMNI_SAVE=1) |
|---|---|---|---|---|---|
| Fig. 2 | `Fig. 2/fig. 2.ipynb` | `python3` | *inline* | `run` | `fig.2a.phytobench-knowledge.bar.pdf` / `fig.2b.phytobench-data.bar.pdf` / `fig.2c.phytobench-analysis.violin.pdf` / … (5 files) |
| Fig. 3 | `Fig. 3/fig. 3.ipynb` | `python3` | `PhytoBench-Paper-for_plot.xlsx` ✓ | `run` | `fig.3d.phytobench-paper.line.pdf` / `fig.3e.phytobench-paper.heatmap.pdf` |
| Ext. Data 5a | `Extended Data Fig. 5/extended_data_fig. 5ab.ipynb` | `ir` (R) | `Phytomni-PaperYear-for_plot.csv` ✓ | `run` | `extended_data_fig.5a.pdf` |
| Ext. Data 5b | `Extended Data Fig. 5/extended_data_fig. 5ab.ipynb` | `ir` (R) | `Phytomni-DocType-for_plot.csv` ⚠ | `skip_until_data` | `extended_data_fig.5b.pdf` |
| Ext. Data 5c | `Extended Data Fig. 5/extended_data_fig. 5c.R` | Rscript | `Phytomni-Multiomics-for_plot.txt` ✓ | `run` | `extended_data_fig.5c.png` |
| Ext. Data 5d | `Extended Data Fig. 5/extended_data_fig. 5d.ipynb` | `python3` | *inline* | `run` | `extended_data_fig.5d.pdf` |
| Ext. Data 6a-c | `Extended Data Fig. 6/extended_data_fig. 6abc.ipynb` | `ir` (R) | `PhytoBench-Knowledge-for_plot.xlsx` ✓ | `run` | `extended_data_fig.6a.pdf` / `extended_data_fig.6b.pdf` |
| Ext. Data 6 radar | `Extended Data Fig. 6/extended_data_fig. 6abc.Rmd` | R / `rmarkdown::render` | `PhytoBench-RAG-for_plot.csv` ⚠ | `skip_until_data` | `extended_data_fig.6-radar-figure1.pdf` / `extended_data_fig.6-radar-figure2.pdf` / `extended_data_fig.6-radar-figure3.pdf` / `extended_data_fig.6-radar-figure4.pdf` |
| Ext. Data 6d,e | `Extended Data Fig. 6/extended_data_fig. 6de.ipynb` | `python3` | *inline* | `run` | `extended_data_fig.6d.pdf` / `extended_data_fig.6e.pdf` |
| Ext. Data 6f,g | `Extended Data Fig. 6/extended_data_fig. 6fg.ipynb` | `python3` | *inline* | `run` | `model_compare_agent_total.pdf` / `model_compare_agent_total_across_speciesv1.pdf` |
| Ext. Data 7 | `Extended Data Fig. 7/extended_data_fig. 7.ipynb` | `ir` (R) | *inline* | `run` | `extended_data_fig.7.pdf` |
| Supp. 1 | `Supplementary Fig. 1/supplementary_fig. 1.ipynb` | `python3` | `Phyto-Chatbot-Pretrain.loss.json` ✓, … (4 files) | `run` | `supplementary_fig.1a.chatbot_pretrain_loss.line.pdf` / `supplementary_fig.1b.reasoner_pretrain_loss.line.pdf` / `supplementary_fig.1c.chatbot_sft_loss.line.pdf` / `supplementary_fig.1d.reasoner_sft_loss.line.pdf` |
| Supp. 6 | `Supplementary Fig. 6/supplementary_fig.6.ipynb` | `ir` (R) | *inline* | `run` | `supplementary_fig.6a.pdf` / `supplementary_fig.6b.pdf` |
| Supp. 17 | `Supplementary Fig. 17/supplementary_fig.17.ipynb` | `ir` (R) | *inline* | `run` | `supplementary_fig.17.pdf` |
| Ext. Data 6ab (deprecated notebook) | `Extended Data Fig. 6/extended_data_fig.6ab.ipynb` | none | *inline* | `deprecated` | — |
| Supp. 7 | `Supplementary Fig. 7/supplementary_fig. 7.py` | `python3` (script) | `PhytoBench-Data-for_plot.xlsx` ✓ | `run` | `model_accuracy_by_species.pdf` |
| Supp. 8 | `Supplementary Fig. 8/supplementary_fig. 8.ipynb` | `python3` | *inline* | `run` | `model_compare_agent_split.pdf` |
| Supp. 9 | `Supplementary Fig. 9/supplementary_fig. 9.ipynb` | `python3` | *inline* | `run` | `model_compare_agent_split_across_speciesv1.pdf` |
| Supp. 10-13 | `Supplementary Fig. 10-13/supplementary_fig. 10-13.ipynb` | `python3` | `score.tsv` ✓, `score.well_studied.tsv` ✓, `score.uncharacterized.tsv` ✓ | `run` | `fig.2d.phytobench-gene.percent.bar.pdf` / `fig.2e.phytobench-gene.prob.heatmap.pdf` / `fig.2f.phytobench-gene.score.bar.pdf` / … (5 files) |
| Supp. 14 | `Supplementary Fig. 14/supplementary_fig. 14.ipynb` | `python3` | *inline* | `run` | `supplementary_fig.6a.model.paperbench-mp.line.pdf` / `supplementary_fig.6b.model.paperbench-as.line.pdf` / `supplementary_fig.6c.model.paperbench-cr.line.pdf` / … (5 files) |
| Supp. 19 | `Supplementary Fig. 19/supplementary_fig. 19.ipynb` | `python3` | *inline* | `run` | `supplementary_fig.19.pdf` |
| Supp. 24 | `Supplementary Fig. 24/supplementary_fig. 24.ipynb` | `python3` | *inline* | `run` | `supplementary_fig.13.phytobench-review.polar.pdf` |
<!-- END:REPRODUCE_MATRIX -->

### How to run

- **Python notebooks** need only the Python environment from [Environment setup](#environment-setup). Run headlessly with `$NBX "<file>"`, or open the file in `jupyter lab` and run all cells.
- **R notebooks** (`5ab`, `6abc`, `7`; plus the `6abc.Rmd` R Markdown) need the `ir` kernel / an R install — install the kernel once with `R -e "IRkernel::installspec()"`. Without it, `nbconvert` reports `No such kernel`.
- **The R script** (`5c.R`) runs standalone with `Rscript`; its save (`ggsave`) is gated behind `PHYTOMNI_SAVE` like every other file.
- **The Python script** (`Supplementary Fig. 7/supplementary_fig. 7.py`) also runs standalone with `python3` (not via `$NBX` — it is a script, not a notebook). It reads `PhytoBench-Data-for_plot.xlsx` by a bare relative path, so run it from inside its directory: `cd "Supplementary Fig. 7" && python3 "supplementary_fig. 7.py"`.
- **Figure-saving is gated behind `PHYTOMNI_SAVE`.** A default run renders each figure inline and writes nothing. Set `PHYTOMNI_SAVE=1` to emit every figure into that directory's `output/` (gitignored; filenames follow `<figure>.<panel>.pdf`/`.png`). `reproduce.sh` does this for you — see [One command](#one-command) above.

## Agent evaluation (not a figure)

`AnalystAgent Evaluation/evaluation_scripts.ipynb` is **not a figure-reproduction notebook** — it benchmarks the Phytomni analyst agent over 10 bioinformatics tasks and writes JSON run-logs to `submit_log/` (no PDF/PNG). It requires, beyond the figure environment:

- **`mcp_server_phytomni`** — install manually from <https://github.com/Phytomni/Phytomni-Bot> (not on PyPI; not in this repo's manifests).
- **`huggingface_hub`** — `pip install huggingface_hub`. On first run the notebook auto-downloads the benchmark data from <https://huggingface.co/datasets/Phytomni/PhytoBench-Analysis> into `AnalystAgent Evaluation/PhytoBench-Analysis/`.
- **A live Phytomni agent backend** the notebook submits tasks to (`retrieve_plan_submit`).

The downloaded data and `submit_log/` are gitignored. This harness cannot run from a clone with the figure environment alone.

`DataAgent Evaluation/` is a second, separate evaluation harness (a rewrite → NL2SQL pipeline for the Data agent) — also **not a figure**. Its scripts read `../data`, `../output`, `../result` by bare relative paths, so run them from inside `DataAgent Evaluation/src/`: `exp_rewrite.py` rewrites questions from `../data/PhytoBench-Data.xlsx` into `../output/`, then `exp_nl2sql.py` runs those through a NL2SQL service into `../result/`. Beyond the figure environment it requires:

- **`langchain`, `openai`, `httpx`, `tqdm`, `pyyaml`, `requests`** — none are in this repo's manifests; `pip install` them separately.
- **A `prompt_template.yaml` in `src/`** — not shipped in the repo; `rewrite.py` loads it at import time, so the scripts fail immediately without it. Supply your own.
- **Private backends** — the LLM endpoint, IAM token service, NL2SQL service URL, and RAG knowledge-search URL are all `Change_to_your_*` placeholders you must fill in.

The `output/` and `result/` directories it creates are gitignored. Like the AnalystAgent harness, this cannot run from a clone with the figure environment alone.

## Help

Please post in the GitHub issues or contact the authors with any questions about the repository, requests for more data, or additional information about the results.
