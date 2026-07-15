# AGENTS.md

Agent guidance for this repository (`CLAUDE.md` is a symlink here). See `README.md` for full setup.

## Repository purpose

This repo contains the figure-reproduction code for the Phytomni manuscript ("An agentic AI for scientific discovery and design in plant research"). It is **not** an application — there is no build step or deployable package. Each top-level directory corresponds 1:1 to a figure or panel set in the paper (e.g. `Fig. 2/`, `Extended Data Fig. 6/`, `Supplementary Fig. 10-13/`) and is self-contained: notebooks/scripts in one directory do not import from another. (Exception: `AnalystAgent Evaluation/evaluation_scripts.ipynb` is an agent-evaluation harness, **not a figure** — it imports the external `mcp_server_phytomni` package, auto-downloads data from HuggingFace, and needs a live agent backend; see README's "Agent evaluation" section.)

**Reproducibility SSOT:** [`reproduce.manifest.yaml`](reproduce.manifest.yaml) is the authoritative target list (status, data deps, expected artifacts). The README reproduce table is **generated** from it (`python -m scripts.render_reproduce_matrix`); CI fails on drift. `./reproduce.sh` reads the manifest; `./reproduce.sh --check` is smoke mode (✓/✘/⊘). Figure runs and eval probes both write per-target logs to `logs/<target-id>.log` (e.g. `logs/ext-data-5c.log`, `logs/eval-analyst.log`; gitignored).

Quickest path: `uv sync` (Python only, from `pyproject.toml`) or `conda env create -f environment.yml` (Python + R in one env). **Python ≥ 3.12, R 4.0+.** The `plotly==6.0.1` / `kaleido==0.2.1` pins are **mandatory** — the plotly notebooks call `pio.kaleido.scope.default_format`, which kaleido v1.x removed. `uv.lock` **is committed** (reproducibility repo): `uv sync --frozen` reproduces the Python environment bit-for-bit. **`renv.lock`** records the R package fingerprint used when figures were validated (including GitHub `ggradar`); `renv::restore()` is best-effort under conda — see README.

Unit tests: `uv run pytest tests/ -v` (manifest loader, skip logic, artifact checks).

## Things that catch you out

**Spaces and literal dots in every path.** Directory names are `Fig. 2`, `Extended Data Fig. 5`, etc., and notebooks are named `fig. 2.ipynb`, `extended_data_fig. 6de.ipynb`. Always quote paths in shell commands:
```bash
cd "Extended Data Fig. 5" && Rscript "extended_data_fig. 5c.R"
jupyter nbconvert --to notebook --execute "Fig. 2/fig. 2.ipynb"
```

**Notebook filenames mostly follow a `fig. N` / `supplementary_fig. N` / `extended_data_fig. N` pattern — but `ls` the directory first anyway.** Multi-panel directories carry a suffix you can't derive (`extended_data_fig. 5ab.ipynb`, `5d.ipynb`, `6abc.ipynb`, `6de.ipynb`, `6fg.ipynb`, plus the `extended_data_fig. 6abc.Rmd` R Markdown), and one file breaks the pattern entirely (`AnalystAgent Evaluation/evaluation_scripts.ipynb`); `Supplementary Fig. 7/supplementary_fig. 7.py` is a `.py` script (not `.ipynb`) but now follows the `supplementary_fig. N` naming. (Historical note: after the 2026 renumbering, notebook filenames were normalized to match their parent directory number — the old Title-Case-verbatim names no longer exist.)

**Some `.ipynb` files are R notebooks, not Python.** They declare the standard `ir` kernel (install once: `R -e "IRkernel::installspec()"`; headless `nbconvert` fails without it):
- `Extended Data Fig. 5/extended_data_fig. 5ab.ipynb` — R (tidyverse, treemapify, RColorBrewer)
- `Extended Data Fig. 6/extended_data_fig. 6abc.ipynb` — R (readxl, ggplot2, tidyr)
- `Extended Data Fig. 7/extended_data_fig. 7.ipynb` — R (ggplot2, dplyr), inline data
- `Extended Data Fig. 6/extended_data_fig. 6abc.Rmd` — R **Markdown** (tidyverse, scales, ggradar), reads `PhytoBench-RAG-for_plot.csv`

All other `.ipynb` files are Python (the R notebooks / R Markdown are listed above). Most use plotly (`Fig. 2`, `Supplementary Fig. 14`, `Supplementary Fig. 10-13`, `Supplementary Fig. 24`); some use matplotlib/seaborn (`Extended Data Fig. 5/5d.ipynb`, `Extended Data Fig. 6/6de.ipynb` + `6fg.ipynb`, `Supplementary Fig. 8`, `Supplementary Fig. 9`); `Supplementary Fig. 19` mixes both.

**Figure-saving is gated behind the `PHYTOMNI_SAVE` env var.** A default run (unset) renders inline and writes **nothing**; set `PHYTOMNI_SAVE=1` to emit each figure into that file's own `output/` subdir (e.g. `Fig. 2/output/fig.2a.*.pdf`). Every file follows this — Python notebooks (`if SAVE_FIGS:` blocks), the R notebooks/`.Rmd` (`if (save_figs)`), the standalone `5c.R`, and `Supplementary Fig. 7/supplementary_fig. 7.py`. `output/` is gitignored.

**A second standalone script (Python).** `Supplementary Fig. 7/supplementary_fig. 7.py` is a Python script (not a notebook), run with `python3` (not `$NBX`). It reads same-dir `PhytoBench-Data-for_plot.xlsx` by a bare relative path, so `cd "Supplementary Fig. 7"` first. Like every other file it now follows `PHYTOMNI_SAVE` gating (previously it saved unconditionally; `5c.R` previously auto-saved via `ggsave` — both are gated now, no file auto-saves).

**A second non-figure eval harness (committed, cannot run from a clone).** `DataAgent Evaluation/` is a rewrite→NL2SQL evaluation pipeline (peer to `AnalystAgent Evaluation/`), tracked in git but reproducible only with private backends. Run scripts from inside `DataAgent Evaluation/src/` — they use `../data`, `../output`, `../result` bare relative paths: `exp_rewrite.py` reads `../data/PhytoBench-Data.xlsx` → writes `../output/`; `exp_nl2sql.py` then reads `../output/` → writes `../result/`. Blockers: (1) `prompt_template.yaml` is NOT in the repo yet `rewrite.py` opens it at import time, so a bare `python exp_rewrite.py` raises `FileNotFoundError` before any backend is even reached; (2) every backend is a `Change_to_your_*` placeholder (OpenAI-compatible LLM in `model/config.py`, IAM token endpoints in `utils.py`/`model/get_token.py`, NL2SQL service URL, RAG KS URL); (3) deps `langchain`, `openai`, `httpx`, `tqdm`, `pyyaml`, `requests` are in NEITHER manifest. Not a figure; will not reproduce from the figure env. `./reproduce.sh --check` runs **probes only** for eval targets (always ⊘ on a bare clone).

**Honest 5a / 5b split.** `extended_data_fig. 5ab.ipynb` is shared, but the manifest defines two targets: `ext-data-5a` (`status: run`, needs `Phytomni-PaperYear-for_plot.csv`) and `ext-data-5b` (`status: skip_until_data`, needs `Phytomni-DocType-for_plot.csv` not yet in the repo). A `have_5b <- file.exists(...)` guard in the notebook lets 5a run when 5b's CSV is absent; smoke reports 5b as ⊘ until the CSV ships. Check `output/extended_data_fig.5b.pdf` to confirm 5b actually emitted.

**Pending-data and deprecated targets.** `ext-data-6-radar` (`extended_data_fig. 6abc.Rmd`) is `skip_until_data` until `PhytoBench-RAG-for_plot.csv` arrives. `ext-data-6ab-deprecated` is `deprecated` (orphan notebook, not executed).

All other data-consuming notebooks read via relative paths and run end-to-end from a fresh clone. Data files follow the `<DatasetName>-for_plot.<ext>` convention. In-repo data files:
- `Fig. 3/PhytoBench-Paper-for_plot.xlsx`
- `Extended Data Fig. 5/Phytomni-Multiomics-for_plot.txt`, `Phytomni-PaperYear-for_plot.csv`
- `Extended Data Fig. 6/PhytoBench-Knowledge-for_plot.xlsx` (consumed by `6abc.ipynb` via relative path — works out of the box)
- `Fig. 2/PhytoBench-Gene-BERTScore-for_plot.tsv` and `Fig. 2/PhytoBench-Gene-hallucination-for_plot.tsv` (frozen five-model inputs for Fig. 2g-h)
- `Supplementary Fig. 10-13/PhytoBench-Gene-for_plot/frozen/` (authoritative schema-2 snapshot: 12 aggregate TSVs plus `provenance.json`; includes rank distributions, Plackett-Luce scores/pairwise probabilities and their 10,000-bootstrap 95% confidence intervals, Fleiss-κ, per-gene Kendall W, top-1 consensus, panel composition, and assignment summaries; it also contains the four compact Claude metric tables and provenance record used to assemble Fig. 2g-h)
- `Supplementary Fig. 10-13/PhytoBench-Gene-for_plot/gene_categories.tsv` (public study-status mapping used when regenerating the frozen aggregates from the anonymized ranking release)
- `DeepGenomeAgent Evaluation/supplementary/Supplementary_Data_Expert_Rankings.tsv` (full public anonymized table: 600 expert–gene assignments from 120 experts, one complete five-model ranking per row), with `Supplementary_Data_Expert_Rankings_Codebook.tsv` and `expert_panel_category_map.tsv` alongside it
- `Supplementary Fig. 10-13/PhytoBench-Gene-for_plot/legacy/score*.tsv` (legacy four-model row-level inputs retained for historical provenance; not consumed by the current notebook or manifest)
- `Supplementary Fig. 7/PhytoBench-Data-for_plot.xlsx` (consumed by `supplementary_fig. 7.py` via bare relative path — run from inside the dir)
- `DataAgent Evaluation/data/PhytoBench-Data.xlsx` (consumed by `src/exp_rewrite.py` via the `../data/` relative path — run from inside `src/`)

For the expert-ranking audit, Fleiss-κ treats each gene×model pair as an item,
R1–R5 as categories, and the three assigned experts as raters. The frozen table
retains all 58 locked scopes; Supplementary Fig. 11 plots only the 1 primary and
12 locked-secondary scopes. `Research_domains` and `Study_species` are
multi-select panel dimensions and therefore use all 120 experts as their
denominator (their percentages can exceed 100% in total); single-select
dimensions use nonmissing respondents. The COI result is a self-report: all 120
experts selected `No`, so describe it as "no conflicts were declared," not as
independent exclusion of conflicts.

The five newer Supplementary notebooks (`Supplementary Fig. 8`, `9`, `14`, `19`, `24`) carry their data **inline** (hardcoded Python dicts / numpy arrays). `Fig. 2a-c` remain inline, while `Fig. 2g-h` read their tracked five-model metric tables. (The R notebook `Extended Data Fig. 7/extended_data_fig. 7.ipynb` likewise carries its inline data as an R `data.frame`.)

## Working with the notebooks

- Editing: prefer modifying notebook cells via `jupyter` rather than hand-editing JSON. When asked to add/change a panel, locate the markdown header (`### Fig. 2d`, etc.) and edit the code cell directly below it.
- Executing headlessly: `jupyter nbconvert --to notebook --execute --inplace "<path>"`.
- The Python plotly notebooks emit static images via `pio.write_image` / `fig.write_image`, which requires `kaleido==0.2.1` (now pinned in both manifests; v1.x removed the `pio.kaleido.scope` API the notebooks rely on).
- The pandas floor is `pandas>=2.1` in both manifests (verified on 3.0.2). pandas 3.0 removed `DataFrame.applymap`; the notebooks were patched to `.map` — don't reintroduce `.applymap` when editing cells.
