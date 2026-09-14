# Phytomni Manuscript Code

Figure-reproduction, scoring, and evaluation materials for "An agentic AI for scientific discovery and design in plant research".

The application and agent runtime live in the separate [Phytomni-Bot repository](https://github.com/Phytomni/Phytomni-Bot). This repository is not an application or a deployable package. Figure directories are self-contained; [reproduce.manifest.yaml](reproduce.manifest.yaml) is the authoritative target, input, and artifact registry.

## Choose your workflow

### 1. Reproduce figures from public frozen inputs

No private workbook, model download, API key, or live model call is needed. Prepare the [reference environment](#reference-environment), activate it, then run from the repository root:

```bash
./reproduce.sh list
./reproduce.sh validate
./reproduce.sh run --target fig-2 --timeout 600
./reproduce.sh run --timeout 600
```

`list` reports declarations and input availability without checking runtimes.
`validate` checks public source/input contracts and README synchronization without
executing notebooks. `run` generates fresh figures in isolated copies, validates
every selected PDF/PNG, then publishes only declared outputs into the owning
figure's `output/`. Shared notebooks execute once for the selected targets.
Use repeated `--target ID` options to select several targets.

No arguments show help. Invalid commands/targets return 2. Missing required data,
runtime failures, invalid artifacts, or zero executed sources return nonzero.
Declared pending data is reported separately; explicitly selecting an unavailable
target fails. Evaluation availability does not mask figure failures.

[Fig. 2's guide](<Fig. 2/README.md>) explains the left and right halves of panels
g-h. [The ranking-data guide](<Supplementary Fig. 12-15/PhytoBench-Gene-for_plot/README.md>)
covers Fig. 2d-f,g-h (left) and Supplementary Figs. 12-15.
[The SeedBench guide](<Supplementary Fig. 19/README.md>) covers Supplementary
Fig. 19a, including the frozen task summaries and their aggregation provenance.

### 2. Recompute public ranking statistics independently

The public anonymous ranking table and gene-category mapping suffice for the
11 numerical tables. This does not reconstruct private expert demographics or
repeat the original model evaluations. Start with the independently saved
100-replicate preview in the [ranking recomputation guide](<Supplementary Fig. 12-15/PhytoBench-Gene-for_plot/README.md>):

```bash
python -m scripts.recompute_deepgenome_rankings --help
```

The guide provides the complete command, fixed seed, input paths, comparison
rules, and independent `local_exports/ranking-recomputation/<run-id>/` destination.
Preview intervals are provisional. A 10,000-replicate run is an explicit,
potentially lengthy calculation, never part of normal figure reproduction.
Even a full comparison cannot overwrite the publication automatically.

### 3. Repeat original evaluations or update a publication

These workflows require their own inputs, optional dependencies and, for live
evaluations, configured services and potentially billable calls. Follow the
[separate evaluator guides](#agent-evaluation-not-a-figure). First inspect
prerequisites without executing an evaluator:

```bash
./reproduce.sh probe --target eval-expert
```

A probe reports discoverability only, not service health or successful evaluation.
Unavailable prerequisites return nonzero. Neither `run` nor `validate` invokes
an evaluator. A controlled publication update requires the complete private
metadata/lineage checks described in the DeepGenomeAgent guide.

## Reference environment

The formal runner targets Linux. The reference environment uses Ubuntu 24.04,
Python 3.12.13, R 4.3.3, [uv.lock](uv.lock), and [renv.lock](renv.lock).
Package locks do not lock the operating system, fonts, or every rendered byte.
The mandatory `plotly==6.0.1` and `kaleido==0.2.1` pins preserve the static-export
API used by these notebooks; do not upgrade one independently.
The static exporters disable unused MathJax loading so asynchronous external
assets cannot add browser-status text to the PDFs. These charts use plain-text
labels, not MathJax expressions.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python
3.12.13, then explicitly prepare and activate the Python environment:

```bash
uv lock --check
uv sync --frozen --python 3.12.13
source .venv/bin/activate
```

Use a separate environment for optional evaluation dependencies. The figure
runner uses the activated interpreter and does not synchronize packages, install
R dependencies, or fall back to another Python. For Python-only reproduction,
select Python targets from `list`; all-figure execution also requires R.

With R 4.3.3 and its system build dependencies installed, prepare the locked R
library explicitly. The [CI workflow](.github/workflows/reproduce.yml) lists the
Linux system packages and verifies every DESCRIPTION version plus the recorded
ggradar Git commit.

```bash
export R_PROFILE_USER=/dev/null
export R_LIBS_USER="$PWD/renv/library/R-4.3/x86_64-pc-linux-gnu"
Rscript --vanilla -e 'source("renv/activate.R"); renv::restore(library=Sys.getenv("R_LIBS_USER"), lockfile="renv.lock", strict=TRUE, prompt=FALSE)'
```

Restore failure is a failure, not a reason to silently use another library.
The runner creates run-local Python/R kernels, binds the explicit R library,
disables user/site configuration, and does not require global kernel registration.

`conda env create -f environment.yml` is an alternative convenience installation;
it is not equivalent to the locked reference environment. Interactive Jupyter use
is also supported: open the source notebook and run all cells with the appropriate
Python or `ir` kernel. Standalone notebook/script execution saves figures only
when `PHYTOMNI_SAVE=1`; otherwise it displays inline. Quote paths containing spaces.
The manifest runner sets the save flag only for figure execution and retains
executed copies without modifying source notebooks.

## Reproduction matrix

This table is generated from the manifest; do not edit it by hand.
Update it with `python -m scripts.render_reproduce_matrix --write-readme README.md`.
Read-only `validate` and CI reject drift.

<!-- BEGIN:REPRODUCE_MATRIX -->
Legend: ✓ = data ships in the repo · ⚠ = data pending (you must supply it) · *inline* = data is hardcoded in the notebook. Artifacts are written only when `PHYTOMNI_SAVE=1` is set; a default run renders inline and writes nothing.

| Figure | File | Kernel | Input data | Status | Emits (into output/ when PHYTOMNI_SAVE=1) |
|---|---|---|---|---|---|
| Fig. 2a-c,g-h (right) | `Fig. 2/fig. 2.ipynb` | `python3` | `PhytoBench-Knowledge-for_plot.tsv` ✓, … (6 files) | `run` | `fig.2a.phytobench-knowledge.bar.pdf` / `fig.2a.phytobench-knowledge.bar.png` / `fig.2b.phytobench-data.bar.pdf` / … (10 files) |
| Fig. 2d-f,g-h (left) | `Supplementary Fig. 12-15/supplementary_fig. 12-15.ipynb` | `python3` | `rank_distribution.tsv` ✓, … (13 files) | `run` | `fig.2d.phytobench-gene.percent.bar.pdf` / `fig.2d.phytobench-gene.percent.bar.png` / `fig.2e.phytobench-gene.prob.heatmap.pdf` / … (10 files) |
| Fig. 3 | `Fig. 3/fig. 3.ipynb` | `python3` | `PhytoBench-Paper-for_plot.xlsx` ✓ | `run` | `fig.3d.phytobench-paper.line.pdf` / `fig.3d.phytobench-paper.line.png` / `fig.3e.phytobench-paper.heatmap.pdf` / `fig.3e.phytobench-paper.heatmap.png` |
| Ext. Data 5a | `Extended Data Fig. 5/extended_data_fig. 5ab.ipynb` | `ir` (R) | `Phytomni-PaperYear-for_plot.csv` ✓ | `run` | `extended_data_fig.5a.pdf` |
| Ext. Data 5b | `Extended Data Fig. 5/extended_data_fig. 5ab.ipynb` | `ir` (R) | `Phytomni-PaperYear-for_plot.csv` ✓ | `run` | `extended_data_fig.5b.pdf` |
| Ext. Data 5c | `Extended Data Fig. 5/extended_data_fig. 5c.R` | Rscript | `Phytomni-Multiomics-for_plot.txt` ✓ | `run` | `extended_data_fig.5c.png` |
| Ext. Data 5d | `Extended Data Fig. 5/extended_data_fig. 5d.ipynb` | `python3` | *inline* | `run` | `extended_data_fig.5d.pdf` |
| Ext. Data 6a | `Extended Data Fig. 6-7/extended_data_fig. 6ab-7ab.ipynb` | `ir` (R) | *inline* | `run` | `extended_data_fig.6a.pdf` |
| Ext. Data 6b | `Extended Data Fig. 6-7/extended_data_fig. 6ab-7ab.ipynb` | `ir` (R) | *inline* | `run` | `extended_data_fig.6b.pdf` |
| Ext. Data 6d,e | `Extended Data Fig. 6-7/extended_data_fig. 6de-7de.ipynb` | `python3` | *inline* | `run` | `extended_data_fig.6d.pdf` / `extended_data_fig.6e.pdf` |
| Ext. Data 7a | `Extended Data Fig. 6-7/extended_data_fig. 6ab-7ab.ipynb` | `ir` (R) | *inline* | `run` | `extended_data_fig.7a.pdf` |
| Ext. Data 7b | `Extended Data Fig. 6-7/extended_data_fig. 6ab-7ab.ipynb` | `ir` (R) | *inline* | `run` | `extended_data_fig.7b.pdf` |
| Ext. Data 7d,e | `Extended Data Fig. 6-7/extended_data_fig. 6de-7de.ipynb` | `python3` | *inline* | `run` | `extended_data_fig.7d.pdf` / `extended_data_fig.7e.pdf` |
| Ext. Data 7c | `Extended Data Fig. 6-7/extended_data_fig. 7c.ipynb` | `python3` | *inline* | `run` | `extended_data_fig.7c.pdf` |
| Supp. 1 | `Supplementary Fig. 1/supplementary_fig. 1.ipynb` | `python3` | `Phyto-Chatbot-Pretrain.loss.json` ✓, … (4 files) | `run` | `supplementary_fig.1a.chatbot_pretrain_loss.line.pdf` / `supplementary_fig.1a.chatbot_pretrain_loss.line.png` / `supplementary_fig.1b.reasoner_pretrain_loss.line.pdf` / … (8 files) |
| Supp. 6 | `Supplementary Fig. 6/supplementary_fig. 6.ipynb` | `ir` (R) | *inline* | `run` | `supplementary_fig.6a.pdf` / `supplementary_fig.6b.pdf` |
| Supp. 7 | `Supplementary Fig. 7/supplementary_fig. 7.py` | `python3` (script) | `PhytoBench-Data-for_plot.xlsx` ✓ | `run` | `model_accuracy_by_species.pdf` |
| Supp. 12-15 | `Supplementary Fig. 12-15/supplementary_fig. 12-15.ipynb` | `python3` | `rank_distribution.tsv` ✓, … (13 files) | `run` | `supplementary_fig.12.phytobench-gene.well_studied.percent.bar.pdf` / `supplementary_fig.12.phytobench-gene.well_studied.percent.bar.png` / `supplementary_fig.12.phytobench-gene.well_studied.rice.percent.bar.pdf` / … (104 files) |
| Supp. 16 | `Supplementary Fig. 16/supplementary_fig. 16.ipynb` | `python3` | *inline* | `run` | `supplementary_fig.16a.model.paperbench-mp.line.pdf` / `supplementary_fig.16a.model.paperbench-mp.line.png` / `supplementary_fig.16b.model.paperbench-as.line.pdf` / … (10 files) |
| Supp. 19 | `Supplementary Fig. 19/supplementary_fig. 19.ipynb` | `ir` (R) | `SeedBench-for_plot.tsv` ✓, … (4 files) | `run` | `supplementary_fig.19.pdf` / `supplementary_fig.19b.pdf` |
| Supp. 20 | `Supplementary Fig. 20/supplementary_fig. 20.ipynb` | `python3` | *inline* | `run` | `supplementary_fig.20.pdf` |
| Supp. 21 | `Supplementary Fig. 21/supplementary_fig. 21.ipynb` | `python3` | *inline* | `run` | `supplementary_fig.21.pdf` |
| Supp. 26 | `Supplementary Fig. 26/supplementary_fig. 26.ipynb` | `python3` | *inline* | `run` | `supplementary_fig.26.phytobench-review.polar.pdf` / `supplementary_fig.26.phytobench-review.polar.png` |
<!-- END:REPRODUCE_MATRIX -->

## Run evidence and verification

Each invocation has a unique `logs/reproduce/<run-id>/` directory:

- `shareable/summary.json`: target and execution-unit coverage, timestamps,
  status/reason codes, source/input/artifact hashes, environment and publication state.
- `shareable/targets/<target-id>.log`: allowlisted status summaries.
- `shareable/artifacts/`: only this run's validated, selected, published figures.
- `work/`, `inputs/`, `raw/`, and `backups/`: local executed copies,
  diagnostic output and overwritten-file recovery evidence. Do not publish these
  directories wholesale; raw output may contain local paths or sensitive diagnostics.

Existing output files are not proof of a successful run. Validation decodes fresh
images/PDFs; visual inspection and scientific comparisons remain separate checks.
If publication encounters a user edit, it reports a conflict and retains evidence
instead of overwriting the edit during rollback. Publication is not a global
multi-file crash-atomic transaction. Clean up only completed run directories after
checking their reports and backups; never delete another active run's work.

Run the public fast gate and, after R preparation, the required R runtime checks:

```bash
python -m pytest tests/ -m 'not r_runtime' -q
python -m pytest tests/ -m r_runtime -q
```

The private-lineage test is explicitly opt-in and is not required to render public
figures. Local checks do not establish hosted CI, external service availability,
full bootstrap convergence, or an externally published release.

## Scope and author-owned reconciliation

The matrix covers currently supplied plotting sources, not every manuscript panel.
Ext. Data 5a/b share the year CSV and inline disciplinary counts; either selection
requires the CSV because the source executes as a whole. Ext. Data 6a/b and 7a/b
share the current benchmark/radar source; 6d/e and 7d/e share the AnalystAgent source.

Ext. Data 6c has no current plotting source. Some supplied AnalystAgent inline
totals differ from the manuscript's rounded overview. Ext. Data 5d category counts
sum to 160 while its existing count-label formula scales by 167. These scientific
discrepancies need author reconciliation; structural cleanup does not replace data
or recover a retired source to fill a gap.

Fig. 2a retains the manuscript-aligned Phyto-Chatbot values 0.720 and 0.084,
with both original workbook and aligned values recorded in provenance. Corrected
source workbooks are needed before a separate scientific update. See the
[Fig. 2 data dictionary](<Fig. 2/DATA_DICTIONARY.md>) and
[ranking data dictionary](<Supplementary Fig. 12-15/PhytoBench-Gene-for_plot/DATA_DICTIONARY.md>)
for units, denominators, scope, missingness and intervals.

Legacy four-model ranking tables remain historical evidence, not active plotting
inputs. Their removal and publication/archive obligations require an explicit
provenance decision. Software citation authors, preferred version and citation
metadata likewise require confirmation; no DOI or data relicensing is inferred.

A local release candidate should contain only verified current-run figures,
shareable status evidence, individually audited executed sources, and their public
input/manifest/lock provenance. Audit executed notebook outputs before sharing.
Do not include private inputs, raw logs, backups, local plans or credentials.
Local candidate preparation is not publication.

## Agent evaluation (not a figure)

### AnalystAgent Evaluation

[AnalystAgent Evaluation/README.md](<AnalystAgent Evaluation/README.md>):
external Phytomni-Bot runtime, downloaded benchmark data and live agent backend.

### DataAgent Evaluation

[DataAgent Evaluation/README.md](<DataAgent Evaluation/README.md>):
rewrite/NL2SQL pipeline, absent prompt template, private service configuration
and separate runtime dependencies.

### Knowledge&ReviewAgent Evaluation

[Knowledge&ReviewAgent Evaluation/README.md](<Knowledge&ReviewAgent Evaluation/README.md>):
identification, trace and review scorers; parser-only help is offline, but
model-based scoring has additional dependencies and download requirements.

### DeepGenomeAgent Evaluation

[DeepGenomeAgent Evaluation/README.md](<DeepGenomeAgent Evaluation/README.md>):
ranking/hallucination notebooks, private evaluation inputs, provider configuration,
and controlled publication updates. Public frozen rendering and independent
ranking recomputation do not require those private services.
