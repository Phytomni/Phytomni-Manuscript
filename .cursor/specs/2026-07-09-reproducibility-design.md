# Design: Improve repository reproducibility

**Date:** 2026-07-09  
**Status:** Approved for implementation planning  
**Scope:** Figure reproduction + environment locking + eval honesty + reviewer UX  
**Out of scope (this effort):** Docker; running private eval backends; pixel/hash golden tests for PDFs; bundling eval-only Python deps into the figure environment

## Goal

Make a fresh clone produce an honest, auditable figure-reproduction result: every target is either ✓ (ran and emitted non-empty expected artifacts), ✘ (ran but failed or missing/empty artifacts), or ⊘ (skipped with a human-readable reason). Environment pins and R fingerprints are documented and lockable. Eval harnesses are probed and documented, never silently green.

## Decisions (from brainstorming)

| Topic | Decision |
|---|---|
| Priority order | A (figures) → B (env) → C (evals) → D (reviewer UX) |
| Missing data (`Phytomni-DocType-for_plot.csv`, `PhytoBench-RAG-for_plot.csv`) | Arrive in 2–3 days; mark `skip_until_data` until then |
| Architecture | **Manifest-driven (path 2)** — single source of truth; implement incrementally on top of existing `reproduce.sh` / CI |
| Environment | Tighten `environment.yml` pins + add `renv` + docs; **no Docker** |
| Evals | Document + honest probes (⊘ + reason); do not execute private backends |
| Artifact checks | Strict: existence + non-empty for every expected file on executed figure targets |
| Spec location | `.cursor/specs/` (user override of default `docs/superpowers/specs/`) |

## §1 Architecture

### Single source of truth

Add repository-root `reproduce.manifest.yaml`. Every runnable or skippable unit is one target. Consumers read only this file (no second hand-maintained matrix):

1. `reproduce.sh` — execute / skip / log / validate artifacts  
2. README reproduction matrix — generated from the manifest (generator script + CI drift check)  
3. CI (`.github/workflows/reproduce.yml`) — both conda and uv legs run the same runner

### Target schema (minimum fields)

| Field | Meaning |
|---|---|
| `id` | Stable machine id (e.g. `ext-data-5a`) |
| `label` | Human label in summary lines |
| `phase` | `figure` \| `eval` |
| `kind` | `notebook` \| `rscript` \| `py_script` \| `rmd` \| `eval_probe` |
| `path` | Notebook/script/Rmd path (quoted-safe; spaces/dots allowed) |
| `workdir` | Optional cwd for bare relative data paths |
| `kernel` | `py` \| `ir` \| `none` |
| `requires_data` | List of files; missing → ⊘ (not ✘) |
| `requires_toolchain` | e.g. `ir`, `Rscript`, `ggradar` |
| `expected_artifacts` | Paths relative to figure dir or repo root; checked only after a successful run |
| `status` | `run` \| `skip_until_data` \| `deprecated` |
| `probes` | (eval only) checks: file exists, importable module, placeholder still `Change_to_your_*`, etc. |

### Environment layer (orthogonal to runner)

| Layer | Role | Source of truth |
|---|---|---|
| Python (uv) | Bit-for-bit Python env | Existing `uv.lock` (`uv sync --frozen`) |
| Python + R (conda) | One-command install | `environment.yml` with tightened pins |
| R fingerprint | Exact package versions used when figures ran | New `renv.lock` (+ minimal renv bootstrap) |
| GitHub-only | `ggradar` | `remotes::install_github` in CI/docs; then captured in renv |

No Docker in this effort.

### Pending data

- Ext. Data 5b: `Phytomni-DocType-for_plot.csv` → `status: skip_until_data`  
- Ext. Data 6 radar Rmd: `PhytoBench-RAG-for_plot.csv` → `status: skip_until_data`  

When files land in-repo, flip those two targets to `run` (and fill `expected_artifacts` if not already listed). No runner redesign required.

## §2 Figure runner

### Commands

- `PHYTOMNI_SAVE=1 ./reproduce.sh` — run all `phase: figure` targets with `status: run` (and toolchain/data available); print ✓/✘/⊘ summary  
- `./reproduce.sh --check` — same; non-zero exit if any executed figure target is ✘ (⊘ does not fail the run)

Always set `PHYTOMNI_SAVE=1` inside the runner (current behavior). Paths with spaces/dots must be quoted.

### Logging

- Full stdout/stderr per target → `logs/<id>.log` (gitignored)  
- Terminal one-liner includes reason for ⊘/✘ (e.g. missing data, missing artifact name)

### Honest 5a / 5b split

- Two manifest entries sharing `Extended Data Fig. 5/extended_data_fig. 5ab.ipynb`  
- **5a:** requires `Phytomni-PaperYear-for_plot.csv`; expects `extended_data_fig.5a.pdf`  
- **5b:** `skip_until_data` until DocType CSV exists; expects `extended_data_fig.5b.pdf`  
- Keep notebook-internal `have_5b` guard so missing DocType does not abort 5a cells  
- Runner status is authoritative for CI/summary (do not report a bundled “5a,b ✓” when only 5a emitted)

### Radar Rmd

Migrate existing skip logic into the manifest: requires CSV + `Rscript` + `ggradar`; `skip_until_data` until CSV is present.

### Figures not yet in the runner

| Candidate | Rule |
|---|---|
| Supplementary Fig. 1 | Has `Model-Loss-for_plot/`; verify `PHYTOMNI_SAVE` gating; add as `run` with expected artifacts |
| Supplementary Fig. 6 / 17 | Probe whether they run without external data; `run` if yes, else `skip_until_data` or document blocker |
| `extended_data_fig.6ab.ipynb` | Clarify vs `6abc`: if duplicate → `deprecated` + README note; if distinct panel → own target |

New figure notebooks must follow existing `PHYTOMNI_SAVE` → `output/` gating before `status: run`.

### Unchanged conventions

- Figure dirs remain self-contained; relative data paths  
- kaleido `0.2.1` / plotly `6.0.1` pins remain mandatory  
- Do not treat eval directories as figure targets

## §3 Environment (conda pins + renv + docs)

### Division of responsibility

- **conda** = “can install” (Python + R packages from conda-forge, plus pip `colorlover`)  
- **renv** = “which versions” (fingerprint of R packages including `ggradar`)  
- **uv.lock** = Python reproducibility on the uv CI leg  

Reviewers’ default path remains `conda env create -f environment.yml` or `uv sync --frozen` (+ R install notes for uv). Use `renv::restore()` when an exact R fingerprint is required. renv is not required to replace conda for a full standalone R install.

### Concrete changes

1. Tighten pins in `environment.yml` for figure-critical packages (align sensibly with `uv.lock`; keep existing plotly/kaleido pins). Document that `colorlover` is pip-only under conda.  
2. Add `renv.lock` and minimal renv bootstrap (e.g. `.Rprofile` / renv config as needed). Snapshot CRAN figure deps + `ggradar`.  
3. CI: after R package install, prefer `renv::restore()` when compatible with the conda/uv-provided R library layout; if restore proves fragile in CI, document the fallback and still commit `renv.lock` as the fingerprint source of truth for local/reviewer use.  
4. README section: “R version locking (renv)” + explicit “no Docker in this repo (for now)”.

### Explicit non-goals

- No Docker image  
- No figure code changes solely to chase newer package APIs  
- No eval-only Python packages in root `pyproject.toml`

## §4 Eval probes (document + honest ⊘)

### Targets

| Directory | Behavior |
|---|---|
| `AnalystAgent Evaluation/` | Probe only (`mcp_server_phytomni`, HF/data layout, backend placeholders) — never submit live tasks |
| `DataAgent Evaluation/` | Probe only (`prompt_template.yaml`, deps, `Change_to_your_*` placeholders) |
| `KnowledgeAgent Evaluation/` | Probe only (scripts/inputs/config) |
| `Expert Evaluation/` | Probe only; do not require a green figure-like run in this effort |

### Manifest + runner

- `phase: eval`, `kind: eval_probe`  
- Outcomes: **⊘** + reason (expected for a bare clone), or **✓** only if all probe predicates pass (no private network smoke required in this effort)  
- Eval ⊘ must **not** fail `--check`  
- Eval must **never** be reported ✓ when probes fail  
- Print an eval summary block at the end of a normal `reproduce.sh` run (so reviewers see it without a separate flag)  
- Logs: `logs/eval-<id>.log`

### Documentation

README subsection per harness: runnable under figure env? / missing pieces / placeholder list. State clearly that evals are **not** part of manuscript figure reproduction.

### Explicit non-goals

- Do not add langchain/openai/etc. to the figure env manifests  
- Do not commit secrets or fill placeholders  
- Do not implement “minimal runnable eval subset in runner” (that was option C; this effort is B)

## §5 Artifact validation + reviewer path (strict)

### Rules

For each `phase: figure` target that **executed** (not ⊘):

1. Every path in `expected_artifacts` must exist and have size > 0.  
2. Missing or empty → target **✘**, even if the process exit code was 0.  
3. `--check` exits non-zero if any such ✘ occurred.  
4. ⊘ targets are not artifact-checked.  
5. Split targets (5a/5b) only require their own expected files.

Expected artifacts live **only** in the manifest (no parallel `expected-artifacts.txt`).

### Matrix generation

- Provide a small script (e.g. under `scripts/`) that renders the README reproduction matrix from the manifest.  
- README states that `reproduce.manifest.yaml` is authoritative.  
- CI runs the generator and fails on drift against the committed matrix section (or equivalent checked-in generated fragment).

### CI

- Keep dual legs (conda + uv).  
- Both run `reproduce.sh --check`.  
- Keep `upload-artifact` of `**/output/**`.  
- Validation happens inside the runner before upload.

### Reviewer shortest path

Document prominently:

```bash
conda env create -f environment.yml && conda activate phytomni-fig
# or: uv sync --frozen && source .venv/bin/activate  (+ R per README)
PHYTOMNI_SAVE=1 ./reproduce.sh --check
```

Success may include ⊘ lines with reasons. Failure points to `logs/<id>.log` and the missing/empty artifact name. Link out to renv, eval ⊘, and pending-data notes.

### Explicit non-goals

- No PDF pixel or hash golden comparisons  
- No Docker

## Implementation phasing (for the plan skill)

1. **Manifest + runner refactor** — encode current 17 figure targets; add logging; split 5a/5b; strict artifact checks; gitignore `logs/`  
2. **Onboard orphan figures** — Supp. 1 / 6 / 17 / `6ab` clarification  
3. **Environment** — pin `environment.yml`; add renv + docs; adjust CI install steps as needed  
4. **Eval probes + README** — four harnesses; matrix generator + drift check  
5. **Pending data follow-up** (when CSVs arrive) — flip `skip_until_data` → `run`; confirm artifacts in CI  

## Success criteria

- Bare clone + documented env + `./reproduce.sh --check` exits 0 with only expected ⊘ (pending data / eval probes / optional missing toolchain on uv-without-R setups as today).  
- No false green: exit 0 from a notebook without its expected non-empty PDF is ✘.  
- 5b and radar are ⊘ with explicit reasons until data lands; 5a can still ✓.  
- README matrix matches manifest (CI-enforced).  
- `environment.yml` pins + committed `renv.lock` + short docs; no Docker.  
- Eval section never claims figure-env reproducibility for private backends.

## Non-goals (recap)

- Docker  
- Live eval execution / secret materialization  
- Eval deps in figure `pyproject.toml`  
- Visual regression of figures  
- Waiting on the two CSVs before starting phases 1–4
