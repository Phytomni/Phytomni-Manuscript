# Repository Reproducibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a fresh clone produce an honest, auditable figure-reproduction result via a manifest-driven runner (✓/✘/⊘), tightened conda + renv locks, eval probes, and strict non-empty artifact checks.

**Architecture:** Single source of truth `reproduce.manifest.yaml` drives `reproduce.sh` (via a small Python library), the README reproduction matrix (generated + CI drift check), and dual-leg CI. Figure targets execute under `PHYTOMNI_SAVE=1`; missing data/toolchain → ⊘; successful exit without expected non-empty artifacts → ✘. Eval harnesses are probe-only.

**Tech Stack:** Bash `reproduce.sh`, Python 3.12+ (`PyYAML`, `pytest` for runner tests), existing Jupyter/`nbconvert`, R/`renv`, conda `environment.yml`, `uv.lock`, GitHub Actions.

**Spec:** `.cursor/specs/2026-07-09-reproducibility-design.md`

## Global Constraints

- No Docker.
- Do not run private eval backends or fill `Change_to_your_*` placeholders.
- Do not add eval-only deps (`langchain`, `openai`, …) to root `pyproject.toml`.
- Keep mandatory pins: `plotly==6.0.1`, `kaleido==0.2.1`.
- Quote all paths that contain spaces or literal dots.
- Figure saves remain gated on `PHYTOMNI_SAVE=1` → each figure’s `output/`.
- Pending CSVs (`Phytomni-DocType-for_plot.csv`, `PhytoBench-RAG-for_plot.csv`) stay `skip_until_data` until they land; do not block phases 1–4.
- Plans/specs live under `.cursor/` (user preference).
- Prefer TDD for Python library code; frequent commits per task.

## File structure (locked in)

| Path | Responsibility |
|---|---|
| `reproduce.manifest.yaml` | SSOT: all figure + eval targets |
| `scripts/reproduce_lib.py` | Load/validate manifest; skip reasons; run helpers; artifact checks; eval probes |
| `scripts/render_reproduce_matrix.py` | Render README matrix markdown from manifest |
| `reproduce.sh` | Thin CLI: `--check`, set `PHYTOMNI_SAVE`, call lib, print summary, exit codes |
| `tests/test_reproduce_lib.py` | Unit tests for lib (skip/artifact/probe/matrix) |
| `logs/` | Per-target logs (gitignored) |
| `environment.yml` | Tightened conda pins |
| `renv.lock` + `renv/` + `.Rprofile` | R fingerprint |
| `README.md` | Shortest path, renv section, eval honesty, generated matrix |
| `.github/workflows/reproduce.yml` | Dual legs + matrix drift check |
| `.gitignore` | Add `/logs/` |
| `pyproject.toml` / `uv.lock` | Add `pyyaml`, `pytest` (dev) |

**Orphan figure policy (discovered during planning):**

- `Supplementary Fig. 1` — already `PHYTOMNI_SAVE`-gated; onboard as `run`.
- `Supplementary Fig. 6`, `Supplementary Fig. 17`, `extended_data_fig.6ab.ipynb` — kernel `ir_r_env` (not `ir`), inline data, **no** `PHYTOMNI_SAVE` / `output/` saves yet.
- `extended_data_fig.6ab.ipynb` — alternate inline draft overlapping ED Fig. 6 panels already covered by `extended_data_fig. 6abc.ipynb` (xlsx) + radar Rmd → mark **`deprecated`** in manifest + README note (do not execute).
- Supp. 6 / 17 → after gating + kernel fix to `ir`, mark `run`.

**ED Fig. 6c:** notebook only saves `6c.pdf` if `exists("p")`. Required artifacts for the 6a–c target are **only** `6a.pdf` and `6b.pdf`.

---

### Task 1: Reproduce library skeleton + manifest schema tests

**Files:**
- Create: `scripts/reproduce_lib.py`
- Create: `tests/test_reproduce_lib.py`
- Create: `reproduce.manifest.yaml` (minimal stub with 2–3 targets for tests)
- Modify: `pyproject.toml` (add `pyyaml`; optional `[dependency-groups] dev = ["pytest"]` or `[project.optional-dependencies] dev`)
- Modify: `.gitignore` (add `/logs/`)

**Interfaces:**
- Produces:
  - `load_manifest(path: str | Path) -> dict`
  - `iter_targets(manifest: dict, phase: str | None = None) -> list[dict]`
  - `Target` fields used everywhere: `id`, `label`, `phase`, `kind`, `path`, `workdir`, `kernel`, `requires_data`, `requires_toolchain`, `expected_artifacts`, `status`, `probes`
  - Raises `ManifestError` on invalid schema

- [ ] **Step 1: Add PyYAML + pytest to the project**

In `pyproject.toml`, add `"pyyaml>=6.0"` to `dependencies`, and:

```toml
[dependency-groups]
dev = ["pytest>=8.0"]
```

(If the installed `uv` rejects `dependency-groups`, use `[project.optional-dependencies] dev = ["pytest>=8.0"]` instead and document the install command as `uv sync --extra dev`.)

- [ ] **Step 2: Write failing tests for manifest loading**

```python
# tests/test_reproduce_lib.py
from pathlib import Path
import pytest

from scripts.reproduce_lib import ManifestError, iter_targets, load_manifest

FIXTURE = Path(__file__).parent / "fixtures" / "mini.manifest.yaml"


def test_load_manifest_requires_targets(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("version: 1\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="targets"):
        load_manifest(p)


def test_iter_targets_filters_phase(tmp_path: Path):
    p = tmp_path / "m.yaml"
    p.write_text(
        """
version: 1
targets:
  - id: fig-2
    label: Fig. 2
    phase: figure
    kind: notebook
    path: "Fig. 2/fig. 2.ipynb"
    kernel: py
    status: run
    expected_artifacts: ["Fig. 2/output/fig.2a.phytobench-knowledge.bar.pdf"]
  - id: eval-analyst
    label: AnalystAgent
    phase: eval
    kind: eval_probe
    status: run
    probes: []
""",
        encoding="utf-8",
    )
    m = load_manifest(p)
    figs = iter_targets(m, phase="figure")
    assert [t["id"] for t in figs] == ["fig-2"]
```

- [ ] **Step 3: Run tests — expect fail**

Run: `uv sync && uv run pytest tests/test_reproduce_lib.py -v`  
Expected: FAIL (`ModuleNotFoundError: scripts.reproduce_lib` or import error)

- [ ] **Step 4: Implement minimal `scripts/reproduce_lib.py`**

```python
# scripts/reproduce_lib.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_TARGET_KEYS = {"id", "label", "phase", "kind", "status"}


class ManifestError(ValueError):
    pass


def load_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "targets" not in data:
        raise ManifestError("manifest must be a mapping with top-level 'targets'")
    if not isinstance(data["targets"], list):
        raise ManifestError("'targets' must be a list")
    for i, t in enumerate(data["targets"]):
        if not isinstance(t, dict):
            raise ManifestError(f"targets[{i}] must be a mapping")
        missing = REQUIRED_TARGET_KEYS - set(t)
        if missing:
            raise ManifestError(f"targets[{i}] missing keys: {sorted(missing)}")
        t.setdefault("requires_data", [])
        t.setdefault("requires_toolchain", [])
        t.setdefault("expected_artifacts", [])
        t.setdefault("probes", [])
        t.setdefault("workdir", None)
        t.setdefault("kernel", "none")
        t.setdefault("path", None)
    return data


def iter_targets(manifest: dict[str, Any], phase: str | None = None) -> list[dict[str, Any]]:
    out = []
    for t in manifest["targets"]:
        if phase is not None and t.get("phase") != phase:
            continue
        out.append(t)
    return out
```

Add empty `scripts/__init__.py` **or** ensure tests import via `PYTHONPATH=.` / `uv run --directory . pytest` with:

```python
# conftest or pytest.ini
# pytest.ini
[pytest]
pythonpath = .
```

Create `pytest.ini` at repo root with `pythonpath = .`.

- [ ] **Step 5: Run tests — expect pass**

Run: `uv run pytest tests/test_reproduce_lib.py -v`  
Expected: PASS

- [ ] **Step 6: Gitignore logs + commit**

Append to `.gitignore`:

```
# reproduce.sh per-target logs
/logs/
```

```bash
git add pyproject.toml uv.lock pytest.ini scripts/reproduce_lib.py tests/test_reproduce_lib.py .gitignore
git commit -m "feat: add reproduce_lib manifest loader and tests"
```

---

### Task 2: Skip reasons + artifact validation helpers

**Files:**
- Modify: `scripts/reproduce_lib.py`
- Modify: `tests/test_reproduce_lib.py`

**Interfaces:**
- Consumes: `load_manifest`, target dicts from Task 1
- Produces:
  - `check_skip(target: dict, repo_root: Path, *, have_ir: bool, have_rscript: bool, have_ggradar: bool) -> str | None`  
    Returns human reason string if should ⊘, else `None`
  - `validate_artifacts(target: dict, repo_root: Path) -> list[str]`  
    Returns list of problem strings (`missing: …`, `empty: …`); empty list means OK
  - Status handling: `status == "skip_until_data"` → skip with reason mentioning pending data; `status == "deprecated"` → skip with deprecated reason; never execute deprecated

- [ ] **Step 1: Write failing tests**

```python
from scripts.reproduce_lib import check_skip, validate_artifacts


def test_skip_until_data_status(tmp_path: Path):
    t = {
        "id": "ext-data-5b",
        "label": "Ext. Data 5b",
        "phase": "figure",
        "kind": "notebook",
        "status": "skip_until_data",
        "path": "Extended Data Fig. 5/extended_data_fig. 5ab.ipynb",
        "requires_data": ["Extended Data Fig. 5/Phytomni-DocType-for_plot.csv"],
        "requires_toolchain": [],
        "expected_artifacts": [],
    }
    reason = check_skip(t, tmp_path, have_ir=True, have_rscript=True, have_ggradar=True)
    assert reason is not None
    assert "skip_until_data" in reason or "pending" in reason.lower() or "DocType" in reason


def test_skip_missing_data_file(tmp_path: Path):
    t = {
        "id": "fig-3",
        "label": "Fig. 3",
        "phase": "figure",
        "kind": "notebook",
        "status": "run",
        "requires_data": ["Fig. 3/PhytoBench-Paper-for_plot.xlsx"],
        "requires_toolchain": [],
        "expected_artifacts": [],
    }
    assert check_skip(t, tmp_path, have_ir=True, have_rscript=True, have_ggradar=True)


def test_validate_artifacts_missing_and_empty(tmp_path: Path):
    out = tmp_path / "Fig. 2" / "output"
    out.mkdir(parents=True)
    empty = out / "empty.pdf"
    empty.write_bytes(b"")
    good = out / "good.pdf"
    good.write_bytes(b"%PDF")
    t = {
        "expected_artifacts": [
            "Fig. 2/output/missing.pdf",
            "Fig. 2/output/empty.pdf",
            "Fig. 2/output/good.pdf",
        ]
    }
    problems = validate_artifacts(t, tmp_path)
    assert any("missing.pdf" in p for p in problems)
    assert any("empty.pdf" in p for p in problems)
    assert not any("good.pdf" in p for p in problems)
```

- [ ] **Step 2: Run tests — expect fail**

Run: `uv run pytest tests/test_reproduce_lib.py::test_skip_until_data_status tests/test_reproduce_lib.py::test_validate_artifacts_missing_and_empty -v`  
Expected: FAIL (functions not defined)

- [ ] **Step 3: Implement helpers**

```python
def check_skip(
    target: dict[str, Any],
    repo_root: Path,
    *,
    have_ir: bool,
    have_rscript: bool,
    have_ggradar: bool,
) -> str | None:
    status = target.get("status", "run")
    if status == "deprecated":
        return "deprecated: not executed"
    if status == "skip_until_data":
        missing = [p for p in target.get("requires_data", []) if not (repo_root / p).is_file()]
        if missing:
            return f"data pending: {missing[0]}"
        # if data somehow present, fall through to normal checks
    for rel in target.get("requires_data", []):
        if not (repo_root / rel).is_file():
            return f"data missing: {rel}"
    tools = set(target.get("requires_toolchain", []))
    if "ir" in tools and not have_ir:
        return 'ir kernel missing: R -e "IRkernel::installspec()"'
    if "Rscript" in tools and not have_rscript:
        return "Rscript not found"
    if "ggradar" in tools and not have_ggradar:
        return "ggradar missing: remotes::install_github('ricardo-bion/ggradar')"
    return None


def validate_artifacts(target: dict[str, Any], repo_root: Path) -> list[str]:
    problems: list[str] = []
    for rel in target.get("expected_artifacts", []):
        path = repo_root / rel
        if not path.is_file():
            problems.append(f"missing artifact: {rel}")
        elif path.stat().st_size <= 0:
            problems.append(f"empty artifact: {rel}")
    return problems
```

Note: For `skip_until_data`, **always** ⊘ until status is flipped to `run` in the manifest (even if a local file appears early). Simpler rule matching the spec:

```python
    if status == "skip_until_data":
        return "skip_until_data (pending author data)"
```

Use this simpler rule in the implementation (update the first test’s assertion accordingly: `assert "skip_until_data" in reason`).

- [ ] **Step 4: Run tests — expect pass**

Run: `uv run pytest tests/test_reproduce_lib.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/reproduce_lib.py tests/test_reproduce_lib.py
git commit -m "feat: add skip reasons and artifact validation helpers"
```

---

### Task 3: Full figure manifest (current runnable set + honest 5a/5b)

**Files:**
- Create/overwrite: `reproduce.manifest.yaml`
- Modify: `tests/test_reproduce_lib.py` (load real manifest smoke test)

**Interfaces:**
- Produces: complete `phase: figure` entries for every target currently in `reproduce.sh`, with 5a/5b split and radar `skip_until_data`

**Artifact path convention:** all `expected_artifacts` are repo-root-relative (`"Fig. 2/output/….pdf"`). Prefer **PDF** as the required artifact (PNG may also exist; do not require PNG unless a target only writes PNG — e.g. Ext. Data 5c).

- [ ] **Step 1: Write `reproduce.manifest.yaml` with these figure targets**

Use this inventory (ids stable):

| id | label | kind | path | workdir | kernel | toolchain | data | status | expected_artifacts (PDF unless noted) |
|---|---|---|---|---|---|---|---|---|---|
| `fig-2` | Fig. 2 | notebook | `Fig. 2/fig. 2.ipynb` | — | py | — | — | run | `fig.2a…bar.pdf`, `fig.2b…bar.pdf`, `fig.2c…violin.pdf`, `fig.2g…well_studied.bar.pdf`, `fig.2h…uncharacterized.bar.pdf` under `Fig. 2/output/` |
| `fig-3` | Fig. 3 | notebook | `Fig. 3/fig. 3.ipynb` | — | py | — | `Fig. 3/PhytoBench-Paper-for_plot.xlsx` | run | `fig.3d…line.pdf`, `fig.3e…heatmap.pdf` |
| `ext-data-5a` | Ext. Data 5a | notebook | `Extended Data Fig. 5/extended_data_fig. 5ab.ipynb` | — | ir | ir | `…/Phytomni-PaperYear-for_plot.csv` | run | `…/output/extended_data_fig.5a.pdf` |
| `ext-data-5b` | Ext. Data 5b | notebook | same as 5a | — | ir | ir | `…/Phytomni-DocType-for_plot.csv` | **skip_until_data** | `…/output/extended_data_fig.5b.pdf` |
| `ext-data-5c` | Ext. Data 5c | rscript | `Extended Data Fig. 5/extended_data_fig. 5c.R` | `Extended Data Fig. 5` | none | Rscript | `…/Phytomni-Multiomics-for_plot.txt` | run | `…/output/extended_data_fig.5c.png` |
| `ext-data-5d` | Ext. Data 5d | notebook | `…/extended_data_fig. 5d.ipynb` | — | py | — | — | run | `…/output/extended_data_fig.5d.pdf` |
| `ext-data-6abc` | Ext. Data 6a-c | notebook | `…/extended_data_fig. 6abc.ipynb` | — | ir | ir | `…/PhytoBench-Knowledge-for_plot.xlsx` | run | **`6a.pdf` + `6b.pdf` only** (not 6c) |
| `ext-data-6-radar` | Ext. Data 6 radar | rmd | `…/extended_data_fig. 6abc.Rmd` | — | none | Rscript, ggradar | `…/PhytoBench-RAG-for_plot.csv` | **skip_until_data** | four files `extended_data_fig.6-radar-{figure1,figure2,figure3,figure4}.pdf` |
| `ext-data-6de` | Ext. Data 6d,e | notebook | `…/6de.ipynb` | — | py | — | — | run | `6d.pdf`, `6e.pdf` |
| `ext-data-6fg` | Ext. Data 6f,g | notebook | `…/6fg.ipynb` | — | py | — | — | run | `model_compare_agent_total.pdf`, `model_compare_agent_total_across_speciesv1.pdf` |
| `ext-data-7` | Ext. Data 7 | notebook | `…/extended_data_fig. 7.ipynb` | — | ir | ir | — | run | `extended_data_fig.7.pdf` |
| `supp-7` | Supp. 7 | py_script | `Supplementary Fig. 7/supplementary_fig. 7.py` | `Supplementary Fig. 7` | none | — | `…/PhytoBench-Data-for_plot.xlsx` | run | `model_accuracy_by_species.pdf` |
| `supp-8` | Supp. 8 | notebook | `…/supplementary_fig. 8.ipynb` | — | py | — | — | run | `model_compare_agent_split.pdf` |
| `supp-9` | Supp. 9 | notebook | `…/supplementary_fig. 9.ipynb` | — | py | — | — | run | `model_compare_agent_split_across_speciesv1.pdf` |
| `supp-10-13` | Supp. 10-13 | notebook | `…/supplementary_fig. 10-13.ipynb` | — | py | — | dir `PhytoBench-Gene-for_plot/` (list representative `requires_data` files that must exist, e.g. overall score tsvs already in repo) | run | Require the three non-loop PDFs (`fig.2d…`, `fig.2e…`, `fig.2f…`) **plus** at least the aggregate loop outputs for `well_studied` and `uncharacterized` percent/prob/score PDFs (6 files). Full 17×3 expansion is allowed but not required if CI time/noise is a concern — **minimum bar: the 3 non-loop PDFs + `well_studied` + `uncharacterized` percent bars**. Document the choice in a YAML comment. |
| `supp-14` | Supp. 14 | notebook | `…/14.ipynb` | — | py | — | — | run | five PDFs `supplementary_fig.6{a–e}.model.paperbench-{mp,as,cr,ng,pc}.line.pdf` |
| `supp-19` | Supp. 19 | notebook | `…/19.ipynb` | — | py | — | — | run | `supplementary_fig.19.pdf` |
| `supp-24` | Supp. 24 | notebook | `…/24.ipynb` | — | py | — | — | run | `supplementary_fig.13.phytobench-review.polar.pdf` |

Exact Fig. 2 PDF basenames (after `.lower().replace(' ', '_')`):

- `fig.2a.phytobench-knowledge.bar.pdf`
- `fig.2b.phytobench-data.bar.pdf`
- `fig.2c.phytobench-analysis.violin.pdf`
- `fig.2g.phytobench-gene.well_studied.bar.pdf`
- `fig.2h.phytobench-gene.uncharacterized.bar.pdf`

Also add placeholder stubs (can be empty probes) for later tasks:

```yaml
  - id: supp-1
    label: Supp. 1
    phase: figure
    kind: notebook
    path: "Supplementary Fig. 1/supplementary_fig. 1.ipynb"
    kernel: py
    status: run
    requires_data:
      - "Supplementary Fig. 1/Model-Loss-for_plot/Phyto-Chatbot-Pretrain.loss.json"
      - "Supplementary Fig. 1/Model-Loss-for_plot/Phyto-Chatbot-SFT.loss.json"
      - "Supplementary Fig. 1/Model-Loss-for_plot/Phyto-Reasoner-Pretrain.loss.json"
      - "Supplementary Fig. 1/Model-Loss-for_plot/Phyto-Reasoner-SFT.loss.json"
    expected_artifacts:
      - "Supplementary Fig. 1/output/supplementary_fig.1a.chatbot_pretrain_loss.line.pdf"
      - "Supplementary Fig. 1/output/supplementary_fig.1b.reasoner_pretrain_loss.line.pdf"
      - "Supplementary Fig. 1/output/supplementary_fig.1c.chatbot_sft_loss.line.pdf"
      - "Supplementary Fig. 1/output/supplementary_fig.1d.reasoner_sft_loss.line.pdf"
```

Leave Supp. 6 / 17 / deprecated 6ab for Task 5 (or add them now as `status: skip_until_data` / `deprecated` so the manifest already lists them).

- [ ] **Step 2: Add smoke test**

```python
def test_real_manifest_loads():
    root = Path(__file__).resolve().parents[1]
    m = load_manifest(root / "reproduce.manifest.yaml")
    ids = [t["id"] for t in iter_targets(m)]
    assert "fig-2" in ids
    assert "ext-data-5a" in ids
    assert "ext-data-5b" in ids
    five_b = next(t for t in m["targets"] if t["id"] == "ext-data-5b")
    assert five_b["status"] == "skip_until_data"
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_reproduce_lib.py -v`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add reproduce.manifest.yaml tests/test_reproduce_lib.py
git commit -m "feat: add full figure reproduce.manifest.yaml with honest 5a/5b split"
```

---

### Task 4: Wire `reproduce.sh` to the manifest (execute + logs + --check)

**Files:**
- Modify: `reproduce.sh` (replace hardcoded target list)
- Modify: `scripts/reproduce_lib.py` (add `run_figure_target`, `detect_toolchain`, `main`-style API)
- Modify: `tests/test_reproduce_lib.py` (unit-test command construction / dry-run if feasible)

**Interfaces:**
- Produces:
  - `detect_toolchain() -> dict` with `have_ir`, `have_rscript`, `have_ggradar`
  - `run_figure_target(target, repo_root, log_path) -> int` exit code
  - `reproduce_main(argv: list[str], repo_root: Path) -> int` used by CLI

**Execution rules:**

| kind | Command |
|---|---|
| `notebook` | `jupyter nbconvert --to notebook --execute --inplace "<path>"` |
| `rscript` | `cd "<workdir>" && Rscript "<basename>"` |
| `py_script` | `cd "<workdir>" && python3 "<basename>"` |
| `rmd` | `Rscript -e 'rmarkdown::render("<path>")'` from repo root |

- Shared notebook for 5a/5b: **execute the notebook only once per run** when either target would run. Implementation: in `reproduce_main`, group by `(kind, path)` for notebooks; if `ext-data-5a` is runnable, run once; then validate **each** target’s own `expected_artifacts` separately. If only 5b is somehow `run` later, same path. While 5b is `skip_until_data`, only 5a’s artifacts are checked after the shared run.
- Always `export PHYTOMNI_SAVE=1`.
- Logs: `logs/<id>.log` (for shared notebook run, write the same log to `logs/ext-data-5a.log`; 5b skipped gets no run log or a one-line skip log).
- Summary marks: ✓ / ✘ / ⊘ matching current script.
- `--check`: exit `1` if any **executed** figure target is ✘; ⊘ does not fail.

- [ ] **Step 1: Implement `run_figure_target` + `reproduce_main` in `reproduce_lib.py`**

Sketch (complete in implementation):

```python
import os
import subprocess
import shutil


def detect_toolchain() -> dict[str, bool]:
    have_rscript = shutil.which("Rscript") is not None
    have_ir = False
    try:
        out = subprocess.check_output(["jupyter", "kernelspec", "list"], text=True, stderr=subprocess.DEVNULL)
        have_ir = any(line.split()[0] == "ir" for line in out.splitlines() if line.strip())
    except (OSError, subprocess.CalledProcessError):
        have_ir = False
    have_ggradar = False
    if have_rscript:
        have_ggradar = (
            subprocess.call(
                ["Rscript", "-e", "if(!requireNamespace('ggradar', quietly=TRUE)) quit(status=1)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            == 0
        )
    return {"have_ir": have_ir, "have_rscript": have_rscript, "have_ggradar": have_ggradar}


def run_figure_target(target: dict[str, Any], repo_root: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    kind = target["kind"]
    path = target["path"]
    workdir = target.get("workdir")
    with log_path.open("w", encoding="utf-8") as log:
        if kind == "notebook":
            cmd = ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(repo_root / path)]
            cwd = repo_root
        elif kind == "rscript":
            cwd = repo_root / workdir
            cmd = ["Rscript", Path(path).name]
        elif kind == "py_script":
            cwd = repo_root / workdir
            cmd = ["python3", Path(path).name]
        elif kind == "rmd":
            cwd = repo_root
            render = f'rmarkdown::render("{path}")'
            cmd = ["Rscript", "-e", render]
        else:
            log.write(f"unknown kind: {kind}\n")
            return 2
        log.write(f"+ {' '.join(cmd)}\n")
        proc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        return proc.returncode
```

Implement `reproduce_main` to: load manifest → detect toolchain → for each figure target apply `check_skip` → dedupe notebook executions → validate artifacts → print lines → return exit code.

- [ ] **Step 2: Replace `reproduce.sh` body**

```bash
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
```

- [ ] **Step 3: Add `scripts/reproduce_cli.py`**

```python
# scripts/reproduce_cli.py
from __future__ import annotations
import sys
from pathlib import Path
from scripts.reproduce_lib import reproduce_main

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raise SystemExit(reproduce_main(sys.argv[1:], root))

if __name__ == "__main__":
    main()
```

Ensure `scripts` is a package (`scripts/__init__.py` empty).

- [ ] **Step 4: Smoke locally (figure subset)**

Run:

```bash
PHYTOMNI_SAVE=1 ./reproduce.sh --check
```

Expected: existing runnable targets ✓ or ✘ with logs under `logs/`; `ext-data-5b` and `ext-data-6-radar` show ⊘; exit 0 if no ✘.

If a previously green target now ✘ due to artifact list mismatch, fix the manifest names (do not weaken validation).

- [ ] **Step 5: Commit**

```bash
git add reproduce.sh scripts/reproduce_lib.py scripts/reproduce_cli.py scripts/__init__.py tests/
git commit -m "feat: manifest-driven reproduce.sh with logs and strict artifact checks"
```

---

### Task 5: Onboard Supp. 1 (already done in manifest) + gate Supp. 6/17 + deprecate 6ab

**Files:**
- Modify: `Supplementary Fig. 6/supplementary_fig.6.ipynb`
- Modify: `Supplementary Fig. 17/supplementary_fig.17.ipynb`
- Modify: `reproduce.manifest.yaml`
- Modify: `README.md` (short note on deprecated `extended_data_fig.6ab.ipynb`)

**Interfaces:**
- Consumes: runner from Task 4
- Produces: `supp-6`, `supp-17` as `run`; `ext-data-6ab` as `deprecated`

- [ ] **Step 1: Fix kernels to `ir`**

In both Supp. 6 and Supp. 17 notebook metadata, set:

```json
"kernelspec": {
  "name": "ir",
  "display_name": "R",
  "language": "R"
}
```

(Same for language_info as other R notebooks in this repo — copy from `extended_data_fig. 5ab.ipynb`.)

- [ ] **Step 2: Add `PHYTOMNI_SAVE` gating + `ggsave` into `output/`**

At the top of the first code cell in each notebook:

```r
save_figs <- Sys.getenv("PHYTOMNI_SAVE") == "1"
if (save_figs) dir.create("output", showWarnings = FALSE)
```

After each final plot object is built, save with stable names, e.g.:

**Supp. 6:**

- `output/supplementary_fig.6a.pdf`
- `output/supplementary_fig.6b.pdf`

**Supp. 17:**

- `output/supplementary_fig.17.pdf`

Pattern:

```r
if (save_figs) ggsave("output/supplementary_fig.6a.pdf", plot = p, width = 8, height = 6)
```

(Use the actual plot object names already in the notebook.)

Supp. 6b uses `ggradar` — add `ggradar` to that target’s `requires_toolchain`.

- [ ] **Step 3: Update manifest**

```yaml
  - id: supp-6
    label: Supp. 6
    phase: figure
    kind: notebook
    path: "Supplementary Fig. 6/supplementary_fig.6.ipynb"
    kernel: ir
    requires_toolchain: [ir, ggradar]
    status: run
    expected_artifacts:
      - "Supplementary Fig. 6/output/supplementary_fig.6a.pdf"
      - "Supplementary Fig. 6/output/supplementary_fig.6b.pdf"

  - id: supp-17
    label: Supp. 17
    phase: figure
    kind: notebook
    path: "Supplementary Fig. 17/supplementary_fig.17.ipynb"
    kernel: ir
    requires_toolchain: [ir]
    status: run
    expected_artifacts:
      - "Supplementary Fig. 17/output/supplementary_fig.17.pdf"

  - id: ext-data-6ab-deprecated
    label: Ext. Data 6ab (deprecated notebook)
    phase: figure
    kind: notebook
    path: "Extended Data Fig. 6/extended_data_fig.6ab.ipynb"
    status: deprecated
    expected_artifacts: []
```

- [ ] **Step 4: Execute the new targets**

```bash
PHYTOMNI_SAVE=1 ./reproduce.sh --check
```

Expected: `Supp. 6` / `Supp. 17` ✓ (if `ir` + `ggradar` present) or ⊘ with toolchain reason; deprecated line ⊘; no false green.

- [ ] **Step 5: Commit**

```bash
git add "Supplementary Fig. 6" "Supplementary Fig. 17" reproduce.manifest.yaml README.md
git commit -m "feat: onboard Supp. 6/17 with SAVE gating; deprecate orphan 6ab notebook"
```

---

### Task 6: Matrix renderer + README SSOT + CI drift check

**Files:**
- Create: `scripts/render_reproduce_matrix.py`
- Modify: `README.md` (mark matrix as generated; shortest path; pending-data; renv stub heading ok if Task 7 fills it)
- Modify: `.github/workflows/reproduce.yml`
- Modify: `tests/test_reproduce_lib.py`

**Interfaces:**
- Produces: `render_matrix(manifest) -> str` markdown table
- Markers in README:

```markdown
<!-- BEGIN:REPRODUCE_MATRIX -->
...generated...
<!-- END:REPRODUCE_MATRIX -->
```

- [ ] **Step 1: Failing test for renderer**

```python
from scripts.render_reproduce_matrix import render_matrix

def test_render_matrix_contains_ids():
    m = {
        "targets": [
            {
                "id": "fig-2",
                "label": "Fig. 2",
                "phase": "figure",
                "kind": "notebook",
                "path": "Fig. 2/fig. 2.ipynb",
                "kernel": "py",
                "status": "run",
                "requires_data": [],
                "expected_artifacts": ["Fig. 2/output/x.pdf"],
            }
        ]
    }
    md = render_matrix(m)
    assert "Fig. 2" in md
    assert "fig. 2.ipynb" in md
```

- [ ] **Step 2: Implement renderer + CLI**

`scripts/render_reproduce_matrix.py` should print the table to stdout, and support:

```bash
python -m scripts.render_reproduce_matrix --write-readme README.md
```

which replaces the block between the HTML markers.

Columns (keep close to current README): Figure | File | Kernel | Input data | Status | Emits |

- [ ] **Step 3: Insert markers into README and regenerate**

Replace the hand-maintained matrix section with markers + generated content. Add at top of Reproducing section:

> Authoritative target list: [`reproduce.manifest.yaml`](reproduce.manifest.yaml). The table below is generated; CI fails if it drifts.

- [ ] **Step 4: CI drift step**

In both jobs (or a tiny third job), after checkout:

```yaml
      - name: Check README matrix matches manifest
        run: |
          uv sync --frozen
          source .venv/bin/activate
          python -m scripts.render_reproduce_matrix --write-readme README.md
          git diff --exit-code README.md
```

For the conda job, use `python` from the activated env after ensuring PyYAML is installed (it will be once `environment.yml` / pip includes it — Task 7). Until then, run the drift check only on the **uv** leg.

- [ ] **Step 5: Commit**

```bash
git add scripts/render_reproduce_matrix.py README.md tests/ .github/workflows/reproduce.yml
git commit -m "feat: generate README reproduce matrix from manifest with CI drift check"
```

---

### Task 7: Tighten `environment.yml` + add `renv` + docs

**Files:**
- Modify: `environment.yml`
- Create: `renv.lock`, `.Rprofile`, `renv/activate.R` (via `renv::init`)
- Modify: `README.md` (R locking section; no Docker note)
- Modify: `.github/workflows/reproduce.yml` (optional `renv::restore()` attempt)
- Modify: `environment.yml` / conda pip list to include `pyyaml`

**Pin targets (align with current `uv.lock`):**

```yaml
  - pandas=3.0.2
  - matplotlib=3.10.9
  - numpy=2.4.4
  - seaborn=0.13.2
  - scipy=1.17.1   # if available on conda-forge at this version; otherwise pin closest and note
  - openpyxl=3.1.5
  - plotly=6.0.1
  - python-kaleido=0.2.1
  - nbformat>=5.10.4
  - jupyterlab>=4.5.7
  - pip:
      - colorlover>=0.3.0
      - pyyaml>=6.0
```

If exact conda builds are unavailable, pin the major.minor that resolves on `conda-forge` and document any drift vs `uv.lock` in README.

**renv steps (local machine with R):**

```r
install.packages("renv")
renv::init(bare = TRUE)
# ensure figure packages + ggradar present, then:
renv::snapshot()
```

Commit `renv.lock`, `.Rprofile`, and `renv/activate.R` (standard renv layout).

README section:

```markdown
### R version locking (renv)

`environment.yml` installs R packages via conda-forge. `renv.lock` records the exact versions used when figures were validated (including GitHub `ggradar`). After creating the conda (or system R) environment:

```r
renv::restore()
```

This repo does not ship a Docker image.
```

CI: after installing R packages + ggradar, try:

```bash
Rscript -e 'if (requireNamespace("renv", quietly=TRUE)) renv::restore(prompt=FALSE)'
```

If restore conflicts with conda library paths, keep the install-as-today path and leave a workflow comment + README note that `renv.lock` is the fingerprint for reviewers; do **not** fail CI solely because restore is awkward — but do commit the lockfile.

- [ ] **Step 1: Update `environment.yml` pins + pyyaml**
- [ ] **Step 2: Generate `renv.lock` on a machine with the figure R packages**
- [ ] **Step 3: Document in README**
- [ ] **Step 4: Adjust CI install notes**
- [ ] **Step 5: Commit**

```bash
git add environment.yml renv.lock .Rprofile renv README.md .github/workflows/reproduce.yml
git commit -m "chore: pin conda deps and add renv.lock for R fingerprinting"
```

---

### Task 8: Eval probes + README honesty block

**Files:**
- Modify: `reproduce.manifest.yaml` (four `eval_probe` targets)
- Modify: `scripts/reproduce_lib.py` (`run_eval_probes`)
- Modify: `tests/test_reproduce_lib.py`
- Modify: `README.md` (Agent evaluation section refresh)

**Interfaces:**
- Produces: `run_probe(probe: dict, repo_root: Path) -> str | None` (None = pass, else failure reason)
- Probe kinds:
  - `file_exists: { path: ... }`
  - `file_missing: { path: ... }` (useful to assert placeholder template absent → ⊘ reason)
  - `can_import: { module: ... }`
  - `file_contains: { path: ..., substring: "Change_to_your_" }` → if still present, probe fails (expected → overall ⊘)

**Eval outcome policy:** if any probe fails → target **⊘** with concatenated reasons (not ✘, not ✓). Eval ⊘ never fails `--check`. Only mark ✓ if every probe passes (rare on a bare clone).

Suggested targets:

```yaml
  - id: eval-analyst
    label: AnalystAgent Evaluation
    phase: eval
    kind: eval_probe
    status: run
    probes:
      - { type: can_import, module: mcp_server_phytomni }
      - { type: can_import, module: huggingface_hub }

  - id: eval-data
    label: DataAgent Evaluation
    phase: eval
    kind: eval_probe
    status: run
    probes:
      - { type: file_exists, path: "DataAgent Evaluation/src/prompt_template.yaml" }
      - { type: can_import, module: langchain }
      - { type: file_contains_none, path: "DataAgent Evaluation/src/model/config.py", substring: "Change_to_your_" }

  - id: eval-knowledge
    label: KnowledgeAgent Evaluation
    phase: eval
    kind: eval_probe
    status: run
    probes:
      - { type: file_exists, path: "KnowledgeAgent Evaluation/evaluation_id.py" }
      # bare clone has scripts but no required --input dataset in-repo → document as needing user data
      - { type: file_exists, path: "KnowledgeAgent Evaluation/INPUT_DATA_PLACEHOLDER" }  # intentional miss → ⊘

  - id: eval-expert
    label: Expert Evaluation
    phase: eval
    kind: eval_probe
    status: run
    probes:
      - { type: file_exists, path: "Expert Evaluation/score.tsv" }
```

Prefer **not** inventing fake placeholder paths: for Knowledge/Expert, probe for clearly missing inputs (`Expert Evaluation/score.tsv` is missing today → ⊘ “data missing: score.tsv”). For Knowledge, probe `can_import` of heavy deps only if imported at module level; otherwise document “requires `--input` dataset not shipped” via a probe type `always_fail` with message — **avoid `always_fail`**. Use:

```yaml
      - { type: note_requires, message: "CLI requires --input dataset not shipped in this repo" }
```

where `note_requires` always returns that message (counts as failed probe → ⊘). That is honest without fake files.

- [ ] **Step 1: Tests for probes**
- [ ] **Step 2: Implement + append eval summary at end of `reproduce_main`**
- [ ] **Step 3: README — four harnesses, figure-env cannot run them**
- [ ] **Step 4: `./reproduce.sh --check` shows eval ⊘ block, exit 0 if figures OK**
- [ ] **Step 5: Commit**

```bash
git add reproduce.manifest.yaml scripts/reproduce_lib.py tests/ README.md
git commit -m "feat: add honest eval probes and document non-reproducible harnesses"
```

---

### Task 9: End-to-end verification + AGENTS.md/CLAUDE.md touch-up

**Files:**
- Modify: `AGENTS.md` (pointer to manifest + new skip/artifact behavior; keep short)
- Modify: `.github/workflows/reproduce.yml` if any final gaps

- [ ] **Step 1: Run unit tests**

```bash
uv run pytest tests/ -v
```

Expected: PASS

- [ ] **Step 2: Full smoke**

```bash
PHYTOMNI_SAVE=1 ./reproduce.sh --check
```

Expected:
- Figures: ✓ or toolchain ⊘ only where appropriate
- `ext-data-5b`, `ext-data-6-radar`: ⊘ pending data
- `ext-data-6ab-deprecated`: ⊘ deprecated
- Evals: ⊘ with reasons
- Exit code 0 if no figure ✘
- `logs/` populated for executed targets
- `output/` PDFs non-empty for ✓ targets

- [ ] **Step 3: Confirm matrix drift check**

```bash
python -m scripts.render_reproduce_matrix --write-readme README.md
git diff --exit-code README.md
```

- [ ] **Step 4: Update `AGENTS.md` bullets** for manifest SSOT, honest 5a/5b, `logs/`, renv — without duplicating the whole README

- [ ] **Step 5: Final commit**

```bash
git add AGENTS.md .github/workflows/reproduce.yml
git commit -m "docs: align AGENTS.md with manifest-driven reproducibility"
```

---

### Task 10 (follow-up when CSVs arrive): Flip pending data targets

**Files:**
- Add: `Extended Data Fig. 5/Phytomni-DocType-for_plot.csv`
- Add: `Extended Data Fig. 6/PhytoBench-RAG-for_plot.csv`
- Modify: `reproduce.manifest.yaml` (`ext-data-5b`, `ext-data-6-radar` → `status: run`)
- Regenerate README matrix

- [ ] **Step 1: Drop CSVs into the paths above**
- [ ] **Step 2: Set both targets’ `status: run`**
- [ ] **Step 3: `./reproduce.sh --check` — expect 5b + radar ✓ with non-empty PDFs**
- [ ] **Step 4: Regenerate matrix + commit**

```bash
git add "Extended Data Fig. 5/Phytomni-DocType-for_plot.csv" \
        "Extended Data Fig. 6/PhytoBench-RAG-for_plot.csv" \
        reproduce.manifest.yaml README.md
git commit -m "feat: enable Ext. Data 5b and radar targets with shipped plot data"
```

---

## Spec coverage self-check

| Spec requirement | Task |
|---|---|
| Manifest SSOT | 1, 3 |
| Runner execute/skip/log | 4 |
| Honest 5a/5b | 3, 4 |
| Radar skip_until_data | 3 |
| Orphan Supp. 1/6/17 + deprecate 6ab | 3, 5 |
| Strict artifact existence+nonempty | 2, 4 |
| Matrix generate + CI drift | 6 |
| conda pins + renv + docs, no Docker | 7 |
| Eval probes + docs, no private backends | 8 |
| Reviewer shortest path / AGENTS | 6, 9 |
| Pending CSV follow-up | 10 |

## Placeholder / consistency scan

- No TBD left for schema field names: `id`, `label`, `phase`, `kind`, `path`, `workdir`, `kernel`, `requires_data`, `requires_toolchain`, `expected_artifacts`, `status`, `probes`.
- `check_skip` / `validate_artifacts` / `reproduce_main` names consistent across tasks.
- 6c not in required artifacts (matches notebook `exists("p")` guard).
- Eval failures are ⊘, never fail `--check`.
