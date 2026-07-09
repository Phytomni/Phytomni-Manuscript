from pathlib import Path

import pytest

from scripts.reproduce_lib import (
    ManifestError,
    check_skip,
    iter_targets,
    load_manifest,
    validate_artifacts,
)

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
    assert "skip_until_data" in reason


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
