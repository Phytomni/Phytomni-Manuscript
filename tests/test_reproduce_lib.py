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
