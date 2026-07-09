from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.reproduce_lib import (
    ManifestError,
    _notebook_run_key,
    check_skip,
    detect_toolchain,
    iter_targets,
    load_manifest,
    reproduce_main,
    run_figure_target,
    validate_artifacts,
)


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
    reason = check_skip(t, tmp_path, have_ir=True, have_rscript=True, have_ggradar=True)
    assert reason is not None
    assert "Fig. 3/PhytoBench-Paper-for_plot.xlsx" in reason


def test_skip_deprecated_status(tmp_path: Path):
    t = {
        "id": "ext-data-6ab-deprecated",
        "label": "Ext. Data 6ab (deprecated notebook)",
        "phase": "figure",
        "kind": "notebook",
        "status": "deprecated",
        "requires_data": [],
        "requires_toolchain": [],
        "expected_artifacts": [],
    }
    reason = check_skip(t, tmp_path, have_ir=True, have_rscript=True, have_ggradar=True)
    assert reason == "deprecated: not executed"


def test_real_manifest_loads():
    root = Path(__file__).resolve().parents[1]
    m = load_manifest(root / "reproduce.manifest.yaml")
    ids = [t["id"] for t in iter_targets(m)]
    assert "fig-2" in ids
    assert "ext-data-5a" in ids
    assert "ext-data-5b" in ids
    five_b = next(t for t in m["targets"] if t["id"] == "ext-data-5b")
    assert five_b["status"] == "skip_until_data"


def test_notebook_run_key_returns_path_for_notebooks():
    target = {
        "kind": "notebook",
        "path": "Extended Data Fig. 5/extended_data_fig. 5ab.ipynb",
    }
    assert _notebook_run_key(target) == "Extended Data Fig. 5/extended_data_fig. 5ab.ipynb"
    assert _notebook_run_key({"kind": "rscript", "path": "foo.R"}) is None
    assert _notebook_run_key({"kind": "notebook", "path": None}) is None


def test_detect_toolchain_keys():
    with patch("scripts.reproduce_lib.shutil.which", return_value="/usr/bin/Rscript"), patch(
        "scripts.reproduce_lib.subprocess.check_output", return_value="ir    /path\n"
    ), patch("scripts.reproduce_lib.subprocess.call", return_value=0):
        tc = detect_toolchain()
    assert tc == {"have_ir": True, "have_rscript": True, "have_ggradar": True}


def test_run_figure_target_unknown_kind(tmp_path: Path):
    target = {"kind": "eval_probe", "path": None, "workdir": None}
    rc = run_figure_target(target, tmp_path, tmp_path / "logs" / "x.log")
    assert rc == 2
    assert "unknown kind" in (tmp_path / "logs" / "x.log").read_text(encoding="utf-8")


def test_run_figure_target_notebook_command(tmp_path: Path):
    nb = tmp_path / "Fig. 2" / "fig. 2.ipynb"
    nb.parent.mkdir(parents=True)
    nb.write_text("{}", encoding="utf-8")
    target = {
        "kind": "notebook",
        "path": "Fig. 2/fig. 2.ipynb",
        "workdir": None,
    }
    log_path = tmp_path / "logs" / "fig-2.log"
    mock_proc = MagicMock(returncode=0)
    with patch("scripts.reproduce_lib.subprocess.run", return_value=mock_proc) as run:
        rc = run_figure_target(target, tmp_path, log_path)
    assert rc == 0
    cmd = run.call_args[0][0]
    assert cmd[:4] == ["jupyter", "nbconvert", "--to", "notebook"]
    assert str(tmp_path / "Fig. 2/fig. 2.ipynb") in cmd[-1]
    assert "+ jupyter" in log_path.read_text(encoding="utf-8")


def _mini_manifest_yaml(*targets: str) -> str:
    body = "\n".join(targets)
    return f"version: 1\ntargets:\n{body}"


def test_reproduce_main_all_skipped(tmp_path: Path, capsys):
    manifest = tmp_path / "reproduce.manifest.yaml"
    manifest.write_text(
        _mini_manifest_yaml(
            """  - id: ext-data-5b
    label: Ext. Data 5b
    phase: figure
    kind: notebook
    path: "Extended Data Fig. 5/extended_data_fig. 5ab.ipynb"
    status: skip_until_data
    requires_data: []
    requires_toolchain: []
    expected_artifacts: []"""
        ),
        encoding="utf-8",
    )
    with patch(
        "scripts.reproduce_lib.detect_toolchain",
        return_value={"have_ir": True, "have_rscript": True, "have_ggradar": True},
    ):
        rc = reproduce_main(["--check"], tmp_path)
    out = capsys.readouterr().out
    assert rc == 0
    assert "⊘ Ext. Data 5b" in out
    assert (tmp_path / "logs" / "ext-data-5b.log").read_text(encoding="utf-8").startswith(
        "skipped:"
    )


def test_reproduce_main_artifact_fail_marks_x(tmp_path: Path, capsys):
    manifest = tmp_path / "reproduce.manifest.yaml"
    manifest.write_text(
        _mini_manifest_yaml(
            """  - id: fig-2
    label: Fig. 2
    phase: figure
    kind: notebook
    path: "Fig. 2/fig. 2.ipynb"
    status: run
    requires_data: []
    requires_toolchain: []
    expected_artifacts:
      - "Fig. 2/output/fig.2a.phytobench-knowledge.bar.pdf" """
        ),
        encoding="utf-8",
    )
    with patch(
        "scripts.reproduce_lib.detect_toolchain",
        return_value={"have_ir": True, "have_rscript": True, "have_ggradar": True},
    ), patch("scripts.reproduce_lib.run_figure_target", return_value=0):
        rc = reproduce_main(["--check"], tmp_path)
    out = capsys.readouterr().out
    assert rc == 1
    assert "✘ Fig. 2" in out
    assert "missing artifact" in out


def test_reproduce_main_check_exit_without_failures(tmp_path: Path, capsys):
    out_dir = tmp_path / "Fig. 2" / "output"
    out_dir.mkdir(parents=True)
    (out_dir / "fig.2a.phytobench-knowledge.bar.pdf").write_bytes(b"%PDF")
    manifest = tmp_path / "reproduce.manifest.yaml"
    manifest.write_text(
        _mini_manifest_yaml(
            """  - id: fig-2
    label: Fig. 2
    phase: figure
    kind: notebook
    path: "Fig. 2/fig. 2.ipynb"
    status: run
    requires_data: []
    requires_toolchain: []
    expected_artifacts:
      - "Fig. 2/output/fig.2a.phytobench-knowledge.bar.pdf" """
        ),
        encoding="utf-8",
    )
    with patch(
        "scripts.reproduce_lib.detect_toolchain",
        return_value={"have_ir": True, "have_rscript": True, "have_ggradar": True},
    ), patch("scripts.reproduce_lib.run_figure_target", return_value=0):
        rc = reproduce_main(["--check"], tmp_path)
    out = capsys.readouterr().out
    assert rc == 0
    assert "✓ Fig. 2" in out


def test_reproduce_main_shared_notebook_runs_once(tmp_path: Path):
    manifest = tmp_path / "reproduce.manifest.yaml"
    manifest.write_text(
        _mini_manifest_yaml(
            """  - id: ext-data-5a
    label: Ext. Data 5a
    phase: figure
    kind: notebook
    path: "Extended Data Fig. 5/extended_data_fig. 5ab.ipynb"
    status: run
    requires_data: []
    requires_toolchain: []
    expected_artifacts: []
  - id: ext-data-5b
    label: Ext. Data 5b
    phase: figure
    kind: notebook
    path: "Extended Data Fig. 5/extended_data_fig. 5ab.ipynb"
    status: run
    requires_data: []
    requires_toolchain: []
    expected_artifacts: []"""
        ),
        encoding="utf-8",
    )

    def fake_run(target, repo_root, log_path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("+ mocked\n", encoding="utf-8")
        return 0

    with patch(
        "scripts.reproduce_lib.detect_toolchain",
        return_value={"have_ir": True, "have_rscript": True, "have_ggradar": True},
    ), patch("scripts.reproduce_lib.run_figure_target", side_effect=fake_run) as run:
        reproduce_main([], tmp_path)
    assert run.call_count == 1
    assert (tmp_path / "logs" / "ext-data-5a.log").exists()
    assert "shared execution" in (tmp_path / "logs" / "ext-data-5b.log").read_text(
        encoding="utf-8"
    )


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
