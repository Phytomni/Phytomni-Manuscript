from pathlib import Path
from copy import deepcopy

import pytest

from scripts.render_reproduce_matrix import render_matrix
from scripts.reproduce_lib import (
    ManifestError,
    iter_targets,
    load_manifest,
    run_eval_probes,
    run_probe,
)



def contract_manifest(**changes):
    target = {
        "id": "fig-1",
        "label": "Fig. 1",
        "phase": "figure",
        "kind": "notebook",
        "path": "Fig. 1/fig. 1.ipynb",
        "kernel": "py",
        "status": "run",
        "requires_data": ["Fig. 1/data.tsv"],
        "expected_artifacts": ["Fig. 1/output/fig.1.pdf"],
    }
    target.update(changes)
    return {"version": 1, "targets": [target]}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", "../escape", "id"),
        ("phase", "surprise", "phase"),
        ("kind", "surprise", "kind"),
        ("status", "deprecated", "status"),
        ("kernel", "python-old", "kernel"),
        ("requires_data", "data.tsv", "requires_data"),
        ("requires_data", [None], "requires_data"),
        ("requires_toolchain", ["unknown"], "requires_toolchain"),
        ("expected_artifacts", [], "expected_artifacts"),
        ("path", None, "path"),
        ("path", "/tmp/figure.ipynb", "relative"),
        ("path", "Fig. 1/../figure.ipynb", "relative"),
        ("path", "Fig. 1\\figure.ipynb", "relative"),
        ("requires_data", [".codex/private.tsv"], "public"),
        ("requires_data", ["Fig. 1/output/cached.tsv"], "output"),
        ("requires_data", ["Fig. 2/data.tsv"], "same figure"),
        ("expected_artifacts", ["Fig. 2/output/result.pdf"], "output"),
        ("workdir", "Fig. 2", "workdir"),
        ("unexpected_field", True, "unexpected"),
    ],
)
def test_manifest_contract_rejects_invalid_fields(tmp_path, field, value, message):
    from scripts.reproduce_lib import validate_manifest

    with pytest.raises(ManifestError, match=message):
        validate_manifest(contract_manifest(**{field: value}), tmp_path)


def test_manifest_contract_rejects_version_and_duplicate_ids(tmp_path):
    from scripts.reproduce_lib import validate_manifest

    manifest = contract_manifest()
    for version in (None, True, 2, "1"):
        with pytest.raises(ManifestError, match="version"):
            validate_manifest({**manifest, "version": version}, tmp_path)
    manifest["targets"].append(deepcopy(manifest["targets"][0]))
    with pytest.raises(ManifestError, match="duplicate target id"):
        validate_manifest(manifest, tmp_path)


def test_manifest_contract_checks_pending_declarations(tmp_path):
    from scripts.reproduce_lib import validate_manifest

    manifest = contract_manifest(status="skip_until_data")
    with pytest.raises(ManifestError, match="pending_reason"):
        validate_manifest(manifest, tmp_path)
    target = manifest["targets"][0]
    target["pending_reason"] = "Awaiting the reviewed input table."
    target["pending_inputs"] = ["Fig. 1/other.tsv"]
    with pytest.raises(ManifestError, match="pending_inputs"):
        validate_manifest(manifest, tmp_path)
    target["pending_inputs"] = list(target["requires_data"])
    before = deepcopy(manifest)
    validate_manifest(manifest, tmp_path)
    assert manifest == before


def test_manifest_contract_checks_ownership_and_symlink_escapes(tmp_path):
    from scripts.reproduce_lib import validate_manifest

    manifest = contract_manifest()
    sibling = {**manifest["targets"][0], "id": "fig-1b"}
    manifest["targets"].append(sibling)
    with pytest.raises(ManifestError, match="artifact.*owned"):
        validate_manifest(manifest, tmp_path)
    sibling["expected_artifacts"] = ["Fig. 1/output/fig.1b.pdf"]
    validate_manifest(manifest, tmp_path)
    sibling["kernel"] = "ir"
    sibling["requires_toolchain"] = ["ir"]
    with pytest.raises(ManifestError, match="shared source"):
        validate_manifest(manifest, tmp_path)
    manifest = contract_manifest()
    (tmp_path / "Fig. 1").mkdir()
    (tmp_path / "Fig. 1/data.tsv").symlink_to(tmp_path.parent / "private.tsv")
    with pytest.raises(ManifestError, match="escapes"):
        validate_manifest(manifest, tmp_path)


def test_manifest_contract_allows_unavailable_private_probe_inputs(tmp_path):
    from scripts.reproduce_lib import validate_manifest

    manifest = {
        "version": 1,
        "targets": [{"id": "eval-agent", "label": "Agent", "phase": "eval",
                     "kind": "eval_probe", "status": "run", "probes": [
                         {"type": "file_exists", "path": "Agent Evaluation/private.tsv"},
                         {"type": "can_import", "module": "private_backend"},
                     ]}],
    }
    validate_manifest(manifest, tmp_path)


def test_prerequisites_distinguish_source_failure_pending_data_and_tools(tmp_path):
    from scripts.reproduce_lib import figure_prerequisites

    target = contract_manifest(status="skip_until_data", pending_reason="Author data",
                               pending_inputs=["Fig. 1/data.tsv"])["targets"][0]
    check = figure_prerequisites(target, tmp_path)
    assert check.state == "failed" and "source" in check.reasons[0]
    source = tmp_path / target["path"]
    source.parent.mkdir()
    source.write_text("{}")
    assert figure_prerequisites(target, tmp_path).state == "pending"
    assert figure_prerequisites(target, tmp_path, explicit=True).state == "failed"
    target["status"] = "run"
    assert figure_prerequisites(target, tmp_path).state == "failed"
    (tmp_path / "Fig. 1/data.tsv").write_text("value\n1\n")
    target["requires_toolchain"] = ["ir"]
    assert figure_prerequisites(target, tmp_path).state == "ready"
    assert figure_prerequisites(target, tmp_path, available_tools=set()).state == "failed"
    assert figure_prerequisites(target, tmp_path, available_tools={"ir"}).state == "ready"


def test_selection_preserves_registry_order_and_rejects_unknown_or_wrong_phase():
    from scripts.reproduce_lib import TargetSelectionError, select_targets

    manifest = contract_manifest()
    manifest["targets"].append({**manifest["targets"][0], "id": "fig-1b"})
    manifest["targets"].append({"id": "eval-agent", "phase": "eval"})
    assert [t["id"] for t in select_targets(manifest, ["fig-1b", "fig-1", "fig-1"],
                                            phase="figure")] == ["fig-1", "fig-1b"]
    with pytest.raises(TargetSelectionError, match="unknown"):
        select_targets(manifest, ["unknown"], phase="figure")
    with pytest.raises(TargetSelectionError, match="phase"):
        select_targets(manifest, ["eval-agent"], phase="figure")


def test_probe_discovery_does_not_execute_package_init(tmp_path, monkeypatch):
    package = tmp_path / "probe_backend_never_import"
    package.mkdir()
    (package / "__init__.py").write_text("raise AssertionError('backend imported')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert run_probe({"type": "can_import", "module": package.name}, tmp_path) is None
    reason = run_probe({"type": "can_import", "module": package.name + ".child"}, tmp_path)
    assert reason is not None and "top-level" in reason


def test_readme_matrix_check_never_writes(tmp_path):
    from scripts.render_reproduce_matrix import check_readme_matrix

    manifest = contract_manifest()
    readme = tmp_path / "README.md"
    readme.write_text("Heading\n<!-- BEGIN:REPRODUCE_MATRIX -->\nold\n<!-- END:REPRODUCE_MATRIX -->\n")
    original = readme.read_bytes()
    assert check_readme_matrix(readme, manifest, tmp_path)
    assert readme.read_bytes() == original
    expected = "Heading\n<!-- BEGIN:REPRODUCE_MATRIX -->\n" + render_matrix(manifest, tmp_path) + "\n<!-- END:REPRODUCE_MATRIX -->\n"
    readme.write_text(expected)
    assert check_readme_matrix(readme, manifest, tmp_path) == []
    assert readme.read_text() == expected


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


def test_real_manifest_loads():
    root = Path(__file__).resolve().parents[1]
    m = load_manifest(root / "reproduce.manifest.yaml")
    ids = [t["id"] for t in iter_targets(m)]
    assert "fig-2" in ids
    assert "ext-data-5a" in ids
    assert "ext-data-5b" in ids
    five_b = next(t for t in m["targets"] if t["id"] == "ext-data-5b")
    assert five_b["status"] == "run"
    eval_ids = [t["id"] for t in iter_targets(m, phase="eval")]
    assert eval_ids == ["eval-analyst", "eval-data", "eval-knowledge", "eval-expert"]


def test_evaluation_directory_references_match_canonical_names():
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root / "reproduce.manifest.yaml")
    eval_targets = {target["id"]: target for target in iter_targets(manifest, phase="eval")}
    expected = {
        "eval-knowledge": (
            "Knowledge&ReviewAgent Evaluation",
            "Knowledge&ReviewAgent Evaluation/evaluation_id.py",
        ),
        "eval-expert": (
            "DeepGenomeAgent Evaluation",
            "DeepGenomeAgent Evaluation/score.tsv",
        ),
    }

    for target_id, (label, probe_path) in expected.items():
        target = eval_targets[target_id]
        assert target["label"] == label
        assert any(probe.get("path") == probe_path for probe in target["probes"])

    readme = (root / "README.md").read_text(encoding="utf-8")
    for label, _ in expected.values():
        assert f"### {label}" in readme
        assert f"<{label}/README.md>" in readme
        assert (root / label / "README.md").is_file()


def test_real_manifest_eval_targets_probe_skip_on_bare_clone():
    root = Path(__file__).resolve().parents[1]
    m = load_manifest(root / "reproduce.manifest.yaml")
    eval_ids = [t["id"] for t in iter_targets(m, phase="eval")]
    assert eval_ids == ["eval-analyst", "eval-data", "eval-knowledge", "eval-expert"]
    for target in iter_targets(m, phase="eval"):
        assert run_eval_probes(target, root) is not None


def test_run_probe_file_exists_passes_and_fails(tmp_path: Path):
    present = tmp_path / "present.txt"
    present.write_text("ok", encoding="utf-8")
    assert run_probe({"type": "file_exists", "path": "present.txt"}, tmp_path) is None
    reason = run_probe({"type": "file_exists", "path": "missing.txt"}, tmp_path)
    assert reason is not None
    assert "missing.txt" in reason


def test_run_probe_can_import(tmp_path: Path):
    assert run_probe({"type": "can_import", "module": "pathlib"}, tmp_path) is None
    reason = run_probe({"type": "can_import", "module": "definitely_not_a_real_module_xyz"}, tmp_path)
    assert reason is not None
    assert "definitely_not_a_real_module_xyz" in reason


def test_run_probe_file_contains_fails_when_substring_present(tmp_path: Path):
    cfg = tmp_path / "config.py"
    cfg.write_text('api_key = "change_to_your_api_key"\n', encoding="utf-8")
    reason = run_probe(
        {"type": "file_contains_none", "path": "config.py", "substring": "Change_to_your_"},
        tmp_path,
    )
    assert reason is not None
    assert "placeholder present" in reason
    cleaned = tmp_path / "clean.py"
    cleaned.write_text("api_key = 'real'\n", encoding="utf-8")
    assert (
        run_probe(
            {"type": "file_contains_none", "path": "clean.py", "substring": "Change_to_your_"},
            tmp_path,
        )
        is None
    )


def test_run_probe_note_requires_always_fails():
    reason = run_probe(
        {"type": "note_requires", "message": "CLI requires --input dataset not shipped in this repo"},
        Path("/unused"),
    )
    assert reason == "CLI requires --input dataset not shipped in this repo"


def test_run_eval_probes_concatenates_failures(tmp_path: Path):
    target = {
        "probes": [
            {"type": "file_exists", "path": "missing.yaml"},
            {"type": "can_import", "module": "definitely_not_a_real_module_xyz"},
        ]
    }
    reason = run_eval_probes(target, tmp_path)
    assert reason is not None
    assert "missing.yaml" in reason
    assert "definitely_not_a_real_module_xyz" in reason


def test_run_eval_probes_all_pass(tmp_path: Path):
    (tmp_path / "ok.txt").write_text("configured\n", encoding="utf-8")
    target = {
        "probes": [
            {"type": "file_exists", "path": "ok.txt"},
            {"type": "can_import", "module": "pathlib"},
        ]
    }
    assert run_eval_probes(target, tmp_path) is None
