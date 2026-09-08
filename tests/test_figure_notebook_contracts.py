import json
from pathlib import Path
from unittest.mock import patch

import nbformat
import pytest

from scripts.reproduce_lib import validate_figure_sources, validate_static


def source_fixture(root: Path, *, kernel="py", name="Fig. 1", code="value = 1"):
    folder = root / name
    folder.mkdir(exist_ok=True)
    path = folder / "figure.ipynb"
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(code)],
        metadata={"kernelspec": {"name": "python3" if kernel == "py" else "ir",
                                 "display_name": kernel}},
    )
    path.write_text(nbformat.writes(notebook), encoding="utf-8")
    return {"id": name.lower().replace(". ", "-"), "label": name, "phase": "figure",
            "kind": "notebook", "path": f"{name}/figure.ipynb", "kernel": kernel,
            "status": "run", "requires_toolchain": ["ir"] if kernel == "ir" else [],
            "requires_data": [], "expected_artifacts": [f"{name}/output/figure.pdf"]}


def test_notebook_structure_checks_do_not_launch_tools_or_change_files(tmp_path):
    target = source_fixture(tmp_path, kernel="ir", code="# Comment\nvalue <- 1")
    source = tmp_path / target["path"]
    before = source.read_bytes()
    with patch("subprocess.run", side_effect=AssertionError("subprocess started")), patch(
        "subprocess.check_output", side_effect=AssertionError("subprocess started")
    ):
        assert validate_figure_sources({"version": 1, "targets": [target]}, tmp_path) == []
    assert source.read_bytes() == before


def test_source_missing_is_invalid_even_for_pending_r_target(tmp_path):
    target = source_fixture(tmp_path, kernel="ir")
    target.update(status="skip_until_data", pending_reason="Awaiting author input",
                  pending_inputs=["Fig. 1/data.tsv"], requires_data=["Fig. 1/data.tsv"])
    (tmp_path / target["path"]).unlink()
    problems = validate_figure_sources({"version": 1, "targets": [target]}, tmp_path)
    assert any("source missing" in message for message in problems)


def test_notebook_invalid_schema_and_wrong_kernel_are_reported_without_repair(tmp_path):
    target = source_fixture(tmp_path)
    source = tmp_path / target["path"]
    notebook = json.loads(source.read_text())
    notebook["metadata"]["kernelspec"]["name"] = "old_private_kernel"
    source.write_text(json.dumps(notebook))
    problems = validate_figure_sources({"version": 1, "targets": [target]}, tmp_path)
    assert any("kernel" in message for message in problems)
    notebook["cells"][0]["execution_count"] = "invalid"
    source.write_text(json.dumps(notebook))
    before = source.read_bytes()
    problems = validate_figure_sources({"version": 1, "targets": [target]}, tmp_path)
    assert any("schema" in message for message in problems)
    assert source.read_bytes() == before


def test_structure_gate_does_not_enforce_later_presentation_policy(tmp_path):
    target = source_fixture(tmp_path, code="# Existing comment\nprint('data')")
    source = tmp_path / target["path"]
    notebook = json.loads(source.read_text())
    notebook["cells"][0]["execution_count"] = 1
    notebook["cells"][0]["outputs"] = [{"output_type": "stream", "name": "stdout", "text": "data\n"}]
    source.write_text(json.dumps(notebook))
    assert validate_figure_sources({"version": 1, "targets": [target]}, tmp_path) == []


def test_source_coverage_detects_orphans_but_not_declared_support_or_evaluators(tmp_path):
    target = source_fixture(tmp_path)
    helper = tmp_path / "Fig. 1/helper.py"
    helper.write_text("value = 1\n")
    target["requires_data"] = ["Fig. 1/helper.py"]
    evaluation = tmp_path / "Agent Evaluation"
    evaluation.mkdir()
    (evaluation / "evaluate.ipynb").write_text("not a figure")
    output = tmp_path / "Fig. 1/output"
    output.mkdir()
    (output / "executed.ipynb").write_text("old execution")
    manifest = {"version": 1, "targets": [target]}
    assert validate_figure_sources(manifest, tmp_path) == []
    (tmp_path / "Fig. 1/unregistered.ipynb").write_text("{}")
    assert any("unregistered figure source" in p for p in validate_figure_sources(manifest, tmp_path))


def test_selected_data_validation_still_checks_global_sources(tmp_path):
    first = source_fixture(tmp_path)
    second = source_fixture(tmp_path, name="Fig. 2")
    second["requires_data"] = ["Fig. 2/not-yet-present.tsv"]
    manifest = {"version": 1, "targets": [first, second]}
    assert validate_static(manifest, tmp_path, target_ids=[first["id"]]) == []
    assert any("required data missing" in p for p in validate_static(manifest, tmp_path))
    (tmp_path / second["path"]).unlink()
    assert any("source missing" in p for p in validate_static(manifest, tmp_path, target_ids=[first["id"]]))


def test_pending_data_is_reportable_not_a_required_data_failure(tmp_path):
    target = source_fixture(tmp_path)
    target.update(status="skip_until_data", pending_reason="Author data",
                  pending_inputs=["Fig. 1/data.tsv"], requires_data=["Fig. 1/data.tsv"])
    assert validate_static({"version": 1, "targets": [target]}, tmp_path) == []


@pytest.mark.parametrize("cells", [None, [None], [3], [{"id": []}]])
def test_malformed_cell_containers_report_failure_instead_of_crashing(tmp_path, cells):
    target = source_fixture(tmp_path)
    source = tmp_path / target["path"]
    notebook = json.loads(source.read_text())
    notebook["cells"] = cells
    source.write_text(json.dumps(notebook))
    assert any("schema" in p for p in validate_figure_sources({"version": 1, "targets": [target]}, tmp_path))
