import json
import os
import sys
import signal
from pathlib import Path

import nbformat
import pytest
import psutil
from PIL import Image

from scripts import reproduce_execution as execution


def figure_target(root: Path, *, tid="fig-1", source="Fig. 1/figure.py", artifact="Fig. 1/output/figure.png", data=None):
    path = root / source
    path.parent.mkdir(parents=True, exist_ok=True)
    code = "from PIL import Image\nImage.new('RGB', (20, 10), 'blue').save('output/figure.png')\n"
    kind = "notebook" if path.suffix == ".ipynb" else "py_script"
    if kind == "notebook":
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)],
            metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
        path.write_text(nbformat.writes(notebook))
    else:
        path.write_text(code)
    return {"id": tid, "label": tid, "phase": "figure", "kind": kind,
            "path": source, "workdir": str(path.parent.relative_to(root)),
            "kernel": "py" if kind == "notebook" else "none", "status": "run",
            "requires_data": data or [], "expected_artifacts": [artifact]}


def run(root, targets, **kwargs):
    return execution.run_targets({"version": 1, "targets": targets}, root, **kwargs)


def test_minimal_snapshot_uses_wip_and_excludes_undeclared_files(tmp_path):
    target = figure_target(tmp_path, data=["Fig. 1/input.txt"])
    (tmp_path / "Fig. 1/input.txt").write_text("current working tree")
    (tmp_path / "Fig. 1/private.txt").write_text("not declared")
    source = tmp_path / target["path"]
    source.write_text(source.read_text() + "assert open('input.txt').read() == 'current working tree'\nassert not __import__('pathlib').Path('private.txt').exists()\n")
    before = source.read_bytes()
    result = run(tmp_path, [target])
    assert result.exit_code == 0
    assert source.read_bytes() == before
    assert Image.open(tmp_path / target["expected_artifacts"][0]).size == (20, 10)
    record = result.summary["targets"][0]
    assert record["status"] == "succeeded"
    assert set(record["input_hashes"]) == {target["path"], "Fig. 1/input.txt"}
    assert result.summary["environment"]["python_version"] == sys.version.split()[0]
    assert (result.run_dir / "shareable/summary.json").is_file()


def test_shared_source_executes_once_and_publishes_selected_artifacts_only(tmp_path):
    first = figure_target(tmp_path)
    second = {**first, "id": "fig-1b", "expected_artifacts": ["Fig. 1/output/second.png"]}
    source = tmp_path / first["path"]
    source.write_text(source.read_text() + "Image.new('RGB', (12, 12)).save('output/second.png')\n")
    selected = run(tmp_path, [first, second], target_ids=[first["id"]])
    assert selected.exit_code == 0
    assert not (tmp_path / second["expected_artifacts"][0]).exists()
    assert not (selected.run_dir / "shareable/artifacts" / second["expected_artifacts"][0]).exists()
    both = run(tmp_path, [first, second])
    assert both.exit_code == 0
    assert both.summary["coverage"]["executed_units"] == 1
    assert both.summary["coverage"]["succeeded_targets"] == 2


def test_unselected_pending_sibling_does_not_add_its_missing_data(tmp_path):
    first = figure_target(tmp_path)
    second = {**first, "id": "fig-1b", "status": "skip_until_data",
              "requires_data": ["Fig. 1/future.tsv"], "pending_inputs": ["Fig. 1/future.tsv"],
              "pending_reason": "Awaiting author input", "expected_artifacts": ["Fig. 1/output/future.png"]}
    assert run(tmp_path, [first, second], target_ids=[first["id"]]).exit_code == 0
    pending = run(tmp_path, [first, second], target_ids=[second["id"]])
    assert pending.exit_code == 1
    assert pending.summary["coverage"]["executed_units"] == 0


def test_noop_cannot_pass_using_existing_outputs(tmp_path):
    target = figure_target(tmp_path)
    output = tmp_path / target["expected_artifacts"][0]
    output.parent.mkdir()
    Image.new("RGB", (3, 4)).save(output)
    before = output.read_bytes()
    (tmp_path / target["path"]).write_text("pass\n")
    result = run(tmp_path, [target])
    assert result.exit_code == 1
    assert result.summary["targets"][0]["reason_code"] == "invalid_artifact"
    assert output.read_bytes() == before


def test_failure_and_secret_diagnostics_do_not_reach_shareable_output(tmp_path, monkeypatch):
    target = figure_target(tmp_path)
    (tmp_path / target["path"]).write_text("import os\nassert 'DEEPGENOME_API_KEY' not in os.environ\nraise RuntimeError('PRIVATE_SENTINEL')\n")
    monkeypatch.setenv("DEEPGENOME_API_KEY", "API_SECRET_SENTINEL")
    result = run(tmp_path, [target])
    assert result.exit_code == 1
    for path in (result.run_dir / "shareable").rglob("*"):
        if path.is_file():
            assert b"PRIVATE_SENTINEL" not in path.read_bytes()
            assert b"API_SECRET_SENTINEL" not in path.read_bytes()
    assert any(b"PRIVATE_SENTINEL" in p.read_bytes() for p in (result.run_dir / "raw").rglob("*.log"))


def test_timeout_terminates_process_group_and_preserves_old_output(tmp_path):
    target = figure_target(tmp_path)
    output = tmp_path / target["expected_artifacts"][0]
    output.parent.mkdir()
    output.write_bytes(b"old output")
    (tmp_path / target["path"]).write_text(
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "Path('child.pid').write_text(str(child.pid))\ntime.sleep(60)\n")
    result = run(tmp_path, [target], timeout=1)
    assert result.exit_code == 1
    assert result.summary["targets"][0]["reason_code"] == "timeout"
    assert output.read_bytes() == b"old output"
    pid = int(next((result.run_dir / "work").rglob("child.pid")).read_text())
    stat = Path(f"/proc/{pid}/stat")
    assert not stat.exists() or stat.read_text().split()[2] == "Z"


def test_notebook_timeout_terminates_detached_kernel(tmp_path):
    target = figure_target(tmp_path, source="Fig. 1/figure.ipynb")
    source = tmp_path / target["path"]
    notebook = nbformat.read(source, as_version=4)
    notebook.cells[0].source = "import os, time\nfrom pathlib import Path\nPath('kernel.pid').write_text(str(os.getpid()))\ntime.sleep(60)"
    source.write_text(nbformat.writes(notebook))
    result = run(tmp_path, [target], timeout=8)
    pid = int(next((result.run_dir / "work").rglob("kernel.pid")).read_text())
    try:
        assert result.summary["targets"][0]["reason_code"] == "timeout"
        assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    finally:
        if psutil.pid_exists(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_fast_parent_exit_cannot_leave_a_reparented_detached_child(tmp_path):
    target = figure_target(tmp_path)
    source = tmp_path / target["path"]
    source.write_text(source.read_text() +
        "import subprocess, sys, os\nfrom pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)\n"
        "Path('child.pid').write_text(str(child.pid))\nos._exit(0)\n")
    result = run(tmp_path, [target])
    pid = int(next((result.run_dir / "work").rglob("child.pid")).read_text())
    try:
        assert result.exit_code == 0
        assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    finally:
        if psutil.pid_exists(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("content, suffix", [(b"ordinary text", ".pdf"), (b"\x89PNG\r\n\x1a\n", ".png")])
def test_artifact_parser_rejects_invalid_files(tmp_path, content, suffix):
    path = tmp_path / ("figure" + suffix)
    path.write_bytes(content)
    with pytest.raises(execution.ArtifactError):
        execution.inspect_artifact(path)


def test_pdf_parser_accepts_real_page_content(tmp_path):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=100, height=80)
    stream = DecodedStreamObject()
    stream.set_data(b"0 0 m 10 10 l S\n")
    page.replace_contents(stream)
    path = tmp_path / "figure.pdf"
    writer.write(path)
    info = execution.inspect_artifact(path)
    assert info["pages"] == 1 and info["format"] == "pdf"


def test_publication_failure_restores_prior_outputs_and_preserves_unrelated_files(tmp_path, monkeypatch):
    first = figure_target(tmp_path)
    second = {**first, "id": "fig-1b", "expected_artifacts": ["Fig. 1/output/second.png"]}
    source = tmp_path / first["path"]
    source.write_text(source.read_text() + "Image.new('RGB', (12, 12)).save('output/second.png')\n")
    folder = tmp_path / "Fig. 1/output"
    folder.mkdir()
    (folder / "figure.png").write_bytes(b"old first")
    (folder / "second.png").write_bytes(b"old second")
    (folder / "user.txt").write_text("untouched")
    replace = execution.os.replace
    def fail_second(src, dst):
        if Path(dst) == folder / "second.png":
            raise OSError("injected publication failure")
        replace(src, dst)
    monkeypatch.setattr(execution.os, "replace", fail_second)
    result = run(tmp_path, [first, second])
    assert result.exit_code == 1
    assert (folder / "figure.png").read_bytes() == b"old first"
    assert (folder / "second.png").read_bytes() == b"old second"
    assert (folder / "user.txt").read_text() == "untouched"
    assert all(t["publication_status"] == "rolled_back" for t in result.summary["targets"])


def test_rollback_never_overwrites_subsequent_user_edit(tmp_path, monkeypatch):
    first = figure_target(tmp_path)
    second = {**first, "id": "fig-1b", "expected_artifacts": ["Fig. 1/output/second.png"]}
    source = tmp_path / first["path"]
    source.write_text(source.read_text() + "Image.new('RGB', (12, 12)).save('output/second.png')\n")
    folder = tmp_path / "Fig. 1/output"
    folder.mkdir()
    (folder / "figure.png").write_bytes(b"old first")
    replace = execution.os.replace
    def fail_after_edit(src, dst):
        if Path(dst) == folder / "second.png":
            (folder / "figure.png").write_bytes(b"new user edit")
            raise OSError("later failure")
        replace(src, dst)
    monkeypatch.setattr(execution.os, "replace", fail_after_edit)
    result = run(tmp_path, [first, second])
    assert result.exit_code == 1
    assert (folder / "figure.png").read_bytes() == b"new user edit"
    assert all(t["publication_status"] == "conflict" for t in result.summary["targets"])


def test_concurrent_input_change_blocks_publication(tmp_path, monkeypatch):
    target = figure_target(tmp_path)
    original_execute = execution.execute_unit
    def change_input(*args, **kwargs):
        outcome = original_execute(*args, **kwargs)
        (tmp_path / target["path"]).write_text("# edited while running\n")
        return outcome
    monkeypatch.setattr(execution, "execute_unit", change_input)
    result = run(tmp_path, [target])
    assert result.exit_code == 1
    assert not (tmp_path / target["expected_artifacts"][0]).exists()
    assert result.summary["targets"][0]["reason_code"] == "publication_conflict"


@pytest.mark.parametrize("location", ["destination", "work-parent"])
def test_artifact_symlinks_never_pass_validation_or_publication(tmp_path, monkeypatch, location):
    target = figure_target(tmp_path)
    original_execute = execution.execute_unit
    outside = tmp_path / "figure.png"
    Image.new("RGB", (30, 40), "red").save(outside)
    outside_before = outside.read_bytes()
    if location == "destination":
        output = tmp_path / target["expected_artifacts"][0]
        output.parent.mkdir()
        alias = output.parent / "alias.png"
        alias.write_bytes(b"user alias")
        output.symlink_to(alias.name)
    else:
        def change_output_parent(unit, work_root, *args, **kwargs):
            outcome = original_execute(unit, work_root, *args, **kwargs)
            folder = work_root / "Fig. 1/output"
            for path in folder.iterdir():
                path.unlink()
            folder.rmdir()
            folder.symlink_to(tmp_path, target_is_directory=True)
            return outcome
        monkeypatch.setattr(execution, "execute_unit", change_output_parent)
    result = run(tmp_path, [target])
    assert result.exit_code == 1
    assert outside.read_bytes() == outside_before
    if location == "destination":
        assert alias.read_bytes() == b"user alias" and output.is_symlink()


def test_changed_artifact_between_validation_and_install_is_not_published(tmp_path, monkeypatch):
    target = figure_target(tmp_path)
    original_publish = execution.publish_unit
    def change_artifact(unit, root, work_root, *args, **kwargs):
        (work_root / unit.artifacts[0]).write_bytes(b"changed after validation")
        return original_publish(unit, root, work_root, *args, **kwargs)
    monkeypatch.setattr(execution, "publish_unit", change_artifact)
    result = run(tmp_path, [target])
    assert result.exit_code == 1
    assert not (tmp_path / target["expected_artifacts"][0]).exists()


def test_shareable_export_failure_reports_already_published_files_truthfully(tmp_path, monkeypatch):
    target = figure_target(tmp_path)
    stage = execution._stage_file
    def fail_export(source, destination):
        if "export-staging" in Path(destination).parts:
            raise OSError("export failed")
        stage(source, destination)
    monkeypatch.setattr(execution, "_stage_file", fail_export)
    result = run(tmp_path, [target])
    assert result.exit_code == 1
    record = result.summary["targets"][0]
    assert record["publication_status"] == "succeeded"
    assert record["reason_code"] == "artifact_export_failed"
    assert record["artifacts"] and (tmp_path / target["expected_artifacts"][0]).exists()


@pytest.mark.r_runtime
@pytest.mark.parametrize("kind", ["notebook", "rscript"])
def test_prepared_r_library_executes_without_profile_or_environment_fallback(tmp_path, kind, monkeypatch):
    library = os.environ.get("R_LIBS_USER")
    assert library and Path(library).is_dir(), "Prepare explicit R_LIBS_USER before R runtime tests"
    folder = tmp_path / "Fig. 1"
    folder.mkdir()
    home = tmp_path / "poisoned-home"
    home.mkdir()
    (home / ".Rprofile").write_text('stop("Unexpected user R profile")\n')
    monkeypatch.setenv("HOME", str(home))
    code = ('stopifnot(Sys.getenv("R_PROFILE_USER") %in% c("", "/dev/null"))\n'
            'stopifnot(normalizePath(.libPaths()[1]) == normalizePath(Sys.getenv("R_LIBS_USER")))\n'
            'stopifnot(!any(grepl("/site-library$", .libPaths())))\n'
            'pdf("output/figure.pdf", width=4, height=3)\nplot(1:3, 1:3)\ndev.off()\n')
    source = folder / ("figure.ipynb" if kind == "notebook" else "figure.R")
    if kind == "notebook":
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)],
            metadata={"kernelspec": {"name": "ir", "display_name": "R"}})
        source.write_text(nbformat.writes(notebook))
    else:
        source.write_text(code)
    target = {"id": "fig-1", "label": "R fixture", "phase": "figure", "kind": kind,
              "path": source.relative_to(tmp_path).as_posix(), "workdir": "Fig. 1",
              "kernel": "ir" if kind == "notebook" else "none", "status": "run",
              "requires_toolchain": ["ir"] if kind == "notebook" else ["Rscript"],
              "expected_artifacts": ["Fig. 1/output/figure.pdf"]}
    before = source.read_bytes()
    result = run(tmp_path, [target], timeout=60)
    assert result.exit_code == 0
    assert source.read_bytes() == before
    assert result.summary["environment"]["r_version"] == "4.3.3"


@pytest.mark.r_runtime
def test_empty_explicit_r_library_cannot_fall_back_to_system_packages(tmp_path, monkeypatch):
    library = tmp_path / "empty-library"
    library.mkdir()
    monkeypatch.setenv("R_LIBS_USER", str(library))
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    _, available, _ = execution._environment(run_dir, require_r=True)
    assert "ir" not in available and "ggradar" not in available


def test_python_notebook_binds_prepared_interpreter_and_preserves_source(tmp_path, monkeypatch):
    home = tmp_path / "poisoned-home"
    (home / ".jupyter").mkdir(parents=True)
    sentinel = home / "config-executed"
    (home / ".jupyter/jupyter_nbconvert_config.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('user config executed')\n")
    monkeypatch.setenv("HOME", str(home))
    target = figure_target(tmp_path, source="Fig. 1/figure.ipynb")
    source = tmp_path / target["path"]
    notebook = nbformat.read(source, as_version=4)
    notebook.cells[0].source += f"\nimport sys\nassert sys.prefix == {sys.prefix!r}\n"
    source.write_text(nbformat.writes(notebook))
    before = source.read_bytes()
    result = run(tmp_path, [target], timeout=30)
    assert result.exit_code == 0
    assert source.read_bytes() == before
    assert not sentinel.exists()


def test_report_locates_exact_inputs_diagnostics_and_backups(tmp_path):
    target = figure_target(tmp_path)
    old_output = tmp_path / target["expected_artifacts"][0]
    old_output.parent.mkdir()
    old_output.write_bytes(b"old user figure")
    result = run(tmp_path, [target])
    record = result.summary["targets"][0]
    assert record["execution_unit"]
    assert (result.run_dir / record["raw_log"]).is_file()
    assert (result.run_dir / record["inputs_root"] / target["path"]).is_file()
    assert record["backup_status"] == "retained"
    assert (result.run_dir / record["backups"][0]["path"]).read_bytes() == b"old user figure"
