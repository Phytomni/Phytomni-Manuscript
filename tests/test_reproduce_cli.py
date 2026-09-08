from pathlib import Path
from unittest.mock import patch

import nbformat
import pytest
import yaml

from scripts.render_reproduce_matrix import render_matrix
from scripts.reproduce_cli import main, parse_request, run_read_only_command


def manifest_fixture(root: Path):
    figure = root / "Fig. 1"
    figure.mkdir()
    notebook = nbformat.v4.new_notebook(metadata={"kernelspec": {
        "name": "python3", "display_name": "Python 3"}})
    (figure / "figure.ipynb").write_text(nbformat.writes(notebook))
    manifest = {"version": 1, "targets": [
        {"id": "fig-1", "label": "Fig. 1", "phase": "figure", "kind": "notebook",
         "path": "Fig. 1/figure.ipynb", "kernel": "py", "status": "run",
         "expected_artifacts": ["Fig. 1/output/figure.pdf"]},
        {"id": "eval-agent", "label": "Agent", "phase": "eval", "kind": "eval_probe",
         "status": "run", "probes": [{"type": "file_exists", "path": "Agent/private.tsv"}]},
    ]}
    (root / "README.md").write_text("<!-- BEGIN:REPRODUCE_MATRIX -->\n" + render_matrix(manifest, root)
                                  + "\n<!-- END:REPRODUCE_MATRIX -->\n")
    return manifest


@pytest.mark.parametrize("argv", [
    ["--check"], ["unknown"], ["run", "--target", "unknown"],
    ["run", "--target", "eval-agent"], ["probe", "--target", "fig-1"],
    ["run", "--timeout", "0"], ["run", "--timeout", "-1"],
    ["run", "--timeout", "nan"], ["run", "--time", "2"],
    ["validate", "--timeout", "1"], ["list", "--target", "fig-1"],
])
def test_invalid_requests_exit_two_before_any_execution(tmp_path, argv):
    manifest = manifest_fixture(tmp_path)
    with patch("subprocess.run", side_effect=AssertionError("executed")), pytest.raises(SystemExit) as error:
        parse_request(argv, manifest)
    assert error.value.code == 2


def test_parser_selects_targets_in_manifest_order(tmp_path):
    manifest = manifest_fixture(tmp_path)
    args = parse_request(["run", "--target", "fig-1", "--target", "fig-1", "--timeout", "12"], manifest)
    assert args.command == "run" and args.timeout == 12
    assert [t["id"] for t in args.targets] == ["fig-1"]
    assert parse_request([], manifest).command is None


@pytest.mark.parametrize("command", ["list", "validate", "probe"])
def test_read_only_handlers_leave_tree_unchanged_and_launch_nothing(tmp_path, command, capsys):
    manifest = manifest_fixture(tmp_path)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    args = parse_request([command], manifest)
    with patch("subprocess.run", side_effect=AssertionError("executed")), patch(
        "subprocess.Popen", side_effect=AssertionError("toolchain checked")
    ):
        status = run_read_only_command(args, manifest, tmp_path)
    assert status == (1 if command == "probe" else 0)
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before
    output = capsys.readouterr().out
    if command == "list":
        assert "toolchain not checked" in output and "eval-agent" in output
    elif command == "probe":
        assert "unavailable" in output and "evaluation not executed" in output


def test_validate_reports_stale_readme_and_missing_source_as_failure(tmp_path, capsys):
    manifest = manifest_fixture(tmp_path)
    args = parse_request(["validate"], manifest)
    (tmp_path / "README.md").write_text("stale")
    (tmp_path / "Fig. 1/figure.ipynb").unlink()
    assert run_read_only_command(args, manifest, tmp_path) == 1
    output = capsys.readouterr().out
    assert "README" in output and "source missing" in output


def test_no_argument_internal_handler_only_shows_help(tmp_path, capsys):
    manifest = manifest_fixture(tmp_path)
    assert run_read_only_command(parse_request([], manifest), manifest, tmp_path) == 0
    assert "usage:" in capsys.readouterr().out


def test_read_only_handler_does_not_accept_run(tmp_path):
    manifest = manifest_fixture(tmp_path)
    with pytest.raises(ValueError, match="read-only"):
        run_read_only_command(parse_request(["run"], manifest), manifest, tmp_path)


def test_public_no_argument_and_invalid_flag_do_not_read_manifest_or_run(tmp_path, capsys):
    with patch("scripts.reproduce_cli.load_manifest", side_effect=AssertionError("manifest read")):
        assert main([], repo_root=tmp_path) == 0
        with pytest.raises(SystemExit) as error:
            main(["--check"], repo_root=tmp_path)
        assert error.value.code == 2
    assert "usage:" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("command", ["list", "validate", "probe"])
def test_public_read_only_dispatch_has_no_runtime_or_environment_side_effects(tmp_path, monkeypatch, command):
    manifest = manifest_fixture(tmp_path)
    (tmp_path / "reproduce.manifest.yaml").write_text(yaml.safe_dump(manifest))
    monkeypatch.delenv("PHYTOMNI_SAVE", raising=False)
    with patch("subprocess.Popen", side_effect=AssertionError("process started")):
        assert main([command], repo_root=tmp_path) == (1 if command == "probe" else 0)
    assert not (tmp_path / "logs").exists()
    assert "PHYTOMNI_SAVE" not in __import__("os").environ


def test_public_run_dispatch_uses_fresh_executor_and_returns_failure(tmp_path):
    manifest = manifest_fixture(tmp_path)
    (tmp_path / "reproduce.manifest.yaml").write_text(yaml.safe_dump(manifest))
    with patch("scripts.reproduce_execution.run_targets") as run:
        run.return_value.exit_code = 1
        assert main(["run", "--target", "fig-1", "--timeout", "5"], repo_root=tmp_path) == 1
    assert run.call_args.kwargs == {"target_ids": ["fig-1"], "timeout": 5}


def test_public_wrong_phase_selection_is_argument_error_without_logs(tmp_path):
    manifest = manifest_fixture(tmp_path)
    (tmp_path / "reproduce.manifest.yaml").write_text(yaml.safe_dump(manifest))
    with pytest.raises(SystemExit) as error:
        main(["run", "--target", "eval-agent"], repo_root=tmp_path)
    assert error.value.code == 2 and not (tmp_path / "logs").exists()


def test_malformed_yaml_is_reported_as_validation_failure(tmp_path):
    (tmp_path / "reproduce.manifest.yaml").write_text("targets: [\n")
    assert main(["validate"], repo_root=tmp_path) == 1
