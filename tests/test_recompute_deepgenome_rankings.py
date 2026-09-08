from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import freeze_deepgenome_rankings as freezer
from scripts import recompute_deepgenome_rankings as cli


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "DeepGenomeAgent Evaluation/supplementary/Supplementary_Data_Expert_Rankings.tsv"
CATEGORIES = ROOT / "Supplementary Fig. 12-15/PhytoBench-Gene-for_plot/gene_categories.tsv"
REFERENCE = CATEGORIES.parent / "frozen"


@pytest.fixture(autouse=True)
def deny_unstubbed_cli_computation(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("CLI contract tests must not run the production-cohort calculation")

    monkeypatch.setattr(cli, "compute_ranking_tables", unexpected)


def arguments(output: Path, *, replicates: int = 20, reference: bool = True) -> list[str]:
    result = [
        "--score-tsv", str(SOURCE), "--gene-categories", str(CATEGORIES),
        "--seed", "20260714", "--bootstrap-replicates", str(replicates),
        "--max-failed-fits", "10", "--output-dir", str(output),
    ]
    if reference:
        result.extend(["--reference-dir", str(REFERENCE)])
    return result


def snapshot(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir() if path.is_file()
    }


@pytest.fixture
def computed_tables() -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(REFERENCE / filename, sep="\t", dtype=str, keep_default_na=False)
        for name, filename in freezer.NUMERICAL_OUTPUT_FILENAMES.items()
    }


@pytest.fixture
def computation(monkeypatch, computed_tables):
    calls = []

    def compute(source, categories, config, *, progress=None):
        calls.append((source.copy(), categories.copy(), config))
        if progress is not None:
            for successful in (0, config.successful_replicates):
                progress({"phase": "pl_bootstrap", "attempted": successful,
                          "successful": successful, "failed": 0,
                          "total": config.successful_replicates})
        return {name: table.copy() for name, table in computed_tables.items()}, {
            "plackett_luce": {"AttemptedReplicates": config.successful_replicates,
                              "SuccessfulReplicates": config.successful_replicates,
                              "FailedFits": 0, "HalfRunStability": {"Applied": False},
                              "MonteCarloPrecision": {"Applied": False}},
            "agreement": {"attempted_replicates": config.successful_replicates,
                          "successful_replicates": config.successful_replicates,
                          "invalid_replicates": 0},
        }

    monkeypatch.setattr(cli, "compute_ranking_tables", compute)
    return calls


@pytest.mark.parametrize("missing", [
    "--score-tsv", "--gene-categories", "--seed", "--bootstrap-replicates",
    "--max-failed-fits", "--output-dir",
])
def test_cli_requires_explicit_inputs_and_configuration(tmp_path, missing):
    argv = arguments(tmp_path / "run")
    index = argv.index(missing)
    del argv[index:index + 2]
    with pytest.raises(SystemExit) as error:
        cli.main(argv)
    assert error.value.code == 2


def test_cli_rejects_unknown_flags(tmp_path):
    with pytest.raises(SystemExit) as error:
        cli.main(arguments(tmp_path / "run") + ["--allow-small-cohort"])
    assert error.value.code == 2


def test_cli_rejects_abbreviated_flags(tmp_path, computation):
    argv = arguments(tmp_path / "run")
    argv[argv.index("--bootstrap-replicates")] = "--bootstrap-rep"
    with pytest.raises(SystemExit) as error:
        cli.main(argv)
    assert error.value.code == 2


@pytest.mark.parametrize("destination", [
    REFERENCE, REFERENCE.parent, REFERENCE / "preview", ROOT,
    ROOT.parent, ROOT / "Fig. 2", ROOT / "Fig. 2/PhytoBench-Gene-metrics-for_plot/run",
    SOURCE.parent / "run", ROOT / "scripts/run",
])
def test_output_preflight_rejects_publication_paths_before_reads(
    destination, monkeypatch, computation,
):
    def unexpected_read(*args, **kwargs):
        raise AssertionError("Output guard must run before input reads")

    monkeypatch.setattr(Path, "read_bytes", unexpected_read)
    assert cli.main(arguments(destination)) == 1
    assert not computation


def test_output_preflight_rejects_symlink_aliases(tmp_path, computation):
    alias = tmp_path / "alias"
    alias.symlink_to(REFERENCE, target_is_directory=True)
    before = snapshot(REFERENCE)
    assert cli.main(arguments(alias / "preview")) == 1
    assert snapshot(REFERENCE) == before
    assert not computation


def test_reference_provenance_leaf_symlink_is_rejected_before_read(
    tmp_path, monkeypatch, computation,
):
    reference = tmp_path / "reference"
    reference.mkdir()
    linked = reference / "provenance.json"
    linked.symlink_to(REFERENCE / "provenance.json")
    read_bytes = Path.read_bytes

    def checked_read(path):
        assert path != linked, "Reference provenance symlink must not be read"
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", checked_read)
    argv = arguments(tmp_path / "run")
    argv[argv.index("--reference-dir") + 1] = str(reference)
    assert cli.main(argv) == 1
    assert not computation


@pytest.mark.parametrize("malformed", [[], {"schema_version": 2, "model_columns": list(freezer.MODEL_COLUMNS), "outputs": []}])
def test_malformed_reference_provenance_is_rejected(tmp_path, computation, malformed):
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "provenance.json").write_text(json.dumps(malformed))
    argv = arguments(tmp_path / "run")
    argv[argv.index("--reference-dir") + 1] = str(reference)
    assert cli.main(argv) == 1
    assert not computation


def test_output_preflight_preserves_unrelated_files(tmp_path, computation):
    output = tmp_path / "run"
    output.mkdir()
    sentinel = output / "notes.txt"
    sentinel.write_text("user data\n")
    assert cli.main(arguments(output)) == 1
    assert sentinel.read_text() == "user data\n"
    assert not computation


def test_output_must_not_contain_source_inputs(tmp_path, computation):
    source = tmp_path / "public.tsv"
    source.write_bytes(SOURCE.read_bytes())
    argv = arguments(tmp_path)
    argv[argv.index("--score-tsv") + 1] = str(source)
    before = source.read_bytes()
    assert cli.main(argv) == 1
    assert source.read_bytes() == before
    assert not computation


@pytest.mark.parametrize("mutation", ["missing_assignment", "columns", "rank", "expert_id"])
def test_production_cohort_and_schema_rejected_before_computation(
    tmp_path, computation, mutation,
):
    source = pd.read_csv(SOURCE, sep="\t", dtype=str)
    if mutation == "missing_assignment":
        source = source.iloc[:-1]
    elif mutation == "columns":
        source = source[source.columns[::-1]]
    elif mutation == "rank":
        source.loc[0, "Claude"] = source.loc[0, "Gemini"]
    else:
        source.loc[source.AnonymousExpertID == "E001", "AnonymousExpertID"] = "E121"
    source_path = tmp_path / "source.tsv"
    source.to_csv(source_path, sep="\t", index=False)
    output = tmp_path / "run"
    argv = arguments(output)
    argv[argv.index("--score-tsv") + 1] = str(source_path)
    assert cli.main(argv) == 1
    assert not output.exists()
    assert not computation


def test_preview_uses_only_public_inputs_and_preserves_reference(
    tmp_path, monkeypatch, computation, capsys,
):
    before = snapshot(REFERENCE)
    original_read_bytes = Path.read_bytes

    def public_reads_only(path):
        allowed = {SOURCE, CATEGORIES, REFERENCE / "provenance.json"}
        allowed.update(REFERENCE / name for name in freezer.NUMERICAL_OUTPUT_FILENAMES.values())
        allowed.update((ROOT / "scripts").glob("*.py"))
        assert path in allowed, "Unexpected private or unrelated input read"
        return original_read_bytes(path)

    with monkeypatch.context() as guarded:
        guarded.setattr(Path, "read_bytes", public_reads_only)
        guarded.setattr(pd, "read_excel", lambda *a, **k: pytest.fail("Private workbook read"))
        output = tmp_path / "run"
        assert cli.main(arguments(output)) == 0
    assert snapshot(REFERENCE) == before
    assert set(path.name for path in output.iterdir()) == {
        *freezer.NUMERICAL_OUTPUT_FILENAMES.values(),
        "recomputation-provenance.json", "comparison.json",
    }
    provenance = json.loads((output / "recomputation-provenance.json").read_text())
    assert provenance["cohort"] == {"assignments": 600, "experts": 120, "genes": 200}
    assert provenance["private_panel_reconstructed"] is False
    assert provenance["publication_eligible"] is False
    assert provenance["run_mode"] == "preview"
    assert provenance["bootstrap_diagnostics"]["plackett_luce"]["MonteCarloPrecision"] == {"Applied": False}
    assert provenance["inputs"]["public_ranking_release"]["sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert "versions" in provenance
    assert str(ROOT) not in json.dumps(provenance)
    comparison = json.loads((output / "comparison.json").read_text())
    assert comparison["mode"] == "point_estimates_and_counts"
    assert comparison["passed"] is True
    assert len(comparison["tables"]) == 11
    original = pd.read_csv(SOURCE, sep="\t", dtype=str)
    pd.testing.assert_frame_equal(computation[0][0], original)
    stderr = capsys.readouterr().err
    assert "10%" in stderr and "100%" in stderr


def test_preview_does_not_compare_bootstrap_intervals(tmp_path, computation, computed_tables):
    computed_tables["pl_scores_ci"].loc[0, "CI95Lower"] = "-9999"
    computed_tables["pl_scores_ci"].loc[0, "SuccessfulReplicates"] = "20"
    output = tmp_path / "run"
    assert cli.main(arguments(output)) == 0
    report = json.loads((output / "comparison.json").read_text())
    assert "CI95Lower" not in report["tables"]["pl_scores_ci.tsv"]["compared_columns"]


def test_point_estimate_mismatch_reports_field_and_returns_one(
    tmp_path, computation, computed_tables,
):
    before = snapshot(REFERENCE)
    computed_tables["pl_scores"].loc[0, "Elo"] = "9999"
    output = tmp_path / "run"
    assert cli.main(arguments(output)) == 1
    report = json.loads((output / "comparison.json").read_text())
    difference = report["tables"]["pl_scores.tsv"]["differences"][0]
    assert difference["column"] == "Elo" and difference["row"] == 1
    assert difference["absolute_difference"] > 0
    assert snapshot(REFERENCE) == before


@pytest.mark.parametrize("change", [None, "interval", "order"])
def test_full_matching_configuration_requires_exact_serialized_tables(
    tmp_path, computation, computed_tables, change,
):
    if change == "interval":
        computed_tables["pl_scores_ci"].loc[0, "CI95Lower"] = "-9999"
    elif change == "order":
        computed_tables["pl_scores_ci"] = computed_tables["pl_scores_ci"].iloc[::-1].reset_index(drop=True)
    output = tmp_path / "run"
    assert cli.main(arguments(output, replicates=10000)) == (0 if change is None else 1)
    if change != "order":
        report = json.loads((output / "comparison.json").read_text())
        assert report["mode"] == "full_exact"
        assert report["passed"] is (change is None)
        assert report["tables"]["pl_scores_ci.tsv"]["serialized_equal"] is (change is None)


def test_failed_computation_leaves_no_output(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("numerical failure")

    monkeypatch.setattr(cli, "compute_ranking_tables", fail)
    output = tmp_path / "run"
    assert cli.main(arguments(output)) == 1
    assert not output.exists()


def test_changed_source_during_computation_prevents_publication(tmp_path, monkeypatch, computation):
    source = tmp_path / "source.tsv"
    source.write_bytes(SOURCE.read_bytes())
    original_compute = cli.compute_ranking_tables

    def mutate_source(*args, **kwargs):
        result = original_compute(*args, **kwargs)
        source.write_bytes(source.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(cli, "compute_ranking_tables", mutate_source)
    output = tmp_path / "run"
    argv = arguments(output)
    argv[argv.index("--score-tsv") + 1] = str(source)
    assert cli.main(argv) == 1
    assert not output.exists()


def test_destination_changed_during_computation_is_preserved(tmp_path, monkeypatch, computation):
    output = tmp_path / "run"
    original_compute = cli.compute_ranking_tables

    def create_user_file(*args, **kwargs):
        result = original_compute(*args, **kwargs)
        output.mkdir()
        (output / "notes.txt").write_text("concurrent user file\n")
        return result

    monkeypatch.setattr(cli, "compute_ranking_tables", create_user_file)
    assert cli.main(arguments(output)) == 1
    assert (output / "notes.txt").read_text() == "concurrent user file\n"
    assert len(list(output.iterdir())) == 1


def test_full_configuration_mismatch_does_not_silently_use_preview(tmp_path, computation):
    output = tmp_path / "run"
    argv = arguments(output, replicates=10000)
    argv[argv.index("--seed") + 1] = "20260715"
    assert cli.main(argv) == 1
    report = json.loads((output / "comparison.json").read_text())
    assert report["mode"] == "full_exact"
    assert report["full_comparison_eligible"] is False
    assert report["eligibility_failures"]


def test_no_reference_emits_no_comparison(tmp_path, computation):
    output = tmp_path / "run"
    output.mkdir()
    assert cli.main(arguments(output, reference=False)) == 0
    assert not (output / "comparison.json").exists()


def test_publication_failure_preserves_empty_destination(tmp_path, monkeypatch, computation):
    output = tmp_path / "run"
    output.mkdir()

    def fail(*args, **kwargs):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(cli.os, "replace", fail)
    assert cli.main(arguments(output)) == 1
    assert list(output.iterdir()) == []
    assert list(tmp_path.iterdir()) == [output]
