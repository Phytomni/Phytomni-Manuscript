"""Recompute public ranking statistics into an independent, nonpublication run."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import platform
import shutil
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from scripts.deepgenome_ranking_statistics import (
    AGREEMENT_SPECIES,
    AGREEMENT_STUDY_STATUSES,
    BOOTSTRAP_METADATA_COLUMNS,
    MODEL_COLUMNS,
    PL_PRODUCTION_REPLICATES,
    BootstrapConfig,
    summarize_assignments,
)
from scripts.freeze_deepgenome_rankings import (
    NUMERICAL_OUTPUT_FILENAMES,
    OUTPUT_SCHEMAS,
    OUTPUT_UNIQUE_KEYS,
    ROOT,
    _is_same_or_ancestor,
    _lexical_absolute,
    _reject_symlink_components,
    compute_ranking_tables,
    prepare_public_rankings,
    sha256_bytes,
    table_bytes,
)


LOCAL_RUN_ROOT = ROOT / "local_exports" / "ranking-recomputation"
EXPECTED_ROWS = {
    "rank_distribution": 450, "pl_scores": 90, "pl_pairwise": 450,
    "pl_scores_ci": 270, "rank_distribution_ci": 450, "pl_pairwise_ci": 360,
    "fleiss_kappa": 58, "kendall_by_gene": 200,
    "ordinal_agreement_summary": 18, "top1_consensus": 54,
    "assignment_summary": 18,
}
PREVIEW_EXCLUDED_COLUMNS = {
    *BOOTSTRAP_METADATA_COLUMNS, "Elo_L", "Elo_U",
    "SuccessfulReplicates", "FailedFits",
}


def preflight_output(output: Path, inputs: tuple[Path, ...]) -> Path:
    """Allow a fresh run, never a publication or an input-containing directory."""
    output = _lexical_absolute(output)
    _reject_symlink_components(output)
    if _is_same_or_ancestor(output, ROOT):
        raise ValueError("Output cannot be the repository or any of its ancestors.")
    if _is_same_or_ancestor(ROOT, output) and (
        not _is_same_or_ancestor(LOCAL_RUN_ROOT, output) or output == LOCAL_RUN_ROOT
    ):
        raise ValueError(
            "Repository output must be a fresh run below local_exports/ranking-recomputation."
        )
    for path in inputs:
        lexical = _lexical_absolute(path)
        resolved = lexical.resolve()
        if any(
            _is_same_or_ancestor(output, candidate)
            or _is_same_or_ancestor(candidate, output)
            for candidate in (lexical, resolved)
        ):
            raise ValueError("Output cannot overlap a source or reference location.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be a new or empty independent run directory.")
    return output


def validate_production_cohort(source: pd.DataFrame, categories: pd.DataFrame) -> dict[str, int]:
    frame = prepare_public_rankings(source, categories)
    counts = {
        "assignments": len(frame),
        "experts": int(frame["Expert"].nunique()),
        "genes": len(frame[["Species", "Gene"]].drop_duplicates()),
    }
    if counts != {"assignments": 600, "experts": 120, "genes": 200}:
        raise ValueError("The public cohort must contain 600 assignments, 120 experts and 200 genes.")
    summary = summarize_assignments(frame, MODEL_COLUMNS)
    overall = summary.loc[summary["Scope"] == "overall"].iloc[0]
    if tuple(overall[["MinGenesPerExpert", "MaxGenesPerExpert", "MinExpertsPerGene", "MaxExpertsPerGene"]]) != (5, 5, 3, 3):
        raise ValueError("The public cohort requires five genes per expert and three experts per gene.")
    return counts


def _read_table(payload: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(payload), sep="\t", dtype=str, keep_default_na=False)


def read_reference(directory: Path) -> tuple[dict[str, bytes], dict[str, object], bytes]:
    _reject_symlink_components(_lexical_absolute(directory))
    provenance_path = directory / "provenance.json"
    if provenance_path.is_symlink() or not provenance_path.is_file():
        raise ValueError("Reference provenance must be a regular file, not a symbolic link.")
    provenance_bytes = provenance_path.read_bytes()
    provenance = json.loads(provenance_bytes)
    if not isinstance(provenance, dict) or provenance.get("schema_version") != 2 or provenance.get("model_columns") != list(MODEL_COLUMNS):
        raise ValueError("Reference must be a current five-model schema-2 publication.")
    if not all(isinstance(provenance.get(key), dict) for key in ("outputs", "inputs", "bootstrap")):
        raise ValueError("Reference provenance must contain output, input and bootstrap mappings.")
    recorded = provenance.get("outputs", {})
    payloads = {}
    for name, filename in NUMERICAL_OUTPUT_FILENAMES.items():
        path = directory / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError("Reference numerical tables must be regular files.")
        payload = path.read_bytes()
        if sha256_bytes(payload) != recorded.get(filename):
            raise ValueError("Reference numerical output hash is inconsistent.")
        table = _read_table(payload)
        if tuple(table.columns) != OUTPUT_SCHEMAS[name] or len(table) != EXPECTED_ROWS[name]:
            raise ValueError("Reference numerical output schema or cohort size is inconsistent.")
        payloads[filename] = payload
    return payloads, provenance, provenance_bytes


def compare_tables(
    payloads: dict[str, bytes], reference: dict[str, bytes], *, full: bool,
    eligibility_failures: list[str],
) -> dict[str, object]:
    report = {
        "mode": "full_exact" if full else "point_estimates_and_counts",
        "full_comparison_eligible": not eligibility_failures,
        "eligibility_failures": eligibility_failures,
        "private_panel_compared": False,
        "tables": {},
    }
    for filename, payload in payloads.items():
        actual = _read_table(payload)
        expected = _read_table(reference[filename])
        columns = [
            column for column in expected.columns
            if full or (column not in PREVIEW_EXCLUDED_COLUMNS and "CI" not in column)
        ]
        schema_equal = tuple(actual.columns) == tuple(expected.columns)
        differences = []
        for column in columns:
            if column not in actual:
                continue
            for row in range(min(len(actual), len(expected))):
                value, reference_value = actual.iloc[row][column], expected.iloc[row][column]
                if value == reference_value:
                    continue
                difference = {"row": row + 1, "column": column,
                              "actual": value, "reference": reference_value}
                try:
                    delta = abs(float(value) - float(reference_value))
                    if math.isfinite(delta):
                        difference["absolute_difference"] = delta
                except ValueError:
                    pass
                differences.append(difference)
        serialized_equal = payload == reference[filename]
        passed = schema_equal and len(actual) == len(expected) and not differences
        if full:
            passed = passed and serialized_equal
        report["tables"][filename] = {
            "passed": passed, "schema_equal": schema_equal,
            "actual_rows": len(actual), "reference_rows": len(expected),
            "compared_columns": columns, "serialized_equal": serialized_equal,
            "differences": differences,
        }
    report["passed"] = all(table["passed"] for table in report["tables"].values()) and (
        not full or not eligibility_failures
    )
    return report


def milestone_observer():
    reported = {}

    def observe(event):
        phase = event["phase"]
        milestone = min(10, 10 * event["successful"] // event["total"])
        previous = reported.get(phase, -1)
        for value in range(previous + 1, milestone + 1):
            print(
                f"{phase}: {value * 10}% reached; successful={event['successful']}/"
                f"{event['total']}, attempted={event['attempted']}, failed={event['failed']}",
                file=sys.stderr, flush=True,
            )
        reported[phase] = milestone

    return observe


def _publish_run(output: Path, payloads: dict[str, bytes], inputs: tuple[Path, ...]) -> None:
    preflight_output(output, inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(output.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for filename, payload in payloads.items():
            (staging / filename).write_bytes(payload)
        preflight_output(output, inputs)
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def recompute(args: argparse.Namespace) -> bool:
    input_paths = (args.score_tsv, args.gene_categories)
    protected_inputs = (*input_paths, *((args.reference_dir,) if args.reference_dir else ()))
    output = preflight_output(args.output_dir, protected_inputs)
    config = BootstrapConfig(args.bootstrap_replicates, args.seed, args.max_failed_fits)
    input_payloads = {path: path.read_bytes() for path in input_paths}
    source, categories = (_read_table(input_payloads[path]) for path in input_paths)
    cohort = validate_production_cohort(source, categories)
    input_hashes = {
        "public_ranking_release": {"sha256": sha256_bytes(input_payloads[args.score_tsv])},
        "gene_categories": {"sha256": sha256_bytes(input_payloads[args.gene_categories])},
    }
    reference = None
    reference_provenance = None
    reference_provenance_bytes = None
    if args.reference_dir:
        reference, reference_provenance, reference_provenance_bytes = read_reference(args.reference_dir)
    tables, diagnostics = compute_ranking_tables(
        source, categories, config, progress=milestone_observer(),
    )
    if set(tables) != set(NUMERICAL_OUTPUT_FILENAMES):
        raise ValueError("The public computation must produce exactly eleven numerical tables.")
    payloads = {}
    for name, filename in NUMERICAL_OUTPUT_FILENAMES.items():
        table = tables[name]
        if tuple(table.columns) != OUTPUT_SCHEMAS[name] or len(table) != EXPECTED_ROWS[name]:
            raise ValueError("A computed numerical table has an invalid schema or row count.")
        if table.duplicated(list(OUTPUT_UNIQUE_KEYS[name])).any():
            raise ValueError("A computed numerical table has duplicate keys.")
        payloads[filename] = table_bytes(table)
    full = config.successful_replicates >= PL_PRODUCTION_REPLICATES
    settings = {
        "master_seed": config.seed, "successful_replicates": config.successful_replicates,
        "maximum_failed_fits": config.max_failed_fits,
    }
    comparison = None
    if reference is not None:
        failures = []
        if not full:
            failures.append("Fewer than 10000 successful bootstrap replicates; intervals are provisional.")
        if any(reference_provenance.get("bootstrap", {}).get(key) != value for key, value in settings.items()):
            failures.append("Bootstrap settings differ from the reference.")
        if any(reference_provenance.get("inputs", {}).get(key) != value for key, value in input_hashes.items()):
            failures.append("Public input hashes differ from the reference; anonymous IDs are not remapped.")
        comparison = compare_tables(payloads, reference, full=full, eligibility_failures=failures)
    provenance = {
        "schema_version": 1, "artifact_type": "public_ranking_recomputation",
        "run_mode": "full" if full else "preview", "publication_eligible": False,
        "private_panel_reconstructed": False, "live_model_evaluation_performed": False,
        "inputs": input_hashes, "bootstrap": settings, "bootstrap_diagnostics": diagnostics,
        "cohort": cohort, "model_columns": list(MODEL_COLUMNS),
        "computational_species_order": list(AGREEMENT_SPECIES),
        "computational_study_status_order": list(AGREEMENT_STUDY_STATUSES),
        "versions": {"python": platform.python_version(), **{
            package: version(package) for package in ("numpy", "pandas", "scipy")
        }},
        "code": {name: {"sha256": sha256_bytes((ROOT / "scripts" / name).read_bytes())}
                 for name in ("recompute_deepgenome_rankings.py", "freeze_deepgenome_rankings.py",
                              "deepgenome_ranking_statistics.py")},
        "outputs": {filename: {"sha256": sha256_bytes(payload), "rows": len(tables[name])}
                    for name, filename in NUMERICAL_OUTPUT_FILENAMES.items()
                    for payload in (payloads[filename],)},
        "comparison_mode": comparison["mode"] if comparison else None,
        "comparison_passed": comparison["passed"] if comparison else None,
    }
    if reference is not None:
        provenance["reference"] = {
            "provenance_sha256": sha256_bytes(reference_provenance_bytes),
            "numerical_output_sha256": {filename: sha256_bytes(payload) for filename, payload in reference.items()},
        }
    for path, original in input_payloads.items():
        if path.read_bytes() != original:
            raise ValueError("A public input changed during computation; no output was published.")
    if reference is not None:
        current_reference, _, current_provenance = read_reference(args.reference_dir)
        if current_reference != reference or current_provenance != reference_provenance_bytes:
            raise ValueError("The reference changed during computation; no output was published.")
    payloads["recomputation-provenance.json"] = (json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if comparison is not None:
        payloads["comparison.json"] = (json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _publish_run(output, payloads, protected_inputs)
    return comparison is None or comparison["passed"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--score-tsv", type=Path, required=True)
    parser.add_argument("--gene-categories", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, required=True)
    parser.add_argument("--max-failed-fits", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        passed = recompute(args)
    except (ValueError, RuntimeError, OSError, KeyError, IndexError) as error:
        message = str(error) if isinstance(error, ValueError) else type(error).__name__
        print(f"Recomputation failed: {message}", file=sys.stderr)
        return 1
    if not passed:
        print("Reference comparison failed; inspect the independent comparison.json.", file=sys.stderr)
        return 1
    print("Independent recomputation complete; publication files were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
