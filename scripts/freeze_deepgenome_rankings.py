from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scripts.deepgenome_ranking_statistics import (
    AGREEMENT_SPECIES,
    AGREEMENT_STUDY_STATUSES,
    ASSIGNMENT_SUMMARY_COLUMNS,
    FLEISS_BOOTSTRAP_COLUMNS,
    GENE_ORDINAL_COLUMNS,
    MODEL_COLUMNS,
    ORDINAL_BOOTSTRAP_COLUMNS,
    PANEL_SUMMARY_COLUMNS,
    PL_INTERVAL_ANALYSES,
    PL_PAIRWISE_COLUMNS,
    PL_RANK_COLUMNS,
    PL_SCORE_COLUMNS,
    REFERENCE_MODEL,
    TOP1_BOOTSTRAP_COLUMNS,
    TOP1_PATTERNS,
    BootstrapConfig,
    agreement_scope_registry,
    bootstrap_agreement,
    bootstrap_plackett_luce_statistics,
    fit_plackett_luce,
    gene_ordinal_agreement,
    ordinal_scope_registry,
    parse_rankings,
    ranking_scope_registry,
    summarize_assignments,
    summarize_expert_panel,
)
from scripts.release_deepgenome_rankings import (
    PUBLIC_COLUMNS,
    RELEASE_ID_PATTERN,
)


ROOT = Path(__file__).resolve().parents[1]
PL_NOTEBOOK = ROOT / "DeepGenomeAgent Evaluation" / "score_plackett_luce.ipynb"
STATISTICS_MODULE = Path(__file__).with_name("deepgenome_ranking_statistics.py")
PANEL_AUDIT_MODULE = Path(__file__).with_name("release_deepgenome_rankings.py")
FREEZER_MODULE = Path(__file__)
DEFAULT_MODEL_COLUMNS = MODEL_COLUMNS
STUDY_STATUSES = AGREEMENT_STUDY_STATUSES
SPECIES_ORDER = AGREEMENT_SPECIES
REPORTING_MATRIX_STATEMENT = (
    "The reporting matrix was locked before the final bootstrap reanalysis "
    "and before manuscript interpretation."
)
OUTPUT_FILENAMES = {
    "rank_distribution": "rank_distribution.tsv",
    "pl_scores": "pl_scores.tsv",
    "pl_pairwise": "pl_pairwise.tsv",
    "pl_scores_ci": "pl_scores_ci.tsv",
    "rank_distribution_ci": "rank_distribution_ci.tsv",
    "pl_pairwise_ci": "pl_pairwise_ci.tsv",
    "fleiss_kappa": "fleiss_kappa.tsv",
    "kendall_by_gene": "kendall_by_gene.tsv",
    "ordinal_agreement_summary": "ordinal_agreement_summary.tsv",
    "top1_consensus": "top1_consensus.tsv",
    "expert_panel_summary": "expert_panel_summary.tsv",
    "assignment_summary": "assignment_summary.tsv",
}
OUTPUT_SCHEMAS = {
    "rank_distribution": (
        "Scope",
        "StudyStatus",
        "Species",
        "Model",
        "Rank",
        "Count",
        "Fraction",
        "N",
    ),
    "pl_scores": (
        "Scope",
        "StudyStatus",
        "Species",
        "Model",
        "Elo",
        "Elo_L",
        "Elo_U",
        "N",
    ),
    "pl_pairwise": (
        "Scope",
        "StudyStatus",
        "Species",
        "RowModel",
        "ColumnModel",
        "Probability",
        "N",
    ),
    "pl_scores_ci": PL_SCORE_COLUMNS,
    "rank_distribution_ci": PL_RANK_COLUMNS,
    "pl_pairwise_ci": PL_PAIRWISE_COLUMNS,
    "fleiss_kappa": FLEISS_BOOTSTRAP_COLUMNS,
    "kendall_by_gene": GENE_ORDINAL_COLUMNS,
    "ordinal_agreement_summary": ORDINAL_BOOTSTRAP_COLUMNS,
    "top1_consensus": TOP1_BOOTSTRAP_COLUMNS,
    "expert_panel_summary": PANEL_SUMMARY_COLUMNS,
    "assignment_summary": ASSIGNMENT_SUMMARY_COLUMNS,
}
OUTPUT_UNIQUE_KEYS = {
    "rank_distribution": ("Scope", "Model", "Rank"),
    "pl_scores": ("Scope", "Model"),
    "pl_pairwise": ("Scope", "RowModel", "ColumnModel"),
    "pl_scores_ci": ("Scope", "Model", "IntervalAnalysis"),
    "rank_distribution_ci": ("Scope", "Model", "Rank"),
    "pl_pairwise_ci": ("Scope", "RowModel", "ColumnModel"),
    "fleiss_kappa": ("ScopeID",),
    "kendall_by_gene": ("Species", "Gene"),
    "ordinal_agreement_summary": ("ScopeID",),
    "top1_consensus": ("ScopeID", "Top1AgreementPattern"),
    "expert_panel_summary": ("Dimension", "PublicCategory"),
    "assignment_summary": ("Scope",),
}
OUTPUT_SORT_KEYS = {
    "rank_distribution": ("Scope", "Model", "Rank"),
    "pl_scores": ("Scope", "Model"),
    "pl_pairwise": ("Scope", "RowModel", "ColumnModel"),
    "pl_scores_ci": ("Scope", "Model", "IntervalAnalysis"),
    "rank_distribution_ci": ("Scope", "Model", "Rank"),
    "pl_pairwise_ci": ("Scope", "RowModel", "ColumnModel"),
    "fleiss_kappa": ("ScopeID",),
    "kendall_by_gene": ("Species", "Gene"),
    "ordinal_agreement_summary": ("ScopeID",),
    "top1_consensus": ("ScopeID", "Top1AgreementPattern"),
    "expert_panel_summary": (
        "Dimension",
        "DisplayOrder",
        "PublicCategory",
    ),
    "assignment_summary": ("Scope",),
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def input_provenance(payload: bytes) -> dict[str, str]:
    return {"sha256": sha256_bytes(payload)}


def validate_public_ranking_release(source: pd.DataFrame) -> None:
    if tuple(source.columns) != PUBLIC_COLUMNS:
        raise ValueError(
            "The public ranking release must use the exact canonical schema."
        )
    if source.empty:
        raise ValueError("The public ranking release is empty.")
    valid_cells = source.map(
        lambda value: (
            isinstance(value, str)
            and bool(value)
            and value == value.strip()
        )
    )
    if not valid_cells.to_numpy().all():
        raise ValueError(
            "The public ranking release must contain nonempty trimmed strings."
        )
    if set(source["Species"]) != set(AGREEMENT_SPECIES):
        raise ValueError(
            "The public ranking release must use canonical species values."
        )
    if set(source["StudyStatus"]) != set(AGREEMENT_STUDY_STATUSES):
        raise ValueError(
            "The public ranking release must use canonical study statuses."
        )

    release_ids = source["AnonymousExpertID"].drop_duplicates().tolist()
    if not all(RELEASE_ID_PATTERN.fullmatch(value) for value in release_ids):
        raise ValueError(
            "The public ranking release must use canonical anonymous IDs."
        )
    expected_ids = {
        f"E{index:03d}" for index in range(1, len(release_ids) + 1)
    }
    if set(release_ids) != expected_ids:
        raise ValueError(
            "The public ranking release must use contiguous anonymous IDs."
        )


def validate_score_frame(
    frame: pd.DataFrame,
    model_columns: tuple[str, ...],
) -> None:
    required = {"Species", "Gene", "Expert", *model_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing score columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("The score TSV is empty.")
    identifiers = ["Species", "Gene", "Expert"]
    if frame[identifiers].isna().any().any():
        raise ValueError("Score identifiers must all be nonmissing.")
    if frame.duplicated(identifiers).any():
        raise ValueError("Duplicate Species/Gene/Expert judgments are not allowed.")

    expected_ranks = {f"R{rank}" for rank in range(1, len(model_columns) + 1)}
    valid = frame[list(model_columns)].apply(
        lambda row: set(row.tolist()) == expected_ranks,
        axis=1,
    )
    if not valid.all():
        bad_rows = [str(index + 2) for index in frame.index[~valid][:10]]
        raise ValueError(
            "Every model row must be a complete permutation of "
            f"{sorted(expected_ranks)}; invalid TSV rows: {', '.join(bad_rows)}"
        )


def attach_gene_categories(
    score_frame: pd.DataFrame,
    categories: pd.DataFrame,
) -> pd.DataFrame:
    required = {"Species", "Gene", "StudyStatus"}
    missing = sorted(required - set(categories.columns))
    if missing:
        raise ValueError(f"Missing gene-category columns: {', '.join(missing)}")
    if categories.duplicated(["Species", "Gene"]).any():
        raise ValueError("Gene categories must be unique by Species/Gene.")
    unknown_statuses = sorted(set(categories["StudyStatus"]) - set(STUDY_STATUSES))
    if unknown_statuses:
        raise ValueError("Gene categories contain an unknown StudyStatus value.")

    working = score_frame.copy()
    has_source_status = "StudyStatus" in working.columns
    if has_source_status:
        working = working.rename(columns={"StudyStatus": "_SourceStudyStatus"})
    merged = working.merge(
        categories[["Species", "Gene", "StudyStatus"]],
        on=["Species", "Gene"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if merged["StudyStatus"].isna().any():
        raise ValueError("Every score gene must have one public study status.")
    if has_source_status:
        if not merged["_SourceStudyStatus"].equals(merged["StudyStatus"]):
            raise ValueError(
                "Public ranking and gene-category study statuses must match."
            )
        merged = merged.drop(columns="_SourceStudyStatus")

    score_genes = set(zip(score_frame["Species"], score_frame["Gene"], strict=True))
    category_genes = set(zip(categories["Species"], categories["Gene"], strict=True))
    if score_genes != category_genes:
        raise ValueError(
            "Gene categories must describe exactly the Species/Gene pairs "
            "in the score TSV."
        )
    return merged


def ordered_species(values: Iterable[str]) -> list[str]:
    present = set(values)
    known = [species for species in SPECIES_ORDER if species in present]
    unknown = sorted(present - set(known), key=str.casefold)
    return [*known, *unknown]


def canonicalize_analysis_frame(frame: pd.DataFrame) -> pd.DataFrame:
    species_order = {
        species: index for index, species in enumerate(AGREEMENT_SPECIES)
    }
    status_order = {
        status: index
        for index, status in enumerate(AGREEMENT_STUDY_STATUSES)
    }
    working = frame.assign(
        _SpeciesOrder=frame["Species"].map(species_order),
        _StatusOrder=frame["StudyStatus"].map(status_order),
    )
    if working[["_SpeciesOrder", "_StatusOrder"]].isna().any().any():
        raise ValueError(
            "Ranking analysis requires canonical species and study statuses."
        )
    return (
        working.sort_values(
            ["_SpeciesOrder", "_StatusOrder", "Gene", "Expert"],
            kind="stable",
        )
        .drop(columns=["_SpeciesOrder", "_StatusOrder"])
        .reset_index(drop=True)
    )


def scope_frames(frame: pd.DataFrame) -> list[tuple[str, str, str, pd.DataFrame]]:
    species_values = ordered_species(frame["Species"].unique())
    scopes: list[tuple[str, str, str, pd.DataFrame]] = [
        ("overall", "all", "all", frame)
    ]
    for status in STUDY_STATUSES:
        status_frame = frame[frame["StudyStatus"] == status]
        if status_frame.empty:
            raise ValueError(f"No rows are available for StudyStatus {status!r}.")
        scopes.append((status, status, "all", status_frame))
        for species in species_values:
            subset = status_frame[status_frame["Species"] == species]
            if subset.empty:
                continue
            scopes.append(
                (
                    f"{status}.{species.casefold().replace(' ', '_')}",
                    status,
                    species,
                    subset,
                )
            )
    for species in species_values:
        subset = frame[frame["Species"] == species]
        scopes.append(
            (
                species.casefold().replace(" ", "_"),
                "all",
                species,
                subset,
            )
        )
    return scopes


def point_estimate_tables(
    frame: pd.DataFrame,
    model_columns: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    if REFERENCE_MODEL not in model_columns:
        raise ValueError(
            f"Model columns must contain reference model {REFERENCE_MODEL!r}."
        )

    rank_records: list[dict[str, Any]] = []
    score_records: list[dict[str, Any]] = []
    pairwise_records: list[dict[str, Any]] = []
    for scope, study_status, species, subset in scope_frames(frame):
        rankings, skipped = parse_rankings(subset, model_columns)
        if skipped:
            raise ValueError(
                f"Scope {scope!r} skipped {skipped} rows after strict validation."
            )
        fit = fit_plackett_luce(rankings, list(model_columns))
        optimizer = fit["optimizer_result"]
        if not optimizer.success:
            raise RuntimeError(
                f"Plackett-Luce optimization failed for {scope!r}: "
                f"{optimizer.message}"
            )

        row_count = len(subset)
        for model in model_columns:
            counts = subset[model].value_counts()
            for rank in range(1, len(model_columns) + 1):
                rank_label = f"R{rank}"
                count = int(counts.get(rank_label, 0))
                rank_records.append(
                    {
                        "Scope": scope,
                        "StudyStatus": study_status,
                        "Species": species,
                        "Model": model,
                        "Rank": rank_label,
                        "Count": count,
                        "Fraction": count / row_count,
                        "N": row_count,
                    }
                )

        fit_models = list(fit["models"])
        fit_index = {model: index for index, model in enumerate(fit_models)}
        for model in model_columns:
            index = fit_index[model]
            score_records.append(
                {
                    "Scope": scope,
                    "StudyStatus": study_status,
                    "Species": species,
                    "Model": model,
                    "Elo": float(fit["elo"][index]),
                    "Elo_L": float(fit["elo_lower"][index]),
                    "Elo_U": float(fit["elo_upper"][index]),
                    "N": row_count,
                }
            )
            for column_model in model_columns:
                column_index = fit_index[column_model]
                probability = fit["pairwise_probabilities"][index, column_index]
                pairwise_records.append(
                    {
                        "Scope": scope,
                        "StudyStatus": study_status,
                        "Species": species,
                        "RowModel": model,
                        "ColumnModel": column_model,
                        "Probability": float(probability),
                        "N": row_count,
                    }
                )
    return {
        "rank_distribution": pd.DataFrame(rank_records),
        "pl_scores": pd.DataFrame(score_records),
        "pl_pairwise": pd.DataFrame(pairwise_records),
    }


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonicalize_output_table(
    name: str,
    frame: pd.DataFrame,
    category_map: pd.DataFrame,
) -> pd.DataFrame:
    expected_schema = OUTPUT_SCHEMAS[name]
    if tuple(frame.columns) != tuple(expected_schema):
        raise RuntimeError(f"Output table {name!r} has an unexpected schema.")
    unique_keys = list(OUTPUT_UNIQUE_KEYS[name])
    if frame.duplicated(unique_keys).any():
        raise RuntimeError(f"Output table {name!r} has duplicate canonical keys.")
    if frame.loc[:, unique_keys].isna().any().any():
        raise RuntimeError(f"Output table {name!r} has a missing canonical key.")

    ranking_scopes = tuple(scope.scope for scope in ranking_scope_registry())
    fleiss_scopes = tuple(
        scope.scope_id
        for scope in agreement_scope_registry(
            AGREEMENT_SPECIES,
            AGREEMENT_STUDY_STATUSES,
            MODEL_COLUMNS,
        )
    )
    ordinal_scopes = tuple(scope.scope_id for scope in ordinal_scope_registry())
    dimensions = tuple(category_map["Dimension"].drop_duplicates())
    ordered_values: dict[str, tuple[object, ...]] = {
        "Model": MODEL_COLUMNS,
        "RowModel": MODEL_COLUMNS,
        "ColumnModel": MODEL_COLUMNS,
        "Rank": tuple(f"R{rank}" for rank in range(1, 6)),
        "IntervalAnalysis": PL_INTERVAL_ANALYSES,
        "Species": AGREEMENT_SPECIES,
        "Top1AgreementPattern": TOP1_PATTERNS,
        "Dimension": dimensions,
    }
    if "Scope" in frame.columns:
        ordered_values["Scope"] = ranking_scopes
    if "ScopeID" in frame.columns:
        ordered_values["ScopeID"] = (
            fleiss_scopes if name == "fleiss_kappa" else ordinal_scopes
        )

    working = frame.copy()
    sort_columns: list[str] = []
    helper_columns: list[str] = []
    for index, key in enumerate(OUTPUT_SORT_KEYS[name]):
        if key not in ordered_values:
            sort_columns.append(key)
            continue
        helper = f"_CanonicalOrder{index}"
        order = {value: position for position, value in enumerate(ordered_values[key])}
        working[helper] = working[key].map(order)
        if working[helper].isna().any():
            raise RuntimeError(
                f"Output table {name!r} has a noncanonical {key} value."
            )
        sort_columns.append(helper)
        helper_columns.append(helper)
    return (
        working.sort_values(sort_columns, kind="stable")
        .drop(columns=helper_columns)
        .reset_index(drop=True)
    )


def write_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        sep="\t",
        index=False,
        na_rep="",
        float_format="%.15g",
        lineterminator="\n",
    )


def _pl_failure_reason_code(reason: object) -> str:
    normalized = str(reason).casefold()
    if "zero effective weight" in normalized:
        return "zero_effective_weight"
    if "optimization failed" in normalized:
        return "optimizer_failure"
    if "nonfinite" in normalized:
        return "nonfinite_output"
    return "numerical_failure"


def public_pl_diagnostics(diagnostics: dict[str, object]) -> dict[str, object]:
    return {
        "AttemptedReplicates": _json_safe(
            diagnostics.get("AttemptedReplicates", 0)
        ),
        "SuccessfulReplicates": _json_safe(
            diagnostics.get("SuccessfulReplicates", 0)
        ),
        "FailedFits": _json_safe(diagnostics.get("FailedFits", 0)),
        "FailureReasonCodes": [
            _pl_failure_reason_code(reason)
            for reason in diagnostics.get("FailureReasons", ())
        ],
        "SeedStream": _json_safe(diagnostics.get("SeedStream", "")),
        "ExpertSeedSpawnKey": _json_safe(
            diagnostics.get("ExpertSeedSpawnKey", ())
        ),
        "GeneSeedSpawnKey": _json_safe(
            diagnostics.get("GeneSeedSpawnKey", ())
        ),
        "HalfRunStability": _json_safe(
            diagnostics.get("HalfRunStability", {"Applied": False})
        ),
    }


def validate_staged_publication(staging_dir: Path) -> None:
    expected_files = {*OUTPUT_FILENAMES.values(), "provenance.json"}
    entries = list(staging_dir.iterdir())
    if (
        {entry.name for entry in entries} != expected_files
        or not all(entry.is_file() for entry in entries)
    ):
        raise RuntimeError(
            "The staged frozen publication must contain exactly 13 files."
        )

    provenance_bytes = (staging_dir / "provenance.json").read_bytes()
    try:
        provenance = json.loads(provenance_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("The staged provenance is invalid.") from error
    recorded_outputs = provenance.get("outputs")
    if not isinstance(recorded_outputs, dict) or set(recorded_outputs) != set(
        OUTPUT_FILENAMES.values()
    ):
        raise RuntimeError("The staged provenance output registry is incomplete.")
    for filename, recorded_digest in recorded_outputs.items():
        payload = (staging_dir / filename).read_bytes()
        if not payload or sha256_bytes(payload) != recorded_digest:
            raise RuntimeError("A staged output hash is inconsistent.")


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _publish_staged_directory(staging_dir: Path, output_dir: Path) -> None:
    if output_dir.exists() and (
        not output_dir.is_dir() or output_dir.is_symlink()
    ):
        raise ValueError("The frozen output path must be a directory.")

    backup_dir: Path | None = None
    if output_dir.exists():
        backup_dir = output_dir.parent / (
            f".{output_dir.name}.backup-{uuid.uuid4().hex}"
        )
        os.replace(output_dir, backup_dir)
    try:
        os.replace(staging_dir, output_dir)
    except Exception:
        if backup_dir is not None:
            try:
                os.replace(backup_dir, output_dir)
            except Exception as rollback_error:
                raise RuntimeError(
                    "Frozen output publication and rollback both failed."
                ) from rollback_error
        raise
    if backup_dir is not None:
        _remove_path(backup_dir)


def freeze_rankings(
    *,
    score_path: Path,
    gene_categories_path: Path,
    expert_metadata_path: Path,
    private_lineage_score_path: Path | None,
    panel_category_map_path: Path,
    output_dir: Path,
    model_columns: tuple[str, ...] = DEFAULT_MODEL_COLUMNS,
    expert_column: str = "AnonymousExpertID",
    bootstrap_config: BootstrapConfig | None = None,
) -> dict[str, Path]:
    config = bootstrap_config or BootstrapConfig()
    model_columns = tuple(model_columns)
    if expert_column != "AnonymousExpertID" or model_columns != MODEL_COLUMNS:
        raise ValueError(
            "Frozen ranking analysis requires the canonical public schema."
        )
    score_path = score_path.resolve()
    gene_categories_path = gene_categories_path.resolve()
    expert_metadata_path = expert_metadata_path.resolve()
    panel_category_map_path = panel_category_map_path.resolve()
    lineage_path = (
        private_lineage_score_path.resolve()
        if private_lineage_score_path is not None
        else None
    )
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and (
        not output_dir.is_dir() or output_dir.is_symlink()
    ):
        raise ValueError("The frozen output path must be a directory.")

    input_snapshots: dict[str, bytes | None] = {
        "public_ranking_release": score_path.read_bytes(),
        "private_lineage_score": (
            lineage_path.read_bytes() if lineage_path is not None else None
        ),
        "gene_categories": gene_categories_path.read_bytes(),
        "expert_metadata": expert_metadata_path.read_bytes(),
        "panel_category_map": panel_category_map_path.read_bytes(),
        "statistical_module": STATISTICS_MODULE.read_bytes(),
        "panel_category_audit_module": PANEL_AUDIT_MODULE.read_bytes(),
        "freezer_module": FREEZER_MODULE.read_bytes(),
        "narrative_notebook": PL_NOTEBOOK.read_bytes(),
    }
    input_records = {
        name: input_provenance(payload) if payload is not None else None
        for name, payload in input_snapshots.items()
    }

    source = pd.read_csv(
        BytesIO(input_snapshots["public_ranking_release"]),
        sep="\t",
        dtype=str,
    )
    validate_public_ranking_release(source)
    score_frame = source.rename(columns={"AnonymousExpertID": "Expert"})
    validate_score_frame(score_frame, model_columns)

    categories = pd.read_csv(
        BytesIO(input_snapshots["gene_categories"]),
        sep="\t",
        dtype=str,
    )
    categorized = attach_gene_categories(score_frame, categories)
    metadata = pd.read_excel(
        BytesIO(input_snapshots["expert_metadata"]),
        dtype=object,
    )
    category_map = pd.read_csv(
        BytesIO(input_snapshots["panel_category_map"]),
        sep="\t",
        dtype=str,
    )
    panel = summarize_expert_panel(metadata, category_map)
    if metadata["Expert_ID"].nunique() != categorized["Expert"].nunique():
        raise ValueError(
            "Expert metadata and the public ranking release must have equal "
            "panel sizes."
        )
    metadata_species_counts = (
        metadata.groupby("Species", dropna=False)["Expert_ID"].nunique()
    )
    ranking_species_counts = (
        categorized.groupby("Species", dropna=False)["Expert"].nunique()
    )
    canonical_species = list(AGREEMENT_SPECIES)
    if (
        set(metadata_species_counts.index) != set(canonical_species)
        or set(ranking_species_counts.index) != set(canonical_species)
        or not metadata_species_counts.reindex(canonical_species).equals(
            ranking_species_counts.reindex(canonical_species)
        )
    ):
        raise ValueError(
            "Expert metadata and the public ranking release must have equal "
            "aggregate panel sizes by species."
        )
    assignment = summarize_assignments(categorized, model_columns)
    categorized = canonicalize_analysis_frame(categorized)

    tables = point_estimate_tables(categorized, model_columns)
    pl_bootstrap = bootstrap_plackett_luce_statistics(
        categorized,
        model_columns,
        config,
    )
    agreement = bootstrap_agreement(categorized, model_columns, config)
    tables.update(pl_bootstrap)
    tables.update(
        {
            "fleiss_kappa": agreement["fleiss_kappa"],
            "kendall_by_gene": gene_ordinal_agreement(
                categorized,
                model_columns,
            ),
            "ordinal_agreement_summary": agreement["ordinal_summary"],
            "top1_consensus": agreement["top1_consensus"],
            "expert_panel_summary": panel,
            "assignment_summary": assignment,
        }
    )
    if set(tables) != set(OUTPUT_FILENAMES):
        raise RuntimeError("The freezer did not construct the complete output set.")
    tables = {
        name: canonicalize_output_table(name, table, category_map)
        for name, table in tables.items()
    }

    pl_diagnostics = pl_bootstrap["pl_scores_ci"].attrs.get(
        "bootstrap_diagnostics",
        {},
    )
    fleiss_first = agreement["fleiss_kappa"].iloc[0]
    provenance = {
        "schema_version": 2,
        "reporting_matrix_status": "locked_before_final_bootstrap_reanalysis",
        "reporting_matrix_statement": REPORTING_MATRIX_STATEMENT,
        "pilot_kappa_values_viewed": True,
        "agreement_item_definition": "Species + Gene + Model",
        "agreement_categories": ["R1", "R2", "R3", "R4", "R5"],
        "fleiss_scope_count": len(agreement["fleiss_kappa"]),
        "ordinal_scope_count": len(agreement["ordinal_summary"]),
        "model_columns": list(model_columns),
        "reference_model": REFERENCE_MODEL,
        "bootstrap": {
            "master_seed": config.seed,
            "successful_replicates": config.successful_replicates,
            "maximum_failed_fits": config.max_failed_fits,
            "primary_pl_interval": "crossed_expert_gene_percentile",
            "agreement_interval": "stratified_gene_block_percentile",
        },
        "bootstrap_diagnostics": {
            "plackett_luce": public_pl_diagnostics(pl_diagnostics),
            "agreement": {
                "attempted_replicates": int(
                    fleiss_first["BootstrapAttempted"]
                ),
                "successful_replicates": int(
                    fleiss_first["BootstrapReplicates"]
                ),
                "invalid_replicates": int(fleiss_first["BootstrapInvalid"]),
                "seed_stream": fleiss_first["SeedStream"],
            },
        },
        "inputs": input_records,
    }
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    staging_dir.chmod(0o755)
    try:
        staging_paths = {
            key: staging_dir / filename
            for key, filename in OUTPUT_FILENAMES.items()
        }
        for key in OUTPUT_FILENAMES:
            write_table(tables[key], staging_paths[key])
        provenance["outputs"] = {
            path.name: sha256_bytes(path.read_bytes())
            for path in staging_paths.values()
        }
        (staging_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_staged_publication(staging_dir)
        _publish_staged_directory(staging_dir, output_dir)
    finally:
        _remove_path(staging_dir)

    paths = {
        key: output_dir / filename
        for key, filename in OUTPUT_FILENAMES.items()
    }
    paths["provenance"] = output_dir / "provenance.json"
    return paths


def parse_model_columns(value: str) -> tuple[str, ...]:
    columns = tuple(column.strip() for column in value.split(",") if column.strip())
    if columns != MODEL_COLUMNS:
        raise ValueError("Model columns must use the canonical order.")
    return columns


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze reviewer-reproducible aggregate statistics from the public "
            "DeepGenome ranking release and private panel metadata."
        )
    )
    parser.add_argument("--score-tsv", type=Path, required=True)
    parser.add_argument("--expert-column", default="AnonymousExpertID")
    parser.add_argument("--gene-categories", type=Path, required=True)
    parser.add_argument("--expert-metadata", type=Path, required=True)
    parser.add_argument("--private-lineage-score", type=Path, required=True)
    parser.add_argument("--panel-category-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-columns",
        default=",".join(DEFAULT_MODEL_COLUMNS),
        help="Comma-separated ranking columns.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-failed-fits", type=int, default=10)
    args = parser.parse_args()
    if args.expert_column != "AnonymousExpertID":
        parser.error("Expert column must use the canonical public schema.")
    try:
        model_columns = parse_model_columns(args.model_columns)
    except ValueError:
        parser.error("Model columns must use the canonical order.")
    paths = freeze_rankings(
        score_path=args.score_tsv,
        gene_categories_path=args.gene_categories,
        expert_metadata_path=args.expert_metadata,
        private_lineage_score_path=args.private_lineage_score,
        panel_category_map_path=args.panel_category_map,
        output_dir=args.output_dir,
        model_columns=model_columns,
        expert_column=args.expert_column,
        bootstrap_config=BootstrapConfig(
            successful_replicates=args.bootstrap_replicates,
            seed=args.seed,
            max_failed_fits=args.max_failed_fits,
        ),
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
