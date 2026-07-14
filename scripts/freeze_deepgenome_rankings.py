from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PL_NOTEBOOK = ROOT / "DeepGenomeAgent Evaluation" / "score_plackett_luce.ipynb"
DEFAULT_MODEL_COLUMNS = ("Gemini", "Grok", "OpenAI", "Phytomni", "Claude")
STUDY_STATUSES = ("well_studied", "uncharacterized")
SPECIES_ORDER = ("Rice", "Maize", "Wheat", "Soybean", "Arabidopsis")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plackett_luce_core(
    notebook_path: Path = PL_NOTEBOOK,
) -> dict[str, Any]:
    notebook = nbformat.read(notebook_path, as_version=4)
    sources = [
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "plackett-luce-core" in cell.metadata.get("tags", [])
    ]
    if not sources:
        raise ValueError(
            f"No plackett-luce-core cell found in {notebook_path}"
        )
    namespace: dict[str, Any] = {"__name__": "deepgenome_plackett_luce_core"}
    source = "\n\n".join(sources)
    exec(compile(source, str(notebook_path), "exec"), namespace)
    return namespace


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
    if frame.duplicated(["Species", "Gene", "Expert"]).any():
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
        raise ValueError(
            "Unknown StudyStatus values: " + ", ".join(unknown_statuses)
        )

    merged = score_frame.merge(
        categories[["Species", "Gene", "StudyStatus"]],
        on=["Species", "Gene"],
        how="left",
        validate="many_to_one",
    )
    if merged["StudyStatus"].isna().any():
        missing_genes = (
            merged.loc[merged["StudyStatus"].isna(), ["Species", "Gene"]]
            .drop_duplicates()
            .head(10)
        )
        labels = [
            f"{row.Species}/{row.Gene}"
            for row in missing_genes.itertuples(index=False)
        ]
        raise ValueError("Unclassified score genes: " + ", ".join(labels))

    score_genes = set(zip(score_frame["Species"], score_frame["Gene"], strict=True))
    category_genes = set(zip(categories["Species"], categories["Gene"], strict=True))
    if score_genes != category_genes:
        raise ValueError(
            "Gene categories must describe exactly the Species/Gene pairs in the score TSV."
        )
    return merged


def ordered_species(values: Iterable[str]) -> list[str]:
    present = set(values)
    known = [species for species in SPECIES_ORDER if species in present]
    unknown = sorted(present - set(known), key=str.casefold)
    return [*known, *unknown]


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


def freeze_rankings(
    *,
    score_path: Path,
    gene_categories_path: Path,
    output_dir: Path,
    model_columns: tuple[str, ...] = DEFAULT_MODEL_COLUMNS,
    notebook_path: Path = PL_NOTEBOOK,
) -> dict[str, Path]:
    score_path = score_path.resolve()
    gene_categories_path = gene_categories_path.resolve()
    output_dir = output_dir.resolve()
    score_frame = pd.read_csv(score_path, sep="\t", dtype=str)
    validate_score_frame(score_frame, model_columns)
    categories = pd.read_csv(gene_categories_path, sep="\t", dtype=str)
    categorized = attach_gene_categories(score_frame, categories)

    core = load_plackett_luce_core(notebook_path)
    parse_rankings = core["parse_rankings"]
    fit_plackett_luce = core["fit_plackett_luce"]
    reference_model = core["REFERENCE_MODEL"]
    if reference_model not in model_columns:
        raise ValueError(
            f"Model columns must contain reference model {reference_model!r}."
        )

    rank_records: list[dict[str, Any]] = []
    score_records: list[dict[str, Any]] = []
    pairwise_records: list[dict[str, Any]] = []
    scope_records: list[dict[str, Any]] = []
    total_skipped = 0

    for scope, study_status, species, subset in scope_frames(categorized):
        rankings, skipped = parse_rankings(subset, model_columns)
        if skipped:
            raise ValueError(
                f"Scope {scope!r} skipped {skipped} rows after strict validation."
            )
        total_skipped += skipped
        fit = fit_plackett_luce(rankings, list(model_columns))
        optimizer = fit["optimizer_result"]
        if not optimizer.success:
            raise RuntimeError(
                f"Plackett-Luce optimization failed for {scope!r}: {optimizer.message}"
            )

        row_count = len(subset)
        scope_records.append(
            {
                "scope": scope,
                "study_status": study_status,
                "species": species,
                "rows": row_count,
            }
        )
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

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "rank_distribution": output_dir / "rank_distribution.tsv",
        "pl_scores": output_dir / "pl_scores.tsv",
        "pl_pairwise": output_dir / "pl_pairwise.tsv",
        "provenance": output_dir / "provenance.json",
    }
    pd.DataFrame(rank_records).to_csv(
        paths["rank_distribution"],
        sep="\t",
        index=False,
        float_format="%.15g",
        lineterminator="\n",
    )
    pd.DataFrame(score_records).to_csv(
        paths["pl_scores"],
        sep="\t",
        index=False,
        float_format="%.15g",
        lineterminator="\n",
    )
    pd.DataFrame(pairwise_records).to_csv(
        paths["pl_pairwise"],
        sep="\t",
        index=False,
        na_rep="",
        float_format="%.15g",
        lineterminator="\n",
    )

    provenance = {
        "schema_version": 1,
        "source": {
            "name": score_path.name,
            "sha256": sha256(score_path),
            "rows": len(score_frame),
        },
        "gene_categories": {
            "name": gene_categories_path.name,
            "sha256": sha256(gene_categories_path),
            "rows": len(categories),
        },
        "scoring_notebook": {
            "name": notebook_path.name,
            "sha256": sha256(notebook_path),
        },
        "model_columns": list(model_columns),
        "reference_model": reference_model,
        "used_rows": len(score_frame),
        "skipped_rows": total_skipped,
        "scopes": scope_records,
        "uncertainty_method": "reference-parameter approximation",
    }
    paths["provenance"].write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def parse_model_columns(value: str) -> tuple[str, ...]:
    columns = tuple(column.strip() for column in value.split(",") if column.strip())
    if len(columns) < 2 or len(columns) != len(set(columns)):
        raise argparse.ArgumentTypeError(
            "Model columns must contain at least two unique names."
        )
    return columns


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze privacy-safe aggregate figure inputs from a private "
            "DeepGenome ranking TSV."
        )
    )
    parser.add_argument("--score-tsv", type=Path, required=True)
    parser.add_argument("--gene-categories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-columns",
        type=parse_model_columns,
        default=DEFAULT_MODEL_COLUMNS,
        help="Comma-separated ranking columns.",
    )
    args = parser.parse_args()
    paths = freeze_rankings(
        score_path=args.score_tsv,
        gene_categories_path=args.gene_categories,
        output_dir=args.output_dir,
        model_columns=args.model_columns,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
