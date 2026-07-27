from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


OUTPUT_FILENAMES = {
    "knowledge": "PhytoBench-Knowledge-for_plot.tsv",
    "data": "PhytoBench-Data-for_plot.tsv",
    "analysis": "PhytoBench-Analysis-for_plot.tsv",
    "provenance": "PhytoBench-Core-for_plot-provenance.json",
}

MODEL_ORDER = (
    "Phyto-Reasoner",
    "Phyto-Chatbot",
    "GPT-5",
    "o3",
    "Gemini-2.5-Pro",
    "Claude-Opus-4.1",
    "Grok-3-Beta",
    "Deepseek-V3",
    "Deepseek-R1",
)

PRIMARY_ANALYSIS_SPECIES = "Oryza_sativa"
FIG2A_MANUSCRIPT_SOURCE = {
    "panel": "Fig. 2a",
    "source_document": "2025-11-31329A-Z_Article_File-20260727.working.md",
    "source_sha256": (
        "2e25c9e62d396c22ef764f8f70cccfdb143617de0f73361de80f95e4613ec964"
    ),
}
FIG2A_MANUSCRIPT_VALUES = {
    "Phyto-Chatbot": {
        "IdentificationAccuracy": 0.72,
        "TraceBLEU4": 0.084,
    }
}
ANALYSIS_SCORE_COLUMNS = {
    "PlanScore": "Plan_Score",
    "ToolScore": "Tool_Score",
    "ParameterScore": "Parameter_Score",
    "RateScore": "Rate_Score",
}

SOURCE_COLUMNS = {
    "Phyto-Reasoner": {
        "knowledge_identification": ("Phyto-Reasoner (hybrid + rerank)",),
        "knowledge_trace": ("Phyto-Reasoner (semantic)",),
        "data": ("Phyto-Reasoner (Knowledge Agent)",),
        "analysis": ("Phyto-Reasoner (Knowledge Agent)",),
    },
    "Phyto-Chatbot": {
        "knowledge_identification": ("Phyto-Chatbot (hybrid + rerank)",),
        "knowledge_trace": ("Phyto-Chatbot (semantic)",),
        "data": ("Phyto-Chatbot (Knowledge Agent)",),
        "analysis": (
            "Phyto-ChatBot (Knowledge Agent)",
            "Phyto-ChatBot (Konwledge Agent)",
        ),
    },
    "GPT-5": {
        "knowledge_identification": ("GPT-5",),
        "knowledge_trace": ("GPT-5",),
        "data": ("GPT-5",),
        "analysis": ("GPT-5",),
    },
    "o3": {
        "knowledge_identification": ("o3",),
        "knowledge_trace": ("o3",),
        "data": ("o3",),
        "analysis": ("o3",),
    },
    "Gemini-2.5-Pro": {
        "knowledge_identification": ("Gemini-2.5-Pro",),
        "knowledge_trace": ("Gemini-2.5-Pro",),
        "data": ("Gemini-2.5-Pro",),
        "analysis": ("Gemini-2.5-Pro",),
    },
    "Claude-Opus-4.1": {
        "knowledge_identification": ("Claude-4.1-Opus",),
        "knowledge_trace": ("Claude-4.1-Opus",),
        "data": ("Claude-Opus-4.1",),
        "analysis": ("Claude-Opus-4.1",),
    },
    "Grok-3-Beta": {
        "knowledge_identification": ("Grok-3-Beta",),
        "knowledge_trace": ("Grok-3-Beta",),
        "data": ("Grok-3-Beta",),
        "analysis": ("Grok-3-Beta",),
    },
    "Deepseek-V3": {
        "knowledge_identification": ("DeepSeek-V3",),
        "knowledge_trace": ("DeepSeek-V3",),
        "data": ("DeepSeek-V3",),
        "analysis": ("DeepSeek-V3",),
    },
    "Deepseek-R1": {
        "knowledge_identification": ("DeepSeek-R1",),
        "knowledge_trace": ("DeepSeek-R1",),
        "data": ("DeepSeek-R1",),
        "analysis": ("DeepSeek-R1",),
    },
}


def _clean_header(value: object) -> str:
    return " ".join(str(value).strip().split())


def _read_workbook(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Workbook does not exist: {path}")
    frame = pd.read_excel(path, header=[1, 2])
    if frame.empty:
        raise ValueError(f"Workbook has no data rows: {path}")
    return frame


def _matching_columns(
    frame: pd.DataFrame,
    model_aliases: Iterable[str],
    metric: str,
) -> list[tuple[object, object]]:
    aliases = {_clean_header(alias) for alias in model_aliases}
    return [
        column
        for column in frame.columns
        if _clean_header(column[0]) in aliases
        and _clean_header(column[1]) == metric
    ]


def _metric_series(
    frame: pd.DataFrame,
    model_aliases: Iterable[str],
    metric: str,
) -> pd.Series:
    matches = _matching_columns(frame, model_aliases, metric)
    if len(matches) != 1:
        aliases = ", ".join(model_aliases)
        raise ValueError(
            f"Expected one {metric!r} column for {aliases}; found {len(matches)}."
        )
    return frame[matches[0]]


def _base_series(frame: pd.DataFrame, name: str) -> pd.Series:
    matches = [
        column
        for column in frame.columns
        if _clean_header(column[0]) == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one base column named {name!r}; found {len(matches)}."
        )
    return frame[matches[0]]


def _numeric_series(
    series: pd.Series,
    *,
    label: str,
    lower: float,
    upper: float,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        invalid_count = int(numeric.isna().sum())
        raise ValueError(
            f"{label} contains {invalid_count} missing or nonnumeric values."
        )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} contains nonfinite values.")
    invalid = ~numeric.between(lower, upper)
    if invalid.any():
        bad = numeric.loc[invalid].head().tolist()
        raise ValueError(
            f"{label} must be within [{lower}, {upper}]; examples: {bad}"
        )
    return numeric.astype(float)


def _judgment_series(series: pd.Series, *, label: str) -> pd.Series:
    def normalize(value: object) -> float:
        if isinstance(value, (bool, np.bool_)):
            return float(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            if value in (0, 1):
                return float(value)
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in {"true", "1"}:
                return 1.0
            if normalized in {"false", "0"}:
                return 0.0
        return np.nan

    normalized = series.map(normalize)
    if normalized.isna().any():
        raise ValueError(
            f"{label} contains {int(normalized.isna().sum())} invalid judgments."
        )
    return normalized.astype(float)


def _validate_unique_key(
    frame: pd.DataFrame,
    series: pd.Series,
    *,
    label: str,
) -> None:
    if series.isna().any():
        raise ValueError(f"{label} contains missing identifiers.")
    if series.duplicated().any():
        raise ValueError(f"{label} contains duplicate identifiers.")
    if len(series) != len(frame):
        raise ValueError(f"{label} does not align with the workbook rows.")


def freeze_core_metrics(
    *,
    knowledge_identification_path: Path,
    knowledge_trace_path: Path,
    data_path: Path,
    analysis_path: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    knowledge_identification = _read_workbook(knowledge_identification_path)
    knowledge_trace = _read_workbook(knowledge_trace_path)
    data = _read_workbook(data_path)
    analysis = _read_workbook(analysis_path)

    _validate_unique_key(
        knowledge_identification,
        _base_series(knowledge_identification, "ID"),
        label="Knowledge-identification ID",
    )
    _validate_unique_key(
        knowledge_trace,
        _base_series(knowledge_trace, "ID"),
        label="Knowledge-trace ID",
    )
    _validate_unique_key(
        data,
        _base_series(data, "Number"),
        label="Data-benchmark Number",
    )

    knowledge_rows: list[dict[str, object]] = []
    data_rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        sources = SOURCE_COLUMNS[model]
        identification = _judgment_series(
            _metric_series(
                knowledge_identification,
                sources["knowledge_identification"],
                "Judgment",
            ),
            label=f"{model} knowledge-identification judgment",
        )
        trace_bleu4 = _numeric_series(
            _metric_series(
                knowledge_trace,
                sources["knowledge_trace"],
                "BLEU-4",
            ),
            label=f"{model} knowledge-trace BLEU-4",
            lower=0,
            upper=1,
        )
        data_judgment = _judgment_series(
            _metric_series(data, sources["data"], "Judgment"),
            label=f"{model} data judgment",
        )
        knowledge_rows.append(
            {
                "Model": model,
                "DisplayLabel": model,
                "IdentificationAccuracy": identification.mean(),
                "IdentificationN": len(identification),
                "TraceBLEU4": trace_bleu4.mean(),
                "TraceN": len(trace_bleu4),
            }
        )
        data_rows.append(
            {
                "Model": model,
                "DisplayLabel": model,
                "Accuracy": data_judgment.mean(),
                "N": len(data_judgment),
            }
        )

    knowledge_output = pd.DataFrame(knowledge_rows)
    alignment_overrides: dict[str, dict[str, dict[str, float]]] = {}
    for model, metrics in FIG2A_MANUSCRIPT_VALUES.items():
        model_rows = knowledge_output["Model"].eq(model)
        if int(model_rows.sum()) != 1:
            raise ValueError(
                f"Expected one Fig. 2a row for manuscript-aligned model {model}."
            )
        alignment_overrides[model] = {}
        for metric, manuscript_value in metrics.items():
            source_value = float(knowledge_output.loc[model_rows, metric].iloc[0])
            alignment_overrides[model][metric] = {
                "source_workbook_value": source_value,
                "manuscript_value": manuscript_value,
            }
            knowledge_output.loc[model_rows, metric] = manuscript_value

    task = _base_series(analysis, "Task").ffill()
    query = _base_series(analysis, "Query").ffill()
    species = _base_series(analysis, "Species")
    rep = _base_series(analysis, "Rep")
    identifiers = (task, query, species, rep)
    if any(series.isna().any() for series in identifiers):
        raise ValueError("Analysis observation identifiers contain missing values.")
    if query.groupby(task, sort=False).nunique().gt(1).any():
        raise ValueError("Each analysis task must map to one query.")

    observation_id = (
        task.astype(str).str.strip()
        + "|"
        + species.astype(str).str.strip()
        + "|"
        + rep.astype(str).str.strip()
    )
    _validate_unique_key(
        analysis,
        observation_id,
        label="Analysis Task-Species-Rep key",
    )

    normalized_species = species.astype(str).str.strip()
    primary_scope = normalized_species.eq(PRIMARY_ANALYSIS_SPECIES)
    primary_task_counts = task.loc[primary_scope].value_counts()
    if (
        int(primary_scope.sum()) != 50
        or len(primary_task_counts) != 10
        or not primary_task_counts.eq(5).all()
    ):
        raise ValueError(
            "The primary Fig. 2c scope must contain 10 tasks with 5 runs each."
        )

    analysis_frames: list[pd.DataFrame] = []
    for model in MODEL_ORDER:
        component_scores = {
            output_name: _numeric_series(
                _metric_series(
                    analysis,
                    SOURCE_COLUMNS[model]["analysis"],
                    source_name,
                ),
                label=f"{model} analysis {source_name}",
                lower=0,
                upper=25,
            )
            for output_name, source_name in ANALYSIS_SCORE_COLUMNS.items()
        }
        total_score = _numeric_series(
            _metric_series(
                analysis,
                SOURCE_COLUMNS[model]["analysis"],
                "Total_Score",
            ),
            label=f"{model} analysis Total_Score",
            lower=0,
            upper=100,
        )
        calculated_total = sum(component_scores.values())
        if not np.allclose(
            total_score.to_numpy(dtype=float),
            calculated_total.to_numpy(dtype=float),
            rtol=0,
            atol=1e-9,
        ):
            raise ValueError(
                f"{model} Total_Score does not equal the four component scores."
            )

        model_frame = pd.DataFrame(
            {
                "Task": task.astype(str).str.strip(),
                "Species": normalized_species,
                "Rep": rep.astype(str).str.strip(),
                "ObservationID": observation_id,
                "Model": model,
                "DisplayLabel": model,
                **component_scores,
                "TotalScore": total_score,
            }
        )
        analysis_frames.append(
            model_frame.loc[primary_scope].reset_index(drop=True)
        )

    outputs = {
        "knowledge": knowledge_output,
        "data": pd.DataFrame(data_rows),
        "analysis": pd.concat(analysis_frames, ignore_index=True),
    }
    provenance: dict[str, object] = {
        "schema_version": 2,
        "metric_definitions": {
            "IdentificationAccuracy": "mean binary judgment",
            "TraceBLEU4": "mean BLEU-4",
            "Accuracy": "mean binary judgment",
            "PlanScore": "task planning accuracy, 0-25",
            "ToolScore": "tool selection precision, 0-25",
            "ParameterScore": "parameter-setting rationality, 0-25",
            "RateScore": "overall task execution success, 0-25",
            "TotalScore": (
                "PlanScore + ToolScore + ParameterScore + RateScore, 0-100"
            ),
        },
        "model_order": list(MODEL_ORDER),
        "analysis_scope": {
            "species": PRIMARY_ANALYSIS_SPECIES,
            "tasks": len(primary_task_counts),
            "repeats_per_task": int(primary_task_counts.iloc[0]),
            "rows": int(primary_scope.sum()),
            "excluded_cross_species_rows": int((~primary_scope).sum()),
        },
        "source_rows": {
            "knowledge_identification": len(knowledge_identification),
            "knowledge_trace": len(knowledge_trace),
            "data": len(data),
            "analysis": len(analysis),
        },
        "source_sha256": {
            "knowledge_identification": _sha256(
                knowledge_identification_path
            ),
            "knowledge_trace": _sha256(knowledge_trace_path),
            "data": _sha256(data_path),
            "analysis": _sha256(analysis_path),
        },
        "source_columns": SOURCE_COLUMNS,
        "manuscript_alignment": {
            **FIG2A_MANUSCRIPT_SOURCE,
            "reason": (
                "The private Supplementary Data 3-new and 4-new workbooks have "
                "not yet been corrected to the manuscript values."
            ),
            "overrides": alignment_overrides,
        },
    }
    return outputs, provenance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(
    outputs: dict[str, pd.DataFrame],
    provenance: dict[str, object],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    for key in ("knowledge", "data", "analysis"):
        path = output_dir / OUTPUT_FILENAMES[key]
        outputs[key].to_csv(
            path,
            sep="\t",
            index=False,
            float_format="%.15g",
            lineterminator="\n",
        )
        output_paths[key] = path

    finalized_provenance = dict(provenance)
    finalized_provenance["outputs"] = {
        path.name: _sha256(path) for path in output_paths.values()
    }
    provenance_path = output_dir / OUTPUT_FILENAMES["provenance"]
    provenance_path.write_text(
        json.dumps(finalized_provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the private Fig. 2a-c workbooks into public plot inputs."
    )
    parser.add_argument("--knowledge-id", type=Path, required=True)
    parser.add_argument("--knowledge-trace", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    outputs, provenance = freeze_core_metrics(
        knowledge_identification_path=args.knowledge_id,
        knowledge_trace_path=args.knowledge_trace,
        data_path=args.data,
        analysis_path=args.analysis,
    )
    write_outputs(outputs, provenance, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
