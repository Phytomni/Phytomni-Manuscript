"""Input validation helpers for the Fig. 2 gene-metric workflow.

The source workbook and Claude response archive are deliberately kept outside
the tracked data boundary.  This module validates their schemas and returns
in-memory values so downstream scoring code never needs to extract private
response files to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Iterable
from zipfile import ZipFile

import numpy as np
import pandas as pd


MODEL_ORDER = ["Phytomni", "Gemini", "Claude", "OpenAI", "Grok"]
DISPLAY_LABELS = {
    "Phytomni": "Phytomni",
    "Gemini": "Gemini Deep Research",
    "Claude": "Claude deep research",
    "OpenAI": "ChatGPT Agent mode",
    "Grok": "Grok DeepSearch",
}
EXPECTED_HISTORICAL_MEANS = {
    "bertscore": {
        "Phytomni": 0.5616412407159809,
        "Gemini": 0.519520597755909,
        "OpenAI": 0.5082760798931122,
        "Grok": 0.5571489349007607,
    },
    "hallucination": {
        "Phytomni": 0.12216996785802783,
        "Gemini": 0.43429823819272273,
        "OpenAI": 0.4562070516040867,
        "Grok": 0.6445983597035421,
    },
}


_SOURCE_COLUMNS = [
    "Species",
    "GeneID",
    "StudyStatus",
    "Query",
    "PhytomniBERTScorePrecision",
    "PhytomniHallucination",
    "GeminiBERTScorePrecision",
    "GeminiHallucination",
    "OpenAIBERTScorePrecision",
    "OpenAIHallucination",
    "GrokBERTScorePrecision",
    "GrokHallucination",
]
_SOURCE_POSITIONS = [0, 1, 2, 3, 7, 8, 12, 13, 17, 18, 22, 23]
_MODEL_ANCHORS = {4: "Phytomni", 9: "Gemini", 14: "OpenAI", 19: "Grok"}
_METRIC_LABELS = {
    4: "Human-1",
    5: "Human-2",
    6: "Human-3",
    7: "BERTScore precision",
    8: "Hallucination",
    9: "Human-1",
    10: "Human-2",
    11: "Human-3",
    12: "BERTScore precision",
    13: "Hallucination",
    14: "Human-1",
    15: "Human-2",
    16: "Human-3",
    17: "BERTScore precision",
    18: "Hallucination",
    19: "Human-1",
    20: "Human-2",
    21: "Human-3",
    22: "BERTScore precision",
    23: "Hallucination",
}
_STATUS_ALIASES = {
    "well-studied": "well_studied",
    "well_studied": "well_studied",
    "uncharacterized": "uncharacterized",
}
_HISTORICAL_COLUMNS = {
    "bertscore": {
        "Phytomni": "PhytomniBERTScorePrecision",
        "Gemini": "GeminiBERTScorePrecision",
        "OpenAI": "OpenAIBERTScorePrecision",
        "Grok": "GrokBERTScorePrecision",
    },
    "hallucination": {
        "Phytomni": "PhytomniHallucination",
        "Gemini": "GeminiHallucination",
        "OpenAI": "OpenAIHallucination",
        "Grok": "GrokHallucination",
    },
}
_HISTORICAL_STATUSES = {
    "bertscore": "well_studied",
    "hallucination": "uncharacterized",
}

_CLAUDE_LOG_FILENAME = "claude__{gene}__rep_1-rep_2-rep_3.json"
_PAIR_COLUMNS = [
    "Model",
    "Gene",
    "SourceResponse",
    "TargetResponse",
    "SupportRatio",
    "ContradictionRatio",
    "EntailmentEstablished",
    "JudgmentLogSHA256",
]
_GENE_HALLUCINATION_COLUMNS = [
    "Model",
    "Gene",
    "MeanDirectionalContradictionRatio",
    "HighContradiction",
]
_KNOWN_ARCHIVE_ANOMALIES = [
    "Os01g0107900-R1 is byte-identical to Os01g0107900-R3.",
    "Os06g0665200-R1 is byte-identical to Os06g0665200-R3.",
    "Zm00001eb140160-R1 is byte-identical to Zm00001eb063410-R1, and its content describes the latter gene.",
]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def _normalise_status(value: object) -> str:
    if pd.isna(value):
        return ""
    return _STATUS_ALIASES.get(str(value).strip().lower(), "")


def load_gene_categories(path: Path) -> pd.DataFrame:
    """Load and validate the authoritative gene-to-study-status table."""

    _require_file(path, "Gene-category table")
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = ["Species", "Gene", "StudyStatus"]
    if frame.columns.tolist() != required:
        raise ValueError(
            "Gene-category table must contain exactly the columns "
            f"{required}, got {frame.columns.tolist()}"
        )
    for column in required:
        frame[column] = frame[column].astype("string").str.strip()
    frame["StudyStatus"] = frame["StudyStatus"].map(_normalise_status)
    if frame.empty:
        raise ValueError("Gene-category table is empty")
    if frame["Gene"].eq("").any() or frame["Gene"].duplicated().any():
        raise ValueError("Gene-category table contains empty or duplicate gene IDs")
    if frame["Species"].eq("").any():
        raise ValueError("Gene-category table contains empty species values")
    if frame["StudyStatus"].eq("").any():
        raise ValueError("Gene-category table contains an unsupported study status")
    return frame


def _cell(frame: pd.DataFrame, row: int, column: int) -> object:
    try:
        return frame.iat[row, column]
    except IndexError as exc:
        raise ValueError("Source workbook does not contain the expected header layout") from exc


def load_source_workbook(path: Path) -> pd.DataFrame:
    """Read the merged-header source workbook into normalized columns."""

    _require_file(path, "Source workbook")
    raw = pd.read_excel(path, header=None)
    if raw.shape[0] < 4 or raw.shape[1] < 24:
        raise ValueError("Source workbook does not contain the expected 24-column layout")

    expected_metadata = ["Species", "Gene ID", "Gene type", "Query"]
    observed_metadata = [str(_cell(raw, 1, column)).strip() for column in range(4)]
    if observed_metadata != expected_metadata:
        raise ValueError(
            f"Source workbook metadata header mismatch: {observed_metadata!r}"
        )

    for column, expected_model in _MODEL_ANCHORS.items():
        observed = str(_cell(raw, 1, column)).strip()
        if observed != expected_model:
            raise ValueError(
                f"Source workbook model header mismatch at column {column}: "
                f"expected {expected_model!r}, got {observed!r}"
            )
    for column, expected_metric in _METRIC_LABELS.items():
        observed = str(_cell(raw, 2, column)).strip()
        if observed != expected_metric:
            raise ValueError(
                f"Source workbook metric header mismatch at column {column}: "
                f"expected {expected_metric!r}, got {observed!r}"
            )

    selected = raw.iloc[3:, _SOURCE_POSITIONS].copy()
    selected.columns = _SOURCE_COLUMNS
    selected = selected.loc[~selected.iloc[:, :4].isna().all(axis=1)].reset_index(drop=True)
    for column in ("Species", "GeneID", "StudyStatus", "Query"):
        selected[column] = selected[column].map(
            lambda value: "" if pd.isna(value) else str(value).strip()
        )
    selected["StudyStatus"] = selected["StudyStatus"].map(_normalise_status)

    if selected.empty:
        raise ValueError("Source workbook contains no data rows")
    if selected["GeneID"].eq("").any() or selected["GeneID"].duplicated().any():
        raise ValueError("Source workbook contains empty or duplicate gene IDs")
    if selected["Query"].eq("").any():
        raise ValueError("Source workbook contains an empty query")
    if selected["StudyStatus"].eq("").any():
        raise ValueError("Source workbook contains an unsupported study status")
    return selected


def _expected_archive_members(categories: pd.DataFrame) -> dict[str, tuple[str, str]]:
    expected: dict[str, tuple[str, str]] = {}
    for row in categories.itertuples(index=False):
        if row.StudyStatus == "well_studied":
            key = (row.Gene, "single")
            member = f"Claude/Claude-{row.Gene}.md"
            expected[member] = key
        elif row.StudyStatus == "uncharacterized":
            for replicate in range(1, 4):
                key = (row.Gene, f"R{replicate}")
                member = f"Claude/Claude-{row.Gene}-R{replicate}.md"
                expected[member] = key
        else:  # pragma: no cover - load_gene_categories rejects this first.
            raise ValueError(f"Unsupported study status: {row.StudyStatus}")
    return expected


def _validate_zip_member_name(name: str) -> None:
    if "\\" in name:
        raise ValueError("Claude archive member contract violation: backslash in member path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"Claude archive member contract violation: unsafe member path {name!r}"
        )


def load_claude_archive(
    path: Path,
    categories: pd.DataFrame,
) -> dict[tuple[str, str], str]:
    """Validate a Claude ZIP and return Markdown text keyed by gene/replicate."""

    _require_file(path, "Claude response archive")
    expected = _expected_archive_members(categories)
    try:
        with ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError(
                    "Claude archive member contract violation: duplicate member names"
                )
            for name in names:
                _validate_zip_member_name(name)
            file_names = [name for name in names if not name.endswith("/")]
            md_names = {name for name in file_names if name.lower().endswith(".md")}
            non_md_names = set(file_names) - md_names
            if non_md_names or md_names != set(expected):
                raise ValueError(
                    "Claude archive member contract violation: expected exactly "
                    f"{len(expected)} Markdown members, got {len(md_names)}"
                )
            responses: dict[tuple[str, str], str] = {}
            for member, key in expected.items():
                try:
                    content = archive.read(member).decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        "Claude archive member contract violation: non-UTF-8 content "
                        f"in {member!r}"
                    ) from exc
                if not content.strip():
                    raise ValueError(
                        "Claude archive member contract violation: empty content "
                        f"in {member!r}"
                    )
                responses[key] = content
    except ValueError:
        raise
    except (OSError, EOFError) as exc:
        raise ValueError(f"Claude archive member contract violation: unreadable ZIP {path}") from exc
    return responses


def _require_historical_source(source: pd.DataFrame) -> None:
    """Validate the fixed 100/100 source cohort used by the baseline means."""

    required_columns = {
        "GeneID",
        "StudyStatus",
        "Query",
        *(
            column
            for metric_columns in _HISTORICAL_COLUMNS.values()
            for column in metric_columns.values()
        ),
    }
    missing = sorted(required_columns.difference(source.columns))
    if missing:
        raise ValueError(f"Historical source is missing columns: {missing}")
    counts = source["StudyStatus"].value_counts(dropna=False).to_dict()
    expected_counts = {"well_studied": 100, "uncharacterized": 100}
    if counts != expected_counts:
        raise ValueError(
            "Historical source must contain exactly 100 well_studied and 100 "
            f"uncharacterized rows, got {counts}"
        )


def historical_model_means(source: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return unweighted baseline means after validating the frozen cohort.

    The workbook contains one row per gene, with BERTScore cells populated for
    the 100 well-studied genes and hallucination cells populated for the 100
    uncharacterized genes.  Every gene in the relevant status contributes
    equally to its model mean.  The values are checked against the historical
    reference before a downstream caller accepts any new metric output.
    """

    _require_historical_source(source)
    means: dict[str, dict[str, float]] = {metric: {} for metric in _HISTORICAL_COLUMNS}
    for metric, model_columns in _HISTORICAL_COLUMNS.items():
        status = _HISTORICAL_STATUSES[metric]
        selected = source.loc[source["StudyStatus"].eq(status)]
        for model, column in model_columns.items():
            values = pd.to_numeric(selected[column], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Historical source contains non-finite {metric} values in {column}")
            if ((values < 0) | (values > 1)).any():
                raise ValueError(f"Historical source contains out-of-range {metric} values in {column}")
            mean = float(values.mean())
            expected = EXPECTED_HISTORICAL_MEANS[metric][model]
            if not math.isfinite(mean) or abs(mean - expected) > 1e-12:
                raise ValueError(
                    f"Historical {metric} mean for {model} changed: "
                    f"expected {expected:.16g}, got {mean:.16g}"
                )
            means[metric][model] = mean
    return means


def create_bertscorer(device: str, batch_size: int) -> object:
    """Lazily construct the BERTScore evaluator used by Knowledge&Review."""

    if batch_size < 1:
        raise ValueError("BERTScore batch_size must be positive")
    try:
        from bert_score import BERTScorer
    except ImportError as exc:  # pragma: no cover - dependency is optional at import time.
        raise RuntimeError(
            "BERTScore is unavailable; install the deepgenome-eval extra before scoring"
        ) from exc
    return BERTScorer(
        model_type="bert-base-uncased",
        num_layers=9,
        batch_size=batch_size,
        nthreads=1,
        all_layers=False,
        idf=False,
        device=device,
        rescale_with_baseline=False,
        lang="en",
    )


def _tensor_values(value: object) -> list[float]:
    """Convert a BERTScore tensor-like result into Python floats."""

    current = value
    for method_name in ("detach", "cpu"):
        method = getattr(current, method_name, None)
        if callable(method):
            current = method()
    tolist = getattr(current, "tolist", None)
    if callable(tolist):
        current = tolist()
    if isinstance(current, (str, bytes)):
        raise ValueError("BERTScore precision result is not numeric")
    if np.isscalar(current):
        current = [current]
    try:
        return [float(item) for item in current]  # type: ignore[union-attr]
    except (TypeError, ValueError) as exc:
        raise ValueError("BERTScore precision result is not a one-dimensional sequence") from exc


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def calculate_claude_bertscore(
    source: pd.DataFrame,
    responses: dict[tuple[str, str], str],
    scorer: object,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Score Claude's well-studied responses against workbook queries.

    BERTScore receives the raw Markdown response as the candidate and the
    corresponding query as the reference.  Only hashes are retained in the
    returned table so response and query text never enter tracked artifacts.
    """

    if batch_size < 1:
        raise ValueError("BERTScore batch_size must be positive")
    required = {"GeneID", "StudyStatus", "Query"}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"Claude BERTScore source is missing columns: {missing}")
    selected = (
        source.loc[source["StudyStatus"].eq("well_studied"), ["GeneID", "Query"]]
        .sort_values("GeneID", kind="mergesort")
        .reset_index(drop=True)
    )
    if selected.empty:
        raise ValueError("Claude BERTScore source contains no well_studied genes")

    candidates: list[str] = []
    references: list[str] = []
    for row in selected.itertuples(index=False):
        gene = str(row.GeneID)
        query = str(row.Query)
        if not query.strip():
            raise ValueError(f"Empty query for Claude BERTScore gene {gene}")
        key = (gene, "single")
        if key not in responses:
            raise ValueError(f"Missing Claude response for {gene}")
        response = responses[key]
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"Empty Claude response for {gene}")
        candidates.append(response)
        references.append(query)

    score_method = getattr(scorer, "score", None)
    if not callable(score_method):
        raise TypeError("BERTScore scorer must provide a callable score method")
    score_result = score_method(candidates, references, batch_size=batch_size)
    try:
        precision_result = score_result[0]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError("BERTScore scorer returned no precision component") from exc
    precision = _tensor_values(precision_result)
    if len(precision) != len(selected):
        raise ValueError(
            "BERTScore scorer returned an unexpected number of precision values: "
            f"expected {len(selected)}, got {len(precision)}"
        )
    for value in precision:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"BERTScore precision must be finite and in [0, 1], got {value!r}")

    records = []
    for row, value, query, response in zip(
        selected.itertuples(index=False), precision, references, candidates, strict=True
    ):
        records.append(
            {
                "Model": "Claude",
                "Gene": str(row.GeneID),
                "BERTScorePrecision": float(value),
                "QuerySHA256": _sha256_text(query),
                "ResponseSHA256": _sha256_text(response),
            }
        )
    return pd.DataFrame(
        records,
        columns=["Model", "Gene", "BERTScorePrecision", "QuerySHA256", "ResponseSHA256"],
    )


def load_hallucination_core(notebook_path: Path) -> dict[str, object]:
    """Load the canonical hallucination helpers from a tagged notebook cell.

    The freezer intentionally executes only the ``hallucination-core`` cell.
    This keeps the validator and aggregation semantics in one place while
    avoiding notebook input, API, or live-run cells.
    """

    _require_file(notebook_path, "Hallucination notebook")
    try:
        import nbformat

        notebook = nbformat.read(notebook_path, as_version=4)
    except Exception as exc:  # pragma: no cover - malformed notebooks are rare.
        raise ValueError(f"Unable to read hallucination notebook: {notebook_path}") from exc
    cells = [
        cell
        for cell in notebook.cells
        if cell.get("id") == "hallucination-core"
    ]
    if len(cells) != 1:
        raise ValueError(
            "Hallucination notebook must contain exactly one hallucination-core cell"
        )
    namespace: dict[str, object] = {"__name__": "_fig2_hallucination_core"}
    try:
        exec("".join(cells[0].get("source", [])), namespace)
    except Exception as exc:
        raise ValueError("Unable to load the hallucination-core cell") from exc
    required = (
        "validate_judgment_records",
        "formal_metadata_is_complete",
        "extract_gene_contradiction",
        "sanitize_api_base_url",
        "sha256_text",
        "JUDGE_PROMPT_SHA256",
    )
    missing = [name for name in required if name not in namespace]
    if missing:
        raise ValueError(
            "hallucination-core is missing required helpers: " + ", ".join(missing)
        )
    return namespace


def _response_text(
    responses: dict[tuple[str, str], str],
    gene: str,
    response_id: str,
) -> str:
    """Resolve both archive (R1) and notebook (rep_1) response IDs."""

    candidates = [response_id]
    if response_id.startswith("rep_"):
        candidates.append("R" + response_id.removeprefix("rep_"))
    elif response_id.startswith("R"):
        candidates.append("rep_" + response_id.removeprefix("R"))
    for candidate in candidates:
        value = responses.get((gene, candidate))
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"Missing Claude response text for {gene}/{response_id}")


def _log_path(judgment_dir: Path, gene: str) -> Path:
    return judgment_dir / _CLAUDE_LOG_FILENAME.format(gene=gene)


def _metadata_record(records: list[dict[str, object]]) -> dict[str, object]:
    metadata = [
        record
        for record in records
        if isinstance(record, dict) and record.get("type") == "metadata"
    ]
    if len(metadata) != 1:
        raise ValueError("Claude judgment log must contain exactly one metadata record")
    return metadata[0]


def compact_claude_judgments(
    judgment_dir: Path,
    genes: list[str],
    responses: dict[tuple[str, str], str],
    core: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate complete Claude logs and return compact pair/gene tables.

    Validation is deliberately strict: every requested gene needs one current
    formal log, six unique directed summaries, matching response hashes, and
    no API-error records.  The canonical notebook validator and extractor are
    called from ``core`` so this function cannot drift from notebook semantics.
    """

    if not judgment_dir.is_dir():
        raise FileNotFoundError(f"Judgment directory does not exist: {judgment_dir}")
    if not genes:
        raise ValueError("100 valid Claude judgment logs required; no genes supplied")
    if list(genes) != sorted(genes) or len(set(genes)) != len(genes):
        raise ValueError("Claude uncharacterized genes must be sorted and unique")
    validator = core.get("validate_judgment_records")
    formal = core.get("formal_metadata_is_complete")
    extractor = core.get("extract_gene_contradiction")
    sha_text = core.get("sha256_text", _sha256_text)
    if not all(callable(function) for function in (validator, formal, extractor, sha_text)):
        raise TypeError("hallucination-core does not expose callable validation helpers")

    pair_records: list[dict[str, object]] = []
    gene_records: list[dict[str, object]] = []
    invalid: list[str] = []
    expected_pairs = {
        (source, target)
        for source in range(3)
        for target in range(3)
        if source != target
    }
    high_threshold = float(core.get("GENE_HIGH_CONTRADICTION_THRESHOLD", 0.6))
    for gene in genes:
        path = _log_path(judgment_dir, gene)
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
            errors = validator(records, expected_response_count=3)
            if errors:
                raise ValueError("; ".join(str(error) for error in errors))
            if not formal(records, "claude", gene, expected_response_count=3):
                raise ValueError("formal metadata is incomplete")
            metadata = _metadata_record(records)
            response_ids = metadata.get("response_ids")
            if response_ids != ["rep_1", "rep_2", "rep_3"]:
                raise ValueError("metadata response IDs must be rep_1, rep_2, rep_3")
            response_hashes = metadata.get("response_sha256")
            expected_hashes = [
                sha_text(_response_text(responses, gene, response_id))
                for response_id in response_ids
            ]
            if response_hashes != expected_hashes:
                raise ValueError("response hash is stale")
            summaries = [
                record
                for record in records
                if isinstance(record, dict)
                and record.get("type") == "version_pair_summary"
            ]
            observed_pairs = {
                tuple(record.get("version_pair", []))
                for record in summaries
            }
            if len(summaries) != 6 or observed_pairs != expected_pairs:
                raise ValueError("judgment log must contain six unique ordered pairs")
            log_hash = sha256_file(path)
            for summary in sorted(
                summaries,
                key=lambda item: tuple(item["version_pair"]),  # type: ignore[index]
            ):
                source_index, target_index = summary["version_pair"]  # type: ignore[misc]
                pair_records.append(
                    {
                        "Model": "Claude",
                        "Gene": gene,
                        "SourceResponse": response_ids[source_index],
                        "TargetResponse": response_ids[target_index],
                        "SupportRatio": float(summary["support_ratio"]),
                        "ContradictionRatio": float(summary["contra_ratio"]),
                        "EntailmentEstablished": bool(
                            summary["entailment_established"]
                        ),
                        "JudgmentLogSHA256": log_hash,
                    }
                )
            extracted = extractor(records)
            if extracted is None:
                raise ValueError("canonical extractor returned no aggregate")
            extracted_mean = float(extracted[0])  # type: ignore[index]
            direct_mean = float(
                sum(float(summary["contra_ratio"]) for summary in summaries) / 6
            )
            if not math.isclose(extracted_mean, direct_mean, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError("canonical extractor disagrees with pair-row mean")
            gene_records.append(
                {
                    "Model": "Claude",
                    "Gene": gene,
                    "MeanDirectionalContradictionRatio": direct_mean,
                    "HighContradiction": direct_mean >= high_threshold,
                }
            )
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
            invalid.append(f"{gene}: {exc}")
    if invalid:
        details = "; ".join(invalid[:3])
        raise ValueError(
            "100 valid Claude judgment logs required; invalid or missing logs: "
            + details
        )
    pairs = pd.DataFrame(pair_records, columns=_PAIR_COLUMNS)
    genes_frame = pd.DataFrame(gene_records, columns=_GENE_HALLUCINATION_COLUMNS)
    return pairs, genes_frame


def _validated_metric_mean(frame: pd.DataFrame, column: str, label: str) -> float:
    if "Model" in frame and not frame["Model"].astype(str).eq("Claude").all():
        raise ValueError(f"Claude {label} table contains non-Claude rows")
    if "Gene" in frame and frame["Gene"].duplicated().any():
        raise ValueError(f"Claude {label} table contains duplicate genes")
    if column not in frame:
        raise ValueError(f"Claude {label} table is missing {column}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if len(values) == 0 or not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError(f"Claude {label} values must be finite and in [0, 1]")
    return float(values.mean())


def build_figure_tables(
    source: pd.DataFrame,
    bertscore: pd.DataFrame,
    hallucination: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble fixed-order five-model Fig. 2g and Fig. 2h inputs."""

    historical = historical_model_means(source)
    claude_bert = _validated_metric_mean(bertscore, "BERTScorePrecision", "BERTScore")
    claude_hallucination = _validated_metric_mean(
        hallucination,
        "MeanDirectionalContradictionRatio",
        "hallucination",
    )
    bert_values = {
        **historical["bertscore"],
        "Claude": claude_bert,
    }
    hallucination_values = {
        **historical["hallucination"],
        "Claude": claude_hallucination,
    }
    bert_plot = pd.DataFrame(
        [
            {
                "Model": model,
                "DisplayLabel": DISPLAY_LABELS[model],
                "BERTScorePrecision": bert_values[model],
            }
            for model in MODEL_ORDER
        ],
        columns=["Model", "DisplayLabel", "BERTScorePrecision"],
    )
    hallucination_plot = pd.DataFrame(
        [
            {
                "Model": model,
                "DisplayLabel": DISPLAY_LABELS[model],
                "MeanDirectionalContradictionRatio": hallucination_values[model],
            }
            for model in MODEL_ORDER
        ],
        columns=["Model", "DisplayLabel", "MeanDirectionalContradictionRatio"],
    )
    return bert_plot, hallucination_plot


def _package_versions() -> dict[str, str]:
    names = ("bert-score", "nbformat", "networkx", "nltk", "numpy", "pandas", "torch")
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _safe_input_record(label: str, path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    return {"label": label, "sha256": sha256_file(path)}


def _frame_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(
        sep="\t",
        index=False,
        lineterminator="\n",
        float_format="%.15g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_provenance(
    source: pd.DataFrame,
    bertscore: pd.DataFrame,
    hallucination_pairs: pd.DataFrame,
    hallucination: pd.DataFrame,
    source_workbook: Path | None = None,
    claude_archive: Path | None = None,
    gene_categories: Path | None = None,
    judgment_dir: Path | None = None,
    hallucination_notebook: Path | None = None,
    freezer_script: Path | None = None,
    core: dict[str, object] | None = None,
    judge: dict[str, object] | None = None,
    anomalies: list[str] | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, object]:
    """Build non-secret, checksummed lineage for the frozen metrics."""

    historical = historical_model_means(source)
    judge_payload = {
        "api_base_url": "https://www.dmxapi.cn/v1",
        "model": "deepseek-v3.2-exp",
        "temperature": 0,
        "max_tokens": 10,
        "max_concurrent": 32,
    }
    if judge:
        for key, value in judge.items():
            if key.casefold() not in {"api_key", "key", "token", "authorization"}:
                judge_payload[key] = value
    if core and isinstance(core.get("JUDGE_PROMPT_SHA256"), str):
        judge_payload["prompt_sha256"] = core["JUDGE_PROMPT_SHA256"]
    inputs = {
        "source_workbook": _safe_input_record("source workbook", source_workbook),
        "claude_archive": _safe_input_record("Claude response archive", claude_archive),
        "gene_categories": _safe_input_record("gene category table", gene_categories),
        "hallucination_notebook": _safe_input_record("canonical hallucination notebook", hallucination_notebook),
        "freezer_script": _safe_input_record("figure-metric freezer", freezer_script),
    }
    if judgment_dir is not None and judgment_dir.is_dir():
        log_hashes = {
            path.name: sha256_file(path)
            for path in sorted(judgment_dir.glob(_CLAUDE_LOG_FILENAME.format(gene="*")))
            if path.is_file()
        }
        inputs["judgment_logs"] = {
            "label": "Claude judgment logs",
            "count": len(log_hashes),
            "sha256_by_name": log_hashes,
        }
    inputs = {key: value for key, value in inputs.items() if value is not None}
    counts = {
        "well_studied_genes": int(source["StudyStatus"].eq("well_studied").sum()),
        "uncharacterized_genes": int(source["StudyStatus"].eq("uncharacterized").sum()),
        "valid_judgment_logs": int(hallucination["Gene"].nunique()),
        "directed_pair_rows": int(len(hallucination_pairs)),
    }
    bert_plot, hallucination_plot = build_figure_tables(
        source,
        bertscore,
        hallucination,
    )
    final_values = {
        "bertscore": {
            row.Model: float(row.BERTScorePrecision)
            for row in bert_plot.itertuples()
        },
        "hallucination": {
            row.Model: float(row.MeanDirectionalContradictionRatio)
            for row in hallucination_plot.itertuples()
        },
    }
    return {
        "schema_version": 1,
        "generated_at_utc": generated_at_utc
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "platform": platform.python_version(),
        "inputs": inputs,
        "cohort": {
            "well_studied_genes": int(source["StudyStatus"].eq("well_studied").sum()),
            "uncharacterized_genes": int(source["StudyStatus"].eq("uncharacterized").sum()),
            "source_frame_sha256": _frame_sha256(source),
        },
        "bertscore": {
            "model_type": "bert-base-uncased",
            "num_layers": 9,
            "lang": "en",
            "all_layers": False,
            "idf": False,
            "rescale_with_baseline": False,
            "nthreads": 1,
            "batch_size": 16,
            "rows": len(bertscore),
            "table_sha256": _frame_sha256(bertscore),
        },
        "judge": judge_payload,
        "package_versions": _package_versions(),
        "counts": counts,
        "anomalies": list(anomalies or _KNOWN_ARCHIVE_ANOMALIES),
        "tables": {
            "bertscore_by_gene_sha256": _frame_sha256(bertscore),
            "hallucination_pairs_sha256": _frame_sha256(hallucination_pairs),
            "hallucination_by_gene_sha256": _frame_sha256(hallucination),
        },
        "historical_values": historical,
        "final_values": final_values,
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_frame(path: Path, frame: pd.DataFrame) -> None:
    payload = frame.to_csv(
        sep="\t",
        index=False,
        lineterminator="\n",
        float_format="%.15g",
    ).encode("utf-8")
    _atomic_write_bytes(path, payload)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write_bytes(path, serialized.encode("utf-8"))


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--source-workbook", type=Path, required=True)
    parser.add_argument("--claude-archive", type=Path, required=True)
    parser.add_argument("--gene-categories", type=Path, required=True)
    parser.add_argument("--judgment-dir", type=Path)
    parser.add_argument("--hallucination-notebook", type=Path)
    parser.add_argument("--frozen-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    source = load_source_workbook(args.source_workbook)
    categories = load_gene_categories(args.gene_categories)
    responses = load_claude_archive(args.claude_archive, categories)
    if args.validate_only:
        well_studied = int((categories["StudyStatus"] == "well_studied").sum())
        uncharacterized = int((categories["StudyStatus"] == "uncharacterized").sum())
        print(
            f"{len(source)} source rows; {well_studied} well-studied responses; "
            f"{uncharacterized} complete response triplets; validation passed"
        )
        return 0

    required_outputs = {
        "--judgment-dir": args.judgment_dir,
        "--hallucination-notebook": args.hallucination_notebook,
        "--frozen-dir": args.frozen_dir,
        "--figure-dir": args.figure_dir,
    }
    missing_outputs = [name for name, value in required_outputs.items() if value is None]
    if missing_outputs:
        raise SystemExit(
            "Normal freezer mode requires: " + ", ".join(missing_outputs)
        )
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    categories = categories.sort_values("Gene", kind="mergesort").reset_index(drop=True)
    uncharacterized = categories.loc[
        categories["StudyStatus"].eq("uncharacterized"), "Gene"
    ].tolist()
    if len(uncharacterized) != 100 or uncharacterized != sorted(uncharacterized):
        raise ValueError(
            "Exactly 100 sorted uncharacterized genes are required for formal aggregation"
        )
    source_genes = set(source["GeneID"])
    category_genes = set(categories["Gene"])
    if source_genes != category_genes:
        raise ValueError("Source workbook and gene-category table contain different gene cohorts")
    source_status = dict(zip(source["GeneID"], source["StudyStatus"], strict=True))
    category_status = dict(zip(categories["Gene"], categories["StudyStatus"], strict=True))
    if source_status != category_status:
        raise ValueError("Source workbook and gene-category table have inconsistent study statuses")
    historical_model_means(source)
    core = load_hallucination_core(args.hallucination_notebook)
    device = args.device
    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    scorer = create_bertscorer(device=device, batch_size=args.batch_size)
    bertscore = calculate_claude_bertscore(
        source,
        responses,
        scorer,
        batch_size=args.batch_size,
    )
    pairs, hallucination = compact_claude_judgments(
        args.judgment_dir,
        uncharacterized,
        responses,
        core,
    )
    bert_plot, hallucination_plot = build_figure_tables(
        source,
        bertscore,
        hallucination,
    )
    provenance = build_provenance(
        source=source,
        bertscore=bertscore,
        hallucination_pairs=pairs,
        hallucination=hallucination,
        source_workbook=args.source_workbook,
        claude_archive=args.claude_archive,
        gene_categories=args.gene_categories,
        judgment_dir=args.judgment_dir,
        hallucination_notebook=args.hallucination_notebook,
        freezer_script=Path(__file__).resolve(),
        core=core,
    )
    frozen_outputs = {
        args.frozen_dir / "PhytoBench-Gene-Claude-BERTScore-by-gene.tsv": (
            bertscore,
            "frame",
        ),
        args.frozen_dir / "PhytoBench-Gene-Claude-hallucination-pairs.tsv": (
            pairs,
            "frame",
        ),
        args.frozen_dir / "PhytoBench-Gene-Claude-hallucination-by-gene.tsv": (
            hallucination,
            "frame",
        ),
        args.frozen_dir / "PhytoBench-Gene-Claude-metrics-provenance.json": (
            provenance,
            "json",
        ),
        args.figure_dir / "PhytoBench-Gene-BERTScore-for_plot.tsv": (
            bert_plot,
            "frame",
        ),
        args.figure_dir / "PhytoBench-Gene-hallucination-for_plot.tsv": (
            hallucination_plot,
            "frame",
        ),
    }
    for path, (payload, kind) in frozen_outputs.items():
        if kind == "frame":
            _atomic_write_frame(path, payload)  # type: ignore[arg-type]
        else:
            _atomic_write_json(path, payload)  # type: ignore[arg-type]
    print(
        f"Published {len(bertscore)} Claude BERTScore rows, "
        f"{len(pairs)} directed hallucination rows, and five-model figure tables"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
