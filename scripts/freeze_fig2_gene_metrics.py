"""Input validation helpers for the Fig. 2 gene-metric workflow.

The source workbook and Claude response archive are deliberately kept outside
the tracked data boundary.  This module validates their schemas and returns
in-memory values so downstream scoring code never needs to extract private
response files to disk.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
from typing import Iterable
from zipfile import ZipFile

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


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--source-workbook", type=Path, required=True)
    parser.add_argument("--claude-archive", type=Path, required=True)
    parser.add_argument("--gene-categories", type=Path, required=True)
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
    else:
        print(f"Validated {len(source)} source rows and {len(responses)} Claude responses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
