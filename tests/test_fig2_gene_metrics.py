from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import openpyxl
import pytest

from scripts.freeze_fig2_gene_metrics import (
    load_claude_archive,
    load_gene_categories,
    load_source_workbook,
)


@pytest.fixture
def gene_categories(tmp_path: Path) -> Path:
    path = tmp_path / "gene_categories.tsv"
    path.write_text(
        "Species\tGene\tStudyStatus\n"
        "Rice\tWELL1\twell_studied\n"
        "Rice\tUN1\tuncharacterized\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def source_workbook(tmp_path: Path) -> Path:
    path = tmp_path / "source.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Supplementary Data 7"
    sheet.append(["Evaluation scores for the gene benchmark."] + [None] * 23)
    sheet.append(
        [
            "Species",
            "Gene ID",
            "Gene type",
            "Query",
            "Phytomni",
            None,
            None,
            None,
            None,
            "Gemini",
            None,
            None,
            None,
            None,
            "OpenAI",
            None,
            None,
            None,
            None,
            "Grok",
            None,
            None,
            None,
            None,
        ]
    )
    sheet.append(
        [
            None,
            None,
            None,
            None,
            "Human-1",
            "Human-2",
            "Human-3",
            "BERTScore precision",
            "Hallucination",
            "Human-1",
            "Human-2",
            "Human-3",
            "BERTScore precision",
            "Hallucination",
            "Human-1",
            "Human-2",
            "Human-3",
            "BERTScore precision",
            "Hallucination",
            "Human-1",
            "Human-2",
            "Human-3",
            "BERTScore precision",
            "Hallucination",
        ]
    )
    sheet.append(
        [
            "Rice",
            "WELL1",
            "Well-studied",
            "[Species Name: rice] Describe WELL1.",
            "R1",
            "R2",
            "R3",
            0.51,
            0.10,
            "R1",
            "R2",
            "R3",
            0.41,
            0.20,
            "R1",
            "R2",
            "R3",
            0.31,
            0.30,
            "R1",
            "R2",
            "R3",
            0.21,
            0.40,
        ]
    )
    sheet.append(
        [
            "Rice",
            "UN1",
            "Uncharacterized",
            "[Species Name: rice] Describe UN1.",
            "R1",
            "R2",
            "R3",
            0.52,
            0.11,
            "R1",
            "R2",
            "R3",
            0.42,
            0.21,
            "R1",
            "R2",
            "R3",
            0.32,
            0.31,
            "R1",
            "R2",
            "R3",
            0.22,
            0.41,
        ]
    )
    for start, end in (("E", "I"), ("J", "N"), ("O", "S"), ("T", "X")):
        sheet.merge_cells(f"{start}2:{end}2")
    workbook.save(path)
    return path


def _write_archive(path: Path, *, include_extra: bool = False, empty_member: bool = False) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("Claude/", "")
        archive.writestr("Claude/Claude-WELL1.md", "# WELL1\n")
        archive.writestr("Claude/Claude-UN1-R1.md", "" if empty_member else "# UN1 R1\n")
        archive.writestr("Claude/Claude-UN1-R2.md", "# UN1 R2\n")
        archive.writestr("Claude/Claude-UN1-R3.md", "# UN1 R3\n")
        if include_extra:
            archive.writestr("Claude/Claude-EXTRA.md", "# EXTRA\n")


@pytest.fixture
def claude_archive(tmp_path: Path) -> Path:
    path = tmp_path / "Claude.zip"
    _write_archive(path)
    return path


def test_source_workbook_parses_merged_header_layout(source_workbook: Path) -> None:
    frame = load_source_workbook(source_workbook)
    assert frame.columns.tolist() == [
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
    assert frame["GeneID"].is_unique
    assert not frame["Query"].str.strip().eq("").any()
    assert frame["StudyStatus"].tolist() == ["well_studied", "uncharacterized"]


def test_claude_archive_requires_exact_cohort_members(
    claude_archive: Path,
    gene_categories: Path,
) -> None:
    categories = load_gene_categories(gene_categories)
    responses = load_claude_archive(claude_archive, categories)
    assert responses[("WELL1", "single")].startswith("# WELL1")
    assert [responses[("UN1", f"R{i}")] for i in range(1, 4)] == [
        "# UN1 R1\n",
        "# UN1 R2\n",
        "# UN1 R3\n",
    ]


@pytest.mark.parametrize(
    ("include_extra", "empty_member"),
    [(True, False), (False, True)],
)
def test_claude_archive_rejects_missing_extra_or_empty_members(
    tmp_path: Path,
    gene_categories: Path,
    include_extra: bool,
    empty_member: bool,
) -> None:
    archive_path = tmp_path / "invalid.zip"
    _write_archive(
        archive_path,
        include_extra=include_extra,
        empty_member=empty_member,
    )
    categories = load_gene_categories(gene_categories)
    with pytest.raises(ValueError, match="Claude archive member contract"):
        load_claude_archive(archive_path, categories)
