from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import openpyxl
import pandas as pd
import pytest

from scripts.freeze_fig2_gene_metrics import (
    EXPECTED_HISTORICAL_MEANS,
    calculate_claude_bertscore,
    create_bertscorer,
    historical_model_means,
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


def _historical_source() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for status in ("well_studied", "uncharacterized"):
        for index in range(100):
            row: dict[str, object] = {
                "Species": "Rice",
                "GeneID": f"{status}-{index:03d}",
                "StudyStatus": status,
                "Query": f"query {status} {index}",
            }
            for metric, model_values in EXPECTED_HISTORICAL_MEANS.items():
                suffix = "BERTScorePrecision" if metric == "bertscore" else "Hallucination"
                for model, value in model_values.items():
                    target_status = "well_studied" if metric == "bertscore" else "uncharacterized"
                    row[f"{model}{suffix}"] = value if status == target_status else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def test_historical_means_are_unweighted_and_match_reference_values() -> None:
    means = historical_model_means(_historical_source())
    for metric, expected_models in EXPECTED_HISTORICAL_MEANS.items():
        assert means[metric] == pytest.approx(expected_models, abs=1e-12)


def test_historical_means_reject_nonfinite_target_cells() -> None:
    source = _historical_source()
    source.loc[0, "PhytomniBERTScorePrecision"] = float("nan")
    with pytest.raises(ValueError, match="non-finite bertscore"):
        historical_model_means(source)


class _TensorLike:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def detach(self) -> "_TensorLike":
        return self

    def cpu(self) -> "_TensorLike":
        return self

    def tolist(self) -> list[float]:
        return self.values


class _RecordingScorer:
    def __init__(self, precision: list[float]) -> None:
        self.precision = precision
        self.calls: list[tuple[list[str], list[str], int]] = []

    def score(
        self,
        candidates: list[str],
        references: list[str],
        *,
        batch_size: int,
    ) -> tuple[_TensorLike, None, None]:
        self.calls.append((candidates, references, batch_size))
        return _TensorLike(self.precision), None, None


def test_claude_bertscore_uses_response_as_candidate_and_query_as_reference() -> None:
    source = pd.DataFrame(
        [
            {
                "Species": "Rice",
                "GeneID": "WELL2",
                "StudyStatus": "well_studied",
                "Query": "query 2",
            },
            {
                "Species": "Rice",
                "GeneID": "WELL1",
                "StudyStatus": "well_studied",
                "Query": "query 1",
            },
        ]
    )
    responses = {
        ("WELL1", "single"): "# WELL1 response\n",
        ("WELL2", "single"): "# WELL2 response\n",
    }
    scorer = _RecordingScorer(precision=[0.51, 0.53])

    result = calculate_claude_bertscore(source, responses, scorer, batch_size=16)

    assert scorer.calls == [
        (["# WELL1 response\n", "# WELL2 response\n"], ["query 1", "query 2"], 16)
    ]
    assert result.columns.tolist() == [
        "Model",
        "Gene",
        "BERTScorePrecision",
        "QuerySHA256",
        "ResponseSHA256",
    ]
    assert result["BERTScorePrecision"].tolist() == [0.51, 0.53]
    assert result["Gene"].tolist() == ["WELL1", "WELL2"]
    assert result["QuerySHA256"].tolist() == [
        hashlib.sha256(b"query 1").hexdigest(),
        hashlib.sha256(b"query 2").hexdigest(),
    ]
    assert result["ResponseSHA256"].tolist() == [
        hashlib.sha256(b"# WELL1 response\n").hexdigest(),
        hashlib.sha256(b"# WELL2 response\n").hexdigest(),
    ]


def test_bertscorer_configuration_matches_review_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, object] = {}

    class FakeBERTScorer:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

    fake_module = types.ModuleType("bert_score")
    fake_module.BERTScorer = FakeBERTScorer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bert_score", fake_module)

    create_bertscorer(device="cpu", batch_size=16)

    assert captured_kwargs == {
        "model_type": "bert-base-uncased",
        "num_layers": 9,
        "batch_size": 16,
        "nthreads": 1,
        "all_layers": False,
        "idf": False,
        "device": "cpu",
        "rescale_with_baseline": False,
        "lang": "en",
    }


@pytest.mark.parametrize("invalid_precision", [float("nan"), -0.01, 1.01])
def test_claude_bertscore_rejects_nonfinite_or_out_of_range_precision(
    invalid_precision: float,
) -> None:
    source = pd.DataFrame(
        [
            {
                "GeneID": "WELL1",
                "StudyStatus": "well_studied",
                "Query": "query 1",
            }
        ]
    )
    responses = {("WELL1", "single"): "response 1"}
    scorer = _RecordingScorer(precision=[invalid_precision])
    with pytest.raises(ValueError, match=r"precision must be finite and in \[0, 1\]"):
        calculate_claude_bertscore(source, responses, scorer)
