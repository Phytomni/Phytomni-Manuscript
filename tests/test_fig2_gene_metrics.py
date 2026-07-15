from __future__ import annotations

import hashlib
import json
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
    build_figure_tables,
    build_provenance,
    compact_claude_judgments,
    create_bertscorer,
    historical_model_means,
    load_hallucination_core,
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


def _write_formal_log(
    directory: Path,
    gene: str,
    responses: dict[tuple[str, str], str],
    notebook_path: Path,
    *,
    ratios: list[float] | None = None,
) -> None:
    core = load_hallucination_core(notebook_path)
    ratios = ratios or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    response_ids = ["rep_1", "rep_2", "rep_3"]
    metadata = {
        "type": "metadata",
        "model_id": "claude",
        "gene_id": gene,
        "response_ids": response_ids,
        "api_base_url": "https://www.dmxapi.cn/v1",
        "judge_model": "deepseek-v3.2-exp",
        "judge_prompt_sha256": core["JUDGE_PROMPT_SHA256"],
        "response_sha256": [
            hashlib.sha256(responses[(gene, f"R{i}")].encode()).hexdigest()
            for i in range(1, 4)
        ],
    }
    pairs = [
        [source, target]
        for source in range(3)
        for target in range(3)
        if source != target
    ]
    records = [metadata]
    records.extend(
        {
            "type": "version_pair_summary",
            "version_pair": pair,
            "support_ratio": 1.0 - ratio,
            "contra_ratio": ratio,
            "entailment_established": False,
        }
        for pair, ratio in zip(pairs, ratios, strict=True)
    )
    path = directory / f"claude__{gene}__rep_1-rep_2-rep_3.json"
    path.write_text(json.dumps(records), encoding="utf-8")


@pytest.fixture
def hallucination_notebook() -> Path:
    return Path("DeepGenomeAgent Evaluation/score_hallucination.ipynb")


@pytest.fixture
def formal_judgment_fixture(
    tmp_path: Path,
    hallucination_notebook: Path,
) -> tuple[Path, list[str], dict[tuple[str, str], str]]:
    directory = tmp_path / "judgments"
    directory.mkdir()
    genes = ["UN1"]
    responses = {
        ("UN1", "R1"): "UN1 response one.",
        ("UN1", "R2"): "UN1 response two.",
        ("UN1", "R3"): "UN1 response three.",
    }
    _write_formal_log(directory, "UN1", responses, hallucination_notebook)
    return directory, genes, responses


def test_compaction_writes_six_pairs_and_one_gene_row(
    formal_judgment_fixture: tuple[Path, list[str], dict[tuple[str, str], str]],
    hallucination_notebook: Path,
) -> None:
    directory, genes, responses = formal_judgment_fixture
    pairs, gene_rows = compact_claude_judgments(
        directory,
        genes,
        responses,
        load_hallucination_core(hallucination_notebook),
    )
    assert len(pairs) == 6
    assert pairs.columns.tolist() == [
        "Model", "Gene", "SourceResponse", "TargetResponse",
        "SupportRatio", "ContradictionRatio", "EntailmentEstablished",
        "JudgmentLogSHA256",
    ]
    assert gene_rows.to_dict("records") == [{
        "Model": "Claude", "Gene": "UN1",
        "MeanDirectionalContradictionRatio": pytest.approx(0.35),
        "HighContradiction": False,
    }]


def test_compaction_rejects_any_missing_invalid_or_hash_stale_gene(
    formal_judgment_fixture: tuple[Path, list[str], dict[tuple[str, str], str]],
    hallucination_notebook: Path,
) -> None:
    directory, genes, responses = formal_judgment_fixture
    responses[("UN1", "R1")] = "changed response"
    with pytest.raises(ValueError, match="100 valid Claude judgment logs required"):
        compact_claude_judgments(
            directory,
            genes,
            responses,
            load_hallucination_core(hallucination_notebook),
        )


def test_figure_tables_preserve_historical_values_and_fixed_order() -> None:
    source = _historical_source()
    bert_rows = pd.DataFrame(
        [{"Model": "Claude", "Gene": "well_studied-000", "BERTScorePrecision": 0.55}]
    )
    hallucination_rows = pd.DataFrame(
        [{
            "Model": "Claude",
            "Gene": "uncharacterized-000",
            "MeanDirectionalContradictionRatio": 0.35,
        }]
    )
    bert_plot, hallucination_plot = build_figure_tables(
        source, bert_rows, hallucination_rows
    )
    assert bert_plot["Model"].tolist() == ["Phytomni", "Gemini", "Claude", "OpenAI", "Grok"]
    assert hallucination_plot["Model"].tolist() == ["Phytomni", "Gemini", "Claude", "OpenAI", "Grok"]
    assert bert_plot.loc[2, "DisplayLabel"] == "Claude deep research"
    assert hallucination_plot.set_index("Model").loc["Phytomni"].iloc[-1] == pytest.approx(
        0.12216996785802783
    )


def test_provenance_contains_complete_non_secret_lineage(
    formal_judgment_fixture: tuple[Path, list[str], dict[tuple[str, str], str]],
    hallucination_notebook: Path,
) -> None:
    directory, genes, responses = formal_judgment_fixture
    pairs, gene_rows = compact_claude_judgments(
        directory,
        genes,
        responses,
        load_hallucination_core(hallucination_notebook),
    )
    source = _historical_source()
    bert_rows = pd.DataFrame(
        [{"Model": "Claude", "Gene": "well_studied-000", "BERTScorePrecision": 0.55}]
    )
    provenance = build_provenance(
        source=source,
        bertscore=bert_rows,
        hallucination_pairs=pairs,
        hallucination=gene_rows,
        judgment_dir=directory,
        hallucination_notebook=hallucination_notebook,
        core=load_hallucination_core(hallucination_notebook),
    )
    assert provenance["judge"]["model"] == "deepseek-v3.2-exp"
    assert provenance["judge"]["temperature"] == 0
    assert provenance["judge"]["max_tokens"] == 10
    assert provenance["counts"] == {
        "well_studied_genes": 100, "uncharacterized_genes": 100,
        "valid_judgment_logs": 1, "directed_pair_rows": 6,
    }
    assert "api_key" not in json.dumps(provenance).lower()
