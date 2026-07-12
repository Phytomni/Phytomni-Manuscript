import ast
import json
import re
from pathlib import Path

import nbformat
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "DeepGenomeAgent Evaluation"
HALLUCINATION_NOTEBOOK = EVAL_DIR / "score_hallucination.ipynb"
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{16,}")


def load_notebook(path: Path) -> nbformat.NotebookNode:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    return notebook


def notebook_text(notebook: nbformat.NotebookNode) -> str:
    return json.dumps(notebook, ensure_ascii=False, sort_keys=True)


def tagged_source(notebook: nbformat.NotebookNode, tag: str) -> str:
    sources = [
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and tag in cell.metadata.get("tags", [])
    ]
    assert sources, f"No code cell tagged {tag}"
    return "\n\n".join(sources)


def execute_tagged_source(path: Path, tag: str) -> dict[str, object]:
    notebook = load_notebook(path)
    namespace: dict[str, object] = {"__name__": "notebook_contract_test"}
    source = tagged_source(notebook, tag)
    exec(compile(source, str(path), "exec"), namespace)
    return namespace


def assert_clean_notebook(path: Path, required_headings: list[str]) -> None:
    notebook = load_notebook(path)
    serialized = notebook_text(notebook)
    assert not CJK_PATTERN.search(serialized)
    assert not SECRET_PATTERN.search(serialized)
    assert "/mnt/c/Users/" not in serialized
    assert "dmxapi" not in serialized.lower()
    assert all(cell.source.strip() for cell in notebook.cells)
    assert all(
        cell.execution_count is None and not cell.outputs
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    code = "\n\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    ast.parse(code)
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    positions = [markdown.index(heading) for heading in required_headings]
    assert positions == sorted(positions)


def test_hallucination_notebook_contract() -> None:
    assert HALLUCINATION_NOTEBOOK.exists()
    assert not (EVAL_DIR / "test_hallucination.ipynb").exists()
    assert not (EVAL_DIR / "cal_hallucination.ipynb").exists()
    assert_clean_notebook(
        HALLUCINATION_NOTEBOOK,
        [
            "# Cross-Response Hallucination Scoring",
            "## Scope and Limitations",
            "## Method",
            "## Reproducibility Configuration",
            "## Load and Validate Inputs",
            "## Pairwise Entailment Judgments",
            "## Semantic-Entropy Diagnostic",
            "## Aggregate Contradiction Scores",
            "## Reproducibility Checks",
            "## Results",
        ],
    )
    text = notebook_text(load_notebook(HALLUCINATION_NOTEBOOK))
    for name in [
        "DEEPGENOME_QUERY_WORKBOOK",
        "DEEPGENOME_RESPONSE_ROOT",
        "DEEPGENOME_JUDGMENT_DIR",
        "DEEPGENOME_API_BASE_URL",
        "DEEPGENOME_API_KEY",
        "DEEPGENOME_JUDGE_MODEL",
    ]:
        assert name in text


def test_hallucination_core_equivalence() -> None:
    namespace = execute_tagged_source(HALLUCINATION_NOTEBOOK, "hallucination-core")
    entropy = namespace["normalized_semantic_entropy"]
    edge = namespace["edge_from_ratios"]
    parse = namespace["parse_judge_label"]
    extract = namespace["extract_gene_contradiction"]
    validate = namespace["validate_judgment_records"]

    assert entropy([[0, 1, 2]], 3) == 0.0
    assert entropy([[0], [1], [2]], 3) == 1.0
    assert np.isclose(entropy([[0, 1], [2]], 3), 0.5793801642856949)
    assert edge(0.6, 0.2)
    assert not edge(np.nextafter(0.6, 0.0), 0.2)
    assert not edge(0.6, np.nextafter(0.2, 1.0))
    assert parse("ENTAILMENT") == "entailment"
    assert parse("contradiction") == "contradiction"
    assert parse("uncertain") == "neutral"

    summaries = [
        {
            "type": "version_pair_summary",
            "version_pair": [index, index + 1],
            "contra_ratio": value,
        }
        for index, value in enumerate([0.1, 0.2, 0.3])
    ]
    summary_result = extract(summaries)
    assert summary_result is not None
    assert np.isclose(summary_result[0], 0.2)
    assert summary_result[1:] == (3, "version_pair_summary")
    labels = [
        {"label": "entailment"},
        {"label": "contradiction"},
        {"label": "neutral"},
        {"label": "contradiction"},
    ]
    assert extract(labels) == (0.5, 4, "window_labels")
    assert validate([{"error": "timeout", "label": "neutral"}])


def test_hallucination_aggregation_omits_scores_without_evidence(
    tmp_path: Path,
) -> None:
    namespace = execute_tagged_source(HALLUCINATION_NOTEBOOK, "hallucination-core")
    aggregate = namespace["aggregate_model_logs"]
    invalid_log = tmp_path / "phytomni__invalid__rep_1-rep_2-rep_3.json"
    invalid_log.write_text("not valid JSON", encoding="utf-8")

    result = aggregate("phytomni", ["missing", "invalid"], tmp_path)

    assert result["valid_log_count"] == 0
    assert result["missing_log_count"] == 1
    assert result["invalid_log_count"] == 1
    assert result["high_contradiction_rate"] is None
    assert result["average_contradiction_ratio"] is None
    assert result["mean_contradiction_ratio"] is None


def test_hallucination_aggregation_runs_after_live_phase() -> None:
    notebook = load_notebook(HALLUCINATION_NOTEBOOK)
    code = "\n\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    live_marker = "live_results = await run_live_evaluation()"
    aggregation_marker = (
        "model_log_summaries = aggregate_configured_model_logs("
    )

    assert live_marker in code
    assert aggregation_marker in code
    assert code.index(live_marker) < code.index(aggregation_marker)


def test_hallucination_semantic_risk_threshold_is_strict() -> None:
    namespace = execute_tagged_source(HALLUCINATION_NOTEBOOK, "hallucination-core")
    assert "semantic_entropy_is_high_risk" in namespace
    is_high_risk = namespace["semantic_entropy_is_high_risk"]

    assert not is_high_risk(0.6)
    assert is_high_risk(np.nextafter(0.6, 1.0))
