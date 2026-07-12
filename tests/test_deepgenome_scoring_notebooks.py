import ast
import hashlib
import json
import re
from itertools import permutations
from pathlib import Path
import tomllib

import nbformat
import numpy as np
import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "DeepGenomeAgent Evaluation"
HALLUCINATION_NOTEBOOK = EVAL_DIR / "score_hallucination.ipynb"
PL_NOTEBOOK = EVAL_DIR / "score_plackett_luce.ipynb"
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{16,}")
RESPONSE_IDS = ["rep_1", "rep_2", "rep_3"]


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


def code_cell_source(path: Path, cell_id: str) -> str:
    notebook = load_notebook(path)
    sources = [
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and cell.id == cell_id
    ]
    assert len(sources) == 1, f"Expected one code cell with ID {cell_id}"
    return sources[0]


def execute_tagged_source(path: Path, tag: str) -> dict[str, object]:
    notebook = load_notebook(path)
    namespace: dict[str, object] = {"__name__": "notebook_contract_test"}
    source = tagged_source(notebook, tag)
    exec(compile(source, str(path), "exec"), namespace)
    return namespace


def metadata_record(response_ids: list[str] | None = None) -> dict[str, object]:
    return {
        "type": "metadata",
        "response_ids": response_ids or RESPONSE_IDS.copy(),
    }


def complete_summary_records(
    contradiction_ratios: list[float] | None = None,
) -> list[dict[str, object]]:
    pairs = list(permutations(range(3), 2))
    ratios = contradiction_ratios or [0.2] * len(pairs)
    assert len(ratios) == len(pairs)
    return [
        {
            "type": "version_pair_summary",
            "version_pair": list(pair),
            "support_ratio": 0.0,
            "contra_ratio": ratio,
            "entailment_established": False,
        }
        for pair, ratio in zip(pairs, ratios, strict=True)
    ]


def complete_primary_log(
    contradiction_ratios: list[float] | None = None,
) -> list[dict[str, object]]:
    return [metadata_record(), *complete_summary_records(contradiction_ratios)]


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
    assert "mean_directional_contradiction_ratio" in text
    assert "high_contradiction_gene_fraction" in text

    detector_source = code_cell_source(
        HALLUCINATION_NOTEBOOK,
        "hallucination-detector",
    )
    analyze_source = detector_source.split(
        "async def analyze_responses",
        maxsplit=1,
    )[1].split("def ordered_judgment_records", maxsplit=1)[0]
    assert '"hallucination_risk": semantic_entropy' in analyze_source


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

    primary_log = complete_primary_log([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])
    assert not validate(primary_log)
    summary_result = extract(primary_log)
    assert summary_result is not None
    assert np.isclose(summary_result[0], 0.2)
    assert summary_result[1:] == (6, "version_pair_summary")
    fallback_log = [
        metadata_record(),
        {"label": "entailment"},
        {"label": "contradiction"},
        {"label": "neutral"},
        {"label": "contradiction"},
    ]
    assert not validate(fallback_log)
    assert extract(fallback_log) == (0.5, 4, "window_labels")
    assert validate([{"error": "timeout", "label": "neutral"}])


def test_hallucination_rejects_structurally_incomplete_evidence() -> None:
    namespace = execute_tagged_source(HALLUCINATION_NOTEBOOK, "hallucination-core")
    validate = namespace["validate_judgment_records"]
    extract = namespace["extract_gene_contradiction"]

    missing_metadata = complete_summary_records()
    duplicate_response_ids = [
        metadata_record(["rep_1", "rep_1", "rep_3"]),
        *complete_summary_records(),
    ]
    five_summaries = [metadata_record(), *complete_summary_records()[:5]]
    duplicate_pairs = complete_primary_log()
    duplicate_pairs[-1] = duplicate_pairs[1].copy()
    malformed_pair = complete_primary_log()
    malformed_pair[1] = {**malformed_pair[1], "version_pair": [0, 3]}
    missing_ratio = complete_primary_log()
    missing_ratio[1] = {
        key: value
        for key, value in missing_ratio[1].items()
        if key != "contra_ratio"
    }
    nonnumeric_ratio = complete_primary_log()
    nonnumeric_ratio[1] = {**nonnumeric_ratio[1], "support_ratio": "0.5"}
    nonfinite_ratio = complete_primary_log()
    nonfinite_ratio[1] = {**nonfinite_ratio[1], "contra_ratio": float("nan")}
    out_of_range_ratio = complete_primary_log()
    out_of_range_ratio[1] = {**out_of_range_ratio[1], "contra_ratio": 1.01}
    nonboolean_entailment = complete_primary_log()
    nonboolean_entailment[1] = {
        **nonboolean_entailment[1],
        "entailment_established": 0,
    }
    invalid_fallback_label = [metadata_record(), {"label": "unknown"}]
    empty_fallback = [metadata_record()]

    malformed_logs = {
        "missing metadata": missing_metadata,
        "duplicate response IDs": duplicate_response_ids,
        "one through five summaries": five_summaries,
        "duplicate ordered pairs": duplicate_pairs,
        "malformed ordered pair": malformed_pair,
        "missing numeric ratio": missing_ratio,
        "nonnumeric ratio": nonnumeric_ratio,
        "nonfinite ratio": nonfinite_ratio,
        "out-of-range ratio": out_of_range_ratio,
        "nonboolean entailment": nonboolean_entailment,
        "invalid fallback label": invalid_fallback_label,
        "empty fallback": empty_fallback,
    }
    for case, records in malformed_logs.items():
        assert validate(records), case
        assert extract(records) is None, case


def test_hallucination_sanitizes_and_records_live_provenance() -> None:
    namespace = execute_tagged_source(HALLUCINATION_NOTEBOOK, "hallucination-core")
    sanitize = namespace["sanitize_api_base_url"]
    build_metadata = namespace["build_judgment_metadata"]

    unsafe_url = (
        "https://private-user:private-password@judge.example:8443/v1"
        "?api_key=private#fragment"
    )
    assert sanitize(unsafe_url) == "https://judge.example:8443/v1"

    metadata = build_metadata(
        model_id="phytomni",
        gene_id="GENE1",
        question="What does this gene do?",
        responses=["first response", "second response", "third response"],
        response_ids=RESPONSE_IDS,
        api_base_url=unsafe_url,
        judge_model="judge-model-v1",
        execution_timestamp_utc="2026-07-13T01:02:03Z",
        package_versions={"networkx": "3.6.1", "openai": "2.0.0"},
    )

    assert metadata["type"] == "metadata"
    assert metadata["api_base_url"] == "https://judge.example:8443/v1"
    assert metadata["judge_model"] == "judge-model-v1"
    assert metadata["execution_timestamp_utc"] == "2026-07-13T01:02:03Z"
    assert metadata["package_versions"] == {
        "networkx": "3.6.1",
        "openai": "2.0.0",
    }
    assert metadata["question_sha256"] == hashlib.sha256(
        b"What does this gene do?"
    ).hexdigest()
    assert metadata["response_sha256"] == [
        hashlib.sha256(response.encode("utf-8")).hexdigest()
        for response in ["first response", "second response", "third response"]
    ]
    assert metadata["response_ids"] == RESPONSE_IDS
    serialized = json.dumps(metadata, sort_keys=True)
    for private_value in [
        "private-user",
        "private-password",
        "api_key=private",
        "What does this gene do?",
        "first response",
    ]:
        assert private_value not in serialized


def test_hallucination_live_preflight_fails_before_client_construction(
    tmp_path: Path,
) -> None:
    namespace = execute_tagged_source(HALLUCINATION_NOTEBOOK, "hallucination-core")
    validate_live_configuration = namespace["validate_live_configuration"]

    with pytest.raises(RuntimeError) as error_info:
        validate_live_configuration(
            query_workbook=None,
            response_root=None,
            judgment_dir=None,
            api_base_url=None,
            api_key=None,
            judge_model=None,
            punkt_tab_available=False,
        )
    message = str(error_info.value)
    for name in [
        "DEEPGENOME_QUERY_WORKBOOK",
        "DEEPGENOME_RESPONSE_ROOT",
        "DEEPGENOME_JUDGMENT_DIR",
        "DEEPGENOME_API_BASE_URL",
        "DEEPGENOME_API_KEY",
        "DEEPGENOME_JUDGE_MODEL",
        "punkt_tab",
    ]:
        assert name in message

    workbook = tmp_path / "queries.xlsx"
    workbook.touch()
    response_root = tmp_path / "responses"
    response_root.mkdir()
    judgment_dir = tmp_path / "judgments"
    judgment_dir.mkdir()
    assert (
        validate_live_configuration(
            query_workbook=workbook,
            response_root=response_root,
            judgment_dir=judgment_dir,
            api_base_url="https://judge.example/v1",
            api_key="private-key",
            judge_model="judge-model-v1",
            punkt_tab_available=True,
        )
        is None
    )

    runner_source = code_cell_source(
        HALLUCINATION_NOTEBOOK,
        "hallucination-live-runner",
    )
    assert runner_source.index("validate_live_configuration(") < runner_source.index(
        "detector = AsyncLongTextHallucinationDetector("
    )
    assert "SKIPPED: Live judging configuration is incomplete" not in runner_source


def test_hallucination_gene_loader_matches_authoritative_baseline(
    tmp_path: Path,
) -> None:
    namespace = execute_tagged_source(
        HALLUCINATION_NOTEBOOK,
        "hallucination-input-core",
    )
    gene_list = tmp_path / "genes.txt"
    gene_list.write_text(
        "this first line is always skipped\n"
        "arabidopsis   GENE_B extra\n"
        "malformed-single-field\n"
        "maize\tGENE_A\tignored\n"
        "rice GENE_B\n",
        encoding="utf-8",
    )

    assert namespace["load_gene_ids"](gene_list) == ["GENE_A", "GENE_B"]


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
    assert result["high_contradiction_gene_fraction"] is None
    assert result["mean_directional_contradiction_ratio"] is None


def test_hallucination_malformed_summary_stubs_cannot_score(
    tmp_path: Path,
) -> None:
    namespace = execute_tagged_source(HALLUCINATION_NOTEBOOK, "hallucination-core")
    aggregate = namespace["aggregate_model_logs"]
    malformed_log = tmp_path / "phytomni__GENE1__rep_1-rep_2-rep_3.json"
    malformed_log.write_text(
        json.dumps(
            [
                metadata_record(),
                *[
                    {
                        "type": "version_pair_summary",
                        "version_pair": list(pair),
                    }
                    for pair in permutations(range(3), 2)
                ],
            ]
        ),
        encoding="utf-8",
    )

    result = aggregate("phytomni", ["GENE1"], tmp_path)

    assert result["valid_log_count"] == 0
    assert result["invalid_log_count"] == 1
    assert result["high_contradiction_gene_fraction"] is None
    assert result["mean_directional_contradiction_ratio"] is None


def test_hallucination_gene_threshold_is_inclusive(tmp_path: Path) -> None:
    namespace = execute_tagged_source(HALLUCINATION_NOTEBOOK, "hallucination-core")
    aggregate = namespace["aggregate_model_logs"]
    boundary_log = tmp_path / "phytomni__GENE1__rep_1-rep_2-rep_3.json"
    boundary_log.write_text(
        json.dumps(complete_primary_log([0.6] * 6)),
        encoding="utf-8",
    )

    result = aggregate("phytomni", ["GENE1"], tmp_path)

    assert result["valid_log_count"] == 1
    assert result["high_contradiction_count"] == 1
    assert result["high_contradiction_gene_fraction"] == 1.0
    assert result["mean_directional_contradiction_ratio"] == 0.6


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


def test_plackett_luce_notebook_contract() -> None:
    assert PL_NOTEBOOK.exists()
    assert not (EVAL_DIR / "cal_score.ipynb").exists()
    assert_clean_notebook(
        PL_NOTEBOOK,
        [
            "# Plackett-Luce Scoring",
            "## Scope and Method",
            "## Reproducibility Configuration",
            "## Load and Validate Rankings",
            "## Fit the Plackett-Luce Model",
            "## Elo-Like Scores and Uncertainty",
            "## Pairwise Win Probabilities",
            "## Reproducibility Checks",
            "## Results and Optional Export",
        ],
    )
    text = notebook_text(load_notebook(PL_NOTEBOOK))
    assert "DEEPGENOME_SCORE_TSV" in text
    assert "DEEPGENOME_SAVE_RESULTS" in text
    assert "Claude" in text
    assert "reference-parameter approximation" in text


def test_plackett_luce_numerical_equivalence() -> None:
    namespace = execute_tagged_source(PL_NOTEBOOK, "plackett-luce-core")
    models = ["Gemini", "Grok", "OpenAI", "Phytomni"]
    rankings = [list(order) for order in permutations(models)]
    rankings.extend([["Phytomni", "OpenAI", "Grok", "Gemini"]] * 6)

    xi0 = {model: 0.0 for model in models}
    log_likelihood, gradient = namespace["pl_loglik_and_grad"](
        xi0,
        rankings,
        models,
        "Gemini",
    )
    assert np.isclose(log_likelihood, -95.34161491043835, atol=1e-12)
    np.testing.assert_allclose(gradient, [-0.5, 2.5, 4.5], atol=1e-12)

    fit = namespace["fit_plackett_luce"](rankings, models)
    assert np.isclose(fit["negative_log_likelihood"], 93.43927901604626, atol=1e-6)
    np.testing.assert_allclose(
        [fit["xi"][model] for model in models],
        [0.0, 0.3126965119, 0.4923752789, 0.6228591999],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        fit["elo"],
        [1437.9857450, 1492.3066929, 1523.5200917, 1546.1874704],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        fit["elo_standard_error"],
        [0.0, 56.4574659, 58.3740115, 59.6279150],
        atol=1e-5,
    )
    probabilities = fit["pairwise_probabilities"]
    assert np.isclose(probabilities[3, 0], 0.6508685493, atol=1e-6)
    assert np.isclose(probabilities[3, 2], 0.5325747750, atol=1e-6)
    assert np.isclose(probabilities[2, 1], 0.5447992300, atol=1e-6)


def test_plackett_luce_parse_rankings_preserves_permissive_semantics() -> None:
    namespace = execute_tagged_source(PL_NOTEBOOK, "plackett-luce-core")
    frame = pd.DataFrame(
        {
            "Gemini": ["R10", "R1", "R5"],
            "Grok": ["R2", "not-a-rank", "R5"],
            "OpenAI": ["R2", "R3", "R5"],
            "Phytomni": ["R-3", "R4", "R5"],
            "Claude": ["R-99", "R0", "R0"],
        }
    )

    rankings, skipped = namespace["parse_rankings"](frame)

    assert skipped == 1
    assert rankings == [
        ["Phytomni", "Grok", "OpenAI", "Gemini"],
        ["Gemini", "Grok", "OpenAI", "Phytomni"],
    ]
    assert all("Claude" not in ranking for ranking in rankings)


def test_plackett_luce_output_tables_have_stable_contract() -> None:
    namespace = execute_tagged_source(PL_NOTEBOOK, "plackett-luce-core")
    models = ["Gemini", "Grok", "OpenAI", "Phytomni"]
    fit = {
        "models": models,
        "elo": np.array([1400.0, 1600.0, 1500.0, 1550.0]),
        "elo_lower": np.array([1390.0, 1590.0, 1490.0, 1540.0]),
        "elo_upper": np.array([1410.0, 1610.0, 1510.0, 1560.0]),
        "pairwise_probabilities": np.arange(16, dtype=float).reshape(4, 4),
    }

    score_table, probability_table = namespace["elo_outputs"](fit)

    assert list(score_table.columns) == ["Model", "Elo", "Elo_L", "Elo_U"]
    assert score_table["Model"].tolist() == [
        "Grok",
        "Phytomni",
        "OpenAI",
        "Gemini",
    ]
    assert probability_table.index.tolist() == models
    assert probability_table.columns.tolist() == models
    assert probability_table.index.name == "Model"


def test_deepgenome_base_test_dependencies_include_networkx() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "networkx>=3.4.2" in metadata["dependency-groups"]["dev"]
    assert "networkx>=3.4.2" in metadata["project"]["optional-dependencies"][
        "deepgenome-eval"
    ]
    assert not any(
        dependency.startswith("networkx")
        for dependency in metadata["project"]["dependencies"]
    )


def test_deepgenome_manifest_probes_canonical_notebooks() -> None:
    manifest = yaml.safe_load((ROOT / "reproduce.manifest.yaml").read_text())
    target = next(item for item in manifest["targets"] if item["id"] == "eval-expert")
    paths = {
        probe["path"]
        for probe in target["probes"]
        if probe.get("type") == "file_exists"
    }
    assert paths == {
        "DeepGenomeAgent Evaluation/score_hallucination.ipynb",
        "DeepGenomeAgent Evaluation/score_plackett_luce.ipynb",
        "DeepGenomeAgent Evaluation/score.tsv",
    }


def test_readme_documents_deepgenome_scoring_contract() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for text in [
        "score_hallucination.ipynb",
        "score_plackett_luce.ipynb",
        "uv sync --frozen --extra deepgenome-eval",
        "mean_directional_contradiction_ratio",
        "high_contradiction_gene_fraction",
        "cross-response inconsistency",
        "DEEPGENOME_SCORE_TSV",
        "fails before constructing the judge client",
    ]:
        assert text in readme
