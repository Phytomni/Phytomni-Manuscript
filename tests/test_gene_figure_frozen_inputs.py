import hashlib
import json
import re
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from scripts.deepgenome_ranking_statistics import PL_INTERVAL_ANALYSES
from scripts.freeze_deepgenome_rankings import (
    OUTPUT_FILENAMES,
    OUTPUT_SCHEMAS,
    OUTPUT_UNIQUE_KEYS,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "Supplementary Fig. 10-13"
DATA_DIR = FIGURE_DIR / "PhytoBench-Gene-for_plot" / "frozen"
NOTEBOOK = FIGURE_DIR / "supplementary_fig. 10-13.ipynb"
FIG2_DIR = ROOT / "Fig. 2"
FIG2_NOTEBOOK = FIG2_DIR / "fig. 2.ipynb"
FIG2_BERTSCORE = FIG2_DIR / "PhytoBench-Gene-BERTScore-for_plot.tsv"
MODEL_ORDER = ["Phytomni", "Gemini", "Claude", "OpenAI", "Grok"]
MODEL_COLUMNS = ["Gemini", "Grok", "OpenAI", "Phytomni", "Claude"]
STRATIFIED_SCOPES = (
    "well_studied",
    "well_studied.rice",
    "well_studied.maize",
    "well_studied.wheat",
    "well_studied.soybean",
    "well_studied.arabidopsis",
    "uncharacterized",
    "uncharacterized.rice",
    "uncharacterized.maize",
    "uncharacterized.wheat",
    "uncharacterized.soybean",
    "uncharacterized.arabidopsis",
    "rice",
    "maize",
    "wheat",
    "soybean",
    "arabidopsis",
)
EXPECTED_OUTPUT_FILENAMES = {
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
EXPECTED_RANKING_FROZEN_FILES = {
    "rank_distribution.tsv",
    "pl_scores.tsv",
    "pl_pairwise.tsv",
    "pl_scores_ci.tsv",
    "rank_distribution_ci.tsv",
    "pl_pairwise_ci.tsv",
    "fleiss_kappa.tsv",
    "kendall_by_gene.tsv",
    "ordinal_agreement_summary.tsv",
    "top1_consensus.tsv",
    "expert_panel_summary.tsv",
    "assignment_summary.tsv",
    "provenance.json",
}
EXPECTED_CLAUDE_FROZEN_FILES = {
    "PhytoBench-Gene-Claude-BERTScore-by-gene.tsv",
    "PhytoBench-Gene-Claude-hallucination-pairs.tsv",
    "PhytoBench-Gene-Claude-hallucination-by-gene.tsv",
    "PhytoBench-Gene-Claude-metrics-provenance.json",
}
EXPECTED_FROZEN_FILES = (
    EXPECTED_RANKING_FROZEN_FILES | EXPECTED_CLAUDE_FROZEN_FILES
)
EXPECTED_OUTPUT_SCHEMAS = {
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
    "pl_scores_ci": (
        "Scope",
        "StudyStatus",
        "Species",
        "Model",
        "IntervalAnalysis",
        "Estimate",
        "CI95Lower",
        "CI95Upper",
        "NJudgments",
        "NExperts",
        "NGenes",
        "SuccessfulReplicates",
        "FailedFits",
        "SeedStream",
    ),
    "rank_distribution_ci": (
        "Scope",
        "StudyStatus",
        "Species",
        "Model",
        "Rank",
        "Fraction",
        "CI95Lower",
        "CI95Upper",
        "Count",
        "NJudgments",
        "NExperts",
        "NGenes",
        "SuccessfulReplicates",
        "FailedFits",
        "SeedStream",
    ),
    "pl_pairwise_ci": (
        "Scope",
        "StudyStatus",
        "Species",
        "RowModel",
        "ColumnModel",
        "Probability",
        "CI95Lower",
        "CI95Upper",
        "NJudgments",
        "NExperts",
        "NGenes",
        "SuccessfulReplicates",
        "FailedFits",
        "SeedStream",
    ),
    "fleiss_kappa": (
        "ScopeID",
        "AnalysisTier",
        "ScopeFamily",
        "Species",
        "StudyStatus",
        "Model",
        "NGenes",
        "NItems",
        "RatingsPerItem",
        "NRatings",
        "NContributingExperts",
        "ObservedAgreement",
        "ExpectedAgreement",
        "FleissKappa",
        "RankR1Share",
        "RankR2Share",
        "RankR3Share",
        "RankR4Share",
        "RankR5Share",
        "CILower",
        "CIUpper",
        "BootstrapAttempted",
        "BootstrapReplicates",
        "BootstrapInvalid",
        "BootstrapUnit",
        "BootstrapStrata",
        "SeedStream",
    ),
    "kendall_by_gene": (
        "Species",
        "Gene",
        "StudyStatus",
        "NExperts",
        "NModels",
        "KendallW",
        "MeanPairwiseKendallTau",
        "Top1AgreementPattern",
    ),
    "ordinal_agreement_summary": (
        "ScopeID",
        "AnalysisTier",
        "ScopeFamily",
        "Species",
        "StudyStatus",
        "NGenes",
        "NContributingExperts",
        "KendallWMean",
        "KendallWMedian",
        "KendallWQ1",
        "KendallWQ3",
        "KendallWMeanCILower",
        "KendallWMeanCIUpper",
        "KendallWMedianCILower",
        "KendallWMedianCIUpper",
        "MeanPairwiseKendallTauMean",
        "MeanPairwiseKendallTauMedian",
        "MeanPairwiseKendallTauQ1",
        "MeanPairwiseKendallTauQ3",
        "MeanPairwiseKendallTauMeanCILower",
        "MeanPairwiseKendallTauMeanCIUpper",
        "MeanPairwiseKendallTauMedianCILower",
        "MeanPairwiseKendallTauMedianCIUpper",
        "BootstrapAttempted",
        "BootstrapReplicates",
        "BootstrapInvalid",
        "BootstrapUnit",
        "BootstrapStrata",
        "SeedStream",
    ),
    "top1_consensus": (
        "ScopeID",
        "AnalysisTier",
        "ScopeFamily",
        "Species",
        "StudyStatus",
        "Top1AgreementPattern",
        "Count",
        "Fraction",
        "FractionCILower",
        "FractionCIUpper",
        "NGenes",
        "NContributingExperts",
        "BootstrapAttempted",
        "BootstrapReplicates",
        "BootstrapInvalid",
        "BootstrapUnit",
        "BootstrapStrata",
        "SeedStream",
    ),
    "expert_panel_summary": (
        "Dimension",
        "PublicCategory",
        "DisplayOrder",
        "N",
        "DenominatorN",
        "Percent",
        "MissingN",
        "PercentageBasis",
    ),
    "assignment_summary": (
        "Scope",
        "StudyStatus",
        "Species",
        "NExperts",
        "NGenes",
        "NJudgments",
        "MinGenesPerExpert",
        "MaxGenesPerExpert",
        "MinExpertsPerGene",
        "MaxExpertsPerGene",
    ),
}
EXPECTED_OUTPUT_UNIQUE_KEYS = {
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
EXPECTED_ROW_COUNTS = {
    "rank_distribution": 450,
    "pl_scores": 90,
    "pl_pairwise": 450,
    "pl_scores_ci": 270,
    "rank_distribution_ci": 450,
    "pl_pairwise_ci": 360,
    "fleiss_kappa": 58,
    "kendall_by_gene": 200,
    "ordinal_agreement_summary": 18,
    "top1_consensus": 54,
    "expert_panel_summary": 44,
    "assignment_summary": 18,
}
EXPECTED_INPUT_SHA256 = {
    "public_ranking_release": (
        "9c8fefddc4cdc83da8701ba7e142d6f62a158824c19576ac3155664d34cb51b2"
    ),
    "private_lineage_score": (
        "bf24408d8e3d68ca11cc7319b25c407a29d2e26301fdc812896dd451818adcbe"
    ),
    "gene_categories": (
        "bcb5695c5efaba3faeedd9efa636accd4cbba69a98a996de2eea2900a692fe52"
    ),
    "expert_metadata": (
        "dc8048fed6200709f40646d5ed62d4e7aab5d94d3c4a528a07cde85e7243242f"
    ),
    "panel_category_map": (
        "5d4d110830859149ee16fc7f5c49dc1ba6117a7b4adc1c30df0c1132ff937489"
    ),
    "narrative_notebook": (
        "8e2f1c1b83aa96c7261928fd2cc9e59327e305ce9ec19dfb3654edbf73aabec8"
    ),
    "statistical_module": (
        "7c90b123ef407f0ae1ed421fe8693dfae84938533f1602f29cf6ec00ac5b0b84"
    ),
    "panel_category_audit_module": (
        "d5a1dcb7535c85c1945431e9ecdb8ab13ba990eb61f7c6137075bbdbe20103fe"
    ),
    "freezer_module": (
        "30eb8a9618af07ab338696ee8823f4c4cccd5e43ce4d0ca427cf081f725a5962"
    ),
}
PANEL_DIMENSIONS = {
    "Species",
    "Country/Region",
    "Institution_type",
    "Current_position",
    "Years_experience",
    "Gender",
    "Research_domains",
    "Study_species",
    "Annotation_experience",
    "Ai_experience",
    "Conflict_interest",
}


def notebook_source() -> str:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    assert all(
        cell.execution_count is None and not cell.outputs
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    return "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )


def test_frozen_tables_are_complete_and_traceable() -> None:
    assert OUTPUT_FILENAMES == EXPECTED_OUTPUT_FILENAMES
    assert OUTPUT_SCHEMAS == EXPECTED_OUTPUT_SCHEMAS
    assert OUTPUT_UNIQUE_KEYS == EXPECTED_OUTPUT_UNIQUE_KEYS
    assert DATA_DIR.is_dir()
    assert {path.name for path in DATA_DIR.iterdir()} == (
        EXPECTED_FROZEN_FILES
    )

    tables = {
        name: pd.read_csv(DATA_DIR / filename, sep="\t")
        for name, filename in OUTPUT_FILENAMES.items()
    }
    for name, table in tables.items():
        assert tuple(table.columns) == EXPECTED_OUTPUT_SCHEMAS[name]
        assert len(table) == EXPECTED_ROW_COUNTS[name]
        assert not table.duplicated(
            list(EXPECTED_OUTPUT_UNIQUE_KEYS[name])
        ).any()
        assert {
            "Expert",
            "Expert_ID",
            "AnonymousExpertID",
        }.isdisjoint(table.columns)

    ranks = tables["rank_distribution"]
    scores = tables["pl_scores"]
    pl_scores_ci = tables["pl_scores_ci"]
    rank_ci = tables["rank_distribution_ci"]
    pairwise_ci = tables["pl_pairwise_ci"]
    fleiss = tables["fleiss_kappa"]
    kendall = tables["kendall_by_gene"]
    ordinal = tables["ordinal_agreement_summary"]
    top1 = tables["top1_consensus"]
    panel = tables["expert_panel_summary"]
    assignment = tables["assignment_summary"]

    claude_bertscore = pd.read_csv(
        DATA_DIR / "PhytoBench-Gene-Claude-BERTScore-by-gene.tsv",
        sep="\t",
    )
    assert claude_bertscore.columns.tolist() == [
        "Model",
        "Gene",
        "BERTScorePrecision",
        "QuerySHA256",
        "ResponseSHA256",
    ]
    assert len(claude_bertscore) == 100
    assert claude_bertscore["Model"].eq("Claude").all()
    assert claude_bertscore["Gene"].is_unique
    assert claude_bertscore["BERTScorePrecision"].between(0.0, 1.0).all()

    claude_hallucination = pd.read_csv(
        DATA_DIR / "PhytoBench-Gene-Claude-hallucination-by-gene.tsv",
        sep="\t",
    )
    assert claude_hallucination.columns.tolist() == [
        "Species",
        "Gene",
        "StudyStatus",
        "Model",
        "DirectionalPairCount",
        "MeanDirectionalContradictionRatio",
        "HighContradiction",
    ]
    assert len(claude_hallucination) == 100
    assert claude_hallucination["Model"].eq("Claude").all()
    assert claude_hallucination["Gene"].is_unique
    assert claude_hallucination["DirectionalPairCount"].eq(6).all()
    assert claude_hallucination[
        "MeanDirectionalContradictionRatio"
    ].between(0.0, 1.0).all()

    claude_pairs = pd.read_csv(
        DATA_DIR / "PhytoBench-Gene-Claude-hallucination-pairs.tsv",
        sep="\t",
    )
    assert len(claude_pairs) == 600
    assert claude_pairs["Model"].eq("Claude").all()
    assert (
        claude_pairs[["Gene", "SourceResponseID", "TargetResponseID"]]
        .duplicated()
        .sum()
        == 0
    )
    assert claude_pairs["WindowJudgmentCount"].gt(0).all()
    assert claude_pairs["ContradictionRatio"].between(0.0, 1.0).all()

    assert ranks["Scope"].nunique() == 18
    assert set(ranks["Model"]) == set(MODEL_ORDER)
    np.testing.assert_allclose(
        ranks.groupby(["Scope", "Model"])["Fraction"].sum(), 1.0
    )

    assert PL_INTERVAL_ANALYSES == (
        "crossed_expert_gene",
        "expert_cluster",
        "gene_cluster",
    )
    assert set(pl_scores_ci["IntervalAnalysis"]) == {
        "crossed_expert_gene",
        "expert_cluster",
        "gene_cluster",
    }
    for frame, estimate in (
        (pl_scores_ci, "Estimate"),
        (rank_ci, "Fraction"),
        (pairwise_ci, "Probability"),
    ):
        assert (frame["CI95Lower"] <= frame[estimate]).all()
        assert (frame[estimate] <= frame["CI95Upper"]).all()

    assert fleiss["ScopeID"].is_unique
    assert fleiss["ScopeFamily"].value_counts().to_dict() == {
        "model_species": 25,
        "species_study_status": 10,
        "model_study_status": 10,
        "species": 5,
        "model": 5,
        "study_status": 2,
        "overall": 1,
    }
    assert not fleiss["ScopeFamily"].str.contains(
        "model_species_study_status|species_model_study_status"
    ).any()
    assert (fleiss["NRatings"] == 3 * fleiss["NItems"]).all()
    np.testing.assert_allclose(
        fleiss["FleissKappa"],
        (fleiss["ObservedAgreement"] - fleiss["ExpectedAgreement"])
        / (1.0 - fleiss["ExpectedAgreement"]),
        rtol=0.0,
        atol=1e-12,
    )

    assert ordinal["ScopeID"].is_unique
    assert kendall[["Species", "Gene"]].drop_duplicates().shape[0] == 200
    assert kendall["KendallW"].between(0.0, 1.0).all()
    assert kendall["MeanPairwiseKendallTau"].between(-1.0, 1.0).all()
    assert top1["ScopeID"].nunique() == 18
    assert top1.groupby("ScopeID").size().eq(3).all()
    np.testing.assert_allclose(top1.groupby("ScopeID")["Fraction"].sum(), 1.0)
    overall_assignment = assignment.loc[
        assignment["Scope"] == "overall"
    ].squeeze()
    assert overall_assignment[["NExperts", "NGenes", "NJudgments"]].tolist() == [
        120,
        200,
        600,
    ]
    assert (assignment["MinExpertsPerGene"] == 3).all()
    assert (assignment["MaxExpertsPerGene"] == 3).all()
    assert (assignment["NJudgments"] == 3 * assignment["NGenes"]).all()

    assert set(panel["Dimension"]) == PANEL_DIMENSIONS
    assert (panel["N"] >= 5).all()
    assert (panel["DenominatorN"] > 0).all()
    assert (panel["MissingN"] >= 0).all()
    np.testing.assert_allclose(
        panel["Percent"], 100.0 * panel["N"] / panel["DenominatorN"]
    )
    for dimension, selected in panel.groupby("Dimension", sort=False):
        assert selected["DenominatorN"].nunique() == 1
        assert selected["MissingN"].nunique() == 1
        assert selected["PercentageBasis"].nunique() == 1
        if dimension in {"Research_domains", "Study_species"}:
            assert selected["PercentageBasis"].iat[0] == "all_experts"
            assert selected["DenominatorN"].iat[0] == 120
        else:
            assert selected["PercentageBasis"].iat[0] == "nonmissing_experts"
            assert selected["N"].sum() == selected["DenominatorN"].iat[0]
            assert (
                selected["DenominatorN"].iat[0] + selected["MissingN"].iat[0]
                == 120
            )
    category_map = pd.read_csv(
        ROOT
        / "DeepGenomeAgent Evaluation"
        / "supplementary"
        / "expert_panel_category_map.tsv",
        sep="\t",
    )
    public_panel_categories = category_map[
        ["Dimension", "PublicCategory", "DisplayOrder"]
    ].drop_duplicates()
    pd.testing.assert_frame_equal(
        panel[["Dimension", "PublicCategory", "DisplayOrder"]]
        .sort_values(["Dimension", "DisplayOrder", "PublicCategory"])
        .reset_index(drop=True),
        public_panel_categories.sort_values(
            ["Dimension", "DisplayOrder", "PublicCategory"]
        ).reset_index(drop=True),
        check_dtype=False,
    )
    assert panel.loc[
        panel["Dimension"] == "Conflict_interest"
    ].to_dict("records") == [
        {
            "Dimension": "Conflict_interest",
            "PublicCategory": "No",
            "DisplayOrder": 1,
            "N": 120,
            "DenominatorN": 120,
            "Percent": 100.0,
            "MissingN": 0,
            "PercentageBasis": "nonmissing_experts",
        }
    ]
    panel_species = panel.loc[
        panel["Dimension"] == "Species",
        ["PublicCategory", "N"],
    ].set_index("PublicCategory")["N"]
    assignment_species = assignment.loc[
        assignment["StudyStatus"].eq("all")
        & assignment["Species"].ne("all"),
        ["Species", "NExperts"],
    ].set_index("Species")["NExperts"]
    pd.testing.assert_series_equal(
        panel_species.sort_index(),
        assignment_species.sort_index(),
        check_names=False,
    )

    frozen_payload = b"\n".join(
        (DATA_DIR / filename).read_bytes()
        for filename in OUTPUT_FILENAMES.values()
    )
    for forbidden in (
        b"AnonymousExpertID",
        b"Expert_ID",
        b"raw identifier",
        b"free text",
    ):
        assert forbidden not in frozen_payload
    assert re.search(rb"\bE\d{3}\b", frozen_payload) is None

    provenance = json.loads((DATA_DIR / "provenance.json").read_text())
    assert provenance["schema_version"] == 2
    assert provenance["reporting_matrix_status"] == (
        "locked_before_final_bootstrap_reanalysis"
    )
    assert provenance["reporting_matrix_statement"] == (
        "The reporting matrix was locked before the final bootstrap reanalysis "
        "and before manuscript interpretation."
    )
    assert provenance["monte_carlo_qc_amendment_statement"] == (
        "The Monte Carlo quality-control rule was amended after the "
        "prespecified half-run tolerance failed and before final result "
        "interpretation; the estimands, reporting matrix, resampling scheme, "
        "seed, and replicate target remained unchanged."
    )
    assert provenance["pilot_kappa_values_viewed"] is True
    assert provenance["fleiss_scope_count"] == 58
    assert provenance["ordinal_scope_count"] == 18
    assert provenance["model_columns"] == MODEL_COLUMNS
    assert provenance["bootstrap"] == {
        "master_seed": 20260714,
        "successful_replicates": 10_000,
        "maximum_failed_fits": 10,
        "primary_pl_interval": "crossed_expert_gene_percentile",
        "agreement_interval": "stratified_gene_block_percentile",
    }

    assert set(provenance["inputs"]) == set(EXPECTED_INPUT_SHA256)
    for name, digest in EXPECTED_INPUT_SHA256.items():
        assert provenance["inputs"][name] == {"sha256": digest}

    pl_diagnostics = provenance["bootstrap_diagnostics"]["plackett_luce"]
    assert pl_diagnostics["SuccessfulReplicates"] == 10_000
    assert 0 <= pl_diagnostics["FailedFits"] <= 10
    assert pl_diagnostics["AttemptedReplicates"] == (
        pl_diagnostics["SuccessfulReplicates"]
        + pl_diagnostics["FailedFits"]
    )
    assert "FailureReasons" not in pl_diagnostics
    half_run = pl_diagnostics["HalfRunStability"]
    assert half_run["Applied"] is True
    assert half_run["Interpretation"] == "descriptive_nonblocking"
    np.testing.assert_allclose(
        [
            half_run["ScoreMaxBoundDifference"],
            half_run["ProbabilityMaxBoundDifference"],
        ],
        [12.483900008886167, 0.013095437435215185],
        rtol=0.0,
        atol=1e-12,
    )
    assert "threshold" not in str(half_run).casefold()
    assert "passed" not in str(half_run).casefold()
    precision = pl_diagnostics["MonteCarloPrecision"]
    assert precision["Applied"] is True
    assert precision["Method"] == "binomial_order_statistic"
    assert precision["Replicates"] == 10_000
    assert precision["TailProbability"] == 0.025
    assert precision["PercentileProbabilityStandardError"] == (
        np.sqrt(0.025 * 0.975 / 10_000)
    )
    assert precision["TailProbabilityRelativeStandardError"] == (
        np.sqrt(0.025 * 0.975 / 10_000) / 0.025
    )
    assert precision["EndpointCounts"] == {
        "CrossedScore": 180,
        "NonredundantPairwiseProbability": 360,
        "Total": 540,
    }
    assert precision["PairwiseDirectionRule"] == (
        "canonical_model_index_row_less_than_column"
    )
    assert precision["Pointwise95"]["RankBrackets"] == {
        "CI95Lower": [220, 282],
        "CI95Upper": [9719, 9781],
    }
    assert precision["BonferroniFamilywise95"]["RankBrackets"] == {
        "CI95Lower": [191, 314],
        "CI95Upper": [9687, 9810],
    }
    for family in ("Pointwise95", "BonferroniFamilywise95"):
        for metric_family in ("Score", "PairwiseProbability"):
            metrics = precision[family][metric_family]
            assert metrics["MaximumBracketDistance"]["Value"] >= 0.0
            assert metrics["MaximumRelativeCIWidth"]["Value"] >= 0.0
            assert metrics["UndefinedRelativeCIWidthCount"] >= 0
    assert "threshold" not in str(precision).casefold()
    assert "passed" not in str(precision).casefold()
    agreement_diagnostics = provenance["bootstrap_diagnostics"]["agreement"]
    assert agreement_diagnostics["successful_replicates"] == 10_000
    assert 0 <= agreement_diagnostics["invalid_replicates"] <= 10
    assert agreement_diagnostics["attempted_replicates"] == (
        agreement_diagnostics["successful_replicates"]
        + agreement_diagnostics["invalid_replicates"]
    )

    assert set(provenance["outputs"]) == set(OUTPUT_FILENAMES.values())
    for filename, digest in provenance["outputs"].items():
        assert hashlib.sha256((DATA_DIR / filename).read_bytes()).hexdigest() == (
            digest
        )
    assert not {
        "source",
        "rows",
        "used_rows",
        "skipped_rows",
        "scopes",
    } & set(provenance)

    overall = scores[scores["Scope"] == "overall"].set_index("Model")
    np.testing.assert_allclose(
        overall.loc[MODEL_ORDER, "Elo"],
        [1612.2511259530193, 1547.6162663668567, 1532.4668216858088,
         1487.5809511985851, 1320.0848347957306],
        atol=1e-10,
    )

    claude_provenance_path = (
        DATA_DIR / "PhytoBench-Gene-Claude-metrics-provenance.json"
    )
    claude_provenance = json.loads(claude_provenance_path.read_text())
    assert claude_provenance["schema_version"] == 1
    assert claude_provenance["counts"] == {
        "archive_member_count": 400,
        "directed_pair_rows": 600,
        "extra_judgment_logs": 0,
        "invalid_judgment_logs": 0,
        "missing_judgment_logs": 0,
        "uncharacterized_genes": 100,
        "valid_judgment_logs": 100,
        "well_studied_genes": 100,
    }
    assert claude_provenance["judge"] == {
        "api_base_url": "https://api.modelarts-maas.com/v2",
        "max_concurrent": 4,
        "max_tokens": 10,
        "model": "deepseek-v3.2",
        "prompt_sha256": (
            "d2cf7c33fe97c307be3ac471f40b2260d48b42068ef4fd6bdf3cfe9d76b72e2e"
        ),
        "temperature": 0,
        "window_size_sentences": 3,
        "window_stride_sentences": 2,
    }
    for table_name, digest_key in (
        (
            "PhytoBench-Gene-Claude-BERTScore-by-gene.tsv",
            "bertscore_by_gene_sha256",
        ),
        (
            "PhytoBench-Gene-Claude-hallucination-by-gene.tsv",
            "hallucination_by_gene_sha256",
        ),
        (
            "PhytoBench-Gene-Claude-hallucination-pairs.tsv",
            "hallucination_pairs_sha256",
        ),
    ):
        assert hashlib.sha256((DATA_DIR / table_name).read_bytes()).hexdigest() == (
            claude_provenance["tables"][digest_key]
        )


def test_supplementary_notebook_only_plots_frozen_five_model_results() -> None:
    source = notebook_source()

    for filename in EXPECTED_RANKING_FROZEN_FILES:
        assert filename in source
    for model in MODEL_ORDER:
        assert f'"{model}"' in source
    assert "calculate_pl_elo" not in source
    assert "pl_loglik_and_grad" not in source
    assert "sp[-4:]" not in source
    assert "np.ix_([3, 0, 2, 1]" not in source
    assert 'y=100.0 * rank_matrix[rank_label]' in source
    assert "PhytoBench-Gene-for_plot/score.tsv" not in source


def test_supplementary_notebook_encodes_ci_and_agreement_figures() -> None:
    source = notebook_source()

    assert "schema_version" in source
    assert "successful_replicates" in source
    assert '"crossed_expert_gene"' in source
    assert "CI95Lower" in source
    assert "CI95Upper" in source
    assert "expert_panel_summary" in source
    assert "assignment_summary" in source
    assert "fleiss_kappa" in source
    assert "kendall_by_gene" in source
    assert "top1_consensus" in source
    assert "1.0 / 3.0" in source
    assert "No conflicts of interest were declared" in source
    assert "supplementary_fig.10.phytobench-gene" in source
    assert "supplementary_fig.11.expert-panel-and-agreement" in source
    assert "supplementary_fig.12.phytobench-gene" in source
    assert "supplementary_fig.13.phytobench-gene" in source
    assert "supplementary_fig.7.phytobench-gene" not in source
    assert "supplementary_fig.8.phytobench-gene" not in source
    assert "supplementary_fig.9.phytobench-gene" not in source


def test_pairwise_heatmap_shows_point_and_range_with_blank_diagonal(
    monkeypatch,
) -> None:
    pairwise = pd.read_csv(DATA_DIR / "pl_pairwise.tsv", sep="\t")
    diagonal = pairwise[pairwise["RowModel"] == pairwise["ColumnModel"]]

    assert len(diagonal) == 18 * len(MODEL_ORDER)
    assert diagonal["Probability"].isna().all()

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace: dict[str, object] = {}
    required_cells = {
        "ranking-setup",
        "ranking-configuration",
        "frozen-validation",
        "main-plot-functions",
    }
    monkeypatch.chdir(FIGURE_DIR)
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.id in required_cells:
            exec(compile(cell.source, f"<{cell.id}>", "exec"), namespace)

    figure = namespace["pairwise_probability_figure"]("overall")
    probabilities = np.asarray(figure.data[0].z, dtype=float)
    text = np.asarray(figure.data[0].text, dtype=str)
    off_diagonal = ~np.eye(len(MODEL_ORDER), dtype=bool)

    assert np.isnan(np.diag(probabilities)).all()
    assert np.isfinite(probabilities[off_diagonal]).all()
    assert (np.diag(text) == "").all()
    intervals = (
        pd.read_csv(DATA_DIR / "pl_pairwise_ci.tsv", sep="\t")
        .loc[lambda frame: frame["Scope"].eq("overall")]
        .set_index(["RowModel", "ColumnModel"])
    )
    for row_index, row_model in enumerate(MODEL_ORDER):
        for column_index, column_model in enumerate(MODEL_ORDER):
            if row_model == column_model:
                continue
            interval = intervals.loc[(row_model, column_model)]
            assert text[row_index, column_index] == (
                f"{probabilities[row_index, column_index]:.2f}<br>"
                f"[{interval['CI95Lower']:.2f}, {interval['CI95Upper']:.2f}]"
            )


def test_pairwise_heatmap_preserves_the_original_tropic_presentation(
    monkeypatch,
) -> None:
    from plotly.colors import get_colorscale

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace: dict[str, object] = {}
    required_cells = {
        "ranking-setup",
        "ranking-configuration",
        "frozen-validation",
        "main-plot-functions",
    }
    monkeypatch.chdir(FIGURE_DIR)
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.id in required_cells:
            exec(compile(cell.source, f"<{cell.id}>", "exec"), namespace)

    figure = namespace["pairwise_probability_figure"]("overall")
    heatmap = figure.data[0]
    expected_scale = get_colorscale("TroPic")

    assert heatmap.type == "heatmap"
    np.testing.assert_allclose(
        [stop for stop, _ in heatmap.colorscale],
        [stop for stop, _ in expected_scale],
    )
    assert [color for _, color in heatmap.colorscale] == [
        color for _, color in expected_scale
    ]
    assert heatmap.xgap is None
    assert heatmap.ygap is None
    assert figure.layout.yaxis.scaleanchor == "x"
    assert figure.layout.yaxis.autorange != "reversed"
    assert heatmap.colorbar.title.text is None
    assert figure.layout.width == 1080
    assert figure.layout.height == 1080
    assert figure.layout.title.text is None
    visible_text = np.asarray(heatmap.text, dtype=str)
    assert all(
        "<br>[" in value and value.endswith("]")
        for value in visible_text[~np.eye(len(MODEL_ORDER), dtype=bool)]
    )


def test_elo_figure_preserves_bars_and_adds_asymmetric_intervals(
    monkeypatch,
) -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace: dict[str, object] = {}
    required_cells = {
        "ranking-setup",
        "ranking-configuration",
        "frozen-validation",
        "main-plot-functions",
    }
    monkeypatch.chdir(FIGURE_DIR)
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.id in required_cells:
            exec(compile(cell.source, f"<{cell.id}>", "exec"), namespace)

    assert "elo_score_figure" in namespace
    supplementary = namespace["elo_score_figure"]("overall")
    trace = supplementary.data[0]
    expected_colors = [
        "rgb(31,113,179)",
        "rgb(208,210,211)",
        "rgb(208,210,211)",
        "rgb(208,210,211)",
        "rgb(208,210,211)",
    ]

    assert len(supplementary.data) == 1
    assert trace.type == "bar"
    assert trace.orientation is None
    assert list(trace.x) == [
        "Phytomni",
        "Gemini Deep Research",
        "Claude deep research",
        "ChatGPT Agent mode",
        "Grok DeepSearch",
    ]
    assert list(trace.marker.color) == expected_colors
    assert trace.marker.line.color == "rgb(0,0,0)"
    assert trace.marker.line.width == 2
    assert trace.textposition == "outside"
    assert trace.error_y.symmetric is False
    assert np.asarray(trace.error_y.array, dtype=float).shape == (5,)
    assert np.asarray(trace.error_y.arrayminus, dtype=float).shape == (5,)
    assert (np.asarray(trace.error_y.array, dtype=float) > 0).all()
    assert (np.asarray(trace.error_y.arrayminus, dtype=float) > 0).all()
    expected = (
        pd.read_csv(DATA_DIR / "pl_scores_ci.tsv", sep="\t")
        .loc[
            lambda frame: frame["Scope"].eq("overall")
            & frame["IntervalAnalysis"].eq("crossed_expert_gene")
        ]
        .set_index("Model")
        .reindex(MODEL_ORDER)
    )
    np.testing.assert_allclose(trace.y, expected["Estimate"])
    np.testing.assert_allclose(
        trace.error_y.array,
        expected["CI95Upper"] - expected["Estimate"],
    )
    np.testing.assert_allclose(
        trace.error_y.arrayminus,
        expected["Estimate"] - expected["CI95Lower"],
    )
    assert tuple(supplementary.layout.yaxis.range) == (1000, 2000)
    assert supplementary.layout.width == 1080
    assert supplementary.layout.height == 1080
    assert supplementary.layout.title.text is None
    assert not supplementary.layout.shapes
    assert not supplementary.layout.annotations

    captured: dict[str, object] = {}
    namespace["render_figure"] = (
        lambda figure, file_prefix: captured.__setitem__(file_prefix, figure)
    )
    render_cell = next(cell for cell in notebook.cells if cell.id == "figure-2-render")
    exec(compile(render_cell.source, "<figure-2-render>", "exec"), namespace)
    main = captured["fig.2f.phytobench-gene.score.bar"]
    assert tuple(main.layout.yaxis.range) == (1200, 1700)


def test_supplementary_figure_11_uses_manuscript_species_order(
    monkeypatch,
) -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace: dict[str, object] = {}
    required_cells = {
        "ranking-setup",
        "ranking-configuration",
        "frozen-validation",
        "main-plot-functions",
        "agreement-plot-functions",
    }
    monkeypatch.chdir(FIGURE_DIR)
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.id in required_cells:
            exec(compile(cell.source, f"<{cell.id}>", "exec"), namespace)

    figure = namespace["expert_agreement_figure"]()
    expected = ["Rice", "Wheat", "Maize", "Soybean", "Arabidopsis"]
    panel_dimensions, panel_categories = figure.data[0].y
    for dimension in ("Species", "Study species†"):
        categories = [
            category
            for row_dimension, category in zip(
                panel_dimensions,
                panel_categories,
                strict=True,
            )
            if row_dimension == dimension
        ]
        assert categories[:5] == expected

    kappa_species = [
        trace.y[0]
        for trace in figure.data
        if (
            len(trace.y) == 1
            and trace.y[0] in expected
            and "κ=" in trace.hovertemplate
        )
    ]
    assert kappa_species == expected
    assert list(figure.layout.yaxis2.categoryarray)[1:6] == expected
    kendall_w_species = [
        trace.y[0]
        for trace in figure.data
        if (
            len(trace.y) == 1
            and trace.y[0] in expected
            and "Mean W=" in trace.hovertemplate
        )
    ]
    assert kendall_w_species == expected
    assert list(figure.layout.yaxis3.categoryarray)[1:6] == expected
    unanimous = next(
        trace for trace in figure.data if trace.name == "Top-1: Unanimous"
    )
    assert [label.split(" (", maxsplit=1)[0] for label in unanimous.y[1:6]] == expected


def test_rendered_figure_contract_preserves_precision_and_readability(
    monkeypatch,
) -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace: dict[str, object] = {}
    required_cells = {
        "ranking-setup",
        "ranking-configuration",
        "frozen-validation",
        "main-plot-functions",
        "agreement-plot-functions",
    }
    monkeypatch.chdir(FIGURE_DIR)
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.id in required_cells:
            exec(compile(cell.source, f"<{cell.id}>", "exec"), namespace)

    supplementary = namespace["expert_agreement_figure"]()
    assert supplementary.layout.xaxis3.title.text == (
        "Mean Kendall's W (95% CI)"
    )
    assert len(supplementary.layout.shapes) == 2
    kendall_w_source = pd.read_csv(
        DATA_DIR / "ordinal_agreement_summary.tsv",
        sep="\t",
    ).set_index("ScopeID")
    agreement_scope_ids = [
        "overall",
        "species.rice",
        "species.wheat",
        "species.maize",
        "species.soybean",
        "species.arabidopsis",
        "study_status.well_studied",
        "study_status.uncharacterized",
    ]
    kendall_w_traces = [
        trace
        for trace in supplementary.data
        if (
            len(trace.y) == 1
            and trace.hovertemplate
            and "Mean W=" in trace.hovertemplate
        )
    ]
    np.testing.assert_allclose(
        [trace.x[0] for trace in kendall_w_traces],
        kendall_w_source.loc[agreement_scope_ids, "KendallWMean"],
    )
    kendall_w_baseline = next(
        shape
        for shape in supplementary.layout.shapes
        if shape.xref == "x3"
    )
    assert np.isclose(kendall_w_baseline.x0, 1 / 3)
    assert np.isclose(kendall_w_baseline.x1, 1 / 3)
    top1_traces = {
        trace.name: trace
        for trace in supplementary.data
        if trace.name and trace.name.startswith("Top-1:")
    }
    unanimous = top1_traces["Top-1: Unanimous"]
    majority = top1_traces["Top-1: 2-of-3 majority"]
    assert unanimous.text[0] == "12.5%"
    assert unanimous.text[2] == "5.0%"
    assert majority.text[0] == "53.5%"

    model_kappa_traces = [
        trace
        for trace in supplementary.data
        if (
            len(trace.y) == 1
            and trace.y[0]
            in {
                "Phytomni",
                "Gemini Deep Research",
                "Claude deep research",
                "ChatGPT Agent mode",
                "Grok DeepSearch",
            }
            and "κ=" in trace.hovertemplate
        )
    ]
    assert not model_kappa_traces

    rank_figure = namespace["rank_distribution_figure"]("overall")
    assert rank_figure.layout.legend.x <= 0.98


def test_supplementary_notebook_uses_requested_claude_label_and_order() -> None:
    source = notebook_source()

    assert 'MODEL_ORDER = ["Phytomni", "Gemini", "Claude", "OpenAI", "Grok"]' in source
    assert '"Claude": "Claude deep research"' in source


def test_fig2g_and_fig2h_read_frozen_five_model_metrics() -> None:
    bertscore = pd.read_csv(FIG2_BERTSCORE, sep="\t")
    assert bertscore.columns.tolist() == [
        "Model",
        "DisplayLabel",
        "BERTScorePrecision",
    ]
    assert bertscore["Model"].tolist() == [
        "Phytomni",
        "Gemini",
        "Claude",
        "OpenAI",
        "Grok",
    ]
    assert bertscore["DisplayLabel"].tolist() == [
        "Phytomni",
        "Gemini Deep Research",
        "Claude deep research",
        "ChatGPT Agent mode",
        "Grok DeepSearch",
    ]

    hallucination = pd.read_csv(
        FIG2_DIR / "PhytoBench-Gene-hallucination-for_plot.tsv",
        sep="\t",
    )
    assert hallucination.columns.tolist() == [
        "Model",
        "DisplayLabel",
        "MeanDirectionalContradictionRatio",
    ]
    assert hallucination["Model"].tolist() == MODEL_ORDER
    assert hallucination["DisplayLabel"].tolist() == (
        bertscore["DisplayLabel"].tolist()
    )

    notebook = nbformat.read(FIG2_NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    assert all(
        cell.source.strip()
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    assert "PhytoBench-Gene-BERTScore-for_plot.tsv" in source
    assert "PhytoBench-Gene-hallucination-for_plot.tsv" in source
    assert "if HALLUCINATION_DATA.is_file():" not in source
    assert "SKIPPED: Fig. 2h" not in source
    assert "range': [0.47, 0.58]" in source
    assert '"range": [0.1, 0.7]' in source
    assert "0.561641241" not in source
    assert "0.12216996785802783" not in source


def test_manifest_connects_fig2def_and_marks_fig2h_runnable() -> None:
    import yaml

    manifest = yaml.safe_load((ROOT / "reproduce.manifest.yaml").read_text())
    targets = {target["id"]: target for target in manifest["targets"]}

    fig2def = targets["fig-2def"]
    assert fig2def["path"] == (
        "Supplementary Fig. 10-13/supplementary_fig. 10-13.ipynb"
    )
    expected_frozen_inputs = {
        "Supplementary Fig. 10-13/PhytoBench-Gene-for_plot/frozen/"
        f"{filename}"
        for filename in EXPECTED_RANKING_FROZEN_FILES
    }
    assert set(fig2def["requires_data"]) == expected_frozen_inputs
    assert set(fig2def["expected_artifacts"]) == {
        "Supplementary Fig. 10-13/output/"
        "fig.2d.phytobench-gene.percent.bar.pdf",
        "Supplementary Fig. 10-13/output/"
        "fig.2e.phytobench-gene.prob.heatmap.pdf",
        "Supplementary Fig. 10-13/output/"
        "fig.2f.phytobench-gene.score.bar.pdf",
    }

    supplementary = targets["supp-10-13"]
    assert set(supplementary["requires_data"]) == expected_frozen_inputs
    expected_supplementary_artifacts = {
        "Supplementary Fig. 10-13/output/"
        f"supplementary_fig.10.phytobench-gene.{scope}.percent.bar.pdf"
        for scope in STRATIFIED_SCOPES
    }
    expected_supplementary_artifacts.add(
        "Supplementary Fig. 10-13/output/"
        "supplementary_fig.11.expert-panel-and-agreement.pdf"
    )
    expected_supplementary_artifacts.update(
        "Supplementary Fig. 10-13/output/"
        f"supplementary_fig.12.phytobench-gene.{scope}.prob.heatmap.pdf"
        for scope in STRATIFIED_SCOPES
    )
    expected_supplementary_artifacts.update(
        "Supplementary Fig. 10-13/output/"
        f"supplementary_fig.13.phytobench-gene.{scope}.score.bar.pdf"
        for scope in STRATIFIED_SCOPES
    )
    assert set(supplementary["expected_artifacts"]) == (
        expected_supplementary_artifacts
    )

    fig2h = targets["fig-2h"]
    assert fig2h["status"] == "run"
    assert fig2h["requires_data"] == [
        "Fig. 2/PhytoBench-Gene-BERTScore-for_plot.tsv",
        "Fig. 2/PhytoBench-Gene-hallucination-for_plot.tsv"
    ]
    assert fig2h["expected_artifacts"] == [
        "Fig. 2/output/fig.2h.phytobench-gene.uncharacterized.bar.pdf"
    ]
    fig2 = targets["fig-2"]
    assert all("fig.2h" not in path for path in fig2["expected_artifacts"])
