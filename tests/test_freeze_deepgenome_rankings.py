import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.freeze_deepgenome_rankings as freeze_module
from scripts.release_deepgenome_rankings import PUBLIC_COLUMNS
from scripts.deepgenome_ranking_statistics import (
    ASSIGNMENT_SUMMARY_COLUMNS,
    FLEISS_BOOTSTRAP_COLUMNS,
    GENE_ORDINAL_COLUMNS,
    ORDINAL_BOOTSTRAP_COLUMNS,
    PANEL_SUMMARY_COLUMNS,
    PL_INTERVAL_ANALYSES,
    PL_PAIRWISE_COLUMNS,
    PL_RANK_COLUMNS,
    PL_SCORE_COLUMNS,
    TOP1_BOOTSTRAP_COLUMNS,
    TOP1_PATTERNS,
    BootstrapConfig,
)
from scripts.freeze_deepgenome_rankings import (
    FREEZER_MODULE,
    PANEL_AUDIT_MODULE,
    PL_NOTEBOOK,
    STATISTICS_MODULE,
    freeze_rankings,
    main,
)


MODELS = ("Gemini", "Grok", "OpenAI", "Phytomni", "Claude")
SPECIES = ("Rice", "Maize", "Wheat", "Soybean", "Arabidopsis")
STATUSES = ("well_studied", "uncharacterized")
EXPECTED_OUTPUTS = {
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
BOOTSTRAP_CONFIG = BootstrapConfig(
    successful_replicates=20,
    seed=20260714,
    max_failed_fits=2,
)
RANKING_SCOPE_ORDER = (
    "overall",
    *(
        scope
        for status in STATUSES
        for scope in (
            status,
            *(f"{status}.{species.casefold()}" for species in SPECIES),
        )
    ),
    *(species.casefold() for species in SPECIES),
)
FLEISS_SCOPE_ORDER = (
    "overall",
    *(f"species.{species.casefold()}" for species in SPECIES),
    *(f"study_status.{status}" for status in STATUSES),
    *(f"model.{model.casefold()}" for model in MODELS),
    *(
        f"species_study_status.{species.casefold()}.{status}"
        for species in SPECIES
        for status in STATUSES
    ),
    *(
        f"model_study_status.{model.casefold()}.{status}"
        for model in MODELS
        for status in STATUSES
    ),
    *(
        f"model_species.{model.casefold()}.{species.casefold()}"
        for model in MODELS
        for species in SPECIES
    ),
)
ORDINAL_SCOPE_ORDER = (
    "overall",
    *(f"species.{species.casefold()}" for species in SPECIES),
    *(f"study_status.{status}" for status in STATUSES),
    *(
        f"species_study_status.{species.casefold()}.{status}"
        for species in SPECIES
        for status in STATUSES
    ),
)


def write_crossed_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    assignments = (
        (0, 1, 2),
        (3, 4, 5),
        (0, 1, 3),
        (2, 4, 5),
        (0, 1, 4),
        (2, 3, 5),
        (0, 1, 5),
        (2, 3, 4),
        (0, 2, 4),
        (1, 3, 5),
    )
    score_rows: list[dict[str, str]] = []
    category_rows: list[dict[str, str]] = []
    for species_index, species in enumerate(SPECIES):
        for gene_index, expert_indices in enumerate(assignments):
            status = STATUSES[gene_index // 5]
            gene = f"{species[:2]}-{gene_index + 1:02d}"
            category_rows.append(
                {
                    "Species": species,
                    "Gene": gene,
                    "StudyStatus": status,
                }
            )
            shift_offsets = (
                (0, 0, 1) if gene_index % 5 < 3 else (0, 1, 2)
            )
            for expert_slot, expert_index in enumerate(expert_indices):
                shift = (
                    species_index
                    + gene_index
                    + shift_offsets[expert_slot]
                ) % len(MODELS)
                order = MODELS[shift:] + MODELS[:shift]
                row = {
                    "AnonymousExpertID": (
                        f"E{species_index * 6 + expert_index + 1:03d}"
                    ),
                    "Species": species,
                    "Gene": gene,
                    "StudyStatus": status,
                }
                row.update(
                    {
                        model: f"R{rank}"
                        for rank, model in enumerate(order, start=1)
                    }
                )
                score_rows.append(row)

    score = pd.DataFrame(score_rows)
    assert tuple(score.columns) == PUBLIC_COLUMNS
    assert (
        score.groupby(["Species", "AnonymousExpertID"]).size() == 5
    ).all()
    assert (
        score.groupby(["Species", "Gene"])["AnonymousExpertID"].nunique()
        == 3
    ).all()
    score_path = tmp_path / "public_rankings.tsv"
    score.to_csv(score_path, sep="\t", index=False)

    categories_path = tmp_path / "gene_categories.tsv"
    pd.DataFrame(category_rows).to_csv(
        categories_path,
        sep="\t",
        index=False,
    )

    countries = ("China", "Germany", "USA", "Australia")
    metadata_rows: list[dict[str, object]] = []
    for index in range(30):
        metadata_rows.append(
            {
                "Expert_ID": f"private-panel-{index + 1:03d}",
                "Species": SPECIES[index // 6],
                "Country/Region": countries[index % len(countries)],
                "Institution_type": (
                    "University" if index % 2 == 0 else "Research institute"
                ),
                "Current_position": (
                    "Full Professor / Full researcher"
                    if index % 2 == 0
                    else "Postdoc / Assistant researcher"
                ),
                "Years_experience": "3-5" if index % 2 == 0 else "6–10",
                "Gender": (
                    None
                    if index == 29
                    else ("Female" if index % 2 == 0 else "Male")
                ),
                "Research_domains": (
                    "['Functional genomics', 'Molecular biology']"
                    if index % 2 == 0
                    else "['Crop genetics & breeding']"
                ),
                "Study_species": (
                    "['Rice', 'Maize']"
                    if index % 2 == 0
                    else "['Wheat']"
                ),
                "Annotation_experience": (
                    "Frequent" if index % 2 == 0 else "Occasional"
                ),
                "Ai_experience": (
                    "Frequent" if index % 2 == 0 else "Occasional"
                ),
                "Conflict_interest": "No",
            }
        )
    metadata_path = tmp_path / "expert_metadata.xlsx"
    pd.DataFrame(metadata_rows).to_excel(metadata_path, index=False)

    map_rows = [
        ("Species", species, species, order)
        for order, species in enumerate(SPECIES, start=1)
    ]
    map_rows.extend(
        [
            ("Country/Region", "China", "Asia", 1),
            ("Country/Region", "Germany", "Europe", 2),
            ("Country/Region", "USA", "North America", 3),
            ("Country/Region", "Australia", "Other regions", 4),
            ("Institution_type", "University", "University", 1),
            (
                "Institution_type",
                "Research institute",
                "Research institute",
                2,
            ),
            (
                "Current_position",
                "Full Professor / Full researcher",
                "Senior faculty or researcher",
                1,
            ),
            (
                "Current_position",
                "Postdoc / Assistant researcher",
                "Early-career researcher",
                2,
            ),
            ("Years_experience", "3-5", "<= 5 years", 1),
            ("Years_experience", "6–10", "6-10 years", 2),
            ("Gender", "Female", "Female", 1),
            ("Gender", "Male", "Male", 2),
            (
                "Research_domains",
                "Functional genomics",
                "Genomics and bioinformatics",
                1,
            ),
            (
                "Research_domains",
                "Molecular biology",
                "Molecular and developmental biology",
                2,
            ),
            (
                "Research_domains",
                "Crop genetics & breeding",
                "Crop genetics and breeding",
                3,
            ),
            ("Study_species", "Rice", "Rice", 1),
            ("Study_species", "Maize", "Maize", 2),
            ("Study_species", "Wheat", "Wheat", 3),
            ("Annotation_experience", "Frequent", "Frequent", 1),
            ("Annotation_experience", "Occasional", "Occasional", 2),
            ("Ai_experience", "Frequent", "Frequent", 1),
            ("Ai_experience", "Occasional", "Occasional", 2),
            ("Conflict_interest", "No", "No declared conflict", 1),
        ]
    )
    category_map_path = tmp_path / "panel_category_map.tsv"
    pd.DataFrame(
        map_rows,
        columns=[
            "Dimension",
            "SourceValue",
            "PublicCategory",
            "DisplayOrder",
        ],
    ).to_csv(category_map_path, sep="\t", index=False)
    return score_path, categories_path, metadata_path, category_map_path


def freeze_fixture(tmp_path: Path, output_dir: Path) -> tuple[Path, ...]:
    paths = write_crossed_fixture(tmp_path)
    score_path, categories_path, metadata_path, category_map_path = paths
    freeze_rankings(
        score_path=score_path,
        gene_categories_path=categories_path,
        expert_metadata_path=metadata_path,
        private_lineage_score_path=None,
        panel_category_map_path=category_map_path,
        output_dir=output_dir,
        model_columns=MODELS,
        bootstrap_config=BOOTSTRAP_CONFIG,
    )
    return paths


def directory_bytes(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_freezer_writes_deterministic_reviewer_tables(tmp_path: Path) -> None:
    output_dir = tmp_path / "frozen"
    (
        score_path,
        categories_path,
        metadata_path,
        category_map_path,
    ) = freeze_fixture(tmp_path, output_dir)
    first_bytes = {
        path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
    }
    (output_dir / "stale-unexpected-file.txt").write_text(
        "must be removed by transactional publication\n",
        encoding="utf-8",
    )
    freeze_rankings(
        score_path=score_path,
        gene_categories_path=categories_path,
        expert_metadata_path=metadata_path,
        private_lineage_score_path=None,
        panel_category_map_path=category_map_path,
        output_dir=output_dir,
        model_columns=MODELS,
        bootstrap_config=BOOTSTRAP_CONFIG,
    )
    second_bytes = {
        path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
    }

    assert first_bytes == second_bytes
    assert set(first_bytes) == EXPECTED_OUTPUTS

    ranks = pd.read_csv(output_dir / "rank_distribution.tsv", sep="\t")
    scores = pd.read_csv(output_dir / "pl_scores.tsv", sep="\t")
    pairwise = pd.read_csv(output_dir / "pl_pairwise.tsv", sep="\t")
    pl_scores_ci = pd.read_csv(output_dir / "pl_scores_ci.tsv", sep="\t")
    rank_ci = pd.read_csv(output_dir / "rank_distribution_ci.tsv", sep="\t")
    pairwise_ci = pd.read_csv(output_dir / "pl_pairwise_ci.tsv", sep="\t")
    fleiss = pd.read_csv(output_dir / "fleiss_kappa.tsv", sep="\t")
    kendall = pd.read_csv(output_dir / "kendall_by_gene.tsv", sep="\t")
    ordinal = pd.read_csv(
        output_dir / "ordinal_agreement_summary.tsv",
        sep="\t",
    )
    top1 = pd.read_csv(output_dir / "top1_consensus.tsv", sep="\t")
    panel = pd.read_csv(output_dir / "expert_panel_summary.tsv", sep="\t")
    assignment = pd.read_csv(output_dir / "assignment_summary.tsv", sep="\t")

    expected_schemas = {
        "rank_distribution": [
            "Scope",
            "StudyStatus",
            "Species",
            "Model",
            "Rank",
            "Count",
            "Fraction",
            "N",
        ],
        "pl_scores": [
            "Scope",
            "StudyStatus",
            "Species",
            "Model",
            "Elo",
            "Elo_L",
            "Elo_U",
            "N",
        ],
        "pl_pairwise": [
            "Scope",
            "StudyStatus",
            "Species",
            "RowModel",
            "ColumnModel",
            "Probability",
            "N",
        ],
        "pl_scores_ci": list(PL_SCORE_COLUMNS),
        "rank_distribution_ci": list(PL_RANK_COLUMNS),
        "pl_pairwise_ci": list(PL_PAIRWISE_COLUMNS),
        "fleiss_kappa": list(FLEISS_BOOTSTRAP_COLUMNS),
        "kendall_by_gene": list(GENE_ORDINAL_COLUMNS),
        "ordinal_agreement_summary": list(ORDINAL_BOOTSTRAP_COLUMNS),
        "top1_consensus": list(TOP1_BOOTSTRAP_COLUMNS),
        "expert_panel_summary": list(PANEL_SUMMARY_COLUMNS),
        "assignment_summary": list(ASSIGNMENT_SUMMARY_COLUMNS),
    }
    tables = {
        "rank_distribution": ranks,
        "pl_scores": scores,
        "pl_pairwise": pairwise,
        "pl_scores_ci": pl_scores_ci,
        "rank_distribution_ci": rank_ci,
        "pl_pairwise_ci": pairwise_ci,
        "fleiss_kappa": fleiss,
        "kendall_by_gene": kendall,
        "ordinal_agreement_summary": ordinal,
        "top1_consensus": top1,
        "expert_panel_summary": panel,
        "assignment_summary": assignment,
    }
    for name, table in tables.items():
        assert list(table.columns) == expected_schemas[name]

    assert len(ranks) == 18 * len(MODELS) ** 2
    assert len(scores) == 18 * len(MODELS)
    assert len(pairwise) == 18 * len(MODELS) ** 2
    assert len(pl_scores_ci) == 18 * len(MODELS) * 3
    assert len(rank_ci) == 18 * len(MODELS) ** 2
    assert len(pairwise_ci) == 18 * len(MODELS) * (len(MODELS) - 1)
    assert len(fleiss) == 58
    assert fleiss["ScopeID"].is_unique
    assert len(kendall) == 50
    assert ordinal["ScopeID"].nunique() == 18
    assert len(top1) == 18 * 3
    assert len(assignment) == 18

    assert list(
        ranks[["Scope", "Model", "Rank"]].itertuples(index=False, name=None)
    ) == [
        (scope, model, f"R{rank}")
        for scope in RANKING_SCOPE_ORDER
        for model in MODELS
        for rank in range(1, 6)
    ]
    assert list(
        scores[["Scope", "Model"]].itertuples(index=False, name=None)
    ) == [
        (scope, model)
        for scope in RANKING_SCOPE_ORDER
        for model in MODELS
    ]
    assert list(
        pairwise[
            ["Scope", "RowModel", "ColumnModel"]
        ].itertuples(index=False, name=None)
    ) == [
        (scope, row_model, column_model)
        for scope in RANKING_SCOPE_ORDER
        for row_model in MODELS
        for column_model in MODELS
    ]
    assert list(
        pl_scores_ci[
            ["Scope", "Model", "IntervalAnalysis"]
        ].itertuples(index=False, name=None)
    ) == [
        (scope, model, analysis)
        for scope in RANKING_SCOPE_ORDER
        for model in MODELS
        for analysis in PL_INTERVAL_ANALYSES
    ]
    assert list(fleiss["ScopeID"]) == list(FLEISS_SCOPE_ORDER)
    assert list(ordinal["ScopeID"]) == list(ORDINAL_SCOPE_ORDER)
    assert list(
        top1[["ScopeID", "Top1AgreementPattern"]].itertuples(
            index=False,
            name=None,
        )
    ) == [
        (scope, pattern)
        for scope in ORDINAL_SCOPE_ORDER
        for pattern in TOP1_PATTERNS
    ]
    assert list(assignment["Scope"]) == list(RANKING_SCOPE_ORDER)
    assert not kendall.duplicated(["Species", "Gene"]).any()
    assert list(
        kendall[["Species", "Gene"]].itertuples(index=False, name=None)
    ) == sorted(
        kendall[["Species", "Gene"]].itertuples(index=False, name=None),
        key=lambda row: (SPECIES.index(row[0]), row[1]),
    )

    np.testing.assert_allclose(
        ranks.groupby(["Scope", "Model"])["Fraction"].sum().to_numpy(),
        1.0,
    )
    primary_scores = pl_scores_ci[
        pl_scores_ci["IntervalAnalysis"] == "crossed_expert_gene"
    ]
    pd.testing.assert_series_equal(
        scores.set_index(["Scope", "Model"])["Elo"].sort_index(),
        primary_scores.set_index(["Scope", "Model"])["Estimate"].sort_index(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        ranks.set_index(["Scope", "Model", "Rank"])["Fraction"].sort_index(),
        rank_ci.set_index(["Scope", "Model", "Rank"])["Fraction"].sort_index(),
        check_names=False,
    )
    point_off_diagonal = pairwise.dropna(subset=["Probability"])
    pd.testing.assert_series_equal(
        point_off_diagonal.set_index(
            ["Scope", "RowModel", "ColumnModel"]
        )["Probability"].sort_index(),
        pairwise_ci.set_index(
            ["Scope", "RowModel", "ColumnModel"]
        )["Probability"].sort_index(),
        check_names=False,
    )
    assert pairwise["Probability"].isna().sum() == 18 * len(MODELS)
    assert np.isfinite(point_off_diagonal["Probability"]).all()
    for frame, estimate, lower, upper in (
        (pl_scores_ci, "Estimate", "CI95Lower", "CI95Upper"),
        (rank_ci, "Fraction", "CI95Lower", "CI95Upper"),
        (pairwise_ci, "Probability", "CI95Lower", "CI95Upper"),
    ):
        assert np.isfinite(frame[[estimate, lower, upper]].to_numpy()).all()
        assert (frame[lower] <= frame[upper]).all()
        assert (frame[lower] <= frame[estimate]).all()
        assert (frame[estimate] <= frame[upper]).all()
    assert (fleiss["NRatings"] == 3 * fleiss["NItems"]).all()
    assert fleiss["ScopeFamily"].value_counts().to_dict() == {
        "model_species": 25,
        "species_study_status": 10,
        "model_study_status": 10,
        "species": 5,
        "model": 5,
        "study_status": 2,
        "overall": 1,
    }
    assert fleiss["AnalysisTier"].value_counts().to_dict() == {
        "locked_exploratory": 45,
        "locked_secondary": 12,
        "primary": 1,
    }
    assert not {
        "model_species_study_status",
        "species_model_study_status",
    } & set(fleiss["ScopeFamily"])
    np.testing.assert_allclose(
        fleiss["FleissKappa"],
        (fleiss["ObservedAgreement"] - fleiss["ExpectedAgreement"])
        / (1.0 - fleiss["ExpectedAgreement"]),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        fleiss[[f"RankR{rank}Share" for rank in range(1, 6)]].sum(axis=1),
        1.0,
    )
    assert np.isfinite(
        fleiss[["FleissKappa", "CILower", "CIUpper"]].to_numpy()
    ).all()
    assert (fleiss["CILower"] <= fleiss["CIUpper"]).all()
    assert (fleiss["BootstrapReplicates"] == 20).all()
    assert (ordinal["BootstrapReplicates"] == 20).all()
    assert (top1["BootstrapReplicates"] == 20).all()
    assert kendall["KendallW"].between(0.0, 1.0).all()
    assert kendall["MeanPairwiseKendallTau"].between(-1.0, 1.0).all()
    assert set(kendall["Top1AgreementPattern"]) <= set(TOP1_PATTERNS)
    for prefix in ("KendallW", "MeanPairwiseKendallTau"):
        for statistic in ("Mean", "Median"):
            lower = f"{prefix}{statistic}CILower"
            upper = f"{prefix}{statistic}CIUpper"
            assert np.isfinite(ordinal[[lower, upper]].to_numpy()).all()
            assert (ordinal[lower] <= ordinal[upper]).all()
    top1_grouped = top1.groupby("ScopeID", sort=False)
    np.testing.assert_allclose(top1_grouped["Fraction"].sum(), 1.0)
    assert (
        top1_grouped["Count"].sum().to_numpy()
        == top1_grouped["NGenes"].first().to_numpy()
    ).all()
    assert (assignment["MinExpertsPerGene"] == 3).all()
    assert (assignment["MaxExpertsPerGene"] == 3).all()
    assert (assignment["NJudgments"] == 3 * assignment["NGenes"]).all()
    assert (assignment["MinGenesPerExpert"] > 0).all()
    assert (
        assignment["MinGenesPerExpert"]
        <= assignment["MaxGenesPerExpert"]
    ).all()
    assert assignment.iloc[0].to_dict() == {
        "Scope": "overall",
        "StudyStatus": "all",
        "Species": "all",
        "NExperts": 30,
        "NGenes": 50,
        "NJudgments": 150,
        "MinGenesPerExpert": 5,
        "MaxGenesPerExpert": 5,
        "MinExpertsPerGene": 3,
        "MaxExpertsPerGene": 3,
    }

    assert (panel["N"] >= 5).all()
    assert (panel["MissingN"] >= 0).all()
    np.testing.assert_allclose(
        panel["Percent"],
        100.0 * panel["N"] / panel["DenominatorN"],
    )
    assert set(
        panel.loc[panel["Dimension"] == "Gender", "MissingN"]
    ) == {1}
    country_sources = {"China", "Germany", "USA", "Australia"}
    country_categories = set(
        panel.loc[
            panel["Dimension"] == "Country/Region",
            "PublicCategory",
        ]
    )
    assert country_categories.isdisjoint(country_sources)
    category_map = pd.read_csv(category_map_path, sep="\t")
    dimension_order = {
        dimension: index
        for index, dimension in enumerate(
            category_map["Dimension"].drop_duplicates()
        )
    }
    expected_panel_order = (
        category_map[
            ["Dimension", "PublicCategory", "DisplayOrder"]
        ]
        .drop_duplicates(["Dimension", "PublicCategory"])
        .assign(
            _DimensionOrder=lambda frame: frame["Dimension"].map(
                dimension_order
            )
        )
        .sort_values(
            ["_DimensionOrder", "DisplayOrder", "PublicCategory"],
            kind="stable",
        )
    )
    assert list(
        panel[
            ["Dimension", "PublicCategory", "DisplayOrder"]
        ].itertuples(index=False, name=None)
    ) == list(
        expected_panel_order[
            ["Dimension", "PublicCategory", "DisplayOrder"]
        ].itertuples(index=False, name=None)
    )
    multiselect = panel["Dimension"].isin(
        ["Research_domains", "Study_species"]
    )
    assert set(panel.loc[multiselect, "PercentageBasis"]) == {"all_experts"}
    assert set(panel.loc[~multiselect, "PercentageBasis"]) == {
        "nonmissing_experts"
    }
    for _, selected in panel.loc[~multiselect].groupby(
        "Dimension",
        sort=False,
    ):
        assert selected["N"].sum() == selected["DenominatorN"].iloc[0]
        assert selected["MissingN"].nunique() == 1
    assert panel.loc[
        panel["Dimension"] == "Conflict_interest"
    ].to_dict("records") == [
        {
            "Dimension": "Conflict_interest",
            "PublicCategory": "No declared conflict",
            "DisplayOrder": 1,
            "N": 30,
            "DenominatorN": 30,
            "Percent": 100.0,
            "MissingN": 0,
            "PercentageBasis": "nonmissing_experts",
        }
    ]
    assert "SourceValue" not in panel.columns
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

    forbidden_identifier_columns = {
        "Expert",
        "Expert_ID",
        "AnonymousExpertID",
    }
    for table in (
        ranks,
        scores,
        pairwise,
        pl_scores_ci,
        rank_ci,
        pairwise_ci,
        fleiss,
        kendall,
        ordinal,
        top1,
        panel,
        assignment,
    ):
        assert forbidden_identifier_columns.isdisjoint(table.columns)
    for name, payload in first_bytes.items():
        if name.endswith(".tsv"):
            assert b"private-panel-" not in payload

    provenance = json.loads((output_dir / "provenance.json").read_text())
    assert provenance["schema_version"] == 2
    assert (
        provenance["reporting_matrix_status"]
        == "locked_before_final_bootstrap_reanalysis"
    )
    assert provenance["pilot_kappa_values_viewed"] is True
    assert provenance["reporting_matrix_statement"] == (
        "The reporting matrix was locked before the final bootstrap "
        "reanalysis and before manuscript interpretation."
    )
    assert provenance["agreement_item_definition"] == "Species + Gene + Model"
    assert provenance["agreement_categories"] == [
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
    ]
    assert provenance["fleiss_scope_count"] == 58
    assert provenance["ordinal_scope_count"] == 18
    assert provenance["bootstrap"] == {
        "master_seed": 20260714,
        "successful_replicates": 20,
        "maximum_failed_fits": 2,
        "primary_pl_interval": "crossed_expert_gene_percentile",
        "agreement_interval": "stratified_gene_block_percentile",
    }
    assert provenance["inputs"]["public_ranking_release"]["sha256"] == (
        hashlib.sha256(score_path.read_bytes()).hexdigest()
    )
    assert provenance["inputs"]["gene_categories"]["sha256"] == (
        hashlib.sha256(categories_path.read_bytes()).hexdigest()
    )
    assert provenance["inputs"]["expert_metadata"]["sha256"] == (
        hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    )
    assert provenance["inputs"]["panel_category_map"]["sha256"] == (
        hashlib.sha256(category_map_path.read_bytes()).hexdigest()
    )
    assert provenance["inputs"]["private_lineage_score"] is None
    assert provenance["inputs"]["statistical_module"]["sha256"] == (
        hashlib.sha256(STATISTICS_MODULE.read_bytes()).hexdigest()
    )
    assert provenance["inputs"]["panel_category_audit_module"]["sha256"] == (
        hashlib.sha256(PANEL_AUDIT_MODULE.read_bytes()).hexdigest()
    )
    assert provenance["inputs"]["freezer_module"]["sha256"] == (
        hashlib.sha256(FREEZER_MODULE.read_bytes()).hexdigest()
    )
    assert provenance["inputs"]["narrative_notebook"]["sha256"] == (
        hashlib.sha256(PL_NOTEBOOK.read_bytes()).hexdigest()
    )
    for record in provenance["inputs"].values():
        if record is not None:
            assert set(record) == {"sha256"}
    pl_diagnostics = provenance["bootstrap_diagnostics"]["plackett_luce"]
    assert pl_diagnostics["AttemptedReplicates"] >= 20
    assert pl_diagnostics["SuccessfulReplicates"] == 20
    assert pl_diagnostics["FailedFits"] <= 2
    assert "FailureReasons" not in pl_diagnostics
    assert isinstance(pl_diagnostics["FailureReasonCodes"], list)
    assert set(pl_diagnostics["FailureReasonCodes"]) <= {
        "zero_effective_weight",
        "optimizer_failure",
        "nonfinite_output",
        "numerical_failure",
    }
    assert pl_diagnostics["SeedStream"] == "pl_expert_gene_components"
    assert isinstance(pl_diagnostics["ExpertSeedSpawnKey"], list)
    assert isinstance(pl_diagnostics["GeneSeedSpawnKey"], list)
    assert pl_diagnostics["HalfRunStability"] == {"Applied": False}
    agreement_diagnostics = provenance["bootstrap_diagnostics"]["agreement"]
    assert agreement_diagnostics == {
        "attempted_replicates": int(fleiss["BootstrapAttempted"].iloc[0]),
        "successful_replicates": 20,
        "invalid_replicates": int(fleiss["BootstrapInvalid"].iloc[0]),
        "seed_stream": "agreement_gene_blocks",
    }
    assert set(pl_scores_ci["FailedFits"]) == {pl_diagnostics["FailedFits"]}
    assert set(provenance["outputs"]) == EXPECTED_OUTPUTS - {"provenance.json"}
    for name, digest in provenance["outputs"].items():
        assert digest == hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
    serialized_provenance = json.dumps(provenance)
    assert "private-panel-" not in serialized_provenance
    assert str(tmp_path) not in serialized_provenance
    assert "crosswalk" not in serialized_provenance.casefold()
    assert "preregister" not in serialized_provenance.casefold()
    assert not {"rows", "used_rows", "skipped_rows", "scopes"} & set(
        provenance
    )


def test_freezer_rejects_incomplete_rankings(tmp_path: Path) -> None:
    score_path, categories_path, metadata_path, category_map_path = (
        write_crossed_fixture(tmp_path)
    )
    score = pd.read_csv(score_path, sep="\t")
    score.loc[0, "Claude"] = "R4"
    score.to_csv(score_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="complete permutation"):
        freeze_rankings(
            score_path=score_path,
            gene_categories_path=categories_path,
            expert_metadata_path=metadata_path,
            private_lineage_score_path=None,
            panel_category_map_path=category_map_path,
            output_dir=tmp_path / "frozen",
            model_columns=MODELS,
            bootstrap_config=BOOTSTRAP_CONFIG,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_column",
        "reordered_columns",
        "malformed_id",
        "noncontiguous_id",
        "whitespace_id",
        "invalid_status",
    ],
)
def test_freezer_requires_exact_canonical_public_release_before_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    score_path, categories_path, metadata_path, category_map_path = (
        write_crossed_fixture(tmp_path)
    )
    score = pd.read_csv(score_path, sep="\t", dtype=str)
    private_sentinel = "private-invalid-sentinel"
    bad_value: str | None = None
    if mutation == "extra_column":
        score["Unexpected"] = private_sentinel
        bad_value = private_sentinel
    elif mutation == "reordered_columns":
        score = score.loc[:, [*score.columns[1:], score.columns[0]]]
    elif mutation == "malformed_id":
        score.loc[score["AnonymousExpertID"] == "E001", "AnonymousExpertID"] = (
            private_sentinel
        )
        bad_value = private_sentinel
    elif mutation == "noncontiguous_id":
        score.loc[score["AnonymousExpertID"] == "E001", "AnonymousExpertID"] = (
            "E999"
        )
        bad_value = "E999"
    elif mutation == "whitespace_id":
        score.loc[score["AnonymousExpertID"] == "E001", "AnonymousExpertID"] = (
            " E001"
        )
        bad_value = " E001"
    else:
        score.loc[0, "StudyStatus"] = private_sentinel
        bad_value = private_sentinel
    score.to_csv(score_path, sep="\t", index=False)

    def unexpected_bootstrap(*args: object, **kwargs: object) -> object:
        raise AssertionError("Invalid public input reached the bootstrap")

    monkeypatch.setattr(
        freeze_module,
        "bootstrap_plackett_luce_statistics",
        unexpected_bootstrap,
    )
    with pytest.raises(ValueError) as error:
        freeze_rankings(
            score_path=score_path,
            gene_categories_path=categories_path,
            expert_metadata_path=metadata_path,
            private_lineage_score_path=None,
            panel_category_map_path=category_map_path,
            output_dir=tmp_path / "frozen",
            model_columns=MODELS,
            bootstrap_config=BOOTSTRAP_CONFIG,
        )
    assert private_sentinel not in str(error.value)
    if bad_value is not None:
        assert bad_value not in str(error.value)


@pytest.mark.parametrize(
    ("expert_column", "model_columns"),
    [
        ("Expert", MODELS),
        ("AnonymousExpertID", tuple(reversed(MODELS))),
    ],
)
def test_freezer_rejects_noncanonical_analysis_configuration_before_io(
    tmp_path: Path,
    expert_column: str,
    model_columns: tuple[str, ...],
) -> None:
    missing = tmp_path / "must-not-be-read"
    with pytest.raises(ValueError, match="canonical"):
        freeze_rankings(
            score_path=missing,
            gene_categories_path=missing,
            expert_metadata_path=missing,
            private_lineage_score_path=None,
            panel_category_map_path=missing,
            output_dir=tmp_path / "frozen",
            model_columns=model_columns,
            expert_column=expert_column,
            bootstrap_config=BOOTSTRAP_CONFIG,
        )


@pytest.mark.parametrize("failure_point", ["write", "validation", "swap"])
def test_freezer_transaction_rolls_back_existing_output_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    score_path, categories_path, metadata_path, category_map_path = (
        write_crossed_fixture(tmp_path)
    )
    output_dir = tmp_path / "frozen"
    output_dir.mkdir()
    (output_dir / "rank_distribution.tsv").write_bytes(b"previous ranks\n")
    (output_dir / "provenance.json").write_bytes(b"previous provenance\n")
    (output_dir / "keep.bin").write_bytes(b"previous extra bytes\x00\x01")
    previous = directory_bytes(output_dir)

    def injected_failure(*args: object, **kwargs: object) -> object:
        raise RuntimeError(f"injected {failure_point} failure")

    if failure_point == "write":
        original_write = freeze_module.write_table

        def fail_after_write(frame: pd.DataFrame, path: Path) -> None:
            original_write(frame, path)
            injected_failure()

        monkeypatch.setattr(freeze_module, "write_table", fail_after_write)
    elif failure_point == "validation":
        monkeypatch.setattr(
            freeze_module,
            "validate_staged_publication",
            injected_failure,
            raising=False,
        )
    else:
        original_replace = freeze_module.os.replace
        failed = False

        def fail_new_directory_swap(source: Path, destination: Path) -> None:
            nonlocal failed
            source_path = Path(source)
            destination_path = Path(destination)
            if (
                not failed
                and destination_path == output_dir
                and ".staging-" in source_path.name
            ):
                failed = True
                injected_failure()
            original_replace(source, destination)

        monkeypatch.setattr(
            freeze_module.os,
            "replace",
            fail_new_directory_swap,
        )

    with pytest.raises(RuntimeError, match=f"injected {failure_point} failure"):
        freeze_rankings(
            score_path=score_path,
            gene_categories_path=categories_path,
            expert_metadata_path=metadata_path,
            private_lineage_score_path=None,
            panel_category_map_path=category_map_path,
            output_dir=output_dir,
            model_columns=MODELS,
            bootstrap_config=BootstrapConfig(
                successful_replicates=1,
                seed=20260714,
                max_failed_fits=2,
            ),
        )

    assert directory_bytes(output_dir) == previous
    assert not list(tmp_path.glob(".frozen.staging-*"))
    assert not list(tmp_path.glob(".frozen.backup-*"))


def test_freezer_hashes_and_parses_the_same_single_read_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score_path, categories_path, metadata_path, category_map_path = (
        write_crossed_fixture(tmp_path)
    )
    original_score = score_path.read_bytes()
    original_read_bytes = Path.read_bytes
    score_reads = 0

    def replace_after_read(path: Path) -> bytes:
        nonlocal score_reads
        payload = original_read_bytes(path)
        if path.resolve() == score_path.resolve():
            score_reads += 1
            if score_reads == 1:
                score_path.write_text(
                    "corrupted after immutable snapshot\n",
                    encoding="utf-8",
                )
        return payload

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    output_dir = tmp_path / "frozen"
    freeze_rankings(
        score_path=score_path,
        gene_categories_path=categories_path,
        expert_metadata_path=metadata_path,
        private_lineage_score_path=None,
        panel_category_map_path=category_map_path,
        output_dir=output_dir,
        model_columns=MODELS,
        bootstrap_config=BootstrapConfig(
            successful_replicates=1,
            seed=20260714,
            max_failed_fits=2,
        ),
    )

    provenance = json.loads((output_dir / "provenance.json").read_text())
    assert score_reads == 1
    assert provenance["inputs"]["public_ranking_release"]["sha256"] == (
        hashlib.sha256(original_score).hexdigest()
    )
    assert len(pd.read_csv(output_dir / "rank_distribution.tsv", sep="\t")) == (
        18 * len(MODELS) ** 2
    )


def test_freezer_validates_lineage_before_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score_path, categories_path, metadata_path, category_map_path = (
        write_crossed_fixture(tmp_path)
    )

    def unexpected_bootstrap(*args: object, **kwargs: object) -> object:
        raise AssertionError("Bootstrap must not start before input hashing")

    monkeypatch.setattr(
        freeze_module,
        "bootstrap_plackett_luce_statistics",
        unexpected_bootstrap,
    )
    with pytest.raises(FileNotFoundError):
        freeze_rankings(
            score_path=score_path,
            gene_categories_path=categories_path,
            expert_metadata_path=metadata_path,
            private_lineage_score_path=tmp_path / "missing-private-score.tsv",
            panel_category_map_path=category_map_path,
            output_dir=tmp_path / "frozen",
            model_columns=MODELS,
            bootstrap_config=BOOTSTRAP_CONFIG,
        )


def test_freezer_rejects_aggregate_species_panel_mismatch(
    tmp_path: Path,
) -> None:
    score_path, categories_path, metadata_path, category_map_path = (
        write_crossed_fixture(tmp_path)
    )
    metadata = pd.read_excel(metadata_path)
    metadata.loc[0, "Species"] = "Maize"
    metadata.to_excel(metadata_path, index=False)

    with pytest.raises(ValueError, match="aggregate panel sizes by species"):
        freeze_rankings(
            score_path=score_path,
            gene_categories_path=categories_path,
            expert_metadata_path=metadata_path,
            private_lineage_score_path=None,
            panel_category_map_path=category_map_path,
            output_dir=tmp_path / "frozen",
            model_columns=MODELS,
            bootstrap_config=BOOTSTRAP_CONFIG,
        )


def test_freezer_rejects_public_study_status_mismatch(tmp_path: Path) -> None:
    score_path, categories_path, metadata_path, category_map_path = (
        write_crossed_fixture(tmp_path)
    )
    score = pd.read_csv(score_path, sep="\t")
    score.loc[0, "StudyStatus"] = "uncharacterized"
    score.to_csv(score_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="study statuses must match"):
        freeze_rankings(
            score_path=score_path,
            gene_categories_path=categories_path,
            expert_metadata_path=metadata_path,
            private_lineage_score_path=None,
            panel_category_map_path=category_map_path,
            output_dir=tmp_path / "frozen",
            model_columns=MODELS,
            bootstrap_config=BOOTSTRAP_CONFIG,
        )


def test_cli_requires_private_lineage_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    score_path, categories_path, metadata_path, category_map_path = (
        write_crossed_fixture(tmp_path)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_deepgenome_rankings",
            "--score-tsv",
            str(score_path),
            "--gene-categories",
            str(categories_path),
            "--expert-metadata",
            str(metadata_path),
            "--panel-category-map",
            str(category_map_path),
            "--output-dir",
            str(tmp_path / "frozen"),
        ],
    )

    with pytest.raises(SystemExit):
        main()

    assert "--private-lineage-score" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--expert-column", "Expert"),
        ("--model-columns", ",".join(reversed(MODELS))),
    ],
)
def test_cli_rejects_noncanonical_analysis_configuration_without_echoing_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flag: str,
    value: str,
) -> None:
    score_path, categories_path, metadata_path, category_map_path = (
        write_crossed_fixture(tmp_path)
    )
    lineage_path = tmp_path / "private-lineage.tsv"
    lineage_path.write_text("private lineage fixture\n", encoding="utf-8")

    def unexpected_freeze(**kwargs: object) -> dict[str, Path]:
        raise AssertionError("Noncanonical CLI configuration reached the freezer")

    monkeypatch.setattr(freeze_module, "freeze_rankings", unexpected_freeze)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_deepgenome_rankings",
            "--score-tsv",
            str(score_path),
            "--gene-categories",
            str(categories_path),
            "--expert-metadata",
            str(metadata_path),
            "--private-lineage-score",
            str(lineage_path),
            "--panel-category-map",
            str(category_map_path),
            "--output-dir",
            str(tmp_path / "frozen"),
            flag,
            value,
        ],
    )

    with pytest.raises(SystemExit):
        main()

    error = capsys.readouterr().err
    assert "canonical" in error.casefold()
    if flag == "--model-columns":
        assert value not in error


def test_cli_propagates_production_inputs_and_bootstrap_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    score_path, categories_path, metadata_path, category_map_path = (
        write_crossed_fixture(tmp_path)
    )
    lineage_path = tmp_path / "private-lineage.tsv"
    lineage_path.write_text("private lineage fixture\n", encoding="utf-8")
    output_dir = tmp_path / "frozen"
    received: dict[str, object] = {}

    def fake_freeze_rankings(**kwargs: object) -> dict[str, Path]:
        received.update(kwargs)
        return {"provenance": output_dir / "provenance.json"}

    monkeypatch.setattr(freeze_module, "freeze_rankings", fake_freeze_rankings)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_deepgenome_rankings",
            "--score-tsv",
            str(score_path),
            "--expert-column",
            "AnonymousExpertID",
            "--gene-categories",
            str(categories_path),
            "--expert-metadata",
            str(metadata_path),
            "--private-lineage-score",
            str(lineage_path),
            "--panel-category-map",
            str(category_map_path),
            "--output-dir",
            str(output_dir),
            "--model-columns",
            ",".join(MODELS),
            "--bootstrap-replicates",
            "7",
            "--seed",
            "9",
            "--max-failed-fits",
            "1",
        ],
    )

    main()

    assert received == {
        "score_path": score_path,
        "gene_categories_path": categories_path,
        "expert_metadata_path": metadata_path,
        "private_lineage_score_path": lineage_path,
        "panel_category_map_path": category_map_path,
        "output_dir": output_dir,
        "model_columns": MODELS,
        "expert_column": "AnonymousExpertID",
        "bootstrap_config": BootstrapConfig(
            successful_replicates=7,
            seed=9,
            max_failed_fits=1,
        ),
    }
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "provenance.json" in captured.out
    assert str(lineage_path) not in captured.out
