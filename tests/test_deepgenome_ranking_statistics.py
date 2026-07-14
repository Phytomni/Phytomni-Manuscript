from collections import Counter
from dataclasses import FrozenInstanceError
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.deepgenome_ranking_statistics as ranking_statistics
from scripts.deepgenome_ranking_statistics import (
    FLEISS_COLUMNS,
    MODEL_COLUMNS,
    AgreementScope,
    BootstrapConfig,
    agreement_scope_registry,
    bootstrap_agreement,
    collapse_weighted_rankings,
    elo_outputs,
    fit_plackett_luce,
    fleiss_kappa_from_counts,
    fleiss_point_estimates,
    gene_bootstrap_multiplicities,
    parse_rankings,
    pl_loglik_and_grad,
)


SPECIES = ("Rice", "Maize", "Wheat", "Soybean", "Arabidopsis")
STATUSES = ("well_studied", "uncharacterized")
FLEISS_1971_COUNTS = np.array(
    [
        [0, 0, 0, 0, 14],
        [0, 2, 6, 4, 2],
        [0, 0, 3, 5, 6],
        [0, 3, 9, 2, 0],
        [2, 2, 8, 1, 1],
        [7, 7, 0, 0, 0],
        [3, 2, 6, 3, 0],
        [2, 5, 3, 2, 2],
        [6, 5, 2, 1, 0],
        [0, 2, 2, 3, 7],
    ]
)


def categorized_ranking_fixture() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for species_index, species in enumerate(SPECIES):
        for status_index, status in enumerate(STATUSES):
            for gene_index in range(2):
                gene = f"{species[:2]}-{status_index}-{gene_index}"
                for expert_index in range(3):
                    shift = (
                        species_index
                        + status_index
                        + 2 * gene_index
                        + expert_index
                    ) % len(MODEL_COLUMNS)
                    order = (
                        MODEL_COLUMNS[shift:] + MODEL_COLUMNS[:shift]
                    )
                    row = {
                        "Species": species,
                        "Gene": gene,
                        "Expert": f"expert-{expert_index + 1}",
                        "StudyStatus": status,
                    }
                    row.update(
                        {
                            model: f"R{rank}"
                            for rank, model in enumerate(order, start=1)
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def crossed_fixture() -> pd.DataFrame:
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
    rows: list[dict[str, str]] = []
    for species_index, species in enumerate(SPECIES):
        for gene_index, expert_indices in enumerate(assignments):
            status = STATUSES[gene_index // 5]
            gene = f"{species[:2]}-{gene_index + 1:02d}"
            shift_offsets = (
                (0, 0, 1) if gene_index % 5 < 3 else (0, 1, 2)
            )
            for expert_slot, expert_index in enumerate(expert_indices):
                shift = (
                    species_index
                    + gene_index
                    + shift_offsets[expert_slot]
                ) % 5
                order = MODEL_COLUMNS[shift:] + MODEL_COLUMNS[:shift]
                row = {
                    "Species": species,
                    "Gene": gene,
                    "Expert": f"{species}-expert-{expert_index + 1}",
                    "StudyStatus": status,
                }
                row.update(
                    {
                        model: f"R{rank}"
                        for rank, model in enumerate(order, start=1)
                    }
                )
                rows.append(row)

    frame = pd.DataFrame(rows)
    rating_cells = (
        frame.groupby(["Species", "Gene"], sort=False).size()
        * len(MODEL_COLUMNS)
    )
    assert (rating_cells == 15).all()
    assert (
        frame.groupby(["Species", "Expert"], sort=False).size() == 5
    ).all()
    assert (
        frame.groupby(["Species", "Gene"], sort=False)["Expert"].nunique()
        == 3
    ).all()
    assert (
        frame[["Species", "StudyStatus"]].drop_duplicates().shape[0] == 10
    )
    return frame


def test_bootstrap_config_has_locked_defaults_and_rejects_invalid_values(
) -> None:
    assert BootstrapConfig() == BootstrapConfig(
        successful_replicates=10_000,
        seed=20260714,
        max_failed_fits=10,
    )

    for field, value in (
        ("successful_replicates", 0),
        ("successful_replicates", -1),
        ("successful_replicates", 1.5),
        ("successful_replicates", True),
        ("seed", -1),
        ("seed", 1.5),
        ("seed", False),
        ("max_failed_fits", -1),
        ("max_failed_fits", 1.5),
        ("max_failed_fits", True),
    ):
        with pytest.raises(ValueError, match=field):
            BootstrapConfig(**{field: value})


def test_gene_bootstrap_multiplicities_are_stratified_and_deterministic(
    crossed_fixture: pd.DataFrame,
) -> None:
    genes = (
        crossed_fixture[["Species", "Gene", "StudyStatus"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    first_rng = np.random.default_rng(
        np.random.SeedSequence(20260714).spawn(1)[0]
    )
    second_rng = np.random.default_rng(
        np.random.SeedSequence(20260714).spawn(1)[0]
    )

    first = gene_bootstrap_multiplicities(genes, first_rng)
    second = gene_bootstrap_multiplicities(genes, second_rng)

    pd.testing.assert_series_equal(first, second)
    assert first.index.equals(genes.index)
    assert pd.api.types.is_integer_dtype(first.dtype)
    assert (first >= 0).all()
    original_sizes = genes.groupby(
        ["Species", "StudyStatus"], sort=False
    ).size()
    sampled_sizes = first.groupby(
        [genes["Species"], genes["StudyStatus"]], sort=False
    ).sum()
    pd.testing.assert_series_equal(sampled_sizes, original_sizes)
    assert (original_sizes == 5).all()


def test_agreement_bootstrap_is_shared_and_reproducible(
    crossed_fixture: pd.DataFrame,
) -> None:
    config = BootstrapConfig(
        successful_replicates=25,
        seed=20260714,
        max_failed_fits=0,
    )

    first = bootstrap_agreement(crossed_fixture, MODEL_COLUMNS, config)
    second = bootstrap_agreement(crossed_fixture, MODEL_COLUMNS, config)

    for name in ("fleiss_kappa", "ordinal_summary", "top1_consensus"):
        pd.testing.assert_frame_equal(first[name], second[name])
        assert first[name].to_csv(index=False) == second[name].to_csv(
            index=False
        )
    assert len(first["fleiss_kappa"]) == 58
    assert len(first["ordinal_summary"]) == 18
    assert first["ordinal_summary"]["ScopeID"].nunique() == 18
    assert len(first["top1_consensus"]) == 18 * 3
    assert (first["fleiss_kappa"]["BootstrapReplicates"] == 25).all()


def test_agreement_bootstrap_uses_one_gene_draw_per_replicate(
    crossed_fixture: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_draws: list[pd.Series] = []
    implementation = ranking_statistics.gene_bootstrap_multiplicities

    def record_draw(
        genes: pd.DataFrame,
        rng: np.random.Generator,
    ) -> pd.Series:
        multiplicities = implementation(genes, rng)
        observed_draws.append(multiplicities.copy())
        return multiplicities

    monkeypatch.setattr(
        ranking_statistics,
        "gene_bootstrap_multiplicities",
        record_draw,
    )
    result = bootstrap_agreement(
        crossed_fixture,
        MODEL_COLUMNS,
        BootstrapConfig(successful_replicates=7, max_failed_fits=0),
    )

    assert len(observed_draws) == 7
    for draw in observed_draws:
        genes = (
            crossed_fixture[["Species", "Gene", "StudyStatus"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        stratum_totals = draw.groupby(
            [genes["Species"], genes["StudyStatus"]], sort=False
        ).sum()
        assert (stratum_totals == 5).all()
    seed_streams = {
        tuple(output["SeedStream"].unique())
        for output in result.values()
    }
    assert seed_streams == {("agreement_gene_blocks",)}


def test_agreement_bootstrap_has_stable_schemas_and_unclipped_intervals(
    crossed_fixture: pd.DataFrame,
) -> None:
    config = BootstrapConfig(
        successful_replicates=25,
        seed=20260714,
        max_failed_fits=0,
    )
    result = bootstrap_agreement(crossed_fixture, MODEL_COLUMNS, config)
    fleiss = result["fleiss_kappa"]
    ordinal = result["ordinal_summary"]
    top1 = result["top1_consensus"]

    metadata = (
        "BootstrapAttempted",
        "BootstrapReplicates",
        "BootstrapInvalid",
        "BootstrapUnit",
        "BootstrapStrata",
        "SeedStream",
    )
    assert tuple(fleiss.columns) == (
        *FLEISS_COLUMNS,
        "CILower",
        "CIUpper",
        *metadata,
    )
    assert tuple(ordinal.columns) == (
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
        *metadata,
    )
    assert tuple(top1.columns) == (
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
        *metadata,
    )

    for output in result.values():
        assert (output["BootstrapAttempted"] == 25).all()
        assert (output["BootstrapReplicates"] == 25).all()
        assert (output["BootstrapInvalid"] == 0).all()
        assert (output["BootstrapUnit"] == "gene").all()
        assert (
            output["BootstrapStrata"] == "Species x StudyStatus"
        ).all()
        assert (output["SeedStream"] == "agreement_gene_blocks").all()

    interval_pairs = [
        (fleiss, "CILower", "CIUpper"),
        (ordinal, "KendallWMeanCILower", "KendallWMeanCIUpper"),
        (
            ordinal,
            "KendallWMedianCILower",
            "KendallWMedianCIUpper",
        ),
        (
            ordinal,
            "MeanPairwiseKendallTauMeanCILower",
            "MeanPairwiseKendallTauMeanCIUpper",
        ),
        (
            ordinal,
            "MeanPairwiseKendallTauMedianCILower",
            "MeanPairwiseKendallTauMedianCIUpper",
        ),
        (top1, "FractionCILower", "FractionCIUpper"),
    ]
    for output, lower, upper in interval_pairs:
        assert np.isfinite(output[[lower, upper]]).all().all()
        assert (output[lower] <= output[upper]).all()

    registry = agreement_scope_registry(SPECIES, STATUSES, MODEL_COLUMNS)
    point_fleiss = fleiss_point_estimates(
        crossed_fixture,
        registry,
        MODEL_COLUMNS,
    )
    pd.testing.assert_frame_equal(
        fleiss.loc[:, FLEISS_COLUMNS],
        point_fleiss,
    )
    assert (fleiss["FleissKappa"] < 0).any()
    assert (fleiss["CILower"] < 0).any()
    assert (
        (fleiss["CILower"] < 0) & (fleiss["CIUpper"] > 0)
    ).any()

    gene_rows = ranking_statistics.gene_ordinal_agreement(
        crossed_fixture,
        MODEL_COLUMNS,
    )
    overall = ordinal.loc[ordinal["ScopeID"] == "overall"].iloc[0]
    assert np.isclose(overall["KendallWMean"], gene_rows["KendallW"].mean())
    assert np.isclose(
        overall["KendallWMedian"], gene_rows["KendallW"].median()
    )
    assert np.isclose(
        overall["MeanPairwiseKendallTauMean"],
        gene_rows["MeanPairwiseKendallTau"].mean(),
    )

    patterns = ("unanimous", "majority_2_of_3", "all_different")
    observed_pattern_order = top1.groupby("ScopeID", sort=False)[
        "Top1AgreementPattern"
    ].apply(tuple)
    assert all(value == patterns for value in observed_pattern_order)
    assert (
        top1.loc[
            top1["Top1AgreementPattern"] == "unanimous", "Count"
        ]
        == 0
    ).all()
    assert np.allclose(
        top1.groupby("ScopeID", sort=False)["Fraction"].sum(),
        1.0,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_expert", "exactly three experts"),
        ("duplicate_expert_gene", "duplicate expert/gene rows"),
        ("incomplete_ranking", "complete no-tie R1-R5 ranking"),
        ("missing_stratum", "all 10 Species x StudyStatus strata"),
        ("noncanonical_species", "canonical species and study statuses"),
    ],
)
def test_agreement_bootstrap_validates_canonical_assignment_boundary(
    crossed_fixture: pd.DataFrame,
    mutation: str,
    message: str,
) -> None:
    frame = crossed_fixture.copy()
    if mutation == "missing_expert":
        frame = frame.drop(frame.index[0])
    elif mutation == "duplicate_expert_gene":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif mutation == "incomplete_ranking":
        frame.loc[0, "Gemini"] = "R2"
    elif mutation == "missing_stratum":
        frame = frame.loc[
            ~(
                (frame["Species"] == SPECIES[0])
                & (frame["StudyStatus"] == STATUSES[0])
            )
        ]
    elif mutation == "noncanonical_species":
        frame.loc[frame["Species"] == SPECIES[0], "Species"] = "Barley"

    with pytest.raises(ValueError, match=message):
        bootstrap_agreement(
            frame,
            MODEL_COLUMNS,
            BootstrapConfig(successful_replicates=1),
        )


def test_agreement_bootstrap_uses_species_gene_composite_blocks(
    crossed_fixture: pd.DataFrame,
) -> None:
    frame = crossed_fixture.copy()
    for species in SPECIES:
        first_gene = frame.loc[frame["Species"] == species, "Gene"].iloc[0]
        frame.loc[
            (frame["Species"] == species) & (frame["Gene"] == first_gene),
            "Gene",
        ] = "SHARED_GENE_LABEL"

    result = bootstrap_agreement(
        frame,
        MODEL_COLUMNS,
        BootstrapConfig(successful_replicates=5, max_failed_fits=0),
    )

    assert result["fleiss_kappa"].loc[
        result["fleiss_kappa"]["ScopeID"] == "overall", "NGenes"
    ].item() == 50


def test_kendall_w_perfect_and_zero_concordance() -> None:
    identical = np.array([[1, 2, 3, 4, 5]] * 3)
    zero_sum = np.array(
        [
            [1, 2, 3, 4, 5],
            [3, 4, 5, 1, 2],
            [5, 3, 1, 4, 2],
        ]
    )

    assert np.isclose(ranking_statistics.kendall_w(identical), 1.0)
    assert np.isclose(zero_sum.sum(axis=0), 9.0).all()
    assert np.isclose(ranking_statistics.kendall_w(zero_sum), 0.0)


def test_mean_pairwise_kendall_tau_identical_and_reversed() -> None:
    ascending = np.array([1, 2, 3, 4, 5])

    assert np.isclose(
        ranking_statistics.mean_pairwise_kendall_tau(
            np.vstack([ascending, ascending])
        ),
        1.0,
    )
    assert np.isclose(
        ranking_statistics.mean_pairwise_kendall_tau(
            np.vstack([ascending, ascending[::-1]])
        ),
        -1.0,
    )


@pytest.mark.parametrize(
    ("rank_matrix", "message"),
    [
        (np.array([1, 2, 3]), "at least two raters and items"),
        (np.array([[1, 2, 3]]), "at least two raters and items"),
        (np.array([[1], [1]]), "at least two raters and items"),
        (
            np.array([[1, 1, 3], [1, 2, 3]]),
            "complete no-tie ranking",
        ),
        (
            np.array([[1, 2, 4], [1, 2, 3]]),
            "complete no-tie ranking",
        ),
        (
            np.array([[1, 2, np.nan], [1, 2, 3]]),
            "complete no-tie ranking",
        ),
    ],
)
@pytest.mark.parametrize(
    "function_name",
    ["kendall_w", "mean_pairwise_kendall_tau"],
)
def test_kendall_statistics_reject_invalid_rank_matrices(
    rank_matrix: np.ndarray,
    message: str,
    function_name: str,
) -> None:
    function = getattr(ranking_statistics, function_name)

    with pytest.raises(ValueError, match=message):
        function(rank_matrix)


def test_top1_patterns_are_exhaustive() -> None:
    assert ranking_statistics.top1_pattern(["A", "A", "A"]) == "unanimous"
    assert (
        ranking_statistics.top1_pattern(["A", "A", "B"])
        == "majority_2_of_3"
    )
    assert (
        ranking_statistics.top1_pattern(["A", "B", "C"])
        == "all_different"
    )


@pytest.mark.parametrize(
    "top_models",
    [
        [],
        ["A", "B"],
        ["A", "B", "C", "D"],
        ["A", "B", None],
        ["A", np.nan, "C"],
        ["A", pd.NA, "C"],
        "ABC",
    ],
)
def test_top1_rejects_inputs_other_than_three_nonmissing_labels(
    top_models: object,
) -> None:
    with pytest.raises(ValueError, match="exactly three nonmissing labels"):
        ranking_statistics.top1_pattern(top_models)


def test_ordinal_scope_registry_has_exact_18_canonical_scopes() -> None:
    registry = ranking_statistics.ordinal_scope_registry(SPECIES, STATUSES)
    expected = [
        AgreementScope("overall", "primary", "overall"),
        *[
            AgreementScope(
                f"species.{species.casefold()}",
                "locked_secondary",
                "species",
                species=species,
            )
            for species in SPECIES
        ],
        *[
            AgreementScope(
                f"study_status.{status}",
                "locked_secondary",
                "study_status",
                study_status=status,
            )
            for status in STATUSES
        ],
        *[
            AgreementScope(
                f"species_study_status.{species.casefold()}.{status}",
                "locked_exploratory",
                "species_study_status",
                species=species,
                study_status=status,
            )
            for species in SPECIES
            for status in STATUSES
        ],
    ]

    assert registry == tuple(expected)
    assert len(registry) == 18
    assert len({scope.scope_id for scope in registry}) == 18
    assert Counter(scope.scope_family for scope in registry) == {
        "overall": 1,
        "species": 5,
        "study_status": 2,
        "species_study_status": 10,
    }
    assert all(scope.model == "all" for scope in registry)


def test_ordinal_scope_registry_rejects_noncanonical_dimensions() -> None:
    with pytest.raises(ValueError, match="canonical species and status"):
        ranking_statistics.ordinal_scope_registry(SPECIES[:-1], STATUSES)
    with pytest.raises(ValueError, match="canonical species and status"):
        ranking_statistics.ordinal_scope_registry(
            SPECIES,
            tuple(reversed(STATUSES)),
        )


def test_gene_ordinal_agreement_has_stable_schema_and_order_invariance() -> None:
    frame = categorized_ranking_fixture()

    expected = ranking_statistics.gene_ordinal_agreement(frame, MODEL_COLUMNS)
    permuted_models = tuple(reversed(MODEL_COLUMNS))
    shuffled = frame.loc[
        :,
        ["Species", "Gene", "Expert", "StudyStatus", *permuted_models],
    ].sample(frac=1.0, random_state=20260714)
    observed = ranking_statistics.gene_ordinal_agreement(
        shuffled,
        permuted_models,
    )

    pd.testing.assert_frame_equal(observed, expected)
    assert tuple(expected.columns) == (
        "Species",
        "Gene",
        "StudyStatus",
        "NExperts",
        "NModels",
        "KendallW",
        "MeanPairwiseKendallTau",
        "Top1AgreementPattern",
    )
    assert (
        len(expected)
        == frame[["Species", "Gene"]].drop_duplicates().shape[0]
        == 20
    )
    assert (expected["NExperts"] == 3).all()
    assert (expected["NModels"] == 5).all()
    assert expected["KendallW"].between(0.0, 1.0).all()
    assert expected["MeanPairwiseKendallTau"].between(-1.0, 1.0).all()
    assert set(expected["Top1AgreementPattern"]) == {"all_different"}


def test_gene_ordinal_agreement_rejects_duplicate_expert_gene_rows() -> None:
    frame = categorized_ranking_fixture()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate expert/gene rows"):
        ranking_statistics.gene_ordinal_agreement(duplicated, MODEL_COLUMNS)


def test_gene_ordinal_agreement_uses_species_gene_composite_key() -> None:
    frame = categorized_ranking_fixture()
    rice_gene = frame.loc[
        (frame["Species"] == "Rice")
        & (frame["StudyStatus"] == "well_studied"),
        "Gene",
    ].iloc[0]
    maize_gene = frame.loc[
        (frame["Species"] == "Maize")
        & (frame["StudyStatus"] == "well_studied"),
        "Gene",
    ].iloc[0]
    shared = frame.loc[
        frame["Gene"].isin([rice_gene, maize_gene])
    ].copy()
    shared["Gene"] = "SHARED_GENE_LABEL"

    observed = ranking_statistics.gene_ordinal_agreement(
        shared,
        MODEL_COLUMNS,
    )

    assert observed[["Species", "Gene"]].to_dict("records") == [
        {"Species": "Maize", "Gene": "SHARED_GENE_LABEL"},
        {"Species": "Rice", "Gene": "SHARED_GENE_LABEL"},
    ]
    assert (observed["NExperts"] == 3).all()


@pytest.mark.parametrize("invalid_rank", [None, "R2", "R6"])
def test_gene_ordinal_agreement_rejects_incomplete_rankings(
    invalid_rank: object,
) -> None:
    frame = categorized_ranking_fixture()
    frame.loc[0, "Gemini"] = invalid_rank

    with pytest.raises(ValueError, match="complete no-tie R1-R5 ranking"):
        ranking_statistics.gene_ordinal_agreement(frame, MODEL_COLUMNS)


@pytest.mark.parametrize(
    ("column", "replacement", "message"),
    [
        ("Species", "NotRice", "exactly three experts"),
        ("StudyStatus", "not_the_status", "mixed study status"),
    ],
)
def test_gene_ordinal_agreement_rejects_mixed_gene_metadata(
    column: str,
    replacement: str,
    message: str,
) -> None:
    frame = categorized_ranking_fixture()
    frame.loc[0, column] = replacement

    with pytest.raises(ValueError, match=message):
        ranking_statistics.gene_ordinal_agreement(frame, MODEL_COLUMNS)


def test_gene_ordinal_agreement_requires_three_experts_and_five_models() -> None:
    frame = categorized_ranking_fixture()
    first_gene = frame.loc[0, "Gene"]
    missing_expert = frame.drop(
        frame.index[(frame["Gene"] == first_gene)][0]
    )

    with pytest.raises(ValueError, match="exactly three experts"):
        ranking_statistics.gene_ordinal_agreement(
            missing_expert,
            MODEL_COLUMNS,
        )
    with pytest.raises(ValueError, match="exactly five model columns"):
        ranking_statistics.gene_ordinal_agreement(
            frame,
            MODEL_COLUMNS[:-1],
        )


def test_fleiss_matches_published_fixture() -> None:
    result = fleiss_kappa_from_counts(FLEISS_1971_COUNTS)

    assert np.isclose(result["observed_agreement"], 0.378021978021978)
    assert np.isclose(result["expected_agreement"], 0.212755102040816)
    assert np.isclose(result["fleiss_kappa"], 0.20993070442195522)
    np.testing.assert_allclose(
        result["rank_marginals"],
        FLEISS_1971_COUNTS.sum(axis=0) / FLEISS_1971_COUNTS.sum(),
    )


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        (np.array([1, 1]), "at least two items and categories"),
        (np.array([[1, 1]]), "at least two items and categories"),
        (np.array([[1], [1]]), "at least two items and categories"),
        (
            np.array([[2, 0], [np.nan, 2]]),
            "finite and non-negative",
        ),
        (
            np.array([[2, 0], [np.inf, 2]]),
            "finite and non-negative",
        ),
        (np.array([[2, 0], [-1, 3]]), "finite and non-negative"),
        (np.array([[1.5, 0.5], [1, 1]]), "integer-valued"),
        (
            np.array([[2, 1, 0], [1, 1, 0]]),
            "same number of ratings",
        ),
        (np.array([[1, 0], [0, 1]]), "at least two ratings"),
        (
            np.array([[2, 0], [2, 0]]),
            "expected agreement must be less than one",
        ),
    ],
)
def test_fleiss_rejects_invalid_counts(
    counts: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fleiss_kappa_from_counts(counts)


def test_fleiss_preserves_negative_values() -> None:
    counts = np.array(
        [
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1],
        ]
    )

    result = fleiss_kappa_from_counts(counts)

    assert np.isclose(result["fleiss_kappa"], -0.25)


def test_agreement_registry_has_exact_order_tiers_and_dimensions() -> None:
    registry = agreement_scope_registry(SPECIES, STATUSES, MODEL_COLUMNS)
    expected = [
        AgreementScope("overall", "primary", "overall"),
        *[
            AgreementScope(
                f"species.{species.casefold()}",
                "locked_secondary",
                "species",
                species=species,
            )
            for species in SPECIES
        ],
        *[
            AgreementScope(
                f"study_status.{status}",
                "locked_secondary",
                "study_status",
                study_status=status,
            )
            for status in STATUSES
        ],
        *[
            AgreementScope(
                f"model.{model.casefold()}",
                "locked_secondary",
                "model",
                model=model,
            )
            for model in MODEL_COLUMNS
        ],
        *[
            AgreementScope(
                f"species_study_status.{species.casefold()}.{status}",
                "locked_exploratory",
                "species_study_status",
                species=species,
                study_status=status,
            )
            for species in SPECIES
            for status in STATUSES
        ],
        *[
            AgreementScope(
                f"model_study_status.{model.casefold()}.{status}",
                "locked_exploratory",
                "model_study_status",
                study_status=status,
                model=model,
            )
            for model in MODEL_COLUMNS
            for status in STATUSES
        ],
        *[
            AgreementScope(
                f"model_species.{model.casefold()}.{species.casefold()}",
                "locked_exploratory",
                "model_species",
                species=species,
                model=model,
            )
            for model in MODEL_COLUMNS
            for species in SPECIES
        ],
    ]

    assert registry == tuple(expected)
    assert len(registry) == 58
    assert len({scope.scope_id for scope in registry}) == 58
    assert Counter(scope.scope_family for scope in registry) == {
        "overall": 1,
        "species": 5,
        "study_status": 2,
        "model": 5,
        "species_study_status": 10,
        "model_study_status": 10,
        "model_species": 25,
    }
    assert all(
        sum(
            value != "all"
            for value in (scope.species, scope.study_status, scope.model)
        )
        <= 2
        for scope in registry
    )
    assert all(
        value is not None
        for scope in registry
        for value in (scope.species, scope.study_status, scope.model)
    )
    with pytest.raises(FrozenInstanceError):
        registry[0].scope_id = "changed"


def test_agreement_registry_rejects_noncanonical_dimensions() -> None:
    with pytest.raises(ValueError, match="canonical species, status, and model"):
        agreement_scope_registry(SPECIES[:-1], STATUSES, MODEL_COLUMNS)
    with pytest.raises(ValueError, match="canonical species, status, and model"):
        agreement_scope_registry(
            tuple(reversed(SPECIES)),
            STATUSES,
            MODEL_COLUMNS,
        )


def test_fleiss_point_estimates_rejects_noncanonical_registry() -> None:
    frame = categorized_ranking_fixture()
    registry = agreement_scope_registry(SPECIES, STATUSES, MODEL_COLUMNS)

    with pytest.raises(ValueError, match="exact canonical 58-scope registry"):
        fleiss_point_estimates(frame, registry[:-1], MODEL_COLUMNS)
    with pytest.raises(ValueError, match="exact canonical 58-scope registry"):
        fleiss_point_estimates(
            frame,
            (registry[1], registry[0], *registry[2:]),
            MODEL_COLUMNS,
        )


def test_fleiss_point_estimates_rejects_triple_interaction() -> None:
    frame = categorized_ranking_fixture()
    forbidden = AgreementScope(
        "forbidden.triple",
        "locked_exploratory",
        "model_species_study_status",
        species=SPECIES[0],
        study_status=STATUSES[0],
        model=MODEL_COLUMNS[0],
    )

    with pytest.raises(ValueError, match="three restricted dimensions"):
        fleiss_point_estimates(frame, (forbidden,), MODEL_COLUMNS)


def test_fleiss_point_estimates_have_stable_schema_and_marginals() -> None:
    frame = categorized_ranking_fixture()
    registry = agreement_scope_registry(SPECIES, STATUSES, MODEL_COLUMNS)

    estimates = fleiss_point_estimates(frame, registry, MODEL_COLUMNS)

    assert tuple(estimates.columns) == FLEISS_COLUMNS
    assert estimates["ScopeID"].tolist() == [
        scope.scope_id for scope in registry
    ]
    assert len(estimates) == 58
    assert estimates["ScopeID"].is_unique
    assert (estimates["NRatings"] == 3 * estimates["NItems"]).all()
    assert (estimates["RatingsPerItem"] == 3).all()
    assert (estimates["NContributingExperts"] == 3).all()

    rank_columns = [f"RankR{rank}Share" for rank in range(1, 6)]
    all_model = estimates[estimates["Model"] == "all"]
    np.testing.assert_allclose(all_model[rank_columns], 0.2, atol=1e-15)
    np.testing.assert_allclose(
        all_model["ExpectedAgreement"],
        0.2,
        atol=1e-15,
    )
    assert (all_model["NItems"] == 5 * all_model["NGenes"]).all()

    gemini_scope = estimates.loc[
        estimates["ScopeID"] == "model.gemini"
    ].iloc[0]
    observed_marginals = (
        frame["Gemini"]
        .value_counts(normalize=True)
        .reindex([f"R{rank}" for rank in range(1, 6)], fill_value=0.0)
    )
    np.testing.assert_allclose(
        gemini_scope[rank_columns].to_numpy(dtype=float),
        observed_marginals.to_numpy(),
    )
    assert np.isclose(
        gemini_scope["ExpectedAgreement"],
        np.square(observed_marginals).sum(),
    )
    assert gemini_scope["ExpectedAgreement"] > 0.2


def historical_fixture() -> tuple[list[str], list[list[str]]]:
    models = ["Gemini", "Grok", "OpenAI", "Phytomni"]
    rankings = [list(order) for order in permutations(models)]
    rankings.extend([["Phytomni", "OpenAI", "Grok", "Gemini"]] * 6)
    return models, rankings


def test_pl_point_estimates_match_historical_fixture() -> None:
    models, rankings = historical_fixture()

    fit = fit_plackett_luce(rankings, models)

    assert np.isclose(
        fit["negative_log_likelihood"],
        93.43927901604626,
        rtol=0.0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        [fit["xi"][model] for model in models],
        [
            0.0,
            0.3126965119014714,
            0.49237527893347205,
            0.6228591998928711,
        ],
        rtol=0.0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        fit["elo"],
        [
            1437.9857450188267,
            1492.3066928705082,
            1523.5200916853792,
            1546.1874704252862,
        ],
        rtol=0.0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        fit["elo_standard_error"],
        [0.0, 56.457465939984594, 58.374011490354356, 59.62791502570149],
        rtol=0.0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        fit["elo_lower"],
        [
            1437.9857450188267,
            1381.6500596281385,
            1409.1070291642845,
            1429.3167569749112,
        ],
        rtol=0.0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        fit["elo_upper"],
        [
            1437.9857450188267,
            1602.963326112878,
            1637.9331542064738,
            1663.0581838756611,
        ],
        rtol=0.0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        fit["pairwise_probabilities"],
        [
            [np.nan, 0.42245668772758843, 0.3793341724914228, 0.34913145067062423],
            [0.5775433122724115, np.nan, 0.45520077001606085, 0.4230750290714746],
            [0.6206658275085771, 0.5447992299839391, np.nan, 0.4674252249724183],
            [0.6508685493293758, 0.5769249709285255, 0.5325747750275817, np.nan],
        ],
        rtol=0.0,
        atol=1e-8,
    )


def test_integer_weights_equal_expanded_rankings() -> None:
    models = list(MODEL_COLUMNS)
    unique = [models, list(reversed(models))]

    weighted = fit_plackett_luce(
        unique,
        models,
        weights=np.array([3.0, 2.0]),
    )
    expanded = fit_plackett_luce(
        [unique[0]] * 3 + [unique[1]] * 2,
        models,
    )

    assert np.isclose(
        weighted["negative_log_likelihood"],
        expanded["negative_log_likelihood"],
        atol=1e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        [weighted["xi"][model] for model in weighted["models"]],
        [expanded["xi"][model] for model in expanded["models"]],
        atol=1e-8,
        rtol=0.0,
    )
    for key in (
        "covariance",
        "elo",
        "elo_standard_error",
        "elo_lower",
        "elo_upper",
        "pairwise_probabilities",
    ):
        np.testing.assert_allclose(
            weighted[key],
            expanded[key],
            atol=1e-8,
            rtol=0.0,
        )


def test_collapse_weighted_rankings_sums_duplicate_weights() -> None:
    rankings = [
        ["Gemini", "Claude", "Grok"],
        ["Grok", "Claude", "Gemini"],
        ["Gemini", "Claude", "Grok"],
        ["Gemini", "Claude", "Grok"],
    ]

    collapsed, weights = collapse_weighted_rankings(
        rankings,
        weights=np.array([1.5, 2.0, 0.5, 0.0]),
    )

    assert collapsed == rankings[:2]
    np.testing.assert_array_equal(weights, np.array([2.0, 2.0]))


def test_collapse_weighted_rankings_counts_unweighted_duplicates() -> None:
    rankings = [["Gemini", "Claude"], ["Claude", "Gemini"]]

    collapsed, weights = collapse_weighted_rankings(
        [rankings[0], rankings[1], rankings[0]],
    )

    assert collapsed == rankings
    np.testing.assert_array_equal(weights, np.array([2.0, 1.0]))


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (np.array([1.0]), "one value per ranking"),
        (np.array([1.0, -1.0]), "finite, non-negative, and non-zero"),
        (np.array([1.0, np.nan]), "finite, non-negative, and non-zero"),
        (np.array([0.0, 0.0]), "finite, non-negative, and non-zero"),
    ],
)
def test_pl_rejects_invalid_ranking_weights(
    weights: np.ndarray,
    message: str,
) -> None:
    models = ["Gemini", "Claude"]
    rankings = [models, list(reversed(models))]

    with pytest.raises(ValueError, match=message):
        fit_plackett_luce(rankings, models, weights=weights)


def test_pl_likelihood_preserves_historical_fixture() -> None:
    models, rankings = historical_fixture()
    xi = {model: 0.0 for model in models}

    log_likelihood, gradient = pl_loglik_and_grad(
        xi,
        rankings,
        models,
        "Gemini",
    )

    assert np.isclose(
        log_likelihood,
        -95.34161491043835,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        gradient,
        [-0.5, 2.5, 4.5],
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("model_count", [2, 3, 5, 6])
def test_pl_likelihood_supports_arbitrary_model_counts(
    model_count: int,
) -> None:
    models = ["Gemini", *[f"Model_{index}" for index in range(1, model_count)]]
    ranking = list(reversed(models))
    xi = {model: 0.0 for model in models}

    log_likelihood, gradient = pl_loglik_and_grad(
        xi,
        [ranking],
        models,
        "Gemini",
    )

    assert np.isclose(
        log_likelihood,
        -sum(np.log(stage_size) for stage_size in range(2, model_count + 1)),
        atol=1e-12,
    )
    expected_gradient = [
        1.0
        - sum(
            1.0 / (model_count - stage)
            for stage in range(ranking.index(model) + 1)
        )
        for model in models
        if model != "Gemini"
    ]
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-12)


def test_parse_rankings_preserves_permissive_semantics() -> None:
    frame = pd.DataFrame(
        {
            "Gemini": ["R10", "R1", "R5"],
            "Grok": ["R2", "not-a-rank", "R5"],
            "OpenAI": ["R2", "R3", "R5"],
            "Phytomni": ["R-3", "R4", "R5"],
            "Claude": ["R-99", "R0", "R0"],
        }
    )

    rankings, skipped = parse_rankings(frame)

    assert skipped == 1
    assert rankings == [
        ["Claude", "Phytomni", "Grok", "OpenAI", "Gemini"],
        ["Claude", "Gemini", "Grok", "OpenAI", "Phytomni"],
    ]


def test_elo_output_tables_preserve_stable_contract() -> None:
    models = ["Gemini", "Grok", "OpenAI", "Phytomni"]
    fit = {
        "models": models,
        "elo": np.array([1400.0, 1600.0, 1500.0, 1550.0]),
        "elo_lower": np.array([1390.0, 1590.0, 1490.0, 1540.0]),
        "elo_upper": np.array([1410.0, 1610.0, 1510.0, 1560.0]),
        "pairwise_probabilities": np.arange(16, dtype=float).reshape(4, 4),
    }

    score_table, probability_table = elo_outputs(fit)

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


RANKING_SCOPE_KEYS = (
    ("overall", "all", "all"),
    ("well_studied", "well_studied", "all"),
    *(
        (f"well_studied.{species.casefold()}", "well_studied", species)
        for species in SPECIES
    ),
    ("uncharacterized", "uncharacterized", "all"),
    *(
        (
            f"uncharacterized.{species.casefold()}",
            "uncharacterized",
            species,
        )
        for species in SPECIES
    ),
    *((species.casefold(), "all", species) for species in SPECIES),
)
INTERVAL_ANALYSES = (
    "crossed_expert_gene",
    "expert_cluster",
    "gene_cluster",
)


def _production_gene_cluster_abnormal_case(
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(__file__).resolve().parents[1]
    release = pd.read_csv(
        root
        / "DeepGenomeAgent Evaluation"
        / "supplementary"
        / "Supplementary_Data_Expert_Rankings.tsv",
        sep="\t",
        dtype=str,
    ).rename(columns={"AnonymousExpertID": "Expert"})
    categories = pd.read_csv(
        root
        / "Supplementary Fig. 10-13"
        / "PhytoBench-Gene-for_plot"
        / "gene_categories.tsv",
        sep="\t",
        dtype=str,
    ).set_index(["Species", "Gene"])["StudyStatus"]
    release_keys = pd.MultiIndex.from_frame(release[["Species", "Gene"]])
    assert release["StudyStatus"].tolist() == categories.reindex(
        release_keys
    ).tolist()

    working = ranking_statistics._validate_pl_bootstrap_frame(
        release,
        MODEL_COLUMNS,
    )
    registry = ranking_statistics.ranking_scope_registry()
    assert registry[11].scope == "uncharacterized.soybean"
    scope_indices = ranking_statistics._ranking_scope_indices(
        working,
        registry,
    )
    rank_lookup = {f"R{rank}": rank - 1 for rank in range(1, 6)}
    rank_numbers = np.array(
        [
            [rank_lookup[value] for value in row]
            for row in working.loc[:, MODEL_COLUMNS].itertuples(
                index=False,
                name=None,
            )
        ],
        dtype=int,
    )
    encoded_rankings = np.argsort(rank_numbers, axis=1, kind="stable")
    unique_rankings, row_permutations = np.unique(
        encoded_rankings,
        axis=0,
        return_inverse=True,
    )
    *_, initial_vectors = ranking_statistics._scope_point_statistics(
        working,
        MODEL_COLUMNS,
        registry,
        scope_indices,
        encoded_rankings,
    )

    genes = (
        working[["Species", "Gene", "StudyStatus"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    gene_keys = pd.MultiIndex.from_frame(genes[["Species", "Gene"]])
    row_gene_indices = gene_keys.get_indexer(
        pd.MultiIndex.from_frame(working[["Species", "Gene"]])
    )
    gene_seed = np.random.SeedSequence(20260714).spawn(2)[1]
    gene_rng = np.random.default_rng(gene_seed)
    gene_counts = None
    for _ in range(391):
        gene_counts = ranking_statistics.sample_within_strata(
            genes,
            id_columns=("Species", "Gene"),
            strata=("Species", "StudyStatus"),
            rng=gene_rng,
        ).to_numpy(dtype=float)
    assert gene_counts is not None

    row_indices = scope_indices[11]
    selected_weights = gene_counts[row_gene_indices][row_indices]
    permutation_weights = np.bincount(
        row_permutations[row_indices],
        weights=selected_weights,
        minlength=len(unique_rankings),
    )
    retained = permutation_weights > 0
    return (
        unique_rankings[retained],
        permutation_weights[retained],
        initial_vectors[11],
    )


def test_bootstrap_pl_fit_retries_production_abnormal_line_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rankings, weights, initial_vector = (
        _production_gene_cluster_abnormal_case()
    )
    implementation = ranking_statistics.minimize
    calls: list[tuple[np.ndarray, str, dict[str, object]]] = []

    def record_minimize(fun, x0, *, jac, method, options):
        calls.append((np.asarray(x0).copy(), method, dict(options)))
        return implementation(
            fun,
            x0,
            jac=jac,
            method=method,
            options=options,
        )

    monkeypatch.setattr(ranking_statistics, "minimize", record_minimize)
    fit = ranking_statistics._bootstrap_pl_fit(
        rankings,
        weights,
        MODEL_COLUMNS,
        initial_vector,
    )

    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0], initial_vector)
    np.testing.assert_array_equal(calls[1][0], initial_vector)
    assert calls[0][1] == calls[1][1] == "L-BFGS-B"
    assert calls[0][2] == ranking_statistics.OPTIMIZER_OPTIONS
    assert calls[1][2] == {
        **ranking_statistics.OPTIMIZER_OPTIONS,
        "maxls": 50,
    }
    assert fit["optimizer_result"].success
    assert np.isfinite(fit["elo"]).all()
    probabilities = fit["pairwise_probabilities"]
    off_diagonal = ~np.eye(len(MODEL_COLUMNS), dtype=bool)
    assert np.isfinite(probabilities[off_diagonal]).all()


@pytest.mark.parametrize(
    ("message", "fun", "vector"),
    [
        ("STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT", 1.0, np.zeros(4)),
        ("ABNORMAL: ", np.inf, np.zeros(4)),
        ("ABNORMAL: ", 1.0, np.full(4, np.nan)),
    ],
)
def test_bootstrap_pl_fit_only_retries_finite_abnormal_results(
    monkeypatch: pytest.MonkeyPatch,
    message: str,
    fun: float,
    vector: np.ndarray,
) -> None:
    calls = 0

    def failed_minimize(*args, **kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            success=False,
            fun=fun,
            x=vector.copy(),
            message=message,
        )

    monkeypatch.setattr(ranking_statistics, "minimize", failed_minimize)
    with pytest.raises(RuntimeError, match="optimization failed"):
        ranking_statistics._bootstrap_pl_fit(
            np.array([[0, 1, 2, 3, 4]]),
            np.array([1.0]),
            MODEL_COLUMNS,
            np.zeros(4),
        )
    assert calls == 1


def test_bootstrap_pl_fit_rejects_failed_abnormal_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def failed_minimize(*args, **kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            success=False,
            fun=1.0,
            x=np.zeros(4),
            message="ABNORMAL: ",
        )

    monkeypatch.setattr(ranking_statistics, "minimize", failed_minimize)
    with pytest.raises(RuntimeError, match="optimization failed"):
        ranking_statistics._bootstrap_pl_fit(
            np.array([[0, 1, 2, 3, 4]]),
            np.array([1.0]),
            MODEL_COLUMNS,
            np.zeros(4),
        )
    assert calls == 2


@pytest.fixture(scope="module")
def crossed_pl_bootstrap_result(
    crossed_fixture: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return ranking_statistics.bootstrap_plackett_luce_statistics(
        crossed_fixture,
        MODEL_COLUMNS,
        BootstrapConfig(
            successful_replicates=30,
            seed=20260714,
            max_failed_fits=2,
        ),
    )


def test_crossed_pl_bootstrap_has_exact_schemas_cardinalities_and_order(
    crossed_pl_bootstrap_result: dict[str, pd.DataFrame],
) -> None:
    scores = crossed_pl_bootstrap_result["pl_scores_ci"]
    ranks = crossed_pl_bootstrap_result["rank_distribution_ci"]
    pairwise = crossed_pl_bootstrap_result["pl_pairwise_ci"]

    assert tuple(scores.columns) == (
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
    )
    assert tuple(ranks.columns) == (
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
    )
    assert tuple(pairwise.columns) == (
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
    )
    assert len(scores) == 18 * 5 * 3
    assert len(ranks) == 18 * 5 * 5
    assert len(pairwise) == 18 * 5 * 4

    score_keys = list(
        scores[["Scope", "StudyStatus", "Species", "Model", "IntervalAnalysis"]]
        .itertuples(index=False, name=None)
    )
    assert score_keys == [
        (*scope, model, analysis)
        for scope in RANKING_SCOPE_KEYS
        for model in MODEL_COLUMNS
        for analysis in INTERVAL_ANALYSES
    ]
    rank_keys = list(
        ranks[["Scope", "StudyStatus", "Species", "Model", "Rank"]]
        .itertuples(index=False, name=None)
    )
    assert rank_keys == [
        (*scope, model, f"R{rank}")
        for scope in RANKING_SCOPE_KEYS
        for model in MODEL_COLUMNS
        for rank in range(1, 6)
    ]
    pair_keys = list(
        pairwise[
            ["Scope", "StudyStatus", "Species", "RowModel", "ColumnModel"]
        ].itertuples(index=False, name=None)
    )
    assert pair_keys == [
        (*scope, row_model, column_model)
        for scope in RANKING_SCOPE_KEYS
        for row_model in MODEL_COLUMNS
        for column_model in MODEL_COLUMNS
        if row_model != column_model
    ]


def test_crossed_pl_bootstrap_has_finite_centered_ordered_intervals(
    crossed_pl_bootstrap_result: dict[str, pd.DataFrame],
) -> None:
    scores = crossed_pl_bootstrap_result["pl_scores_ci"]
    ranks = crossed_pl_bootstrap_result["rank_distribution_ci"]
    pairwise = crossed_pl_bootstrap_result["pl_pairwise_ci"]

    assert tuple(scores["IntervalAnalysis"].drop_duplicates()) == (
        INTERVAL_ANALYSES
    )
    assert np.isfinite(
        scores[["Estimate", "CI95Lower", "CI95Upper"]]
    ).all().all()
    assert np.isfinite(
        ranks[["Fraction", "CI95Lower", "CI95Upper"]]
    ).all().all()
    assert np.isfinite(
        pairwise[["Probability", "CI95Lower", "CI95Upper"]]
    ).all().all()
    for output, point in (
        (scores, "Estimate"),
        (ranks, "Fraction"),
        (pairwise, "Probability"),
    ):
        assert (output["CI95Lower"] <= output["CI95Upper"]).all()
        assert (output["CI95Lower"] <= output[point]).all()
        assert (output[point] <= output["CI95Upper"]).all()

    means = scores.groupby(
        ["Scope", "IntervalAnalysis"], sort=False
    )["Estimate"].mean()
    assert (means.sub(1500.0).abs() < 1e-8).all()
    assert (ranks["Fraction"].between(0.0, 1.0)).all()
    assert (pairwise["Probability"].between(0.0, 1.0)).all()
    assert (pairwise["RowModel"] != pairwise["ColumnModel"]).all()
    for output in crossed_pl_bootstrap_result.values():
        assert (output["SuccessfulReplicates"] == 30).all()
        assert (output["FailedFits"] >= 0).all()
        assert output["SeedStream"].nunique() == 1
        assert output["SeedStream"].iloc[0]


def _ranking_scope_frame(
    frame: pd.DataFrame,
    study_status: str,
    species: str,
) -> pd.DataFrame:
    selected = frame
    if study_status != "all":
        selected = selected.loc[selected["StudyStatus"] == study_status]
    if species != "all":
        selected = selected.loc[selected["Species"] == species]
    return selected


def test_crossed_pl_bootstrap_preserves_unweighted_point_estimates(
    crossed_fixture: pd.DataFrame,
    crossed_pl_bootstrap_result: dict[str, pd.DataFrame],
) -> None:
    scores = crossed_pl_bootstrap_result["pl_scores_ci"]
    ranks = crossed_pl_bootstrap_result["rank_distribution_ci"]
    pairwise = crossed_pl_bootstrap_result["pl_pairwise_ci"]

    for scope, study_status, species in RANKING_SCOPE_KEYS:
        selected = _ranking_scope_frame(
            crossed_fixture,
            study_status,
            species,
        )
        rankings, skipped = parse_rankings(selected, MODEL_COLUMNS)
        assert skipped == 0
        fit = fit_plackett_luce(rankings, list(MODEL_COLUMNS))
        fit_index = {
            model: index for index, model in enumerate(fit["models"])
        }
        selected_scores = scores.loc[
            (scores["Scope"] == scope)
            & (scores["IntervalAnalysis"] == "crossed_expert_gene")
        ]
        np.testing.assert_allclose(
            selected_scores["Estimate"],
            [fit["elo"][fit_index[model]] for model in MODEL_COLUMNS],
            atol=1e-8,
            rtol=0.0,
        )
        selected_pairwise = pairwise.loc[pairwise["Scope"] == scope]
        np.testing.assert_allclose(
            selected_pairwise["Probability"],
            [
                fit["pairwise_probabilities"][
                    fit_index[row_model], fit_index[column_model]
                ]
                for row_model in MODEL_COLUMNS
                for column_model in MODEL_COLUMNS
                if row_model != column_model
            ],
            atol=1e-8,
            rtol=0.0,
        )
        selected_ranks = ranks.loc[ranks["Scope"] == scope]
        expected_counts = [
            int((selected[model] == f"R{rank}").sum())
            for model in MODEL_COLUMNS
            for rank in range(1, 6)
        ]
        assert selected_ranks["Count"].tolist() == expected_counts
        np.testing.assert_allclose(
            selected_ranks["Fraction"],
            np.asarray(expected_counts) / len(selected),
        )

        expected_experts = selected[["Species", "Expert"]].drop_duplicates()
        expected_genes = selected[["Species", "Gene"]].drop_duplicates()
        for output in (selected_scores, selected_ranks, selected_pairwise):
            assert (output["NJudgments"] == len(selected)).all()
            assert (output["NExperts"] == len(expected_experts)).all()
            assert (output["NGenes"] == len(expected_genes)).all()


def test_crossed_pl_bootstrap_is_same_seed_reproducible(
    crossed_fixture: pd.DataFrame,
    crossed_pl_bootstrap_result: dict[str, pd.DataFrame],
) -> None:
    repeated = ranking_statistics.bootstrap_plackett_luce_statistics(
        crossed_fixture,
        MODEL_COLUMNS,
        BootstrapConfig(
            successful_replicates=30,
            seed=20260714,
            max_failed_fits=2,
        ),
    )

    assert repeated.keys() == crossed_pl_bootstrap_result.keys()
    for name in repeated:
        pd.testing.assert_frame_equal(
            repeated[name], crossed_pl_bootstrap_result[name]
        )
        assert repeated[name].to_csv(index=False) == (
            crossed_pl_bootstrap_result[name].to_csv(index=False)
        )


def test_sample_within_strata_preserves_cluster_draw_sizes(
    crossed_fixture: pd.DataFrame,
) -> None:
    experts = crossed_fixture[["Species", "Expert"]].drop_duplicates()
    genes = crossed_fixture[
        ["Species", "Gene", "StudyStatus"]
    ].drop_duplicates()
    seed = np.random.SeedSequence(20260714)
    expert_seed, gene_seed = seed.spawn(2)

    expert_counts = ranking_statistics.sample_within_strata(
        experts,
        id_columns=("Expert",),
        strata=("Species",),
        rng=np.random.default_rng(expert_seed),
    )
    gene_counts = ranking_statistics.sample_within_strata(
        genes,
        id_columns=("Species", "Gene"),
        strata=("Species", "StudyStatus"),
        rng=np.random.default_rng(gene_seed),
    )

    assert expert_counts.index.equals(experts.index)
    assert gene_counts.index.equals(genes.index)
    pd.testing.assert_series_equal(
        expert_counts.groupby(experts["Species"], sort=False).sum(),
        experts.groupby("Species", sort=False).size(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        gene_counts.groupby(
            [genes["Species"], genes["StudyStatus"]], sort=False
        ).sum(),
        genes.groupby(["Species", "StudyStatus"], sort=False).size(),
        check_names=False,
    )


def test_crossed_pl_bootstrap_reuses_one_expert_and_gene_draw_per_attempt(
    crossed_fixture: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[tuple[str, ...], pd.DataFrame, pd.Series]] = []
    implementation = ranking_statistics.sample_within_strata

    def record_draw(
        units: pd.DataFrame,
        *,
        id_columns: tuple[str, ...],
        strata: tuple[str, ...],
        rng: np.random.Generator,
    ) -> pd.Series:
        draw = implementation(
            units,
            id_columns=id_columns,
            strata=strata,
            rng=rng,
        )
        observed.append((strata, units.copy(), draw.copy()))
        return draw

    monkeypatch.setattr(
        ranking_statistics,
        "sample_within_strata",
        record_draw,
    )
    ranking_statistics.bootstrap_plackett_luce_statistics(
        crossed_fixture,
        MODEL_COLUMNS,
        BootstrapConfig(successful_replicates=3, max_failed_fits=0),
    )

    assert [strata for strata, _, _ in observed] == [
        ("Species",),
        ("Species", "StudyStatus"),
    ] * 3
    for strata, units, draw in observed:
        sampled_sizes = draw.groupby(
            [units[column] for column in strata],
            sort=False,
        ).sum()
        original_sizes = units.groupby(list(strata), sort=False).size()
        pd.testing.assert_series_equal(
            sampled_sizes,
            original_sizes,
            check_names=False,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_expert", "exactly three experts"),
        ("duplicate_expert_gene", "duplicate expert/gene rows"),
        ("incomplete_ranking", "complete no-tie R1-R5 ranking"),
        ("missing_stratum", "all 10 Species x StudyStatus strata"),
        ("noncanonical_species", "canonical species and study statuses"),
        ("cross_species_expert", "exactly one species"),
        ("unbalanced_expert", "balanced expert assignments"),
    ],
)
def test_crossed_pl_bootstrap_validates_exact_assignment_boundary(
    crossed_fixture: pd.DataFrame,
    mutation: str,
    message: str,
) -> None:
    frame = crossed_fixture.copy()
    if mutation == "missing_expert":
        frame = frame.drop(frame.index[0])
    elif mutation == "duplicate_expert_gene":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif mutation == "incomplete_ranking":
        frame.loc[0, "Gemini"] = "R2"
    elif mutation == "missing_stratum":
        frame = frame.loc[
            ~(
                (frame["Species"] == SPECIES[0])
                & (frame["StudyStatus"] == STATUSES[0])
            )
        ]
    elif mutation == "noncanonical_species":
        frame.loc[frame["Species"] == SPECIES[0], "Species"] = "Barley"
    elif mutation == "cross_species_expert":
        frame.loc[0, "Expert"] = frame.loc[
            frame["Species"] == SPECIES[1], "Expert"
        ].iloc[0]
    elif mutation == "unbalanced_expert":
        frame.loc[0, "Expert"] = f"{SPECIES[0]}-expert-4"

    with pytest.raises(ValueError, match=message):
        ranking_statistics.bootstrap_plackett_luce_statistics(
            frame,
            MODEL_COLUMNS,
            BootstrapConfig(successful_replicates=1),
        )


def test_crossed_pl_bootstrap_uses_species_gene_composite_keys(
    crossed_fixture: pd.DataFrame,
) -> None:
    frame = crossed_fixture.copy()
    for species in SPECIES:
        first_gene = frame.loc[frame["Species"] == species, "Gene"].iloc[0]
        frame.loc[
            (frame["Species"] == species) & (frame["Gene"] == first_gene),
            "Gene",
        ] = "SHARED_GENE_LABEL"

    result = ranking_statistics.bootstrap_plackett_luce_statistics(
        frame,
        MODEL_COLUMNS,
        BootstrapConfig(successful_replicates=2, max_failed_fits=0),
    )
    overall = result["pl_scores_ci"].loc[
        result["pl_scores_ci"]["Scope"] == "overall"
    ]
    assert (overall["NGenes"] == 50).all()


def test_crossed_pl_bootstrap_records_failures_and_enforces_threshold(
    crossed_fixture: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation = ranking_statistics._bootstrap_pl_fit
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic optimizer failure")
        return implementation(*args, **kwargs)

    monkeypatch.setattr(ranking_statistics, "_bootstrap_pl_fit", fail_once)
    result = ranking_statistics.bootstrap_plackett_luce_statistics(
        crossed_fixture,
        MODEL_COLUMNS,
        BootstrapConfig(successful_replicates=2, max_failed_fits=1),
    )
    for output in result.values():
        assert (output["SuccessfulReplicates"] == 2).all()
        assert (output["FailedFits"] == 1).all()
        diagnostics = output.attrs["bootstrap_diagnostics"]
        assert diagnostics["AttemptedReplicates"] == 3
        assert diagnostics["SuccessfulReplicates"] == 2
        assert diagnostics["FailedFits"] == 1
        assert diagnostics["FailureReasons"] == (
            "synthetic optimizer failure",
        )
        assert diagnostics["HalfRunStability"] == {"Applied": False}

    def always_fail(*args, **kwargs):
        raise RuntimeError("synthetic optimizer failure")

    monkeypatch.setattr(ranking_statistics, "_bootstrap_pl_fit", always_fail)
    with pytest.raises(RuntimeError, match="exceeded max_failed_fits"):
        ranking_statistics.bootstrap_plackett_luce_statistics(
            crossed_fixture,
            MODEL_COLUMNS,
            BootstrapConfig(successful_replicates=1, max_failed_fits=2),
        )


def test_validate_half_run_stability_is_descriptive_and_nonblocking() -> None:
    diagnostics = ranking_statistics.validate_half_run_stability(
        np.array([[1400.0, 1600.0], [1450.0, 1550.0]]),
        np.array([[1412.5, 1587.5], [1451.0, 1549.0]]),
        np.array([[0.25, 0.75], [0.40, 0.60]]),
        np.array([[0.2631, 0.7369], [0.399, 0.601]]),
    )
    assert diagnostics == {
        "Interpretation": "descriptive_nonblocking",
        "ScoreMaxBoundDifference": 12.5,
        "ProbabilityMaxBoundDifference": pytest.approx(0.0131),
    }
    serialized = str(diagnostics).casefold()
    assert "threshold" not in serialized
    assert "passed" not in serialized


@pytest.mark.parametrize(
    ("first_scores", "second_scores", "first_probabilities", "message"),
    [
        (
            np.array([[1400.0, 1600.0]]),
            np.array([[1400.0]]),
            np.array([[0.25, 0.75]]),
            "Score half-run bounds must have matching shapes",
        ),
        (
            np.array([[1400.0, np.nan]]),
            np.array([[1400.0, 1600.0]]),
            np.array([[0.25, 0.75]]),
            "finite",
        ),
        (
            np.array([[1400.0, 1600.0]]),
            np.array([[1400.0, 1600.0]]),
            np.array([[0.25, np.inf]]),
            "finite",
        ),
    ],
)
def test_validate_half_run_stability_fails_closed(
    first_scores: np.ndarray,
    second_scores: np.ndarray,
    first_probabilities: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ranking_statistics.validate_half_run_stability(
            first_scores,
            second_scores,
            first_probabilities,
            np.array([[0.25, 0.75]]),
        )


def _production_shaped_mc_samples(
    replicates: int = 1_000,
) -> tuple[np.ndarray, np.ndarray]:
    draw = np.linspace(-2.0, 2.0, replicates)
    scores = np.empty((replicates, 18, 3, 5), dtype=float)
    for scope_index in range(18):
        for analysis_index in range(3):
            for model_index in range(5):
                scale = 1.0 + scope_index / 20.0 + model_index / 10.0
                scores[:, scope_index, analysis_index, model_index] = (
                    1500.0
                    + 10.0 * scope_index
                    + model_index
                    + scale * (draw + 0.1 * draw**3)
                )
    scores[:, :, 1, :] += 1_000_000.0 * draw[:, None, None]
    scores[:, :, 2, :] -= 1_000_000.0 * draw[:, None, None]

    latent = np.empty((replicates, 18, 5), dtype=float)
    for scope_index in range(18):
        for model_index in range(5):
            latent[:, scope_index, model_index] = (
                0.2 * model_index
                + (0.05 + scope_index / 100.0) * draw
                * (model_index + 1)
            )
    pairs = [
        (row, column)
        for row in range(5)
        for column in range(5)
        if row != column
    ]
    probabilities = np.empty((replicates, 18, 20), dtype=float)
    for pair_index, (row, column) in enumerate(pairs):
        difference = latent[:, :, row] - latent[:, :, column]
        probabilities[:, :, pair_index] = 1.0 / (1.0 + np.exp(-difference))
    return scores, probabilities


def test_monte_carlo_precision_uses_exact_displayed_endpoint_family() -> None:
    scores, probabilities = _production_shaped_mc_samples()
    diagnostics = ranking_statistics.monte_carlo_precision_diagnostics(
        scores,
        probabilities,
    )

    assert diagnostics["Method"] == "binomial_order_statistic"
    assert diagnostics["Replicates"] == 1_000
    assert diagnostics["TailProbability"] == 0.025
    assert diagnostics["PercentileProbabilityStandardError"] == pytest.approx(
        np.sqrt(0.025 * 0.975 / 1_000)
    )
    assert diagnostics["TailProbabilityRelativeStandardError"] == pytest.approx(
        np.sqrt(0.025 * 0.975 / 1_000) / 0.025
    )
    assert diagnostics["EndpointCounts"] == {
        "CrossedScore": 180,
        "NonredundantPairwiseProbability": 360,
        "Total": 540,
    }
    assert diagnostics["PairwiseDirectionRule"] == (
        "canonical_model_index_row_less_than_column"
    )
    assert diagnostics["Pointwise95"]["RankBrackets"] == {
        "CI95Lower": [16, 36],
        "CI95Upper": [965, 985],
    }
    assert diagnostics["BonferroniFamilywise95"]["RankBrackets"] == {
        "CI95Lower": [8, 47],
        "CI95Upper": [954, 993],
    }
    assert diagnostics["Pointwise95"]["EndpointAlpha"] == 0.05
    assert diagnostics["BonferroniFamilywise95"][
        "EndpointAlpha"
    ] == pytest.approx(0.05 / 540)

    for family in ("Pointwise95", "BonferroniFamilywise95"):
        for metric_family in ("Score", "PairwiseProbability"):
            metrics = diagnostics[family][metric_family]
            assert set(metrics) == {
                "MaximumBracketDistance",
                "MaximumRelativeCIWidth",
                "UndefinedRelativeCIWidthCount",
            }
            distance = metrics["MaximumBracketDistance"]
            relative = metrics["MaximumRelativeCIWidth"]
            assert distance["Value"] >= 0.0
            assert relative["Value"] >= 0.0
            assert distance["Bound"] in {"lower", "upper"}
            assert relative["Bound"] in {"lower", "upper"}
            assert distance["Scope"] in {
                scope.scope
                for scope in ranking_statistics.ranking_scope_registry()
            }
            if metric_family == "Score":
                assert distance["Analysis"] == "crossed_expert_gene"
                assert distance["Model"] in MODEL_COLUMNS
            else:
                assert MODEL_COLUMNS.index(distance["RowModel"]) < (
                    MODEL_COLUMNS.index(distance["ColumnModel"])
                )

    serialized = str(diagnostics).casefold()
    assert "threshold" not in serialized
    assert "passed" not in serialized


def test_monte_carlo_precision_ignores_one_way_sensitivity_noise() -> None:
    scores, probabilities = _production_shaped_mc_samples()
    baseline = scores.copy()
    baseline[:, :, 1:, :] = baseline[:, :, :1, :]
    assert ranking_statistics.monte_carlo_precision_diagnostics(
        scores,
        probabilities,
    ) == ranking_statistics.monte_carlo_precision_diagnostics(
        baseline,
        probabilities,
    )


def test_monte_carlo_precision_deduplicates_reciprocal_pairs() -> None:
    scores, probabilities = _production_shaped_mc_samples()
    pairs = [
        (row, column)
        for row in range(5)
        for column in range(5)
        if row != column
    ]
    for pair_index, (row, column) in enumerate(pairs):
        reverse_index = pairs.index((column, row))
        np.testing.assert_allclose(
            probabilities[:, :, pair_index]
            + probabilities[:, :, reverse_index],
            1.0,
            rtol=0.0,
            atol=1e-12,
        )
    diagnostics = ranking_statistics.monte_carlo_precision_diagnostics(
        scores,
        probabilities,
    )
    assert diagnostics["EndpointCounts"][
        "NonredundantPairwiseProbability"
    ] == 18 * 10 * 2


def test_monte_carlo_precision_fails_closed_on_invalid_arrays() -> None:
    scores, probabilities = _production_shaped_mc_samples()
    with pytest.raises(ValueError, match="score samples.*shape"):
        ranking_statistics.monte_carlo_precision_diagnostics(
            scores[:, :, :, :4],
            probabilities,
        )
    with pytest.raises(ValueError, match="same number of replicates"):
        ranking_statistics.monte_carlo_precision_diagnostics(
            scores,
            probabilities[:-1],
        )

    nonfinite_scores = scores.copy()
    nonfinite_scores[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ranking_statistics.monte_carlo_precision_diagnostics(
            nonfinite_scores,
            probabilities,
        )

    nonreciprocal = probabilities.copy()
    nonreciprocal[:, 0, 0] *= 0.9
    with pytest.raises(ValueError, match="reciprocal"):
        ranking_statistics.monte_carlo_precision_diagnostics(
            scores,
            nonreciprocal,
        )


def test_crossed_bootstrap_attaches_nonblocking_production_qc(
    crossed_fixture: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ranking_statistics, "PL_PRODUCTION_REPLICATES", 2)
    outputs = ranking_statistics.bootstrap_plackett_luce_statistics(
        crossed_fixture,
        MODEL_COLUMNS,
        BootstrapConfig(successful_replicates=2, max_failed_fits=0),
    )
    for output in outputs.values():
        diagnostics = output.attrs["bootstrap_diagnostics"]
        assert diagnostics["HalfRunStability"]["Applied"] is True
        assert diagnostics["HalfRunStability"]["Interpretation"] == (
            "descriptive_nonblocking"
        )
        precision = diagnostics["MonteCarloPrecision"]
        assert precision["Applied"] is True
        assert precision["EndpointCounts"] == {
            "CrossedScore": 180,
            "NonredundantPairwiseProbability": 360,
            "Total": 540,
        }
        assert "threshold" not in str(precision).casefold()
        assert "passed" not in str(precision).casefold()
