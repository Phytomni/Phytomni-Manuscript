from collections import Counter
from dataclasses import FrozenInstanceError
from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from scripts.deepgenome_ranking_statistics import (
    FLEISS_COLUMNS,
    MODEL_COLUMNS,
    AgreementScope,
    agreement_scope_registry,
    collapse_weighted_rankings,
    elo_outputs,
    fit_plackett_luce,
    fleiss_kappa_from_counts,
    fleiss_point_estimates,
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
