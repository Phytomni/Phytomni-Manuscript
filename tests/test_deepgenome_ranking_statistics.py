from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from scripts.deepgenome_ranking_statistics import (
    MODEL_COLUMNS,
    collapse_weighted_rankings,
    elo_outputs,
    fit_plackett_luce,
    parse_rankings,
    pl_loglik_and_grad,
)


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
        atol=1e-6,
    )
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

    assert np.isclose(log_likelihood, -95.34161491043835, atol=1e-12)
    np.testing.assert_allclose(gradient, [-0.5, 2.5, 4.5], atol=1e-12)


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
