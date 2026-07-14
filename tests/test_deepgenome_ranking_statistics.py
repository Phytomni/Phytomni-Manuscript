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
