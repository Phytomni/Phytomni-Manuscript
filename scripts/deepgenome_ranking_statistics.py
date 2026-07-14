from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


MODEL_COLUMNS = ("Gemini", "Grok", "OpenAI", "Phytomni", "Claude")
AGREEMENT_SPECIES = ("Rice", "Maize", "Wheat", "Soybean", "Arabidopsis")
AGREEMENT_STUDY_STATUSES = ("well_studied", "uncharacterized")
REFERENCE_MODEL = "Gemini"
OPTIMIZER_OPTIONS = {"ftol": 1e-10, "gtol": 1e-8, "maxiter": 1000}
HESSIAN_EPSILON = 1e-5
ELO_SCALE = 400.0 / np.log(10.0)
ELO_CENTER = 1500.0
CI_Z = 1.96
FLEISS_COLUMNS = (
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
)


@dataclass(frozen=True)
class AgreementScope:
    scope_id: str
    analysis_tier: str
    scope_family: str
    species: str = "all"
    study_status: str = "all"
    model: str = "all"


def agreement_scope_registry(
    species: tuple[str, ...],
    study_statuses: tuple[str, ...],
    models: tuple[str, ...],
) -> tuple[AgreementScope, ...]:
    if (
        species != AGREEMENT_SPECIES
        or study_statuses != AGREEMENT_STUDY_STATUSES
        or models != MODEL_COLUMNS
    ):
        raise ValueError(
            "Locked agreement analysis requires canonical species, status, "
            "and model values in their fixed order."
        )

    scopes = [AgreementScope("overall", "primary", "overall")]
    scopes.extend(
        AgreementScope(
            f"species.{value.casefold()}",
            "locked_secondary",
            "species",
            species=value,
        )
        for value in species
    )
    scopes.extend(
        AgreementScope(
            f"study_status.{value.casefold()}",
            "locked_secondary",
            "study_status",
            study_status=value,
        )
        for value in study_statuses
    )
    scopes.extend(
        AgreementScope(
            f"model.{value.casefold()}",
            "locked_secondary",
            "model",
            model=value,
        )
        for value in models
    )
    scopes.extend(
        AgreementScope(
            f"species_study_status.{species_value.casefold()}."
            f"{status_value.casefold()}",
            "locked_exploratory",
            "species_study_status",
            species=species_value,
            study_status=status_value,
        )
        for species_value in species
        for status_value in study_statuses
    )
    scopes.extend(
        AgreementScope(
            f"model_study_status.{model_value.casefold()}."
            f"{status_value.casefold()}",
            "locked_exploratory",
            "model_study_status",
            study_status=status_value,
            model=model_value,
        )
        for model_value in models
        for status_value in study_statuses
    )
    scopes.extend(
        AgreementScope(
            f"model_species.{model_value.casefold()}."
            f"{species_value.casefold()}",
            "locked_exploratory",
            "model_species",
            species=species_value,
            model=model_value,
        )
        for model_value in models
        for species_value in species
    )
    return tuple(scopes)


def fleiss_kappa_from_counts(counts: np.ndarray) -> dict[str, object]:
    matrix = np.asarray(counts, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError(
            "Fleiss counts require at least two items and categories."
        )
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Fleiss counts must be finite and non-negative.")
    if not np.equal(matrix, np.floor(matrix)).all():
        raise ValueError("Fleiss counts must be integer-valued.")

    ratings = matrix.sum(axis=1)
    if not np.equal(ratings, ratings[0]).all():
        raise ValueError("Every item must have the same number of ratings.")
    if ratings[0] < 2:
        raise ValueError("Every item requires at least two ratings.")

    ratings_per_item = float(ratings[0])
    item_agreement = (matrix * (matrix - 1.0)).sum(axis=1) / (
        ratings_per_item * (ratings_per_item - 1.0)
    )
    marginals = matrix.sum(axis=0) / matrix.sum()
    observed = float(item_agreement.mean())
    expected = float(np.square(marginals).sum())
    if expected >= 1.0:
        raise ValueError("Fleiss expected agreement must be less than one.")
    return {
        "observed_agreement": observed,
        "expected_agreement": expected,
        "fleiss_kappa": float((observed - expected) / (1.0 - expected)),
        "rank_marginals": marginals,
    }


def _scope_frame(frame: pd.DataFrame, scope: AgreementScope) -> pd.DataFrame:
    selected = frame
    if scope.species != "all":
        selected = selected.loc[selected["Species"] == scope.species]
    if scope.study_status != "all":
        selected = selected.loc[
            selected["StudyStatus"] == scope.study_status
        ]
    return selected


def _validate_agreement_registry(
    registry: tuple[AgreementScope, ...],
) -> None:
    allowed_tiers = {"primary", "locked_secondary", "locked_exploratory"}
    allowed_families = {
        "overall",
        "species",
        "study_status",
        "model",
        "species_study_status",
        "model_study_status",
        "model_species",
    }
    for scope in registry:
        restricted = sum(
            value != "all"
            for value in (scope.species, scope.study_status, scope.model)
        )
        if restricted == 3:
            raise ValueError(
                "Fleiss agreement forbids scopes with three restricted "
                "dimensions."
            )
        if scope.analysis_tier not in allowed_tiers:
            raise ValueError(
                f"Unknown agreement analysis tier: {scope.analysis_tier!r}."
            )
        if scope.scope_family not in allowed_families:
            raise ValueError(
                f"Unknown agreement scope family: {scope.scope_family!r}."
            )

    canonical = agreement_scope_registry(
        AGREEMENT_SPECIES,
        AGREEMENT_STUDY_STATUSES,
        MODEL_COLUMNS,
    )
    if registry != canonical:
        raise ValueError(
            "Fleiss point estimates require the exact canonical 58-scope "
            "registry."
        )


def fleiss_point_estimates(
    frame: pd.DataFrame,
    registry: tuple[AgreementScope, ...],
    model_columns: tuple[str, ...] = MODEL_COLUMNS,
) -> pd.DataFrame:
    _validate_agreement_registry(registry)
    if model_columns != MODEL_COLUMNS:
        raise ValueError(
            "Fleiss point estimates require the canonical model columns."
        )
    required = {
        "Species",
        "Gene",
        "Expert",
        "StudyStatus",
        *model_columns,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Missing required agreement columns: " + ", ".join(missing)
        )
    ranks = tuple(f"R{rank}" for rank in range(1, 6))
    expected_ranks = set(ranks)
    for row in frame.loc[:, model_columns].itertuples(index=False, name=None):
        if set(row) != expected_ranks or len(row) != len(set(row)):
            raise ValueError(
                "Every agreement row must contain one complete R1-R5 ranking."
            )

    records: list[dict[str, object]] = []
    for scope in registry:
        selected = _scope_frame(frame, scope)
        if selected.empty:
            raise ValueError(f"Agreement scope {scope.scope_id!r} is empty.")
        selected_models = (
            model_columns if scope.model == "all" else (scope.model,)
        )
        unknown_models = set(selected_models).difference(model_columns)
        if unknown_models:
            raise ValueError(
                "Agreement scope references unknown models: "
                + ", ".join(sorted(unknown_models))
            )

        long = selected.melt(
            id_vars=["Species", "Gene", "Expert", "StudyStatus"],
            value_vars=list(selected_models),
            var_name="Model",
            value_name="Rank",
        )
        rating_key = ["Species", "Gene", "Model", "Expert"]
        if long.duplicated(rating_key).any():
            raise ValueError(
                f"Agreement scope {scope.scope_id!r} contains duplicate ratings."
            )
        item_key = ["Species", "Gene", "Model"]
        counts = (
            long.groupby(item_key + ["Rank"], sort=True)
            .size()
            .unstack("Rank", fill_value=0)
            .reindex(columns=ranks, fill_value=0)
        )
        agreement = fleiss_kappa_from_counts(counts.to_numpy())
        marginals = np.asarray(agreement["rank_marginals"], dtype=float)
        records.append(
            {
                "ScopeID": scope.scope_id,
                "AnalysisTier": scope.analysis_tier,
                "ScopeFamily": scope.scope_family,
                "Species": scope.species,
                "StudyStatus": scope.study_status,
                "Model": scope.model,
                "NGenes": int(
                    selected[["Species", "Gene"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "NItems": int(counts.shape[0]),
                "RatingsPerItem": int(counts.sum(axis=1).iloc[0]),
                "NRatings": int(counts.to_numpy().sum()),
                "NContributingExperts": int(long["Expert"].nunique()),
                "ObservedAgreement": agreement["observed_agreement"],
                "ExpectedAgreement": agreement["expected_agreement"],
                "FleissKappa": agreement["fleiss_kappa"],
                **{
                    f"RankR{rank}Share": float(marginals[rank - 1])
                    for rank in range(1, 6)
                },
            }
        )
    result = pd.DataFrame.from_records(records, columns=FLEISS_COLUMNS)
    if result["ScopeID"].duplicated().any():
        raise ValueError("Agreement registry contains duplicate scope IDs.")
    return result


def resolve_model_columns(setting: str | None) -> tuple[str, ...]:
    if setting is None or not setting.strip():
        return MODEL_COLUMNS
    model_columns = tuple(
        column.strip() for column in setting.split(",") if column.strip()
    )
    if len(model_columns) < 2:
        raise ValueError("Model configuration requires at least two columns.")
    if len(set(model_columns)) != len(model_columns):
        raise ValueError("Model configuration contains duplicate column names.")
    if REFERENCE_MODEL not in model_columns:
        raise ValueError(
            "Configured columns must include reference model "
            f"{REFERENCE_MODEL!r}."
        )
    return model_columns


def parse_rankings(
    frame: pd.DataFrame,
    model_columns: tuple[str, ...] = MODEL_COLUMNS,
) -> tuple[list[list[str]], int]:
    missing = [column for column in model_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required model columns: {', '.join(missing)}")
    rankings: list[list[str]] = []
    skipped = 0
    for _, row in frame.iterrows():
        rank_map: dict[str, int] = {}
        for model in model_columns:
            value = row[model]
            if isinstance(value, str) and value.startswith("R"):
                try:
                    rank_map[model] = int(value[1:])
                except ValueError:
                    continue
        if len(rank_map) != len(model_columns):
            skipped += 1
            continue
        ordered = sorted(rank_map.items(), key=lambda item: item[1])
        rankings.append([model for model, _ in ordered])
    return rankings, skipped


def normalize_weights(size: int, weights: np.ndarray | None) -> np.ndarray:
    result = np.ones(size) if weights is None else np.asarray(weights, dtype=float)
    if result.shape != (size,):
        raise ValueError("Ranking weights must have one value per ranking.")
    if not np.isfinite(result).all() or (result < 0).any() or result.sum() <= 0:
        raise ValueError(
            "Ranking weights must be finite, non-negative, and non-zero."
        )
    return result


def collapse_weighted_rankings(
    rankings: list[list[str]],
    weights: np.ndarray | None = None,
) -> tuple[list[list[str]], np.ndarray]:
    observation_weights = normalize_weights(len(rankings), weights)
    totals: dict[tuple[str, ...], float] = {}
    for ranking, weight in zip(rankings, observation_weights, strict=True):
        key = tuple(ranking)
        totals[key] = totals.get(key, 0.0) + float(weight)
    collapsed = [ranking for ranking, weight in totals.items() if weight > 0]
    collapsed_weights = np.array(
        [totals[ranking] for ranking in collapsed],
        dtype=float,
    )
    return [list(ranking) for ranking in collapsed], collapsed_weights


def pl_loglik_and_grad(
    xi: dict[str, float],
    rankings: list[list[str]],
    models: list[str],
    reference_model: str,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    observation_weights = normalize_weights(len(rankings), weights)
    free_models = [model for model in models if model != reference_model]
    index = {model: position for position, model in enumerate(free_models)}
    theta = {model: np.exp(xi[model]) for model in models}

    log_likelihood = 0.0
    gradient = np.zeros(len(free_models), dtype=float)

    for ranking, weight in zip(rankings, observation_weights, strict=True):
        remaining = ranking[:]
        for selected_model in ranking[:-1]:
            denominator = sum(theta[model] for model in remaining)
            log_likelihood += weight * (
                np.log(theta[selected_model]) - np.log(denominator)
            )
            for model in remaining:
                if model != reference_model:
                    gradient[index[model]] += weight * (
                        float(model == selected_model)
                        - theta[model] / denominator
                    )
            remaining = remaining[1:]

    return log_likelihood, gradient


def pack_xi_vector(
    xi: dict[str, float],
    models: list[str],
    reference_model: str,
) -> np.ndarray:
    free_models = [model for model in models if model != reference_model]
    return np.array([xi[model] for model in free_models], dtype=float)


def unpack_xi_vector(
    vector: np.ndarray,
    models: list[str],
    reference_model: str,
) -> dict[str, float]:
    free_models = [model for model in models if model != reference_model]
    return {
        model: (
            0.0
            if model == reference_model
            else float(vector[free_models.index(model)])
        )
        for model in models
    }


def negative_pl_objective(
    vector: np.ndarray,
    rankings: list[list[str]],
    models: list[str],
    reference_model: str,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    xi = unpack_xi_vector(vector, models, reference_model)
    log_likelihood, gradient = pl_loglik_and_grad(
        xi,
        rankings,
        models,
        reference_model,
        weights,
    )
    return -log_likelihood, -gradient


def central_difference_hessian(
    function,
    vector: np.ndarray,
    epsilon: float = HESSIAN_EPSILON,
) -> np.ndarray:
    size = len(vector)
    hessian = np.zeros((size, size))
    for index in range(size):
        plus = vector.copy()
        plus[index] += epsilon
        _, gradient_plus = function(plus)
        minus = vector.copy()
        minus[index] -= epsilon
        _, gradient_minus = function(minus)
        hessian[:, index] = (gradient_plus - gradient_minus) / (2.0 * epsilon)
    return hessian


def covariance_from_hessian(
    hessian: np.ndarray,
    models: list[str],
    reference_model: str,
) -> np.ndarray:
    try:
        free_covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        free_covariance = np.linalg.pinv(hessian)
    covariance = np.zeros((len(models), len(models)))
    free_models = [model for model in models if model != reference_model]
    full_index = {model: index for index, model in enumerate(models)}
    for row, row_model in enumerate(free_models):
        for column, column_model in enumerate(free_models):
            covariance[
                full_index[row_model], full_index[column_model]
            ] = free_covariance[row, column]
    return covariance


def elo_from_xi(
    xi: dict[str, float],
    models: list[str],
) -> np.ndarray:
    xi_vector = np.array([xi[model] for model in models])
    elo_raw = ELO_SCALE * xi_vector
    offset = ELO_CENTER - np.mean(elo_raw)
    return elo_raw + offset


def elo_confidence_intervals(
    elo: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elo_standard_error = ELO_SCALE * np.sqrt(np.diag(covariance))
    elo_lower = elo - CI_Z * elo_standard_error
    elo_upper = elo + CI_Z * elo_standard_error
    return elo_standard_error, elo_lower, elo_upper


def pairwise_win_probabilities(
    xi: dict[str, float],
    models: list[str],
) -> np.ndarray:
    probabilities = np.zeros((len(models), len(models)))
    for row, row_model in enumerate(models):
        for column, column_model in enumerate(models):
            if row == column:
                probabilities[row, column] = np.nan
            else:
                probabilities[row, column] = 1.0 / (
                    1.0 + np.exp(xi[column_model] - xi[row_model])
                )
    return probabilities


def fit_plackett_luce(
    rankings: list[list[str]],
    models: list[str],
    weights: np.ndarray | None = None,
) -> dict[str, object]:
    models = sorted(models)
    reference_model = REFERENCE_MODEL
    if reference_model not in models:
        raise ValueError(
            f"Reference model {reference_model!r} is absent from the model list."
        )
    if not rankings:
        raise ValueError("At least one complete ranking is required.")
    observation_weights = normalize_weights(len(rankings), weights)

    initial_xi = {model: 0.0 for model in models}
    initial_vector = pack_xi_vector(
        initial_xi,
        models,
        reference_model,
    )

    def objective(vector: np.ndarray) -> tuple[float, np.ndarray]:
        return negative_pl_objective(
            vector,
            rankings,
            models,
            reference_model,
            observation_weights,
        )

    result = minimize(
        lambda vector: objective(vector)[0],
        initial_vector,
        jac=lambda vector: objective(vector)[1],
        method="L-BFGS-B",
        options=OPTIMIZER_OPTIONS,
    )
    xi_hat = unpack_xi_vector(result.x, models, reference_model)
    hessian = central_difference_hessian(
        objective,
        result.x,
        HESSIAN_EPSILON,
    )
    covariance = covariance_from_hessian(
        hessian,
        models,
        reference_model,
    )
    elo = elo_from_xi(xi_hat, models)
    elo_standard_error, elo_lower, elo_upper = elo_confidence_intervals(
        elo,
        covariance,
    )
    pairwise_probabilities = pairwise_win_probabilities(xi_hat, models)

    return {
        "models": models,
        "reference_model": reference_model,
        "optimizer_result": result,
        "negative_log_likelihood": float(result.fun),
        "xi": xi_hat,
        "covariance": covariance,
        "elo": elo,
        "elo_standard_error": elo_standard_error,
        "elo_lower": elo_lower,
        "elo_upper": elo_upper,
        "pairwise_probabilities": pairwise_probabilities,
    }


def elo_outputs(
    fit: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = fit["models"]
    score_table = pd.DataFrame(
        {
            "Model": models,
            "Elo": fit["elo"],
            "Elo_L": fit["elo_lower"],
            "Elo_U": fit["elo_upper"],
        }
    ).sort_values("Elo", ascending=False, ignore_index=True)
    probability_table = pd.DataFrame(
        fit["pairwise_probabilities"],
        index=models,
        columns=models,
    )
    probability_table.index.name = "Model"
    return score_table, probability_table
