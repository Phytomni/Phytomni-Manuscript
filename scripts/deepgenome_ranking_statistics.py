from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import kendalltau

from scripts.release_deepgenome_rankings import (
    MULTISELECT_DIMENSIONS,
    audit_panel_category_map,
)


MODULE_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
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
GENE_ORDINAL_COLUMNS = (
    "Species",
    "Gene",
    "StudyStatus",
    "NExperts",
    "NModels",
    "KendallW",
    "MeanPairwiseKendallTau",
    "Top1AgreementPattern",
)
BOOTSTRAP_METADATA_COLUMNS = (
    "BootstrapAttempted",
    "BootstrapReplicates",
    "BootstrapInvalid",
    "BootstrapUnit",
    "BootstrapStrata",
    "SeedStream",
)
FLEISS_BOOTSTRAP_COLUMNS = (
    *FLEISS_COLUMNS,
    "CILower",
    "CIUpper",
    *BOOTSTRAP_METADATA_COLUMNS,
)
ORDINAL_BOOTSTRAP_COLUMNS = (
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
    *BOOTSTRAP_METADATA_COLUMNS,
)
TOP1_BOOTSTRAP_COLUMNS = (
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
    *BOOTSTRAP_METADATA_COLUMNS,
)
TOP1_PATTERNS = ("unanimous", "majority_2_of_3", "all_different")
AGREEMENT_SEED_STREAM = "agreement_gene_blocks"
BOOTSTRAP_STRATA_LABEL = "Species x StudyStatus"
PL_INTERVAL_ANALYSES = (
    "crossed_expert_gene",
    "expert_cluster",
    "gene_cluster",
)
PL_BOOTSTRAP_SEED_STREAM = "pl_expert_gene_components"
PL_PRODUCTION_REPLICATES = 10_000
PL_SCORE_COLUMNS = (
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
PL_RANK_COLUMNS = (
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
PL_PAIRWISE_COLUMNS = (
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
PANEL_SUMMARY_COLUMNS = (
    "Dimension",
    "PublicCategory",
    "DisplayOrder",
    "N",
    "DenominatorN",
    "Percent",
    "MissingN",
    "PercentageBasis",
)
ASSIGNMENT_SUMMARY_COLUMNS = (
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
)


@dataclass(frozen=True)
class AgreementScope:
    scope_id: str
    analysis_tier: str
    scope_family: str
    species: str = "all"
    study_status: str = "all"
    model: str = "all"


@dataclass(frozen=True)
class RankingScope:
    scope: str
    study_status: str
    species: str


@dataclass(frozen=True)
class BootstrapConfig:
    successful_replicates: int = 10_000
    seed: int = 20260714
    max_failed_fits: int = 10

    def __post_init__(self) -> None:
        values = (
            ("successful_replicates", self.successful_replicates, 1),
            ("seed", self.seed, 0),
            ("max_failed_fits", self.max_failed_fits, 0),
        )
        for name, value, minimum in values:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(
                    f"{name} must be an integer greater than or equal to "
                    f"{minimum}."
                )


def gene_bootstrap_multiplicities(
    genes: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.Series:
    required = {"Species", "StudyStatus"}
    missing = sorted(required.difference(genes.columns))
    if missing:
        raise ValueError(
            "Missing gene bootstrap strata columns: " + ", ".join(missing)
        )
    if genes.empty:
        raise ValueError("Gene bootstrap requires at least one gene.")
    if genes[["Species", "StudyStatus"]].isna().any().any():
        raise ValueError("Gene bootstrap strata must be nonmissing.")

    multiplicities = np.zeros(len(genes), dtype=int)
    groups = genes.groupby(
        ["Species", "StudyStatus"],
        sort=False,
        dropna=False,
    ).indices
    for positions in groups.values():
        positions = np.asarray(positions, dtype=int)
        sampled = rng.choice(positions, size=len(positions), replace=True)
        multiplicities += np.bincount(sampled, minlength=len(genes))
    return pd.Series(multiplicities, index=genes.index, dtype=int)


def sample_within_strata(
    units: pd.DataFrame,
    *,
    id_columns: tuple[str, ...],
    strata: tuple[str, ...],
    rng: np.random.Generator,
) -> pd.Series:
    """Draw cluster units with replacement within fixed strata."""
    if not id_columns or not strata:
        raise ValueError("Bootstrap IDs and strata must both be nonempty.")
    key_columns = tuple(dict.fromkeys((*strata, *id_columns)))
    missing = sorted(set(key_columns).difference(units.columns))
    if missing:
        raise ValueError(
            "Missing cluster bootstrap columns: " + ", ".join(missing)
        )
    if units.empty:
        raise ValueError("Cluster bootstrap requires at least one unit.")
    if units.loc[:, key_columns].isna().any().any():
        raise ValueError("Cluster bootstrap IDs and strata must be nonmissing.")
    if units.duplicated(list(key_columns)).any():
        raise ValueError(
            "Cluster bootstrap units must be unique within each stratum."
        )

    multiplicities = np.zeros(len(units), dtype=int)
    grouped_positions = units.groupby(
        list(strata),
        sort=False,
        dropna=False,
        observed=True,
    ).indices
    for positions in grouped_positions.values():
        positions = np.asarray(positions, dtype=int)
        sampled = rng.choice(positions, size=len(positions), replace=True)
        multiplicities += np.bincount(sampled, minlength=len(units))
    return pd.Series(multiplicities, index=units.index, dtype=int)


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


def ordinal_scope_registry(
    species: tuple[str, ...] = AGREEMENT_SPECIES,
    study_statuses: tuple[str, ...] = AGREEMENT_STUDY_STATUSES,
) -> tuple[AgreementScope, ...]:
    if species != AGREEMENT_SPECIES or study_statuses != AGREEMENT_STUDY_STATUSES:
        raise ValueError(
            "Locked ordinal analysis requires canonical species and status "
            "values in their fixed order."
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
    return tuple(scopes)


def ranking_scope_registry() -> tuple[RankingScope, ...]:
    """Return the frozen ranking scopes in their publication order."""
    scopes = [RankingScope("overall", "all", "all")]
    for status in AGREEMENT_STUDY_STATUSES:
        scopes.append(RankingScope(status, status, "all"))
        scopes.extend(
            RankingScope(
                f"{status}.{species.casefold().replace(' ', '_')}",
                status,
                species,
            )
            for species in AGREEMENT_SPECIES
        )
    scopes.extend(
        RankingScope(
            species.casefold().replace(" ", "_"),
            "all",
            species,
        )
        for species in AGREEMENT_SPECIES
    )
    return tuple(scopes)


def _is_missing_metadata_value(value: object) -> bool:
    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def summarize_expert_panel(
    metadata: pd.DataFrame,
    category_map: pd.DataFrame,
    *,
    expert_column: str = "Expert_ID",
    minimum_count: int = 5,
) -> pd.DataFrame:
    """Return privacy-reviewed panel composition aggregates."""
    if expert_column not in metadata.columns:
        raise ValueError("The expert metadata table is missing its expert column.")
    if metadata[expert_column].isna().any():
        raise ValueError("Expert identifiers must not be missing.")
    if metadata[expert_column].duplicated().any():
        raise ValueError("Expert metadata must contain one row per expert.")

    audited = audit_panel_category_map(
        metadata,
        category_map,
        expert_column=expert_column,
        minimum_count=minimum_count,
    )
    total_experts = int(metadata[expert_column].nunique())
    dimensions = category_map["Dimension"].drop_duplicates().tolist()
    public_order = (
        category_map[
            ["Dimension", "PublicCategory", "DisplayOrder"]
        ]
        .drop_duplicates(["Dimension", "PublicCategory"])
        .set_index(["Dimension", "PublicCategory"])["DisplayOrder"]
        .astype(int)
    )
    country_map = category_map.loc[
        category_map["Dimension"] == "Country/Region"
    ]
    if set(country_map["SourceValue"]) & set(country_map["PublicCategory"]):
        raise ValueError(
            "Country/Region summaries must use aggregate public categories."
        )

    records: list[dict[str, object]] = []
    for dimension in dimensions:
        missing_count = sum(
            _is_missing_metadata_value(value)
            for value in metadata[dimension].tolist()
        )
        is_multiselect = dimension in MULTISELECT_DIMENSIONS
        denominator = (
            total_experts if is_multiselect else total_experts - missing_count
        )
        if denominator <= 0:
            raise ValueError(
                "Panel percentages require a positive expert denominator."
            )
        selected = audited.loc[audited["Dimension"] == dimension]
        if not is_multiselect and int(selected["N"].sum()) != denominator:
            raise ValueError(
                "Single-select panel categories must cover every nonmissing expert."
            )
        for row in selected.itertuples(index=False):
            records.append(
                {
                    "Dimension": dimension,
                    "PublicCategory": row.PublicCategory,
                    "DisplayOrder": int(
                        public_order.loc[(dimension, row.PublicCategory)]
                    ),
                    "N": int(row.N),
                    "DenominatorN": denominator,
                    "Percent": 100.0 * int(row.N) / denominator,
                    "MissingN": missing_count,
                    "PercentageBasis": (
                        "all_experts"
                        if is_multiselect
                        else "nonmissing_experts"
                    ),
                }
            )
    result = pd.DataFrame.from_records(records, columns=PANEL_SUMMARY_COLUMNS)
    if (result["N"] < minimum_count).any():
        raise ValueError("Every category must meet the minimum public group size.")
    return result


def summarize_assignments(
    frame: pd.DataFrame,
    model_columns: tuple[str, ...] = MODEL_COLUMNS,
) -> pd.DataFrame:
    """Summarize the crossed expert-gene assignment design without IDs."""
    working = _validate_pl_bootstrap_frame(frame, tuple(model_columns))
    records: list[dict[str, object]] = []
    for scope in ranking_scope_registry():
        selected = working
        if scope.study_status != "all":
            selected = selected.loc[
                selected["StudyStatus"] == scope.study_status
            ]
        if scope.species != "all":
            selected = selected.loc[selected["Species"] == scope.species]
        genes_per_expert = selected.groupby(
            ["Species", "Expert"],
            sort=False,
        )["Gene"].nunique()
        experts_per_gene = selected.groupby(
            ["Species", "Gene"],
            sort=False,
        )["Expert"].nunique()
        records.append(
            {
                "Scope": scope.scope,
                "StudyStatus": scope.study_status,
                "Species": scope.species,
                "NExperts": int(
                    selected[["Species", "Expert"]].drop_duplicates().shape[0]
                ),
                "NGenes": int(
                    selected[["Species", "Gene"]].drop_duplicates().shape[0]
                ),
                "NJudgments": len(selected),
                "MinGenesPerExpert": int(genes_per_expert.min()),
                "MaxGenesPerExpert": int(genes_per_expert.max()),
                "MinExpertsPerGene": int(experts_per_gene.min()),
                "MaxExpertsPerGene": int(experts_per_gene.max()),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=ASSIGNMENT_SUMMARY_COLUMNS,
    )


def _validated_rank_matrix(rank_matrix: np.ndarray) -> np.ndarray:
    try:
        ranks = np.asarray(rank_matrix, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Each Kendall W row must be a complete no-tie ranking."
        ) from error
    if ranks.ndim != 2 or ranks.shape[0] < 2 or ranks.shape[1] < 2:
        raise ValueError("Kendall W requires at least two raters and items.")
    expected = np.arange(1, ranks.shape[1] + 1, dtype=float)
    if not all(np.array_equal(np.sort(row), expected) for row in ranks):
        raise ValueError(
            "Each Kendall W row must be a complete no-tie ranking."
        )
    return ranks


def kendall_w(rank_matrix: np.ndarray) -> float:
    ranks = _validated_rank_matrix(rank_matrix)
    n_raters, n_items = ranks.shape
    rank_sums = ranks.sum(axis=0)
    squared_deviation = np.square(rank_sums - rank_sums.mean()).sum()
    return float(
        12
        * squared_deviation
        / (n_raters**2 * (n_items**3 - n_items))
    )


def mean_pairwise_kendall_tau(rank_matrix: np.ndarray) -> float:
    ranks = _validated_rank_matrix(rank_matrix)
    pairwise_taus = [
        float(kendalltau(first, second).statistic)
        for first, second in combinations(ranks, 2)
    ]
    return float(np.mean(pairwise_taus))


def top1_pattern(top_models: Sequence[str]) -> str:
    if isinstance(top_models, (str, bytes)):
        raise ValueError("Top-1 agreement requires exactly three nonmissing labels.")
    try:
        labels = list(top_models)
    except TypeError as error:
        raise ValueError(
            "Top-1 agreement requires exactly three nonmissing labels."
        ) from error
    if len(labels) != 3 or pd.isna(labels).any():
        raise ValueError("Top-1 agreement requires exactly three nonmissing labels.")

    counts = pd.Series(labels, dtype=object).value_counts()
    if counts.iloc[0] == 3:
        return "unanimous"
    if counts.iloc[0] == 2:
        return "majority_2_of_3"
    return "all_different"


def gene_ordinal_agreement(
    frame: pd.DataFrame,
    model_columns: tuple[str, ...] = MODEL_COLUMNS,
) -> pd.DataFrame:
    model_columns = tuple(model_columns)
    if len(model_columns) != 5 or len(set(model_columns)) != 5:
        raise ValueError(
            "Gene ordinal agreement requires exactly five model columns."
        )

    identifier_columns = ("Species", "Gene", "Expert", "StudyStatus")
    required = {*identifier_columns, *model_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Missing required ordinal agreement columns: "
            + ", ".join(missing)
        )
    if frame.loc[:, identifier_columns].isna().any().any():
        raise ValueError(
            "Gene ordinal agreement identifiers must all be nonmissing."
        )
    item_columns = ["Species", "Gene"]
    if frame.duplicated([*item_columns, "Expert"]).any():
        raise ValueError(
            "Gene ordinal agreement contains duplicate expert/gene rows."
        )

    gene_metadata = frame.groupby(
        item_columns,
        sort=False,
        dropna=False,
    )[["StudyStatus"]].nunique(dropna=False)
    if (gene_metadata["StudyStatus"] != 1).any():
        raise ValueError(
            "Gene ordinal agreement contains mixed study status within a gene."
        )

    gene_experts = frame.groupby(
        item_columns,
        sort=False,
        dropna=False,
    )["Expert"].agg(["size", "nunique"])
    if (gene_experts["size"] != 3).any() or (
        gene_experts["nunique"] != 3
    ).any():
        raise ValueError(
            "Gene ordinal agreement requires exactly three experts per gene."
        )

    expected_ranks = {f"R{rank}" for rank in range(1, 6)}
    for row in frame.loc[:, model_columns].itertuples(index=False, name=None):
        if (
            any(pd.isna(value) for value in row)
            or len(set(row)) != 5
            or set(row) != expected_ranks
        ):
            raise ValueError(
                "Every expert/gene row must contain one complete no-tie "
                "R1-R5 ranking."
            )

    rank_values = {f"R{rank}": rank for rank in range(1, 6)}
    records: list[dict[str, object]] = []
    for (species, gene), selected in frame.groupby(
        item_columns,
        sort=True,
        dropna=False,
    ):
        rank_matrix = np.array(
            [
                [rank_values[value] for value in row]
                for row in selected.loc[:, model_columns].itertuples(
                    index=False,
                    name=None,
                )
            ],
            dtype=float,
        )
        top_models = [
            model_columns[row.index("R1")]
            for row in selected.loc[:, model_columns].itertuples(
                index=False,
                name=None,
            )
        ]
        records.append(
            {
                "Species": species,
                "Gene": gene,
                "StudyStatus": selected["StudyStatus"].iloc[0],
                "NExperts": int(selected["Expert"].nunique()),
                "NModels": len(model_columns),
                "KendallW": kendall_w(rank_matrix),
                "MeanPairwiseKendallTau": mean_pairwise_kendall_tau(
                    rank_matrix
                ),
                "Top1AgreementPattern": top1_pattern(top_models),
            }
        )

    return (
        pd.DataFrame.from_records(records, columns=GENE_ORDINAL_COLUMNS)
        .sort_values(["Species", "Gene", "StudyStatus"], kind="stable")
        .reset_index(drop=True)
    )


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


def _validate_agreement_bootstrap_frame(
    frame: pd.DataFrame,
    model_columns: tuple[str, ...],
) -> pd.DataFrame:
    if model_columns != MODEL_COLUMNS:
        raise ValueError(
            "Agreement bootstrap requires the canonical model columns."
        )
    identifiers = ("Species", "Gene", "Expert", "StudyStatus")
    required = {*identifiers, *model_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Missing required agreement bootstrap columns: "
            + ", ".join(missing)
        )
    if frame.loc[:, identifiers].isna().any().any():
        raise ValueError(
            "Agreement bootstrap identifiers must all be nonmissing."
        )
    if set(frame["Species"]) != set(AGREEMENT_SPECIES) or set(
        frame["StudyStatus"]
    ) != set(AGREEMENT_STUDY_STATUSES):
        raise ValueError(
            "Agreement bootstrap requires canonical species and study "
            "statuses."
        )
    observed_strata = set(
        frame[["Species", "StudyStatus"]].itertuples(index=False, name=None)
    )
    expected_strata = {
        (species, status)
        for species in AGREEMENT_SPECIES
        for status in AGREEMENT_STUDY_STATUSES
    }
    if observed_strata != expected_strata:
        raise ValueError(
            "Agreement bootstrap requires all 10 Species x StudyStatus "
            "strata."
        )
    return gene_ordinal_agreement(frame, model_columns)


def _gene_rank_count_array(
    frame: pd.DataFrame,
    genes: pd.DataFrame,
    model_columns: tuple[str, ...],
) -> np.ndarray:
    gene_index = pd.MultiIndex.from_frame(genes[["Species", "Gene"]])
    row_keys = pd.MultiIndex.from_frame(frame[["Species", "Gene"]])
    row_gene_indices = gene_index.get_indexer(row_keys)
    if (row_gene_indices < 0).any():
        raise ValueError("Agreement bootstrap could not index every gene block.")

    rank_lookup = {f"R{rank}": rank - 1 for rank in range(1, 6)}
    rank_indices = (
        frame.loc[:, model_columns]
        .replace(rank_lookup)
        .to_numpy(dtype=int)
    )
    counts = np.zeros((len(genes), len(model_columns), 5), dtype=int)
    np.add.at(
        counts,
        (
            np.repeat(row_gene_indices, len(model_columns)),
            np.tile(np.arange(len(model_columns)), len(frame)),
            rank_indices.ravel(),
        ),
        1,
    )
    return counts


def _scope_gene_indices(
    genes: pd.DataFrame,
    scope: AgreementScope,
) -> np.ndarray:
    mask = np.ones(len(genes), dtype=bool)
    if scope.species != "all":
        mask &= genes["Species"].to_numpy() == scope.species
    if scope.study_status != "all":
        mask &= genes["StudyStatus"].to_numpy() == scope.study_status
    return np.flatnonzero(mask)


def _fleiss_bootstrap_values(
    draws: np.ndarray,
    genes: pd.DataFrame,
    rank_counts: np.ndarray,
    registry: tuple[AgreementScope, ...],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.empty((len(draws), len(registry)), dtype=float)
    valid = np.ones(len(draws), dtype=bool)
    item_agreement = (
        rank_counts * (rank_counts - 1)
    ).sum(axis=2) / 6.0
    for scope_index, scope in enumerate(registry):
        gene_indices = _scope_gene_indices(genes, scope)
        model_indices = (
            np.arange(len(MODEL_COLUMNS))
            if scope.model == "all"
            else np.array([MODEL_COLUMNS.index(scope.model)])
        )
        weights = draws[:, gene_indices]
        item_count = weights.sum(axis=1) * len(model_indices)
        category_counts = rank_counts[gene_indices][:, model_indices].sum(
            axis=1
        )
        weighted_categories = weights @ category_counts
        marginals = weighted_categories / (3.0 * item_count[:, None])
        observed = (
            weights
            @ item_agreement[gene_indices][:, model_indices].sum(axis=1)
        ) / item_count
        expected = np.square(marginals).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            kappa = (observed - expected) / (1.0 - expected)
        values[:, scope_index] = kappa
        valid &= (
            (item_count > 0)
            & np.isfinite(kappa)
            & np.isfinite(expected)
            & (expected < 1.0)
        )
    return values, valid


def _weighted_medians(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[:, order]
    cumulative = np.cumsum(sorted_weights, axis=1)
    totals = cumulative[:, -1]
    lower_ranks = (totals - 1) // 2
    upper_ranks = totals // 2
    lower_indices = (cumulative > lower_ranks[:, None]).argmax(axis=1)
    upper_indices = (cumulative > upper_ranks[:, None]).argmax(axis=1)
    return (
        sorted_values[lower_indices] + sorted_values[upper_indices]
    ) / 2.0


def _bootstrap_metadata(
    attempted: int,
    config: BootstrapConfig,
) -> dict[str, object]:
    return {
        "BootstrapAttempted": attempted,
        "BootstrapReplicates": config.successful_replicates,
        "BootstrapInvalid": attempted - config.successful_replicates,
        "BootstrapUnit": "gene",
        "BootstrapStrata": BOOTSTRAP_STRATA_LABEL,
        "SeedStream": AGREEMENT_SEED_STREAM,
    }


def bootstrap_agreement(
    frame: pd.DataFrame,
    model_columns: tuple[str, ...] = MODEL_COLUMNS,
    config: BootstrapConfig | None = None,
) -> dict[str, pd.DataFrame]:
    model_columns = tuple(model_columns)
    config = config or BootstrapConfig()
    gene_rows = _validate_agreement_bootstrap_frame(frame, model_columns)
    genes = gene_rows.loc[:, ["Species", "Gene", "StudyStatus"]].copy()
    registry = agreement_scope_registry(
        AGREEMENT_SPECIES,
        AGREEMENT_STUDY_STATUSES,
        model_columns,
    )
    ordinal_registry = ordinal_scope_registry()
    point_fleiss = fleiss_point_estimates(frame, registry, model_columns)
    rank_counts = _gene_rank_count_array(frame, genes, model_columns)

    seed_sequence = np.random.SeedSequence(config.seed)
    rng = np.random.default_rng(seed_sequence.spawn(1)[0])
    candidate_draws = [
        gene_bootstrap_multiplicities(genes, rng).to_numpy(dtype=int)
        for _ in range(config.successful_replicates)
    ]
    draw_matrix = np.vstack(candidate_draws)
    kappa_matrix, valid = _fleiss_bootstrap_values(
        draw_matrix,
        genes,
        rank_counts,
        registry,
    )
    accepted_draws = [row for row in draw_matrix[valid]]
    accepted_kappas = [row for row in kappa_matrix[valid]]
    invalid = int((~valid).sum())
    if invalid > config.max_failed_fits:
        raise RuntimeError(
            "Agreement bootstrap exceeded max_failed_fits before reaching "
            "the requested successful replicates."
        )
    while len(accepted_draws) < config.successful_replicates:
        draw = gene_bootstrap_multiplicities(genes, rng).to_numpy(dtype=int)
        kappa, is_valid = _fleiss_bootstrap_values(
            draw[None, :],
            genes,
            rank_counts,
            registry,
        )
        if is_valid[0]:
            accepted_draws.append(draw)
            accepted_kappas.append(kappa[0])
        else:
            invalid += 1
            if invalid > config.max_failed_fits:
                raise RuntimeError(
                    "Agreement bootstrap exceeded max_failed_fits before "
                    "reaching the requested successful replicates."
                )

    draws = np.vstack(accepted_draws)
    kappas = np.vstack(accepted_kappas)
    attempted = config.successful_replicates + invalid
    metadata = _bootstrap_metadata(attempted, config)
    lower, upper = np.quantile(kappas, [0.025, 0.975], axis=0)
    fleiss = point_fleiss.assign(
        CILower=lower,
        CIUpper=upper,
        **metadata,
    ).loc[:, FLEISS_BOOTSTRAP_COLUMNS]

    ordinal_records: list[dict[str, object]] = []
    top1_records: list[dict[str, object]] = []
    for scope in ordinal_registry:
        gene_indices = _scope_gene_indices(genes, scope)
        selected = gene_rows.iloc[gene_indices]
        weights = draws[:, gene_indices]
        totals = weights.sum(axis=1)
        expert_count = int(
            _scope_frame(frame, scope)["Expert"].nunique()
        )
        ordinal_record: dict[str, object] = {
            "ScopeID": scope.scope_id,
            "AnalysisTier": scope.analysis_tier,
            "ScopeFamily": scope.scope_family,
            "Species": scope.species,
            "StudyStatus": scope.study_status,
            "NGenes": len(selected),
            "NContributingExperts": expert_count,
        }
        for source, prefix in (
            ("KendallW", "KendallW"),
            (
                "MeanPairwiseKendallTau",
                "MeanPairwiseKendallTau",
            ),
        ):
            point_values = selected[source].to_numpy(dtype=float)
            q1, median, q3 = np.quantile(point_values, [0.25, 0.5, 0.75])
            replicate_means = (weights @ point_values) / totals
            replicate_medians = _weighted_medians(point_values, weights)
            mean_lower, mean_upper = np.quantile(
                replicate_means, [0.025, 0.975]
            )
            median_lower, median_upper = np.quantile(
                replicate_medians, [0.025, 0.975]
            )
            ordinal_record.update(
                {
                    f"{prefix}Mean": float(point_values.mean()),
                    f"{prefix}Median": float(median),
                    f"{prefix}Q1": float(q1),
                    f"{prefix}Q3": float(q3),
                    f"{prefix}MeanCILower": float(mean_lower),
                    f"{prefix}MeanCIUpper": float(mean_upper),
                    f"{prefix}MedianCILower": float(median_lower),
                    f"{prefix}MedianCIUpper": float(median_upper),
                }
            )
        ordinal_record.update(metadata)
        ordinal_records.append(ordinal_record)

        observed_patterns = selected["Top1AgreementPattern"]
        for pattern in TOP1_PATTERNS:
            indicator = (
                observed_patterns.to_numpy(dtype=object) == pattern
            ).astype(float)
            replicate_fractions = (weights @ indicator) / totals
            fraction_lower, fraction_upper = np.quantile(
                replicate_fractions, [0.025, 0.975]
            )
            count = int(indicator.sum())
            top1_records.append(
                {
                    "ScopeID": scope.scope_id,
                    "AnalysisTier": scope.analysis_tier,
                    "ScopeFamily": scope.scope_family,
                    "Species": scope.species,
                    "StudyStatus": scope.study_status,
                    "Top1AgreementPattern": pattern,
                    "Count": count,
                    "Fraction": count / len(selected),
                    "FractionCILower": float(fraction_lower),
                    "FractionCIUpper": float(fraction_upper),
                    "NGenes": len(selected),
                    "NContributingExperts": expert_count,
                    **metadata,
                }
            )

    ordinal = pd.DataFrame.from_records(
        ordinal_records,
        columns=ORDINAL_BOOTSTRAP_COLUMNS,
    )
    top1 = pd.DataFrame.from_records(
        top1_records,
        columns=TOP1_BOOTSTRAP_COLUMNS,
    )
    return {
        "fleiss_kappa": fleiss,
        "ordinal_summary": ordinal,
        "top1_consensus": top1,
    }


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


def validate_half_run_stability(
    first_score_bounds: np.ndarray,
    second_score_bounds: np.ndarray,
    first_probability_bounds: np.ndarray,
    second_probability_bounds: np.ndarray,
) -> dict[str, float]:
    """Validate production-bootstrap bounds against the other half-run."""
    first_scores = np.asarray(first_score_bounds, dtype=float)
    second_scores = np.asarray(second_score_bounds, dtype=float)
    first_probabilities = np.asarray(first_probability_bounds, dtype=float)
    second_probabilities = np.asarray(second_probability_bounds, dtype=float)
    if first_scores.shape != second_scores.shape:
        raise ValueError("Score half-run bounds must have matching shapes.")
    if first_probabilities.shape != second_probabilities.shape:
        raise ValueError(
            "Probability half-run bounds must have matching shapes."
        )

    with np.errstate(invalid="ignore"):
        score_difference = float(
            np.max(np.abs(first_scores - second_scores))
        )
        probability_difference = float(
            np.max(
                np.abs(first_probabilities - second_probabilities)
            )
        )
    unstable: list[str] = []
    if not np.isfinite(score_difference) or score_difference >= 2.0:
        unstable.append("score bounds")
    if (
        not np.isfinite(probability_difference)
        or probability_difference >= 0.01
    ):
        unstable.append("probability bounds")
    if unstable:
        raise RuntimeError(
            "Unstable half-run bootstrap metrics: " + ", ".join(unstable)
        )
    return {
        "ScoreMaxBoundDifference": score_difference,
        "ProbabilityMaxBoundDifference": probability_difference,
    }


def _bootstrap_pl_fit(
    rankings: np.ndarray,
    weights: np.ndarray,
    models: tuple[str, ...],
    initial_vector: np.ndarray | None = None,
) -> dict[str, object]:
    """Fit a point-only PL model to collapsed, encoded rankings."""
    encoded = np.asarray(rankings, dtype=int)
    observation_weights = np.asarray(weights, dtype=float)
    if encoded.ndim != 2 or encoded.shape[1] != len(models):
        raise RuntimeError("Bootstrap rankings have an invalid shape.")
    if observation_weights.shape != (len(encoded),):
        raise RuntimeError("Bootstrap weights do not match rankings.")
    if (
        not np.isfinite(observation_weights).all()
        or (observation_weights < 0).any()
        or observation_weights.sum() <= 0
    ):
        raise RuntimeError("Bootstrap fit has invalid or zero weights.")

    unique_rankings, inverse = np.unique(
        encoded,
        axis=0,
        return_inverse=True,
    )
    collapsed_weights = np.bincount(
        inverse,
        weights=observation_weights,
        minlength=len(unique_rankings),
    )
    retained = collapsed_weights > 0
    encoded = unique_rankings[retained]
    observation_weights = collapsed_weights[retained]

    reference_index = models.index(REFERENCE_MODEL)
    free_indices = np.array(
        [index for index in range(len(models)) if index != reference_index],
        dtype=int,
    )
    if initial_vector is None:
        start = np.zeros(len(free_indices), dtype=float)
    else:
        start = np.asarray(initial_vector, dtype=float)
        if start.shape != (len(free_indices),) or not np.isfinite(start).all():
            raise RuntimeError("Bootstrap PL initial values are invalid.")

    def objective(vector: np.ndarray) -> tuple[float, np.ndarray]:
        xi = np.zeros(len(models), dtype=float)
        xi[free_indices] = vector
        log_likelihood = 0.0
        gradient = np.zeros(len(models), dtype=float)
        for stage in range(len(models) - 1):
            remaining = encoded[:, stage:]
            logits = xi[remaining]
            log_denominator = logsumexp(logits, axis=1)
            selected = encoded[:, stage]
            log_likelihood += float(
                observation_weights
                @ (xi[selected] - log_denominator)
            )
            np.add.at(gradient, selected, observation_weights)
            expected = observation_weights[:, None] * np.exp(
                logits - log_denominator[:, None]
            )
            np.add.at(
                gradient,
                remaining.ravel(),
                -expected.ravel(),
            )
        return -log_likelihood, -gradient[free_indices]

    result = minimize(
        objective,
        start,
        jac=True,
        method="L-BFGS-B",
        options=OPTIMIZER_OPTIONS,
    )
    if (
        not result.success
        or not np.isfinite(result.fun)
        or not np.isfinite(result.x).all()
    ):
        raise RuntimeError(
            "Bootstrap Plackett-Luce optimization failed: "
            f"{result.message}"
        )

    xi = np.zeros(len(models), dtype=float)
    xi[free_indices] = result.x
    elo = ELO_SCALE * xi
    elo += ELO_CENTER - float(elo.mean())
    probabilities = expit(xi[:, None] - xi[None, :])
    np.fill_diagonal(probabilities, np.nan)
    off_diagonal = ~np.eye(len(models), dtype=bool)
    if (
        not np.isfinite(elo).all()
        or not np.isfinite(probabilities[off_diagonal]).all()
    ):
        raise RuntimeError(
            "Bootstrap Plackett-Luce fit produced nonfinite outputs."
        )
    return {
        "optimizer_result": result,
        "elo": elo,
        "pairwise_probabilities": probabilities,
    }


def _validate_pl_bootstrap_frame(
    frame: pd.DataFrame,
    model_columns: tuple[str, ...],
) -> pd.DataFrame:
    _validate_agreement_bootstrap_frame(frame, model_columns)
    expert_species = frame.groupby(
        "Expert",
        sort=False,
        dropna=False,
    )["Species"].nunique(dropna=False)
    if (expert_species != 1).any():
        raise ValueError(
            "Crossed ranking bootstrap requires each expert to belong to "
            "exactly one species."
        )
    expert_assignments = frame.groupby(
        ["Species", "Expert"],
        sort=False,
        dropna=False,
    ).size()
    if not all(
        group.nunique() == 1
        for _, group in expert_assignments.groupby(level="Species")
    ):
        raise ValueError(
            "Crossed ranking bootstrap requires balanced expert assignments "
            "within every species."
        )

    species_order = {
        species: index for index, species in enumerate(AGREEMENT_SPECIES)
    }
    status_order = {
        status: index
        for index, status in enumerate(AGREEMENT_STUDY_STATUSES)
    }
    working = frame.copy()
    working["_SpeciesOrder"] = working["Species"].map(species_order)
    working["_StatusOrder"] = working["StudyStatus"].map(status_order)
    return (
        working.sort_values(
            [
                "_SpeciesOrder",
                "_StatusOrder",
                "Gene",
                "Expert",
            ],
            kind="stable",
        )
        .drop(columns=["_SpeciesOrder", "_StatusOrder"])
        .reset_index(drop=True)
    )


def _ranking_scope_indices(
    frame: pd.DataFrame,
    registry: tuple[RankingScope, ...],
) -> tuple[np.ndarray, ...]:
    species = frame["Species"].to_numpy(dtype=object)
    statuses = frame["StudyStatus"].to_numpy(dtype=object)
    result: list[np.ndarray] = []
    for scope in registry:
        mask = np.ones(len(frame), dtype=bool)
        if scope.study_status != "all":
            mask &= statuses == scope.study_status
        if scope.species != "all":
            mask &= species == scope.species
        indices = np.flatnonzero(mask)
        if not len(indices):
            raise ValueError(f"Ranking scope {scope.scope!r} is empty.")
        result.append(indices)
    return tuple(result)


def _scope_point_statistics(
    frame: pd.DataFrame,
    model_columns: tuple[str, ...],
    registry: tuple[RankingScope, ...],
    scope_indices: tuple[np.ndarray, ...],
    encoded_rankings: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, ...],
]:
    point_scores = np.empty(
        (len(registry), len(model_columns)), dtype=float
    )
    point_probabilities = np.empty(
        (len(registry), len(model_columns), len(model_columns)), dtype=float
    )
    point_rank_counts = np.empty(
        (len(registry), len(model_columns), len(model_columns)), dtype=int
    )
    scope_metadata = np.empty((len(registry), 3), dtype=int)
    initial_vectors: list[np.ndarray] = []
    free_models = [
        model for model in model_columns if model != REFERENCE_MODEL
    ]

    for scope_index, row_indices in enumerate(scope_indices):
        selected = frame.iloc[row_indices]
        rankings = [
            [model_columns[index] for index in ranking]
            for ranking in encoded_rankings[row_indices]
        ]
        collapsed, weights = collapse_weighted_rankings(rankings)
        fit = fit_plackett_luce(
            collapsed,
            list(model_columns),
            weights=weights,
        )
        optimizer = fit["optimizer_result"]
        if not optimizer.success:
            raise RuntimeError(
                "Point Plackett-Luce optimization failed for scope "
                f"{registry[scope_index].scope!r}: {optimizer.message}"
            )
        fit_index = {
            model: index for index, model in enumerate(fit["models"])
        }
        point_scores[scope_index] = [
            fit["elo"][fit_index[model]] for model in model_columns
        ]
        point_probabilities[scope_index] = [
            [
                fit["pairwise_probabilities"][
                    fit_index[row_model], fit_index[column_model]
                ]
                for column_model in model_columns
            ]
            for row_model in model_columns
        ]
        initial_vectors.append(
            np.array([fit["xi"][model] for model in free_models])
        )
        for model_index, model in enumerate(model_columns):
            counts = selected[model].value_counts()
            point_rank_counts[scope_index, model_index] = [
                int(counts.get(f"R{rank}", 0))
                for rank in range(1, len(model_columns) + 1)
            ]
        scope_metadata[scope_index] = (
            len(selected),
            len(selected[["Species", "Expert"]].drop_duplicates()),
            len(selected[["Species", "Gene"]].drop_duplicates()),
        )
    return (
        point_scores,
        point_probabilities,
        point_rank_counts,
        scope_metadata,
        tuple(initial_vectors),
    )


def bootstrap_plackett_luce_statistics(
    frame: pd.DataFrame,
    model_columns: tuple[str, ...] = MODEL_COLUMNS,
    config: BootstrapConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Bootstrap crossed expert/gene uncertainty for the 18 PL scopes."""
    model_columns = tuple(model_columns)
    config = config or BootstrapConfig()
    if model_columns != MODEL_COLUMNS:
        raise ValueError(
            "Crossed ranking bootstrap requires canonical model columns."
        )
    working = _validate_pl_bootstrap_frame(frame, model_columns)
    registry = ranking_scope_registry()
    scope_indices = _ranking_scope_indices(working, registry)

    rank_lookup = {f"R{rank}": rank - 1 for rank in range(1, 6)}
    rank_numbers = np.array(
        [
            [rank_lookup[value] for value in row]
            for row in working.loc[:, model_columns].itertuples(
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
    (
        point_scores,
        point_probabilities,
        point_rank_counts,
        scope_metadata,
        initial_vectors,
    ) = _scope_point_statistics(
        working,
        model_columns,
        registry,
        scope_indices,
        encoded_rankings,
    )

    experts = (
        working[["Species", "Expert"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    genes = (
        working[["Species", "Gene", "StudyStatus"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    expert_keys = pd.MultiIndex.from_frame(experts[["Species", "Expert"]])
    gene_keys = pd.MultiIndex.from_frame(genes[["Species", "Gene"]])
    row_expert_indices = expert_keys.get_indexer(
        pd.MultiIndex.from_frame(working[["Species", "Expert"]])
    )
    row_gene_indices = gene_keys.get_indexer(
        pd.MultiIndex.from_frame(working[["Species", "Gene"]])
    )
    if (row_expert_indices < 0).any() or (row_gene_indices < 0).any():
        raise ValueError("Crossed bootstrap could not index every cluster.")

    seed_sequence = np.random.SeedSequence(config.seed)
    expert_seed, gene_seed = seed_sequence.spawn(2)
    expert_rng = np.random.default_rng(expert_seed)
    gene_rng = np.random.default_rng(gene_seed)
    score_samples: list[np.ndarray] = []
    rank_samples: list[np.ndarray] = []
    pairwise_samples: list[np.ndarray] = []
    failed_fits = 0
    failure_reasons: list[str] = []
    off_diagonal_pairs = [
        (row, column)
        for row in range(len(model_columns))
        for column in range(len(model_columns))
        if row != column
    ]

    while len(score_samples) < config.successful_replicates:
        expert_counts = sample_within_strata(
            experts,
            id_columns=("Expert",),
            strata=("Species",),
            rng=expert_rng,
        ).to_numpy(dtype=float)
        gene_counts = sample_within_strata(
            genes,
            id_columns=("Species", "Gene"),
            strata=("Species", "StudyStatus"),
            rng=gene_rng,
        ).to_numpy(dtype=float)
        expert_weights = expert_counts[row_expert_indices]
        gene_weights = gene_counts[row_gene_indices]
        analysis_weights = (
            expert_weights * gene_weights,
            expert_weights,
            gene_weights,
        )

        replicate_scores = np.empty(
            (
                len(registry),
                len(PL_INTERVAL_ANALYSES),
                len(model_columns),
            ),
            dtype=float,
        )
        replicate_ranks = np.empty(
            (len(registry), len(model_columns), len(model_columns)),
            dtype=float,
        )
        replicate_probabilities = np.empty(
            (len(registry), len(off_diagonal_pairs)), dtype=float
        )
        try:
            for scope_index, row_indices in enumerate(scope_indices):
                for analysis_index, row_weights in enumerate(
                    analysis_weights
                ):
                    selected_weights = row_weights[row_indices]
                    permutation_weights = np.bincount(
                        row_permutations[row_indices],
                        weights=selected_weights,
                        minlength=len(unique_rankings),
                    )
                    retained = permutation_weights > 0
                    if not retained.any():
                        raise RuntimeError(
                            "Bootstrap scope has zero effective weight."
                        )
                    fit = _bootstrap_pl_fit(
                        unique_rankings[retained],
                        permutation_weights[retained],
                        model_columns,
                        initial_vectors[scope_index],
                    )
                    replicate_scores[
                        scope_index, analysis_index
                    ] = fit["elo"]
                    if analysis_index == 0:
                        probabilities = fit["pairwise_probabilities"]
                        replicate_probabilities[scope_index] = [
                            probabilities[row, column]
                            for row, column in off_diagonal_pairs
                        ]
                        total_weight = float(selected_weights.sum())
                        if total_weight <= 0:
                            raise RuntimeError(
                                "Bootstrap scope has zero effective weight."
                            )
                        for model_index in range(len(model_columns)):
                            replicate_ranks[scope_index, model_index] = (
                                np.bincount(
                                    rank_numbers[
                                        row_indices, model_index
                                    ],
                                    weights=selected_weights,
                                    minlength=len(model_columns),
                                )
                                / total_weight
                            )
            if (
                not np.isfinite(replicate_scores).all()
                or not np.isfinite(replicate_ranks).all()
                or not np.isfinite(replicate_probabilities).all()
            ):
                raise RuntimeError(
                    "Bootstrap replicate produced nonfinite outputs."
                )
            score_samples.append(replicate_scores)
            rank_samples.append(replicate_ranks)
            pairwise_samples.append(replicate_probabilities)
        except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            failed_fits += 1
            failure_reasons.append(str(error))
            if failed_fits > config.max_failed_fits:
                raise RuntimeError(
                    "Crossed ranking bootstrap exceeded max_failed_fits "
                    "before reaching the requested successful replicates."
                ) from error

    scores_array = np.stack(score_samples)
    ranks_array = np.stack(rank_samples)
    pairwise_array = np.stack(pairwise_samples)
    score_lower, score_upper = np.quantile(
        scores_array,
        [0.025, 0.975],
        axis=0,
    )
    rank_lower, rank_upper = np.quantile(
        ranks_array,
        [0.025, 0.975],
        axis=0,
    )
    pairwise_lower, pairwise_upper = np.quantile(
        pairwise_array,
        [0.025, 0.975],
        axis=0,
    )

    half_run_diagnostics: dict[str, object] = {"Applied": False}
    if config.successful_replicates >= PL_PRODUCTION_REPLICATES:
        half = config.successful_replicates // 2
        first_score_bounds = np.quantile(
            scores_array[:half], [0.025, 0.975], axis=0
        )
        second_score_bounds = np.quantile(
            scores_array[half:], [0.025, 0.975], axis=0
        )
        first_probability_bounds = np.quantile(
            pairwise_array[:half], [0.025, 0.975], axis=0
        )
        second_probability_bounds = np.quantile(
            pairwise_array[half:], [0.025, 0.975], axis=0
        )
        half_run_diagnostics = {
            "Applied": True,
            **validate_half_run_stability(
                first_score_bounds,
                second_score_bounds,
                first_probability_bounds,
                second_probability_bounds,
            ),
        }

    common_metadata = {
        "SuccessfulReplicates": config.successful_replicates,
        "FailedFits": failed_fits,
        "SeedStream": PL_BOOTSTRAP_SEED_STREAM,
    }
    score_records: list[dict[str, object]] = []
    rank_records: list[dict[str, object]] = []
    pairwise_records: list[dict[str, object]] = []
    for scope_index, scope in enumerate(registry):
        n_judgments, n_experts, n_genes = scope_metadata[scope_index]
        scope_metadata_record = {
            "Scope": scope.scope,
            "StudyStatus": scope.study_status,
            "Species": scope.species,
            "NJudgments": int(n_judgments),
            "NExperts": int(n_experts),
            "NGenes": int(n_genes),
            **common_metadata,
        }
        for model_index, model in enumerate(model_columns):
            for analysis_index, analysis in enumerate(
                PL_INTERVAL_ANALYSES
            ):
                score_records.append(
                    {
                        **scope_metadata_record,
                        "Model": model,
                        "IntervalAnalysis": analysis,
                        "Estimate": float(
                            point_scores[scope_index, model_index]
                        ),
                        "CI95Lower": float(
                            score_lower[
                                scope_index,
                                analysis_index,
                                model_index,
                            ]
                        ),
                        "CI95Upper": float(
                            score_upper[
                                scope_index,
                                analysis_index,
                                model_index,
                            ]
                        ),
                    }
                )
            for rank_index in range(len(model_columns)):
                count = int(
                    point_rank_counts[
                        scope_index, model_index, rank_index
                    ]
                )
                rank_records.append(
                    {
                        **scope_metadata_record,
                        "Model": model,
                        "Rank": f"R{rank_index + 1}",
                        "Fraction": count / int(n_judgments),
                        "CI95Lower": float(
                            rank_lower[
                                scope_index, model_index, rank_index
                            ]
                        ),
                        "CI95Upper": float(
                            rank_upper[
                                scope_index, model_index, rank_index
                            ]
                        ),
                        "Count": count,
                    }
                )
            pair_offset = model_index * (len(model_columns) - 1)
            column_offset = 0
            for column_index, column_model in enumerate(model_columns):
                if model_index == column_index:
                    continue
                pair_index = pair_offset + column_offset
                column_offset += 1
                pairwise_records.append(
                    {
                        **scope_metadata_record,
                        "RowModel": model,
                        "ColumnModel": column_model,
                        "Probability": float(
                            point_probabilities[
                                scope_index, model_index, column_index
                            ]
                        ),
                        "CI95Lower": float(
                            pairwise_lower[scope_index, pair_index]
                        ),
                        "CI95Upper": float(
                            pairwise_upper[scope_index, pair_index]
                        ),
                    }
                )

    outputs = {
        "pl_scores_ci": pd.DataFrame.from_records(
            score_records,
            columns=PL_SCORE_COLUMNS,
        ),
        "rank_distribution_ci": pd.DataFrame.from_records(
            rank_records,
            columns=PL_RANK_COLUMNS,
        ),
        "pl_pairwise_ci": pd.DataFrame.from_records(
            pairwise_records,
            columns=PL_PAIRWISE_COLUMNS,
        ),
    }
    diagnostics = {
        "AttemptedReplicates": (
            config.successful_replicates + failed_fits
        ),
        "SuccessfulReplicates": config.successful_replicates,
        "FailedFits": failed_fits,
        "FailureReasons": tuple(failure_reasons),
        "SeedStream": PL_BOOTSTRAP_SEED_STREAM,
        "ExpertSeedSpawnKey": expert_seed.spawn_key,
        "GeneSeedSpawnKey": gene_seed.spawn_key,
        "HalfRunStability": half_run_diagnostics,
    }
    for output in outputs.values():
        output.attrs["bootstrap_diagnostics"] = diagnostics.copy()
    return outputs
