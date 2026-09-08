# Ranking aggregate dictionary

The executable definitions are `OUTPUT_SCHEMAS`, `OUTPUT_UNIQUE_KEYS`, and
`OUTPUT_SORT_KEYS` in `scripts/freeze_deepgenome_rankings.py`, with numerical
definitions in `scripts/deepgenome_ranking_statistics.py`. Paths below are
relative to the repository root unless linked. This document explains those
contracts; it is not a second target or statistical-scope registry.

## Population, units, and interpretation

The public release contains 600 complete expert-gene assignments: 120 experts
each rank five genes, and 200 genes each have three assigned experts. Each row
is a forced, no-tie ranking of five systems, R1 (best) through R5. Preserve
the released E001-E120 identifiers and exact source-column order for seeded
recomputation. The gene-category table defines well-studied versus
uncharacterized strata; all five species are retained.

Eleven numerical/assignment tables can be computed without private metadata.
The publication adds `expert_panel_summary.tsv`, making 12 TSVs plus
`provenance.json`. Public recomputation does not reconstruct that private-input
aggregate or replace the publication. The four compact Claude metric files
belong to [Fig. 2](<../../Fig. 2/README.md>), not this frozen directory.

Fractions and probabilities are on 0-1 scales; `Percent` is on 0-100. Counts are
nonnegative integers, not percentages. An `all` scope value means aggregation,
not missing data. Valid source rankings and identifiers cannot be missing.
An empty or nonfinite inferential result must not be imputed to zero; bootstrap
invalidity is counted and the existing computation validity rules apply.

### Scope and identity fields

- `Scope` and `ScopeID`: canonical aggregation identifiers. `StudyStatus` and
  `Species` identify the included strata; `Gene` is the public gene identifier.
- `AnalysisTier`: `primary`, `locked_secondary`, or `locked_exploratory`, as
  fixed before plotting. `ScopeFamily` identifies which dimensions define a scope.
- `Model`, `RowModel`, `ColumnModel`: system identifiers, not display names.
  Pairwise `Probability` means the row model beats the column model; the point
  table retains self-comparison rows with blank/NaN Probability, while the
  interval table excludes self-comparisons.
- `Rank`: R1-R5. `Top1AgreementPattern`: unanimous, two-of-three majority, or all
  different, using the exact serialized category names in the table.

### Counts, proportions, and assignment balance

- `N` in the three point-estimate tables is the number of complete assignments
  in the scope. In `expert_panel_summary`, `N` instead counts experts in a category.
- `Count` in rank tables counts assignments giving that model the specified
  rank; `Fraction = Count / N` (or `NJudgments`). In `top1_consensus`, `Count`
  counts genes and `Fraction = Count / NGenes`.
- `NJudgments`: complete expert-gene rankings; `NGenes`: distinct species-gene
  pairs. `NExperts` counts contributing experts (species-expert pairs in the
  assignment summary); `NContributingExperts` is the agreement-scope expert count.
- `NModels`: number of systems ranked. `NItems`: gene-model items in Fleiss
  analysis; `RatingsPerItem` is three here and `NRatings` totals item ratings.
- `MinGenesPerExpert`, `MaxGenesPerExpert`, `MinExpertsPerGene`, and
  `MaxExpertsPerGene`: within-scope assignment extrema, not the global totals.

### Plackett-Luce scores and intervals

- `Elo` and `Estimate`: Elo-like scores, transformed by 400/log(10) from fitted
  log-strengths and centered at 1500 across the five models within each scope.
  They are relative ranking scores, not a calibrated probability or percent.
- `Elo_L` and `Elo_U` in `pl_scores.tsv` preserve the original Hessian-based
  approximate 95% intervals. Current plotted uncertainty instead comes from
  the bootstrap tables; these two interval definitions must not be conflated.
- `IntervalAnalysis` identifies the retained bootstrap sensitivity analysis.
  The primary figure intervals use the crossed expert-gene analysis.
- `CI95Lower` and `CI95Upper`: percentile 95% limits for the row's score,
  probability, or fraction. `SuccessfulReplicates`, `FailedFits`, and
  `SeedStream` record computation counts and the derived deterministic stream.

### Agreement and consensus

- `ObservedAgreement`, `ExpectedAgreement`: observed and marginal-chance rank
  agreement. `FleissKappa` is their chance-corrected agreement; negative values
  are legitimate and are retained. `RankR1Share` through `RankR5Share` are the
  marginal proportions of item ratings. `CILower` and `CIUpper` are bootstrap
  limits for kappa, not limits for those marginal proportions.
- `KendallW`: per-gene concordance in [0,1]. `MeanPairwiseKendallTau` is the mean
  of the three pairwise rank correlations in [-1,1]. With three no-tie rankings,
  the random-ranking baseline for W is 1/3, not zero.
- `KendallWMean`, `KendallWMedian`, `KendallWQ1`, and `KendallWQ3` summarize
  the distribution across genes. The corresponding
  `MeanPairwiseKendallTauMean`, `MeanPairwiseKendallTauMedian`,
  `MeanPairwiseKendallTauQ1`, and `MeanPairwiseKendallTauQ3` summarize tau.
  Mean/median fields ending in `CILower` or `CIUpper` are percentile 95% limits
  for that summary, not per-gene measurement errors.
- `FractionCILower`, `FractionCIUpper`: limits for each top-1 pattern fraction.
- `BootstrapAttempted`, `BootstrapReplicates`, `BootstrapInvalid`: total draws,
  accepted draws, and invalid draws. `BootstrapUnit` and `BootstrapStrata`
  identify the gene-block resampling and species-by-study-status stratification.

The production configuration uses master seed 20260714 and 10,000 successful
replicates, with a maximum of 10 failed PL fits. Computational registry ordering
must not be changed to match a plot's display order. Reduced-replicate previews
are not replacement confidence intervals; production-only diagnostics with
`Applied: false` are not passed checks. Half-run stability remains descriptive
and nonblocking. Agreement measures expert preference consistency, not factual
correctness, and within-expert cross-gene stability is not a frozen table.

### Expert-panel aggregate

`Dimension` and `PublicCategory` identify a privacy-reviewed grouping;
`DisplayOrder` determines category presentation. `DenominatorN` is the number
used for `Percent = 100 * N / DenominatorN`; `MissingN` records missing responses.
`PercentageBasis` distinguishes the denominator policy. Multi-select
`Research_domains` and `Study_species` use all 120 experts and can sum above
100%. Single-select dimensions use nonmissing respondents. All 120 experts
self-reported no conflict: no conflicts were declared, not independently ruled
out. Individual demographics and identity crosswalks are not public inputs.

## Navigation and provenance

Rank distribution feeds Fig. 2d and Supplementary Fig. 12; pairwise probabilities
feed Fig. 2e and Supplementary Fig. 14; scores feed Fig. 2f, Fig. 2g-h (left), and
Supplementary Fig. 15. Panel composition, kappa, mean W, and top-1 consensus form
the four Supplementary Fig. 13 panels. It displays eight non-model kappa scopes;
all 58 locked estimates remain available. Consult
[`reproduce.manifest.yaml`](../../reproduce.manifest.yaml) for executable targets
and artifact ownership, and [the input guide](README.md) for commands.

`provenance.json` stores public input/output hashes, schema and settings, code
identity, and numerical diagnostics. A relocation or code-hash change is not a
new experiment. Numerical TSVs use deterministic ordering and 15-significant-
digit serialization; report differences before considering a reference update.

## Exact serialized schemas

Column and key order follows the executable definitions.

### `rank_distribution.tsv`

Columns: `Scope`, `StudyStatus`, `Species`, `Model`, `Rank`, `Count`, `Fraction`, `N`.

Unique key: `Scope`, `Model`, `Rank`.

Sort keys: `Scope`, `Model`, `Rank`.

### `pl_scores.tsv`

Columns: `Scope`, `StudyStatus`, `Species`, `Model`, `Elo`, `Elo_L`, `Elo_U`, `N`.

Unique key: `Scope`, `Model`.

Sort keys: `Scope`, `Model`.

### `pl_pairwise.tsv`

Columns: `Scope`, `StudyStatus`, `Species`, `RowModel`, `ColumnModel`, `Probability`, `N`.

Unique key: `Scope`, `RowModel`, `ColumnModel`.

Sort keys: `Scope`, `RowModel`, `ColumnModel`.

### `pl_scores_ci.tsv`

Columns: `Scope`, `StudyStatus`, `Species`, `Model`, `IntervalAnalysis`, `Estimate`, `CI95Lower`, `CI95Upper`, `NJudgments`, `NExperts`, `NGenes`, `SuccessfulReplicates`, `FailedFits`, `SeedStream`.

Unique key: `Scope`, `Model`, `IntervalAnalysis`.

Sort keys: `Scope`, `Model`, `IntervalAnalysis`.

### `rank_distribution_ci.tsv`

Columns: `Scope`, `StudyStatus`, `Species`, `Model`, `Rank`, `Fraction`, `CI95Lower`, `CI95Upper`, `Count`, `NJudgments`, `NExperts`, `NGenes`, `SuccessfulReplicates`, `FailedFits`, `SeedStream`.

Unique key: `Scope`, `Model`, `Rank`.

Sort keys: `Scope`, `Model`, `Rank`.

### `pl_pairwise_ci.tsv`

Columns: `Scope`, `StudyStatus`, `Species`, `RowModel`, `ColumnModel`, `Probability`, `CI95Lower`, `CI95Upper`, `NJudgments`, `NExperts`, `NGenes`, `SuccessfulReplicates`, `FailedFits`, `SeedStream`.

Unique key: `Scope`, `RowModel`, `ColumnModel`.

Sort keys: `Scope`, `RowModel`, `ColumnModel`.

### `fleiss_kappa.tsv`

Columns: `ScopeID`, `AnalysisTier`, `ScopeFamily`, `Species`, `StudyStatus`, `Model`, `NGenes`, `NItems`, `RatingsPerItem`, `NRatings`, `NContributingExperts`, `ObservedAgreement`, `ExpectedAgreement`, `FleissKappa`, `RankR1Share`, `RankR2Share`, `RankR3Share`, `RankR4Share`, `RankR5Share`, `CILower`, `CIUpper`, `BootstrapAttempted`, `BootstrapReplicates`, `BootstrapInvalid`, `BootstrapUnit`, `BootstrapStrata`, `SeedStream`.

Unique key: `ScopeID`.

Sort keys: `ScopeID`.

### `kendall_by_gene.tsv`

Columns: `Species`, `Gene`, `StudyStatus`, `NExperts`, `NModels`, `KendallW`, `MeanPairwiseKendallTau`, `Top1AgreementPattern`.

Unique key: `Species`, `Gene`.

Sort keys: `Species`, `Gene`.

### `ordinal_agreement_summary.tsv`

Columns: `ScopeID`, `AnalysisTier`, `ScopeFamily`, `Species`, `StudyStatus`, `NGenes`, `NContributingExperts`, `KendallWMean`, `KendallWMedian`, `KendallWQ1`, `KendallWQ3`, `KendallWMeanCILower`, `KendallWMeanCIUpper`, `KendallWMedianCILower`, `KendallWMedianCIUpper`, `MeanPairwiseKendallTauMean`, `MeanPairwiseKendallTauMedian`, `MeanPairwiseKendallTauQ1`, `MeanPairwiseKendallTauQ3`, `MeanPairwiseKendallTauMeanCILower`, `MeanPairwiseKendallTauMeanCIUpper`, `MeanPairwiseKendallTauMedianCILower`, `MeanPairwiseKendallTauMedianCIUpper`, `BootstrapAttempted`, `BootstrapReplicates`, `BootstrapInvalid`, `BootstrapUnit`, `BootstrapStrata`, `SeedStream`.

Unique key: `ScopeID`.

Sort keys: `ScopeID`.

### `top1_consensus.tsv`

Columns: `ScopeID`, `AnalysisTier`, `ScopeFamily`, `Species`, `StudyStatus`, `Top1AgreementPattern`, `Count`, `Fraction`, `FractionCILower`, `FractionCIUpper`, `NGenes`, `NContributingExperts`, `BootstrapAttempted`, `BootstrapReplicates`, `BootstrapInvalid`, `BootstrapUnit`, `BootstrapStrata`, `SeedStream`.

Unique key: `ScopeID`, `Top1AgreementPattern`.

Sort keys: `ScopeID`, `Top1AgreementPattern`.

### `expert_panel_summary.tsv`

Columns: `Dimension`, `PublicCategory`, `DisplayOrder`, `N`, `DenominatorN`, `Percent`, `MissingN`, `PercentageBasis`.

Unique key: `Dimension`, `PublicCategory`.

Sort keys: `Dimension`, `DisplayOrder`, `PublicCategory`.

### `assignment_summary.tsv`

Columns: `Scope`, `StudyStatus`, `Species`, `NExperts`, `NGenes`, `NJudgments`, `MinGenesPerExpert`, `MaxGenesPerExpert`, `MinExpertsPerGene`, `MaxExpertsPerGene`.

Unique key: `Scope`.

Sort keys: `Scope`.
