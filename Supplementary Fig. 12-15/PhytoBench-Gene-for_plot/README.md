# PhytoBench-Gene figure inputs

`frozen/` is the authoritative schema-2 figure-input snapshot for Fig. 2d-f and
Supplementary Fig. 12-15. The plotting notebook reads exactly these 13 files:

- Point estimates: `rank_distribution.tsv`, `pl_scores.tsv`, and
  `pl_pairwise.tsv`.
- 95% bootstrap intervals: `rank_distribution_ci.tsv`, `pl_scores_ci.tsv`, and
  `pl_pairwise_ci.tsv`.
- Agreement: `fleiss_kappa.tsv`, `kendall_by_gene.tsv`,
  `ordinal_agreement_summary.tsv`, and `top1_consensus.tsv`.
- Panel audit: `expert_panel_summary.tsv` and `assignment_summary.tsv`.
- Provenance: `provenance.json`.

The production snapshot uses 10,000 successful bootstrap replicates with the
recorded master seed. Plackett-Luce 95% confidence intervals use the crossed
expert–gene percentile analysis; agreement confidence intervals use a
species×study-status-stratified gene-block percentile bootstrap.

In `fleiss_kappa.tsv`, each item is a gene×model pair, R1–R5 are the rating
categories, and three experts rate every item. The table preserves 58 locked
scopes: 1 primary overall scope, 12 locked secondary scopes (five species, two
study-status groups, and five models), and 45 locked exploratory interaction
scopes. Supplementary Fig. 13 plots the overall, five-species, and two-study-
status scopes; model-specific and exploratory estimates remain in the full
auditable table. Per-gene Kendall W and top-1 consensus provide complementary
ordinal summaries.

`gene_categories.tsv` maps the 200 benchmark genes to the `well_studied` and
`uncharacterized` strata used by the aggregate freezer.

The full anonymized scoring table is tracked at
`../../DeepGenomeAgent Evaluation/supplementary/Supplementary_Data_Expert_Rankings.tsv`;
its codebook and the public expert-panel category map are in the same directory.
The release contains 600 expert–gene assignments from 120 anonymized experts,
with one complete five-model ranking per row. Row-level demographics are not
released. In `expert_panel_summary.tsv`, the multi-select `Research_domains`
and `Study_species` dimensions use all 120 experts as the denominator and can
sum to more than 100%; single-select dimensions use nonmissing respondents.
All 120 experts self-reported `No` for conflict of interest, which supports the
statement that no conflicts were declared but is not an independent COI audit.

Private expert metadata, the private lineage `score.tsv`, individual model
responses, and judgment logs are not tracked. They are required only to rebuild
the frozen snapshot, not to reproduce the figures from a clone. The root README
contains the complete deterministic freeze command.

The `legacy/score*.tsv` files are legacy four-model inputs retained for
historical provenance. They are not read by the current plotting notebook or
by `reproduce.manifest.yaml`. Do not use them to regenerate the current
five-model panels.

See the repository README for the deterministic freeze and direct plotting
commands.
