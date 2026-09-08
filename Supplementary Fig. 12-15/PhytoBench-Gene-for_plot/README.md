# PhytoBench-Gene figure inputs

See the [data dictionary](DATA_DICTIONARY.md) for every serialized field,
unique key, denominator, interval method, and source-to-panel relationship.
Commands below run from the repository root. The
[manifest](../../reproduce.manifest.yaml) is the executable target authority.

## Render the accepted figures

```bash
uv sync --frozen
source .venv/bin/activate
./reproduce.sh run --target fig-2def --target supp-12-15
```

The runner renders the tracked aggregate inputs through isolated notebook
execution and validates fresh outputs before publication. It does not recompute
bootstrap statistics or contact a model. Panel files go to
`Supplementary Fig. 12-15/output/`; the separate Fig. 2a-c and Fig. 2g-h (right)
panels are covered by the [Fig. 2 guide](<../../Fig. 2/README.md>).

## Frozen publication ownership

`frozen/` is the authoritative schema-2 figure-input snapshot for Fig. 2d-f,
Fig. 2g-h (left), and Supplementary Fig. 12-15. The plotting notebook reads
exactly these 13 files:

- Point estimates: `rank_distribution.tsv`, `pl_scores.tsv`, and
  `pl_pairwise.tsv`.
- 95% bootstrap intervals: `rank_distribution_ci.tsv`, `pl_scores_ci.tsv`, and
  `pl_pairwise_ci.tsv`.
- Agreement: `fleiss_kappa.tsv`, `kendall_by_gene.tsv`,
  `ordinal_agreement_summary.tsv`, and `top1_consensus.tsv`.
- Panel audit: `expert_panel_summary.tsv` and `assignment_summary.tsv`.
- Provenance: `provenance.json`.

Only these 13 ranking-publication files belong in `frozen/`. The three compact
Claude metric TSVs and their historical provenance JSON are owned separately by
`Fig. 2/PhytoBench-Gene-metrics-for_plot/`; see the
[Fig. 2 data guide](<../../Fig. 2/README.md>). Their relocation did not change
their bytes or recorded results. Do not direct the gene-metric freezer into
this ranking publication directory.

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
[Supplementary_Data_Expert_Rankings.tsv](<../../DeepGenomeAgent Evaluation/supplementary/Supplementary_Data_Expert_Rankings.tsv>);
its [codebook](<../../DeepGenomeAgent Evaluation/supplementary/Supplementary_Data_Expert_Rankings_Codebook.tsv>)
and [public expert-panel category map](<../../DeepGenomeAgent Evaluation/supplementary/expert_panel_category_map.tsv>)
are in the same directory.
The release contains 600 expert–gene assignments from 120 anonymized experts,
with one complete five-model ranking per row. Row-level demographics are not
released. In `expert_panel_summary.tsv`, the multi-select `Research_domains`
and `Study_species` dimensions use all 120 experts as the denominator and can
sum to more than 100%; single-select dimensions use nonmissing respondents.
All 120 experts self-reported `No` for conflict of interest, which supports the
statement that no conflicts were declared but is not an independent COI audit.

## Recompute public ranking statistics

The public score release and `gene_categories.tsv` are sufficient to recompute
11 numerical and assignment tables. Private expert metadata, the private lineage
`score.tsv`, model responses, judgment logs, credentials, and network services
are not required. This command does not reconstruct `expert_panel_summary.tsv`
or individual demographics and cannot replace the 13-file publication snapshot.

Start with one 100-replicate preview in a fresh independent directory:

```bash
uv run --frozen --no-sync python -m scripts.recompute_deepgenome_rankings \
  --score-tsv "DeepGenomeAgent Evaluation/supplementary/Supplementary_Data_Expert_Rankings.tsv" \
  --gene-categories "Supplementary Fig. 12-15/PhytoBench-Gene-for_plot/gene_categories.tsv" \
  --seed 20260714 --bootstrap-replicates 100 --max-failed-fits 10 \
  --output-dir "local_exports/ranking-recomputation/preview-001" \
  --reference-dir "Supplementary Fig. 12-15/PhytoBench-Gene-for_plot/frozen"
```

All six non-reference arguments are required; `--reference-dir` is optional.
The CLI checks the canonical source-column order, original anonymous expert
IDs, complete R1–R5 permutations, category join, and the 600-assignment,
120-expert, 200-gene cohort. It reports aggregate progress at 10% milestones.
It preserves the computational species order recorded by the implementation;
the figure's display-species order must not be applied to bootstrap sampling.

The output is exactly 11 TSVs plus `recomputation-provenance.json`, and
`comparison.json` when a reference is provided. These are nonpublication
artifacts: `publication_eligible` is always false. Within the repository, use
a fresh run below the ignored `local_exports/ranking-recomputation/` directory.
The CLI rejects source/reference overlap, publication locations and their
ancestors or descendants, symlink aliases, and nonempty destinations. Inputs
and references are rechecked before independent results are written.

Preview comparisons cover point estimates and counts only. Their confidence
intervals are provisional and are not compared with the 10,000-replicate
reference intervals. Production-only half-run and Monte Carlo diagnostics
remain `Applied: false`, not passed. A successful preview does not establish
full numerical equivalence.

After reviewing preview cost and obtaining separate authorization, use
`--bootstrap-replicates 10000` with a different fresh directory such as
`local_exports/ranking-recomputation/full-001`. Keep `--seed 20260714`,
`--max-failed-fits 10`, and the same public inputs and reference. Full comparison
requires matching input hashes and configuration and compares every numerical
table's schema, row order, values, and deterministic TSV bytes. Half-run
stability remains descriptive and nonblocking; no new Monte Carlo pass
threshold is imposed. Passing this comparison is not permission to publish
a new snapshot or rerun the original model evaluations.

Unknown or missing arguments return exit code 2. Invalid inputs, computation
errors, and requested-reference mismatches return 1; a successful run returns 0.
Comparison mismatches retain the independent report with differing fields and
values, while the reference remains unchanged. Hashes, settings, cohort counts,
diagnostics, package versions, and the exclusion of private-panel reconstruction
are recorded in the run provenance.

## Author-only publication refresh

Refreshing all 12 publication TSVs and `provenance.json` is a separate,
explicitly authorized operation. It requires the public ranking release and
mappings plus private expert metadata and private ranking lineage. The private
inputs permit the controlled demographic aggregation and lineage checks; they
are not needed for the 11-table public numerical recomputation above.
Individual model responses and hallucination-judgment logs are not inputs to
this ranking freezer. Those belong to the separate
[original scoring workflows](<../../DeepGenomeAgent Evaluation/README.md>).

Only after authorizing a production refresh, run:

```bash
uv run --frozen --no-sync python -m scripts.freeze_deepgenome_rankings \
  --score-tsv "DeepGenomeAgent Evaluation/supplementary/Supplementary_Data_Expert_Rankings.tsv" \
  --expert-column AnonymousExpertID \
  --gene-categories "Supplementary Fig. 12-15/PhytoBench-Gene-for_plot/gene_categories.tsv" \
  --expert-metadata /absolute/path/to/expert_metadata.xlsx \
  --private-lineage-score /absolute/path/to/private/score.tsv \
  --panel-category-map "DeepGenomeAgent Evaluation/supplementary/expert_panel_category_map.tsv" \
  --bootstrap-replicates 10000 --seed 20260714 --max-failed-fits 10 \
  --output-dir "Supplementary Fig. 12-15/PhytoBench-Gene-for_plot/frozen"
```

This private publication workflow is not available from a public clone and
must not be used for a quick preview. Preserve the public COI self-declaration
wording and approved demographic aggregation; do not release row-level
metadata or a private-to-public expert crosswalk.

## Historical inputs

The `legacy/score*.tsv` files are legacy four-model inputs retained for
historical provenance. They are not read by the current plotting notebook or
by `reproduce.manifest.yaml`. Do not use them to regenerate the current
five-model panels.

See the [repository guide](../../README.md) for environment setup and the
separate figure, public recomputation, and original evaluation workflows.
