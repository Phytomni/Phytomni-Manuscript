# PhytoBench-Gene figure inputs

`frozen/` is the authoritative figure-input snapshot for Fig. 2d-f and
Supplementary Fig. 10-13. It contains only aggregate rank distributions,
Plackett-Luce scores, pairwise probabilities, and provenance for the validated
five-model run. The new private five-model `score.tsv`, individual responses,
and judgment logs are not tracked.

`gene_categories.tsv` maps the 200 benchmark genes to the `well_studied` and
`uncharacterized` strata used by the aggregate freezer.

The `score*.tsv` files in this directory are legacy four-model inputs retained
for historical provenance. They are not read by the current plotting notebook
or by `reproduce.manifest.yaml`. Do not use them to regenerate the current
five-model panels.

See the repository README for the deterministic freeze and direct plotting
commands.
