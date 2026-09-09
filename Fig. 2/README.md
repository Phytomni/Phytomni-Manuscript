# Fig. 2 inputs and outputs

`fig. 2.ipynb` renders Fig. 2a-c and the right-hand Fig. 2g-h panels from the
tracked TSV files in this directory. The authoritative target and input list is
[`reproduce.manifest.yaml`](../reproduce.manifest.yaml). Fig. 2d-f and the left-hand
Fig. 2g-h Elo panels are rendered by the Supplementary Fig. 12-15 notebook.

## Render frozen panels

From the repository root, install the locked figure environment and run both
targets to obtain the complete Fig. 2 panel set:

```bash
uv sync --frozen
source .venv/bin/activate
./reproduce.sh run --target fig-2 --target fig-2def
```

The runner sets `PHYTOMNI_SAVE=1`, executes isolated copies, and checks newly
generated artifacts before publishing them. The Fig. 2 notebook writes Fig. 2a-c
and Fig. 2g-h (right) into `Fig. 2/output/`. Ranking panels are written into
`Supplementary Fig. 12-15/output/`; these include Fig. 2d-f and Fig. 2g-h (left)
with `.score.bar` filenames for the latter Elo bars.

The left-hand Fig. 2g-h panels reuse the well-studied and uncharacterized Elo
summaries, respectively, with the Fig. 2f y-axis range `[1200, 1700]`.
The right-hand panels show BERTScore precision and internal hallucination rate,
respectively, with ranges `[0.47, 0.58]` and `[0.1, 0.7]`. Rendering uses the
frozen inputs; it does not refit rankings, contact a model, or require private
workbooks. See the [ranking-data guide](<../Supplementary Fig. 12-15/PhytoBench-Gene-for_plot/README.md>)
for public numerical recomputation and the separately controlled publication
refresh.

## Core benchmark inputs and author-only refresh

Fig. 2a-c read `PhytoBench-Knowledge-for_plot.tsv`,
`PhytoBench-Data-for_plot.tsv`, and `PhytoBench-Analysis-for_plot.tsv`, rather
than embedded arrays. Their field definitions are in the
[metric dictionary](DATA_DICTIONARY.md).

The Fig. 2c source contains 50 primary runs (10 scenarios × 5 runs) per model,
excluding the 60 cross-species extension runs. The violin plot shows the
distribution of per-run composite scores, with an embedded box and mean line.
Planning, tool selection, parameter setting, and execution are each scored from
0 to 25. The recorded 0–100 total is their sum, equivalently the mean of the
four normalized component scores multiplied by 100. The components remain in
the input table but are not plotted separately.

Fig. 2a deliberately aligns the Phyto-Chatbot values to the 20260727 manuscript:
`IdentificationAccuracy = 0.720` and `TraceBLEU4 = 0.084`. The reviewed
Supplementary Data 4 now agrees with 0.720; Supplementary Data 5 still gives
Trace BLEU-4 = 0.06023096459723756, whose discrepancy remains unresolved.
The freezer preserves both the workbook-computed and manuscript-aligned
values; do not replace the aligned values with current workbook means during
a repository cleanup. Fig. 2b and Fig. 2c remain derived directly from their
private workbooks.

Only for an authorized source-data refresh, provide the four private workbooks:

```bash
uv run --frozen --no-sync python -m scripts.freeze_fig2_core_metrics \
  --knowledge-id /path/to/knowledge-identification.xlsx \
  --knowledge-trace /path/to/knowledge-trace.xlsx \
  --data /path/to/data-benchmark.xlsx \
  --analysis /path/to/analysis-benchmark.xlsx \
  --output-dir "Fig. 2"
```

The freezer validates row identifiers, completeness, numeric ranges, model
column mappings, and the primary Analysis cohort. It writes the three TSVs and
`PhytoBench-Core-for_plot-provenance.json`, recording source-file and output
hashes, row counts, scope, metric definitions, and model order without exposing
private source paths. This mutating refresh is not part of figure reproduction.
The reviewed source headers use `Claude-Opus-4.1` and, for Analysis,
`Phyto-ChatBot (Knowledge Agent)`; retired spelling aliases are not accepted.

## Gene-metric data ownership

`PhytoBench-Gene-metrics-for_plot/` belongs exclusively to
`scripts/freeze_fig2_gene_metrics.py` and contains four frozen files:

- `PhytoBench-Gene-Claude-BERTScore-by-gene.tsv`: precision for 100 well-studied
  genes, with query and response hashes.
- `PhytoBench-Gene-Claude-hallucination-by-gene.tsv`: mean directional
  contradiction ratios for 100 uncharacterized genes.
- `PhytoBench-Gene-Claude-hallucination-pairs.tsv`: the corresponding 600 ordered
  response-pair summaries.
- `PhytoBench-Gene-Claude-metrics-provenance.json`: settings, input/output hashes,
  historical model values, and non-secret metadata from the recorded evaluation.

These three TSVs and their provenance were relocated without recalculation.
Their hashes and historical run metadata are unchanged. They support the final
five-model `PhytoBench-Gene-BERTScore-for_plot.tsv` and
`PhytoBench-Gene-hallucination-for_plot.tsv` read by the notebook; the notebook
does not load private responses or rerun a judge. Cross-response contradiction
measures inconsistency, not independently verified factual incorrectness.

For an authorized metric refresh, pass
`--frozen-dir "Fig. 2/PhytoBench-Gene-metrics-for_plot"` and
`--figure-dir "Fig. 2"` to the gene-metric freezer from the repository root.
Its other arguments require the private source workbook, response archive,
and recorded judgment logs. Neither output argument may point into the
[ranking publication](<../Supplementary Fig. 12-15/PhytoBench-Gene-for_plot/README.md>).
Reproducing the figures from their frozen tables does not require a refresh.

See the [repository README](../README.md) for the locked environment and figure
commands. Saved figures go to this directory's `output/` only when
`PHYTOMNI_SAVE=1` is set. Keep documentation outside the strict four-file
gene-metric directory.

The [metric dictionary](DATA_DICTIONARY.md) defines all public table fields,
denominators, the primary Analysis cohort, and the preserved Fig. 2a
manuscript/source-workbook discrepancy.
