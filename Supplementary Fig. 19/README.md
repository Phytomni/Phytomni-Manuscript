# Supplementary Fig. 19: SeedBench inputs

The notebook renders the four SeedBench task summaries, corresponding to panel
a in the current manuscript caption. It does not render the AA-Omniscience
hallucination panel. The existing `output/supplementary_fig.19.pdf` basename,
four task facets, eight-model colors and order, 0-1 axis, and 14 x 8 inch size
are preserved. Display spellings use `o3` and `Claude-Opus-4.1`.

## Reproduce the figure

Prepare the locked Python and R environments following the repository
[README](../README.md), activate Python, and set `R_LIBS_USER` to the prepared
R library. Then run from the repository root:

```bash
./reproduce.sh run --target supp-19 --timeout 600
```

The safe runner reads the frozen files below and writes a fresh PDF to
`output/`. It does not need the private workbook, an API key, or an evaluator.
Tracked notebooks retain no executed outputs. The executable input/output
contract is [reproduce.manifest.yaml](../reproduce.manifest.yaml).

## Frozen data dictionary

`SeedBench-for_plot.tsv` has 32 rows: four tasks by eight models. Each
(`Task`, `Model`) pair is unique, and every task contains the same eight models.

| Field | Definition |
|---|---|
| `Task` | QA-1, QA-2, QA-3, or QA-4 |
| `Metric` | Accuracy, Macro-F1, or ROUGE-L F1, as specified below |
| `Model` | Canonical model identifier and figure label |
| `Score` | Task-specific summary on the 0-1 scale, at the existing four-decimal precision |
| `N` | Number of questions: 235, 218, 269, and 277 for QA-1 through QA-4 |

QA-1 and QA-2 retain the supplied summary values exactly. QA-1 is single-choice
accuracy. The retained QA-2 definition first averages per-sample precision and
recall and then takes their harmonic mean; it is not the average of per-item
F1. This refresh does not redefine that metric or alter either summary row.

For QA-3 and QA-4, the author confirmed the full 269- and 277-question cohorts
and their existing per-item ROUGE-L F1 scores as final. The summaries are the
equal-weight arithmetic means of every stored score, including zeros, with
no filtering or special weights. Rounding to four decimal places occurs only
after averaging, preserving the original summary and plot precision. These
are aggregates of recorded scores, not a new ROUGE implementation or a repeat
of model inference. Different task metrics must not be averaged together.

`SeedBench-for_plot-provenance.json` records source-file and output hashes,
row counts, source-column mappings, aggregation rules, and unrounded QA-3/4
means and zero counts. It contains no raw prompts, responses, question IDs,
private source paths, or credentials. The private reviewed Data 21 workbook
and author-only correction audit remain outside Git.

## Author-only source refresh

Only after an explicitly approved data revision, run:

```bash
python -m scripts.freeze_seedbench_metrics \
  --source /path/to/reviewed-source.xlsx \
  --output-dir "Supplementary Fig. 19"
```

The freezer validates the current source schema, task counts, identifiers,
scores, and QA-3/4 agreement between per-item means and the reviewed summary.
It does not repair the workbook. Missing or inconsistent inputs fail rather
than silently changing cohorts or falling back to historical inline values.
Future source changes require both the workbook summary and frozen publication
inputs to be reconciled before figure regeneration.
