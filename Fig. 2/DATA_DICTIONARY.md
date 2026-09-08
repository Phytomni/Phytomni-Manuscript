# Fig. 2 metric dictionary

The notebook reads five tracked plot-ready TSVs. It does not run evaluations or
contact a model. [The input guide](README.md) explains ownership and output
locations; [`reproduce.manifest.yaml`](../reproduce.manifest.yaml) is the target
authority. Raw workbooks, model responses, and judge logs remain private.

| Table | Fields and units | Panel |
|---|---|---|
| `PhytoBench-Knowledge-for_plot.tsv` | `Model`, `DisplayLabel`; `IdentificationAccuracy` in [0,1] and denominator `IdentificationN`; `TraceBLEU4` in [0,1] and denominator `TraceN` | 2a |
| `PhytoBench-Data-for_plot.tsv` | `Model`, `DisplayLabel`; `Accuracy` in [0,1], question count `N` | 2b |
| `PhytoBench-Analysis-for_plot.tsv` | `Task`, `Species`, repeat `Rep`, `ObservationID`, `Model`, `DisplayLabel`; unique row key (`ObservationID`, `Model`); `PlanScore`, `ToolScore`, `ParameterScore`, `RateScore` each 0-25; `TotalScore` is their 0-100 sum | 2c |
| `PhytoBench-Gene-BERTScore-for_plot.tsv` | `Model`, `DisplayLabel`, mean `BERTScorePrecision` in [0,1] | 2g right |
| `PhytoBench-Gene-hallucination-for_plot.tsv` | `Model`, `DisplayLabel`, `MeanDirectionalContradictionRatio` in [0,1] | 2h right |

`Model` is a stable computational identifier; `DisplayLabel` is the manuscript
label, including Claude Research. The Analysis table contains 50 primary rice
observations per model (10 scenarios x 5 repeats; 450 model-observation rows
across nine models), excluding the 60 cross-species observations per model.
The main panel shows each of the four component dimensions; a component is not
a percentage or the 0-100 total. Identifiers, expected columns, model mappings,
and valid numerical ranges are checked by the corresponding freezer and tests.
Missing required observations are errors, not zero scores.

## Frozen compact Claude records

`PhytoBench-Gene-metrics-for_plot/` contains exactly three TSVs and one provenance
JSON. These are upstream records, not additional files the plot notebook reads.

- `PhytoBench-Gene-Claude-BERTScore-by-gene.tsv`: `Model`, `Gene`, and
  `BERTScorePrecision` for 100 well-studied genes; `QuerySHA256` and
  `ResponseSHA256` identify the private compared texts without releasing them.
- `PhytoBench-Gene-Claude-hallucination-by-gene.tsv`: `Species`, `Gene`,
  `StudyStatus`, `Model`, six ordered comparisons in `DirectionalPairCount`,
  their mean `MeanDirectionalContradictionRatio`, and `HighContradiction`
  under the recorded inclusive threshold.
- `PhytoBench-Gene-Claude-hallucination-pairs.tsv`: the same gene/model/stratum
  identifiers plus `SourceResponseID` and `TargetResponseID` for each ordered
  pair. `WindowJudgmentCount`, `EntailmentCount`, `ContradictionCount`, and
  `NeutralCount` are window-level counts. `SupportRatio` and
  `ContradictionRatio` divide the corresponding counts by `WindowJudgmentCount`;
  `JudgmentLogSHA256` identifies the frozen private judgment log. There are 600
  pairs over 100 uncharacterized genes, not 600 independent genes.
- `PhytoBench-Gene-Claude-metrics-provenance.json`: exact scorer/judge settings,
  historical other-model metrics, gene counts, input/output hashes, and recorded
  handling of archive anomalies. Its original bytes are preserved on relocation.

Gene-metric fractions must not be interpreted as direct factual-error rates:
consistent repeated false statements can pass a cross-response consistency
test. The label describes the manuscript's operational internal-hallucination
metric. A fresh run against an external model alias is not guaranteed to reproduce
frozen entailment labels. BERTScore precision is a text-similarity metric, not
expert preference or an Elo score.

## Source reconciliation and preserved provenance

`PhytoBench-Core-for_plot-provenance.json` records private workbook hashes,
row counts, scope, metric definitions, model order, and output hashes without
private paths. Fig. 2a deliberately retains the manuscript-aligned Phyto-Chatbot
values 0.720 (identification) and 0.084 (Trace BLEU-4). The current source-workbook
means differ; both computed and aligned values remain documented. Correcting
the source workbooks and accepting a scientific refresh is a separate author
decision. Do not silently replace aligned values during repository cleanup.

Elo, ranking distributions, pairwise probabilities, and expert agreement are
owned by the [ranking aggregate dictionary](<../Supplementary Fig. 12-15/PhytoBench-Gene-for_plot/DATA_DICTIONARY.md>).
They supply Fig. 2d-f and 2g-h (left). Rendering frozen tables, recomputing public
rankings, and rerunning private original evaluations are different workflows.
