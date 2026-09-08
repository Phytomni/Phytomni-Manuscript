# DeepGenomeAgent Evaluation

This directory contains evaluation notebooks, not figure-rendering targets.
Commands below run from the repository root. For rendering existing results,
use the [repository guide](../README.md), [Fig. 2 input guide](<../Fig. 2/README.md>),
or [ranking-data guide](<../Supplementary Fig. 12-15/PhytoBench-Gene-for_plot/README.md>).

## Public ranking data and numerical recomputation

The [anonymized scoring table](supplementary/Supplementary_Data_Expert_Rankings.tsv)
contains 600 expert–gene assignments from 120 experts and 200 genes. Each row
is one complete, no-tie R1–R5 ranking of five models. The
[codebook](supplementary/Supplementary_Data_Expert_Rankings_Codebook.tsv) defines
the fields; the [public category map](supplementary/expert_panel_category_map.tsv)
describes privacy-reviewed demographic aggregation, not row-level demographics.

`scripts.recompute_deepgenome_rankings` recomputes 11 numerical and assignment
tables using only this public scoring table and the public gene-category map.
It does not require private metadata, lineage files, responses, judgment logs,
credentials, or model services. See the
[public recomputation command](<../Supplementary Fig. 12-15/PhytoBench-Gene-for_plot/README.md#recompute-public-ranking-statistics>)
for a 100-replicate preview and the separately authorized 10,000-replicate full
comparison. Both write independent nonpublication outputs, never the accepted
frozen snapshot. Public recomputation cannot reconstruct individual demographics
or the private-panel aggregation workflow; the released panel summary remains
a frozen public figure input.

## Plackett–Luce scoring notebook

`score_plackett_luce.ipynb` converts expert rankings into Plackett–Luce
log-strengths, Elo-like scores and confidence intervals, and pairwise win
probabilities. Its default models are `Gemini`, `Grok`, `OpenAI`, `Phytomni`, and
`Claude`; the likelihood supports any configured count of at least two models.
The manuscript recomputation and publication workflows deliberately require the
fixed five-model public-release schema and preserve its anonymous expert IDs.

For an offline repository check, use the locked base environment. Without a
private score path, the notebook performs deterministic checks, reports
`SKIPPED`, and does not fit or invent benchmark results:

```bash
uv sync --frozen
uv run --frozen --no-sync pytest tests/test_deepgenome_scoring_notebooks.py -v
mkdir -p /tmp/deepgenome-notebook-runs
uv run --frozen --no-sync jupyter nbconvert --to notebook --execute \
  --output-dir=/tmp/deepgenome-notebook-runs \
  "DeepGenomeAgent Evaluation/score_plackett_luce.ipynb"
```

Set `DEEPGENOME_SCORE_TSV=/absolute/path/to/score.tsv` to fit a private ranking
table. Set `DEEPGENOME_MODEL_COLUMNS=Gemini,Claude,Model_X` for a different
comma-separated model set, retaining `Gemini` as the reference model.
`DEEPGENOME_EXPORT_DIR=/absolute/path/to/export-dir` opts into writing aggregate
TSVs; exports are disabled by default and should use a temporary or explicitly
managed data directory. This configurable notebook workflow is distinct from
the strict public-cohort CLI and the
[author-only publication refresh](<../Supplementary Fig. 12-15/PhytoBench-Gene-for_plot/README.md#author-only-publication-refresh>).

## Cross-response hallucination scoring

`score_hallucination.ipynb` measures cross-response inconsistency. For each
gene and model, it compares three repeated responses over every ordered pair,
records window-level entailment judgments, and clusters mutually entailing
responses to calculate normalized semantic entropy. Structurally complete logs
are summarized with the primary `mean_directional_contradiction_ratio` and
inclusive-threshold `high_contradiction_gene_fraction`.

Neither cross-response consistency nor semantic entropy directly verifies
factual truth: a false claim repeated consistently across all three responses
can score as consistent. Exact numerical reproduction requires the recorded
judgment logs. Rerunning a drifting external judge alias can change entailment
labels and therefore does not guarantee the same results. The private query
workbook, response corpus, and judgment logs are not shipped. Compact frozen
Claude metrics used by the figures are documented in the
[Fig. 2 input guide](<../Fig. 2/README.md#gene-metric-data-ownership>).

### Aggregate recorded judgment logs

The default notebook execution is offline. Its optional analysis dependencies
are installed with `deepgenome-eval`. To aggregate existing logs without
contacting a service, leave live judging disabled and provide the log directory:

```bash
uv sync --frozen --extra deepgenome-eval
mkdir -p /tmp/deepgenome-notebook-runs
DEEPGENOME_JUDGMENT_DIR=/absolute/path/to/frozen/judgment-logs \
  uv run --frozen --no-sync --extra deepgenome-eval jupyter nbconvert \
  --to notebook --execute --output-dir=/tmp/deepgenome-notebook-runs \
  "DeepGenomeAgent Evaluation/score_hallucination.ipynb"
```

### Run a new live evaluation

This is a separate, explicitly opted-in evaluation, not a prerequisite for
public figure reproduction. Install the extra and NLTK sentence tokenizer,
create the private log directory, and provide the seven settings below:

```bash
uv sync --frozen --extra deepgenome-eval
uv run --frozen --no-sync --extra deepgenome-eval python -m nltk.downloader punkt_tab
mkdir -p /absolute/path/to/judgment-logs
mkdir -p /tmp/deepgenome-notebook-runs
DEEPGENOME_QUERY_WORKBOOK=/absolute/path/to/queries.xlsx \
DEEPGENOME_RESPONSE_ROOT=/absolute/path/to/response-corpus \
DEEPGENOME_JUDGMENT_DIR=/absolute/path/to/judgment-logs \
DEEPGENOME_API_BASE_URL=https://judge.example/v1 \
DEEPGENOME_API_KEY=replace-with-private-key \
DEEPGENOME_JUDGE_MODEL=replace-with-pinned-model \
DEEPGENOME_RUN_LIVE_JUDGING=1 \
  uv run --frozen --no-sync --extra deepgenome-eval jupyter nbconvert \
  --to notebook --execute --output-dir=/tmp/deepgenome-notebook-runs \
  "DeepGenomeAgent Evaluation/score_hallucination.ipynb"
```

With live judging disabled, a missing judgment directory or absence of valid
logs reports `SKIPPED` instead of fabricating a score. With
`DEEPGENOME_RUN_LIVE_JUDGING=1`, missing private inputs or API settings, or an
unavailable NLTK `punkt_tab`, raises an actionable error and fails before
constructing the judge client. Each live log records a sanitized API base URL,
judge model, UTC timestamp, relevant package versions, and SHA-256 checksums
for the question and ordered responses. It never serializes the API key or
URL credentials, query, or fragment. Keep real credentials out of tracked files.

The examples write executed copies to
`/tmp/deepgenome-notebook-runs/score_plackett_luce.ipynb` and
`/tmp/deepgenome-notebook-runs/score_hallucination.ipynb`. The tracked canonical
notebooks remain output-free; none of these examples executes them in place.
