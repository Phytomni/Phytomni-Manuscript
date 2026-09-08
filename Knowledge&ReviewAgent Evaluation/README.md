# Knowledge&ReviewAgent Evaluation

This is an evaluation workflow, not a public figure-rendering target. Commands
below use the repository root unless stated otherwise. See the
[repository guide](../README.md) for figure reproduction.

`Knowledge&ReviewAgent Evaluation/` contains CLI evaluators (`evaluation_id.py`, `evaluation_trace.py`) for gene-ID and trace-QA benchmarks. Both require a **`--input` dataset** (`.xlsx`/`.csv`) that is **not shipped in this repo** — supply your own benchmark file. They use pandas/matplotlib (available in the figure Python env) but cannot produce manuscript figures without your input data and model outputs.


## Review and trace metrics

`evaluation_review.py` evaluates review answers against a supplied reference:
BLEU, ROUGE, BERTScore, embedding similarity, and optional BLEURT. It writes
checkpoints, detailed metric tables, summaries, and optional PNG plots to the
required `--output-dir`. `evaluation_trace.py` has a similar interface for
trace-QA answers. Input columns are selected explicitly through `--models`,
`--reference-column` (default `answer`), and `--id-column` (default `ID`).
Neither evaluator generates missing benchmark answers.

Beyond the base figure environment, these neural evaluators require `nltk`,
`regex`, `rouge`, `scikit-learn`, `tqdm`, `torch`, `transformers`, and `bert-score`.
BLEURT additionally requires its package and a supplied checkpoint. These are
not automatically installed by figure reproduction. The DeepGenome extra
covers only some of these dependencies, not the entire review runtime.

Inspect the CLI without loading models or reading a private dataset:

```bash
python "Knowledge&ReviewAgent Evaluation/evaluation_id.py" --help
python "Knowledge&ReviewAgent Evaluation/evaluation_trace.py" --help
python "Knowledge&ReviewAgent Evaluation/evaluation_review.py" --help
```

A neural-metric run may download encoder weights and use a GPU; time and memory
depend on the model and response lengths. Prepare the encoder and tokenizer
locally before an offline run, and set Hugging Face offline mode as well as
`--local-files-only`; the BERTScore loader is a separate model-loading path.
Record the exact checkpoint, layer settings, input hashes, and runtime versions
for comparison. In `evaluation_review.py`, BERTScore uses layer 9 by default.
An omitted BLEURT checkpoint produces a zero placeholder, not a measured zero
quality score. Do not interpret that column as an evaluated metric.

The command-specific checkpoints are useful for resuming a run, but they are
not a replacement for the tracked frozen figure-input tables. Keep private
responses and generated outputs outside Git, for example in `local_exports/`.
