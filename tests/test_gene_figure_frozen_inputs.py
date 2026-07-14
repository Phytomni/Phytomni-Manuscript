import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "Supplementary Fig. 10-13"
DATA_DIR = FIGURE_DIR / "PhytoBench-Gene-for_plot" / "frozen"
NOTEBOOK = FIGURE_DIR / "supplementary_fig. 10-13.ipynb"
FIG2_DIR = ROOT / "Fig. 2"
FIG2_NOTEBOOK = FIG2_DIR / "fig. 2.ipynb"
FIG2_BERTSCORE = FIG2_DIR / "PhytoBench-Gene-BERTScore-for_plot.tsv"
MODEL_ORDER = ["Phytomni", "Gemini", "Claude", "OpenAI", "Grok"]
SOURCE_SHA256 = "bf24408d8e3d68ca11cc7319b25c407a29d2e26301fdc812896dd451818adcbe"


def notebook_source() -> str:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    assert all(
        cell.execution_count is None and not cell.outputs
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    return "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )


def test_frozen_tables_are_complete_and_traceable() -> None:
    provenance = json.loads((DATA_DIR / "provenance.json").read_text())
    ranks = pd.read_csv(DATA_DIR / "rank_distribution.tsv", sep="\t")
    scores = pd.read_csv(DATA_DIR / "pl_scores.tsv", sep="\t")
    pairwise = pd.read_csv(DATA_DIR / "pl_pairwise.tsv", sep="\t")

    assert provenance["source"]["sha256"] == SOURCE_SHA256
    assert provenance["source"]["rows"] == 600
    assert provenance["used_rows"] == 600
    assert provenance["skipped_rows"] == 0
    assert provenance["model_columns"] == [
        "Gemini",
        "Grok",
        "OpenAI",
        "Phytomni",
        "Claude",
    ]
    assert ranks["Scope"].nunique() == 18
    assert len(ranks) == 18 * 5 * 5
    assert len(scores) == 18 * 5
    assert len(pairwise) == 18 * 5 * 5
    assert set(ranks["Model"]) == set(MODEL_ORDER)
    np.testing.assert_allclose(
        ranks.groupby(["Scope", "Model"])["Fraction"].sum(),
        1.0,
    )

    overall = scores[scores["Scope"] == "overall"].set_index("Model")
    np.testing.assert_allclose(
        overall.loc[MODEL_ORDER, "Elo"],
        [1612.2511259530193, 1547.6162663668567, 1532.4668216858088,
         1487.5809511985851, 1320.0848347957306],
        atol=1e-10,
    )


def test_supplementary_notebook_only_plots_frozen_five_model_results() -> None:
    source = notebook_source()

    for filename in (
        "rank_distribution.tsv",
        "pl_scores.tsv",
        "pl_pairwise.tsv",
        "provenance.json",
    ):
        assert filename in source
    for model in MODEL_ORDER:
        assert f'"{model}"' in source
    assert "calculate_pl_elo" not in source
    assert "pl_loglik_and_grad" not in source
    assert "sp[-4:]" not in source
    assert "np.ix_([3, 0, 2, 1]" not in source
    assert 'y=100.0 * rank_matrix[rank_label]' in source
    assert "PhytoBench-Gene-for_plot/score.tsv" not in source


def test_supplementary_notebook_uses_requested_claude_label_and_order() -> None:
    source = notebook_source()

    assert 'MODEL_ORDER = ["Phytomni", "Gemini", "Claude", "OpenAI", "Grok"]' in source
    assert '"Claude": "Claude deep research"' in source


def test_fig2g_reads_frozen_metric_and_fig2h_is_pending() -> None:
    bertscore = pd.read_csv(FIG2_BERTSCORE, sep="\t")
    assert bertscore.columns.tolist() == [
        "Model",
        "DisplayLabel",
        "BERTScorePrecision",
    ]
    assert bertscore["Model"].tolist() == [
        "Phytomni",
        "Gemini",
        "OpenAI",
        "Grok",
    ]

    notebook = nbformat.read(FIG2_NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    assert all(
        cell.source.strip()
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    assert "PhytoBench-Gene-BERTScore-for_plot.tsv" in source
    assert "PhytoBench-Gene-hallucination-for_plot.tsv" in source
    assert "if HALLUCINATION_DATA.is_file():" in source
    assert "SKIPPED: Fig. 2h" in source
    assert "0.561641241" not in source
    assert "0.12216996785802783" not in source


def test_manifest_connects_fig2def_and_marks_fig2h_pending() -> None:
    import yaml

    manifest = yaml.safe_load((ROOT / "reproduce.manifest.yaml").read_text())
    targets = {target["id"]: target for target in manifest["targets"]}

    fig2def = targets["fig-2def"]
    assert fig2def["path"] == (
        "Supplementary Fig. 10-13/supplementary_fig. 10-13.ipynb"
    )
    assert set(fig2def["requires_data"]) == {
        "Supplementary Fig. 10-13/PhytoBench-Gene-for_plot/frozen/"
        "rank_distribution.tsv",
        "Supplementary Fig. 10-13/PhytoBench-Gene-for_plot/frozen/"
        "pl_scores.tsv",
        "Supplementary Fig. 10-13/PhytoBench-Gene-for_plot/frozen/"
        "pl_pairwise.tsv",
        "Supplementary Fig. 10-13/PhytoBench-Gene-for_plot/frozen/"
        "provenance.json",
    }
    assert len(fig2def["expected_artifacts"]) == 3

    fig2h = targets["fig-2h"]
    assert fig2h["status"] == "skip_until_data"
    assert fig2h["requires_data"] == [
        "Fig. 2/PhytoBench-Gene-BERTScore-for_plot.tsv",
        "Fig. 2/PhytoBench-Gene-hallucination-for_plot.tsv"
    ]
    fig2 = targets["fig-2"]
    assert all("fig.2h" not in path for path in fig2["expected_artifacts"])
