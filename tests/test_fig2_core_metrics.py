import hashlib
import importlib
import json
from pathlib import Path

import nbformat
import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "Fig. 2"
NOTEBOOK = FIGURE_DIR / "fig. 2.ipynb"
FREEZER = ROOT / "scripts" / "freeze_fig2_core_metrics.py"
KNOWLEDGE_INPUT = FIGURE_DIR / "PhytoBench-Knowledge-for_plot.tsv"
DATA_INPUT = FIGURE_DIR / "PhytoBench-Data-for_plot.tsv"
ANALYSIS_INPUT = FIGURE_DIR / "PhytoBench-Analysis-for_plot.tsv"
PROVENANCE = FIGURE_DIR / "PhytoBench-Core-for_plot-provenance.json"

MODEL_ORDER = [
    "Phyto-Reasoner",
    "Phyto-Chatbot",
    "GPT-5",
    "o3",
    "Gemini-2.5-Pro",
    "Claude-Opus-4.1",
    "Grok-3-Beta",
    "Deepseek-V3",
    "Deepseek-R1",
]

EXPECTED_IDENTIFICATION_ACCURACY = {
    "Phyto-Reasoner": 0.7757142857142857,
    "Phyto-Chatbot": 0.72,
    "GPT-5": 0.4514285714285714,
    "o3": 0.4114285714285714,
    "Gemini-2.5-Pro": 0.42714285714285716,
    "Claude-Opus-4.1": 0.3914285714285714,
    "Grok-3-Beta": 0.2671428571428571,
    "Deepseek-V3": 0.29285714285714287,
    "Deepseek-R1": 0.2842857142857143,
}
EXPECTED_TRACE_BLEU4 = {
    "Phyto-Reasoner": 0.09010893858062388,
    "Phyto-Chatbot": 0.084,
    "GPT-5": 0.011369437572386475,
    "o3": 0.008644131287009985,
    "Gemini-2.5-Pro": 0.013907710816595295,
    "Claude-Opus-4.1": 0.01524805980535732,
    "Grok-3-Beta": 0.028513309628754507,
    "Deepseek-V3": 0.02114828752802387,
    "Deepseek-R1": 0.017716771685954372,
}
EXPECTED_DATA_ACCURACY = {
    "Phyto-Reasoner": 0.884313725490196,
    "Phyto-Chatbot": 0.8794117647058823,
    "GPT-5": 0.5480392156862746,
    "o3": 0.5588235294117647,
    "Gemini-2.5-Pro": 0.4264705882352941,
    "Claude-Opus-4.1": 0.5558823529411765,
    "Grok-3-Beta": 0.4892156862745098,
    "Deepseek-V3": 0.4872549019607843,
    "Deepseek-R1": 0.5137254901960784,
}
EXPECTED_ANALYSIS_MEAN_TOTAL_SCORES = {
    "Phyto-Reasoner": 71.93,
    "Phyto-Chatbot": 63.94,
    "GPT-5": 68.225,
    "o3": 60.59,
    "Gemini-2.5-Pro": 61.38666666666667,
    "Claude-Opus-4.1": 69.79,
    "Grok-3-Beta": 58.4,
    "Deepseek-V3": 59.54333333333334,
    "Deepseek-R1": 59.515,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_fig2_core_metric_freezer_is_importable_and_declares_outputs() -> None:
    assert FREEZER.is_file(), "The Fig. 2a-c freezer script is missing."
    module = importlib.import_module("scripts.freeze_fig2_core_metrics")
    assert module.OUTPUT_FILENAMES == {
        "knowledge": KNOWLEDGE_INPUT.name,
        "data": DATA_INPUT.name,
        "analysis": ANALYSIS_INPUT.name,
        "provenance": PROVENANCE.name,
    }


def test_fig2_core_frozen_inputs_match_locked_publication_values() -> None:
    required = [KNOWLEDGE_INPUT, DATA_INPUT, ANALYSIS_INPUT, PROVENANCE]
    missing = [path.name for path in required if not path.is_file()]
    assert not missing, f"Missing Fig. 2a-c frozen inputs: {missing}"

    knowledge = pd.read_csv(KNOWLEDGE_INPUT, sep="\t")
    assert knowledge["Model"].tolist() == MODEL_ORDER
    assert knowledge["IdentificationN"].tolist() == [700] * len(MODEL_ORDER)
    assert knowledge["TraceN"].tolist() == [300] * len(MODEL_ORDER)
    for row in knowledge.itertuples(index=False):
        assert row.IdentificationAccuracy == pytest.approx(
            EXPECTED_IDENTIFICATION_ACCURACY[row.Model]
        )
        assert row.TraceBLEU4 == pytest.approx(EXPECTED_TRACE_BLEU4[row.Model])

    data = pd.read_csv(DATA_INPUT, sep="\t")
    assert data["Model"].tolist() == MODEL_ORDER
    assert data["N"].tolist() == [1020] * len(MODEL_ORDER)
    for row in data.itertuples(index=False):
        assert row.Accuracy == pytest.approx(EXPECTED_DATA_ACCURACY[row.Model])

    analysis = pd.read_csv(ANALYSIS_INPUT, sep="\t")
    assert analysis.shape == (450, 11)
    assert analysis["Model"].drop_duplicates().tolist() == MODEL_ORDER
    assert analysis.groupby("Model", sort=False).size().tolist() == [50] * 9
    assert analysis["ObservationID"].nunique() == 50
    assert not analysis.duplicated(["ObservationID", "Model"]).any()
    assert analysis["Species"].eq("Oryza_sativa").all()
    assert analysis["Task"].nunique() == 10
    assert analysis.groupby(["Model", "Task"]).size().eq(5).all()
    component_columns = [
        "PlanScore",
        "ToolScore",
        "ParameterScore",
        "RateScore",
    ]
    assert analysis[component_columns].ge(0).all().all()
    assert analysis[component_columns].le(25).all().all()
    assert analysis["TotalScore"].between(0, 100).all()
    pd.testing.assert_series_equal(
        analysis[component_columns].sum(axis=1),
        analysis["TotalScore"],
        check_names=False,
    )
    means = analysis.groupby("Model", sort=False)["TotalScore"].mean()
    for model, expected in EXPECTED_ANALYSIS_MEAN_TOTAL_SCORES.items():
        assert means.loc[model] == pytest.approx(expected)

    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert provenance["schema_version"] == 2
    assert provenance["source_rows"] == {
        "knowledge_identification": 700,
        "knowledge_trace": 300,
        "data": 1020,
        "analysis": 110,
    }
    assert provenance["analysis_scope"] == {
        "species": "Oryza_sativa",
        "tasks": 10,
        "repeats_per_task": 5,
        "rows": 50,
        "excluded_cross_species_rows": 60,
    }
    alignment = provenance["manuscript_alignment"]
    assert alignment["panel"] == "Fig. 2a"
    assert alignment["source_document"] == (
        "2025-11-31329A-Z_Article_File-20260727.working.md"
    )
    assert alignment["source_sha256"] == (
        "2e25c9e62d396c22ef764f8f70cccfdb143617de0f73361de80f95e4613ec964"
    )
    assert alignment["overrides"]["Phyto-Chatbot"] == {
        "IdentificationAccuracy": {
            "source_workbook_value": pytest.approx(0.6328571428571429),
            "manuscript_value": 0.72,
        },
        "TraceBLEU4": {
            "source_workbook_value": pytest.approx(0.06023096459723756),
            "manuscript_value": 0.084,
        },
    }
    assert provenance["outputs"] == {
        KNOWLEDGE_INPUT.name: _sha256(KNOWLEDGE_INPUT),
        DATA_INPUT.name: _sha256(DATA_INPUT),
        ANALYSIS_INPUT.name: _sha256(ANALYSIS_INPUT),
    }


def test_fig2_notebook_reads_core_frozen_inputs_without_inline_arrays() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    for filename in (
        KNOWLEDGE_INPUT.name,
        DATA_INPUT.name,
        ANALYSIS_INPUT.name,
    ):
        assert filename in source
    assert "gi_y_list = [" not in source
    assert "bi_list = [" not in source
    assert "y=[50,25,50" not in source
    assert "GoalCompletionPercent" not in source
    assert "TotalScore" in source
    assert "Total score (0–100)" in source
    assert "width=2100" in source
    assert "height=784" in source


def test_fig2_manifest_requires_core_frozen_inputs() -> None:
    manifest = yaml.safe_load(
        (ROOT / "reproduce.manifest.yaml").read_text(encoding="utf-8")
    )
    target = next(item for item in manifest["targets"] if item["id"] == "fig-2")
    required = {Path(path).name for path in target["requires_data"]}
    assert {
        KNOWLEDGE_INPUT.name,
        DATA_INPUT.name,
        ANALYSIS_INPUT.name,
        PROVENANCE.name,
    } <= required
