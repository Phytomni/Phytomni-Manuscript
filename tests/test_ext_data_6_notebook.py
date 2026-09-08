import ast
import hashlib
import json
import re
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "Extended Data Fig. 6-7"
KNOWLEDGE_NOTEBOOK = FIGURE_DIR / "extended_data_fig. 6ab-7ab.ipynb"
ANALYST_NOTEBOOK = FIGURE_DIR / "extended_data_fig. 6de-7de.ipynb"


def notebook_source(path: Path) -> str:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    return "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )


def inline_r_table(source: str, variable: str) -> dict[str, list]:
    """Read the notebook's literal, multiline R vectors without starting R."""
    match = re.search(
        rf"{re.escape(variable)} <- data.frame\((.*?)stringsAsFactors = FALSE",
        source,
        flags=re.DOTALL,
    )
    assert match is not None, f"R table {variable} is missing"
    vectors = re.findall(
        r"^\s*([A-Za-z0-9_.]+|\x60[^\x60]+\x60)\s*=\s*c\((.*?)^\s*\),",
        match.group(1),
        flags=re.MULTILINE | re.DOTALL,
    )
    result = {
        name.strip("\x60"): ast.literal_eval(f"[{values}]")
        for name, values in vectors
    }
    assert result, f"No literal vectors in {variable}"
    assert len({len(values) for values in result.values()}) == 1
    return result


def content_hash(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def test_knowledge_tables_preserve_the_accepted_inline_values() -> None:
    source = notebook_source(KNOWLEDGE_NOTEBOOK)
    bar = inline_r_table(source, "data_selected")
    radar = inline_r_table(source, "data_selected_radar")

    assert content_hash(bar) == (
        "8a0d53c8794b52a133092fe4ba8e28dbffb9731a52d9b1379fa604559e2a9806"
    )
    assert content_hash(radar) == (
        "ea5fa9d641fe4a1f74c3881a1075010b60443f4251fadcfd7678bdeef7ae52a3"
    )
    assert len(bar["model"]) == 23
    assert len(radar["model_name"]) == 26
    assert set(bar) == {
        "model", "figure1", "figure2", "Arabidopsis.thaliana", "Oryza.sativa",
        "Zea.mays", "Glycine.max", "Triticum.Aestivum", "Mix",
    }
    assert set(radar) == {
        "model_name", "figure1", "figure2", "BLEU-1", "BLEU-2", "BLEU-3",
        "BLEU-4", "ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1",
        "BERTScore Precision", "BERTScore F1", "BERTScore Recall",
    }
    for table, model_column in ((bar, "model"), (radar, "model_name")):
        for flag, expected_models in (("figure1", 9), ("figure2", 16)):
            selected = {
                model for model, enabled in zip(table[model_column], table[flag])
                if enabled == "yes"
            }
            assert len(selected) == expected_models


def test_knowledge_selection_and_metric_validation_remain_explicit() -> None:
    source = notebook_source(KNOWLEDGE_NOTEBOOK)
    assert 'filter(!!sym(fig_flag) == "yes")' in source
    assert "select(group, all_of(radar_metrics))" in source
    assert "scale_fill_manual(values = target_colors[current_models])" in source
    assert "mutate(across(-group, ~ rescale(.)))" in source
    assert "read_excel" not in source


def test_knowledge_notebook_saves_all_four_current_panels() -> None:
    source = notebook_source(KNOWLEDGE_NOTEBOOK)
    assert 'save_figs <- Sys.getenv("PHYTOMNI_SAVE") == "1"' in source
    for panel in ("6a", "6b", "7a", "7b"):
        assert f'if (save_figs) ggsave("output/extended_data_fig.{panel}.pdf"' in source
    assert "ir_r_env" not in KNOWLEDGE_NOTEBOOK.read_text()


def test_analyst_tables_preserve_every_score_and_accepted_scope() -> None:
    source = notebook_source(ANALYST_NOTEBOOK)
    tree = ast.parse(source)
    tables = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name):
            name = statement.targets[0].id
            if name.endswith("_plot_data"):
                tables[name] = ast.literal_eval(statement.value)
    assert len(tables) == 8
    assert content_hash(tables) == (
        "1ebe8f395824f538e24fe5bb68b05e4333e00a17c39fb99993c3c3e62867d98c"
    )
    for name, values in tables.items():
        assert len(values) == 15
        assert {len(scores) for scores in values.values()} == (
            {5} if name.startswith("spa_") else {10}
        )
    assert source.count('"plan_score": plot_plan_df.loc[model_index] * 4') == 4
    assert source.count("cmap=tropic_cmap") == 4
    orders = [
        ast.literal_eval(statement.value)
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "species_order"
    ]
    assert len(orders) == 4
    assert orders[0] == orders[2] == [
        "Oryza sativa", "Triticum aestivum", "Zea mays",
        "Glycine max", "Arabidopsis thaliana",
    ]


def test_analyst_notebook_saves_comparison_and_ablation_panels() -> None:
    source = notebook_source(ANALYST_NOTEBOOK)
    for panel in ("6d", "6e", "7d", "7e"):
        assert f'if SAVE_FIGS:\n    fig.savefig("output/extended_data_fig.{panel}.pdf")' in source
    assert source.count("figsize=(8, 12), dpi=300") == 2
    assert source.count("figsize=(30, 10), dpi=300") == 2
