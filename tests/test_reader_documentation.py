import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import shlex
import yaml

from scripts.freeze_deepgenome_rankings import (
    OUTPUT_SCHEMAS,
    OUTPUT_SORT_KEYS,
    OUTPUT_UNIQUE_KEYS,
)


ROOT = Path(__file__).resolve().parents[1]


def test_root_guide_commands_use_actual_target_ids_and_parsers():
    from scripts.reproduce_cli import parse_request

    document = (ROOT / "README.md").read_text()
    manifest = yaml.safe_load((ROOT / "reproduce.manifest.yaml").read_text())
    commands = [line for line in document.splitlines() if line.startswith("./reproduce.sh ")]
    assert len(commands) == 5
    parsed = [parse_request(shlex.split(line)[1:], manifest) for line in commands]
    assert [item.command for item in parsed] == ["list", "validate", "run", "run", "probe"]
    assert 'source .venv/bin/activate' in document
    assert './reproduce.sh --check' not in document


def test_ranking_dictionary_tracks_exact_serialized_fields_and_keys():
    document = (
        ROOT / "Supplementary Fig. 12-15/PhytoBench-Gene-for_plot/DATA_DICTIONARY.md"
    ).read_text()
    for name, columns in OUTPUT_SCHEMAS.items():
        section = document.split(f"### `{name}.tsv`\n", 1)[1].split("\n### ", 1)[0]
        for label, expected in (
            ("Columns", columns),
            ("Unique key", OUTPUT_UNIQUE_KEYS[name]),
            ("Sort keys", OUTPUT_SORT_KEYS[name]),
        ):
            line = next(line for line in section.splitlines() if line.startswith(label + ":"))
            assert tuple(re.findall(r"`([^`]+)`", line)) == expected


def test_fig2_dictionary_names_every_public_metric_field():
    document = (ROOT / "Fig. 2/DATA_DICTIONARY.md").read_text()
    tables = list((ROOT / "Fig. 2").glob("*for_plot.tsv"))
    tables += list((ROOT / "Fig. 2/PhytoBench-Gene-metrics-for_plot").glob("*.tsv"))
    assert len(tables) == 8
    for table in tables:
        assert f"`{table.name}`" in document
        for column in pd.read_csv(table, sep="\t", nrows=0).columns:
            assert f"`{column}`" in document


def test_pairwise_dictionary_describes_blank_self_comparisons():
    root = ROOT / "Supplementary Fig. 12-15/PhytoBench-Gene-for_plot"
    document = (root / "DATA_DICTIONARY.md").read_text()
    point = pd.read_csv(root / "frozen/pl_pairwise.tsv", sep="\t")
    self_rows = point.loc[point["RowModel"] == point["ColumnModel"], "Probability"]
    assert len(self_rows) and self_rows.isna().all()
    assert "blank/NaN" in document
    assert "0.5 diagonal" not in document


@pytest.mark.parametrize("script", ["evaluation_id.py", "evaluation_trace.py", "evaluation_review.py"])
def test_documented_evaluation_help_works_without_runtime_or_input(tmp_path, script):
    result = subprocess.run(
        [sys.executable, "-S", str(ROOT / "Knowledge&ReviewAgent Evaluation" / script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--input" in result.stdout and "--output-dir" in result.stdout
    assert list(tmp_path.iterdir()) == []
