"""Locked numerical and rendering boundaries for the SeedBench source refresh."""
import hashlib
import json
from pathlib import Path

import nbformat
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
FIGURE = ROOT / 'Supplementary Fig. 19'
MODELS = ['Phyto-Reasoner', 'GPT-5', 'o3', 'Gemini-2.5-Pro', 'Claude-Opus-4.1',
          'DeepSeek-R1', 'DeepSeek-V3', 'Grok-3-Beta']
EXPECTED = {
    'QA-1': [0.7447, 0.7021, 0.7021, 0.7021, 0.6979, 0.6851, 0.6766, 0.6298],
    'QA-2': [0.8403, 0.7807, 0.7602, 0.7612, 0.7679, 0.7471, 0.7591, 0.7550],
    'QA-3': [0.4150, 0.3776, 0.3816, 0.3586, 0.3710, 0.3112, 0.3149, 0.2575],
    'QA-4': [0.4202, 0.2095, 0.2030, 0.1681, 0.2497, 0.2580, 0.3478, 0.0487],
}


def test_seedbench_frozen_scores_preserve_qa12_and_reconcile_qa34():
    table = pd.read_csv(FIGURE / 'SeedBench-for_plot.tsv', sep='\t')
    assert list(table.columns) == ['Task', 'Metric', 'Model', 'Score', 'N']
    assert table.shape == (32, 5)
    assert not table.duplicated(['Task', 'Model']).any()
    for task, n in zip(EXPECTED, [235, 218, 269, 277], strict=True):
        rows = table.loc[table.Task.eq(task)]
        assert rows.Model.tolist() == MODELS
        assert rows.Score.tolist() == pytest.approx(EXPECTED[task], abs=1e-12)
        assert rows.N.tolist() == [n] * 8


def test_seedbench_notebook_reads_frozen_data_and_preserves_plot_geometry():
    notebook = nbformat.read(FIGURE / 'supplementary_fig. 19.ipynb', as_version=4)
    nbformat.validate(notebook)
    code = '\n'.join(c.source for c in notebook.cells if c.cell_type == 'code')
    assert 'SeedBench-for_plot.tsv' in code
    assert 'tribble' not in code
    assert '0.4061' not in code and '0.4589' not in code
    assert 'Claude-4.1-Opus' not in code and '"O3"' not in code
    for token in ['geom_bar(', 'ncol = 4', 'limits = c(0, 1)',
                  'width = 14, height = 8', 'output/supplementary_fig.19.pdf',
                  'AA-Omniscience-for_plot.tsv', 'output/supplementary_fig.19b.pdf']:
        assert token in code
    for color in ['#1e72b2', '#F2C6DE', '#F29C9A', '#E59E65', '#D6BA9E',
                  '#9C9A9C', '#4D9489', '#96CDA0']:
        assert color in code
    assert all(not c.outputs and c.execution_count is None
               for c in notebook.cells if c.cell_type == 'code')


def test_aa_omniscience_frozen_table_reconciles_rates_and_manifest() -> None:
    table = pd.read_csv(FIGURE / 'AA-Omniscience-for_plot.tsv', sep='\t')
    assert table.shape == (8, 9)
    assert table.Model.tolist() == [
        'Phyto-Reasoner', 'Claude-Opus-4.1', 'GPT-5', 'DeepSeek-V3',
        'o3', 'Grok-3-Beta', 'Gemini-2.5-Pro', 'DeepSeek-R1']
    expected = table.Incorrect / (table.Incorrect + table.Partial + table.Not_attempted)
    assert table.Hallucination_rate.tolist() == pytest.approx(expected.tolist(), abs=1e-12)
    assert table.Hallucination_rate_percent.tolist() == [67, 67, 75, 75, 100, 100, 100, 100]
    manifest = yaml.safe_load((ROOT / 'reproduce.manifest.yaml').read_text())
    target = next(t for t in manifest['targets'] if t['id'] == 'supp-19')
    assert target['requires_data'] == [
        'Supplementary Fig. 19/SeedBench-for_plot.tsv',
        'Supplementary Fig. 19/SeedBench-for_plot-provenance.json',
        'Supplementary Fig. 19/AA-Omniscience-for_plot.tsv',
        'Supplementary Fig. 19/AA-Omniscience-for_plot-provenance.json']
    assert target['expected_artifacts'] == [
        'Supplementary Fig. 19/output/supplementary_fig.19.pdf',
        'Supplementary Fig. 19/output/supplementary_fig.19b.pdf']
    provenance = json.loads((FIGURE / 'AA-Omniscience-for_plot-provenance.json').read_text())
    assert provenance['source_sha256'] == 'd29546b4ec774fb50237d311b0225820c3f4e0311ea1f77fdec0f17ce2cd3ce6'
    assert provenance['input_sha256'] == hashlib.sha256((FIGURE / 'AA-Omniscience-for_plot.tsv').read_bytes()).hexdigest()


def test_seedbench_manifest_requires_frozen_table_and_provenance():
    manifest = yaml.safe_load((ROOT / 'reproduce.manifest.yaml').read_text())
    target = next(t for t in manifest['targets'] if t['id'] == 'supp-19')
    assert target['requires_data'] == [
        'Supplementary Fig. 19/SeedBench-for_plot.tsv',
        'Supplementary Fig. 19/SeedBench-for_plot-provenance.json',
        'Supplementary Fig. 19/AA-Omniscience-for_plot.tsv',
        'Supplementary Fig. 19/AA-Omniscience-for_plot-provenance.json',
    ]
    provenance = (FIGURE / 'SeedBench-for_plot-provenance.json').read_text()
    assert hashlib.sha256((FIGURE / 'SeedBench-for_plot.tsv').read_bytes()).hexdigest() in provenance
    json.loads(provenance)
    for private in ['/home/', 'local_exports', '1-3_0001', '1-4_0001']:
        assert private not in provenance
