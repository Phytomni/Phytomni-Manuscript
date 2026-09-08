"""Public Plotly figures do not need MathJax or its asynchronous loading footer."""

import ast
from pathlib import Path

import nbformat
import plotly.io as pio
import pytest
from IPython.core.inputtransformer2 import TransformerManager

from scripts.reproduce_lib import load_manifest


ROOT = Path(__file__).resolve().parents[1]


def _code_tree(source):
    return ast.parse(TransformerManager().transform_cell(source))


def _plotly_export_notebooks():
    paths = dict.fromkeys(
        target["path"]
        for target in load_manifest(ROOT / "reproduce.manifest.yaml")["targets"]
        if target["phase"] == "figure" and target["kind"] == "notebook"
        and target.get("kernel") == "py"
    )
    for path in paths:
        notebook = nbformat.read(ROOT / path, as_version=4)
        trees = [_code_tree(cell.source) for cell in notebook.cells if cell.cell_type == "code"]
        if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "write_image" for tree in trees for node in ast.walk(tree)):
            yield path


@pytest.mark.parametrize("path", list(_plotly_export_notebooks()))
def test_export_setup_effectively_disables_unused_mathjax(path, monkeypatch):
    notebook = nbformat.read(ROOT / path, as_version=4)
    assignments = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for node in _code_tree(cell.source).body:
            if isinstance(node, ast.Assign) and any(
                ast.unparse(target) in {"pio.kaleido.scope.mathjax", "pio.kaleido.scope.default_format"}
                for target in node.targets
            ):
                assignments.append(node)
    assert any(ast.unparse(target) == "pio.kaleido.scope.mathjax"
               for assignment in assignments for target in assignment.targets), path
    monkeypatch.setattr(pio.kaleido.scope, "mathjax", "file:///unused-mathjax-sentinel.js")
    monkeypatch.setattr(pio.kaleido.scope, "default_format", "png")
    setup = ast.fix_missing_locations(ast.Module(body=assignments, type_ignores=[]))
    exec(compile(setup, path, "exec"), {"pio": pio})
    assert pio.kaleido.scope.mathjax is None
    assert pio.kaleido.scope.default_format == "pdf"
    assert not any(arg.startswith("--mathjax") for arg in pio.kaleido.scope._build_proc_args())
