"""Reader-facing contracts for the completed figure-notebook cleanup."""

import io
import re
import tokenize
from pathlib import Path

import nbformat
import pytest
import yaml
from pygments import lex
from pygments.lexers import get_lexer_by_name
from pygments.token import Comment


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = yaml.safe_load((ROOT / "reproduce.manifest.yaml").read_text())
NOTEBOOK_PATHS = sorted({
    target["path"] for target in MANIFEST["targets"]
    if target["phase"] == "figure" and target["kind"] == "notebook"
})
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def r_comments(source: str) -> list[str]:
    """Use the installed R lexer so quoted research data are not comments."""
    return [value for token, value in lex(source, get_lexer_by_name("r")) if token in Comment]


def test_r_comment_detection_preserves_quoted_research_data() -> None:
    source = 'x <- "raw # 数据"\ny <- "multiline\n# 原始数据"\nz <- 1 # authored comment\n# standalone comment\n'
    assert r_comments(source) == ["# authored comment", "# standalone comment"]


@pytest.mark.parametrize("relative", NOTEBOOK_PATHS)
def test_figure_sources_are_clean_and_explain_reproduction(relative: str) -> None:
    notebook = nbformat.read(ROOT / relative, as_version=4)
    nbformat.validate(notebook)
    assert all(cell.source.strip() for cell in notebook.cells)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []
    assert notebook.cells[0].cell_type == "markdown"
    opening = notebook.cells[0].source.lower()
    for topic in ("purpose", "inputs", "run", "outputs", "provenance"):
        assert re.search(rf"\b{topic}\b", opening), f"Missing {topic} guide in {relative}"


@pytest.mark.parametrize("relative", NOTEBOOK_PATHS)
def test_figure_markdown_and_authored_comments_use_english(relative: str) -> None:
    notebook = nbformat.read(ROOT / relative, as_version=4)
    kernel = notebook.metadata.kernelspec.name
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            assert CJK.search(cell.source) is None
        elif cell.cell_type == "code" and kernel == "python3":
            comments = [
                token.string
                for token in tokenize.generate_tokens(io.StringIO(cell.source).readline)
                if token.type == tokenize.COMMENT
            ]
            assert CJK.search("\n".join(comments)) is None
        elif cell.cell_type == "code" and kernel == "ir":
            comments = r_comments(cell.source)
            assert CJK.search("\n".join(comments)) is None
