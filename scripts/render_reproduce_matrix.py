from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from scripts.reproduce_lib import iter_targets, load_manifest

BEGIN_MARKER = "<!-- BEGIN:REPRODUCE_MATRIX -->"
END_MARKER = "<!-- END:REPRODUCE_MATRIX -->"

TABLE_HEADER = (
    "| Figure | File | Kernel | Input data | Status | Emits (into output/ when PHYTOMNI_SAVE=1) |"
)
TABLE_SEP = "|---|---|---|---|---|---|"


def _format_kernel(target: dict[str, Any]) -> str:
    kind = target.get("kind", "notebook")
    kernel = target.get("kernel", "none")
    if kind == "notebook":
        if kernel == "py":
            return "`python3`"
        if kernel == "ir":
            return "`ir` (R)"
        return kernel
    if kind == "rscript":
        return "Rscript"
    if kind == "py_script":
        return "`python3` (script)"
    if kind == "rmd":
        return "R / `rmarkdown::render`"
    return "—"


def _format_input_data(target: dict[str, Any], repo_root: Path | None) -> str:
    requires = target.get("requires_data") or []
    if not requires:
        return "*inline*"
    parts: list[str] = []
    for rel in requires:
        name = Path(rel).name
        if repo_root is not None and (repo_root / rel).is_file():
            parts.append(f"`{name}` ✓")
        else:
            parts.append(f"`{name}` ⚠")
    if len(parts) == 1:
        return parts[0]
    if len(parts) <= 3:
        return ", ".join(parts)
    return f"{parts[0]}, … ({len(parts)} files)"


def _format_emits(target: dict[str, Any]) -> str:
    artifacts = target.get("expected_artifacts") or []
    if not artifacts:
        return "—"
    basenames = [Path(a).name for a in artifacts]
    if len(basenames) <= 4:
        return " / ".join(f"`{name}`" for name in basenames)
    shown = " / ".join(f"`{name}`" for name in basenames[:3])
    return f"{shown} / … ({len(basenames)} files)"


def _format_file(target: dict[str, Any]) -> str:
    path = target.get("path")
    if not path:
        return "—"
    return f"`{path}`"


def _format_status(target: dict[str, Any]) -> str:
    return target.get("status", "run")


def render_matrix(manifest: dict[str, Any], repo_root: Path | None = None) -> str:
    """Render the reproduction matrix markdown table from a manifest dict."""
    rows: list[str] = [
        "Legend: ✓ = data ships in the repo · ⚠ = data pending (you must supply it) · "
        "*inline* = data is hardcoded in the notebook. Artifacts are written only when "
        "`PHYTOMNI_SAVE=1` is set; a default run renders inline and writes nothing.",
        "",
        TABLE_HEADER,
        TABLE_SEP,
    ]
    for target in iter_targets(manifest, phase="figure"):
        row = (
            f"| {target['label']} "
            f"| {_format_file(target)} "
            f"| {_format_kernel(target)} "
            f"| {_format_input_data(target, repo_root)} "
            f"| `{_format_status(target)}` "
            f"| {_format_emits(target)} |"
        )
        rows.append(row)
    return "\n".join(rows)


def _matrix_block(matrix_md: str) -> str:
    return f"{BEGIN_MARKER}\n{matrix_md}\n{END_MARKER}"


def update_readme_matrix(readme_text: str, matrix_md: str) -> str:
    """Replace or insert the reproduction matrix block in README text."""
    if BEGIN_MARKER in readme_text and END_MARKER in readme_text:
        before, rest = readme_text.split(BEGIN_MARKER, 1)
        _, after = rest.split(END_MARKER, 1)
        return f"{before}{_matrix_block(matrix_md)}{after}"

    section = f"\n### Reproduction matrix\n\n{_matrix_block(matrix_md)}\n"

    repro_matrix = "### Reproduction matrix"
    if repro_matrix in readme_text:
        start = readme_text.index(repro_matrix)
        rest = readme_text[start + len(repro_matrix) :]
        next_h3 = rest.find("\n### ")
        if next_h3 != -1:
            end = start + len(repro_matrix) + next_h3
            return readme_text[:start] + section.lstrip("\n") + readme_text[end:]
        return readme_text[:start] + section.lstrip("\n")

    one_cmd = "### One command"
    if one_cmd in readme_text:
        start = readme_text.index(one_cmd)
        rest = readme_text[start + len(one_cmd) :]
        next_h3 = rest.find("\n### ")
        insert_at = start + len(one_cmd) + next_h3 if next_h3 != -1 else len(readme_text)
        return readme_text[:insert_at] + section + readme_text[insert_at:]

    repro = "## Reproducing the figures"
    if repro in readme_text:
        idx = readme_text.index(repro) + len(repro)
        return readme_text[:idx] + section + readme_text[idx:]

    return readme_text + section


def write_readme_matrix(readme_path: Path, manifest_path: Path, repo_root: Path) -> None:
    manifest = load_manifest(manifest_path)
    matrix_md = render_matrix(manifest, repo_root=repo_root)
    readme_text = readme_path.read_text(encoding="utf-8")
    updated = update_readme_matrix(readme_text, matrix_md)
    readme_path.write_text(updated, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render README reproduction matrix from manifest")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to reproduce.manifest.yaml (default: repo root)",
    )
    parser.add_argument(
        "--write-readme",
        type=Path,
        metavar="README.md",
        help="Update the marked block in README.md",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = args.manifest or (repo_root / "reproduce.manifest.yaml")
    manifest = load_manifest(manifest_path)
    matrix_md = render_matrix(manifest, repo_root=repo_root)

    if args.write_readme:
        readme_path = args.write_readme
        if not readme_path.is_absolute():
            readme_path = repo_root / readme_path
        write_readme_matrix(readme_path, manifest_path, repo_root)
    else:
        print(matrix_md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
