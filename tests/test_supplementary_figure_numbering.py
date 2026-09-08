from pathlib import Path

import nbformat
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_uses_0717_supplementary_figure_numbering() -> None:
    manifest = yaml.safe_load((ROOT / "reproduce.manifest.yaml").read_text())
    targets = {target["id"]: target for target in manifest["targets"]}

    expected_paths = {
        "supp-12-15": (
            "Supp. 12-15",
            "Supplementary Fig. 12-15/supplementary_fig. 12-15.ipynb",
        ),
        "supp-16": (
            "Supp. 16",
            "Supplementary Fig. 16/supplementary_fig. 16.ipynb",
        ),
        "supp-19": (
            "Supp. 19",
            "Supplementary Fig. 19/supplementary_fig. 19.ipynb",
        ),
        "supp-20": (
            "Supp. 20",
            "Supplementary Fig. 20/supplementary_fig. 20.ipynb",
        ),
        "supp-21": (
            "Supp. 21",
            "Supplementary Fig. 21/supplementary_fig. 21.ipynb",
        ),
        "supp-26": (
            "Supp. 26",
            "Supplementary Fig. 26/supplementary_fig. 26.ipynb",
        ),
    }
    removed_target_ids = {
        "supp-8",
        "supp-9",
        "supp-10-13",
        "supp-14",
        "supp-17",
        "supp-24",
        "ext-data-6fg",
        "ext-data-6ab-deprecated",
        "ext-data-6abc",
        "ext-data-6-radar",
        "ext-data-7",
    }

    assert expected_paths.keys() <= targets.keys()
    assert removed_target_ids.isdisjoint(targets)
    for target_id, (label, path) in expected_paths.items():
        assert targets[target_id]["label"] == label
        assert targets[target_id]["path"] == path
        assert (ROOT / path).is_file()

    assert targets["fig-2def"]["path"] == expected_paths["supp-12-15"][1]
    for panel in ("6a", "6b", "7a", "7b"):
        target = targets[f"ext-data-{panel}"]
        assert target["path"] == (
            "Extended Data Fig. 6-7/extended_data_fig. 6ab-7ab.ipynb"
        )
        assert target["expected_artifacts"] == [
            f"Extended Data Fig. 6-7/output/extended_data_fig.{panel}.pdf"
        ]
    assert targets["ext-data-6de"]["path"] == (
        "Extended Data Fig. 6-7/extended_data_fig. 6de-7de.ipynb"
    )
    assert targets["ext-data-7de"]["path"] == targets["ext-data-6de"]["path"]
    assert targets["ext-data-7c"]["path"] == (
        "Extended Data Fig. 6-7/extended_data_fig. 7c.ipynb"
    )


def test_notebook_sources_and_outputs_use_0717_numbers() -> None:
    expected_output_markers = {
        "Supplementary Fig. 12-15/supplementary_fig. 12-15.ipynb": (
            "supplementary_fig.12.phytobench-gene",
            "supplementary_fig.13.expert-panel-and-agreement",
            "supplementary_fig.14.phytobench-gene",
            "supplementary_fig.15.phytobench-gene",
        ),
        "Supplementary Fig. 16/supplementary_fig. 16.ipynb": (
            "supplementary_fig.16",
        ),
        "Supplementary Fig. 19/supplementary_fig. 19.ipynb": (
            "supplementary_fig.19.pdf",
        ),
        "Supplementary Fig. 20/supplementary_fig. 20.ipynb": (
            "supplementary_fig.20.pdf",
        ),
        "Supplementary Fig. 21/supplementary_fig. 21.ipynb": (
            "supplementary_fig.21.pdf",
        ),
        "Supplementary Fig. 26/supplementary_fig. 26.ipynb": (
            "supplementary_fig.26.phytobench-review.polar",
        ),
    }

    for relative_path, markers in expected_output_markers.items():
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        for marker in markers:
            assert marker in source

    assert not (ROOT / "Supplementary Fig. 8").exists()
    assert not (ROOT / "Supplementary Fig. 9").exists()
    assert not (ROOT / "Extended Data Fig. 6").exists()
    assert not (ROOT / "Extended Data Fig. 7").exists()
    assert not (ROOT / "Extended Data Fig. 6-7" / "extended_data_fig. 6fg.ipynb").exists()
    assert not (ROOT / "Extended Data Fig. 6-7" / "extended_data_fig.6ab.ipynb").exists()

    knowledge = (
        ROOT / "Extended Data Fig. 6-7" / "extended_data_fig. 6ab-7ab.ipynb"
    ).read_text(encoding="utf-8")
    for panel in ("6a", "6b", "7a", "7b"):
        assert f"extended_data_fig.{panel}.pdf" in knowledge

    de = (
        ROOT / "Extended Data Fig. 6-7" / "extended_data_fig. 6de-7de.ipynb"
    ).read_text(encoding="utf-8")
    for panel in ("6d", "6e", "7d", "7e"):
        assert f"extended_data_fig.{panel}.pdf" in de
    assert "model_compare_agent_total.pdf" not in de

    seven_c = (
        ROOT / "Extended Data Fig. 6-7" / "extended_data_fig. 7c.ipynb"
    ).read_text(encoding="utf-8")
    assert "extended_data_fig.7c.pdf" in seven_c
    assert "GeneTuring" not in seven_c


def test_edf5d_saves_its_populated_figure_before_inline_show() -> None:
    path = ROOT / "Extended Data Fig. 5" / "extended_data_fig. 5d.ipynb"
    notebook = nbformat.read(path, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    save = 'fig.savefig("output/extended_data_fig.5d.pdf")'
    assert save in source
    assert source.index(save) < source.index("plt.show()")
