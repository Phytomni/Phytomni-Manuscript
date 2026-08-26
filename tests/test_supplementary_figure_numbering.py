from pathlib import Path

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
    }

    assert expected_paths.keys() <= targets.keys()
    assert removed_target_ids.isdisjoint(targets)
    for target_id, (label, path) in expected_paths.items():
        assert targets[target_id]["label"] == label
        assert targets[target_id]["path"] == path
        assert (ROOT / path).is_file()

    assert targets["fig-2def"]["path"] == expected_paths["supp-12-15"][1]
    assert targets["ext-data-6abc"]["label"] == "Ext. Data 6a, 7a"
    assert targets["ext-data-6abc"]["expected_artifacts"] == [
        "Extended Data Fig. 6/output/extended_data_fig.6a.pdf",
        "Extended Data Fig. 6/output/extended_data_fig.7a.pdf",
    ]
    assert targets["ext-data-6de"]["path"] == (
        "Extended Data Fig. 6/extended_data_fig. 6de.ipynb"
    )
    assert targets["ext-data-7c"]["path"] == (
        "Extended Data Fig. 7/extended_data_fig. 7c.ipynb"
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
    assert not (ROOT / "Extended Data Fig. 6" / "extended_data_fig. 6fg.ipynb").exists()

    abc = (
        ROOT / "Extended Data Fig. 6" / "extended_data_fig. 6abc.ipynb"
    ).read_text(encoding="utf-8")
    assert "extended_data_fig.6a.pdf" in abc
    assert "extended_data_fig.7a.pdf" in abc
    assert "extended_data_fig.6b.pdf" not in abc

    de = (
        ROOT / "Extended Data Fig. 6" / "extended_data_fig. 6de.ipynb"
    ).read_text(encoding="utf-8")
    assert "extended_data_fig.6d.pdf" in de
    assert "extended_data_fig.6e.pdf" in de
    assert "model_compare_agent_total.pdf" not in de

    seven_c = (
        ROOT / "Extended Data Fig. 7" / "extended_data_fig. 7c.ipynb"
    ).read_text(encoding="utf-8")
    assert "extended_data_fig.7c.pdf" in seven_c
    assert "GeneTuring" not in seven_c
