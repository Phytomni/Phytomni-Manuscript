from io import BytesIO
from itertools import permutations
import os
from pathlib import Path
import stat
import subprocess

import numpy as np
import pandas as pd
import pytest

import scripts.deepgenome_ranking_statistics as ranking_statistics
import scripts.release_deepgenome_rankings as release_module
from scripts.release_deepgenome_rankings import (
    build_release,
    initialize_crosswalk,
    main,
)


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("Gemini", "Grok", "OpenAI", "Phytomni", "Claude")
PUBLIC_COLUMNS = [
    "AnonymousExpertID",
    "Species",
    "Gene",
    "StudyStatus",
    *MODELS,
]
SUPPLEMENTARY = ROOT / "DeepGenomeAgent Evaluation" / "supplementary"
PRODUCTION_ATTACHMENT = (
    SUPPLEMENTARY / "Supplementary_Data_Expert_Rankings.tsv"
)
CANONICAL_SPECIES = ("Rice", "Maize", "Wheat", "Soybean", "Arabidopsis")
CANONICAL_STATUSES = ("well_studied", "uncharacterized")


def assert_ignored(relative_path: str) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", relative_path],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0, relative_path


def test_private_expert_inputs_are_ignored() -> None:
    assert_ignored("DeepGenomeAgent Evaluation/evaluation_expert_metadata-0714.xlsx")
    assert_ignored("DeepGenomeAgent Evaluation/private/expert_release_crosswalk.tsv")


@pytest.fixture
def private_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    score = pd.DataFrame(
        [
            {
                "Expert": "private-alpha",
                "Species": "Arabidopsis",
                "Gene": "AT1G01010",
                **{model: f"R{rank}" for rank, model in enumerate(MODELS, 1)},
            },
            {
                "Expert": "private-beta",
                "Species": "Rice",
                "Gene": "Os01g0100100",
                **{
                    model: f"R{rank}"
                    for rank, model in enumerate(reversed(MODELS), 1)
                },
            },
        ]
    )
    categories = pd.DataFrame(
        [
            ("Arabidopsis", "AT1G01010", "well_studied"),
            ("Rice", "Os01g0100100", "uncharacterized"),
        ],
        columns=["Species", "Gene", "StudyStatus"],
    )
    return score, categories


def write_crosswalk(tmp_path: Path, score: pd.DataFrame) -> Path:
    crosswalk = tmp_path / "crosswalk.tsv"
    raw_ids = sorted(score["Expert"].unique())
    release_ids = list(
        reversed(
            [f"E{index:03d}" for index in range(1, len(raw_ids) + 1)]
        )
    )
    pd.DataFrame(
        {
            "Expert": raw_ids,
            "AnonymousExpertID": release_ids,
        }
    ).to_csv(crosswalk, sep="\t", index=False)
    crosswalk.chmod(0o600)
    return crosswalk


def write_crosswalk_rows(
    path: Path,
    rows: list[tuple[str, str]],
    *,
    mode: int = 0o600,
) -> Path:
    pd.DataFrame(rows, columns=["Expert", "AnonymousExpertID"]).to_csv(
        path,
        sep="\t",
        index=False,
    )
    path.chmod(mode)
    return path


def write_cli_inputs(
    tmp_path: Path,
    score: pd.DataFrame,
    categories: pd.DataFrame,
) -> tuple[Path, Path]:
    score_path = tmp_path / "score.tsv"
    categories_path = tmp_path / "gene_categories.tsv"
    score.to_csv(score_path, sep="\t", index=False)
    categories.to_csv(categories_path, sep="\t", index=False)
    return score_path, categories_path


def canonical_tsv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        sep="\t",
        index=False,
        lineterminator="\n",
    ).encode("utf-8")


def synthetic_full_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking_orders = list(permutations(MODELS))
    score_rows: list[dict[str, str]] = []
    category_rows: list[tuple[str, str, str]] = []

    for species_index, species in enumerate(CANONICAL_SPECIES):
        experts = [
            f"synthetic-{species.casefold()}-{index:02d}"
            for index in range(1, 25)
        ]
        for gene_index in range(40):
            gene = f"SYN{species_index + 1}G{gene_index + 1:03d}"
            status = CANONICAL_STATUSES[gene_index // 20]
            category_rows.append((species, gene, status))
            for rater_offset in range(3):
                expert = experts[(gene_index * 3 + rater_offset) % len(experts)]
                order = ranking_orders[
                    (species_index * 120 + gene_index * 3 + rater_offset)
                    % len(ranking_orders)
                ]
                ranks = {
                    model: f"R{rank}"
                    for rank, model in enumerate(order, start=1)
                }
                score_rows.append(
                    {
                        "Expert": expert,
                        "Species": species,
                        "Gene": gene,
                        **ranks,
                    }
                )

    score = pd.DataFrame(score_rows)
    categories = pd.DataFrame(
        category_rows,
        columns=["Species", "Gene", "StudyStatus"],
    )
    return score, categories


def ranking_orders_from_frame(frame: pd.DataFrame) -> list[list[str]]:
    return [
        sorted(MODELS, key=lambda model: int(getattr(row, model)[1:]))
        for row in frame.itertuples(index=False)
    ]


def rank_counts(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.loc[:, list(MODELS)]
        .melt(var_name="Model", value_name="Rank")
        .groupby(["Model", "Rank"], sort=True)
        .size()
        .rename("Count")
    )


def read_production_attachment() -> tuple[bytes, pd.DataFrame]:
    raw = PRODUCTION_ATTACHMENT.read_bytes()
    frame = pd.read_csv(
        BytesIO(raw),
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    return raw, frame


def test_release_replaces_every_private_id(
    tmp_path: Path,
    private_fixture: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    score, categories = private_fixture
    crosswalk = write_crosswalk(tmp_path, score)
    raw_ids = set(score["Expert"])

    release = build_release(score, categories, crosswalk)

    assert list(release.columns) == PUBLIC_COLUMNS
    assert "Expert" not in release.columns
    assert release["AnonymousExpertID"].nunique() == score["Expert"].nunique()
    assert not raw_ids & set(release.astype(str).stack())
    assert len(release) == len(score)

    expected = (
        score.merge(pd.read_csv(crosswalk, sep="\t"), on="Expert")
        .merge(categories, on=["Species", "Gene"], validate="many_to_one")
        .loc[:, PUBLIC_COLUMNS]
        .sort_values(["AnonymousExpertID", "Species", "Gene"], kind="mergesort")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(release, expected)


def test_initialize_crosswalk_rejects_repository_path(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(mode=0o700)

    with pytest.raises(ValueError, match="outside the repository"):
        initialize_crosswalk(
            ["private-alpha", "private-beta"],
            repo_root / "private" / "crosswalk.tsv",
            repo_root,
        )


def test_initialize_crosswalk_uses_random_release_ids_and_private_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shuffled: list[list[str]] = []

    class RecordingSystemRandom:
        def shuffle(self, values: list[str]) -> None:
            shuffled.append(list(values))
            values.reverse()

    monkeypatch.setattr(
        release_module.secrets,
        "SystemRandom",
        RecordingSystemRandom,
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir(mode=0o700)
    crosswalk = tmp_path / "private" / "crosswalk.tsv"

    result = initialize_crosswalk(
        ["private-beta", "private-alpha"],
        crosswalk,
        repo_root,
    )

    assert result is None
    assert shuffled == [["E001", "E002"]]
    assert stat.S_IMODE(crosswalk.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(crosswalk.stat().st_mode) == 0o600
    frame = pd.read_csv(crosswalk, sep="\t", dtype=str)
    assert list(frame.columns) == ["Expert", "AnonymousExpertID"]
    assert frame["Expert"].tolist() == ["private-alpha", "private-beta"]
    assert set(frame["AnonymousExpertID"]) == {"E001", "E002"}


def test_initialize_crosswalk_refuses_overwrite(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(mode=0o700)
    crosswalk = tmp_path / "crosswalk.tsv"
    crosswalk.write_text("do not replace\n", encoding="utf-8")
    crosswalk.chmod(0o600)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        initialize_crosswalk(["private-alpha"], crosswalk, repo_root)

    assert crosswalk.read_text(encoding="utf-8") == "do not replace\n"


def test_initialize_crosswalk_rejects_unsafe_parent_permissions(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(mode=0o700)
    unsafe_parent = tmp_path / "shared"
    unsafe_parent.mkdir(mode=0o755)
    unsafe_parent.chmod(0o755)

    with pytest.raises(ValueError, match="parent directory permissions"):
        initialize_crosswalk(
            ["private-alpha"],
            unsafe_parent / "crosswalk.tsv",
            repo_root,
        )


def test_initialize_crosswalk_rejects_duplicate_raw_ids(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(mode=0o700)

    with pytest.raises(ValueError, match="unique"):
        initialize_crosswalk(
            ["private-alpha", "private-alpha"],
            tmp_path / "crosswalk.tsv",
            repo_root,
        )


def test_build_release_rejects_repository_crosswalk(
    tmp_path: Path,
    private_fixture: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    score, categories = private_fixture
    repo_root = tmp_path / "repo"
    repo_root.mkdir(mode=0o700)
    crosswalk = repo_root / "crosswalk.tsv"
    write_crosswalk_rows(
        crosswalk,
        [("private-alpha", "E001"), ("private-beta", "E002")],
    )

    with pytest.raises(ValueError, match="outside the repository"):
        build_release(score, categories, crosswalk, repo_root=repo_root)


def test_build_release_requires_mode_0600_crosswalk(
    tmp_path: Path,
    private_fixture: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    score, categories = private_fixture
    crosswalk = write_crosswalk(tmp_path, score)
    crosswalk.chmod(0o640)

    with pytest.raises(ValueError, match="0600"):
        build_release(score, categories, crosswalk)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [("private-alpha", "E001"), ("private-alpha", "E002")],
            "raw Expert IDs must be unique",
        ),
        ([("private-alpha", "E001")], "exactly match"),
        (
            [
                ("private-alpha", "E001"),
                ("private-beta", "E002"),
                ("private-gamma", "E003"),
            ],
            "exactly match",
        ),
        (
            [("private-alpha", "E001"), ("private-beta", "E001")],
            "release IDs must be unique",
        ),
        (
            [("private-alpha", "E01"), ("private-beta", "E002")],
            "E###",
        ),
        (
            [("private-alpha", "E001"), ("private-beta", "E003")],
            "contiguous",
        ),
    ],
    ids=[
        "duplicate-raw",
        "missing-raw",
        "extra-raw",
        "duplicate-release",
        "malformed-release",
        "noncontiguous-release",
    ],
)
def test_build_release_rejects_invalid_crosswalk_ids(
    tmp_path: Path,
    private_fixture: tuple[pd.DataFrame, pd.DataFrame],
    rows: list[tuple[str, str]],
    message: str,
) -> None:
    score, categories = private_fixture
    crosswalk = write_crosswalk_rows(tmp_path / "crosswalk.tsv", rows)

    with pytest.raises(ValueError, match=message):
        build_release(score, categories, crosswalk)


def test_build_release_rejects_crosswalk_columns_beyond_id_pair(
    tmp_path: Path,
    private_fixture: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    score, categories = private_fixture
    crosswalk = tmp_path / "crosswalk.tsv"
    pd.DataFrame(
        {
            "Expert": ["private-alpha", "private-beta"],
            "AnonymousExpertID": ["E001", "E002"],
            "Country": ["hidden", "hidden"],
        }
    ).to_csv(crosswalk, sep="\t", index=False)
    crosswalk.chmod(0o600)

    with pytest.raises(ValueError, match="exactly these columns"):
        build_release(score, categories, crosswalk)


def test_build_release_rejects_duplicate_gene_categories(
    tmp_path: Path,
    private_fixture: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    score, categories = private_fixture
    categories = pd.concat([categories, categories.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="unique by Species/Gene"):
        build_release(score, categories, write_crosswalk(tmp_path, score))


@pytest.mark.parametrize("case", ["missing", "extra"])
def test_build_release_requires_exact_gene_category_set(
    tmp_path: Path,
    private_fixture: tuple[pd.DataFrame, pd.DataFrame],
    case: str,
) -> None:
    score, categories = private_fixture
    if case == "missing":
        categories = categories.iloc[:-1].copy()
    else:
        categories = pd.concat(
            [
                categories,
                pd.DataFrame(
                    [("Wheat", "TraesCS1A01G000100", "well_studied")],
                    columns=categories.columns,
                ),
            ],
            ignore_index=True,
        )

    with pytest.raises(ValueError, match="exactly the Species/Gene pairs"):
        build_release(score, categories, write_crosswalk(tmp_path, score))


@pytest.mark.parametrize("replacement", ["R4", "R6", None])
def test_build_release_rejects_invalid_ranking_permutations(
    tmp_path: Path,
    private_fixture: tuple[pd.DataFrame, pd.DataFrame],
    replacement: str | None,
) -> None:
    score, categories = private_fixture
    score.loc[0, "Claude"] = replacement

    with pytest.raises(ValueError, match="complete permutation"):
        build_release(score, categories, write_crosswalk(tmp_path, score))


def test_build_release_final_scan_rejects_raw_id_in_any_public_cell(
    tmp_path: Path,
    private_fixture: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    score, categories = private_fixture
    score.loc[0, "Gene"] = "private-alpha"
    categories.loc[
        (categories["Species"] == "Arabidopsis")
        & (categories["Gene"] == "AT1G01010"),
        "Gene",
    ] = "private-alpha"

    with pytest.raises(ValueError, match="raw expert identifier"):
        build_release(score, categories, write_crosswalk(tmp_path, score))


def test_build_release_final_scan_rejects_raw_id_embedded_in_public_cell(
    tmp_path: Path,
    private_fixture: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    score, categories = private_fixture
    leaked_gene = "gene-private-alpha-suffix"
    score.loc[0, "Gene"] = leaked_gene
    categories.loc[
        (categories["Species"] == "Arabidopsis")
        & (categories["Gene"] == "AT1G01010"),
        "Gene",
    ] = leaked_gene

    with pytest.raises(ValueError, match="raw expert identifier"):
        build_release(score, categories, write_crosswalk(tmp_path, score))


def invoke_crosswalk_mode(
    mode: str,
    path: Path,
    repo_root: Path,
    score: pd.DataFrame,
    categories: pd.DataFrame,
) -> None:
    if mode == "initialize":
        initialize_crosswalk(sorted(score["Expert"].unique()), path, repo_root)
    else:
        build_release(score, categories, path, repo_root=repo_root)


@pytest.mark.parametrize("mode", ["initialize", "use"])
def test_crosswalk_rejects_lexical_repository_containment(
    tmp_path: Path,
    private_fixture: tuple[pd.DataFrame, pd.DataFrame],
    mode: str,
) -> None:
    score, categories = private_fixture
    repo_root = tmp_path / "repo"
    repo_root.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    escape = repo_root / "escape"
    escape.symlink_to(outside, target_is_directory=True)
    path = escape / "crosswalk.tsv"
    if mode == "use":
        write_crosswalk_rows(
            outside / "crosswalk.tsv",
            [("private-alpha", "E001"), ("private-beta", "E002")],
        )

    with pytest.raises(ValueError, match="lexically contained"):
        invoke_crosswalk_mode(mode, path, repo_root, score, categories)


@pytest.mark.parametrize("mode", ["initialize", "use"])
def test_crosswalk_rejects_resolved_repository_containment(
    tmp_path: Path,
    private_fixture: tuple[pd.DataFrame, pd.DataFrame],
    mode: str,
) -> None:
    score, categories = private_fixture
    repo_root = tmp_path / "repo"
    private = repo_root / "private"
    private.mkdir(parents=True, mode=0o700)
    outside_link = tmp_path / "outside-link"
    outside_link.symlink_to(private, target_is_directory=True)
    path = outside_link / "crosswalk.tsv"
    if mode == "use":
        write_crosswalk_rows(
            private / "crosswalk.tsv",
            [("private-alpha", "E001"), ("private-beta", "E002")],
        )

    with pytest.raises(ValueError, match="resolves inside"):
        invoke_crosswalk_mode(mode, path, repo_root, score, categories)


@pytest.mark.parametrize("mode", ["initialize", "use"])
def test_crosswalk_rejects_symlink_leaf(
    tmp_path: Path,
    private_fixture: tuple[pd.DataFrame, pd.DataFrame],
    mode: str,
) -> None:
    score, categories = private_fixture
    repo_root = tmp_path / "repo"
    repo_root.mkdir(mode=0o700)
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    target_parent = tmp_path / "target"
    target_parent.mkdir(mode=0o700)
    target = write_crosswalk_rows(
        target_parent / "real.tsv",
        [("private-alpha", "E001"), ("private-beta", "E002")],
    )
    path = parent / "crosswalk.tsv"
    path.symlink_to(target)

    with pytest.raises(ValueError, match="symbolic link"):
        invoke_crosswalk_mode(mode, path, repo_root, score, categories)


@pytest.mark.parametrize("mode", ["initialize", "use"])
def test_crosswalk_rejects_symlink_ancestor(
    tmp_path: Path,
    private_fixture: tuple[pd.DataFrame, pd.DataFrame],
    mode: str,
) -> None:
    score, categories = private_fixture
    repo_root = tmp_path / "repo"
    repo_root.mkdir(mode=0o700)
    real_parent = tmp_path / "real-private"
    real_parent.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked-private"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    path = linked_parent / "crosswalk.tsv"
    if mode == "use":
        write_crosswalk_rows(
            real_parent / "crosswalk.tsv",
            [("private-alpha", "E001"), ("private-beta", "E002")],
        )

    with pytest.raises(ValueError, match="symbolic link"):
        invoke_crosswalk_mode(mode, path, repo_root, score, categories)


def test_build_release_rejects_unsafe_crosswalk_parent_permissions(
    tmp_path: Path,
    private_fixture: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    score, categories = private_fixture
    parent = tmp_path / "shared"
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)
    crosswalk = write_crosswalk_rows(
        parent / "crosswalk.tsv",
        [("private-alpha", "E001"), ("private-beta", "E002")],
    )

    with pytest.raises(ValueError, match="parent directory permissions"):
        build_release(score, categories, crosswalk)


def test_build_release_rejects_nonregular_crosswalk(
    tmp_path: Path,
    private_fixture: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    score, categories = private_fixture
    crosswalk = tmp_path / "crosswalk.tsv"
    crosswalk.mkdir(mode=0o700)

    with pytest.raises(ValueError, match="regular file"):
        build_release(score, categories, crosswalk)


def test_build_release_is_deterministically_sorted_and_drops_demographics(
    tmp_path: Path,
    private_fixture: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    score, categories = private_fixture
    score["Country"] = ["private-country-a", "private-country-b"]
    crosswalk = write_crosswalk(tmp_path, score)

    first = build_release(score, categories, crosswalk)
    second = build_release(
        score.iloc[::-1].reset_index(drop=True),
        categories.iloc[::-1].reset_index(drop=True),
        crosswalk,
    )

    pd.testing.assert_frame_equal(first, second)
    assert "Country" not in first.columns


def test_cli_modes_are_mutually_exclusive(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--score-tsv",
                "score.tsv",
                "--gene-categories",
                "categories.tsv",
                "--crosswalk",
                "crosswalk.tsv",
                "--initialize-crosswalk",
                "--use-crosswalk",
            ]
        )

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "private-alpha" not in captured.out + captured.err


def test_cli_initialization_writes_only_crosswalk_and_never_prints_ids(
    tmp_path: Path,
    private_fixture: tuple[pd.DataFrame, pd.DataFrame],
    capsys: pytest.CaptureFixture[str],
) -> None:
    score, categories = private_fixture
    score_path, _ = write_cli_inputs(tmp_path, score, categories)
    crosswalk = tmp_path / "private" / "crosswalk.tsv"

    result = main(
        [
            "--score-tsv",
            str(score_path),
            "--crosswalk",
            str(crosswalk),
            "--initialize-crosswalk",
        ]
    )

    assert result == 0
    assert crosswalk.exists()
    assert not (tmp_path / "release.tsv").exists()
    output = capsys.readouterr().out
    assert "private-alpha" not in output
    assert "private-beta" not in output
    assert "AnonymousExpertID" not in output


def test_cli_use_mode_requires_gene_categories(
    tmp_path: Path,
    private_fixture: tuple[pd.DataFrame, pd.DataFrame],
    capsys: pytest.CaptureFixture[str],
) -> None:
    score, categories = private_fixture
    score_path, _ = write_cli_inputs(tmp_path, score, categories)
    crosswalk = write_crosswalk(tmp_path, score)

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--score-tsv",
                str(score_path),
                "--crosswalk",
                str(crosswalk),
                "--use-crosswalk",
                "--output",
                str(tmp_path / "release.tsv"),
            ]
        )

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "private-alpha" not in captured.out + captured.err


def test_cli_use_mode_writes_only_requested_output_and_never_prints_ids(
    tmp_path: Path,
    private_fixture: tuple[pd.DataFrame, pd.DataFrame],
    capsys: pytest.CaptureFixture[str],
) -> None:
    score, categories = private_fixture
    score_path, categories_path = write_cli_inputs(tmp_path, score, categories)
    crosswalk = write_crosswalk(tmp_path, score)
    output_path = tmp_path / "release.tsv"
    before = set(tmp_path.rglob("*"))

    result = main(
        [
            "--score-tsv",
            str(score_path),
            "--gene-categories",
            str(categories_path),
            "--crosswalk",
            str(crosswalk),
            "--use-crosswalk",
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    assert set(tmp_path.rglob("*")) - before == {output_path}
    assert list(pd.read_csv(output_path, sep="\t").columns) == PUBLIC_COLUMNS
    output = capsys.readouterr().out
    assert "private-alpha" not in output
    assert "private-beta" not in output
    assert "AnonymousExpertID" not in output


def test_production_attachment_has_exact_public_schema_and_canonical_bytes() -> None:
    raw, release = read_production_attachment()

    assert list(release.columns) == PUBLIC_COLUMNS
    assert "Expert" not in release.columns
    assert len(release) == 600
    assert not release.isna().any().any()
    assert not release.eq("").any().any()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 601
    assert canonical_tsv_bytes(release) == raw

    expected_order = release.sort_values(
        ["AnonymousExpertID", "Species", "Gene"],
        kind="mergesort",
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(release, expected_order)

    expected_ranks = {f"R{rank}" for rank in range(1, 6)}
    assert all(
        len(set(row)) == 5 and set(row) == expected_ranks
        for row in release.loc[:, list(MODELS)].itertuples(
            index=False,
            name=None,
        )
    )


def test_production_attachment_has_complete_balanced_panel_assignments() -> None:
    _, release = read_production_attachment()

    expected_release_ids = {
        f"E{index:03d}" for index in range(1, 121)
    }
    assert set(release["AnonymousExpertID"]) == expected_release_ids
    assert release["AnonymousExpertID"].nunique() == 120
    assert (
        release["AnonymousExpertID"].value_counts(sort=False) == 5
    ).all()
    assert (
        release.groupby("AnonymousExpertID", sort=False)["Species"].nunique()
        == 1
    ).all()

    items = release.groupby(["Species", "Gene"], sort=False)
    assert items.ngroups == 200
    assert (items.size() == 3).all()
    assert (items["AnonymousExpertID"].nunique() == 3).all()
    assert (items["StudyStatus"].nunique() == 1).all()

    assert set(release["Species"]) == set(CANONICAL_SPECIES)
    assert set(release["StudyStatus"]) == set(CANONICAL_STATUSES)
    assert (
        release.groupby("Species", sort=False)["Gene"].nunique() == 40
    ).all()
    assert (
        release.groupby("Species", sort=False)["AnonymousExpertID"].nunique()
        == 24
    ).all()
    assert (
        release.groupby(["Species", "StudyStatus"], sort=False)["Gene"]
        .nunique()
        .eq(20)
        .all()
    )


@pytest.mark.skipif(
    os.environ.get("PHYTOMNI_RUN_PRIVATE_LINEAGE") != "1",
    reason="private production lineage is an explicit opt-in gate",
)
def test_private_production_lineage_matches_committed_attachment() -> None:
    crosswalk_value = os.environ.get("PHYTOMNI_EXPERT_CROSSWALK")
    if not crosswalk_value:
        pytest.fail(
            "Private lineage configuration failed: "
            "PHYTOMNI_EXPERT_CROSSWALK is required.",
            pytrace=False,
        )

    try:
        score = pd.read_csv(
            ROOT / "DeepGenomeAgent Evaluation" / "score.tsv",
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )
        categories = pd.read_csv(
            ROOT
            / "Supplementary Fig. 10-13"
            / "PhytoBench-Gene-for_plot"
            / "gene_categories.tsv",
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )
        committed_bytes, committed = read_production_attachment()
        regenerated = build_release(
            score,
            categories,
            Path(crosswalk_value),
        )
        byte_equivalent = (
            canonical_tsv_bytes(regenerated) == committed_bytes
        )

        raw_ids = set(score["Expert"])
        privacy_safe = not any(
            raw_id in public_cell
            for public_cell in committed.astype(str).to_numpy().ravel()
            for raw_id in raw_ids
        )

        raw = score.merge(
            categories,
            on=["Species", "Gene"],
            how="left",
            validate="many_to_one",
        ).loc[:, ["Species", "Gene", "Expert", "StudyStatus", *MODELS]]
        public = committed.rename(
            columns={"AnonymousExpertID": "Expert"}
        ).loc[:, ["Species", "Gene", "Expert", "StudyStatus", *MODELS]]
        rank_counts_equal = rank_counts(raw).equals(rank_counts(public))

        raw_fit = ranking_statistics.fit_plackett_luce(
            ranking_orders_from_frame(raw),
            list(MODELS),
        )
        public_fit = ranking_statistics.fit_plackett_luce(
            ranking_orders_from_frame(public),
            list(MODELS),
        )
        pl_models_equal = raw_fit["models"] == public_fit["models"]
        pl_fits_succeeded = bool(
            raw_fit["optimizer_result"].success
            and public_fit["optimizer_result"].success
        )
        pl_max_delta = (
            float(
                np.max(
                    np.abs(
                        np.asarray(raw_fit["elo"], dtype=float)
                        - np.asarray(public_fit["elo"], dtype=float)
                    )
                )
            )
            if pl_models_equal
            else float("inf")
        )

        registry = ranking_statistics.agreement_scope_registry(
            CANONICAL_SPECIES,
            CANONICAL_STATUSES,
            MODELS,
        )
        raw_fleiss = ranking_statistics.fleiss_point_estimates(
            raw,
            registry,
            MODELS,
        )
        public_fleiss = ranking_statistics.fleiss_point_estimates(
            public,
            registry,
            MODELS,
        )
        fleiss_exact_columns = [
            "ScopeID",
            "AnalysisTier",
            "ScopeFamily",
            "Species",
            "StudyStatus",
            "Model",
            "NGenes",
            "NItems",
            "RatingsPerItem",
            "NRatings",
            "NContributingExperts",
        ]
        fleiss_numeric_columns = [
            "ObservedAgreement",
            "ExpectedAgreement",
            "FleissKappa",
            "RankR1Share",
            "RankR2Share",
            "RankR3Share",
            "RankR4Share",
            "RankR5Share",
        ]
        fleiss_count_equal = (
            len(raw_fleiss) == len(public_fleiss) == 58
        )
        fleiss_keys_equal = raw_fleiss.loc[
            :, fleiss_exact_columns
        ].equals(public_fleiss.loc[:, fleiss_exact_columns])
        fleiss_max_delta = float(
            np.max(
                np.abs(
                    raw_fleiss.loc[
                        :, fleiss_numeric_columns
                    ].to_numpy(dtype=float)
                    - public_fleiss.loc[
                        :, fleiss_numeric_columns
                    ].to_numpy(dtype=float)
                )
            )
        )

        raw_gene = ranking_statistics.gene_ordinal_agreement(raw, MODELS)
        public_gene = ranking_statistics.gene_ordinal_agreement(
            public,
            MODELS,
        )
        gene_exact_columns = [
            "Species",
            "Gene",
            "StudyStatus",
            "NExperts",
            "NModels",
            "Top1AgreementPattern",
        ]
        gene_numeric_columns = [
            "KendallW",
            "MeanPairwiseKendallTau",
        ]
        gene_count_equal = len(raw_gene) == len(public_gene) == 200
        gene_keys_equal = raw_gene.loc[:, gene_exact_columns].equals(
            public_gene.loc[:, gene_exact_columns]
        )
        gene_max_delta = float(
            np.max(
                np.abs(
                    raw_gene.loc[
                        :, gene_numeric_columns
                    ].to_numpy(dtype=float)
                    - public_gene.loc[
                        :, gene_numeric_columns
                    ].to_numpy(dtype=float)
                )
            )
        )
    except Exception as error:
        pytest.fail(
            "Private lineage execution failed: "
            f"{type(error).__name__}.",
            pytrace=False,
        )

    if not byte_equivalent:
        pytest.fail(
            "Private lineage byte equivalence failed.",
            pytrace=False,
        )
    if not privacy_safe:
        pytest.fail(
            "Private lineage privacy scan failed.",
            pytrace=False,
        )
    if not rank_counts_equal:
        pytest.fail(
            "Private lineage rank-count equivalence failed.",
            pytrace=False,
        )
    if (
        not pl_models_equal
        or not pl_fits_succeeded
        or not np.isfinite(pl_max_delta)
        or pl_max_delta > 1e-8
    ):
        pytest.fail(
            "Private lineage overall PL equivalence failed: "
            f"max_abs_delta={pl_max_delta:.17g}.",
            pytrace=False,
        )
    if not fleiss_count_equal:
        pytest.fail(
            "Private lineage Fleiss scope-count equivalence failed: "
            f"private_count={len(raw_fleiss)}, "
            f"public_count={len(public_fleiss)}.",
            pytrace=False,
        )
    if not fleiss_keys_equal:
        pytest.fail(
            "Private lineage Fleiss key equivalence failed.",
            pytrace=False,
        )
    if (
        not np.isfinite(fleiss_max_delta)
        or fleiss_max_delta > 1e-15
    ):
        pytest.fail(
            "Private lineage Fleiss numeric equivalence failed: "
            f"max_abs_delta={fleiss_max_delta:.17g}.",
            pytrace=False,
        )
    if not gene_count_equal:
        pytest.fail(
            "Private lineage gene-count equivalence failed: "
            f"private_count={len(raw_gene)}, "
            f"public_count={len(public_gene)}.",
            pytrace=False,
        )
    if not gene_keys_equal:
        pytest.fail(
            "Private lineage gene key/Top1 equivalence failed.",
            pytrace=False,
        )
    if not np.isfinite(gene_max_delta) or gene_max_delta > 1e-15:
        pytest.fail(
            "Private lineage gene numeric equivalence failed: "
            f"max_abs_delta={gene_max_delta:.17g}.",
            pytrace=False,
        )


def test_synthetic_full_panel_regeneration_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    score, categories = synthetic_full_panel()
    crosswalk = write_crosswalk(tmp_path, score)

    first = build_release(score, categories, crosswalk)
    second = build_release(
        score.iloc[::-1].reset_index(drop=True),
        categories.iloc[::-1].reset_index(drop=True),
        crosswalk,
    )

    assert stat.S_IMODE(crosswalk.stat().st_mode) == 0o600
    assert canonical_tsv_bytes(first) == canonical_tsv_bytes(second)


def test_synthetic_release_preserves_all_locked_ranking_statistics(
    tmp_path: Path,
) -> None:
    score, categories = synthetic_full_panel()
    crosswalk = write_crosswalk(tmp_path, score)
    release = build_release(score, categories, crosswalk)
    raw = score.merge(
        categories,
        on=["Species", "Gene"],
        how="left",
        validate="many_to_one",
    ).loc[:, ["Species", "Gene", "Expert", "StudyStatus", *MODELS]]
    public = release.rename(
        columns={"AnonymousExpertID": "Expert"}
    ).loc[:, ["Species", "Gene", "Expert", "StudyStatus", *MODELS]]

    pd.testing.assert_series_equal(rank_counts(public), rank_counts(raw))

    raw_fit = ranking_statistics.fit_plackett_luce(
        ranking_orders_from_frame(raw),
        list(MODELS),
    )
    release_fit = ranking_statistics.fit_plackett_luce(
        ranking_orders_from_frame(public),
        list(MODELS),
    )
    assert raw_fit["models"] == release_fit["models"]
    np.testing.assert_allclose(
        release_fit["elo"],
        raw_fit["elo"],
        rtol=0.0,
        atol=1e-8,
    )

    registry = ranking_statistics.agreement_scope_registry(
        CANONICAL_SPECIES,
        CANONICAL_STATUSES,
        MODELS,
    )
    raw_fleiss = ranking_statistics.fleiss_point_estimates(
        raw,
        registry,
        MODELS,
    )
    release_fleiss = ranking_statistics.fleiss_point_estimates(
        public,
        registry,
        MODELS,
    )
    assert len(raw_fleiss) == len(release_fleiss) == 58
    pd.testing.assert_frame_equal(
        release_fleiss,
        raw_fleiss,
        check_exact=False,
        rtol=0.0,
        atol=1e-15,
    )

    raw_gene = ranking_statistics.gene_ordinal_agreement(raw, MODELS)
    release_gene = ranking_statistics.gene_ordinal_agreement(public, MODELS)
    assert len(raw_gene) == len(release_gene) == 200
    pd.testing.assert_frame_equal(
        release_gene,
        raw_gene,
        check_exact=False,
        rtol=0.0,
        atol=1e-15,
    )


def test_public_ranking_codebook_is_exact() -> None:
    codebook = pd.read_csv(
        SUPPLEMENTARY / "Supplementary_Data_Expert_Rankings_Codebook.tsv",
        sep="\t",
        dtype=str,
    )
    expected = pd.DataFrame(
        [
            ("AnonymousExpertID", "Random release identifier with no public crosswalk", "E001-E120"),
            ("Species", "Benchmark species", "Arabidopsis; Maize; Rice; Soybean; Wheat"),
            ("Gene", "Public benchmark gene identifier", "Non-empty string"),
            ("StudyStatus", "Gene knowledge stratum", "well_studied; uncharacterized"),
            ("Gemini", "Exact rank position", "R1-R5"),
            ("Grok", "Exact rank position", "R1-R5"),
            ("OpenAI", "Exact rank position", "R1-R5"),
            ("Phytomni", "Exact rank position", "R1-R5"),
            ("Claude", "Exact rank position", "R1-R5"),
        ],
        columns=["Column", "Description", "AllowedValues"],
    )
    pd.testing.assert_frame_equal(codebook, expected)


def test_panel_category_map_has_complete_controlled_schema() -> None:
    category_map = pd.read_csv(
        SUPPLEMENTARY / "expert_panel_category_map.tsv",
        sep="\t",
        dtype=str,
    )

    assert list(category_map.columns) == [
        "Dimension",
        "SourceValue",
        "PublicCategory",
        "DisplayOrder",
    ]
    assert not category_map.duplicated(["Dimension", "SourceValue"]).any()
    assert category_map["DisplayOrder"].str.fullmatch(r"[1-9][0-9]*").all()
    assert not any(
        "expert" in column.casefold() or "count" in column.casefold()
        for column in category_map.columns
    )

    expected_sources = {
        "Species": {"Arabidopsis", "Maize", "Rice", "Soybean", "Wheat"},
        "Country/Region": {
            "Australia", "Austria", "Belgium", "Canada", "China", "Denmark",
            "Egypt", "Germany", "Japan", "Malaysia", "Netherlands", "Nigeria",
            "Norway", "Philippines", "Singapore", "Spain", "United Kingdom",
            "USA", "Vietnam", "Western Australia",
        },
        "Institution_type": {"Research institute", "University"},
        "Current_position": {
            "Associate Professor / Associate researcher",
            "Full Professor / Full researcher",
            "Master's student", "Other", "PhD student",
            "Postdoc / Assistant researcher",
        },
        "Years_experience": {
            "< 3", "3-5", "3–5", "6–10", "11–20", "> 20",
        },
        "Gender": {"Female", "Male", "Prefer not to say"},
        "Research_domains": {
            "Bioinformatics / Computational biology", "Crop genetics & breeding",
            "Developmental biology", "Epigenetics", "Functional genomics",
            "Metabolomics / Proteomics", "Molecular biology", "Other",
            "Plant hormone / Signal transduction", "Plant pathology / Stress biology",
            "Population / Evolutionary genetics",
        },
        "Study_species": {
            "Arabidopsis", "Cotton", "Fruit / Tree crops", "Maize", "Other",
            "Rice", "Soybean", "Vegetable crops", "Wheat",
        },
        "Annotation_experience": {"Frequent", "Occasional"},
        "Ai_experience": {"Frequent", "Occasional"},
        "Conflict_interest": {"No"},
    }
    actual_sources = {
        dimension: set(group["SourceValue"])
        for dimension, group in category_map.groupby("Dimension")
    }
    assert actual_sources == expected_sources


def test_panel_category_map_contains_required_privacy_merges() -> None:
    category_map = pd.read_csv(
        SUPPLEMENTARY / "expert_panel_category_map.tsv",
        sep="\t",
        dtype=str,
    ).set_index(["Dimension", "SourceValue"])["PublicCategory"]

    country_groups = {
        "Asia": {"China", "Japan", "Malaysia", "Philippines", "Singapore", "Vietnam"},
        "Europe": {
            "Austria", "Belgium", "Denmark", "Germany", "Netherlands", "Norway",
            "Spain", "United Kingdom",
        },
        "North America": {"Canada", "USA"},
        "Other regions": {"Australia", "Western Australia", "Egypt", "Nigeria"},
    }
    for public_category, sources in country_groups.items():
        assert {
            source
            for source in sources
            if category_map.loc[("Country/Region", source)] == public_category
        } == sources

    assert category_map.loc[("Current_position", "Master's student")] == "Other career stages"
    assert category_map.loc[("Current_position", "Other")] == "Other career stages"
    assert category_map.loc[("Years_experience", "< 3")] == "<= 5 years"
    assert category_map.loc[("Years_experience", "3-5")] == "<= 5 years"
    assert category_map.loc[("Years_experience", "3–5")] == "<= 5 years"
    for source in (
        "Population / Evolutionary genetics",
        "Metabolomics / Proteomics",
        "Other",
    ):
        assert category_map.loc[("Research_domains", source)] == "Population, metabolomics, or other"
    for source in ("Cotton", "Fruit / Tree crops", "Other"):
        assert category_map.loc[("Study_species", source)] == "Other crop systems"


def synthetic_category_map(
    rows: list[tuple[str, str, str, int]],
) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["Dimension", "SourceValue", "PublicCategory", "DisplayOrder"],
    )


def test_panel_category_audit_deduplicates_expert_public_categories() -> None:
    metadata = pd.DataFrame(
        {
            "Expert_ID": [f"expert-{index}" for index in range(1, 6)],
            "Research_domains": [
                "['Alpha', 'Alpha', 'Beta']",
                "['Alpha']",
                "['Alpha']",
                "['Alpha']",
                "['Alpha']",
            ],
        }
    )
    category_map = synthetic_category_map(
        [
            ("Research_domains", "Alpha", "Shared domain", 1),
            ("Research_domains", "Beta", "Shared domain", 1),
        ]
    )

    result = release_module.audit_panel_category_map(metadata, category_map)

    expected = pd.DataFrame(
        [("Research_domains", "Shared domain", 5)],
        columns=["Dimension", "PublicCategory", "N"],
    )
    pd.testing.assert_frame_equal(result, expected)
    assert not any("expert" in column.casefold() for column in result.columns)


def test_panel_category_audit_rejects_public_group_below_minimum() -> None:
    metadata = pd.DataFrame(
        {
            "Expert_ID": [f"expert-{index}" for index in range(1, 5)],
            "Research_domains": ["['Alpha']"] * 4,
        }
    )
    category_map = synthetic_category_map(
        [("Research_domains", "Alpha", "Small group", 1)]
    )

    with pytest.raises(ValueError, match="minimum public group size"):
        release_module.audit_panel_category_map(metadata, category_map)


def test_panel_category_audit_excludes_scalar_missing_values() -> None:
    metadata = pd.DataFrame(
        {
            "Expert_ID": [f"expert-{index}" for index in range(1, 7)],
            "Gender": ["Female", "Female", "Female", "Female", "Female", None],
        }
    )
    category_map = synthetic_category_map(
        [("Gender", "Female", "Female", 1)]
    )

    result = release_module.audit_panel_category_map(metadata, category_map)

    assert result.to_dict("records") == [
        {"Dimension": "Gender", "PublicCategory": "Female", "N": 5}
    ]


def test_panel_category_audit_requires_exact_map_columns() -> None:
    metadata = pd.DataFrame(
        {"Expert_ID": [f"expert-{index}" for index in range(1, 6)], "Gender": ["Female"] * 5}
    )
    category_map = synthetic_category_map(
        [("Gender", "Female", "Female", 1)]
    )
    category_map["Count"] = 5

    with pytest.raises(ValueError, match="exactly these columns"):
        release_module.audit_panel_category_map(metadata, category_map)


def test_panel_category_audit_rejects_duplicate_source_mapping() -> None:
    metadata = pd.DataFrame(
        {"Expert_ID": [f"expert-{index}" for index in range(1, 6)], "Gender": ["Female"] * 5}
    )
    category_map = synthetic_category_map(
        [
            ("Gender", "Female", "Female", 1),
            ("Gender", "Female", "Women", 2),
        ]
    )

    with pytest.raises(ValueError, match="unique by Dimension/SourceValue"):
        release_module.audit_panel_category_map(metadata, category_map)


def test_panel_category_audit_rejects_unmapped_observed_source() -> None:
    metadata = pd.DataFrame(
        {
            "Expert_ID": [f"expert-{index}" for index in range(1, 6)],
            "Gender": ["Female", "Female", "Female", "Female", "Unmapped"],
        }
    )
    category_map = synthetic_category_map(
        [("Gender", "Female", "Female", 1)]
    )

    with pytest.raises(ValueError, match="observed source values must be mapped"):
        release_module.audit_panel_category_map(metadata, category_map)


@pytest.mark.parametrize(
    "invalid_cell",
    ["not a literal", "'Alpha'", "['Alpha', 3]", "[]"],
)
def test_panel_category_audit_rejects_invalid_multiselect_cells(
    invalid_cell: str,
) -> None:
    metadata = pd.DataFrame(
        {
            "Expert_ID": [f"expert-{index}" for index in range(1, 6)],
            "Study_species": [invalid_cell, "['Rice']", "['Rice']", "['Rice']", "['Rice']"],
        }
    )
    category_map = synthetic_category_map(
        [("Study_species", "Rice", "Rice", 1)]
    )

    with pytest.raises(ValueError, match="valid list of non-empty strings"):
        release_module.audit_panel_category_map(metadata, category_map)
