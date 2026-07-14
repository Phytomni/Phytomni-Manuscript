import hashlib
import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.freeze_deepgenome_rankings import freeze_rankings


MODELS = ("Gemini", "Grok", "OpenAI", "Phytomni", "Claude")


def write_balanced_fixture(tmp_path: Path) -> tuple[Path, Path]:
    genes = [
        ("Arabidopsis", "AT_well", "well_studied"),
        ("Arabidopsis", "AT_unknown", "uncharacterized"),
        ("Rice", "OS_well", "well_studied"),
        ("Rice", "OS_unknown", "uncharacterized"),
    ]
    rows: list[dict[str, str]] = []
    for species, gene, _ in genes:
        for expert_index, order in enumerate(permutations(MODELS)):
            row = {
                "Species": species,
                "Gene": gene,
                "Expert": f"expert-{species}-{gene}-{expert_index}",
            }
            row.update(
                {model: f"R{rank}" for rank, model in enumerate(order, start=1)}
            )
            rows.append(row)

    score_path = tmp_path / "score.tsv"
    pd.DataFrame(rows).to_csv(score_path, sep="\t", index=False)
    categories_path = tmp_path / "gene_categories.tsv"
    pd.DataFrame(
        genes,
        columns=["Species", "Gene", "StudyStatus"],
    ).to_csv(categories_path, sep="\t", index=False)
    return score_path, categories_path


def test_freezer_writes_deterministic_figure_tables(tmp_path: Path) -> None:
    score_path, categories_path = write_balanced_fixture(tmp_path)
    output_dir = tmp_path / "frozen"

    freeze_rankings(
        score_path=score_path,
        gene_categories_path=categories_path,
        output_dir=output_dir,
        model_columns=MODELS,
    )
    first_bytes = {
        path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
    }
    freeze_rankings(
        score_path=score_path,
        gene_categories_path=categories_path,
        output_dir=output_dir,
        model_columns=MODELS,
    )
    second_bytes = {
        path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
    }

    assert first_bytes == second_bytes
    assert set(first_bytes) == {
        "pl_pairwise.tsv",
        "pl_scores.tsv",
        "provenance.json",
        "rank_distribution.tsv",
    }

    ranks = pd.read_csv(output_dir / "rank_distribution.tsv", sep="\t")
    scores = pd.read_csv(output_dir / "pl_scores.tsv", sep="\t")
    pairwise = pd.read_csv(output_dir / "pl_pairwise.tsv", sep="\t")
    scopes = set(ranks["Scope"])

    assert scopes == {
        "overall",
        "arabidopsis",
        "rice",
        "well_studied",
        "well_studied.arabidopsis",
        "well_studied.rice",
        "uncharacterized",
        "uncharacterized.arabidopsis",
        "uncharacterized.rice",
    }
    assert len(ranks) == len(scopes) * len(MODELS) ** 2
    assert len(scores) == len(scopes) * len(MODELS)
    assert len(pairwise) == len(scopes) * len(MODELS) ** 2
    assert set(ranks["Model"]) == set(MODELS)
    assert set(ranks["Rank"]) == {"R1", "R2", "R3", "R4", "R5"}
    np.testing.assert_allclose(
        ranks.groupby(["Scope", "Model"])["Fraction"].sum().to_numpy(),
        1.0,
    )
    np.testing.assert_allclose(scores["Elo"], 1500.0, atol=1e-8)
    off_diagonal = pairwise["RowModel"] != pairwise["ColumnModel"]
    np.testing.assert_allclose(
        pairwise.loc[off_diagonal, "Probability"],
        0.5,
        atol=1e-10,
    )
    assert pairwise.loc[~off_diagonal, "Probability"].isna().all()

    provenance = json.loads((output_dir / "provenance.json").read_text())
    assert provenance["schema_version"] == 1
    assert provenance["source"]["sha256"] == hashlib.sha256(
        score_path.read_bytes()
    ).hexdigest()
    assert provenance["source"]["rows"] == 480
    assert provenance["used_rows"] == 480
    assert provenance["skipped_rows"] == 0
    assert provenance["model_columns"] == list(MODELS)
    assert provenance["reference_model"] == "Gemini"


def test_freezer_rejects_incomplete_rankings(tmp_path: Path) -> None:
    score_path, categories_path = write_balanced_fixture(tmp_path)
    score = pd.read_csv(score_path, sep="\t")
    score.loc[0, "Claude"] = "R4"
    score.to_csv(score_path, sep="\t", index=False)

    try:
        freeze_rankings(
            score_path=score_path,
            gene_categories_path=categories_path,
            output_dir=tmp_path / "frozen",
            model_columns=MODELS,
        )
    except ValueError as error:
        assert "complete permutation" in str(error)
    else:
        raise AssertionError("Invalid rankings must fail before freezing")
