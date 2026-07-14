from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import secrets
import stat
from collections.abc import Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_COLUMNS = ("Gemini", "Grok", "OpenAI", "Phytomni", "Claude")
PUBLIC_COLUMNS = (
    "AnonymousExpertID",
    "Species",
    "Gene",
    "StudyStatus",
    *MODEL_COLUMNS,
)
STUDY_STATUSES = frozenset({"well_studied", "uncharacterized"})
RELEASE_ID_PATTERN = re.compile(r"E[0-9]{3}")


def _outside_repository(path: Path, repo_root: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved_root = repo_root.expanduser().resolve()
    if resolved == resolved_root or resolved_root in resolved.parents:
        raise ValueError("The expert crosswalk must be stored outside the repository.")
    return resolved


def _validate_identifier_values(values: Sequence[object]) -> list[str]:
    identifiers = list(values)
    if not identifiers:
        raise ValueError("At least one raw expert identifier is required.")
    if any(
        not isinstance(identifier, str)
        or not identifier
        or identifier != identifier.strip()
        or "\t" in identifier
        or "\n" in identifier
        or "\r" in identifier
        for identifier in identifiers
    ):
        raise ValueError("Raw expert identifiers must be non-empty, trimmed strings.")
    return identifiers


def _private_parent(path: Path) -> None:
    parent = path.parent
    if parent.exists():
        if not parent.is_dir():
            raise ValueError("The crosswalk parent must be a directory.")
    else:
        parent.mkdir(parents=True, mode=0o700)
    if stat.S_IMODE(parent.stat().st_mode) != 0o700:
        raise ValueError("Crosswalk parent directory permissions must be exactly 0700.")


def initialize_crosswalk(
    expert_ids: Sequence[str],
    path: Path,
    repo_root: Path = ROOT,
) -> None:
    """Create a private, randomly assigned raw-to-release identifier crosswalk."""

    resolved = _outside_repository(path, repo_root)
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite existing crosswalk: {resolved}")

    raw_ids = _validate_identifier_values(expert_ids)
    if len(raw_ids) != len(set(raw_ids)):
        raise ValueError("Raw expert identifiers must be unique.")
    if len(raw_ids) > 999:
        raise ValueError("At most 999 expert identifiers can be released as E### IDs.")

    _private_parent(resolved)
    release_ids = [f"E{index:03d}" for index in range(1, len(raw_ids) + 1)]
    secrets.SystemRandom().shuffle(release_ids)
    frame = pd.DataFrame(
        {
            "Expert": sorted(raw_ids),
            "AnonymousExpertID": release_ids,
        }
    )

    descriptor = os.open(
        resolved,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, sep="\t", index=False, lineterminator="\n")
        resolved.chmod(0o600)
    except BaseException:
        resolved.unlink(missing_ok=True)
        raise

    if stat.S_IMODE(resolved.stat().st_mode) != 0o600:
        resolved.unlink(missing_ok=True)
        raise PermissionError("Crosswalk file permissions must be exactly 0600.")


def _validate_score(score: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    required = {"Expert", "Species", "Gene", *MODEL_COLUMNS}
    missing = sorted(required - set(score.columns))
    if missing:
        raise ValueError("The private score table is missing required release columns.")
    if score.empty:
        raise ValueError("The private score table is empty.")

    frame = score.copy()
    identifiers = _validate_identifier_values(frame["Expert"].tolist())
    raw_ids = set(identifiers)
    for column in ("Species", "Gene"):
        values = frame[column].tolist()
        if any(
            not isinstance(value, str)
            or not value
            or value != value.strip()
            for value in values
        ):
            raise ValueError(f"{column} values must be non-empty, trimmed strings.")
    if frame.duplicated(["Species", "Gene", "Expert"]).any():
        raise ValueError("Duplicate Species/Gene/Expert judgments are not allowed.")

    expected_ranks = {f"R{rank}" for rank in range(1, len(MODEL_COLUMNS) + 1)}
    valid_rankings = frame[list(MODEL_COLUMNS)].apply(
        lambda row: set(row.tolist()) == expected_ranks,
        axis=1,
    )
    if not valid_rankings.all():
        raise ValueError("Every ranking row must be a complete permutation of R1-R5.")
    return frame, raw_ids


def _validate_categories(
    categories: pd.DataFrame,
    score: pd.DataFrame,
) -> pd.DataFrame:
    required = {"Species", "Gene", "StudyStatus"}
    if not required.issubset(categories.columns):
        raise ValueError("The gene category table is missing required columns.")
    frame = categories.loc[:, ["Species", "Gene", "StudyStatus"]].copy()
    if frame.empty:
        raise ValueError("The gene category table is empty.")
    if frame.duplicated(["Species", "Gene"]).any():
        raise ValueError("Gene categories must be unique by Species/Gene.")
    for column in ("Species", "Gene", "StudyStatus"):
        if any(
            not isinstance(value, str)
            or not value
            or value != value.strip()
            for value in frame[column].tolist()
        ):
            raise ValueError("Gene category values must be non-empty, trimmed strings.")
    if not set(frame["StudyStatus"]).issubset(STUDY_STATUSES):
        raise ValueError("StudyStatus must be well_studied or uncharacterized.")

    score_genes = set(zip(score["Species"], score["Gene"], strict=True))
    category_genes = set(zip(frame["Species"], frame["Gene"], strict=True))
    if score_genes != category_genes:
        raise ValueError(
            "Gene categories must describe exactly the Species/Gene pairs in the score table."
        )
    return frame


def _load_crosswalk(
    path: Path,
    raw_ids: set[str],
    repo_root: Path,
) -> pd.DataFrame:
    resolved = _outside_repository(path, repo_root)
    if not resolved.is_file():
        raise FileNotFoundError("The external expert crosswalk does not exist.")
    if stat.S_IMODE(resolved.stat().st_mode) != 0o600:
        raise ValueError("The external expert crosswalk must have exact mode 0600.")

    frame = pd.read_csv(resolved, sep="\t", dtype=str, keep_default_na=False)
    expected_columns = ["Expert", "AnonymousExpertID"]
    if list(frame.columns) != expected_columns:
        raise ValueError(
            "The crosswalk must contain exactly these columns in order: "
            "Expert, AnonymousExpertID."
        )
    crosswalk_raw = _validate_identifier_values(frame["Expert"].tolist())
    if len(crosswalk_raw) != len(set(crosswalk_raw)):
        raise ValueError("Crosswalk raw Expert IDs must be unique.")
    if set(crosswalk_raw) != raw_ids:
        raise ValueError("Crosswalk raw Expert IDs must exactly match the score table.")

    release_ids = frame["AnonymousExpertID"].tolist()
    if len(release_ids) != len(set(release_ids)):
        raise ValueError("Crosswalk release IDs must be unique.")
    if not all(RELEASE_ID_PATTERN.fullmatch(value) for value in release_ids):
        raise ValueError("Crosswalk release IDs must use the E### format.")
    expected_release_ids = {
        f"E{index:03d}" for index in range(1, len(release_ids) + 1)
    }
    if set(release_ids) != expected_release_ids:
        raise ValueError("Crosswalk release IDs must be a contiguous E### set from E001.")
    return frame


def build_release(
    score: pd.DataFrame,
    categories: pd.DataFrame,
    crosswalk_path: Path,
    *,
    repo_root: Path = ROOT,
) -> pd.DataFrame:
    """Build the canonical public ranking table after all privacy checks pass."""

    score_frame, raw_ids = _validate_score(score)
    category_frame = _validate_categories(categories, score_frame)
    crosswalk = _load_crosswalk(crosswalk_path, raw_ids, repo_root)

    release = score_frame.merge(
        crosswalk,
        on="Expert",
        how="left",
        validate="many_to_one",
    ).merge(
        category_frame,
        on=["Species", "Gene"],
        how="left",
        validate="many_to_one",
    )
    release = release.loc[:, list(PUBLIC_COLUMNS)]
    if release.isna().any().any():
        raise ValueError("The public release contains missing values.")

    public_cells = set(release.astype(str).to_numpy().ravel())
    if raw_ids & public_cells:
        raise ValueError("A raw expert identifier leaked into a public release cell.")

    return release.sort_values(
        ["AnonymousExpertID", "Species", "Gene"],
        kind="mergesort",
    ).reset_index(drop=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a private random crosswalk or a public ranking release."
    )
    parser.add_argument("--score-tsv", type=Path, required=True)
    parser.add_argument("--gene-categories", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--initialize-crosswalk", action="store_true")
    modes.add_argument("--use-crosswalk", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def _redact_raw_ids(message: str, raw_ids: Sequence[str]) -> str:
    safe = message
    for raw_id in sorted(raw_ids, key=len, reverse=True):
        safe = safe.replace(raw_id, "[redacted]")
    return safe


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    raw_ids: list[str] = []
    try:
        score = pd.read_csv(
            args.score_tsv,
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )
        if "Expert" not in score.columns:
            raise ValueError("The private score table is missing the Expert column.")
        raw_ids = _validate_identifier_values(score["Expert"].tolist())
        unique_raw_ids = sorted(set(raw_ids))

        if args.initialize_crosswalk:
            if args.output is not None:
                parser.error("--output is only valid with --use-crosswalk.")
            initialize_crosswalk(unique_raw_ids, args.crosswalk, ROOT)
            print(f"Initialized a private crosswalk for {len(unique_raw_ids)} experts.")
            return 0

        if args.output is None:
            parser.error("--output is required with --use-crosswalk.")
        categories = pd.read_csv(
            args.gene_categories,
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )
        release = build_release(score, categories, args.crosswalk)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        release.to_csv(
            args.output,
            sep="\t",
            index=False,
            lineterminator="\n",
        )
        print(f"Wrote {len(release)} anonymized ranking rows.")
        return 0
    except (FileExistsError, FileNotFoundError, PermissionError, ValueError) as error:
        parser.error(_redact_raw_ids(str(error), raw_ids))


if __name__ == "__main__":
    raise SystemExit(main())
