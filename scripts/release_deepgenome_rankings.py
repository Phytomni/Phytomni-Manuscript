from __future__ import annotations

import argparse
import ast
import errno
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
PANEL_CATEGORY_MAP_COLUMNS = (
    "Dimension",
    "SourceValue",
    "PublicCategory",
    "DisplayOrder",
)
MULTISELECT_DIMENSIONS = frozenset({"Research_domains", "Study_species"})
DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
)


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            file_stat = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(file_stat.st_mode):
            raise ValueError("Crosswalk paths cannot contain a symbolic link.")


def _validated_crosswalk_path(path: Path, repo_root: Path) -> Path:
    lexical = _lexical_absolute(path)
    lexical_root = _lexical_absolute(repo_root)
    if _is_within(lexical, lexical_root):
        raise ValueError(
            "Crosswalk path is lexically contained in the repository; "
            "crosswalks must remain outside the repository."
        )

    resolved = lexical.resolve(strict=False)
    resolved_root = lexical_root.resolve(strict=False)
    if _is_within(resolved, resolved_root):
        raise ValueError(
            "Crosswalk path resolves inside the repository; "
            "crosswalks must remain outside the repository."
        )
    _reject_symlink_components(lexical)
    if not lexical.name:
        raise ValueError("The crosswalk path must name a file.")
    return lexical


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


def _open_private_parent(path: Path, *, create: bool) -> int:
    descriptor = os.open(path.anchor, DIRECTORY_OPEN_FLAGS)
    try:
        for component in path.parent.parts[1:]:
            try:
                next_descriptor = os.open(
                    component,
                    DIRECTORY_OPEN_FLAGS,
                    dir_fd=descriptor,
                )
            except FileNotFoundError as error:
                if not create:
                    raise ValueError(
                        "The crosswalk parent directory does not exist."
                    ) from error
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                try:
                    next_descriptor = os.open(
                        component,
                        DIRECTORY_OPEN_FLAGS,
                        dir_fd=descriptor,
                    )
                except OSError as open_error:
                    if open_error.errno in {errno.ELOOP, errno.ENOTDIR}:
                        raise ValueError(
                            "Crosswalk paths cannot contain a symbolic link."
                        ) from open_error
                    raise
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ValueError(
                        "Crosswalk paths cannot contain a symbolic link."
                    ) from error
                raise
            os.close(descriptor)
            descriptor = next_descriptor

        parent_stat = os.fstat(descriptor)
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise ValueError("The crosswalk parent must be a real directory.")
        if parent_stat.st_uid != os.getuid():
            raise ValueError(
                "The crosswalk parent directory must be owned by the current user."
            )
        if stat.S_IMODE(parent_stat.st_mode) != 0o700:
            raise ValueError(
                "Crosswalk parent directory permissions must be exactly 0700."
            )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _validate_crosswalk_file_stat(file_stat: os.stat_result) -> None:
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError("The external expert crosswalk must be a regular file.")
    if file_stat.st_uid != os.getuid():
        raise ValueError("The external expert crosswalk must be owned by the current user.")
    if stat.S_IMODE(file_stat.st_mode) != 0o600:
        raise ValueError("The external expert crosswalk must have exact mode 0600.")


def initialize_crosswalk(
    expert_ids: Sequence[str],
    path: Path,
    repo_root: Path = ROOT,
) -> None:
    """Create a private, randomly assigned raw-to-release identifier crosswalk."""

    raw_ids = _validate_identifier_values(expert_ids)
    if len(raw_ids) != len(set(raw_ids)):
        raise ValueError("Raw expert identifiers must be unique.")
    if len(raw_ids) > 999:
        raise ValueError("At most 999 expert identifiers can be released as E### IDs.")

    crosswalk_path = _validated_crosswalk_path(path, repo_root)
    release_ids = [f"E{index:03d}" for index in range(1, len(raw_ids) + 1)]
    secrets.SystemRandom().shuffle(release_ids)
    frame = pd.DataFrame(
        {
            "Expert": sorted(raw_ids),
            "AnonymousExpertID": release_ids,
        }
    )

    parent_descriptor = _open_private_parent(crosswalk_path, create=True)
    descriptor: int | None = None
    created = False
    try:
        try:
            descriptor = os.open(
                crosswalk_path.name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=parent_descriptor,
            )
        except FileExistsError as error:
            raise FileExistsError(
                f"Refusing to overwrite existing crosswalk: {crosswalk_path}"
            ) from error
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise ValueError(
                    "Crosswalk paths cannot contain a symbolic link."
                ) from error
            raise
        created = True
        handle = os.fdopen(descriptor, "w", encoding="utf-8", newline="")
        descriptor = None
        with handle:
            _validate_crosswalk_file_stat(os.fstat(handle.fileno()))
            frame.to_csv(handle, sep="\t", index=False, lineterminator="\n")
            handle.flush()
            _validate_crosswalk_file_stat(os.fstat(handle.fileno()))
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                os.unlink(crosswalk_path.name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        raise
    finally:
        os.close(parent_descriptor)


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
    crosswalk_path = _validated_crosswalk_path(path, repo_root)
    parent_descriptor = _open_private_parent(crosswalk_path, create=False)
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                crosswalk_path.name,
                os.O_RDONLY
                | os.O_NOFOLLOW
                | os.O_NONBLOCK
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_descriptor,
            )
        except FileNotFoundError as error:
            raise FileNotFoundError(
                "The external expert crosswalk does not exist."
            ) from error
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise ValueError(
                    "Crosswalk paths cannot contain a symbolic link."
                ) from error
            raise
        _validate_crosswalk_file_stat(os.fstat(descriptor))
        handle = os.fdopen(descriptor, "r", encoding="utf-8", newline="")
        descriptor = None
        with handle:
            frame = pd.read_csv(
                handle,
                sep="\t",
                dtype=str,
                keep_default_na=False,
            )
            _validate_crosswalk_file_stat(os.fstat(handle.fileno()))
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)

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


def _is_missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _parse_multiselect_cell(value: object) -> list[str]:
    if not isinstance(value, str):
        raise ValueError(
            "Multi-select metadata cells must be a valid list of non-empty strings."
        )
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError(
            "Multi-select metadata cells must be a valid list of non-empty strings."
        ) from error
    if not isinstance(parsed, list) or not parsed or any(
        not isinstance(item, str) or not item or item != item.strip()
        for item in parsed
    ):
        raise ValueError(
            "Multi-select metadata cells must be a valid list of non-empty strings."
        )
    return parsed


def audit_panel_category_map(
    metadata: pd.DataFrame,
    category_map: pd.DataFrame,
    expert_column: str = "Expert_ID",
    minimum_count: int = 5,
) -> pd.DataFrame:
    """Audit controlled panel categories and return aggregate counts only."""

    if not isinstance(minimum_count, int) or isinstance(minimum_count, bool):
        raise ValueError("minimum_count must be a positive integer.")
    if minimum_count < 1:
        raise ValueError("minimum_count must be a positive integer.")
    if list(category_map.columns) != list(PANEL_CATEGORY_MAP_COLUMNS):
        raise ValueError(
            "The panel category map must contain exactly these columns in order: "
            "Dimension, SourceValue, PublicCategory, DisplayOrder."
        )
    if category_map.empty:
        raise ValueError("The panel category map is empty.")
    if category_map.duplicated(["Dimension", "SourceValue"]).any():
        raise ValueError(
            "Panel category mappings must be unique by Dimension/SourceValue."
        )

    map_frame = category_map.copy()
    for column in ("Dimension", "SourceValue", "PublicCategory"):
        if any(
            not isinstance(value, str)
            or not value
            or value != value.strip()
            for value in map_frame[column].tolist()
        ):
            raise ValueError("Panel category map values must be non-empty strings.")
    try:
        display_order = pd.to_numeric(map_frame["DisplayOrder"], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("DisplayOrder values must be positive integers.") from error
    if any(float(value) != int(value) or int(value) < 1 for value in display_order):
        raise ValueError("DisplayOrder values must be positive integers.")
    map_frame["DisplayOrder"] = display_order.astype(int)
    order_counts = map_frame.groupby(
        ["Dimension", "PublicCategory"],
        sort=False,
    )["DisplayOrder"].nunique()
    if (order_counts != 1).any():
        raise ValueError(
            "Each public category must have one consistent DisplayOrder."
        )

    if expert_column not in metadata.columns:
        raise ValueError("The expert metadata table is missing its expert column.")
    expert_values = metadata[expert_column].tolist()
    if any(_is_missing(value) for value in expert_values):
        raise ValueError("Expert identifiers must not be missing.")
    try:
        for value in expert_values:
            hash(value)
    except TypeError as error:
        raise ValueError("Expert identifiers must be scalar values.") from error

    dimensions = map_frame["Dimension"].drop_duplicates().tolist()
    missing_dimensions = [
        dimension for dimension in dimensions if dimension not in metadata.columns
    ]
    if missing_dimensions:
        raise ValueError(
            "The expert metadata table is missing a mapped dimension column."
        )

    mapping = {
        (row.Dimension, row.SourceValue): row.PublicCategory
        for row in map_frame.itertuples(index=False)
    }
    expert_categories: set[tuple[str, str, object]] = set()
    for dimension in dimensions:
        for expert, raw_value in zip(
            expert_values,
            metadata[dimension].tolist(),
            strict=True,
        ):
            if _is_missing(raw_value):
                continue
            if dimension in MULTISELECT_DIMENSIONS:
                source_values = set(_parse_multiselect_cell(raw_value))
            else:
                if (
                    not isinstance(raw_value, str)
                    or not raw_value
                    or raw_value != raw_value.strip()
                ):
                    raise ValueError(
                        "Scalar metadata cells must be non-empty strings when present."
                    )
                source_values = {raw_value}
            for source_value in source_values:
                public_category = mapping.get((dimension, source_value))
                if public_category is None:
                    raise ValueError(
                        "All observed source values must be mapped before release."
                    )
                expert_categories.add((dimension, public_category, expert))

    counts: dict[tuple[str, str], int] = {}
    for dimension, public_category, _ in expert_categories:
        key = (dimension, public_category)
        counts[key] = counts.get(key, 0) + 1

    public_groups = map_frame[
        ["Dimension", "PublicCategory", "DisplayOrder"]
    ].drop_duplicates(["Dimension", "PublicCategory"])
    dimension_order = {dimension: index for index, dimension in enumerate(dimensions)}
    public_groups = public_groups.assign(
        _dimension_order=public_groups["Dimension"].map(dimension_order)
    ).sort_values(
        ["_dimension_order", "DisplayOrder", "PublicCategory"],
        kind="mergesort",
    )
    result_rows: list[dict[str, object]] = []
    for row in public_groups.itertuples(index=False):
        count = counts.get((row.Dimension, row.PublicCategory), 0)
        if count < minimum_count:
            raise ValueError(
                "Every category must meet the minimum public group size."
            )
        result_rows.append(
            {
                "Dimension": row.Dimension,
                "PublicCategory": row.PublicCategory,
                "N": count,
            }
        )
    return pd.DataFrame(
        result_rows,
        columns=["Dimension", "PublicCategory", "N"],
    )


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

    public_cells = release.astype(str).to_numpy().ravel()
    if any(
        raw_id in public_cell
        for public_cell in public_cells
        for raw_id in raw_ids
    ):
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
    parser.add_argument("--gene-categories", type=Path)
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
        if args.gene_categories is None:
            parser.error("--gene-categories is required with --use-crosswalk.")
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
