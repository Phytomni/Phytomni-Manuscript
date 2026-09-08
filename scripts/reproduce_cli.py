from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from scripts.render_reproduce_matrix import check_readme_matrix
from scripts.reproduce_lib import (
    ManifestError,
    TargetSelectionError,
    figure_prerequisites,
    load_manifest,
    run_eval_probes,
    select_targets,
    validate_manifest,
    validate_static,
)


def _positive_seconds(value: str) -> int:
    try:
        seconds = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("timeout must be a positive integer") from None
    if seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be a positive integer")
    return seconds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce public manuscript figures", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("list", help="List target declarations without runtime probes", allow_abbrev=False)
    for name, help_text in (
        ("validate", "Check public sources, inputs, and generated documentation"),
        ("run", "Execute selected figure sources and publish validated artifacts"),
        ("probe", "Inspect evaluation prerequisites without executing evaluators"),
    ):
        command = commands.add_parser(name, help=help_text, allow_abbrev=False)
        command.add_argument("--target", dest="target_ids", action="append", metavar="ID")
        if name == "run":
            command.add_argument("--timeout", type=_positive_seconds, default=600, metavar="SECONDS")
    return parser


def parse_request(argv: list[str], manifest: dict[str, Any]) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    phase = "eval" if args.command == "probe" else "figure" if args.command in {"run", "validate"} else None
    try:
        args.targets = select_targets(manifest, getattr(args, "target_ids", None), phase=phase)
    except TargetSelectionError as exc:
        parser.error(str(exc))
    return args


def run_read_only_command(args: argparse.Namespace, manifest: dict[str, Any], repo_root: Path) -> int:
    """Read-only commands neither prepare toolchains nor run figure/evaluation code."""
    if args.command is None:
        build_parser().print_help()
        return 0
    if args.command not in {"list", "validate", "probe"}:
        raise ValueError("not a read-only command")
    try:
        validate_manifest(manifest, repo_root)
        if args.command == "list":
            for target in args.targets:
                if target["phase"] == "eval":
                    availability = "probe not run; evaluation not executed"
                else:
                    result = figure_prerequisites(target, repo_root)
                    availability = f"{result.state}; toolchain not checked"
                    if result.reasons:
                        availability += "; " + "; ".join(result.reasons)
                print(f"{target['id']}: {target['label']} [{availability}]")
                for artifact in target.get("expected_artifacts", []):
                    print(f"  output: {artifact}")
            return 0
        if args.command == "validate":
            problems = validate_static(manifest, repo_root, target_ids=getattr(args, "target_ids", None))
            problems.extend(check_readme_matrix(repo_root / "README.md", manifest, repo_root))
            for problem in problems:
                print(problem)
            if not problems:
                print("Public source/input contracts and README matrix are valid; runtime not checked.")
            return int(bool(problems))
        unavailable = 0
        for target in args.targets:
            reason = run_eval_probes(target, repo_root)
            if reason:
                unavailable += 1
            print(f"{target['id']}: {'unavailable' if reason else 'prerequisites discoverable'}; evaluation not executed")
            if reason:
                print(f"  {reason}")
        return int(bool(unavailable))
    except (ManifestError, OSError) as exc:
        print(f"Validation failed: {exc}")
        return 1


def main(argv: list[str] | None = None, *, repo_root: Path | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    initial = parser.parse_args(argv)
    if initial.command is None:
        parser.print_help()
        return 0
    root = Path(__file__).resolve().parents[1] if repo_root is None else repo_root.resolve()
    try:
        manifest = load_manifest(root / "reproduce.manifest.yaml")
        validate_manifest(manifest, root)
    except (ManifestError, OSError) as exc:
        print(f"Manifest validation failed: {exc}", file=sys.stderr)
        return 1
    args = parse_request(argv, manifest)
    if args.command != "run":
        return run_read_only_command(args, manifest, root)
    from scripts.reproduce_execution import run_targets

    return run_targets(manifest, root, target_ids=args.target_ids, timeout=args.timeout).exit_code


if __name__ == "__main__":
    raise SystemExit(main())
