from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

REQUIRED_TARGET_KEYS = {"id", "label", "phase", "kind", "status"}


class ManifestError(ValueError):
    pass


def load_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "targets" not in data:
        raise ManifestError("manifest must be a mapping with top-level 'targets'")
    if not isinstance(data["targets"], list):
        raise ManifestError("'targets' must be a list")
    for i, t in enumerate(data["targets"]):
        if not isinstance(t, dict):
            raise ManifestError(f"targets[{i}] must be a mapping")
        missing = REQUIRED_TARGET_KEYS - set(t)
        if missing:
            raise ManifestError(f"targets[{i}] missing keys: {sorted(missing)}")
        t.setdefault("requires_data", [])
        t.setdefault("requires_toolchain", [])
        t.setdefault("expected_artifacts", [])
        t.setdefault("probes", [])
        t.setdefault("workdir", None)
        t.setdefault("kernel", "none")
        t.setdefault("path", None)
    return data


def iter_targets(manifest: dict[str, Any], phase: str | None = None) -> list[dict[str, Any]]:
    out = []
    for t in manifest["targets"]:
        if phase is not None and t.get("phase") != phase:
            continue
        out.append(t)
    return out


def check_skip(
    target: dict[str, Any],
    repo_root: Path,
    *,
    have_ir: bool,
    have_rscript: bool,
    have_ggradar: bool,
) -> str | None:
    status = target.get("status", "run")
    if status == "deprecated":
        return "deprecated: not executed"
    if status == "skip_until_data":
        return "skip_until_data (pending author data)"
    for rel in target.get("requires_data", []):
        if not (repo_root / rel).is_file():
            return f"data missing: {rel}"
    tools = set(target.get("requires_toolchain", []))
    if "ir" in tools and not have_ir:
        return 'ir kernel missing: R -e "IRkernel::installspec()"'
    if "Rscript" in tools and not have_rscript:
        return "Rscript not found"
    if "ggradar" in tools and not have_ggradar:
        return "ggradar missing: remotes::install_github('ricardo-bion/ggradar')"
    return None


def validate_artifacts(target: dict[str, Any], repo_root: Path) -> list[str]:
    problems: list[str] = []
    for rel in target.get("expected_artifacts", []):
        path = repo_root / rel
        if not path.is_file():
            problems.append(f"missing artifact: {rel}")
        elif path.stat().st_size <= 0:
            problems.append(f"empty artifact: {rel}")
    return problems


def detect_toolchain() -> dict[str, bool]:
    have_rscript = shutil.which("Rscript") is not None
    have_ir = False
    try:
        out = subprocess.check_output(
            ["jupyter", "kernelspec", "list"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        have_ir = any(
            line.split()[0] == "ir" for line in out.splitlines() if line.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        have_ir = False
    have_ggradar = False
    if have_rscript:
        have_ggradar = (
            subprocess.call(
                [
                    "Rscript",
                    "-e",
                    "if(!requireNamespace('ggradar', quietly=TRUE)) quit(status=1)",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            == 0
        )
    return {"have_ir": have_ir, "have_rscript": have_rscript, "have_ggradar": have_ggradar}


def run_figure_target(target: dict[str, Any], repo_root: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    kind = target["kind"]
    path = target["path"]
    workdir = target.get("workdir")
    with log_path.open("w", encoding="utf-8") as log:
        if kind == "notebook":
            cmd = [
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                str(repo_root / path),
            ]
            cwd = repo_root
        elif kind == "rscript":
            cwd = repo_root / workdir
            cmd = ["Rscript", Path(path).name]
        elif kind == "py_script":
            cwd = repo_root / workdir
            cmd = ["python3", Path(path).name]
        elif kind == "rmd":
            cwd = repo_root
            render = f'rmarkdown::render("{path}")'
            cmd = ["Rscript", "-e", render]
        else:
            log.write(f"unknown kind: {kind}\n")
            return 2
        log.write(f"+ {' '.join(cmd)}\n")
        proc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        return proc.returncode


def _notebook_key(target: dict[str, Any]) -> tuple[str, str] | None:
    if target.get("kind") != "notebook":
        return None
    path = target.get("path")
    if not path:
        return None
    return ("notebook", path)


def _write_skip_log(log_path: Path, reason: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"skipped: {reason}\n", encoding="utf-8")


def _format_line(mark: str, label: str, note: str | None = None) -> str:
    if note:
        return f"{mark} {label}  ({note})"
    return f"{mark} {label}"


def run_probe(probe: dict[str, Any], repo_root: Path) -> str | None:
    probe_type = probe.get("type")
    if probe_type == "file_exists":
        rel = probe["path"]
        if not (repo_root / rel).is_file():
            return f"file missing: {rel}"
        return None
    if probe_type == "file_missing":
        rel = probe["path"]
        if (repo_root / rel).is_file():
            return f"file present: {rel}"
        return None
    if probe_type == "can_import":
        module = probe["module"]
        try:
            __import__(module)
        except ImportError:
            return f"import failed: {module}"
        return None
    if probe_type in ("file_contains", "file_contains_none"):
        rel = probe["path"]
        path = repo_root / rel
        if not path.is_file():
            return f"file missing: {rel}"
        text = path.read_text(encoding="utf-8", errors="replace")
        substring = probe["substring"]
        if substring.lower() in text.lower():
            return f"placeholder present: {rel}"
        return None
    if probe_type == "note_requires":
        return probe["message"]
    return f"unknown probe type: {probe_type}"


def run_eval_probes(target: dict[str, Any], repo_root: Path) -> str | None:
    reasons: list[str] = []
    for probe in target.get("probes", []):
        reason = run_probe(probe, repo_root)
        if reason:
            reasons.append(reason)
    if reasons:
        return "; ".join(reasons)
    return None


def reproduce_main(argv: list[str], repo_root: Path) -> int:
    os.environ["PHYTOMNI_SAVE"] = "1"
    check_mode = "--check" in argv

    manifest = load_manifest(repo_root / "reproduce.manifest.yaml")
    toolchain = detect_toolchain()
    have_ir = toolchain["have_ir"]
    have_rscript = toolchain["have_rscript"]
    have_ggradar = toolchain["have_ggradar"]

    targets = iter_targets(manifest, phase="figure")
    logs_dir = repo_root / "logs"

    lines: list[str] = []
    pass_count = 0
    fail_count = 0
    skip_count = 0

    notebook_runs: dict[tuple[str, str], tuple[int, Path]] = {}

    for target in targets:
        tid = target["id"]
        label = target["label"]
        skip_reason = check_skip(
            target,
            repo_root,
            have_ir=have_ir,
            have_rscript=have_rscript,
            have_ggradar=have_ggradar,
        )

        if skip_reason:
            _write_skip_log(logs_dir / f"{tid}.log", skip_reason)
            lines.append(_format_line("⊘", label, skip_reason))
            skip_count += 1
            continue

        kind = target["kind"]
        log_path = logs_dir / f"{tid}.log"
        nb_key = _notebook_key(target)

        if nb_key is not None and nb_key in notebook_runs:
            run_rc, primary_log = notebook_runs[nb_key]
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                f"shared execution; see {primary_log.relative_to(repo_root)}\n",
                encoding="utf-8",
            )
        elif nb_key is not None:
            run_rc = run_figure_target(target, repo_root, log_path)
            notebook_runs[nb_key] = (run_rc, log_path)
        else:
            run_rc = run_figure_target(target, repo_root, log_path)

        artifact_problems = validate_artifacts(target, repo_root)

        if run_rc != 0:
            lines.append(_format_line("✘", label, f"exit {run_rc}"))
            fail_count += 1
        elif artifact_problems:
            lines.append(_format_line("✘", label, artifact_problems[0]))
            fail_count += 1
        else:
            lines.append(_format_line("✓", label))
            pass_count += 1

    for line in lines:
        print(line)
    total = pass_count + fail_count + skip_count
    print("————————————————————————————————————————————")
    print(f"{total} targets / {pass_count} ok / {fail_count} failed / {skip_count} skipped")

    eval_targets = iter_targets(manifest, phase="eval")
    if eval_targets:
        print("————————————————————————————————————————————")
        print("Agent evaluation (not figures — probes only)")
        eval_pass = 0
        eval_skip = 0
        for target in eval_targets:
            tid = target["id"]
            label = target["label"]
            log_path = logs_dir / f"eval-{tid}.log"
            skip_reason = run_eval_probes(target, repo_root)
            if skip_reason:
                _write_skip_log(log_path, skip_reason)
                print(_format_line("⊘", label, skip_reason))
                eval_skip += 1
            else:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("all probes passed\n", encoding="utf-8")
                print(_format_line("✓", label))
                eval_pass += 1
        print("————————————————————————————————————————————")
        print(f"{len(eval_targets)} eval targets / {eval_pass} ok / {eval_skip} skipped")

    if check_mode and fail_count > 0:
        return 1
    return 0
