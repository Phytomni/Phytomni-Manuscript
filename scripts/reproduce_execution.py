"""Execute declared public figure inputs in isolated, source-preserving copies."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection

from PIL import Image
from pypdf import PdfReader

from scripts.reproduce_lib import (
    ManifestError, TargetSelectionError, figure_prerequisites, repository_path,
    select_targets, validate_figure_sources, validate_manifest,
)


@dataclass(frozen=True)
class ExecutionUnit:
    source: str
    targets: tuple[dict[str, Any], ...]
    inputs: tuple[str, ...]
    artifacts: tuple[str, ...]

    @property
    def key(self) -> str:
        return hashlib.sha256(self.source.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    timed_out: bool = False


@dataclass(frozen=True)
class RunResult:
    exit_code: int
    run_dir: Path
    summary: dict[str, Any]


class ArtifactError(ValueError):
    pass


class PublicationError(RuntimeError):
    def __init__(self, reason_code: str, status: str):
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.status = status


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _fingerprint(path: Path) -> str | None:
    if path.is_symlink():
        raise PublicationError("publication_conflict", "not_attempted")
    if not path.exists():
        return None
    if not path.is_file():
        raise PublicationError("publication_conflict", "not_attempted")
    return _hash(path)


def _artifact_path(root: Path, rel: str) -> Path:
    """Keep lexical paths so resolving a symlink cannot hide its presence."""
    path = root
    for part in Path(rel).parts:
        path = path / part
        if path.is_symlink():
            raise PublicationError("publication_conflict", "not_attempted")
    repository_path(root, rel, "artifact")
    return path


def build_execution_units(targets: Collection[dict[str, Any]]) -> list[ExecutionUnit]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for target in targets:
        groups.setdefault(target["path"], []).append(target)
    return [ExecutionUnit(
        source, tuple(members),
        tuple(dict.fromkeys([source] + [rel for t in members for rel in t.get("requires_data", [])])),
        tuple(rel for t in members for rel in t["expected_artifacts"]),
    ) for source, members in groups.items()]


def inspect_artifact(path: Path) -> dict[str, Any]:
    """Decode publication files; file validity is not scientific/visual validity."""
    try:
        if not path.is_file() or path.stat().st_size == 0 or path.is_symlink():
            raise ArtifactError("artifact missing, empty, or symbolic")
        suffix = path.suffix.lower()
        if suffix == ".png":
            with Image.open(path) as image:
                if image.format != "PNG":
                    raise ArtifactError("artifact is not PNG")
                image.verify()
            with Image.open(path) as image:
                image.load()
                width, height = image.size
            if width <= 0 or height <= 0:
                raise ArtifactError("empty image dimensions")
            details = {"format": "png", "width": width, "height": height}
        elif suffix == ".pdf":
            reader = PdfReader(path, strict=True)
            if reader.is_encrypted or not reader.pages:
                raise ArtifactError("encrypted or empty PDF")
            for page in reader.pages:
                contents = page.get_contents()
                if float(page.mediabox.width) <= 0 or float(page.mediabox.height) <= 0 or contents is None or not contents.get_data().strip():
                    raise ArtifactError("PDF page has no content or positive geometry")
            details = {"format": "pdf", "pages": len(reader.pages)}
        else:
            raise ArtifactError("unsupported artifact format")
        return {**details, "size_bytes": path.stat().st_size, "sha256": _hash(path)}
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError("artifact cannot be decoded") from exc


def _git_identity(root: Path) -> dict[str, Any]:
    if not (root / ".git").exists():
        return {"revision": None, "dirty": None, "identity": "source/input hashes"}
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                  text=True, check=True, timeout=5).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"],
                                cwd=root, capture_output=True, text=True, check=True, timeout=5).stdout
        if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
            raise ValueError("invalid Git revision")
        return {"revision": revision, "dirty": bool(status), "identity": "Git and source/input hashes"}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None, "identity": "source/input hashes"}


def _environment(run_dir: Path, *, require_r: bool) -> tuple[dict[str, str], set[str], dict[str, Any]]:
    allowed = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "LD_LIBRARY_PATH",
               "FONTCONFIG_PATH", "FONTCONFIG_FILE")
    environment = {key: os.environ[key] for key in allowed if key in os.environ}
    environment.update({
        "PATH": str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", os.defpath),
        "PHYTOMNI_SAVE": "1", "PYTHONNOUSERSITE": "1", "PYTHONIOENCODING": "utf-8",
        "IPYTHONDIR": str(run_dir / "ipython"), "MPLCONFIGDIR": str(run_dir / "cache/matplotlib"),
        "XDG_CACHE_HOME": str(run_dir / "cache"), "JUPYTER_DATA_DIR": str(run_dir / "jupyter"),
        "JUPYTER_PATH": str(run_dir / "jupyter"), "JUPYTER_RUNTIME_DIR": str(run_dir / "runtime"),
        "JUPYTER_NO_CONFIG": "1",
        "R_PROFILE_USER": "/dev/null", "R_ENVIRON_USER": "/dev/null",
        "RENV_CONFIG_AUTOLOADER_ENABLED": "FALSE",
    })
    packages = {}
    for name in ("nbconvert", "nbformat", "ipykernel", "numpy", "pandas", "plotly", "kaleido", "matplotlib", "Pillow", "pypdf", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    metadata: dict[str, Any] = {"python_version": sys.version.split()[0], "python_environment":
                              "virtualenv" if sys.prefix != sys.base_prefix else "base",
                              "os": platform.system(), "os_release": platform.release(),
                              "architecture": platform.machine(), "packages": packages}
    tools: set[str] = set()
    kernel = {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
              "display_name": "Prepared Python", "language": "python"}
    kernel_dir = run_dir / "jupyter/kernels/python3"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(json.dumps(kernel), encoding="utf-8")
    if not require_r:
        return environment, tools, metadata
    library = os.environ.get("R_LIBS_USER", "")
    if not library or not Path(library).is_absolute() or not Path(library).is_dir():
        metadata["r_toolchain"] = "unavailable: set R_LIBS_USER to an explicitly prepared library"
        return environment, tools, metadata
    empty_site = run_dir / "empty-r-site"
    empty_site.mkdir()
    environment.update({"R_LIBS_USER": library, "R_LIBS": library, "R_LIBS_SITE": str(empty_site)})
    rscript = shutil.which("Rscript", path=environment["PATH"])
    r = shutil.which("R", path=environment["PATH"])
    if rscript is None or r is None:
        metadata["r_toolchain"] = "unavailable: R and Rscript are required"
        return environment, tools, metadata
    expression = ('cat("R=", as.character(getRversion()), "\\n", sep=""); '
                  'for (p in c("IRkernel", "ggradar")) '
                  'cat(p, "=", if (requireNamespace(p, lib.loc=Sys.getenv("R_LIBS_USER"), quietly=TRUE)) '
                  'as.character(packageVersion(p, lib.loc=Sys.getenv("R_LIBS_USER"))) else "missing", "\\n", sep="")')
    try:
        outcome = subprocess.run([rscript, "--vanilla", "-e", expression], env=environment,
                                 capture_output=True, text=True, timeout=30, check=False)
        (run_dir / "raw/r-toolchain.log").write_text(outcome.stdout + outcome.stderr, encoding="utf-8")
        values = dict(line.split("=", 1) for line in outcome.stdout.splitlines() if "=" in line)
        if outcome.returncode or not re.fullmatch(r"\d+\.\d+\.\d+", values.get("R", "")):
            metadata["r_toolchain"] = "unavailable: prepared R library probe failed"
            return environment, tools, metadata
        metadata["r_version"] = values["R"]
        metadata["r_library"] = "explicit R_LIBS_USER; user profiles disabled"
        tools.add("Rscript")
        for package, tool in (("IRkernel", "ir"), ("ggradar", "ggradar")):
            version = values.get(package, "")
            if re.fullmatch(r"[0-9]+(?:[.-][0-9]+)+", version):
                tools.add(tool)
                metadata.setdefault("r_packages", {})[package] = version
        if "ir" in tools:
            directory = run_dir / "jupyter/kernels/ir"
            directory.mkdir(parents=True)
            (directory / "kernel.json").write_text(json.dumps({
                "argv": [r, "--vanilla", "--slave", "-e", "IRkernel::main()", "--args", "{connection_file}"],
                "display_name": "Prepared R", "language": "R",
            }), encoding="utf-8")
    except (OSError, subprocess.SubprocessError):
        metadata["r_toolchain"] = "unavailable: prepared R library probe failed"
    return environment, tools, metadata


def execute_unit(unit: ExecutionUnit, work_root: Path, raw_log: Path, *,
                 timeout: int, environment: dict[str, str]) -> ProcessResult:
    target = unit.targets[0]
    source = work_root / unit.source
    cwd = source.parent
    if target["kind"] == "notebook":
        command = [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                   "--ExecutePreprocessor.kernel_name=" + ("python3" if target["kernel"] == "py" else "ir"),
                   f"--ExecutePreprocessor.timeout={timeout}", str(source)]
    elif target["kind"] == "py_script":
        command = [sys.executable, str(source)]
    elif target["kind"] == "rscript":
        command = [shutil.which("Rscript", path=environment["PATH"]), "--vanilla", str(source)]
    else:
        raise ValueError("unsupported execution kind")
    command = [sys.executable, str(Path(__file__).with_name("reproduce_worker.py")), *command]
    with raw_log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return ProcessResult(process.wait(timeout=timeout))
        except subprocess.TimeoutExpired:
            return ProcessResult(1, timed_out=True)
        finally:
            _terminate_group(process)


def _terminate_group(process: subprocess.Popen) -> None:
    # The dedicated subreaper also cleans kernels that start separate sessions.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=2)


def _unchanged(root: Path, inputs: dict[str, str]) -> bool:
    try:
        return all(_fingerprint(repository_path(root, rel, "input")) == digest for rel, digest in inputs.items())
    except (ManifestError, OSError, PublicationError):
        return False


def _stage_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())


def publish_unit(unit: ExecutionUnit, root: Path, work_root: Path, run_dir: Path,
                 inputs: dict[str, str], destinations: dict[str, str | None],
                 artifacts: dict[str, dict[str, Any]]) -> str:
    lock_dir = root / "logs/reproduce/locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = repository_path(root, f"logs/reproduce/locks/{unit.key}.lock", "publication lock")
    backup_root = run_dir / "backups" / unit.key
    installed: list[str] = []
    staged: list[Path] = []
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PublicationError("publication_conflict", "not_attempted") from None
        try:
            for rel in unit.artifacts:
                destination = _artifact_path(root, rel)
                if not _unchanged(root, inputs) or _fingerprint(destination) != destinations[rel]:
                    raise PublicationError("publication_conflict", "not_attempted")
                if destinations[rel] is not None:
                    _stage_file(destination, backup_root / rel)
                    if _hash(backup_root / rel) != destinations[rel]:
                        raise PublicationError("publication_conflict", "not_attempted")
                staged_path = destination.with_name(f".{destination.name}.{run_dir.name}.tmp")
                staged.append(staged_path)
                _stage_file(_artifact_path(work_root, rel), staged_path)
                if _hash(staged_path) != artifacts[rel]["sha256"]:
                    raise PublicationError("publication_conflict", "not_attempted")
                destination = _artifact_path(root, rel)
                if not _unchanged(root, inputs) or _fingerprint(destination) != destinations[rel]:
                    raise PublicationError("publication_conflict", "not_attempted")
                os.replace(staged_path, destination)
                installed.append(rel)
            return "succeeded"
        except Exception as exc:
            conflict = isinstance(exc, PublicationError)
            rollback_failed = False
            for rel in reversed(installed):
                try:
                    destination = _artifact_path(root, rel)
                    if _fingerprint(destination) != artifacts[rel]["sha256"]:
                        conflict = True
                        continue
                    if destinations[rel] is None:
                        destination.unlink()
                    else:
                        restore = destination.with_name(f".{destination.name}.{run_dir.name}.restore")
                        staged.append(restore)
                        _stage_file(backup_root / rel, restore)
                        destination = _artifact_path(root, rel)
                        if _fingerprint(destination) != artifacts[rel]["sha256"]:
                            conflict = True
                            continue
                        os.replace(restore, destination)
                except (PublicationError, ManifestError):
                    conflict = True
                except Exception:
                    rollback_failed = True
            status = "partial" if rollback_failed else "conflict" if conflict else "rolled_back"
            raise PublicationError("publication_conflict" if conflict else "publication_failed", status) from exc
        finally:
            for path in staged:
                try:
                    rel = path.relative_to(root).as_posix()
                    _artifact_path(root, rel).unlink(missing_ok=True)
                except (OSError, ManifestError, PublicationError):
                    pass
            fcntl.flock(lock, fcntl.LOCK_UN)


def _finish(run_dir: Path, summary: dict[str, Any], started: float) -> RunResult:
    summary["finished_utc"] = _utc()
    summary["duration_seconds"] = round(time.monotonic() - started, 3)
    records = summary["targets"]
    summary["coverage"].update({
        "selected_targets": len(records),
        "succeeded_targets": sum(t["status"] == "succeeded" for t in records),
        "failed_targets": sum(t["status"] == "failed" for t in records),
        "pending_targets": sum(t["status"] == "pending" for t in records),
    })
    code = summary.get("exit_code", 0)
    if summary["coverage"]["failed_targets"] or not summary["coverage"]["executed_units"]:
        code = code or 1
    summary["exit_code"] = code
    shareable = run_dir / "shareable"
    (shareable / "targets").mkdir(parents=True, exist_ok=True)
    for record in records:
        (shareable / "targets" / f"{record['id']}.log").write_text(
            f"{record['id']}: {record['status']} ({record['reason_code']})\n"
            f"Publication: {record['publication_status']}\n", encoding="utf-8")
    (shareable / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    coverage = summary["coverage"]
    print(f"{coverage['selected_targets']} targets / {coverage['executed_units']} executions / "
          f"{coverage['succeeded_targets']} succeeded / {coverage['failed_targets']} failed / {coverage['pending_targets']} pending")
    print(f"Shareable summary: logs/reproduce/{run_dir.name}/shareable/summary.json")
    return RunResult(code, run_dir, summary)


def run_targets(manifest: dict[str, Any], repo_root: Path, *,
                target_ids: Collection[str] | None = None, timeout: int = 600) -> RunResult:
    """Run only declared public figures; unavailable private evaluations are not run."""
    root = repo_root.resolve()
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("timeout must be a positive integer")
    started = time.monotonic()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:10]
    run_dir = repository_path(root, f"logs/reproduce/{run_id}", "run directory")
    (run_dir / "raw").mkdir(parents=True)
    (run_dir / "work").mkdir()
    summary: dict[str, Any] = {"schema_version": 1, "run_id": run_id, "operation": "public_figure_reproduction",
        "started_utc": _utc(), "source_identity": _git_identity(root), "environment": {},
        "coverage": {"executed_units": 0}, "targets": [], "exit_code": 0}
    try:
        validate_manifest(manifest, root)
        selected = select_targets(manifest, target_ids, phase="figure")
    except (ManifestError, TargetSelectionError):
        (run_dir / "raw/preflight.log").write_text(traceback.format_exc(), encoding="utf-8")
        summary.update(exit_code=1, reason_code="invalid_manifest_or_selection")
        return _finish(run_dir, summary, started)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["manifest_sha256"] = _hash(run_dir / "manifest.json")
    records = {target["id"]: {"id": target["id"], "label": target["label"], "source": target["path"],
        "status": "failed", "reason_code": "not_executed", "publication_status": "not_attempted",
        "export_status": "not_attempted", "backup_status": "not_created", "backups": [],
        "execution_unit": None, "raw_log": None, "inputs_root": None, "work_root": None,
        "started_utc": None, "finished_utc": None, "duration_seconds": 0.0, "input_hashes": {}, "artifacts": []}
        for target in selected}
    summary["targets"] = list(records.values())
    problems = validate_figure_sources(manifest, root)
    if problems:
        (run_dir / "raw/preflight.log").write_text("\n".join(problems) + "\n", encoding="utf-8")
        for record in records.values():
            record["reason_code"] = "invalid_source_contract"
        return _finish(run_dir, summary, started)
    require_r = any(t.get("kernel") == "ir" or t["kind"] == "rscript" for t in selected)
    environment, available_tools, metadata = _environment(run_dir, require_r=require_r)
    summary["environment"] = metadata
    runnable = []
    for target in selected:
        check = figure_prerequisites(target, root, available_tools=available_tools, explicit=bool(target_ids))
        if check.state == "ready":
            runnable.append(target)
        else:
            record = records[target["id"]]
            record.update(status=check.state, reason_code="pending_author_data" if check.state == "pending" else "prerequisite_failed",
                          reasons=list(check.reasons))
    for unit in build_execution_units(runnable):
        unit_started = time.monotonic()
        members = [records[t["id"]] for t in unit.targets]
        for record in members:
            record.update(started_utc=_utc(), execution_unit=unit.key, raw_log=f"raw/{unit.key}.log",
                          inputs_root=f"inputs/{unit.key}", work_root=f"work/{unit.key}")
        print(f"Starting {', '.join(t['id'] for t in unit.targets)}: {unit.source}", flush=True)
        work_root = run_dir / "work" / unit.key
        output_dir = work_root / Path(unit.source).parent / "output"
        output_dir.mkdir(parents=True)
        raw_log = run_dir / "raw" / f"{unit.key}.log"
        exporting = False
        try:
            inputs = {}
            for rel in unit.inputs:
                source = repository_path(root, rel, "input")
                archive = run_dir / "inputs" / unit.key / rel
                _stage_file(source, archive)
                destination = work_root / rel
                _stage_file(archive, destination)
                inputs[rel] = _hash(destination)
            for record in members:
                record["input_hashes"] = dict(inputs)
            destinations = {rel: _fingerprint(_artifact_path(root, rel)) for rel in unit.artifacts}
            summary["coverage"]["executed_units"] += 1
            outcome = execute_unit(unit, work_root, raw_log, timeout=timeout, environment=environment)
            if outcome.timed_out or outcome.returncode:
                for record in members:
                    record["reason_code"] = "timeout" if outcome.timed_out else "execution_failed"
                continue
            artifacts = {rel: inspect_artifact(_artifact_path(work_root, rel)) for rel in unit.artifacts}
            publication = publish_unit(unit, root, work_root, run_dir, inputs, destinations, artifacts)
            for target, record in zip(unit.targets, members, strict=True):
                record.update(publication_status=publication,
                              artifacts=[{"path": rel, **artifacts[rel]} for rel in target["expected_artifacts"]])
            exporting = True
            for rel in unit.artifacts:
                staged_export = run_dir / "export-staging" / rel
                _stage_file(_artifact_path(work_root, rel), staged_export)
                if _hash(staged_export) != artifacts[rel]["sha256"]:
                    raise ArtifactError("export no longer matches the validated artifact")
            for rel in unit.artifacts:
                exported = run_dir / "shareable/artifacts" / rel
                exported.parent.mkdir(parents=True, exist_ok=True)
                os.replace(run_dir / "export-staging" / rel, exported)
            for record in members:
                record.update(status="succeeded", reason_code="completed", export_status="succeeded")
        except ArtifactError:
            for record in members:
                record["reason_code"] = "artifact_export_failed" if exporting else "invalid_artifact"
                if exporting:
                    record["export_status"] = "failed"
            with raw_log.open("a", encoding="utf-8") as log:
                log.write(traceback.format_exc())
        except PublicationError as exc:
            for record in members:
                if exporting:
                    record.update(reason_code="artifact_export_failed", export_status="failed")
                else:
                    record.update(reason_code=exc.reason_code, publication_status=exc.status)
            with raw_log.open("a", encoding="utf-8") as log:
                log.write(traceback.format_exc())
        except Exception:
            for record in members:
                record["reason_code"] = "artifact_export_failed" if exporting else "runner_error"
                if exporting:
                    record["export_status"] = "failed"
            with raw_log.open("a", encoding="utf-8") as log:
                log.write(traceback.format_exc())
        finally:
            for target, record in zip(unit.targets, members, strict=True):
                record.update(finished_utc=_utc(), duration_seconds=round(time.monotonic() - unit_started, 3))
                for rel in target["expected_artifacts"]:
                    backup = run_dir / "backups" / unit.key / rel
                    if backup.is_file():
                        record["backups"].append({"path": backup.relative_to(run_dir).as_posix(),
                                                  "sha256": _hash(backup)})
                if record["backups"]:
                    record["backup_status"] = "retained"
            state = "succeeded" if all(r["status"] == "succeeded" for r in members) else "failed"
            print(f"Finished {unit.source}: {state}; logs/reproduce/{run_id}/raw/{unit.key}.log", flush=True)
    return _finish(run_dir, summary, started)
