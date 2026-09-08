from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Literal

import yaml

REQUIRED_TARGET_KEYS = {"id", "label", "phase", "kind", "status"}
TARGET_KEYS = REQUIRED_TARGET_KEYS | {
    "path", "workdir", "kernel", "requires_data", "requires_toolchain",
    "expected_artifacts", "probes", "pending_reason", "pending_inputs",
}
FIGURE_KINDS = {"notebook": ".ipynb", "py_script": ".py", "rscript": ".R"}
TOOLCHAIN_NAMES = {"ir", "Rscript", "ggradar"}
PRIVATE_ROOTS = {".git", ".codex", ".cursor", ".agents", ".venv", "local_exports", "logs"}


class ManifestError(ValueError):
    pass


class TargetSelectionError(ValueError):
    pass


@dataclass(frozen=True)
class PrerequisiteResult:
    state: Literal["ready", "pending", "failed"]
    reasons: tuple[str, ...] = ()


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 for c in value):
        raise ManifestError(f"{field} must be a nonempty single-line string")
    return value


def _string_list(target: dict[str, Any], field: str) -> list[str]:
    values = target.get(field, [])
    if not isinstance(values, list):
        raise ManifestError(f"{field} must be a list")
    for value in values:
        _text(value, field)
    if len(values) != len(set(values)):
        raise ManifestError(f"{field} contains duplicate entries")
    return values


def repository_path(repo_root: Path, value: Any, field: str) -> Path:
    """Resolve a canonical repository-relative path without allowing escapes."""
    value = _text(value, field)
    path = Path(value)
    if path.is_absolute() or "\\" in value or ".." in path.parts or path.as_posix() != value or value == ".":
        raise ManifestError(f"{field} must be a canonical relative path: {value}")
    try:
        resolved = (repo_root / path).resolve()
    except (OSError, RuntimeError):
        raise ManifestError(f"{field} cannot be resolved: {value}") from None
    if not resolved.is_relative_to(repo_root.resolve()):
        raise ManifestError(f"{field} escapes the repository: {value}")
    return resolved


def _public_path(repo_root: Path, value: Any, field: str) -> Path:
    resolved = repository_path(repo_root, value, field)
    relative = resolved.relative_to(repo_root.resolve())
    if not relative.parts or Path(value).parts[0] in PRIVATE_ROOTS or relative.parts[0] in PRIVATE_ROOTS:
        raise ManifestError(f"{field} must refer to public figure files: {value}")
    return resolved


def _validate_probes(probes: Any, repo_root: Path) -> None:
    if not isinstance(probes, list):
        raise ManifestError("probes must be a list")
    fields = {
        "file_exists": {"path"}, "can_import": {"module"},
        "file_contains_none": {"path", "substring"}, "note_requires": {"message"},
    }
    for probe in probes:
        if not isinstance(probe, dict) or not isinstance(probe.get("type"), str) or probe["type"] not in fields:
            raise ManifestError("unsupported probe type")
        required = fields[probe["type"]] | {"type"}
        if set(probe) != required:
            raise ManifestError(f"invalid fields for {probe['type']} probe")
        for key in required:
            _text(probe[key], f"probe {key}")
        if "path" in probe:
            repository_path(repo_root, probe["path"], "probe path")
        if "module" in probe and not re.fullmatch(r"[A-Za-z_]\w*", probe["module"]):
            raise ManifestError("probe module must name a top-level module")


def validate_manifest(manifest: dict[str, Any], repo_root: Path) -> None:
    """Validate declarations without importing code or checking installed tools.

    Source existence/schema and selected data availability are checked separately,
    so a missing private evaluation prerequisite is not a malformed registry.
    """
    if not isinstance(manifest, dict) or set(manifest) != {"version", "targets"}:
        raise ManifestError("manifest must contain only version and targets")
    if type(manifest["version"]) is not int or manifest["version"] != 1:
        raise ManifestError("unsupported manifest version")
    if not isinstance(manifest["targets"], list) or not manifest["targets"]:
        raise ManifestError("targets must be a nonempty list")
    ids: set[str] = set()
    sources: dict[Path, tuple[str, str, str, str]] = {}
    inputs: set[Path] = set()
    artifacts: dict[Path, str] = {}
    for target in manifest["targets"]:
        if not isinstance(target, dict):
            raise ManifestError("each target must be a mapping")
        if REQUIRED_TARGET_KEYS - set(target):
            raise ManifestError(f"target missing keys: {sorted(REQUIRED_TARGET_KEYS - set(target))}")
        if set(target) - TARGET_KEYS:
            raise ManifestError(f"unexpected target fields: {sorted(set(target) - TARGET_KEYS)}")
        tid = _text(target["id"], "id")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", tid):
            raise ManifestError(f"invalid target id: {tid}")
        if tid in ids:
            raise ManifestError(f"duplicate target id: {tid}")
        ids.add(tid)
        _text(target["label"], "label")
        phase = _text(target["phase"], "phase")
        kind = _text(target["kind"], "kind")
        status = _text(target["status"], "status")
        kernel = _text(target.get("kernel", "none"), "kernel")
        if phase not in {"figure", "eval"}:
            raise ManifestError(f"unsupported phase: {phase}")
        if status not in {"run", "skip_until_data"}:
            raise ManifestError(f"unsupported status: {status}")
        if kernel not in {"py", "ir", "none"}:
            raise ManifestError(f"unsupported kernel: {kernel}")
        data = _string_list(target, "requires_data")
        tools = _string_list(target, "requires_toolchain")
        outputs = _string_list(target, "expected_artifacts")
        pending = _string_list(target, "pending_inputs")
        if set(tools) - TOOLCHAIN_NAMES:
            raise ManifestError(f"unsupported requires_toolchain: {tools}")
        _validate_probes(target.get("probes", []), repo_root)
        if phase == "eval":
            if kind != "eval_probe" or status != "run" or kernel != "none":
                raise ManifestError(f"invalid evaluation kind/status/kernel: {tid}")
            if data or tools or outputs or pending or target.get("pending_reason") or target.get("path") or target.get("workdir"):
                raise ManifestError(f"evaluation targets declare prerequisites only through probes: {tid}")
            continue
        if kind not in FIGURE_KINDS:
            raise ManifestError(f"unsupported figure kind: {kind}")
        source = _public_path(repo_root, target.get("path"), "path")
        relative_source = Path(target["path"])
        if relative_source.suffix != FIGURE_KINDS[kind]:
            raise ManifestError(f"source suffix does not match kind: {tid}")
        parent = relative_source.parent
        if "output" in relative_source.parts:
            raise ManifestError(f"source cannot be inside output: {tid}")
        if kind == "notebook" and kernel not in {"py", "ir"} or kind != "notebook" and kernel != "none":
            raise ManifestError(f"kernel does not match kind: {tid}")
        required_tools = {"ir"} if kernel == "ir" else {"Rscript"} if kind == "rscript" else set()
        if not required_tools.issubset(tools):
            raise ManifestError(f"requires_toolchain omits {sorted(required_tools)}: {tid}")
        workdir = target.get("workdir")
        if kind in {"rscript", "py_script"} and workdir is None:
            raise ManifestError(f"workdir is required for scripts: {tid}")
        if workdir is not None:
            _public_path(repo_root, workdir, "workdir")
            if Path(workdir) != parent:
                raise ManifestError(f"workdir must be the source directory: {tid}")
        descriptor = (target["path"], kind, kernel, workdir or str(parent))
        if source in sources and sources[source] != descriptor:
            raise ManifestError(f"shared source has inconsistent execution declarations: {tid}")
        sources[source] = descriptor
        inputs.add(source)
        for rel in data:
            path = _public_path(repo_root, rel, "requires_data")
            if "output" in Path(rel).parts:
                raise ManifestError(f"requires_data cannot read output caches: {rel}")
            if not Path(rel).is_relative_to(parent) or not path.is_relative_to(source.parent):
                raise ManifestError(f"requires_data must remain in the same figure directory: {rel}")
            inputs.add(path)
        if not outputs:
            raise ManifestError(f"expected_artifacts must not be empty: {tid}")
        for rel in outputs:
            path = _public_path(repo_root, rel, "expected_artifacts")
            if not Path(rel).is_relative_to(parent / "output") or not path.is_relative_to(source.parent / "output"):
                raise ManifestError(f"expected_artifacts must be in the source output directory: {rel}")
            if path in artifacts:
                raise ManifestError(f"artifact already owned by {artifacts[path]}: {rel}")
            artifacts[path] = tid
        if target.get("probes"):
            raise ManifestError(f"figure targets cannot declare evaluation probes: {tid}")
        if status == "skip_until_data":
            _text(target.get("pending_reason"), "pending_reason")
            if not pending or not set(pending).issubset(data):
                raise ManifestError("pending_inputs must be a nonempty subset of requires_data")
        elif pending or target.get("pending_reason"):
            raise ManifestError(f"run targets cannot retain pending declarations: {tid}")
    if inputs & artifacts.keys():
        raise ManifestError("artifact paths collide with source/input paths")


def select_targets(manifest: dict[str, Any], target_ids: Collection[str] | None = None,
                   *, phase: str | None = None) -> list[dict[str, Any]]:
    targets = manifest["targets"]
    by_id = {t["id"]: t for t in targets}
    requested = set(target_ids or ())
    unknown = requested - by_id.keys()
    if unknown:
        raise TargetSelectionError(f"unknown target IDs: {', '.join(sorted(unknown))}")
    if phase is not None and any(by_id[tid]["phase"] != phase for tid in requested):
        raise TargetSelectionError(f"selected targets must belong to the {phase} phase")
    return [t for t in targets if (not requested or t["id"] in requested)
            and (phase is None or t["phase"] == phase)]


def figure_prerequisites(target: dict[str, Any], repo_root: Path, *,
                         available_tools: Collection[str] | None = None,
                         explicit: bool = False) -> PrerequisiteResult:
    """Classify prerequisites; omitted tools mean that runtime is not being checked."""
    source = repository_path(repo_root, target["path"], "path")
    if not source.is_file():
        return PrerequisiteResult("failed", (f"source missing: {target['path']}",))
    missing = [rel for rel in target.get("requires_data", [])
               if not repository_path(repo_root, rel, "requires_data").is_file()]
    if target["status"] == "skip_until_data":
        undeclared = set(missing) - set(target["pending_inputs"])
        if undeclared:
            return PrerequisiteResult("failed", tuple(f"required data missing: {rel}" for rel in sorted(undeclared)))
        return PrerequisiteResult("failed" if explicit else "pending",
                                  (target["pending_reason"], "author promotion to run is required"))
    reasons = [f"required data missing: {rel}" for rel in missing]
    if available_tools is not None:
        reasons.extend(f"required toolchain missing: {name}" for name in target.get("requires_toolchain", [])
                       if name not in available_tools)
    return PrerequisiteResult("failed" if reasons else "ready", tuple(reasons))


def validate_figure_sources(manifest: dict[str, Any], repo_root: Path) -> list[str]:
    """Check source/schema/kernel coverage, not presentation or installed kernels."""
    import nbformat

    validate_manifest(manifest, repo_root)
    problems: list[str] = []
    seen: set[str] = set()
    declared: set[str] = set()
    for target in iter_targets(manifest, phase="figure"):
        rel = target["path"]
        declared.add(rel)
        declared.update(target.get("requires_data", []))
        if rel in seen:
            continue
        seen.add(rel)
        path = repository_path(repo_root, rel, "path")
        if not path.is_file():
            problems.append(f"source missing: {rel}")
            continue
        if target["kind"] != "notebook":
            continue
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(notebook, dict):
                raise ValueError("notebook must be a mapping")
            error = next(nbformat.validator.iter_validate(notebook), None)
            if error is not None:
                raise ValueError("invalid notebook schema")
            ids = [cell["id"] for cell in notebook.get("cells", []) if "id" in cell]
            if len(ids) != len(set(ids)):
                raise ValueError("invalid notebook schema")
        except (OSError, UnicodeError, ValueError, nbformat.ValidationError):
            problems.append(f"invalid notebook schema: {rel}")
            continue
        expected = "python3" if target["kernel"] == "py" else "ir"
        if notebook.get("metadata", {}).get("kernelspec", {}).get("name") != expected:
            problems.append(f"notebook kernel must be {expected}: {rel}")
    for directory in sorted(repo_root.iterdir()):
        if not directory.is_dir() or not re.fullmatch(r"(?:Fig|Extended Data Fig|Supplementary Fig)\. \d+(?:-\d+)?", directory.name):
            continue
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(repo_root)
            if any(part.startswith(".") or part in {"output", "__pycache__"} for part in relative.parts[1:]):
                continue
            if path.suffix not in FIGURE_KINDS.values() or not path.is_file():
                continue
            if relative.as_posix() not in declared:
                problems.append(f"unregistered figure source: {relative.as_posix()}")
    return problems


def validate_static(manifest: dict[str, Any], repo_root: Path, *,
                    target_ids: Collection[str] | None = None) -> list[str]:
    """Validate global sources and selected public inputs without runtime probes."""
    problems = validate_figure_sources(manifest, repo_root)
    for target in select_targets(manifest, target_ids, phase="figure"):
        result = figure_prerequisites(target, repo_root)
        if result.state == "failed":
            problems.extend(result.reasons)
    return list(dict.fromkeys(problems))


def load_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ManifestError("manifest is not valid YAML") from exc
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


def run_probe(probe: dict[str, Any], repo_root: Path) -> str | None:
    probe_type = probe.get("type")
    if probe_type == "file_exists":
        rel = probe["path"]
        if not (repo_root / rel).is_file():
            return f"file missing: {rel}"
        return None
    if probe_type == "can_import":
        module = probe["module"]
        if not isinstance(module, str) or not re.fullmatch(r"[A-Za-z_]\w*", module):
            return "module discovery requires a top-level module name"
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            return f"module not discoverable: {module}"
        return None
    if probe_type == "file_contains_none":
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
