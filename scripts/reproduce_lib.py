from __future__ import annotations

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
