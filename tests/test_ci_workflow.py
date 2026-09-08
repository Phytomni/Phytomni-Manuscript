import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_reference_restore_checks_lock_identity_in_the_explicit_library():
    workflow = yaml.safe_load((ROOT / ".github/workflows/reproduce.yml").read_text())
    job = workflow["jobs"]["figures"]
    step = next(s for s in job["steps"] if "renv::restore" in s.get("run", ""))
    source = step["run"]
    assert 'source("renv/activate.R")' in source
    assert 'library=library_path, lockfile="renv.lock", strict=TRUE' in source
    assert 'lib.loc=library_path' in source
    assert 'actual$Version, locked$Packages[[name]]$Version' in source
    assert 'locked$Packages$ggradar$RemoteSha' in source
    assert 'getRversion()) == locked$R$Version' in source
    lock = json.loads((ROOT / "renv.lock").read_text())
    assert lock["R"]["Version"] == "4.3.3"
    assert lock["Packages"]["ggradar"]["RemoteSha"] == "f99517ae4df2903d3dbd4116af91b24009a5ac93"


@pytest.mark.r_runtime
def test_explicit_reference_library_matches_every_locked_description_version():
    library = os.environ.get("R_LIBS_USER", "")
    assert Path(library).is_absolute() and Path(library).is_dir(), "Prepare and set R_LIBS_USER first"
    assert shutil.which("Rscript"), "Prepare R 4.3.3 before selecting r_runtime tests"
    lock = json.loads((ROOT / "renv.lock").read_text())
    result = subprocess.run(
        ["Rscript", "--vanilla", "-e", 'cat(as.character(getRversion()))'],
        env={**os.environ, "R_PROFILE_USER": "/dev/null"}, capture_output=True,
        text=True, timeout=15, check=True,
    )
    assert result.stdout == lock["R"]["Version"]
    for name, expected in lock["Packages"].items():
        description = (Path(library) / name / "DESCRIPTION").read_text()
        fields = dict(line.split(": ", 1) for line in description.splitlines()
                      if ": " in line and not line.startswith(" "))
        assert fields["Version"] == expected["Version"], name
        if name == "ggradar":
            assert fields["RemoteSha"] == expected["RemoteSha"]
