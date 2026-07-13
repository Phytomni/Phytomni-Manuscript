import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "reproduce.yml"


def test_r_jobs_install_locked_ggradar_without_github_credentials() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "renv.lock").read_text(encoding="utf-8"))
    ggradar_sha = lock["Packages"]["ggradar"]["RemoteSha"]

    for job_name in ("conda", "uv"):
        run_blocks = [
            step["run"]
            for step in workflow["jobs"][job_name]["steps"]
            if "ggradar" in step.get("name", "").lower()
        ]
        assert len(run_blocks) == 1, job_name
        run_block = run_blocks[0]
        ggradar_guard = 'if (!requireNamespace("ggradar", quietly=TRUE))'

        assert ggradar_guard in run_block, job_name
        assert ggradar_sha in run_block, job_name
        assert "https://github.com/ricardo-bion/ggradar/archive/" in run_block
        assert "repos=NULL" in run_block
        assert 'type="source"' in run_block
        assert "remotes::" not in run_block
        assert "GITHUB_PAT" not in run_block
        assert "GITHUB_TOKEN" not in run_block


def test_conda_job_falls_back_when_optional_renv_restore_fails() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    conda_steps = workflow["jobs"]["conda"]["steps"]
    restore_steps = [
        step
        for step in conda_steps
        if "renv::restore(prompt=FALSE)" in step.get("run", "")
    ]

    assert len(restore_steps) == 1
    restore_step = restore_steps[0]
    assert restore_step["id"] == "conda_renv_restore"
    assert restore_step["continue-on-error"] is True

    fallback_steps = [
        step
        for step in conda_steps
        if "steps.conda_renv_restore.outcome == 'failure'"
        in step.get("if", "")
    ]
    assert len(fallback_steps) == 1
    assert (
        'RENV_CONFIG_AUTOLOADER_ENABLED=FALSE' in fallback_steps[0]["run"]
    )
    assert "R_PROFILE_USER=/dev/null" in fallback_steps[0]["run"]
    assert "$GITHUB_ENV" in fallback_steps[0]["run"]

    conda_environment = yaml.safe_load(
        (ROOT / "environment.yml").read_text(encoding="utf-8")
    )
    dependencies = set(
        dependency
        for dependency in conda_environment["dependencies"]
        if isinstance(dependency, str)
    )
    assert {
        "r-remotes",
        "r-irkernel",
        "r-tidyverse",
        "r-treemapify",
    } <= dependencies


def test_uv_job_installs_each_missing_r_dependency_independently() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    install_step = next(
        step
        for step in workflow["jobs"]["uv"]["steps"]
        if step.get("name") == "Install R packages + kernel + ggradar"
    )
    run_block = install_step["run"]

    assert 'if (!requireNamespace("tidyverse"' not in run_block
    assert "missing <- packages[!vapply(" in run_block
    assert "if (length(missing)) install.packages(missing," in run_block
    for package in (
        "tidyverse",
        "scales",
        "treemapify",
        "viridis",
        "readxl",
        "RColorBrewer",
        "rmarkdown",
        "IRkernel",
    ):
        assert f'"{package}"' in run_block


def test_uv_job_activates_python_env_before_registering_r_kernel() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    install_step = next(
        step
        for step in workflow["jobs"]["uv"]["steps"]
        if step.get("name") == "Install R packages + kernel + ggradar"
    )
    run_block = install_step["run"]

    activation = "source .venv/bin/activate"
    registration = "IRkernel::installspec(user = TRUE)"
    assert activation in run_block
    assert run_block.index(activation) < run_block.index(registration)


def test_r_jobs_activate_the_project_library_from_figure_directories() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    for job_name in ("conda", "uv"):
        environment = workflow["jobs"][job_name]["env"]
        assert environment["RENV_PROJECT"] == "${{ github.workspace }}"
        assert (
            environment["R_PROFILE_USER"]
            == "${{ github.workspace }}/.Rprofile"
        )

    r_profile = (ROOT / ".Rprofile").read_text(encoding="utf-8")
    assert 'Sys.getenv("RENV_PROJECT"' in r_profile
    assert 'file.path(project, "renv", "activate.R")' in r_profile
    assert 'source("renv/activate.R")' not in r_profile


def test_conda_fallback_starts_r_without_profile_recursion() -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is not installed in the Python-only test environment")

    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    conda_job = workflow["jobs"]["conda"]
    fallback_step = next(
        step
        for step in conda_job["steps"]
        if step.get("name") == "Fall back to conda R library"
    )
    environment = os.environ.copy()
    environment.update(
        {
            name: value.replace("${{ github.workspace }}", str(ROOT))
            for name, value in conda_job["env"].items()
        }
    )
    environment.update(
        dict(
            re.findall(
                r'echo "([A-Z_]+)=([^"\n]+)" >> "\$GITHUB_ENV"',
                fallback_step["run"],
            )
        )
    )

    result = subprocess.run(
        [rscript, "-e", 'cat("startup-ok\\n")'],
        cwd=ROOT / "Extended Data Fig. 5",
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "startup-ok\n"
