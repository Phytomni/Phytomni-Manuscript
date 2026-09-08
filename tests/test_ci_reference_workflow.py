from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def workflow():
    return yaml.safe_load((ROOT / ".github/workflows/reproduce.yml").read_text())


def test_fast_public_gate_precedes_figure_jobs_without_r_setup():
    jobs = workflow()["jobs"]
    fast = jobs["checks"]
    blocks = "\n".join(step.get("run", "") for step in fast["steps"])
    assert blocks.index("uv lock --check") < blocks.index("uv sync --frozen")
    assert blocks.index("uv sync --frozen") < blocks.index("./reproduce.sh validate")
    assert "pytest tests/" in blocks
    assert "-m 'not r_runtime'" in blocks
    assert "Rscript" not in blocks
    assert all("setup-r" not in step.get("uses", "") for step in fast["steps"])
    assert jobs["figures"]["needs"] == "checks"


def test_formal_figure_job_uses_fixed_r_and_fatal_locked_restore():
    job = workflow()["jobs"]["figures"]
    assert job["runs-on"] == "ubuntu-24.04"
    r_setup = next(step for step in job["steps"] if "setup-r" in step.get("uses", ""))
    assert r_setup["with"]["r-version"] == "4.3.3"
    restore = next(step for step in job["steps"] if "renv::restore" in step.get("run", ""))
    assert "strict=TRUE" in restore["run"]
    assert "continue-on-error" not in restore
    assert "packageDescription" in restore["run"]
    assert "RemoteSha" in restore["run"]
    assert job["env"]["R_PROFILE_USER"] == "/dev/null"
    assert "R_LIBS_USER" in job["env"]
    blocks = "\n".join(step.get("run", "") for step in job["steps"])
    assert "./reproduce.sh run --timeout 600" in blocks
    assert "-m r_runtime" in blocks
    assert "--check" not in blocks.replace("uv lock --check", "")
    assert "|| true" not in blocks
    assert "remotes::" not in blocks
    assert not any("fallback" in step.get("name", "").lower() for step in job["steps"])


def test_ci_uploads_only_curated_current_run_evidence():
    uploads = [step for job in workflow()["jobs"].values() for step in job["steps"]
               if "upload-artifact" in step.get("uses", "")]
    assert len(uploads) == 2
    paths = {step["with"]["path"].strip() for step in uploads}
    assert paths == {
        "logs/reproduce/*/shareable/artifacts/**",
        "logs/reproduce/*/shareable/summary.json\nlogs/reproduce/*/shareable/targets/*.log",
    }
    assert all(step["if"] == "always()" for step in uploads)
    for path in paths:
        assert "/work/" not in path and "/backups/" not in path
        assert "**/output/**" not in path and "logs/**" not in path


def test_ci_uses_one_formal_environment_not_a_conda_fallback():
    jobs = workflow()["jobs"]
    assert set(jobs) == {"checks", "figures"}
    for job in jobs.values():
        setup = next(step for step in job["steps"] if "setup-uv" in step.get("uses", ""))
        assert setup["with"]["python-version"] == "3.12.13"
