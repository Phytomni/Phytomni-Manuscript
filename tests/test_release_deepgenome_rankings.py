from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def assert_ignored(relative_path: str) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", relative_path],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0, relative_path


def test_private_expert_inputs_are_ignored() -> None:
    assert_ignored("DeepGenomeAgent Evaluation/evaluation_expert_metadata-0714.xlsx")
    assert_ignored("DeepGenomeAgent Evaluation/private/expert_release_crosswalk.tsv")
