"""Diagnostic isolation is checked in a fresh original-runtime process."""

import json
import subprocess
import sys
from pathlib import Path


def test_seeded_original_policy_replay_is_read_only_and_logging_preserves_actions():
    repo = Path(__file__).resolve().parents[1]
    if not (repo / "experiments/naming_review/original_revision").is_dir():
        import pytest

        pytest.skip("Local original-policy diagnostic fixture is unavailable")
    result = subprocess.run(
        [sys.executable, "-B", "-m", "tools.policy_diagnostics", "--phase", "self-check"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    checks = json.loads(result.stdout)
    assert checks["seeded_replay"] and checks["tracing_preserves_actions_and_outcome"]
    assert checks["weights_unchanged"] and checks["global_rng_restored"] and checks["masks_valid"]
    assert checks["decisions"] > 0
