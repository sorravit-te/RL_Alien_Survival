"""Saved-trace plots must not execute or load any game/policy runtime."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from game.interaction import segment_clear
from game.map import WorldMap
from tools.policy_diagnostic_report import (
    connector_segments,
    draw_samples,
    file_hashes,
    plot_saved_traces,
)


def geometry():
    world = WorldMap(("#####", "#.#.#", "#...#", "#####"), ((1.5, 1.5),))
    return SimpleNamespace(world=world, segment_clear=segment_clear)


def test_wall_crossing_connector_omitted_and_open_connector_dashed():
    points = [(1.5, 1.5), (3.5, 1.5), (3.5, 2.5)]
    allowed = connector_segments(points, geometry())
    assert allowed == [(points[1], points[2])]
    fig, ax = plt.subplots()
    try:
        assert draw_samples(ax, points, geometry(), "red", "actor") == 1
        assert len(ax.lines) == 1 and ax.lines[0].get_linestyle() == "--"
        assert list(ax.lines[0].get_xdata()) == [3.5, 3.5]
        assert list(ax.lines[0].get_ydata()) == [1.5, 2.5]
    finally:
        plt.close(fig)


def test_wall_corner_touch_omitted():
    assert not connector_segments([(1.5, 1.5), (2.5, 2.5)], geometry())


def test_plot_outputs_must_be_new_and_separate(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    with pytest.raises(ValueError, match="separate"):
        plot_saved_traces(source, source)
    with pytest.raises(ValueError, match="separate"):
        plot_saved_traces(source, source / "nested")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="New output"):
        plot_saved_traces(source, existing)
    assert list(source.iterdir()) == [] and list(existing.iterdir()) == []


def test_plot_only_reproduces_pictures_without_runtime_models_or_input_writes(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    source = repo / "experiments/policy_diagnostics/action_sampling_v1"
    if not source.is_dir():
        pytest.skip("Local saved-trace fixture unavailable")
    original = repo / "experiments/naming_review/original_revision"
    before = file_hashes(source)
    export_before = file_hashes(original)
    destination = tmp_path / "rendered"
    script = r"""import importlib.abc, pathlib, sys
pilot=pathlib.Path(sys.argv[1]).resolve()
class DenyRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('rl','game','ai','torch','gymnasium','sb3_contrib','stable_baselines3') or fullname=='tools.policy_diagnostics':
            raise AssertionError('Runtime import attempted: '+fullname)
sys.meta_path.insert(0,DenyRuntime())
def deny_pilot_files(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        name=args[0].decode() if isinstance(args[0],bytes) else args[0]
        if pathlib.Path(name).resolve().is_relative_to(pilot):
            raise AssertionError('Original pilot/test file opened: '+name)
sys.addaudithook(deny_pilot_files)
from tools.policy_diagnostic_report import main
sys.argv=['plot-only','--input',sys.argv[2],'--output',sys.argv[3]]
main()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            script,
            str(repo / "experiments/2c/pilot_20261024"),
            str(source),
            str(destination),
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    counters = json.loads(result.stdout)
    assert counters["episodes_run"] == counters["checkpoint_loads"] == 0
    assert file_hashes(source) == before and file_hashes(original) == export_before
    for name in ("paths_validation_0003.png", "approach_validation_0043.png"):
        assert (destination / name).stat().st_size > 1000
    provenance = json.loads((destination / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["input_unchanged"] and provenance["original_static_source_files_unchanged"]
    assert provenance["input_sha256"] == before
    assert (
        provenance["plotter_sha256"]
        == hashlib.sha256((repo / "tools/policy_diagnostic_report.py").read_bytes()).hexdigest()
    )
    assert provenance["wait_capture"]["action"] == 30
    assert abs(provenance["wait_capture"]["separation"] - 0.4) < 1e-6
