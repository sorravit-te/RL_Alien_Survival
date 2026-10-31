"""Choosing best.zip by the hunt as well as the bots, without touching what training sees."""

import random
from fractions import Fraction
from time import perf_counter

import numpy as np
import pytest
import torch
from stable_baselines3.common.logger import configure

import rl.evaluate
import rl.hunt
from rl.evaluate import evaluate_suite, selection_hunt
from rl.evaluation_protocol import (
    CATCH_RULE,
    HUNT_RULE,
    HUNT_SELECTION,
    SELECTION,
    trained_score,
    validate_protocol,
)
from rl.hunt import CASES, FIRST_SCENARIO_SEED, SELECTION_SCENARIO_SEED, stations
from rl.pipeline import (
    ManifestEnv,
    new_model,
    parameter_hash,
    prepare_run,
    read_json,
    read_run,
    save_checkpoint,
    write_json,
)
from rl.train import PilotCallback, complete_sampled_pilot

SHIP_CONFIGS = ("configs/ship_deck_training_pilot.json", "configs/ship_deck_training_full.json")


def suite(bots, *hunt):
    summary = {"trained_sampled": {"per_case": [{"captured": bots, "unresolved": 0, "repeats": 90}]}}
    if hunt:
        names = ("noisy", "silent", "hidden")
        summary["hunt"] = {
            name: {"caught": caught, "cases": cases}
            for name, (caught, cases) in zip(names, hunt, strict=True)
        }
    return summary


def test_the_score_is_the_mean_of_the_bots_and_each_hunt():
    summary = suite(90, (32, 32), (16, 32), (0, 37))
    assert trained_score(summary, 10, 1)[0] == Fraction(1 + 1 + Fraction(1, 2) + 0, 4)


def test_a_wide_hunter_beats_a_slightly_better_bot_catcher():
    camper = suite(90, (32, 32), (10, 32), (12, 37))
    hunter = suite(86, (30, 32), (29, 32), (31, 37))
    assert trained_score(hunter, 60, 2) > trained_score(camper, 20, 1)


def test_a_summary_without_a_hunt_is_scored_by_the_bots_alone():
    assert trained_score(suite(45), 10, 1)[0] == Fraction(1, 2)


def test_an_equal_score_keeps_the_earlier_checkpoint():
    summary = suite(45, (16, 32), (16, 32), (16, 32))
    assert trained_score(summary, 10, 1) > trained_score(summary, 20, 2)


@pytest.mark.parametrize("path", SHIP_CONFIGS)
def test_the_ship_is_chosen_by_the_hunt(path):
    config = read_json(path)
    protocol = read_json(config["evaluation_protocol"])
    validate_protocol(config, protocol)
    assert protocol["selection"] == HUNT_SELECTION and config["selection"] == HUNT_RULE


def test_a_rule_that_disagrees_with_the_protocol_is_refused():
    config = read_json(SHIP_CONFIGS[0])
    protocol = read_json(config["evaluation_protocol"])
    with pytest.raises(ValueError, match="selection"):
        validate_protocol({**config, "selection": CATCH_RULE}, protocol)
    with pytest.raises(ValueError, match="selection"):
        validate_protocol(config, {**protocol, "selection": SELECTION})


def test_selection_never_sees_the_seeds_the_hunt_test_reports():
    reported = FIRST_SCENARIO_SEED + max(CASES, len(stations("ship_deck_v1")))
    assert SELECTION_SCENARIO_SEED >= reported


def test_the_full_run_only_changes_how_long_it_trains():
    pilot, full = (read_json(path) for path in SHIP_CONFIGS)
    assert full["requested_steps"] > pilot["requested_steps"]
    same = ("name", "requested_steps")
    assert {k: v for k, v in pilot.items() if k not in same} == {
        k: v for k, v in full.items() if k not in same
    }


@pytest.fixture
def short_hunt(monkeypatch):
    def short(choose, scenario, *, first_seed):
        return real(choose, scenario, cases=2, noisy_seconds=2, first_seed=first_seed)

    real = rl.evaluate.hunt_test
    monkeypatch.setattr(rl.hunt, "SILENT_SECONDS", 2)
    monkeypatch.setattr(rl.evaluate, "hunt_test", short)


@pytest.fixture
def hunting_run(tmp_path, short_hunt):
    config = read_json("configs/scenario_randomization_training_pilot.json")
    protocol = read_json(config["evaluation_protocol"])
    counts = {"train": 2, "validation": 2, "test": 1}
    config.update(
        manifest_seed=9100327,
        manifest_counts=counts,
        requested_steps=32,
        evaluation_interval=16,
        progress_interval=16,
        max_episode_seconds=1,
        selection=HUNT_RULE,
    )
    protocol.update(manifest_seed=9100327, manifest_counts=counts, selection=HUNT_SELECTION)
    spec = tmp_path / "hunting_protocol.json"
    write_json(spec, protocol)
    config["evaluation_protocol"] = str(spec)
    prepare_run(tmp_path / "run", config)
    config, manifest, source = read_run(tmp_path / "run")
    env = ManifestEnv(config, manifest["splits"]["train"])
    try:
        yield tmp_path / "run", config, manifest, source, new_model(config, env)
    finally:
        env.close()


def streams():
    return random.getstate(), np.random.get_state()[1].tolist(), torch.get_rng_state().tolist()


def test_the_hunt_is_the_same_each_time_and_leaves_training_streams_alone(hunting_run):
    _, config, _, _, model = hunting_run
    before, weights = streams(), parameter_hash(model)
    first, again = selection_hunt(model, config), selection_hunt(model, config)
    assert first == again and set(first) == {"noisy", "silent", "hidden"}
    assert streams() == before and parameter_hash(model) == weights


def test_the_hunt_runs_on_trained_stages_only(hunting_run):
    run, config, manifest, _, model = hunting_run
    schedule = read_json(run / "sampling_schedule.json")["validation"]
    cases = manifest["splits"]["validation"]
    models = {"trained": model, "untrained": model}
    trained, _ = evaluate_suite(
        config, cases, run / "t", schedule=schedule, models=models, stage="trained"
    )
    baseline, _ = evaluate_suite(
        config, cases, run / "b", schedule=schedule, models=models, stage="baseline"
    )
    assert "hunt" in trained and "hunt" not in baseline
    assert read_json(run / "t/summary.json")["hunt"] == trained["hunt"]


def test_a_selected_hunt_survives_the_fresh_replay(hunting_run):
    run, config, manifest, source, model = hunting_run
    save_checkpoint(run / "checkpoints/untrained.zip", model, config, source, role="untrained")
    untrained = read_json(run / "checkpoints/untrained.metadata.json")
    model.set_logger(configure(str(run / "tensorboard"), ["csv"]))
    callback = PilotCallback(run, config, manifest["splits"]["validation"], source)
    callback.init_callback(model)
    try:
        model.num_timesteps, model._n_updates = 16, 10
        callback.validate_checkpoint(run / "checkpoints/last.zip", "last")
        result = complete_sampled_pilot(
            run, config, manifest, source, model, untrained, callback, {}, perf_counter(), 0
        )
    finally:
        callback.close()
    assert result["fresh_load_reproduced"]
    assert read_json(run / "selection.json")["best_validation"]["hunt"] == callback.best_summary["hunt"]
    header = (run / "validation_curve.csv").read_text().splitlines()[0]
    assert "hunt_hidden_caught" in header
