"""Real SB3 smoke tests, separate from the 20k pilot and never its evidence."""

import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from game.map import MapObject, WorldMap
from rl.env import AlienEnv
from rl.evaluate import HeuristicSelector, evaluate, frozen_evaluation, paired_catch_difference
from rl.pipeline import (
    CHECKPOINT_FORMAT,
    ManifestEnv,
    digest,
    load_checkpoint,
    make_manifest,
    new_model,
    parameter_hash,
    prepare_run,
    read_json,
    read_run,
    save_checkpoint,
    source_state,
    write_json,
)
from rl.train import baselines, pilot


@pytest.fixture
def config():
    result = read_json(Path(__file__).resolve().parents[1] / "configs/training_pilot.json")
    result.update(
        manifest_counts={"train": 4, "validation": 2, "test": 2},
        requested_steps=32,
        evaluation_interval=16,
        progress_interval=16,
        max_episode_seconds=1,
    )
    result["ppo"].update(n_steps=16, batch_size=16, n_epochs=1)
    return result


def test_manifest_deterministic_disjoint_both_seed_kinds_and_fixed_map_warning(config):
    first = make_manifest(config["manifest_seed"], config["manifest_counts"])
    assert first == make_manifest(config["manifest_seed"], config["manifest_counts"])
    values = [
        row[key]
        for cases in first["splits"].values()
        for row in cases
        for key in ("reset_seed", "scenario_seed", "bot_seed", "selector_seed")
    ]
    assert len(values) == len(set(values))
    assert len(first["map"]["spawn_points"]) == 2
    assert "NOT layout" in first["map"]["warning"]


def test_training_manifest_reset_cycle_and_cap(config):
    cases = make_manifest(config["manifest_seed"], config["manifest_counts"])["splits"]["train"]
    env = ManifestEnv(config, cases)
    try:
        for i in range(6):
            _, info = env.reset()
            assert info["scenario_seed"] == cases[i % len(cases)]["scenario_seed"]
            assert info["bot_seed"] == cases[i % len(cases)]["bot_seed"]
            assert env.unwrapped.elapsed_ticks == 0
            assert env.unwrapped.core.snapshot().perception.last_seen is None
            env.step(30)
            _, _, terminal, truncated, info = env.step(30)
            assert not terminal and truncated and info["elapsed_ticks"] == 30
            assert info["manifest_id"] == cases[i % len(cases)]["id"]
        env.reset(seed=3)
        assert env.current_case == cases[0]
        with pytest.raises(ValueError, match="manifest owns"):
            env.reset(options={"bot_seed": 1})
    finally:
        env.close()


def test_frozen_config_manifest_and_source_rejected(tmp_path, config, monkeypatch):
    run = tmp_path / "run"
    prepare_run(run, config)
    assert read_run(run)[0] == config
    manifest = read_json(run / "manifest.json")
    changed = deepcopy(manifest)
    changed["splits"]["test"][0]["bot_seed"] += 1
    write_json(run / "manifest.json", changed)
    with pytest.raises(ValueError, match="manifest changed"):
        read_run(run)
    write_json(run / "manifest.json", manifest)
    identity = read_json(run / "run_identity.json")
    monkeypatch.setattr("rl.pipeline.source_state", lambda: {"source_sha256": "different"})
    with pytest.raises(ValueError, match="Source changed"):
        read_run(run)
    assert identity["config_sha256"] == digest(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("contract_version", "bad"),
        # Synthetic retired identifiers must be rejected; archived files stay untouched.
        ("contract_version", "alien-vector-2b-v1"),
        ("format", "alien-maskable-checkpoint-2c-v1"),
        ("observation_shape", [351]),
        ("action_n", 31),
        ("config_sha256", "bad"),
        ("manifest_sha256", "bad"),
        ("dependencies", {}),
    ],
)
def test_checkpoint_metadata_mismatch_has_no_fallback(tmp_path, config, field, value):
    env = AlienEnv(max_episode_seconds=1)
    try:
        model = new_model(config, env)
        source = source_state()
        path = tmp_path / "model.zip"
        save_checkpoint(path, model, config, source, role="test")
        meta = read_json(path.with_suffix(".metadata.json"))
        meta[field] = value
        write_json(path.with_suffix(".metadata.json"), meta)
        with pytest.raises(ValueError, match="metadata mismatch"):
            load_checkpoint(path, config, source)
    finally:
        env.close()


def test_current_checkpoint_identifiers_save_and_load_without_training(tmp_path, config):
    from rl.contract import VERSION
    from rl.pipeline import map_identity

    assert VERSION == "alien-vector-v2"
    assert CHECKPOINT_FORMAT == "alien-maskable-checkpoint-v1"
    assert map_identity()["name"] == "small_test_map_v1"
    manifest = make_manifest(config["manifest_seed"], config["manifest_counts"])
    assert manifest["map"]["name"] == "small_test_map_v1"
    env = AlienEnv(max_episode_seconds=1)
    try:
        model = new_model(config, env)
        source = source_state()
        path = tmp_path / "model.zip"
        metadata = save_checkpoint(path, model, config, source, role="test")
        assert metadata["format"] == CHECKPOINT_FORMAT
        assert metadata["contract_version"] == VERSION
        assert metadata["manifest_sha256"] == digest(manifest)
        assert metadata["map"]["name"] == "small_test_map_v1"
        loaded, checked = load_checkpoint(path, config, source)
        assert digest(checked) == digest(metadata)
        assert loaded.num_timesteps == model.num_timesteps == 0
        assert parameter_hash(loaded) == parameter_hash(model)
    finally:
        env.close()


def test_checkpoint_source_and_archive_mismatch(tmp_path, config):
    env = AlienEnv(max_episode_seconds=1)
    try:
        model = new_model(config, env)
        source = source_state()
        path = tmp_path / "model.zip"
        save_checkpoint(path, model, config, source, role="test")
        with pytest.raises(ValueError, match="source"):
            load_checkpoint(path, config, {**source, "source_sha256": "bad"})
        path.write_bytes(path.read_bytes() + b"changed")
        with pytest.raises(ValueError, match="file hash"):
            load_checkpoint(path, config, source)
    finally:
        env.close()


def test_real_masked_train_save_fresh_load_and_eval_do_not_touch_training_env(
    tmp_path, config, monkeypatch
):
    cases = make_manifest(config["manifest_seed"], config["manifest_counts"])["splits"]["train"]
    env = ManifestEnv(config, cases)
    applied = []
    original_step = env.step

    def checked_step(action):
        assert env.action_masks()[int(action)]
        applied.append(int(action))
        return original_step(action)

    monkeypatch.setattr(env, "step", checked_step)
    try:
        model = new_model(config, env)
        before = parameter_hash(model)
        model.learn(total_timesteps=32, use_masking=True)
        assert len(applied) == 32 and model.num_timesteps == 32
        assert parameter_hash(model) != before
        path = tmp_path / "model.zip"
        source = source_state()
        save_checkpoint(path, model, config, source, role="test")
        loaded, meta = load_checkpoint(path, config, source)
        assert meta["actual_steps"] == 32 and parameter_hash(loaded) == parameter_hash(model)
        state = (
            env.cursor,
            env.current_case,
            env.unwrapped.core.snapshot(),
            env.unwrapped.elapsed_ticks,
            model._last_obs.copy(),
        )
        seen = []
        predict = model.predict

        def checked_predict(obs, **kwargs):
            assert kwargs["deterministic"] is True
            mask = kwargs["action_masks"]
            assert mask.shape == (32,) and mask.dtype == np.bool_
            seen.append(mask.copy())
            return predict(obs, **kwargs)

        monkeypatch.setattr(model, "predict", checked_predict)
        first, rows = evaluate("trained", config, cases[:2], tmp_path / "first.csv", model=model)
        second, loaded_rows = evaluate(
            "trained", config, cases[:2], tmp_path / "second.csv", model=loaded
        )
        assert seen and first == second and rows == loaded_rows
        assert (
            env.cursor,
            env.current_case,
            env.unwrapped.core.snapshot(),
            env.unwrapped.elapsed_ticks,
        ) == state[:4]
        assert np.array_equal(model._last_obs, state[4])
        assert len(applied) == 32 and parameter_hash(loaded) == parameter_hash(model)
    finally:
        env.close()


def test_frozen_evaluation_preserves_global_random_streams(config):
    env = AlienEnv(max_episode_seconds=1)
    try:
        model = new_model(config, env)
        model.policy.set_training_mode(True)
        py, npstate, tstate = random.getstate(), np.random.get_state(), torch.get_rng_state()
        with frozen_evaluation(model):
            random.random()
            np.random.random()
            torch.rand(3)
        assert random.getstate() == py
        assert np.array_equal(np.random.get_state()[1], npstate[1])
        assert torch.equal(torch.get_rng_state(), tstate)
        assert model.policy.training
    finally:
        env.close()


@pytest.mark.parametrize("terminal", [False, True])
def test_actual_maskable_ppo_bootstraps_only_time_limit(terminal):
    world = WorldMap(
        ("." * 16,) * 16,
        ((2.5, 2.5),),
        (MapObject("card", "keycard", 3.0, 2.5), MapObject("exit", "exit", 3.0, 2.5))
        if terminal
        else (),
        alien_spawn=(12.5, 12.5, 0),
    )
    env = DummyVecEnv(
        [
            lambda: AlienEnv(
                world,
                max_episode_seconds=2 / 30 if terminal else 1 / 30,
                bot_kind="explorer" if terminal else "stationary_debug",
            )
        ]
    )
    try:
        torch.set_num_threads(1)
        model = MaskablePPO(
            "MlpPolicy", env, n_steps=2, batch_size=2, n_epochs=1, gamma=0.995, device="cpu", seed=4
        )
        model.policy.predict_values = lambda obs: torch.full((obs.shape[0], 1), 5.0)
        model.learn(total_timesteps=2, use_masking=True)
        # True escape is -10; unresolved cap is reward0 plus gamma * terminal V.
        assert np.allclose(model.rollout_buffer.rewards, -10 if terminal else 0.995 * 5)
        info = env.buf_infos[0]
        assert info["TimeLimit.truncated"] is (not terminal)
        assert info["terminal_observation"].shape == (352,)
        assert info["outcome"] == ("escaped" if terminal else "unresolved")
        assert env.envs[0].elapsed_ticks == 0  # vec-env auto-reset after done
    finally:
        env.close()


def test_heuristic_sees_only_allowed_view_and_paired_statistics(config):
    env = AlienEnv(max_episode_seconds=1)
    try:
        env.reset(seed=1)
        selector = HeuristicSelector(env.config.target_arrival_distance)
        action = selector.choose(env._allowed_view())
        assert action < 20 and env.action_masks()[action]
        rows = [{"id": "a", "outcome": "captured"}, {"id": "b", "outcome": "escaped"}]
        comparison = paired_catch_difference(rows, rows)
        assert comparison["difference"] == 0 and comparison["paired_bootstrap95"] == [0.0, 0.0]
    finally:
        env.close()


def test_complete_small_pipeline_has_best_last_test_guard_and_tensorboard(tmp_path, config):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run = tmp_path / "smoke"
    prepare_run(run, config)
    baselines(run)
    result = pilot(run)
    assert result["actual_steps"] == 32 and result["best_steps"] == 16  # exact tie keeps earliest
    assert result["fresh_load_reproduced"]
    assert (run / "checkpoints/best.zip").exists() and (run / "checkpoints/last.zip").exists()
    assert read_json(run / "checkpoints/best.metadata.json")["actual_steps"] == 16
    assert read_json(run / "checkpoints/last.metadata.json")["actual_steps"] == 32
    assert (run / "learning_curves.png").stat().st_size > 1000
    assert len(result["test"]) == 5
    events = EventAccumulator(str(run / "tensorboard")).Reload()
    assert "validation/catch_rate" in events.Tags()["scalars"]
    assert "train/episode_return100" in events.Tags()["scalars"]
    with pytest.raises(FileExistsError):
        pilot(run)
    with pytest.raises(FileExistsError):
        baselines(run)
