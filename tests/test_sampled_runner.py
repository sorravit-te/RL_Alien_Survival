"""Sampled runner contract tests: tiny synthetic cases, real untrained PPO, zero learning."""

import random
from copy import deepcopy

import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import configure

from rl.evaluate import evaluate, evaluate_suite, summarize_cases
from rl.evaluation_protocol import frozen_protocol, trained_score, validate_protocol
from rl.pipeline import (
    ManifestEnv,
    digest,
    file_hash,
    load_checkpoint,
    make_manifest,
    new_model,
    parameter_hash,
    prepare_run,
    read_json,
    read_run,
    save_checkpoint,
    write_json,
)
from rl.train import PilotCallback, final_sampled_test


@pytest.fixture(autouse=True)
def zero_learning(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Learning/optimizer updates forbidden in integration checks")

    monkeypatch.setattr(MaskablePPO, "learn", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)


@pytest.fixture
def prepared(tmp_path):
    config = read_json("configs/evasive_training_pilot.json")
    protocol = read_json(config["evaluation_protocol"])
    # Distinct tiny synthetic split, not any reviewed population or future evaluation cases.
    config.update(
        manifest_seed=9100327,
        manifest_counts={"train": 2, "validation": 2, "test": 1},
        requested_steps=32,
        evaluation_interval=16,
        progress_interval=16,
        max_episode_seconds=1,
    )
    protocol.update(
        manifest_seed=config["manifest_seed"], manifest_counts=config["manifest_counts"]
    )
    spec = tmp_path / "synthetic_protocol.json"
    write_json(spec, protocol)
    config["evaluation_protocol"] = str(spec)
    run = tmp_path / "run"
    prepare_run(run, config)
    return run, *read_run(run)


@pytest.fixture
def policy(prepared):
    _, config, manifest, _ = prepared
    env = ManifestEnv(config, manifest["splits"]["train"])
    model = new_model(config, env)
    try:
        yield model, env
    finally:
        env.close()


def snapshot_rng():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def assert_rng_equal(before):
    now = snapshot_rng()
    assert now[0] == before[0]
    assert now[1][0] == before[1][0]
    assert np.array_equal(now[1][1], before[1][1])
    assert now[1][2:] == before[1][2:]
    assert torch.equal(now[2], before[2])


def test_freeze_actual_protocol_schedule_budget_and_ignore_later_path_changes(prepared):
    run, config, manifest, _ = prepared
    protocol = frozen_protocol(config)
    schedule = read_json(run / "sampling_schedule.json")
    for split in ("validation", "test"):
        for case, entry in zip(manifest["splits"][split], schedule[split], strict=True):
            assert entry["sampling_seeds"] == [
                int(np.random.SeedSequence([case["selector_seed"], seed]).generate_state(1)[0])
                for seed in protocol["sampling_repeat_seeds"]
            ]
    budget = read_json(run / "evaluation_budget.json")
    assert budget["baseline_validation_episodes"] == 18 and budget["final_test_episodes"] == 13
    assert (
        read_json(run / "run_identity.json")["evaluation_mode"]
        == "sampled_primary_argmax_secondary"
    )
    # The mutable external path is no longer the authority.
    write_json(config["evaluation_protocol"], {"changed": True})
    assert read_run(run)[0] == config
    assert config["frozen_evaluation"]["protocol_file_sha256"] == file_hash(
        run / "evaluation_protocol.json"
    )


@pytest.mark.parametrize(
    "name",
    [
        "evaluation_protocol.json",
        "sampling_schedule.json",
        "evaluation_budget.json",
        "manifest.json",
        "config.json",
    ],
)
def test_frozen_input_mutation_rejected(prepared, name):
    run, _, _, _ = prepared
    value = read_json(run / name)
    value["tampered"] = True
    write_json(run / name, value)
    with pytest.raises(ValueError, match="changed"):
        read_run(run)


@pytest.mark.parametrize(
    "bad",
    [
        "availability",
        "missing_controller",
        "repeat_bool",
        "seed_bool",
        "settings",
        "cap",
        "counts",
        "budget",
        "primary",
        "selection",
        "training_seed",
        "missing_ppo",
        "settings_bool",
        "env_count_bool",
    ],
)
def test_invalid_protocol_fails_before_run_creation(tmp_path, bad):
    c = read_json("configs/evasive_training_pilot.json")
    p = read_json(c["evaluation_protocol"])
    if bad == "availability":
        p["controllers"][1]["available"] = "fallback"
    elif bad == "missing_controller":
        p["controllers"].pop()
    elif bad == "repeat_bool":
        p["controllers"][-1]["repeats"] = True
    elif bad == "seed_bool":
        p["sampling_repeat_seeds"][0] = True
    elif bad == "settings":
        c["survivor_threat_settings"].pop("memory_seconds")
    elif bad == "cap":
        c["max_episode_seconds"] = -1
    elif bad == "counts":
        c["manifest_counts"]["validation"] = 0
    elif bad == "budget":
        c["requested_steps"] = 0
    elif bad == "primary":
        p["primary_inference"] = "argmax"
    elif bad == "training_seed":
        c.pop("training_seed")
    elif bad == "missing_ppo":
        c["ppo"].pop("learning_rate")
    elif bad == "settings_bool":
        c["survivor_threat_settings"]["memory_seconds"] = True
    elif bad == "env_count_bool":
        c["n_envs"] = True
    else:
        p["selection"]["primary"] = "return"
    path = tmp_path / "bad_protocol.json"
    write_json(path, p)
    c["evaluation_protocol"] = str(path)
    with pytest.raises(ValueError):
        prepare_run(tmp_path / "never_created", c)
    assert not (tmp_path / "never_created").exists()


def test_actual_review_config_validates_without_running_it():
    c = read_json("configs/evasive_training_pilot.json")
    validate_protocol(c, read_json(c["evaluation_protocol"]))
    with pytest.raises(ValueError, match="prepared frozen"):
        frozen_protocol(c)


def test_real_sampled_masked_replay_restores_rng_mode_env_and_rollout(
    prepared, policy, monkeypatch
):
    run, config, manifest, source = prepared
    model, env = policy
    env.reset()
    state = (
        env.cursor,
        env.current_case,
        env.unwrapped.core.snapshot(),
        env.unwrapped.elapsed_ticks,
    )
    model._last_obs = np.arange(352, dtype=np.float32)[None, :]
    last_obs = model._last_obs.copy()
    rollout = deepcopy(model.rollout_buffer.__dict__)
    before_hash = parameter_hash(model)
    model.policy.set_training_mode(True)
    rng = snapshot_rng()
    original = model.predict
    manual_seed = torch.manual_seed
    seed_calls = []

    def checked_seed(seed):
        seed_calls.append(seed)
        return manual_seed(seed)

    monkeypatch.setattr(torch, "manual_seed", checked_seed)
    calls = []

    def checked(obs, **kwargs):
        mask = kwargs["action_masks"]
        assert mask.dtype == np.bool_ and mask.shape == (32,) and mask.any()
        action, extra = original(obs, **kwargs)
        assert mask[int(action)]
        calls.append(kwargs["deterministic"])
        return action, extra

    monkeypatch.setattr(model, "predict", checked)
    schedule = read_json(run / "sampling_schedule.json")["validation"]
    first, rows = evaluate_suite(
        config,
        manifest["splits"]["validation"],
        run / "check_first",
        schedule=schedule,
        models={"trained": model},
        stage="trained",
    )
    assert calls.count(False) == 12 and calls.count(True) == 4
    assert seed_calls == [e["sampling_seeds"][repeat] for repeat in range(3) for e in schedule] + [
        e["sampling_seeds"][0] for e in schedule
    ]
    assert_rng_equal(rng)
    assert model.policy.training and parameter_hash(model) == before_hash
    assert (
        env.cursor,
        env.current_case,
        env.unwrapped.core.snapshot(),
        env.unwrapped.elapsed_ticks,
    ) == state
    assert np.array_equal(model._last_obs, last_obs)
    for key, value in rollout.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(model.rollout_buffer.__dict__[key], value)
    assert model.num_timesteps == model._n_updates == 0
    metadata = save_checkpoint(
        run / "checkpoints/check.zip", model, config, source, role="untrained"
    )
    loaded, _ = load_checkpoint(run / "checkpoints/check.zip", config, source)
    second, reloaded = evaluate_suite(
        config,
        manifest["splits"]["validation"],
        run / "check_second",
        schedule=schedule,
        models={"trained": loaded},
        stage="trained",
    )
    assert first == second and rows == reloaded
    assert metadata["evaluation_identity"] == config["frozen_evaluation"]
    assert metadata["evaluated_at_decision_steps"] is None
    with pytest.raises(ValueError, match="explicit frozen"):
        evaluate(
            "trained", config, manifest["splits"]["validation"], run / "forbidden.csv", model=model
        )


def test_baseline_controllers_repeats_seeds_and_masked_uniform(prepared, policy, monkeypatch):
    run, config, manifest, _ = prepared
    model, _ = policy
    schedule = read_json(run / "sampling_schedule.json")["validation"]
    summary, rows = evaluate_suite(
        config,
        manifest["splits"]["validation"],
        run / "baseline_fixture",
        schedule=schedule,
        models={"untrained": model},
        stage="baseline",
    )
    assert {name: len(entries) for name, entries in rows.items()} == {
        "random_valid_mask": 6,
        "untrained_sampled": 6,
        "untrained_deterministic": 2,
        "heuristic": 2,
        "wait": 2,
    }
    # Uniform valid-mask actions match the local generator for the whole tiny episode.
    from rl.pipeline import make_env

    env = make_env(config)
    try:
        import hashlib

        for row in rows["random_valid_mask"]:
            case = next(c for c in manifest["splits"]["validation"] if c["id"] == row["id"])
            env.reset(
                seed=case["reset_seed"],
                options={"scenario_seed": case["scenario_seed"], "bot_seed": case["bot_seed"]},
            )
            rng = np.random.default_rng(row["sampling_seed"])
            actions = []
            while True:
                expected = int(rng.choice(np.flatnonzero(env.action_masks())))
                actions.append(expected)
                _, _, terminal, truncated, _ = env.step(expected)
                if terminal or truncated:
                    break
            assert row["action_sha256"] == hashlib.sha256(bytes(actions)).hexdigest()
    finally:
        env.close()
    for row in rows["untrained_sampled"]:
        other = next(
            r
            for r in rows["random_valid_mask"]
            if r["id"] == row["id"] and r["repeat_index"] == row["repeat_index"]
        )
        assert row["sampling_seed"] == other["sampling_seed"]
    assert summary["untrained_sampled"]["cases"] == 2
    assert len(summary["untrained_sampled"]["per_repeat"]) == 3
    with pytest.raises(ValueError, match="available"):
        evaluate_suite(
            config,
            manifest["splits"]["validation"],
            run / "missing_model",
            schedule=schedule,
            models={},
            stage="baseline",
        )
    with pytest.raises(ValueError, match="schedule"):
        evaluate_suite(
            config,
            manifest["splits"]["validation"],
            run / "bad_schedule",
            schedule=schedule[::-1],
            models={"untrained": model},
            stage="baseline",
        )


def test_case_cluster_mean_and_trained_only_selection_ignores_secondary():
    rows = [
        {
            "id": i,
            "outcome": result,
            "return": 0,
            "decision_steps": 1,
            "elapsed_ticks": 1,
            "repeat_index": j,
        }
        for i, outcomes in (("a", ["captured"] * 3), ("b", ["escaped"]))
        for j, result in enumerate(outcomes)
    ]
    s = summarize_cases(rows, "sampled_masked")
    assert s["case_mean_sampled_catch_rate"] == 0.5  # pooled 3/4 would be wrong
    suite = {"trained_sampled": s, "trained_deterministic": {"case_mean_catch_rate": 1.0}}
    assert trained_score(suite, 0, 0) is None
    assert trained_score(suite, 10, 0) is None
    a = trained_score(suite, 10, 1)
    suite["trained_deterministic"]["case_mean_catch_rate"] = 0
    assert trained_score(suite, 10, 1) == a
    assert a > trained_score(suite, 20, 2)
    suite["trained_sampled"]["per_case"][1]["unresolved"] = 1
    assert a > trained_score(suite, 10, 1)
    with pytest.raises(KeyError):
        trained_score({"untrained_sampled": s}, 10, 1)


def init_callback(prepared, policy):
    run, config, manifest, source = prepared
    model, _ = policy
    model.set_logger(configure(folder=None, format_strings=[]))
    cb = PilotCallback(run, config, manifest["splits"]["validation"], source)
    cb.init_callback(model)
    return cb


def test_callback_fixed_weights_timing_best_last_and_fresh_replay(prepared, policy, monkeypatch):
    run, config, manifest, source = prepared
    model, env = policy
    cb = init_callback(prepared, policy)
    original = parameter_hash(model)
    # Controlled counters simulate callback lifecycle; no optimizer is ever called.
    model.num_timesteps, model._n_updates = 16, 0
    try:
        cb.validate_checkpoint(run / "checkpoints/initial_interval.zip", "interval")
        assert cb.best_steps is None
        env.reset()
        env_state = (env.cursor, env.unwrapped.core.snapshot())
        monkeypatch.setattr(
            env, "step", lambda *a, **k: pytest.fail("callback touched training env")
        )
        monkeypatch.setattr(
            env, "reset", lambda *a, **k: pytest.fail("callback reset training env")
        )
        model.num_timesteps, model._n_updates = 32, 10
        cb.num_timesteps = 32
        cb.locals = {"dones": [False], "infos": [{}]}
        assert cb._on_step()
        assert cb.best_steps == 32
        metadata = read_json(run / "checkpoints/best.metadata.json")
        assert metadata["evaluated_at_decision_steps"] == 32
        assert metadata["evaluated_at_optimizer_update_epochs"] == 10
        assert (
            metadata["evaluation_context"]
            == "rollout_callback_before_current_rollout_optimizer_update"
        )
        assert (env.cursor, env.unwrapped.core.snapshot()) == env_state
        best_hash = file_hash(run / "checkpoints/best.zip")
        model.num_timesteps, model._n_updates = 48, 20
        cb.validate_checkpoint(run / "checkpoints/last.zip", "last")
        assert cb.best_steps == 32 and file_hash(run / "checkpoints/best.zip") == best_hash
        assert (
            read_json(run / "checkpoints/last.metadata.json")["evaluation_context"]
            == "after_learning_optimizer_update"
        )
        loaded, _ = load_checkpoint(run / "checkpoints/best.zip", config, source)
        summary, rows = evaluate_suite(
            config,
            manifest["splits"]["validation"],
            run / "fresh_replay",
            schedule=read_json(run / "sampling_schedule.json")["validation"],
            models={"trained": loaded},
            stage="trained",
        )
        assert summary == cb.best_summary and rows == cb.best_rows
        assert parameter_hash(model) == original
    finally:
        cb.close()


def selection_fixture(prepared, policy):
    run, config, manifest, source = prepared
    model, _ = policy
    save_checkpoint(run / "checkpoints/untrained.zip", model, config, source, role="untrained")
    cb = init_callback(prepared, policy)
    try:
        model.num_timesteps, model._n_updates = (
            16,
            10,
        )  # synthetic trained counters, fixed initial weights
        cb.validate_checkpoint(run / "checkpoints/step.zip", "interval")
        loaded, _ = load_checkpoint(run / "checkpoints/best.zip", config, source)
        summary, rows = evaluate_suite(
            config,
            manifest["splits"]["validation"],
            run / "validation/best_reloaded",
            schedule=read_json(run / "sampling_schedule.json")["validation"],
            models={"trained": loaded},
            stage="trained",
        )
        assert summary == cb.best_summary and rows == cb.best_rows
        write_json(
            run / "selection.json",
            {
                "best_steps": 16,
                "best_optimizer_update_epochs": 10,
                "best_checkpoint_sha256": file_hash(run / "checkpoints/best.zip"),
                "validation_action_rows_sha256": digest(rows),
                "rule": config["frozen_evaluation"]["protocol"]["selection"],
                "best_validation": cb.best_summary,
                "fresh_load_reproduced_actions_and_episodes": True,
            },
        )
    finally:
        cb.close()


def test_final_test_requires_selection_and_runs_once_on_synthetic_split(prepared, policy):
    run, _, _, _ = prepared
    with pytest.raises(FileNotFoundError):
        final_sampled_test(run)
    assert not (run / "test_started.txt").exists()
    selection_fixture(prepared, policy)
    summary, rows = final_sampled_test(run)
    assert sum(len(r) for r in rows.values()) == 13
    assert len(summary) == 7 and read_json(run / "test_status.json")["status"] == "complete"
    with pytest.raises(FileExistsError):
        final_sampled_test(run)


def test_failed_test_stays_partial_and_never_retries(prepared, policy, monkeypatch):
    run, _, _, _ = prepared
    selection_fixture(prepared, policy)

    def fail(*args, **kwargs):
        raise RuntimeError("controlled interrupted test")

    monkeypatch.setattr("rl.train.evaluate_suite", fail)
    with pytest.raises(RuntimeError, match="controlled interrupted"):
        final_sampled_test(run)
    assert read_json(run / "test_status.json")["status"] == "partial"
    with pytest.raises(FileExistsError):
        final_sampled_test(run)


def test_checkpoint_evaluation_identity_mutation_rejected(prepared, policy):
    run, config, _, source = prepared
    model, _ = policy
    path = run / "checkpoints/new.zip"
    save_checkpoint(path, model, config, source, role="untrained")
    metadata = read_json(path.with_suffix(".metadata.json"))
    metadata["evaluation_identity"]["schedule_sha256"] = "bad"
    write_json(path.with_suffix(".metadata.json"), metadata)
    with pytest.raises(ValueError, match="evaluation_identity"):
        load_checkpoint(path, config, source)


def test_legacy_without_protocol_is_explicitly_deterministic(tmp_path, monkeypatch):
    config = read_json("configs/training_pilot.json")
    config.update(
        manifest_seed=9100328,
        manifest_counts={"train": 1, "validation": 1, "test": 1},
        max_episode_seconds=1 / 30,
    )
    run = tmp_path / "legacy"
    prepare_run(run, config)
    assert read_run(run)[0] == config
    assert read_json(run / "run_identity.json")["evaluation_mode"] == "legacy_deterministic_argmax"
    env = ManifestEnv(
        config, make_manifest(config["manifest_seed"], config["manifest_counts"])["splits"]["train"]
    )
    try:
        model = new_model(config, env)
        predict = model.predict

        def checked(*args, **kwargs):
            assert kwargs["deterministic"] is True
            return predict(*args, **kwargs)

        monkeypatch.setattr(model, "predict", checked)
        _, rows = evaluate("untrained", config, env.cases, run / "legacy.csv", model=model)
        assert "inference" not in rows[0]  # legacy metric and CSV contracts stay intact
    finally:
        env.close()


@pytest.mark.parametrize(
    "field",
    [
        "actual_steps",
        "optimizer_update_epochs",
        "evaluated_at_decision_steps",
        "evaluated_at_optimizer_update_epochs",
        "evaluation_context",
    ],
)
def test_checkpoint_timing_metadata_is_checked_on_load(prepared, policy, field):
    run, config, _, source = prepared
    model, _ = policy
    model.num_timesteps, model._n_updates = 16, 10  # controlled counters, fixed weights
    path = run / "checkpoints/timing.zip"
    save_checkpoint(
        path,
        model,
        config,
        source,
        role="interval",
        validation={"fixture": True},
        evaluation_context="rollout_callback_before_current_rollout_optimizer_update",
    )
    metadata = read_json(path.with_suffix(".metadata.json"))
    metadata[field] = "wrong" if field == "evaluation_context" else 999
    write_json(path.with_suffix(".metadata.json"), metadata)
    with pytest.raises(ValueError, match="counters mismatch|timing mismatch"):
        load_checkpoint(path, config, source)


def test_failed_evaluation_restores_rng_and_training_mode(prepared, policy, monkeypatch):
    run, config, manifest, _ = prepared
    model, _ = policy
    model.policy.set_training_mode(True)
    before = snapshot_rng()

    def fail(*args, **kwargs):
        random.random()
        np.random.random()
        torch.rand(1)
        raise RuntimeError("controlled predict failure")

    monkeypatch.setattr(model, "predict", fail)
    with pytest.raises(RuntimeError, match="controlled predict failure"):
        evaluate_suite(
            config,
            manifest["splits"]["validation"],
            run / "interrupted_validation",
            schedule=read_json(run / "sampling_schedule.json")["validation"],
            models={"trained": model},
            stage="trained",
        )
    assert_rng_equal(before)
    assert model.policy.training


def test_selection_replay_mutation_blocks_test_before_claim(prepared, policy):
    run, _, _, _ = prepared
    selection_fixture(prepared, policy)
    rows_path = run / "validation/best_reloaded/episodes.json"
    rows = read_json(rows_path)
    rows["trained_sampled"][0]["action_sha256"] = "tampered"
    write_json(rows_path, rows)
    with pytest.raises(ValueError, match="selected, replay-verified"):
        final_sampled_test(run)
    assert not (run / "test_started.txt").exists()


def test_post_learning_orchestration_with_fixed_weights_and_synthetic_counters(prepared, policy):
    from time import perf_counter

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from rl.train import complete_sampled_pilot

    run, config, manifest, source = prepared
    model, _ = policy
    before = parameter_hash(model)
    untrained = save_checkpoint(
        run / "checkpoints/untrained.zip", model, config, source, role="untrained"
    )
    cb = init_callback(prepared, policy)
    model.set_logger(configure(str(run / "tensorboard"), ["tensorboard"]))
    try:
        # These are lifecycle counters only. The session guard prevents any real update.
        model.num_timesteps, model._n_updates = 16, 10
        cb.validate_checkpoint(run / "checkpoints/last.zip", "last")
        result = complete_sampled_pilot(
            run, config, manifest, source, model, untrained, cb, {}, perf_counter(), 0
        )
        assert result["fresh_load_reproduced"] and not result["weights_changed"]
        assert parameter_hash(model) == before
        assert read_json(run / "test_status.json")["status"] == "complete"
        selection = read_json(run / "selection.json")
        assert selection["best_steps"] == 16 and selection["best_optimizer_update_epochs"] == 10
        tags = EventAccumulator(str(run / "tensorboard")).Reload().Tags()["scalars"]
        assert "validation/sampled_primary_case_mean_catch_rate" in tags
        assert "validation/argmax_secondary_case_mean_catch_rate" in tags
        assert "validation/evaluated_optimizer_update_epochs" in tags
        header = (run / "validation_curve.csv").read_text().splitlines()[0]
        assert "sampled_primary_case_mean_catch_rate" in header and "argmax_secondary" in header
    finally:
        cb.close()


@pytest.mark.parametrize("outcome", ["captured", "unresolved"])
def test_exact_checkpoint_tie_with_82_of_150_distributed_differently(outcome):
    # Both have 82 successes across 50 cases x 3 repeats. Different arrangements
    # provoke unequal float means through the real aggregation function.
    distributions = ([2] * 32 + [1] * 18, [3] * 27 + [1] + [0] * 22)
    if outcome == "unresolved":
        distributions = distributions[::-1]  # lower float unresolved must not win either
    summaries = []
    for counts in distributions:
        rows = [
            {
                "id": str(i),
                "outcome": outcome if repeat < count else "escaped",
                "return": 0,
                "decision_steps": 1,
                "elapsed_ticks": 1,
                "repeat_index": repeat,
            }
            for i, count in enumerate(counts)
            for repeat in range(3)
        ]
        summaries.append(summarize_cases(rows, "sampled_masked"))
    field = (
        "case_mean_sampled_catch_rate"
        if outcome == "captured"
        else "case_mean_sampled_unresolved_rate"
    )
    assert summaries[0][field] != summaries[1][field]
    assert summaries[0][outcome] == summaries[1][outcome] == 82
    first = trained_score({"trained_sampled": summaries[0]}, 100, 10)
    later = trained_score({"trained_sampled": summaries[1]}, 200, 20)
    assert first[:2] == later[:2]
    assert first > later  # an exact tie must retain the earlier trained checkpoint


def test_exact_selection_preserves_case_weighting_and_tiny_real_improvement():
    from fractions import Fraction

    unequal = {
        "trained_sampled": {
            "per_case": [
                {"captured": 3, "unresolved": 0, "repeats": 3},
                {"captured": 0, "unresolved": 1, "repeats": 1},
            ]
        }
    }
    assert trained_score(unequal, 100, 10) == (Fraction(1, 2), -Fraction(1, 2), -100)
    # No tolerance or rounded float may suppress an actual score improvement.
    first = {"trained_sampled": {"per_case": [{"captured": 1, "unresolved": 0, "repeats": 10**18}]}}
    better = deepcopy(first)
    better["trained_sampled"]["per_case"][0]["captured"] = 2
    assert trained_score(better, 200, 20) > trained_score(first, 100, 10)
