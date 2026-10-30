"""Pure/static and mocked integration only: no native actor, policy or episode execution."""

from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction

import pytest

from game.investigation import InvestigationSnapshot
from game.map import DEFAULT_SCENARIO, RANDOM_SCENARIO, scenario_map, small_test_map
from game.perception import PerceptionSnapshot
from game.state import AlienState, CoreConfig, SurvivorState, WorldSnapshot
from rl import pipeline
from rl.env import AlienEnv
from rl.evaluation_protocol import evaluation_budget, trained_score, validate_protocol
from rl.search_metrics import SearchMetrics, summarize_search
from rl.train import validate_fresh_start

ENV_STEP = AlienEnv.step
ENV_INIT = AlienEnv.__init__  # Called only with a namespace and fully mocked actors below.

LOOP_HASH = "af5142aeeee14782b5a3fc668f9ea97432c5bff72f8f498b920c3a4aad7f850f"
DEFAULT_HASH = "7aeafcc0b15f5361f4baefa420ceb5f17fc5f38eccd82fbdd52c6e9ca215736a"
RANDOM_HASH = "f578a895cf84f31b185068456c69b00c92a68371a837430fd93d965ac3a0fef1"

# Independent expected fixture ships with this test; no ignored experiment data.
EXPECTED_LOOP_WORLD = {
    "alien_spawn_options": [],
    "rows": [
        "################",
        "#..............#",
        "#..............#",
        "#..............#",
        "#....######....#",
        "#....######....#",
        "#....######....#",
        "#....######....#",
        "#....######....#",
        "#....######....#",
        "#....######....#",
        "#..............#",
        "#........##....#",
        "#........##....#",
        "#........##....#",
        "################",
    ],
    "spawn_points": [[2.5, 11.5], [2.5, 13.5]],
    "objects": [
        {"object_id": "keycard", "kind": "keycard", "x": 2.5, "y": 2.5},
        {"object_id": "exit", "kind": "exit", "x": 13.5, "y": 13.5},
    ],
    "alien_spawn": [13.5, 2.5, 3.141592653589793],
    "patrol_waypoints": [
        [2.5, 2.5],
        [2.5, 13.5],
        [13.5, 13.5],
        [13.5, 2.5],
        [4.5, 11.5],
        [11.5, 11.5],
    ],
}


@pytest.fixture(autouse=True)
def forbid_runtime(monkeypatch):
    import torch

    from game.core import GameCore
    from rl.bots.evasive import EvasiveExplorerBot
    from rl.bots.explorer import ExplorerBot
    from rl.env import AlienEnv

    def forbidden(*a, **kw):
        pytest.fail("Native actor/model/rollout execution forbidden in pure readiness tests")

    for cls, names in (
        (GameCore, ("__init__", "reset", "step")),
        (AlienEnv, ("__init__", "reset", "step")),
        (EvasiveExplorerBot, ("__init__", "reset", "command")),
        (ExplorerBot, ("__init__", "reset", "command")),
        (pipeline.MaskablePPO, ("__init__", "load", "predict", "learn")),
        (torch.optim.Adam, ("step",)),
    ):
        for name in names:
            monkeypatch.setattr(cls, name, forbidden)


@pytest.fixture
def config():
    return pipeline.read_json("configs/scenario_randomization_training_pilot.json")


def test_exact_reviewed_world_and_default_backward_identity():
    assert asdict(scenario_map()) == asdict(small_test_map())
    assert pipeline.map_identity()["sha256"] == DEFAULT_HASH
    world = scenario_map("occluded_loop_dev_v1")
    assert pipeline.digest(asdict(world)) == LOOP_HASH
    assert pipeline.digest(EXPECTED_LOOP_WORLD) == LOOP_HASH
    assert pipeline.read_json_value(asdict(world)) == EXPECTED_LOOP_WORLD
    assert pipeline.map_identity("occluded_loop_dev_v1")["name"] == "occluded_loop_dev_v1"
    assert world.spawn_points == ((2.5, 11.5), (2.5, 13.5))
    assert world.width == world.height == 16


@pytest.mark.parametrize("name", [None, "", "occluded", "small_test_map", 1, True])
def test_unknown_scene_fails_closed(name):
    with pytest.raises(ValueError, match="Unknown scenario"):
        scenario_map(name)


def test_factory_routes_once_with_explicit_scene(config, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "AlienEnv", lambda **kw: calls.append(kw) or kw)
    assert pipeline.make_env(config)["scenario"] == RANDOM_SCENARIO
    assert len(calls) == 1


def test_protocol_budgets_and_static_settings(config):
    p = pipeline.read_json(config["evaluation_protocol"])
    validate_protocol(config, p)
    assert pipeline.validate_environment_config(config)["sha256"] == RANDOM_HASH
    budget = evaluation_budget(config)
    assert budget == {
        "rollout_rounded_decisions": 40192,
        "baseline_validation_episodes": 450,
        "checkpoint_validation_calls": 5,
        "checkpoint_validation_episodes": 1000,
        "fresh_best_validation_replay_episodes": 200,
        "final_test_episodes": 650,
        "max_episode_seconds": 600,
    }
    p["scenario"] = DEFAULT_SCENARIO
    with pytest.raises(ValueError, match="scenario mismatch"):
        validate_protocol(config, p)


def test_pure_prepare_freezes_scenario_provenance_and_disjoint_seeds(config, tmp_path):
    run = tmp_path / "prepared"
    manifest, source = pipeline.prepare_run(run, config)
    frozen, stored, actual = pipeline.read_run(run)
    assert pipeline.read_json_value(manifest) == stored and actual == source
    assert stored["map"]["sha256"] == RANDOM_HASH
    assert stored["map"]["name"] == frozen["scenario"]
    assert frozen["fresh_run"] and frozen["defer_final_test"]
    assert not list(run.rglob("*.zip"))
    seeds = [
        c[k]
        for cases in stored["splits"].values()
        for c in cases
        for k in ("reset_seed", "scenario_seed", "bot_seed", "selector_seed")
    ]
    assert len(seeds) == len(set(seeds))
    assert frozen["frozen_evaluation"]["protocol"]["scenario"] == frozen["scenario"]


@pytest.mark.parametrize("tamper", ["name", "sha256", "spawn_points", "case_seed"])
def test_manifest_semantic_mismatch_even_if_outer_digest_updated(config, tmp_path, tamper):
    run = tmp_path / "prepared"
    pipeline.prepare_run(run, config)
    manifest = pipeline.read_json(run / "manifest.json")
    if tamper == "case_seed":
        manifest["splits"]["train"][0]["bot_seed"] += 1
    else:
        manifest["map"][tamper] = "wrong"
    pipeline.write_json(run / "manifest.json", manifest)
    identity = pipeline.read_json(run / "run_identity.json")
    identity["manifest_sha256"] = pipeline.digest(manifest)
    pipeline.write_json(run / "run_identity.json", identity)
    with pytest.raises(ValueError, match="scenario identity mismatch"):
        pipeline.read_run(run)


@pytest.mark.parametrize("tamper", ["name", "sha256", "spawn_points", "missing"])
def test_checkpoint_scene_mismatch_rejected_before_any_load(config, tmp_path, monkeypatch, tamper):
    # Synthetic fake save only; no actual model is created or loaded.
    from types import SimpleNamespace

    pipeline.prepare_run(tmp_path / "run", config)
    config, _, source = pipeline.read_run(tmp_path / "run")
    model = SimpleNamespace(
        num_timesteps=0,
        _n_updates=0,
        save=lambda p: __import__("pathlib").Path(p).write_text("synthetic only"),
    )
    monkeypatch.setattr(pipeline, "parameter_hash", lambda m: "synthetic-parameter-hash")
    monkeypatch.setattr(pipeline.MaskablePPO, "load", lambda *a, **k: pytest.fail("no load"))
    path = tmp_path / "synthetic.zip"
    metadata = pipeline.save_checkpoint(path, model, config, source, role="untrained")
    if tamper == "missing":
        metadata.pop("map")
    else:
        metadata["map"][tamper] = "wrong"
    pipeline.write_json(path.with_suffix(".metadata.json"), metadata)
    with pytest.raises(ValueError, match="mismatch: map"):
        pipeline.load_checkpoint(path, config, source)


@pytest.mark.parametrize(
    "key,value",
    [
        ("actual_steps", 1),
        ("optimizer_update_epochs", 1),
        ("role", "last"),
        ("training_seed", 0),
        ("config_sha256", "wrong"),
    ],
)
def test_fresh_run_rejects_nonzero_or_foreign_policy(config, key, value):
    metadata = {
        "role": "untrained",
        "actual_steps": 0,
        "optimizer_update_epochs": 0,
        "training_seed": config["training_seed"],
        "config_sha256": pipeline.digest(config),
    }
    validate_fresh_start(config, metadata)
    metadata[key] = value
    with pytest.raises(ValueError, match="Fresh run"):
        validate_fresh_start(config, metadata)


def test_exact_fraction_selection_parity_ignores_float_and_argmax():
    # Same 82/150 count, distinct distribution; displayed IEEE floats disagree.
    def summary(counts, display):
        return {
            "trained_sampled": {
                "per_case": [{"captured": c, "unresolved": 0, "repeats": 3} for c in counts],
                "case_mean_sampled_catch_rate": display,
            },
            "trained_deterministic": {"case_mean_catch_rate": 1},
        }

    a = summary([3] * 27 + [1] + [0] * 22, 0.5466666666666665)
    b = summary([2] * 32 + [1] * 18, 0.5466666666666666)
    assert trained_score(a, 5000, 10)[:2] == (Fraction(82, 150), 0)
    assert trained_score(b, 5000, 10) == trained_score(a, 5000, 10)
    assert trained_score(b, 10000, 20) < trained_score(a, 5000, 10)
    assert trained_score(a, 5000, 0) is None


def snap(tick=0, **kw):
    values = dict(
        tick=tick,
        seed=0,
        survivor=SurvivorState(5.3, 11.5),
        world_map=scenario_map("occluded_loop_dev_v1"),
        config=CoreConfig(),
        alien=AlienState(3.4372358, 7.0352116, 1.0),
        perception=PerceptionSnapshot(True, 6, (5.1, 11.5), 0, 29),
    )
    values.update(kw)
    return WorldSnapshot(**values)


def test_wall_loss_eligible_continue_cancel_and_no_mutation():
    metrics = SearchMetrics()
    before = snap(185)
    lost = replace(before, tick=186, perception=replace(before.perception, visible=False))
    original = asdict(before)
    metrics.tick(before, lost)
    active = InvestigationSnapshot("travelling", lost.perception.last_seen, 29)
    started = replace(
        lost, investigation=active, alien_target_source="last_seen", alien_target=active.target
    )
    metrics.decision(lost, started, True, 21)
    metrics.decision(replace(started, tick=195), replace(started, tick=195), True, 21)
    reacquired = replace(
        started,
        tick=210,
        perception=before.perception,
        investigation=replace(active, phase="cancelled"),
    )
    metrics.tick(started, reacquired)
    result = metrics.finish(reacquired)
    assert result["wall_loss_opportunities"] == 1
    assert result["eligible_action21"] == {"numerator": 2, "denominator": 2, "rate": 1}
    assert result["same_primitive_unseen_continuation"]["numerator"] == 1
    assert result["cancel_on_reacquisition"]["rate"] == 1
    assert result["windows"][0]["end_reason"] == "reacquired"
    assert asdict(before) == original
    assert metrics.finish(reacquired) == result


def test_no_wall_opportunity_is_na_and_fov_loss_excluded():
    metrics = SearchMetrics()
    # Native range/FOV fail; wall alone is not the cause.
    before = replace(snap(), alien=AlienState(3.4372358, 7.0352116, -2.0))
    after = replace(before, tick=1, perception=replace(before.perception, visible=False))
    metrics.tick(before, after)
    result = metrics.finish(after)
    assert result["confirmed_losses"] == 1
    assert result["wall_loss_opportunities"] == 0
    for key in (
        "eligible_action21",
        "same_primitive_unseen_continuation",
        "cancel_on_reacquisition",
    ):
        assert result[key] == {"numerator": 0, "denominator": 0, "rate": None}
    assert summarize_search([{"search_metrics": result}])["eligible_action21"]["rate"] is None


def test_switch_restart_terminal_and_aggregate_denominators():
    metrics = SearchMetrics()
    before = snap()
    lost = replace(before, tick=1, perception=replace(before.perception, visible=False))
    metrics.tick(before, lost)
    active = InvestigationSnapshot("travelling", lost.perception.last_seen, 29)
    start = replace(
        lost, investigation=active, alien_target=active.target, alien_target_source="last_seen"
    )
    metrics.decision(lost, start, True, 21)
    metrics.decision(start, replace(start, alien_target=(2.5, 2.5)), True, 0)
    # Revision/rotation reset must not count as same primitive continuation.
    metrics.decision(
        start, replace(start, investigation=replace(active, rotation_ticks=1)), True, 21
    )
    terminal = replace(start, tick=222, outcome="captured")
    metrics.tick(start, terminal)
    m = metrics.finish(terminal)
    empty = SearchMetrics().finish(snap())
    summary = summarize_search([{"search_metrics": m}, {"search_metrics": empty}])
    assert m["same_primitive_unseen_continuation"] == {"numerator": 0, "denominator": 2, "rate": 0}
    assert m["target_switching"]["numerator"] == 2
    assert summary["episodes_with_wall_loss"]["rate"] == 0.5
    assert summary["eligible_action21"]["numerator"] == 2
    assert summary["eligible_action21"]["denominator"] == 3
    assert m["windows"][0]["end_reason"] == "captured"


@pytest.mark.parametrize("value", [None, "wrong", DEFAULT_HASH])
def test_config_hash_mismatch_before_factory_or_prepare(config, tmp_path, value):
    config["scenario_sha256"] = value
    with pytest.raises(ValueError, match="full-world hash mismatch"):
        pipeline.make_env(config)
    with pytest.raises(ValueError, match="full-world hash mismatch"):
        pipeline.prepare_run(tmp_path / "never-created", config)
    assert not (tmp_path / "never-created").exists()


def test_evaluate_serializes_passive_metrics_using_only_fake_environment(
    config, tmp_path, monkeypatch
):
    import csv
    import json

    from rl import evaluate as evaluation

    class FakeEnv:
        def __init__(self):
            self.config = CoreConfig()
            self.closed = False

        def reset(self, **kw):
            self.steps = 0
            return None, {}

        def action_masks(self):
            import numpy as np

            mask = np.zeros(32, dtype=bool)
            mask[30] = True
            return mask

        def step(self, action):
            assert action == 30
            self.steps += 1
            return (
                None,
                0,
                False,
                True,
                {
                    "outcome": "unresolved",
                    "elapsed_ticks": 15,
                    "invalid_action": False,
                    "reward_components": {"capture": 0, "escape": 0, "first_sight": 0},
                    "search_metrics": SearchMetrics().finish(snap(15)),
                },
            )

        def close(self):
            self.closed = True

    fake = FakeEnv()
    monkeypatch.setattr(evaluation, "make_env", lambda c: fake)
    case = {
        "id": "synthetic_only",
        "scenario_seed": 1,
        "bot_seed": 2,
        "reset_seed": 3,
        "selector_seed": 4,
    }
    path = tmp_path / "fake.csv"
    _, rows = evaluation.evaluate(
        "wait",
        config,
        [case],
        path,
        inference="constant_wait",
        episode_seeds=[5],
        repeat_index=0,
        controller_name="wait",
    )
    with path.open() as handle:
        saved = next(csv.DictReader(handle))
    assert json.loads(saved["search_metrics"]) == rows[0]["search_metrics"]
    assert fake.closed and fake.steps == 1


def test_suite_reuses_paired_schedule_and_attaches_case_denominators(config, tmp_path, monkeypatch):
    from rl import evaluate as evaluation
    from rl.evaluation_protocol import sampling_schedule

    run = tmp_path / "prepared"
    pipeline.prepare_run(run, config)
    c, manifest, _ = pipeline.read_run(run)
    cases = manifest["splits"]["validation"][:2]
    calls = []

    def fake_evaluate(kind, config, cases, path, **kw):
        calls.append((kind, kw["inference"], kw["repeat_index"], kw["episode_seeds"]))
        rows = [
            {
                **case,
                "return": 0,
                "outcome": "unresolved",
                "decision_steps": 1,
                "elapsed_ticks": 15,
                "repeat_index": kw["repeat_index"],
                "evaluation_role": "baseline",
                "search_metrics": SearchMetrics().finish(snap()),
            }
            for case in cases
        ]
        return {}, rows

    monkeypatch.setattr(evaluation, "evaluate", fake_evaluate)
    schedule = sampling_schedule(c["frozen_evaluation"]["protocol"], manifest)["validation"][:2]
    summary, rows = evaluation.evaluate_suite(
        c,
        cases,
        tmp_path / "fake_suite",
        schedule=schedule,
        models={"untrained": object()},
        stage="baseline",
    )
    assert len(calls) == 9
    assert {k: len(v) for k, v in rows.items()} == {
        "random_valid_mask": 6,
        "untrained_sampled": 6,
        "untrained_deterministic": 2,
        "heuristic": 2,
        "wait": 2,
    }
    assert calls[0][3] == calls[3][3]  # paired random/untrained repeat zero
    assert summary["untrained_sampled"]["search"]["episodes"] == 6
    for case in summary["untrained_sampled"]["per_case"]:
        assert case["search"]["episodes"] == 3
        assert case["search"]["eligible_action21"]["rate"] is None


def test_completion_defers_heldout_without_calling_test_or_real_load(config, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from rl import train

    config = deepcopy(config)
    config["frozen_evaluation"] = {"protocol": {"selection": "synthetic"}}
    summary, rows = {"synthetic": True}, {"synthetic": []}
    model = SimpleNamespace(num_timesteps=20224, _n_updates=790)
    callback = SimpleNamespace(
        best_score=(1,),
        best_summary=summary,
        best_rows=rows,
        best_steps=5000,
        counts={"captured": 1, "escaped": 0, "unresolved": 0},
    )
    pipeline.write_json(tmp_path / "sampling_schedule.json", {"validation": []})
    metadata = {"optimizer_update_epochs": 790, "policy_sha256": "trained"}
    monkeypatch.setattr(train, "load_checkpoint", lambda *a: (object(), metadata))
    monkeypatch.setattr(train, "evaluate_suite", lambda *a, **kw: (summary, rows))
    monkeypatch.setattr(train, "file_hash", lambda p: "synthetic")
    monkeypatch.setattr(train, "parameter_hash", lambda m: "trained")
    monkeypatch.setattr(train, "final_sampled_test", lambda *a: pytest.fail("held-out forbidden"))
    result = train.complete_sampled_pilot(
        tmp_path,
        config,
        {"splits": {"validation": []}},
        {},
        model,
        {"policy_sha256": "untrained"},
        callback,
        {},
        0,
        1,
    )
    assert result["test"] is None
    assert result["test_status"] == "deferred_pending_separate_authorization"
    assert not (tmp_path / "test_started.txt").exists()


def test_mock_checkpoint_save_records_exact_scene_and_zero_provenance(
    config, tmp_path, monkeypatch
):
    # Text marker and fake save only; no policy serialization or checkpoint load.
    from types import SimpleNamespace

    pipeline.prepare_run(tmp_path / "run", config)
    config = pipeline.read_run(tmp_path / "run")[0]
    model = SimpleNamespace(
        num_timesteps=0,
        _n_updates=0,
        save=lambda p: __import__("pathlib").Path(p).write_text("synthetic only"),
    )
    monkeypatch.setattr(pipeline, "parameter_hash", lambda m: "synthetic-parameter-hash")
    metadata = pipeline.save_checkpoint(
        tmp_path / "fake.zip", model, config, {"source_sha256": "synthetic"}, role="untrained"
    )
    assert metadata["map"]["sha256"] == RANDOM_HASH
    assert metadata["config"]["scenario_sha256"] == RANDOM_HASH
    assert metadata["manifest_sha256"] == pipeline.digest(
        pipeline.make_manifest(
            config["manifest_seed"], config["manifest_counts"], config["scenario"]
        )
    )
    validate_fresh_start(config, metadata)


@pytest.mark.parametrize("name", [None, DEFAULT_SCENARIO, "occluded_loop_dev_v1"])
def test_environment_constructor_routing_with_mocked_actors_only(name, monkeypatch):
    from types import SimpleNamespace

    from rl import env as environment

    captured = []

    def fake_core(world, config, **kw):
        captured.append(world)
        return object()

    monkeypatch.setattr(environment, "GameCore", fake_core)
    monkeypatch.setattr(environment, "ExplorerBot", lambda *a: object())
    assembly = SimpleNamespace()  # No real AlienEnv instance, native Core or bot.
    ENV_INIT(assembly, scenario=name)
    expected = DEFAULT_SCENARIO if name is None else name
    assert captured == [scenario_map(expected)]
    assert assembly.world == scenario_map(expected)
    assert assembly.scenario == expected
    assert assembly.metrics is None
    assert assembly.action_space.n == 32 and assembly.observation_space.shape == (352,)
    with pytest.raises(ValueError, match="never both"):
        ENV_INIT(SimpleNamespace(), world_map=small_test_map(), scenario="occluded_loop_dev_v1")


def test_environment_step_wiring_with_fake_core_only(monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from game.state import SurvivorCommand
    from rl import env as environment

    before = snap(185)
    terminal = replace(
        before, tick=186, outcome="captured", perception=replace(before.perception, visible=False)
    )
    state = [before]
    calls = []

    def fake_step(command):
        calls.append(command)
        state[0] = terminal
        return terminal

    fake_core = SimpleNamespace(
        snapshot=lambda: state[0], step=fake_step, set_alien_target=lambda *a, **kw: None
    )
    candidates = [SimpleNamespace(valid=True, target=None, evidence_revision=None)] * 32
    monkeypatch.setattr(environment, "build_candidates", lambda view: candidates)
    monkeypatch.setattr(environment, "committed_waypoint", lambda view: None)
    monkeypatch.setattr(environment, "build_observation", lambda view: np.zeros(352, np.float32))
    monkeypatch.setattr(environment, "observe_survivor", lambda *args: "filtered-fake-view")
    command = SurvivorCommand()
    assembly = SimpleNamespace(
        _done=False,
        action_space=SimpleNamespace(contains=lambda a: a == 30),
        _allowed_view=lambda: object(),
        core=fake_core,
        config=CoreConfig(),
        metrics=SearchMetrics(),
        world=before.world_map,
        bot_kind="explorer",
        bot=SimpleNamespace(command=lambda view: command, label="synthetic"),
        elapsed_ticks=0,
        cap_ticks=18000,
        visit_ticks=[None] * 6,
        reward_tracker=SimpleNamespace(collect=lambda *a: {"capture": 1}),
        action_masks=lambda: np.ones(32, dtype=bool),
    )
    obs, reward, terminated, truncated, info = ENV_STEP(assembly, 30)
    assert len(calls) == 1 and calls[0] is command
    assert obs.shape == (352,) and reward == 1 and terminated and not truncated
    assert assembly.elapsed_ticks == 1 and assembly._done
    assert info["search_metrics"]["wall_loss_opportunities"] == 1
    window = info["search_metrics"]["windows"][0]
    assert window["loss_tick"] == 185 and window["snapshot_tick"] == 186
    assert window["episode_outcome"] == "captured"
