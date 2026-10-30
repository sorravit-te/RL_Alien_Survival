"""Bounded headless mechanics and mocked identity tests; no real policy execution."""

from copy import deepcopy
from dataclasses import replace
from math import cos, sin
from types import SimpleNamespace

import numpy as np
import pytest

from app.rl_controller import HumanRLController
from game.investigation import InvestigationSnapshot
from game.map import MapObject, WorldMap
from game.perception import PerceptionSnapshot
from game.state import SurvivorCommand, SurvivorState
from rl import pipeline
from rl.candidates import build_candidates, candidate_masks, search_is_restricted
from rl.contract import SEARCH_CONTROL, VERSION, search_contract
from rl.env import AlienEnv
from rl.observations import build_observation
from rl.scripted_controllers import HeuristicSelector
from rl.search_metrics import summarize_search

CLUE = (5.5, 5.5)
PILOT_CONFIG = "configs/scenario_randomization_training_pilot.json"


def arena(survivor=(17.5, 17.5), objects=()):
    return WorldMap(
        ("." * 20,) * 20,
        (survivor,),
        objects,
        alien_spawn=(*CLUE, 0),
        patrol_waypoints=((8.5, 5.5),),
    )


@pytest.fixture
def env():
    e = AlienEnv(arena(), bot_kind="stationary_debug", max_episode_seconds=10)
    e.reset(seed=914, options={"scenario_seed": 199, "bot_seed": 297})
    e.core._perception = PerceptionSnapshot(last_seen=CLUE, last_seen_age_ticks=300, revision=1)
    yield e
    e.close()


def seed(core, target=CLUE, revision=1):
    core._perception = PerceptionSnapshot(
        last_seen=target, last_seen_age_ticks=300, revision=revision
    )


def mask(e):
    return np.flatnonzero(e.action_masks()).tolist()


def test_start_continue_cancel_and_retarget_delay(env):
    assert mask(env) == [0, 21, 30]
    env.step(21)
    before = env.core.snapshot()
    assert before.investigation.rotation_ticks == 15 and mask(env) == [21, 30]
    candidates = build_candidates(env._allowed_view())
    env.core.set_alien_target(
        candidates[21].target,
        source="last_seen",
        evidence_revision=candidates[21].evidence_revision,
    )
    assert env.core.snapshot() == before  # selection preserves all primitive/path fields
    _, reward, _, _, info = env.step(30)
    assert reward == 0 and not info["invalid_action"]
    assert env.core.snapshot().investigation == InvestigationSnapshot()
    assert mask(env) == [0, 21, 30] and env.core.snapshot().tick == 30
    old = env.core.snapshot().alien
    env.step(0)
    assert env.core.snapshot().alien.x > old.x
    control = env.metrics.control
    assert control.counts["policy_cancels"] == 1
    assert (
        control.retargets[0]["cancel_tick"] == 15
        and control.retargets[0]["earliest_retarget_tick"] == 30
    )
    assert control.retargets[0]["actual_delay_ticks"] == 15


def test_masked_retarget_is_explicit_invalid_wait_and_no_reward(env):
    env.step(21)
    _, reward, _, _, info = env.step(0)
    assert info["invalid_action"] and info["applied_action"] == 30 and info["requested_action"] == 0
    assert reward == 0 and env.metrics.control.counts["masked_retarget_requests"] == 1
    assert env.metrics.control.counts["policy_cancels"] == 1
    assert env.metrics.control.events[-1]["reason"] == "masked_action_fallback"


def test_partial_restart_loses_progress_without_marking_completed(env):
    env.step(21)
    env.step(21)
    assert env.core.snapshot().investigation.rotation_ticks == 30
    env.step(30)
    env.step(21)
    s = env.core.snapshot()
    assert s.investigation.rotation_ticks == 15 and s.investigation.investigated_revision is None
    assert env.metrics.control.counts["starts"] == 1 and env.metrics.control.counts["restarts"] == 1


def test_final_rotation_sample_then_completion_and_reset(env):
    env.cap_ticks = 60
    for _ in range(3):
        env.step(21)
    s = env.core.snapshot()
    assert (
        s.tick == 45
        and s.investigation.rotation_ticks == 45
        and s.investigation.phase == "sweeping"
    )
    assert mask(env) == [21, 30]
    _, _, terminal, truncated, info = env.step(21)
    assert not terminal and truncated
    assert s.tick == 45 and env.core.snapshot().investigation.phase == "completed"
    assert env.core.snapshot().investigation.investigated_revision == 1 and mask(env) == [0, 30]
    audit = info["search_metrics"]["control"]
    assert audit["counts"]["completed_sweeps"] == 1 and audit["events"][-1]["tick"] == 46
    assert audit["attempts"][0]["duration_ticks"] == 46
    with pytest.raises(RuntimeError):
        env.step(21)
    env.reset(seed=914)
    assert env.core.snapshot().investigation == InvestigationSnapshot()
    assert all(v == 0 for v in env.metrics.control.counts.values())


@pytest.mark.parametrize("travelling", [False, True])
def test_reacquisition_interrupts_at_tick6_not_policy_boundary(travelling, monkeypatch):
    e = AlienEnv(arena((8.5, 5.5)), bot_kind="stationary_debug")
    try:
        e.reset(seed=1)
        seed(e.core, (10.5, 5.5) if travelling else CLUE)
        records = []
        original = e.core.step

        def spy(command):
            b = e.core.snapshot()
            a = original(command)
            records.append((b, a))
            return a

        monkeypatch.setattr(e.core, "step", spy)
        e.step(21)
        b, a = next((b, a) for b, a in records if a.investigation.phase == "cancelled")
        assert a.tick == 6 and a.tick % 15 != 0
        # The old primitive stops at once; the written helper chases the confirmed point.
        assert a.alien_target_source == "visible_handoff" and a.alien_target == (8.5, 5.5)
        assert (a.alien.x, a.alien.y) != (b.alien.x, b.alien.y)
        assert 0 in mask(e) and 20 in mask(e) and 21 not in mask(e)
        cancel = next(x for x in e.metrics.control.events if x["kind"] == "helper_cancel")
        assert cancel["reason"] == "confirmed_visible"
    finally:
        e.close()


def test_new_unseen_revision_interrupts_old_primitive_one_core_tick(env):
    env.step(21)
    before = env.core.snapshot()
    env.core._perception = replace(before.perception, revision=2)
    a = env.core.step(SurvivorCommand())
    assert a.tick == 16 and a.investigation.phase == "cancelled" and not a.navigation_path
    assert a.investigation.revision == 1 and a.perception.revision == 2
    assert mask(env) == [0, 21, 30]  # fresh public clue can START; old primitive is gone


def test_end_rotation_reacquisition_precedes_completion(env, monkeypatch):
    original = env.core.step

    def insert(command):
        s = env.core.snapshot()
        if s.tick == 40:
            env.core._survivor = SurvivorState(
                CLUE[0] + 3 * cos(s.alien.yaw), CLUE[1] + 3 * sin(s.alien.yaw)
            )
        return original(command)

    monkeypatch.setattr(env.core, "step", insert)
    for _ in range(4):
        env.step(21)
    assert env.core.snapshot().investigation.phase == "cancelled"
    assert env.core.snapshot().investigation.investigated_revision is None
    assert env.metrics.control.counts["completed_sweeps"] == 0
    event = next(e for e in env.metrics.control.events if e["kind"] == "helper_cancel")
    assert event["tick"] == 46


@pytest.mark.parametrize(
    "change",
    [
        "visible",
        "missing",
        "zero",
        "boolean",
        "nan",
        "investigated",
        "completed",
        "old_revision",
        "wrong_target",
    ],
)
def test_invalid_or_stale_active_clue_never_forces_continuation(env, change):
    env.step(21)
    v = env._allowed_view()
    inv = v.investigation
    if change == "visible":
        v = replace(v, visible=True)
    if change == "missing":
        v = replace(v, last_seen=None)
    if change == "zero":
        v = replace(v, clue_revision=0)
    if change == "boolean":
        v = replace(v, clue_revision=True)
    if change == "nan":
        v = replace(v, last_seen=(float("nan"), 5.5))
    if change == "investigated":
        v = replace(v, investigation=replace(inv, investigated_revision=1))
    if change == "completed":
        v = replace(v, investigation=replace(inv, phase="completed"))
    if change == "old_revision":
        v = replace(v, investigation=replace(inv, revision=9))
    if change == "wrong_target":
        v = replace(v, investigation=replace(inv, target=(10.5, 5.5)))
    assert not search_is_restricted(v)
    m = candidate_masks(build_candidates(v))
    assert m[0] and m[30] and m.any()
    assert m[HeuristicSelector(env.config.target_arrival_distance).choose(v)]
    if change not in ("old_revision", "wrong_target"):
        assert not m[21]


def test_public_hidden_world_and_adapter_mask_observation_parity():
    envs = [AlienEnv(arena(p), bot_kind="stationary_debug") for p in ((17.5, 17.5), (16.5, 17.5))]
    try:
        for e in envs:
            e.reset(seed=7)
            seed(e.core)
        for _ in range(7):
            views = [e._allowed_view() for e in envs]
            assert views[0] == views[1] and build_candidates(views[0]) == build_candidates(views[1])
            np.testing.assert_array_equal(build_observation(views[0]), build_observation(views[1]))
            app = object.__new__(HumanRLController)
            app.core = envs[0].core
            app.config = envs[0].config
            app.last_action = envs[0].last_action
            app.visit_ticks = envs[0].visit_ticks
            np.testing.assert_array_equal(app.observation(), build_observation(views[0]))
            np.testing.assert_array_equal(app.action_masks(), envs[0].action_masks())
            np.testing.assert_array_equal(app.observation()[-32:].astype(bool), app.action_masks())
            for e in envs:
                e.step(21)
    finally:
        for e in envs:
            e.close()


def test_terminal_freezes_and_audit_closes_then_reset_clears():
    objects = (MapObject("card", "keycard", 17.5, 17.5), MapObject("exit", "exit", 17.5, 17.5))
    e = AlienEnv(arena(objects=objects), bot_kind="stationary_debug")
    try:
        e.reset(seed=2)
        seed(e.core)
        e.bot = SimpleNamespace(
            command=lambda view: SurvivorCommand(interact=True),
            reset=lambda rng: None,
            label="fixture",
        )
        _, _, term, _, info = e.step(21)
        assert term and e.core.snapshot().outcome == "escaped" and e.core.snapshot().tick == 2
        audit = info["search_metrics"]["control"]
        assert (
            audit["attempts"][0]["end_reason"] == "terminal" and e.metrics.control.pending is None
        )
        frozen = e.core.snapshot()
        with pytest.raises(RuntimeError):
            e.step(21)
        assert e.core.step(SurvivorCommand()) == replace(frozen, events=())
        e.reset(seed=2)
        assert (
            e.core.snapshot().outcome is None
            and e.core.snapshot().investigation == InvestigationSnapshot()
        )
    finally:
        e.close()


def test_waypoint_arrival_after_cancel_does_not_scan(env):
    env.step(21)
    assert search_contract()["contract_version"] == VERSION
    env.step(30)
    for _ in range(5):
        env.step(0)
    s = env.core.snapshot()
    assert (
        (s.alien.x, s.alien.y) == (8.5, 5.5)
        and s.alien.yaw == 0
        and s.investigation.phase == "idle"
    )


def test_heuristic_and_uniform_valid_controls_share_public_binary_choice(env):
    env.step(21)
    v = env._allowed_view()
    valid = np.flatnonzero(candidate_masks(build_candidates(v)))
    assert valid.tolist() == [21, 30] and len(valid) == 2
    assert HeuristicSelector(env.config.target_arrival_distance).choose(v) == 21
    assert 30 in valid  # Wait cancels; no forced-only continuation


@pytest.mark.parametrize("key", ["search_control", "contract_version"])
def test_old_contract_names_rejected_before_actors(key, monkeypatch):
    monkeypatch.setattr("rl.env.GameCore", lambda *a, **k: pytest.fail("actor must not construct"))
    cfg = pipeline.read_json(PILOT_CONFIG)
    cfg[key] = "legacy" if key == "search_control" else "alien-vector-v2"
    with pytest.raises(ValueError, match=key):
        pipeline.make_env(cfg)


def prepared(tmp_path):
    cfg = pipeline.read_json(PILOT_CONFIG)
    run = tmp_path / "new_run"
    pipeline.prepare_run(run, cfg)
    return run, *pipeline.read_run(run)


def test_static_prepare_protocol_and_identity_no_actor(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "AlienEnv", lambda **kw: kw)
    run, cfg, manifest, source = prepared(tmp_path)
    identity = pipeline.read_json(run / "run_identity.json")
    assert identity["semantic_contract"] == search_contract()
    assert pipeline.make_env(cfg)["bot_kind"] == "evasive_mix"
    assert cfg["frozen_evaluation"]["protocol"]["search_control"] == SEARCH_CONTROL
    bad = deepcopy(identity)
    bad["semantic_contract"]["contract_version"] = "alien-vector-v2"
    pipeline.write_json(run / "run_identity.json", bad)
    with pytest.raises(ValueError, match="semantic"):
        pipeline.read_run(run)


@pytest.mark.parametrize("field", ["mode", "version"])
def test_protocol_mismatch_rejected(tmp_path, field):
    from rl.evaluation_protocol import validate_protocol

    cfg = pipeline.read_json(PILOT_CONFIG)
    p = pipeline.read_json(cfg["evaluation_protocol"])
    p["search_control" if field == "mode" else "contract_version"] = (
        "legacy" if field == "mode" else "alien-vector-v3-continue-cancel"
    )
    with pytest.raises(ValueError):
        validate_protocol(cfg, p)


@pytest.mark.parametrize("bad", ["contract", "mode", "source"])
def test_checkpoint_identity_fails_closed_before_real_load(tmp_path, monkeypatch, bad):
    _, cfg, _, source = prepared(tmp_path)
    model = SimpleNamespace(
        num_timesteps=0,
        _n_updates=0,
        save=lambda p: __import__("pathlib").Path(p).write_bytes(b"fake"),
    )
    monkeypatch.setattr(pipeline, "parameter_hash", lambda m: "fake-hash")
    path = tmp_path / "fake.zip"
    meta = pipeline.save_checkpoint(path, model, cfg, source, role="untrained")
    assert meta["search_control"] == SEARCH_CONTROL and meta["contract_version"] == VERSION
    if bad == "contract":
        meta["contract_version"] = "alien-vector-v3-continue-cancel"
    if bad == "mode":
        meta["search_control"] = "legacy"
    if bad == "source":
        meta["source"] = {"source_sha256": "old-runtime"}
    pipeline.write_json(path.with_suffix(".metadata.json"), meta)
    monkeypatch.setattr(
        pipeline.MaskablePPO, "load", lambda *a, **k: pytest.fail("real load forbidden")
    )
    with pytest.raises(ValueError):
        pipeline.load_checkpoint(path, cfg, source)


def test_metadata_mock_roundtrip_keeps_new_mode(tmp_path, monkeypatch):
    _, cfg, _, source = prepared(tmp_path)
    model = SimpleNamespace(
        num_timesteps=0,
        _n_updates=0,
        save=lambda p: __import__("pathlib").Path(p).write_bytes(b"fake"),
    )
    monkeypatch.setattr(pipeline, "parameter_hash", lambda m: "fake-hash")
    path = tmp_path / "fake.zip"
    pipeline.save_checkpoint(path, model, cfg, source, role="untrained")
    calls = []
    monkeypatch.setattr(pipeline.MaskablePPO, "load", lambda *a, **k: calls.append((a, k)) or model)
    loaded, meta = pipeline.load_checkpoint(path, cfg, source)
    assert loaded is model and len(calls) == 1 and meta["contract_version"] == VERSION


def test_versioned_control_metrics_summary_and_idempotent_finish(env):
    env.cap_ticks = 30
    env.step(21)
    _, _, _, done, info = env.step(30)
    assert done
    m = info["search_metrics"]
    assert m["version"] == "wall-search-metrics-v4-waypoint-commitment"
    assert m["control"]["version"] == "continue-cancel-audit-v2-pursuit-tracking"
    assert env.metrics.finish(env.core.snapshot()) == m
    summary = summarize_search([{"search_metrics": m}])
    assert summary["control"]["counts"]["policy_cancels"] == 1


def test_new_revision_helper_audit_at_tick16(env, monkeypatch):
    env.step(21)
    original = env.core.step

    def replace_public_revision(command):
        if env.core.snapshot().tick == 15:
            env.core._perception = replace(env.core.snapshot().perception, revision=2)
        return original(command)

    monkeypatch.setattr(env.core, "step", replace_public_revision)
    env.step(21)
    event = next(e for e in env.metrics.control.events if e["kind"] == "helper_cancel")
    assert event["tick"] == 16 and event["reason"] == "new_revision"
    assert env.metrics.control.counts["helper_cancels"] == 1
    assert mask(env) == [0, 21, 30]


def test_headless_loader_routes_mode_and_named_world_with_mocked_load(monkeypatch):
    from app.rl_controller import load_human_rl
    from game.map import scenario_map

    cfg = pipeline.read_json(PILOT_CONFIG)
    fake = SimpleNamespace(policy=SimpleNamespace(set_training_mode=lambda value: None))
    meta = {"role": "best_validation", "actual_steps": 123, "optimizer_update_epochs": 10}
    selection = {
        "best_checkpoint_sha256": "fake",
        "best_steps": 123,
        "best_optimizer_update_epochs": 10,
        "fresh_load_reproduced_actions_and_episodes": True,
    }
    monkeypatch.setattr(pipeline, "read_run", lambda path: (cfg, {}, {}))
    monkeypatch.setattr(pipeline, "load_checkpoint", lambda *a, **kw: (fake, meta))
    monkeypatch.setattr(pipeline, "read_json", lambda path: selection)
    monkeypatch.setattr(pipeline, "file_hash", lambda path: "fake")
    driver = load_human_rl("synthetic-run", "synthetic-checkpoint", 123)
    assert driver.core.world_map == scenario_map(cfg["scenario"])
    assert driver.observation().shape == (352,)


def test_evaluation_rows_serialize_search_audit(env, tmp_path, monkeypatch):
    from rl import evaluate as evaluation

    env.cap_ticks = 1
    _, _, _, _, info = env.step(30)
    fake = SimpleNamespace(
        config=env.config,
        action_masks=lambda: np.ones(32, dtype=bool),
        reset=lambda **kw: (np.zeros(352, np.float32), {}),
        step=lambda action: (np.zeros(352, np.float32), 0.0, False, True, info),
        close=lambda: None,
    )
    monkeypatch.setattr(evaluation, "make_env", lambda cfg: fake)
    cfg = {}
    case = {
        "id": "synthetic_0",
        "scenario_seed": 1,
        "bot_seed": 2,
        "reset_seed": 3,
        "selector_seed": 4,
    }
    _, rows = evaluation.evaluate(
        "wait",
        cfg,
        [case],
        tmp_path / "fake.csv",
        inference="constant_wait",
        episode_seeds=[5],
        repeat_index=0,
        controller_name="wait",
    )
    assert (
        rows[0]["search_metrics"]["control"]["version"]
        == "continue-cancel-audit-v2-pursuit-tracking"
    )
    assert "control" in (tmp_path / "fake.csv").read_text()
