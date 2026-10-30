"""Confirmed-clue investigation contracts; fake policies and synthetic worlds only."""

import json
import random
from dataclasses import FrozenInstanceError, replace
from math import ceil, cos, hypot, pi, sin

import numpy as np
import pygame
import pytest
import torch
from test_human_rl import Policy
from test_recording import same_rng

from app.loop import FixedStepLoop
from app.play import PlaySession
from app.recorder import GameplayRecorder, read_records, read_tick_commands
from app.rl_controller import HumanRLController, rng_state
from game.core import GameCore
from game.investigation import InvestigationSnapshot
from game.map import MapObject, WorldMap
from game.perception import PerceptionSnapshot, update_perception
from game.state import AlienState, CoreConfig, SurvivorCommand, SurvivorState
from rl.candidates import AllowedAlienView, build_candidates, candidate_masks
from rl.contract import BASE_FIELDS, VERSION, WAIT
from rl.env import AlienEnv
from rl.observations import build_observation

CLUE = (5.5, 5.5)


def world(survivor=(17.5, 17.5), rows=None, objects=()):
    return WorldMap(
        rows or ("." * 20,) * 20,
        (survivor,),
        objects,
        alien_spawn=(*CLUE, 0),
        patrol_waypoints=((8.5, 5.5),),
    )


def seed_clue(core, target=CLUE, revision=1):
    # Explicit fixture of a prior CONFIRMED sample; no synthetic hidden truth lookup.
    core._perception = PerceptionSnapshot(
        last_seen=target, last_seen_age_ticks=300, revision=revision
    )


def make_core(*, scene=None, config=None, target=CLUE):
    core = GameCore(scene or world(), config, alien_control="external")
    core.reset(7799301)
    seed_clue(core, target)
    return core


def select(core):
    memory = core.snapshot().perception
    core.set_alien_target(memory.last_seen, source="last_seen", evidence_revision=memory.revision)


def advance(core, n):
    for _ in range(n):
        core.step(SurvivorCommand())
    return core.snapshot()


def view(core):
    s = core.snapshot()
    a, m = s.alien, s.perception
    return AllowedAlienView(
        (a.x, a.y),
        a.yaw,
        a.speed_mode == "chase",
        21,
        m.visible,
        m.last_seen,
        m.last_seen_age_ticks / 30 if m.last_seen_age_ticks is not None else None,
        s.world_map.patrol_waypoints,
        (None,) * len(s.world_map.patrol_waypoints),
        bool(s.navigation_path),
        m.revision,
        s.investigation,
    )


def test_complete_sweep_samples_final_orientation_before_exhausting_only_that_revision():
    core = make_core()
    select(core)
    for n in range(1, 46):
        if n % 15 == 1:
            select(core)  # Same-revision action21 must not restart.
        s = advance(core, 1)
        assert s.investigation.phase == "sweeping" and s.investigation.rotation_ticks == n
        assert s.investigation.investigated_revision is None
        assert s.investigation.progress == pytest.approx(n / 45)
        assert (s.alien.x, s.alien.y) == CLUE
        assert ((s.alien.yaw - n * 2 * pi / 45 + pi) % (2 * pi) - pi) == pytest.approx(0, abs=1e-12)
        assert s.perception.revision == 1 and s.perception.last_seen_age_ticks == 300 + n
        assert candidate_masks(build_candidates(view(core)))[21]
    assert s.investigation.progress == 1 and s.investigation.phase != "completed"
    s = advance(core, 1)
    assert s.tick == 46 and s.investigation.phase == "completed"
    assert s.investigation.investigated_revision == 1 and s.perception.last_seen == CLUE
    masks = candidate_masks(build_candidates(view(core)))
    assert not masks[21] and masks[0] and masks[WAIT] and masks.any()
    before = core.snapshot()
    with pytest.raises(ValueError, match="already been investigated"):
        select(core)
    assert core.snapshot() == before
    with pytest.raises(FrozenInstanceError):
        s.investigation.revision = 99


def test_walk_reaches_entire_path_then_sweeps_and_unreachable_clue_never_exhausts():
    target = (6.5, 5.5)
    core = make_core(target=target)
    select(core)
    for _ in range(30):
        before = core.snapshot()
        after = advance(core, 1)
        if after.investigation.phase == "sweeping":
            assert (
                hypot(after.alien.x - target[0], after.alien.y - target[1])
                <= core.config.target_arrival_distance
            )
            assert after.navigation_path == ()
            assert (before.alien.x, before.alien.y) == (after.alien.x, after.alien.y)
            break
        assert (
            hypot(after.alien.x - before.alien.x, after.alien.y - before.alien.y)
            <= core.config.alien_search_speed / 30 + 1e-12
        )
    else:
        pytest.fail("never reached clue")
    blocked = make_core(
        scene=world(rows=tuple(".......#............" for _ in range(20))), target=(10.5, 5.5)
    )
    select(blocked)
    s = advance(blocked, 100)
    assert s.navigation_path == () and s.investigation.phase == "travelling"
    assert s.investigation.rotation_ticks == 0 and s.investigation.investigated_revision is None


@pytest.mark.parametrize(
    "source,target", [("wait", None), ("waypoint", (8.5, 5.5)), ("visible", (8.5, 5.5))]
)
def test_other_action_discards_partial_sweep_and_reselect_starts_at_zero(source, target):
    core = make_core()
    select(core)
    advance(core, 37)
    core.set_alien_target(target, source=source)
    s = core.snapshot()
    assert s.investigation == InvestigationSnapshot()
    select(core)
    assert core.snapshot().investigation.rotation_ticks == 0
    assert advance(core, 1).investigation.rotation_ticks == 1


@pytest.mark.parametrize("rate", [2 * pi / 3, pi / 2, 2.0])
def test_rate_config_caps_last_rotation_and_waits_for_tail_sensor(rate):
    core = make_core(config=CoreConfig(investigation_turn_speed=rate))
    select(core)
    total = ceil(2 * pi / (rate / 30) - 1e-12)
    s = advance(core, total)
    assert s.investigation.rotation_ticks == total and s.investigation.phase == "sweeping"
    assert ((s.alien.yaw + pi) % (2 * pi) - pi) == pytest.approx(0, abs=1e-12)
    assert advance(core, 1).investigation.phase == "completed"


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_turn_speed_rejected(bad):
    with pytest.raises(ValueError):
        CoreConfig(investigation_turn_speed=bad)


def test_revision_is_generated_only_by_confirmed_samples_and_same_coordinate_is_fresh():
    scene = world()
    config = CoreConfig()
    alien = AlienState(5.5, 5.5, 0)
    survivor = SurvivorState(8.5, 5.5)
    memory = PerceptionSnapshot()
    for _ in range(5):
        memory = update_perception(scene, alien, survivor, memory, config)
        assert memory.revision is None
    memory = update_perception(scene, alien, survivor, memory, config)
    assert memory.visible and memory.revision == 1
    for _ in range(10):
        memory = update_perception(scene, alien, SurvivorState(17.5, 17.5), memory, config)
        assert not memory.visible and memory.revision == 1
    for _ in range(6):
        memory = update_perception(scene, alien, survivor, memory, config)
    assert memory.visible and memory.revision == 2 and memory.last_seen == (8.5, 5.5)


def test_fresh_same_position_reenables_exhausted_clue_after_actual_confirmed_sensing():
    core = make_core()
    select(core)
    advance(core, 46)
    assert not candidate_masks(build_candidates(view(core)))[21]
    core.set_alien_target((8.5, 5.5), source="waypoint")
    advance(core, 45)
    core._alien = replace(core.snapshot().alien, yaw=pi)
    core._survivor = SurvivorState(*CLUE)
    s = advance(core, 6)
    assert s.perception.visible and s.perception.revision == 2 and s.perception.last_seen == CLUE
    assert s.investigation.investigated_revision == 1
    assert not candidate_masks(build_candidates(view(core)))[21]  # visible: chase, not search
    core._survivor = SurvivorState(17.5, 17.5)
    advance(core, 1)
    assert candidate_masks(build_candidates(view(core)))[21]
    select(core)
    assert (
        core.snapshot().investigation.revision == 2
        and core.snapshot().investigation.rotation_ticks == 0
    )


@pytest.mark.parametrize("phase", ["travelling", "sweeping"])
def test_reacquisition_cancels_immediately_and_hands_off_to_helper_chase(phase):
    scene = world(survivor=(8.5, 5.5) if phase == "travelling" else (5.5, 8.5))
    core = make_core(scene=scene, target=(10.5, 5.5) if phase == "travelling" else CLUE)
    select(core)
    for _ in range(40):
        before = core.snapshot()
        s = advance(core, 1)
        if s.perception.visible:
            assert before.investigation.phase == phase
            assert s.investigation.phase == "cancelled" and s.investigation.revision == 1
            assert s.perception.revision == 2
            assert s.investigation.investigated_revision is None
            assert s.alien_target == s.perception.last_seen
            assert s.alien_target_source == "visible_handoff"
            assert s.navigation_path and (s.alien.x, s.alien.y) != (before.alien.x, before.alien.y)
            break
    else:
        pytest.fail("never reacquired")
    later = advance(core, 3)
    assert later.investigation.revision == 1 and later.perception.revision > 2
    assert later.alien_target_source == "visible_handoff"
    assert (later.alien.x, later.alien.y) != (s.alien.x, s.alien.y)
    with pytest.raises(ValueError, match="current confirmed"):
        core.set_alien_target(s.alien_target, source="last_seen", evidence_revision=1)


@pytest.mark.parametrize("insertion_tick", [20, 40])
def test_confirmation_before_end_or_on_final_sample_wins_completion(insertion_tick):
    core = make_core()
    select(core)
    advance(core, insertion_tick)
    angle = core.snapshot().alien.yaw
    # At40 ticks yaw320deg; a point at that bearing remains within FOV through final sample46.
    core._survivor = SurvivorState(CLUE[0] + 3 * cos(angle), CLUE[1] + 3 * sin(angle))
    for i in range(6):
        s = advance(core, 1)
        assert s.investigation.investigated_revision is None
        if i < 5:
            assert not s.perception.visible
    assert s.perception.visible and s.investigation.phase == "cancelled"
    assert s.tick == insertion_tick + 6
    if insertion_tick == 40:
        assert s.tick == 46  # Last sample, not an earlier shortcut/hidden-truth check.


def test_sensor_observes_pre_rotation_fov_and_wall_prevents_reacquisition():
    core = make_core(scene=world(survivor=(5.5, 8.5)))
    select(core)
    for _ in range(4):
        s = advance(core, 1)
        assert s.perception.confirmation_ticks == 0
    s = advance(core, 1)
    assert s.perception.confirmation_ticks == 1 and not s.perception.visible
    assert s.investigation.phase == "sweeping" and s.perception.revision == 1
    wall = make_core(
        scene=world(survivor=(8.5, 5.5), rows=tuple(".......#............" for _ in range(20)))
    )
    select(wall)
    s = advance(wall, 46)
    assert not s.perception.visible and s.perception.revision == 1
    assert s.investigation.phase == "completed"  # No hidden-coordinate shortcut through wall.


def test_paired_hidden_worlds_have_identical_full_observations_masks_and_helper_states():
    cores = [make_core(scene=world(survivor=p)) for p in ((17.5, 17.5), (16.5, 17.5))]
    for core in cores:
        select(core)
    for _ in range(95):
        for core in cores:
            advance(core, 1)
        assert view(cores[0]) == view(cores[1])
        np.testing.assert_array_equal(
            build_observation(view(cores[0])), build_observation(view(cores[1]))
        )
        np.testing.assert_array_equal(
            candidate_masks(build_candidates(view(cores[0]))),
            candidate_masks(build_candidates(view(cores[1]))),
        )
        assert candidate_masks(build_candidates(view(cores[0]))).any()


def test_observation_mapping_exposes_phase_target_progress_current_revision_and_exhaustion():
    core = make_core()
    select(core)
    advance(core, 22)
    values = dict(zip(BASE_FIELDS, build_observation(view(core))[:32]))
    assert VERSION == "alien-vector-v6-scenario-randomization" and len(values) == 32
    assert values["investigation_active"] == values["investigation_sweeping"] == 1
    assert values["investigation_progress"] == pytest.approx(22 / 45)
    assert values["investigation_target_valid"] == 1
    assert values["investigation_revision_current"] == 1 and values["last_seen_investigated"] == 0
    advance(core, 24)
    values = dict(zip(BASE_FIELDS, build_observation(view(core))[:32]))
    assert values["investigation_complete"] == values["last_seen_investigated"] == 1
    assert values["investigation_active"] == 0 and values["investigation_progress"] == 1


def test_terminal_and_reset_freeze_and_clear_all_evidence_investigation_state():
    objects = (MapObject("card", "keycard", 17.5, 17.5), MapObject("exit", "exit", 17.5, 17.5))
    core = make_core(scene=world(objects=objects))
    select(core)
    core.step(SurvivorCommand(interact=True))
    s = core.step(SurvivorCommand(interact=True))
    assert s.outcome == "escaped" and s.investigation.rotation_ticks == 2
    for _ in range(10):
        assert core.step(SurvivorCommand()) == replace(s, events=())
    with pytest.raises(RuntimeError, match="terminal"):
        core.set_alien_target(None, source="wait")
    s = core.reset(7799301)
    assert s.investigation == InvestigationSnapshot() and s.perception == PerceptionSnapshot()
    assert s.alien.yaw == 0 and s.tick == 0 and s.outcome is None


class InvestigatingPolicy(Policy):
    def predict(self, obs, *, action_masks, deterministic):
        # Consume private streams to detect accidental logger/primitive RNG draws.
        random.random()
        np.random.random()
        torch.rand(())
        self.action = 21 if action_masks[21] else 0
        return super().predict(obs, action_masks=action_masks, deterministic=deterministic)


def app_driver(scene=None, policy=None):
    app = HumanRLController(
        GameCore(scene or world(), alien_control="external"), policy or InvestigatingPolicy(), 991
    )
    app.reset(7799301)
    seed_clue(app.core)
    app.checkpoint_identity = {"synthetic_fixture": True, "contract_version": VERSION}
    return app


def test_recorder_on_off_parity_full_primitive_rng_truth_separation_and_command_reconstruction(
    tmp_path, monkeypatch
):
    import app.recorder as recorder

    root = tmp_path / "experiments"
    root.mkdir()
    monkeypatch.setattr(recorder, "EXPERIMENTS_ROOT", root)
    apps = app_driver(), app_driver()
    rec = GameplayRecorder(root / "trace", apps[1])
    apps[1].recorder = rec
    commands = [SurvivorCommand() for _ in range(120)]
    for command in commands:
        before = rng_state()
        assert apps[0].step(command) == apps[1].step(command)
        same_rng(before, rng_state())
        same_rng(apps[0].sampler.state, apps[1].sampler.state)
        np.testing.assert_array_equal(apps[0].observation(), apps[1].observation())
        np.testing.assert_array_equal(apps[0].action_masks(), apps[1].action_masks())
    rec.close(completed=True, reason="synthetic_end")
    records = list(read_records(rec.directory))
    ticks = [r for r in records if r["type"] == "tick"]
    assert ticks[44]["investigation_after"]["phase"] == "sweeping"
    assert ticks[45]["investigation_after"]["phase"] == "completed"
    decisions = [r for r in records if r["type"] == "decision"]
    assert [r["tick"] for r in decisions] == list(range(0, 120, 15))
    assert [r["action"] for r in decisions] == [21] * 4 + [0] * 4
    assert apps[0].decisions == apps[1].decisions == 8
    assert all("diagnostic_truth_only" not in r["allowed_view"] for r in decisions)
    assert all("survivor_after" not in r["allowed_view"] for r in decisions)
    assert all(r["allowed_view"]["clue_revision"] == 1 for r in decisions)
    assert ticks[45]["sensor_before_movement"]["sample_tick_start"] == 45
    assert "survivor_after" in ticks[45]["diagnostic_truth_only"]
    assert (
        json.loads((rec.directory / "manifest.json").read_text())["format"]
        == "human-rl-command-recording-v2"
    )
    assert read_tick_commands(rec.directory, 0) == commands
    replay = app_driver()
    for command in read_tick_commands(rec.directory, 0):
        replay.step(command)
    assert replay.snapshot() == apps[0].snapshot()


@pytest.mark.parametrize("scenario", ["complete", "interrupt", "reacquire"])
def test_actual_env_human_adapter_full_obs_mask_and_snapshot_parity_at_decisions_and_every_posttick(
    scenario,
):
    scene = world(survivor=(5.5, 8.5)) if scenario == "reacquire" else world()
    env = AlienEnv(scene, bot_kind="stationary_debug", max_episode_seconds=6)
    env.reset(seed=1, options={"scenario_seed": 7799301, "bot_seed": 7799302})
    seed_clue(env.core)
    policy = Policy()
    app = app_driver(scene, policy)
    states = []
    previews = []
    original = env.core.step

    class Human:
        label = "synthetic_human_input"

        def command(self, _):
            v = env._allowed_view()
            previews.append((v, build_observation(v), env.action_masks().copy()))
            return SurvivorCommand()

    def step(command):
        s = original(command)
        states.append(s)
        return s

    env.core.step = step
    env.bot = Human()
    try:
        for block in range(12):
            assert app.allowed_view() == env._allowed_view()
            np.testing.assert_array_equal(app.observation(), build_observation(env._allowed_view()))
            np.testing.assert_array_equal(app.action_masks(), env.action_masks())
            masks = env.action_masks()
            action = (
                WAIT
                if scenario == "interrupt" and block == 2
                else 20
                if scenario == "reacquire" and masks[20]
                else 21
                if masks[21]
                else 0
            )
            policy.action = action
            offset = len(states)
            _, _, terminal, truncated, _ = env.step(action)
            for index in range(offset, len(states)):
                assert app.step(SurvivorCommand()) == states[index]
                v, o, m = (
                    previews[index + 1]
                    if index + 1 < len(states)
                    else (
                        env._allowed_view(),
                        build_observation(env._allowed_view()),
                        env.action_masks(),
                    )
                )
                assert app.allowed_view() == v
                np.testing.assert_array_equal(app.observation(), o)
                np.testing.assert_array_equal(app.action_masks(), m)
            assert app.snapshot() == env.core.snapshot()
            if terminal or truncated:
                break
        assert app.decisions == len(policy.calls)
    finally:
        env.close()


@pytest.mark.parametrize("fps", [30, 60, 144])
def test_clock_independent_sweep_completion_at_tick46(fps):
    app = app_driver()
    loop = FixedStepLoop(app)
    for _ in range(fps * 4):
        loop.advance(1 / fps)
    assert loop.current.tick == 120 and app.decisions == 8
    assert loop.current.investigation.investigated_revision == 1
    assert loop.current.alien_target_source == "waypoint"  # Chosen by fake policy at60, not helper.


def test_pause_focus_resume_and_reset_freeze_sweep_no_time_or_rng_and_clear_input(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.display.init()
    pygame.display.set_mode((320, 180))
    try:
        app = app_driver()
        loop = FixedStepLoop(app)
        session = PlaySession(loop)
        session.focused = True
        esc = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)
        session.events([esc])
        for _ in range(37):
            loop.advance(loop.dt)
        before = app.snapshot()
        state = app.sampler.state
        loop.inputs.key("e", True)
        loop.inputs.mouse(100, -100)
        session.events([pygame.event.Event(pygame.WINDOWFOCUSLOST)])
        loop.advance(100)
        session.events([pygame.event.Event(pygame.WINDOWFOCUSGAINED)])
        assert loop.paused and app.snapshot() == before
        same_rng(state, app.sampler.state)
        assert loop.inputs.consume(loop.dt) == SurvivorCommand()
        session.events([esc])
        loop.advance(loop.dt)
        assert app.snapshot().investigation.rotation_ticks == 38
        session.events([esc])
        loop.advance(100)
        assert app.snapshot().investigation.rotation_ticks == 38
        session.events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)])
        assert app.snapshot().investigation == InvestigationSnapshot()
        assert app.snapshot().perception == PerceptionSnapshot() and app.phase == app.decisions == 0
        assert loop.display_pitch() == 0 and loop.inputs.consume(loop.dt) == SurvivorCommand()
    finally:
        pygame.quit()


@pytest.mark.parametrize("revision", [None, 0, True, 2])
def test_core_rejects_unconfirmed_or_mismatched_revision_without_mutation(revision):
    core = make_core()
    before = core.snapshot()
    with pytest.raises(ValueError, match="current confirmed"):
        core.set_alien_target(CLUE, source="last_seen", evidence_revision=revision)
    assert core.snapshot() == before


def test_heuristic_uses_same_sweep_then_explicit_selector_patrol_and_fresh_revision_rules():
    from rl.evaluate import HeuristicSelector

    core = GameCore(world(), alien_control="heuristic")
    seed_clue(core)
    advance(core, 45)
    assert core.snapshot().investigation.phase == "sweeping"
    advance(core, 1)
    assert core.snapshot().investigation.phase == "completed"
    advance(core, 14)
    assert core.snapshot().alien_target_source == "last_seen"
    advance(core, 1)
    assert core.snapshot().alien_target_source == "patrol"
    selector = HeuristicSelector(core.config.target_arrival_distance)
    exhausted = view(core)
    assert selector.choose(exhausted) == 0
    assert selector.choose(replace(exhausted, clue_revision=2)) == 21


def test_env_exhausted_action21_is_masked_invalid_applies_wait_without_penalty():
    env = AlienEnv(world(), bot_kind="stationary_debug")
    env.reset(seed=1, options={"scenario_seed": 7799301, "bot_seed": 7799302})
    seed_clue(env.core)
    try:
        for _ in range(4):
            env.step(21)
        assert env.core.snapshot().investigation.phase == "completed" and not env.action_masks()[21]
        before = env.core.snapshot().alien
        _, reward, terminal, truncated, info = env.step(21)
        assert info["invalid_action"] and info["applied_action"] == WAIT
        assert reward == 0 and not terminal and not truncated
        assert env.core.snapshot().alien == before
        assert env.core.snapshot().investigation.investigated_revision == 1
        assert env.core.snapshot().perception.last_seen == CLUE
    finally:
        env.close()


def test_adapter_terminal_freezes_primitive_policy_clock_and_reset_clears_everything():
    objects = (MapObject("card", "keycard", 17.5, 17.5), MapObject("exit", "exit", 17.5, 17.5))
    app = app_driver(world(objects=objects))
    app.step(SurvivorCommand(interact=True))
    s = app.step(SurvivorCommand(interact=True))
    state = app.sampler.state
    assert s.outcome == "escaped" and s.investigation.rotation_ticks == 2
    for _ in range(20):
        assert app.step(SurvivorCommand()) == replace(s, events=())
    same_rng(state, app.sampler.state)
    assert app.decisions == len(app.model.calls) == 1 and app.phase == 2
    app.reset(7799301)
    assert app.snapshot().investigation == InvestigationSnapshot()
    assert app.snapshot().perception == PerceptionSnapshot()
    assert app.decisions == app.phase == 0
