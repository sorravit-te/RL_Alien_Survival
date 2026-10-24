"""Filtered threat memory, safe retreat, mission resumption and unchanged Alien boundary."""

from dataclasses import replace
from math import hypot, pi

import numpy as np
import pytest

from game.core import GameCore
from game.map import WorldMap, small_test_map
from game.state import AlienState, CoreConfig, SurvivorState
from rl.env import AlienEnv
from rl.evasive_survivor import (
    EvasionConfig,
    EvasiveExplorerBot,
    SurvivorThreatView,
    observe_threat,
)
from rl.observations import build_observation
from rl.pipeline import make_env, prepare_run, read_json
from rl.survivor_bots import ExplorerBot, SurvivorView, observe_survivor


def arena():
    return WorldMap(("." * 16,) * 12, ((5.5, 5.5),), alien_spawn=(7.5, 5.5, pi))


def bot(world=None):
    world = arena() if world is None else world
    result = EvasiveExplorerBot(world.rows, CoreConfig())
    result.reset(np.random.default_rng(4))
    return result


@pytest.mark.parametrize("kind", ["wall", "outside_fov", "outside_range"])
def test_hidden_alien_pairs_give_identical_views_and_commands(kind):
    world = arena()
    survivor = SurvivorState(3.5, 3.5, 0)
    if kind == "wall":
        world = WorldMap(tuple(".....#.........." for _ in range(12)), ((3.5, 3.5),))
        positions = ((8.5, 3.5), (10.5, 4.5))
    elif kind == "outside_fov":
        positions = ((1.5, 3.5), (1.5, 4.5))
    else:
        positions = ((13.5, 3.5), (14.5, 4.5))
    views = [
        observe_threat(world, survivor, False, AlienState(*p), EvasionConfig()) for p in positions
    ]
    assert views[0] == views[1] and views[0].visible_alien is None
    bots = [bot(world), bot(world)]
    assert bots[0].command(views[0]) == bots[1].command(views[1])


def test_memory_refreshes_only_when_visible_expires_at_limit_and_reset_clears():
    subject = bot()
    mission = SurvivorView(5.5, 5.5, 0, False, ())
    subject.command(SurvivorThreatView(mission, (7.5, 5.5)))
    assert subject.last_seen == (7.5, 5.5) and subject.last_seen_age_ticks == 0 and subject.fleeing
    for age in range(1, 60):
        subject.command(SurvivorThreatView(mission))
        assert subject.last_seen == (7.5, 5.5) and subject.last_seen_age_ticks == age
    command = subject.command(SurvivorThreatView(mission))
    assert subject.last_seen is None and subject.last_seen_age_ticks is None and not subject.fleeing
    assert not command.running and subject.escape_target is None
    subject.command(SurvivorThreatView(mission, (8.0, 5.5)))
    assert subject.last_seen == (8.0, 5.5)
    subject.reset(np.random.default_rng(4))
    assert subject.last_seen is None and not subject.fleeing and subject.tick == 0
    assert subject.escape_target is None and subject.navigator.path == ()


def test_hidden_true_position_does_not_refresh_existing_memory():
    world = arena()
    survivor = SurvivorState(5.5, 5.5, 0)
    subjects = [bot(), bot()]
    for subject in subjects:
        subject.command(SurvivorThreatView(observe_survivor(world, survivor, False), (7.5, 5.5)))
    for _ in range(12):
        views = [
            observe_threat(world, survivor, False, AlienState(*p), EvasionConfig())
            for p in ((1.5, 5.5), (1.5, 7.5))
        ]
        assert views[0] == views[1]
        assert subjects[0].command(views[0]) == subjects[1].command(views[1])
        assert subjects[0].last_seen == subjects[1].last_seen == (7.5, 5.5)
    assert subjects[0].last_seen_age_ticks == 12


def test_visible_stationary_threat_causes_immediate_retreat_with_core_commands():
    world = arena()
    core = GameCore(world, alien_control="external")
    subject = bot(world)
    previous = 2.0
    for _ in range(20):
        before = core.snapshot()
        view = observe_threat(
            world, before.survivor, before.has_keycard, before.alien, subject.settings
        )
        command = subject.command(view)
        assert command.running
        after = core.step(command)
        distance = hypot(after.survivor.x - after.alien.x, after.survivor.y - after.alien.y)
        assert distance > previous and after.outcome is None
        assert not world.overlaps(after.survivor.x, after.survivor.y, core.config.radius)
        previous = distance
    assert previous > 4.0


def test_paths_crossing_perceived_threat_rejected_and_nearby_outward_escape_allowed():
    subject = bot()
    assert not subject._safe_path((5.5, 5.5), ((9.5, 5.5),), (7.5, 5.5))
    assert subject._safe_path((5.5, 5.5), ((4.5, 5.5), (4.5, 2.5)), (7.5, 5.5))
    assert subject._safe_path((7.0, 5.5), ((6.0, 5.5),), (7.5, 5.5))
    assert not subject._safe_path((7.0, 5.5), ((8.0, 5.5),), (7.5, 5.5))


def test_reached_target_does_not_bypass_minimum_hold_and_expiry_overrides_hold():
    subject = bot()
    mission = SurvivorView(5.5, 5.5, 0, False, ())
    subject.command(SurvivorThreatView(mission, (7.5, 5.5)))
    target, plans = subject.escape_target, subject.replans
    assert target is not None
    arrived = replace(mission, x=target[0], y=target[1])
    subject.command(SurvivorThreatView(arrived))
    assert subject.fleeing and subject.escape_target == target and subject.replans == plans
    subject.last_seen_age_ticks = 59
    subject.command(SurvivorThreatView(arrived))
    assert subject.last_seen is None and not subject.fleeing


@pytest.mark.parametrize("seed", [0, 2])
def test_without_threat_new_bot_matches_explorer_mission_and_escapes(seed):
    world = small_test_map()
    core = GameCore(world, CoreConfig(alien_enabled=False))
    core.reset(seed)
    subject = EvasiveExplorerBot(world.rows, core.config)
    original = ExplorerBot(world.rows, core.config)
    for controller in (subject, original):
        controller.reset(np.random.default_rng(seed))
    for _ in range(5000):
        before = core.snapshot()
        mission = observe_survivor(world, before.survivor, before.has_keycard)
        command = subject.command(SurvivorThreatView(mission))
        assert command == original.command(mission)
        after = core.step(command)
        assert not world.overlaps(after.survivor.x, after.survivor.y, core.config.radius)
        if after.outcome:
            break
    assert after.outcome == "escaped"


def test_threat_memory_is_not_an_alien_policy_input_and_default_explorer_unchanged():
    original = AlienEnv()
    evasive = AlienEnv(bot_kind="evasive_explorer")
    try:
        a, _ = original.reset(seed=42)
        b, _ = evasive.reset(seed=42)
        assert np.array_equal(a, b) and np.array_equal(
            original.action_masks(), evasive.action_masks()
        )
        assert a.shape == (352,) and evasive.action_space.n == 32
        evasive.bot.last_seen = (99.0, 99.0)
        evasive.bot.last_seen_age_ticks = 12
        assert np.array_equal(b, build_observation(evasive._allowed_view()))
        assert type(original.bot) is ExplorerBot and original.bot_kind == "explorer"
    finally:
        original.close()
        evasive.close()


def test_new_bot_seeded_environment_replay_and_wall_safety():
    def replay():
        env = AlienEnv(bot_kind="evasive_explorer", max_episode_seconds=5)
        try:
            obs, _ = env.reset(seed=707)
            rng = np.random.default_rng(30)
            rows = []
            for _ in range(20):
                action = int(rng.choice(np.flatnonzero(env.action_masks())))
                obs, reward, terminated, truncated, info = env.step(action)
                snapshot = env.core.snapshot()
                assert not env.world.overlaps(
                    snapshot.survivor.x, snapshot.survivor.y, env.config.radius
                )
                rows.append((obs.tobytes(), reward, terminated, truncated, snapshot))
                if terminated or truncated:
                    break
            return rows
        finally:
            env.close()

    assert replay() == replay()


def test_unknown_bot_wrong_settings_and_unfrozen_config_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown bot"):
        AlienEnv(bot_kind="secret")
    with pytest.raises(ValueError, match="require evasive"):
        AlienEnv(evasion_settings=EvasionConfig())
    config = read_json("configs/evasive_training_pilot.json")
    env = make_env(config)
    assert env.bot_kind == "evasive_explorer" and env.bot.settings == EvasionConfig()
    env.close()
    from rl.evaluation_protocol import frozen_protocol

    with pytest.raises(ValueError, match="frozen protocol"):
        frozen_protocol(config)
    prepare_run(tmp_path / "prepared", config)
    assert (tmp_path / "prepared/evaluation_protocol.json").exists()
    config["evaluation_protocol"] = ""
    config["bot_kind"] = "secret"
    with pytest.raises(ValueError, match="supported mission bot"):
        prepare_run(tmp_path / "invalid", config)


def test_bot_rejects_world_truth_as_command_input():
    with pytest.raises(TypeError, match="filtered"):
        bot().command(GameCore(arena()).snapshot())


def test_sampled_protocol_validates_modes_seeds_and_keeps_legacy_explicit(tmp_path):
    from copy import deepcopy

    from tools.check_survivor_bots import prepare, validate_protocol

    training = read_json("configs/evasive_training_pilot.json")
    protocol = read_json("configs/sampled_policy_evaluation.json")
    validate_protocol(training, protocol)
    original = read_json("configs/training_pilot.json")
    assert (
        training["ppo"] == original["ppo"]
        and training["requested_steps"] == original["requested_steps"]
    )
    for mutate in ("primary", "repeat_seed", "controller_mode"):
        changed = deepcopy(protocol)
        if mutate == "primary":
            changed["primary_inference"] = "deterministic_argmax"
        elif mutate == "repeat_seed":
            changed["sampling_repeat_seeds"] = [1, 1, 2]
        else:
            changed["controllers"][1]["inference"] = "argmax_masked"
        with pytest.raises(ValueError):
            validate_protocol(training, changed)
    output = tmp_path / "protocol"
    prepare(output)
    assert read_json(output / "smoke_protocol.json")["episodes"] == 120
    assert len(read_json(output / "future_manifest.json")["splits"]["test"]) == 50
    assert (
        read_json(output / "evaluation_protocol.json")["status"]
        == "integrated_pending_runner_review"
    )
    with pytest.raises(FileExistsError):
        prepare(output)
    supported = {k: v for k, v in training.items() if k != "evaluation_protocol"}
    prepare_run(tmp_path / "supported_bot", supported)
    assert read_json(tmp_path / "supported_bot/config.json")["bot_kind"] == "evasive_explorer"


def test_retreat_around_obstacle_uses_clearance_and_core_collision():
    world = WorldMap(
        (
            "############",
            "#..........#",
            "#..........#",
            "#...#......#",
            "#...#......#",
            "#..........#",
            "#..........#",
            "############",
        ),
        ((5.5, 4.5),),
        alien_spawn=(7.5, 4.5, pi),
    )
    core = GameCore(world, alien_control="external")
    subject = bot(world)
    for _ in range(90):
        before = core.snapshot()
        command = subject.command(
            observe_threat(
                world, before.survivor, before.has_keycard, before.alien, subject.settings
            )
        )
        after = core.step(command)
        assert after.outcome is None
        assert not world.overlaps(after.survivor.x, after.survivor.y, core.config.radius)
    assert subject.flee_ticks > 0


def test_safe_corridor_detour_may_initially_approach_without_entering_threat_zone():
    world = small_test_map()
    subject = bot(world)
    subject.last_seen = (9.5, 7.5)
    target, path = subject._plan((10.5, 3.5))
    assert target is not None and path
    assert subject._safe_path((10.5, 3.5), path, subject.last_seen)
    assert not subject._safe_path((7.0, 5.5), ((8.0, 5.5),), (7.5, 5.5))
    # Reproduce the saved no-route state as a controlled fixture, not another smoke case.
    fixture = replace(world, spawn_points=((10.5, 3.5),))
    core = GameCore(fixture, alien_control="external")
    core._survivor = replace(core.snapshot().survivor, yaw=1.676148649762207)
    subject.reset(np.random.default_rng(4))
    subject.last_seen = (9.5, 7.5)
    subject.last_seen_age_ticks = 0
    subject.fleeing = True
    initial = core.snapshot().survivor
    for _ in range(60):
        before = core.snapshot()
        cmd = subject.command(
            observe_threat(
                fixture, before.survivor, before.has_keycard, before.alien, subject.settings
            )
        )
        after = core.step(cmd)
        assert after.outcome is None
        assert not fixture.overlaps(after.survivor.x, after.survivor.y, core.config.radius)
    assert hypot(after.survivor.x - initial.x, after.survivor.y - initial.y) > 1.0
    assert subject.no_safe_route_count == 0


def test_mission_navigation_cache_is_invalidated_after_retreat():
    subject = bot()
    mission = SurvivorView(5.5, 5.5, 0, False, ())
    subject.command(SurvivorThreatView(mission, (7.5, 5.5)))
    old_navigator = subject.explorer.navigator
    old_navigator.target = (12.5, 5.5)
    old_navigator.path = ((10.5, 5.5), (12.5, 5.5))
    subject.last_seen_age_ticks = 59
    subject.command(SurvivorThreatView(mission))
    assert not subject.fleeing
    assert subject.explorer.navigator is not old_navigator
    assert subject.explorer.navigator.target is None and subject.explorer.navigator.path == ()
