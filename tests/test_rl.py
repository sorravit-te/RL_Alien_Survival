"""Policy boundary, external targets, bot knowledge and Gym API regressions."""

import subprocess
import sys
from dataclasses import replace
from math import hypot, pi, radians

import numpy as np
import pytest

from game.core import GameCore
from game.map import MapObject, WorldMap, small_test_map
from game.perception import PerceptionSnapshot
from game.state import CoreConfig, GameEvent, SurvivorCommand
from rl.candidates import AllowedAlienView, build_candidates, candidate_masks
from rl.contract import BASE_FIELDS, CANDIDATE_FIELDS, OBSERVATION_SIZE, VERSION, WAIT
from rl.env import AlienEnv
from rl.observations import build_observation
from rl.rewards import RewardTracker
from rl.survivor_bots import ExplorerBot, observe_survivor


def arena(objects=(), survivor=(2.5, 2.5), alien=(10.5, 10.5, 0), waypoints=()):
    return WorldMap(("." * 16,) * 16, (survivor,), objects, alien, waypoints)


def test_external_targets_never_call_heuristic_and_wait_clears_old_path(monkeypatch):
    import game.core as module

    def forbidden(*args):
        pytest.fail("Heuristic was called in external mode")

    monkeypatch.setattr(module, "choose_target", forbidden)
    world = arena(waypoints=((12.5, 10.5), (10.5, 12.5)))
    core = GameCore(world, alien_control="external")
    core.set_alien_target((12.5, 10.5), source="waypoint")
    for _ in range(31):
        snapshot = core.step(SurvivorCommand())
        assert snapshot.alien_target == (12.5, 10.5)
        assert snapshot.alien_target_source == "waypoint"
    before = snapshot.alien
    core.set_alien_target(None, source="wait")
    for _ in range(31):
        snapshot = core.step(SurvivorCommand())
        assert snapshot.alien == before
        assert snapshot.navigation_path == ()
        assert snapshot.alien_target is None
    with pytest.raises(RuntimeError):
        GameCore(world).set_alien_target(None)


def test_multiple_valid_actions_change_paths_and_actual_movement():
    world = arena(waypoints=((12.5, 10.5), (10.5, 12.5)))
    first, second = (
        AlienEnv(world, bot_kind="stationary_debug"),
        AlienEnv(world, bot_kind="stationary_debug"),
    )
    first.reset(seed=4)
    second.reset(seed=4)
    assert first.action_masks()[0] and first.action_masks()[1]
    first.step(0)
    second.step(1)
    a, b = first.core.snapshot(), second.core.snapshot()
    assert a.alien_target != b.alien_target and a.navigation_path != b.navigation_path
    assert (a.alien.x, a.alien.y) != (b.alien.x, b.alien.y)
    first.step(WAIT)
    assert first.core.snapshot().navigation_path == ()


def test_fixed_action_contract_zero_missing_and_float32_observation():
    env = AlienEnv(max_episode_seconds=1)
    observation, _ = env.reset(seed=3)
    assert observation.shape == (OBSERVATION_SIZE,) == (352,)
    assert observation.dtype == np.float32 and env.observation_space.contains(observation)
    masks = env.action_masks()
    assert masks.dtype == np.bool_ and masks.shape == (32,) and masks[WAIT]
    assert np.flatnonzero(masks).tolist() == [0, 1, 2, 3, 4, 5, 30]
    offset = len(BASE_FIELDS)
    for index in list(range(6, 30)) + [31]:
        if index in (20, 21):
            assert not masks[index]
        width = len(CANDIDATE_FIELDS)
        assert np.all(observation[offset + index * width : offset + (index + 1) * width] == 0)
    assert np.array_equal(observation[-32:].astype(bool), masks)


def test_hidden_world_pair_yields_same_obs_masks_candidates_and_target():
    world = arena(survivor=(2.5, 2.5), alien=(10.5, 10.5, 0), waypoints=((12.5, 10.5),))
    changed = replace(
        world,
        spawn_points=((3.5, 2.5),),
        objects=(
            MapObject("hidden_card", "keycard", 2.5, 3.5),
            MapObject("hidden_exit", "exit", 2.5, 4.5),
        ),
    )
    first, second = (
        AlienEnv(world, bot_kind="stationary_debug"),
        AlienEnv(changed, bot_kind="stationary_debug"),
    )
    a, _ = first.reset(seed=7)
    b, _ = second.reset(seed=7)
    assert np.array_equal(a, b)
    assert first._allowed_view() == second._allowed_view()
    assert build_candidates(first._allowed_view()) == build_candidates(second._allowed_view())
    assert np.array_equal(first.action_masks(), second.action_masks())
    first.step(0)
    second.step(0)
    assert first.core.snapshot().alien_target == second.core.snapshot().alien_target
    assert np.array_equal(
        build_observation(first._allowed_view()), build_observation(second._allowed_view())
    )


def test_preconfirmation_count_is_not_policy_knowledge():
    view = AllowedAlienView(
        (5.5, 5.5), 0, False, WAIT, False, None, None, ((7.5, 5.5),), (None,), False
    )
    a, b = PerceptionSnapshot(False, 0), PerceptionSnapshot(False, 5)
    # Boundary extracts only confirmed fields, not confirmation_ticks.
    env = AlienEnv(arena())
    env.reset(seed=1)
    env.core._perception = a
    first = env._allowed_view()
    env.core._perception = b
    second = env._allowed_view()
    assert first == second
    assert np.array_equal(build_observation(first), build_observation(second))
    assert not candidate_masks(build_candidates(view))[20]


def test_confirmed_and_last_seen_candidates_are_snapshots_with_valid_flags():
    view = AllowedAlienView((5.5, 5.5), 0, True, 20, True, (7.5, 5.5), 0, (), (), True)
    candidates = build_candidates(view)
    assert candidates[20].valid and candidates[21].valid
    hidden = replace(view, visible=False, last_seen_age_seconds=2)
    updated = build_candidates(hidden)
    assert not updated[20].valid and updated[21].target == (7.5, 5.5)
    assert updated[21].clue_age == 2
    assert build_observation(hidden).shape == (352,)


def test_masked_action_becomes_wait_without_penalty_and_bad_space_raises():
    env = AlienEnv(arena(), bot_kind="stationary_debug", max_episode_seconds=1)
    env.reset(seed=1)
    before = env.core.snapshot().alien
    _, reward, terminated, truncated, info = env.step(24)
    assert reward == 0 and not terminated and not truncated
    assert info["invalid_action"] and info["applied_action"] == WAIT
    assert env.core.snapshot().alien == before
    assert env.core.snapshot().alien_target is None
    with pytest.raises(ValueError):
        env.step(32)


def test_reward_tracker_once_no_time_or_proximity_components():
    tracker = RewardTracker()
    assert sum(tracker.collect(True, ()).values()) == 0.2
    assert sum(tracker.collect(True, ()).values()) == 0
    event = GameEvent("captured", "alien", 10)
    assert sum(tracker.collect(False, (event,)).values()) == 10
    assert sum(tracker.collect(True, (event,)).values()) == 0
    assert sum(RewardTracker().collect(False, (GameEvent("escaped", "exit", 1),)).values()) == -10


def test_first_sight_reward_is_retained_if_sight_is_lost_before_decision_end():
    world = replace(
        small_test_map(),
        spawn_points=((5.5, 2.5),),
        alien_spawn=(2.5, 2.5, 0),
        patrol_waypoints=((2.5, 2.5),),
    )
    env = AlienEnv(world, CoreConfig(run_speed=60), max_episode_seconds=2)
    env.reset(seed=4)

    class ScriptedSensorFixture:
        label = "scripted_sensor_fixture"
        index = 0

        def command(self, view):
            index = self.index
            self.index += 1
            return (
                SurvivorCommand(strafe=1, running=True)
                if index in (5, 8)
                else SurvivorCommand(forward=1, running=True)
                if index in (6, 7)
                else SurvivorCommand()
            )

    env.bot = ScriptedSensorFixture()
    _, reward, terminated, truncated, info = env.step(WAIT)
    assert reward == 0.2 and info["reward_components"]["first_sight"] == 0.2
    assert info["step_ticks"] == 15 and not terminated and not truncated
    assert not env.core.snapshot().perception.visible
    assert env.step(WAIT)[1] == 0


def test_capture_reward_first_sight_and_early_termination():
    env = AlienEnv(
        arena(
            objects=(MapObject("card", "keycard", 6.5, 2.5),),
            survivor=(2.5, 2.5),
            alien=(5.5, 2.5, pi),
        ),
        max_episode_seconds=5,
    )
    env.reset(seed=1)
    total = 0
    components = {"capture": 0.0, "escape": 0.0, "first_sight": 0.0}
    for _ in range(10):
        _, reward, terminal, truncated, info = env.step(WAIT)
        total += reward
        for key, value in info["reward_components"].items():
            components[key] += value
        if terminal or truncated:
            break
    assert terminal and not truncated and info["outcome"] == "captured"
    assert total == pytest.approx(10.2)
    assert components == {"capture": 10.0, "escape": 0.0, "first_sight": 0.2}
    with pytest.raises(RuntimeError):
        env.step(WAIT)


def test_escape_reward_early_terminal_and_terminal_priority_at_cap():
    world = arena(objects=(MapObject("a", "keycard", 3.5, 2.5), MapObject("z", "exit", 3.5, 2.5)))
    env = AlienEnv(world, max_episode_seconds=2 / 30)
    env.reset(seed=3)
    _, reward, terminal, truncated, info = env.step(WAIT)
    assert terminal and not truncated and reward == -10
    assert info["step_ticks"] == info["elapsed_ticks"] == 2
    assert info["outcome"] == "escaped" and info["reward_components"]["escape"] == -10


def test_cap_unresolved_partial_step_and_no_countdown_in_observation():
    short = AlienEnv(arena(), bot_kind="stationary_debug", max_episode_seconds=0.5)
    long = AlienEnv(arena(), bot_kind="stationary_debug", max_episode_seconds=0.6)
    short.reset(seed=8)
    long.reset(seed=8)
    a, reward, terminal, truncated, info = short.step(WAIT)
    b, _, _, long_truncated, _ = long.step(WAIT)
    assert np.array_equal(a, b) and not long_truncated
    assert reward == 0 and not terminal and truncated and info["outcome"] == "unresolved"
    assert short.core.snapshot().outcome is None
    _, reward, terminal, truncated, info = long.step(WAIT)
    assert info["step_ticks"] == 3 and info["elapsed_ticks"] == 18
    assert reward == 0 and not terminal and truncated


def test_seeded_reset_step_replay_and_separate_bot_scenario_rngs():
    env = AlienEnv(max_episode_seconds=2)

    def replay():
        obs, initial = env.reset(seed=42)
        records = [obs.copy()]
        for action in (0, 30, 2, 24):
            obs, reward, terminal, truncated, info = env.step(action)
            records.append((obs.copy(), reward, terminal, truncated, info["elapsed_ticks"]))
            if terminal or truncated:
                break
        return initial, records

    a, first = replay()
    b, second = replay()
    assert a["contract_version"] == VERSION
    assert a["scenario_seed"] == b["scenario_seed"] and a["bot_seed"] == b["bot_seed"]
    np.testing.assert_array_equal(first[0], second[0])
    for x, y in zip(first[1:], second[1:]):
        np.testing.assert_array_equal(x[0], y[0])
        assert x[1:] == y[1:]
    env.reset(seed=42, options={"bot_seed": 1})
    spawn = env.core.snapshot().survivor
    env.reset(seed=42, options={"bot_seed": 999})
    assert env.core.snapshot().survivor == spawn
    env.reset(seed=42, options={"scenario_seed": 1})
    bot_seed = env.bot_seed
    env.reset(seed=42, options={"scenario_seed": 2})
    assert env.bot_seed == bot_seed
    assert not env.reward_tracker.seen_paid and not env.reward_tracker.terminal_paid
    assert (
        env.elapsed_ticks == 0
        and env.last_action == WAIT
        and env.core.snapshot().alien_target is None
    )


def test_bot_hidden_object_pair_has_same_command_and_no_known_object():
    first = small_test_map()
    second = replace(first, objects=(MapObject("changed", "keycard", 5.5, 4.5),))
    state = GameCore(first, CoreConfig(alien_enabled=False)).snapshot().survivor
    a, b = observe_survivor(first, state, False), observe_survivor(second, state, False)
    assert a == b and not a.visible_objects
    bots = [ExplorerBot(world.rows, CoreConfig()) for world in (first, second)]
    for bot in bots:
        bot.reset(np.random.default_rng(8))
    assert bots[0].command(a) == bots[1].command(b)
    assert bots[0].known == bots[1].known == {}
    assert bots[0].geometry.objects == ()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_explorer_moves_and_escapes_without_danger_using_human_rules(seed):
    world = small_test_map()
    core = GameCore(world, CoreConfig(alien_enabled=False))
    core.reset(seed)
    bot = ExplorerBot(world.rows, core.config)
    bot.reset(np.random.default_rng(seed))
    moved = False
    for _ in range(6000):
        before = core.snapshot()
        command = bot.command(observe_survivor(world, before.survivor, before.has_keycard))
        assert abs(command.yaw_delta) <= radians(120) / 30 + 1e-12
        snapshot = core.step(command)
        displacement = hypot(
            snapshot.survivor.x - before.survivor.x, snapshot.survivor.y - before.survivor.y
        )
        assert displacement <= core.config.walk_speed / 30 + 1e-9
        moved |= displacement > 0
        if snapshot.outcome:
            break
    assert moved and snapshot.has_keycard and snapshot.outcome == "escaped"
    assert bot.label == "explorer_scripted"


def test_env_import_and_step_do_not_load_pygame_or_assets():
    code = "from rl.env import AlienEnv; import sys; e=AlienEnv(); e.reset(seed=1); e.step(30); assert 'pygame' not in sys.modules"
    subprocess.run([sys.executable, "-B", "-c", code], check=True)


def test_gymnasium_and_sb3_environment_checkers():
    from gymnasium.utils.env_checker import check_env
    from stable_baselines3.common.env_checker import check_env as sb3_check_env

    env = AlienEnv(max_episode_seconds=1)
    check_env(env, skip_render_check=True)
    sb3_check_env(env, warn=True)


def test_first_sight_reward_and_helper_memory_reset_for_new_episode():
    env = AlienEnv(
        arena(survivor=(5.5, 2.5), alien=(2.5, 2.5, 0)),
        bot_kind="stationary_debug",
        max_episode_seconds=1,
    )
    env.reset(seed=2)
    assert env.step(WAIT)[1] == 0.2
    assert env.step(WAIT)[1] == 0
    env.reset(seed=2)
    assert not env.reward_tracker.seen_paid
    assert env.visit_ticks == [None] * len(env.world.patrol_waypoints)
    assert env.core.snapshot().perception.last_seen is None
    assert env.step(WAIT)[1] == 0.2


def test_builders_reject_full_world_and_external_target_owns_immutable_copy():
    core = GameCore(arena(), alien_control="external")
    with pytest.raises(TypeError, match="AllowedAlienView"):
        build_observation(core.snapshot())
    with pytest.raises(TypeError, match="AllowedAlienView"):
        build_candidates(core.snapshot())
    target = [12.5, 10.5]
    core.set_alien_target(target)
    target[0] = 2.5
    assert core.snapshot().alien_target == (12.5, 10.5)
    with pytest.raises(ValueError, match="source"):
        core.set_alien_target(None, source=[])
