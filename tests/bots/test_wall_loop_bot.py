"""Wall-loop evasive bot, per-episode evasive mix, and the pursuit helper against them."""

from math import atan2, cos, hypot, pi, sin

import numpy as np
import pytest

from game.interaction import segment_clear
from game.map import scenario_map
from game.state import CoreConfig
from rl.bots.evasive import EvasionConfig, EvasiveExplorerBot
from rl.bots.views import SurvivorThreatView, SurvivorView
from rl.bots.wall_loop import EvasiveMixBot, WallLoopEvasiveBot
from rl.env import AlienEnv
from rl.scripted_controllers import HeuristicSelector

SCENE = "occluded_loop_dev_v1"


def threat_view(position, yaw, alien):
    return SurvivorThreatView(SurvivorView(*position, yaw, False, ()), alien)


def test_loop_settings_derive_from_configured_evasion():
    world, config = scenario_map(SCENE), CoreConfig()
    plain = EvasiveExplorerBot(world.rows, config)
    loop = WallLoopEvasiveBot(world.rows, config)
    assert plain.settings == EvasionConfig() and not plain.replan_on_arrival
    assert loop.replan_on_arrival
    assert loop.settings.enter_distance == plain.settings.safe_distance
    assert loop.settings.safe_distance == plain.settings.sight_range
    assert loop.settings.memory_seconds == 2 * plain.settings.memory_seconds
    assert loop.memory_ticks == 2 * plain.memory_ticks


def test_loop_flee_picks_occluded_cover_and_watches_threat():
    world, config = scenario_map(SCENE), CoreConfig()
    bot = WallLoopEvasiveBot(world.rows, config)
    bot.reset(np.random.default_rng(0))
    # North of the central block, Alien approaching from the west along the corridor.
    position, alien = (6.5, 2.5), (3.5, 2.5)
    command = bot.command(threat_view(position, 0.0, alien))
    assert bot.fleeing and command.running
    cover = bot.escape_target
    assert not segment_clear(world, *alien, *cover)
    assert hypot(cover[0] - alien[0], cover[1] - alien[1]) > hypot(
        position[0] - alien[0], position[1] - alien[1]
    )
    # The turn heads toward the threat, while displacement retreats from it.
    wanted = atan2(alien[1] - position[1], alien[0] - position[0])
    assert abs((0.0 + command.yaw_delta - wanted + pi) % (2 * pi) - pi) < abs(
        (0.0 - wanted + pi) % (2 * pi) - pi
    )
    yaw = command.yaw_delta
    move = (
        cos(yaw) * command.forward - sin(yaw) * command.strafe,
        sin(yaw) * command.forward + cos(yaw) * command.strafe,
    )
    assert move[0] * (position[0] - alien[0]) + move[1] * (position[1] - alien[1]) > 0


def test_loop_bot_accepts_only_filtered_views():
    bot = WallLoopEvasiveBot(scenario_map(SCENE).rows, CoreConfig())
    bot.reset(np.random.default_rng(0))
    with pytest.raises(TypeError):
        bot.command(SurvivorView(2.5, 2.5, 0.0, False, ()))


def test_mix_choice_is_seeded_and_balanced():
    world, config = scenario_map(SCENE), CoreConfig()
    mix = EvasiveMixBot(world.rows, config)
    labels = []
    for seed in range(200):
        mix.reset(np.random.default_rng(seed), bot_seed=seed)
        labels.append(mix.label)
    again = EvasiveMixBot(world.rows, config)
    again.reset(np.random.default_rng(7), bot_seed=7)
    assert again.label == labels[7]
    assert set(labels) == {EvasiveExplorerBot.label, WallLoopEvasiveBot.label}
    assert 0.35 < labels.count(WallLoopEvasiveBot.label) / len(labels) < 0.65


def rollout(env, bot_seed, steps=40):
    env.reset(seed=11, options={"scenario_seed": 5, "bot_seed": bot_seed})
    selector = HeuristicSelector(env.config.target_arrival_distance)
    trace = []
    for _ in range(steps):
        _, _, term, trunc, info = env.step(selector.choose(env._allowed_view()))
        s = env.core.snapshot()
        trace.append((s.survivor, s.alien, info["outcome"]))
        if term or trunc:
            break
    return trace, info["bot"]


def test_plain_mix_episode_replays_plain_evasive_bot_exactly():
    mix = AlienEnv(scenario=SCENE, bot_kind="evasive_mix")
    plain = AlienEnv(scenario=SCENE, bot_kind="evasive_explorer")
    try:
        seed = next(
            s for s in range(50) if np.random.default_rng([s, 1]).random() >= 0.5
        )  # a plain-bot episode
        mixed, label = rollout(mix, seed)
        assert label == EvasiveExplorerBot.label
        assert mixed == rollout(plain, seed)[0]
    finally:
        mix.close()
        plain.close()


def heuristic_outcomes(seeds):
    env = AlienEnv(scenario=SCENE, bot_kind="wall_loop_evasive", max_episode_seconds=60)
    outcomes = []
    try:
        for i in seeds:
            env.reset(seed=1000 + i, options={"scenario_seed": 2000 + i, "bot_seed": 3000 + i})
            selector = HeuristicSelector(env.config.target_arrival_distance)
            done = False
            while not done:
                _, _, term, trunc, info = env.step(selector.choose(env._allowed_view()))
                done = term or trunc
            outcomes.append(info["outcome"])
    finally:
        env.close()
    return outcomes


def test_pipeline_accepts_evasive_mix_pilot_config_and_rejects_debug_bot():
    from rl.evaluation_protocol import validate_protocol
    from rl.pipeline import make_env, read_json, validate_environment_config

    config = read_json("configs/pursuit_tracking_training_pilot.json")
    protocol = read_json(config["evaluation_protocol"])
    validate_protocol(config, protocol)
    validate_environment_config(config)
    env = make_env(config)
    try:
        assert env.bot_kind == "evasive_mix"
        assert env.bot.settings == EvasionConfig(**config["survivor_threat_settings"])
    finally:
        env.close()
    with pytest.raises(ValueError, match="mission bot"):
        validate_environment_config({**config, "bot_kind": "stationary_debug"})
    with pytest.raises(ValueError, match="evasive mission bot"):
        validate_protocol({**config, "bot_kind": "explorer"}, protocol)


def test_tracking_pursuit_catches_wall_loop_bot():
    # Before tracking pursuit the heuristic caught 0/20 wall-loop episodes (plan.md v1.6).
    assert heuristic_outcomes(range(3)) == ["captured"] * 3
