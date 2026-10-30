"""The Runner rushes its search at run speed; other bots and mixes stay as they were."""

import numpy as np

from game.core import GameCore
from game.map import WorldMap
from game.ship_deck import SHIP_SCENARIO
from game.state import CoreConfig
from rl.bots.evasive import EvasiveExplorerBot, RunnerEvasiveBot
from rl.bots.explorer import ExplorerBot
from rl.bots.views import SurvivorThreatView, observe_survivor
from rl.bots.wall_loop import EvasiveMixBot, RunnerMixBot
from rl.contract import WAIT
from rl.env import THREAT_BOT_KINDS, AlienEnv
from rl.pipeline import TRAINING_BOT_KINDS

HALL = (
    "##############",
    "#............#",
    "##############",
)
CONFIG = CoreConfig(alien_enabled=False)


def walk_to(bot, target, ticks=400):
    """Ticks until the Survivor reaches target, driven by the bot's own commands."""
    core = GameCore(WorldMap(HALL, ((1.5, 1.5),)), CONFIG)
    core.reset(1)
    bot.reset(np.random.default_rng(1))
    bot.order, bot.scanning, bot.explore_target = (target,), 0, target
    commands = []
    for tick in range(ticks):
        snapshot = core.snapshot()
        view = observe_survivor(snapshot.world_map, snapshot.survivor, False)
        if abs(snapshot.survivor.x - target[0]) < 1e-6:
            return tick, commands
        commands.append(bot.command(view))
        core.step(commands[-1])
    raise AssertionError("never arrived")


def test_runner_covers_ground_at_run_speed():
    walked, _ = walk_to(ExplorerBot(HALL, CONFIG), (11.5, 1.5))
    ran, commands = walk_to(ExplorerBot(HALL, CONFIG, running=True), (11.5, 1.5))
    assert ran < walked * 0.75
    assert any(command.running and command.forward > 0 for command in commands)


def test_plain_explorer_never_runs():
    _, commands = walk_to(ExplorerBot(HALL, CONFIG), (11.5, 1.5))
    assert not any(command.running for command in commands)


def test_runner_stops_exactly_on_its_target():
    bot = ExplorerBot(HALL, CONFIG, running=True)
    ticks, _ = walk_to(bot, (4.5, 1.5))
    assert ticks > 0


def test_runner_is_an_evasive_bot_that_searches_at_a_run():
    bot = RunnerEvasiveBot(HALL, CONFIG)
    bot.reset(np.random.default_rng(1))
    bot.explorer.scanning = 0
    core = GameCore(WorldMap(HALL, ((1.5, 1.5),)), CONFIG)
    snapshot = core.snapshot()
    view = observe_survivor(snapshot.world_map, snapshot.survivor, False)
    commands = [bot.command(SurvivorThreatView(view)) for _ in range(3)]
    assert bot.explorer.running and bot.label == "runner_evasive_scripted"
    assert isinstance(commands[0].running, bool)
    assert not EvasiveExplorerBot(HALL, CONFIG).explorer.running


def test_runner_mix_gives_the_runner_a_third_of_episodes():
    mix = RunnerMixBot(HALL, CONFIG)
    picks = []
    for seed in range(300):
        mix.reset(np.random.default_rng(seed), bot_seed=seed)
        picks.append(mix.active is mix.runner)
    assert 0.25 < sum(picks) / len(picks) < 0.42


def test_runner_mix_repeats_for_a_seed():
    one, two = RunnerMixBot(HALL, CONFIG), RunnerMixBot(HALL, CONFIG)
    for seed in range(20):
        one.reset(np.random.default_rng(seed), bot_seed=seed)
        two.reset(np.random.default_rng(seed), bot_seed=seed)
        assert one.label == two.label


def test_runner_mix_keeps_evasive_mix_choice_on_other_episodes():
    mix, plain = RunnerMixBot(HALL, CONFIG), EvasiveMixBot(HALL, CONFIG)
    checked = 0
    for seed in range(60):
        mix.reset(np.random.default_rng(seed), bot_seed=seed)
        plain.reset(np.random.default_rng(seed), bot_seed=seed)
        if mix.active is not mix.runner:
            assert mix.label == plain.label
            assert mix.active.explorer.order == plain.active.explorer.order
            checked += 1
    assert checked > 20


def test_mix_passes_doors_to_the_runner():
    door = (6, 1)
    mix = RunnerMixBot(HALL, CONFIG, doors=(door,))
    assert mix.runner.explorer.stations == mix.plain.explorer.stations
    assert mix.runner.explorer.walk_steps


def test_new_kinds_can_be_trained():
    for kind in ("runner_evasive", "runner_mix"):
        assert kind in THREAT_BOT_KINDS and kind in TRAINING_BOT_KINDS


def test_environment_runs_a_runner_episode_on_the_ship():
    env = AlienEnv(scenario=SHIP_SCENARIO, bot_kind="runner_mix", max_episode_seconds=5)
    env.reset(seed=1, options={"scenario_seed": 7000, "bot_seed": 5})
    done = False
    while not done:
        *_, terminated, truncated, info = env.step(WAIT)
        done = terminated or truncated
    assert info["outcome"] == "unresolved"
