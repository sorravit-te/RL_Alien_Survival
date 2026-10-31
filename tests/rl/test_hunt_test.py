"""The hunt test's fixed Survivors and its scoring, plus the coverage reward."""

from math import pi

import numpy as np

from game.core import GameCore
from game.map import WorldMap
from game.state import CoreConfig
from rl.bots.hunt_targets import Pacer, StillTarget
from rl.bots.views import observe_survivor
from rl.contract import WAIT
from rl.env import AlienEnv
from rl.hunt import hunt_test, stations, trial
from rl.rewards import COVERAGE_REWARD, coverage_reward
from tools.hunt_test import written_alien_chooser

HALL = (
    "##############",
    "#............#",
    "#............#",
    "##############",
)
SCENE = "occluded_loop_random_v1"


def walk(bot, ticks):
    core = GameCore(WorldMap(HALL, ((6.5, 1.5),)), CoreConfig(alien_enabled=False))
    core.reset(1)
    bot.reset(np.random.default_rng(1))
    commands = []
    for _ in range(ticks):
        snapshot = core.snapshot()
        commands.append(bot.command(observe_survivor(snapshot.world_map, snapshot.survivor, False)))
        snapshot = core.step(commands[-1])
    return snapshot, commands


def test_pacer_picks_the_longest_line_and_stays_near_its_spawn():
    snapshot, commands = walk(Pacer(WorldMap(HALL, ((6.5, 1.5),))), 400)
    assert commands[1].running and commands[1].forward == 1.0
    assert sum(abs(c.yaw_delta) == pi for c in commands) >= 8
    assert 1.0 < snapshot.survivor.x < 13.0 and snapshot.survivor.y == 1.5


def test_still_target_never_moves():
    snapshot, _ = walk(StillTarget(), 60)
    assert (snapshot.survivor.x, snapshot.survivor.y) == (6.5, 1.5)


def test_a_waiting_alien_catches_nobody_silent():
    result = trial(lambda env, obs, mask: WAIT, SCENE, lambda world: StillTarget(), 5, cases=2)
    assert result == {
        "cases": 2,
        "caught": 0,
        "median_seconds_to_catch": None,
        "noise_offered": 0,
        "noise_followed": 0,
    }


def test_written_alien_hunts_a_noisy_target():
    result = hunt_test(written_alien_chooser(), SCENE, cases=6, noisy_seconds=30)
    assert result["noisy"]["caught"] > 0
    assert result["hidden"]["cases"] == len(stations(SCENE))


def test_a_hidden_survivor_waits_at_each_station_in_turn():
    spots = stations(SCENE)[:2]
    starts = []

    def note_where_it_hides(env, obs, mask):
        survivor = env.core.snapshot().survivor
        if env.elapsed_ticks == 0:
            starts.append((survivor.x, survivor.y))
        return WAIT

    trial(note_where_it_hides, SCENE, lambda world: StillTarget(), 1, len(spots), spots=spots)
    assert starts == [tuple(spot) for spot in spots]


def test_first_visit_pays():
    assert coverage_reward([None], [10], 10, 30) == COVERAGE_REWARD


def test_staying_at_a_waypoint_pays_nothing_more():
    assert coverage_reward([10], [11], 11, 30) == 0


def test_a_quick_return_pays_nothing():
    assert coverage_reward([10], [1000], 1000, 30) == 0


def test_a_return_after_a_minute_pays_again():
    assert coverage_reward([10], [1810], 1810, 30) == COVERAGE_REWARD


def test_each_reached_waypoint_pays_once():
    assert coverage_reward([None, None, 5], [7, None, 5], 7, 30) == COVERAGE_REWARD


def test_environment_reports_the_coverage_reward():
    env = AlienEnv(scenario=SCENE, bot_kind="stationary_debug")
    env.reset(seed=1, options={"scenario_seed": 1, "bot_seed": 1})
    point = env.world.patrol_waypoints[0]
    env.core._alien = env.core._alien.__class__(*point)
    *_, info = env.step(WAIT)
    assert info["reward_components"]["coverage"] == COVERAGE_REWARD
