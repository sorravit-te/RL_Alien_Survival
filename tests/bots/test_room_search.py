"""A deck with doors is searched from stations, nearest first; other scenes stay as they were."""

import numpy as np
import pytest

from game.interaction import segment_clear
from game.map import WorldMap, scenario_map
from game.ship_deck import DOORS, SHIP_SCENARIO
from game.state import CoreConfig
from rl.bots.evasive import EvasiveExplorerBot
from rl.bots.explorer import ExplorerBot
from rl.bots.stations import STATION_SIGHT, Walls, plan_stations
from rl.bots.views import SurvivorView
from rl.bots.wall_loop import EvasiveMixBot, WallLoopEvasiveBot
from rl.env import AlienEnv

ROOMS = (
    "#########",
    "#...#...#",
    "#...#...#",
    "#.......#",
    "#...#...#",
    "#...#...#",
    "#########",
)
DOOR = (4, 3)
SHIP = scenario_map(SHIP_SCENARIO)


def explorer(rows=ROOMS, doors=(DOOR,), seed=1):
    bot = ExplorerBot(rows, CoreConfig(), doors=doors)
    bot.reset(np.random.default_rng(seed))
    return bot


def view_at(x, y):
    return SurvivorView(x, y, 0.0, False, ())


def test_each_side_of_a_shut_door_has_its_own_station():
    stations = explorer().stations
    assert any(x < 4 for x, _ in stations) and any(x > 5 for x, _ in stations)


def test_stations_see_the_whole_ship_and_never_stand_in_a_doorway():
    bot = ExplorerBot(SHIP.rows, CoreConfig(), doors=DOORS)
    door_points = {(x + 0.5, y + 0.5) for x, y in DOORS}
    assert not door_points & set(bot.stations)
    assert len(bot.stations) < len(bot.points) / 5
    solid = {(x, y) for y, row in enumerate(SHIP.rows) for x, tile in enumerate(row) if tile == "#"}
    walls = Walls(frozenset(solid | set(DOORS)))
    for point in set(bot.points) - door_points:
        assert any(
            np.hypot(point[0] - x, point[1] - y) <= STATION_SIGHT
            and segment_clear(walls, *point, x, y)
            for x, y in bot.stations
        ), point


def test_stations_come_from_geometry_alone():
    plain = SHIP.rows
    bot = ExplorerBot(plain, CoreConfig(), doors=DOORS)
    other = WorldMap(plain, ((3.5, 3.5),))
    points = tuple(p for p in bot.points if not other.overlaps(*p, CoreConfig().radius))
    assert plan_stations(plain, points, DOORS)[0] == bot.stations


def test_search_goes_to_the_nearest_station_first():
    bot = explorer()
    bot.bias = dict.fromkeys(bot.order, 0.0)
    bot.scanning = 0
    bot.command(view_at(2.5, 3.5))
    here = (2.5, 3.5)
    nearest = min(bot.stations, key=lambda station: bot.walk_steps[station][here])
    assert bot.explore_target == nearest


def test_seeds_search_in_different_orders():
    routes = set()
    for seed in range(12):
        bot = explorer(SHIP.rows, DOORS, seed)
        bot.scanning = 0
        bot.command(view_at(19.5, 19.5))
        routes.add(bot.explore_target)
    assert len(routes) > 2


def test_a_finished_search_starts_over():
    bot = explorer()
    bot.visited.update(bot.stations)
    bot.scanning = 0
    bot.command(view_at(2.5, 3.5))
    assert bot.explore_target == bot.order[0] and bot.visited == set()


def test_a_station_the_bot_was_chased_from_waits_for_the_next_sweep():
    bot = EvasiveExplorerBot(ROOMS, CoreConfig(), doors=(DOOR,))
    bot.reset(np.random.default_rng(1))
    bot.explorer.explore_target = target = bot.explorer.stations[0]
    bot._begin_flee()
    assert bot.explorer.explore_target is None and target in bot.explorer.visited


def test_a_scene_without_doors_keeps_its_target_through_a_flee():
    bot = EvasiveExplorerBot(ROOMS, CoreConfig())
    bot.reset(np.random.default_rng(1))
    bot.explorer.explore_target = target = bot.explorer.stations[0]
    bot._begin_flee()
    assert bot.explorer.explore_target == target and not bot.explorer.visited


def test_without_doors_every_floor_point_stays_a_target_in_random_order():
    bot = explorer(doors=())
    other = ExplorerBot(ROOMS, CoreConfig())
    other.reset(np.random.default_rng(1))
    assert bot.stations == bot.points and bot.walk_steps == {}
    assert bot.order == other.order
    bot.scanning = 0
    bot.command(view_at(2.5, 3.5))
    assert bot.explore_target == bot.order[0]


@pytest.mark.parametrize("kind", [EvasiveExplorerBot, WallLoopEvasiveBot])
def test_evasive_bots_search_from_stations(kind):
    bot = kind(ROOMS, CoreConfig(), doors=(DOOR,))
    assert bot.explorer.stations == explorer().stations


def test_mix_hands_doors_to_both_bots():
    bot = EvasiveMixBot(ROOMS, CoreConfig(), doors=(DOOR,))
    assert bot.plain.explorer.stations == bot.wall_loop.explorer.stations == explorer().stations


def test_env_passes_the_deck_doors_to_its_bot():
    env = AlienEnv(scenario=SHIP_SCENARIO, bot_kind="evasive_mix")
    assert env.bot.plain.explorer.stations != env.bot.plain.explorer.points
    assert AlienEnv(bot_kind="explorer").bot.walk_steps == {}
