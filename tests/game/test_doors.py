"""Automatic doors block sight while shut and never stand in anyone's way."""

import pytest

from game.core import GameCore
from game.doors import sight_world
from game.interaction import segment_clear
from game.map import WorldMap, scenario_map
from game.ship_deck import DOORS, SHIP_SCENARIO
from game.state import CoreConfig, SurvivorCommand, SurvivorState
from rl.bots.views import observe_threat

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
CONFIG = CoreConfig(alien_enabled=False)


def core_at(x, y, *, yaw=0.0, doors=(DOOR,), config=CONFIG):
    world = WorldMap(ROOMS, ((x, y),), doors=doors)
    core = GameCore(world, config)
    core.reset(1)
    core._survivor = SurvivorState(x, y, yaw)
    return core


def wait(core, ticks):
    for _ in range(ticks):
        snapshot = core.step(SurvivorCommand())
    return snapshot


def test_door_starts_shut_and_blocks_sight():
    snapshot = core_at(1.5, 3.5).snapshot()
    assert snapshot.closed_doors == (DOOR,) and snapshot.door_open == (0.0,)
    assert not segment_clear(sight_world(snapshot), 2.5, 3.5, 6.5, 3.5)
    assert segment_clear(snapshot.world_map, 2.5, 3.5, 6.5, 3.5)


def test_door_opens_for_someone_in_its_passage_then_closes():
    core = core_at(3.5, 3.5)
    snapshot = wait(core, 11)
    assert snapshot.door_open == (1.0,) and snapshot.closed_doors == ()
    assert segment_clear(sight_world(snapshot), 2.5, 3.5, 6.5, 3.5)
    core._survivor = SurvivorState(1.5, 3.5)
    assert wait(core, 11).closed_doors == (DOOR,)


def test_door_ignores_someone_behind_the_wall_beside_it():
    assert wait(core_at(3.5, 1.5), 15).door_open == (0.0,)


def test_door_stays_open_while_someone_stands_in_it():
    core = core_at(4.5, 3.5)
    wait(core, 40)
    assert core.snapshot().door_open == (1.0,)


def test_a_runner_never_reaches_a_door_before_it_clears():
    core = core_at(2.0, 3.5)
    command = SurvivorCommand(forward=1.0, running=True)
    for _ in range(45):
        snapshot = core.step(command)
        if 3.8 < snapshot.survivor.x < 5.2:
            assert snapshot.closed_doors == ()
    assert snapshot.survivor.x > 4.6  # the door never slowed the run down


def test_alien_cannot_see_through_a_shut_door_but_does_through_an_open_one():
    world = WorldMap(ROOMS, ((1.5, 3.5),), alien_spawn=(7.5, 3.5, 3.1416), doors=(DOOR,))
    config = CoreConfig(alien_sight_range=8.0, door_reach=0.1, door_lateral=0.1)
    core = GameCore(world, config, alien_control="external")
    core.reset(1)
    assert not wait(core, 12).perception.visible
    core._doors.closed = ()
    core._doors.openness = (1.0,)
    assert core.step(SurvivorCommand()).perception.confirmation_ticks == 1


def test_bot_sensors_stop_at_a_shut_door():
    class Settings:
        sight_range, fov_degrees = 8.0, 75.0

    class Alien:
        x, y = 6.5, 3.5

    snapshot = core_at(2.5, 3.5).snapshot()
    survivor = snapshot.survivor
    assert (
        observe_threat(sight_world(snapshot), survivor, False, Alien, Settings).visible_alien
        is None
    )
    seen = observe_threat(snapshot.world_map, survivor, False, Alien, Settings)
    assert seen.visible_alien == (6.5, 3.5)


def test_reset_shuts_every_door():
    core = core_at(3.5, 3.5)
    wait(core, 11)
    assert core.reset(1).door_open == (0.0,)


@pytest.mark.parametrize("scenario", ["small_test_map_v1", "occluded_loop_dev_v1"])
def test_scenes_without_doors_see_as_before(scenario):
    snapshot = GameCore(scenario_map(scenario)).snapshot()
    assert sight_world(snapshot) is snapshot.world_map


def test_every_ship_door_stands_in_a_one_cell_passage():
    world = scenario_map(SHIP_SCENARIO)
    assert world.doors == DOORS and len(DOORS) == 27
    for x, y in DOORS:
        along_x = not world.blocked(x - 1, y) and not world.blocked(x + 1, y)
        along_y = not world.blocked(x, y - 1) and not world.blocked(x, y + 1)
        assert along_x != along_y, (x, y)
