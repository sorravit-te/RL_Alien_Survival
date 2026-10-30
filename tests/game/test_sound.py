"""A running footfall reaches the Alien along the floor, as a coarse region only."""

from math import pi

import pytest

from ai.controller import ControllerState, choose_target
from game.core import GameCore
from game.map import WorldMap, scenario_map
from game.perception import PerceptionSnapshot
from game.ship_deck import SHIP_SCENARIO
from game.sound import HeardSnapshot, carries, listen, region_point
from game.state import AlienState, CoreConfig, SurvivorCommand, SurvivorState

HALL = (
    "##########",
    "#........#",
    "#........#",
    "##########",
)
SPLIT = (
    "###########",
    "#....#....#",
    "#....#....#",
    "#....#....#",
    "#.........#",
    "###########",
)
RUN = SurvivorCommand(forward=1.0, running=True)
WALK = SurvivorCommand(forward=1.0)
LONG = (
    "#################",
    "#...............#",
    "#...............#",
    "#################",
)
# The Alien looks away, so only hearing can tell it anything.
BLIND = CoreConfig(alien_sight_range=1.0, alien_near_range=0.5)
DOOR_ONLY = CoreConfig(
    alien_sight_range=1.0, alien_near_range=0.5, walk_sound_range=0.1, run_sound_range=0.1
)


def core_with(rows, survivor, alien, *, doors=(), config=BLIND, control="external"):
    world = WorldMap(rows, (survivor,), alien_spawn=(*alien, 0.0), doors=doors)
    core = GameCore(world, config, alien_control=control)
    core.reset(1)
    return core


def steps(core, command, ticks):
    for _ in range(ticks):
        snapshot = core.step(command)
    return snapshot


def test_running_in_range_is_heard():
    heard = steps(core_with(HALL, (1.5, 1.5), (7.5, 1.5)), RUN, 10).heard
    assert heard.point is not None and heard.revision >= 1 and heard.age_ticks is not None


def test_walking_is_heard_only_close_by():
    assert steps(core_with(HALL, (1.5, 1.5), (7.5, 1.5)), WALK, 20).heard == HeardSnapshot()
    assert steps(core_with(HALL, (1.5, 1.5), (3.5, 2.5)), WALK, 16).heard.revision >= 1


def test_standing_still_is_silent():
    assert steps(core_with(HALL, (1.5, 1.5), (7.5, 1.5)), SurvivorCommand(), 20).heard.point is None


def test_running_on_the_spot_makes_no_noise():
    command = SurvivorCommand(running=True)
    assert steps(core_with(HALL, (1.5, 1.5), (7.5, 1.5)), command, 20).heard.point is None


def test_running_into_a_wall_makes_no_noise():
    core = core_with(HALL, (1.5, 1.5), (7.5, 1.5))
    core._survivor = SurvivorState(1.2, 1.5, pi)
    assert steps(core, RUN, 20).heard.point is None


def test_a_door_the_survivor_opens_is_heard_past_a_walking_step():
    core = core_with(HALL, (2.2, 1.5), (7.5, 1.5), doors=((4, 1),), config=DOOR_ONLY)
    snapshot = steps(core, WALK, 10)
    assert snapshot.door_open[0] > 0.0
    assert snapshot.heard.point == region_point(core.world_map, 4.5, 1.5, 4)


def test_a_door_is_heard_once_per_opening():
    core = core_with(HALL, (2.2, 1.5), (7.5, 1.5), doors=((4, 1),), config=DOOR_ONLY)
    assert steps(core, WALK, 15).heard.revision == 1


def test_a_door_the_alien_opens_itself_gives_no_clue():
    core = core_with(HALL, (1.5, 1.5), (5.8, 1.5), doors=((4, 1),), config=DOOR_ONLY)
    snapshot = steps(core, SurvivorCommand(), 10)
    assert snapshot.door_open[0] > 0.0 and snapshot.heard.point is None


def test_a_door_too_far_along_the_floor_stays_unheard():
    core = core_with(LONG, (2.2, 1.5), (14.5, 1.5), doors=((4, 1),), config=DOOR_ONLY)
    snapshot = steps(core, WALK, 10)
    assert snapshot.door_open[0] > 0.0 and snapshot.heard.point is None


def test_noise_cannot_cross_a_wall_it_would_have_to_walk_around():
    world = WorldMap(SPLIT, ((1.5, 1.5),))
    assert not carries(world, frozenset(), (4.5, 1.5), (6.5, 1.5), 4.0, 3.0)
    assert carries(world, frozenset(), (4.5, 3.5), (6.5, 3.5), 4.0, 3.0)
    assert carries(world, frozenset(), (4.5, 1.5), (6.5, 1.5), 8.0, 3.0)


def test_a_shut_door_cuts_how_far_noise_carries():
    world = WorldMap(HALL, ((1.5, 1.5),))
    assert carries(world, frozenset(), (1.5, 1.5), (7.5, 1.5), 7.0, 3.0)
    both_rows = frozenset(((4, 1), (4, 2)))
    assert not carries(world, both_rows, (1.5, 1.5), (7.5, 1.5), 7.0, 3.0)
    assert carries(world, frozenset(((4, 1),)), (1.5, 1.5), (7.5, 1.5), 7.0, 3.0)  # other row


def test_a_region_gives_one_floor_point_whatever_spot_made_the_noise():
    world = scenario_map(SHIP_SCENARIO)
    spots = {
        region_point(world, x + 0.5, y + 0.5, 4)
        for y in range(16, 20)
        for x in range(36, 40)
        if not world.blocked(x, y)
    }
    assert len(spots) == 1
    x, y = spots.pop()
    assert not world.blocked(int(x), int(y))


def test_every_floor_cell_has_a_region_point_on_floor():
    world = scenario_map(SHIP_SCENARIO)
    for y, row in enumerate(world.rows):
        for x, tile in enumerate(row):
            if tile == ".":
                px, py = region_point(world, x + 0.5, y + 0.5, 4)
                assert not world.blocked(int(px), int(py))


def test_old_clue_ages_in_place_and_new_noise_replaces_it():
    world = WorldMap(HALL, ((1.5, 1.5),))
    alien = AlienState(7.5, 1.5)
    far = (((1.5, 1.5), 8.0),)
    first = listen(world, frozenset(), alien, far, HeardSnapshot(), BLIND)
    assert (first.age_ticks, first.revision) == (0, 1)
    older = listen(world, frozenset(), alien, (), first, BLIND)
    assert older.point == first.point and older.age_ticks == 1
    fresh = listen(world, frozenset(), alien, (((5.5, 1.5), 8.0),), older, BLIND)
    assert fresh.point != first.point and (fresh.age_ticks, fresh.revision) == (0, 2)


def test_standing_at_the_point_marks_it_checked():
    world = WorldMap(HALL, ((1.5, 1.5),))
    clue = HeardSnapshot((2.5, 1.5), 0, 1)
    alien = AlienState(2.5, 1.5)
    assert listen(world, frozenset(), alien, (), clue, BLIND).checked


def test_heuristic_prefers_sight_then_clue_then_sound_then_patrol():
    alien = AlienState(1.5, 1.5)
    heard = HeardSnapshot((5.5, 1.5), 0, 1)
    waypoints = ((8.5, 1.5),)
    state = ControllerState()
    seen = PerceptionSnapshot(True, 6, (3.5, 1.5), 0, 1)
    lost = PerceptionSnapshot(False, 0, (3.5, 1.5), 9, 1)
    none = PerceptionSnapshot()
    choose = lambda perception, clue, done=None: choose_target(  # noqa: E731
        alien, perception, waypoints, state, 0.15, done, clue
    )
    assert choose(seen, heard).source == "visible"
    assert choose(lost, heard).source == "last_seen"
    assert choose(lost, heard, 1).source == "sound"
    assert choose(none, heard).target == (5.5, 1.5)
    checked = HeardSnapshot((5.5, 1.5), 0, 1, True)
    assert choose(none, checked).source == "patrol"


def test_scripted_alien_walks_to_a_noise_then_returns_to_patrol():
    world = WorldMap(
        HALL, ((1.5, 2.5),), alien_spawn=(7.5, 1.5, 0.0), patrol_waypoints=((8.5, 1.5),)
    )  # one row apart, so the Alien passes the Survivor without touching
    core = GameCore(world, BLIND)
    core.reset(1)
    snapshot = steps(core, RUN, 16)
    assert snapshot.alien_target_source == "sound"
    for _ in range(200):
        snapshot = core.step(SurvivorCommand())
    assert snapshot.heard.checked and snapshot.alien_target_source == "patrol"


@pytest.mark.parametrize(
    "field",
    [
        "run_sound_range",
        "walk_sound_range",
        "door_sound_range",
        "sound_step_seconds",
        "sound_door_cost",
    ],
)
def test_sound_settings_must_be_positive(field):
    with pytest.raises(ValueError):
        CoreConfig(**{field: 0.0})


@pytest.mark.parametrize("size", [0, 1.5])
def test_sound_region_must_be_whole_cells(size):
    with pytest.raises(ValueError):
        CoreConfig(sound_region=size)
