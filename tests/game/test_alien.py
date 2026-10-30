"""Headless sensing, memory, path, continuous capture and lifecycle contracts."""

from dataclasses import FrozenInstanceError, replace
from math import cos, hypot, pi, sin

import pytest

from ai.controller import ControllerState, choose_target
from ai.navigation import Navigator, find_path
from game.capture import capture_fraction
from game.core import GameCore
from game.map import MapObject, WorldMap, small_test_map
from game.motion import segment_has_clearance
from game.perception import PerceptionSnapshot, update_perception
from game.state import AlienState, CoreConfig, SurvivorCommand, SurvivorState


def arena(survivor=(2.5, 2.5), alien=(6.5, 2.5, pi), waypoints=(), rows=None, objects=()):
    return WorldMap(rows or ("." * 16,) * 16, (survivor,), objects, alien, waypoints)


@pytest.mark.parametrize(
    "distance,angle,visible",
    [
        (8, 0, True),
        (8.001, 0, False),
        (4, pi / 3, True),
        (4, pi / 3 + 0.001, False),
        (0.99, pi, True),
        (1.001, pi, False),
        (0.5, -pi, True),
    ],
)
def test_perception_range_fov_near_sensor_and_confirmation(distance, angle, visible):
    world = arena()
    alien = AlienState(5.5, 5.5)
    survivor = SurvivorState(5.5 + distance * cos(angle), 5.5 + distance * sin(angle))
    memory = PerceptionSnapshot()
    for tick in range(6):
        memory = update_perception(world, alien, survivor, memory, CoreConfig())
        assert memory.visible == (visible and tick == 5)
    assert (memory.last_seen is not None) == visible


def test_confirmation_requires_continuous_sensing_and_wall_blocks_near_sensor():
    world = arena(rows=("......", "..#...", "......", "......"))
    alien = AlienState(1.75, 1.5)
    behind_wall = SurvivorState(3.1, 1.5)
    config = CoreConfig(alien_near_range=2)
    memory = PerceptionSnapshot()
    for _ in range(10):
        memory = update_perception(world, alien, behind_wall, memory, config)
    assert memory == PerceptionSnapshot()
    world = arena()
    survivor = SurvivorState(3.5, 1.5)
    for _ in range(5):
        memory = update_perception(world, alien, survivor, memory, config)
    assert not memory.visible
    memory = update_perception(world, alien, SurvivorState(0.2, 10), memory, config)
    assert memory.confirmation_ticks == 0
    for _ in range(5):
        memory = update_perception(world, alien, survivor, memory, config)
        assert not memory.visible
    assert update_perception(world, alien, survivor, memory, config).visible


def test_hidden_positions_cannot_change_memory_targets_or_navigation():
    world = arena()
    alien = AlienState(5.5, 5.5)
    previous = PerceptionSnapshot(True, 6, (7.5, 5.5), 0, revision=1)
    config = CoreConfig()
    memories = [
        update_perception(world, alien, SurvivorState(*p), previous, config)
        for p in ((2.5, 5.5), (3.5, 3.5))
    ]
    assert memories[0] == memories[1] == PerceptionSnapshot(False, 0, (7.5, 5.5), 1, revision=1)
    choices = [choose_target(alien, m, ((10.5, 10.5),), ControllerState(), 0.15) for m in memories]
    assert choices[0] == choices[1]
    assert choices[0].source == "last_seen" and choices[0].target == (7.5, 5.5)
    assert find_path(world, (alien.x, alien.y), choices[0].target, 0.2) == find_path(
        world, (alien.x, alien.y), choices[1].target, 0.2
    )


def test_core_memory_samples_before_movement_and_holds_chase_for_two_seconds():
    config = CoreConfig(run_speed=120, alien_search_speed=1e-6, alien_chase_speed=1e-6)
    core = GameCore(arena((7.5, 5.5), (5.5, 5.5, 0)), config)
    for _ in range(6):
        core.step(SurvivorCommand())
    assert core.snapshot().perception.visible
    moved = core.step(SurvivorCommand(strafe=1, running=True))
    assert moved.perception.last_seen == (7.5, 5.5)  # old position, not this tick's movement
    assert moved.survivor.y == pytest.approx(9.5)
    for age in range(1, 62):
        snapshot = core.step(SurvivorCommand())
        assert snapshot.perception.last_seen == (7.5, 5.5)
        assert snapshot.perception.last_seen_age_ticks == age
        assert snapshot.alien.speed_mode == ("chase" if age <= 60 else "search")


def test_heuristic_selects_only_every_half_second(monkeypatch):
    import game.core as module

    calls = []
    original = module.choose_target

    def record(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(module, "choose_target", record)
    core = GameCore(arena((2.5, 2.5), (11.5, 11.5, 0)))
    for _ in range(31):
        core.step(SurvivorCommand())
    assert len(calls) == 3  # pre-ticks 0,15,30
    # Adds only the investigated revision and heard memory, no truth.
    assert all(len(args) == 7 for args in calls)


def test_path_detours_and_no_diagonal_corner_cutting():
    world = small_test_map()
    start, target = (2.5, 4.5), (9.5, 7.5)
    path = find_path(world, start, target, 0.2)
    assert path and path[-1] == target and len(path) > 2
    for a, b in zip((start,) + path, path):
        assert segment_has_clearance(world, a, b, 0.2)
    corner = WorldMap(("....", "..#.", ".#..", "...."), ((1.5, 1.5),))
    assert not segment_has_clearance(corner, (1.5, 1.5), (2.5, 2.5), 0.2)
    path = find_path(corner, (1.5, 1.5), (2.5, 2.5), 0.2)
    assert len(path) > 2
    for a, b in zip(((1.5, 1.5),) + path, path):
        assert segment_has_clearance(corner, a, b, 0.2)


def test_navigation_respects_radius_and_unreachable_target():
    world = WorldMap(
        ("#######", "#.....#", "#.....#", "###.###", "#.....#", "#.....#", "#######"), ((3.5, 1.8),)
    )
    assert find_path(world, (3.5, 1.8), (3.5, 5.2), 0.2)
    assert not find_path(world, (3.5, 1.8), (3.5, 5.2), 0.6)
    blocked = WorldMap(("...#..",) * 6, ((1.5, 2.5),))
    nav = Navigator()
    nav.set_target(blocked, (1.5, 2.5), (4.5, 2.5), 0.2)
    assert nav.path == () and nav.displacement((1.5, 2.5), 1) == (0, 0)


def test_navigation_controller_actually_reaches_chosen_target_through_core():
    target = (9.5, 7.5)
    world = replace(
        small_test_map(),
        spawn_points=((2.5, 2.5),),
        alien_spawn=(2.5, 4.5, 0),
        patrol_waypoints=(target,),
    )
    core = GameCore(world, CoreConfig(alien_near_range=0.1, alien_sight_range=0.1))
    for _ in range(600):
        snapshot = core.step(SurvivorCommand())
        assert snapshot.outcome is None
        assert not world.overlaps(snapshot.alien.x, snapshot.alien.y, 0.2)
        if hypot(snapshot.alien.x - target[0], snapshot.alien.y - target[1]) < 1e-8:
            break
    else:
        pytest.fail("Alien never reached target")
    assert snapshot.alien_target == target


def test_continuous_capture_detects_stationary_and_crossing_contacts_and_walls():
    world = arena()
    assert capture_fraction(world, (2, 2), (4, 2), (4, 2), (2, 2), 0.4) == pytest.approx(0.4)
    assert capture_fraction(world, (2, 2), (2, 2), (2.4, 2), (2.4, 2), 0.4) == 0
    wall = WorldMap(("...#...",) * 6, ((2.5, 2.5),))
    assert capture_fraction(wall, (2.5, 2.5), (2.5, 2.5), (4.5, 2.5), (4.5, 2.5), 3) is None


def test_fast_opposing_motion_captures_before_swapping_and_terminal_event_once():
    world = arena(waypoints=((2.5, 2.5),))
    core = GameCore(world, CoreConfig(run_speed=120, alien_search_speed=120, alien_chase_speed=120))
    ended = core.step(SurvivorCommand(forward=1, running=True))
    assert ended.outcome == "captured"
    assert ended.survivor.x < ended.alien.x
    assert ended.survivor.x == pytest.approx(4.3)
    assert [e.kind for e in ended.events] == ["captured"]
    for _ in range(10):
        assert core.step(SurvivorCommand(1, 1, 1, True, True)) == replace(ended, events=())


def test_no_capture_or_detection_through_wall_even_with_large_capture_padding():
    core = GameCore(
        arena((2.5, 2.5), (4.5, 2.5, pi), rows=("...#...",) * 6), CoreConfig(capture_padding=3)
    )
    for _ in range(20):
        snapshot = core.step(SurvivorCommand())
        assert snapshot.outcome is None
        assert not snapshot.perception.visible
        assert snapshot.perception.last_seen is None


def escape_scene():
    return arena(
        waypoints=((2.5, 2.5),),
        objects=(MapObject("a", "keycard", 3.5, 2.5), MapObject("z", "exit", 3.5, 2.5)),
    )


def test_early_escape_freezes_alien_but_same_tick_capture_beats_escape():
    core = GameCore(escape_scene())
    assert core.step(SurvivorCommand(interact=True)).has_keycard
    early = core.step(SurvivorCommand(interact=True))
    assert early.outcome == "escaped"
    assert core.step(SurvivorCommand()).alien == early.alien
    core = GameCore(escape_scene())
    core.step(SurvivorCommand(interact=True))
    for _ in range(200):
        before = core.snapshot()
        gap = hypot(before.alien.x - before.survivor.x, before.alien.y - before.survivor.y)
        speed = (
            core.config.alien_chase_speed
            if before.alien.speed_mode == "chase"
            else core.config.alien_search_speed
        )
        if gap <= 0.4 + speed / 30 + 1e-8:
            break
        core.step(SurvivorCommand())
    else:
        pytest.fail("Alien did not approach capture distance")
    ended = core.step(SurvivorCommand(interact=True))
    assert ended.outcome == "captured"
    assert [e.kind for e in ended.events] == ["captured"]


def test_default_alien_spawn_safe_and_reset_clears_every_helper():
    core = GameCore()
    initial = core.reset(0)
    assert initial.alien is not None and core.config.alien_enabled
    assert hypot(initial.alien.x - initial.survivor.x, initial.alien.y - initial.survivor.y) >= 2
    for _ in range(80):
        core.step(SurvivorCommand())
    assert core.reset(0) == initial
    assert core.snapshot().navigation_path == ()
    assert core.snapshot().perception == PerceptionSnapshot()
    assert core.snapshot().alien_target is None
    with pytest.raises(FrozenInstanceError):
        initial.alien.x = 99
    with pytest.raises(FrozenInstanceError):
        initial.perception.last_seen = (99, 99)
    with pytest.raises(ValueError, match="too close"):
        GameCore(arena((2.5, 2.5), (3.5, 2.5, 0)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alien_radius": 0},
        {"alien_search_speed": 0},
        {"alien_fov": 0},
        {"sight_confirmation_seconds": 0},
        {"chase_hold_seconds": -1},
        {"alien_decision_seconds": 0},
        {"capture_padding": -1},
        {"minimum_spawn_distance": -1},
    ],
)
def test_alien_config_validation(kwargs):
    with pytest.raises(ValueError):
        CoreConfig(**kwargs)


def test_deterministic_seen_lost_search_capture_and_escape_replays():
    from tools.replay_alien import run_replays

    assert run_replays() == run_replays()


def test_spawn_cannot_begin_in_contact_even_if_minimum_distance_is_overridden():
    with pytest.raises(ValueError, match="too close"):
        GameCore(arena((2.5, 2.5), (2.8, 2.5, 0)), CoreConfig(minimum_spawn_distance=0))


def test_active_alien_simulation_is_independent_of_render_schedule():
    from app.loop import FixedStepLoop

    def simulate(frames, delta):
        loop = FixedStepLoop(GameCore())
        for _ in range(frames):
            loop.advance(delta)
            loop.display_snapshot()
        return loop.current

    assert simulate(240, 1 / 60) == simulate(576, 1 / 144)
