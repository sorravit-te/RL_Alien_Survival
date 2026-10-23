"""Core-only 1C contract; no rendering/asset dependencies."""

from dataclasses import FrozenInstanceError, replace
from math import pi

import pytest

from game.core import GameCore
from game.interaction import select_target
from game.map import MapObject, WorldMap, small_test_map
from game.state import CoreConfig, SurvivorCommand, SurvivorState
from tools.replay_escape import replay_commands, run_replay


def scene(objects=None, spawn=(2.5, 2.5), rows=None, config=None):
    objects = objects if objects is not None else (
        MapObject("card", "keycard", 3.5, 2.5), MapObject("exit", "exit", 2.5, 3.5))
    return GameCore(WorldMap(rows or ("......",) * 6, (spawn,), objects), config)


def hold(core, count=18, **kwargs):
    snapshots = [core.step(SurvivorCommand(interact=True, **kwargs)) for _ in range(count)]
    return snapshots[-1], tuple(e for s in snapshots for e in s.events)


def test_collect_requires_full_duration_and_emits_only_once():
    core = scene()
    before = core.snapshot()
    pending, events = hold(core, 17)
    assert not pending.has_keycard
    assert pending.interaction_ticks == 17
    assert events == ()
    collected, events = hold(core, 1)
    assert collected.has_keycard
    assert [(e.kind, e.object_id, e.tick) for e in events] == [("keycard_collected", "card", 18)]
    assert collected.interaction_ticks == 0
    assert collected.interaction_target is None
    assert hold(core, 60)[1] == ()
    assert not before.has_keycard


def test_exit_requires_card_and_full_duration_then_freezes_and_reset_restores():
    core = scene()
    core.step(SurvivorCommand(yaw_delta=pi / 2))
    locked, events = hold(core, 60)
    assert locked.interaction_target.kind == "exit"
    assert locked.interaction_ticks == 0
    assert locked.outcome is None
    assert events == ()
    core.step(SurvivorCommand(yaw_delta=-pi / 2))
    hold(core)
    core.step(SurvivorCommand(yaw_delta=pi / 2))
    pending, _ = hold(core, 17)
    assert pending.outcome is None
    escaped, events = hold(core, 1)
    assert escaped.outcome == "escaped"
    assert [e.kind for e in events] == ["escaped"]
    for _ in range(30):
        stopped = core.step(SurvivorCommand(1, 1, pi, True, True))
        assert stopped == replace(escaped, events=())
    assert core.reset(0) == scene().snapshot()
    assert not core.snapshot().has_keycard


@pytest.mark.parametrize("cancel", [SurvivorCommand(), SurvivorCommand(forward=1, interact=True),
                                    SurvivorCommand(strafe=1, interact=True),
                                    SurvivorCommand(yaw_delta=pi, interact=True)])
def test_releasing_moving_or_turning_away_cancels_and_restarts(cancel):
    core = scene()
    hold(core, 10)
    cancelled = core.step(cancel)
    assert cancelled.interaction_ticks == 0
    assert cancelled.interaction_object_id is None
    if cancel.yaw_delta:
        core.step(SurvivorCommand(yaw_delta=-pi))
    pending, events = hold(core, 17)
    assert not pending.has_keycard
    assert events == ()
    assert hold(core, 1)[0].has_keycard


def test_switching_targets_never_transfers_progress():
    core = scene((MapObject("a", "keycard", 3.5, 2.5),
                  MapObject("b", "keycard", 2.5, 3.5)))
    hold(core, 10)
    switched = core.step(SurvivorCommand(yaw_delta=pi / 2, interact=True))
    assert switched.interaction_object_id == "b"
    assert switched.interaction_ticks == 1
    pending, _ = hold(core, 16)
    assert not pending.has_keycard
    collected, events = hold(core, 1)
    assert collected.has_keycard
    assert events[0].object_id == "b"


@pytest.mark.parametrize("spawn,obj,rows", [
    ((1.5, 2.5), (4.5, 2.5), ("......",) * 6),  # too far
    ((2.5, 2.5), (2.5, 3.5), ("......",) * 6),  # outside yaw cone
    ((2.5, 2.5), (3.5, 2.5), ("...#..",) * 6),  # invalid object in wall
])
def test_invalid_range_angle_and_object_placement(spawn, obj, rows):
    objects = (MapObject("card", "keycard", *obj),)
    if obj == (3.5, 2.5):
        with pytest.raises(ValueError, match="traversable"):
            scene(objects, spawn, rows)
    else:
        core = scene(objects, spawn, rows)
        final, events = hold(core, 30)
        assert not final.has_keycard
        assert final.interaction_target is None
        assert not events


def test_wall_blocks_nearby_interaction_and_yaw_independent_of_rendering():
    rows = ("......", "..#...", "..#...", "......")
    objects = (MapObject("card", "keycard", 3.1, 1.5),)
    core = scene(objects, (1.75, 1.5), rows, CoreConfig(interaction_range=2))
    final, events = hold(core, 30)
    assert not final.has_keycard
    assert final.interaction_target is None
    assert final.interaction_ticks == 0
    assert not events


def test_losing_range_or_wall_visibility_cancels_progress():
    objects = (MapObject("card", "keycard", 3.5, 1.5),)
    core = scene(objects, (3.5, 2.5), ("......", "..#...", "..#...", "......"),
                 CoreConfig(interaction_range=1.1))
    core.step(SurvivorCommand(yaw_delta=-pi / 2))
    hold(core, 10)
    # Actual movement invalidates progress even if the object remains in range.
    snap = core.step(SurvivorCommand(forward=-1, interact=True))
    assert snap.interaction_ticks == 0
    for _ in range(4):
        snap = core.step(SurvivorCommand(forward=-1))
    assert snap.interaction_target is None


def test_deterministic_target_priority_angle_then_distance_then_id():
    config = CoreConfig(interaction_range=3)
    survivor = SurvivorState(2.5, 2.5)
    choices = (MapObject("near", "exit", 3.0, 2.6),
               MapObject("far", "keycard", 4.5, 2.5))
    world = WorldMap(("......",) * 6, ((2.5, 2.5),), choices)
    assert select_target(world, survivor, config, False).object_id == "far"
    world = replace(world, objects=(MapObject("b", "exit", 3.5, 2.5),
                                    MapObject("a", "keycard", 3.5, 2.5)))
    assert select_target(world, survivor, config, False).object_id == "a"
    assert select_target(replace(world, objects=world.objects[::-1]),
                         survivor, config, False).object_id == "a"
    world = replace(world, objects=(MapObject("near", "exit", 3., 2.5),
                                    MapObject("far", "keycard", 4., 2.5)))
    assert select_target(world, survivor, config, False).object_id == "near"


@pytest.mark.parametrize("kwargs", [{"interaction_range": 0}, {"interaction_seconds": 0},
                                    {"interaction_half_angle": 0},
                                    {"interaction_half_angle": 4},
                                    {"interaction_range": float("inf")},
                                    {"interaction_seconds": float("nan")}])
def test_interaction_config_validation(kwargs):
    with pytest.raises(ValueError):
        CoreConfig(**kwargs)


def test_configurable_duration_range_and_angle():
    core = scene(config=CoreConfig(interaction_range=1, interaction_half_angle=.1,
                                  interaction_seconds=.1))
    assert not hold(core, 2)[0].has_keycard
    assert hold(core, 1)[0].has_keycard


def test_objects_and_events_are_immutable_and_copied_from_input():
    objects = [MapObject("card", "keycard", 3.5, 2.5)]
    core = scene(objects)
    objects.clear()
    before = core.snapshot()
    with pytest.raises(FrozenInstanceError):
        before.world_map.objects[0].x = 0
    with pytest.raises(FrozenInstanceError):
        before.has_keycard = True
    after, events = hold(core)
    with pytest.raises(FrozenInstanceError):
        events[0].kind = "changed"
    assert not before.has_keycard
    assert after.has_keycard
    assert replace(after, has_keycard=False) != core.snapshot()


def test_same_small_map_has_reachable_fixed_card_and_exit_and_headless_replay():
    world = small_test_map()
    assert len(world.objects) == 2
    assert {o.kind for o in world.objects} == {"keycard", "exit"}
    first, second = run_replay(), run_replay()
    assert first == second
    assert first["outcome"] == "escaped"
    assert [e["kind"] for e in first["events"]] == ["keycard_collected", "escaped"]
    core = GameCore()
    initial = core.reset(1)
    for command in replay_commands():
        core.step(command)
    assert core.reset(1) == initial
    assert not core.snapshot().has_keycard



def test_no_timeout_win_and_blocked_walk_input_still_cancels_interaction():
    rows = ("......", "......", "..#...", "......", "......")
    core = scene((MapObject("card", "keycard", 3.5, 1.8),), (2.5, 1.8), rows)
    for _ in range(600):
        assert core.step(SurvivorCommand()).outcome is None
    hold(core, 10)
    before = core.snapshot().survivor
    # Strafe into the touching wall: no displacement, but walking intent cancels E.
    cancelled = core.step(SurvivorCommand(strafe=1, interact=True))
    assert cancelled.survivor == before
    assert cancelled.interaction_ticks == 0


def test_line_of_sight_rejects_corner_touch():
    from game.interaction import segment_clear

    world = WorldMap(("....", "..#.", "....", "...."), ((.5, .5),))
    assert not segment_clear(world, 1.5, 1.5, 2.5, 2.5)
    assert segment_clear(world, .5, .5, 1.5, 1.5)



@pytest.mark.parametrize('kind', ['keycard', 'exit'])
def test_explicit_interaction_cancel_is_idempotent_and_does_not_step_world(kind):
    core = scene()
    if kind == 'exit':
        hold(core)
        core.step(SurvivorCommand(yaw_delta=pi / 2))
    hold(core, 10)
    before = core.snapshot()
    expected = replace(before, interaction_object_id=None, interaction_ticks=0)
    assert core.cancel_interaction() == expected
    assert core.cancel_interaction() == expected
    assert core.snapshot() == expected
    assert hold(core, 17)[1] == ()
    assert [e.kind for e in hold(core, 1)[1]] == [
        'keycard_collected' if kind == 'keycard' else 'escaped']
