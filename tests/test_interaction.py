"""Core 1C instant-press contract, independent of graphics and assets."""

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


def press(core, count=1):
    # Each True command is one explicit bot press, not a held keyboard state.
    snapshots = [core.step(SurvivorCommand(interact=True)) for _ in range(count)]
    return snapshots[-1], tuple(e for s in snapshots for e in s.events)



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
        final, events = press(core, 30)
        assert not final.has_keycard
        assert final.interaction_target is None
        assert not events



def test_wall_blocks_nearby_interaction_and_yaw_independent_of_rendering():
    rows = ("......", "..#...", "..#...", "......")
    objects = (MapObject("card", "keycard", 3.1, 1.5),)
    core = scene(objects, (1.75, 1.5), rows, CoreConfig(interaction_range=2))
    final, events = press(core, 30)
    assert not final.has_keycard
    assert final.interaction_target is None
    assert not events



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



def test_objects_and_events_are_immutable_and_copied_from_input():
    objects = [MapObject("card", "keycard", 3.5, 2.5)]
    core = scene(objects)
    objects.clear()
    before = core.snapshot()
    with pytest.raises(FrozenInstanceError):
        before.world_map.objects[0].x = 0
    with pytest.raises(FrozenInstanceError):
        before.has_keycard = True
    after, events = press(core)
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
    core = GameCore(config=CoreConfig(alien_enabled=False))
    initial = core.reset(1)
    for command in replay_commands():
        core.step(command)
    assert core.reset(1) == initial
    assert not core.snapshot().has_keycard



def test_line_of_sight_rejects_corner_touch():
    from game.interaction import segment_clear

    world = WorldMap(("....", "..#.", "....", "...."), ((.5, .5),))
    assert not segment_clear(world, 1.5, 1.5, 2.5, 2.5)
    assert segment_clear(world, .5, .5, 1.5, 1.5)



def test_card_is_immediate_and_never_repeats_after_collection():
    core = scene()
    initial = core.snapshot()
    picked = core.step(SurvivorCommand(interact=True))
    assert picked.tick == 1
    assert picked.has_keycard
    assert [(e.kind, e.object_id, e.tick) for e in picked.events] == [
        ('keycard_collected', 'card', 1)]
    assert not initial.has_keycard
    assert picked.interaction_target is None
    for _ in range(30):
        assert core.step(SurvivorCommand(interact=True)).events == ()


def test_exit_requires_card_and_new_press_then_terminal_freezes_and_resets():
    core = scene()
    core.step(SurvivorCommand(yaw_delta=pi / 2))
    denied = core.step(SurvivorCommand(interact=True))
    assert denied.outcome is None
    assert denied.events == ()
    assert denied.interaction_target.kind == 'exit'
    core.step(SurvivorCommand(yaw_delta=-pi / 2, interact=True))
    escaped = core.step(SurvivorCommand(yaw_delta=pi / 2, interact=True))
    assert escaped.outcome == 'escaped'
    assert [e.kind for e in escaped.events] == ['escaped']
    for _ in range(30):
        assert core.step(SurvivorCommand(1, 1, pi, True, True)) == replace(escaped, events=())
    assert core.reset(0) == scene().snapshot()


def test_one_press_cannot_collect_card_and_open_exit_but_two_bot_presses_can():
    core = scene((MapObject('a', 'keycard', 3.5, 2.5),
                  MapObject('z', 'exit', 3.5, 2.5)))
    picked = core.step(SurvivorCommand(interact=True))
    assert picked.has_keycard and picked.outcome is None
    assert len(picked.events) == 1
    for _ in range(5):
        idle = core.step(SurvivorCommand())
        assert idle.outcome is None and idle.events == ()
    escaped = core.step(SurvivorCommand(interact=True))
    assert escaped.outcome == 'escaped' and len(escaped.events) == 1


def test_failed_press_is_not_deferred_until_target_is_eligible():
    core = scene()
    failed = core.step(SurvivorCommand(yaw_delta=pi, interact=True))
    assert not failed.has_keycard
    restored = core.step(SurvivorCommand(yaw_delta=-pi))
    assert restored.interaction_target.kind == 'keycard'
    assert not restored.has_keycard and not restored.events
    assert core.step(SurvivorCommand(interact=True)).has_keycard


@pytest.mark.parametrize('command', [SurvivorCommand(forward=1, interact=True),
                                    SurvivorCommand(strafe=1, interact=True)])
def test_movement_restriction_is_unchanged(command):
    core = scene()
    assert not core.step(command).has_keycard
    assert not core.step(SurvivorCommand()).has_keycard
    assert core.step(SurvivorCommand(interact=True)).has_keycard


@pytest.mark.parametrize('kwargs', [{'interaction_range': 0}, {'interaction_half_angle': 0},
                                    {'interaction_half_angle': 4},
                                    {'interaction_range': float('inf')}])
def test_interaction_config_validation(kwargs):
    with pytest.raises(ValueError):
        CoreConfig(**kwargs)


def test_configurable_range_and_angle_still_gate_instant_press():
    core = scene(config=CoreConfig(interaction_range=.9, interaction_half_angle=.1))
    assert not core.step(SurvivorCommand(interact=True)).has_keycard
    core = scene(config=CoreConfig(interaction_range=1, interaction_half_angle=.1))
    assert core.step(SurvivorCommand(interact=True)).has_keycard


def test_no_timeout_win_and_obsolete_progress_fields_are_removed():
    core = scene()
    for _ in range(600):
        assert core.step(SurvivorCommand()).outcome is None
    assert not hasattr(core.snapshot(), 'interaction_ticks')
    assert not hasattr(core.snapshot(), 'interaction_object_id')
    assert not hasattr(core.config, 'interaction_seconds')



@pytest.mark.parametrize('kind', ['keycard', 'exit'])
def test_human_tap_and_bot_press_have_identical_command_and_core_semantics(kind):
    from app.input import InputState

    human, bot = scene(), scene()
    if kind == 'exit':
        for core in (human, bot):
            core.step(SurvivorCommand(interact=True))
            core.step(SurvivorCommand(yaw_delta=pi / 2))
    inputs = InputState()
    inputs.key('e', True)
    inputs.key('e', False)
    command = inputs.consume(1 / 30)
    assert command == SurvivorCommand(interact=True)
    assert human.step(command) == bot.step(SurvivorCommand(interact=True))
