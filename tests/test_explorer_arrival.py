"""Arrival labels/commands using filtered inputs only; never run physics."""

from math import hypot, pi

import numpy as np
import pytest

from game.interaction import select_target
from game.map import MapObject, WorldMap
from game.state import CoreConfig, SurvivorCommand, SurvivorState
from rl.bots.evasive import EvasiveExplorerBot
from rl.bots.explorer import ExplorerBot
from rl.bots.views import SurvivorThreatView, SurvivorView

ROWS = ("########",) + ("#......#",) * 5 + ("########",)
HERE = (2.5, 2.5)


def explorer(rows=ROWS):
    bot = ExplorerBot(rows, CoreConfig())
    bot.reset(np.random.default_rng(4))
    bot.scanning = 0
    bot.order = (HERE, (5.5, 2.5))
    return bot


@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_new_and_existing_public_arrival_share_scan_and_visited(existing, projected):
    bot = explorer()
    if existing:
        bot.explore_target = HERE
    command = bot.command(SurvivorView(*HERE, pi, False, ()), move_while_turning=projected)
    assert command == SurvivorCommand(yaw_delta=bot.turn_per_tick)
    assert bot.last_command_reason == "arrived"
    assert bot.visited == {HERE} and bot.explore_target is None
    assert bot.scanning == bot.scan_ticks - 1
    # Every arrival scan remains bounded by the original count, no next-target shortcut.
    for _ in range(bot.scan_ticks - 1):
        assert bot.command(SurvivorView(*HERE, pi, False, ())) == command
        assert bot.last_command_reason == "scan" and bot.explore_target is None
    assert bot.scanning == 0
    bot.command(SurvivorView(*HERE, 0, False, ()))
    assert bot.explore_target == (5.5, 2.5) and HERE in bot.visited


@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_near_but_not_arrived_still_walks(existing, projected):
    bot = explorer()
    target = (HERE[0] + 1e-7, HERE[1])
    bot.order = (target,)
    if existing:
        bot.explore_target = target
    command = bot.command(SurvivorView(*HERE, 0, False, ()), move_while_turning=projected)
    assert bot.last_command_reason == "aligned" and bot.scanning == 0
    assert command.forward > 0 and not command.interact and not command.running
    assert hypot(command.forward, command.strafe) <= 1
    assert not bot.visited


@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize("mission", [False, True])
def test_disconnected_target_is_still_no_path(projected, mission):
    rows = ("########",) + ("#..#...#",) * 5 + ("########",)
    bot = explorer(rows)
    target = (5.5, 2.5)
    bot.order = (target,)
    if mission:
        goal = MapObject("previously_seen", "keycard", *target)
        bot.known[goal.object_id] = goal
    command = bot.command(SurvivorView(*HERE, 0, False, ()), move_while_turning=projected)
    assert command == SurvivorCommand() and bot.last_command_reason == "no_path"
    assert bot.visited == (set() if mission else {target})
    assert bot.explore_target is None and bot.scanning == 0
    if mission:
        assert bot.known == {goal.object_id: goal}


@pytest.mark.parametrize("kind,card", [("keycard", False), ("exit", True)])
@pytest.mark.parametrize("projected", [False, True])
def test_tiny_mission_segment_still_requires_yaw_before_press(kind, card, projected):
    bot = explorer()
    goal = MapObject("known", kind, HERE[0] + 1e-10, HERE[1])
    world = WorldMap(ROWS, (HERE,), (goal,))
    bot.known[goal.object_id] = goal  # discovered earlier; currently behind the camera
    view = SurvivorView(*HERE, pi, card, ())
    assert select_target(world, SurvivorState(*HERE, pi), bot.config, card) is None
    command = bot.command(view, move_while_turning=projected)
    assert bot.last_command_reason == "arrived"
    assert command == SurvivorCommand(yaw_delta=-bot.turn_per_tick)
    assert not bot.visited and bot.known == {goal.object_id: goal} and bot.scanning == 0
    # A later *synthetic input* faces the object. No displacement or physics replay.
    assert bot.command(SurvivorView(*HERE, 0, card, ())) == SurvivorCommand(interact=True)
    assert bot.last_command_reason == "interaction"


@pytest.mark.parametrize("yaw", [0, pi / 2, pi])
def test_exact_object_position_uses_existing_core_zero_distance_rule(yaw):
    bot = explorer()
    goal = MapObject("known", "keycard", *HERE)
    world = WorldMap(ROWS, (HERE,), (goal,))
    assert select_target(world, SurvivorState(*HERE, yaw), bot.config, False) == goal
    assert bot.command(SurvivorView(*HERE, yaw, False, (goal,))) == SurvivorCommand(interact=True)
    assert bot.last_command_reason == "interaction" and not bot.visited


@pytest.mark.parametrize("kind", ["public", "unreachable", "tiny_mission", "ready"])
def test_resume_keeps_original_end_contract_and_deadline(kind):
    rows = ("########",) + ("#..#...#",) * 5 + ("########",) if kind == "unreachable" else ROWS
    bot = EvasiveExplorerBot(rows, CoreConfig())
    bot.reset(np.random.default_rng(4))
    bot.fleeing = True
    bot.last_seen = (5.5, 4.5)
    bot.last_seen_age_ticks = 59
    bot.explorer.order = ((5.5, 2.5),) if kind == "unreachable" else (HERE, (5.5, 2.5))
    objects = ()
    if kind in ("tiny_mission", "ready"):
        objects = (MapObject("known", "keycard", HERE[0] + 1e-10, HERE[1]),)
    yaw = 0 if kind == "ready" else pi
    if kind == "tiny_mission":
        bot.explorer.known[objects[0].object_id] = objects[0]
        objects = ()  # remembered objective, not a newly visible object behind the camera
    command = bot.command(SurvivorThreatView(SurvivorView(*HERE, yaw, False, objects)))
    reason = (
        "no_path"
        if kind == "unreachable"
        else "interaction_ready"
        if kind == "ready"
        else "arrived"
    )
    assert not bot.fleeing and not bot.resume_active and bot.resume_end_reason == reason
    assert bot.resume_entry_reason == "memory_expired"
    assert bot.resume_started_tick == bot.resume_ended_tick == bot.tick
    assert bot.resume_deadline_tick == bot.resume_started_tick + 45
    assert bot.last_seen is bot.last_seen_age_ticks is None
    if kind == "public":
        assert command == SurvivorCommand(yaw_delta=bot.explorer.turn_per_tick)
        assert bot.explorer.scanning == bot.explorer.scan_ticks - 1
    elif kind == "tiny_mission":
        assert command == SurvivorCommand(yaw_delta=-bot.explorer.turn_per_tick)
    elif kind == "ready":
        assert command == SurvivorCommand(interact=True)
    else:
        assert command == SurvivorCommand()
