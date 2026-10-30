"""Known-exit interaction standpoints use real Core rules and filtered threat memory."""

from dataclasses import replace
from math import atan2, hypot, pi

import numpy as np
import pytest

from game.core import GameCore
from game.map import MapObject, WorldMap
from game.state import SurvivorCommand
from rl.bots.evasive import EvasiveExplorerBot
from rl.bots.views import SurvivorThreatView, SurvivorView, observe_threat


def fixture():
    world = WorldMap(
        ("." * 12,) * 10,
        ((3.5, 5.5),),
        (MapObject("card", "keycard", 3.5, 5.5), MapObject("exit", "exit", 6.5, 5.5)),
        alien_spawn=(6.5, 5.5, pi),
    )
    core = GameCore(world, alien_control="external")
    bot = EvasiveExplorerBot(world.rows, core.config)
    bot.reset(np.random.default_rng(4))
    return world, core, bot


def filtered(world, core, bot):
    s = core.snapshot()
    return observe_threat(world, s.survivor, s.has_keycard, s.alien, bot.settings)


def test_real_card_pickup_then_safe_known_exit_escapes_stationary_alien():
    world, core, bot = fixture()
    pressed = 0
    for _ in range(120):
        before = core.snapshot()
        command = bot.command(filtered(world, core, bot))
        if command.interact:
            pressed += 1
            if before.has_keycard:
                assert (
                    hypot(before.survivor.x - 6.5, before.survivor.y - 5.5)
                    <= core.config.interaction_range
                )
                angle = abs(
                    (
                        atan2(5.5 - before.survivor.y, 6.5 - before.survivor.x)
                        - before.survivor.yaw
                        + pi
                    )
                    % (2 * pi)
                    - pi
                )
                assert angle <= core.config.interaction_half_angle
        after = core.step(command)
        assert not world.overlaps(after.survivor.x, after.survivor.y, core.config.radius)
        assert (
            hypot(after.survivor.x - after.alien.x, after.survivor.y - after.alien.y)
            >= bot.settings.threat_clearance - 1e-9
        )
        if after.outcome:
            break
    assert after.outcome == "escaped" and after.has_keycard and pressed == 2
    assert bot.exit_approach_steps > 0
    assert bot.exit_standpoint != (6.5, 5.5)


@pytest.mark.parametrize("card,known", [(False, True), (True, False)])
def test_no_card_or_unknown_exit_never_uses_shortcut(card, known):
    _, _, bot = fixture()
    goal = MapObject("exit", "exit", 6.5, 5.5)
    mission = SurvivorView(3.5, 5.5, 0, card, (goal,) if known else ())
    command = bot.command(SurvivorThreatView(mission, (6.5, 5.5)))
    assert not bot.exit_approach_active and bot.exit_approach_attempts == 0
    assert command.running and not command.interact


def test_visible_threat_moving_onto_route_cancels_and_retreats_same_tick():
    world, core, bot = fixture()
    core._has_keycard = True
    first = bot.command(filtered(world, core, bot))
    core.step(first)
    assert bot.exit_approach_active
    core._alien = replace(core.snapshot().alien, x=4.5, y=5.5)
    before = core.snapshot()
    command = bot.command(filtered(world, core, bot))
    assert not bot.exit_approach_active and bot.exit_approach_cancellations == 1
    assert command.running and not command.interact
    assert bot.exit_retry_tick > bot.tick
    after = core.step(command)
    assert hypot(after.survivor.x - 4.5, after.survivor.y - 5.5) > hypot(
        before.survivor.x - 4.5, before.survivor.y - 5.5
    )


def test_at_safe_standpoint_turns_then_interacts_without_walking_to_center():
    world, core, bot = fixture()
    core._has_keycard = True
    core._survivor = replace(core.snapshot().survivor, x=5.5, y=5.5, yaw=pi)
    bot.explorer.known["exit"] = world.objects[1]  # prior filtered mission observation
    bot.last_seen = (6.5, 5.5)
    bot.last_seen_age_ticks = 0
    first = bot.command(filtered(world, core, bot))
    assert not first.interact and first.forward == first.strafe == 0
    core.step(first)
    for _ in range(60):
        command = bot.command(filtered(world, core, bot))
        after = core.step(command)
        assert (after.survivor.x, after.survivor.y) == (5.5, 5.5)
        if after.outcome:
            break
    assert after.outcome == "escaped"


def test_wall_and_range_block_interaction_and_no_route_does_not_force_exit():
    world, core, bot = fixture()
    goal = world.objects[1]
    bot.explorer.known["exit"] = goal
    assert not bot._ready_interaction(SurvivorView(3.5, 5.5, 0, True, ()))
    blocked = WorldMap(
        tuple(".....#......" for _ in range(10)), ((4.5, 5.5),), (goal,), alien_spawn=(6.5, 5.5, pi)
    )
    blocked_core = GameCore(blocked, alien_control="external")
    blocked_core._has_keycard = True
    blocked_bot = EvasiveExplorerBot(blocked.rows, blocked_core.config)
    blocked_bot.reset(np.random.default_rng(4))
    blocked_bot.explorer.known["exit"] = goal
    blocked_bot.last_seen = (6.5, 5.5)
    blocked_bot.last_seen_age_ticks = 0
    command = blocked_bot.command(filtered(blocked, blocked_core, blocked_bot))
    assert not command.interact and not blocked_bot.exit_approach_active
    assert not blocked_bot._ready_interaction(SurvivorView(4.5, 5.5, 0, True, ()))
    # Threat occupies the would-be stand area; no unsafe completion shortcut.
    bot.last_seen = (5.5, 5.5)
    bot.last_seen_age_ticks = 0
    assert not bot._exit_path_safe((3.5, 5.5), ((5.5, 5.5),), (5.5, 5.5), goal)


def test_hidden_motion_does_not_change_approach_or_refresh_memory():
    world, _, _ = fixture()
    bots = []
    for _ in range(2):
        b = EvasiveExplorerBot(world.rows, fixture()[1].config)
        b.reset(np.random.default_rng(4))
        b.command(
            SurvivorThreatView(SurvivorView(3.5, 5.5, 0, True, (world.objects[1],)), (6.5, 5.5))
        )
        bots.append(b)
    # Both real threats are outside the Survivor's FOV: filtered observations are identical.
    from game.state import AlienState, SurvivorState

    views = [
        observe_threat(world, SurvivorState(3.5, 5.5, 0), True, AlienState(*p), b.settings)
        for p, b in zip(((1.5, 5.5), (1.5, 7.5)), bots, strict=True)
    ]
    assert views[0] == views[1]
    assert bots[0].command(views[0]) == bots[1].command(views[1])
    assert bots[0].last_seen == bots[1].last_seen == (6.5, 5.5)
    assert bots[0].last_seen_age_ticks == 1


def test_reset_clears_approach_and_unsafe_current_position_cannot_press_exit():
    world, _, bot = fixture()
    bot.command(
        SurvivorThreatView(SurvivorView(3.5, 5.5, 0, True, (world.objects[1],)), (6.5, 5.5))
    )
    assert bot.exit_approach_active
    bot.last_seen = (5.9, 5.5)
    assert not bot._ready_interaction(SurvivorView(5.5, 5.5, 0, True, ()))
    bot.reset(np.random.default_rng(4))
    assert not bot.exit_approach_active and bot.exit_standpoint is None
    assert bot.exit_goal_id is None and bot.exit_navigator.path == ()
    assert bot.exit_approach_steps == bot.exit_approach_cancellations == 0
    assert bot.exit_motion_audit is None


def test_wall_corner_blocks_e_even_when_exit_is_in_range_and_facing_is_correct():
    rows = [list("." * 12) for _ in range(10)]
    rows[5][5] = "#"
    goal = MapObject("exit", "exit", 5.2, 4.8)
    world = WorldMap(
        tuple("".join(r) for r in rows), ((4.8, 5.2),), (goal,), alien_spawn=(8.5, 4.5, pi)
    )
    core = GameCore(world, alien_control="external")
    core._has_keycard = True
    core._survivor = replace(core.snapshot().survivor, yaw=-pi / 4)
    bot = EvasiveExplorerBot(world.rows, core.config)
    bot.reset(np.random.default_rng(4))
    bot.explorer.known["exit"] = goal
    bot.last_seen = (8.5, 4.5)
    bot.last_seen_age_ticks = 0
    pose = core.snapshot().survivor
    assert hypot(pose.x - goal.x, pose.y - goal.y) < core.config.interaction_range
    assert not bot._ready_interaction(SurvivorView(pose.x, pose.y, pose.yaw, True, ()))
    result = core.step(SurvivorCommand(interact=True))
    assert result.outcome is None and not result.events


@pytest.mark.parametrize("threat", [(9.5, 6.5), (7.5, 4.5)])
@pytest.mark.parametrize("hidden_alien", [(9.5, 7.5), (10.5, 7.5)])
def test_cancel_after_approach_replans_old_side_flee_route_without_wall_stall(hidden_alien, threat):
    from ai.navigation import find_path
    from game.map import small_test_map
    from game.motion import segment_has_clearance

    world = small_test_map()
    core = GameCore(world, alien_control="external")
    core._has_keycard = True
    core._survivor = replace(core.snapshot().survivor, x=5.5, y=3.5, yaw=0)
    core._alien = replace(core.snapshot().alien, x=threat[0], y=threat[1])
    bot = EvasiveExplorerBot(world.rows, core.config)
    bot.reset(np.random.default_rng(4))
    goal = world.objects[1]
    bot.explorer.known[goal.object_id] = goal
    bot.fleeing = True
    bot.flee_started = 0
    bot.flee_entries = 1
    bot.escape_target = (1.5, 3.5)
    bot.navigator.target = bot.escape_target
    bot.navigator.path = find_path(world, (3.5, 3.5), bot.escape_target, core.config.radius)
    assert bot.navigator.path == ((1.5, 3.5),)
    assert not segment_has_clearance(world, (5.5, 3.5), bot.escape_target, core.config.radius)
    bot.last_plan = bot.tick  # hold timer must not delay the required replan
    bot.exit_approach_active = True
    bot.exit_goal_id = goal.object_id
    bot.exit_standpoint = (9.5, 6.5)
    bot.exit_navigator.target = bot.exit_standpoint
    bot.exit_navigator.path = find_path(world, (5.5, 3.5), bot.exit_standpoint, core.config.radius)
    if threat == (9.5, 6.5):
        # Exact review fixture injects filtered input: this point is occluded.
        pose = core.snapshot().survivor
        view = SurvivorThreatView(SurvivorView(pose.x, pose.y, pose.yaw, True, ()), threat)
    else:
        # End-to-end variant: the sensor really sees a threat on the approach route.
        view = filtered(world, core, bot)
        assert view.visible_alien == threat
    command = bot.command(view)
    assert not bot.exit_approach_active and bot.exit_approach_cancellations == 1
    assert bot.replans == 1 and bot.last_plan == bot.tick
    assert bot.flee_entries == 1 and bot.flee_started == 0
    assert command.running and not command.interact
    points = ((5.5, 3.5), *bot.navigator.path)
    assert bot.navigator.path
    assert all(
        segment_has_clearance(world, a, b, core.config.radius) for a, b in zip(points, points[1:])
    )
    initial = core.snapshot().survivor
    after = core.step(command)
    # Move the real Alien behind the camera. Only the identical filtered memory
    # may influence subsequent commands, regardless of its hidden position.
    core._alien = replace(core.snapshot().alien, x=hidden_alien[0], y=hidden_alien[1])
    for _ in range(11):
        view = filtered(world, core, bot)
        assert view.visible_alien is None
        command = bot.command(view)
        after = core.step(command)
        assert after.outcome is None
        assert not world.overlaps(after.survivor.x, after.survivor.y, core.config.radius)
    assert hypot(after.survivor.x - initial.x, after.survivor.y - initial.y) > 1.0
    assert after.survivor.y < 3.0  # routes above the wall, rather than stalling at x=5.2
    assert bot.last_seen == threat and bot.last_seen_age_ticks == 11


@pytest.mark.parametrize("unsafe", [False, True])
def test_all_active_approach_clear_paths_invalidate_retreat_cache(unsafe):
    _, _, bot = fixture()
    bot.fleeing = True
    bot.tick = bot.last_plan = 20
    bot.flee_started = 2
    bot.flee_entries = 1
    bot.exit_approach_active = True
    bot.escape_target = (1.5, 3.5)
    bot.navigator.target = bot.escape_target
    bot.navigator.path = (bot.escape_target,)
    bot._clear_exit_approach(unsafe=unsafe)
    assert bot.escape_target is None
    assert bot.navigator.target is None and bot.navigator.path == ()
    assert bot.tick - bot.last_plan >= bot.hold_ticks
    assert bot.fleeing and bot.flee_started == 2 and bot.flee_entries == 1
