"""Pure RESUME state/motion contracts. Never construct or step Core/Env."""

from math import cos, hypot, pi, sin

import numpy as np
import pytest
from test_flee_resume import arena, retreating

from game.map import MapObject
from game.motion import segment_has_clearance
from game.state import AlienState, CoreConfig, SurvivorCommand, SurvivorState
from rl.evasive_survivor import (
    EvasionConfig,
    EvasiveExplorerBot,
    SurvivorThreatView,
    observe_threat,
)
from rl.survivor_bots import ExplorerBot, SurvivorView


def mission(yaw=pi / 2, objects=(), has_keycard=False):
    return SurvivorView(2.5, 5.5, yaw, has_keycard, objects)


def stub_turning(subject, monkeypatch):
    def command(view, **kwargs):
        subject.explorer.last_command_reason = "turning"
        return SurvivorCommand(yaw_delta=subject.explorer.turn_per_tick)

    monkeypatch.setattr(subject.explorer, "command", command)


def begin_resume(monkeypatch):
    subject = retreating()
    stub_turning(subject, monkeypatch)
    subject.last_seen_age_ticks = 59
    subject.command(SurvivorThreatView(mission()))
    assert subject.resume_active and not subject.fleeing
    return subject


@pytest.mark.parametrize("age", [0, 32, 58])
def test_stale_point_distance_is_not_fresh_separation(age):
    subject = retreating()
    subject.last_seen_age_ticks = age
    subject.command(SurvivorThreatView(mission()))
    assert subject.fleeing and not subject.resume_active
    assert subject.last_seen == (9.5, 5.5) and subject.last_seen_age_ticks == age + 1


@pytest.mark.parametrize("elapsed,released", [(29, False), (30, True)])
def test_fresh_visible_distance_uses_original_minimum_hold(elapsed, released):
    subject = retreating()
    subject.tick = subject.flee_started + elapsed - 1
    subject.command(SurvivorThreatView(mission(yaw=0), (9.5, 5.5)))
    assert subject.fleeing is not released
    assert subject.last_seen == (9.5, 5.5) and subject.last_seen_age_ticks == 0
    if released:
        assert subject.resume_entry_reason == "visible_separation"


def test_expiry_clears_coordinates_and_releases_even_before_hold():
    subject = retreating()
    subject.tick = subject.flee_started
    subject.last_seen_age_ticks = 59
    subject.command(SurvivorThreatView(mission()))
    assert not subject.fleeing
    assert subject.last_seen is None and subject.last_seen_age_ticks is None
    assert subject.resume_entry_reason == "memory_expired"
    for _ in range(4):
        subject.command(SurvivorThreatView(mission()))
        assert subject.last_seen is None and not subject.fleeing


def test_deadline_counts_entry_and_does_not_restart_on_target_changes(monkeypatch):
    subject = begin_resume(monkeypatch)
    start, deadline = subject.resume_started_tick, subject.resume_deadline_tick
    assert subject.resume_limit_ticks == 45 and deadline == start + 45
    for i in range(44):
        subject.explorer.explore_target = (1.5, 1.5) if i % 2 else (4.5, 4.5)
        subject._start_resume("must_not_restart")
        subject.command(SurvivorThreatView(mission()))
        assert subject.resume_active and subject.resume_deadline_tick == deadline
        assert subject.resume_entry_reason == "memory_expired"
    assert subject.tick == deadline - 1
    subject.command(SurvivorThreatView(mission()))
    assert not subject.resume_active and subject.resume_end_reason == "deadline"
    assert subject.resume_ended_tick == deadline
    for _ in range(3):
        subject.command(SurvivorThreatView(mission()))
    assert not subject.resume_active and subject.resume_started_tick == start


@pytest.mark.parametrize("early_return", ["approach", "interaction_on_deadline"])
def test_deadline_precedes_all_early_returns(monkeypatch, early_return):
    subject = begin_resume(monkeypatch)
    deadline = subject.resume_deadline_tick
    if early_return == "approach":
        monkeypatch.setattr(subject, "_exit_approach", lambda view: SurvivorCommand(yaw_delta=0.01))
    else:
        monkeypatch.setattr(subject, "_ready_interaction", lambda view: subject.tick >= deadline)
    for _ in range(45):
        subject.command(SurvivorThreatView(mission()))
    assert subject.tick == deadline and not subject.resume_active
    assert subject.resume_end_reason == "deadline" and subject.resume_ended_tick == deadline


def test_ready_interaction_ends_resume_before_approach(monkeypatch):
    subject = begin_resume(monkeypatch)
    card = MapObject("observed_card", "keycard", 2.5, 6.0)
    calls = []
    monkeypatch.setattr(subject, "_exit_approach", lambda view: calls.append("unexpected"))
    command = subject.command(SurvivorThreatView(mission(objects=(card,))))
    assert command.interact and command.forward == command.strafe == 0
    assert not subject.resume_active and subject.resume_end_reason == "interaction_ready"
    assert not calls and subject.explorer.known[card.object_id] == card


def test_exit_without_card_never_becomes_interaction(monkeypatch):
    subject = begin_resume(monkeypatch)
    exit = MapObject("observed_exit", "exit", 2.5, 6.0)
    command = subject.command(SurvivorThreatView(mission(objects=(exit,))))
    assert not command.interact
    assert subject.explorer.known[exit.object_id] == exit


def test_safe_exit_approach_keeps_priority_and_cannot_reset_clock(monkeypatch):
    subject = begin_resume(monkeypatch)
    start, deadline = subject.resume_started_tick, subject.resume_deadline_tick
    expected = SurvivorCommand(forward=0.2)
    monkeypatch.setattr(subject, "_exit_approach", lambda view: expected)
    monkeypatch.setattr(
        subject.explorer, "command", lambda *a, **k: pytest.fail("Approach lost priority")
    )
    assert subject.command(SurvivorThreatView(mission())) == expected
    assert (
        subject.resume_active
        and subject.resume_started_tick == start
        and subject.resume_deadline_tick == deadline
    )


def test_unsafe_exit_approach_flee_transition_cancels_resume(monkeypatch):
    subject = begin_resume(monkeypatch)

    def unsafe(view):
        subject._begin_flee()
        return SurvivorCommand()

    monkeypatch.setattr(subject, "_exit_approach", unsafe)
    subject.command(SurvivorThreatView(mission()))
    assert subject.fleeing and not subject.resume_active
    assert subject.resume_end_reason == "flee_entered"


@pytest.mark.parametrize("distance,should_flee", [(2.0, True), (5.0, False), (7.0, False)])
def test_current_threat_checked_before_resume_movement(monkeypatch, distance, should_flee):
    subject = begin_resume(monkeypatch)
    start = subject.resume_started_tick
    subject.command(SurvivorThreatView(mission(yaw=0), (2.5 + distance, 5.5)))
    assert subject.fleeing == should_flee
    assert subject.last_seen_age_ticks == 0 and subject.last_seen == (2.5 + distance, 5.5)
    if should_flee:
        assert not subject.resume_active and subject.resume_end_reason == "flee_entered"
    else:
        assert subject.resume_active and subject.resume_started_tick == start


def test_aligned_route_finishes_resume_on_entry_command():
    subject = retreating()
    subject.last_seen_age_ticks = 59
    command = subject.command(SurvivorThreatView(mission(yaw=pi)))
    assert command.forward > 0
    assert not subject.resume_active and subject.resume_end_reason == "aligned"
    assert subject.resume_started_tick == subject.resume_ended_tick == subject.tick


def test_arrival_finishes_resume_and_preserves_arrival_scan():
    subject = retreating()
    subject.last_seen_age_ticks = 59
    subject.command(SurvivorThreatView(mission(yaw=pi / 2)))
    assert subject.resume_active
    view = SurvivorView(1.5, 5.5, pi / 2, False, ())
    command = subject.command(SurvivorThreatView(view))
    assert not subject.resume_active and subject.resume_end_reason == "arrived"
    assert subject.explorer.scanning == subject.explorer.scan_ticks - 1
    assert command.forward == command.strafe == 0


@pytest.mark.parametrize("failure", ["no_path", "no_target"])
def test_route_failure_exits_resume_without_reentry_loop(failure):
    rows = list(arena().rows)
    if failure == "no_path":
        rows = [row[:6] + "#" + row[7:] for row in rows]
    subject = EvasiveExplorerBot(tuple(rows), CoreConfig())
    subject.reset(np.random.default_rng(4))
    subject.fleeing = True
    subject.last_seen = (9.5, 5.5)
    subject.last_seen_age_ticks = 59
    subject.explorer.order = ((8.5, 5.5),) if failure == "no_path" else ()
    command = subject.command(SurvivorThreatView(mission(yaw=0)))
    assert (
        not subject.fleeing and not subject.resume_active and subject.resume_end_reason == failure
    )
    assert command.forward == command.strafe == 0
    start = subject.resume_started_tick
    subject.command(SurvivorThreatView(mission(yaw=0)))
    assert not subject.resume_active and subject.resume_started_tick == start


def test_reset_clears_all_transition_and_threat_state(monkeypatch):
    subject = begin_resume(monkeypatch)
    subject.command(SurvivorThreatView(mission()))
    subject.reset(np.random.default_rng(4))
    assert not subject.resume_active and not subject.fleeing and subject.tick == 0
    assert (
        subject.resume_started_tick
        is subject.resume_deadline_tick
        is subject.resume_ended_tick
        is None
    )
    assert subject.resume_entry_reason is None and subject.resume_end_reason == "reset"
    assert subject.last_seen is subject.last_seen_age_ticks is None
    assert not subject.explorer.known and subject.explorer.scanning == subject.explorer.scan_ticks


@pytest.mark.parametrize("yaw", [n * pi / 4 for n in range(8)])
@pytest.mark.parametrize(
    "vector", [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
)
def test_projected_walk_preserves_global_direction_speed_and_turn_limit(yaw, vector):
    subject = ExplorerBot(arena().rows, CoreConfig())
    subject.reset(np.random.default_rng(4))
    subject.scanning = 0
    subject.order = ((5.5 + vector[0], 5.5 + vector[1]),)
    view = SurvivorView(5.5, 5.5, yaw, False, ())
    command = subject.command(view, move_while_turning=True)
    assert not command.running and not command.interact
    assert hypot(command.forward, command.strafe) <= 1 + 1e-12
    assert abs(command.yaw_delta) <= subject.turn_per_tick + 1e-12
    turned = yaw + command.yaw_delta
    dx = cos(turned) * command.forward - sin(turned) * command.strafe
    dy = sin(turned) * command.forward + cos(turned) * command.strafe
    length = hypot(*vector)
    assert (dx, dy) == pytest.approx((vector[0] / length, vector[1] / length), abs=1e-12)
    step = subject.config.walk_speed / subject.config.tick_hz
    assert segment_has_clearance(
        subject.geometry,
        (view.x, view.y),
        (view.x + dx * step, view.y + dy * step),
        subject.config.radius,
    )


def test_default_no_threat_exploration_keeps_original_command_mode():
    evader = EvasiveExplorerBot(arena().rows, CoreConfig())
    explorer = ExplorerBot(arena().rows, CoreConfig())
    for b in (evader, explorer):
        b.reset(np.random.default_rng(4))
    for t in range(100):
        view = SurvivorView(2.5, 5.5, t * 0.03, False, ())
        assert evader.command(SurvivorThreatView(view)) == explorer.command(view)
        assert not evader.resume_active and not evader.fleeing


@pytest.mark.parametrize("card,known", [(False, True), (True, False)])
def test_real_exit_approach_needs_inventory_and_discovered_exit(card, known):
    subject = retreating()
    goal = MapObject("exit_goal", "exit", 6.5, 5.5)
    command = subject.command(
        SurvivorThreatView(
            mission(yaw=0, objects=(goal,) if known else (), has_keycard=card), (9.5, 5.5)
        )
    )
    assert not subject.fleeing and not subject.exit_approach_active
    assert subject.exit_approach_attempts == 0 and not command.interact
    assert (goal.object_id in subject.explorer.known) == known


def test_real_safe_exit_approach_priority_and_deadline():
    subject = retreating()
    goal = MapObject("exit_goal", "exit", 6.5, 5.5)
    view = SurvivorThreatView(mission(yaw=0, objects=(goal,), has_keycard=True), (9.5, 5.5))
    command = subject.command(view)
    assert subject.resume_active and subject.exit_approach_active
    assert subject.exit_approach_steps > 0 and command.forward > 0 and not command.interact
    deadline = subject.resume_deadline_tick
    # Repeated identical filtered own-state inputs test early-return accounting only.
    # No Core/Env, displacement application or physical episode occurs here.
    for _ in range(45):
        command = subject.command(view)
        assert not command.interact
    assert subject.tick == deadline and not subject.resume_active
    assert subject.resume_end_reason == "deadline" and subject.exit_approach_active


def test_real_new_threat_cancels_exit_approach_and_resume():
    subject = retreating()
    goal = MapObject("exit_goal", "exit", 6.5, 5.5)
    view = mission(yaw=0, objects=(goal,), has_keycard=True)
    subject.command(SurvivorThreatView(view, (9.5, 5.5)))
    assert subject.resume_active and subject.exit_approach_active
    command = subject.command(SurvivorThreatView(view, (3.0, 5.5)))
    assert subject.fleeing and not subject.resume_active and not subject.exit_approach_active
    assert subject.resume_end_reason == "flee_entered" and not command.interact
    assert subject.exit_approach_cancellations > 0


def test_known_exit_behind_corner_cannot_force_press_during_resume():
    rows = list(arena().rows)
    rows[5] = rows[5][:6] + "#" + rows[5][7:]
    subject = EvasiveExplorerBot(tuple(rows), CoreConfig())
    subject.reset(np.random.default_rng(4))
    goal = MapObject("known_exit", "exit", 6.2, 6.2)
    subject.explorer.known[goal.object_id] = goal  # a previously discovered object
    subject.fleeing = True
    subject.last_seen = (9.5, 5.5)
    subject.last_seen_age_ticks = 59
    command = subject.command(SurvivorThreatView(SurvivorView(5.5, 5.5, pi / 4, True, ())))
    assert not command.interact
    assert subject.last_seen is None and subject.resume_entry_reason == "memory_expired"


def test_hidden_world_invariance_during_active_resume(monkeypatch):
    subjects = [begin_resume(monkeypatch), begin_resume(monkeypatch)]
    worlds = [arena((MapObject("hidden_card", "keycard", x, 5.5),)) for x in (6.5, 12.5)]
    own = SurvivorState(2.5, 5.5, pi)
    views = [
        observe_threat(w, own, False, AlienState(x, 5.5), EvasionConfig())
        for w, x in zip(worlds, (9.5, 10.5))
    ]
    assert (
        views[0] == views[1]
        and not views[0].mission.visible_objects
        and views[0].visible_alien is None
    )
    deadlines = [b.resume_deadline_tick for b in subjects]
    assert subjects[0].command(views[0]) == subjects[1].command(views[1])
    for b, d in zip(subjects, deadlines):
        assert b.resume_active and b.resume_deadline_tick == d
        assert b.last_seen is None and b.last_seen_age_ticks is None
        assert not b.explorer.known


def test_newly_discovered_mission_changes_target_without_restarting_resume():
    subject = retreating()
    subject.last_seen_age_ticks = 59
    own = mission(yaw=pi / 8)
    subject.command(SurvivorThreatView(own))
    assert subject.resume_active
    start, deadline = subject.resume_started_tick, subject.resume_deadline_tick
    card = MapObject("discovered_card", "keycard", 5.5, 5.5)
    command = subject.command(SurvivorThreatView(mission(yaw=pi / 8, objects=(card,))))
    assert not command.interact and subject.resume_active
    assert subject.explorer.navigator.target == (card.x, card.y)
    assert subject.resume_started_tick == start and subject.resume_deadline_tick == deadline
