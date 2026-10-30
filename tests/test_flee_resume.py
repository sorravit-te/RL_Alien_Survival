"""Flee-to-mission handoff regressions: filtered inputs, no Core scenarios."""

from math import hypot, pi

import numpy as np
import pytest

from game.core import GameCore
from game.map import MapObject, WorldMap
from game.state import AlienState, CoreConfig, SurvivorState
from rl.bots.evasive import EvasionConfig, EvasiveExplorerBot
from rl.bots.explorer import ExplorerBot
from rl.bots.views import SurvivorThreatView, SurvivorView, observe_threat


@pytest.fixture(autouse=True)
def no_core_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Pure handoff regressions must not construct or step Core")

    monkeypatch.setattr(GameCore, "__init__", forbidden)
    monkeypatch.setattr(GameCore, "step", forbidden)


def arena(objects=()):
    return WorldMap(
        ("#" * 16,) + ("#" + "." * 14 + "#",) * 10 + ("#" * 16,), ((2.5, 5.5),), objects
    )


def retreating():
    subject = EvasiveExplorerBot(arena().rows, CoreConfig())
    subject.reset(np.random.default_rng(4))
    subject.command(SurvivorThreatView(SurvivorView(7.5, 5.5, 0, False, ()), (9.5, 5.5)))
    assert subject.fleeing
    # Pure helper-state fixture: a previously observed threat, elapsed hold and
    # a public route. These are not Core/world ticks or hidden world inputs.
    subject.tick = subject.hold_ticks + 1
    subject.last_seen_age_ticks = 32
    subject.explorer.order = ((1.5, 5.5),) + tuple(
        p for p in subject.explorer.order if p != (1.5, 5.5)
    )
    subject.explorer.navigator.target = (12.5, 5.5)
    subject.explorer.navigator.path = ((10.5, 5.5), (12.5, 5.5))
    return subject


@pytest.mark.parametrize("reason", ["visible_separation", "memory_expiry"])
@pytest.mark.parametrize("pending_scan", [90, 30, 0])
def test_exit_replans_public_route_without_pending_stationary_scan(reason, pending_scan):
    subject = retreating()
    old = subject.explorer.navigator
    subject.explorer.scanning = pending_scan
    if reason == "memory_expiry":
        subject.last_seen_age_ticks = 59
    view = SurvivorView(2.5, 5.5, 0 if reason == "visible_separation" else pi, False, ())
    visible = (9.5, 5.5) if reason == "visible_separation" else None
    command = subject.command(SurvivorThreatView(view, visible))
    assert not subject.fleeing and subject.escape_target is None
    assert subject.explorer.scanning == 0
    assert subject.explorer.navigator is not old and subject.explorer.navigator.path != old.path
    assert subject.explorer.explore_target == subject.explorer.navigator.target == (1.5, 5.5)
    assert (
        hypot(command.forward, command.strafe) > 0 and not command.running and not command.interact
    )
    assert subject.explorer.known == {}
    if reason == "visible_separation":
        assert subject.last_seen == (9.5, 5.5) and subject.last_seen_age_ticks == 0
        assert subject.resume_entry_reason == "visible_separation"
    else:
        assert subject.last_seen is subject.last_seen_age_ticks is None


def test_arrival_scan_remains_enabled_after_resume():
    subject = retreating()
    subject.last_seen_age_ticks = 59
    subject.command(SurvivorThreatView(SurvivorView(2.5, 5.5, pi, False, ())))
    # Synthetic own-state at the public waypoint; no movement simulation.
    command = subject.command(SurvivorThreatView(SurvivorView(1.5, 5.5, pi, False, ())))
    assert subject.explorer.scanning == subject.explorer.scan_ticks - 1
    assert (
        command.forward == command.strafe == 0
        and command.yaw_delta == subject.explorer.turn_per_tick
    )


def test_active_retreat_does_not_discard_scan_or_change_hold_rule():
    subject = retreating()
    subject.tick = subject.flee_started
    command = subject.command(SurvivorThreatView(SurvivorView(2.5, 5.5, pi, False, ())))
    assert subject.fleeing and subject.explorer.scanning == subject.explorer.scan_ticks
    assert command.running


def test_new_visible_threat_refreshes_memory_and_keeps_retreat():
    subject = retreating()
    command = subject.command(SurvivorThreatView(SurvivorView(2.5, 5.5, 0, False, ()), (4.5, 5.5)))
    assert subject.fleeing and subject.last_seen == (4.5, 5.5) and subject.last_seen_age_ticks == 0
    assert subject.explorer.scanning == subject.explorer.scan_ticks and command.running


def test_discovered_interaction_still_takes_priority_after_exit():
    subject = retreating()
    subject.last_seen_age_ticks = 59
    card = MapObject("visible_card", "keycard", 1.9, 5.5)
    command = subject.command(SurvivorThreatView(SurvivorView(2.5, 5.5, pi, False, (card,))))
    assert not subject.fleeing and subject.explorer.scanning == 0
    assert command.interact and command.forward == command.strafe == 0
    assert subject.explorer.known == {card.object_id: card}


def test_hidden_world_changes_cannot_refresh_memory_or_alter_resume():
    objects = [MapObject("unknown", "keycard", x, 5.5) for x in (6.5, 12.5)]
    worlds = [arena((obj,)) for obj in objects]
    survivor = SurvivorState(2.5, 5.5, pi)
    views = [
        observe_threat(w, survivor, False, AlienState(x, 5.5), EvasionConfig())
        for w, x in zip(worlds, (9.5, 10.5))
    ]
    assert (
        views[0] == views[1]
        and views[0].visible_alien is None
        and not views[0].mission.visible_objects
    )
    subjects = [retreating(), retreating()]
    assert subjects[0].command(views[0]) == subjects[1].command(views[1])
    assert subjects[0].last_seen == subjects[1].last_seen == (9.5, 5.5)
    assert subjects[0].last_seen_age_ticks == subjects[1].last_seen_age_ticks == 33
    assert not subjects[0].explorer.known and not subjects[1].explorer.known
    assert subjects[0].fleeing and subjects[1].fleeing
    assert not subjects[0].resume_active and not subjects[1].resume_active


def test_no_threat_still_matches_original_explorer_initial_scan():
    subject = EvasiveExplorerBot(arena().rows, CoreConfig())
    original = ExplorerBot(arena().rows, CoreConfig())
    for b in (subject, original):
        b.reset(np.random.default_rng(4))
    for n in range(10):
        view = SurvivorView(2.5, 5.5, n * subject.explorer.turn_per_tick, False, ())
        assert subject.command(SurvivorThreatView(view)) == original.command(view)
        assert subject.explorer.scanning == original.scanning
