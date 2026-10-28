"""Scripted retry contracts on synthetic filtered inputs; no physics or models."""

from dataclasses import asdict, fields
from math import hypot, pi

import numpy as np
import pytest

import ai.navigation as navigation
import rl.evasive_survivor as evasion
from game.map import MapObject, WorldMap
from game.state import AlienState, CoreConfig, SurvivorCommand, SurvivorState
from rl.evasive_survivor import (
    EvasiveExplorerBot,
    ExplorationRetryRecord,
    SurvivorThreatView,
    observe_threat,
)
from rl.survivor_bots import ExplorerBot, SurvivorView

ROWS = ("#" * 16,) + ("#" + "." * 14 + "#",) * 10 + ("#" * 16,)
TARGET = (8.5, 5.5)
HERE = (2.5, 5.5)
STEP = 2.5 / 30


def view(x=HERE[0], yaw=0, threat=None, card=False, objects=()):
    return SurvivorThreatView(SurvivorView(x, HERE[1], yaw, card, objects), threat)


def subject(seed=4, rows=ROWS, seeded=False):
    bot = EvasiveExplorerBot(rows, CoreConfig())
    bot.reset(np.random.default_rng(seed))
    bot.explorer.scanning = 0
    if seeded:
        bot.explorer.explore_target = TARGET
    else:
        bot.explorer.order = (TARGET, (1.5, 7.5), (12.5, 7.5))
    return bot


def first_interruption(bot, *, fresh=True):
    command = bot.command(view())
    assert command.forward > 0 and not command.running
    if not fresh:
        # Previously filtered sight, still valid; no hidden world state is supplied.
        bot.last_seen, bot.last_seen_age_ticks = (5.5, 5.5), 8
    command = bot.command(view(x=HERE[0] + STEP, threat=(5.5, 5.5) if fresh else None))
    assert bot.fleeing and command.running and hypot(command.forward, command.strafe) > 0
    return bot._retry_anchor_ids[TARGET]


def second_interruption(bot, *, tick=None):
    # Pure clock/memory boundary fixture. Own states are inputs, never applied motion.
    if tick is not None:
        bot.tick = tick - 2
    bot.last_seen_age_ticks = 59
    command = bot.command(view())
    assert not bot.fleeing and command.forward > 0
    command = bot.command(view(x=HERE[0] + STEP, threat=(5.5, 5.5)))
    assert bot.fleeing and command.running


def pending_subject():
    bot = subject()
    anchor_id = first_interruption(bot)
    second_interruption(bot)
    assert bot.retry_records[anchor_id].count == 2 and bot.retry_pending is not None
    return bot, anchor_id


def release(bot, *, x=HERE[0]):
    bot.last_seen_age_ticks = 59
    return bot.command(view(x=x))


@pytest.mark.parametrize("seed", [0, 4, 17])
def test_no_threat_command_and_rng_parity_with_unchanged_explorer(seed):
    bot = EvasiveExplorerBot(ROWS, CoreConfig())
    base = ExplorerBot(ROWS, CoreConfig())
    rngs = [np.random.default_rng(seed), np.random.default_rng(seed)]
    bot.reset(rngs[0])
    base.reset(rngs[1])
    for t in range(160):
        mission = view(yaw=(t % 9) * pi / 4).mission
        assert bot.command(SurvivorThreatView(mission)) == base.command(mission)
        assert not bot.retry_records and bot.retry_pending is None
    assert rngs[0].bit_generator.state == rngs[1].bit_generator.state


@pytest.mark.parametrize("fresh,reason", [(True, "fresh_sight"), (False, "remembered_threat")])
def test_actual_public_motion_and_source_attribution(fresh, reason):
    bot = subject()
    anchor = first_interruption(bot, fresh=fresh)
    assert bot.retry_records[anchor] == ExplorationRetryRecord(anchor, 1, 2, None)
    assert bot.retry_last_interruption == (2, anchor, reason)
    assert bot.retry_pending is None
    assert bot.explorer.explore_target == TARGET


@pytest.mark.parametrize("mode", ["stationary", "turn_only", "scan", "arrival", "no_path"])
def test_non_moving_modes_do_not_create_interruption(mode, monkeypatch):
    bot = subject()
    initial = view(yaw=pi if mode == "turn_only" else 0)
    if mode == "scan":
        bot.explorer.scanning = 10
    if mode == "arrival":
        bot.explorer.explore_target = HERE
    if mode == "no_path":
        monkeypatch.setattr(bot.explorer.navigator, "displacement", lambda *a: (0, 0))
    bot.command(initial)
    bot.command(view(x=HERE[0] if mode == "stationary" else HERE[0] + STEP, threat=(5.5, 5.5)))
    assert not bot.retry_records


def test_expired_memory_and_zero_retreat_do_not_count(monkeypatch):
    bot = subject()
    bot.command(view())
    bot.last_seen, bot.last_seen_age_ticks = (5.5, 5.5), 59
    bot.command(view(x=HERE[0] + STEP))
    assert bot.last_seen is None and not bot.retry_records and not bot.fleeing
    bot = subject()
    bot.command(view())
    monkeypatch.setattr(bot, "_plan", lambda position: (None, ()))
    assert bot.command(view(x=HERE[0] + STEP, threat=(5.5, 5.5))) == SurvivorCommand()
    assert bot.fleeing and not bot.retry_records


def test_held_flee_does_not_renew_or_count_again():
    bot = subject()
    anchor = first_interruption(bot)
    record = bot.retry_records[anchor]
    for _ in range(20):
        bot.command(view(x=HERE[0] + STEP, threat=(5.5, 5.5)))
    assert bot.retry_records[anchor] == record and bot.flee_entries == 1


@pytest.mark.parametrize("delta,count", [(298, 2), (299, 2), (300, 1), (301, 1)])
def test_exact_first_window_boundary(delta, count):
    bot = subject()
    anchor = first_interruption(bot)
    first = bot.tick
    second_interruption(bot, tick=first + delta)
    record = bot.retry_records[anchor]
    assert record.count == count
    assert (bot.retry_pending is not None) == (count == 2)
    assert record.first_tick == (first if count == 2 else first + delta)
    assert record.defer_until == (first + delta + 180 if count == 2 else None)


@pytest.mark.parametrize("offset,active", [(178, True), (179, True), (180, False), (181, False)])
@pytest.mark.parametrize("early", ["approach", "interaction"])
def test_defer_expiry_cleans_pending_before_early_returns(offset, active, early, monkeypatch):
    bot, anchor = pending_subject()
    second = bot.tick
    record = bot.retry_records[anchor]
    if early == "approach":
        monkeypatch.setattr(bot, "_exit_approach", lambda mission: SurvivorCommand())
    else:
        monkeypatch.setattr(bot, "_ready_interaction", lambda mission: True)
    bot.tick = second + offset - 1
    bot.command(view())
    assert (anchor in bot.retry_records) is active
    assert (bot.retry_pending is not None) is active
    if active:
        assert bot.retry_records[anchor] == record
    else:
        assert bot.retry_exit_token is None and bot.retry_last_interruption is None


def test_selected_once_seeded_order_and_no_extra_path_search(monkeypatch):
    bot, anchor = pending_subject()
    calls = []
    original = evasion.find_path
    monkeypatch.setattr(evasion, "find_path", lambda *a: calls.append(a[2]) or original(*a))
    monkeypatch.setattr(navigation, "find_path", evasion.find_path)
    command = release(bot)
    selection = bot.retry_last_selection.copy()
    assert calls == [TARGET, (1.5, 7.5)]
    assert selection["inspected"] == 2 and selection["path_searches"] == 2
    assert selection["candidate_searches"] == 1
    assert bot.explorer.explore_target == (1.5, 7.5)
    assert bot.retry_pending is None and anchor in bot.retry_records
    assert not command.running and not command.interact
    bot.command(view())
    assert bot.retry_last_selection == selection and len(calls) == 2


def test_path_and_id_bounds_fallback_and_next_exit_opportunity(monkeypatch):
    bot, anchor = pending_subject()
    bot.explorer.order = (TARGET,) + tuple(p for p in bot.explorer.points if p != TARGET)
    original = evasion.find_path
    calls = []
    monkeypatch.setattr(
        evasion,
        "find_path",
        lambda *a: calls.append(a[2]) or (original(*a) if a[2] == TARGET else ()),
    )
    monkeypatch.setattr(navigation, "find_path", evasion.find_path)
    release(bot)
    assert len(calls) == 4 and bot.retry_last_selection["path_searches"] == 4
    assert bot.retry_last_selection["inspected"] == 4
    assert bot.retry_last_selection["outcome"] == "fallback"
    assert bot.explorer.explore_target == TARGET and bot.retry_pending is not None
    first_attempt_tick = bot.retry_last_selection["tick"]
    cursor = bot.retry_cursor
    for _ in range(3):
        bot.command(view())
    assert len(calls) == 4 and bot.retry_cursor == cursor
    assert bot.retry_last_selection["tick"] == first_attempt_tick
    monkeypatch.setattr(evasion, "find_path", original)
    bot.command(view(x=HERE[0] + STEP, threat=(5.5, 5.5)))
    record = bot.retry_records[anchor]
    release(bot)
    assert bot.retry_last_selection["outcome"] == "selected"
    assert bot.retry_last_selection["tick"] > first_attempt_tick
    assert bot.retry_records[anchor] == record  # no renewal from fallback encounter


@pytest.mark.parametrize("order_kind", ["empty", "visited", "deferred"])
def test_no_candidates_soft_fallback_never_fakes_visits(order_kind):
    bot, anchor = pending_subject()
    if order_kind == "empty":
        bot.explorer.order = ()
    elif order_kind == "visited":
        bot.explorer.visited.update(p for p in bot.explorer.order if p != TARGET)
    else:
        other = bot._retry_anchor_ids[(1.5, 7.5)]
        bot.retry_records[other] = ExplorationRetryRecord(other, 2, bot.tick, bot.tick + 180)
        bot.explorer.order = (TARGET, (1.5, 7.5))
    visited = set(bot.explorer.visited)
    release(bot)
    assert bot.retry_last_selection["outcome"] == "fallback"
    assert bot.explorer.visited == visited and bot.explorer.explore_target == TARGET
    assert bot.retry_records[anchor].defer_until is not None


def test_sixteen_id_cap_with_no_search_and_cursor_advance():
    bot, _ = pending_subject()
    order = tuple(p for p in bot.explorer.points if p != TARGET)[:20]
    bot.explorer.order = order
    bot.explorer.visited.update(order)
    release(bot)
    assert bot.retry_last_selection["inspected"] == 16
    assert bot.retry_last_selection["path_searches"] == 1 and bot.retry_cursor == 16
    assert bot.retry_last_selection["candidate_searches"] == 0


@pytest.mark.parametrize("near", [0, 5e-9])
def test_due_arrival_keeps_scan_and_clears_referenced_pending(near):
    bot, anchor = pending_subject()
    release(bot, x=TARGET[0] + near)
    assert bot.explorer.last_command_reason == "arrived"
    assert bot.explorer.scanning == bot.explorer.scan_ticks - 1
    assert TARGET in bot.explorer.visited
    assert anchor not in bot.retry_records and bot.retry_pending is None
    assert bot.resume_end_reason == "arrived" and bot.retry_last_selection is None


def test_candidate_at_current_position_preserves_arrival_scan():
    bot, old = pending_subject()
    bot.explorer.order = (TARGET, HERE)
    command = release(bot)
    assert command == SurvivorCommand(yaw_delta=bot.explorer.turn_per_tick)
    assert bot.explorer.last_command_reason == "arrived" and HERE in bot.explorer.visited
    assert bot.explorer.scanning == bot.explorer.scan_ticks - 1
    assert bot.retry_pending is None and old in bot.retry_records


def test_pending_waits_for_scan_without_extending_resume_or_defer(monkeypatch):
    bot, anchor = pending_subject()
    original = bot._exit_approach
    monkeypatch.setattr(bot, "_exit_approach", lambda mission: SurvivorCommand())
    release(bot)
    deadline = bot.resume_deadline_tick
    defer = bot.retry_records[anchor].defer_until
    monkeypatch.setattr(bot, "_exit_approach", original)
    bot.explorer.scanning = 2  # synthetic due scan after the priority early return
    for left in (1, 0):
        command = bot.command(view())
        assert command == SurvivorCommand(yaw_delta=bot.explorer.turn_per_tick)
        assert bot.retry_pending is not None and bot.retry_last_selection is None
        assert bot.explorer.scanning == left
    bot.command(view())
    assert bot.retry_pending is None and bot.retry_last_selection["outcome"] == "selected"
    assert bot.resume_deadline_tick == deadline and bot.retry_records[anchor].defer_until == defer


def test_waiting_pending_expires_during_scan_and_resume_deadline(monkeypatch):
    bot, anchor = pending_subject()
    monkeypatch.setattr(bot, "_exit_approach", lambda mission: SurvivorCommand())
    release(bot)
    deadline = bot.resume_deadline_tick
    bot.tick = deadline - 1
    bot.command(view())
    assert bot.resume_end_reason == "deadline" and bot.retry_pending is not None
    bot.tick = bot.retry_records[anchor].defer_until - 1
    bot.command(view())
    assert not bot.retry_records and bot.retry_pending is None and bot.retry_last_selection is None
    assert bot.resume_deadline_tick == deadline


@pytest.mark.parametrize("reason", ["phase", "wanted", "reset"])
def test_mission_change_discovery_and_reset_clear_pending(reason):
    bot, _ = pending_subject()
    if reason == "reset":
        bot.reset(np.random.default_rng(4))
    elif reason == "phase":
        bot.command(view(card=True))
    else:
        goal = MapObject("newly_seen", "keycard", HERE[0] + 0.1, HERE[1])
        command = bot.command(view(objects=(goal,)))
        assert command == SurvivorCommand(interact=True)
    assert not bot.retry_records and bot.retry_pending is None and bot.retry_exit_token is None
    assert bot.retry_cursor == 0


def test_irrelevant_object_does_not_disclose_or_override_current_mission():
    bot, anchor = pending_subject()
    goal = MapObject("seen_exit", "exit", 12.5, 5.5)
    bot.command(view(objects=(goal,)))
    assert anchor in bot.retry_records and bot.retry_pending is not None
    assert goal.object_id in bot.explorer.known and not bot.retry_last_selection


def test_record_capacity_oldest_tie_eviction_cancels_stale_reference():
    bot = subject()
    ids = sorted(bot._retry_anchor_ids.values())[:5]
    # Bounded metadata fixture; insertion/eviction still enters via filtered walking/retreat.
    for anchor in ids[:4]:
        bot.retry_records[anchor] = ExplorationRetryRecord(anchor, 2, 1, 180)
    oldest = ids[0]
    bot.retry_pending = (oldest, 1)
    first_interruption(bot)
    assert len(bot.retry_records) == 4 and oldest not in bot.retry_records
    assert bot.retry_pending is None


def test_missing_or_replaced_record_cannot_leave_stale_pending(monkeypatch):
    bot, anchor = pending_subject()
    bot.retry_records[anchor] = ExplorationRetryRecord(anchor, 1, 99, None)
    monkeypatch.setattr(bot, "_ready_interaction", lambda mission: True)
    bot.command(view())
    assert bot.retry_pending is None and bot.retry_exit_token is None


def test_history_schema_and_expired_threat_hidden_world_invariance():
    bots = [pending_subject()[0], pending_subject()[0]]
    worlds = [
        WorldMap(ROWS, (HERE,), (MapObject("hidden", "keycard", x, 5.5),)) for x in (9.5, 12.5)
    ]
    views = [
        observe_threat(w, SurvivorState(*HERE, pi), False, AlienState(x, 5.5), b.settings)
        for w, x, b in zip(worlds, (10.5, 13.5), bots)
    ]
    assert views[0] == views[1] and views[0].visible_alien is None
    commands = []
    for bot, filtered in zip(bots, views):
        bot.last_seen_age_ticks = 59
        commands.append(bot.command(filtered))
        assert bot.last_seen is bot.last_seen_age_ticks is None
        assert not bot.explorer.known
        for record in bot.retry_records.values():
            assert {f.name for f in fields(record)} == {
                "anchor_id",
                "count",
                "first_tick",
                "defer_until",
            }
            assert all(value is None or type(value) is int for value in asdict(record).values())
    assert commands[0] == commands[1]
    assert bots[0].retry_records == bots[1].retry_records
    assert bots[0].retry_last_selection == bots[1].retry_last_selection


@pytest.mark.parametrize("seed", [5, 17, 99])
def test_reselection_reuses_reset_seed_order(seed):
    bots = [subject(seed=seed, seeded=True), subject(seed=seed, seeded=True)]
    outputs = []
    for bot in bots:
        first_interruption(bot)
        second_interruption(bot)
        expected = next(p for p in bot.explorer.order if p != TARGET)
        command = release(bot)
        assert bot.retry_last_selection["to_anchor"] == bot._retry_anchor_ids[expected]
        outputs.append((command, bot.retry_last_selection, bot.retry_records))
    assert outputs[0] == outputs[1]


def test_real_unreachable_candidate_then_reachable_alternative():
    rows = tuple(row[:10] + "#" + row[11:] for row in ROWS)
    bot = subject(rows=rows)
    first_interruption(bot)
    second_interruption(bot)
    bot.explorer.order = (TARGET, (12.5, 7.5), (1.5, 7.5))
    release(bot)
    assert bot.retry_last_selection["path_searches"] == 3
    assert bot.retry_last_selection["candidate_searches"] == 2
    assert bot.explorer.explore_target == (1.5, 7.5)
    assert (12.5, 7.5) not in bot.explorer.visited


def test_deferred_record_expires_during_held_flee_without_refresh():
    bot, anchor = pending_subject()
    record = bot.retry_records[anchor]
    for _ in range(179):
        bot.command(view(x=HERE[0] + STEP, threat=(5.5, 5.5)))
        assert bot.retry_records[anchor] == record
    bot.command(view(x=HERE[0] + STEP, threat=(5.5, 5.5)))
    assert anchor not in bot.retry_records and bot.retry_pending is None
    assert bot.flee_entries == 2


def test_real_ready_interaction_and_exit_approach_cannot_add_history():
    bot = subject()
    bot.command(view())
    card = MapObject("seen_card", "keycard", HERE[0] + STEP + 0.1, HERE[1])
    command = bot.command(view(x=HERE[0] + STEP, threat=(5.5, 5.5), objects=(card,)))
    assert command == SurvivorCommand(interact=True) and not bot.retry_records
    bot, _ = pending_subject()
    goal = MapObject("seen_exit", "exit", 6.5, 5.5)
    command = bot.command(view(threat=(9.5, 5.5), card=True, objects=(goal,)))
    assert bot.exit_approach_active and not command.running and not command.interact
    assert not bot.retry_records and bot.retry_pending is None


@pytest.mark.parametrize("early", ["approach", "interaction"])
def test_arrival_reference_cleanup_precedes_priority_early_return(early, monkeypatch):
    bot, anchor = pending_subject()
    if early == "approach":
        monkeypatch.setattr(bot, "_exit_approach", lambda mission: SurvivorCommand())
    else:
        monkeypatch.setattr(bot, "_ready_interaction", lambda mission: True)
    bot.command(view(x=TARGET[0] + 5e-9))
    assert anchor not in bot.retry_records and bot.retry_pending is None
    assert bot.retry_last_selection is None


def test_expired_waiting_pending_never_runs_when_normal_opportunity_returns(monkeypatch):
    bot, anchor = pending_subject()
    original = bot._exit_approach
    monkeypatch.setattr(bot, "_exit_approach", lambda mission: SurvivorCommand())
    release(bot)
    bot.tick = bot.retry_records[anchor].defer_until - 1
    bot.command(view())
    monkeypatch.setattr(bot, "_exit_approach", original)
    bot.command(view())
    assert bot.retry_pending is None and bot.retry_last_selection is None
    assert bot.explorer.explore_target == TARGET
