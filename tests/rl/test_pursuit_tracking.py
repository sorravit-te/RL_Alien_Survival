"""Tracking pursuit helper: confirmed-point chase and reacquisition handoff, no hidden truth."""

from types import SimpleNamespace

import numpy as np

from game.map import WorldMap
from game.perception import PerceptionSnapshot
from game.state import SurvivorCommand, SurvivorState
from rl.candidates import build_candidates
from rl.contract import SEARCH_CONTROL, VERSION, search_contract
from rl.env import AlienEnv
from rl.observations import build_observation

CLUE = (5.5, 5.5)


def arena(survivor=(17.5, 17.5)):
    return WorldMap(
        ("." * 20,) * 20,
        (survivor,),
        alien_spawn=(*CLUE, 0),
        patrol_waypoints=((8.5, 5.5),),
    )


def make_env(survivor=(9.5, 5.5)):
    e = AlienEnv(arena(survivor), bot_kind="stationary_debug")
    e.reset(seed=3)
    return e


def record_ticks(env, monkeypatch, before_tick=None):
    records = []
    original = env.core.step

    def spy(command):
        if before_tick is not None:
            before_tick(env.core.snapshot())
        b = env.core.snapshot()
        a = original(command)
        records.append((b, a))
        return a

    monkeypatch.setattr(env.core, "step", spy)
    return records


def strafing_bot():
    return SimpleNamespace(
        command=lambda view: SurvivorCommand(strafe=1.0), reset=lambda rng: None, label="fixture"
    )


def test_contract_identity():
    contract = search_contract()
    assert contract["search_control"] == SEARCH_CONTROL == "continue_cancel_waypoint_commitment_v1"
    assert contract["contract_version"] == VERSION == "alien-vector-v5-waypoint-commitment"
    assert contract["observation_shape"] == [352] and contract["action_n"] == 32


def test_visible_chase_target_follows_each_confirmed_point(monkeypatch):
    e = make_env()
    try:
        e.step(30)  # stationary Survivor in view: confirm sight before the chase block
        assert e.core.snapshot().perception.visible and 20 in np.flatnonzero(e.action_masks())
        e.bot = strafing_bot()
        records = record_ticks(e, monkeypatch)
        e.step(20)
        targets = [a.alien_target for _, a in records]
        assert all(a.perception.visible for _, a in records)
        # Each tick's target is that tick's pre-motion confirmed sample.
        assert all(a.alien_target == a.perception.last_seen for _, a in records)
        assert len(set(targets)) == len(targets)
    finally:
        e.close()


def test_tracking_stops_at_last_confirmed_point_after_sight_loss(monkeypatch):
    e = make_env()
    try:
        e.step(30)
        e.bot = strafing_bot()

        def hide(snapshot):
            if snapshot.tick == 20:  # move behind the Alien, outside cone and near range
                e.core._survivor = SurvivorState(1.5, 5.5)

        records = record_ticks(e, monkeypatch, hide)
        e.step(20)
        lost = [a for _, a in records if not a.perception.visible]
        assert lost and lost[0].tick == 21
        point = lost[0].perception.last_seen
        assert all(a.alien_target == point and a.alien_target_source == "visible" for a in lost)
        assert all(a.alien_target != (a.survivor.x, a.survivor.y) for a in lost)
    finally:
        e.close()


def test_reacquisition_hands_off_to_helper_chase_without_freezing(monkeypatch):
    e = AlienEnv(arena((8.5, 5.5)), bot_kind="stationary_debug")
    try:
        e.reset(seed=1)
        e.core._perception = PerceptionSnapshot(last_seen=CLUE, last_seen_age_ticks=300, revision=1)
        records = record_ticks(e, monkeypatch)
        e.step(21)
        cancel = next(i for i, (_, a) in enumerate(records) if a.investigation.phase == "cancelled")
        b, a = records[cancel]
        assert a.tick == 6 and a.alien_target_source == "visible_handoff"
        assert a.alien_target == a.perception.last_seen and a.navigation_path
        assert (a.alien.x, a.alien.y) != (b.alien.x, b.alien.y)
        rest = [a for _, a in records[cancel:]]
        assert all(a.alien_target_source == "visible_handoff" for a in rest)
        xs = [a.alien.x for a in rest]
        assert all(later > earlier for earlier, later in zip(xs, xs[1:]))
        counts = e.metrics.control.counts
        assert counts["helper_cancels"] == 1 and counts["helper_pursuit_handoffs"] == 1
        assert counts["policy_cancels"] == 0
        mask = np.flatnonzero(e.action_masks()).tolist()
        assert 20 in mask and 21 not in mask
        e.step(20)  # the next policy decision replaces the helper source
        assert e.core.snapshot().alien_target_source == "visible"
        assert e.core.snapshot().investigation.phase == "idle"
    finally:
        e.close()


def test_hidden_survivor_position_does_not_change_tracking_views():
    envs = [AlienEnv(arena(p), bot_kind="stationary_debug") for p in ((17.5, 17.5), (16.5, 17.5))]
    try:
        for e in envs:
            e.reset(seed=7)
            e.core._perception = PerceptionSnapshot(
                last_seen=CLUE, last_seen_age_ticks=300, revision=1
            )
        for action in (21, 20, 0, 30, 21):
            views = [e._allowed_view() for e in envs]
            assert views[0] == views[1] and build_candidates(views[0]) == build_candidates(views[1])
            np.testing.assert_array_equal(build_observation(views[0]), build_observation(views[1]))
            snapshots = [e.core.snapshot() for e in envs]
            assert snapshots[0].alien == snapshots[1].alien
            assert snapshots[0].alien_target == snapshots[1].alien_target
            for e in envs:
                e.step(action)
    finally:
        for e in envs:
            e.close()
