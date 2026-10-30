"""A waypoint walk holds until arrival, a dead path or new confirmed evidence."""

from dataclasses import replace

import numpy as np
import pytest

from game.map import WorldMap
from rl.alien_view import still_travelling
from rl.candidates import AllowedAlienView, committed_waypoint
from rl.env import AlienEnv
from rl.scripted_controllers import HeuristicSelector

START = (5.5, 5.5)
FAR = (8.5, 5.5)  # 41 ticks of walking: three decisions
ONE_DECISION = (6.6, 5.5)  # the final step lands on the last tick of a decision


def make_env(survivor=(17.5, 17.5), waypoints=(FAR, (5.5, 8.5))):
    world = WorldMap(
        ("." * 20,) * 20, (survivor,), alien_spawn=(*START, 0), patrol_waypoints=waypoints
    )
    env = AlienEnv(world, bot_kind="stationary_debug")
    env.reset(seed=1, options={"scenario_seed": 1, "bot_seed": 2})
    return env


def valid(env):
    return np.flatnonzero(env.action_masks()).tolist()


def walking_view(**changes):
    view = AllowedAlienView(
        START, 0.0, False, 1, False, None, None, (FAR, ONE_DECISION), (None, None), True
    )
    return replace(view, **changes)


def test_walk_is_the_only_choice_until_arrival():
    env = make_env()
    assert valid(env) == [0, 1, 30]
    for _ in range(2):
        env.step(0)
        assert valid(env) == [0]
    env.step(0)
    alien = env.core.snapshot().alien
    assert (alien.x, alien.y) == FAR and valid(env) == [0, 1, 30]


def test_arrival_on_a_decision_boundary_does_not_waste_a_decision():
    env = make_env(waypoints=(ONE_DECISION,))
    env.step(0)
    snapshot = env.core.snapshot()
    assert snapshot.navigation_path  # Core prunes the finished point on the next tick.
    assert not still_travelling(snapshot) and valid(env) == [0, 30]


def test_masked_request_ends_the_walk_and_is_audited():
    env = make_env()
    env.step(0)
    info = env.step(30)[4]
    assert info["invalid_action"] and info["applied_action"] == 30
    assert valid(env) == [0, 1, 30]
    assert env.metrics.patrol.counts["abandoned"] == 1


def test_sighting_opens_every_option_again():
    env = make_env(survivor=(5.5, 14.5), waypoints=((5.5, 12.5),))
    env.step(0)
    assert valid(env) == [0]
    while not env.core.snapshot().perception.visible:
        env.step(0)
    assert valid(env) == [0, 20, 30]


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({}, 1),
        ({"clue_revision": 3, "target_revision": 3}, 1),
        ({"clue_revision": 3, "target_revision": 2}, None),  # seen and lost within a decision
        ({"visible": True}, None),
        ({"navigation_active": False}, None),  # arrived or no path
        ({"current_action": 21}, None),
        ({"current_action": 30}, None),
        ({"current_action": 2}, None),  # not a waypoint in this scene
    ],
)
def test_commitment_rule(change, expected):
    assert committed_waypoint(walking_view(**change)) == expected


def test_heuristic_keeps_the_walk():
    assert HeuristicSelector(0.15).choose(walking_view()) == 1


def test_hidden_survivor_position_changes_nothing_in_a_walk():
    envs = [make_env(survivor=point) for point in ((17.5, 17.5), (16.5, 17.5))]
    for action in (0, 0, 0, 1, 1):
        steps = [env.step(action) for env in envs]
        np.testing.assert_array_equal(steps[0][0], steps[1][0])
        np.testing.assert_array_equal(envs[0].action_masks(), envs[1].action_masks())


def test_audit_counts_walks_arrivals_and_released_walks():
    env = make_env()
    env.cap_ticks = 60
    for _ in range(3):
        env.step(0)
    info = env.step(1)[4]
    patrol = info["search_metrics"]["patrol"]
    assert patrol["counts"] == {
        "walks": 2,
        "arrivals": 1,
        "released_by_clue": 0,
        "forced_choices": 2,
        "abandoned": 0,
    }
    assert patrol["distinct_waypoints_reached"] == 1 and patrol["unfinished_at_end"] == 1

    env = make_env(survivor=(5.5, 14.5), waypoints=((5.5, 12.5),))
    while not env.core.snapshot().perception.visible:
        env.step(0)
    assert env.metrics.patrol.counts["released_by_clue"] == 1
