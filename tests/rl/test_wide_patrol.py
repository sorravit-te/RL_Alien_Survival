"""Anti-camping on the full ship: a free patrol keeps moving, and RL still picks the way."""

from math import hypot

import numpy as np
import pytest
from sb3_contrib import MaskablePPO

from ai.navigation import find_path
from game.map import scenario_map
from game.ship_deck import SHIP_SCENARIO, WAYPOINTS
from game.state import CoreConfig
from rl.candidates import (
    NEAREST_DUE,
    REST_SECONDS,
    AllowedAlienView,
    build_candidates,
    candidate_masks,
)
from rl.contract import WAIT
from rl.env import AlienEnv
from rl.evaluate import evaluate
from rl.pipeline import make_env, read_json
from rl.scripted_controllers import HeuristicSelector

SMALL = tuple((float(x), 2.5) for x in range(1, 7))
ONE_CASE = {"id": "ship", "scenario_seed": 1, "bot_seed": 2, "reset_seed": 3, "selector_seed": 4}


def view(*, ages=None, waypoints=WAYPOINTS, position=(3.5, 20.5), **fields):
    ages = (None,) * len(waypoints) if ages is None else ages
    defaults = {
        "yaw": 0.0, "chasing": False, "current_action": WAIT, "visible": False, "last_seen": None,
        "last_seen_age_seconds": None, "navigation_active": False,
    }  # fmt: skip
    return AllowedAlienView(position, waypoints=waypoints, visit_ages=ages, **{**defaults, **fields})


def offered(state):
    return set(np.flatnonzero(candidate_masks(build_candidates(state))).tolist())


def nearest(position, count):
    order = sorted(
        range(len(WAYPOINTS)),
        key=lambda i: hypot(WAYPOINTS[i][0] - position[0], WAYPOINTS[i][1] - position[1]),
    )
    return set(order[:count])


def test_a_free_patrol_offers_only_the_nearest_waypoints_and_cannot_wait():
    assert offered(view()) == nearest((3.5, 20.5), NEAREST_DUE)


def test_a_waypoint_checked_lately_rests():
    near = sorted(nearest((3.5, 20.5), 1))[0]
    ages = tuple(5.0 if i == near else None for i in range(len(WAYPOINTS)))
    assert near not in offered(view(ages=ages))


def test_a_rested_waypoint_is_offered_again():
    near = sorted(nearest((3.5, 20.5), 1))[0]
    ages = tuple(REST_SECONDS if i == near else None for i in range(len(WAYPOINTS)))
    assert near in offered(view(ages=ages))


def test_the_choice_reaches_across_the_ship_once_nearby_rooms_rest():
    close = nearest((3.5, 20.5), 16)
    ages = tuple(5.0 if i in close else None for i in range(len(WAYPOINTS)))
    assert offered(view(ages=ages)) == set(range(len(WAYPOINTS))) - close


def test_with_everything_checked_lately_the_whole_choice_comes_back():
    choice = offered(view(ages=(5.0,) * len(WAYPOINTS)))
    assert choice == set(range(len(WAYPOINTS))) | {WAIT}


def test_a_walk_in_progress_is_left_alone():
    assert offered(view(current_action=7, navigation_active=True)) == {7}


def test_a_new_noise_keeps_its_own_choice():
    state = view(last_heard=(10.5, 19.5), last_heard_age_seconds=0.0, heard_revision=1)
    assert offered(state) == {22, WAIT}


def test_a_sighting_can_always_be_chased():
    state = view(visible=True, last_seen=(5.5, 20.5), last_seen_age_seconds=0.0)
    choice = offered(state)
    assert 20 in choice and WAIT not in choice


def test_a_small_scene_keeps_its_whole_choice():
    assert offered(view(waypoints=SMALL)) == set(range(len(SMALL))) | {WAIT}


def test_the_written_alien_joins_the_loop_at_the_nearest_waypoint():
    selector = HeuristicSelector(0.15)
    action = selector.choose(view(position=(36.0, 21.0)))
    assert WAYPOINTS[action] == (35.5, 20.5)


def test_a_small_scene_still_starts_the_written_order_at_the_beginning():
    assert HeuristicSelector(0.15).choose(view(waypoints=SMALL, position=(5.5, 2.5))) == 0


def test_the_written_alien_walks_on_past_a_waypoint_not_offered():
    selector = HeuristicSelector(0.15)
    selector.choose(view())
    start = selector.state.patrol_index
    ages = tuple(5.0 if i == start else None for i in range(len(WAYPOINTS)))
    action = selector.choose(view(ages=ages))
    assert action != start and action != WAIT and action in offered(view(ages=ages))


def test_the_written_alien_only_ever_picks_what_is_offered():
    env = AlienEnv(scenario=SHIP_SCENARIO, bot_kind="stationary_debug", max_episode_seconds=120)
    env.reset(seed=1, options={"scenario_seed": 2, "bot_seed": 3})
    selector = HeuristicSelector(env.config.target_arrival_distance)
    done, waits = False, 0
    while not done:
        action = selector.choose(env._allowed_view())
        assert env.action_masks()[action]
        waits += action == WAIT
        *_, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    assert waits == 0


def short_ship_config():
    return {**read_json("configs/ship_deck_training_pilot.json"), "max_episode_seconds": 5}


def test_the_wait_baseline_stands_still_where_a_policy_may_not(tmp_path, monkeypatch):
    made = []

    def remembered_env(config):
        made.append(make_env(config))
        return made[-1]

    monkeypatch.setattr("rl.evaluate.make_env", remembered_env)
    _, rows = evaluate(
        "wait", short_ship_config(), [ONE_CASE], tmp_path / "wait.csv",
        inference="constant_wait", episode_seeds=[5], repeat_index=0, controller_name="wait",
    )  # fmt: skip
    world, alien = made[0].world, made[0].core.snapshot().alien
    assert rows[0]["decision_steps"] > 1
    assert (alien.x, alien.y) in [s[:2] for s in (world.alien_spawn, *world.alien_spawn_options)]


def test_a_policy_that_asks_to_wait_on_a_free_patrol_is_refused(tmp_path, monkeypatch):
    config = short_ship_config()
    env = make_env(config)
    model = MaskablePPO("MlpPolicy", env, device="cpu", **config["ppo"])
    monkeypatch.setattr(model, "predict", lambda obs, **kwargs: (WAIT, None))
    with pytest.raises(RuntimeError, match="masked action"):
        evaluate(
            "trained", config, [ONE_CASE], tmp_path / "policy.csv", model=model,
            inference="sampled_masked", episode_seeds=[5], repeat_index=0, controller_name="trained",
        )  # fmt: skip


def test_the_route_is_a_short_loop_of_the_ship():
    config = CoreConfig()
    world = scenario_map(SHIP_SCENARIO)

    def walk(start, end):
        path = find_path(world, start, end, config.alien_radius)
        return sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip((start, *path), path))

    legs = zip(WAYPOINTS, (*WAYPOINTS[1:], WAYPOINTS[0]))
    assert sum(walk(a, b) for a, b in legs) < 220.0
