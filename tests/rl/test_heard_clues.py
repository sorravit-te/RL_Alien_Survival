"""A noise reaches the policy as a masked target and four observation fields, nothing else."""

from dataclasses import replace

from game.map import WorldMap
from game.state import CoreConfig, SurvivorCommand
from rl.alien_view import allowed_view, target_source
from rl.candidates import AllowedAlienView, build_candidates, committed_waypoint
from rl.contract import BASE_FIELDS, WAIT, WORLD_SCALE
from rl.env import AlienEnv
from rl.observations import build_observation
from rl.scripted_controllers import HeuristicSelector

HALL = (
    "##########",
    "#........#",
    "#........#",
    "##########",
)
SOUND = 22
QUIET = AllowedAlienView(
    (1.5, 1.5), 0.0, False, WAIT, False, None, None, ((5.5, 1.5),), (None,), False
)
HEARD = replace(QUIET, last_heard=(4.5, 1.5), last_heard_age_seconds=3.0, heard_revision=1)


def fields(view):
    values = build_observation(view)
    return {name: float(value) for name, value in zip(BASE_FIELDS, values, strict=False)}


def test_sound_action_is_masked_until_something_is_heard():
    assert not build_candidates(QUIET)[SOUND].valid
    candidate = build_candidates(HEARD)[SOUND]
    assert candidate.valid and candidate.target == (4.5, 1.5) and candidate.kind == 3
    assert candidate.clue_age == 3.0


def test_a_checked_noise_is_masked_until_the_next_one():
    assert not build_candidates(replace(HEARD, heard_checked=True))[SOUND].valid


def test_observation_carries_heard_fields():
    quiet = fields(QUIET)
    assert (quiet["last_heard_valid"], quiet["last_heard_dx"]) == (0.0, 0.0)
    heard = fields(HEARD)
    assert heard["last_heard_valid"] == 1.0
    assert abs(heard["last_heard_dx"] - 3.0 / WORLD_SCALE) < 1e-6
    assert heard["last_heard_dy"] == 0.0
    assert abs(heard["last_heard_age"] - 3.0 / 60.0) < 1e-6
    assert heard["director_valid"] == 0.0 and heard["interaction_valid"] == 0.0


def test_a_new_noise_releases_a_waypoint_commitment():
    walking = replace(QUIET, current_action=0, navigation_active=True)
    assert committed_waypoint(walking) == 0
    noise = replace(walking, last_heard=(4.5, 1.5), last_heard_age_seconds=0.0, heard_revision=1)
    assert committed_waypoint(noise) is None
    assert build_candidates(noise)[SOUND].valid
    assigned = replace(noise, target_heard_revision=1)
    assert committed_waypoint(assigned) == 0


def test_target_source_names_sound():
    assert target_source(SOUND) == "sound"


def test_selector_follows_an_unchecked_noise():
    assert HeuristicSelector(0.15).choose(HEARD) == SOUND
    assert HeuristicSelector(0.15).choose(replace(HEARD, heard_checked=True)) == 0


def env_with_noise():
    world = WorldMap(
        HALL, ((1.5, 1.5),), alien_spawn=(7.5, 1.5, 0.0), patrol_waypoints=((8.5, 1.5),)
    )
    env = AlienEnv(
        world, CoreConfig(alien_sight_range=1.0, alien_near_range=0.5), bot_kind="stationary_debug"
    )
    env.reset(options={"scenario_seed": 1, "bot_seed": 1})
    for _ in range(16):
        env.core.step(SurvivorCommand(forward=1.0, running=True))
    return env


def test_environment_walks_the_alien_to_the_region_not_the_body():
    env = env_with_noise()
    snapshot = env.core.snapshot()
    view = allowed_view(snapshot, WAIT, env.visit_ticks)
    assert view.last_heard == snapshot.heard.point
    assert view.last_heard != (snapshot.survivor.x, snapshot.survivor.y)
    assert env.action_masks()[SOUND]
    env.step(SOUND)
    after = env.core.snapshot()
    assert after.alien_target_source == "sound" and after.alien_target == snapshot.heard.point
