"""Single boundary from a Core snapshot to policy-allowed fields.

Training Env and human play share these helpers, so both see identical fields,
physical ages, target sources and waypoint visit stamps.
"""

from math import hypot

from rl.candidates import AllowedAlienView
from rl.contract import WAIT


def allowed_view(snapshot, last_action, visit_ticks):
    alien, memory, tick_hz = snapshot.alien, snapshot.perception, snapshot.config.tick_hz
    return AllowedAlienView(
        (alien.x, alien.y),
        alien.yaw,
        alien.speed_mode == "chase",
        last_action,
        memory.visible,
        memory.last_seen,
        memory.last_seen_age_ticks / tick_hz if memory.last_seen_age_ticks is not None else None,
        snapshot.world_map.patrol_waypoints,
        tuple(
            (snapshot.tick - tick) / tick_hz if tick is not None else None for tick in visit_ticks
        ),
        bool(snapshot.navigation_path),
        memory.revision,
        snapshot.investigation,
    )


def target_source(action):
    if action == WAIT:
        return "wait"
    if action < 20:
        return "waypoint"
    return "visible" if action == 20 else "last_seen"


def record_waypoint_visits(snapshot, visit_ticks):
    """Stamp each waypoint reached by the post-tick Alien position, including a terminal tick."""
    alien = snapshot.alien
    for index, point in enumerate(snapshot.world_map.patrol_waypoints):
        if hypot(alien.x - point[0], alien.y - point[1]) <= snapshot.config.target_arrival_distance:
            visit_ticks[index] = snapshot.tick
