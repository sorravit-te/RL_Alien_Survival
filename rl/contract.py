"""Frozen 2B contract. Change VERSION for any ordering/shape/normalization change.

Actions0..19 static waypoints,20 confirmed visible snapshot,21 last-seen,
22 sound,23 Director,24..29 lockers,30 wait,31 guard departure. Unimplemented
slots are masked, never removed. A masked action is applied as wait, penalty0;
a value outside Discrete(32) raises ValueError. One decision is15 ticks at30Hz.
Age scaling is physical clue/visit age, never the external episode countdown.
No unconfirmed sight confidence/count is exposed. An unreachable observed floor
point may have no path; navigation never substitutes a heuristic target.
"""

VERSION = "alien-vector-2b-v1"
ACTION_COUNT = 32
WAIT = 30
DECISION_TICKS = 15
WORLD_SCALE = 40.0
DISTANCE_SCALE = 2**0.5 * WORLD_SCALE
AGE_SCALE_SECONDS = 60.0
ACTION_NAMES = tuple(
    [f"waypoint_{i}" for i in range(20)]
    + ["visible", "last_seen", "sound", "director"]
    + [f"locker_{i}" for i in range(6)]
    + ["wait", "guard_departure"]
)
BASE_FIELDS = (
    "self_x",
    "self_y",
    "yaw_cos",
    "yaw_sin",
    "chasing",
    "current_action",
    "visible_valid",
    "visible_dx",
    "visible_dy",
    "last_seen_valid",
    "last_seen_dx",
    "last_seen_dy",
    "last_seen_age",
    "last_heard_valid",
    "last_heard_dx",
    "last_heard_dy",
    "last_heard_age",
    "director_valid",
    "director_zone",
    "director_age",
    "interaction_valid",
    "interaction_progress",
    "navigation_active",
    "stuck_valid",
    "stuck",
    "guard_valid",
    "guard_timer",
    "guard_cooldown",
    "doors_valid",
    "door_state",
    "lockers_valid",
    "locker_result",
)
CANDIDATE_FIELDS = (
    "valid",
    "kind",
    "target_valid",
    "x",
    "y",
    "distance",
    "clue_age",
    "visit_age_valid",
    "visit_age",
)
OBSERVATION_SIZE = len(BASE_FIELDS) + ACTION_COUNT * len(CANDIDATE_FIELDS) + ACTION_COUNT
assert OBSERVATION_SIZE == 352
