"""Frozen vector observation and target-action contract. Change VERSION for semantics/ordering/shape/normalization changes.

Actions0..19 static waypoints,20 confirmed visible snapshot,21 investigate last-seen,
22 sound,23 Director,24..29 lockers,30 wait,31 guard departure. Unimplemented
slots are masked, never removed. A masked action is applied as wait, penalty0;
a value outside Discrete(32) raises ValueError. One decision is15 ticks at30Hz.
Action21 snapshots confirmed evidence revision, walks, then performs a bounded
stationary inspection. Exhausted revision masks21; fresh evidence re-enables it.
Base23..31 now carry investigation state instead of previously unused reservations.
Age scaling is physical clue/visit age, never the external episode countdown.
No unconfirmed sight confidence/count is exposed. An unreachable observed floor
point may have no path; navigation never substitutes a heuristic target.
"""

# Schema identifier for new runs; archived checkpoints retain their recorded version.
VERSION = "alien-vector-v2"
ACTION_COUNT = 32
WAIT = 30
DECISION_TICKS = 15
WORLD_SCALE = 40.0
DISTANCE_SCALE = 2**0.5 * WORLD_SCALE
AGE_SCALE_SECONDS = 60.0
ACTION_NAMES = tuple(
    [f"waypoint_{i}" for i in range(20)]
    + ["visible", "investigate_last_seen", "sound", "director"]
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
    "last_seen_investigated",
    "investigation_active",
    "investigation_sweeping",
    "investigation_progress",
    "investigation_target_valid",
    "investigation_target_dx",
    "investigation_target_dy",
    "investigation_revision_current",
    "investigation_complete",
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
