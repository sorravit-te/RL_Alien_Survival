"""Frozen vector observation and target-action contract. Change VERSION for semantics/ordering/shape/normalization changes.

Actions0..19 static waypoints,20 confirmed visible snapshot,21 investigate last-seen,
22 sound,23 Director,24..29 lockers,30 wait,31 guard departure. Unimplemented
slots are masked, never removed. A masked action is applied as wait, penalty0;
a value outside Discrete(32) raises ValueError. One decision is15 ticks at30Hz.
Action21 snapshots confirmed evidence revision, walks, then performs a bounded
stationary inspection. Exhausted revision masks21; fresh evidence re-enables it.
Base23..31 now carry investigation state instead of previously unused reservations.
Age scaling is physical clue/visit age, never the external episode countdown.
While action21's current clue is active, only 21 (continue) and 30 (cancel) are valid.
Core retargets action20 to each confirmed sensor point and hands reacquisition during
action21 to a written helper chase until the next decision.
A waypoint action holds as the only valid action until arrival, a dead path or new
confirmed evidence; the mask carries this, so the observation shape is unchanged.
No unconfirmed sight confidence/count is exposed. An unreachable observed floor
point may have no path; navigation never substitutes a heuristic target.
"""

# Schema identifier for new runs; archived checkpoints retain their recorded version.
VERSION = "alien-vector-v5-waypoint-commitment"
SEARCH_CONTROL = "continue_cancel_waypoint_commitment_v1"
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


def search_contract():
    """Static/public identity recorded with runs and checkpoints."""
    return {
        "search_control": SEARCH_CONTROL,
        "contract_version": VERSION,
        "observation_shape": [OBSERVATION_SIZE],
        "action_n": ACTION_COUNT,
        "decision_ticks": DECISION_TICKS,
    }
