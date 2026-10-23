"""float32[352]:32 base,32x9 candidates,32 masks; allowed-view input only."""

from math import cos, hypot, sin

import numpy as np

from rl.candidates import AllowedAlienView, build_candidates, candidate_masks
from rl.contract import AGE_SCALE_SECONDS, DISTANCE_SCALE, OBSERVATION_SIZE, WORLD_SCALE


def build_observation(view: AllowedAlienView):
    if not isinstance(view, AllowedAlienView):
        raise TypeError("Observations require AllowedAlienView, not full world state")
    # Flags guard every missing clue/system. Unconfirmed sensor data is absent.
    values = [
        view.position[0] / WORLD_SCALE,
        view.position[1] / WORLD_SCALE,
        cos(view.yaw),
        sin(view.yaw),
        float(view.chasing),
        view.current_action / 31,
    ]
    visible = view.visible and view.last_seen is not None
    dx = (view.last_seen[0] - view.position[0]) / WORLD_SCALE if view.last_seen else 0.0
    dy = (view.last_seen[1] - view.position[1]) / WORLD_SCALE if view.last_seen else 0.0
    values += [float(visible), dx if visible else 0.0, dy if visible else 0.0]
    values += [
        float(view.last_seen is not None),
        dx,
        dy,
        min(1.0, (view.last_seen_age_seconds or 0.0) / AGE_SCALE_SECONDS),
    ]
    values += [0.0] * 9  # last heard4,Director3,interaction2
    values += [float(view.navigation_active)] + [0.0] * 9  # navigation/stuck/guard/doors/lockers
    candidates = build_candidates(view)
    for candidate in candidates:
        if not candidate.valid:
            values += [0.0] * 9
            continue
        point = candidate.target
        values += [
            1.0,
            candidate.kind / 7,
            float(point is not None),
            point[0] / WORLD_SCALE if point else 0.0,
            point[1] / WORLD_SCALE if point else 0.0,
            hypot(point[0] - view.position[0], point[1] - view.position[1]) / DISTANCE_SCALE
            if point
            else 0.0,
            min(1.0, candidate.clue_age / AGE_SCALE_SECONDS),
            float(candidate.visit_age is not None),
            min(1.0, (candidate.visit_age or 0.0) / AGE_SCALE_SECONDS),
        ]
    values += candidate_masks(candidates).astype(np.float32).tolist()
    observation = np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0)
    assert observation.shape == (OBSERVATION_SIZE,)
    return observation
