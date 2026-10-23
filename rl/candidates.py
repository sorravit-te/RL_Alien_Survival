"""Candidates derive solely from explicitly allowed knowledge, never WorldState."""

from dataclasses import dataclass

import numpy as np

from rl.contract import ACTION_COUNT, WAIT


@dataclass(frozen=True)
class AllowedAlienView:
    position: tuple[float, float]
    yaw: float
    chasing: bool
    current_action: int
    visible: bool
    last_seen: tuple[float, float] | None
    last_seen_age_seconds: float | None
    waypoints: tuple[tuple[float, float], ...]
    visit_ages: tuple[float | None, ...]
    navigation_active: bool


@dataclass(frozen=True)
class Candidate:
    valid: bool = False
    kind: int = 0  # waypoint0,visible1,lastseen2,sound3,director4,locker5,wait6,guard7
    target: tuple[float, float] | None = None
    clue_age: float = 0.0
    visit_age: float | None = None


def build_candidates(view: AllowedAlienView):
    if not isinstance(view, AllowedAlienView):
        raise TypeError("Candidates require AllowedAlienView, not full world state")
    if len(view.waypoints) > 20 or len(view.visit_ages) != len(view.waypoints):
        raise ValueError("Waypoint/visit contract mismatch")
    slots = [Candidate() for _ in range(ACTION_COUNT)]
    for index, point in enumerate(view.waypoints):
        slots[index] = Candidate(True, 0, point, 0.0, view.visit_ages[index])
    if view.visible and view.last_seen is not None:
        slots[20] = Candidate(True, 1, view.last_seen)
    if view.last_seen is not None:
        slots[21] = Candidate(True, 2, view.last_seen, view.last_seen_age_seconds or 0.0)
    slots[WAIT] = Candidate(True, 6)
    return tuple(slots)


def candidate_masks(candidates):
    return np.asarray([c.valid for c in candidates], dtype=np.bool_)
