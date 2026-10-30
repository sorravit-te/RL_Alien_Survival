"""Candidates derive solely from explicitly allowed knowledge, never WorldState."""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from game.investigation import InvestigationSnapshot
from rl.contract import (
    ACTION_COUNT,
    CONTINUE_CANCEL,
    LEGACY_SEARCH_CONTROL,
    WAIT,
    validate_search_control,
)


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
    clue_revision: int | None = None
    investigation: InvestigationSnapshot = InvestigationSnapshot()
    search_control: str = LEGACY_SEARCH_CONTROL


@dataclass(frozen=True)
class Candidate:
    valid: bool = False
    kind: int = 0  # waypoint0,visible1,lastseen2,sound3,director4,locker5,wait6,guard7
    target: tuple[float, float] | None = None
    clue_age: float = 0.0
    visit_age: float | None = None
    evidence_revision: int | None = None


def current_search_clue(view):
    """New-mode eligibility uses only confirmed public clue/primitive fields."""
    point, revision, state = view.last_seen, view.clue_revision, view.investigation
    return (
        not view.visible
        and isinstance(point, (tuple, list))
        and len(point) == 2
        and all(type(x) in (int, float) and isfinite(x) for x in point)
        and type(revision) is int
        and revision > 0
        and revision != state.investigated_revision
        and not (state.phase == "completed" and state.revision == revision)
    )


def search_is_restricted(view):
    return (
        view.search_control == CONTINUE_CANCEL
        and current_search_clue(view)
        and view.investigation.active
        and view.investigation.revision == view.clue_revision
        and view.investigation.target == view.last_seen
    )


def build_candidates(view: AllowedAlienView):
    if not isinstance(view, AllowedAlienView):
        raise TypeError("Candidates require AllowedAlienView, not full world state")
    validate_search_control(view.search_control)
    if len(view.waypoints) > 20 or len(view.visit_ages) != len(view.waypoints):
        raise ValueError("Waypoint/visit contract mismatch")
    slots = [Candidate() for _ in range(ACTION_COUNT)]
    for index, point in enumerate(view.waypoints):
        slots[index] = Candidate(True, 0, point, 0.0, view.visit_ages[index])
    if view.visible and view.last_seen is not None:
        slots[20] = Candidate(True, 1, view.last_seen)
    if (
        view.last_seen is not None
        and view.clue_revision is not None
        and view.clue_revision != view.investigation.investigated_revision
    ):
        slots[21] = Candidate(
            True,
            2,
            view.last_seen,
            view.last_seen_age_seconds or 0.0,
            evidence_revision=view.clue_revision,
        )
    if view.search_control == CONTINUE_CANCEL:
        if not current_search_clue(view):
            slots[21] = Candidate()
        if search_is_restricted(view):
            continuation = slots[21]
            slots = [Candidate() for _ in range(ACTION_COUNT)]
            slots[21] = continuation
    slots[WAIT] = Candidate(True, 6)
    return tuple(slots)


def candidate_masks(candidates):
    return np.asarray([c.valid for c in candidates], dtype=np.bool_)
