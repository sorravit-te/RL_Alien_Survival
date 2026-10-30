"""Replaceable target selector; receives no WorldState or Survivor reference."""

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class ControllerState:
    target: tuple[float, float] | None = None
    source: str = "patrol"
    patrol_index: int = 0
    clue_revision: int | None = None  # confirmed evidence when this target was assigned


def choose_target(
    alien, perception, waypoints, previous, arrival_distance, investigated_revision=None
):
    if perception.visible:
        return ControllerState(perception.last_seen, "visible", previous.patrol_index)
    if (
        perception.last_seen is not None
        and perception.revision is not None
        and perception.revision != investigated_revision
    ):
        # Arrival alone does not mean inspected: Core owns the bounded sweep.
        return ControllerState(perception.last_seen, "last_seen", previous.patrol_index)
    if not waypoints:
        return ControllerState((alien.x, alien.y), "patrol", 0)
    index = previous.patrol_index % len(waypoints)
    if hypot(alien.x - waypoints[index][0], alien.y - waypoints[index][1]) <= arrival_distance:
        index = (index + 1) % len(waypoints)
    return ControllerState(waypoints[index], "patrol", index)
