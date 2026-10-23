"""Replaceable target selector; receives no WorldState or Survivor reference."""

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class ControllerState:
    target: tuple[float, float] | None = None
    source: str = "patrol"
    patrol_index: int = 0
    searched_last_seen: tuple[float, float] | None = None


def choose_target(alien, perception, waypoints, previous, arrival_distance):
    if perception.visible:
        return ControllerState(perception.last_seen, "visible", previous.patrol_index)
    searched = previous.searched_last_seen
    if perception.last_seen is not None and searched != perception.last_seen:
        if (
            hypot(alien.x - perception.last_seen[0], alien.y - perception.last_seen[1])
            > arrival_distance
        ):
            return ControllerState(
                perception.last_seen, "last_seen", previous.patrol_index, searched
            )
        searched = perception.last_seen
    if not waypoints:
        return ControllerState((alien.x, alien.y), "patrol", 0, searched)
    index = previous.patrol_index % len(waypoints)
    if hypot(alien.x - waypoints[index][0], alien.y - waypoints[index][1]) <= arrival_distance:
        index = (index + 1) % len(waypoints)
    return ControllerState(waypoints[index], "patrol", index, searched)
