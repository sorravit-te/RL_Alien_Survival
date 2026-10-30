"""Scan stations: a short list of floor points that together see every floor cell.

Built from public geometry only, walls and where the doors stand, never an object or a
spawn. A shut door hides what is behind it, so every room needs a station of its own.
"""

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from math import hypot

from game.interaction import segment_clear

STATION_SIGHT = 7.0  # a room this far away counts as not searched yet


@dataclass(frozen=True)
class Walls:
    solid: frozenset

    def blocked(self, x, y):
        return (x, y) in self.solid


def _sight_lines(walls, spots):
    """Which spots see each other; a line works both ways, so each pair is tested once."""
    seen = {spot: {spot} for spot in spots}
    for index, a in enumerate(spots):
        for b in spots[index + 1 :]:
            if hypot(a[0] - b[0], a[1] - b[1]) <= STATION_SIGHT and segment_clear(walls, *a, *b):
                seen[a].add(b)
                seen[b].add(a)
    return seen


def _walk_steps(start, floor):
    """Steps from start to every floor point, moving one cell at a time."""
    steps, queue = {start: 0}, deque((start,))
    while queue:
        point = queue.popleft()
        x, y = point
        for near in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if near in floor and near not in steps:
                steps[near] = steps[point] + 1
                queue.append(near)
    return steps


@lru_cache(maxsize=2)
def plan_stations(rows, points, doors):
    """Greedy cover of the floor; also each station's walking distance to every point."""
    solid = {(x, y) for y, row in enumerate(rows) for x, tile in enumerate(row) if tile == "#"}
    door_points = {(x + 0.5, y + 0.5) for x, y in doors}
    spots = tuple(point for point in points if point not in door_points)
    seen = _sight_lines(Walls(frozenset(solid | set(doors))), spots)
    stations, unseen = [], set(spots)
    while unseen:
        best = max(spots, key=lambda spot: (len(seen[spot] & unseen), -spot[1], -spot[0]))
        stations.append(best)
        unseen -= seen[best]
    floor = frozenset(points)
    return tuple(stations), {station: _walk_steps(station, floor) for station in stations}
