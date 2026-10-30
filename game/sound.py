"""Noise follows the floor to the Alien and never passes through a wall.

Footfalls and doors are Core events, so they behave the same without a speaker. The Alien
keeps only a coarse region: every point in a region gives one fixed floor point, so
hearing many footfalls can never be averaged into the Survivor's real position.
"""

from dataclasses import dataclass, replace
from heapq import heappop, heappush
from math import floor, hypot, inf, sqrt

DIAGONAL = sqrt(2)
STEPS = tuple(
    (dx, dy, DIAGONAL if dx and dy else 1.0) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy
)


@dataclass(frozen=True)
class HeardSnapshot:
    point: tuple[float, float] | None = None
    age_ticks: int | None = None
    revision: int = 0  # increments on every noise that reaches the Alien
    checked: bool = False  # the Alien finished looking around this point


def carries(world, closed, source, listener, reach, door_cost):
    """Whether a noise at source survives the walk to listener; a shut door costs extra."""
    if hypot(source[0] - listener[0], source[1] - listener[1]) > reach:
        return False
    start = (floor(source[0]), floor(source[1]))
    goal = (floor(listener[0]), floor(listener[1]))
    costs, frontier = {start: 0.0}, [(0.0, start)]
    while frontier:
        cost, cell = heappop(frontier)
        if cell == goal:
            return True
        if cost > costs[cell]:
            continue
        x, y = cell
        for dx, dy, step in STEPS:
            near = (x + dx, y + dy)
            if world.blocked(*near) or (
                dx and dy and (world.blocked(x + dx, y) or world.blocked(x, y + dy))
            ):
                continue
            total = cost + step + (door_cost if near in closed else 0.0)
            if total <= reach and total < costs.get(near, inf):
                costs[near] = total
                heappush(frontier, (total, near))
    return False


def region_point(world, x, y, size):
    """The floor cell nearest the middle of the region around (x, y)."""
    left, top = floor(x) // size * size, floor(y) // size * size
    middle = (left + size / 2, top + size / 2)
    cells = (
        (cell_x + 0.5, cell_y + 0.5)
        for cell_y in range(top, top + size)
        for cell_x in range(left, left + size)
        if not world.blocked(cell_x, cell_y)
    )
    return min(cells, key=lambda p: (hypot(p[0] - middle[0], p[1] - middle[1]), p))


def listen(world, closed, alien, noises, previous, config):
    """One tick of hearing; noises are (point, reach), an old clue ages in place."""
    for source, reach in noises:
        if carries(world, closed, source, (alien.x, alien.y), reach, config.sound_door_cost):
            point = region_point(world, *source, config.sound_region)
            return HeardSnapshot(point, 0, previous.revision + 1)
    if previous.point is None:
        return previous
    return replace(previous, age_ticks=previous.age_ticks + 1)
