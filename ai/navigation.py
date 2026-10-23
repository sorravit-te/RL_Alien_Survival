"""Deterministic clearance-aware A* and movement toward an already chosen target."""

from heapq import heappop, heappush
from math import floor, hypot

from game.motion import segment_has_clearance


def find_path(world, start, target, radius):
    if world.overlaps(*start, radius) or world.overlaps(*target, radius):
        return ()
    if segment_has_clearance(world, start, target, radius):
        return (target,)
    source, goal = (floor(start[0]), floor(start[1])), (floor(target[0]), floor(target[1]))

    def center(cell):
        return (cell[0] + 0.5, cell[1] + 0.5)

    if not segment_has_clearance(world, start, center(source), radius):
        return ()
    if not segment_has_clearance(world, center(goal), target, radius):
        return ()
    frontier = [(0.0, 0.0, source)]
    costs = {source: 0.0}
    parents = {}
    while frontier:
        _, cost, node = heappop(frontier)
        if cost > costs[node]:
            continue
        if node == goal:
            cells = [node]
            while cells[-1] != source:
                cells.append(parents[cells[-1]])
            points = tuple(center(c) for c in reversed(cells)) + (target,)
            return tuple(p for i, p in enumerate(points) if i == 0 or p != points[i - 1])
        x, y = node
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)):
            neighbor = (x + dx, y + dy)
            point = center(neighbor)
            if world.overlaps(*point, radius):
                continue
            if (
                dx
                and dy
                and (
                    world.overlaps(x + dx + 0.5, y + 0.5, radius)
                    or world.overlaps(x + 0.5, y + dy + 0.5, radius)
                )
            ):
                continue
            if not segment_has_clearance(world, center(node), point, radius):
                continue
            new_cost = cost + hypot(dx, dy)
            if new_cost < costs.get(neighbor, float("inf")):
                costs[neighbor] = new_cost
                parents[neighbor] = node
                heappush(
                    frontier,
                    (
                        new_cost + hypot(neighbor[0] - goal[0], neighbor[1] - goal[1]),
                        new_cost,
                        neighbor,
                    ),
                )
    return ()


class Navigator:
    def __init__(self):
        self.target = None
        self.path = ()

    def set_target(self, world, start, target, radius):
        if target != self.target:
            self.target = target
            self.path = find_path(world, start, target, radius) if target is not None else ()

    def displacement(self, position, distance):
        while (
            self.path and hypot(self.path[0][0] - position[0], self.path[0][1] - position[1]) < 1e-9
        ):
            self.path = self.path[1:]
        if not self.path:
            return (0.0, 0.0)
        dx, dy = self.path[0][0] - position[0], self.path[0][1] - position[1]
        length = hypot(dx, dy)
        scale = min(distance, length) / length
        return dx * scale, dy * scale
