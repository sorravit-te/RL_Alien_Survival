"""Automatic sliding doors: they block sight while shut and never block movement.

A door opens for anyone walking up its passage, so a body behind the wall beside it does
not trigger it. Opening beats the fastest runner to the door, which keeps navigation and
collision unchanged and means a door can never close on anyone.
"""

from dataclasses import dataclass

SIGHT_CLEAR = 0.5  # a door stops blocking sight once it is this far open


@dataclass(frozen=True)
class DoorSight:
    """The world as seen by sensors: shut doors count as solid, nothing else changes."""

    world: object
    closed: frozenset

    @property
    def objects(self):
        return self.world.objects

    def blocked(self, x, y):
        return (x, y) in self.closed or self.world.blocked(x, y)


def sight_world(snapshot):
    if not snapshot.closed_doors:
        return snapshot.world_map
    return DoorSight(snapshot.world_map, frozenset(snapshot.closed_doors))


class Doors:
    def __init__(self, world, config):
        self.cells = world.doors
        self.reach = config.door_reach
        self.lateral = config.door_lateral
        self.step = 1 / (config.door_seconds * config.tick_hz)
        # A passage runs along x when both of its x neighbours are floor.
        self.along_x = tuple(
            not world.blocked(x - 1, y) and not world.blocked(x + 1, y) for x, y in self.cells
        )
        self.openness = (0.0,) * len(self.cells)
        self.closed = self.cells
        self.opened = ()  # doors the Survivor started opening this tick

    def _wanted(self, cell, along_x, bodies):
        for x, y in bodies:
            ahead = x - cell[0] - 0.5 if along_x else y - cell[1] - 0.5
            across = y - cell[1] - 0.5 if along_x else x - cell[0] - 0.5
            if abs(ahead) <= self.reach and abs(across) <= self.lateral:
                return True
        return False

    def update(self, bodies):
        """The first body is the Survivor; a door the Alien opens tells it nothing."""
        before = self.openness
        self.openness = tuple(
            min(1.0, amount + self.step)
            if self._wanted(cell, along_x, bodies)
            else max(0.0, amount - self.step)
            for cell, along_x, amount in zip(self.cells, self.along_x, self.openness, strict=True)
        )
        self.closed = tuple(
            cell
            for cell, amount in zip(self.cells, self.openness, strict=True)
            if amount < SIGHT_CLEAR
        )
        self.opened = tuple(
            cell
            for cell, along_x, old, new in zip(
                self.cells, self.along_x, before, self.openness, strict=True
            )
            if old == 0.0 < new and self._wanted(cell, along_x, bodies[:1])
        )
