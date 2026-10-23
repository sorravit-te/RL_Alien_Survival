"""Single immutable geometry source for collision and future rendering."""

from dataclasses import dataclass
from math import floor, isfinite


@dataclass(frozen=True)
class WorldMap:
    rows: tuple[str, ...]
    spawn_points: tuple[tuple[float, float], ...]

    def __post_init__(self):
        # Own immutable copies even when a caller supplies lists.
        object.__setattr__(self, "rows", tuple(self.rows))
        if any(not isinstance(row, str) for row in self.rows):
            raise TypeError("Map rows must be strings")
        object.__setattr__(self, "spawn_points", tuple(tuple(p) for p in self.spawn_points))
        if not self.rows or not self.rows[0]:
            raise ValueError("Map must not be empty")
        if any(len(row) != self.width or set(row) - {"#", "."} for row in self.rows):
            raise ValueError("Map must be rectangular and contain only # or .")
        if not self.spawn_points:
            raise ValueError("Map needs at least one spawn point")
        if any(len(p) != 2 or not all(isfinite(v) for v in p) for p in self.spawn_points):
            raise ValueError("Invalid spawn point")

    @property
    def width(self):
        return len(self.rows[0])

    @property
    def height(self):
        return len(self.rows)

    def blocked(self, x: int, y: int) -> bool:
        return not (0 <= x < self.width and 0 <= y < self.height) or self.rows[y][x] == "#"

    def overlaps(self, x: float, y: float, radius: float) -> bool:
        if x - radius < 0 or y - radius < 0:
            return True
        if x + radius > self.width or y + radius > self.height:
            return True
        for cy in range(floor(y - radius), floor(y + radius) + 1):
            for cx in range(floor(x - radius), floor(x + radius) + 1):
                if self.blocked(cx, cy):
                    dx = x - min(max(x, cx), cx + 1)
                    dy = y - min(max(y, cy), cy + 1)
                    if dx * dx + dy * dy < radius * radius - 1e-12:
                        return True
        return False


def small_test_map() -> WorldMap:
    """Room, tall obstacle and L-shaped corridor; no doors or gameplay objects."""
    return WorldMap(
        (
            "############",
            "#......#####",
            "#......#####",
            "#...#......#",
            "#...#......#",
            "#......##..#",
            "#########..#",
            "#########..#",
            "############",
        ),
        ((2.5, 2.5), (2.5, 4.5)),
    )
