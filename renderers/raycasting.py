"""DDA ray against the grid: the plain depth the ship renderer is checked against."""

from dataclasses import dataclass
from math import floor, inf


@dataclass(frozen=True)
class RayHit:
    distance: float  # parameter along ray; perpendicular depth for camera-plane rays
    cell_x: int
    cell_y: int
    side: int  # 0 = x boundary, 1 = y boundary
    wall_position: float


def cast_ray(world, x, y, dx, dy):
    if dx == 0 and dy == 0:
        raise ValueError('Ray direction must be nonzero')
    cx, cy = floor(x), floor(y)
    delta_x = abs(1 / dx) if dx else inf
    delta_y = abs(1 / dy) if dy else inf
    step_x, step_y = (1 if dx >= 0 else -1), (1 if dy >= 0 else -1)
    next_x = ((cx + 1 - x) if dx >= 0 else (x - cx)) * delta_x
    next_y = ((cy + 1 - y) if dy >= 0 else (y - cy)) * delta_y
    # Out-of-map tiles are solid, just as in collision.
    for _ in range(world.width + world.height + 2):
        if next_x < next_y:
            distance, next_x = next_x, next_x + delta_x
            cx += step_x
            side = 0
        else:
            distance, next_y = next_y, next_y + delta_y
            cy += step_y
            side = 1
        if world.blocked(cx, cy):
            position = y + distance * dy if side == 0 else x + distance * dx
            return RayHit(distance, cx, cy, side, position)
    raise ValueError('Ray origin must be inside the map')
