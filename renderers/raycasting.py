"""DDA wall renderer. Input is a read-only WorldSnapshot; no Core dependency."""

from dataclasses import dataclass
from math import cos, floor, inf, radians, sin, tan
from pathlib import Path

import pygame

WALL_TEXTURE = Path(__file__).resolve().parents[1] / 'assets/stradex/prepared/wall1.png'


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


class RaycastingRenderer:
    def __init__(self, texture_path=WALL_TEXTURE, size=(320, 180), fov=75):
        texture_path = Path(texture_path)
        if not texture_path.is_file():
            raise FileNotFoundError(
                f'Missing wall texture: {texture_path}\n'
                'Place your licensed Stradex pack in assets/stradex/ (including Flat/ and '
                'LICENSE.txt), then run: .venv/Scripts/python.exe -m tools.prepare_assets'
            )
        self.texture = pygame.image.load(str(texture_path))
        self.width, self.height = size
        self.plane_scale = tan(radians(fov) / 2)
        self.focal_length = self.width / (2 * self.plane_scale)
        self.frame = pygame.Surface(size)
        self.depth = ()
        self.columns = [self.texture.subsurface((x, 0, 1, self.texture.get_height()))
                        for x in range(self.texture.get_width())]

    def render(self, snapshot, pitch=0.0):
        # Simulated look only: shift projection vertically, never change rays/depth.
        # A positive pitch looks up, moving the horizon down the screen.
        offset = round(self.focal_length * tan(pitch))
        horizon = self.height // 2 + offset
        split = max(0, min(self.height, horizon))
        self.frame.fill((29, 33, 40))
        self.frame.fill((48, 44, 40), (0, split, self.width, self.height - split))
        camera = snapshot.survivor
        facing_x, facing_y = cos(camera.yaw), sin(camera.yaw)
        plane_x, plane_y = -facing_y * self.plane_scale, facing_x * self.plane_scale
        depths = []
        tw, th = self.texture.get_size()
        for column in range(self.width):
            screen_x = 2 * (column + 0.5) / self.width - 1
            dx, dy = facing_x + plane_x * screen_x, facing_y + plane_y * screen_x
            hit = cast_ray(snapshot.world_map, camera.x, camera.y, dx, dy)
            depths.append(hit.distance)
            height = max(1, round(self.focal_length / max(hit.distance, 1e-6)))
            # One tile is one wall-height; preserve non-square texture aspect by tiling.
            u = (hit.wall_position / (tw / th)) % 1
            texture_x = min(tw - 1, int(u * tw))
            if (hit.side == 0 and dx > 0) or (hit.side == 1 and dy < 0):
                texture_x = tw - 1 - texture_x
            strip = pygame.transform.scale(self.columns[texture_x], (1, height))
            self.frame.blit(strip, (column, (self.height - height) // 2 + offset))
        self.depth = tuple(depths)
        return self.frame
