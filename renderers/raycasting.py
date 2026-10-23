"""DDA wall renderer. Input is a read-only WorldSnapshot; no Core dependency."""

from dataclasses import dataclass
from math import cos, floor, inf, radians, sin, tan
from pathlib import Path

import pygame

WALL_TEXTURE = Path(__file__).resolve().parents[1] / 'assets/stradex/prepared/wall1.png'


@dataclass(frozen=True)
class SpriteObject:
    object_id: str
    kind: str
    x: float
    y: float


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
    def __init__(self, texture_path=WALL_TEXTURE, size=(320, 180), fov=75, material_paths=None):
        texture_path = Path(texture_path)
        if not texture_path.is_file():
            raise FileNotFoundError(
                f'Missing wall texture: {texture_path}\n'
                'Place your licensed Stradex pack in assets/stradex/ (including Flat/ and '
                'LICENSE.txt), then run: .venv/Scripts/python.exe -m tools.prepare_assets'
            )
        self.texture = pygame.image.load(str(texture_path))
        self.material_textures = {"wall": self.texture}
        for material_id, material_path in (material_paths or {}).items():
            if not isinstance(material_id, str) or not material_id:
                raise ValueError("Invalid material ID")
            path = Path(material_path)
            if not path.is_file():
                raise FileNotFoundError(f"Material {material_id}: {path}")
            self.material_textures[material_id] = pygame.image.load(str(path))
        self.material_columns = {
            key: [surface.subsurface((x, 0, 1, surface.get_height()))
                  for x in range(surface.get_width())]
            for key, surface in self.material_textures.items()
        }
        self.width, self.height = size
        self.plane_scale = tan(radians(fov) / 2)
        self.focal_length = self.width / (2 * self.plane_scale)
        self.frame = pygame.Surface(size)
        self.depth = ()
        self.sprites = self._placeholders()
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
        for column in range(self.width):
            screen_x = 2 * (column + 0.5) / self.width - 1
            dx, dy = facing_x + plane_x * screen_x, facing_y + plane_y * screen_x
            hit = cast_ray(snapshot.world_map, camera.x, camera.y, dx, dy)
            depths.append(hit.distance)
            height = max(1, round(self.focal_length / max(hit.distance, 1e-6)))
            material_id = snapshot.world_map.wall_material(hit.cell_x, hit.cell_y)
            texture = self.material_textures.get(material_id, self.texture)
            columns = self.material_columns.get(material_id, self.material_columns["wall"])
            tw, th = texture.get_size()
            # One tile is one wall-height; preserve non-square texture aspect by tiling.
            u = (hit.wall_position / (tw / th)) % 1
            texture_x = min(tw - 1, int(u * tw))
            if (hit.side == 0 and dx > 0) or (hit.side == 1 and dy < 0):
                texture_x = tw - 1 - texture_x
            strip = pygame.transform.scale(columns[texture_x], (1, height))
            self.frame.blit(strip, (column, (self.height - height) // 2 + offset))
        self.depth = tuple(depths)
        self._draw_objects(snapshot, horizon)
        return self.frame

    @staticmethod
    def _placeholders():
        """Self-made pictograms; no purchased sprites or collision side effects."""
        card = pygame.Surface((64, 40), pygame.SRCALPHA)
        pygame.draw.rect(card, (245, 205, 65), (2, 2, 60, 36), border_radius=4)
        pygame.draw.rect(card, (35, 55, 65), (5, 6, 54, 8))
        pygame.draw.rect(card, (65, 170, 195), (10, 19, 15, 12))
        pygame.draw.line(card, (35, 55, 65), (32, 22), (53, 22), 3)
        pygame.draw.line(card, (35, 55, 65), (32, 29), (48, 29), 3)
        exit_sign = pygame.Surface((64, 96), pygame.SRCALPHA)
        pygame.draw.rect(exit_sign, (20, 80, 65), (2, 2, 60, 94))
        pygame.draw.rect(exit_sign, (80, 240, 165), (2, 2, 60, 94), 4)
        pygame.draw.rect(exit_sign, (220, 255, 235), (29, 26, 7, 38))
        pygame.draw.polygon(exit_sign, (220, 255, 235), [(17, 38), (32, 20), (47, 38)])
        pygame.draw.rect(exit_sign, (80, 240, 165), (12, 78, 40, 5))
        alien = pygame.Surface((64,96), pygame.SRCALPHA)
        pygame.draw.ellipse(alien,(110,50,135),(8,4,48,52))
        pygame.draw.polygon(alien,(85,35,110),[(18,40),(46,40),(60,88),(4,88)])
        pygame.draw.ellipse(alien,(165,255,90),(16,24,12,7))
        pygame.draw.ellipse(alien,(165,255,90),(36,24,12,7))
        pygame.draw.rect(alien,(45,18,60),(17,84,10,12))
        pygame.draw.rect(alien,(45,18,60),(37,84,10,12))
        return {"keycard": (card, .50, .3125), "exit": (exit_sign, .65, .975),
                "alien": (alien, .48, .9)}

    def _draw_objects(self, snapshot, horizon):
        camera = snapshot.survivor
        fx, fy = cos(camera.yaw), sin(camera.yaw)
        projected = []
        objects=list(snapshot.world_map.objects)
        if snapshot.alien is not None:
            objects.append(SpriteObject("alien","alien",snapshot.alien.x,snapshot.alien.y))
        for obj in objects:
            if obj.kind == "keycard" and snapshot.has_keycard:
                continue
            rx, ry = obj.x - camera.x, obj.y - camera.y
            depth = rx * fx + ry * fy
            if depth <= .05:
                continue  # behind camera / near plane, avoid unbounded sprite scaling
            across = -rx * fy + ry * fx
            projected.append((depth, obj.object_id, across, obj))
        for depth, _, across, obj in sorted(projected, key=lambda p: (p[0],p[1]), reverse=True):
            sprite, world_width, world_height = self.sprites[obj.kind]
            width = max(1, round(self.focal_length * world_width / depth))
            height = max(1, round(self.focal_length * world_height / depth))
            left = round(self.width / 2 + self.focal_length * across / depth - width / 2)
            # Both pictures stand on the floor; card is raised to eye level.
            base = .35 if obj.kind == "keycard" else 0
            top = round(horizon + self.focal_length * (.5 - base - world_height) / depth)
            for column in range(max(0, left), min(self.width, left + width)):
                if depth >= self.depth[column]:
                    continue
                tx = min(sprite.get_width() - 1,
                         int((column - left) * sprite.get_width() / width))
                strip = sprite.subsurface((tx, 0, 1, sprite.get_height()))
                strip = pygame.transform.scale(strip, (1, height))
                self.frame.blit(strip, (column, top))
