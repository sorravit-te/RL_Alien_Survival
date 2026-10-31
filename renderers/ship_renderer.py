"""Draw the ship as real 3D surfaces over the 2D Game Core.

Read-only like RaycastingRenderer: it draws a WorldSnapshot and owns no game rules. All
motion in the picture comes from snapshot.tick, so a frame replays exactly. Pitch remains
simulated: it slides the horizon and never changes what the Core senses.
"""

from math import cos, exp, pi, radians, sin, tan
from os import cpu_count
from pathlib import Path

import numba
import numpy as np
import pygame

from renderers.alien_sprites import FRAMES as ALIEN_FRAMES
from renderers.alien_sprites import AlienSprites
from renderers.ship_kernels import (
    FOG_DENSITY,
    TORCH_BLUE,
    TORCH_FALLOFF,
    TORCH_GREEN,
    TORCH_POWER,
    TORCH_RED,
    TRI_FIELDS,
    draw_sprite,
    finish_frame,
    project,
    rasterize,
    shade,
)
from renderers.ship_mesh import WALL_TOP, build_mesh
from renderers.ship_theme import FLICKER, LIGHT_GRID, PULSE, STEADY, ShipTheme, stable

TEXTURES = Path(__file__).resolve().parents[1] / "assets/stradex/prepared/ship_textures.npz"
EYE_HEIGHT = 1.12
TORCH_COLOR = np.array([TORCH_RED, TORCH_GREEN, TORCH_BLUE], np.float32)
TICKS_PER_SECOND = 30
RASTER_BANDS = 24
SKY_TURNS = 4  # star texture repeats around one full turn of the camera


def make_sprite(draw, size):
    """Self-made sprite: draw(color, glow) paints two surfaces into RGBA and glow bytes."""
    color = pygame.Surface(size, pygame.SRCALPHA)
    glow = pygame.Surface(size, pygame.SRCALPHA)
    draw(color, glow)
    rgb = pygame.surfarray.array3d(color).swapaxes(0, 1)
    alpha = pygame.surfarray.array_alpha(color).swapaxes(0, 1)
    shine = pygame.surfarray.array_alpha(glow).swapaxes(0, 1)
    shine_rgb = pygame.surfarray.array3d(glow).swapaxes(0, 1)
    rgb = np.where(shine[..., None] > 0, shine_rgb, rgb)
    return np.dstack((rgb, np.maximum(alpha, shine))), shine


def draw_alien(color, glow):
    body = (6, 7, 9, 255)
    pygame.draw.polygon(
        color,
        body,
        [(46, 6), (70, 4), (88, 22), (82, 48), (98, 74), (108, 150), (92, 158), (86, 110), (82, 196),
         (92, 298), (72, 300), (62, 200), (52, 300), (32, 298), (42, 196), (38, 110), (32, 158),
         (16, 150), (26, 74), (42, 48), (36, 22)],
    )  # fmt: skip
    for x, y in ((50, 26), (74, 26)):
        pygame.draw.ellipse(glow, (190, 255, 120, 255), (x - 7, y - 3, 14, 7))


def draw_keycard(color, glow):
    pygame.draw.rect(color, (232, 190, 62, 255), (2, 2, 60, 36), border_radius=4)
    pygame.draw.rect(color, (35, 55, 65, 255), (5, 6, 54, 8))
    pygame.draw.rect(glow, (255, 214, 90, 255), (2, 2, 60, 36), 3, border_radius=4)
    pygame.draw.rect(glow, (120, 230, 255, 255), (10, 19, 15, 12))


def draw_exit(color, glow):
    pygame.draw.rect(color, (12, 34, 28, 255), (2, 2, 60, 46))
    pygame.draw.rect(glow, (90, 255, 170, 255), (2, 2, 60, 46), 3)
    pygame.draw.rect(glow, (220, 255, 235, 255), (29, 16, 7, 24))
    pygame.draw.polygon(glow, (220, 255, 235, 255), [(17, 22), (32, 8), (47, 22)])


class ShipRenderer:
    def __init__(
        self, world, texture_path=TEXTURES, size=(960, 540), fov=75, alien_path=ALIEN_FRAMES
    ):
        texture_path = Path(texture_path)
        if not texture_path.is_file():
            raise FileNotFoundError(
                f"Missing ship textures: {texture_path}\n"
                "Place your licensed Stradex pack in assets/stradex/ (including Materials/), "
                "then run: .venv/Scripts/python.exe -m tools.bake_ship_textures"
            )
        textures = np.load(texture_path)
        levels = [textures[f"level{i}"] for i in range(3)]
        self.texture_size = levels[0].shape[1]
        self.texels = np.concatenate(
            [level.reshape(-1, 4).view(np.uint32).ravel() for level in levels]
        )
        sizes = [level.reshape(-1, 4).shape[0] for level in levels]
        self.starts = np.array((0, sizes[0], sizes[0] + sizes[1]), np.int64)
        self.theme = ShipTheme(world)
        self.mesh = build_mesh(self.theme, textures["names"])
        self.door_rest = self.mesh["vertices"][self.mesh["door_quads"]].copy()
        self.width, self.height = size
        self.plane_scale = tan(radians(fov) / 2)
        self.focal_length = self.width / (2 * self.plane_scale)
        self.frame = pygame.Surface(size)
        self.depth = ()
        self.color = np.zeros((self.height, self.width, 3), np.float32)
        self.inverse_depth = np.zeros((self.height, self.width), np.float64)
        self.ids = np.zeros((self.height, self.width), np.int32)
        quads = len(self.mesh["vertices"])
        self.tris = np.zeros((quads * 3, TRI_FIELDS), np.float64)
        self.tri_quad = np.zeros(quads * 3, np.int32)
        self.pixels = np.zeros((self.width, self.height, 3), np.uint8)
        numba.set_num_threads(max(2, min(numba.config.NUMBA_NUM_THREADS, (cpu_count() or 4) - 2)))
        across = (2 * (np.arange(self.width, dtype=np.float32) + 0.5) / self.width - 1)[None, :]
        rows = np.arange(self.height, dtype=np.float32) + 0.5
        down = ((rows / self.height) * 2 - 1)[:, None]
        self.torch_beam = np.exp(-(across**2 + down**2) / 0.5).astype(np.float32)
        self.vignette = (1 - 0.42 * np.clip(across**2 * 0.75 + down**2 * 0.8, 0, 1)).astype(
            np.float32
        )
        noise = np.random.default_rng(404).normal(0, 1.8, (self.height, self.width))
        self.grain = noise.astype(np.float32)
        # Lifts mid tones so textures read, and keeps a true black toe for the shadows.
        level = np.arange(256, dtype=np.float32) / 255
        curve = np.clip(level**0.8 * (1.02 - 0.16 * (1 - level) ** 6), 0, 1)
        self.tone = (curve * 255 + 0.5).astype(np.uint8)
        self.sprites = {
            "alien": (make_sprite(draw_alien, (124, 304)), 0.72, 1.85, 0.0),
            "keycard": (make_sprite(draw_keycard, (64, 40)), 0.42, 0.26, 0.95),
            "exit": (make_sprite(draw_exit, (64, 50)), 0.62, 0.48, 1.2),
        }
        self.alien_art = AlienSprites(alien_path)
        spawn = self.theme.world.spawn_points[0]
        light, *_ = self._draw(*spawn, 0.0, self.height / 2, 0)  # compile first
        self._draw_sprite(light, self.height / 2, 1.0, 0.0, "exit", *spawn)

    def _light_now(self, tick):
        """Blend the light layers; only their strength changes from frame to frame."""
        seconds = tick / TICKS_PER_SECOND
        burst = stable(tick // 4, 3, 21)
        flicker = 0.1 if burst < 0.1 else 0.5 if burst < 0.26 else 1.0
        pulse = 0.4 + 0.6 * (0.5 + 0.5 * sin(seconds * 2.2)) ** 2
        layers = self.theme.layers
        return layers[STEADY] + layers[FLICKER] * flicker + layers[PULSE] * pulse

    def _draw(self, x, y, yaw, horizon, tick):
        mesh = self.mesh
        fx, fy = cos(yaw), sin(yaw)
        view = (float(x), float(y), fx, fy, self.plane_scale, float(horizon), self.focal_length)
        count = project(
            mesh["vertices"], mesh["uvs"], mesh["normals"], view, EYE_HEIGHT, self.width,
            self.height, self.tris, self.tri_quad,
        )  # fmt: skip
        rasterize(self.tris, count, self.inverse_depth, self.ids, RASTER_BANDS)
        light = self._light_now(tick)
        shade(
            self.color, self.inverse_depth, self.ids, self.tris, self.tri_quad, self.texels,
            self.starts, self.texture_size, mesh["textures"], mesh["tints"], mesh["shifts"],
            mesh["shades"], light, LIGHT_GRID, self.torch_beam, WALL_TOP,
            yaw / (2 * pi) * SKY_TURNS, float(horizon),
        )  # fmt: skip
        return light, fx, fy

    def _place_doors(self, openness):
        """Slide the leaves; a wall hides whatever passes behind its door frame."""
        mesh = self.mesh
        amount = np.asarray(openness, np.float32)[mesh["door_index"]]
        moved = self.door_rest + (mesh["door_slide"] * amount[:, None])[:, None, :]
        mesh["vertices"][mesh["door_quads"]] = moved

    def render(self, snapshot, pitch=0.0):
        camera = snapshot.survivor
        if snapshot.door_open:
            self._place_doors(snapshot.door_open)
        horizon = self.height / 2 + self.focal_length * tan(pitch)
        light, fx, fy = self._draw(camera.x, camera.y, camera.yaw, horizon, snapshot.tick)
        middle = self.inverse_depth[self.height // 2]
        self.depth = tuple((1 / np.maximum(middle, 1e-6)).tolist())
        alien = self.alien_art.sprite(snapshot)
        if alien is not None:
            self.sprites["alien"] = alien
        for sprite in self._visible_sprites(snapshot, fx, fy):
            self._draw_sprite(light, horizon, *sprite)
        finish_frame(
            self.color, self.vignette, self.grain, snapshot.tick * 37 % self.height, self.tone,
            self.pixels,
        )  # fmt: skip
        pygame.surfarray.blit_array(self.frame, self.pixels)
        return self.frame

    def _visible_sprites(self, snapshot, fx, fy):
        camera = snapshot.survivor
        placed = [
            (obj.x, obj.y, obj.kind)
            for obj in snapshot.world_map.objects
            if not (obj.kind == "keycard" and snapshot.has_keycard)
        ]
        if snapshot.alien is not None:
            placed.append((snapshot.alien.x, snapshot.alien.y, "alien"))
        seen = []
        for x, y, kind in placed:
            rx, ry = x - camera.x, y - camera.y
            depth = rx * fx + ry * fy
            if depth > 0.05:
                seen.append((depth, -rx * fy + ry * fx, kind, x, y))
        return sorted(seen, key=lambda item: item[0], reverse=True)

    def _draw_sprite(self, light, horizon, depth, across, kind, x, y):
        (picture, glow), world_w, world_h, base = self.sprites[kind]
        width = max(1, round(self.focal_length * world_w / depth))
        height = max(1, round(self.focal_length * world_h / depth))
        left = round(self.width / 2 + self.focal_length * across / depth - width / 2)
        top = round(horizon + self.focal_length * (EYE_HEIGHT - base - world_h) / depth)
        spot = light[int(y * LIGHT_GRID), int(x * LIGHT_GRID)]
        fog = exp(-depth * FOG_DENSITY)
        near = TORCH_POWER / (1 + TORCH_FALLOFF * depth * depth) * 0.6
        lit = np.asarray(spot * fog + near * TORCH_COLOR, np.float32)
        box = (left, top, width, height)
        draw_sprite(self.color, self.inverse_depth, picture, glow, box, 1 / depth, lit)
