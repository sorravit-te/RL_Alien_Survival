"""Zone palettes and baked ship light. Read-only art data, no game rules.

Every choice is a stable hash of the cell, so the ship looks the same on every run.
Lights are placed from the zone rows, then cast with shadows against the solid grid.
"""

from dataclasses import dataclass

import numpy as np

from game.ship_deck import EXITS
from renderers.ship_zones import ZONE_ROWS

LIGHT_GRID = 4  # light samples per world unit
LIGHT_RADIUS = 10.0
RAY_SAMPLES = 40
LAMP_LOW, LAMP_HIGH = 1.92, 2.3  # lamp height under a plain ceiling and a raised one
EXIT_LAMP = 1.7
TONE_KEPT = 0.3  # how much of a zone's hue its lamps keep; metal should still read as metal
AMBIENT = 3.0  # a dead stretch of corridor stays readable, and the Alien stands out against it
STEADY, FLICKER, PULSE = 0, 1, 2
LIGHT_LAYERS = 3


@dataclass(frozen=True)
class Zone:
    bays: tuple[str, ...]  # panels set into wall recesses
    pillar: str
    lintel: str
    sill: str
    cornice: str
    floors: tuple[str, ...]
    ceilings: tuple[str, ...]
    lamp: str
    color: tuple[float, float, float]
    ambient: tuple[float, float, float]
    spacing: int  # cells between ceiling lights
    dead_share: float  # lights that no longer work
    mode: int = STEADY
    power: float = 1.0
    blood: float = 0.0  # share of panels and floors with blood or claw marks
    bay_glow: tuple[float, float, float] = (0.0, 0.0, 0.0)  # hidden light inside recesses
    stripe: str = ""  # lit guide rail along a corridor; its color names the district


CORRIDORS = "SOWEGAD"  # zones that follow the corridor layout of side strips and a middle
SPINE = Zone(
    ("wall10", "wall10", "wall12", "wall10"), "walltrim2", "ceiling_trim2", "floor_trim1",
    "walltrim3", ("floortrim4", "floor3"), ("ceiling4", "ceiling2"), "ceiling2_lit",
    (0.55, 0.86, 0.92), (0.022, 0.034, 0.042), 5, 0.3, FLICKER, 0.8, 0.1, (0.06, 0.28, 0.33),
    "stripe_spine",
)  # fmt: skip
ZONES = {
    "S": SPINE,
    "D": SPINE,
    "A": SPINE,
    "O": Zone(
        ("wall2", "wall7", "wall11"), "columna1", "ceiling_trim1", "floor_trim1", "walltrim6",
        ("floor1", "floor4"), ("ceiling6", "ceiling2"), "ceiling7", (0.5, 0.68, 1.0),
        (0.016, 0.026, 0.06), 5, 0.2, STEADY, 1.1, 0.0, (0.02, 0.07, 0.22), "stripe_command",
    ),
    "W": Zone(
        ("wall1", "wall7", "wall1"), "walltrim1", "ceiling_trim2", "floor_trim2", "walltrim4",
        ("floor5", "floor2"), ("ceiling2", "ceiling6"), "ceiling7", (1.0, 0.66, 0.32),
        (0.042, 0.026, 0.014), 5, 0.4, FLICKER, 0.9, 0.18, (0.2, 0.09, 0.02), "stripe_crew",
    ),
    "E": Zone(
        ("wall5", "wall7", "wall5"), "trim2", "ceiling_trim5", "floor_trim2", "walltrim3",
        ("floor4", "floor1"), ("ceiling6", "ceiling2"), "ceiling2_lit", (0.78, 1.0, 0.86),
        (0.028, 0.044, 0.034), 5, 0.3, STEADY, 1.05, 0.08, (0.05, 0.16, 0.08), "stripe_medical",
    ),
    "G": Zone(
        ("rack1", "wall12", "wall2"), "columna2", "walltrim5", "floor_trim3", "walltrim5",
        ("floor3", "floor2"), ("ceiling3", "ceiling4"), "ceiling3_lit", (1.0, 0.3, 0.12),
        (0.05, 0.016, 0.01), 5, 0.3, PULSE, 1.3, 0.1, (0.22, 0.05, 0.01), "stripe_engine",
    ),
    "Z": Zone(
        ("wall12", "wall10"), "columna1", "ceiling_trim2", "floor_trim1", "walltrim6",
        ("floor5", "floor1"), ("ceiling2",), "ceiling7", (0.75, 0.9, 1.0),
        (0.03, 0.04, 0.05), 4, 0.0, STEADY, 0.95, 0.0, (0.05, 0.16, 0.24),
    ),
    "B": Zone(
        ("wall2", "wall11", "wall5"), "walltrim1", "ceiling_trim1", "floor_trim1", "walltrim6",
        ("floor4", "floor1"), ("ceiling1", "ceiling6"), "ceiling7", (0.5, 0.72, 1.0),
        (0.018, 0.03, 0.055), 4, 0.25, STEADY, 1.1, 0.0, (0.03, 0.08, 0.2),
    ),
    "L": Zone(
        ("wall5", "wall11", "wall7"), "walltrim1", "ceiling_trim1", "floor_trim2", "walltrim3",
        ("floor1", "floor4"), ("ceiling2", "ceiling6"), "ceiling2_lit", (0.72, 0.92, 1.0),
        (0.025, 0.035, 0.045), 4, 0.35, FLICKER, 0.9, 0.0, (0.05, 0.12, 0.16),
    ),
    "Q": Zone(
        ("wall1", "wall2", "wall7"), "walltrim2", "ceiling_trim2", "floor_trim2", "walltrim4",
        ("floor2", "floor5"), ("ceiling2", "ceiling6"), "ceiling7", (1.0, 0.68, 0.36),
        (0.04, 0.026, 0.016), 4, 0.5, FLICKER, 0.8, 0.22,
    ),
    "M": Zone(
        ("wall2", "wall1", "wall7"), "walltrim2", "ceiling_trim2", "floor_trim3", "walltrim4",
        ("floor1", "floor2"), ("ceiling6", "ceiling2"), "ceiling7", (1.0, 0.76, 0.46),
        (0.04, 0.03, 0.02), 4, 0.5, STEADY, 0.85, 0.12,
    ),
    "H": Zone(
        ("wall5", "wall11", "wall7"), "walltrim1", "ceiling_trim1", "floor_trim2", "walltrim3",
        ("floor4", "floor1"), ("ceiling6", "ceiling2"), "ceiling2_lit", (0.82, 1.0, 0.92),
        (0.028, 0.04, 0.036), 4, 0.4, FLICKER, 0.95, 0.3, (0.04, 0.1, 0.09),
    ),
    "P": Zone(
        ("wall8", "wall7"), "columna1", "ceiling_trim2", "floor_trim1", "walltrim6",
        ("floor2", "floor5"), ("ceiling4",), "ceiling4_lit", (0.4, 1.0, 0.62),
        (0.014, 0.04, 0.026), 5, 0.25, STEADY, 1.0, 0.0, (0.02, 0.14, 0.08),
    ),
    "K": Zone(
        ("wall5", "wall11"), "trim2", "ceiling_trim5", "floortrim4", "walltrim3",
        ("floor1", "floor4"), ("ceiling6",), "ceiling7", (0.55, 0.85, 1.0),
        (0.02, 0.036, 0.055), 5, 0.25, STEADY, 1.0, 0.0, (0.05, 0.14, 0.24),
    ),
    "C": Zone(
        ("wall7", "other2", "wall1"), "columna2", "doorborder2", "floor_trim3", "walltrim5",
        ("floor2", "floor3"), ("ceiling4", "ceiling2"), "ceiling4_lit", (1.0, 0.72, 0.3),
        (0.034, 0.024, 0.012), 6, 0.45, FLICKER, 1.1, 0.12,
    ),
    "R": Zone(
        ("wall12", "rack1", "wall10"), "columna2", "walltrim5", "floor_trim1", "walltrim6",
        ("floor3", "floor2"), ("ceiling3",), "ceiling3_lit", (1.0, 0.16, 0.08),
        (0.07, 0.01, 0.008), 4, 0.1, PULSE, 2.4, 0.0, (0.22, 0.04, 0.02),
    ),
    "U": Zone(
        ("wall8", "wall7"), "columna1", "ceiling_trim2", "floor_trim1", "walltrim6",
        ("floor5", "floor2"), ("ceiling4",), "ceiling2_lit", (0.45, 1.0, 0.7),
        (0.014, 0.036, 0.024), 4, 0.2, STEADY, 1.0, 0.0, (0.03, 0.12, 0.08),
    ),
    "N": Zone(
        ("wall3",), "trim3", "walltrim5", "floor_trim3", "walltrim5", ("floor3",), ("ceiling3",),
        "ceiling3_lit", (0.9, 0.22, 0.12), (0.02, 0.006, 0.004), 4, 0.6, PULSE, 0.6,
    ),
}  # fmt: skip

FURNITURE = {  # solid cells drawn as objects rather than ship walls
    "C": "crate",
    "R": "reactor",
    "Z": "column",
}
SIGNS = {  # (cell x, y, face): a stencil beside the doorway it names
    (17, 17, 1): "sign_crew",
    (21, 17, 0): "sign_mess",
    (21, 23, 0): "sign_medbay",
    (17, 11, 1): "sign_command",
    (21, 11, 0): "sign_command",
    (17, 26, 1): "sign_engine",
    (21, 26, 0): "sign_engine",
    (6, 18, 1): "sign_crewring",
    (33, 18, 0): "sign_medical",
    (22, 18, 0): "sign_plaza",
    (17, 30, 1): "sign_cargo",
    (21, 30, 0): "sign_reactor",
    (17, 9, 1): "sign_bridge",
    (21, 9, 0): "sign_bridge",
    (6, 15, 1): "sign_pods",
    (33, 15, 0): "sign_cryo",
    (17, 33, 1): "sign_shuttle",
    (12, 11, 3): "sign_lab",
    (27, 11, 3): "sign_lab",
}
WARNINGS = {
    (29, 18, 0): "warning_quarantine",
    (9, 24, 0): "warning_listen",
    (29, 22, 1): "warning_seal",
    (17, 23, 1): "warning_listen",
}
CROSSES = ((29, 23, 1), (33, 20, 0))  # green crosses mark the medical side
FACES = ((-1, 0), (1, 0), (0, -1), (0, 1))  # west, east, north, south face of a solid cell
PIPED ={"S": 0.7, "O": 0.6, "W": 0.7, "E": 0.6, "G": 1.0, "C": 0.5, "R": 1.0}  # share of walls
HOT_PIPES = "GR"
CABLED = "SOEGZ"
SPARK_SHARE = 0.6  # dead lights that still short out now and then


def stable(x, y, salt=0):
    """Small deterministic hash in 0..1; Python's hash() changes between runs."""
    value = (x * 73856093 ^ y * 19349663 ^ salt * 83492791) & 0xFFFFFFFF
    value = (value ^ (value >> 13)) * 1274126177 & 0xFFFFFFFF
    return ((value ^ (value >> 16)) & 0xFFFF) / 0xFFFF


def pick(options, x, y, salt):
    return options[int(stable(x, y, salt) * len(options)) % len(options)]


class ShipTheme:
    def __init__(self, world):
        self.world = world
        self.height, self.width = world.height, world.width
        self.solid = np.array([[tile == "#" for tile in row] for row in world.rows], dtype=bool)
        self.blocks = self._free_blocks()
        self.lights = []
        self.light_cells = {}
        self._place_lights()
        self.layers = self._bake_light()
        self.sparks = tuple(
            (x + 0.5, y + 0.5)
            for (x, y), lit in sorted(self.light_cells.items())
            if not lit and stable(x, y, 15) < SPARK_SHARE
        )

    def zone_at(self, x, y):
        return ZONE_ROWS[y][x]

    def is_open(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height and not self.solid[y, x]

    def wall_count(self, x, y):
        return sum(not self.is_open(x + dx, y + dy) for dx, dy in FACES)

    def edge_count(self, x, y):
        """Like wall_count, but a doorway or alcove still counts as the room's edge."""
        return sum(
            not self.is_open(x + dx, y + dy) or self.zone_at(x + dx, y + dy) in "DA"
            for dx, dy in FACES
        )

    def _free_blocks(self):
        """Furniture is a solid island standing inside one room, touching no corridor."""
        blocks = np.zeros_like(self.solid)
        seen = np.zeros_like(self.solid)
        for start_y in range(self.height):
            for start_x in range(self.width):
                if seen[start_y, start_x] or not self.solid[start_y, start_x]:
                    continue
                cells, zones, stack = [], set(), [(start_x, start_y)]
                seen[start_y, start_x] = True
                while stack:
                    x, y = stack.pop()
                    cells.append((x, y))
                    for dx, dy in FACES:
                        nx, ny = x + dx, y + dy
                        if not (0 <= nx < self.width and 0 <= ny < self.height):
                            zones.add("hull")
                        elif not self.solid[ny, nx]:
                            zones.add(self.zone_at(nx, ny))
                        elif not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((nx, ny))
                if len(zones) == 1 and not zones & set(CORRIDORS + "N") and "hull" not in zones:
                    for x, y in cells:
                        blocks[y, x] = True
        return blocks

    def _place_lights(self):
        taken = []
        cells = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.is_open(x, y) and self.zone_at(x, y) not in "DA"
        ]
        cells.sort(key=lambda cell: (stable(*cell, 7), cell))
        cells.sort(key=lambda cell: self.wall_count(*cell))  # prefer the middle of a room
        for x, y in cells:
            zone = ZONES[self.zone_at(x, y)]
            if any(
                max(abs(x - lx), abs(y - ly)) < min(zone.spacing, spacing)
                for lx, ly, spacing in taken
            ):
                continue
            taken.append((x, y, zone.spacing))
            dead = stable(x, y, 11) < zone.dead_share
            self.light_cells[(x, y)] = not dead
            if not dead:
                layer = self._layer_for(zone, x, y)
                lamp = LAMP_HIGH if self.edge_count(x, y) == 0 else LAMP_LOW
                color = muted(zone.color)
                self.lights.append((x + 0.5, y + 0.5, color, zone.power, layer, lamp))
        for _, ex, ey in EXITS:
            # Every exit carries the same lamp, so light never tells which one works.
            self.lights.append((ex, ey, (0.3, 1.0, 0.55), 0.7, STEADY, EXIT_LAMP))

    @staticmethod
    def _layer_for(zone, x, y):
        if zone.mode == FLICKER and stable(x, y, 13) < 0.45:
            return FLICKER
        return PULSE if zone.mode == PULSE else STEADY

    def _bake_light(self):
        """One RGB layer per light behaviour, so flicker only rescales stored light."""
        grid_h, grid_w = self.height * LIGHT_GRID, self.width * LIGHT_GRID
        ys, xs = np.mgrid[0:grid_h, 0:grid_w]
        points_x = (xs + 0.5) / LIGHT_GRID
        points_y = (ys + 0.5) / LIGHT_GRID
        layers = np.zeros((LIGHT_LAYERS, grid_h, grid_w, 3), np.float32)
        aim = np.zeros((grid_h, grid_w, 4), np.float32)  # the lamps that reach a point, by weight
        steps = (np.arange(1, RAY_SAMPLES + 1, dtype=np.float32) / (RAY_SAMPLES + 1))[None, :]
        for lx, ly, color, power, layer, lamp in self.lights:
            distance = np.hypot(points_x - lx, points_y - ly)
            near = distance < LIGHT_RADIUS
            px, py = points_x[near], points_y[near]
            sample_x = lx + (px - lx)[:, None] * steps
            sample_y = ly + (py - ly)[:, None] * steps
            blocked = self.solid[sample_y.astype(np.int32), sample_x.astype(np.int32)].any(axis=1)
            falloff = 1 / (1 + 0.3 * distance[near] ** 2)
            falloff *= np.clip(1 - distance[near] / LIGHT_RADIUS, 0, 1)
            strength = power * 2.1 * falloff * ~blocked
            light = strength[:, None] * np.array(color, np.float32)
            layers[layer, ys[near], xs[near]] += light
            aim[ys[near], xs[near]] += np.stack(
                (strength * (lx - px), strength * (ly - py), strength * lamp, strength), axis=1
            )
        layers[STEADY] += self._ambient_grid() * AMBIENT
        occlusion = self._occlusion_grid()
        for layer in range(LIGHT_LAYERS):
            layers[layer] = soften(layers[layer]) * occlusion[..., None]
        aim = soften(aim)
        weight = np.maximum(aim[..., 3:], 1e-4)
        self.aim = aim[..., :3] / weight
        self.aim[..., 2][aim[..., 3] < 1e-3] = LAMP_LOW  # no lamp here: fill alone, from above
        return layers

    def _ambient_grid(self):
        grid = np.zeros((self.height, self.width, 3), np.float32)
        for y in range(self.height):
            for x in range(self.width):
                if not self.solid[y, x]:
                    grid[y, x] = muted(ZONES[self.zone_at(x, y)].ambient)
        for _ in range(2):  # let each room's tone spill a little past its doorways
            grid = np.maximum(grid, cross_blur(grid) * 0.8)
        return np.repeat(np.repeat(grid, LIGHT_GRID, axis=0), LIGHT_GRID, axis=1)

    def _occlusion_grid(self):
        """Dark contact shadow along every wall foot."""
        fine = np.repeat(np.repeat(~self.solid, LIGHT_GRID, axis=0), LIGHT_GRID, axis=1)
        openness = fine.astype(np.float32)
        for _ in range(3):
            openness = cross_blur(openness)
        return 0.55 + 0.45 * openness**1.5


def muted(color):
    grey = sum(color) / 3
    return tuple(grey + (value - grey) * TONE_KEPT for value in color)


def cross_blur(grid):
    padded = np.pad(grid, ((1, 1), (1, 1)) + ((0, 0),) * (grid.ndim - 2), mode="edge")
    return (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + padded[1:-1, 1:-1]
    ) / 5


def soften(layer):
    for _ in range(2):
        layer = cross_blur(layer)
    return layer
