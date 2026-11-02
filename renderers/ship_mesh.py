"""Build the ship's real 3D surfaces from the 2D deck, in the pack's corridor style.

Corridor walls follow the pack's preview: proud pillars between recessed bays, a tilted
panel in each bay, a sloped lintel above it and a sill below. Ceilings step up over the
middle of a room, floor grates sink, and doorways drop a header. Surfaces may sit behind a
wall line but never reach past where collision stops the player, so Core stays the truth.
"""

from math import cos, radians, sin

import numpy as np

from game.ship_deck import EXITS
from renderers.ship_kernels import SKY
from renderers.ship_theme import (
    CABLED,
    CORRIDORS,
    FACES,
    FURNITURE,
    HOT_PIPES,
    PIPED,
    SIGNS,
    ZONES,
    pick,
    stable,
)

WALL_TOP = 2.05
RAISED = 0.42  # ceiling step over the middle of a room
DOOR_TOP = 1.72
CRAWL_TOP = 1.55
SUNK = 0.12  # grate depth below the deck
PILLAR = 0.42
PILLAR_PROUD = 0.07
BAY_SPAN = 2.0
SILL_TOP = 0.3
PANEL_TOP = 1.5
LINTEL_FOOT = 1.7
PANEL_LOW = 0.16  # recess depth at the foot of a bay panel
PANEL_HIGH = 0.42  # recess depth at its head
WINDOW_LOW, WINDOW_HIGH, WINDOW_DEPTH = 0.42, 1.65, 0.3
WINDOW_SEAL = 0.035  # the dark gasket a pane sits in
DOOR_TRIM, DOOR_COLUMN = 0.26, 0.5  # frame widths beside a doorway
DOOR_FRAME_PROUD, DOOR_COLUMN_PROUD, DOOR_HEADER_PROUD = 0.1, 0.18, 0.14  # past PILLAR_PROUD
FRAME_INSET, FRAME_BACK = 0.01, 0.05  # the frame's inner face must not share a pillar's plane
CAP_RISE = 0.16
EDGE_TEXTURE, EDGE_SHAPE = "trim5", 8.0  # a trim this many times taller than it is wide
RAIL_INNER, RAIL_DEPTH, RAIL_HEAD = 0.4, 0.2, 1.56  # the frame the leaves run in
LEAF_SUNK, LEAF_RAISED = 0.03, 0.08
LEAF_PROFILE = (  # height, distance out from the door's middle plane
    (0.0, LEAF_RAISED),
    (0.34, LEAF_RAISED),
    (0.38, LEAF_SUNK),
    (0.7, LEAF_SUNK),
    (0.74, LEAF_RAISED),
    (0.96, LEAF_RAISED),
    (1.0, LEAF_SUNK),
    (1.5, LEAF_SUNK),
    (1.54, LEAF_RAISED),
    (DOOR_TOP, LEAF_RAISED),
)
SIGN_LEFT, SIGN_RIGHT, SIGN_LOW, SIGN_HIGH = -0.25, 1.25, 1.74, 2.0
SIGN_PROUD = 0.19  # clear of the door frames beside it, inside the player's collision radius
LINE_OFFSET = 0.3  # painted floor line, measured out from the wall
LINE_WIDTH = 0.045
SIDEWAYS = ("columna1", "columna2")  # these images lie down, so a pillar stands them up
# Overhead fittings hang above the Alien's head and inside the player's collision radius,
# so they can never hide it or look like something to walk around.
OVERHEAD_FLOOR = 1.87
PIPE_OUT, PIPE_HEIGHT, PIPE_RADIUS = 0.08, 1.96, 0.06
LINE_OUT, LINE_HEIGHT, LINE_RADIUS = 0.05, 1.895, 0.024
COLLAR_SCALE, COLLAR_WIDTH, COLLAR_SPAN = 1.3, 0.08, 2  # width in units, one every SPAN cells
PIPE_ANGLES = (45, -5, -55, -110)  # the sides a standing player can see from below
PIPE_SHADES = (0.95, 0.8, 0.6)
CABLE_THICKNESS = 0.03
CABLE_SHARE = 0.2


def surface_frame(points, corners, normal):
    """World directions of a texture's right, up and out, so its relief can be read in place."""
    a, b, _, d = points
    edge_u, edge_v = b - a, d - a
    (u0, v0), (u1, v1), _, (u3, v3) = corners
    du1, dv1, du2, dv2 = u1 - u0, v1 - v0, u3 - u0, v3 - v0
    area = du1 * dv2 - du2 * dv1
    if abs(area) < 1e-7:
        right, down = edge_u, edge_v
    else:
        right = (dv2 * edge_u - dv1 * edge_v) / area
        down = (du1 * edge_v - du2 * edge_u) / area
    right = right - normal * np.dot(right, normal)
    down = down - normal * np.dot(down, normal)
    right = right / max(np.linalg.norm(right), 1e-7)
    down = down / max(np.linalg.norm(down), 1e-7)
    return (*right, *-down, *normal)


class Mesh:
    def __init__(self, index):
        self.index = index
        self.vertices, self.uvs, self.normals, self.textures = [], [], [], []
        self.tints, self.shifts, self.shades, self.frames = [], [], [], []
        self.door_parts = []  # (quad, door index, sliding direction) of every door leaf

    def quad(
        self, a, b, c, d, texture, facing, *, repeat=(1.0, 1.0), turned=False, tint=(0, 0, 0),
        reach=0.12, shade=1.0, span=None, rows=None,
    ):  # fmt: skip
        """Corners run bottom-left, bottom-right, top-right, top-left as seen from the front."""
        points = np.array((a, b, c, d), np.float32)
        normal = np.cross(points[1] - points[0], points[3] - points[0])
        length = np.linalg.norm(normal)
        if length < 1e-7:
            return
        normal = normal / length
        if np.dot(normal, facing) < 0:
            normal = -normal
        ru, rv = repeat
        left, right = span or (0, ru)
        upper, lower = rows or (0, rv)
        corners = ((left, lower), (right, lower), (right, upper), (left, upper))
        if turned:
            corners = ((rv, 0), (rv, ru), (0, ru), (0, 0))
        self.vertices.append(points)
        self.uvs.append(corners)
        self.normals.append(normal)
        self.frames.append(surface_frame(points, corners, normal))
        self.textures.append(self.index[texture])
        self.tints.append(tint)
        self.shifts.append((facing[0] * reach, facing[1] * reach))
        self.shades.append(shade)


def cell_ceiling(theme, x, y):
    key = theme.zone_at(x, y)
    if key == "D":
        return DOOR_TOP
    if key == "N":
        return CRAWL_TOP
    if key != "A" and theme.edge_count(x, y) == 0:
        return WALL_TOP + RAISED
    return WALL_TOP


def cell_floor(theme, x, y):
    sunk = floor_name(theme, x, y)[0] == "floor3" and theme.edge_count(x, y) == 0
    return -SUNK if sunk else 0.0


def floor_name(theme, x, y):
    """Texture and direction of a deck tile; corridors run a strip down each side."""
    key = theme.zone_at(x, y)
    zone = ZONES[key]
    if key in CORRIDORS:
        side, middle = zone.floors
        if theme.edge_count(x, y) == 0:
            return middle, False
        if side != "floortrim4":
            return side, False
        west_or_east = not theme.is_open(x - 1, y) or not theme.is_open(x + 1, y)
        return "floortrim4", not west_or_east
    name = pick(zone.floors, x, y, 1)
    if zone.blood and stable(x, y, 2) < zone.blood * 0.5 and f"{name}_blood" in BLOODY_FLOORS:
        return f"{name}_blood", False
    return name, False


BLOODY_FLOORS = ("floor1_blood", "floor2_blood", "floor5_blood")


def ceiling_name(theme, x, y):
    zone = ZONES[theme.zone_at(x, y)]
    if theme.light_cells.get((x, y)):
        return zone.lamp
    if theme.edge_count(x, y) == 0 and "ceiling2" in zone.ceilings:
        return "ceiling2"
    return pick(zone.ceilings, x, y, 3)


def add_cells(mesh, theme):
    for y in range(theme.height):
        for x in range(theme.width):
            if not theme.is_open(x, y):
                continue
            low, high = cell_floor(theme, x, y), cell_ceiling(theme, x, y)
            name, turned = floor_name(theme, x, y)
            mesh.quad(
                (x, y + 1, low), (x + 1, y + 1, low), (x + 1, y, low), (x, y, low), name,
                (0, 0, 1), repeat=(1.0, 0.5 if name == "floortrim4" else 1.0), turned=turned,
                reach=0.0,
            )  # fmt: skip
            lit = bool(theme.light_cells.get((x, y)))
            mesh.quad(
                (x, y, high), (x + 1, y, high), (x + 1, y + 1, high), (x, y + 1, high),
                ceiling_name(theme, x, y), (0, 0, -1), reach=0.0, shade=1.0 if lit else 0.8,
            )  # fmt: skip
            add_steps(mesh, theme, x, y, low, high)


def add_steps(mesh, theme, x, y, low, high):
    """Risers where a neighbouring ceiling or floor sits at another height."""
    for dx, dy in FACES:
        nx, ny = x + dx, y + dy
        if not theme.is_open(nx, ny):
            continue
        left, right = edge_points(x, y, dx, dy)
        facing = (-dx, -dy, 0)
        other_high = cell_ceiling(theme, nx, ny)
        if other_high < high:
            door = theme.zone_at(nx, ny) == "D"
            mesh.quad(
                (*left, other_high), (*right, other_high), (*right, high), (*left, high),
                "doorborder1" if door else "ceiling_trim4", facing, turned=not door,
                shade=0.7, reach=0.3,
            )  # fmt: skip
        other_low = cell_floor(theme, nx, ny)
        if other_low > low:
            mesh.quad(
                (*left, low), (*right, low), (*right, other_low), (*left, other_low),
                "floor_trim1", facing, shade=0.45, reach=0.3,
            )  # fmt: skip


def edge_points(x, y, dx, dy):
    """Ends of the cell edge toward (dx, dy), left then right as seen from inside the cell."""
    if dx > 0:
        return (x + 1, y), (x + 1, y + 1)
    if dx < 0:
        return (x, y + 1), (x, y)
    if dy > 0:
        return (x + 1, y + 1), (x, y + 1)
    return (x, y), (x + 1, y)


def wall_edges(theme):
    """Every open-to-solid cell edge, grouped into straight runs of one kind of wall."""
    groups = {}
    for y in range(theme.height):
        for x in range(theme.width):
            if not theme.is_open(x, y):
                continue
            for dx, dy in FACES:
                nx, ny = x + dx, y + dy
                if theme.is_open(nx, ny):
                    continue
                line = (x + (dx > 0)) if dx else (y + (dy > 0))
                along = y if dx else x
                inside = 0 <= nx < theme.width and 0 <= ny < theme.height
                solid_cell = bool(inside and theme.blocks[ny, nx])
                key = (dx, dy, line, theme.zone_at(x, y), solid_cell)
                groups.setdefault(key, []).append(along)
    runs = []
    for key, spots in groups.items():
        spots.sort()
        start = previous = spots[0]
        for spot in spots[1:] + [None]:
            if spot is None or spot != previous + 1:
                runs.append((*key, start, previous + 1))
                start = spot
            previous = spot
    return runs


class Wall:
    """A straight wall seen from the open side; s runs left to right across the view."""

    def __init__(self, mesh, dx, dy, line, first, last, top, zone, seed):
        self.mesh, self.top, self.zone, self.seed = mesh, top, zone, seed
        self.normal = np.array((-dx, -dy), np.float32)
        if dx > 0:
            self.origin, self.right = np.array((line, first)), np.array((0, 1))
        elif dx < 0:
            self.origin, self.right = np.array((line, last)), np.array((0, -1))
        elif dy > 0:
            self.origin, self.right = np.array((last, line)), np.array((-1, 0))
        else:
            self.origin, self.right = np.array((first, line)), np.array((1, 0))
        self.length = float(last - first)

    def at(self, s, depth, z):
        """A point s along the wall, depth behind its line (negative stands in front)."""
        x, y = self.origin + self.right * s - self.normal * depth
        return float(x), float(y), z

    @property
    def facing(self):
        return float(self.normal[0]), float(self.normal[1]), 0.0

    def side(self, sign):
        return float(self.right[0] * sign), float(self.right[1] * sign), 0.0

    def flat(
        self, s0, s1, z0, z1, texture, *, depth=0.0, repeat=(1.0, 1.0), shade=1.0, reach=0.14,
        turned=False,
    ):  # fmt: skip
        self.mesh.quad(
            self.at(s0, depth, z0), self.at(s1, depth, z0), self.at(s1, depth, z1),
            self.at(s0, depth, z1), texture, self.facing, repeat=repeat, turned=turned,
            shade=shade, reach=reach,
        )  # fmt: skip

    def pillar(self, s0, s1):
        """Stands proud of the wall line, so its sides show from along the corridor."""
        proud = -PILLAR_PROUD
        turned = self.zone.pillar in SIDEWAYS
        self.flat(
            s0, s1, 0, self.top, self.zone.pillar, depth=proud, shade=0.9, reach=0.2, turned=turned,
        )  # fmt: skip
        for s, sign in ((s0, -1), (s1, 1)):
            self.mesh.quad(
                self.at(s, 0, 0), self.at(s, proud, 0), self.at(s, proud, self.top),
                self.at(s, 0, self.top), self.zone.pillar, self.side(sign),
                repeat=(0.2, 1.0), turned=turned, shade=0.6, reach=0.2,
            )  # fmt: skip

    def bay(self, s0, s1, texture):
        """A recess: sloped sill, tilted panel, sloped lintel, then the wall above."""
        zone = self.zone
        glow = zone.bay_glow
        low, high = PANEL_LOW, PANEL_HIGH
        reach = 0.6
        self.mesh.quad(
            self.at(s0, 0, 0), self.at(s1, 0, 0), self.at(s1, low, SILL_TOP),
            self.at(s0, low, SILL_TOP), zone.sill, self.facing, shade=0.8, reach=0.3,
        )  # fmt: skip
        self.mesh.quad(
            self.at(s0, low, SILL_TOP), self.at(s1, low, SILL_TOP), self.at(s1, high, PANEL_TOP),
            self.at(s0, high, PANEL_TOP), texture, self.facing, tint=glow, reach=reach,
        )  # fmt: skip
        self.mesh.quad(
            self.at(s0, high, PANEL_TOP), self.at(s1, high, PANEL_TOP),
            self.at(s1, 0, LINTEL_FOOT), self.at(s0, 0, LINTEL_FOOT), zone.lintel, self.facing,
            tint=tuple(c * 0.5 for c in glow), shade=0.55, reach=reach,
        )  # fmt: skip
        self.flat(s0, s1, LINTEL_FOOT, self.top, zone.cornice, shade=0.85)
        for s, sign in ((s0, 1), (s1, -1)):
            self.mesh.quad(
                self.at(s, 0, 0), self.at(s, low, SILL_TOP), self.at(s, high, PANEL_TOP),
                self.at(s, 0, LINTEL_FOOT), "trim1", self.side(sign), repeat=(0.3, 1.0),
                tint=tuple(c * 0.6 for c in glow), shade=0.62, reach=reach,
            )  # fmt: skip

    def window(self, s0, s1):
        """A framed opening. The sky sits at the back and its four reveals close every gap."""
        low, high, deep = WINDOW_LOW, WINDOW_HIGH, WINDOW_DEPTH
        self.flat(s0, s1, 0, low, "wall9", shade=0.8)
        self.flat(s0, s1, high, self.top, "walltrim6", shade=0.8)
        self.flat(s0, s1, low, high, "window_glass", depth=deep, shade=SKY)
        seal, front = WINDOW_SEAL, deep - 0.012
        for left, right, foot, head in (
            (s0, s1, low, low + seal), (s0, s1, high - seal, high),
            (s0, s0 + seal, low, high), (s1 - seal, s1, low, high),
        ):  # fmt: skip
            self.flat(left, right, foot, head, "pane_seal", depth=front, reach=0.3)
        frame = "walltrim3"
        self.mesh.quad(
            self.at(s0, 0, low), self.at(s1, 0, low), self.at(s1, deep, low),
            self.at(s0, deep, low), frame, (0, 0, 1), shade=0.7, reach=0.3,
        )  # fmt: skip
        self.mesh.quad(
            self.at(s0, deep, high), self.at(s1, deep, high), self.at(s1, 0, high),
            self.at(s0, 0, high), frame, (0, 0, -1), shade=0.5, reach=0.3,
        )  # fmt: skip
        for s, sign in ((s0, 1), (s1, -1)):
            self.mesh.quad(
                self.at(s, 0, low), self.at(s, deep, low), self.at(s, deep, high),
                self.at(s, 0, high), frame, self.side(sign), shade=0.6, reach=0.3,
            )  # fmt: skip

    def floor_line(self):
        """A lit line beside the wall, the same blue everywhere, to follow in the dark."""
        if self.zone.guide_line:
            near, far = -LINE_OFFSET, -LINE_OFFSET - LINE_WIDTH
            self.mesh.quad(
                self.at(0, near, 0.012), self.at(self.length, near, 0.012),
                self.at(self.length, far, 0.012), self.at(0, far, 0.012), "guide_line",
                (0, 0, 1), repeat=(self.length, 1.0), reach=0.0,
            )  # fmt: skip

    def modules(self, choose):
        """Share the run out between pillars and bays, ending on a pillar at both ends."""
        count = max(int((self.length - PILLAR) / BAY_SPAN + 0.5), 1)
        span = (self.length - (count + 1) * PILLAR) / count
        if span < 0.6:
            self.panelled(choose(0))
            return
        s = 0.0
        for index in range(count):
            self.pillar(s, s + PILLAR)
            self.bay(s + PILLAR, s + PILLAR + span, choose(index))
            s += PILLAR + span
        self.pillar(s, s + PILLAR)

    def panelled(self, texture):
        """A short wall has no room for a bay: a kick plate, a panel and a cornice."""
        self.flat(0, self.length, 0, 0.26, self.zone.sill, shade=0.8)
        self.flat(0, self.length, 0.26, self.top - 0.3, texture, repeat=(self.length / 1.6, 1.0))
        self.flat(0, self.length, self.top - 0.3, self.top, self.zone.cornice, shade=0.85)

    def sign_edges(self):
        """The underside and ends of a sign box, so it reads as fixed to the wall."""
        low, high, proud = SIGN_LOW, SIGN_HIGH, -SIGN_PROUD
        self.mesh.quad(
            self.at(SIGN_LEFT, proud, low), self.at(SIGN_RIGHT, proud, low),
            self.at(SIGN_RIGHT, 0, low), self.at(SIGN_LEFT, 0, low), "walltrim5", (0, 0, -1),
            span=(0, 0.5), shade=0.35, reach=0.3,
        )  # fmt: skip
        for s, sign in ((SIGN_LEFT, -1), (SIGN_RIGHT, 1)):
            self.mesh.quad(
                self.at(s, 0, low), self.at(s, proud, low), self.at(s, proud, high),
                self.at(s, 0, high), "walltrim5", self.side(sign), span=(0, 0.2), shade=0.45,
                reach=0.3,
            )  # fmt: skip

    def pipe(self, s0, s1, out, z, radius, texture):
        """A run along the wall, faceted around the sides seen from below."""
        points = [(radians(a), cos(radians(a)), sin(radians(a))) for a in PIPE_ANGLES]
        facets = len(points) - 1
        for index, (near, far) in enumerate(zip(points, points[1:])):
            high_out, high_z = out + radius * near[1], z + radius * near[2]
            low_out, low_z = out + radius * far[1], z + radius * far[2]
            middle = (near[0] + far[0]) / 2
            nx, ny, _ = self.facing
            self.mesh.quad(
                self.at(s0, -low_out, low_z), self.at(s1, -low_out, low_z),
                self.at(s1, -high_out, high_z), self.at(s0, -high_out, high_z), texture,
                (nx * cos(middle), ny * cos(middle), sin(middle)), repeat=(s1 - s0, 1.0),
                rows=(index / facets, (index + 1) / facets), shade=PIPE_SHADES[index], reach=0.3,
            )  # fmt: skip


def worn(zone, name, x, y):
    """Blood and claw marks stay rare, and only appear on panels that have such a copy."""
    if stable(x, y, 8) >= zone.blood:
        return name
    variant = "clawed" if stable(x, y, 9) < 0.5 else "blood"
    candidate = f"{name}_{variant}"
    return candidate if candidate in WORN_NAMES else name


WORN_NAMES = tuple(
    f"{name}_{kind}"
    for name in ("wall10", "wall6", "wall1", "wall4")
    for kind in ("blood", "clawed")
)


def add_walls(mesh, theme):
    for dx, dy, line, key, is_block, first, last in wall_edges(theme):
        zone = ZONES[key]
        x = line if dx else first
        y = first if dx else line
        if key == "D":
            top = DOOR_TOP
        elif key == "N":
            top = CRAWL_TOP
        else:
            top = WALL_TOP
        wall = Wall(mesh, dx, dy, line, first, last, top, zone, stable(x, y, 4))
        if key == "D":
            wall.flat(0, wall.length, 0, top, "walltrim1", repeat=(wall.length * 2, 1.0), shade=0.7)
        elif is_block:
            add_block(wall, FURNITURE[key])
            continue
        elif key == "B" and dy < 0 and line <= 4:
            add_viewport(wall)
        elif key in "NA":
            wall.panelled(pick(zone.bays, x, y, 5))
        else:
            wall.modules(
                lambda i, x=x, y=y, z=zone: worn(z, pick(z.bays, x + i, y + i, 5), x + i, y + i)
            )
            wall.floor_line()
            add_pipes(wall, key, x, y)


def add_pipes(wall, key, x, y):
    if wall.length < 3 or stable(x, y, 14) >= PIPED.get(key, 0.0):
        return
    texture = "pipe_hot" if key in HOT_PIPES else "pipe"
    wall.pipe(0, wall.length, PIPE_OUT, PIPE_HEIGHT, PIPE_RADIUS, texture)
    wall.pipe(0, wall.length, LINE_OUT, LINE_HEIGHT, LINE_RADIUS, "cable")
    for s in np.arange(1.0, wall.length - 0.5, COLLAR_SPAN):
        wall.pipe(
            float(s), float(s) + COLLAR_WIDTH, PIPE_OUT, PIPE_HEIGHT, PIPE_RADIUS * COLLAR_SCALE,
            "bracket",
        )  # fmt: skip


def raised_axis(theme, x, y):
    """The way a raised ceiling runs through a cell, or None where it does not run on."""
    def raised(cx, cy):
        return theme.is_open(cx, cy) and theme.edge_count(cx, cy) == 0
    if raised(x, y - 1) and raised(x, y + 1):
        return (0.0, 1.0)
    if raised(x - 1, y) and raised(x + 1, y):
        return (1.0, 0.0)
    return None


def add_cables(mesh, theme):
    """Slack cables strung under the raised ceiling, all well above anyone's head."""
    for y in range(theme.height):
        for x in range(theme.width):
            if (
                not theme.is_open(x, y)
                or theme.zone_at(x, y) not in CABLED
                or theme.edge_count(x, y) != 0
                or stable(x, y, 17) >= CABLE_SHARE
            ):
                continue
            axis = raised_axis(theme, x, y)
            if axis is not None:
                add_cable(mesh, x, y, axis, cell_ceiling(theme, x, y))


def add_cable(mesh, x, y, axis, ceiling):
    side = (axis[1], axis[0])
    offset = (stable(x, y, 18) - 0.5) * 0.6
    middle = np.array((x + 0.5 + side[0] * offset, y + 0.5 + side[1] * offset))
    sag = 2.02 + stable(x, y, 19) * 0.16
    heights = (ceiling, sag + 0.1, sag, ceiling)
    spots = (-1.4, -0.45, 0.4, 1.4)
    for (s0, z0), (s1, z1) in zip(zip(spots, heights), zip(spots[1:], heights[1:])):
        a, b = middle + np.array(axis) * s0, middle + np.array(axis) * s1
        low = (float(a[0]), float(a[1]), z0), (float(b[0]), float(b[1]), z1)
        high = (*low[0][:2], z0 + CABLE_THICKNESS), (*low[1][:2], z1 + CABLE_THICKNESS)
        for sign in (-1.0, 1.0):
            mesh.quad(
                low[0], low[1], high[1], high[0], "cable", (side[0] * sign, side[1] * sign, 0.0),
                repeat=(s1 - s0, 1.0), reach=0.0,
            )  # fmt: skip


def add_block(wall, kind):
    length = wall.length
    if kind == "crate":
        wall.flat(0, length, 0, 1.0, "other1", repeat=(length, 1.0))
        wall.flat(0, length, 1.0, 1.1, "walltrim5", repeat=(length, 1.0), shade=0.7)
        wall.flat(0, length, 1.1, wall.top, "other2", repeat=(length, 1.0))
    elif kind == "reactor":
        wall.flat(0, length, 0, 0.5, "wall12", repeat=(length, 0.4), shade=0.8)
        wall.flat(0, length, 0.5, 1.5, "rack1", repeat=(length, 1.0))
        wall.flat(0, length, 1.5, wall.top, "columna2", repeat=(length / 2, 1.0), shade=0.8)
    else:
        # The ship's centre: glowing conduits ringed by plate, seen from every corridor.
        wall.flat(0, length, 0, 0.5, "columna1", repeat=(length, 1.0), shade=0.9)
        wall.flat(0, length, 0.5, 1.7, "wall12", repeat=(length, 1.0))
        wall.flat(0, length, 1.7, wall.top + RAISED, "columna2", repeat=(length, 1.0), shade=0.9)


def add_viewport(wall):
    """Windows on the bow: pillars frame deep openings onto open space."""
    count = max(int(wall.length / 2), 1)
    span = (wall.length - (count + 1) * PILLAR) / count
    if span < 0.6:
        wall.panelled("walltrim6")
        return
    s = 0.0
    for _ in range(count):
        wall.pillar(s, s + PILLAR)
        wall.window(s + PILLAR, s + PILLAR + span)
        s += PILLAR + span
    wall.pillar(s, s + PILLAR)


def add_hull_doors(mesh, theme):
    """All three exits look alike; only trying the door tells which one works."""
    for _, ex, ey in EXITS:
        x, y = int(ex), int(ey)
        for face, (dx, dy) in enumerate(FACES):
            if theme.is_open(x + dx, y + dy):
                continue
            line = (x + (dx > 0)) if dx else (y + (dy > 0))
            along = y if dx else x
            wall = Wall(mesh, dx, dy, line, along - 1, along + 2, WALL_TOP, ZONES["S"], face)
            wall.flat(0.45, 2.55, 0, 1.75, "door1", depth=-0.09, shade=0.95)
            wall.flat(0.45, 2.55, 1.75, WALL_TOP, "doorborder1", depth=-0.09, shade=0.8)


def add_signs(mesh, theme):
    for (x, y, face), name in SIGNS.items():
        dx, dy = FACES[face]
        line = (x + (dx > 0)) if dx else (y + (dy > 0))
        along = y if dx else x
        # The wall's outward normal is (dx, dy), so it is built from the open cell's side.
        wall = Wall(mesh, -dx, -dy, line, along, along + 1, WALL_TOP, ZONES["S"], 0)
        wall.flat(SIGN_LEFT, SIGN_RIGHT, SIGN_LOW, SIGN_HIGH, name, depth=-SIGN_PROUD, reach=0.3)
        wall.sign_edges()


def add_doors(mesh, theme):
    """Sliding leaves set deep in a real frame, built to the look of the pack's door bay."""
    for index, (x, y) in enumerate(theme.world.doors):
        along_x = theme.is_open(x - 1, y) and theme.is_open(x + 1, y)
        ahead = (1.0, 0.0) if along_x else (0.0, 1.0)
        for sign in (1.0, -1.0):
            side = DoorSide(mesh, x, y, (ahead[0] * sign, ahead[1] * sign))
            for leaf in (0, 1):
                side.leaf(index, leaf)
            side.rail()
            side.frame(theme)


class DoorSide:
    """One face of a doorway; s runs left to right as seen from this side, d toward the viewer."""

    def __init__(self, mesh, x, y, normal):
        self.mesh, self.x, self.y = mesh, x, y
        self.normal = normal
        self.right = (normal[1], -normal[0])
        self.facing = (*normal, 0.0)

    def at(self, s, d, z):
        return (
            self.x + 0.5 + self.right[0] * s + self.normal[0] * d,
            self.y + 0.5 + self.right[1] * s + self.normal[1] * d,
            z,
        )

    def box(self, s0, s1, d, z0, z1, texture, *, shade=1.0, turned=False, span=None):
        self.mesh.quad(
            self.at(s0, d, z0), self.at(s1, d, z0), self.at(s1, d, z1), self.at(s0, d, z1),
            texture, self.facing, turned=turned, shade=shade, reach=0.3, span=span,
        )  # fmt: skip

    def end(self, s, d0, d1, z0, z1, sign, shade=0.5):
        """A face running back along the passage, cut to size from a tall trim; sign is its look."""
        facing = (self.right[0] * sign, self.right[1] * sign, 0.0)
        width = abs(d1 - d0) / (z1 - z0) * EDGE_SHAPE
        self.mesh.quad(
            self.at(s, d0, z0), self.at(s, d1, z0), self.at(s, d1, z1), self.at(s, d0, z1),
            EDGE_TEXTURE, facing, span=(0, width), shade=shade, reach=0.3,
        )  # fmt: skip

    def leaf(self, index, leaf):
        """Raised plates and sunken panels, sliding apart; the texture runs across them all."""
        s0, s1 = -0.5 + leaf * 0.5, leaf * 0.5
        sign = -1.0 if leaf == 0 else 1.0
        slide = (self.right[0] * sign * 0.5, self.right[1] * sign * 0.5, 0.0)
        u0, u1 = leaf * 0.5, (leaf + 1) * 0.5
        for (z0, d0), (z1, d1) in zip(LEAF_PROFILE, LEAF_PROFILE[1:]):
            if d1 < d0:
                shade = 0.95  # a slope that faces up catches the light
            elif d1 > d0:
                shade = 0.42  # one that faces down stays in shadow
            else:
                shade = 0.9 if d0 > LEAF_SUNK else 0.68
            self.mesh.quad(
                self.at(s0, d0, z0), self.at(s1, d0, z0), self.at(s1, d1, z1), self.at(s0, d1, z1),
                "door1_seam", self.facing, span=(u0, u1),
                rows=(1 - z1 / DOOR_TOP, 1 - z0 / DOOR_TOP), shade=shade, reach=0.3,
            )  # fmt: skip
            self.mesh.door_parts.append((len(self.mesh.vertices) - 1, index, slide))

    def rail(self):
        """Heavy posts and a head beam the leaves slide behind, so they sit in a real frame."""
        for inner, outer, sign in ((-RAIL_INNER, -0.5, 1), (RAIL_INNER, 0.5, -1)):
            left, right = sorted((inner, outer))
            self.box(left, right, RAIL_DEPTH, 0, DOOR_TOP, "doorborder4", shade=0.8)
            self.end(inner, RAIL_DEPTH, 0.0, 0, RAIL_HEAD, sign, shade=0.42)
        self.box(
            -RAIL_INNER, RAIL_INNER, RAIL_DEPTH, RAIL_HEAD, DOOR_TOP, "doorborder3", shade=0.8,
            span=(0, 0.3),
        )  # fmt: skip
        self.mesh.quad(
            self.at(-RAIL_INNER, 0.0, RAIL_HEAD), self.at(RAIL_INNER, 0.0, RAIL_HEAD),
            self.at(RAIL_INNER, RAIL_DEPTH, RAIL_HEAD), self.at(-RAIL_INNER, RAIL_DEPTH, RAIL_HEAD),
            "walltrim5", (0.0, 0.0, -1.0), span=(0, 0.5), shade=0.35, reach=0.3,
        )  # fmt: skip

    def frame(self, theme):
        """Stepped surround on the room side: trim, then a column standing proud of it."""
        line = 0.5
        near_s, trim_s = line - FRAME_INSET, line + DOOR_TRIM
        column_s = trim_s + DOOR_COLUMN
        trim, proud = line + DOOR_FRAME_PROUD, line + DOOR_COLUMN_PROUD
        for sign in (-1.0, 1.0):
            far = self.at(sign * 1.6, 0, 0)
            roomy = not theme.is_open(int(far[0]), int(far[1]))
            self._strip(near_s, trim_s, sign, trim, "trim5", False)
            self.end(sign * near_s, trim, line - FRAME_BACK, 0, DOOR_TOP, -sign)
            if roomy:
                self._strip(trim_s, column_s, sign, proud, "columna1", True)
                self.end(sign * trim_s, proud, trim, 0, DOOR_TOP, -sign)
                # It runs back past the wall line to close the bay recess behind the column.
                self.end(sign * column_s, proud, line - PANEL_HIGH, 0, DOOR_TOP, sign)
                self._cap(sign, trim_s, column_s, proud, line)
        self._header(line, trim_s)

    def _cap(self, sign, near, far, proud, line):
        """The column's sloped head, with a wedge closing each side of the slope."""
        left, right = sorted((sign * near, sign * far))
        top = DOOR_TOP + CAP_RISE
        self.mesh.quad(
            self.at(left, proud, DOOR_TOP), self.at(right, proud, DOOR_TOP),
            self.at(right, line, top), self.at(left, line, top),
            "walltrim4", (*self.normal, 0.6), span=(0, 0.3), shade=0.7, reach=0.3,
        )  # fmt: skip
        for s, look in ((left, -1.0), (right, 1.0)):
            self.mesh.quad(
                self.at(s, proud, DOOR_TOP), self.at(s, line, DOOR_TOP), self.at(s, line, top),
                self.at(s, line, top), EDGE_TEXTURE, (self.right[0] * look, self.right[1] * look, 0.0),
                shade=0.45, reach=0.3,
            )  # fmt: skip

    def _strip(self, near, far, sign, depth, texture, turned):
        left, right = sorted((sign * near, sign * far))
        self.box(left, right, depth, 0, DOOR_TOP, texture, shade=0.9, turned=turned)

    def _header(self, line, span):
        """The beam over the doorway and its trim stands out, with a shadowed underside."""
        depth = line + DOOR_HEADER_PROUD
        top = DOOR_TOP + 0.3
        self.box(-span, span, depth, DOOR_TOP, top, "doorborder1", shade=0.9)
        for sign in (-1.0, 1.0):
            self.end(sign * span, depth, line, DOOR_TOP, top, sign, shade=0.45)
        self.mesh.quad(
            self.at(-span, line, DOOR_TOP), self.at(span, line, DOOR_TOP),
            self.at(span, depth, DOOR_TOP), self.at(-span, depth, DOOR_TOP), "walltrim5",
            (0.0, 0.0, -1.0), shade=0.3, reach=0.3,
        )  # fmt: skip


def add_hazard_lines(mesh, theme):
    """Striped thresholds warn that the ship's dangerous end starts here."""
    for y in range(theme.height):
        for x in range(theme.width):
            if not theme.is_open(x, y) or theme.zone_at(x, y) != "D":
                continue
            if not any(
                theme.is_open(x + dx, y + dy) and theme.zone_at(x + dx, y + dy) in "CRG"
                for dx, dy in FACES
            ):
                continue
            mesh.quad(
                (x, y + 1, 0.01), (x + 1, y + 1, 0.01), (x + 1, y, 0.01), (x, y, 0.01), "hazard",
                (0, 0, 1), reach=0.0,
            )  # fmt: skip


def add_wreck(mesh):
    """One ceiling panel has dropped on its hinge in the crew ring, above the Alien's head."""
    hole = WALL_TOP - 0.01
    mesh.quad(
        (7.1, 17.3, hole), (8.9, 17.3, hole), (8.9, 16.1, hole), (7.1, 16.1, hole), "cable",
        (0, 0, -1), reach=0.0, shade=0.5,
    )  # fmt: skip
    low, high = OVERHEAD_FLOOR + 0.02, OVERHEAD_FLOOR + 0.08
    corners = (7.1, 16.2, low), (8.9, 16.4, high), (8.9, 17.3, WALL_TOP), (7.1, 17.2, WALL_TOP)
    mesh.quad(*corners, "ceiling2", (0.1, -0.2, 0.97), shade=0.75, reach=0.0)
    mesh.quad(*corners, "ceiling4", (-0.1, 0.2, -0.97), shade=0.9, reach=0.0)


def build_mesh(theme, names):
    mesh = Mesh({str(name): i for i, name in enumerate(names)})
    add_cells(mesh, theme)
    add_walls(mesh, theme)
    add_hull_doors(mesh, theme)
    add_signs(mesh, theme)
    add_doors(mesh, theme)
    add_hazard_lines(mesh, theme)
    add_wreck(mesh)
    add_cables(mesh, theme)
    return {
        "vertices": np.array(mesh.vertices, np.float32),
        "uvs": np.array(mesh.uvs, np.float32),
        "normals": np.array(mesh.normals, np.float32),
        "frames": np.array(mesh.frames, np.float32),
        "textures": np.array(mesh.textures, np.int64),
        "tints": np.array(mesh.tints, np.float32),
        "shifts": np.array(mesh.shifts, np.float32),
        "shades": np.array(mesh.shades, np.float32),
        "door_quads": np.array([part[0] for part in mesh.door_parts], np.int64),
        "door_index": np.array([part[1] for part in mesh.door_parts], np.int64),
        "door_slide": np.array([part[2] for part in mesh.door_parts], np.float32),
    }
