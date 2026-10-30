"""Single immutable geometry source for collision and future rendering."""

from dataclasses import dataclass, replace
from math import floor, isfinite, pi

from game.ship_deck import (
    ALIEN_SPAWNS,
    DOORS,
    EXITS,
    KEYCARDS,
    ROWS,
    SHIP_SCENARIO,
    SURVIVOR_SPAWNS,
    WAYPOINTS,
)


@dataclass(frozen=True)
class MapObject:
    object_id: str
    kind: str
    x: float
    y: float

    def __post_init__(self):
        if not isinstance(self.object_id, str) or not self.object_id:
            raise ValueError("Object needs a nonempty string ID")
        if self.kind not in ("keycard", "exit"):
            raise ValueError("Unsupported interaction object")
        if not all(isfinite(v) for v in (self.x, self.y)):
            raise ValueError("Object coordinates must be finite")


@dataclass(frozen=True)
class WorldMap:
    rows: tuple[str, ...]
    spawn_points: tuple[tuple[float, float], ...]
    objects: tuple[MapObject, ...] = ()
    alien_spawn: tuple[float, float, float] | None = None
    patrol_waypoints: tuple[tuple[float, float], ...] = ()
    alien_spawn_options: tuple[tuple[float, float, float], ...] = ()  # beside alien_spawn
    doors: tuple[tuple[int, int], ...] = ()  # floor cells closed by automatic doors

    def __post_init__(self):
        # Own immutable copies even when a caller supplies lists.
        object.__setattr__(self, "rows", tuple(self.rows))
        if any(not isinstance(row, str) for row in self.rows):
            raise TypeError("Map rows must be strings")
        object.__setattr__(self, "spawn_points", tuple(tuple(p) for p in self.spawn_points))
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "doors", tuple(tuple(cell) for cell in self.doors))
        if any(not isinstance(obj, MapObject) for obj in self.objects):
            raise TypeError("Map objects must be immutable MapObject values")
        if len({obj.object_id for obj in self.objects}) != len(self.objects):
            raise ValueError("Object IDs must be unique")
        if self.alien_spawn is not None:
            object.__setattr__(self, "alien_spawn", tuple(self.alien_spawn))
            if len(self.alien_spawn) != 3 or not all(isfinite(v) for v in self.alien_spawn):
                raise ValueError("Alien spawn must contain finite x/y/yaw")
        object.__setattr__(self, "patrol_waypoints", tuple(tuple(p) for p in self.patrol_waypoints))
        object.__setattr__(
            self, "alien_spawn_options", tuple(tuple(p) for p in self.alien_spawn_options)
        )
        if self.alien_spawn_options and self.alien_spawn is None:
            raise ValueError("Alien spawn options need a primary Alien spawn")
        if any(len(p) != 3 or not all(isfinite(v) for v in p) for p in self.alien_spawn_options):
            raise ValueError("Alien spawn must contain finite x/y/yaw")
        if any(len(p) != 2 or not all(isfinite(v) for v in p) for p in self.patrol_waypoints):
            raise ValueError("Patrol waypoint must contain finite x/y")
        if not self.rows or not self.rows[0]:
            raise ValueError("Map must not be empty")
        if any(len(row) != self.width or set(row) - {"#", "."} for row in self.rows):
            raise ValueError("Map must be rectangular and contain only # or .")
        if not self.spawn_points:
            raise ValueError("Map needs at least one spawn point")
        if any(len(p) != 2 or not all(isfinite(v) for v in p) for p in self.spawn_points):
            raise ValueError("Invalid spawn point")

        if any(self.blocked(floor(obj.x), floor(obj.y)) for obj in self.objects):
            raise ValueError("Interaction objects must be on traversable floor")

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
    """Room, tall obstacle and L-shaped corridor with a fixed card and exit point."""
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
        (MapObject("keycard", "keycard", 5.5, 2.5), MapObject("exit", "exit", 9.5, 7.5)),
        alien_spawn=(9.5, 7.5, -pi / 2),
        patrol_waypoints=((9.5, 7.5), (9.5, 4.5), (5.5, 4.5), (5.5, 2.5), (2.5, 2.5), (2.5, 5.5)),
    )


def ship_deck() -> WorldMap:
    objects = tuple(
        MapObject(object_id, kind, x, y)
        for kind, points in (("keycard", KEYCARDS), ("exit", EXITS))
        for object_id, x, y in points
    )
    return WorldMap(
        ROWS,
        SURVIVOR_SPAWNS,
        objects,
        alien_spawn=ALIEN_SPAWNS[0],
        patrol_waypoints=WAYPOINTS,
        alien_spawn_options=ALIEN_SPAWNS[1:],
        doors=DOORS,
    )


DEFAULT_SCENARIO = "small_test_map_v1"
RANDOM_SCENARIO = "occluded_loop_random_v1"
SCENARIOS = (DEFAULT_SCENARIO, "occluded_loop_dev_v1", RANDOM_SCENARIO, SHIP_SCENARIO)


def scenario_map(name=DEFAULT_SCENARIO) -> WorldMap:
    """Explicit opt-in scene factory; the playable default remains small_test_map."""
    if name == DEFAULT_SCENARIO:
        return small_test_map()
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name!r}")
    if name == SHIP_SCENARIO:
        return ship_deck()
    # Reviewed fixed DEV world. Keep object/order/waypoint identity exact.
    world = WorldMap(
        (
            "################",
            "#..............#",
            "#..............#",
            "#..............#",
            "#....######....#",
            "#....######....#",
            "#....######....#",
            "#....######....#",
            "#....######....#",
            "#....######....#",
            "#....######....#",
            "#..............#",
            "#........##....#",
            "#........##....#",
            "#........##....#",
            "################",
        ),
        ((2.5, 11.5), (2.5, 13.5)),
        (MapObject("keycard", "keycard", 2.5, 2.5), MapObject("exit", "exit", 13.5, 13.5)),
        alien_spawn=(13.5, 2.5, pi),
        patrol_waypoints=(
            (2.5, 2.5),
            (2.5, 13.5),
            (13.5, 13.5),
            (13.5, 2.5),
            (4.5, 11.5),
            (11.5, 11.5),
        ),
    )
    if name != RANDOM_SCENARIO:
        return world
    # Same walls and waypoints. Each reset keeps one card and one working exit.
    return replace(
        world,
        spawn_points=((2.5, 11.5), (2.5, 13.5), (13.5, 11.5)),
        objects=(
            MapObject("keycard_west", "keycard", 2.5, 7.5),
            MapObject("keycard_north", "keycard", 7.5, 2.5),
            MapObject("keycard_east", "keycard", 13.5, 7.5),
            MapObject("keycard_south", "keycard", 7.5, 12.5),
            MapObject("exit_northwest", "exit", 2.5, 2.5),
            MapObject("exit_northeast", "exit", 13.5, 2.5),
            MapObject("exit_southeast", "exit", 13.5, 13.5),
        ),
        alien_spawn_options=((2.5, 2.5, 0.0),),
    )
