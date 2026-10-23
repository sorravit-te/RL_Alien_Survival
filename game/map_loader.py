"""Import a grid-aligned Tiled (.tmj) object map without coupling Game Core to Tiled.

Author rectangular walls in an object layer named Walls and point entities in Entities.
One tile = one world unit. Pixel coordinates are converted at load time.
"""
import json
from math import isclose, radians
from pathlib import Path

from game.map import MapObject, WorldMap


def _properties(obj):
    return {p["name"]: p["value"] for p in obj.get("properties", [])}


def _grid_coordinate(value, size, label):
    scaled = value / size
    rounded = round(scaled)
    if not isclose(scaled, rounded, abs_tol=1e-7):
        raise ValueError(f"{label} must align to the Tiled grid")
    return int(rounded)


def load_tiled_map(filename: str | Path) -> WorldMap:
    """Read a finite orthogonal Tiled JSON map with wall rectangles and entity points.

    Accepted entity names: survivor_spawn, alien_spawn, waypoint, keycard, exit.
    Wall rectangles must be grid-aligned, axis-aligned and inside the map.
    Changing artwork or material IDs never changes collision geometry.
    """
    data = json.loads(Path(filename).read_text(encoding="utf-8"))
    if data.get("orientation") != "orthogonal" or data.get("infinite", False):
        raise ValueError("Only finite orthogonal Tiled maps are supported")
    width, height = data["width"], data["height"]
    tile_w, tile_h = data["tilewidth"], data["tileheight"]
    if any(type(v) is not int or v <= 0 for v in (width, height, tile_w, tile_h)):
        raise ValueError("Invalid map dimensions or tile size")
    if tile_w != tile_h:
        raise ValueError("Tiles must be square")

    layers = {layer.get("name"): layer for layer in data.get("layers", [])}
    if "Walls" not in layers or "Entities" not in layers:
        raise ValueError("Tiled map needs Walls and Entities object layers")
    if any(layers[name].get("type") != "objectgroup" for name in ("Walls", "Entities")):
        raise ValueError("Walls and Entities must be object layers")

    cells = [["." for _ in range(width)] for _ in range(height)]
    materials = [["" for _ in range(width)] for _ in range(height)]
    for obj in layers["Walls"].get("objects", []):
        if obj.get("point") or obj.get("ellipse") or obj.get("polygon") or obj.get("polyline"):
            raise ValueError("Walls must be rectangles")
        if obj.get("rotation", 0):
            raise ValueError("Rotated walls are not supported")
        x0 = _grid_coordinate(obj["x"], tile_w, "Wall x")
        y0 = _grid_coordinate(obj["y"], tile_h, "Wall y")
        w = _grid_coordinate(obj["width"], tile_w, "Wall width")
        h = _grid_coordinate(obj["height"], tile_h, "Wall height")
        if w <= 0 or h <= 0 or x0 < 0 or y0 < 0 or x0 + w > width or y0 + h > height:
            raise ValueError("Wall rectangle is empty or outside the map")
        material = _properties(obj).get("material", "wall")
        if not isinstance(material, str) or not material or not all(
            c.islower() or c.isdigit() or c in "_-" for c in material
        ):
            raise ValueError("Material IDs must use lowercase letters, digits, - or _")
        for y in range(y0, y0 + h):
            for x in range(x0, x0 + w):
                if cells[y][x] == "#" and materials[y][x] != material:
                    raise ValueError(f"Conflicting wall materials at ({x}, {y})")
                cells[y][x] = "#"
                materials[y][x] = material

    spawns, objects, waypoints = [], [], []
    alien_spawn = None
    for obj in layers["Entities"].get("objects", []):
        kind = str(obj.get("name", "")).strip().lower()
        if not obj.get("point", False) or obj.get("rotation", 0):
            raise ValueError(f"Entity {kind!r} must be an unrotated point")
        x, y = obj["x"] / tile_w, obj["y"] / tile_h
        if not (0 <= x < width and 0 <= y < height) or cells[int(y)][int(x)] == "#":
            raise ValueError(f"Entity {kind!r} is outside the floor")
        if kind == "survivor_spawn":
            spawns.append((x, y))
        elif kind == "alien_spawn":
            if alien_spawn is not None:
                raise ValueError("Only one alien_spawn is supported")
            angle = _properties(obj).get("yaw_degrees", 0)
            if not isinstance(angle, (int, float)):
                raise ValueError("yaw_degrees must be numeric")
            alien_spawn = (x, y, radians(angle))
        elif kind == "waypoint":
            waypoints.append((x, y))
        elif kind in ("keycard", "exit"):
            objects.append(MapObject(f"{kind}_{obj['id']}", kind, x, y))
        else:
            raise ValueError(f"Unsupported Tiled entity: {kind!r}")
    if not spawns:
        raise ValueError("At least one survivor_spawn is required")
    return WorldMap(
        tuple("".join(row) for row in cells),
        tuple(spawns),
        tuple(objects),
        alien_spawn=alien_spawn,
        patrol_waypoints=tuple(waypoints),
        wall_materials=tuple(tuple(row) for row in materials),
    )
