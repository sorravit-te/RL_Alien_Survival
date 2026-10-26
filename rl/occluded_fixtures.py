"""Versioned DEV blockouts and public routes, independent of production map creation.

Each WorldMap is the sole grid/object catalogue used by Core/Env. The Survivor
bot receives only its rows plus filtered sensors, never this map's object truth.
"""

import json
from dataclasses import dataclass
from math import hypot, isfinite
from pathlib import Path

from ai.navigation import find_path
from game.map import MapObject, WorldMap, small_test_map
from game.state import CoreConfig

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "configs/occluded_dev_fixtures_v2.json"


@dataclass(frozen=True)
class OccludedFixture:
    case_id: str
    layout_id: str
    scenario_seed: int
    bot_seed: int
    exploration_anchors: tuple[tuple[float, float], ...]
    profile: str
    intended_alignment: str
    scene: WorldMap

    def world(self):
        return self.scene


def _point(value, length=2):
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError("Fixture coordinate has wrong shape")
    if not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v) for v in value
    ):
        raise ValueError("Fixture coordinates must be finite numbers")
    return tuple(value)


def read_occluded_fixtures(path=DEFAULT_MANIFEST):
    data = json.loads(Path(path).read_text(encoding="utf8"))
    if data["format"] != "occluded-dev-fixtures-v2" or data["split"] != "development_only":
        raise ValueError("Expected versioned development fixture manifest")
    if not 1 <= len(data["layouts"]) <= 2 or not 1 <= len(data["cases"]) <= 12:
        raise ValueError("At most two small layouts and twelve cases are allowed")
    if (data["episode_cap_ticks"], data["total_core_tick_cap"], data["execution_cap"]) != (
        1800,
        43200,
        24,
    ):
        raise ValueError("Unexpected authorized budget")
    if set(data["controllers"]) != {"wait", "heuristic"}:
        raise ValueError("Only Wait and the shared Heuristic are allowed")
    config, cases, ids, seeds = CoreConfig(), [], set(), set()
    for raw in data["cases"]:
        case_id = raw["case_id"]
        if (
            not isinstance(case_id, str)
            or not case_id
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in case_id)
            or case_id in ids
        ):
            raise ValueError("Case IDs must be unique safe lowercase identifiers")
        ids.add(case_id)
        for key in ("scenario_seed", "bot_seed"):
            seed = raw[key]
            if type(seed) is not int or seed < 0 or seed in seeds:
                raise ValueError("Use unique nonnegative DEV seeds")
            seeds.add(seed)
        layout = data["layouts"][raw["layout_id"]]
        spawn, alien = _point(raw["survivor_spawn"]), _point(raw["alien_spawn"], 3)
        world = WorldMap(
            tuple(layout["rows"]),
            (spawn,),
            tuple(MapObject(**o) for o in layout["objects"]),
            alien,
            tuple(_point(p) for p in layout["patrol_waypoints"]),
        )
        if max(world.width, world.height) > 16:
            raise ValueError("DEV blockout exceeds 16x16")
        if any(c != "#" for c in world.rows[0] + world.rows[-1]) or any(
            row[0] != "#" or row[-1] != "#" for row in world.rows
        ):
            raise ValueError("DEV blockout must have solid outer walls")
        if sorted(o.kind for o in world.objects) != ["exit", "keycard"]:
            raise ValueError("Use exactly one Keycard and exit")
        if not 1 <= len(world.patrol_waypoints) <= 20:
            raise ValueError("Expected existing external catalogue capacity")
        if (
            world.overlaps(*spawn, config.radius)
            or world.overlaps(*alien[:2], config.alien_radius)
            or hypot(spawn[0] - alien[0], spawn[1] - alien[1]) < config.minimum_spawn_distance
        ):
            raise ValueError("Unsafe fixture spawn")
        anchors = tuple(_point(p) for p in raw["exploration_anchors"])
        if not anchors or len(set(anchors)) != len(anchors):
            raise ValueError("Public exploration anchors must be nonempty and unique")
        targets = (*anchors, *((o.x, o.y) for o in world.objects), *world.patrol_waypoints)
        for start, radius in ((spawn, config.radius), (alien[:2], config.alien_radius)):
            if any(not find_path(world, start, target, radius) for target in targets):
                raise ValueError("Disconnected or blocked fixture route/object")
        if raw["layout_id"] == "small_room_control":
            base = small_test_map()
            if (world.rows, world.objects, world.patrol_waypoints) != (
                base.rows,
                base.objects,
                base.patrol_waypoints,
            ):
                raise ValueError("Production-map control must preserve geometry and catalogue")
        cases.append(
            OccludedFixture(
                case_id,
                raw["layout_id"],
                raw["scenario_seed"],
                raw["bot_seed"],
                anchors,
                raw["profile"],
                raw["intended_alignment"],
                world,
            )
        )
    return data, tuple(cases)
