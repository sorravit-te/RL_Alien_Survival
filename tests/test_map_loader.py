"""Tests for user-authored Tiled maps and independent render material metadata."""
import json
from pathlib import Path

import pytest

from game.map import WorldMap
from game.map_loader import load_tiled_map

DEMO = Path(__file__).resolve().parents[1] / "maps" / "station_demo.tmj"


def test_station_demo_loads():
    world = load_tiled_map(DEMO)
    assert (world.width, world.height) == (16, 12)
    assert world.wall_material(1, 4) == "lab"
    assert world.wall_material(11, 4) == "crew"
    assert not world.blocked(2, 2)
    assert len(world.objects) == 2
    assert len(world.patrol_waypoints) == 3
    assert world.alien_spawn is not None


def test_materials_do_not_change_collision():
    world = WorldMap(("###", "#.#", "###"), ((1.5, 1.5),),
                     wall_materials=(("wall", "lab", "wall"),
                                     ("wall", "", "crew"),
                                     ("wall", "wall", "wall")))
    assert world.blocked(2, 1)
    assert world.wall_material(2, 1) == "crew"
    assert world.wall_material(1, 1) == ""


def test_tiled_rejects_wall_outside_bounds(tmp_path):
    data = json.loads(DEMO.read_text(encoding="utf-8"))
    data["layers"][0]["objects"][0]["width"] = 9999
    path = tmp_path / "bad.tmj"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        load_tiled_map(path)


def test_tiled_rejects_solid_spawn(tmp_path):
    data = json.loads(DEMO.read_text(encoding="utf-8"))
    next(item for item in data["layers"][1]["objects"] if item["name"] == "survivor_spawn")["x"] = 0
    path = tmp_path / "bad.tmj"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="outside the floor"):
        load_tiled_map(path)
