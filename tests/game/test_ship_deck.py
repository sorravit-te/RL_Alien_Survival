"""The full deck: a connected, loop-rich layout whose every scenario can start and finish."""

from collections import deque
from math import floor, hypot
from random import Random

import pytest

from ai.navigation import find_path
from game.core import GameCore
from game.interaction import segment_clear
from game.map import scenario_map
from game.ship_deck import (
    ALIEN_SPAWNS,
    EXITS,
    KEYCARDS,
    LOCKER_SPOTS,
    ROWS,
    SHIP_SCENARIO,
    WAYPOINTS,
)
from game.state import CoreConfig
from renderers.ship_theme import SIGNS, WARNINGS
from renderers.ship_zones import ZONE_ROWS

WORLD = scenario_map(SHIP_SCENARIO)
CONFIG = CoreConfig()
FACES = ((-1, 0), (1, 0), (0, -1), (0, 1))


def floor_cells(blocked=()):
    return {
        (x, y)
        for y, row in enumerate(ROWS)
        for x, tile in enumerate(row)
        if tile == "." and (x, y) not in blocked
    }


def reachable(start, cells):
    seen, queue = {start}, deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in FACES:
            cell = (x + dx, y + dy)
            if cell in cells and cell not in seen:
                seen.add(cell)
                queue.append(cell)
    return seen


def cell_of(point):
    return floor(point[0]), floor(point[1])


def test_counts_match_the_plan():
    assert (WORLD.width, WORLD.height) == (40, 40)
    assert len(WAYPOINTS) == 20 and len(KEYCARDS) == 6 and len(EXITS) == 3
    assert len(LOCKER_SPOTS) == 6
    assert len({zone for row in ZONE_ROWS for zone in row}) >= 10


def test_every_floor_cell_connects():
    cells = floor_cells()
    assert reachable(next(iter(cells)), cells) == cells


def test_all_points_stand_clear_of_walls():
    points = (
        *WORLD.spawn_points,
        *(spawn[:2] for spawn in ALIEN_SPAWNS),
        *WAYPOINTS,
        *((x, y) for _, x, y in (*KEYCARDS, *EXITS)),
        *LOCKER_SPOTS,
    )
    assert not any(WORLD.overlaps(*point, CONFIG.radius) for point in points)


@pytest.mark.parametrize("start", WORLD.spawn_points)
def test_each_spawn_reaches_every_card_and_exit(start):
    for _, x, y in (*KEYCARDS, *EXITS):
        assert find_path(WORLD, start, (x, y), CONFIG.radius)


def test_no_spawn_pair_starts_in_sight():
    for sx, sy in WORLD.spawn_points:
        for ax, ay, _ in ALIEN_SPAWNS:
            near = hypot(ax - sx, ay - sy) <= CONFIG.alien_sight_range
            assert not (near and segment_clear(WORLD, sx, sy, ax, ay))


def test_each_exit_faces_a_different_side():
    sides = {
        (
            "west" if x < 10 else "east" if x > 30 else "middle",
            "south" if y > 30 else "north" if y < 10 else "middle",
        )
        for _, x, y in EXITS
    }
    assert len(sides) == 3


@pytest.mark.parametrize("door", [(6, 16), (10, 29), (22, 19), (29, 16), (17, 31), (24, 29)])
def test_main_rooms_survive_one_blocked_doorway(door):
    """No single door traps the player: every waypoint still connects without it."""
    cells = floor_cells(blocked={door})
    seen = reachable(cell_of(WAYPOINTS[6]), cells)
    assert all(cell_of(point) in seen for point in WAYPOINTS)


def test_corridors_form_loops():
    """Removing either ring corridor leaves the spine linked, so a chase can always circle."""
    west = {(x, 17) for x in (7, 8)}
    east = {(x, 17) for x in (30, 31, 32)}
    for blocked in (west, east):
        cells = floor_cells(blocked=blocked)
        seen = reachable(cell_of(WAYPOINTS[6]), cells)
        assert cell_of(WAYPOINTS[8]) in seen and cell_of(WAYPOINTS[12]) in seen


def test_zone_rows_trace_the_same_floor():
    assert len(ZONE_ROWS) == len(ROWS)
    for row, zones in zip(ROWS, ZONE_ROWS, strict=True):
        assert [tile == "#" for tile in row] == [zone == "#" for zone in zones]


def test_painted_signs_face_open_floor():
    for x, y, face in (*SIGNS, *WARNINGS):
        dx, dy = FACES[face]
        assert ROWS[y][x] == "#" and ROWS[y + dy][x + dx] == ".", (x, y, face)


def test_core_starts_every_seeded_scenario_inside_the_ship():
    core = GameCore(WORLD)
    for seed in range(120):
        snapshot = core.reset(seed)
        assert (snapshot.survivor.x, snapshot.survivor.y) in WORLD.spawn_points
        assert sum(obj.kind == "keycard" for obj in snapshot.world_map.objects) == 1
        assert sum(obj.kind == "exit" for obj in snapshot.world_map.objects) == 3


def test_scenarios_cover_all_card_and_exit_choices():
    cards, exits = set(), set()
    core = GameCore(WORLD)
    for seed in Random(7).sample(range(100000), 200):
        snapshot = core.reset(seed)
        cards.add(next(o.object_id for o in snapshot.world_map.objects if o.kind == "keycard"))
        exits.add(core._active_exit)
    assert len(cards) == 6 and len(exits) == 3
