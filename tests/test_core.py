"""Behavioral acceptance tests for the headless 1A contract."""

import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from math import hypot, pi

import pytest

from game.core import GameCore
from game.map import WorldMap, small_test_map
from game.state import CoreConfig, SurvivorCommand


def arena(spawn=(5.0, 5.0)):
    return WorldMap(tuple("." * 20 for _ in range(20)), (spawn,))


def advance(core, command, ticks=30):
    for _ in range(ticks):
        result = core.step(command)
    return result


@pytest.mark.parametrize("running,speed", [(False, 2.5), (True, 4.0)])
@pytest.mark.parametrize(
    "forward,strafe", [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
)
def test_all_directions_have_expected_speed(running, speed, forward, strafe):
    result = advance(GameCore(arena()), SurvivorCommand(forward, strafe, running=running))
    assert hypot(result.survivor.x - 5, result.survivor.y - 5) == pytest.approx(speed)
    assert result.tick == 30
    assert result.survivor.yaw == 0


def test_rotation_applies_before_movement_and_strafe_does_not_turn():
    core = GameCore(arena())
    result = core.step(SurvivorCommand(forward=1, yaw_delta=pi / 2))
    assert result.survivor.x == pytest.approx(5)
    assert result.survivor.y == pytest.approx(5 + 2.5 / 30)
    result = core.step(SurvivorCommand(strafe=1))
    assert result.survivor.x == pytest.approx(5 - 2.5 / 30)
    assert result.survivor.yaw == pytest.approx(pi / 2)
    assert core.step(SurvivorCommand(yaw_delta=-3 * pi)).survivor.yaw == pytest.approx(3 * pi / 2)


def test_replay_and_reset():
    core = GameCore(config=CoreConfig(alien_enabled=False))
    initial = core.reset(123)
    commands = [SurvivorCommand(1, 1, 0.03, True), SurvivorCommand(-1, 0, -0.02)] * 150
    first = [core.step(c) for c in commands]
    assert core.reset(123) == initial
    assert [core.step(c) for c in commands] == first
    assert core.reset(123) == initial


def test_snapshots_are_immutable_and_old_snapshots_stay_unchanged():
    core = GameCore(config=CoreConfig(alien_enabled=False))
    before = core.snapshot()
    with pytest.raises(FrozenInstanceError):
        before.survivor.x = 99
    with pytest.raises(TypeError):
        before.world_map.rows[0] = "broken"
    altered = replace(before, survivor=replace(before.survivor, x=99))
    assert altered != core.snapshot()
    core.step(SurvivorCommand(1))
    assert before.tick == 0
    assert before.survivor != core.snapshot().survivor


@pytest.mark.parametrize("forward,strafe", [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)])
def test_world_edges_are_solid_even_without_border_tiles(forward, strafe):
    core = GameCore(arena())
    result = advance(core, SurvivorCommand(forward, strafe, running=True), 300)
    assert not result.world_map.overlaps(result.survivor.x, result.survivor.y, 0.2)
    assert 0.2 <= result.survivor.x <= 19.8
    assert 0.2 <= result.survivor.y <= 19.8


@pytest.mark.parametrize("speed", [4.0, 120.0])
def test_entire_movement_segment_cannot_cross_wall(speed):
    rows = tuple(".....#......" for _ in range(12))
    core = GameCore(WorldMap(rows, ((3.5, 5.5),)), CoreConfig(run_speed=speed))
    result = advance(core, SurvivorCommand(1, running=True), 90)
    assert result.survivor.x == pytest.approx(4.8)


def test_sliding_along_wall():
    core = GameCore(WorldMap(tuple(".....#......" for _ in range(12)), ((4.8, 3.5),)))
    result = advance(core, SurvivorCommand(1, 1, running=True))
    assert result.survivor.x == pytest.approx(4.8)
    assert result.survivor.y > 6


def test_corner_cannot_be_cut():
    rows = ("....", "..#.", ".#..", "....")
    core = GameCore(WorldMap(rows, ((1.5, 1.5),)))
    for _ in range(120):
        snap = core.step(SurvivorCommand(1, 1, running=True))
        assert not snap.world_map.overlaps(snap.survivor.x, snap.survivor.y, 0.2)
        assert snap.survivor.x < 2 and snap.survivor.y < 2


def test_authored_room_corridor_and_all_spawn_clearances():
    world = small_test_map()
    assert world.blocked(4, 3)
    assert not world.blocked(9, 6)
    for spawn in world.spawn_points:
        assert not world.overlaps(*spawn, 0.2)
    # The room and corridor belong to one connected traversable floor region.
    reached = {(2, 2)}
    pending = [(2, 2)]
    while pending:
        x, y = pending.pop()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (nx, ny) not in reached and not world.blocked(nx, ny):
                reached.add((nx, ny))
                pending.append((nx, ny))
    assert (9, 7) in reached


def test_map_copies_mutable_input():
    rows = ["....", "...."]
    spawns = [[1.5, 1.0]]
    world = WorldMap(rows, spawns)
    rows[0] = "####"
    spawns[0][0] = 99
    assert world.rows[0] == "...."
    assert world.spawn_points == ((1.5, 1.0),)


@pytest.mark.parametrize("row", [list("...."), tuple("...."), {"."}, {".": 0}, None, 123])
@pytest.mark.parametrize("row_index", [0, 1])
def test_map_rejects_non_string_rows(row, row_index):
    rows = ["....", "...."]
    rows[row_index] = row
    with pytest.raises(TypeError, match="Map rows must be strings"):
        WorldMap(rows, ((1.5, 1.0),))


def test_core_imports_without_third_party_libraries():
    code = "from game.core import GameCore; GameCore().step(__import__('game.state', fromlist=['SurvivorCommand']).SurvivorCommand())"
    subprocess.run([sys.executable, "-S", "-c", code], check=True)


@pytest.mark.parametrize(
    "kwargs", [{"forward": float("nan")}, {"strafe": 2}, {"yaw_delta": float("inf")}]
)
def test_invalid_commands_rejected(kwargs):
    with pytest.raises(ValueError):
        SurvivorCommand(**kwargs)


def test_invalid_config_and_spawn_rejected():
    with pytest.raises(ValueError):
        CoreConfig(tick_hz=60)
    with pytest.raises(ValueError):
        CoreConfig(radius=0)
    with pytest.raises(ValueError):
        GameCore(WorldMap(("##", "##"), ((0.5, 0.5),)))


@pytest.mark.parametrize(
    "yaw,spawn,axis,expected",
    [
        (pi, (7.5, 5.5), "x", 6.2),
        (pi / 2, (5.5, 3.5), "y", 4.8),
        (3 * pi / 2, (5.5, 7.5), "y", 6.2),
    ],
)
def test_swept_collision_other_directions(yaw, spawn, axis, expected):
    if axis == "x":
        rows = tuple(".....#......" for _ in range(12))
    else:
        rows = tuple("############" if y == 5 else "." * 12 for y in range(12))
    core = GameCore(WorldMap(rows, (spawn,)), CoreConfig(run_speed=120))
    core.step(SurvivorCommand(yaw_delta=yaw))
    result = advance(core, SurvivorCommand(1, running=True), 5)
    assert getattr(result.survivor, axis) == pytest.approx(expected)


def test_many_commands_never_overlap_authored_geometry():
    from random import Random

    core = GameCore(config=CoreConfig(alien_enabled=False))
    rng = Random(12)
    for _ in range(3000):
        command = SurvivorCommand(
            rng.choice([-1, 0, 1]),
            rng.choice([-1, 0, 1]),
            rng.uniform(-pi, pi),
            rng.choice([True, False]),
        )
        snap = core.step(command)
        assert not snap.world_map.overlaps(snap.survivor.x, snap.survivor.y, 0.2)


def test_render_schedule_cannot_change_fixed_tick_results():
    def simulate(ticks_per_frame):
        core = GameCore(arena())
        for count in ticks_per_frame:
            core.snapshot()  # a renderer can sample before or after ticks
            for _ in range(count):
                core.step(SurvivorCommand(1, yaw_delta=0.01))
        return core.snapshot()

    assert simulate([1] * 30) == simulate([0, 1] * 30) == simulate([3] * 10)
