"""Source-center shortcuts, geometry safety and opt-in truth HUD; no policy replay."""

from dataclasses import replace
from math import hypot, pi

import pygame
import pytest

from ai.navigation import Navigator, find_path
from app.loop import FixedStepLoop
from app.play import alien_debug_labels, hud_labels
from game.core import GameCore
from game.interaction import segment_clear
from game.map import WorldMap, small_test_map
from game.motion import segment_has_clearance
from game.state import CoreConfig, SurvivorCommand

# Recorded decision inputs, not a historical checkpoint or command replay.
CASES = [
    (
        (5.284250658936464, 4.852041980190635),
        (4.714319532401225, 5.8),
        ((5.5, 4.5), (5.5, 5.5), (4.5, 5.5), (4.714319532401225, 5.8)),
    ),
    (
        (3.028123145431911, 4.113180972520039),
        (4.325369675979193, 2.3486982506272196),
        ((3.5, 4.5), (3.5, 3.5), (3.5, 2.5), (4.5, 2.5), (4.325369675979193, 2.3486982506272196)),
    ),
    (
        (8.059999999999993, 3.5),
        (4.050000000000004, 2.75),
        ((8.5, 3.5), (7.5, 3.5), (6.5, 3.5), (5.5, 2.5), (4.5, 2.5), (4.050000000000004, 2.75)),
    ),
]


def length(start, path):
    return sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip((start,) + path, path))


@pytest.mark.parametrize(
    "start,target,old_path", CASES, ids=["human_ep0_795", "human_ep3_210", "synthetic_60"]
)
def test_recorded_reversal_only_removes_initial_center_with_circle_clearance(
    start, target, old_path
):
    world = small_test_map()
    path = find_path(world, start, target, 0.2)
    assert not segment_has_clearance(world, start, target, 0.2)
    assert path == old_path[1:]  # No smoothing/target substitution beyond the first point.
    assert length(start, path) < length(start, old_path)
    direction = (target[0] - start[0], target[1] - start[1])
    assert sum((old_path[0][i] - start[i]) * direction[i] for i in (0, 1)) < 0
    assert sum((path[0][i] - start[i]) * direction[i] for i in (0, 1)) > 0
    for a, b in zip((start,) + path, path):
        assert segment_has_clearance(world, a, b, 0.2)
    assert all(find_path(world, start, target, 0.2) == path for _ in range(5))


@pytest.mark.parametrize("start,target,old_path", CASES)
def test_core_scripted_target_moves_toward_next_point_without_speed_change_and_reset(
    start, target, old_path
):
    world = replace(small_test_map(), alien_spawn=(*start, 0), spawn_points=((9.5, 7.5),))
    config = CoreConfig(alien_sight_range=0.1, alien_near_range=0.1)
    core = GameCore(world, config, alien_control="external")

    def run():
        core.reset(7799201)
        assert core.snapshot().navigation_path == () and core.snapshot().alien_target is None
        core.set_alien_target(target, source="visible")  # Scripted, NOT a policy call.
        assert core.snapshot().navigation_path == old_path[1:]
        before = core.snapshot()
        after = core.step(SurvivorCommand())
        assert after.alien_target == target and after.alien_target_source == "visible"
        delta = (after.alien.x - before.alien.x, after.alien.y - before.alien.y)
        assert hypot(*delta) == pytest.approx(config.alien_search_speed / config.tick_hz)
        assert segment_has_clearance(
            world, start, (after.alien.x, after.alien.y), config.alien_radius
        )
        return after

    assert run() == run()


def test_point_los_at_narrow_corner_is_insufficient_and_center_must_remain():
    world = WorldMap(("......", "..#...", "......", "......", "......"), ((1.78, 2.02),))
    start, target = (1.78, 2.02), (4.5, 1.5)
    path = find_path(world, start, target, 0.2)
    assert path == ((1.5, 2.5), (2.5, 2.5), (3.5, 2.5), target)
    assert segment_clear(world, *start, *path[1])
    assert not segment_has_clearance(world, start, path[1], 0.2)
    for a, b in zip((start,) + path, path):
        assert segment_has_clearance(world, a, b, 0.2)


def test_direct_route_unreachable_reset_and_radius_constraints():
    world = WorldMap(("......",) * 6, ((1.2, 1.2),))
    assert find_path(world, (1.2, 1.2), (4.5, 3.5), 0.2) == ((4.5, 3.5),)
    blocked = WorldMap(("...#..",) * 6, ((1.5, 2.5),))
    nav = Navigator()
    nav.set_target(blocked, (1.5, 2.5), (4.5, 2.5), 0.2)
    assert nav.path == () and nav.displacement((1.5, 2.5), 0.1) == (0, 0)
    nav.set_target(blocked, (1.5, 2.5), None, 0.2)
    assert nav.target is None and nav.path == ()
    narrow = WorldMap(
        ("#######", "#.....#", "#.....#", "###.###", "#.....#", "#.....#", "#######"), ((3.5, 1.8),)
    )
    assert find_path(narrow, (3.5, 1.8), (3.5, 5.2), 0.2)
    assert find_path(narrow, (3.5, 1.8), (3.5, 5.2), 0.6) == ()
    corner = WorldMap(("....", "..#.", ".#..", "...."), ((1.5, 1.5),))
    path = find_path(corner, (1.5, 1.5), (2.5, 2.5), 0.2)
    assert len(path) > 2
    for a, b in zip(((1.5, 1.5),) + path, path):
        assert segment_has_clearance(corner, a, b, 0.2)


@pytest.mark.parametrize(
    "yaw,expected", [(0, "0.0"), (pi / 2, "90.0"), (pi, "180.0"), (3 * pi / 2, "270.0")]
)
def test_debug_displays_actual_tick_yaw_separately_from_sensor_without_mutating_core(yaw, expected):
    core = GameCore()
    core._alien = replace(core.snapshot().alien, yaw=yaw)
    before = core.snapshot()
    loop = FixedStepLoop(core)
    assert not any("DEBUG" in label for label in hud_labels(loop))
    labels = hud_labels(loop, debug_alien=True)
    assert any(f"yaw {expected} deg" in label for label in labels)
    assert any("confirmed-visible=False" in label for label in labels)
    changed = replace(before, perception=replace(before.perception, visible=True))
    assert alien_debug_labels(changed)[0] == alien_debug_labels(before)[0]
    assert "confirmed-visible=True" in alien_debug_labels(changed)[1]
    assert core.snapshot() == before


def test_debug_when_alien_disabled_and_terminal_remains_explicitly_optional():
    core = GameCore(WorldMap(("....",) * 4, ((1.5, 1.5),)))
    assert alien_debug_labels(core.snapshot()) == ["DEBUG Alien disabled"]
    loop = FixedStepLoop(GameCore())
    loop.current = replace(loop.current, outcome="captured")
    assert not any("DEBUG" in s for s in hud_labels(loop))
    assert any("DEBUG truth" in s for s in hud_labels(loop, debug_alien=True))


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_debug_flag_is_off_by_default_and_pygame_closes(monkeypatch, enabled):
    from app import play

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    batches = iter([[], [pygame.event.Event(pygame.QUIT)]])
    monkeypatch.setattr(pygame.event, "get", lambda: next(batches))

    class Renderer:
        def render(self, *args, **kwargs):
            return pygame.Surface((320, 180))

    monkeypatch.setattr(play, "RaycastingRenderer", Renderer)
    seen = []
    original = play.hud_labels

    def labels(loop, *, debug_alien=False):
        seen.append(debug_alien)
        return original(loop, debug_alien=debug_alien)

    monkeypatch.setattr(play, "hud_labels", labels)
    play.main(["--debug-alien"] if enabled else [])
    assert seen == [enabled]
    assert not pygame.display.get_init() and not pygame.font.get_init()
