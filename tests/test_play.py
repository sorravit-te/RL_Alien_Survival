"""1B geometry, input, scheduler and window-event regressions (dummy SDL)."""

import os
from dataclasses import replace
from math import pi

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import pygame
import pytest

from app.input import InputState
from app.loop import FixedStepLoop
from app.play import PlaySession
from game.core import GameCore
from game.map import WorldMap
from game.state import SurvivorCommand, SurvivorState
from renderers.raycasting import RaycastingRenderer, cast_ray


@pytest.fixture
def display():
    pygame.display.init()
    pygame.display.set_mode((960, 540))
    yield
    pygame.quit()


def test_ray_cardinal_directions_and_world_edges():
    world = WorldMap(('....',) * 4, ((1.5, 1.5),))
    for direction, distance, cell, side in [
        ((1, 0), 2.5, (4, 1), 0), ((-1, 0), 1.5, (-1, 1), 0),
        ((0, 1), 2.5, (1, 4), 1), ((0, -1), 1.5, (1, -1), 1),
    ]:
        hit = cast_ray(world, 1.5, 1.5, *direction)
        assert hit.distance == pytest.approx(distance)
        assert (hit.cell_x, hit.cell_y, hit.side) == (*cell, side)
    with pytest.raises(ValueError):
        cast_ray(world, 1.5, 1.5, 0, 0)


def test_ray_stops_at_first_obstacle():
    world = WorldMap(('.....#..',) * 6, ((2.5, 2.5),))
    hit = cast_ray(world, 2.5, 2.5, 1, .1)
    assert hit.distance == pytest.approx(2.5)
    assert (hit.cell_x, hit.cell_y) == (5, 2)
    assert hit.wall_position == pytest.approx(2.75)


def test_front_wall_has_constant_perpendicular_depth_and_no_fisheye(display, tmp_path):
    texture = pygame.Surface((32, 16))  # exercise non-square textures too
    texture.fill((180, 160, 140))
    path = tmp_path / 'wall.png'
    pygame.image.save(texture, path)
    renderer = RaycastingRenderer(path)
    core = GameCore(WorldMap(('.....#',) * 12, ((2.5, 5.5),)))
    before = core.snapshot()
    frame = renderer.render(before)
    assert frame.get_size() == (320, 180)
    assert len(renderer.depth) == 320
    assert renderer.depth == pytest.approx([2.5] * 320)
    tops = [next(y for y in range(90) if frame.get_at((x, y))[:3] == (180, 160, 140))
            for x in (0, 80, 160, 240, 319)]
    assert len(set(tops)) == 1
    assert core.snapshot() == before
    # Rendering between ticks cannot affect the deterministic simulation.
    other = GameCore(before.world_map)
    for _ in range(20):
        renderer.render(core.snapshot())
        assert core.step(SurvivorCommand(.5, .5, .02)) == other.step(
            SurvivorCommand(.5, .5, .02))


def test_missing_texture_explains_filename_and_preparation(tmp_path):
    missing = tmp_path / 'missing-wall.png'
    with pytest.raises(FileNotFoundError) as error:
        RaycastingRenderer(missing)
    assert str(missing) in str(error.value)
    assert '-m tools.prepare_assets' in str(error.value)


def test_mouse_accumulates_until_tick_then_consumed_once():
    inputs = InputState(mouse_sensitivity=.01)
    loop = FixedStepLoop(GameCore(), inputs)
    inputs.mouse(12)
    assert loop.advance(.01) == 0
    inputs.mouse(-2)
    assert loop.advance(.09) == 3
    assert loop.current.survivor.yaw == pytest.approx(.1)
    assert inputs.consume(loop.dt).yaw_delta == 0


def test_opposite_keys_shift_and_fixed_rate_arrow_turn():
    inputs = InputState()
    for key in ('w', 's', 'a', 'd', 'left', 'right', 'shift_right'):
        inputs.key(key, True)
    command = inputs.consume(1 / 30)
    assert (command.forward, command.strafe, command.yaw_delta, command.running) == (0, 0, 0, True)
    inputs.key('left', False)
    assert sum(inputs.consume(1 / 30).yaw_delta for _ in range(30)) == pytest.approx(2 * pi / 3)
    inputs.clear()
    assert inputs.consume(1 / 30) == SurvivorCommand()


def test_fixed_tick_result_independent_of_render_schedule():
    def run(deltas):
        loop = FixedStepLoop(GameCore())
        loop.inputs.key('w', True)
        loop.inputs.key('right', True)
        for elapsed in deltas:
            loop.advance(elapsed)
            loop.display_snapshot()
        return loop.current
    assert run([1 / 60] * 120) == run([1 / 144] * 288) == run([.1] * 20)
    assert run([1 / 60] * 120).tick == 60


def test_slow_frame_has_bounded_catchup_and_no_endless_backlog():
    loop = FixedStepLoop(GameCore())
    assert loop.advance(10) == 5
    assert 0 <= loop.accumulator < loop.dt
    assert loop.dropped_seconds > 9.8
    assert loop.advance(0) == 0


def test_pause_reset_clear_inputs_interpolation_and_time():
    loop = FixedStepLoop(GameCore())
    initial = loop.current
    loop.inputs.key('w', True)
    loop.advance(.05)
    loop.inputs.mouse(200)
    loop.set_paused(True)
    frozen = loop.current
    assert loop.advance(20) == 0
    assert loop.display_snapshot() == frozen
    loop.set_paused(False)
    loop.advance(loop.dt)
    assert loop.current.survivor == frozen.survivor
    loop.inputs.mouse(400)
    loop.reset()
    assert loop.current == loop.previous == initial
    assert loop.accumulator == 0
    assert loop.inputs.consume(loop.dt) == SurvivorCommand()
    assert loop.display_snapshot() == initial


def test_interpolation_yaw_uses_short_arc():
    loop = FixedStepLoop(GameCore())
    loop.previous = replace(loop.current, survivor=replace(loop.current.survivor, yaw=2*pi-.1))
    loop.current = replace(loop.current, survivor=replace(loop.current.survivor, yaw=.1))
    loop.accumulator = loop.dt / 2
    assert loop.display_snapshot().survivor.yaw == pytest.approx(0)


def test_interpolation_never_cuts_across_solid_corner():
    world = WorldMap(('....', '..#.', '.#..', '....'), ((1.5, 1.5),))
    loop = FixedStepLoop(GameCore(world))
    # Two individually valid endpoints whose straight chord intersects a wall.
    loop.previous = replace(loop.current, survivor=SurvivorState(2.5, .5))
    loop.current = replace(loop.current, survivor=SurvivorState(3.5, 1.5))
    for fraction in (.1, .3, .5, .7, .9):
        loop.accumulator = loop.dt * fraction
        camera = loop.display_snapshot().survivor
        assert not world.overlaps(camera.x, camera.y, .2)
    assert loop.display_snapshot().survivor == loop.current.survivor


def event(kind, **attributes):
    return pygame.event.Event(kind, **attributes)


def test_window_pause_focus_reset_and_stale_batch_events(display):
    loop = FixedStepLoop(GameCore())
    session = PlaySession(loop)
    session.focused = True
    session.events([event(pygame.KEYDOWN, key=pygame.K_ESCAPE),
                    event(pygame.KEYDOWN, key=pygame.K_w),
                    event(pygame.MOUSEMOTION, rel=(200, 0))])
    assert not loop.paused
    assert loop.inputs.consume(loop.dt) == SurvivorCommand()
    session.events([event(pygame.KEYDOWN, key=pygame.K_w),
                    event(pygame.MOUSEMOTION, rel=(100, 0))])
    loop.advance(loop.dt)
    assert loop.current.tick == 1
    session.events([event(pygame.WINDOWFOCUSLOST),
                    event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    frozen = loop.current
    assert loop.paused
    assert loop.inputs.consume(loop.dt) == SurvivorCommand()
    session.events([event(pygame.WINDOWFOCUSGAINED)])
    assert loop.paused
    assert loop.advance(10) == 0
    session.events([event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    loop.advance(loop.dt)
    assert loop.current.survivor == frozen.survivor
    session.events([event(pygame.KEYDOWN, key=pygame.K_r),
                    event(pygame.MOUSEMOTION, rel=(900, 0))])
    assert loop.current.tick == 0
    assert loop.inputs.consume(loop.dt) == SurvivorCommand()
    session.events([event(pygame.QUIT)])
    assert not session.running


def test_input_loop_reaches_corridor_end_and_slides_on_walls():
    loop = FixedStepLoop(GameCore())
    loop.core.reset(1)  # spawn (2.5, 2.5)
    loop.synchronize()
    # East in upper room, south beside the tall obstacle, east into corridor, south.
    for key, ticks in [('w', 36), ('d', 24), ('w', 48), ('d', 36)]:
        loop.inputs.clear()
        loop.inputs.key(key, True)
        for _ in range(ticks):
            loop.advance(loop.dt)
            camera = loop.display_snapshot().survivor
            assert not loop.current.world_map.overlaps(camera.x, camera.y, .2)
    assert loop.current.survivor.x > 9
    assert loop.current.survivor.y > 7


@pytest.mark.parametrize('yaw', [0, pi / 2, pi, 3 * pi / 2])
@pytest.mark.parametrize('key,forward,strafe', [('w', 1, 0), ('s', -1, 0),
                                              ('a', 0, -1), ('d', 0, 1)])
def test_device_key_mapping_matches_core_movement_at_each_yaw(yaw, key, forward, strafe):
    world = WorldMap(('.' * 20,) * 20, ((10., 10.),))
    actual, expected = GameCore(world), GameCore(world)
    actual.step(SurvivorCommand(yaw_delta=yaw))
    expected.step(SurvivorCommand(yaw_delta=yaw))
    loop = FixedStepLoop(actual)
    loop.inputs.key(key, True)
    loop.inputs.key('shift_left', True)
    for _ in range(30):
        loop.advance(loop.dt)
        expected.step(SurvivorCommand(forward, strafe, running=True))
    assert loop.current == expected.snapshot()


def test_normal_window_close_releases_pygame(monkeypatch, tmp_path):
    from app import play

    texture_path = tmp_path / 'wall.png'
    pygame.image.save(pygame.Surface((16, 16)), texture_path)
    monkeypatch.setattr(play, 'RaycastingRenderer', lambda: RaycastingRenderer(texture_path))
    monkeypatch.setattr(pygame.event, 'get', lambda: [event(pygame.QUIT)])
    play.main([])
    assert not pygame.display.get_init()
    assert not pygame.font.get_init()


@pytest.mark.parametrize('dy,expected', [(-20, .2), (20, -.2), (-10000, 25*pi/180),
                                       (10000, -25*pi/180)])
def test_pitch_direction_limit_and_separate_sensitivity(dy, expected):
    inputs = InputState(mouse_sensitivity=.002, vertical_sensitivity=.01)
    inputs.mouse(10, dy)
    command = inputs.consume(1 / 30)
    assert command.yaw_delta == pytest.approx(.02)
    assert inputs.pitch == pytest.approx(expected)
    assert (inputs.mouse_dx, inputs.mouse_dy) == (0, 0)
    inputs.consume(1 / 30)
    assert inputs.pitch == pytest.approx(expected)


def test_both_mouse_axes_wait_for_tick_and_are_not_reused_in_catchup():
    inputs = InputState(mouse_sensitivity=.01, vertical_sensitivity=.01)
    loop = FixedStepLoop(GameCore(), inputs)
    inputs.mouse(12, -12)
    assert loop.advance(.01) == 0
    assert inputs.pitch == 0
    inputs.mouse(-2, 2)
    assert loop.advance(.09) == 3
    assert loop.current.survivor.yaw == pytest.approx(.1)
    assert inputs.pitch == pytest.approx(.1)
    loop.advance(.1)
    assert loop.current.survivor.yaw == pytest.approx(.1)
    assert inputs.pitch == pytest.approx(.1)


def test_pitch_is_independent_of_render_frequency():
    def simulate(frames):
        loop = FixedStepLoop(GameCore())
        for _ in range(frames):
            loop.inputs.mouse(30 / frames, -60 / frames)
            loop.advance(1 / frames)
        return loop.inputs.pitch, loop.current
    p60, core60 = simulate(60)
    p144, core144 = simulate(144)
    assert p60 == pytest.approx(p144)
    assert p60 == pytest.approx(.18)
    assert core60.survivor.yaw == pytest.approx(core144.survivor.yaw)


def test_pitch_does_not_change_core_movement_or_collision():
    level, pitched = FixedStepLoop(GameCore()), FixedStepLoop(GameCore())
    for i in range(120):
        for loop in (level, pitched):
            loop.inputs.key('w', True)
            loop.inputs.key('d', True)
            loop.inputs.mouse(2)
        pitched.inputs.mouse(0, -100 if i % 2 else 100)
        level.advance(level.dt)
        pitched.advance(pitched.dt)
        assert pitched.current == level.current


def test_pitch_preserved_on_pause_focus_resume_but_reset_clears_it(display):
    loop = FixedStepLoop(GameCore())
    session = PlaySession(loop)
    session.focused = True
    session.events([event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    session.events([event(pygame.MOUSEMOTION, rel=(0, -40))])
    loop.advance(loop.dt)
    expected = loop.inputs.pitch
    assert expected == pytest.approx(.12)
    session.events([event(pygame.MOUSEMOTION, rel=(100, 500)),
                    event(pygame.WINDOWFOCUSLOST),
                    event(pygame.MOUSEMOTION, rel=(900, 900))])
    assert (loop.inputs.mouse_dx, loop.inputs.mouse_dy) == (0, 0)
    assert loop.inputs.pitch == expected
    loop.advance(20)
    session.events([event(pygame.MOUSEMOTION, rel=(900, 900))])
    session.events([event(pygame.WINDOWFOCUSGAINED)])
    assert loop.paused
    session.events([event(pygame.KEYDOWN, key=pygame.K_ESCAPE),
                    event(pygame.MOUSEMOTION, rel=(900, 900))])
    loop.advance(loop.dt)
    assert loop.inputs.pitch == expected
    assert loop.current.survivor.yaw == 0
    session.events([event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    assert loop.paused
    assert loop.inputs.pitch == expected
    session.events([event(pygame.MOUSEMOTION, rel=(900, 900))])
    session.events([event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    loop.advance(loop.dt)
    assert loop.inputs.pitch == expected
    session.events([event(pygame.KEYDOWN, key=pygame.K_r),
                    event(pygame.MOUSEMOTION, rel=(900, 900))])
    assert loop.inputs.pitch == 0
    loop.advance(loop.dt)
    assert loop.inputs.pitch == 0
    assert loop.current.survivor.yaw == 0


def test_pitch_projection_fully_repaints_extremes_and_preserves_depth(display, tmp_path):
    texture = pygame.Surface((16, 16))
    wall_color = (180, 160, 140)
    texture.fill(wall_color)
    path = tmp_path / 'wall.png'
    pygame.image.save(texture, path)
    renderer = RaycastingRenderer(path)
    core = GameCore(WorldMap(('.' * 24,) * 24, ((12., 12.),)))
    initial = core.snapshot()
    zero = pygame.image.tobytes(renderer.render(initial), 'RGB')
    depths = renderer.depth
    tops = {}
    for pitch in (25*pi/180, -25*pi/180, 0):
        renderer.frame.fill((253, 0, 253))  # detect any stale/unpainted pixels
        frame = renderer.render(initial, pitch=pitch)
        assert all(frame.get_at((x, y))[:3] in
                   ((29, 33, 40), (48, 44, 40), wall_color)
                   for x in range(320) for y in range(180))
        assert renderer.depth == depths
        assert core.snapshot() == initial
        tops[pitch] = next(y for y in range(180)
                           if frame.get_at((160, y))[:3] == wall_color)
    assert tops[25*pi/180] > tops[0] > tops[-25*pi/180]
    assert pygame.image.tobytes(renderer.frame, 'RGB') == zero


@pytest.mark.parametrize('alpha', [0, .25, .5, .75, .999])
def test_display_pitch_interpolates_with_same_alpha_as_yaw(alpha):
    loop = FixedStepLoop(GameCore())
    loop.inputs.mouse(40, -40)
    loop.advance(loop.dt)
    loop.advance(alpha * loop.dt)
    assert loop.previous_pitch == 0
    assert loop.current_pitch == pytest.approx(.12)
    assert loop.display_pitch() == pytest.approx(.12 * alpha)
    assert loop.display_snapshot().survivor.yaw == pytest.approx(.12 * alpha)
    assert loop.inputs.pitch == pytest.approx(.12)
    # Reading display values must never consume mouse or mutate the simulation.
    before = loop.current
    loop.inputs.mouse(0, -20)
    assert loop.display_pitch() == pytest.approx(.12 * alpha)
    assert loop.current == before
    assert loop.inputs.mouse_dy == -20


def test_pitch_interpolation_settles_next_tick_without_smoothing_tail_or_delta_reuse():
    loop = FixedStepLoop(GameCore())
    loop.inputs.mouse(0, -40)
    loop.advance(loop.dt * 1.5)
    assert loop.display_pitch() == pytest.approx(.06)
    loop.advance(loop.dt * .5)
    assert loop.previous_pitch == loop.current_pitch == pytest.approx(.12)
    assert loop.display_pitch() == pytest.approx(.12)
    loop.advance(loop.dt * 4)
    assert loop.display_pitch() == pytest.approx(.12)
    assert loop.inputs.mouse_dy == 0


def test_interpolated_pitch_stays_in_bounds_across_both_limits():
    loop = FixedStepLoop(GameCore())
    loop.inputs.mouse(0, -10000)
    loop.advance(loop.dt)
    assert loop.current_pitch == loop.inputs.pitch_limit
    loop.inputs.mouse(0, 20000)
    loop.advance(loop.dt)
    assert loop.previous_pitch == loop.inputs.pitch_limit
    assert loop.current_pitch == -loop.inputs.pitch_limit
    for alpha in (0, .1, .5, .9, .999):
        loop.accumulator = alpha * loop.dt
        assert -loop.inputs.pitch_limit <= loop.display_pitch() <= loop.inputs.pitch_limit
    loop.accumulator = loop.dt / 2
    assert loop.display_pitch() == pytest.approx(0)


def test_pitch_interpolation_synchronizes_on_pause_focus_resume_and_reset(display):
    loop = FixedStepLoop(GameCore())
    session = PlaySession(loop)
    session.focused = True
    session.events([event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    session.events([event(pygame.MOUSEMOTION, rel=(0, -40))])
    loop.advance(loop.dt * 1.5)
    assert loop.display_pitch() == pytest.approx(.06)
    session.events([event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    expected = loop.inputs.pitch
    assert expected == pytest.approx(.12)
    assert loop.previous_pitch == loop.current_pitch == loop.display_pitch() == expected
    session.events([event(pygame.MOUSEMOTION, rel=(900, 900)),
                    event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    assert loop.previous_pitch == loop.current_pitch == loop.display_pitch() == expected
    loop.advance(loop.dt / 2)
    assert loop.display_pitch() == expected
    # Lose focus in the middle of a new interpolated movement.
    session.events([event(pygame.MOUSEMOTION, rel=(0, -20))])
    loop.advance(loop.dt)
    session.events([event(pygame.WINDOWFOCUSLOST),
                    event(pygame.MOUSEMOTION, rel=(900, 900))])
    expected = loop.inputs.pitch
    assert expected == pytest.approx(.18)
    assert loop.previous_pitch == loop.current_pitch == loop.display_pitch() == expected
    session.events([event(pygame.WINDOWFOCUSGAINED)])
    assert loop.paused
    assert loop.display_pitch() == expected
    session.events([event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    loop.advance(loop.dt / 2)
    assert loop.display_pitch() == expected
    session.events([event(pygame.KEYDOWN, key=pygame.K_r)])
    assert loop.previous_pitch == loop.current_pitch == loop.display_pitch() == 0
    assert loop.inputs.pitch == 0
    loop.advance(loop.dt / 2)
    assert loop.display_pitch() == 0
