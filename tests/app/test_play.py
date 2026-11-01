"""Raycasting geometry, input, scheduler and window-event regressions (dummy SDL)."""

import os
from dataclasses import replace
from math import pi, tan
from random import Random

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import pygame
import pytest

from app.input import InputState
from app.loop import FixedStepLoop, alien_starts_far, round_start
from app.play import PlaySession
from game.core import GameCore
from game.map import WorldMap, scenario_map
from game.ship_deck import SHIP_SCENARIO
from game.state import CoreConfig, SurvivorCommand, SurvivorState
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
    loop = FixedStepLoop(GameCore(config=CoreConfig(alien_enabled=False)), inputs)
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
        loop = FixedStepLoop(GameCore(config=CoreConfig(alien_enabled=False)))
        loop.inputs.key('w', True)
        loop.inputs.key('right', True)
        for elapsed in deltas:
            loop.advance(elapsed)
            loop.display_snapshot()
        return loop.current
    assert run([1 / 60] * 120) == run([1 / 144] * 288) == run([.1] * 20)
    assert run([1 / 60] * 120).tick == 60


def test_slow_frame_has_bounded_catchup_and_no_endless_backlog():
    loop = FixedStepLoop(GameCore(config=CoreConfig(alien_enabled=False)))
    assert loop.advance(10) == 5
    assert 0 <= loop.accumulator < loop.dt
    assert loop.dropped_seconds > 9.8
    assert loop.advance(0) == 0


def test_pause_reset_clear_inputs_interpolation_and_time():
    loop = FixedStepLoop(GameCore(config=CoreConfig(alien_enabled=False)))
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
    loop = FixedStepLoop(GameCore(config=CoreConfig(alien_enabled=False)))
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
    loop = FixedStepLoop(GameCore(config=CoreConfig(alien_enabled=False)))
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
    loop = FixedStepLoop(GameCore(config=CoreConfig(alien_enabled=False)))
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
    play.main(["--windowed"])
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
    loop = FixedStepLoop(GameCore(config=CoreConfig(alien_enabled=False)), inputs)
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
        loop = FixedStepLoop(GameCore(config=CoreConfig(alien_enabled=False)))
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
    level, pitched = FixedStepLoop(GameCore(config=CoreConfig(alien_enabled=False))), FixedStepLoop(GameCore(config=CoreConfig(alien_enabled=False)))
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
    loop = FixedStepLoop(GameCore(config=CoreConfig(alien_enabled=False)))
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
    loop = FixedStepLoop(GameCore(config=CoreConfig(alien_enabled=False)))
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
    loop = FixedStepLoop(GameCore(config=CoreConfig(alien_enabled=False)))
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
    loop = FixedStepLoop(GameCore(config=CoreConfig(alien_enabled=False)))
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
    loop = FixedStepLoop(GameCore(config=CoreConfig(alien_enabled=False)))
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



def test_e_input_and_terminal_mouse_release_restart(display, monkeypatch):
    from app.play import hud_labels
    from game.map import MapObject

    world = WorldMap(('......',) * 6, ((2.5, 2.5),),
                     (MapObject('card', 'keycard', 3.5, 2.5),
                      MapObject('exit', 'exit', 2.5, 3.5)))
    loop = FixedStepLoop(GameCore(world))
    session = PlaySession(loop)
    session.focused = True
    grabs, visibility = [], []
    monkeypatch.setattr(pygame.event, 'set_grab', grabs.append)
    monkeypatch.setattr(pygame.mouse, 'set_visible', visibility.append)
    session.events([event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    assert 'Press E - Keycard' in hud_labels(loop)
    assert all('Hold E' not in text and '%' not in text for text in hud_labels(loop))
    session.events([event(pygame.KEYDOWN, key=pygame.K_e),
                    event(pygame.KEYUP, key=pygame.K_e)])
    loop.advance(loop.dt)
    assert loop.current.has_keycard
    loop.inputs.mouse(0, -40)
    loop.inputs.mouse((pi / 2) / loop.inputs.mouse_sensitivity)
    loop.advance(loop.dt)
    # Two fresh presses queued, but the first completes the game and clears the rest.
    for _ in range(2):
        session.events([event(pygame.KEYDOWN, key=pygame.K_e),
                        event(pygame.KEYUP, key=pygame.K_e)])
    loop.advance(loop.dt)
    assert loop.inputs.pending_interactions == 0
    session.after_step()
    assert loop.current.outcome == 'escaped' and loop.paused
    assert grabs[-1] is False and visibility[-1] is True
    assert 'YOU ESCAPED' in hud_labels(loop)[0]
    session.events([event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    assert loop.paused
    session.events([event(pygame.KEYDOWN, key=pygame.K_r)])
    assert not loop.paused
    assert grabs[-1] is True and visibility[-1] is False
    assert loop.current.outcome is None and not loop.current.has_keycard
    assert loop.display_pitch() == 0
    assert loop.inputs.consume(loop.dt) == SurvivorCommand()


def test_the_ship_view_follows_the_fov_option(monkeypatch):
    from app import play

    seen = []
    monkeypatch.setattr(
        play, "ShipRenderer", lambda world, size, fov: seen.append((size, fov)) or "ship"
    )
    assert play.make_renderer(scenario_map(SHIP_SCENARIO), 90) == "ship"
    assert seen == [(play.SHIP_FRAME, 90)]


@pytest.mark.parametrize("fov", ["59", "121"])
def test_a_field_of_view_that_would_look_wrong_is_refused(fov, capsys):
    from app import play

    with pytest.raises(SystemExit) as error:
        play.main(["--scenario", "ship_deck_v1", "--fov", fov])
    assert error.value.code == 2 and "--fov" in capsys.readouterr().err


def test_the_ship_is_drawn_pixel_for_pixel_and_never_stretched_by_windows(monkeypatch):
    from app import play

    assert play.frame_size(scenario_map(SHIP_SCENARIO)) == play.WINDOW
    monkeypatch.delenv("SDL_WINDOWS_DPI_AWARENESS", raising=False)
    play.keep_window_sharp()
    assert os.environ["SDL_WINDOWS_DPI_AWARENESS"] == "permonitorv2"


def test_no_alien_walks_the_map_alone_and_keeps_new_rounds(monkeypatch):
    from app import play

    def stop_at_the_core(world, config):
        made.append(GameCore(world, config))
        raise SystemExit

    made = []
    monkeypatch.setattr(play, "GameCore", stop_at_the_core)
    with pytest.raises(SystemExit):
        play.main(["--scenario", "ship_deck_v1", "--no-alien"])
    loop = FixedStepLoop(made[0])
    assert loop.current.alien is None
    seed = loop.current.seed
    loop.new_round()
    assert loop.current.alien is None and loop.current.seed != seed


def test_no_alien_refuses_an_rl_controller(capsys):
    from app import play

    with pytest.raises(SystemExit) as error:
        play.main(["--controller", "rl", "--no-alien"])
    assert error.value.code == 2 and "--no-alien" in capsys.readouterr().err


class Draws:
    """Hand new_round the seeds a test needs, in order."""

    def __init__(self, *seeds):
        self.seeds = list(seeds)

    def randrange(self, limit):
        return self.seeds.pop(0)


def test_every_new_round_leaves_the_start_just_played():
    loop = FixedStepLoop(GameCore(scenario_map(SHIP_SCENARIO)))
    loop.rounds = Random(7)
    starts = [round_start(loop.current)]
    for _ in range(40):
        loop.new_round()
        starts.append(round_start(loop.current))
        assert starts[-1] != starts[-2] and loop.current.tick == 0
    assert len(set(starts)) >= 30
    assert len({start[2] for start in starts}) == 6  # every keycard room comes up


def test_new_round_compares_the_start_not_where_the_round_got_to():
    core = GameCore(scenario_map(SHIP_SCENARIO))
    played = round_start(core.reset(1))
    twin = next(s for s in range(2, 9999) if round_start(core.reset(s)) == played)
    other = next(
        s
        for s in range(2, 9999)
        if round_start(snapshot := core.reset(s)) != played and alien_starts_far(snapshot)
    )
    core.reset(1)
    loop = FixedStepLoop(core)
    start = loop.current.survivor
    loop.inputs.key("w", True)
    for _ in range(30):
        loop.advance(loop.dt)
    assert loop.current.survivor != start
    loop.rounds = Draws(twin, twin, other)
    loop.new_round()
    assert loop.current.seed == other


@pytest.mark.parametrize(
    "seed, far",
    [
        (4253250415, False),  # 9.5 cells apart, 10.2 to walk: caught in 6 seconds
        (752671448, False),  # 13.6 apart, 17.2 to walk: caught in 6 seconds
        (1303755778, True),  # 14.9 apart, 19.2 to walk
        (3566102720, True),  # 13.0 apart in a line, but a wall makes it 23.8 to walk
    ],
)
def test_the_alien_start_is_judged_by_walking(seed, far):
    assert alien_starts_far(GameCore(scenario_map(SHIP_SCENARIO)).reset(seed)) is far


def test_new_rounds_leave_out_the_closest_ship_starts():
    loop = FixedStepLoop(GameCore(scenario_map(SHIP_SCENARIO)))
    loop.rounds = Random(5)
    pairs = set()
    for _ in range(150):
        loop.new_round()
        assert alien_starts_far(loop.current)
        pairs.add(round_start(loop.current)[:2])
    assert len(pairs) == 12


def test_a_map_of_close_starts_still_leaves_the_start_just_played():
    world = WorldMap(("." * 12,) * 8, ((1.5, 1.5), (1.5, 6.5)), alien_spawn=(10.5, 3.5, pi),
                     patrol_waypoints=((10.5, 3.5),))
    loop = FixedStepLoop(GameCore(world))
    loop.rounds = Random(3)
    for _ in range(20):
        before = round_start(loop.current)
        assert not alien_starts_far(loop.current)
        loop.new_round()
        assert round_start(loop.current) != before


def test_new_round_on_a_map_with_one_start_still_takes_a_new_seed():
    world = WorldMap(("....",) * 4, ((1.5, 1.5),))
    loop = FixedStepLoop(GameCore(world, CoreConfig(alien_enabled=False)))
    before = loop.current.seed
    loop.new_round()
    assert loop.current.seed != before and loop.current.tick == 0


def test_same_round_replays_the_round_new_round_chose():
    loop = FixedStepLoop(GameCore(scenario_map(SHIP_SCENARIO)))
    loop.rounds = Random(11)
    loop.new_round()
    start = loop.current
    loop.inputs.key("w", True)
    for _ in range(30):
        loop.advance(loop.dt)
    assert loop.current.survivor != start.survivor
    loop.reset()
    assert loop.current == start


def test_new_round_key_waits_for_focus_and_ignores_key_repeat(display):
    loop = FixedStepLoop(GameCore(scenario_map(SHIP_SCENARIO)))
    session = PlaySession(loop)
    seed = loop.current.seed
    session.focused = False
    session.events([event(pygame.KEYDOWN, key=pygame.K_n)])
    assert loop.current.seed == seed
    session.focused = True
    session.events([event(pygame.KEYDOWN, key=pygame.K_n, repeat=True)])
    assert loop.current.seed == seed


def test_new_round_key_keeps_a_pause_and_resumes_after_a_finished_round(display):
    from app.play import hud_labels, window_title

    world = WorldMap(("......",) * 6, ((2.5, 2.5), (3.5, 3.5)))
    loop = FixedStepLoop(GameCore(world))
    session = PlaySession(loop)
    session.focused = True
    session.events([event(pygame.KEYDOWN, key=pygame.K_n)])
    assert loop.paused  # the first round can be swapped before it starts
    assert any("N new round" in text and "R same round" in text for text in hud_labels(loop))
    assert pygame.display.get_caption()[0] == window_title(loop.core)
    loop.current = replace(loop.current, outcome="captured")
    assert any("N new round" in text and "R same round" in text for text in hud_labels(loop))
    session.events([event(pygame.KEYDOWN, key=pygame.K_n)])
    assert not loop.paused and loop.current.outcome is None


def test_a_running_round_keeps_the_screen_clear():
    from app.play import hud_labels, showing_menu

    world = WorldMap(("......",) * 6, ((2.5, 2.5),))
    loop = FixedStepLoop(GameCore(world))
    assert hud_labels(loop) == [] and not showing_menu(loop)


def test_the_pause_menu_holds_what_the_screen_no_longer_shows(display):
    from app.play import hud_labels, showing_menu

    world = WorldMap(("......",) * 6, ((2.5, 2.5),))
    loop = FixedStepLoop(GameCore(world))
    session = PlaySession(loop)
    assert showing_menu(loop)
    text = " ".join(hud_labels(loop))
    for wanted in ("Esc", "WASD", "Q quit", "Keycard: not acquired", "Alien disabled"):
        assert wanted in text
    session.focused = True
    session.events([event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    assert not showing_menu(loop)


def test_a_finished_round_shows_the_menu_before_the_session_pauses_it():
    from app.play import showing_menu

    loop = FixedStepLoop(GameCore(WorldMap(("......",) * 6, ((2.5, 2.5),))))
    loop.current = replace(loop.current, outcome="escaped")
    assert not loop.paused and showing_menu(loop)


def test_q_quits_only_from_the_menu(display):
    loop = FixedStepLoop(GameCore(config=CoreConfig(alien_enabled=False)))
    session = PlaySession(loop)
    session.focused = True
    session.events([event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    session.events([event(pygame.KEYDOWN, key=pygame.K_q)])
    assert session.running  # a round in play ignores a stray press
    session.events([event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    session.events([event(pygame.KEYDOWN, key=pygame.K_q, repeat=True)])
    assert session.running
    session.events([event(pygame.KEYDOWN, key=pygame.K_q)])
    assert not session.running


def test_labels_draw_in_the_middle_only_for_a_menu(display):
    from app.play import LINE_HEIGHT, WINDOW, show_labels

    pygame.font.init()
    fonts = pygame.font.Font(None, 24), pygame.font.Font(None, 46)

    def drawn(menu):
        screen = pygame.Surface(WINDOW)
        screen.fill((200, 200, 200))
        show_labels(screen, fonts, ["one line"], menu)
        return screen

    prompt, menu = drawn(False), drawn(True)
    assert prompt.get_at((5, 5))[:3] == (200, 200, 200)  # play is not dimmed
    assert menu.get_at((5, 5))[0] < 100
    middle = WINDOW[1] // 2
    row = pygame.surfarray.array3d(menu)[:, middle - LINE_HEIGHT // 2 : middle + LINE_HEIGHT]
    assert row.max() > 200  # the title sits at the centre of the screen


@pytest.mark.parametrize("windowed", [True, False])
def test_the_game_fills_the_screen_unless_told_to_stay_in_a_window(monkeypatch, windowed):
    from app import play

    seen = []
    monkeypatch.setattr(pygame.display, "set_mode", lambda *args: seen.append(args))
    monkeypatch.delenv("SDL_MOUSE_RELATIVE_SCALING", raising=False)
    play.open_screen(windowed)
    if windowed:
        assert seen == [(play.WINDOW,)]
    else:
        assert seen == [(play.WINDOW, pygame.SCALED | pygame.FULLSCREEN)]
        assert os.environ["SDL_MOUSE_RELATIVE_SCALING"] == "0"


def test_locked_exit_hint_and_occluded_object_no_hint(display):
    from app.play import hud_labels
    from game.map import MapObject

    core = GameCore(WorldMap(("......",) * 6, ((2.5, 2.5),),
                            (MapObject("exit", "exit", 3.5, 2.5),)))
    loop = FixedStepLoop(core)
    assert 'Exit - Keycard required' in hud_labels(loop)
    core = GameCore(WorldMap(("...#..",) * 6, ((2.5, 2.5),),
                            (MapObject("exit", "exit", 4.5, 2.5),)),
                    CoreConfig(interaction_range=3))
    assert not any('Exit -' in label for label in hud_labels(FixedStepLoop(core)))


def test_sprite_wall_occlusion_and_pitch_shift(display, tmp_path):
    from game.map import MapObject

    path = tmp_path / 'plain.png'
    texture = pygame.Surface((16, 16))
    texture.fill((100, 100, 100))
    pygame.image.save(texture, path)
    renderer = RaycastingRenderer(path)
    rows = tuple('....#...' if y == 4 else '........' for y in range(8))
    hidden = GameCore(WorldMap(rows, ((2.5, 4.5),),
                               (MapObject('card', 'keycard', 5.5, 4.5),)),
                      CoreConfig(interaction_range=5)).snapshot()
    empty = replace(hidden, world_map=replace(hidden.world_map, objects=()))
    baseline = pygame.image.tobytes(renderer.render(empty), 'RGB')
    assert pygame.image.tobytes(renderer.render(hidden), 'RGB') == baseline
    assert hidden.interaction_target is None
    # A wide sign straddles a wall edge: only unobstructed columns may change.
    partial = replace(hidden, world_map=replace(hidden.world_map,
                      objects=(MapObject('exit', 'exit', 5.2, 3.7),)))
    base = renderer.render(empty).copy()
    shown = renderer.render(partial).copy()
    changed = [x for x in range(320) if any(
        shown.get_at((x,y)) != base.get_at((x,y)) for y in range(180))]
    assert changed
    assert all(renderer.depth[x] > 2.7 for x in changed)
    # The unblocked projection has more visible columns than the occluded one.
    open_world = replace(partial.world_map, rows=('........',) * 8)
    unblocked = replace(partial, world_map=open_world)
    unblocked_base = renderer.render(replace(unblocked,
                              world_map=replace(open_world, objects=()))).copy()
    unblocked_frame = renderer.render(unblocked).copy()
    full_changed = [x for x in range(320) if any(
        unblocked_frame.get_at((x,y)) != unblocked_base.get_at((x,y)) for y in range(180))]
    assert len(full_changed) > len(changed)
    # Sprite and wall share precisely the same horizon shift.
    positions = []
    for pitch in (0, .1, -.1):
        no_object = renderer.render(replace(unblocked,
                    world_map=replace(open_world, objects=())), pitch=pitch).copy()
        objects = renderer.render(unblocked, pitch=pitch).copy()
        ys = [y for y in range(180) for x in range(320)
              if objects.get_at((x,y)) != no_object.get_at((x,y))]
        positions.append(min(ys))
    offset = round(renderer.focal_length * tan(.1))
    assert positions[1] - positions[0] == offset
    assert positions[2] - positions[0] == -offset



def test_press_release_between_ticks_is_buffered_once_and_repeat_is_ignored():
    inputs = InputState()
    inputs.key('e', True)
    inputs.key('e', True)  # duplicate down even without a repeat flag
    inputs.key('e', True, repeat=True)
    inputs.key('e', False)
    assert inputs.consume(1 / 30).interact
    assert not inputs.consume(1 / 30).interact
    inputs.key('e', True, repeat=True)  # repeat after transition clear must not enqueue
    assert not inputs.consume(1 / 30).interact


def test_collected_sprite_disappears_and_rendering_cannot_change_game(display, tmp_path):
    from game.map import MapObject

    path = tmp_path / 'wall.png'
    pygame.image.save(pygame.Surface((16, 16)), path)
    renderer = RaycastingRenderer(path)
    core = GameCore(WorldMap(('......',) * 6, ((2.5, 2.5),),
                            (MapObject('card', 'keycard', 3.5, 2.5),)))
    before = core.snapshot()
    visible = pygame.image.tobytes(renderer.render(before), 'RGB')
    depth = renderer.depth
    assert core.snapshot() == before
    collected = core.step(SurvivorCommand(interact=True))
    hidden = pygame.image.tobytes(renderer.render(collected, pitch=.1), 'RGB')
    no_objects = replace(collected, world_map=replace(collected.world_map, objects=()))
    assert hidden == pygame.image.tobytes(renderer.render(no_objects, pitch=.1), 'RGB')
    assert depth == renderer.depth
    assert visible != pygame.image.tobytes(renderer.render(collected), 'RGB')
    assert core.snapshot() == collected



@pytest.mark.parametrize('kind', ['keycard', 'exit'])
@pytest.mark.parametrize('interruption', ['pause', 'focus'])
def test_pause_or_focus_discards_queued_e_without_time_before_new_press(display, kind, interruption):
    from game.map import MapObject

    core = GameCore(WorldMap(('......',) * 6, ((2.5, 2.5),),
                            (MapObject('card', 'keycard', 3.5, 2.5),
                             MapObject('exit', 'exit', 2.5, 3.5))))
    if kind == 'exit':
        core.step(SurvivorCommand(interact=True))
        core.step(SurvivorCommand(yaw_delta=pi / 2))
    loop = FixedStepLoop(core)
    session = PlaySession(loop)
    session.focused = True
    session.events([event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    before = core.snapshot()
    session.events([event(pygame.KEYDOWN, key=pygame.K_e)])
    assert loop.inputs.pending_interactions == 1
    if interruption == 'pause':
        session.events([event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    else:
        session.events([event(pygame.WINDOWFOCUSLOST)])
    assert loop.paused and loop.inputs.pending_interactions == 0
    assert loop.advance(100) == 0 and core.snapshot() == before
    session.events([event(pygame.KEYDOWN, key=pygame.K_e)])
    if interruption == 'focus':
        session.events([event(pygame.WINDOWFOCUSGAINED)])
        assert loop.paused
    session.events([event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    session.events([event(pygame.KEYDOWN, key=pygame.K_e, repeat=True)])
    loop.advance(loop.dt)
    assert loop.current.has_keycard == (kind == 'exit')
    assert loop.current.outcome is None and loop.current.events == ()
    session.events([event(pygame.KEYUP, key=pygame.K_e),
                    event(pygame.KEYDOWN, key=pygame.K_e)])
    loop.advance(loop.dt)
    assert [e.kind for e in loop.current.events] == [
        'keycard_collected' if kind == 'keycard' else 'escaped']



@pytest.mark.parametrize('count', [1, 2, 3])
def test_multiple_taps_between_ticks_preserve_one_pulse_per_press(count):
    inputs = InputState()
    for _ in range(count):
        inputs.key('e', True)
        inputs.key('e', False)
    assert [inputs.consume(1 / 30).interact for _ in range(count + 2)] == [True]*count+[False]*2


def test_holding_e_cannot_chain_card_and_exit_during_catchup(display):
    from game.map import MapObject

    world = WorldMap(('......',)*6, ((2.5,2.5),),
                     (MapObject('a','keycard',3.5,2.5), MapObject('z','exit',3.5,2.5)))
    loop = FixedStepLoop(GameCore(world))
    session = PlaySession(loop)
    session.focused = True
    session.events([event(pygame.KEYDOWN,key=pygame.K_ESCAPE)])
    session.events([event(pygame.KEYDOWN,key=pygame.K_e)])
    loop.advance(loop.dt * 3)
    assert loop.current.has_keycard and loop.current.outcome is None
    session.events([event(pygame.KEYDOWN,key=pygame.K_e,repeat=True)])
    loop.advance(loop.dt)
    assert loop.current.outcome is None
    session.events([event(pygame.KEYUP,key=pygame.K_e),event(pygame.KEYDOWN,key=pygame.K_e)])
    loop.advance(loop.dt)
    assert loop.current.outcome == 'escaped'


def test_reset_discards_queued_e_and_no_press_does_not_collect():
    from game.map import MapObject

    loop=FixedStepLoop(GameCore(WorldMap(('......',)*6,((2.5,2.5),),
                                       (MapObject('card','keycard',3.5,2.5),))))
    loop.inputs.key('e',True)
    loop.inputs.key('e',False)
    assert loop.inputs.pending_interactions == 1
    loop.reset()
    assert loop.inputs.pending_interactions == 0
    loop.advance(loop.dt)
    assert not loop.current.has_keycard and not loop.current.events



def test_alien_sprite_full_partial_occlusion_and_pitch(display,tmp_path):
    from game.state import AlienState

    texture=pygame.Surface((16,16))
    texture.fill((100,100,100))
    path=tmp_path/'plain.png'
    pygame.image.save(texture,path)
    renderer=RaycastingRenderer(path)
    rows=tuple('....#...' if y==4 else '........' for y in range(8))
    core=GameCore(WorldMap(rows,((2.5,4.5),)))
    empty=core.snapshot()
    base=renderer.render(empty).copy()
    hidden=replace(empty,alien=AlienState(5.5,4.5))
    assert pygame.image.tobytes(renderer.render(hidden),'RGB')==pygame.image.tobytes(base,'RGB')
    partial=replace(empty,alien=AlienState(5.2,3.7))
    frame=renderer.render(partial).copy()
    changed=[x for x in range(320) if any(frame.get_at((x,y))!=base.get_at((x,y)) for y in range(180))]
    assert changed and all(renderer.depth[x]>2.7 for x in changed)
    open_world=replace(empty.world_map,rows=('........',)*8)
    clear=replace(empty,world_map=open_world,alien=AlienState(5.5,4.5))
    tops=[]
    for pitch in (0,.1,-.1):
        background=renderer.render(replace(clear,alien=None),pitch=pitch).copy()
        alien=renderer.render(clear,pitch=pitch).copy()
        tops.append(min(y for y in range(180) for x in range(320)
                        if alien.get_at((x,y))!=background.get_at((x,y))))
    offset=round(renderer.focal_length*tan(.1))
    assert tops[1]-tops[0]==offset and tops[2]-tops[0]==-offset
    assert core.snapshot()==empty


def test_near_card_occludes_far_alien_and_near_alien_occludes_card(display,tmp_path):
    from game.map import MapObject
    from game.state import AlienState

    path=tmp_path/'plain.png'
    pygame.image.save(pygame.Surface((16,16)),path)
    renderer=RaycastingRenderer(path)
    core=GameCore(WorldMap(('.'*10,)*8,((2.5,4.5),),
                            (MapObject('card','keycard',3.5,4.5),)))
    snapshot=core.snapshot()
    card=renderer.render(snapshot).copy()
    combined=renderer.render(replace(snapshot,alien=AlienState(6.5,4.5))).copy()
    assert all(combined.get_at((160,y))==card.get_at((160,y)) for y in range(75,110))
    near_alien=replace(snapshot,world_map=replace(snapshot.world_map,objects=()),
                       alien=AlienState(3.5,4.5))
    alien=renderer.render(near_alien).copy()
    far_card=replace(near_alien,world_map=replace(snapshot.world_map,
                           objects=(MapObject('card','keycard',6.5,4.5),)))
    combined=renderer.render(far_card).copy()
    assert all(combined.get_at((160,y))==alien.get_at((160,y)) for y in range(85,100))


def test_alien_pause_focus_freezes_everything_and_caught_restart_resets(display,monkeypatch):
    from app.play import hud_labels
    from game.perception import PerceptionSnapshot

    world=WorldMap(('.'*12,)*8,((2.5,2.5),),alien_spawn=(6.5,2.5,pi),
                   patrol_waypoints=((2.5,2.5),))
    loop=FixedStepLoop(GameCore(world,CoreConfig(run_speed=120,alien_search_speed=120)))
    initial=loop.current
    session=PlaySession(loop)
    session.focused=True
    session.events([event(pygame.KEYDOWN,key=pygame.K_ESCAPE)])
    loop.inputs.mouse(0,-20)
    loop.inputs.key('e',True)
    session.events([event(pygame.WINDOWFOCUSLOST)])
    assert loop.advance(100)==0 and loop.current==initial
    assert loop.inputs.pending_interactions==0
    session.events([event(pygame.WINDOWFOCUSGAINED)])
    assert loop.paused
    session.events([event(pygame.KEYDOWN,key=pygame.K_ESCAPE)])
    loop.inputs.key('w',True)
    loop.inputs.key('shift_left',True)
    grabs=[]
    monkeypatch.setattr(pygame.event,'set_grab',grabs.append)
    loop.advance(loop.dt)
    session.after_step()
    assert loop.current.outcome=='captured' and loop.paused and grabs[-1] is False
    assert 'YOU WERE CAUGHT' in hud_labels(loop)[0]
    assert 'Heuristic test controller - NOT RL' in hud_labels(loop)
    frozen=loop.current
    assert loop.advance(100)==0 and loop.current==frozen
    session.events([event(pygame.KEYDOWN,key=pygame.K_r)])
    assert loop.current==initial and not loop.paused
    assert loop.current.perception==PerceptionSnapshot()
    assert loop.current.alien_target is None and loop.current.navigation_path==()
    assert loop.display_pitch()==0 and loop.inputs.consume(loop.dt)==SurvivorCommand()


def test_alien_display_interpolation_is_read_only_and_safe():
    world=WorldMap(('.'*16,)*16,((2.5,2.5),),alien_spawn=(10.5,10.5,0),
                   patrol_waypoints=((12.5,10.5),))
    loop=FixedStepLoop(GameCore(world))
    loop.advance(loop.dt*1.5)
    current=loop.core.snapshot()
    rendered=loop.display_snapshot()
    assert rendered.alien.x==(loop.previous.alien.x+current.alien.x)/2
    assert not world.overlaps(rendered.alien.x,rendered.alien.y,.2)
    assert loop.core.snapshot()==current
