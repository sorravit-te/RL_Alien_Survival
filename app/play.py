"""First-person Human Survivor vs explicit heuristic or frozen sampled RL Alien."""

import argparse
import json
import os
from math import degrees
from pathlib import Path
from statistics import mean
from time import perf_counter

import pygame

from app.audio import ShipAudio, Speaker
from app.loop import FixedStepLoop
from app.recorder import GameplayRecorder, validate_record_directory
from game.core import GameCore
from game.map import DEFAULT_SCENARIO, SCENARIOS, scenario_map
from game.ship_deck import ROWS as SHIP_ROWS
from game.state import CoreConfig
from renderers.raycasting import RaycastingRenderer
from renderers.ship_renderer import FIELD_OF_VIEW, ShipRenderer

WINDOW = (1067, 600)  # mist and grain hide the softness; a chase needs the frame rate
SHIP_FRAME = WINDOW  # drawn pixel for pixel; only the graphics card stretches it to the screen
MIN_FOV, MAX_FOV = 60, 120  # wider stretches walls at the screen's edges past looking right
MENU_VEIL = 165  # opacity of 255; a paused frame stays visible behind the text, at about a third
TEXT_COLOR, TEXT_BACKING = (240, 240, 240), (15, 18, 23)
TEXT_SIZE, TITLE_SIZE = 24, 46
LINE_HEIGHT = 30
PROMPT_FROM_BOTTOM = 96  # low on the floor, where it hides nothing the player is watching
KEYS = {
    pygame.K_w: "w",
    pygame.K_s: "s",
    pygame.K_a: "a",
    pygame.K_d: "d",
    pygame.K_LEFT: "left",
    pygame.K_RIGHT: "right",
    pygame.K_LSHIFT: "shift_left",
    pygame.K_RSHIFT: "shift_right",
    pygame.K_e: "e",
}


def window_title(core):
    return "Alien Survival - " + getattr(core, "controller_label", "Heuristic Test")


def keep_window_sharp():
    """Windows otherwise blurs the game by stretching it to the desktop's text size."""
    os.environ.setdefault("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2")


def open_screen(windowed):
    """Fill the desktop by letting the graphics card stretch the frame, which costs almost
    nothing; drawing every desktop pixel in the shader would cost two and a half times more."""
    if windowed:
        return pygame.display.set_mode(WINDOW)
    os.environ.setdefault("SDL_RENDER_SCALE_QUALITY", "linear")  # smooth, not blocky
    os.environ.setdefault("SDL_MOUSE_RELATIVE_SCALING", "0")  # the stretch must not slow the mouse
    return pygame.display.set_mode(WINDOW, pygame.SCALED | pygame.FULLSCREEN)


class PlaySession:
    """Window events are isolated here so focus/pause/reset can be regression-tested."""

    def __init__(self, loop):
        self.loop = loop
        self.running = True
        self.focused = bool(pygame.key.get_focused())
        self.set_paused(True, "initial")
        self.sync_mouse()

    def notify(self, event, **data):
        recorder = getattr(self.loop.core, "recorder", None)
        if recorder is not None:
            recorder.lifecycle(event, self.loop.core, **data)

    def set_paused(self, paused, reason):
        self.loop.set_paused(paused)
        self.notify(
            "pause" if paused else "resume",
            reason=reason,
            focused=self.focused,
            pitch=self.loop.current_pitch,
        )

    def sync_mouse(self):
        grabbed = not self.loop.paused and self.focused and not self.loop.current.outcome
        pygame.event.set_grab(grabbed)
        pygame.mouse.set_visible(not grabbed)
        pygame.mouse.get_rel()  # discard warp/relative deltas on every transition
        pygame.event.clear(pygame.MOUSEMOTION)

    def events(self, events):
        transitioned = False
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
                self.notify("window_close")
            elif event.type == pygame.WINDOWFOCUSLOST:
                self.focused = False
                self.notify("focus_lost")
                self.set_paused(True, "focus_lost")
                self.sync_mouse()
                transitioned = True
            elif event.type == pygame.WINDOWFOCUSGAINED:
                self.focused = True  # explicit Esc is still required
                self.notify("focus_gained")
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                if (
                    self.focused
                    and not self.loop.current.outcome
                    and not transitioned
                    and not getattr(event, "repeat", False)
                ):
                    self.set_paused(not self.loop.paused, "escape")
                    self.sync_mouse()
                    transitioned = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                # Full screen has no close button; a running round ignores a stray press.
                if self.focused and self.loop.paused and not getattr(event, "repeat", False):
                    self.running = False
                    self.notify("window_close")
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_r, pygame.K_n):
                if self.focused and not transitioned and not getattr(event, "repeat", False):
                    was_terminal = self.loop.current.outcome is not None
                    if event.key == pygame.K_r:
                        self.loop.reset()
                    else:
                        self.loop.new_round()
                        pygame.display.set_caption(window_title(self.loop.core))
                    if was_terminal:
                        self.set_paused(False, "terminal_reset")
                    self.sync_mouse()
                    transitioned = True
            elif (
                not transitioned
                and not self.loop.paused
                and self.focused
                and not self.loop.current.outcome
            ):
                if event.type in (pygame.KEYDOWN, pygame.KEYUP) and event.key in KEYS:
                    self.loop.inputs.key(
                        KEYS[event.key],
                        event.type == pygame.KEYDOWN,
                        repeat=getattr(event, "repeat", False),
                    )
                elif event.type == pygame.MOUSEMOTION:
                    self.loop.inputs.mouse(*event.rel)
        return transitioned

    def after_step(self):
        if self.loop.current.outcome and not self.loop.paused:
            self.set_paused(True, "terminal")
            self.sync_mouse()


def alien_debug_labels(snapshot):
    """Opt-in privileged diagnostics from the current Core tick, never policy input."""
    if snapshot.alien is None:
        return ["DEBUG Alien disabled"]
    heard = snapshot.heard
    sensor_time = (
        f"sample tick start {snapshot.tick - 1}" if snapshot.tick else "initial; not sampled"
    )
    return [
        f"DEBUG truth | Alien yaw {degrees(snapshot.alien.yaw) % 360:.1f} deg "
        "(0=E 90=S 180=W 270=N)",
        f"DEBUG sensor | confirmed-visible={snapshot.perception.visible} "
        f"({sensor_time}; BEFORE movement)",
        f"DEBUG heard | {heard.point} checked={heard.checked} count={heard.revision} "
        f"sweep={snapshot.noise_sweep.phase}",
    ]


def interaction_prompt(snapshot):
    target = snapshot.interaction_target
    if target is None:
        return []
    if target.object_id in snapshot.ruled_out_exits:
        return ["Exit - Offline"]
    if target.kind == "exit" and not snapshot.has_keycard:
        return ["Exit - Keycard required"]
    return [f"Press E - {'Keycard' if target.kind == 'keycard' else 'Exit'}"]


def showing_menu(loop):
    return loop.paused or loop.current.outcome is not None


def hud_labels(loop, *, debug_alien=False):
    """The screen stays clear in play: only what can be done right now, or the pause menu."""
    snapshot = loop.current
    controller = getattr(loop.core, "controller_label", "Heuristic test controller - NOT RL")
    debug = alien_debug_labels(snapshot) if debug_alien else []
    if snapshot.outcome:
        result = "YOU ESCAPED" if snapshot.outcome == "escaped" else "YOU WERE CAUGHT"
        return [result, "N new round | R same round | Q quit", controller] + debug
    if not loop.paused:
        return interaction_prompt(snapshot) + debug
    return [
        "PAUSED",
        "Esc play | N new round | R same round | Q quit",
        "WASD move | Shift run | E interact | Mouse look | Arrows turn",
        "Keycard: acquired" if snapshot.has_keycard else "Keycard: not acquired",
        controller if snapshot.alien is not None else "Alien disabled (isolated fixture)",
        *interaction_prompt(snapshot),
        *debug,
    ]


def show_labels(screen, fonts, labels, menu):
    """A menu sits in the middle over a dimmed frame; a prompt sits low and in the way of nothing."""
    width, height = screen.get_size()
    if menu:
        veil = pygame.Surface((width, height))
        veil.set_alpha(MENU_VEIL)
        screen.blit(veil, (0, 0))
    top = (height - len(labels) * LINE_HEIGHT) // 2 if menu else height - PROMPT_FROM_BOTTOM
    for line, label in enumerate(labels):
        title = menu and line == 0
        font = fonts[1] if title else fonts[0]
        text = font.render(label, True, TEXT_COLOR, None if menu else TEXT_BACKING)
        y = top + line * LINE_HEIGHT + (TITLE_SIZE // 2 if menu and line else 0)
        screen.blit(text, ((width - text.get_width()) // 2, y))


def frame_size(world):
    return SHIP_FRAME if world.rows == SHIP_ROWS else (320, 180)


def make_renderer(world, fov=FIELD_OF_VIEW):
    """The ship deck has its own textures and light; test scenes keep the plain renderer."""
    if world.rows == SHIP_ROWS:
        return ShipRenderer(world, size=SHIP_FRAME, fov=fov)
    return RaycastingRenderer()


def percentile(values, fraction):
    return sorted(values)[min(len(values) - 1, int((len(values) - 1) * fraction))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("heuristic", "rl"), default="heuristic")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--record-dir", type=Path, help="New local experiments folder; RL only")
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--scenario-seed", type=int, default=0)
    parser.add_argument("--scenario", choices=SCENARIOS, help="Heuristic only; RL uses its run")
    parser.add_argument("--no-alien", action="store_true", help="Walk the map alone; heuristic only")
    parser.add_argument("--windowed", action="store_true", help="Stay in a window, not full screen")
    parser.add_argument(
        "--fov", type=int, default=FIELD_OF_VIEW, help="Degrees across the screen, on the ship"
    )
    parser.add_argument(
        "--debug-alien", action="store_true", help="Privileged yaw/sensor HUD; off by default"
    )
    parser.add_argument("--fps-limit", type=int, default=60, help="0 = uncapped")
    parser.add_argument(
        "--benchmark-seconds",
        type=float,
        default=0,
        help="Auto-close after this duration; print metrics after 1s warmup",
    )
    args = parser.parse_args(argv)
    if args.fps_limit < 0 or args.benchmark_seconds < 0:
        parser.error("Limits must be nonnegative")
    if not MIN_FOV <= args.fov <= MAX_FOV:
        parser.error(f"--fov must be between {MIN_FOV} and {MAX_FOV}")
    if args.record_dir is not None:
        if args.controller != "rl":
            parser.error("--record-dir requires --controller rl")
        try:
            args.record_dir = validate_record_directory(args.record_dir)
        except RuntimeError as exc:
            parser.error(str(exc))
    if args.controller == "rl":
        if args.no_alien:
            parser.error("--no-alien has no RL Alien to run")
        if args.scenario is not None:
            parser.error("--scenario is fixed by the RL run")
        if args.run_dir is None or args.checkpoint is None or args.sampling_seed is None:
            parser.error("RL requires --run-dir, --checkpoint and --sampling-seed")
        try:
            from app.rl_controller import load_human_rl

            core = load_human_rl(
                args.run_dir, args.checkpoint, args.sampling_seed, args.scenario_seed
            )
        except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
            parser.exit(2, f"Cannot start RL controller: {exc}\n")
    else:
        if any(value is not None for value in (args.run_dir, args.checkpoint, args.sampling_seed)):
            parser.error("RL checkpoint options require --controller rl")
        config = CoreConfig(alien_enabled=not args.no_alien)
        core = GameCore(scenario_map(args.scenario or DEFAULT_SCENARIO), config)
        core.reset(args.scenario_seed)
    recorder = None
    completed, end_reason = False, "exception"
    try:
        loop = FixedStepLoop(core)
        if args.record_dir is not None:
            recorder = GameplayRecorder(
                args.record_dir,
                core,
                settings={
                    "tick_hz": core.config.tick_hz,
                    "decision_ticks": 15,
                    "max_catchup_ticks": loop.max_steps,
                    "max_frame_seconds": loop.max_frame_time,
                    "fps_limit": args.fps_limit,
                    "framebuffer": list(frame_size(core.snapshot().world_map)),
                    "window": list(WINDOW),
                    "fullscreen": not args.windowed,
                    "fov_degrees": args.fov,
                    "mouse_sensitivity": loop.inputs.mouse_sensitivity,
                    "vertical_sensitivity": loop.inputs.vertical_sensitivity,
                    "turn_speed": loop.inputs.turn_speed,
                    "pitch_limit": loop.inputs.pitch_limit,
                    "debug_alien": args.debug_alien,
                },
            )
            core.recorder = recorder
            print(f"Recording game to {recorder.directory}")
        keep_window_sharp()
        pygame.display.init()
        pygame.font.init()
        screen = open_screen(args.windowed)
        pygame.display.set_caption(window_title(core))
        renderer = make_renderer(core.snapshot().world_map, args.fov)
        smooth = isinstance(renderer, ShipRenderer)
        fit = pygame.transform.smoothscale if smooth else pygame.transform.scale
        fonts = pygame.font.Font(None, TEXT_SIZE), pygame.font.Font(None, TITLE_SIZE)
        session = PlaySession(loop)
        audio = ShipAudio(Speaker())
        loop.on_tick = audio.hear
        audio.start()
        clock = pygame.time.Clock()
        start = previous = perf_counter()
        frames, work_times = [], []
        previous_measured = None
        while session.running:
            now = perf_counter()
            elapsed, previous = now - previous, now
            transitioned = session.events(pygame.event.get())
            if not session.running:
                end_reason = "window_close"
                break
            # Do not apply pre-pause/reset wall time to a newly resumed simulation.
            if not transitioned:
                loop.advance(elapsed)
            session.after_step()
            audio.follow(loop.paused, loop.current)
            frame = renderer.render(loop.display_snapshot(), pitch=loop.display_pitch())
            if frame.get_size() == WINDOW:
                screen.blit(frame, (0, 0))
            else:
                fit(frame, WINDOW, screen)
            labels = hud_labels(loop, debug_alien=args.debug_alien)
            menu = showing_menu(loop)
            if recorder is not None and menu:
                labels.append(f"Recording: {recorder.directory.name}")
            show_labels(screen, fonts, labels, menu)
            pygame.display.flip()
            work_end = perf_counter()
            if args.benchmark_seconds and now - start >= 1:
                work_times.append((work_end - now) * 1000)
                if previous_measured is not None:
                    frames.append((now - previous_measured) * 1000)
                previous_measured = now
            if args.benchmark_seconds and work_end - start >= args.benchmark_seconds:
                end_reason = "benchmark_end"
                break
            clock.tick(args.fps_limit)
        completed = True
        if frames:
            print(
                json.dumps(
                    {
                        "framebuffer": list(frame_size(core.snapshot().world_map)),
                        "window": list(WINDOW),
                        "fullscreen": not args.windowed,
                        "screen": list(pygame.display.get_window_size()),
                        "fov_degrees": args.fov,
                        "display_driver": pygame.display.get_driver(),
                        "fps_limit": args.fps_limit,
                        "warmup_seconds": 1,
                        "measured_frames": len(frames),
                        "fps": 1000 / mean(frames),
                        "frame_ms_mean": mean(frames),
                        "frame_ms_p95": percentile(frames, 0.95),
                        "work_ms_mean": mean(work_times),
                        "work_ms_p95": percentile(work_times, 0.95),
                        "dropped_simulation_seconds": loop.dropped_seconds,
                        "simulation_tick": loop.current.tick,
                        "paused_at_exit": loop.paused,
                    },
                    indent=2,
                )
            )
    except (OSError, ValueError, RuntimeError, pygame.error) as exc:
        parser.exit(2, f"Game controller stopped: {exc}\n")
    finally:
        try:
            if recorder is not None:
                recorder.close(completed=completed, reason=end_reason)
        except RuntimeError as exc:
            parser.exit(2, f"Game recording stopped: {exc}\n")
        finally:
            try:
                if pygame.display.get_init():
                    pygame.event.set_grab(False)
                    pygame.mouse.set_visible(True)
            finally:
                pygame.quit()


if __name__ == "__main__":
    main()
