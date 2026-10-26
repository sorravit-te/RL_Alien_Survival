"""First-person Human Survivor vs explicit heuristic or frozen sampled RL Alien."""

import argparse
import json
from math import degrees
from pathlib import Path
from statistics import mean
from time import perf_counter

import pygame

from app.loop import FixedStepLoop
from app.recorder import GameplayRecorder, validate_record_directory
from game.core import GameCore
from renderers.raycasting import RaycastingRenderer

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
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.focused and not transitioned and not getattr(event, "repeat", False):
                    was_terminal = self.loop.current.outcome is not None
                    self.loop.reset()
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
    sensor_time = (
        f"sample tick start {snapshot.tick - 1}" if snapshot.tick else "initial; not sampled"
    )
    return [
        f"DEBUG truth | Alien yaw {degrees(snapshot.alien.yaw) % 360:.1f} deg "
        "(0=E 90=S 180=W 270=N)",
        f"DEBUG sensor | confirmed-visible={snapshot.perception.visible} "
        f"({sensor_time}; BEFORE movement)",
    ]


def hud_labels(loop, *, debug_alien=False):
    snapshot = loop.current
    controller = getattr(loop.core, "controller_label", "Heuristic test controller - NOT RL")
    debug = alien_debug_labels(snapshot) if debug_alien else []
    if snapshot.outcome:
        result = "YOU ESCAPED" if snapshot.outcome == "escaped" else "YOU WERE CAUGHT"
        return [f"{result} - R to restart", "Mouse released", controller] + debug
    status = "PAUSED - Esc to play" if loop.paused else "WASD move | Shift run | Press E interact"
    labels = [
        status,
        "Mouse look | Arrows turn | Esc pause | R reset",
        "Keycard: acquired" if snapshot.has_keycard else "Keycard: not acquired",
        controller if snapshot.alien is not None else "Alien disabled (isolated fixture)",
    ]
    target = snapshot.interaction_target
    if target is not None:
        if target.kind == "exit" and not snapshot.has_keycard:
            labels.append("Exit - Keycard required")
        else:
            name = "Keycard" if target.kind == "keycard" else "Exit"
            labels.append(f"Press E - {name}")
    return labels + debug


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
    if args.record_dir is not None:
        if args.controller != "rl":
            parser.error("--record-dir requires --controller rl")
        try:
            args.record_dir = validate_record_directory(args.record_dir)
        except RuntimeError as exc:
            parser.error(str(exc))
    if args.controller == "rl":
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
        core = GameCore()
        core.reset(args.scenario_seed)
    # Initialize only graphics/font; the game currently has no audio system.
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
                    "framebuffer": [320, 180],
                    "window": [960, 540],
                    "fov_degrees": 75,
                    "mouse_sensitivity": loop.inputs.mouse_sensitivity,
                    "vertical_sensitivity": loop.inputs.vertical_sensitivity,
                    "turn_speed": loop.inputs.turn_speed,
                    "pitch_limit": loop.inputs.pitch_limit,
                    "debug_alien": args.debug_alien,
                },
            )
            core.recorder = recorder
            print(f"Recording game to {recorder.directory}")
        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode((960, 540))
        pygame.display.set_caption(
            "Alien Survival - " + getattr(core, "controller_label", "Heuristic Test")
        )
        renderer = RaycastingRenderer()
        font = pygame.font.Font(None, 22)
        session = PlaySession(loop)
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
            frame = renderer.render(loop.display_snapshot(), pitch=loop.display_pitch())
            pygame.transform.scale(frame, screen.get_size(), screen)
            labels = hud_labels(loop, debug_alien=args.debug_alien)
            if recorder is not None:
                labels.append(f"Recording: {recorder.directory.name}")
            for line, label in enumerate(labels):
                text = font.render(label, True, (240, 240, 240), (15, 18, 23))
                screen.blit(text, (12, 12 + line * 24))
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
                        "framebuffer": [320, 180],
                        "window": [960, 540],
                        "fov_degrees": 75,
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
