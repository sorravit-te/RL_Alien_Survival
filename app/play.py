"""First-person room viewer with a heuristic Alien controller: python -m app.play."""

import argparse
import json
from statistics import mean
from time import perf_counter

import pygame

from app.loop import FixedStepLoop
from game.core import GameCore
from renderers.raycasting import RaycastingRenderer

KEYS = {
    pygame.K_w: 'w', pygame.K_s: 's', pygame.K_a: 'a', pygame.K_d: 'd',
    pygame.K_LEFT: 'left', pygame.K_RIGHT: 'right',
    pygame.K_LSHIFT: 'shift_left', pygame.K_RSHIFT: 'shift_right', pygame.K_e: 'e',
}


class PlaySession:
    """Window events are isolated here so focus/pause/reset can be regression-tested."""

    def __init__(self, loop):
        self.loop = loop
        self.running = True
        self.focused = bool(pygame.key.get_focused())
        self.loop.set_paused(True)
        self.sync_mouse()

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
            elif event.type == pygame.WINDOWFOCUSLOST:
                self.focused = False
                self.loop.set_paused(True)
                self.sync_mouse()
                transitioned = True
            elif event.type == pygame.WINDOWFOCUSGAINED:
                self.focused = True  # explicit Esc is still required
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                if (self.focused and not self.loop.current.outcome and not transitioned
                        and not getattr(event, 'repeat', False)):
                    self.loop.set_paused(not self.loop.paused)
                    self.sync_mouse()
                    transitioned = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.focused and not transitioned and not getattr(event, 'repeat', False):
                    was_terminal = self.loop.current.outcome is not None
                    self.loop.reset()
                    if was_terminal:
                        self.loop.set_paused(False)
                    self.sync_mouse()
                    transitioned = True
            elif (not transitioned and not self.loop.paused and self.focused
                  and not self.loop.current.outcome):
                if event.type in (pygame.KEYDOWN, pygame.KEYUP) and event.key in KEYS:
                    self.loop.inputs.key(KEYS[event.key], event.type == pygame.KEYDOWN,
                                         repeat=getattr(event, 'repeat', False))
                elif event.type == pygame.MOUSEMOTION:
                    self.loop.inputs.mouse(*event.rel)
        return transitioned


    def after_step(self):
        if self.loop.current.outcome and not self.loop.paused:
            self.loop.set_paused(True)
            self.sync_mouse()


def hud_labels(loop):
    snapshot = loop.current
    if snapshot.outcome:
        result="YOU ESCAPED" if snapshot.outcome=="escaped" else "YOU WERE CAUGHT"
        return [f"{result} - R to restart", "Mouse released",
                "Heuristic test controller - NOT RL"]
    status = 'PAUSED - Esc to play' if loop.paused else 'WASD move | Shift run | Press E interact'
    labels = [status, 'Mouse look | Arrows turn | Esc pause | R reset',
              'Keycard: acquired' if snapshot.has_keycard else 'Keycard: not acquired',
              'Heuristic test controller - NOT RL' if snapshot.alien is not None
              else 'Alien disabled (isolated fixture)']
    target = snapshot.interaction_target
    if target is not None:
        if target.kind == "exit" and not snapshot.has_keycard:
            labels.append('Exit - Keycard required')
        else:
            name = "Keycard" if target.kind == "keycard" else "Exit"
            labels.append(f'Press E - {name}')
    return labels


def percentile(values, fraction):
    return sorted(values)[min(len(values) - 1, int((len(values) - 1) * fraction))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=str, default=None, help='Tiled .tmj map file')
    parser.add_argument('--materials', type=str, default=None, help='Local JSON material-ID to texture path mapping')
    parser.add_argument('--fps-limit', type=int, default=60, help='0 = uncapped')
    parser.add_argument('--benchmark-seconds', type=float, default=0,
                        help='Auto-close after this duration; print metrics after 1s warmup')
    args = parser.parse_args(argv)
    if args.fps_limit < 0 or args.benchmark_seconds < 0:
        parser.error('Limits must be nonnegative')
    # Initialize only graphics/font; the game currently has no audio system.
    pygame.display.init()
    pygame.font.init()
    try:
        screen = pygame.display.set_mode((960, 540))
        pygame.display.set_caption('Alien Survival - Heuristic Test')
        material_paths = None
        if args.materials:
            from pathlib import Path
            path = Path(args.materials).resolve()
            materials = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(materials, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in materials.items()):
                raise ValueError("Materials must be an object of material IDs to texture file paths")
            material_paths = {key: (path.parent / value).resolve() for key, value in materials.items()}
        renderer = RaycastingRenderer(material_paths=material_paths)
        font = pygame.font.Font(None, 22)
        if args.map:
            from game.map_loader import load_tiled_map
            world = load_tiled_map(args.map)
            loop = FixedStepLoop(GameCore(world))
        else:
            loop = FixedStepLoop(GameCore())
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
                break
            # Do not apply pre-pause/reset wall time to a newly resumed simulation.
            if not transitioned:
                loop.advance(elapsed)
            session.after_step()
            frame = renderer.render(loop.display_snapshot(), pitch=loop.display_pitch())
            pygame.transform.scale(frame, screen.get_size(), screen)
            for line, label in enumerate(hud_labels(loop)):
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
                break
            clock.tick(args.fps_limit)
        if frames:
            print(json.dumps({
                'framebuffer': [320, 180], 'window': [960, 540], 'fov_degrees': 75,
                'display_driver': pygame.display.get_driver(), 'fps_limit': args.fps_limit,
                'warmup_seconds': 1, 'measured_frames': len(frames),
                'fps': 1000 / mean(frames), 'frame_ms_mean': mean(frames),
                'frame_ms_p95': percentile(frames, .95),
                'work_ms_mean': mean(work_times), 'work_ms_p95': percentile(work_times, .95),
                'dropped_simulation_seconds': loop.dropped_seconds,
                'simulation_tick': loop.current.tick, 'paused_at_exit': loop.paused,
            }, indent=2))
    finally:
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)
        pygame.quit()


if __name__ == '__main__':
    main()
