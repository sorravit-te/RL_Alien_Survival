"""Wall-looping retreat and the per-episode training mixes of scripted bots."""

from dataclasses import replace
from math import atan2, ceil, hypot

import numpy as np

from ai.navigation import find_path
from game.interaction import segment_clear
from rl.bots.evasive import EvasiveExplorerBot, RunnerEvasiveBot


class WallLoopEvasiveBot(EvasiveExplorerBot):
    """Retreats to the nearest reachable point whose view of the remembered threat is
    blocked, so repeated chases circle a wall block. Same filtered view and speeds."""

    label = "wall_loop_evasive_scripted"
    replan_on_arrival = True  # keep running: a reached cover point is not a hiding place
    cover_route_limit = 8.0  # world units of route to the next cover point

    def __init__(self, rows, config, settings=None, *, doors=()):
        super().__init__(rows, config, settings, doors=doors)
        # Flee from any sighting inside the old safe band, keep fleeing until the threat
        # is at sight range, and remember it twice as long, like a player circling a block.
        s = self.settings
        if s.safe_distance < s.sight_range:
            s = replace(s, enter_distance=s.safe_distance, safe_distance=s.sight_range)
        self.settings = replace(s, memory_seconds=2 * s.memory_seconds)
        self.memory_ticks = ceil(self.settings.memory_seconds * config.tick_hz)

    def _flee_facing(self, position, dx, dy):
        # Watch the remembered threat while retreating (backpedal/strafe at run speed).
        threat = self.last_seen
        if threat is None or hypot(threat[0] - position[0], threat[1] - position[1]) < 1e-9:
            return atan2(dy, dx)
        return atan2(threat[1] - position[1], threat[0] - position[0])

    def _plan(self, position):
        threat = self.last_seen
        distance = hypot(position[0] - threat[0], position[1] - threat[1])
        limit = self.cover_route_limit
        # Farthest-from-threat cover first; a straight line bounds the route from below.
        covers = sorted(
            (-hypot(p[0] - threat[0], p[1] - threat[1]), p[1], p[0], p)
            for p in self.explorer.points
            if not segment_clear(self.explorer.geometry, *threat, *p)  # out of threat's sight
            and hypot(p[0] - threat[0], p[1] - threat[1]) > distance + 0.25
            and hypot(p[0] - position[0], p[1] - position[1]) <= limit
        )
        for _, _, _, point in covers:
            path = find_path(self.explorer.geometry, position, point, self.config.radius)
            length = sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip((position, *path), path))
            if length <= limit and self._safe_path(position, path, threat):
                return point, path
        # No safe cover in reach: keep the parent's farthest safe retreat.
        return super()._plan(position)


class EvasiveMixBot:
    """Per-episode choice between the plain and wall-loop evasive bots.

    The choice uses its own stream derived from bot_seed; the chosen bot is reset with
    the unchanged bot_seed stream, so a plain episode replays EvasiveExplorerBot exactly.
    """

    wall_loop_probability = 0.5

    def __init__(self, rows, config, settings=None, *, doors=()):
        self.plain = EvasiveExplorerBot(rows, config, settings, doors=doors)
        self.wall_loop = WallLoopEvasiveBot(rows, config, settings, doors=doors)
        self.settings = self.plain.settings
        self.active = self.plain

    @property
    def label(self):
        return self.active.label

    def reset(self, rng, *, bot_seed):
        choice = np.random.default_rng([bot_seed, 1]).random()
        self.active = self.wall_loop if choice < self.wall_loop_probability else self.plain
        self.active.reset(rng)

    def command(self, view):
        return self.active.command(view)


class RunnerMixBot(EvasiveMixBot):
    """One episode in three goes to the Runner; the rest replay EvasiveMixBot exactly."""

    runner_probability = 1 / 3

    def __init__(self, rows, config, settings=None, *, doors=()):
        super().__init__(rows, config, settings, doors=doors)
        self.runner = RunnerEvasiveBot(rows, config, settings, doors=doors)

    def reset(self, rng, *, bot_seed):
        if np.random.default_rng([bot_seed, 2]).random() < self.runner_probability:
            self.active = self.runner
            self.active.reset(rng)
        else:
            super().reset(rng, bot_seed=bot_seed)
