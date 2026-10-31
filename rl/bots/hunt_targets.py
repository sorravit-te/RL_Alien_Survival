"""Fixed Survivors for testing a hunter, independent of how the training bots play."""

from math import floor, pi

from game.state import SurvivorCommand

FLIP_TICKS = 40
LONGEST_LOOK = 8
HEADINGS = ((1, 0), (0, 1), (-1, 0), (0, -1))  # yaw 0, 90, 180, 270 degrees


class Pacer:
    """Runs back and forth along the longest straight line from its spawn, and never leaves."""

    label = "noisy_pacer"

    def __init__(self, world):
        self.world = world

    def reset(self, rng):
        self.ticks = 0
        self.aim = None

    def _longest_heading(self, x, y):
        lengths = []
        for index, (dx, dy) in enumerate(HEADINGS):
            length, cx, cy = 0, floor(x), floor(y)
            while length < LONGEST_LOOK and not self.world.blocked(cx + dx, cy + dy):
                cx, cy, length = cx + dx, cy + dy, length + 1
            lengths.append((length, -index))
        return -max(lengths)[1] * pi / 2

    def command(self, view):
        if self.aim is None:
            self.aim = self._longest_heading(view.x, view.y)
            return SurvivorCommand(yaw_delta=(self.aim - view.yaw + pi) % (2 * pi) - pi)
        self.ticks += 1
        if self.ticks % FLIP_TICKS == 0:
            return SurvivorCommand(yaw_delta=pi)
        return SurvivorCommand(forward=1.0, running=True)


class StillTarget:
    """Stands silently where it spawned; only patrol can ever find it."""

    label = "silent_still"

    def reset(self, rng):
        pass

    def command(self, view):
        return SurvivorCommand()
