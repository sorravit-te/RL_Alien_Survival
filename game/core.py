"""Fixed-tick headless movement. No wall clock, window or asset dependencies."""

from math import ceil, cos, floor, hypot, pi, sin, sqrt
from random import Random

from game.map import WorldMap, small_test_map
from game.state import CoreConfig, SurvivorCommand, SurvivorState, WorldSnapshot


class GameCore:
    def __init__(self, world_map: WorldMap | None = None, config: CoreConfig | None = None):
        self.world_map = world_map if world_map is not None else small_test_map()
        self.config = config if config is not None else CoreConfig()
        for x, y in self.world_map.spawn_points:
            if self.world_map.overlaps(x, y, self.config.radius):
                raise ValueError("Spawn does not have character clearance")
        self.reset(0)

    def reset(self, seed: int | None = None) -> WorldSnapshot:
        """Explicit seed replays a spawn; None requests a fresh local RNG seed."""
        self._seed = seed
        x, y = Random(seed).choice(self.world_map.spawn_points)
        self._survivor = SurvivorState(x, y)
        self._tick = 0
        return self.snapshot()

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(self._tick, self._seed, self._survivor, self.world_map, self.config)

    def step(self, command: SurvivorCommand) -> WorldSnapshot:
        """Advance exactly 1/30 second; rotate first, then move in the new frame."""
        yaw = (self._survivor.yaw + command.yaw_delta) % (2 * pi)
        forward, strafe = command.forward, command.strafe
        scale = max(1.0, hypot(forward, strafe))
        speed = self.config.run_speed if command.running else self.config.walk_speed
        distance = speed / self.config.tick_hz / scale
        dx = (cos(yaw) * forward - sin(yaw) * strafe) * distance
        dy = (sin(yaw) * forward + cos(yaw) * strafe) * distance
        x, y = self._survivor.x, self._survivor.y
        # Small axis sweeps allow wall sliding without diagonal corner cutting.
        count = max(1, ceil(hypot(dx, dy) / (self.config.radius / 4)))
        for _ in range(count):
            x = self._sweep_axis(x, y, dx / count, horizontal=True)
            y = self._sweep_axis(y, x, dy / count, horizontal=False)
        self._survivor = SurvivorState(x, y, yaw)
        self._tick += 1
        return self.snapshot()

    def _sweep_axis(self, position, other, delta, *, horizontal):
        """Clamp the entire axis segment against rounded, circle-expanded tiles."""
        radius = self.config.radius
        extent = self.world_map.width if horizontal else self.world_map.height
        target = min(max(position + delta, radius), extent - radius)
        if delta == 0:
            return target
        for across in range(floor(other - radius), floor(other + radius) + 1):
            gap = abs(other - min(max(other, across), across + 1))
            if gap >= radius:
                continue
            padding = sqrt(max(0.0, radius * radius - gap * gap))
            for along in range(
                floor(min(position, target) - radius), floor(max(position, target) + radius) + 1
            ):
                cx, cy = (along, across) if horizontal else (across, along)
                if not self.world_map.blocked(cx, cy):
                    continue
                lower, upper = along - padding, along + 1 + padding
                if delta > 0 and position <= lower + 1e-12 and target > lower:
                    target = lower
                elif delta < 0 and position >= upper - 1e-12 and target < upper:
                    target = upper
        return target
