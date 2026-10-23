"""Immutable commands, configuration and world snapshots.

Coordinates: x right, y down; yaw 0 faces +x, positive yaw turns toward +y.
Commands are sampled once per physics tick. yaw_delta is radians per tick,
not angular velocity; input adapters must accumulate mouse motion accordingly.
"""

from dataclasses import dataclass
from math import isfinite

from game.map import WorldMap


@dataclass(frozen=True)
class CoreConfig:
    tick_hz: int = 30
    walk_speed: float = 2.5
    run_speed: float = 4.0
    radius: float = 0.2

    def __post_init__(self):
        if self.tick_hz != 30:
            raise ValueError("The core contract requires 30 Hz")
        if not all(isfinite(v) and v > 0 for v in (self.walk_speed, self.run_speed, self.radius)):
            raise ValueError("Speeds and radius must be finite and positive")


@dataclass(frozen=True)
class SurvivorCommand:
    forward: float = 0.0
    strafe: float = 0.0  # positive = right
    yaw_delta: float = 0.0
    running: bool = False

    def __post_init__(self):
        if not all(isfinite(v) for v in (self.forward, self.strafe, self.yaw_delta)):
            raise ValueError("Command values must be finite")
        if abs(self.forward) > 1 or abs(self.strafe) > 1:
            raise ValueError("Movement axes must be in [-1, 1]")


@dataclass(frozen=True)
class SurvivorState:
    x: float
    y: float
    yaw: float = 0.0


@dataclass(frozen=True)
class WorldSnapshot:
    tick: int
    seed: int | None
    survivor: SurvivorState
    world_map: "WorldMap"
    config: CoreConfig
