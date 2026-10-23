"""Immutable commands, configuration and world snapshots.

Coordinates: x right, y down; yaw 0 faces +x, positive yaw turns toward +y.
Commands are sampled once per physics tick. yaw_delta is radians per tick,
not angular velocity; input adapters must accumulate mouse motion accordingly.
"""

from dataclasses import dataclass
from math import isfinite, pi

from game.map import MapObject, WorldMap


@dataclass(frozen=True)
class CoreConfig:
    tick_hz: int = 30
    walk_speed: float = 2.5
    run_speed: float = 4.0
    radius: float = 0.2
    interaction_range: float = 1.25
    interaction_half_angle: float = pi / 6

    def __post_init__(self):
        if self.tick_hz != 30:
            raise ValueError("The core contract requires 30 Hz")
        if not all(isfinite(v) and v > 0 for v in (self.walk_speed, self.run_speed, self.radius)):
            raise ValueError("Speeds and radius must be finite and positive")
        if not isfinite(self.interaction_range) or self.interaction_range <= 0:
            raise ValueError("Interaction range must be finite and positive")
        if not isfinite(self.interaction_half_angle) or not 0 < self.interaction_half_angle <= pi:
            raise ValueError("Interaction half-angle must be in (0, pi]")


@dataclass(frozen=True)
class SurvivorCommand:
    forward: float = 0.0
    strafe: float = 0.0  # positive = right
    yaw_delta: float = 0.0
    running: bool = False
    interact: bool = False  # one new interaction press for this tick, never a held state

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
class GameEvent:
    kind: str
    object_id: str
    tick: int


@dataclass(frozen=True)
class WorldSnapshot:
    tick: int
    seed: int | None
    survivor: SurvivorState
    world_map: "WorldMap"
    config: CoreConfig
    has_keycard: bool = False
    interaction_target: MapObject | None = None
    outcome: str | None = None
    events: tuple[GameEvent, ...] = ()
