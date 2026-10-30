"""Immutable commands, configuration and world snapshots.

Coordinates: x right, y down; yaw 0 faces +x, positive yaw turns toward +y.
Commands are sampled once per physics tick. yaw_delta is radians per tick,
not angular velocity; input adapters must accumulate mouse motion accordingly.
"""

from dataclasses import dataclass
from math import isfinite, pi

from game.investigation import InvestigationSnapshot
from game.map import MapObject, WorldMap
from game.perception import PerceptionSnapshot
from game.sound import HeardSnapshot


@dataclass(frozen=True)
class CoreConfig:
    tick_hz: int = 30
    walk_speed: float = 2.5
    run_speed: float = 4.0
    radius: float = 0.2
    interaction_range: float = 1.25
    interaction_half_angle: float = pi / 6
    alien_enabled: bool = True
    alien_radius: float = 0.2
    alien_search_speed: float = 2.2
    alien_chase_speed: float = 4.4
    alien_fov: float = 2 * pi / 3
    alien_sight_range: float = 8.0
    alien_near_range: float = 1.0
    sight_confirmation_seconds: float = 0.2
    chase_hold_seconds: float = 2.0
    alien_decision_seconds: float = 0.5
    investigation_turn_speed: float = 4 * pi / 3  # rad/s; a 360-degree sweep takes 1.5s
    target_arrival_distance: float = 0.15
    capture_padding: float = 0.0  # contact distance = both radii + this explicit padding
    minimum_spawn_distance: float = 2.0
    door_reach: float = 1.7  # how far up its passage a body opens a door
    door_lateral: float = 0.9
    door_seconds: float = 0.35
    run_sound_range: float = 8.0  # walking cells a running footfall carries
    sound_step_seconds: float = 0.5
    sound_door_cost: float = 3.0  # cells a shut door takes off that reach
    sound_region: int = 4  # the Alien learns a noise's region, this many cells wide

    def __post_init__(self):
        if self.tick_hz != 30:
            raise ValueError("The core contract requires 30 Hz")
        if not all(isfinite(v) and v > 0 for v in (self.walk_speed, self.run_speed, self.radius)):
            raise ValueError("Speeds and radius must be finite and positive")
        if not isfinite(self.interaction_range) or self.interaction_range <= 0:
            raise ValueError("Interaction range must be finite and positive")
        if not isfinite(self.interaction_half_angle) or not 0 < self.interaction_half_angle <= pi:
            raise ValueError("Interaction half-angle must be in (0, pi]")
        values = (
            self.alien_radius,
            self.alien_search_speed,
            self.alien_chase_speed,
            self.alien_sight_range,
            self.alien_near_range,
            self.sight_confirmation_seconds,
            self.chase_hold_seconds,
            self.alien_decision_seconds,
            self.target_arrival_distance,
            self.investigation_turn_speed,
            self.door_reach,
            self.door_lateral,
            self.door_seconds,
            self.run_sound_range,
            self.sound_step_seconds,
            self.sound_door_cost,
        )
        if not all(isfinite(v) and v > 0 for v in values):
            raise ValueError("Alien distances, speeds and times must be finite and positive")
        if type(self.sound_region) is not int or self.sound_region <= 0:
            raise ValueError("Sound region must be a positive whole number of cells")
        if not isfinite(self.alien_fov) or not 0 < self.alien_fov <= 2 * pi:
            raise ValueError("Alien FOV must be in (0, 2*pi]")
        if self.alien_near_range > self.alien_sight_range:
            raise ValueError("Near sensor range cannot exceed sight range")
        if not all(
            isfinite(v) and v >= 0 for v in (self.capture_padding, self.minimum_spawn_distance)
        ):
            raise ValueError("Capture padding and spawn distance must be finite and nonnegative")


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
class AlienState:
    x: float
    y: float
    yaw: float = 0.0
    speed_mode: str = "search"


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
    alien: AlienState | None = None
    perception: PerceptionSnapshot = PerceptionSnapshot()
    alien_target: tuple[float, float] | None = None
    alien_target_source: str = "patrol"
    navigation_path: tuple[tuple[float, float], ...] = ()
    investigation: InvestigationSnapshot = InvestigationSnapshot()
    alien_target_revision: int | None = None
    ruled_out_exits: tuple[str, ...] = ()  # Survivor knowledge, never Alien input
    door_open: tuple[float, ...] = ()  # 0 shut to 1 open, in world_map.doors order
    closed_doors: tuple[tuple[int, int], ...] = ()
    heard: HeardSnapshot = HeardSnapshot()
    alien_target_heard_revision: int = 0
