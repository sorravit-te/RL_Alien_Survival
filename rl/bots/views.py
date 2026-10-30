"""Filtered Survivor sensors and the shared movement command builder.

Object and threat perception are explicit privileged boundaries: only these functions
read world or Alien truth, and bots receive their frozen views. FOV75 and LoS match the
human horizontal camera rules; simulated pitch has no interaction meaning. Bots use
center visibility (conservative vs a partially visible human sprite).
"""

from dataclasses import dataclass
from math import atan2, cos, hypot, isfinite, radians, sin

from game.interaction import angle_difference, segment_clear
from game.state import SurvivorCommand


@dataclass(frozen=True)
class SurvivorView:
    x: float
    y: float
    yaw: float
    has_keycard: bool
    visible_objects: tuple
    ruled_out: tuple = ()  # exits this Survivor already tried


def move_command(dx, dy, yaw, turn, config, *, running=False):
    """Command for world displacement (dx, dy); Core rotates first, then moves in that frame."""
    yaw += turn
    scale = config.tick_hz / (config.run_speed if running else config.walk_speed)
    return SurvivorCommand(
        forward=max(-1.0, min(1.0, (dx * cos(yaw) + dy * sin(yaw)) * scale)),
        strafe=max(-1.0, min(1.0, (-dx * sin(yaw) + dy * cos(yaw)) * scale)),
        yaw_delta=turn,
        running=running,
    )


def observe_survivor(world, survivor, has_keycard, ruled_out=()):
    visible = []
    for obj in world.objects:
        if (obj.kind == "keycard" and has_keycard) or obj.object_id in ruled_out:
            continue
        dx, dy = obj.x - survivor.x, obj.y - survivor.y
        angle = abs(angle_difference(atan2(dy, dx), survivor.yaw)) if hypot(dx, dy) else 0
        if angle <= radians(75) / 2 and segment_clear(world, survivor.x, survivor.y, obj.x, obj.y):
            visible.append(obj)
    return SurvivorView(
        survivor.x,
        survivor.y,
        survivor.yaw,
        has_keycard,
        tuple(sorted(visible, key=lambda obj: obj.object_id)),
        tuple(ruled_out),
    )


@dataclass(frozen=True)
class SurvivorThreatView:
    mission: SurvivorView
    visible_alien: tuple[float, float] | None = None

    def __post_init__(self):
        if not isinstance(self.mission, SurvivorView):
            raise TypeError("Threat view requires filtered SurvivorView")
        if self.visible_alien is not None:
            point = tuple(self.visible_alien)
            if len(point) != 2 or not all(isfinite(v) for v in point):
                raise ValueError("Visible threat must be a finite point")
            object.__setattr__(self, "visible_alien", point)


def observe_threat(world, survivor, has_keycard, alien, settings, ruled_out=()):
    """Privileged sensing boundary; hidden Alien coordinates never leave this function."""
    point = None
    if alien is not None:
        dx, dy = alien.x - survivor.x, alien.y - survivor.y
        angle = abs(angle_difference(atan2(dy, dx), survivor.yaw))
        if (
            hypot(dx, dy) <= settings.sight_range
            and angle <= radians(settings.fov_degrees) / 2
            and segment_clear(world, survivor.x, survivor.y, alien.x, alien.y)
        ):
            point = (alien.x, alien.y)
    return SurvivorThreatView(observe_survivor(world, survivor, has_keycard, ruled_out), point)
