"""Privileged sensing boundary. Only confirmed snapshots leave this module."""

from dataclasses import dataclass
from math import atan2, ceil, hypot

from game.interaction import angle_difference, segment_clear


@dataclass(frozen=True)
class PerceptionSnapshot:
    visible: bool = False
    confirmation_ticks: int = 0
    last_seen: tuple[float, float] | None = None
    last_seen_age_ticks: int | None = None
    revision: int | None = None  # increments on confirmed samples only, including same position


def update_perception(world, alien, survivor, previous, config):
    dx, dy = survivor.x - alien.x, survivor.y - alien.y
    distance = hypot(dx, dy)
    angle = abs(angle_difference(atan2(dy, dx), alien.yaw)) if distance else 0
    in_sensor = distance <= config.alien_near_range or (
        distance <= config.alien_sight_range and angle <= config.alien_fov / 2
    )
    sensed = in_sensor and segment_clear(world, alien.x, alien.y, survivor.x, survivor.y)
    required = max(1, ceil(config.sight_confirmation_seconds * config.tick_hz - 1e-12))
    count = min(required, previous.confirmation_ticks + 1) if sensed else 0
    visible = count >= required
    return PerceptionSnapshot(
        visible,
        count,
        (survivor.x, survivor.y) if visible else previous.last_seen,
        0
        if visible
        else (
            previous.last_seen_age_ticks + 1 if previous.last_seen_age_ticks is not None else None
        ),
        (previous.revision or 0) + 1 if visible else previous.revision,
    )
