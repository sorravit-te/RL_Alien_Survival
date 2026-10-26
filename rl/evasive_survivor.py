"""LoS-filtered Survivor threat view and a small scripted retreating Explorer.

Only observe_threat receives true Alien state. The bot receives a frozen view,
keeps expiring last-seen memory, and never receives Core/WorldSnapshot/AlienState.
"""

from dataclasses import dataclass
from math import atan2, ceil, cos, hypot, isfinite, pi, radians, sin

from ai.navigation import Navigator, find_path
from game.interaction import segment_clear
from game.state import SurvivorCommand
from rl.survivor_bots import ExplorerBot, SurvivorView, observe_survivor


@dataclass(frozen=True)
class EvasionConfig:
    sight_range: float = 8.0
    fov_degrees: float = 75.0
    memory_seconds: float = 2.0
    enter_distance: float = 4.0
    safe_distance: float = 6.0
    target_hold_seconds: float = 1.0
    threat_clearance: float = 0.8

    def __post_init__(self):
        if not all(isfinite(v) and v > 0 for v in vars(self).values()):
            raise ValueError("Evasion settings must be finite and positive")
        if (
            not self.enter_distance < self.safe_distance <= self.sight_range
            or self.fov_degrees > 180
        ):
            raise ValueError("Evasion requires enter < safe <= sight range and FOV <=180")


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


def observe_threat(world, survivor, has_keycard, alien, settings):
    """Privileged sensing boundary; hidden Alien coordinates never leave this function."""
    point = None
    if alien is not None:
        dx, dy = alien.x - survivor.x, alien.y - survivor.y
        angle = abs((atan2(dy, dx) - survivor.yaw + pi) % (2 * pi) - pi)
        if (
            hypot(dx, dy) <= settings.sight_range
            and angle <= radians(settings.fov_degrees) / 2
            and segment_clear(world, survivor.x, survivor.y, alien.x, alien.y)
        ):
            point = (alien.x, alien.y)
    return SurvivorThreatView(observe_survivor(world, survivor, has_keycard), point)


def point_segment_distance(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    t = (
        max(
            0.0,
            min(
                1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (dx * dx + dy * dy)
            ),
        )
        if dx or dy
        else 0.0
    )
    return hypot(start[0] + t * dx - point[0], start[1] + t * dy - point[1])


class EvasiveExplorerBot:
    label = "evasive_explorer_scripted"

    def __init__(self, rows, config, settings=None):
        self.explorer = ExplorerBot(rows, config)
        self.config = config
        self.settings = settings if settings is not None else EvasionConfig()
        if (
            self.settings.threat_clearance
            <= config.radius + config.alien_radius + config.capture_padding
        ):
            raise ValueError("Threat clearance must exceed physical capture distance")
        self.memory_ticks = ceil(self.settings.memory_seconds * config.tick_hz)
        self.hold_ticks = ceil(self.settings.target_hold_seconds * config.tick_hz)

    def reset(self, rng):
        self.explorer.reset(rng)
        self.last_seen = None
        self.last_seen_age_ticks = None
        self.fleeing = False
        self.tick = 0
        self.flee_started = 0
        self.last_plan = -self.hold_ticks
        self.escape_target = None
        self.navigator = Navigator()
        self.no_safe_route_count = 0
        self.flee_entries = 0
        self.flee_ticks = 0
        self.replans = 0
        self.exit_approach_active = False
        self.exit_standpoint = self.exit_goal_id = None
        self.exit_navigator = Navigator()
        self.exit_retry_tick = 0
        self.exit_approach_attempts = 0
        self.exit_approach_cancellations = 0
        self.exit_approach_steps = 0
        self.exit_motion_audit = None

    def _safe_path(self, start, path, threat):
        if not path:
            return False
        points = (start, *path)
        # Outside the clearance zone a safe detour may initially approach a little.
        # Rejecting every such first step can trap a bot at a harmless corridor corner.
        for a, b in zip(points, points[1:]):
            initial = hypot(a[0] - threat[0], a[1] - threat[1])
            bound = min(self.settings.threat_clearance, initial)
            if point_segment_distance(threat, a, b) < bound - 1e-9:
                return False
            if (
                initial < self.settings.threat_clearance - 1e-9
                and hypot(b[0] - threat[0], b[1] - threat[1]) <= initial + 1e-9
                and a != b
            ):
                return False
        return True

    def _plan(self, position):
        threat = self.last_seen
        distance = hypot(position[0] - threat[0], position[1] - threat[1])
        points = sorted(
            self.explorer.points,
            key=lambda p: (-hypot(p[0] - threat[0], p[1] - threat[1]), p[1], p[0]),
        )
        detour = None
        for point in points:
            if hypot(point[0] - threat[0], point[1] - threat[1]) <= distance + 0.25:
                continue
            path = find_path(self.explorer.geometry, position, point, self.config.radius)
            if self._safe_path(position, path, threat):
                first = next(
                    (p for p in path if hypot(p[0] - position[0], p[1] - position[1]) > 1e-8), None
                )
                outward = (
                    first is not None
                    and (
                        (first[0] - position[0]) * (position[0] - threat[0])
                        + (first[1] - position[1]) * (position[1] - threat[1])
                    )
                    >= -1e-9
                )
                if outward:
                    return point, path
                if detour is None:
                    detour = (point, path)
        # Safe detours are preferable to freezing when every route starts slightly inward.
        return detour if detour is not None else (None, ())

    def _ready_interaction(self, view):
        wanted = "exit" if view.has_keycard else "keycard"
        for obj in self.explorer.known.values():
            dx, dy = obj.x - view.x, obj.y - view.y
            angle = abs((atan2(dy, dx) - view.yaw + pi) % (2 * pi) - pi)
            if (
                obj.kind == "exit"
                and self.last_seen is not None
                and hypot(view.x - self.last_seen[0], view.y - self.last_seen[1])
                < self.settings.threat_clearance - 1e-9
            ):
                continue
            if (
                obj.kind == wanted
                and hypot(dx, dy) <= self.config.interaction_range
                and angle <= self.config.interaction_half_angle
                and segment_clear(self.explorer.geometry, view.x, view.y, obj.x, obj.y)
            ):
                return True
        return False

    def _begin_flee(self):
        if not self.fleeing:
            self.fleeing = True
            self.flee_started = self.tick
            self.last_plan = -self.hold_ticks
            self.flee_entries += 1

    def _clear_exit_approach(self, unsafe=False):
        if self.exit_approach_active:
            self.explorer.navigator = Navigator()
            # Approach can move us across an obstacle from the cached flee origin.
            # Replan this tick without restarting the existing flee episode.
            self.navigator = Navigator()
            self.escape_target = None
            self.last_plan = self.tick - self.hold_ticks
        self.exit_approach_active = False
        self.exit_standpoint = self.exit_goal_id = None
        self.exit_navigator = Navigator()
        if unsafe:
            self.exit_approach_cancellations += 1
            self.exit_retry_tick = self.tick + self.hold_ticks
            self._begin_flee()

    def _exit_path_safe(self, position, path, point, goal):
        if (
            self.last_seen is None
            or hypot(position[0] - self.last_seen[0], position[1] - self.last_seen[1])
            < self.settings.threat_clearance - 1e-9
            or hypot(point[0] - self.last_seen[0], point[1] - self.last_seen[1])
            < self.settings.threat_clearance - 1e-9
            or self.explorer.geometry.overlaps(*point, self.config.radius)
            or not 1e-8
            < hypot(point[0] - goal.x, point[1] - goal.y)
            <= self.config.interaction_range - 1e-6
            or not segment_clear(self.explorer.geometry, *point, goal.x, goal.y)
        ):
            return False
        return self._safe_path(position, path, self.last_seen)

    def _exit_approach(self, view):
        # Knowledge comes only from previously filtered mission/threat observations.
        goals = sorted(
            (o for o in self.explorer.known.values() if o.kind == "exit"), key=lambda o: o.object_id
        )
        if not view.has_keycard or not goals or self.last_seen is None:
            self._clear_exit_approach()
            return None
        if self.tick < self.exit_retry_tick:
            return None
        position = (view.x, view.y)
        if self.exit_approach_active:
            goal = next((o for o in goals if o.object_id == self.exit_goal_id), None)
            arrived = (
                hypot(position[0] - self.exit_standpoint[0], position[1] - self.exit_standpoint[1])
                < 1e-8
            )
            remaining = (position,) if arrived else self.exit_navigator.path
            if goal is None or not self._exit_path_safe(
                position, remaining, self.exit_standpoint, goal
            ):
                self._clear_exit_approach(unsafe=True)
                return None
        else:
            self.exit_approach_attempts += 1
            choices = []
            for goal in goals:
                for point in (position, *self.explorer.points):
                    if (
                        not 1e-8
                        < hypot(point[0] - goal.x, point[1] - goal.y)
                        <= self.config.interaction_range - 1e-6
                    ):
                        continue
                    path = find_path(self.explorer.geometry, position, point, self.config.radius)
                    if not self._exit_path_safe(position, path, point, goal):
                        continue
                    length = sum(
                        hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip((position, *path), path)
                    )
                    choices.append((length, goal.object_id, point[1], point[0], point, path, goal))
            if not choices:
                self.exit_retry_tick = self.tick + self.hold_ticks
                self._begin_flee()
                return None
            _, self.exit_goal_id, _, _, self.exit_standpoint, path, goal = min(
                choices, key=lambda x: x[:4]
            )
            self.exit_navigator.target, self.exit_navigator.path = self.exit_standpoint, path
            self.exit_approach_active = True
        dx, dy = self.exit_navigator.displacement(
            position, self.config.walk_speed / self.config.tick_hz
        )
        # This conservative rectangle includes Core's X/Y substeps and clipped movement.
        nearest_x = max(
            min(position[0], position[0] + dx),
            min(self.last_seen[0], max(position[0], position[0] + dx)),
        )
        nearest_y = max(
            min(position[1], position[1] + dy),
            min(self.last_seen[1], max(position[1], position[1] + dy)),
        )
        clearance = hypot(nearest_x - self.last_seen[0], nearest_y - self.last_seen[1])
        if clearance < self.settings.threat_clearance - 1e-9:
            self._clear_exit_approach(unsafe=True)
            return None
        facing = atan2(goal.y - view.y, goal.x - view.x)
        error = (facing - view.yaw + pi) % (2 * pi) - pi
        turn = max(-self.explorer.turn_per_tick, min(self.explorer.turn_per_tick, error))
        yaw = view.yaw + turn
        scale = self.config.tick_hz / self.config.walk_speed
        if hypot(dx, dy) > 1e-12:
            self.exit_approach_steps += 1
            self.exit_motion_audit = {
                "threat": self.last_seen,
                "clearance": self.settings.threat_clearance,
                "predicted_min_clearance": clearance,
            }
        return SurvivorCommand(
            forward=max(-1.0, min(1.0, (dx * cos(yaw) + dy * sin(yaw)) * scale)),
            strafe=max(-1.0, min(1.0, (-dx * sin(yaw) + dy * cos(yaw)) * scale)),
            yaw_delta=turn,
        )

    def command(self, view):
        if not isinstance(view, SurvivorThreatView):
            raise TypeError("Evasive bot accepts only filtered SurvivorThreatView")
        self.tick += 1
        self.exit_motion_audit = None
        if view.visible_alien is not None:
            self.last_seen, self.last_seen_age_ticks = view.visible_alien, 0
        elif self.last_seen is not None:
            self.last_seen_age_ticks += 1
            if self.last_seen_age_ticks >= self.memory_ticks:
                self.last_seen = self.last_seen_age_ticks = None
        mission = view.mission
        for obj in mission.visible_objects:
            self.explorer.known[obj.object_id] = obj
        position = (mission.x, mission.y)
        distance = (
            hypot(position[0] - self.last_seen[0], position[1] - self.last_seen[1])
            if self.last_seen is not None
            else float("inf")
        )
        if self.fleeing and (
            self.last_seen is None
            or (
                distance >= self.settings.safe_distance
                and self.tick - self.flee_started >= self.hold_ticks
            )
        ):
            self.fleeing = False
            self.escape_target = None
            self.navigator = Navigator()
            # Retreat changed the mission path's starting position; discard its cached route.
            self.explorer.navigator = Navigator()
            # Resume a public mission/exploration route instead of paying a pending
            # stationary scan immediately after retreat. Arrival scans still apply.
            self.explorer.scanning = 0
        if (
            not self.fleeing
            and self.last_seen is not None
            and distance <= self.settings.enter_distance
        ):
            self._begin_flee()
        if self._ready_interaction(mission):
            return SurvivorCommand(interact=True)
        approach = self._exit_approach(mission)
        if approach is not None:
            return approach
        if not self.fleeing:
            return self.explorer.command(mission)
        self.flee_ticks += 1
        arrived = (
            self.escape_target is not None
            and hypot(position[0] - self.escape_target[0], position[1] - self.escape_target[1])
            < 1e-8
        )
        unsafe = (
            self.escape_target is not None
            and not arrived
            and not self._safe_path(position, self.navigator.path, self.last_seen)
        )
        if unsafe or (
            (self.escape_target is None or arrived)
            and self.tick - self.last_plan >= self.hold_ticks
        ):
            self.escape_target, path = self._plan(position)
            self.navigator.target, self.navigator.path = self.escape_target, path
            self.last_plan = self.tick
            self.replans += 1
            self.no_safe_route_count += self.escape_target is None
        dx, dy = self.navigator.displacement(position, self.config.run_speed / self.config.tick_hz)
        if hypot(dx, dy) < 1e-12:
            return SurvivorCommand()
        error = (atan2(dy, dx) - mission.yaw + pi) % (2 * pi) - pi
        turn = max(-self.explorer.turn_per_tick, min(self.explorer.turn_per_tick, error))
        yaw = mission.yaw + turn
        scale = self.config.tick_hz / self.config.run_speed
        # Core applies yaw first. Local backward/strafe movement retreats immediately while turning.
        return SurvivorCommand(
            forward=max(-1.0, min(1.0, (dx * cos(yaw) + dy * sin(yaw)) * scale)),
            strafe=max(-1.0, min(1.0, (-dx * sin(yaw) + dy * cos(yaw)) * scale)),
            yaw_delta=turn,
            running=True,
        )
