"""A small scripted retreating Explorer over the LoS-filtered threat view.

The bot keeps expiring last-seen memory and never receives Core/WorldSnapshot/AlienState.
"""

from dataclasses import dataclass
from math import atan2, ceil, hypot, isfinite, pi

from ai.navigation import Navigator, find_path
from game.interaction import angle_difference, segment_clear
from game.state import SurvivorCommand
from rl.bots.explorer import ExplorerBot
from rl.bots.views import SurvivorThreatView, move_command


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


@dataclass(frozen=True)
class ExplorationRetryRecord:
    """Public task history, never a threat position or a danger map."""

    anchor_id: int
    count: int
    first_tick: int
    defer_until: int | None


class EvasiveExplorerBot:
    retry_window_commands = 300
    retry_defer_commands = 180
    retry_record_limit = 4
    retry_anchor_check_limit = 16
    retry_path_limit = 4
    replan_on_arrival = False  # plain bot waits the hold time at a reached retreat point

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
        self._retry_anchor_ids = {point: i for i, point in enumerate(self.explorer.points)}
        self.memory_ticks = ceil(self.settings.memory_seconds * config.tick_hz)
        self.hold_ticks = ceil(self.settings.target_hold_seconds * config.tick_hz)
        # At most one half-turn at the existing turn rate; a ceiling, not an alignment guarantee.
        self.resume_limit_ticks = max(1, ceil(pi / self.explorer.turn_per_tick - 1e-12))

    def reset(self, rng):
        self.explorer.reset(rng)
        self.last_seen = None
        self.last_seen_age_ticks = None
        self.fleeing = False
        self.tick = 0
        self.resume_active = False
        self.resume_started_tick = self.resume_deadline_tick = None
        self.resume_entry_reason = None
        self.resume_end_reason = "reset"
        self.resume_ended_tick = None
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
        self.retry_records = {}
        self.retry_pending = self.retry_exit_token = self.retry_attempted_exit = None
        self.retry_cursor = 0
        self.retry_phase = None
        self.retry_last_interruption = self.retry_last_selection = None
        self._retry_previous_position = self._retry_previous_anchor = None
        self._retry_entry_candidate = self._retry_public_command_anchor = None

    def _start_resume(self, reason):
        # Only the FLEE exit calls this. Target changes never restart the clock.
        if self.resume_active:
            return
        self.resume_active = True
        self.resume_started_tick = self.tick
        self.resume_deadline_tick = self.tick + self.resume_limit_ticks
        self.resume_entry_reason = reason
        self.resume_end_reason = self.resume_ended_tick = None

    def _end_resume(self, reason):
        if self.resume_active:
            self.resume_active = False
            self.resume_end_reason = reason
            self.resume_ended_tick = self.tick

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
            angle = abs(angle_difference(atan2(dy, dx), view.yaw))
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
            self._end_resume("flee_entered")
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
        error = angle_difference(facing, view.yaw)
        turn = max(-self.explorer.turn_per_tick, min(self.explorer.turn_per_tick, error))
        if hypot(dx, dy) > 1e-12:
            self.exit_approach_steps += 1
            self.exit_motion_audit = {
                "threat": self.last_seen,
                "clearance": self.settings.threat_clearance,
                "predicted_min_clearance": clearance,
            }
        return move_command(dx, dy, view.yaw, turn, self.config)

    def _retry_clear_pending(self):
        self.retry_pending = None
        self.retry_exit_token = None

    def _retry_drop(self, anchor_id):
        record = self.retry_records.pop(anchor_id, None)
        if record is not None and self.retry_pending == (anchor_id, record.first_tick):
            self._retry_clear_pending()
        if (
            self.retry_last_interruption is not None
            and self.retry_last_interruption[1] == anchor_id
        ):
            self.retry_last_interruption = None
        if (
            self.retry_last_selection is not None
            and self.retry_last_selection["from_anchor"] == anchor_id
        ):
            self.retry_last_selection = None

    def _retry_clear(self):
        self.retry_records.clear()
        self._retry_clear_pending()
        self.retry_last_interruption = self.retry_last_selection = None
        self.retry_cursor = 0

    def _retry_wanted_known(self, mission):
        wanted = "exit" if mission.has_keycard else "keycard"
        return any(obj.kind == wanted for obj in self.explorer.known.values())

    def _retry_cleanup(self, mission):
        # All time/mission invalidation runs before any command early return.
        if self.retry_phase is not None and self.retry_phase != mission.has_keycard:
            self._retry_clear()
        self.retry_phase = mission.has_keycard
        if self._retry_wanted_known(mission):
            self._retry_clear()
        for anchor_id, record in tuple(self.retry_records.items()):
            expiry = (
                record.defer_until
                if record.defer_until is not None
                else record.first_tick + self.retry_window_commands
            )
            point = self.explorer.points[anchor_id]
            reached = (
                self.explorer.explore_target == point
                and hypot(mission.x - point[0], mission.y - point[1]) < 1e-8
            )
            if self.tick >= expiry or reached:
                self._retry_drop(anchor_id)
        if self.retry_pending is not None:
            anchor_id, first_tick = self.retry_pending
            record = self.retry_records.get(anchor_id)
            if record is None or record.first_tick != first_tick or record.defer_until is None:
                self._retry_clear_pending()

    def _retry_interrupted_anchor(self, view):
        mission = view.mission
        target = self.explorer.explore_target
        anchor_id = self._retry_anchor_ids.get(target)
        if (
            anchor_id is None
            or anchor_id != self._retry_previous_anchor
            or self._retry_previous_position is None
            or self.explorer.scanning
            or not self.explorer.navigator.path
            or self._retry_wanted_known(mission)
            or hypot(mission.x - target[0], mission.y - target[1]) < 1e-8
            or hypot(
                mission.x - self._retry_previous_position[0],
                mission.y - self._retry_previous_position[1],
            )
            <= 1e-8
            or self.last_seen is None
            or self.last_seen_age_ticks is None
            or not 0 <= self.last_seen_age_ticks < self.memory_ticks
        ):
            return None
        return anchor_id, "fresh_sight" if view.visible_alien is not None else "remembered_threat"

    def _retry_record_interruption(self, anchor_id, reason):
        record = self.retry_records.get(anchor_id)
        # Deferred records cannot be renewed, including repeated fallback encounters.
        if record is not None and record.defer_until is not None:
            return
        if record is None:
            if len(self.retry_records) >= self.retry_record_limit:
                oldest = min(self.retry_records.values(), key=lambda r: (r.first_tick, r.anchor_id))
                self._retry_drop(oldest.anchor_id)
            record = ExplorationRetryRecord(anchor_id, 1, self.tick, None)
        else:
            record = ExplorationRetryRecord(
                anchor_id, 2, record.first_tick, self.tick + self.retry_defer_commands
            )
            self.retry_pending = (anchor_id, record.first_tick)
            self.retry_exit_token = None
        self.retry_records[anchor_id] = record
        self.retry_last_interruption = (self.tick, anchor_id, reason)

    def _retry_reselect(self, mission):
        if (
            self.retry_pending is None
            or self.retry_exit_token is None
            or self.retry_attempted_exit == self.retry_exit_token
            or self.explorer.scanning
        ):
            return
        old_id, _ = self.retry_pending
        target = self.explorer.explore_target
        if self._retry_anchor_ids.get(target) != old_id:
            self._retry_clear_pending()
            return
        if hypot(mission.x - target[0], mission.y - target[1]) < 1e-8:
            return  # Explorer's original arrival/scan handler owns this command.
        self.retry_attempted_exit = self.retry_exit_token
        inspected = 0
        # Reserve one of the four searches for the original target's fallback route.
        # Explorer reuses either this path or the chosen candidate path, never a fifth search.
        fallback = find_path(
            self.explorer.geometry, (mission.x, mission.y), target, self.config.radius
        )
        self.explorer.navigator = Navigator()
        self.explorer.navigator.target, self.explorer.navigator.path = target, fallback
        searches = 1
        selected_id = None
        order = self.explorer.order
        for _ in range(min(self.retry_anchor_check_limit, len(order))):
            index = self.retry_cursor % len(order)
            point = order[index]
            self.retry_cursor = (index + 1) % len(order)
            inspected += 1
            anchor_id = self._retry_anchor_ids.get(point)
            record = self.retry_records.get(anchor_id)
            if (
                anchor_id is None
                or anchor_id == old_id
                or point in self.explorer.visited
                or (record is not None and record.defer_until is not None)
            ):
                continue
            searches += 1
            path = find_path(
                self.explorer.geometry, (mission.x, mission.y), point, self.config.radius
            )
            if path:
                # Public route only; reuse this exact search, not a fifth path computation.
                self.explorer.explore_target = point
                self.explorer.navigator = Navigator()
                self.explorer.navigator.target, self.explorer.navigator.path = point, path
                selected_id = anchor_id
                self._retry_clear_pending()
                break
            if searches >= self.retry_path_limit:
                break
        self.retry_last_selection = {
            "tick": self.tick,
            "from_anchor": old_id,
            "to_anchor": selected_id,
            "inspected": inspected,
            "path_searches": searches,
            "candidate_searches": searches - 1,
            "outcome": "selected" if selected_id is not None else "fallback",
        }

    def command(self, view):
        if not isinstance(view, SurvivorThreatView):
            raise TypeError("Evasive bot accepts only filtered SurvivorThreatView")
        self._retry_entry_candidate = None
        self._retry_public_command_anchor = None
        command = self._command(view)
        # Only a genuine direct entry which actually emits retreat can add history.
        if (
            self._retry_entry_candidate is not None
            and command.running
            and hypot(command.forward, command.strafe) > 1e-12
        ):
            self._retry_record_interruption(*self._retry_entry_candidate)
        self._retry_previous_position = (view.mission.x, view.mission.y)
        self._retry_previous_anchor = (
            self._retry_public_command_anchor
            if not command.running and hypot(command.forward, command.strafe) > 1e-12
            else None
        )
        return command

    def _flee_facing(self, position, dx, dy):
        return atan2(dy, dx)  # face the retreat direction

    def _command(self, view):
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
        # Entry command counts as RESUME command1: S..S+limit-1 are active.
        # Expire BEFORE all interaction/approach early returns on command S+limit.
        if self.resume_active and self.tick >= self.resume_deadline_tick:
            self._end_resume("deadline")
        mission = view.mission
        for obj in mission.visible_objects:
            self.explorer.known[obj.object_id] = obj
        for object_id in mission.ruled_out:
            self.explorer.known.pop(object_id, None)
        position = (mission.x, mission.y)
        self._retry_cleanup(mission)
        distance = (
            hypot(position[0] - self.last_seen[0], position[1] - self.last_seen[1])
            if self.last_seen is not None
            else float("inf")
        )
        if self.fleeing and (
            self.last_seen is None
            or (
                view.visible_alien is not None
                and distance >= self.settings.safe_distance
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
            self._start_resume("memory_expired" if self.last_seen is None else "visible_separation")
            if self.retry_pending is not None:
                self.retry_exit_token = self.flee_entries
        if (
            not self.fleeing
            and self.last_seen is not None
            and distance <= self.settings.enter_distance
        ):
            self._retry_entry_candidate = self._retry_interrupted_anchor(view)
            self._begin_flee()
        if self._ready_interaction(mission):
            self._end_resume("interaction_ready")
            return SurvivorCommand(interact=True)
        approach = self._exit_approach(mission)
        if approach is not None:
            return approach
        if not self.fleeing:
            if not self._retry_wanted_known(mission):
                self._retry_reselect(mission)
            command = self.explorer.command(mission, move_while_turning=self.resume_active)
            reason = self.explorer.last_command_reason
            if reason == "arrived":
                for anchor_id in tuple(self.retry_records):
                    point = self.explorer.points[anchor_id]
                    if hypot(position[0] - point[0], position[1] - point[1]) < 1e-8:
                        self._retry_drop(anchor_id)
            if not self._retry_wanted_known(mission) and self.explorer.navigator.path:
                self._retry_public_command_anchor = self._retry_anchor_ids.get(
                    self.explorer.explore_target
                )
            if reason in ("aligned", "arrived", "interaction", "no_path", "no_target"):
                self._end_resume(reason)
            return command
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
            and (
                self.tick - self.last_plan >= self.hold_ticks
                or (arrived and self.replan_on_arrival)
            )
        ):
            self.escape_target, path = self._plan(position)
            self.navigator.target, self.navigator.path = self.escape_target, path
            self.last_plan = self.tick
            self.replans += 1
            self.no_safe_route_count += self.escape_target is None
        dx, dy = self.navigator.displacement(position, self.config.run_speed / self.config.tick_hz)
        if hypot(dx, dy) < 1e-12:
            return SurvivorCommand()
        error = angle_difference(self._flee_facing(position, dx, dy), mission.yaw)
        turn = max(-self.explorer.turn_per_tick, min(self.explorer.turn_per_tick, error))
        # Local backward/strafe movement retreats immediately while turning.
        return move_command(dx, dy, mission.yaw, turn, self.config, running=True)
