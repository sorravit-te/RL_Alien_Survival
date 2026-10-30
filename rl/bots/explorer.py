"""Explorer uses known static geometry and visible objects only, no Alien action."""

from math import atan2, ceil, floor, hypot, inf, pi, radians

from ai.navigation import Navigator
from game.interaction import angle_difference, segment_clear
from game.map import WorldMap
from game.state import SurvivorCommand
from rl.bots.stations import plan_stations
from rl.bots.views import move_command

ROUTE_JITTER = 8.0  # steps a station may lose to a nearer one, so routes differ by seed


class ExplorerBot:
    label = "explorer_scripted"

    def __init__(self, rows, config, *, doors=(), running=False):
        # Copy geometry only. No undiscovered object positions, spawn list or Alien state.
        floors = tuple(
            (x + 0.5, y + 0.5)
            for y, row in enumerate(rows)
            for x, tile in enumerate(row)
            if tile == "."
        )
        if not floors:
            raise ValueError("Explorer needs walkable geometry")
        self.geometry = WorldMap(tuple(rows), (floors[0],))
        self.config = config
        self.running = running
        self.speed = config.run_speed if running else config.walk_speed
        self.points = tuple(p for p in floors if not self.geometry.overlaps(*p, config.radius))
        # A deck with doors is searched room by room; other scenes roam every floor point.
        self.stations, self.walk_steps = (
            plan_stations(tuple(rows), self.points, tuple(doors)) if doors else (self.points, {})
        )
        self.turn_per_tick = radians(120) / config.tick_hz
        self.scan_ticks = ceil(2 * pi / self.turn_per_tick)

    def reset(self, rng):
        self.order = tuple(self.stations[int(i)] for i in rng.permutation(len(self.stations)))
        self.bias = {p: i * ROUTE_JITTER / len(self.order) for i, p in enumerate(self.order)}
        self.known = {}
        self.visited = set()
        self.explore_target = None
        self.scanning = self.scan_ticks
        self.navigator = Navigator()
        self.last_command_reason = "reset"

    def _next_target(self, position):
        waiting = [p for p in self.order if p not in self.visited]
        if not waiting or not self.walk_steps:
            return waiting[0] if waiting else None
        here = (floor(position[0]) + 0.5, floor(position[1]) + 0.5)
        return min(waiting, key=lambda p: self.walk_steps[p].get(here, inf) + self.bias[p])

    def postpone_target(self):
        """A deck search leaves a station it was chased from until the next sweep."""
        if self.walk_steps and self.explore_target is not None:
            self.visited.add(self.explore_target)
            self.explore_target = None

    def command(self, view, *, move_while_turning=False):
        """Default commands stay unchanged; RESUME may project walking through yaw."""
        self.last_command_reason = None
        for obj in view.visible_objects:
            self.known[obj.object_id] = obj
        for object_id in view.ruled_out:
            self.known.pop(object_id, None)
        wanted = "exit" if view.has_keycard else "keycard"
        discovered = [obj for obj in self.known.values() if obj.kind == wanted]
        mission = (
            min(discovered, key=lambda obj: (hypot(obj.x - view.x, obj.y - view.y), obj.object_id))
            if discovered
            else None
        )
        position = (view.x, view.y)
        if mission is not None:
            target = (mission.x, mission.y)
            dx, dy = mission.x - view.x, mission.y - view.y
            angle = angle_difference(atan2(dy, dx), view.yaw) if hypot(dx, dy) else 0.0
            if (
                hypot(dx, dy) <= self.config.interaction_range
                and abs(angle) <= self.config.interaction_half_angle
                and segment_clear(self.geometry, view.x, view.y, *target)
            ):
                self.last_command_reason = "interaction"
                return SurvivorCommand(interact=True)  # one new press; Core commits it now
            self.scanning = 0
        else:
            # Select once, then apply the same arrival rules to new and existing targets.
            if self.explore_target is None and not self.scanning:
                self.explore_target = self._next_target(position)
                if self.explore_target is None:
                    if not self.order:
                        self.last_command_reason = "no_target"
                        return SurvivorCommand()
                    self.visited.clear()
                    self.explore_target = self.order[0]
            arrived = (
                self.explore_target is not None
                and hypot(view.x - self.explore_target[0], view.y - self.explore_target[1]) < 1e-8
            )
            if arrived:
                self.visited.add(self.explore_target)
                self.explore_target = None
                self.scanning = self.scan_ticks
            if self.scanning:
                self.scanning -= 1
                self.last_command_reason = "arrived" if arrived else "scan"
                return SurvivorCommand(yaw_delta=self.turn_per_tick)
            target = self.explore_target
        self.navigator.set_target(self.geometry, position, target, self.config.radius)
        dx, dy = self.navigator.displacement(position, self.speed / self.config.tick_hz)
        if hypot(dx, dy) < 1e-12:
            if mission is not None and hypot(target[0] - view.x, target[1] - view.y) < 1e-8:
                # Navigator can prune a tiny final segment before the yaw requirement is met.
                # Arrival still ends RESUME; turn toward E without inventing a route failure.
                self.last_command_reason = "arrived"
                turn = max(-self.turn_per_tick, min(self.turn_per_tick, angle))
                return SurvivorCommand(yaw_delta=turn)
            self.last_command_reason = "no_path"
            if mission is None:
                self.visited.add(target)
                self.explore_target = None
            return SurvivorCommand()
        error = angle_difference(atan2(dy, dx), view.yaw)
        turn = max(-self.turn_per_tick, min(self.turn_per_tick, error))
        aligned = abs(error) <= self.turn_per_tick + 1e-12
        self.last_command_reason = "aligned" if aligned else "turning"
        if move_while_turning:
            # Preserve the navigator's global walk displacement while turning.
            return move_command(dx, dy, view.yaw, turn, self.config, running=self.running)
        if not aligned:
            return SurvivorCommand(yaw_delta=turn)
        distance = hypot(dx, dy)
        return SurvivorCommand(
            forward=min(1.0, distance * self.config.tick_hz / self.speed),
            yaw_delta=turn,
            running=self.running,
        )


class StationaryDebugBot:
    label = "stationary_debug_only"

    def reset(self, rng):
        pass

    def command(self, view):
        return SurvivorCommand()
