"""Explorer uses known static geometry and visible objects only, no Alien action.

Object perception is an explicit privileged boundary. Bot.command receives the
filtered view, not WorldSnapshot/WorldMap objects. FOV75 and LoS match the human
horizontal camera rules; simulated pitch has no interaction meaning. The bot
uses center visibility (conservative vs a partially visible human sprite).
"""

from dataclasses import dataclass
from math import atan2, ceil, hypot, pi, radians

from ai.navigation import Navigator
from game.interaction import segment_clear
from game.map import WorldMap
from game.state import SurvivorCommand


@dataclass(frozen=True)
class SurvivorView:
    x: float
    y: float
    yaw: float
    has_keycard: bool
    visible_objects: tuple


def observe_survivor(world, survivor, has_keycard):
    visible = []
    for obj in world.objects:
        if obj.kind == "keycard" and has_keycard:
            continue
        dx, dy = obj.x - survivor.x, obj.y - survivor.y
        angle = abs((atan2(dy, dx) - survivor.yaw + pi) % (2 * pi) - pi) if hypot(dx, dy) else 0
        if angle <= radians(75) / 2 and segment_clear(world, survivor.x, survivor.y, obj.x, obj.y):
            visible.append(obj)
    return SurvivorView(
        survivor.x,
        survivor.y,
        survivor.yaw,
        has_keycard,
        tuple(sorted(visible, key=lambda obj: obj.object_id)),
    )


class ExplorerBot:
    label = "explorer_scripted"

    def __init__(self, rows, config):
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
        self.points = tuple(p for p in floors if not self.geometry.overlaps(*p, config.radius))
        self.turn_per_tick = radians(120) / config.tick_hz
        self.scan_ticks = ceil(2 * pi / self.turn_per_tick)

    def reset(self, rng):
        self.order = tuple(self.points[int(i)] for i in rng.permutation(len(self.points)))
        self.known = {}
        self.visited = set()
        self.explore_target = None
        self.scanning = self.scan_ticks
        self.navigator = Navigator()

    def command(self, view):
        for obj in view.visible_objects:
            self.known[obj.object_id] = obj
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
            angle = (atan2(dy, dx) - view.yaw + pi) % (2 * pi) - pi if hypot(dx, dy) else 0.0
            if (
                hypot(dx, dy) <= self.config.interaction_range
                and abs(angle) <= self.config.interaction_half_angle
                and segment_clear(self.geometry, view.x, view.y, *target)
            ):
                return SurvivorCommand(interact=True)  # one new press; Core commits it now
            self.scanning = 0
        else:
            if (
                self.explore_target is not None
                and hypot(view.x - self.explore_target[0], view.y - self.explore_target[1]) < 1e-8
            ):
                self.visited.add(self.explore_target)
                self.explore_target = None
                self.scanning = self.scan_ticks
            if self.scanning:
                self.scanning -= 1
                return SurvivorCommand(yaw_delta=self.turn_per_tick)
            if self.explore_target is None:
                self.explore_target = next((p for p in self.order if p not in self.visited), None)
                if self.explore_target is None:
                    self.visited.clear()
                    self.explore_target = self.order[0]
            target = self.explore_target
        self.navigator.set_target(self.geometry, position, target, self.config.radius)
        dx, dy = self.navigator.displacement(position, self.config.walk_speed / self.config.tick_hz)
        if hypot(dx, dy) < 1e-12:
            if mission is None:
                self.visited.add(target)
                self.explore_target = None
            return SurvivorCommand()
        error = (atan2(dy, dx) - view.yaw + pi) % (2 * pi) - pi
        turn = max(-self.turn_per_tick, min(self.turn_per_tick, error))
        if abs(error) > self.turn_per_tick + 1e-12:
            return SurvivorCommand(yaw_delta=turn)
        distance = hypot(dx, dy)
        return SurvivorCommand(
            forward=min(1.0, distance * self.config.tick_hz / self.config.walk_speed),
            yaw_delta=turn,
        )


class StationaryDebugBot:
    label = "stationary_debug_only"

    def reset(self, rng):
        pass

    def command(self, view):
        return SurvivorCommand()
