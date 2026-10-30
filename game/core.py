"""Fixed-tick headless movement. No wall clock, window or asset dependencies."""

from dataclasses import replace
from math import atan2, ceil, cos, floor, hypot, isfinite, pi, sin
from random import Random

from ai.controller import ControllerState, choose_target
from ai.navigation import Navigator
from game.capture import capture_fraction
from game.doors import Doors, DoorSight
from game.interaction import segment_clear, select_target
from game.investigation import LastSeenInvestigation
from game.map import WorldMap, small_test_map
from game.motion import sweep_axis
from game.perception import PerceptionSnapshot, update_perception
from game.sound import HeardSnapshot, listen
from game.state import (
    AlienState,
    CoreConfig,
    GameEvent,
    SurvivorCommand,
    SurvivorState,
    WorldSnapshot,
)


class GameCore:
    def __init__(
        self,
        world_map: WorldMap | None = None,
        config: CoreConfig | None = None,
        *,
        alien_control: str = "heuristic",
    ):
        if alien_control not in ("heuristic", "external"):
            raise ValueError("Alien control must be heuristic or external")
        self.alien_control = alien_control
        self.world_map = world_map if world_map is not None else small_test_map()
        self.config = config if config is not None else CoreConfig()
        for x, y in self.world_map.spawn_points:
            if self.world_map.overlaps(x, y, self.config.radius):
                raise ValueError("Spawn does not have character clearance")
        if self.config.alien_enabled and self.world_map.alien_spawn is not None:
            for ax, ay, _ in self._alien_spawns():
                if self.world_map.overlaps(ax, ay, self.config.alien_radius):
                    raise ValueError("Alien spawn does not have character clearance")
                for x, y in self.world_map.spawn_points:
                    distance = hypot(ax - x, ay - y)
                    contact = (
                        self.config.radius + self.config.alien_radius + self.config.capture_padding
                    )
                    if distance < self.config.minimum_spawn_distance or (
                        distance <= contact and segment_clear(self.world_map, x, y, ax, ay)
                    ):
                        raise ValueError("Alien spawn is too close to Survivor spawn")
            if any(
                self.world_map.overlaps(*p, self.config.alien_radius)
                for p in self.world_map.patrol_waypoints
            ):
                raise ValueError("Patrol waypoint lacks Alien clearance")
        self.reset(0)

    def _alien_spawns(self):
        return (self.world_map.alien_spawn, *self.world_map.alien_spawn_options)

    def _scenario_objects(self, rng):
        """Keep one card and every exit in play; the working exit stays private."""
        objects = self.world_map.objects
        cards = [obj for obj in objects if obj.kind == "keycard"]
        exits = [obj for obj in objects if obj.kind == "exit"]
        self._active_exit = rng.choice(exits).object_id if exits else None
        if len(cards) < 2:
            return self.world_map
        return replace(self.world_map, objects=(rng.choice(cards), *exits))

    def reset(self, seed: int | None = None) -> WorldSnapshot:
        """Explicit seed replays a spawn; None requests a fresh local RNG seed."""
        self._seed = seed
        rng = Random(seed)
        x, y = rng.choice(self.world_map.spawn_points)
        self._survivor = SurvivorState(x, y)
        self._episode_map = self._scenario_objects(rng)
        self._ruled_out_exits = ()
        self._doors = Doors(self.world_map, self.config)
        self._tick = 0
        self._has_keycard = False
        self._outcome = None
        self._events = ()
        self._alien = (
            AlienState(*rng.choice(self._alien_spawns()))
            if self.config.alien_enabled and self.world_map.alien_spawn is not None
            else None
        )
        self._perception = PerceptionSnapshot()
        self._heard = HeardSnapshot()
        self._controller = ControllerState()
        self._navigator = Navigator()
        self._investigation = LastSeenInvestigation(self.config)
        self._noise_sweep = LastSeenInvestigation(self.config)
        return self.snapshot()

    def snapshot(self) -> WorldSnapshot:
        target = (
            None
            if self._outcome
            else select_target(self._episode_map, self._survivor, self.config, self._has_keycard)
        )
        return WorldSnapshot(
            self._tick,
            self._seed,
            self._survivor,
            self._episode_map,
            self.config,
            self._has_keycard,
            target,
            self._outcome,
            self._events,
            self._alien,
            self._perception,
            self._controller.target,
            self._controller.source,
            self._navigator.path,
            self._investigation.state,
            self._controller.clue_revision,
            self._ruled_out_exits,
            self._doors.openness,
            self._doors.closed,
            self._heard,
            self._controller.heard_revision,
            self._noise_sweep.state,
        )

    def set_alien_target(self, target, *, source="external", evidence_revision=None):
        """External controller owns a copied target snapshot; None explicitly stops.

        Last-seen investigation additionally requires the current confirmed target/revision.
        Other targets retain the caller-owned knowledge boundary. No hidden Survivor lookup
        or heuristic fallback occurs here. Unreachable floor targets yield an empty path.
        """
        if self.alien_control != "external":
            raise RuntimeError("External targets require external control mode")
        if self._alien is None:
            raise RuntimeError("No active Alien")
        if self._outcome is not None:
            raise RuntimeError("Cannot set Alien target after terminal")
        if not isinstance(source, str) or not source:
            raise ValueError("Target source must be a nonempty string")
        if target is not None:
            target = tuple(float(value) for value in target)
            if len(target) != 2 or not all(isfinite(v) for v in target):
                raise ValueError("Target must be a finite x/y snapshot")
            if self.world_map.blocked(floor(target[0]), floor(target[1])):
                raise ValueError("Target must be on floor")
        self._assign_alien_target(target, source, evidence_revision)

    def _assign_alien_target(self, target, source, evidence_revision=None):
        if source == "last_seen":
            if (
                type(evidence_revision) is not int
                or evidence_revision <= 0
                or evidence_revision != self._perception.revision
                or target != self._perception.last_seen
            ):
                raise ValueError(
                    "Investigation requires the current confirmed clue target/revision"
                )
            self._investigation.select(target, evidence_revision)
        else:
            self._investigation.cancel()
        if source == "sound":
            if target != self._heard.point or self._heard.checked:
                raise ValueError("Sound targets only the latest unchecked noise")
            self._noise_sweep.select(target, self._heard.revision)
        else:
            self._noise_sweep.cancel()
        self._controller = replace(
            self._controller,
            target=target,
            source=source,
            clue_revision=self._perception.revision,
            heard_revision=self._heard.revision,
        )
        self._navigator.set_target(
            self.world_map, (self._alien.x, self._alien.y), target, self.config.alien_radius
        )

    def step(self, command: SurvivorCommand) -> WorldSnapshot:
        """Advance exactly 1/30 second; rotate first, then move in the new frame."""
        self._events = ()  # events belong to this step, including post-terminal calls
        if self._outcome is not None:
            return self.snapshot()
        # Both commands use pre-tick state. Sensor memory is sampled before movement.
        alien_delta = self._alien_displacement()
        yaw = (self._survivor.yaw + command.yaw_delta) % (2 * pi)
        scale = max(1.0, hypot(command.forward, command.strafe))
        speed = self.config.run_speed if command.running else self.config.walk_speed
        distance = speed / self.config.tick_hz / scale
        survivor_delta = (
            (cos(yaw) * command.forward - sin(yaw) * command.strafe) * distance,
            (sin(yaw) * command.forward + cos(yaw) * command.strafe) * distance,
        )
        start = self._survivor
        captured = self._move_both(survivor_delta, alien_delta, yaw)
        bodies = [(self._survivor.x, self._survivor.y)]
        if self._alien is not None:
            bodies.append((self._alien.x, self._alien.y))
        self._doors.update(bodies)
        self._hear(start, command.running)
        self._tick += 1
        if captured:
            self._outcome = "captured"
            self._events = (GameEvent("captured", "alien", self._tick),)
        else:
            # E resolves at the end of the tick; any earlier/simultaneous capture wins.
            self._interact(command)
        return self.snapshot()

    def _hear(self, start, running):
        """Footfalls land on a fixed beat; standing or walking into a wall makes none."""
        if self._alien is None:
            return
        period = max(1, ceil(self.config.sound_step_seconds * self.config.tick_hz - 1e-12))
        moved = hypot(self._survivor.x - start.x, self._survivor.y - start.y) > 1e-9
        noises = [((x + 0.5, y + 0.5), self.config.door_sound_range) for x, y in self._doors.opened]
        if moved and self._tick % period == 0:
            reach = self.config.run_sound_range if running else self.config.walk_sound_range
            noises.insert(0, ((self._survivor.x, self._survivor.y), reach))
        self._heard = listen(
            self.world_map,
            frozenset(self._doors.closed),
            self._alien,
            noises,
            self._heard,
            self.config,
        )

    def _alien_displacement(self):
        if self._alien is None:
            return (0.0, 0.0)
        sight = DoorSight(self._episode_map, frozenset(self._doors.closed))
        self._perception = update_perception(
            sight, self._alien, self._survivor, self._perception, self.config
        )
        age = self._perception.last_seen_age_ticks
        chasing = age is not None and age <= ceil(
            self.config.chase_hold_seconds * self.config.tick_hz
        )
        self._alien = replace(self._alien, speed_mode="chase" if chasing else "search")
        period = max(1, ceil(self.config.alien_decision_seconds * self.config.tick_hz - 1e-12))
        if self.alien_control == "heuristic" and self._tick % period == 0:
            self._controller = choose_target(
                self._alien,
                self._perception,
                self.world_map.patrol_waypoints,
                self._controller,
                self.config.target_arrival_distance,
                self._investigation.state.investigated_revision,
                self._heard,
            )
            self._assign_alien_target(
                self._controller.target,
                self._controller.source,
                self._perception.revision if self._controller.source == "last_seen" else None,
            )
        # Prune points already reached before checking the path-complete condition.
        self._navigator.displacement((self._alien.x, self._alien.y), 0.0)
        sweep, memory = self._active_sweep()
        state = sweep.state
        arrived = (
            state.target is not None
            and not self._navigator.path
            and hypot(self._alien.x - state.target[0], self._alien.y - state.target[1])
            <= self.config.target_arrival_distance
        )
        turn, hold = sweep.advance(memory, arrived=arrived)
        if state.phase != "cancelled" and sweep.state.phase == "cancelled":
            # Reacquisition/new evidence cancels old navigation, retaining chosen target in diagnostics.
            self._navigator.set_target(
                self.world_map, (self._alien.x, self._alien.y), None, self.config.alien_radius
            )
        if sweep is self._noise_sweep and sweep.state.investigated_revision == self._heard.revision:
            self._heard = replace(self._heard, checked=True)
        hold = self._track_confirmed_sighting(hold)
        if hold:
            self._alien = replace(self._alien, yaw=(self._alien.yaw + turn) % (2 * pi))
            return (0.0, 0.0)
        speed = self.config.alien_chase_speed if chasing else self.config.alien_search_speed
        return self._navigator.displacement(
            (self._alien.x, self._alien.y), speed / self.config.tick_hz
        )

    def _active_sweep(self):
        """A newer noise never cancels a noise sweep: the next decision simply retargets."""
        if self._controller.source != "sound":
            return self._investigation, self._perception
        sweep = self._noise_sweep
        return sweep, replace(self._perception, revision=sweep.state.revision)

    def _track_confirmed_sighting(self, hold):
        """Written pursuit helper; reads only this tick's pre-motion confirmed sample.

        A chase target follows each confirmed point. Reacquisition that cancelled a
        last-seen primitive becomes a helper chase until the next decision. After sight
        is lost the target stays at the last confirmed point, never the hidden Survivor.
        """
        memory, source = self._perception, self._controller.source
        if (
            memory.visible
            and source in ("last_seen", "sound")
            and self._active_sweep()[0].state.phase == "cancelled"
        ):
            source = "visible_handoff"
        if source not in ("visible", "visible_handoff"):
            return hold
        if memory.visible:
            self._controller = replace(self._controller, target=memory.last_seen, source=source)
            self._navigator.set_target(
                self.world_map,
                (self._alien.x, self._alien.y),
                memory.last_seen,
                self.config.alien_radius,
            )
        return False if source == "visible_handoff" else hold

    def _move_both(self, survivor_delta, alien_delta, yaw):
        s = (self._survivor.x, self._survivor.y)
        a = (self._alien.x, self._alien.y) if self._alien else None
        original_a = a
        radius = min(self.config.radius, self.config.alien_radius) if a else self.config.radius
        count = max(1, ceil(max(hypot(*survivor_delta), hypot(*alien_delta)) / (radius / 4)))
        captured = False
        for _ in range(count):
            # Within each bounded substep: simultaneous X sweeps, then simultaneous Y sweeps.
            for horizontal in (True, False):
                axis = 0 if horizontal else 1
                other = 1 - axis
                next_s = list(s)
                next_s[axis] = sweep_axis(
                    self.world_map,
                    self.config.radius,
                    s[axis],
                    s[other],
                    survivor_delta[axis] / count,
                    horizontal=horizontal,
                )
                next_s = tuple(next_s)
                if a is not None:
                    next_a = list(a)
                    next_a[axis] = sweep_axis(
                        self.world_map,
                        self.config.alien_radius,
                        a[axis],
                        a[other],
                        alien_delta[axis] / count,
                        horizontal=horizontal,
                    )
                    next_a = tuple(next_a)
                    limit = (
                        self.config.radius + self.config.alien_radius + self.config.capture_padding
                    )
                    t = capture_fraction(self.world_map, s, next_s, a, next_a, limit)
                    if t is not None:
                        s = tuple(s[i] + t * (next_s[i] - s[i]) for i in (0, 1))
                        a = tuple(a[i] + t * (next_a[i] - a[i]) for i in (0, 1))
                        captured = True
                        break
                    a = next_a
                s = next_s
            if captured:
                break
        self._survivor = SurvivorState(*s, yaw)
        if a is not None:
            dx, dy = a[0] - original_a[0], a[1] - original_a[1]
            facing = atan2(dy, dx) % (2 * pi) if hypot(dx, dy) > 1e-12 else self._alien.yaw
            self._alien = replace(self._alien, x=a[0], y=a[1], yaw=facing)
        return captured

    def _interact(self, command):
        target = select_target(self._episode_map, self._survivor, self.config, self._has_keycard)
        eligible = (
            command.interact
            and command.forward == 0
            and command.strafe == 0
            and target is not None
            and (target.kind != "exit" or self._has_keycard)
        )
        if not eligible:
            return
        if target.kind == "keycard":
            self._has_keycard = True
            kind = "keycard_collected"
        elif target.object_id != self._active_exit:
            kind = "exit_offline"
            if target.object_id not in self._ruled_out_exits:
                self._ruled_out_exits += (target.object_id,)
        else:
            self._outcome = "escaped"
            kind = "escaped"
        self._events = (GameEvent(kind, target.object_id, self._tick),)
