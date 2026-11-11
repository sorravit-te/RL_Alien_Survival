"""Bounded fixed-step scheduling and collision-safe display interpolation."""

from dataclasses import replace
from math import floor, hypot, pi
from random import Random

from ai.navigation import find_path
from app.input import InputState
from game.core import GameCore
from game.interaction import angle_difference
from game.motion import segment_has_clearance

NEW_ROUND_TRIES = 64  # a map with a single start stops looking for a different one
SEED_LIMIT = 2**32  # the range EpisodeSampler accepts
START_GRACE_SECONDS = 5.0  # to look around before a searching Alien can have you in sight range
# Cells from the start to the card and on to the nearest exit. On the ship 3 of the 24 start and
# card pairs fall short: the card waits beside the start with an exit behind it, a 4 to 6 second run.
SHORTEST_ESCAPE = 30.0


def round_start(snapshot):
    """What a player can tell apart between rounds; the working exit stays private to Core."""
    alien = snapshot.alien
    cards = tuple(obj.object_id for obj in snapshot.world_map.objects if obj.kind == "keycard")
    return (snapshot.survivor.x, snapshot.survivor.y), alien and (alien.x, alien.y), cards


def alien_starts_far(snapshot):
    """Measured by walking, since a wall can part two close starts; human rounds only."""
    alien, survivor, config = snapshot.alien, snapshot.survivor, snapshot.config
    if alien is None:
        return True
    walk = walk_length(
        snapshot.world_map, (alien.x, alien.y), (survivor.x, survivor.y), config.alien_radius
    )
    return walk >= config.alien_sight_range + START_GRACE_SECONDS * config.alien_search_speed


def escape_takes_long(snapshot):
    """Start, card, then the nearest exit, walked; human rounds only.

    Which exit works stays private to Core and never counts here: a round left out for it would
    tell the player that the exit beside a close card is the one that works.
    """
    world, survivor, radius = snapshot.world_map, snapshot.survivor, snapshot.config.radius
    card = next((obj for obj in world.objects if obj.kind == "keycard"), None)
    exits = [obj for obj in world.objects if obj.kind == "exit"]
    if card is None or not exits:
        return True
    to_card = walk_length(world, (survivor.x, survivor.y), (card.x, card.y), radius)
    onward = min(walk_length(world, (card.x, card.y), (out.x, out.y), radius) for out in exits)
    return to_card + onward >= SHORTEST_ESCAPE


def walk_length(world, start, end, radius):
    """Along the route a body of this radius walks; no route at all counts as no distance."""
    route = find_path(world, start, end, radius)
    return sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip((start, *route), route))


class FixedStepLoop:
    def __init__(self, core, inputs=None):
        self.core = core
        self.inputs = inputs if inputs is not None else InputState()
        self.dt = 1 / core.config.tick_hz
        self.max_steps = 5
        self.max_frame_time = 0.25
        self.paused = False
        self.dropped_seconds = 0.0
        self.on_tick = None
        self.rounds = Random()  # new rounds differ per session; a recording keeps each seed
        self.synchronize()

    def synchronize(self):
        self.accumulator = 0.0
        self.inputs.clear()
        self.previous = self.current = self.core.snapshot()
        self.previous_pitch = self.current_pitch = self.inputs.pitch

    def set_paused(self, paused):
        self.paused = paused
        self.synchronize()

    def reset(self, seed=None):
        self.core.reset(self.current.seed if seed is None else seed)
        self.inputs.pitch = 0.0
        self.synchronize()

    def new_round(self):
        """Leave the played start, card and Alien spawn; a new sampling seed varies its choices.

        It also passes over rounds with the Alien close or the card and an exit beside the start.
        """
        core = getattr(self.core, "core", self.core)
        preview = GameCore(core.world_map, core.config, alien_control=core.alien_control)
        played = round_start(preview.reset(self.current.seed))
        seed = self.rounds.randrange(SEED_LIMIT)
        for _ in range(NEW_ROUND_TRIES):
            candidate = self.rounds.randrange(SEED_LIMIT)
            start = preview.reset(candidate)
            if round_start(start) != played:
                seed = candidate
                if alien_starts_far(start) and escape_takes_long(start):
                    break
        if hasattr(self.core, "sampling_seed"):
            self.core.sampling_seed = self.rounds.randrange(SEED_LIMIT)
        self.reset(seed)

    def advance(self, elapsed):
        if self.paused:
            return 0
        elapsed = max(0.0, elapsed)
        accepted = min(elapsed, self.max_frame_time)
        self.dropped_seconds += elapsed - accepted
        self.accumulator += accepted
        count = 0
        while self.accumulator + 1e-12 >= self.dt and count < self.max_steps:
            self.previous = self.current
            self.previous_pitch = self.current_pitch
            self.current = self.core.step(self.inputs.consume(self.dt))
            if self.on_tick is not None:
                self.on_tick(self.current)
            self.current_pitch = self.inputs.pitch
            self.accumulator = max(0.0, self.accumulator - self.dt)
            count += 1
            if self.current.outcome is not None:
                self.inputs.clear()
                self.accumulator = 0.0
                self.previous = self.current
                self.previous_pitch = self.current_pitch
                break
        # Retain the fractional tick, discard whole ticks beyond the work budget.
        if self.accumulator >= self.dt:
            skipped = int(self.accumulator / self.dt)
            self.accumulator -= skipped * self.dt
            self.dropped_seconds += skipped * self.dt
        return count

    def display_pitch(self):
        """Interpolate tick endpoints with the same alpha as yaw; no smoothing history."""
        if self.paused:
            return self.current_pitch
        alpha = self.accumulator / self.dt
        return self.previous_pitch + (self.current_pitch - self.previous_pitch) * alpha

    def display_snapshot(self):
        if self.paused:
            return self.current
        alpha = self.accumulator / self.dt
        before, after = self.previous.survivor, self.current.survivor
        x = before.x + (after.x - before.x) * alpha
        y = before.y + (after.y - before.y) * alpha
        # Core wall sliding may follow a bent path. Never cut across its corner.
        radius = self.current.config.radius
        world = self.current.world_map
        # Conservative swept bounding box: if any solid tile intersects it,
        # display the proven Core endpoint instead of interpolating a corner cut.
        left, right = min(before.x, after.x) - radius, max(before.x, after.x) + radius
        top, bottom = min(before.y, after.y) - radius, max(before.y, after.y) + radius
        safe = not any(
            world.blocked(cx, cy)
            and left < cx + 1 and right > cx and top < cy + 1 and bottom > cy
            for cy in range(floor(top), floor(bottom) + 1)
            for cx in range(floor(left), floor(right) + 1)
        )
        if not safe or world.overlaps(x, y, radius):
            x, y = after.x, after.y
        turn = angle_difference(after.yaw, before.yaw)
        camera = replace(after, x=x, y=y, yaw=(before.yaw + turn * alpha) % (2 * pi))
        alien=self.current.alien
        prior=self.previous.alien
        if alien is not None and prior is not None:
            ax=prior.x+(alien.x-prior.x)*alpha
            ay=prior.y+(alien.y-prior.y)*alpha
            if not segment_has_clearance(world,(prior.x,prior.y),(alien.x,alien.y),
                                         self.current.config.alien_radius):
                ax,ay=alien.x,alien.y
            turn=(alien.yaw-prior.yaw+pi)%(2*pi)-pi
            alien=replace(alien,x=ax,y=ay,yaw=(prior.yaw+turn*alpha)%(2*pi))
        return replace(self.current, survivor=camera, alien=alien)
