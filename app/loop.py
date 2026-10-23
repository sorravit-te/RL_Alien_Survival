"""Bounded fixed-step scheduling and collision-safe display interpolation."""

from dataclasses import replace
from math import floor, pi

from app.input import InputState


class FixedStepLoop:
    def __init__(self, core, inputs=None):
        self.core = core
        self.inputs = inputs if inputs is not None else InputState()
        self.dt = 1 / core.config.tick_hz
        self.max_steps = 5
        self.max_frame_time = 0.25
        self.paused = False
        self.dropped_seconds = 0.0
        self.synchronize()

    def synchronize(self):
        self.accumulator = 0.0
        self.inputs.clear()
        self.previous = self.current = self.core.snapshot()
        self.previous_pitch = self.current_pitch = self.inputs.pitch

    def set_paused(self, paused):
        self.paused = paused
        self.synchronize()

    def reset(self):
        self.core.reset(self.current.seed)
        self.inputs.pitch = 0.0
        self.synchronize()

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
        turn = (after.yaw - before.yaw + pi) % (2 * pi) - pi
        camera = replace(after, x=x, y=y, yaw=(before.yaw + turn * alpha) % (2 * pi))
        return replace(self.current, survivor=camera)
