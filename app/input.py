"""Accumulate device input; consume relative mouse movement once per tick."""

from math import radians

from game.state import SurvivorCommand


class InputState:
    def __init__(self, mouse_sensitivity=0.003, turn_speed=radians(120),
                 vertical_sensitivity=0.003):
        self.mouse_sensitivity = mouse_sensitivity
        self.turn_speed = turn_speed
        self.vertical_sensitivity = vertical_sensitivity  # radians per vertical mouse pixel
        self.pitch_limit = radians(25)
        self.pitch = 0.0  # App-only simulated camera angle; positive = look up
        self.clear()

    def clear(self):
        self.keys = set()
        self.mouse_dx = self.mouse_dy = 0.0
        self.pending_interactions = 0

    def key(self, name, down, repeat=False):
        if name == 'e' and down:
            if repeat:
                return
            if name not in self.keys:
                self.pending_interactions += 1
        if down:
            self.keys.add(name)
        else:
            self.keys.discard(name)

    def mouse(self, dx, dy=0):
        self.mouse_dx += dx
        self.mouse_dy += dy

    def consume(self, dt):
        yaw = self.mouse_dx * self.mouse_sensitivity
        self.pitch = max(-self.pitch_limit, min(
            self.pitch_limit, self.pitch - self.mouse_dy * self.vertical_sensitivity))
        self.mouse_dx = self.mouse_dy = 0.0
        yaw += (('right' in self.keys) - ('left' in self.keys)) * self.turn_speed * dt
        interact = self.pending_interactions > 0
        if interact:
            self.pending_interactions -= 1
        return SurvivorCommand(
            forward=('w' in self.keys) - ('s' in self.keys),
            strafe=('d' in self.keys) - ('a' in self.keys),
            yaw_delta=yaw,
            running='shift_left' in self.keys or 'shift_right' in self.keys,
            interact=interact,
        )
