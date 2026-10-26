"""Written investigation primitive. Receives confirmed memory, never Survivor truth."""

from dataclasses import dataclass, replace
from math import ceil, pi


@dataclass(frozen=True)
class InvestigationSnapshot:
    phase: str = "idle"  # idle, travelling, sweeping, completed, cancelled
    target: tuple[float, float] | None = None
    revision: int | None = None
    investigated_revision: int | None = None
    rotation_ticks: int = 0
    total_rotation_ticks: int = 0

    @property
    def active(self):
        return self.phase in ("travelling", "sweeping")

    @property
    def progress(self):
        return self.rotation_ticks / self.total_rotation_ticks if self.total_rotation_ticks else 0.0


class LastSeenInvestigation:
    def __init__(self, config):
        self.angle_per_tick = config.investigation_turn_speed / config.tick_hz
        self.total_ticks = max(1, ceil(2 * pi / self.angle_per_tick - 1e-12))
        self.state = InvestigationSnapshot()

    def select(self, target, revision):
        state = self.state
        if revision == state.investigated_revision:
            raise ValueError("This evidence revision has already been investigated")
        if state.active and state.revision == revision and state.target == target:
            return  # A repeated action21 continues the same primitive.
        self.state = InvestigationSnapshot(
            "travelling", target, revision, state.investigated_revision, 0, self.total_ticks
        )

    def cancel(self):
        # Interrupted partial sweeps are discarded, not marked investigated.
        self.state = InvestigationSnapshot(investigated_revision=self.state.investigated_revision)

    def advance(self, memory, *, arrived):
        """Called AFTER this tick's sole pre-motion sensor sample.

        Return (yaw_delta, hold_position). A full rotation is sampled on the NEXT
        tick before completion; confirmed reacquisition always wins that check.
        """
        state = self.state
        if state.phase == "idle":
            return 0.0, False
        if state.phase == "cancelled":
            return 0.0, True
        if memory.visible or memory.revision != state.revision:
            # Preserve the chosen target/revision separately from fresh sensor memory.
            self.state = replace(state, phase="cancelled", rotation_ticks=0)
            return 0.0, True  # Stop the old primitive; do not substitute chase.
        if state.phase == "completed":
            return 0.0, True
        if state.phase == "travelling":
            if not arrived:
                return 0.0, False
            state = replace(state, phase="sweeping")
        if state.rotation_ticks >= state.total_rotation_ticks:
            self.state = replace(state, phase="completed", investigated_revision=state.revision)
            return 0.0, True
        angle = min(2 * pi, (state.rotation_ticks + 1) * self.angle_per_tick) - min(
            2 * pi, state.rotation_ticks * self.angle_per_tick
        )
        self.state = replace(state, rotation_ticks=state.rotation_ticks + 1)
        return angle, True
