"""Per-tick event rewards accumulated by the adapter, never inside Game Core."""

COVERAGE_REWARD = 0.1
COVERAGE_AGE_SECONDS = 60.0


def coverage_reward(before, after, tick, tick_hz):
    """Small pay for reaching a waypoint nobody checked lately, so patrol stays wide."""
    age = COVERAGE_AGE_SECONDS * tick_hz
    return COVERAGE_REWARD * sum(
        new is not None and new != old and (old is None or tick - old >= age)
        for old, new in zip(before, after, strict=True)
    )


class RewardTracker:
    def __init__(self):
        self.seen_paid = False
        self.terminal_paid = False

    def collect(self, visible, events):
        components = {"capture": 0.0, "escape": 0.0, "first_sight": 0.0}
        if visible and not self.seen_paid:
            components["first_sight"] = 0.2
            self.seen_paid = True
        if not self.terminal_paid:
            for event in events:
                if event.kind in ("captured", "escaped"):
                    components["capture" if event.kind == "captured" else "escape"] = (
                        10.0 if event.kind == "captured" else -10.0
                    )
                    self.terminal_paid = True
                    break
        return components
