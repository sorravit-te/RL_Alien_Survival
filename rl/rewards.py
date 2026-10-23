"""Per-tick event rewards accumulated by the adapter, never inside Game Core."""


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
