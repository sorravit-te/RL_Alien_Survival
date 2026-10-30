"""Passive waypoint-walk audit. Never used by policy/reward/selection."""

from math import hypot


class PatrolMetrics:
    version = "waypoint-commitment-audit-v1"

    def __init__(self):
        self.counts = dict.fromkeys(
            ("walks", "arrivals", "released_by_clue", "forced_choices", "abandoned"), 0
        )
        self.reached = set()
        self.target = None

    def decision(self, after, action, committed):
        if committed and action < 20:
            self.counts["forced_choices"] += 1
            return
        # Anything else while a walk is open means the commitment failed to hold.
        self.counts["abandoned"] += self.target is not None
        self.target = None
        if action < 20 and after.navigation_path:
            self.counts["walks"] += 1
            self.target = after.alien_target

    def tick(self, before, after):
        if self.target is None:
            return
        reach = after.config.target_arrival_distance
        if hypot(after.alien.x - self.target[0], after.alien.y - self.target[1]) <= reach:
            self.counts["arrivals"] += 1
            self.reached.add(self.target)
            self.target = None
        elif after.perception.visible or after.perception.revision != before.perception.revision:
            self.counts["released_by_clue"] += 1
            self.target = None

    def finish(self):
        return {
            "version": self.version,
            "counts": dict(self.counts),
            "distinct_waypoints_reached": len(self.reached),
            "unfinished_at_end": int(self.target is not None),
        }
