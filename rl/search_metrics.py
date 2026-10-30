"""Passive episode diagnostics; privileged geometry never enters a policy or reward.

Loss is detected from completed perception transitions, classified using the
pre-motion pose used by that tick's sole native sensor. Only wall-only losses
(with native distance/FOV passing) form the conditional search population.
"""

from copy import deepcopy
from math import atan2, hypot, pi

from game.interaction import segment_clear
from rl.contract import (
    CONTINUE_CANCEL_TRACKING,
    LEGACY_SEARCH_CONTROL,
    restricts_search,
    validate_search_control,
)

VERSION = "wall-search-metrics-v1"


def ratio(numerator, denominator):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def wall_only_loss(before):
    a, s, c = before.alien, before.survivor, before.config
    dx, dy = s.x - a.x, s.y - a.y
    distance = hypot(dx, dy)
    angle = abs((atan2(dy, dx) - a.yaw + pi) % (2 * pi) - pi) if distance else 0
    in_sensor = distance <= c.alien_near_range or (
        distance <= c.alien_sight_range and angle <= c.alien_fov / 2
    )
    return in_sensor and not segment_clear(before.world_map, a.x, a.y, s.x, s.y)


class SearchMetrics:
    def __init__(self, search_control=LEGACY_SEARCH_CONTROL):
        self.search_control = validate_search_control(search_control)
        self.control = None
        if restricts_search(search_control):
            from rl.search_control_metrics import SearchControlMetrics

            self.control = SearchControlMetrics(tracking=search_control == CONTINUE_CANCEL_TRACKING)
        self.losses = 0
        self.windows = []
        self.current = None
        self.previous_target = None
        self.finished = False

    def tick(self, before, after):
        if self.control is not None:
            self.control.tick(before, after)
        if before.perception.visible and not after.perception.visible:
            self.losses += 1
            if wall_only_loss(before):
                self.current = {
                    "loss_tick": before.tick,
                    "snapshot_tick": after.tick,
                    "revision": after.perception.revision,
                    "eligible_choices": 0,
                    "action21_choices": 0,
                    "continuation_opportunities": 0,
                    "same_primitive_continuations": 0,
                    "target_comparisons": 0,
                    "target_switches": 0,
                    "reacquired": False,
                    "cancel_opportunities": 0,
                    "cancelled_on_reacquisition": 0,
                    "end_tick": None,
                    "end_reason": None,
                }
                self.windows.append(self.current)
                self.previous_target = None
        w = self.current
        if w is not None and after.perception.visible:
            w["reacquired"] = True
            if before.investigation.active:
                w["cancel_opportunities"] += 1
                w["cancelled_on_reacquisition"] += int(after.investigation.phase == "cancelled")
            self._end(after.tick, "reacquired")
        if after.outcome is not None and self.current is not None:
            self._end(after.tick, after.outcome)

    def _end(self, tick, reason):
        self.current["end_tick"] = tick
        self.current["end_reason"] = reason
        self.current = None
        self.previous_target = None

    def decision(self, before, after, eligible21, action, *, requested_action=None):
        if self.control is not None:
            self.control.decision(
                before,
                after,
                eligible21,
                action,
                action if requested_action is None else requested_action,
            )
        w = self.current
        if w is None or before.perception.visible:
            return
        target = (after.alien_target_source, after.alien_target)
        if self.previous_target is not None:
            w["target_comparisons"] += 1
            w["target_switches"] += int(target != self.previous_target)
        self.previous_target = target
        if not eligible21:
            return
        w["eligible_choices"] += 1
        w["action21_choices"] += int(action == 21)
        state = before.investigation
        same_clue = (
            state.active
            and state.revision == before.perception.revision
            and state.target == before.perception.last_seen
        )
        if same_clue:
            w["continuation_opportunities"] += 1
            # Selection itself must preserve every primitive field, not restart it.
            w["same_primitive_continuations"] += int(action == 21 and after.investigation == state)

    def finish(self, snapshot):
        if not self.finished:
            if self.current is not None:
                self._end(snapshot.tick, snapshot.outcome or "unresolved")
            self.finished = True
        windows = deepcopy(self.windows)
        for window in windows:
            window["episode_outcome"] = snapshot.outcome or "unresolved"
            window["episode_end_tick"] = snapshot.tick

        def total(key):
            return sum(w[key] for w in windows)

        result = {
            "version": VERSION
            if self.control is None
            else "wall-search-metrics-v3-pursuit-tracking"
            if self.control.tracking
            else "wall-search-metrics-v2-continue-cancel",
            "confirmed_losses": self.losses,
            "wall_loss_opportunities": len(windows),
            "eligible_action21": ratio(total("action21_choices"), total("eligible_choices")),
            "same_primitive_unseen_continuation": ratio(
                total("same_primitive_continuations"), total("continuation_opportunities")
            ),
            "cancel_on_reacquisition": ratio(
                total("cancelled_on_reacquisition"), total("cancel_opportunities")
            ),
            "target_switching": ratio(total("target_switches"), total("target_comparisons")),
            "windows": windows,
        }
        if self.control is not None:
            result["control"] = self.control.finish(snapshot)
        return result


def summarize_search(episodes):
    """Pooled conditional counts plus episode opportunity denominator, never impute N/A."""
    metrics = [e["search_metrics"] for e in episodes]
    result = {
        "episodes": len(metrics),
        "episodes_with_wall_loss": ratio(
            sum(m["wall_loss_opportunities"] > 0 for m in metrics), len(metrics)
        ),
        "wall_loss_opportunities": sum(m["wall_loss_opportunities"] for m in metrics),
        "confirmed_losses": sum(m["confirmed_losses"] for m in metrics),
        "wall_loss_episode_outcomes": {
            outcome: sum(
                m["wall_loss_opportunities"] > 0 and m["windows"][0]["episode_outcome"] == outcome
                for m in metrics
            )
            for outcome in ("captured", "escaped", "unresolved")
        },
    }
    for key in (
        "eligible_action21",
        "same_primitive_unseen_continuation",
        "cancel_on_reacquisition",
        "target_switching",
    ):
        result[key] = ratio(
            sum(m[key]["numerator"] for m in metrics), sum(m[key]["denominator"] for m in metrics)
        )
    controls = [m.get("control") for m in metrics]
    if any(c is not None for c in controls):
        if not all(c is not None for c in controls):
            raise ValueError("Cannot pool legacy and continue/cancel metric populations")
        result["control"] = {
            "version": "continue-cancel-audit-v1",
            "counts": {k: sum(c["counts"][k] for c in controls) for k in controls[0]["counts"]},
            "attempts": sum(len(c["attempts"]) for c in controls),
            "actual_retarget_delays_ticks": [
                r["actual_delay_ticks"]
                for c in controls
                for r in c["retargets"]
                if "actual_delay_ticks" in r
            ],
            "outcomes": {
                name: sum(c["outcome"] == name for c in controls)
                for name in ("captured", "escaped", "unresolved")
            },
        }
    result["note"] = "Pooled conditional counts; case clusters/repeats remain in per-episode rows."
    return result
