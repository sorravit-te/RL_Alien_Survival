"""Passive explicit-search execution audit. Never used by policy/reward/selection."""

from copy import deepcopy

from rl.contract import DECISION_TICKS


class SearchControlMetrics:
    version = "continue-cancel-audit-v2-pursuit-tracking"

    def __init__(self):
        self.counts = dict.fromkeys(
            (
                "starts",
                "restarts",
                "restricted_choices",
                "continues",
                "policy_cancels",
                "masked_retarget_requests",
                "helper_cancels",
                "helper_after_completion",
                "completed_sweeps",
                "active_but_unrestricted",
                "helper_pursuit_handoffs",
            ),
            0,
        )
        self.seen = set()
        self.events = []
        self.attempt = None
        self.attempts = []
        self.pending = None
        self.retargets = []
        self.finished = False

    def event(self, tick, kind, **data):
        self.events.append({"tick": tick, "kind": kind, **data})

    def end(self, tick, reason, outcome=None):
        if self.attempt is not None:
            self.attempt.update(
                end_tick=tick,
                end_reason=reason,
                outcome=outcome,
                duration_ticks=tick - self.attempt["start_tick"],
            )
            self.attempts.append(self.attempt)
            self.attempt = None

    def decision(self, before, after, eligible21, action, requested_action):
        state, m = before.investigation, before.perception
        restricted = (
            state.active
            and eligible21
            and not m.visible
            and state.revision == m.revision
            and state.target == m.last_seen
        )
        if self.pending is not None and action != 30:
            self.retargets.append(
                {
                    **self.pending,
                    "retarget_tick": before.tick,
                    "actual_delay_ticks": before.tick - self.pending["cancel_tick"],
                    "action": action,
                }
            )
            self.pending = None
        if restricted:
            self.counts["restricted_choices"] += 1
            self.event(
                before.tick,
                "restricted_choice",
                valid_actions=[21, 30],
                requested_action=requested_action,
                applied_action=action,
                revision=state.revision,
                target=state.target,
            )
            if requested_action not in (21, 30):
                self.counts["masked_retarget_requests"] += 1
        elif state.active:
            self.counts["active_but_unrestricted"] += 1
        preserve = action == 21 and state == after.investigation and state.active
        if preserve:
            self.counts["continues"] += 1
        elif action == 21 and after.investigation.active:
            new = after.investigation
            key = (new.revision, new.target)
            restart = key in self.seen
            self.seen.add(key)
            self.end(before.tick, "replaced_by_start")
            self.counts["restarts" if restart else "starts"] += 1
            self.attempt = {
                "start_tick": before.tick,
                "revision": new.revision,
                "target": new.target,
                "restart": restart,
                "sweep_tick": None,
            }
            self.event(
                before.tick,
                "restart" if restart else "start",
                revision=new.revision,
                target=new.target,
            )
        if state.active and action != 21 and after.investigation.phase == "idle":
            reason = (
                "masked_action_fallback"
                if requested_action != action
                else "explicit_wait"
                if action == 30
                else "retarget"
            )
            self.counts["policy_cancels"] += 1
            self.event(
                before.tick,
                "policy_cancel",
                reason=reason,
                revision=state.revision,
                target=state.target,
            )
            self.end(before.tick, "policy_cancel")
            self.pending = {
                "cancel_tick": before.tick,
                "earliest_retarget_tick": before.tick + DECISION_TICKS,
                "mechanical_delay_ticks": DECISION_TICKS,
                "reason": reason,
            }

    def tick(self, before, after):
        b, a = before.investigation, after.investigation
        if self.attempt is not None and b.phase == "travelling" and a.phase == "sweeping":
            self.attempt["sweep_tick"] = after.tick
        if a.phase == "cancelled" and b.phase != "cancelled":
            reason = (
                "confirmed_visible"
                if after.perception.visible
                else "new_revision"
                if after.perception.revision != b.revision
                else "other_public_condition"
            )
            self.counts["helper_cancels" if b.active else "helper_after_completion"] += 1
            self.event(
                after.tick,
                "helper_cancel",
                reason=reason,
                revision=b.revision,
                new_revision=after.perception.revision,
                was_active=b.active,
            )
            self.end(after.tick, reason, after.outcome)
        if (
            after.alien_target_source == "visible_handoff"
            and before.alien_target_source != "visible_handoff"
        ):
            # Written helper chase after reacquisition; never counted as a policy action.
            self.counts["helper_pursuit_handoffs"] += 1
            self.event(after.tick, "helper_pursuit_handoff", revision=after.perception.revision)
        if a.phase == "completed" and b.phase != "completed":
            self.counts["completed_sweeps"] += 1
            self.event(
                after.tick, "completed_sweep", revision=a.revision, rotation_ticks=a.rotation_ticks
            )
            self.end(after.tick, "completed", after.outcome)
        if after.outcome is not None:
            self.end(after.tick, "terminal", after.outcome)

    def finish(self, snapshot):
        if not self.finished:
            self.end(
                snapshot.tick,
                "terminal" if snapshot.outcome else "external_cap",
                snapshot.outcome or "unresolved",
            )
            if self.pending is not None:
                self.retargets.append(
                    {
                        **self.pending,
                        "retarget_tick": None,
                        "end_tick": snapshot.tick,
                        "end_outcome": snapshot.outcome or "unresolved",
                    }
                )
                self.pending = None
            for attempt in self.attempts:
                attempt["episode_outcome"] = snapshot.outcome or "unresolved"
                attempt["episode_end_tick"] = snapshot.tick
            self.finished = True
        return deepcopy(
            {
                "version": self.version,
                "counts": self.counts,
                "events": self.events,
                "attempts": self.attempts,
                "retargets": self.retargets,
                "outcome": snapshot.outcome or "unresolved",
                "episode_end_tick": snapshot.tick,
                "note": "Restrictions and primitive execution are written mechanics; continuation alone is not learned improvement. Actual retarget delay may exceed15ticks if policy voluntarily waits.",
            }
        )
