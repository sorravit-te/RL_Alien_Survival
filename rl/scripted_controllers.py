"""Model-free catalogue selector, shared by evaluation and dev checks."""

from dataclasses import replace
from math import hypot
from types import SimpleNamespace

from ai.controller import ControllerState, choose_target
from rl.candidates import (
    SHIP_SIZED_PATROL,
    build_candidates,
    committed_waypoint,
    search_is_restricted,
)
from rl.contract import WAIT


class HeuristicSelector:
    """Adapt patrol/chase rules to masked target actions using only allowed knowledge.

    Select every 15 physics ticks using the latest completed perception sample.
    The internal game controller samples on the next physics tick, so its sensor
    timing differs even though both use the same target-selection rules.
    """

    def __init__(self, arrival_distance):
        self.state = ControllerState()
        self.arrival_distance = arrival_distance

    def choose(self, view):
        if self.state.target is None and len(view.waypoints) >= SHIP_SIZED_PATROL:
            # Join the loop at the nearest waypoint, so rounds do not all open alike.
            x, y = view.position
            near = min(
                range(len(view.waypoints)),
                key=lambda i: hypot(view.waypoints[i][0] - x, view.waypoints[i][1] - y),
            )
            self.state = replace(self.state, patrol_index=near)
        if search_is_restricted(view):
            return 21
        committed = committed_waypoint(view)
        if committed is not None:
            return committed
        alien = SimpleNamespace(x=view.position[0], y=view.position[1])
        perception = SimpleNamespace(
            visible=view.visible, last_seen=view.last_seen, revision=view.clue_revision
        )
        self.state = choose_target(
            alien,
            perception,
            view.waypoints,
            self.state,
            self.arrival_distance,
            view.investigation.investigated_revision,
            SimpleNamespace(point=view.last_heard, checked=view.heard_checked),
        )
        if self.state.source == "visible":
            action = 20
        elif self.state.source == "last_seen":
            action = 21
        elif self.state.source == "sound":
            action = 22
        else:
            action = self.state.patrol_index if view.waypoints else WAIT
        slots = build_candidates(view)
        if slots[action].valid:
            return action
        for step in range(1, len(view.waypoints)):  # walk on past a waypoint not offered
            index = (self.state.patrol_index + step) % len(view.waypoints)
            if slots[index].valid:
                self.state = replace(self.state, patrol_index=index)
                return index
        return WAIT
