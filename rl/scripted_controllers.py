"""Model-free catalogue selector, shared by evaluation and dev checks."""

from types import SimpleNamespace

from ai.controller import ControllerState, choose_target
from rl.candidates import build_candidates, search_is_restricted
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
        if search_is_restricted(view):
            return 21
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
        )
        if self.state.source == "visible":
            action = 20
        elif self.state.source == "last_seen":
            action = 21
        else:
            action = self.state.patrol_index if view.waypoints else WAIT
        if not build_candidates(view)[action].valid:
            return WAIT
        return action
