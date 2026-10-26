"""Model-free catalogue selector, shared by evaluation and dev checks."""

from types import SimpleNamespace

from ai.controller import ControllerState, choose_target
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
            return 20
        if self.state.source == "last_seen":
            return 21
        return self.state.patrol_index if view.waypoints else WAIT
