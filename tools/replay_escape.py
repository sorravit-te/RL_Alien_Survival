"""Deterministic, asset-free 1C headless completion: python -m tools.replay_escape."""

import json
from dataclasses import asdict
from math import pi

from game.core import GameCore
from game.state import CoreConfig, SurvivorCommand


def replay_commands():
    route = []
    for command, ticks in [
        (SurvivorCommand(forward=1), 24),
        (SurvivorCommand(interact=True), 1),
        (SurvivorCommand(forward=1), 12),
        (SurvivorCommand(strafe=1), 24),
        (SurvivorCommand(forward=1), 48),
        (SurvivorCommand(strafe=1), 24),
        (SurvivorCommand(yaw_delta=pi / 2), 1),
        (SurvivorCommand(interact=True), 1),
    ]:
        route.extend([command] * ticks)
    return tuple(route)


def run_replay():
    core = GameCore(config=CoreConfig(alien_enabled=False))
    core.reset(1)
    records, events = [], []
    for command in replay_commands():
        snapshot = core.step(command)
        events.extend(asdict(event) for event in snapshot.events)
        records.append({"command": asdict(command), "tick": snapshot.tick,
                        "survivor": asdict(snapshot.survivor),
                        "has_keycard": snapshot.has_keycard,
                        "outcome": snapshot.outcome})
    if snapshot.outcome != "escaped":
        raise RuntimeError("Headless route failed to escape")
    return {"mode": "headless fixed-tick replay (not keyboard input)", "seed": 1,
            "interaction_mode": "single_press", "alien_enabled": False,
            "outcome": snapshot.outcome, "events": events, "steps": records}


if __name__ == "__main__":
    print(json.dumps(run_replay(), indent=2))
