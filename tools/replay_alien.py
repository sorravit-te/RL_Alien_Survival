"""Deterministic 2A headless evidence; heuristic controller, not RL."""

import argparse
import json
from dataclasses import asdict, replace
from math import hypot, pi
from statistics import mean
from time import perf_counter_ns

from game.core import GameCore
from game.map import MapObject, WorldMap, small_test_map
from game.state import CoreConfig, SurvivorCommand


def record(snapshot):
    return {
        "tick": snapshot.tick,
        "survivor": asdict(snapshot.survivor),
        "alien": asdict(snapshot.alien) if snapshot.alien else None,
        "perception": asdict(snapshot.perception),
        "target": snapshot.alien_target,
        "target_source": snapshot.alien_target_source,
        "outcome": snapshot.outcome,
        "has_keycard": snapshot.has_keycard,
        "events": [asdict(e) for e in snapshot.events],
    }


def run_replays():
    world = replace(
        small_test_map(),
        spawn_points=((5.5, 2.5),),
        alien_spawn=(2.5, 2.5, 0),
        patrol_waypoints=((2.5, 2.5),),
    )
    core = GameCore(world, CoreConfig(run_speed=60))
    trace = []
    for index in range(120):
        command = (
            SurvivorCommand(strafe=1, running=True)
            if index in (5, 8)
            else SurvivorCommand(forward=1, running=True)
            if index in (6, 7)
            else SurvivorCommand()
        )
        trace.append(record(core.step(command)))
    assert any(r["perception"]["visible"] for r in trace)
    assert any(r["target_source"] == "last_seen" for r in trace)
    assert any(
        r["target_source"] == "last_seen"
        and hypot(r["alien"]["x"] - 5.5, r["alien"]["y"] - 2.5) < 0.15
        for r in trace
    )
    assert all(r["perception"]["last_seen"] in (None, (5.5, 2.5)) for r in trace)
    assert all(r["outcome"] is None for r in trace)
    arena = WorldMap(
        ("." * 16,) * 8, ((2.5, 2.5),), alien_spawn=(6.5, 2.5, pi), patrol_waypoints=((2.5, 2.5),)
    )
    core = GameCore(arena, CoreConfig(run_speed=120, alien_search_speed=120, alien_chase_speed=120))
    capture = record(core.step(SurvivorCommand(forward=1, running=True)))
    assert capture["outcome"] == "captured"
    core = GameCore(
        replace(
            arena, objects=(MapObject("a", "keycard", 3.5, 2.5), MapObject("z", "exit", 3.5, 2.5))
        )
    )
    escape = [record(core.step(SurvivorCommand(interact=True))) for _ in range(2)]
    assert escape[-1]["outcome"] == "escaped"
    return {
        "controller": "heuristic test controller - NOT RL",
        "sight_loss_search": {
            "fixture": "same small map; run speed60 only to script rapid concealment",
            "trace": trace,
        },
        "crossing_capture": {"fixture": "open test room, opposing speed120", "state": capture},
        "escape_with_alien": {"fixture": "open test room with adjacent card/exit", "trace": escape},
    }


def benchmark(ticks):
    core = GameCore()
    durations = []
    resets = 0
    for _ in range(ticks):
        if core.snapshot().outcome:
            core.reset(0)
            resets += 1
        start = perf_counter_ns()
        core.step(SurvivorCommand())
        durations.append((perf_counter_ns() - start) / 1e6)
    ordered = sorted(durations)
    return {
        "mode": "active default Core.step; headless; no rendering; reset costs excluded",
        "ticks": ticks,
        "resets": resets,
        "mean_ms": mean(durations),
        "p95_ms": ordered[int((len(ordered) - 1) * 0.95)],
        "max_ms": max(durations),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-ticks", type=int, default=0)
    args = parser.parse_args()
    if args.benchmark_ticks < 0:
        parser.error("benchmark-ticks must be nonnegative")
    report = run_replays()
    if args.benchmark_ticks:
        report["benchmark"] = benchmark(args.benchmark_ticks)
    print(json.dumps(report, indent=2))
