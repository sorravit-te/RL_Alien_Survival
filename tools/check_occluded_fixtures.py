"""One frozen DEV batch: alternate placements plus a small opaque-loop blockout.

No policy models/baselines/training. Reuses shared external Env, original bot,
Wait/Heuristic selectors and precharged persistent accounting. No automatic retry.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from math import hypot
from pathlib import Path

from tools.check_lost_sight_fixtures import (
    FORBIDDEN,
    BudgetLedger,
    NoModels,
    summarize_execution,
    write_json,
)


def summarize_inspection(case, controller, ticks, decisions, world, config):
    result = summarize_execution(case, controller, ticks, decisions, world, config)
    result["layout_id"] = case.layout_id
    for span in result["loss_windows"]:
        start, end, revision = span["loss_sensor_tick"], span["end_sensor_tick"], span["revision"]
        rows = ticks[start : min(end + 1, len(ticks))]
        phases = []
        arrival, sweep, completed, travel = None, None, None, None
        for row in rows:
            before, after = row["investigation_before"], row["investigation_after"]
            matching = before.get("revision") == revision or after.get("revision") == revision
            if not matching:
                continue
            t = row["tick_before"]
            if (
                before["phase"] == "travelling"
                and before.get("revision") == revision
                and travel is None
            ):
                travel = t
            a = row["alien_after"]
            clue = span["clue"]
            if (
                arrival is None
                and not row["navigation_path"]
                and hypot(a["x"] - clue[0], a["y"] - clue[1]) <= config.target_arrival_distance
            ):
                arrival = t + 1
            if after["phase"] == "sweeping" and after.get("revision") == revision and sweep is None:
                sweep = t
            if (
                after["phase"] == "completed"
                and after.get("revision") == revision
                and completed is None
            ):
                completed = t
            phases.append(
                {
                    "tick_before": t,
                    "before": before["phase"],
                    "after": after["phase"],
                    "rotation_ticks_after": after.get("rotation_ticks", 0),
                }
            )
        reacquisition_stage = None
        interrupted = False
        if span["ending"] == "reacquired":
            row = ticks[end]
            inv = row["investigation_before"]
            reacquisition_stage = (
                inv["phase"]
                if inv.get("revision") == revision
                and inv["phase"] in ("travelling", "sweeping", "completed")
                else "no_active_inspection"
            )
            interrupted = (
                reacquisition_stage in ("travelling", "sweeping")
                and row["investigation_after"]["phase"] == "cancelled"
                and row["perception"]["visible"]
            )
        span.update(
            travel_start_tick=travel,
            geometric_clue_arrival_snapshot_tick=arrival,
            sweep_start_tick=sweep,
            sweep_complete_tick=completed,
            reacquisition_stage=reacquisition_stage,
            interrupted_by_fresh_confirmed_clue=interrupted,
            inspection_trace=phases,
            inspection_status="swept"
            if sweep is not None
            else "travel_only"
            if travel is not None
            else "not_applicable_no_inspection_chosen",
        )
    result["sweep_reacquisitions"] = sum(
        s["reacquisition_stage"] == "sweeping" for s in result["loss_windows"]
    )
    result["travel_reacquisitions"] = sum(
        s["reacquisition_stage"] == "travelling" for s in result["loss_windows"]
    )
    result["inspection_coverage_ready"] = bool(
        result["sweep_starts"] and (result["sweep_completions"] or result["sweep_reacquisitions"])
    )
    result["stationary_ticks"] = sum(
        hypot(
            row["diagnostic_only_survivor_after"]["x"]
            - row["diagnostic_only_survivor_before"]["x"],
            row["diagnostic_only_survivor_after"]["y"]
            - row["diagnostic_only_survivor_before"]["y"],
        )
        < 1e-10
        for row in ticks
    )
    result["bot_no_safe_route_count"] = ticks[-1]["bot_diagnostic"]["no_safe_route_count"]
    return result


def run_checks(output):
    sys.meta_path.insert(0, NoModels())
    from game.core import GameCore
    from rl.contract import WAIT
    from rl.dev_fixtures import FixtureMissionBot
    from rl.env import AlienEnv
    from rl.occluded_fixtures import DEFAULT_MANIFEST, read_occluded_fixtures
    from rl.scripted_controllers import HeuristicSelector

    output = Path(output)
    frozen_path = output / "frozen_fixtures.json"
    frozen = frozen_path.read_bytes()
    if frozen != DEFAULT_MANIFEST.read_bytes():
        raise RuntimeError("Fixture definitions changed since freeze")
    data, cases = read_occluded_fixtures(frozen_path)
    root = Path(__file__).resolve().parents[1]
    freeze = json.loads((output / "execution_source_freeze.json").read_text())
    for rel, expected in freeze["files"].items():
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Execution source changed after freeze: " + rel)
    if (
        data["episode_cap_ticks"] != 1800
        or data["execution_cap"] != 24
        or data["total_core_tick_cap"] != 43200
    ):
        raise ValueError("Unexpected budget")
    ledger = BudgetLedger(
        output / "budget_ledger.jsonl", tick_cap=43200, execution_cap=24, local_cap=1800
    )
    results = []
    original_step = GameCore.step
    try:
        for case in cases:
            for controller in ("wait", "heuristic"):
                ledger.begin(case.case_id, controller)
                env = AlienEnv(
                    world_map=case.world(), max_episode_seconds=60, bot_kind="evasive_explorer"
                )
                env.bot = FixtureMissionBot(env.world.rows, env.config, case.exploration_anchors)
                env.reset(
                    seed=case.scenario_seed,
                    options={"scenario_seed": case.scenario_seed, "bot_seed": case.bot_seed},
                )
                selector = HeuristicSelector(env.config.target_arrival_distance)
                rows, decisions = [], []
                trace = (output / (case.case_id + "_" + controller + "_ticks.jsonl")).open(
                    "x", encoding="utf8"
                )

                def tracked_step(core, command):
                    if core is not env.core:
                        raise RuntimeError("Unbudgeted Core instance")
                    ledger.tick()
                    before = core.snapshot()
                    after = original_step(core, command)
                    row = {
                        "tick_before": before.tick,
                        "tick_after": after.tick,
                        "action": env.last_action,
                        "alien_before": asdict(before.alien),
                        "alien_after": asdict(after.alien),
                        "diagnostic_only_survivor_before": asdict(before.survivor),
                        "diagnostic_only_survivor_after": asdict(after.survivor),
                        "perception": asdict(after.perception),
                        "investigation_before": asdict(before.investigation),
                        "investigation_after": asdict(after.investigation),
                        "known_objects": sorted(env.bot.explorer.known),
                        "bot_diagnostic": {
                            "fleeing": env.bot.fleeing,
                            "explore_target": env.bot.explorer.explore_target,
                            "flee_target": env.bot.escape_target,
                            "no_safe_route_count": env.bot.no_safe_route_count,
                        },
                        "command": asdict(command),
                        "events": [asdict(e) for e in after.events],
                        "outcome": after.outcome,
                        "target": after.alien_target,
                        "navigation_path": after.navigation_path,
                    }
                    trace.write(json.dumps(row) + "\n")
                    trace.flush()
                    rows.append(row)
                    return after

                GameCore.step = tracked_step
                try:
                    while True:
                        view = env._allowed_view()
                        masks = env.action_masks().tolist()
                        action = WAIT if controller == "wait" else selector.choose(view)
                        if not masks[action]:
                            raise RuntimeError("Scripted selector chose a masked action")
                        decisions.append(
                            {
                                "tick": env.core.snapshot().tick,
                                "allowed_sensor_tick": env.core.snapshot().tick - 1
                                if env.core.snapshot().tick
                                else None,
                                "action": action,
                                "masks": masks,
                                "view": asdict(view),
                            }
                        )
                        _, _, terminal, truncated, _ = env.step(action)
                        if terminal or truncated:
                            break
                finally:
                    GameCore.step = original_step
                    trace.close()
                    env.close()
                metrics = summarize_inspection(
                    case, controller, rows, decisions, env.world, env.config
                )
                write_json(
                    output / (case.case_id + "_" + controller + "_decisions.json"), decisions
                )
                results.append(metrics)
                write_json(output / "case_results.json", results)
                ledger.end(metrics["outcome"])
    finally:
        GameCore.step = original_step
        ledger.close()
    if any(n in sys.modules for n in FORBIDDEN):
        raise RuntimeError("Forbidden model module imported")
    write_json(
        output / "execution_audit.json",
        {
            "scenario_executions": ledger.executions,
            "core_tick_attempts": ledger.ticks,
            "unit_test_core_ticks": 0,
            "cap_ticks": 43200,
            "unused_ticks": 43200 - ledger.ticks,
            "policy_predictions": 0,
            "model_loads": 0,
            "learn_calls": 0,
            "optimizer_updates": 0,
            "held_out_resets": 0,
            "forbidden_imports_present": [],
            "manifest_sha256": hashlib.sha256(frozen).hexdigest(),
            "scope": "This check process only; scripts share AlienEnv catalogue/masks, Core primitives and navigation. Not an RL baseline or policy quality measurement.",
        },
    )
    print(
        json.dumps(
            {
                "executions": ledger.executions,
                "ticks": ledger.ticks,
                "outcomes": dict(Counter(r["outcome"] for r in results)),
                "loss_windows": sum(len(r["loss_windows"]) for r in results),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run_checks(parser.parse_args().output)


if __name__ == "__main__":
    main()
