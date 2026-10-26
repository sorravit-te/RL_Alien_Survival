"""Bounded Wait/Heuristic DEV fixture check. Never import/load/predict a model.

The persistent ledger charges each attempted Core.step BEFORE execution, including
failures, and every started scenario. Existing ledgers cannot be rerun. Unit tests
for this feature prohibit Core.step and therefore consume zero simulation budget.
"""

import argparse
import hashlib
import importlib.abc
import json
import sys
from collections import Counter
from dataclasses import asdict
from math import hypot
from pathlib import Path

FORBIDDEN = ("torch", "stable_baselines3", "sb3_contrib", "rl.pipeline", "rl.evaluate")


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == n or fullname.startswith(n + ".") for n in FORBIDDEN):
            raise RuntimeError("Model imports prohibited in fixture checks: " + fullname)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


class BudgetLedger:
    def __init__(self, path, tick_cap=21600, execution_cap=24, local_cap=900):
        self.path = Path(path)
        # Exclusive create: restarting must not erase prior attempts or restore budget.
        self.stream = self.path.open("x", encoding="utf8")
        self.ticks = self.executions = 0
        self.tick_cap, self.execution_cap = tick_cap, execution_cap
        self.local_cap = local_cap
        self.active = None
        self.charge(
            "limits",
            tick_cap=tick_cap,
            execution_cap=execution_cap,
            unit_test_ticks=0,
            local_cap=local_cap,
        )

    def charge(self, kind, **fields):
        self.stream.write(json.dumps({"kind": kind, **fields}) + "\n")
        self.stream.flush()

    def begin(self, case_id, controller):
        if self.active is not None or self.executions >= self.execution_cap:
            raise RuntimeError("Execution budget exhausted or previous execution unfinished")
        self.executions += 1
        self.active = (case_id, controller)
        self.local_ticks = 0
        self.charge(
            "execution_start", number=self.executions, case_id=case_id, controller=controller
        )

    def tick(self):
        if self.active is None or self.ticks >= self.tick_cap or self.local_ticks >= self.local_cap:
            raise RuntimeError("Core tick budget exhausted or no active scenario")
        self.ticks += 1
        self.local_ticks += 1
        self.charge(
            "core_tick_attempt",
            total=self.ticks,
            local=self.local_ticks,
            case_id=self.active[0],
            controller=self.active[1],
        )

    def end(self, outcome):
        self.charge(
            "execution_end",
            case_id=self.active[0],
            controller=self.active[1],
            ticks=self.local_ticks,
            outcome=outcome,
        )
        self.active = None

    def close(self):
        self.stream.close()


def summarize_execution(case, controller, ticks, decisions, world, config):
    from game.motion import segment_has_clearance

    spans, open_span = [], None
    was_visible = False
    first_sight = None
    for row in ticks:
        memory = row["perception"]
        visible, t = memory["visible"], row["tick_before"]
        if visible and first_sight is None:
            first_sight = t
        if not visible and was_visible:
            clue = memory["last_seen"]
            distances = [hypot(clue[0] - p[0], clue[1] - p[1]) for p in world.patrol_waypoints]
            nearest = min(range(len(distances)), key=distances.__getitem__)
            clear = segment_has_clearance(
                world, tuple(clue), world.patrol_waypoints[nearest], config.alien_radius
            )
            distance = distances[nearest]
            alignment = (
                "arrival_waypoint"
                if clear and distance <= config.target_arrival_distance
                else "near_waypoint"
                if clear and distance <= 1.0
                else "offset"
            )
            open_span = {
                "loss_sensor_tick": t,
                "available_tick": t + 1,
                "clue": clue,
                "revision": memory["revision"],
                "nearest_waypoint_distance": distance,
                "realised_alignment": alignment,
            }
        if visible and not was_visible and open_span is not None:
            open_span.update(
                end_sensor_tick=t, ending="reacquired", reacquired_revision=memory["revision"]
            )
            spans.append(open_span)
            open_span = None
        was_visible = visible
    outcome = ticks[-1]["outcome"] or "unresolved"
    if open_span is not None:
        open_span.update(end_sensor_tick=len(ticks), ending=outcome, reacquired_revision=None)
        spans.append(open_span)
    for span in spans:
        eligible = [
            d
            for d in decisions
            if span["available_tick"] <= d["tick"] <= span["end_sensor_tick"]
            and not d["view"]["visible"]
            and d["view"]["last_seen"] is not None
        ]
        span["loss_duration_ticks"] = span["end_sensor_tick"] - span["loss_sensor_tick"]
        span["first_informed_decision_tick"] = eligible[0]["tick"] if eligible else None
        span["available_to_decision_ticks"] = (
            eligible[0]["tick"] - span["available_tick"] if eligible else None
        )
        span["loss_informed_decisions"] = len(eligible)
        span["actions"] = [d["action"] for d in eligible]
    known_times = {}
    pickups, escapes, arrivals = [], [], []
    previous_waypoint_inside = set()
    wait_runs, wait_start = [], None
    transitions = []
    previous_inv = {"phase": "idle"}
    for row in ticks:
        for key in row["known_objects"]:
            known_times.setdefault(key, row["tick_before"])
        for e in row["events"]:
            if e["kind"] == "keycard_collected":
                pickups.append(row["tick_after"])
            elif e["kind"] == "escaped":
                escapes.append(row["tick_after"])
        action = row["action"]
        inside = set()
        if action < len(world.patrol_waypoints):
            p = world.patrol_waypoints[action]
            a = row["alien_after"]
            if hypot(a["x"] - p[0], a["y"] - p[1]) <= config.target_arrival_distance:
                inside.add(action)
                if action not in previous_waypoint_inside:
                    arrivals.append({"waypoint": action, "tick_after": row["tick_after"]})
        previous_waypoint_inside = inside
        if action == 30 and wait_start is None:
            wait_start = row["tick_before"]
        if action != 30 and wait_start is not None:
            wait_runs.append(row["tick_before"] - wait_start)
            wait_start = None
        # Account separately for cancellation by action assignment and by sensor update.
        for stage, current in (
            ("action_assignment", row["investigation_before"]),
            ("sensor_motion", row["investigation_after"]),
        ):
            old, new = previous_inv["phase"], current["phase"]
            if old != new:
                transitions.append(
                    {
                        "tick_before": row["tick_before"],
                        "stage": stage,
                        "from": old,
                        "to": new,
                        "revision": current.get("revision"),
                    }
                )
            previous_inv = current
    if wait_start is not None:
        wait_runs.append(len(ticks) - wait_start)
    wp_choices = [d["action"] for d in decisions if d["action"] < 20]
    compressed = [a for i, a in enumerate(wp_choices) if i == 0 or wp_choices[i - 1] != a]
    revisits = len(compressed) - len(set(compressed))
    arrival_counts = Counter(a["waypoint"] for a in arrivals)
    return {
        "case_id": case.case_id,
        "profile": case.profile,
        "intended_alignment": case.intended_alignment,
        "controller": controller,
        "scenario_seed": case.scenario_seed,
        "bot_seed": case.bot_seed,
        "ticks": len(ticks),
        "seconds": len(ticks) / config.tick_hz,
        "outcome": outcome,
        "card_discovery_tick": known_times.get("keycard"),
        "exit_discovery_tick": known_times.get("exit"),
        "card_pickup_ticks": pickups,
        "escape_ticks": escapes,
        "first_confirmed_sensor_tick": first_sight,
        "no_sight": first_sight is None,
        "no_loss": not spans,
        "no_card": not pickups,
        "timeout": outcome == "unresolved",
        "capture_before_loss": outcome == "captured" and not spans,
        "loss_windows": spans,
        "reacquisitions": sum(s["ending"] == "reacquired" for s in spans),
        "clue_alignment_counts": dict(Counter(s["realised_alignment"] for s in spans)),
        "decisions": len(decisions),
        "action_counts": dict(Counter(d["action"] for d in decisions)),
        "action21_eligible_decisions": sum(d["masks"][21] for d in decisions),
        "action21_chosen": sum(d["action"] == 21 for d in decisions),
        "investigation_transitions": transitions,
        "sweep_starts": sum(t["to"] == "sweeping" for t in transitions),
        "sweep_completions": sum(t["to"] == "completed" for t in transitions),
        "investigation_cancellations": sum(
            t["from"] in ("travelling", "sweeping") and t["to"] in ("idle", "cancelled")
            for t in transitions
        ),
        "waypoint_arrivals": arrivals,
        "waypoint_arrival_revisits": sum(max(0, n - 1) for n in arrival_counts.values()),
        "nonconsecutive_waypoint_reselections": revisits,
        "wait_ticks": sum(wait_runs),
        "longest_wait_ticks": max(wait_runs, default=0),
        "wait_runs_ticks": wait_runs,
        "conditional_metrics_status": "applicable" if spans else "not_applicable_no_loss",
        "post_loss_choices_status": "applicable"
        if any(s["loss_informed_decisions"] for s in spans)
        else "not_applicable_no_loss_choice",
    }


def run_checks(output):
    sys.meta_path.insert(0, NoModels())
    from game.core import GameCore
    from rl.contract import WAIT
    from rl.dev_fixtures import DEFAULT_MANIFEST, FixtureMissionBot, read_fixtures
    from rl.env import AlienEnv
    from rl.scripted_controllers import HeuristicSelector

    output = Path(output)
    frozen_path = output / "frozen_fixtures.json"
    frozen = frozen_path.read_bytes()
    if frozen != DEFAULT_MANIFEST.read_bytes():
        raise RuntimeError("Fixture definitions changed since freeze")
    data, cases = read_fixtures(frozen_path)
    if (
        data["episode_cap_ticks"] != 900
        or data["execution_cap"] != 24
        or data["total_core_tick_cap"] != 21600
    ):
        raise ValueError("Unexpected budget")
    ledger = BudgetLedger(output / "budget_ledger.jsonl")
    results = []
    original_step = GameCore.step
    try:
        for case in cases:
            for controller in ("wait", "heuristic"):
                ledger.begin(case.case_id, controller)
                env = AlienEnv(
                    world_map=case.world(), max_episode_seconds=30, bot_kind="evasive_explorer"
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
                metrics = summarize_execution(
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
            "cap_ticks": 21600,
            "unused_ticks": 21600 - ledger.ticks,
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
    parser.add_argument(
        "--output",
        required=True,
        help="Existing ignored directory containing frozen_fixtures.json; no overwrites/reruns",
    )
    run_checks(parser.parse_args().output)


if __name__ == "__main__":
    main()
