"""Prepare a review-only sampled evaluation protocol and run at most120 dev bot episodes."""

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from math import hypot
from pathlib import Path
from time import perf_counter

import numpy as np

from rl.evaluate import HeuristicSelector, summarize
from rl.evasive_survivor import EvasionConfig
from rl.pipeline import (
    dependencies,
    file_hash,
    live_log,
    make_env,
    make_manifest,
    read_json,
    source_state,
    write_json,
)


def validate_protocol(training, evaluation):
    if (
        training["bot_kind"] != "evasive_explorer"
        or training["evaluation_protocol"] != "configs/sampled_policy_evaluation.json"
    ):
        raise ValueError("Explicit evasive bot and sampled protocol are required")
    EvasionConfig(**training["survivor_threat_settings"])
    if (
        evaluation["primary_inference"] != "sampled_masked_policy"
        or evaluation["secondary_inference"] != "deterministic_argmax_diagnostic"
    ):
        raise ValueError("Sampled inference must be primary, argmax secondary")
    seeds = evaluation["sampling_repeat_seeds"]
    if (
        len(seeds) != 3
        or len(set(seeds)) != 3
        or any(not isinstance(s, int) or s < 0 for s in seeds)
    ):
        raise ValueError("Exactly3 distinct sampling repeat seeds required")
    expected = {
        "random_valid_mask": ("uniform_valid", 3),
        "untrained_sampled": ("sampled_masked", 3),
        "untrained_deterministic": ("argmax_masked", 1),
        "heuristic": ("scripted_catalog", 1),
        "wait": ("constant_wait", 1),
        "trained_sampled": ("sampled_masked", 3),
        "trained_deterministic": ("argmax_masked", 1),
    }
    if (
        len(evaluation["controllers"]) != len(expected)
        or {c["name"]: (c["inference"], c["repeats"]) for c in evaluation["controllers"]}
        != expected
    ):
        raise ValueError("Controller/inference/repeat specification mismatch")
    if evaluation["selection"] != {
        "split": "validation",
        "primary": "case_mean_sampled_catch_rate",
        "tie_1": "min_sampled_unresolved_rate",
        "tie_2": "earliest_trained_checkpoint",
        "exclude": "test outcomes and untrained baseline cannot select trained best",
    }:
        raise ValueError("Validation-primary selection rules changed")
    if (
        training["manifest_seed"] != evaluation["manifest_seed"]
        or training["manifest_counts"] != evaluation["manifest_counts"]
    ):
        raise ValueError("Training and evaluation manifests disagree")
    smoke = evaluation["smoke"]
    if (
        smoke["cases"] != 20
        or smoke["controllers"] != ["random", "heuristic", "wait"]
        or smoke["bots"] != ["explorer", "evasive_explorer"]
        or smoke["episodes"] != 120
    ):
        raise ValueError("Only the predeclared120-episode dev smoke is supported")


def prepare(output):
    training = read_json("configs/evasive_training_pilot.json")
    evaluation = read_json("configs/sampled_policy_evaluation.json")
    validate_protocol(training, evaluation)
    future = make_manifest(evaluation["manifest_seed"], evaluation["manifest_counts"])
    dev = make_manifest(
        evaluation["smoke"]["manifest_seed"], {"train": 20, "validation": 1, "test": 1}
    )["splits"]["train"]
    dev = [{**case, "id": f"dev_{i:04d}"} for i, case in enumerate(dev)]
    seed_fields = ("reset_seed", "scenario_seed", "bot_seed", "selector_seed")
    future_seeds = {c[k] for split in future["splits"].values() for c in split for k in seed_fields}
    assert not future_seeds.intersection({c[k] for c in dev for k in seed_fields})
    schedule = {
        split: [
            {
                "case_id": case["id"],
                "sampling_seeds": [
                    int(np.random.SeedSequence([case["selector_seed"], s]).generate_state(1)[0])
                    for s in evaluation["sampling_repeat_seeds"]
                ],
            }
            for case in future["splits"][split]
        ]
        for split in ("validation", "test")
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "training_config.json", training)
    write_json(output / "evaluation_protocol.json", evaluation)
    write_json(output / "future_manifest.json", future)
    write_json(output / "future_sampling_schedule.json", schedule)
    write_json(
        output / "threat_settings.json",
        asdict(EvasionConfig(**training["survivor_threat_settings"])),
    )
    smoke = {
        "config": evaluation["smoke"],
        "cases": dev,
        "episodes": 120,
        "future_evaluation_is_review_only": True,
        "future_cases_disjoint_from_dev": True,
    }
    write_json(output / "smoke_protocol.json", smoke)
    identity = {
        "source": source_state(),
        "dependencies": dependencies(),
        "driver_sha256": file_hash(__file__),
        "input_hashes": {p.name: file_hash(p) for p in output.glob("*.json")},
    }
    write_json(output / "identity.json", identity)
    print(
        "PREPARED review-only future protocol and120 dev episodes; no training or test evaluation.",
        flush=True,
    )


def run(output):
    identity = read_json(output / "identity.json")
    if (
        identity["source"]["source_sha256"] != source_state()["source_sha256"]
        or identity["dependencies"] != dependencies()
        or identity["driver_sha256"] != file_hash(__file__)
    ):
        raise ValueError("Prepared smoke source/dependencies/driver changed")
    if any(file_hash(output / name) != h for name, h in identity["input_hashes"].items()):
        raise ValueError("Frozen smoke/protocol input changed")
    protocol = read_json(output / "smoke_protocol.json")
    training = read_json(output / "training_config.json")
    with (output / "started.txt").open("x", encoding="utf-8") as f:
        f.write("One smoke only; max120 episodes. No training or checkpoint evaluation.\n")
    started = perf_counter()
    rows = []
    traces = output / "traces"
    traces.mkdir()
    with (
        live_log(output / "run.log"),
        (output / "episodes.jsonl").open("x", encoding="utf-8", buffering=1) as episodes,
    ):
        for kind in protocol["config"]["bots"]:
            for controller in protocol["config"]["controllers"]:
                config = {
                    **training,
                    "bot_kind": kind,
                    "max_episode_seconds": protocol["config"]["max_episode_seconds"],
                }
                if kind == "explorer":
                    config.pop("survivor_threat_settings")
                env = make_env(config)
                try:
                    for case in protocol["cases"]:
                        env.reset(
                            seed=case["reset_seed"],
                            options={
                                "scenario_seed": case["scenario_seed"],
                                "bot_seed": case["bot_seed"],
                            },
                        )
                        selector = HeuristicSelector(env.config.target_arrival_distance)
                        rng = np.random.default_rng(case["selector_seed"])
                        total = 0.0
                        decisions = 0
                        counts = Counter()
                        audit_stats = {
                            "ticks": 0,
                            "unsafe_perceived_motion": 0,
                            "capture_ticks": 0,
                            "min_perceived_distance": None,
                            "min_actual_distance": None,
                        }
                        original_step = env.core.step

                        def audited_step(command):
                            # Logger-only: the bot already chose this tick's command.
                            audit = getattr(env.bot, "exit_motion_audit", None)
                            state = original_step(command)
                            for event in state.events:
                                counts[event.kind] += 1
                            if audit is not None:
                                audit_stats["ticks"] += 1
                                distance = hypot(
                                    state.survivor.x - audit["threat"][0],
                                    state.survivor.y - audit["threat"][1],
                                )
                                actual = hypot(
                                    state.survivor.x - state.alien.x,
                                    state.survivor.y - state.alien.y,
                                )
                                audit_stats["unsafe_perceived_motion"] += (
                                    distance < audit["clearance"] - 1e-8
                                )
                                audit_stats["capture_ticks"] += state.outcome == "captured"
                                audit_stats["min_perceived_distance"] = min(
                                    distance,
                                    audit_stats["min_perceived_distance"]
                                    if audit_stats["min_perceived_distance"] is not None
                                    else distance,
                                )
                                audit_stats["min_actual_distance"] = min(
                                    actual,
                                    audit_stats["min_actual_distance"]
                                    if audit_stats["min_actual_distance"] is not None
                                    else actual,
                                )
                            return state

                        env.core.step = audited_step
                        with (traces / f"{kind}_{controller}_{case['id']}.jsonl").open(
                            "x", encoding="utf-8"
                        ) as trace:
                            while True:
                                mask = env.action_masks()
                                action = (
                                    int(rng.choice(np.flatnonzero(mask)))
                                    if controller == "random"
                                    else 30
                                    if controller == "wait"
                                    else selector.choose(env._allowed_view())
                                )
                                assert mask[action]
                                _, reward, terminal, truncated, info = env.step(action)
                                total += reward
                                decisions += 1
                                s = env.core.snapshot()
                                assert not env.world.overlaps(
                                    s.survivor.x, s.survivor.y, env.config.radius
                                )
                                trace.write(
                                    json.dumps(
                                        {
                                            "decision": decisions,
                                            "action": action,
                                            "tick": s.tick,
                                            "survivor": asdict(s.survivor),
                                            "alien": asdict(s.alien),
                                            "has_keycard": s.has_keycard,
                                            "fleeing": getattr(env.bot, "fleeing", False),
                                            "last_seen": getattr(env.bot, "last_seen", None),
                                            "last_seen_age_ticks": getattr(
                                                env.bot, "last_seen_age_ticks", None
                                            ),
                                            "escape_target": getattr(
                                                env.bot, "escape_target", None
                                            ),
                                            "exit_approach_active": getattr(
                                                env.bot, "exit_approach_active", False
                                            ),
                                            "exit_standpoint": getattr(
                                                env.bot, "exit_standpoint", None
                                            ),
                                            "exit_motion_audit": dict(audit_stats),
                                            "outcome": info["outcome"],
                                        }
                                    )
                                    + "\n"
                                )
                                if terminal or truncated:
                                    break
                        env.core.step = original_step
                        state_reason = (
                            (
                                "safe_exit_approach_incomplete"
                                if getattr(env.bot, "exit_approach_active", False)
                                else "no_safe_retreat_route"
                                if getattr(env.bot, "fleeing", False)
                                and env.bot.escape_target is None
                                else "repeated_threat_reencounters"
                                if getattr(env.bot, "flee_entries", 0) > 1
                                else "mission_incomplete"
                            )
                            if truncated
                            else None
                        )
                        row = {
                            "id": case["id"],
                            "bot": kind,
                            "controller": controller,
                            "outcome": info["outcome"],
                            "return": total,
                            "decision_steps": decisions,
                            "elapsed_ticks": info["elapsed_ticks"],
                            "scenario_seed": case["scenario_seed"],
                            "bot_seed": case["bot_seed"],
                            "selector_seed": case["selector_seed"],
                            "truncation_reason": info["truncation_reason"],
                            "behavior_at_cap": state_reason,
                            "flee_entries": getattr(env.bot, "flee_entries", 0),
                            "flee_ticks": getattr(env.bot, "flee_ticks", 0),
                            "no_safe_route_count": getattr(env.bot, "no_safe_route_count", 0),
                            "replans": getattr(env.bot, "replans", 0),
                            "has_keycard": s.has_keycard,
                            "event_counts": dict(counts),
                            "exit_approach_steps": getattr(env.bot, "exit_approach_steps", 0),
                            "exit_approach_cancellations": getattr(
                                env.bot, "exit_approach_cancellations", 0
                            ),
                            "exit_approach_attempts": getattr(env.bot, "exit_approach_attempts", 0),
                            "exit_motion_audit": dict(audit_stats),
                        }
                        rows.append(row)
                        episodes.write(json.dumps(row) + "\n")
                    group = [r for r in rows if r["bot"] == kind and r["controller"] == controller]
                    summary = summarize(group)
                    print(
                        f"SMOKE {len(rows)}/120 {kind}/{controller} Catch/Escape/Unresolved="
                        f"{summary['captured']}/{summary['escaped']}/{summary['unresolved']} wall={perf_counter() - started:.1f}s",
                        flush=True,
                    )
                finally:
                    env.close()
        assert len(rows) == 120
        summary = {
            f"{kind}/{controller}": {
                **summarize(
                    [r for r in rows if r["bot"] == kind and r["controller"] == controller]
                ),
                "truncation_reasons": dict(
                    Counter(
                        r["truncation_reason"]
                        for r in rows
                        if r["bot"] == kind
                        and r["controller"] == controller
                        and r["truncation_reason"]
                    )
                ),
                "behavior_at_cap": dict(
                    Counter(
                        r["behavior_at_cap"]
                        for r in rows
                        if r["bot"] == kind
                        and r["controller"] == controller
                        and r["behavior_at_cap"]
                    )
                ),
            }
            for kind in protocol["config"]["bots"]
            for controller in protocol["config"]["controllers"]
        }
        write_json(output / "summary.json", summary)
        with (output / "episodes.csv").open("x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(
            output / "execution.json",
            {
                "episodes": len(rows),
                "wall_seconds": perf_counter() - started,
                "cap_seconds": protocol["config"]["max_episode_seconds"],
                "trained_policy_episodes": 0,
                "training_updates": 0,
                "test_episodes": 0,
                "same_runtime_comparison": True,
                "new_dev_cases_only": True,
            },
        )
        print(
            "COMPLETE dev smoke only; no final policy performance claim. Stop for review.",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "run"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.phase == "prepare" else run)(args.output)


if __name__ == "__main__":
    main()
