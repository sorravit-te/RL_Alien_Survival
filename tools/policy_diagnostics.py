"""Read-only frozen-policy diagnostics using a verified original runtime export.

Run in a fresh process: current rl/game/ai modules must not already be imported.
Truth snapshots are read only AFTER action selection, never passed to a selector.
"""

import argparse
import csv
import gzip
import hashlib
import importlib
import json
import random
import sys
from collections import Counter
from dataclasses import asdict
from math import hypot
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
ORIGINAL = REPO / "experiments/naming_review/original_revision"
PILOT = REPO / "experiments/2c/pilot_20261024"
REPEAT_SEEDS = (1103, 2207, 3301)
VALIDATION_MODES = (
    "random",
    "untrained_deterministic",
    "untrained_stochastic",
    "best_deterministic",
    "best_stochastic",
    "last_deterministic",
    "last_stochastic",
    "heuristic",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_runtime():
    if any(name.split(".")[0] in ("rl", "game", "ai") for name in sys.modules):
        raise RuntimeError("Use a fresh process; refusing to mix renamed and original runtimes")
    sys.path.insert(0, str(ORIGINAL))
    pipeline = importlib.import_module("rl.pipeline")
    evaluation = importlib.import_module("rl.evaluate")
    candidates = importlib.import_module("rl.candidates")
    assert Path(pipeline.__file__).resolve() == ORIGINAL / "rl/pipeline.py"
    config, manifest, fingerprint = pipeline.read_run(PILOT)
    torch.set_num_threads(config["torch_threads"])
    models, metadata = {}, {}
    for kind in ("untrained", "best", "last"):
        models[kind], metadata[kind] = pipeline.load_checkpoint(
            PILOT / f"checkpoints/{kind}.zip", config, fingerprint
        )
    return SimpleNamespace(
        p=pipeline,
        e=evaluation,
        c=candidates,
        config=config,
        manifest=manifest,
        source=fingerprint,
        models=models,
        metadata=metadata,
    )


def protected_hashes():
    # Opaque integrity hashes only: historical test outcomes are never parsed or analyzed.
    paths = [p for folder in (PILOT, ORIGINAL) for p in folder.rglob("*") if p.is_file()]
    paths += [REPO / "experiments/naming_review/original_revision.zip"]
    return {str(p.relative_to(REPO)): sha(p) for p in sorted(paths)}


def jobs(cases):
    for split, modes in (
        ("validation", VALIDATION_MODES),
        ("train", ("best_deterministic", "last_deterministic")),
    ):
        for mode in modes:
            for case in cases[split]:
                repeats = range(3) if mode.endswith("_stochastic") else range(1)
                for repeat in repeats:
                    seed = (
                        int(
                            np.random.SeedSequence(
                                [case["selector_seed"], REPEAT_SEEDS[repeat]]
                            ).generate_state(1)[0]
                        )
                        if mode.endswith("_stochastic")
                        else case["selector_seed"]
                    )
                    yield split, mode, case, repeat, seed


def prepare(output):
    runtime = original_runtime()
    if output.parent.resolve() != (REPO / "experiments/policy_diagnostics").resolve():
        raise ValueError("Output must be a new named directory in experiments/policy_diagnostics")
    cases = {
        "validation": runtime.manifest["splits"]["validation"],
        "train": runtime.manifest["splits"]["train"][:50],
    }
    assert len(cases["validation"]) == len(cases["train"]) == 50
    planned = [
        {"split": s, "mode": m, "case_id": c["id"], "repeat": r, "selector_seed": seed}
        for s, m, c, r, seed in jobs(cases)
    ]
    assert len(planned) == 800
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "version": "frozen_policy_diagnostics_v1",
        "source": runtime.source,
        "config": runtime.config,
        "cases": cases,
        "jobs": planned,
        "expected_episodes": 800,
        "original_config_sha256": sha(PILOT / "config.json"),
        "original_manifest_sha256": sha(PILOT / "manifest.json"),
        "original_export": str(ORIGINAL),
        "dependencies": runtime.p.dependencies(),
        "checkpoints": runtime.metadata,
        "driver_sha256": sha(__file__),
        "reporter_sha256": sha(REPO / "tools/policy_diagnostic_report.py"),
        "repeat_seeds": REPEAT_SEEDS,
        "training_seeds": 1,
        "sampling": "Three fixed sampling repeats per stochastic case; NOT training seeds.",
        "metrics": [
            "catch/escape/unresolved",
            "return",
            "length",
            "masked entropy/probabilities",
            "actions/wait/clue selection",
            "target repeat/hold/switch/ABAB",
            "net movement/target progress",
            "logger-only actor trajectories",
        ],
        "definitions": {
            "hold": "target distance <= arrival distance and net displacement <1e-6",
            "stalled": "target distance > arrival distance and net displacement <1e-6, nonterminal",
            "ABAB": "four successive actions A,B,A,B where A != B; a switching indicator, not proof of a loop",
            "survivor_toward_alien": "Survivor net displacement projected toward PRE-step Alien position; descriptive geometry only",
            "uncertainty": "Paired case-cluster bootstrap; retain all three sampling repeats within a case.",
        },
        "example_selection": [
            "first validation case where best deterministic fails and best stochastic repeat0 catches",
            "first validation case where random catches and best deterministic fails",
            "first captured episode with stationary Alien and positive Survivor approach",
        ],
        "boundary": "Selectors get only allowed observation/mask/view. Truth reads occur after action is fixed; logger data never feeds policy/bot.",
        "protected_before": protected_hashes(),
    }
    runtime.p.write_json(output / "protocol.json", protocol)
    runtime.p.write_json(output / "protocol_digest.json", {"sha256": runtime.p.digest(protocol)})
    print(
        f"PREPARED {output} : 800 episodes, fixed train/validation only; no test evaluation.",
        flush=True,
    )


def policy_details(model, obs, mask):
    # Read after selection; feature/probability inspection must consume no random samples.
    state = torch.get_rng_state().clone()
    with torch.no_grad():
        tensor = model.policy.obs_to_tensor(obs)[0]
        distribution = model.policy.get_distribution(tensor, action_masks=mask).distribution
        probabilities = distribution.probs[0].cpu().numpy().astype(float)
        entropy = float(distribution.entropy()[0].cpu())
    assert torch.equal(state, torch.get_rng_state())
    assert np.isclose(probabilities.sum(), 1.0) and np.all(probabilities[~mask] < 1e-7)
    return probabilities, entropy


def run_episode(runtime, mode, case, seed, *, trace=None, inspect_policy=True):
    env = runtime.p.make_env(runtime.config)
    model = runtime.models[mode.split("_")[0]] if "_" in mode else None
    selector = runtime.e.HeuristicSelector(env.config.target_arrival_distance)
    rng = np.random.default_rng(seed)
    counts = Counter()
    actions = []
    rows = []
    totals = Counter()
    reward_parts = Counter()
    action_digest = hashlib.sha256()
    start = perf_counter()
    try:
        with runtime.e.frozen_evaluation(model):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            obs, _ = env.reset(
                seed=case["reset_seed"],
                options={"scenario_seed": case["scenario_seed"], "bot_seed": case["bot_seed"]},
            )
            spawn = None
            previous_target = None
            while True:
                mask = env.action_masks()
                view = env._allowed_view()
                candidates = runtime.c.build_candidates(view)
                if mode == "random":
                    action = int(rng.choice(np.flatnonzero(mask)))
                elif mode == "heuristic":
                    action = selector.choose(view)
                else:
                    action, _ = model.predict(
                        obs, action_masks=mask, deterministic=mode.endswith("_deterministic")
                    )
                    action = int(action)
                assert mask[action]
                # Everything below is logger-only. Action is fixed before truth is accessed.
                if model is not None and inspect_policy:
                    probabilities, entropy = policy_details(model, obs, mask)
                else:
                    probabilities = (
                        mask.astype(float) / mask.sum() if mode == "random" else np.eye(32)[action]
                    )
                    entropy = float(
                        -np.sum(
                            probabilities[probabilities > 0]
                            * np.log(probabilities[probabilities > 0])
                        )
                    )
                before = env.core.snapshot()
                if spawn is None:
                    spawn = [before.survivor.x, before.survivor.y]
                target = candidates[action].target
                saved_obs = obs.copy()
                obs, reward, terminated, truncated, info = env.step(action)
                after = env.core.snapshot()
                assert not info["invalid_action"] and info["applied_action"] == action
                a0 = (before.alien.x, before.alien.y)
                a1 = (after.alien.x, after.alien.y)
                s0 = (before.survivor.x, before.survivor.y)
                s1 = (after.survivor.x, after.survivor.y)
                amove, smove = (
                    hypot(a1[0] - a0[0], a1[1] - a0[1]),
                    hypot(s1[0] - s0[0], s1[1] - s0[1]),
                )
                distance = hypot(s0[0] - a0[0], s0[1] - a0[1])
                approach = (
                    (s1[0] - s0[0]) * (a0[0] - s0[0]) + (s1[1] - s0[1]) * (a0[1] - s0[1])
                ) / max(distance, 1e-12)
                target_distance = hypot(target[0] - a0[0], target[1] - a0[1]) if target else None
                progress = (
                    target_distance - hypot(target[0] - a1[0], target[1] - a1[1]) if target else 0.0
                )
                counts["decisions"] += 1
                counts[f"action_{action}"] += 1
                counts["wait"] += action == 30
                counts["visible_available"] += bool(mask[20])
                counts["other_while_visible"] += bool(mask[20]) and action not in (20, 21)
                counts["clue"] += action in (20, 21)
                counts["visible"] += action == 20
                counts["last_seen"] += action == 21
                counts["repeat_action"] += bool(actions) and action == actions[-1]
                counts["repeat_target"] += target is not None and target == previous_target
                counts["target_switch"] += (
                    target is not None and previous_target is not None and target != previous_target
                )
                counts["hold"] += (
                    target is not None
                    and target_distance <= env.config.target_arrival_distance
                    and amove < 1e-6
                )
                counts["stalled"] += (
                    target is not None
                    and target_distance > env.config.target_arrival_distance
                    and amove < 1e-6
                    and not (terminated or truncated)
                )
                actions.append(action)
                counts["alternation"] += (
                    len(actions) >= 4
                    and actions[-4] == actions[-2]
                    and actions[-3] == actions[-1]
                    and actions[-1] != actions[-2]
                )
                counts["stationary_capture"] += (
                    info["outcome"] == "captured" and amove < 1e-8 and approach > 1e-6
                )
                totals.update(
                    {
                        "return": reward,
                        "entropy_sum": entropy,
                        "max_probability_sum": float(probabilities.max()),
                        "selected_probability_sum": float(probabilities[action]),
                        "alien_net_distance": amove,
                        "survivor_net_distance": smove,
                        "target_progress": progress,
                        "survivor_approach": approach,
                    }
                )
                reward_parts.update(info["reward_components"])
                action_digest.update(bytes([action]))
                record = {
                    "decision": len(actions),
                    "action": action,
                    "observation": saved_obs.tolist(),
                    "allowed_view": asdict(view),
                    "candidates": [asdict(c) for c in candidates],
                    "mask": mask.tolist(),
                    "probabilities": probabilities.tolist(),
                    "entropy": entropy,
                    "argmax": int(probabilities.argmax()),
                    "target": target,
                    "net_alien_movement": amove,
                    "net_survivor_movement": smove,
                    "target_progress": progress,
                    "survivor_approach": approach,
                    "reward_components": info["reward_components"],
                    "outcome": info["outcome"],
                    "step_ticks": info["step_ticks"],
                    "logger_truth": {
                        "before_alien": a0,
                        "after_alien": a1,
                        "before_survivor": s0,
                        "after_survivor": s1,
                        "tick": after.tick,
                        "has_keycard": after.has_keycard,
                        "perception": asdict(after.perception),
                        "navigation_path": after.navigation_path,
                    },
                }
                if trace is not None:
                    trace.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                if inspect_policy:
                    rows.append({"action": action, "mask_valid": bool(mask[action])})
                previous_target = target
                if terminated or truncated:
                    break
        result = {
            "id": case["id"],
            "mode": mode,
            "selector_seed": seed,
            "spawn": spawn,
            "scenario_seed": case["scenario_seed"],
            "bot_seed": case["bot_seed"],
            "outcome": info["outcome"],
            "elapsed_ticks": info["elapsed_ticks"],
            "action_sha256": action_digest.hexdigest(),
            **counts,
            **totals,
            "reward_components": dict(reward_parts),
            "wall_seconds": perf_counter() - start,
        }
        return result, rows
    finally:
        env.close()


def self_check():
    runtime = original_runtime()
    case = runtime.manifest["splits"]["validation"][0]
    model = runtime.models["best"]
    before = runtime.p.parameter_hash(model)
    py, npstate, tstate = random.getstate(), np.random.get_state(), torch.get_rng_state()
    import io

    buffer = io.StringIO()
    first, _ = run_episode(runtime, "best_stochastic", case, 704, trace=buffer)
    second, _ = run_episode(runtime, "best_stochastic", case, 704)
    plain, _ = run_episode(runtime, "best_stochastic", case, 704, inspect_policy=False)
    for row in (second, plain):
        for key in ("action_sha256", "outcome", "elapsed_ticks", "return"):
            assert first[key] == row[key], key
    assert before == runtime.p.parameter_hash(model)
    assert py == random.getstate() and np.array_equal(npstate[1], np.random.get_state()[1])
    assert torch.equal(tstate, torch.get_rng_state())
    records = [json.loads(line) for line in buffer.getvalue().splitlines()]
    assert all(r["mask"][r["action"]] and len(r["observation"]) == 352 for r in records)
    assert "pygame" not in sys.modules
    print(
        json.dumps(
            {
                "seeded_replay": True,
                "tracing_preserves_actions_and_outcome": True,
                "weights_unchanged": True,
                "global_rng_restored": True,
                "masks_valid": True,
                "decisions": first["decisions"],
                "outcome": first["outcome"],
            }
        ),
        flush=True,
    )


def run(output):
    runtime = original_runtime()
    protocol = runtime.p.read_json(output / "protocol.json")
    assert (
        runtime.p.digest(protocol) == runtime.p.read_json(output / "protocol_digest.json")["sha256"]
    )
    assert protocol["driver_sha256"] == sha(__file__)
    assert protocol["reporter_sha256"] == sha(REPO / "tools/policy_diagnostic_report.py")
    assert protocol["protected_before"] == protected_hashes()
    with (output / "started.txt").open("x", encoding="utf-8") as handle:
        handle.write("Single diagnostic run; no automatic reruns.\n")
    with runtime.p.live_log(output / "run.log"):
        start = perf_counter()
        episodes = []
        before_weights = {k: runtime.p.parameter_hash(m) for k, m in runtime.models.items()}
        trace_dir = output / "traces"
        trace_dir.mkdir()
        with (output / "episodes.jsonl").open("x", encoding="utf-8", buffering=1) as results:
            for index, (split, mode, case, repeat, seed) in enumerate(jobs(protocol["cases"])):
                planned = protocol["jobs"][index]
                assert planned == {
                    "split": split,
                    "mode": mode,
                    "case_id": case["id"],
                    "repeat": repeat,
                    "selector_seed": seed,
                }
                trace_path = trace_dir / f"{split}_{mode}_{case['id']}_repeat{repeat}.jsonl.gz"
                with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
                    row, _ = run_episode(runtime, mode, case, seed, trace=handle)
                row.update(split=split, repeat=repeat, trace=str(trace_path.relative_to(output)))
                episodes.append(row)
                results.write(json.dumps(row, allow_nan=False) + "\n")
                results.flush()
                if (index + 1) % 25 == 0 or index + 1 == 800:
                    subset = [r for r in episodes if r["split"] == split and r["mode"] == mode]
                    outcomes = Counter(r["outcome"] for r in subset)
                    print(
                        f"DIAGNOSTIC {index + 1}/800 {split}/{mode} n={len(subset)} "
                        f"Catch/Escape/Unresolved={outcomes['captured']}/{outcomes['escaped']}/{outcomes['unresolved']} "
                        f"wall={perf_counter() - start:.1f}s",
                        flush=True,
                    )
        assert len(episodes) == 800 and protocol["protected_before"] == protected_hashes()
        assert before_weights == {k: runtime.p.parameter_hash(m) for k, m in runtime.models.items()}
        scalar_fields = [
            k for k, v in episodes[0].items() if isinstance(v, (str, int, float, bool))
        ]
        with (output / "episodes.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=scalar_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(episodes)
        runtime.p.write_json(
            output / "execution.json",
            {
                "episodes": len(episodes),
                "wall_seconds": perf_counter() - start,
                "weights_unchanged": True,
                "protected_files_unchanged": True,
                "sampling_repeats": 3,
                "training_seeds": 1,
                "physics_training_updates": 0,
                "held_out_test_episodes": 0,
            },
        )
        # Reporter consumes diagnostic outputs only, plus static map from original source.
        from tools.policy_diagnostic_report import report

        report(output, runtime)
        print(
            "COMPLETE: review.txt; no model updates, no held-out test. Stop for review.", flush=True
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "run", "self-check"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "self-check":
        self_check()
    elif args.output is None:
        parser.error("--output is required")
    elif args.phase == "prepare":
        prepare(args.output.resolve())
    else:
        run(args.output.resolve())


if __name__ == "__main__":
    main()
