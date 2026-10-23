"""Frozen masked evaluation. Heuristic adapts only the policy-allowed view."""

import argparse
import contextlib
import csv
import hashlib
import random
from math import sqrt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ai.controller import ControllerState, choose_target
from rl.contract import WAIT
from rl.pipeline import load_checkpoint, make_env, parameter_hash, read_run, write_json


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
        perception = SimpleNamespace(visible=view.visible, last_seen=view.last_seen)
        self.state = choose_target(
            alien, perception, view.waypoints, self.state, self.arrival_distance
        )
        if self.state.source == "visible":
            return 20
        if self.state.source == "last_seen":
            return 21
        return self.state.patrol_index if view.waypoints else WAIT


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    radius = z * sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize(rows):
    counts = {
        name: sum(r["outcome"] == name for r in rows)
        for name in ("captured", "escaped", "unresolved")
    }
    n = len(rows)
    return {
        "episodes": n,
        **counts,
        "catch_rate": counts["captured"] / n,
        "escape_rate": counts["escaped"] / n,
        "unresolved_rate": counts["unresolved"] / n,
        "catch_wilson95": wilson(counts["captured"], n),
        "mean_return": float(np.mean([r["return"] for r in rows])),
        "mean_decision_steps": float(np.mean([r["decision_steps"] for r in rows])),
        "mean_seconds": float(np.mean([r["elapsed_ticks"] / 30 for r in rows])),
    }


@contextlib.contextmanager
def frozen_evaluation(model):
    """Evaluation must not perturb weights, training mode or training random streams."""
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    training = model.policy.training if model is not None else None
    before = parameter_hash(model) if model is not None else None
    try:
        if model is not None:
            model.policy.set_training_mode(False)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if model is not None:
            model.policy.set_training_mode(training)
            if parameter_hash(model) != before:
                raise RuntimeError("Evaluation changed model parameters")


def evaluate(kind, config, cases, output_csv, *, model=None, label="validation"):
    if kind not in ("random", "untrained", "heuristic", "wait", "trained"):
        raise ValueError("Unknown evaluation controller")
    if kind in ("untrained", "trained") and model is None:
        raise ValueError("A checkpoint is required; no heuristic fallback")
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    fields = [
        "id",
        "scenario_seed",
        "bot_seed",
        "outcome",
        "return",
        "decision_steps",
        "elapsed_ticks",
        "capture",
        "escape",
        "first_sight",
        "action_sha256",
    ]
    env = make_env(config)  # Never reuse/reset the training environment.
    try:
        with frozen_evaluation(model), path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, case in enumerate(cases):
                obs, _ = env.reset(
                    seed=case["reset_seed"],
                    options={"scenario_seed": case["scenario_seed"], "bot_seed": case["bot_seed"]},
                )
                selector = HeuristicSelector(env.config.target_arrival_distance)
                rng = np.random.default_rng(case["selector_seed"])
                total, decisions = 0.0, 0
                components = {"capture": 0.0, "escape": 0.0, "first_sight": 0.0}
                actions = hashlib.sha256()
                while True:
                    mask = env.action_masks()
                    if kind == "random":
                        action = int(rng.choice(np.flatnonzero(mask)))
                    elif kind == "wait":
                        action = WAIT
                    elif kind == "heuristic":
                        action = selector.choose(env._allowed_view())
                    else:
                        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                        action = int(action)
                    if not mask[action]:
                        raise RuntimeError(f"{kind} selected a masked action {action}")
                    actions.update(bytes([action]))
                    obs, reward, terminated, truncated, info = env.step(action)
                    if info["invalid_action"]:
                        raise RuntimeError("Action mask and applied action disagree")
                    total += reward
                    decisions += 1
                    for name, value in info["reward_components"].items():
                        components[name] += value
                    if terminated or truncated:
                        break
                row = {
                    "id": case["id"],
                    "scenario_seed": case["scenario_seed"],
                    "bot_seed": case["bot_seed"],
                    "outcome": info["outcome"],
                    "return": total,
                    "decision_steps": decisions,
                    "elapsed_ticks": info["elapsed_ticks"],
                    **components,
                    "action_sha256": actions.hexdigest(),
                }
                rows.append(row)
                writer.writerow(row)
                handle.flush()
                if (index + 1) % 10 == 0 or index + 1 == len(cases):
                    s = summarize(rows)
                    print(
                        f"EVAL {label:12s} {kind:9s} episodes={index + 1}/{len(cases)} "
                        f"return={s['mean_return']:.3f} Catch/Escape/Unresolved="
                        f"{s['captured']}/{s['escaped']}/{s['unresolved']}",
                        flush=True,
                    )
    finally:
        env.close()
    return summarize(rows), rows


def paired_catch_difference(first, second):
    if [r["id"] for r in first] != [r["id"] for r in second]:
        raise ValueError("Paired comparison requires identical case order")
    delta = np.array(
        [
            int(a["outcome"] == "captured") - int(b["outcome"] == "captured")
            for a, b in zip(first, second, strict=True)
        ]
    )
    rng = np.random.default_rng(704)
    draws = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
    return {
        "difference": float(delta.mean()),
        "paired_bootstrap95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "episodes": len(delta),
        "note": "Episode resampling only; one training seed, same small map.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config, manifest, source = read_run(args.run_dir)
    torch.set_num_threads(config["torch_threads"])
    model, _ = load_checkpoint(args.checkpoint, config, source)
    summary, _ = evaluate(
        "trained", config, manifest["splits"]["validation"], args.output, model=model
    )
    write_json(args.output.with_suffix(".json"), summary)


if __name__ == "__main__":
    main()
