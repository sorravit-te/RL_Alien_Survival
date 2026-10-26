"""Frozen masked evaluation. Heuristic adapts only the policy-allowed view."""

import argparse
import contextlib
import csv
import hashlib
import random
from math import sqrt
from pathlib import Path

import numpy as np
import torch

from rl.contract import WAIT
from rl.pipeline import load_checkpoint, make_env, parameter_hash, read_run, write_json
from rl.scripted_controllers import HeuristicSelector as HeuristicSelector


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


def evaluate(
    kind,
    config,
    cases,
    output_csv,
    *,
    model=None,
    label="validation",
    inference=None,
    episode_seeds=None,
    repeat_index=None,
    controller_name=None,
):
    if config.get("evaluation_protocol") and inference is None:
        raise ValueError("Sampled runs require explicit frozen evaluation inference")
    explicit = inference is not None
    inference = inference or "argmax_masked"
    if inference not in (
        "sampled_masked",
        "argmax_masked",
        "uniform_valid",
        "scripted_catalog",
        "constant_wait",
    ):
        raise ValueError("Unsupported inference mode")
    if explicit and (episode_seeds is None or len(episode_seeds) != len(cases)):
        raise ValueError("A frozen per-episode seed schedule is required")
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
    if explicit:
        fields += [
            "controller",
            "inference",
            "repeat_index",
            "reset_seed",
            "selector_seed",
            "sampling_seed",
            "evaluation_role",
        ]
    evaluation_role = (
        "sampled_primary"
        if kind in ("trained", "untrained") and inference == "sampled_masked"
        else "argmax_secondary"
        if kind in ("trained", "untrained") and explicit
        else "baseline"
    )
    env = make_env(config)  # Never reuse/reset the training environment.
    try:
        with (
            frozen_evaluation(model),
            path.open("x" if explicit else "w", newline="", encoding="utf-8") as handle,
        ):
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, case in enumerate(cases):
                seed = episode_seeds[index] if explicit else case["selector_seed"]
                if explicit:
                    # Once per episode, never per decision. Outer context restores all streams.
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                obs, _ = env.reset(
                    seed=case["reset_seed"],
                    options={"scenario_seed": case["scenario_seed"], "bot_seed": case["bot_seed"]},
                )
                selector = HeuristicSelector(env.config.target_arrival_distance)
                rng = np.random.default_rng(seed)
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
                        action, _ = model.predict(
                            obs, action_masks=mask, deterministic=inference != "sampled_masked"
                        )
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
                if explicit:
                    row.update(
                        controller=controller_name,
                        inference=inference,
                        repeat_index=repeat_index,
                        reset_seed=case["reset_seed"],
                        selector_seed=case["selector_seed"],
                        sampling_seed=seed,
                        evaluation_role=evaluation_role,
                    )
                rows.append(row)
                writer.writerow(row)
                handle.flush()
                if (index + 1) % 10 == 0 or index + 1 == len(cases):
                    s = summarize(rows)
                    print(
                        f"EVAL {label:12s} {controller_name or kind} "
                        f"{evaluation_role if explicit else 'legacy_deterministic'} episodes={index + 1}/{len(cases)} "
                        f"return={s['mean_return']:.3f} Catch/Escape/Unresolved="
                        f"{s['captured']}/{s['escaped']}/{s['unresolved']}",
                        flush=True,
                    )
    finally:
        env.close()
    return summarize(rows), rows


def summarize_cases(rows, inference):
    """Retain repeat clusters; never treat action samples as independent training seeds."""
    groups = {}
    for row in rows:
        groups.setdefault(row["id"], []).append(row)
    if not groups:
        raise ValueError("Empty evaluation cases")
    per_case = []
    for case_id, cluster in groups.items():
        per_case.append(
            {
                "id": case_id,
                "repeats": len(cluster),
                **{
                    name: sum(r["outcome"] == name for r in cluster)
                    for name in ("captured", "escaped", "unresolved")
                },
                **{
                    name + "_rate": float(np.mean([r["outcome"] == name for r in cluster]))
                    for name in ("captured", "escaped", "unresolved")
                },
                "case_mean_return": float(np.mean([r["return"] for r in cluster])),
                "case_mean_decision_steps": float(np.mean([r["decision_steps"] for r in cluster])),
            }
        )
    result = {
        "inference": inference,
        "cases": len(groups),
        "episodes": len(rows),
        **{
            name: sum(r["outcome"] == name for r in rows)
            for name in ("captured", "escaped", "unresolved")
        },
        "case_mean_catch_rate": float(np.mean([r["captured_rate"] for r in per_case])),
        "case_mean_escape_rate": float(np.mean([r["escaped_rate"] for r in per_case])),
        "case_mean_unresolved_rate": float(np.mean([r["unresolved_rate"] for r in per_case])),
        "case_mean_return": float(np.mean([r["case_mean_return"] for r in per_case])),
        "case_mean_decision_steps": float(
            np.mean([r["case_mean_decision_steps"] for r in per_case])
        ),
        "per_case": per_case,
        "per_repeat": {
            str(i): summarize([r for r in rows if r["repeat_index"] == i])
            for i in sorted({r["repeat_index"] for r in rows})
        },
        "statistics_unit": "case cluster; three action samples are not three training seeds",
    }
    for repeat in result["per_repeat"].values():
        repeat["episode_mean_return"] = repeat.pop("mean_return")
        repeat["episode_mean_decision_steps"] = repeat.pop("mean_decision_steps")
        repeat["episode_mean_simulation_seconds"] = repeat.pop("mean_seconds")
    if inference == "sampled_masked":
        result["case_mean_sampled_catch_rate"] = result["case_mean_catch_rate"]
        result["case_mean_sampled_unresolved_rate"] = result["case_mean_unresolved_rate"]
    return result


def evaluate_suite(config, cases, output_dir, *, schedule, models, stage):
    from rl.evaluation_protocol import frozen_protocol

    protocol = frozen_protocol(config)
    if protocol is None:
        raise ValueError("Sampled suite requires a frozen protocol")
    if stage not in ("baseline", "trained", "final"):
        raise ValueError("Unknown evaluation stage")
    expected = [
        {
            "case_id": c["id"],
            "sampling_seeds": [
                int(np.random.SeedSequence([c["selector_seed"], seed]).generate_state(1)[0])
                for seed in protocol["sampling_repeat_seeds"]
            ],
        }
        for c in cases
    ]
    if not cases or schedule != expected:
        raise ValueError("Frozen sampling schedule/case order mismatch")
    controllers = [
        c
        for c in protocol["controllers"]
        if stage == "final" or (c["name"].startswith("trained_") == (stage == "trained"))
    ]
    required = {
        "trained" if c["name"].startswith("trained_") else "untrained"
        for c in controllers
        if c["inference"] in ("sampled_masked", "argmax_masked")
    }
    if any(models.get(name) is None for name in required):
        raise ValueError("All declared policy controllers must be available; no fallback")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    summaries, all_rows = {}, {}
    for controller in controllers:
        name, inference = controller["name"], controller["inference"]
        kind = (
            "trained"
            if name.startswith("trained_")
            else "untrained"
            if name.startswith("untrained_")
            else "random"
            if name == "random_valid_mask"
            else name
        )
        rows = []
        for repeat in range(controller["repeats"]):
            _, repeat_rows = evaluate(
                kind,
                config,
                cases,
                output_dir / f"{name}_repeat_{repeat}.csv",
                model=models.get(kind),
                label=stage,
                inference=inference,
                episode_seeds=[entry["sampling_seeds"][repeat] for entry in schedule],
                repeat_index=repeat,
                controller_name=name,
            )
            rows.extend(repeat_rows)
        summaries[name] = summarize_cases(rows, inference)
        summaries[name]["evaluation_role"] = rows[0]["evaluation_role"]
        all_rows[name] = rows
    write_json(output_dir / "summary.json", summaries)
    write_json(output_dir / "episodes.json", all_rows)
    return summaries, all_rows


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
    if config.get("evaluation_protocol"):
        from rl.pipeline import read_json

        summary, _ = evaluate_suite(
            config,
            manifest["splits"]["validation"],
            args.output,
            schedule=read_json(args.run_dir / "sampling_schedule.json")["validation"],
            models={"trained": model},
            stage="trained",
        )
    else:
        summary, _ = evaluate(
            "trained", config, manifest["splits"]["validation"], args.output, model=model
        )
    write_json(args.output.with_suffix(".json"), summary)


if __name__ == "__main__":
    main()
