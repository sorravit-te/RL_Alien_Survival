"""Validated sampled evaluation specification, frozen schedules and case-level selection."""

from dataclasses import asdict
from fractions import Fraction
from math import ceil, isfinite

import numpy as np

from rl.bots.evasive import EvasionConfig
from rl.contract import search_contract

CONTROLLER_SPECS = {
    "random_valid_mask": ("uniform_valid", 3, "baseline"),
    "untrained_sampled": ("sampled_masked", 3, "new compatible initial policy"),
    "untrained_deterministic": ("argmax_masked", 1, "new compatible initial policy"),
    "heuristic": ("scripted_catalog", 1, "baseline"),
    "wait": ("constant_wait", 1, "baseline"),
    "trained_sampled": ("sampled_masked", 3, "future compatible checkpoint only"),
    "trained_deterministic": ("argmax_masked", 1, "future compatible checkpoint only"),
}
SELECTION = {
    "split": "validation",
    "primary": "case_mean_sampled_catch_rate",
    "tie_1": "min_sampled_unresolved_rate",
    "tie_2": "earliest_trained_checkpoint",
    "exclude": "test outcomes and untrained baseline cannot select trained best",
}
HUNT_SELECTION = {**SELECTION, "primary": "mean_of_sampled_catch_rate_and_hunt_catch_rates"}
SELECTION_TIES = "then min unresolved, then earliest trained checkpoint"
CATCH_RULE = f"max validation primary sampled case-mean catch rate, {SELECTION_TIES}"
HUNT_RULE = f"max mean of validation sampled catch rate and hunt catch rates, {SELECTION_TIES}"
SEED_RULE = (
    "SeedSequence([case.selector_seed, repeat_seed]).generate_state(1)[0]; "
    "seed once per episode; isolate and restore global RNG streams"
)


def positive_integer(value):
    return type(value) is int and value > 0


def validate_protocol(training, evaluation):
    try:
        contract = search_contract()
        for spec in (training, evaluation):
            for key in ("search_control", "contract_version"):
                if spec.get(key) != contract[key]:
                    raise ValueError(f"Sampled protocol requires the current explicit {key}")
        if "scenario" in training or "scenario" in evaluation:
            from game.map import DEFAULT_SCENARIO
            from rl.pipeline import configured_map_identity

            scenario = training.get("scenario", DEFAULT_SCENARIO)
            identity = configured_map_identity(training)
            if (
                evaluation.get("scenario") != scenario
                or evaluation.get("scenario_sha256") != identity["sha256"]
            ):
                raise ValueError("Training and evaluation scenario mismatch")
        from rl.env import THREAT_BOT_KINDS

        if (
            training["bot_kind"] not in THREAT_BOT_KINDS
            or type(training["n_envs"]) is not int
            or training["n_envs"] != 1
        ):
            raise ValueError("Sampled protocol requires one explicit evasive mission bot")
        settings = training["survivor_threat_settings"]
        if set(settings) != set(asdict(EvasionConfig())):
            raise ValueError("Every evasion setting must be explicit")
        if any(type(v) not in (int, float) for v in settings.values()):
            raise ValueError("Evasion settings must be explicit numeric values")
        EvasionConfig(**settings)
        seed = training["training_seed"]
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Training seed must be an explicit uint32")
        if (
            evaluation["primary_inference"] != "sampled_masked_policy"
            or evaluation["secondary_inference"] != "deterministic_argmax_diagnostic"
        ):
            raise ValueError("Sampled inference must be primary, argmax secondary")
        seeds = evaluation["sampling_repeat_seeds"]
        if (
            len(seeds) != 3
            or len(set(seeds)) != 3
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
        ):
            raise ValueError("Exactly three distinct uint32 repeat seeds required")
        specs = evaluation["controllers"]
        if (
            len(specs) != len(CONTROLLER_SPECS)
            or {c["name"]: (c["inference"], c["repeats"], c["available"]) for c in specs}
            != CONTROLLER_SPECS
            or any(not positive_integer(c["repeats"]) for c in specs)
        ):
            raise ValueError("Controller availability/inference/repeats mismatch")
        if (
            evaluation["selection"] not in (SELECTION, HUNT_SELECTION)
            or evaluation["selector_seed_rule"] != SEED_RULE
        ):
            raise ValueError("Selection or selector seed rules changed")
        counts = training["manifest_counts"]
        if (
            set(counts) != {"train", "validation", "test"}
            or any(not positive_integer(v) for v in counts.values())
            or counts != evaluation["manifest_counts"]
            or training["manifest_seed"] != evaluation["manifest_seed"]
            or type(training["manifest_seed"]) is not int
            or not 0 <= training["manifest_seed"] < 2**32
        ):
            raise ValueError("Training and evaluation manifests disagree or are invalid")
        for name in (
            "requested_steps",
            "evaluation_interval",
            "progress_interval",
            "torch_threads",
        ):
            if not positive_integer(training[name]):
                raise ValueError(f"Invalid budget: {name}")
        cap = training["max_episode_seconds"]
        if isinstance(cap, bool) or not isfinite(cap) or cap <= 0:
            raise ValueError("Invalid episode time budget")
        ppo = training["ppo"]
        for name in ("n_steps", "batch_size", "n_epochs"):
            if not positive_integer(ppo[name]):
                raise ValueError(f"Invalid PPO budget: {name}")
        if ppo["n_steps"] < 2 or ppo["batch_size"] < 2:
            raise ValueError("PPO rollout and batch budgets must exceed one")
        for name in ("learning_rate", "gamma", "gae_lambda", "clip_range", "max_grad_norm"):
            value = ppo[name]
            if type(value) not in (int, float) or not isfinite(value) or value <= 0:
                raise ValueError(f"Invalid explicit PPO parameter: {name}")
        if ppo["gamma"] > 1 or ppo["gae_lambda"] > 1:
            raise ValueError("Discount parameters must not exceed one")
        for name in ("ent_coef", "vf_coef"):
            value = ppo[name]
            if type(value) not in (int, float) or not isfinite(value) or value < 0:
                raise ValueError(f"Invalid explicit PPO parameter: {name}")
        if type(ppo["normalize_advantage"]) is not bool:
            raise ValueError("normalize_advantage must be explicit boolean")
        architecture = ppo["policy_kwargs"]["net_arch"]
        if (
            not isinstance(architecture, list)
            or not architecture
            or any(not positive_integer(width) for width in architecture)
        ):
            raise ValueError("Explicit positive MLP architecture required")
        rule = HUNT_RULE if evaluation["selection"] == HUNT_SELECTION else CATCH_RULE
        if training["selection"] != rule:
            raise ValueError("Training selection disagrees with sampled protocol")
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f"Incomplete or invalid sampled protocol: {exc}") from exc


def sampling_schedule(protocol, manifest):
    return {
        split: [
            {
                "case_id": c["id"],
                "sampling_seeds": [
                    int(np.random.SeedSequence([c["selector_seed"], seed]).generate_state(1)[0])
                    for seed in protocol["sampling_repeat_seeds"]
                ],
            }
            for c in manifest["splits"][split]
        ]
        for split in ("validation", "test")
    }


def evaluation_budget(config):
    counts = config["manifest_counts"]
    rounded = ceil(config["requested_steps"] / config["ppo"]["n_steps"]) * config["ppo"]["n_steps"]
    validations = rounded // config["evaluation_interval"] + 1
    budget = {
        "rollout_rounded_decisions": rounded,
        "baseline_validation_episodes": counts["validation"] * 9,
        "checkpoint_validation_calls": validations,
        "checkpoint_validation_episodes": counts["validation"] * 4 * validations,
        "fresh_best_validation_replay_episodes": counts["validation"] * 4,
        "final_test_episodes": counts["test"] * 13,
        "max_episode_seconds": config["max_episode_seconds"],
    }
    if "imitation" in config:
        budget["imitated_validation_episodes"] = counts["validation"] * 4
    return budget


def frozen_protocol(config):
    from game.map import DEFAULT_SCENARIO
    from rl.pipeline import digest, make_manifest

    frozen = config.get("frozen_evaluation")
    if not frozen:
        raise ValueError("Sampled evaluation requires a prepared frozen protocol")
    protocol = frozen["protocol"]
    validate_protocol(config, protocol)
    manifest = make_manifest(
        config["manifest_seed"], config["manifest_counts"], config.get("scenario", DEFAULT_SCENARIO)
    )
    if (
        digest(protocol) != frozen["protocol_sha256"]
        or digest(sampling_schedule(protocol, manifest)) != frozen["schedule_sha256"]
        or digest(evaluation_budget(config)) != frozen["budget_sha256"]
    ):
        raise ValueError("Frozen evaluation protocol/schedule/budget changed")
    return protocol


def trained_score(summary, decision_steps, optimizer_epochs):
    """Compare exact per-case count ratios; displayed floats never decide selection."""
    if decision_steps <= 0 or optimizer_epochs <= 0:
        return None
    cases = summary["trained_sampled"]["per_case"]
    if not cases:
        raise ValueError("Selection requires nonempty per-case counts")
    catch = sum(Fraction(case["captured"], case["repeats"]) for case in cases) / len(cases)
    unresolved = sum(Fraction(case["unresolved"], case["repeats"]) for case in cases) / len(cases)
    hunts = summary.get("hunt", {}).values()
    catch = (catch + sum(Fraction(t["caught"], t["cases"]) for t in hunts)) / (1 + len(hunts))
    return catch, -unresolved, -decision_steps
