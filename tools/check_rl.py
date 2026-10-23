"""Check the real2B adapter and seeded masked-random rollouts. Never trains a model."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
from gymnasium.utils.env_checker import check_env
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from rl.contract import (
    ACTION_NAMES,
    AGE_SCALE_SECONDS,
    BASE_FIELDS,
    CANDIDATE_FIELDS,
    DISTANCE_SCALE,
    OBSERVATION_SIZE,
    VERSION,
    WORLD_SCALE,
)
from rl.env import AlienEnv

ROOT = Path(__file__).resolve().parents[1]


def validate(episodes=12, seed=20261024, cap_seconds=600):
    if episodes <= 0:
        raise ValueError("Need at least one episode")
    checker_env = AlienEnv(max_episode_seconds=1)
    check_env(checker_env, skip_render_check=True)
    sb3_check_env(checker_env, warn=True)
    assert is_masking_supported(checker_env)
    results = []
    traces = []
    times = []
    counts = {"captured": 0, "escaped": 0, "unresolved": 0}
    for episode in range(episodes):
        env = AlienEnv(max_episode_seconds=cap_seconds)
        reset_seed = seed + episode
        observation, initial = env.reset(seed=reset_seed)
        rng = np.random.default_rng(np.random.SeedSequence([seed, episode, 99]))
        total = 0.0
        decisions = 0
        components = {"capture": 0.0, "escape": 0.0, "first_sight": 0.0}
        while True:
            masks = get_action_masks(env)
            assert masks.dtype == np.bool_ and masks.shape == (32,) and masks.any()
            assert env.observation_space.contains(observation)
            action = int(rng.choice(np.flatnonzero(masks)))
            before = observation.copy()
            start = perf_counter()
            observation, reward, terminated, truncated, info = env.step(action)
            times.append((perf_counter() - start) * 1000)
            decisions += 1
            total += reward
            assert not info["invalid_action"]
            for name, value in info["reward_components"].items():
                components[name] += value
            traces.append(
                {
                    "episode": episode,
                    "decision": decisions,
                    "observation": before.tolist(),
                    "valid_actions": np.flatnonzero(masks).tolist(),
                    "action": action,
                    "reward": reward,
                    "reward_components": info["reward_components"],
                    "elapsed_ticks": info["elapsed_ticks"],
                    "step_ticks": info["step_ticks"],
                    "outcome": info["outcome"],
                }
            )
            if terminated or truncated:
                outcome = info["outcome"]
                counts[outcome] += 1
                results.append(
                    {
                        "episode": episode,
                        "reset_seed": reset_seed,
                        "scenario_seed": initial["scenario_seed"],
                        "bot_seed": initial["bot_seed"],
                        "bot": initial["bot"],
                        "decisions": decisions,
                        "physics_ticks": info["elapsed_ticks"],
                        "return": total,
                        "reward_components": components,
                        "outcome": outcome,
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                )
                break
        env.close()
    ordered = sorted(times)
    files = sorted(p for folder in ("rl", "game", "ai") for p in (ROOT / folder).glob("*.py"))
    source_hash = hashlib.sha256(
        b"".join(p.relative_to(ROOT).as_posix().encode() + p.read_bytes() for p in files)
    ).hexdigest()
    return {
        "evidence": "integration only; seeded masked-random selector; NO TRAINING/LEARNING CLAIM",
        "checkers": {"gymnasium": "passed", "sb3": "passed", "sb3_contrib_mask_utils": "passed"},
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("gymnasium", "numpy", "stable_baselines3", "sb3_contrib")
        },
        "source_sha256": source_hash,
        "contract": {
            "version": VERSION,
            "action_names": ACTION_NAMES,
            "base_fields": BASE_FIELDS,
            "candidate_fields": CANDIDATE_FIELDS,
            "observation_shape": [OBSERVATION_SIZE],
            "dtype": "float32",
            "bounds": [-1, 1],
            "world_scale": WORLD_SCALE,
            "distance_scale": DISTANCE_SCALE,
            "age_seconds_scale": AGE_SCALE_SECONDS,
            "decision_ticks": 15,
            "physics_hz": 30,
        },
        "seed": seed,
        "cap_seconds": cap_seconds,
        "episodes": results,
        "counts": counts,
        "total_decisions": sum(row["decisions"] for row in results),
        "total_physics_ticks": sum(row["physics_ticks"] for row in results),
        "decision_cost_ms": {
            "conditions": "env.step only; includes Core/bot/obs; excludes reset/checkers/training/render",
            "mean": mean(times),
            "p95": ordered[int((len(ordered) - 1) * 0.95)],
            "max": max(times),
        },
        "trace": traces,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--cap-seconds", type=float, default=600)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/2b/integration.json")
    args = parser.parse_args()
    report = validate(args.episodes, args.seed, args.cap_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "checkers",
                    "counts",
                    "total_decisions",
                    "total_physics_ticks",
                    "decision_cost_ms",
                )
            },
            indent=2,
        )
    )
    print(f"Evidence saved: {args.output}")
