"""Reproducible training manifests, checkpoint metadata, logging and episode wrapper."""

import contextlib
import hashlib
import io
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO

from game.map import small_test_map
from rl.contract import ACTION_COUNT, OBSERVATION_SIZE, VERSION
from rl.env import AlienEnv
from rl.evasive_survivor import EvasionConfig

ROOT = Path(__file__).resolve().parents[1]
# New checkpoints use this format; loading requires an exact match.
CHECKPOINT_FORMAT = "alien-maskable-checkpoint-v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependencies():
    return {
        "python": platform.python_version(),
        **{
            name: version(name)
            for name in (
                "numpy",
                "torch",
                "gymnasium",
                "stable_baselines3",
                "sb3_contrib",
                "tensorboard",
                "matplotlib",
            )
        },
    }


def source_state():
    files = sorted(p for folder in ("game", "ai", "rl") for p in (ROOT / folder).glob("*.py"))
    files += [ROOT / "requirements.lock.txt"]
    source_hash = hashlib.sha256(
        b"".join(p.relative_to(ROOT).as_posix().encode() + p.read_bytes() for p in files)
    ).hexdigest()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    return {"revision": revision, "source_sha256": source_hash, "dirty_status": dirty}


def map_identity():
    world = small_test_map()
    return {
        "name": "small_test_map_v1",
        "sha256": digest(asdict(world)),
        "spawn_points": world.spawn_points,
        "card_exit_fixed": True,
        "warning": "Two real spawn points only; seeds are NOT layout/spawn holdout.",
    }


def make_manifest(seed, counts):
    rng = np.random.default_rng(seed)
    used = set()

    def unique():
        while True:
            value = int(rng.integers(0, 2**31))
            if value not in used:
                used.add(value)
                return value

    splits = {}
    for split in ("train", "validation", "test"):
        if counts[split] <= 0:
            raise ValueError("Each split must be nonempty")
        splits[split] = [
            {
                "id": f"{split}_{i:04d}",
                "reset_seed": unique(),
                "scenario_seed": unique(),
                "bot_seed": unique(),
                "selector_seed": unique(),
            }
            for i in range(counts[split])
        ]
    return {
        "seed": seed,
        "map": map_identity(),
        "splits": splits,
        "warning": "Disjoint numeric seeds, same fixed layout/card/exit and two spawns.",
    }


def make_env(config):
    settings = config.get("survivor_threat_settings")
    return AlienEnv(
        max_episode_seconds=config["max_episode_seconds"],
        bot_kind=config["bot_kind"],
        evasion_settings=EvasionConfig(**settings) if settings is not None else None,
    )


class ManifestEnv(gym.Wrapper):
    def __init__(self, config, cases):
        super().__init__(make_env(config))
        self.cases = tuple(cases)
        self.cursor = 0
        self.current_case = None

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("Training manifest owns reset options")
        if seed is not None:
            self.cursor = 0
        self.current_case = self.cases[self.cursor % len(self.cases)]
        self.cursor += 1
        case = self.current_case
        return self.env.reset(
            seed=case["reset_seed"],
            options={"scenario_seed": case["scenario_seed"], "bot_seed": case["bot_seed"]},
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = {**info, "manifest_id": self.current_case["id"]}
        return obs, reward, terminated, truncated, info

    def action_masks(self):
        return self.env.action_masks()


def new_model(config, env):
    torch.set_num_threads(config["torch_threads"])
    return MaskablePPO(
        "MlpPolicy", env, seed=config["training_seed"], device="cpu", verbose=0, **config["ppo"]
    )


def parameter_hash(model):
    return hashlib.sha256(
        b"".join(
            t.detach().cpu().contiguous().numpy().tobytes()
            for t in model.policy.state_dict().values()
        )
    ).hexdigest()


def save_checkpoint(
    path,
    model,
    config,
    fingerprint,
    *,
    role,
    elapsed_seconds=0.0,
    validation=None,
    evaluation_context=None,
):
    from rl.evaluation_protocol import frozen_protocol

    protocol = frozen_protocol(config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "role": role,
        "runtime": {
            "device": "cpu",
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch_threads": torch.get_num_threads(),
        },
        "source": fingerprint,
        "config_sha256": digest(config),
        "config": config,
        "manifest_sha256": digest(
            make_manifest(config["manifest_seed"], config["manifest_counts"])
        ),
        "contract_version": VERSION,
        "observation_shape": [OBSERVATION_SIZE],
        "action_n": ACTION_COUNT,
        "map": map_identity(),
        "dependencies": dependencies(),
        "training_seed": config["training_seed"],
        "bot_kind": config["bot_kind"],
        "ppo_parameters": config["ppo"],
        "actual_steps": int(model.num_timesteps),
        "optimizer_update_epochs": int(model._n_updates),
        "elapsed_seconds": float(elapsed_seconds),
        "validation": validation,
        "checkpoint_sha256": file_hash(path),
        "policy_sha256": parameter_hash(model),
    }
    if protocol is not None:
        metadata["evaluation_identity"] = config["frozen_evaluation"]
        metadata["evaluated_at_decision_steps"] = int(model.num_timesteps) if validation else None
        metadata["evaluated_at_optimizer_update_epochs"] = (
            int(model._n_updates) if validation else None
        )
        metadata["evaluation_context"] = evaluation_context
    write_json(path.with_suffix(".metadata.json"), metadata)
    return metadata


def load_checkpoint(path, config, fingerprint, env=None):
    from rl.evaluation_protocol import frozen_protocol

    protocol = frozen_protocol(config)
    path = Path(path)
    metadata = read_json(path.with_suffix(".metadata.json"))
    expected = {
        "contract_version": VERSION,
        "observation_shape": [OBSERVATION_SIZE],
        "action_n": ACTION_COUNT,
        "config_sha256": digest(config),
        "config": config,
        "manifest_sha256": digest(
            make_manifest(config["manifest_seed"], config["manifest_counts"])
        ),
        "dependencies": dependencies(),
        "format": CHECKPOINT_FORMAT,
    }
    if protocol is not None:
        expected["evaluation_identity"] = config["frozen_evaluation"]
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f"Checkpoint metadata mismatch: {name}")
    if metadata["source"]["source_sha256"] != fingerprint["source_sha256"]:
        raise ValueError("Checkpoint metadata mismatch: source")
    if metadata["map"]["sha256"] != map_identity()["sha256"]:
        raise ValueError("Checkpoint metadata mismatch: map")
    if file_hash(path) != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint file hash mismatch")
    model = MaskablePPO.load(str(path), env=env, device="cpu")
    if parameter_hash(model) != metadata["policy_sha256"]:
        raise ValueError("Loaded policy parameter hash mismatch")
    if protocol is not None:
        if (
            int(model.num_timesteps) != metadata["actual_steps"]
            or int(model._n_updates) != metadata["optimizer_update_epochs"]
        ):
            raise ValueError("Checkpoint decision/optimizer counters mismatch")
        if metadata["validation"] is not None and (
            metadata.get("evaluated_at_decision_steps") != int(model.num_timesteps)
            or metadata.get("evaluated_at_optimizer_update_epochs") != int(model._n_updates)
            or metadata.get("evaluation_context")
            not in (
                "rollout_callback_before_current_rollout_optimizer_update",
                "after_learning_optimizer_update",
            )
        ):
            raise ValueError("Checkpoint evaluation timing mismatch")
    return model, metadata


def prepare_run(run_dir, config):
    if config["n_envs"] != 1 or config["bot_kind"] not in ("explorer", "evasive_explorer"):
        raise ValueError(
            "Training requires one environment and an explicitly supported mission bot"
        )
    from copy import deepcopy

    from rl.evaluation_protocol import evaluation_budget, sampling_schedule, validate_protocol

    config = deepcopy(config)
    protocol_bytes = None
    if "evaluation_protocol" in config:
        if not config["evaluation_protocol"]:
            raise ValueError("Remove evaluation_protocol for explicit legacy deterministic mode")
        protocol_bytes = (ROOT / config["evaluation_protocol"]).read_bytes()
        protocol = json.loads(protocol_bytes)
        validate_protocol(config, protocol)
    else:
        protocol = None
    manifest = make_manifest(config["manifest_seed"], config["manifest_counts"])
    if protocol is not None:
        schedule = sampling_schedule(protocol, manifest)
        budget = evaluation_budget(config)
        config["frozen_evaluation"] = {
            "protocol": protocol,
            "protocol_sha256": digest(protocol),
            "protocol_file_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "schedule_sha256": digest(schedule),
            "budget_sha256": digest(budget),
        }
    # Fail before creating any run if bot/settings cannot be constructed.
    env = make_env(config)
    env.close()
    fingerprint = source_state()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "manifest.json", manifest)
    identity = {
        "source": fingerprint,
        "config_sha256": digest(config),
        "manifest_sha256": digest(manifest),
        "dependencies": dependencies(),
        "evaluation_mode": "sampled_primary_argmax_secondary"
        if protocol
        else "legacy_deterministic_argmax",
    }
    if protocol is not None:
        (run_dir / "evaluation_protocol.json").write_bytes(protocol_bytes)
        write_json(run_dir / "sampling_schedule.json", schedule)
        write_json(run_dir / "evaluation_budget.json", budget)
        identity["evaluation_files"] = {
            name: file_hash(run_dir / name)
            for name in (
                "evaluation_protocol.json",
                "sampling_schedule.json",
                "evaluation_budget.json",
            )
        }
        identity["readiness"] = "integrated; prepare does not authorize learning"
    write_json(run_dir / "run_identity.json", identity)
    return manifest, fingerprint


def read_run(run_dir):
    run_dir = Path(run_dir)
    config, manifest, identity = (
        read_json(run_dir / name) for name in ("config.json", "manifest.json", "run_identity.json")
    )
    if (
        digest(config) != identity["config_sha256"]
        or digest(manifest) != identity["manifest_sha256"]
    ):
        raise ValueError("Frozen run config/manifest changed")
    if source_state()["source_sha256"] != identity["source"]["source_sha256"]:
        raise ValueError("Source changed after run preparation")
    if dependencies() != identity["dependencies"]:
        raise ValueError("Dependencies changed after run preparation")
    from rl.evaluation_protocol import evaluation_budget, frozen_protocol, sampling_schedule

    protocol = frozen_protocol(config)
    if protocol is not None:
        expected_files = {
            "evaluation_protocol.json",
            "sampling_schedule.json",
            "evaluation_budget.json",
        }
        files = identity.get("evaluation_files", {})
        if set(files) != expected_files or any(
            file_hash(run_dir / n) != h for n, h in files.items()
        ):
            raise ValueError("Frozen evaluation files changed")
        frozen = config["frozen_evaluation"]
        if (
            read_json(run_dir / "evaluation_protocol.json") != protocol
            or file_hash(run_dir / "evaluation_protocol.json") != frozen["protocol_file_sha256"]
            or read_json(run_dir / "sampling_schedule.json")
            != sampling_schedule(protocol, manifest)
            or read_json(run_dir / "evaluation_budget.json") != evaluation_budget(config)
        ):
            raise ValueError("Frozen evaluation schedule/protocol/budget changed")
    return config, manifest, identity["source"]


class _Tee(io.TextIOBase):
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


@contextlib.contextmanager
def live_log(path):
    with Path(path).open("a", encoding="utf-8", buffering=1) as log:
        tee = _Tee(sys.stdout, log)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            yield
