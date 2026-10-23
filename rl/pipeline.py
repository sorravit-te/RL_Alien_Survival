"""2C reproducible manifests, metadata, logging and training-only episode wrapper."""

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

ROOT = Path(__file__).resolve().parents[1]


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
        "name": "small_test_map_1c_2a",
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
    return AlienEnv(max_episode_seconds=config["max_episode_seconds"], bot_kind=config["bot_kind"])


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
    path, model, config, fingerprint, *, role, elapsed_seconds=0.0, validation=None
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    metadata = {
        "format": "alien-maskable-checkpoint-2c-v1",
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
    write_json(path.with_suffix(".metadata.json"), metadata)
    return metadata


def load_checkpoint(path, config, fingerprint, env=None):
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
        "format": "alien-maskable-checkpoint-2c-v1",
    }
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
    return model, metadata


def prepare_run(run_dir, config):
    if config["n_envs"] != 1 or config["bot_kind"] != "explorer":
        raise ValueError("This pilot supports one real Explorer training environment")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = make_manifest(config["manifest_seed"], config["manifest_counts"])
    fingerprint = source_state()
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "manifest.json", manifest)
    write_json(
        run_dir / "run_identity.json",
        {
            "source": fingerprint,
            "config_sha256": digest(config),
            "manifest_sha256": digest(manifest),
            "dependencies": dependencies(),
        },
    )
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
