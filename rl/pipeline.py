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

from game.map import DEFAULT_SCENARIO, scenario_map
from game.state import CoreConfig
from rl.bots.evasive import EvasionConfig
from rl.contract import ACTION_COUNT, OBSERVATION_SIZE, search_contract
from rl.env import THREAT_BOT_KINDS, AlienEnv

# Mission bots a training run may use; stationary_debug is a mechanics fixture only.
TRAINING_BOT_KINDS = ("explorer", *THREAT_BOT_KINDS)

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


def source_files():
    """Every behavior-defining file, including packages nested below game/ai/rl."""
    files = sorted(p for folder in ("game", "ai", "rl") for p in (ROOT / folder).rglob("*.py"))
    return [*files, ROOT / "requirements.lock.txt"]


def source_state():
    source_hash = hashlib.sha256(
        b"".join(p.relative_to(ROOT).as_posix().encode() + p.read_bytes() for p in source_files())
    ).hexdigest()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    return {"revision": revision, "source_sha256": source_hash, "dirty_status": dirty}


def map_identity(scenario=DEFAULT_SCENARIO):
    world = scenario_map(scenario)
    return {
        "name": scenario,
        "sha256": digest(asdict(world)),
        "spawn_points": world.spawn_points,
        "card_exit_fixed": True,
        "warning": "Two real spawn points only; seeds are NOT layout/spawn holdout.",
    }


def configured_map_identity(config):
    identity = map_identity(config.get("scenario", DEFAULT_SCENARIO))
    if "scenario" in config and config.get("scenario_sha256") != identity["sha256"]:
        raise ValueError("Configured scenario full-world hash mismatch")
    return identity


def configured_search_contract(config):
    """Only the current contract can be trained or loaded; old run names are rejected."""
    identity = search_contract()
    for key in ("search_control", "contract_version"):
        if key in config and config[key] != identity[key]:
            raise ValueError(f"Configured {key} is not the current contract")
    return identity


def make_manifest(seed, counts, scenario=DEFAULT_SCENARIO):
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
        "map": map_identity(scenario),
        "splits": splits,
        "warning": "Disjoint numeric seeds, same fixed layout/card/exit and two spawns.",
    }


def read_json_value(value):
    return json.loads(json.dumps(value))


def validate_environment_config(config):
    """Static readiness only; runtime navigation/actors require a separately approved smoke."""
    from math import isfinite

    from ai.navigation import find_path

    configured_search_contract(config)
    configured_map_identity(config)
    world = scenario_map(config.get("scenario", DEFAULT_SCENARIO))
    core = CoreConfig()
    cap = config["max_episode_seconds"]
    if isinstance(cap, bool) or not isfinite(cap) or cap <= 0:
        raise ValueError("Episode cap must be finite and positive")
    if config["bot_kind"] not in TRAINING_BOT_KINDS:
        raise ValueError("Training requires a mission bot")
    for flag in ("fresh_run", "defer_final_test"):
        if flag in config and type(config[flag]) is not bool:
            raise ValueError(f"{flag} must be boolean")
    settings = config.get("survivor_threat_settings")
    if settings is not None:
        if config["bot_kind"] not in THREAT_BOT_KINDS:
            raise ValueError("Evasion settings require evasive bot kinds")
        EvasionConfig(**settings)
    if max(world.width, world.height) > 40 or len(world.patrol_waypoints) > 20:
        raise ValueError("World exceeds observation contract")
    if world.alien_spawn is None:
        raise ValueError("Scenario requires Alien spawn")
    for point in (*world.spawn_points, world.alien_spawn[:2], *world.patrol_waypoints):
        if world.overlaps(*point, core.alien_radius):
            raise ValueError("Scenario lacks body clearance")
    for point in world.patrol_waypoints:
        if not find_path(world, world.alien_spawn[:2], point, core.alien_radius):
            raise ValueError("Static policy waypoint is unreachable")
    return map_identity(config.get("scenario", DEFAULT_SCENARIO))


def make_env(config):
    configured_search_contract(config)
    configured_map_identity(config)
    settings = config.get("survivor_threat_settings")
    return AlienEnv(
        max_episode_seconds=config["max_episode_seconds"],
        bot_kind=config["bot_kind"],
        evasion_settings=EvasionConfig(**settings) if settings is not None else None,
        scenario=config.get("scenario", DEFAULT_SCENARIO),
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

    configured_map_identity(config)
    frozen_protocol(config)
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
            make_manifest(
                config["manifest_seed"],
                config["manifest_counts"],
                config.get("scenario", DEFAULT_SCENARIO),
            )
        ),
        "contract_version": configured_search_contract(config)["contract_version"],
        "observation_shape": [OBSERVATION_SIZE],
        "action_n": ACTION_COUNT,
        "map": configured_map_identity(config),
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
        "search_control": configured_search_contract(config)["search_control"],
        "evaluation_identity": config["frozen_evaluation"],
        "evaluated_at_decision_steps": int(model.num_timesteps) if validation else None,
        "evaluated_at_optimizer_update_epochs": int(model._n_updates) if validation else None,
        "evaluation_context": evaluation_context,
    }
    write_json(path.with_suffix(".metadata.json"), metadata)
    return metadata


def load_checkpoint(path, config, fingerprint, env=None):
    from rl.evaluation_protocol import frozen_protocol

    configured_map_identity(config)
    frozen_protocol(config)
    path = Path(path)
    metadata = read_json(path.with_suffix(".metadata.json"))
    contract = configured_search_contract(config)
    expected = {
        "contract_version": contract["contract_version"],
        "search_control": contract["search_control"],
        "observation_shape": [OBSERVATION_SIZE],
        "action_n": ACTION_COUNT,
        "config_sha256": digest(config),
        "config": config,
        "manifest_sha256": digest(
            make_manifest(
                config["manifest_seed"],
                config["manifest_counts"],
                config.get("scenario", DEFAULT_SCENARIO),
            )
        ),
        "dependencies": dependencies(),
        "format": CHECKPOINT_FORMAT,
        "evaluation_identity": config["frozen_evaluation"],
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f"Checkpoint metadata mismatch: {name}")
    if metadata["source"]["source_sha256"] != fingerprint["source_sha256"]:
        raise ValueError("Checkpoint metadata mismatch: source")
    if digest(metadata.get("map")) != digest(configured_map_identity(config)):
        raise ValueError("Checkpoint metadata mismatch: map")
    if env is not None:
        actual_env = env.unwrapped
        if digest(asdict(actual_env.world)) != metadata["map"][
            "sha256"
        ] or actual_env.scenario != config.get("scenario", DEFAULT_SCENARIO):
            raise ValueError("Checkpoint metadata mismatch: environment scenario")
    if file_hash(path) != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint file hash mismatch")
    model = MaskablePPO.load(str(path), env=env, device="cpu")
    if parameter_hash(model) != metadata["policy_sha256"]:
        raise ValueError("Loaded policy parameter hash mismatch")
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
    if config["n_envs"] != 1 or config["bot_kind"] not in TRAINING_BOT_KINDS:
        raise ValueError(
            "Training requires one environment and an explicitly supported mission bot"
        )
    from copy import deepcopy

    from rl.evaluation_protocol import evaluation_budget, sampling_schedule, validate_protocol

    config = deepcopy(config)
    if not config.get("evaluation_protocol"):
        raise ValueError("Training requires an evaluation_protocol file")
    protocol_bytes = (ROOT / config["evaluation_protocol"]).read_bytes()
    protocol = json.loads(protocol_bytes)
    validate_protocol(config, protocol)
    manifest = make_manifest(
        config["manifest_seed"], config["manifest_counts"], config.get("scenario", DEFAULT_SCENARIO)
    )
    schedule = sampling_schedule(protocol, manifest)
    budget = evaluation_budget(config)
    config["frozen_evaluation"] = {
        "protocol": protocol,
        "protocol_sha256": digest(protocol),
        "protocol_file_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "schedule_sha256": digest(schedule),
        "budget_sha256": digest(budget),
    }
    # Preparation is static: no Core construction/reset/rollout or policy creation.
    validate_environment_config(config)
    fingerprint = source_state()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "manifest.json", manifest)
    (run_dir / "evaluation_protocol.json").write_bytes(protocol_bytes)
    write_json(run_dir / "sampling_schedule.json", schedule)
    write_json(run_dir / "evaluation_budget.json", budget)
    identity = {
        "source": fingerprint,
        "config_sha256": digest(config),
        "manifest_sha256": digest(manifest),
        "dependencies": dependencies(),
        "evaluation_mode": "sampled_primary_argmax_secondary",
        "semantic_contract": configured_search_contract(config),
        "evaluation_files": {
            name: file_hash(run_dir / name)
            for name in (
                "evaluation_protocol.json",
                "sampling_schedule.json",
                "evaluation_budget.json",
            )
        },
        "readiness": "integrated; prepare does not authorize learning",
    }
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
    configured_map_identity(config)
    if identity.get("semantic_contract") != configured_search_contract(config):
        raise ValueError("Frozen search semantic contract changed")
    expected_manifest = make_manifest(
        config["manifest_seed"], config["manifest_counts"], config.get("scenario", DEFAULT_SCENARIO)
    )
    if manifest != read_json_value(expected_manifest):
        raise ValueError("Frozen manifest/scenario identity mismatch")
    if source_state()["source_sha256"] != identity["source"]["source_sha256"]:
        raise ValueError("Source changed after run preparation")
    if dependencies() != identity["dependencies"]:
        raise ValueError("Dependencies changed after run preparation")
    from rl.evaluation_protocol import evaluation_budget, frozen_protocol, sampling_schedule

    protocol = frozen_protocol(config)
    expected_files = {
        "evaluation_protocol.json",
        "sampling_schedule.json",
        "evaluation_budget.json",
    }
    files = identity.get("evaluation_files", {})
    if set(files) != expected_files or any(file_hash(run_dir / n) != h for n, h in files.items()):
        raise ValueError("Frozen evaluation files changed")
    frozen = config["frozen_evaluation"]
    if (
        read_json(run_dir / "evaluation_protocol.json") != protocol
        or file_hash(run_dir / "evaluation_protocol.json") != frozen["protocol_file_sha256"]
        or read_json(run_dir / "sampling_schedule.json") != sampling_schedule(protocol, manifest)
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
