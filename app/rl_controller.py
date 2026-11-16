"""Human-input adapter for the frozen Alien target contract; no Env or Survivor bot."""

import os
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from game.core import GameCore
from game.map import DEFAULT_SCENARIO, scenario_map
from game.state import CoreConfig
from rl.alien_view import allowed_view, record_waypoint_visits, target_source
from rl.candidates import build_candidates, candidate_masks
from rl.contract import ACTION_COUNT, DECISION_TICKS, OBSERVATION_SIZE, WAIT
from rl.observations import build_observation

# Loading checks the source with git; tools/bootstrap.ps1 fetches this copy when Windows has none.
PORTABLE_GIT = Path(__file__).resolve().parents[1] / ".git-portable" / "cmd"


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


@contextmanager
def isolated_globals():
    before = rng_state()
    try:
        yield
    finally:
        restore_rng(before)


class EpisodeSampler:
    """Private CPU RNG streams: seed once on reset, resume them at each decision."""

    def __init__(self, model):
        self.model = model

    def reset(self, seed):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Sampling seed must be an integer in [0, 2**32)")
        self.seed = seed
        with isolated_globals():
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.state = rng_state()

    def predict(self, observation, mask):
        policy = getattr(self.model, "policy", None)
        training = policy.training if policy is not None else None
        with isolated_globals():
            restore_rng(self.state)
            try:
                if policy is not None:
                    policy.set_training_mode(False)
                action, _ = self.model.predict(observation, action_masks=mask, deterministic=False)
                self.state = rng_state()
            finally:
                if policy is not None:
                    policy.set_training_mode(training)
        if np.asarray(action).shape != () or not np.issubdtype(
            np.asarray(action).dtype, np.integer
        ):
            raise ValueError("RL policy returned a non-integer action")
        action = int(action)
        if not 0 <= action < ACTION_COUNT or not mask[action]:
            raise ValueError(f"RL policy selected invalid/masked action {action}; stopping")
        return action


class HumanRLController:
    """Core-compatible tick driver; decisions happen before each block of 15 ticks."""

    def __init__(
        self,
        core,
        model,
        sampling_seed,
        *,
        label="RL sampled",
    ):
        if core.alien_control != "external" or core.snapshot().alien is None:
            raise ValueError("RL play requires an active external-control Alien")
        self.core, self.model = core, model
        self.config = core.config
        self.label = label
        self.sampler = EpisodeSampler(model)
        self.sampling_seed = sampling_seed
        self.recorder = None
        self.reset(core.snapshot().seed)

    @property
    def controller_label(self):
        return f"{self.label} | seed {self.sampling_seed}"

    def reset(self, seed):
        if self.recorder is not None:
            self.recorder.before_reset(self, seed)
        self.core.reset(seed)
        self.phase = self.decisions = 0
        self.last_action = WAIT
        self.visit_ticks = [None] * len(self.core.world_map.patrol_waypoints)
        self.sampler.reset(self.sampling_seed)
        if self.recorder is not None:
            self.recorder.after_reset(self)
        return self.snapshot()

    def snapshot(self):
        return self.core.snapshot()

    def allowed_view(self):
        return allowed_view(self.core.snapshot(), self.last_action, self.visit_ticks)

    def observation(self):
        return build_observation(self.allowed_view())

    def action_masks(self):
        return candidate_masks(build_candidates(self.allowed_view()))

    def step(self, human_command):
        before = self.snapshot()
        phase_before = self.phase
        if before.outcome is not None:
            # Preserve Core post-terminal event clearing without advancing policy time.
            return self.core.step(human_command)
        if self.phase == 0:
            view = self.allowed_view()
            candidates = build_candidates(view)
            observation, mask = build_observation(view), candidate_masks(candidates)
            if observation.shape != (OBSERVATION_SIZE,):
                raise ValueError("RL observation contract mismatch")
            action = self.sampler.predict(observation, mask)
            self.last_action = action
            chosen = candidates[action]
            self.core.set_alien_target(
                chosen.target,
                source=target_source(action),
                evidence_revision=chosen.evidence_revision,
            )
            self.decisions += 1
            if self.recorder is not None:
                self.recorder.decision(self, before, view, observation, mask, action, chosen.target)
        snapshot = self.core.step(human_command)
        record_waypoint_visits(snapshot, self.visit_ticks)
        self.phase = (self.phase + 1) % DECISION_TICKS
        if self.recorder is not None:
            self.recorder.tick(self, human_command, before, snapshot, phase_before)
        return snapshot


def load_human_rl(run_dir, checkpoint, sampling_seed, scenario_seed=0):
    """Exact existing provenance checks; selected best only, never heuristic fallback."""
    from rl.pipeline import file_hash, load_checkpoint, read_json, read_run

    run_dir, checkpoint = Path(run_dir), Path(checkpoint)
    if PORTABLE_GIT.is_dir() and str(PORTABLE_GIT) not in os.environ["PATH"]:
        os.environ["PATH"] = f"{PORTABLE_GIT}{os.pathsep}{os.environ['PATH']}"
    with isolated_globals():
        config, _, source = read_run(run_dir)
        torch.set_num_threads(config["torch_threads"])
        model, metadata = load_checkpoint(checkpoint, config, source)
    selection = read_json(run_dir / "selection.json")
    if (
        metadata["role"] != "best_validation"
        or file_hash(checkpoint) != selection["best_checkpoint_sha256"]
        or metadata["actual_steps"] != selection["best_steps"]
        or metadata["optimizer_update_epochs"] != selection["best_optimizer_update_epochs"]
        or not selection["fresh_load_reproduced_actions_and_episodes"]
    ):
        raise ValueError("RL play requires the selected, replay-verified best checkpoint")
    model.policy.set_training_mode(False)
    core = GameCore(
        scenario_map(config.get("scenario", DEFAULT_SCENARIO)),
        CoreConfig(),
        alien_control="external",
    )
    core.reset(scenario_seed)
    label = f"RL sampled | {run_dir.name}/{checkpoint.name} | {metadata['actual_steps']} steps"
    driver = HumanRLController(core, model, sampling_seed, label=label)
    # Copy provenance returned by the strict loader, without a second policy load/draw.
    driver.checkpoint_identity = {
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "verified_metadata": metadata,
        "selection": selection,
    }
    return driver
