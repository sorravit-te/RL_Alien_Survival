"""Human-input adapter for the frozen Alien target contract; no Env or Survivor bot."""

import random
from contextlib import contextmanager
from math import hypot
from pathlib import Path

import numpy as np
import torch

from game.core import GameCore
from game.map import DEFAULT_SCENARIO, scenario_map
from rl.candidates import AllowedAlienView, build_candidates, candidate_masks
from rl.contract import (
    ACTION_COUNT,
    DECISION_TICKS,
    LEGACY_SEARCH_CONTROL,
    OBSERVATION_SIZE,
    WAIT,
    core_pursuit,
    validate_search_control,
)
from rl.observations import build_observation


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
        search_control=LEGACY_SEARCH_CONTROL,
    ):
        if core.alien_control != "external" or core.snapshot().alien is None:
            raise ValueError("RL play requires an active external-control Alien")
        self.search_control = validate_search_control(search_control)
        if getattr(core, "pursuit", "snapshot") != core_pursuit(search_control):
            raise ValueError("Core pursuit helper does not match the policy search_control")
        self.core, self.model = core, model
        self.config = core.config
        self.controller_label = label
        self.sampler = EpisodeSampler(model)
        self.sampling_seed = sampling_seed
        self.recorder = None
        self.reset(core.snapshot().seed)

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
        # Same permitted fields and physical ages as AlienEnv._allowed_view.
        snapshot = self.core.snapshot()
        alien, memory = snapshot.alien, snapshot.perception
        return AllowedAlienView(
            (alien.x, alien.y),
            alien.yaw,
            alien.speed_mode == "chase",
            self.last_action,
            memory.visible,
            memory.last_seen,
            memory.last_seen_age_ticks / self.config.tick_hz
            if memory.last_seen_age_ticks is not None
            else None,
            self.core.world_map.patrol_waypoints,
            tuple(
                (snapshot.tick - tick) / self.config.tick_hz if tick is not None else None
                for tick in self.visit_ticks
            ),
            bool(snapshot.navigation_path),
            memory.revision,
            snapshot.investigation,
            self.search_control,
        )

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
            source = (
                "wait"
                if action == WAIT
                else "waypoint"
                if action < 20
                else "visible"
                if action == 20
                else "last_seen"
            )
            self.core.set_alien_target(
                chosen.target, source=source, evidence_revision=chosen.evidence_revision
            )
            self.decisions += 1
            if self.recorder is not None:
                self.recorder.decision(self, before, view, observation, mask, action, chosen.target)
        snapshot = self.core.step(human_command)
        # Env records arrivals from POST-tick position, including a terminal tick.
        for index, point in enumerate(self.core.world_map.patrol_waypoints):
            if (
                hypot(snapshot.alien.x - point[0], snapshot.alien.y - point[1])
                <= self.config.target_arrival_distance
            ):
                self.visit_ticks[index] = snapshot.tick
        self.phase = (self.phase + 1) % DECISION_TICKS
        if self.recorder is not None:
            self.recorder.tick(self, human_command, before, snapshot, phase_before)
        return snapshot


def load_human_rl(run_dir, checkpoint, sampling_seed, scenario_seed=0):
    """Exact existing provenance checks; selected best only, never heuristic fallback."""
    from rl.pipeline import file_hash, load_checkpoint, read_json, read_run

    run_dir, checkpoint = Path(run_dir), Path(checkpoint)
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
    search_control = config.get("search_control", LEGACY_SEARCH_CONTROL)
    core = GameCore(
        scenario_map(config.get("scenario", DEFAULT_SCENARIO)),
        alien_control="external",
        pursuit=core_pursuit(search_control),
    )
    core.reset(scenario_seed)
    label = f"RL sampled | {run_dir.name}/{checkpoint.name} | {metadata['actual_steps']} steps | seed {sampling_seed}"
    driver = HumanRLController(
        core,
        model,
        sampling_seed,
        label=label,
        search_control=search_control,
    )
    # Copy provenance returned by the strict loader, without a second policy load/draw.
    driver.checkpoint_identity = {
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "verified_metadata": metadata,
        "selection": selection,
    }
    return driver
