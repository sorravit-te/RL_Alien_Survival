"""Score how well an Alien hunts, with Survivors that do not depend on the training bots.

Noisy: the Survivor paces at a run in one spot. Does the Alien come to it?
Silent: the Survivor stands still in one spot. Does patrol alone ever find it?
Hidden: a silent Survivor at every station in turn, so no part of the ship counts double.
"""

from collections import Counter
from dataclasses import replace
from statistics import median

import torch

from rl.bots.explorer import ExplorerBot
from rl.bots.hunt_targets import Pacer, StillTarget
from rl.env import AlienEnv

CASES = 32
FIRST_SCENARIO_SEED = 8000  # tools.hunt_test reports these, so they never choose a checkpoint
SELECTION_SCENARIO_SEED = 9000
SAMPLING_SEED = 4242
NOISY_SECONDS, SILENT_SECONDS = 90, 180


def still_target(world):
    return StillTarget()


def stations(scenario):
    env = AlienEnv(scenario=scenario, bot_kind="stationary_debug")
    return ExplorerBot(env.world.rows, env.config, doors=env.world.doors).stations


def trial(
    choose, scenario, make_target, seconds, cases=CASES, *, first_seed=FIRST_SCENARIO_SEED, spots=None
):
    outcomes, times, offered, followed = Counter(), [], 0, 0
    env = AlienEnv(scenario=scenario, bot_kind="stationary_debug", max_episode_seconds=seconds)
    env.bot = make_target(env.world)
    for index in range(cases):
        obs, _ = env.reset(seed=index, options={"scenario_seed": first_seed + index, "bot_seed": index})
        if spots is not None:
            env.core._survivor = replace(env.core._survivor, x=spots[index][0], y=spots[index][1])
        done, info = False, {}
        while not done:
            mask = env.action_masks()
            action = choose(env, obs, mask)
            if mask.sum() > 1 and mask[22]:
                offered += 1
                followed += action == 22
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        outcomes[info["outcome"]] += 1
        if info["outcome"] == "captured":
            times.append(info["elapsed_ticks"] / env.config.tick_hz)
    return {
        "cases": cases,
        "caught": outcomes["captured"],
        "median_seconds_to_catch": median(times) if times else None,
        "noise_offered": offered,
        "noise_followed": followed,
    }


def policy_chooser(model):
    torch.manual_seed(SAMPLING_SEED)

    def choose(env, obs, mask):
        action, _ = model.predict(obs, action_masks=mask, deterministic=False)
        return int(action)

    return choose


def hunt_test(choose, scenario, *, cases=CASES, noisy_seconds=NOISY_SECONDS, first_seed=FIRST_SCENARIO_SEED):
    spots = stations(scenario)
    return {
        "noisy": trial(choose, scenario, Pacer, noisy_seconds, cases, first_seed=first_seed),
        "silent": trial(choose, scenario, still_target, SILENT_SECONDS, cases, first_seed=first_seed),
        "hidden": trial(
            choose, scenario, still_target, SILENT_SECONDS, len(spots), first_seed=first_seed, spots=spots
        ),
    }
