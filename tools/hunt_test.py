"""Score how well an Alien hunts, with Survivors that do not depend on the training bots.

Noisy: the Survivor paces at a run in one spot. Does the Alien come to it?
Silent: the Survivor stands still in one spot. Does patrol alone ever find it?

Usage: python -m tools.hunt_test --run-dir experiments/<run>/run --checkpoint <file.zip>
Without --checkpoint it scores the written Alien on the run's scenario.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median

from rl.bots.hunt_targets import Pacer, StillTarget
from rl.env import AlienEnv
from rl.pipeline import load_checkpoint, read_run
from rl.scripted_controllers import HeuristicSelector

CASES = 32
FIRST_SCENARIO_SEED = 8000
SAMPLING_SEED = 4242
NOISY_SECONDS, SILENT_SECONDS = 90, 180


def trial(choose, scenario, make_target, seconds, cases=CASES):
    outcomes, times, offered, followed = Counter(), [], 0, 0
    env = AlienEnv(scenario=scenario, bot_kind="stationary_debug", max_episode_seconds=seconds)
    env.bot = make_target(env.world)
    for index in range(cases):
        obs, _ = env.reset(
            seed=index, options={"scenario_seed": FIRST_SCENARIO_SEED + index, "bot_seed": index}
        )
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
    model.set_random_seed(SAMPLING_SEED)

    def choose(env, obs, mask):
        action, _ = model.predict(obs, action_masks=mask, deterministic=False)
        return int(action)

    return choose


def written_alien_chooser():
    selector = {}

    def choose(env, obs, mask):
        if env.elapsed_ticks == 0:
            selector["now"] = HeuristicSelector(env.config.target_arrival_distance)
        return selector["now"].choose(env._allowed_view())

    return choose


def hunt_test(choose, scenario, *, cases=CASES, noisy_seconds=NOISY_SECONDS):
    return {
        "noisy": trial(choose, scenario, Pacer, noisy_seconds, cases),
        "silent": trial(choose, scenario, lambda world: StillTarget(), SILENT_SECONDS, cases),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    config, _, source = read_run(args.run_dir)
    if args.checkpoint is None:
        name, choose = "written_alien", written_alien_chooser()
    else:
        model, _ = load_checkpoint(args.checkpoint, config, source)
        name, choose = args.checkpoint.stem, policy_chooser(model)
    result = hunt_test(choose, config["scenario"])
    output = args.run_dir / "hunt_test" / f"{name}.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    noisy, silent = result["noisy"], result["silent"]
    print(
        f"{name}: noisy {noisy['caught']}/{noisy['cases']} "
        f"follows {noisy['noise_followed']}/{noisy['noise_offered']}, "
        f"silent {silent['caught']}/{silent['cases']}"
    )


if __name__ == "__main__":
    main()
