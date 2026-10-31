"""Report how well an Alien hunts; rl/hunt.py describes the trials.

Usage: python -m tools.hunt_test --run-dir experiments/<run>/run --checkpoint <file.zip>
Without --checkpoint it scores the written Alien on the run's scenario.
"""

import argparse
import json
from pathlib import Path

from rl.hunt import hunt_test, policy_chooser
from rl.pipeline import load_checkpoint, read_run
from rl.scripted_controllers import HeuristicSelector


def written_alien_chooser():
    selector = {}

    def choose(env, obs, mask):
        if env.elapsed_ticks == 0:
            selector["now"] = HeuristicSelector(env.config.target_arrival_distance)
        return selector["now"].choose(env._allowed_view())

    return choose


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
    noisy, silent, hidden = result["noisy"], result["silent"], result["hidden"]
    print(
        f"{name}: noisy {noisy['caught']}/{noisy['cases']} "
        f"follows {noisy['noise_followed']}/{noisy['noise_offered']}, "
        f"silent {silent['caught']}/{silent['cases']}, "
        f"hidden {hidden['caught']}/{hidden['cases']}"
    )


if __name__ == "__main__":
    main()
