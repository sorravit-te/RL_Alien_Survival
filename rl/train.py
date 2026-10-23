"""CPU training pilot: prepare -> baselines -> pilot. Each phase runs explicitly once."""

import argparse
import csv
from collections import deque
from math import ceil
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from rl.evaluate import evaluate, paired_catch_difference
from rl.pipeline import (
    ManifestEnv,
    live_log,
    load_checkpoint,
    new_model,
    parameter_hash,
    prepare_run,
    read_json,
    read_run,
    save_checkpoint,
    write_json,
)

CONTROLLERS = ("random", "untrained", "heuristic", "wait")


def claim_phase(run, phase):
    # Exclusive file creation prevents two trainers or accidental repeated held-out tests.
    with (run / f"{phase}_started.txt").open("x", encoding="utf-8") as handle:
        handle.write("Started once; inspect run.log before any manual recovery.\n")


def baselines(run):
    config, manifest, source = read_run(run)
    claim_phase(run, "baselines")
    env = ManifestEnv(config, manifest["splits"]["train"])
    try:
        model = new_model(config, env)
        save_checkpoint(run / "checkpoints/untrained.zip", model, config, source, role="untrained")
        results = {}
        for kind in CONTROLLERS:
            results[kind], _ = evaluate(
                kind,
                config,
                manifest["splits"]["validation"],
                run / f"validation/baseline_{kind}.csv",
                model=model if kind == "untrained" else None,
            )
        write_json(run / "baseline_summary.json", results)
        print("BASELINES COMPLETE. Review ceiling diagnostics before invoking pilot.", flush=True)
        return results
    finally:
        env.close()


class PilotCallback(BaseCallback):
    def __init__(self, run, config, cases, source):
        super().__init__()
        self.run, self.config, self.cases, self.source = run, config, cases, source
        self.started = perf_counter()
        self.counts = dict.fromkeys(("captured", "escaped", "unresolved"), 0)
        self.returns = deque(maxlen=100)
        self.next_progress = config["progress_interval"]
        self.next_eval = config["evaluation_interval"]
        self.best_score = None
        self.best_steps = None
        self.best_rows = None
        self.best_summary = None
        self.episode_handle = (run / "train_episodes.csv").open("w", newline="", encoding="utf-8")
        self.episodes = csv.DictWriter(
            self.episode_handle,
            fieldnames=[
                "steps",
                "episode",
                "manifest_id",
                "return",
                "decision_steps",
                "elapsed_ticks",
                "outcome",
            ],
        )
        self.episodes.writeheader()
        self.curve_handle = (run / "validation_curve.csv").open("w", newline="", encoding="utf-8")
        self.curve = csv.DictWriter(
            self.curve_handle,
            fieldnames=[
                "steps",
                "elapsed_seconds",
                "episodes",
                "captured",
                "escaped",
                "unresolved",
                "mean_return",
            ],
        )
        self.curve.writeheader()

    def elapsed(self):
        return perf_counter() - self.started

    def validate_checkpoint(self, path, role):
        elapsed = self.elapsed()
        summary, rows = evaluate(
            "trained",
            self.config,
            self.cases,
            self.run / f"validation/step_{self.model.num_timesteps:08d}.csv",
            model=self.model,
            label=f"step{self.model.num_timesteps}",
        )
        save_checkpoint(
            path,
            self.model,
            self.config,
            self.source,
            role=role,
            elapsed_seconds=elapsed,
            validation=summary,
        )
        self.curve.writerow(
            {
                "steps": self.model.num_timesteps,
                "elapsed_seconds": self.elapsed(),
                **{
                    key: summary[key]
                    for key in ("episodes", "captured", "escaped", "unresolved", "mean_return")
                },
            }
        )
        self.curve_handle.flush()
        for name in ("catch_rate", "escape_rate", "unresolved_rate", "mean_return"):
            self.logger.record(f"validation/{name}", summary[name])
        self.logger.dump(self.model.num_timesteps)
        score = (summary["captured"], -summary["unresolved"])
        if self.best_score is None or score > self.best_score:
            self.best_score, self.best_steps = score, self.model.num_timesteps
            self.best_rows, self.best_summary = rows, summary
            save_checkpoint(
                self.run / "checkpoints/best.zip",
                self.model,
                self.config,
                self.source,
                role="best_validation",
                elapsed_seconds=self.elapsed(),
                validation=summary,
            )
            print(f"SELECTION best trained checkpoint = {self.best_steps} steps", flush=True)
        return summary

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done:
                episode = info["episode"]
                self.counts[info["outcome"]] += 1
                self.returns.append(episode["r"])
                self.episodes.writerow(
                    {
                        "steps": self.num_timesteps,
                        "episode": sum(self.counts.values()),
                        "manifest_id": info["manifest_id"],
                        "return": episode["r"],
                        "decision_steps": episode["l"],
                        "elapsed_ticks": info["elapsed_ticks"],
                        "outcome": info["outcome"],
                    }
                )
                self.episode_handle.flush()
        if self.num_timesteps >= self.next_progress:
            elapsed = self.elapsed()
            mean = float(np.mean(self.returns)) if self.returns else 0.0
            print(
                f"TRAIN steps={self.num_timesteps} episodes={sum(self.counts.values())} "
                f"elapsed={elapsed:.1f}s decision_steps/s={self.num_timesteps / max(elapsed, 1e-9):.1f} "
                f"return100={mean:.3f} Catch/Escape/Unresolved="
                f"{self.counts['captured']}/{self.counts['escaped']}/{self.counts['unresolved']}",
                flush=True,
            )
            self.logger.record("train/episode_return100", mean)
            for name, value in self.counts.items():
                self.logger.record(f"train/episodes_{name}", value)
            self.logger.dump(self.num_timesteps)
            self.next_progress += self.config["progress_interval"]
        if self.num_timesteps >= self.next_eval:
            self.validate_checkpoint(
                self.run / f"checkpoints/step_{self.num_timesteps:08d}.zip", "interval"
            )
            self.next_eval += self.config["evaluation_interval"]
        return True

    def close(self):
        self.episode_handle.close()
        self.curve_handle.close()


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_curves(run, baseline, test):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes, curve = read_rows(run / "train_episodes.csv"), read_rows(run / "validation_curve.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    returns = np.array([float(row["return"]) for row in episodes])
    steps = [int(row["steps"]) for row in episodes]
    axes[0].scatter(steps, returns, s=2, alpha=0.18)
    window = min(100, len(returns))
    if window:
        axes[0].plot(steps[window - 1 :], np.convolve(returns, np.ones(window) / window, "valid"))
    axes[0].set(
        title="Training return (100-episode mean)", xlabel="Decision steps", ylabel="Raw return"
    )
    axes[1].plot(
        [int(row["steps"]) for row in curve],
        [int(row["captured"]) / int(row["episodes"]) for row in curve],
        marker="o",
        label="Trained validation",
    )
    for kind in ("random", "untrained", "heuristic", "wait"):
        axes[1].axhline(baseline[kind]["catch_rate"], linestyle="--", label=kind)
    axes[1].set(
        title="Validation (50 episodes/checkpoint)",
        xlabel="Decision steps",
        ylabel="Catch rate",
        ylim=(-0.05, 1.05),
    )
    axes[1].legend(fontsize=7)
    for index, (kind, result) in enumerate(test.items()):
        rate = result["catch_rate"]
        low, high = result["catch_wilson95"]
        axes[2].errorbar(index, rate, yerr=[[rate - low], [high - rate]], fmt="o", capsize=4)
    axes[2].set_xticks(range(len(test)), test.keys(), rotation=25)
    axes[2].set(
        title="Final test: catch with Wilson 95% CI", ylabel="Catch rate", ylim=(-0.05, 1.05)
    )
    fig.tight_layout()
    fig.savefig(run / "learning_curves.png", dpi=150)
    plt.close(fig)


def pilot(run):
    config, manifest, source = read_run(run)
    baseline = read_json(run / "baseline_summary.json")
    claim_phase(run, "pilot")
    torch.set_num_threads(config["torch_threads"])
    env = Monitor(
        ManifestEnv(config, manifest["splits"]["train"]),
        filename=str(run / "sb3_train.monitor.csv"),
        info_keywords=("outcome", "elapsed_ticks"),
    )
    callback = None
    started = perf_counter()
    try:
        model, untrained_meta = load_checkpoint(
            run / "checkpoints/untrained.zip", config, source, env
        )
        model.set_logger(configure(str(run / "tensorboard"), ["csv", "tensorboard"]))
        callback = PilotCallback(run, config, manifest["splits"]["validation"], source)
        expected = (
            ceil(config["requested_steps"] / config["ppo"]["n_steps"]) * config["ppo"]["n_steps"]
        )
        print(
            f"PILOT CPU threads={config['torch_threads']} envs=1 requested={config['requested_steps']} "
            f"rollout-rounded={expected}; throughput is decisions/s, NOT renderer FPS.",
            flush=True,
        )
        model.learn(
            total_timesteps=config["requested_steps"],
            callback=callback,
            use_masking=True,
            reset_num_timesteps=True,
        )
        callback.validate_checkpoint(run / "checkpoints/last.zip", "last")
        if model.num_timesteps != expected:
            raise RuntimeError("Unexpected rollout step count")
        training_seconds = perf_counter() - started
        # Fresh deserialization must reproduce the selected validation trajectories/actions.
        best, best_meta = load_checkpoint(run / "checkpoints/best.zip", config, source)
        reproduced, reproduced_rows = evaluate(
            "trained",
            config,
            manifest["splits"]["validation"],
            run / "validation/best_reloaded.csv",
            model=best,
            label="reload_check",
        )
        if reproduced_rows != callback.best_rows or reproduced != callback.best_summary:
            raise RuntimeError("Fresh checkpoint validation replay did not reproduce")
        write_json(
            run / "selection.json",
            {
                "best_steps": callback.best_steps,
                "rule": config["selection"],
                "best_validation": reproduced,
                "fresh_load_reproduced_actions_and_episodes": True,
            },
        )
        # Select first, then use held-out test only once. Never feed test results into selection.
        claim_phase(run, "test")
        untrained, _ = load_checkpoint(run / "checkpoints/untrained.zip", config, source)
        test, rows = {}, {}
        for kind in (*CONTROLLERS, "trained"):
            test[kind], rows[kind] = evaluate(
                kind,
                config,
                manifest["splits"]["test"],
                run / f"test/{kind}.csv",
                model=best if kind == "trained" else untrained if kind == "untrained" else None,
                label="final_test",
            )
        result = {
            "requested_steps": config["requested_steps"],
            "actual_steps": model.num_timesteps,
            "training_seconds_including_validation": training_seconds,
            "total_pilot_seconds": perf_counter() - started,
            "decision_steps_per_second_including_validation": model.num_timesteps
            / training_seconds,
            "best_steps": callback.best_steps,
            "baseline_validation": baseline,
            "best_validation": reproduced,
            "test": test,
            "test_trained_minus_untrained": paired_catch_difference(
                rows["trained"], rows["untrained"]
            ),
            "test_trained_minus_random": paired_catch_difference(rows["trained"], rows["random"]),
            "weights_changed": parameter_hash(model) != untrained_meta["policy_sha256"],
            "fresh_load_reproduced": True,
            "training_episodes": sum(callback.counts.values()),
            "training_outcomes": callback.counts,
            "limitations": [
                "Single training seed; pilot results do not establish generalization.",
                "Fixed layout/card/exit, two real spawns. Numeric seed holdout is not layout generalization.",
                "Heuristic selects from the action catalog using the latest completed perception; internal game sensing occurs on the next physics tick.",
                "Weights/reward alone do not prove learning; baseline ceilings may limit this task.",
            ],
        }
        write_json(run / "results.json", result)
        plot_curves(run, baseline, test)
        print(
            f"PILOT COMPLETE actual_steps={model.num_timesteps} best_steps={callback.best_steps} "
            f"train_wall={training_seconds:.2f}s; see results.json. App integration is a separate step.",
            flush=True,
        )
        return result
    finally:
        if callback is not None:
            callback.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "baselines", "pilot"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/training_pilot.json"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare_run(args.run_dir, read_json(args.config))
        print(f"PREPARED config and manifest: {args.run_dir}", flush=True)
        return
    with live_log(args.run_dir / "run.log"):
        try:
            (baselines if args.phase == "baselines" else pilot)(args.run_dir)
        except Exception:
            import traceback

            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
