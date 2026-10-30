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

from rl.evaluate import evaluate_suite, frozen_evaluation
from rl.evaluation_protocol import trained_score
from rl.pipeline import (
    ManifestEnv,
    digest,
    file_hash,
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
        results, _ = evaluate_suite(
            config,
            manifest["splits"]["validation"],
            run / "validation/baselines",
            schedule=read_json(run / "sampling_schedule.json")["validation"],
            models={"untrained": model},
            stage="baseline",
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
                "optimizer_update_epochs",
                "evaluation_context",
                "elapsed_seconds",
                "sampled_primary_case_mean_catch_rate",
                "sampled_primary_case_mean_unresolved_rate",
                "sampled_primary_case_mean_return",
                "argmax_secondary_case_mean_catch_rate",
            ],
        )
        self.curve.writeheader()

    def elapsed(self):
        return perf_counter() - self.started

    def validate_checkpoint(self, path, role):
        config, _, _ = read_run(self.run)  # Recheck frozen files before evaluation.
        if config != self.config:
            raise ValueError("Callback frozen config changed")
        context = (
            "rollout_callback_before_current_rollout_optimizer_update"
            if role == "interval"
            else "after_learning_optimizer_update"
        )
        steps, epochs = int(self.model.num_timesteps), int(self.model._n_updates)
        summary, rows = evaluate_suite(
            self.config,
            self.cases,
            self.run / f"validation/step_{steps:08d}_epochs_{epochs:06d}_{role}",
            schedule=read_json(self.run / "sampling_schedule.json")["validation"],
            models={"trained": self.model},
            stage="trained",
        )
        save_checkpoint(
            path,
            self.model,
            self.config,
            self.source,
            role=role,
            elapsed_seconds=self.elapsed(),
            validation=summary,
            evaluation_context=context,
        )
        primary, secondary = summary["trained_sampled"], summary["trained_deterministic"]
        self.curve.writerow(
            {
                "steps": steps,
                "optimizer_update_epochs": epochs,
                "evaluation_context": context,
                "elapsed_seconds": self.elapsed(),
                "sampled_primary_case_mean_catch_rate": primary["case_mean_sampled_catch_rate"],
                "sampled_primary_case_mean_unresolved_rate": primary[
                    "case_mean_sampled_unresolved_rate"
                ],
                "sampled_primary_case_mean_return": primary["case_mean_return"],
                "argmax_secondary_case_mean_catch_rate": secondary["case_mean_catch_rate"],
            }
        )
        self.curve_handle.flush()
        for prefix, values in (("sampled_primary", primary), ("argmax_secondary", secondary)):
            for name in ("case_mean_catch_rate", "case_mean_unresolved_rate", "case_mean_return"):
                self.logger.record(f"validation/{prefix}_{name}", values[name])
        self.logger.record("validation/evaluated_optimizer_update_epochs", epochs)
        self.logger.dump(steps)
        score = trained_score(summary, steps, epochs)
        if score is not None and (self.best_score is None or score > self.best_score):
            self.best_score, self.best_steps = score, steps
            self.best_rows, self.best_summary = rows, summary
            save_checkpoint(
                self.run / "checkpoints/best.zip",
                self.model,
                self.config,
                self.source,
                role="best_validation",
                elapsed_seconds=self.elapsed(),
                validation=summary,
                evaluation_context=context,
            )
            print(
                f"SELECTION sampled-primary best trained checkpoint: decisions={steps} "
                f"optimizer_update_epochs={epochs}",
                flush=True,
            )
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


def validate_fresh_start(config, metadata):
    """Fresh pilot must originate from this run's compatible, unoptimized baseline."""
    if config.get("fresh_run") and (
        metadata.get("role") != "untrained"
        or metadata.get("actual_steps") != 0
        or metadata.get("optimizer_update_epochs") != 0
        or metadata.get("training_seed") != config["training_seed"]
        or metadata.get("config_sha256") != digest(config)
    ):
        raise ValueError("Fresh run requires its own zero-step, zero-update untrained policy")


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
        validate_fresh_start(config, untrained_meta)
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
        return complete_sampled_pilot(
            run,
            config,
            manifest,
            source,
            model,
            untrained_meta,
            callback,
            baseline,
            started,
            training_seconds,
        )
    finally:
        if callback is not None:
            callback.close()
        env.close()


def final_sampled_test(run):
    """Exclusive, fail-closed final test. A failed claimed test stays partial, never retries."""
    config, manifest, source = read_run(run)
    selection = read_json(run / "selection.json")
    best_path = run / "checkpoints/best.zip"
    metadata = read_json(best_path.with_suffix(".metadata.json"))
    replay_dir = run / "validation/best_reloaded"
    replay_summary = read_json(replay_dir / "summary.json")
    replay_rows = read_json(replay_dir / "episodes.json")
    if (
        selection.get("fresh_load_reproduced_actions_and_episodes") is not True
        or selection.get("best_checkpoint_sha256") != file_hash(best_path)
        or selection.get("best_steps") != metadata["actual_steps"]
        or selection.get("best_optimizer_update_epochs") != metadata["optimizer_update_epochs"]
        or metadata.get("role") != "best_validation"
        or selection.get("best_validation") != metadata["validation"]
        or replay_summary != metadata["validation"]
        or selection.get("validation_action_rows_sha256") != digest(replay_rows)
        or selection.get("rule") != config["frozen_evaluation"]["protocol"]["selection"]
        or trained_score(
            selection["best_validation"],
            selection["best_steps"],
            selection["best_optimizer_update_epochs"],
        )
        is None
    ):
        raise ValueError("A selected, replay-verified trained checkpoint is required before test")
    claim_phase(run, "test")
    write_json(run / "test_status.json", {"status": "started", "retry_allowed": False})
    try:
        with frozen_evaluation(None):
            best, _ = load_checkpoint(best_path, config, source)
            untrained, _ = load_checkpoint(run / "checkpoints/untrained.zip", config, source)
            summary, rows = evaluate_suite(
                config,
                manifest["splits"]["test"],
                run / "test/final_sampled",
                schedule=read_json(run / "sampling_schedule.json")["test"],
                models={"trained": best, "untrained": untrained},
                stage="final",
            )
        write_json(run / "test_summary.json", summary)
        write_json(run / "test_status.json", {"status": "complete", "retry_allowed": False})
        return summary, rows
    except BaseException as exc:
        write_json(
            run / "test_status.json",
            {"status": "partial", "retry_allowed": False, "error": f"{type(exc).__name__}: {exc}"},
        )
        raise


def complete_sampled_pilot(
    run,
    config,
    manifest,
    source,
    model,
    untrained_meta,
    callback,
    baseline,
    started,
    training_seconds,
):
    if callback.best_score is None:
        raise RuntimeError("No optimizer-updated checkpoint eligible for selection; test forbidden")
    with frozen_evaluation(None):
        best, metadata = load_checkpoint(run / "checkpoints/best.zip", config, source)
        reproduced, rows = evaluate_suite(
            config,
            manifest["splits"]["validation"],
            run / "validation/best_reloaded",
            schedule=read_json(run / "sampling_schedule.json")["validation"],
            models={"trained": best},
            stage="trained",
        )
    if reproduced != callback.best_summary or rows != callback.best_rows:
        raise RuntimeError(
            "Fresh sampled/argmax checkpoint replay did not reproduce actions/outcomes"
        )
    write_json(
        run / "selection.json",
        {
            "best_steps": callback.best_steps,
            "best_optimizer_update_epochs": metadata["optimizer_update_epochs"],
            "best_checkpoint_sha256": file_hash(run / "checkpoints/best.zip"),
            "rule": config["frozen_evaluation"]["protocol"]["selection"],
            "best_validation": reproduced,
            "validation_action_rows_sha256": digest(rows),
            "fresh_load_reproduced_actions_and_episodes": True,
        },
    )
    # New scene pilots stop before held-out runtime unless separately invoked.
    if config.get("defer_final_test", False):
        test = None
    else:
        test, _ = final_sampled_test(run)
    result = {
        "evaluation_mode": "sampled_primary_argmax_secondary",
        "requested_steps": config["requested_steps"],
        "actual_steps": model.num_timesteps,
        "optimizer_update_epochs": model._n_updates,
        "training_seconds_including_validation": training_seconds,
        "total_pilot_seconds": perf_counter() - started,
        "best_steps": callback.best_steps,
        "best_optimizer_update_epochs": metadata["optimizer_update_epochs"],
        "baseline_validation": baseline,
        "best_validation": reproduced,
        "test": test,
        "test_status": "deferred_pending_separate_authorization" if test is None else "complete",
        "weights_changed": parameter_hash(model) != untrained_meta["policy_sha256"],
        "fresh_load_reproduced": True,
        "training_episodes": sum(callback.counts.values()),
        "training_outcomes": callback.counts,
        "limitations": [
            "Three sampling repeats are not three training seeds.",
            "One fixed map, two spawns; no layout generalization claim.",
            "Argmax is secondary and never selects best.",
        ],
    }
    write_json(run / "results.json", result)
    print("PILOT COMPLETE sampled-primary; review case-cluster results.json", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("prepare", "baselines", "pilot", "final-test"), required=True
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pursuit_tracking_training_pilot.json")
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare_run(args.run_dir, read_json(args.config))
        print(f"PREPARED config and manifest: {args.run_dir}", flush=True)
        return
    with live_log(args.run_dir / "run.log"):
        try:
            {"baselines": baselines, "pilot": pilot, "final-test": final_sampled_test}[args.phase](
                args.run_dir
            )
        except Exception:
            import traceback

            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
