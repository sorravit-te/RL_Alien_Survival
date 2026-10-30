"""Opt-in local game trace. No policy sampling, replay execution or outside-window capture."""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from math import atan2
from pathlib import Path
from time import perf_counter

from game.interaction import angle_difference, segment_clear
from game.state import SurvivorCommand

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1] / "experiments"


class RecordingError(RuntimeError):
    pass


def validate_record_directory(path):
    path = Path(path).resolve()
    root = EXPERIMENTS_ROOT.resolve()
    if path == root or not path.is_relative_to(root):
        raise RecordingError("Record directory must be a new folder inside this repo's experiments")
    if path.exists():
        raise RecordingError(
            f"Recording already exists: {path}. Choose a new folder; nothing overwritten"
        )
    return path


class GameplayRecorder:
    def __init__(self, directory, driver, *, settings=None):
        self.directory = validate_record_directory(directory)
        identity = getattr(driver, "checkpoint_identity", None)
        if not identity:
            raise RecordingError("Recording requires the loaded controller's checkpoint identity")
        self.driver = driver
        self.stream = None
        self.failed = False
        self.closed = False
        self.sequence = 0
        self.last_flushed_sequence = -1
        self.owns_directory = False
        self.episode = -1
        self.active = False
        self.started = self.last_flush = perf_counter()
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
            self.owns_directory = True
            self.stream = (self.directory / "events.jsonl").open(
                "x", encoding="utf-8", buffering=65536, newline="\n"
            )
            snapshot = driver.snapshot()
            self._json(
                "manifest.json",
                {
                    "format": "human-rl-command-recording-v2",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "scenario_seed": snapshot.seed,
                    "sampling_seed": driver.sampling_seed,
                    "policy": identity,
                    "controller": driver.controller_label,
                    "app_sha256": {
                        p.name: sha256(p.read_bytes()).hexdigest()
                        for p in sorted(Path(__file__).parent.glob("*.py"))
                    },
                    "physics_config": asdict(driver.config),
                    "settings": settings or {},
                    "diagnostic_world_only": asdict(snapshot.world_map),
                    "timing": {
                        "decision": "At tick start, before target change and Core.step",
                        "allowed_view_sensor": "Previous Core tick sensor (initial state at tick 0)",
                        "tick_sensor": "after.perception sampled at tick_before BEFORE investigation rotation and movement",
                        "geometry": "Diagnostic truth computed separately at both tick endpoints",
                    },
                    "probabilities": "omitted; recorder never calls or samples policy",
                    "privacy": "Only game commands/window lifecycle/game state; no raw desktop events",
                    "incomplete_rule": "Missing status or status other than completed means incomplete/partial",
                },
            )
            self._status("recording", "session_open")
            self.begin_episode(driver, reason="initial")
        except (OSError, ValueError, TypeError, RecordingError) as exc:
            self.failed = True
            if self.owns_directory:
                self._best_effort_status("partial", str(exc))
            if self.stream is not None:
                try:
                    self.stream.close()
                except OSError:
                    pass
            raise RecordingError(f"Cannot start recording at {self.directory}: {exc}") from exc

    def _json(self, name, value):
        temporary = self.directory / (name + ".tmp")
        with temporary.open("w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
        temporary.replace(self.directory / name)

    def _status(self, status, reason):
        self._json(
            "status.json",
            {
                "status": status,
                "reason": reason,
                "last_accepted_sequence": self.sequence - 1,
                "last_flushed_sequence": self.last_flushed_sequence,
                "episode_index": self.episode,
                "episode_active": self.active,
            },
        )

    def _best_effort_status(self, status, reason):
        try:
            self._status(status, reason)
        except OSError:
            pass  # Initial recording status remains an explicit incomplete marker.

    def _emit(self, kind, **data):
        if self.closed or self.failed:
            raise RecordingError("Recording is closed/failed; game must stop")
        try:
            now = perf_counter()
            record = {
                "sequence": self.sequence,
                "type": kind,
                "episode": self.episode,
                "session_seconds": now - self.started,
                **data,
            }
            self.stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
            self.sequence += 1
            if now - self.last_flush >= 0.5:
                self.flush()
        except (OSError, ValueError, TypeError) as exc:
            self.failed = True
            self._best_effort_status("partial", str(exc))
            raise RecordingError(
                f"Recording failed at {self.directory}: {exc}; game stopped, partial files retained"
            ) from exc

    def flush(self):
        try:
            self.stream.flush()
            self.last_flushed_sequence = self.sequence - 1
            self.last_flush = perf_counter()
        except OSError as exc:
            self.failed = True
            self._best_effort_status("partial", str(exc))
            raise RecordingError(
                f"Cannot flush recording {self.directory}: {exc}; game stopped"
            ) from exc

    def begin_episode(self, driver, *, reason):
        self.episode += 1
        self.active = True
        snapshot = driver.snapshot()
        self._emit(
            "episode_start",
            reason=reason,
            tick=snapshot.tick,
            phase=driver.phase,
            scenario_seed=snapshot.seed,
            sampling_seed=driver.sampling_seed,
            diagnostic_initial_state_only=asdict(snapshot),
        )
        self.flush()

    def end_episode(self, driver, *, reason, status):
        if not self.active:
            return
        snapshot = driver.snapshot()
        self._emit(
            "episode_end",
            reason=reason,
            status=status,
            tick=snapshot.tick,
            phase=driver.phase,
            outcome=snapshot.outcome,
        )
        self.active = False

    def before_reset(self, driver, seed):
        self.end_episode(driver, reason="reset", status="partial")
        self._emit(
            "reset_boundary",
            tick=driver.snapshot().tick,
            phase=driver.phase,
            requested_scenario_seed=seed,
            sampling_seed=driver.sampling_seed,
        )
        self.flush()

    def after_reset(self, driver):
        self.begin_episode(driver, reason="reset")

    def lifecycle(self, event, driver, **data):
        self._emit(event, tick=driver.snapshot().tick, phase=driver.phase, **data)
        self.flush()

    def decision(self, driver, before, view, observation, mask, action, target):
        after_target = driver.snapshot()
        self._emit(
            "decision",
            tick=before.tick,
            phase_before=0,
            allowed_view=asdict(view),
            allowed_view_sensor_tick_start=before.tick - 1 if before.tick else None,
            observation=observation.tolist(),
            mask=mask.tolist(),
            action=action,
            chosen_target=target,
            alien_before=asdict(before.alien),
            prior_target=before.alien_target,
            path_before=before.navigation_path,
            path_after_target=after_target.navigation_path,
            source_after_target=after_target.alien_target_source,
            investigation_before=asdict(before.investigation),
            investigation_after_target=asdict(after_target.investigation),
        )

    @staticmethod
    def _geometry(snapshot):
        alien, survivor = snapshot.alien, snapshot.survivor
        bearing = abs(
            angle_difference(atan2(survivor.y - alien.y, survivor.x - alien.x), alien.yaw)
        )
        return {
            "line_of_sight": segment_clear(
                snapshot.world_map, alien.x, alien.y, survivor.x, survivor.y
            ),
            "bearing_error_radians": bearing,
        }

    def tick(self, driver, command, before, after, phase_before):
        # after.perception is the sensor result sampled from BEFORE movement by Core.
        self._emit(
            "tick",
            tick_before=before.tick,
            tick_after=after.tick,
            phase_before=phase_before,
            phase_after=driver.phase,
            command=asdict(command),
            last_action=driver.last_action,
            alien_before=asdict(before.alien),
            alien_after=asdict(after.alien),
            chosen_target=after.alien_target,
            path_after=after.navigation_path,
            visit_ticks=list(driver.visit_ticks),
            investigation_before=asdict(before.investigation),
            investigation_after=asdict(after.investigation),
            outcome=after.outcome,
            sensor_before_movement={
                "sample_tick_start": before.tick,
                **asdict(after.perception),
                "movement_speed_mode": after.alien.speed_mode,
            },
            diagnostic_truth_only={
                "survivor_before": asdict(before.survivor),
                "survivor_after": asdict(after.survivor),
                "geometry_before_movement": self._geometry(before),
                "geometry_after_movement": self._geometry(after),
            },
            core_events=[asdict(e) for e in after.events],
        )
        if after.outcome is not None:
            self.end_episode(driver, reason="terminal", status="completed")
            self.flush()

    def close(self, *, completed, reason):
        if self.closed:
            return
        failure = None
        try:
            if not self.failed:
                self.end_episode(self.driver, reason=reason, status="partial")
                self._emit(
                    "session_end", status="completed" if completed else "partial", reason=reason
                )
                self.flush()
        except RecordingError as exc:
            failure = exc
        finally:
            try:
                if self.stream is not None:
                    self.stream.close()
            except OSError as exc:
                self.failed = True
                failure = RecordingError(f"Recording close failed: {exc}")
            self.closed = True
            try:
                self._status("completed" if completed and not self.failed else "partial", reason)
            except OSError as exc:
                failure = RecordingError(
                    f"Recording status write failed at {self.directory}: {exc}"
                )
        if failure is not None:
            raise failure


def read_records(directory, *, allow_partial_tail=False):
    """Read local data only; never load a policy or advance any simulation."""
    with (Path(directory) / "events.jsonl").open(encoding="utf-8") as f:
        expected = 0
        for line in f:
            if not line.endswith("\n") and allow_partial_tail:
                return
            record = json.loads(line)
            if record["sequence"] != expected:
                raise ValueError("Recording sequence is not contiguous")
            expected += 1
            yield record


def read_tick_commands(directory, episode):
    """Offline command extraction only, including a partial final decision block."""
    commands = []
    for record in read_records(directory, allow_partial_tail=True):
        if record["type"] == "tick" and record["episode"] == episode:
            if record["tick_before"] != len(commands) or record["tick_after"] != len(commands) + 1:
                raise ValueError("Tick command stream is not contiguous")
            commands.append(SurvivorCommand(**record["command"]))
    return commands
