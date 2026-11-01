"""Recorder integration with fake sampled policies only; never loads frozen checkpoints."""

import json
from dataclasses import asdict, replace

import numpy as np
import pygame
import pytest
import torch
from test_human_rl import Policy, arena

from app import recorder as recording
from app.loop import FixedStepLoop
from app.play import PlaySession
from app.recorder import GameplayRecorder, RecordingError, read_records, read_tick_commands
from app.rl_controller import HumanRLController, rng_state
from game.core import GameCore
from game.state import CoreConfig, SurvivorCommand


@pytest.fixture
def root(tmp_path, monkeypatch):
    path = tmp_path / "experiments"
    path.mkdir()
    monkeypatch.setattr(recording, "EXPERIMENTS_ROOT", path)
    return path


def driver(*, world=None, config=None, sampled=True, action=0):
    app = HumanRLController(
        GameCore(world or arena(), config, alien_control="external"),
        Policy(action, sampled=sampled),
        991,
    )
    app.reset(7799101)
    app.checkpoint_identity = {"synthetic_fixture": True, "source_sha256": "fake-only"}
    return app


def attach(path, app):
    result = GameplayRecorder(path, app, settings={"synthetic": True})
    app.recorder = result
    return result


def same_rng(a, b):
    assert a[0] == b[0]
    assert a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1])
    assert a[1][2:] == b[1][2:]
    assert torch.equal(a[2], b[2])


def test_recording_preserves_every_tick_allowed_inputs_actions_and_private_global_rng(root):
    plain, logged = driver(), driver()
    rec = attach(root / "parity", logged)
    commands = [
        SurvivorCommand(forward=0.2, strafe=(-1) ** i * 0.3, yaw_delta=0.01, interact=i == 5)
        for i in range(46)
    ]
    for command in commands:
        before = rng_state()
        assert plain.step(command) == logged.step(command)
        same_rng(before, rng_state())
        same_rng(plain.sampler.state, logged.sampler.state)
        assert plain.phase == logged.phase and plain.visit_ticks == logged.visit_ticks
        assert plain.allowed_view() == logged.allowed_view()
    rec.close(completed=True, reason="synthetic_end")
    assert len(plain.model.calls) == len(logged.model.calls) == 4
    for a, b in zip(plain.model.calls, logged.model.calls):
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
    records = list(read_records(rec.directory))
    decisions = [r for r in records if r["type"] == "decision"]
    assert [r["tick"] for r in decisions] == [0, 15, 30, 45]
    assert [r["allowed_view_sensor_tick_start"] for r in decisions] == [None, 14, 29, 44]
    assert all("diagnostic_truth_only" not in r for r in decisions)
    assert read_tick_commands(rec.directory, 0) == commands
    ticks = [r for r in records if r["type"] == "tick"]
    assert all(r["sensor_before_movement"]["sample_tick_start"] == r["tick_before"] for r in ticks)
    assert all(r["tick_after"] == r["tick_before"] + 1 for r in ticks)
    assert len(ticks) == 46 and records[-2]["status"] == "partial"
    manifest = json.loads((rec.directory / "manifest.json").read_text())
    assert manifest["policy"] == logged.checkpoint_identity
    assert manifest["scenario_seed"] == 7799101 and manifest["sampling_seed"] == 991
    assert manifest["app_sha256"] and "geometry" in manifest["timing"]
    assert json.loads((rec.directory / "status.json").read_text())["status"] == "completed"
    # Reconstruct with only recorded commands and a fresh FAKE sampled policy.
    replay = driver()
    for command in read_tick_commands(rec.directory, 0):
        replay.step(command)
    assert replay.snapshot() == logged.snapshot()
    same_rng(replay.sampler.state, logged.sampler.state)


@pytest.mark.parametrize("fps", [30, 60, 144])
def test_fixedstep_recorded_and_disabled_parity_at_multiple_render_rates(root, fps):
    apps = driver(), driver()
    rec = attach(root / str(fps), apps[1])
    loops = [FixedStepLoop(a) for a in apps]
    for loop in loops:
        loop.inputs.key("w", True)
        loop.inputs.key("right", True)
        loop.inputs.mouse(12, -20)
        loop.inputs.key("e", True)
        loop.inputs.key("e", False)
    for _ in range(fps * 2):
        for loop in loops:
            loop.advance(1 / fps)
        assert loops[0].current == loops[1].current
        assert loops[0].display_pitch() == loops[1].display_pitch()
    rec.close(completed=True, reason="synthetic_end")
    commands = read_tick_commands(rec.directory, 0)
    assert len(commands) == 60 and sum(c.interact for c in commands) == 1
    assert apps[0].decisions == apps[1].decisions == 4
    assert [p.name for p in root.iterdir()] == [str(fps)]


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.display.init()
    pygame.font.init()
    pygame.display.set_mode((320, 180))
    yield
    pygame.quit()


def test_focus_pause_resume_reset_records_boundaries_without_advancing_or_sampling(root, window):
    app = driver()
    rec = attach(root / "lifecycle", app)
    loop = FixedStepLoop(app)
    session = PlaySession(loop)
    session.focused = True
    escape = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)
    session.events([escape])
    for _ in range(7):
        loop.advance(loop.dt)
    before, state = app.snapshot(), app.sampler.state
    loop.inputs.key("e", True)
    loop.inputs.mouse(100, -100)
    session.events([pygame.event.Event(pygame.WINDOWFOCUSLOST)])
    loop.advance(99)
    session.events(
        [
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_e),
            pygame.event.Event(pygame.MOUSEMOTION, rel=(500, 500)),
        ]
    )
    session.events([pygame.event.Event(pygame.WINDOWFOCUSGAINED)])
    assert loop.paused and app.snapshot() == before and app.phase == 7
    same_rng(state, app.sampler.state)
    assert loop.inputs.consume(loop.dt) == SurvivorCommand()
    session.events([escape])
    for _ in range(8):
        loop.advance(loop.dt)
    assert app.phase == 0 and app.decisions == 1
    session.events([escape])  # Manual pause also consumes no simulation ticks.
    loop.advance(100)
    session.events([escape])
    loop.advance(loop.dt)
    assert app.decisions == 2
    session.events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)])
    assert loop.current.tick == app.phase == app.decisions == 0
    assert loop.display_pitch() == 0 and loop.inputs.consume(loop.dt) == SurvivorCommand()
    loop.advance(loop.dt)
    rec.close(completed=True, reason="synthetic_end")
    records = list(read_records(rec.directory))
    assert len(read_tick_commands(rec.directory, 0)) == 16
    assert len(read_tick_commands(rec.directory, 1)) == 1
    assert [r["scenario_seed"] for r in records if r["type"] == "episode_start"] == [7799101] * 2
    assert [r["tick"] for r in records if r["type"] == "focus_lost"] == [7]
    assert [r["tick"] for r in records if r["type"] == "focus_gained"] == [7]
    ends = [r for r in records if r["type"] == "episode_end"]
    assert ends[0]["reason"] == "reset" and ends[0]["status"] == "partial"


def test_new_round_records_both_seeds_and_same_round_replays_them(root, window):
    app = driver()
    rec = attach(root / "rounds", app)
    loop = FixedStepLoop(app)
    session = PlaySession(loop)
    session.focused = True
    key = {name: pygame.event.Event(pygame.KEYDOWN, key=name) for name in (pygame.K_n, pygame.K_r)}

    def play():
        actions = []
        for _ in range(46):
            app.step(SurvivorCommand())
            actions.append(app.last_action)
        return actions

    play()
    session.events([key[pygame.K_n]])
    seeds = app.snapshot().seed, app.sampling_seed
    assert seeds[0] != 7799101 and seeds[1] != 991
    assert app.controller_label.endswith(f"seed {app.sampling_seed}")
    assert app.controller_label in pygame.display.get_caption()[0]
    first = play()
    session.events([key[pygame.K_r]])
    assert (app.snapshot().seed, app.sampling_seed) == seeds and play() == first
    rec.close(completed=True, reason="synthetic_end")
    records = list(read_records(rec.directory))
    starts = [r for r in records if r["type"] == "episode_start"]
    assert [(r["scenario_seed"], r["sampling_seed"]) for r in starts] == [
        (7799101, 991),
        seeds,
        seeds,
    ]
    boundaries = [r for r in records if r["type"] == "reset_boundary"]
    assert [(r["requested_scenario_seed"], r["sampling_seed"]) for r in boundaries] == [seeds] * 2


def test_early_capture_full_episode_no_extra_terminal_draw_and_reset(root):
    world = replace(arena(), alien_spawn=(6.5, 2.5, 0), patrol_waypoints=((2.5, 2.5),))
    app = driver(
        world=world, config=CoreConfig(alien_search_speed=120, alien_chase_speed=120), sampled=False
    )
    rec = attach(root / "capture", app)
    app.step(SurvivorCommand())
    assert app.snapshot().outcome == "captured" and app.snapshot().tick == 1
    app.step(SurvivorCommand())
    assert app.phase == app.decisions == 1 and len(app.model.calls) == 1
    app.reset(7799101)
    assert app.phase == app.decisions == 0
    app.step(SurvivorCommand())
    rec.close(completed=True, reason="synthetic_end")
    ends = [r for r in read_records(rec.directory) if r["type"] == "episode_end"]
    assert len(ends) == 2 and all(r["status"] == "completed" for r in ends)
    assert all(r["tick"] == 1 and r["outcome"] == "captured" for r in ends)
    assert (
        len(read_tick_commands(rec.directory, 0)) == len(read_tick_commands(rec.directory, 1)) == 1
    )


def test_sensor_geometry_have_distinct_time_coordinates(root):
    app = driver(sampled=False, action=1)
    rec = attach(root / "sensor", app)
    before = app.snapshot()
    after = app.step(SurvivorCommand(forward=1, yaw_delta=0.5))
    rec.close(completed=True, reason="synthetic_end")
    tick = next(r for r in read_records(rec.directory) if r["type"] == "tick")
    assert tick["sensor_before_movement"]["visible"] == after.perception.visible
    assert tick["diagnostic_truth_only"]["survivor_before"] == asdict(before.survivor)
    assert tick["diagnostic_truth_only"]["survivor_after"] == asdict(after.survivor)
    assert tick["diagnostic_truth_only"]["geometry_before_movement"] == rec._geometry(before)
    assert tick["diagnostic_truth_only"]["geometry_after_movement"] == rec._geometry(after)
    assert rec._geometry(before) != rec._geometry(after)


@pytest.mark.parametrize("completed", [False, True])
def test_graceful_partial_and_completed_sessions_keep_partial_last_episode(root, completed):
    app = driver()
    rec = attach(root / "close", app)
    for _ in range(7):
        app.step(SurvivorCommand())
    rec.close(completed=completed, reason="window_close" if completed else "exception")
    rec.close(completed=completed, reason="idempotent")
    status = json.loads((rec.directory / "status.json").read_text())
    assert status["status"] == ("completed" if completed else "partial")
    assert rec.closed and rec.stream.closed and len(read_tick_commands(rec.directory, 0)) == 7
    records = list(read_records(rec.directory))
    assert records[-2]["type"] == "episode_end" and records[-2]["status"] == "partial"
    assert status["last_flushed_sequence"] == records[-1]["sequence"]


def test_existing_or_outside_directory_refused_before_mutation(root):
    existing = root / "existing"
    existing.mkdir()
    (existing / "status.json").write_text("original")
    for path in (existing, root, root.parent / "outside"):
        with pytest.raises(RecordingError):
            attach(path, driver())
    assert (existing / "status.json").read_text() == "original"
    assert not (root.parent / "outside").exists()
    assert len(list(existing.iterdir())) == 1


def test_mkdir_race_never_changes_other_recording(root, monkeypatch):
    app = driver()
    from pathlib import Path

    original = Path.mkdir

    def race(path, *args, **kwargs):
        if path.name == "race":
            original(path)
            (path / "status.json").write_text("other owner's file")
            raise FileExistsError("race fixture")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", race)
    with pytest.raises(RecordingError, match="race fixture"):
        attach(root / "race", app)
    assert (root / "race/status.json").read_text() == "other owner's file"


class BrokenStream:
    def __init__(self, stream, operation):
        self.stream, self.operation = stream, operation

    def write(self, text):
        if self.operation == "write":
            raise OSError("disk fixture unavailable")
        return self.stream.write(text)

    def flush(self):
        if self.operation == "flush":
            raise OSError("disk fixture unavailable")
        self.stream.flush()

    def close(self):
        self.stream.close()


@pytest.mark.parametrize("operation", ["write", "flush"])
def test_io_failure_is_visible_stops_controller_and_retains_partial(root, operation):
    app = driver()
    rec = attach(root / operation, app)
    app.step(SurvivorCommand())
    stream = rec.stream
    rec.stream = BrokenStream(stream, operation)
    with pytest.raises(RecordingError, match="disk fixture"):
        if operation == "write":
            app.step(SurvivorCommand())
        else:
            rec.flush()
    rec.close(completed=False, reason="exception")
    assert stream.closed and rec.failed
    assert json.loads((rec.directory / "status.json").read_text())["status"] == "partial"
    assert len(app.model.calls) == 1  # No controller fallback or extra sampling.
    assert len(read_tick_commands(rec.directory, 0)) == 1


def test_partial_trailing_record_is_readable_but_midstream_corruption_rejected(root):
    app = driver()
    rec = attach(root / "tail", app)
    app.step(SurvivorCommand())
    rec.close(completed=False, reason="exception")
    path = rec.directory / "events.jsonl"
    with path.open("a") as stream:
        stream.write('{"sequence":')
    assert len(read_tick_commands(rec.directory, 0)) == 1
    with pytest.raises(json.JSONDecodeError):
        list(read_records(rec.directory))
    with path.open("a") as stream:
        stream.write("\n")
    with pytest.raises(json.JSONDecodeError):
        list(read_records(rec.directory, allow_partial_tail=True))


def cli(path=None):
    args = [
        "--controller",
        "rl",
        "--run-dir",
        "fake",
        "--checkpoint",
        "fake.zip",
        "--sampling-seed",
        "991",
        "--scenario-seed",
        "7799101",
        "--windowed",
    ]
    return args if path is None else args + ["--record-dir", str(path)]


@pytest.mark.parametrize(
    "mode", ["disabled", "close", "renderer_error", "logger_error", "init_error", "tick_error"]
)
def test_cli_recording_cleanup_no_real_policy_or_assets(root, monkeypatch, capsys, mode):
    from app import play

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = driver()
    monkeypatch.setattr("app.rl_controller.load_human_rl", lambda *a: app)
    monkeypatch.setattr(pygame.event, "get", lambda: [pygame.event.Event(pygame.QUIT)])

    class Renderer:
        def __init__(self):
            if mode == "renderer_error":
                raise RuntimeError("renderer fixture failed")

        def render(self, *a, **kw):
            return pygame.Surface((320, 180))

    monkeypatch.setattr(play, "RaycastingRenderer", Renderer)
    if mode == "tick_error":
        monkeypatch.setattr(pygame.key, "get_focused", lambda: True)
        batches = iter([[pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)], []])
        monkeypatch.setattr(pygame.event, "get", lambda: next(batches))
        counter = iter(i * 0.04 for i in range(20))
        monkeypatch.setattr(play, "perf_counter", lambda: next(counter))
    original_emit = GameplayRecorder._emit

    def broken_emit(self, kind, **data):
        if mode == "tick_error" and kind == "tick":
            self.stream = BrokenStream(self.stream, "write")
        return original_emit(self, kind, **data)

    monkeypatch.setattr(GameplayRecorder, "_emit", broken_emit)
    original = GameplayRecorder._json

    def broken_json(self, name, data):
        if mode == "init_error" and name == "manifest.json":
            raise OSError("manifest fixture failed")
        return original(self, name, data)

    monkeypatch.setattr(GameplayRecorder, "_json", broken_json)
    original_event = GameplayRecorder.lifecycle

    def broken_lifecycle(self, *a, **kw):
        if mode == "logger_error":
            self.stream = BrokenStream(self.stream, "write")
        return original_event(self, *a, **kw)

    monkeypatch.setattr(GameplayRecorder, "lifecycle", broken_lifecycle)
    path = root / mode
    if mode.endswith("error"):
        with pytest.raises(SystemExit) as error:
            play.main(cli(path))
        assert error.value.code == 2 and "fixture" in capsys.readouterr().err
    else:
        play.main(cli(None if mode == "disabled" else path))
    assert not pygame.display.get_init() and not pygame.font.get_init()
    assert len(app.model.calls) == (1 if mode == "tick_error" else 0)
    if mode == "disabled":
        assert not list(root.iterdir()) and app.recorder is None
    else:
        status = json.loads((path / "status.json").read_text())
        assert status["status"] == ("completed" if mode == "close" else "partial")
        if app.recorder is not None:
            assert app.recorder.closed


def test_bad_cli_record_path_rejected_before_loading(root, monkeypatch):
    from app import play

    def unexpected(*a):
        pytest.fail("Must reject recording path before any policy load")

    monkeypatch.setattr("app.rl_controller.load_human_rl", unexpected)
    with pytest.raises(SystemExit) as error:
        play.main(cli(root))
    assert error.value.code == 2
    with pytest.raises(SystemExit):
        play.main(["--record-dir", str(root / "heuristic")])
    assert not list(root.iterdir())


def test_periodic_flush_is_bounded_and_stream_keeps_all_complete_records(root, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(recording, "perf_counter", lambda: now[0])
    app = driver()
    rec = attach(root / "buffer", app)
    app.step(SurvivorCommand())
    assert rec.last_flushed_sequence == 0  # Initial boundary only; ticks are buffered.
    now[0] = 0.51
    app.step(SurvivorCommand())
    assert rec.last_flushed_sequence == rec.sequence - 1
    assert len(read_tick_commands(rec.directory, 0)) == 2
    rec.close(completed=True, reason="synthetic_end")


def test_constructor_flush_failure_closes_handle_and_marks_partial(root, monkeypatch):
    opened = []
    original = GameplayRecorder.begin_episode

    def fail_initial_flush(self, *args, **kwargs):
        opened.append(self.stream)
        self.stream = BrokenStream(self.stream, "flush")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GameplayRecorder, "begin_episode", fail_initial_flush)
    with pytest.raises(RecordingError, match="disk fixture"):
        attach(root / "init_flush", driver())
    assert opened[0].closed
    assert json.loads((root / "init_flush/status.json").read_text())["status"] == "partial"


def test_valid_strict_loader_exposes_verified_identity_without_second_load(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import rl.pipeline as pipeline
    from app.rl_controller import load_human_rl

    checkpoint = tmp_path / "fake.zip"
    checkpoint.write_bytes(b"fake model fixture; never deserialized")
    metadata = {
        "role": "best_validation",
        "actual_steps": 20224,
        "optimizer_update_epochs": 790,
        "source": {"source_sha256": "fake"},
        "contract_version": "fake-schema",
        "checkpoint_sha256": pipeline.file_hash(checkpoint),
    }
    selection = {
        "best_checkpoint_sha256": pipeline.file_hash(checkpoint),
        "best_steps": 20224,
        "best_optimizer_update_epochs": 790,
        "fresh_load_reproduced_actions_and_episodes": True,
    }
    pipeline.write_json(tmp_path / "selection.json", selection)
    monkeypatch.setattr(pipeline, "read_run", lambda p: ({"torch_threads": 1}, {}, {}))
    loads = []
    policy = Policy()
    policy.policy = SimpleNamespace(set_training_mode=lambda value: None)

    def fake_load(*args):
        loads.append(args)
        return policy, metadata

    monkeypatch.setattr(pipeline, "load_checkpoint", fake_load)
    app = load_human_rl(tmp_path, checkpoint, 991, 7799101)
    assert len(loads) == 1 and not app.model.calls
    assert app.checkpoint_identity["verified_metadata"] == metadata
    assert app.checkpoint_identity["selection"] == selection
    assert app.snapshot().seed == 7799101
