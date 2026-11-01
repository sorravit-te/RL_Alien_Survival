"""App-only contract parity using synthetic commands/worlds; no training/population replay."""

import random
from dataclasses import replace
from math import pi

import numpy as np
import pytest
import torch

from app.loop import FixedStepLoop
from app.rl_controller import HumanRLController, load_human_rl, rng_state
from game.core import GameCore
from game.map import MapObject, WorldMap
from game.state import CoreConfig, SurvivorCommand
from rl.contract import WAIT
from rl.env import AlienEnv
from rl.observations import build_observation


class Policy:
    def __init__(self, action=WAIT, sampled=False):
        self.action, self.sampled, self.calls = action, sampled, []

    def predict(self, obs, *, action_masks, deterministic):
        assert deterministic is False
        assert obs.shape == (352,) and obs.dtype == np.float32
        assert action_masks.shape == (32,) and action_masks.dtype == np.bool_
        self.calls.append((obs.copy(), action_masks.copy()))
        if self.sampled:
            # Exercise all three private streams; consumer noise must not alter choices.
            random.random()
            np.random.random()
            valid = np.flatnonzero(action_masks)
            action = int(valid[torch.randint(len(valid), ()).item()])
        else:
            action = self.action
        return np.int64(action), None


class ScriptedHuman:
    label = "synthetic_human_input"

    def __init__(self, commands):
        self.commands = commands
        self.cursor = 0

    def command(self, view):
        command = self.commands[self.cursor]
        self.cursor += 1
        return command


def arena():
    return WorldMap(
        ("." * 16,) * 16,
        ((2.5, 2.5),),
        alien_spawn=(10.5, 10.5, pi),
        patrol_waypoints=((10.5, 10.5), (12.5, 10.5), (10.5, 12.5)),
    )


@pytest.mark.parametrize("action", [WAIT, 0, 1])
@pytest.mark.parametrize("cap", [1, 17, 46])
def test_full_env_parity_predecision_and_posttick_with_partial_block(action, cap):
    world = arena()
    commands = [
        SurvivorCommand(forward=0.2, strafe=(-1) ** i * 0.3, yaw_delta=0.01) for i in range(cap)
    ]
    env = AlienEnv(world, max_episode_seconds=cap / 30, bot_kind="stationary_debug")
    env.reset(seed=1122, options={"scenario_seed": 7799001, "bot_seed": 7799002})
    env.bot = ScriptedHuman(commands)
    env_states = []
    original_step = env.core.step

    def record(command):
        snapshot = original_step(command)
        env_states.append(snapshot)
        return snapshot

    env.core.step = record
    policy = Policy(action)
    core = GameCore(world, alien_control="external")
    core.reset(7799001)
    app = HumanRLController(core, policy, 123)
    try:
        while env.elapsed_ticks < cap:
            assert app.allowed_view() == env._allowed_view()
            np.testing.assert_array_equal(app.observation(), build_observation(env._allowed_view()))
            np.testing.assert_array_equal(app.action_masks(), env.action_masks())
            assert app.visit_ticks == env.visit_ticks and app.last_action == env.last_action
            before = env.elapsed_ticks
            _, _, terminal, truncated, info = env.step(action)
            for i in range(before, env.elapsed_ticks):
                assert app.step(commands[i]) == env_states[i]
            assert app.snapshot() == env.core.snapshot()
            assert app.snapshot().alien_target == env.core.snapshot().alien_target
            assert app.visit_ticks == env.visit_ticks and app.last_action == env.last_action
            if terminal or truncated:
                break
        assert app.decisions == (cap + 14) // 15
        assert app.phase == cap % 15
        assert app.visit_ticks[0] is not None if action in (WAIT, 0) else True
    finally:
        env.close()


def test_terminal_early_block_and_reset_clear_everything():
    world = WorldMap(
        ("." * 12,) * 8,
        ((2.5, 2.5),),
        (MapObject("card", "keycard", 2.5, 2.5), MapObject("exit", "exit", 2.5, 2.5)),
        alien_spawn=(9.5, 5.5, pi),
        patrol_waypoints=((9.5, 5.5),),
    )
    env = AlienEnv(world, bot_kind="stationary_debug")
    env.reset(seed=1, options={"scenario_seed": 7799003, "bot_seed": 7799004})
    commands = [SurvivorCommand(interact=True)] * 15
    env.bot = ScriptedHuman(commands)
    app = HumanRLController(GameCore(world, alien_control="external"), Policy(), 123)
    app.reset(7799003)
    _, _, terminal, _, info = env.step(WAIT)
    for i in range(info["step_ticks"]):
        app.step(commands[i])
    assert terminal and app.snapshot() == env.core.snapshot() and app.snapshot().tick == 2
    before = app.snapshot()
    app.step(SurvivorCommand())
    assert app.snapshot() == replace(before, events=()) and app.decisions == 1
    app.reset(7799003)
    assert app.phase == app.decisions == 0 and app.last_action == WAIT
    assert app.visit_ticks == [None] and app.snapshot().navigation_path == ()
    assert app.snapshot().perception.last_seen is None and not app.snapshot().has_keycard
    env.close()


def test_visible_target_is_preblock_snapshot_not_tracking_live_human():
    from game.perception import PerceptionSnapshot

    core = GameCore(arena(), alien_control="external")
    core._perception = PerceptionSnapshot(visible=True, last_seen=(2.5, 2.5), last_seen_age_ticks=0)
    app = HumanRLController(core, Policy(20), 321)
    # Set a controlled confirmed observation after reset, as if from a prior Core tick.
    core._perception = PerceptionSnapshot(visible=True, last_seen=(2.5, 2.5), last_seen_age_ticks=0)
    app.step(SurvivorCommand(strafe=1))
    assert app.snapshot().alien_target == (2.5, 2.5)
    for _ in range(14):
        app.step(SurvivorCommand(strafe=1))
    assert app.snapshot().survivor.y != 2.5 and app.snapshot().alien_target == (2.5, 2.5)
    assert app.decisions == 1


@pytest.mark.parametrize("occluded", [False, True])
def test_hidden_truth_invariance_after_actual_sensor_ticks(occluded):
    world = arena()
    if occluded:
        rows = list(world.rows)
        rows[10] = rows[10][:9] + "#" + rows[10][10:]
        world = replace(world, rows=tuple(rows))
    apps = []
    for index, point in enumerate(((8.5, 10.5), (8.5, 11.5))):
        objects = (
            MapObject("card", "keycard", 2.5 + index, 2.5),
            MapObject("exit", "exit", 4.5 + index, 4.5),
        )
        core = GameCore(replace(world, objects=objects), alien_control="external")
        app = HumanRLController(core, Policy(), 33)
        core._survivor = replace(core.snapshot().survivor, x=point[0], y=point[1])
        core._alien = replace(core.snapshot().alien, yaw=pi if occluded else 0)
        core._has_keycard = bool(index)
        apps.append(app)
    for _ in range(16):
        for app in apps:
            app.step(SurvivorCommand())
            assert not app.snapshot().perception.visible
            assert app.snapshot().perception.last_seen is None
        assert apps[0].allowed_view() == apps[1].allowed_view()
        np.testing.assert_array_equal(apps[0].observation(), apps[1].observation())
        np.testing.assert_array_equal(apps[0].action_masks(), apps[1].action_masks())


def test_rng_is_private_seed_once_and_reset_reproduces(monkeypatch):
    policy = Policy(sampled=True)
    app = HumanRLController(GameCore(arena(), alien_control="external"), policy, 991)
    seed_calls = []
    original = torch.manual_seed

    def record(seed):
        seed_calls.append(seed)
        return original(seed)

    monkeypatch.setattr(torch, "manual_seed", record)

    def replay(noise):
        app.reset(7799010)
        actions = []
        for i in range(45):
            if noise:
                random.random()
                np.random.random()
                torch.rand(3)
            before = rng_state()
            app.step(SurvivorCommand())
            after = rng_state()
            assert before[0] == after[0] and np.array_equal(before[1][1], after[1][1])
            assert torch.equal(before[2], after[2])
            actions.append(app.last_action)
        return actions, app.snapshot()

    a = replay(False)
    b = replay(True)
    assert a == b and seed_calls == [991, 991]


@pytest.mark.parametrize("fps", [30, 60, 144])
def test_fixed_step_two_hz_pause_reset_fps_independence(fps):
    app = HumanRLController(GameCore(arena(), alien_control="external"), Policy(), 123)
    loop = FixedStepLoop(app)
    for _ in range(fps * 2):
        loop.advance(1 / fps)
    assert loop.current.tick == 60 and app.decisions == 4 and app.phase == 0
    before = loop.current
    calls = len(app.model.calls)
    loop.set_paused(True)
    loop.advance(999)
    assert loop.current == before and len(app.model.calls) == calls and app.phase == 0
    loop.reset()
    assert loop.current.tick == 0 and app.phase == app.decisions == 0
    assert loop.inputs.consume(loop.dt) == SurvivorCommand()


@pytest.mark.parametrize("action", [31, 32, -1])
def test_invalid_policy_action_stops_without_core_tick_or_heuristic(action):
    app = HumanRLController(GameCore(arena(), alien_control="external"), Policy(action), 123)
    with pytest.raises(ValueError, match="invalid/masked"):
        app.step(SurvivorCommand())
    assert app.snapshot().tick == 0 and app.snapshot().alien_target is None


def test_load_failure_has_no_core_fallback(monkeypatch, tmp_path):
    import rl.pipeline

    def fail(*args, **kwargs):
        raise ValueError("source mismatch fixture")

    monkeypatch.setattr(rl.pipeline, "read_run", fail)
    with pytest.raises(ValueError, match="source mismatch"):
        load_human_rl(tmp_path, tmp_path / "missing.zip", 123)


def test_cli_missing_args_and_load_failure_stop_before_window(monkeypatch, capsys):
    from app import play

    with pytest.raises(SystemExit) as error:
        play.main(["--controller", "rl"])
    assert error.value.code == 2
    monkeypatch.setattr(
        "app.rl_controller.load_human_rl",
        lambda *args: (_ for _ in ()).throw(ValueError("metadata mismatch fixture")),
    )
    with pytest.raises(SystemExit) as error:
        play.main(
            [
                "--controller",
                "rl",
                "--run-dir",
                "fixture",
                "--checkpoint",
                "fixture.zip",
                "--sampling-seed",
                "1",
            ]
        )
    assert error.value.code == 2 and "metadata mismatch" in capsys.readouterr().err


def test_capture_early_and_no_extra_decision_after_terminal():
    world = WorldMap(
        ("." * 12,) * 8, ((2.5, 2.5),), alien_spawn=(6.5, 2.5, pi), patrol_waypoints=((2.5, 2.5),)
    )
    config = CoreConfig(alien_search_speed=120, alien_chase_speed=120)
    env = AlienEnv(world, config, bot_kind="stationary_debug")
    env.reset(seed=1, options={"scenario_seed": 7799011, "bot_seed": 7799012})
    app = HumanRLController(GameCore(world, config, alien_control="external"), Policy(0), 55)
    app.reset(7799011)
    _, _, terminal, _, info = env.step(0)
    for _ in range(info["step_ticks"]):
        app.step(SurvivorCommand())
    assert (
        terminal and app.snapshot() == env.core.snapshot() and app.snapshot().outcome == "captured"
    )
    assert app.snapshot().tick == 1 and app.decisions == 1 and app.phase == 1
    app.step(SurvivorCommand())
    assert app.decisions == 1 and app.phase == 1
    env.close()


def test_pygame_focus_pause_preserves_partial_decision_clock_and_retry(monkeypatch, tmp_path):
    import pygame

    from app.play import PlaySession, hud_labels

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.display.init()
    pygame.font.init()
    pygame.display.set_mode((320, 180))
    try:
        app = HumanRLController(
            GameCore(arena(), alien_control="external"), Policy(sampled=True), 123
        )
        loop = FixedStepLoop(app)
        session = PlaySession(loop)
        session.focused = True
        session.events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
        for _ in range(7):
            loop.advance(loop.dt)
        before = app.snapshot()
        state = app.sampler.state
        calls = len(app.model.calls)
        loop.inputs.key("e", True)
        loop.inputs.mouse(100, 100)
        session.events([pygame.event.Event(pygame.WINDOWFOCUSLOST)])
        loop.advance(99)
        assert app.snapshot() == before and app.phase == 7 and len(app.model.calls) == calls
        assert torch.equal(state[2], app.sampler.state[2]) and loop.inputs.pending_interactions == 0
        session.events([pygame.event.Event(pygame.WINDOWFOCUSGAINED)])
        assert loop.paused
        session.events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
        for _ in range(8):
            loop.advance(loop.dt)
        assert app.phase == 0 and app.decisions == 1
        loop.advance(loop.dt)
        assert app.decisions == 2
        session.events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)])
        assert app.snapshot().tick == app.phase == app.decisions == 0
        assert loop.display_pitch() == 0 and loop.inputs.pending_interactions == 0
        assert not any("RL sampled" in label for label in hud_labels(loop))
        session.events([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
        assert any("RL sampled" in label for label in hud_labels(loop))
    finally:
        pygame.quit()


def test_rl_window_close_releases_pygame_with_placeholder_texture(monkeypatch, tmp_path):
    import pygame

    from app import play
    from renderers.raycasting import RaycastingRenderer

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.display.init()
    texture = tmp_path / "wall.png"
    pygame.image.save(pygame.Surface((16, 16)), texture)
    app = HumanRLController(GameCore(arena(), alien_control="external"), Policy(), 123)
    monkeypatch.setattr("app.rl_controller.load_human_rl", lambda *a: app)
    monkeypatch.setattr(play, "RaycastingRenderer", lambda: RaycastingRenderer(texture))
    monkeypatch.setattr(pygame.event, "get", lambda: [pygame.event.Event(pygame.QUIT)])
    play.main(
        [
            "--controller",
            "rl",
            "--run-dir",
            "fixture",
            "--checkpoint",
            "fixture.zip",
            "--sampling-seed",
            "123",
            "--windowed",
        ]
    )
    assert not pygame.display.get_init() and not pygame.font.get_init()


@pytest.mark.parametrize("field", ["role", "hash", "steps", "epochs", "replay"])
def test_selected_best_binding_failure_prevents_core_creation(monkeypatch, tmp_path, field):
    import rl.pipeline as pipeline

    path = tmp_path / "model.zip"
    path.write_bytes(b"synthetic checkpoint fixture")
    selection = {
        "best_checkpoint_sha256": pipeline.file_hash(path),
        "best_steps": 20224,
        "best_optimizer_update_epochs": 790,
        "fresh_load_reproduced_actions_and_episodes": True,
    }
    metadata = {"role": "best_validation", "actual_steps": 20224, "optimizer_update_epochs": 790}
    if field == "role":
        metadata["role"] = "untrained"
    elif field == "hash":
        selection["best_checkpoint_sha256"] = "wrong"
    elif field == "steps":
        metadata["actual_steps"] = 0
    elif field == "epochs":
        metadata["optimizer_update_epochs"] = 0
    else:
        selection["fresh_load_reproduced_actions_and_episodes"] = False
    pipeline.write_json(tmp_path / "selection.json", selection)
    monkeypatch.setattr(pipeline, "read_run", lambda p: ({"torch_threads": 1}, {}, {}))
    monkeypatch.setattr(pipeline, "load_checkpoint", lambda *a: (Policy(), metadata))

    def no_core(*args, **kwargs):
        pytest.fail("Invalid metadata reached Core construction")

    monkeypatch.setattr("app.rl_controller.GameCore", no_core)
    with pytest.raises(ValueError, match="selected, replay-verified"):
        load_human_rl(tmp_path, path, 123)
