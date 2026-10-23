"""Gymnasium Alien-target environment. No rendering/assets/model/training imports.

Invalid masked actions become WAIT with no penalty; out-of-space values raise.
A step is15 Core ticks unless terminal/cap intervenes; terminal beats the cap.
Only _allowed_view extracts policy-permitted fields. Builder/candidates never
receive full world state. Info is diagnostic, not part of policy observation.
"""

from math import ceil, hypot, isfinite

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ai.navigation import find_path
from game.core import GameCore
from game.map import small_test_map
from game.state import CoreConfig
from rl.candidates import AllowedAlienView, build_candidates, candidate_masks
from rl.contract import ACTION_COUNT, DECISION_TICKS, OBSERVATION_SIZE, VERSION, WAIT, WORLD_SCALE
from rl.observations import build_observation
from rl.rewards import RewardTracker
from rl.survivor_bots import ExplorerBot, StationaryDebugBot, observe_survivor


class AlienEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self, world_map=None, config=None, *, max_episode_seconds=600, bot_kind="explorer"
    ):
        self.world = world_map if world_map is not None else small_test_map()
        self.config = config if config is not None else CoreConfig()
        if not isfinite(max_episode_seconds) or max_episode_seconds <= 0:
            raise ValueError("Episode cap must be finite and positive")
        if max(self.world.width, self.world.height) > WORLD_SCALE:
            raise ValueError("Map exceeds fixed world-coordinate normalization contract")
        if len(self.world.patrol_waypoints) > 20:
            raise ValueError("Current contract permits at most20 waypoints")
        if not self.config.alien_enabled or self.world.alien_spawn is None:
            raise ValueError("RL environment requires an active Alien")
        self.core = GameCore(self.world, self.config, alien_control="external")
        start = self.world.alien_spawn[:2]
        for point in self.world.patrol_waypoints:
            if not find_path(self.world, start, point, self.config.alien_radius):
                raise ValueError("Static policy waypoint is unreachable")
        if bot_kind not in ("explorer", "stationary_debug"):
            raise ValueError("Unknown bot kind")
        self.bot = (
            ExplorerBot(self.world.rows, self.config)
            if bot_kind == "explorer"
            else StationaryDebugBot()
        )
        self.cap_ticks = ceil(max_episode_seconds * self.config.tick_hz - 1e-12)
        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self.render_mode = None
        self._done = True
        self.last_action = WAIT
        self.visit_ticks = [None] * len(self.world.patrol_waypoints)

    def _allowed_view(self):
        snapshot = self.core.snapshot()
        alien = snapshot.alien
        memory = snapshot.perception
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
            self.world.patrol_waypoints,
            tuple(
                (snapshot.tick - tick) / self.config.tick_hz if tick is not None else None
                for tick in self.visit_ticks
            ),
            bool(snapshot.navigation_path),
        )

    def action_masks(self):
        return candidate_masks(build_candidates(self._allowed_view()))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if set(options) - {"scenario_seed", "bot_seed"}:
            raise ValueError("Only scenario_seed and bot_seed reset options are supported")
        root = np.random.SeedSequence(int(self.np_random.integers(0, 2**63)))
        scenario, bot = root.spawn(2)
        self.scenario_seed = int(options.get("scenario_seed", scenario.generate_state(1)[0]))
        self.bot_seed = int(options.get("bot_seed", bot.generate_state(1)[0]))
        if min(self.scenario_seed, self.bot_seed) < 0:
            raise ValueError("Seeds must be nonnegative")
        self.core.reset(self.scenario_seed)
        self.bot.reset(np.random.default_rng(self.bot_seed))
        self.reward_tracker = RewardTracker()
        self.last_action = WAIT
        self.visit_ticks = [None] * len(self.world.patrol_waypoints)
        self.elapsed_ticks = 0
        self._done = False
        return build_observation(self._allowed_view()), {
            "contract_version": VERSION,
            "scenario_seed": self.scenario_seed,
            "bot_seed": self.bot_seed,
            "bot": self.bot.label,
            "outcome": None,
            "elapsed_ticks": 0,
            "action_masks": self.action_masks(),
        }

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset before stepping a completed environment")
        if not self.action_space.contains(action):
            raise ValueError("Action is outside Discrete(32)")
        action = int(action)
        candidates = build_candidates(self._allowed_view())
        invalid = not candidates[action].valid
        applied = WAIT if invalid else action
        self.last_action = applied
        chosen = candidates[applied]
        self.core.set_alien_target(
            chosen.target,
            source=(
                "wait"
                if applied == WAIT
                else "waypoint"
                if applied < 20
                else "visible"
                if applied == 20
                else "last_seen"
            ),
        )
        components = {"capture": 0.0, "escape": 0.0, "first_sight": 0.0}
        ticks = 0
        snapshot = self.core.snapshot()
        while ticks < DECISION_TICKS and self.elapsed_ticks < self.cap_ticks:
            # Both sides decide from the same pre-tick state. Bot sees no Alien action/state.
            before = self.core.snapshot()
            view = observe_survivor(self.world, before.survivor, before.has_keycard)
            command = self.bot.command(view)
            snapshot = self.core.step(command)
            ticks += 1
            self.elapsed_ticks += 1
            for name, value in self.reward_tracker.collect(
                snapshot.perception.visible, snapshot.events
            ).items():
                components[name] += value
            for index, point in enumerate(self.world.patrol_waypoints):
                if (
                    hypot(snapshot.alien.x - point[0], snapshot.alien.y - point[1])
                    <= self.config.target_arrival_distance
                ):
                    self.visit_ticks[index] = snapshot.tick
            if snapshot.outcome is not None:
                break
        terminated = snapshot.outcome is not None
        truncated = not terminated and self.elapsed_ticks >= self.cap_ticks
        self._done = terminated or truncated
        info = {
            "contract_version": VERSION,
            "bot": self.bot.label,
            "requested_action": action,
            "applied_action": applied,
            "invalid_action": invalid,
            "reward_components": components,
            "outcome": snapshot.outcome if terminated else "unresolved" if truncated else None,
            "elapsed_ticks": self.elapsed_ticks,
            "step_ticks": ticks,
            "action_masks": self.action_masks(),
        }
        return (
            build_observation(self._allowed_view()),
            float(sum(components.values())),
            terminated,
            truncated,
            info,
        )
