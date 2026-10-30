"""Gymnasium Alien-target environment. No rendering/assets/model/training imports.

Invalid masked actions become WAIT with no penalty; out-of-space values raise.
A step is15 Core ticks unless terminal/cap intervenes; terminal beats the cap.
Only rl.alien_view extracts policy-permitted fields. Builder/candidates never
receive full world state. Info is diagnostic, not part of policy observation.
"""

from math import ceil, isfinite

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ai.navigation import find_path
from game.core import GameCore
from game.doors import sight_world
from game.map import DEFAULT_SCENARIO, scenario_map
from game.state import CoreConfig
from rl.alien_view import allowed_view, record_waypoint_visits, target_source
from rl.bots.evasive import EvasiveExplorerBot
from rl.bots.explorer import ExplorerBot, StationaryDebugBot
from rl.bots.views import observe_survivor, observe_threat
from rl.bots.wall_loop import EvasiveMixBot, WallLoopEvasiveBot
from rl.candidates import build_candidates, candidate_masks, committed_waypoint
from rl.contract import (
    ACTION_COUNT,
    DECISION_TICKS,
    OBSERVATION_SIZE,
    VERSION,
    WAIT,
    WORLD_SCALE,
)
from rl.observations import build_observation
from rl.rewards import RewardTracker
from rl.search_metrics import SearchMetrics

BOT_KINDS = ("explorer", "evasive_explorer", "stationary_debug", "wall_loop_evasive", "evasive_mix")
# Bots that receive the LoS-filtered threat view and accept EvasionConfig.
THREAT_BOT_KINDS = ("evasive_explorer", "wall_loop_evasive", "evasive_mix")
THREAT_BOTS = {
    "evasive_explorer": EvasiveExplorerBot,
    "wall_loop_evasive": WallLoopEvasiveBot,
    "evasive_mix": EvasiveMixBot,
}


class AlienEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        world_map=None,
        config=None,
        *,
        max_episode_seconds=600,
        bot_kind="explorer",
        evasion_settings=None,
        scenario=None,
    ):
        if world_map is not None and scenario is not None:
            raise ValueError("Use a named scenario or a custom world, never both")
        self.scenario = DEFAULT_SCENARIO if world_map is None and scenario is None else scenario
        self.metrics = None
        self.world = (
            world_map
            if world_map is not None
            else (scenario_map() if scenario is None else scenario_map(scenario))
        )
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
        if bot_kind not in BOT_KINDS:
            raise ValueError("Unknown bot kind")
        self.bot_kind = bot_kind
        if evasion_settings is not None and bot_kind not in THREAT_BOT_KINDS:
            raise ValueError("Evasion settings require evasive bot kinds")
        self.bot = (
            THREAT_BOTS[bot_kind](
                self.world.rows, self.config, evasion_settings, doors=self.world.doors
            )
            if bot_kind in THREAT_BOT_KINDS
            else ExplorerBot(self.world.rows, self.config, doors=self.world.doors)
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
        return allowed_view(self.core.snapshot(), self.last_action, self.visit_ticks)

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
        if self.bot_kind == "evasive_mix":
            self.bot.reset(np.random.default_rng(self.bot_seed), bot_seed=self.bot_seed)
        else:
            self.bot.reset(np.random.default_rng(self.bot_seed))
        self.reward_tracker = RewardTracker()
        self.last_action = WAIT
        self.visit_ticks = [None] * len(self.world.patrol_waypoints)
        self.metrics = SearchMetrics()
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
        view = self._allowed_view()
        candidates = build_candidates(view)
        invalid = not candidates[action].valid
        applied = WAIT if invalid else action
        self.last_action = applied
        chosen = candidates[applied]
        before_selection = self.core.snapshot()
        self.core.set_alien_target(
            chosen.target,
            source=target_source(applied),
            evidence_revision=chosen.evidence_revision,
        )
        self.metrics.decision(
            before_selection,
            self.core.snapshot(),
            candidates[21].valid,
            applied,
            requested_action=action,
            committed=committed_waypoint(view) is not None,
        )
        components = {"capture": 0.0, "escape": 0.0, "first_sight": 0.0}
        ticks = 0
        snapshot = self.core.snapshot()
        while ticks < DECISION_TICKS and self.elapsed_ticks < self.cap_ticks:
            # Both sides decide from the same pre-tick state. Only the threat sensor reads Alien truth.
            before = self.core.snapshot()
            view = (
                observe_threat(
                    sight_world(before),
                    before.survivor,
                    before.has_keycard,
                    before.alien,
                    self.bot.settings,
                    before.ruled_out_exits,
                )
                if self.bot_kind in THREAT_BOT_KINDS
                else observe_survivor(
                    sight_world(before),
                    before.survivor,
                    before.has_keycard,
                    before.ruled_out_exits,
                )
            )
            command = self.bot.command(view)
            snapshot = self.core.step(command)
            self.metrics.tick(before, snapshot)
            ticks += 1
            self.elapsed_ticks += 1
            for name, value in self.reward_tracker.collect(
                snapshot.perception.visible, snapshot.events
            ).items():
                components[name] += value
            record_waypoint_visits(snapshot, self.visit_ticks)
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
            "truncation_reason": "external_time_cap" if truncated else None,
            "action_masks": self.action_masks(),
        }
        if self._done:
            info["search_metrics"] = self.metrics.finish(snapshot)
        return (
            build_observation(self._allowed_view()),
            float(sum(components.values())),
            terminated,
            truncated,
            info,
        )
