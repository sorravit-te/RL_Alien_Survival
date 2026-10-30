"""Each reset draws spawns, one card and one working exit; the Alien learns none of them."""

from collections import Counter

import numpy as np
import pytest

from game.core import GameCore
from game.map import RANDOM_SCENARIO, MapObject, WorldMap, scenario_map
from game.state import CoreConfig, SurvivorCommand, SurvivorState
from rl.bots.explorer import ExplorerBot
from rl.bots.views import observe_survivor
from rl.env import AlienEnv


def core_for(seed, *, alien=True):
    core = GameCore(scenario_map(RANDOM_SCENARIO), CoreConfig(alien_enabled=alien))
    core.reset(seed)
    return core


def kinds(snapshot):
    return Counter(obj.kind for obj in snapshot.world_map.objects)


def test_each_reset_keeps_one_card_and_every_exit():
    snapshot = core_for(5).snapshot()
    assert kinds(snapshot) == {"keycard": 1, "exit": 3}


def test_seed_replays_the_same_scenario_and_all_options_occur():
    seen = set()
    for seed in range(1000):
        a, b = core_for(seed), core_for(seed)
        sa, sb = a.snapshot(), b.snapshot()
        assert sa == sb and a._active_exit == b._active_exit
        card = next(o.object_id for o in sa.world_map.objects if o.kind == "keycard")
        seen.add((sa.survivor.x, sa.survivor.y, sa.alien.x, card, a._active_exit))
    assert len(seen) == 3 * 2 * 4 * 3


def test_fixed_scenes_keep_their_object_lists():
    world = scenario_map("occluded_loop_dev_v1")
    assert GameCore(world).snapshot().world_map is world


@pytest.mark.parametrize("active", [True, False])
def test_exit_needs_a_card_and_only_the_working_one_escapes(active):
    world = WorldMap(
        ("." * 9,) * 9,
        ((4.5, 4.5),),
        (MapObject("card", "keycard", 5.5, 4.5), MapObject("z", "exit", 5.5, 4.5)),
    )
    core = GameCore(world, CoreConfig(alien_enabled=False))
    core.reset(1)
    core._active_exit = "z" if active else "other"
    core.step(SurvivorCommand(interact=True))
    assert core.snapshot().has_keycard and core.snapshot().outcome is None
    snapshot = core.step(SurvivorCommand(interact=True))
    if active:
        assert snapshot.outcome == "escaped" and snapshot.ruled_out_exits == ()
    else:
        assert snapshot.outcome is None and snapshot.ruled_out_exits == ("z",)
        assert [e.kind for e in snapshot.events] == ["exit_offline"]
        again = core.step(SurvivorCommand(interact=True))
        assert again.ruled_out_exits == ("z",)
        assert core.reset(1).ruled_out_exits == ()


def test_exit_without_a_card_reveals_nothing():
    world = WorldMap(("." * 9,) * 9, ((4.5, 4.5),), (MapObject("a", "exit", 5.5, 4.5),))
    core = GameCore(world, CoreConfig(alien_enabled=False))
    core.reset(1)
    core._active_exit = "other"
    assert core.step(SurvivorCommand(interact=True)).ruled_out_exits == ()


def test_alien_input_does_not_depend_on_the_working_exit_or_card():
    def observation(seed, active, card_x):
        env = AlienEnv(scenario=RANDOM_SCENARIO, bot_kind="stationary_debug")
        obs, _ = env.reset(seed=1, options={"scenario_seed": seed, "bot_seed": 9})
        env.core._active_exit = active
        env.core._survivor = SurvivorState(env.core.snapshot().survivor.x, 13.5)
        return obs, env.action_masks(), card_x

    base = observation(3, "exit_northwest", 0)
    other = observation(3, "exit_southeast", 1)
    np.testing.assert_array_equal(base[0], other[0])
    np.testing.assert_array_equal(base[1], other[1])
    names = set(env_fields())
    assert not {"objects", "active_exit", "keycard"} & names


def env_fields():
    from dataclasses import fields

    from rl.candidates import AllowedAlienView

    return [field.name for field in fields(AllowedAlienView)]


@pytest.mark.parametrize("seed", range(12))
def test_explorer_finds_the_card_then_tries_exits_until_one_works(seed):
    core = core_for(seed, alien=False)
    bot = ExplorerBot(core.world_map.rows, core.config)
    bot.reset(np.random.default_rng(seed))
    snapshot = core.snapshot()
    for _ in range(30 * 600):
        view = observe_survivor(
            snapshot.world_map, snapshot.survivor, snapshot.has_keycard, snapshot.ruled_out_exits
        )
        snapshot = core.step(bot.command(view))
        if snapshot.outcome:
            break
    assert snapshot.outcome == "escaped" and core._active_exit not in snapshot.ruled_out_exits
