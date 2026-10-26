"""Pure fixture, knowledge-boundary and accounting tests: zero Core steps."""

from dataclasses import asdict, replace
from random import Random

import numpy as np
import pytest

from ai.navigation import find_path
from game.core import GameCore
from game.investigation import InvestigationSnapshot
from game.map import MapObject, small_test_map
from game.motion import segment_has_clearance
from game.state import AlienState, CoreConfig, SurvivorState
from rl.candidates import AllowedAlienView, build_candidates, candidate_masks
from rl.contract import ACTION_COUNT, OBSERVATION_SIZE, VERSION
from rl.dev_fixtures import FixtureMissionBot, read_fixtures
from rl.evasive_survivor import EvasionConfig, SurvivorThreatView, observe_threat
from rl.scripted_controllers import HeuristicSelector
from rl.survivor_bots import SurvivorView
from tools.check_lost_sight_fixtures import BudgetLedger, summarize_execution


@pytest.fixture(autouse=True)
def prohibit_simulation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail(
            "Fixture unit tests must not advance Core; all runs use the shared budget ledger"
        )

    monkeypatch.setattr(GameCore, "step", forbidden)


@pytest.mark.parametrize("case", read_fixtures()[1], ids=lambda c: c.case_id)
def test_frozen_case_geometry_routes_seeds(case):
    base, world, config = small_test_map(), case.world(), CoreConfig()
    assert (world.rows, world.objects, world.patrol_waypoints, world.alien_spawn) == (
        base.rows,
        base.objects,
        base.patrol_waypoints,
        base.alien_spawn,
    )
    assert world.spawn_points == (base.spawn_points[case.survivor_spawn_index],)
    assert Random(case.scenario_seed).choice(world.spawn_points) == Random(
        case.scenario_seed
    ).choice(world.spawn_points)
    for point in (*world.spawn_points, *case.exploration_anchors):
        for obj in world.objects:
            path = find_path(world, point, (obj.x, obj.y), config.radius)
            assert path
            assert all(
                segment_has_clearance(world, a, b, config.radius)
                for a, b in zip((point, *path), path)
            )
    bots = [FixtureMissionBot(world.rows, config, case.exploration_anchors) for _ in range(2)]
    for b in bots:
        b.reset(np.random.default_rng(case.bot_seed))
        assert not b.explorer.known
        assert not b.explorer.geometry.objects
        assert b.explorer.order[: len(case.exploration_anchors)] == case.exploration_anchors
    assert bots[0].explorer.order == bots[1].explorer.order


def bot():
    b = FixtureMissionBot(small_test_map().rows, CoreConfig(), ((5.5, 2.5), (9.5, 4.5)))
    b.reset(np.random.default_rng(17))
    return b


def test_anchor_is_not_object_discovery_or_interaction():
    b = bot()
    view = SurvivorThreatView(SurvivorView(5.5, 2.5, 0, False, ()))
    assert not b.command(view).interact
    assert not b.explorer.known
    # Even at an exit-shaped exploration anchor, inventory alone reveals no goal.
    b = bot()
    assert not b.command(SurvivorThreatView(SurvivorView(9.5, 4.5, 0, True, ()))).interact
    assert not b.explorer.known


def test_discovered_mission_overrides_exploration_and_uses_press_command():
    b = bot()
    card = MapObject("card_custom", "keycard", 3.25, 2.5)
    command = b.command(SurvivorThreatView(SurvivorView(2.5, 2.5, 0, False, (card,))))
    assert command.interact and command.forward == command.strafe == 0
    assert b.explorer.known == {card.object_id: card}
    # Core inventory is the only signal that collection completed; visible exit is then eligible.
    exit = MapObject("exit_custom", "exit", 3.5, 2.5)
    command = b.command(SurvivorThreatView(SurvivorView(2.5, 2.5, 0, True, (exit,))))
    assert command.interact


def test_hidden_world_changes_preserve_input_and_commands():
    base = small_test_map()
    survivor = SurvivorState(2.5, 4.5)
    worlds = [
        replace(base, objects=(MapObject("keycard", "keycard", x, 2.5), base.objects[1]))
        for x in (5.5, 6.5)
    ]
    aliens = [AlienState(x, 4.5) for x in (5.5, 6.5)]
    bots = [bot(), bot()]
    views = [observe_threat(w, survivor, False, a, EvasionConfig()) for w, a in zip(worlds, aliens)]
    assert views[0] == views[1]
    assert views[0].visible_alien is None and not views[0].mission.visible_objects
    for _ in range(30):
        assert bots[0].command(views[0]) == bots[1].command(views[1])
    assert not bots[0].explorer.known
    assert not bots[1].explorer.known


def test_bot_rejects_unfiltered_snapshot_shaped_input():
    with pytest.raises(TypeError, match="filtered SurvivorThreatView"):
        bot().command(SurvivorState(2.5, 2.5))


@pytest.mark.parametrize("anchors", [(), ((4.5, 3.5),), ((5.5, 2.5), (5.5, 2.5))])
def test_invalid_anchors_fail_without_running_a_scenario(anchors):
    with pytest.raises(ValueError):
        FixtureMissionBot(small_test_map().rows, CoreConfig(), anchors)


def view(**changes):
    result = AllowedAlienView(
        (9.5, 7.5),
        0,
        False,
        30,
        False,
        None,
        None,
        small_test_map().patrol_waypoints,
        (None,) * 6,
        False,
    )
    return replace(result, **changes)


@pytest.mark.parametrize(
    "v,expected",
    [
        (view(), 1),
        (view(visible=True, last_seen=(5.5, 2.5), clue_revision=1), 20),
        (view(last_seen=(5.5, 2.5), clue_revision=1), 21),
        (
            view(
                last_seen=(5.5, 2.5),
                clue_revision=1,
                investigation=InvestigationSnapshot(investigated_revision=1),
            ),
            1,
        ),
    ],
)
def test_shared_heuristic_selector_uses_allowed_catalogue(v, expected):
    selector = HeuristicSelector(CoreConfig().target_arrival_distance)
    action = selector.choose(v)
    assert action == expected
    assert candidate_masks(build_candidates(v))[action]
    assert VERSION == "alien-vector-v2" and OBSERVATION_SIZE == 352 and ACTION_COUNT == 32


def test_ledger_caps_charges_and_refuses_restart(tmp_path):
    # Accounting unit operations, not Core execution; isolated temporary ledger.
    p = tmp_path / "ledger.jsonl"
    ledger = BudgetLedger(p, tick_cap=1, execution_cap=1)
    ledger.begin("unit_only", "wait")
    ledger.tick()
    with pytest.raises(RuntimeError):
        ledger.tick()
    ledger.end("unit_only")
    with pytest.raises(RuntimeError):
        ledger.begin("unit_only", "wait")
    ledger.close()
    with pytest.raises(FileExistsError):
        BudgetLedger(p)


def test_loss_timing_keeps_reacquisition_tick_choice_and_no_trigger_denominator():
    config, world = CoreConfig(), small_test_map()
    case = read_fixtures()[1][0]
    ticks = []
    for t, visible in enumerate((True, False, False, True)):
        ticks.append(
            {
                "tick_before": t,
                "tick_after": t + 1,
                "action": 30,
                "alien_after": {"x": 9.5, "y": 7.5},
                "perception": {
                    "visible": visible,
                    "last_seen": [5.5, 2.5],
                    "revision": 2 if t == 3 else 1,
                },
                "known_objects": [],
                "events": [],
                "outcome": None,
                "investigation_before": {"phase": "idle"},
                "investigation_after": {"phase": "idle"},
            }
        )
    decision_view = asdict(view(last_seen=(5.5, 2.5), clue_revision=1))
    decisions = [{"tick": 3, "action": 30, "view": decision_view, "masks": [True] * 32}]
    result = summarize_execution(case, "wait", ticks, decisions, world, config)
    span = result["loss_windows"][0]
    assert span["loss_sensor_tick"] == 1 and span["available_tick"] == 2
    assert span["first_informed_decision_tick"] == 3
    assert span["available_to_decision_ticks"] == 1
    assert span["ending"] == "reacquired" and span["loss_duration_ticks"] == 2
    assert result["reacquisitions"] == 1 and result["outcome"] == "unresolved"
    assert result["no_card"] and result["wait_ticks"] == 4
    quiet = [{**row, "perception": {**row["perception"], "visible": False}} for row in ticks]
    result = summarize_execution(case, "wait", quiet, decisions, world, config)
    assert result["no_sight"] and result["no_loss"]
    assert result["conditional_metrics_status"] == "not_applicable_no_loss"
