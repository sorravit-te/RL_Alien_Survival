"""Pure v2 layout/filter/stage checks. Both Core construction and step prohibited."""

import json
from dataclasses import replace
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
from rl.dev_fixtures import FixtureMissionBot
from rl.evasive_survivor import EvasionConfig, SurvivorThreatView, observe_threat
from rl.occluded_fixtures import DEFAULT_MANIFEST, read_occluded_fixtures
from rl.scripted_controllers import HeuristicSelector
from rl.survivor_bots import SurvivorView
from tools.check_lost_sight_fixtures import BudgetLedger
from tools.check_occluded_fixtures import summarize_inspection


@pytest.fixture(autouse=True)
def no_scenarios(monkeypatch):
    def prohibited(*args, **kwargs):
        pytest.fail("Pure tests may not create or step a Core scenario")

    monkeypatch.setattr(GameCore, "__init__", prohibited)
    monkeypatch.setattr(GameCore, "step", prohibited)


@pytest.mark.parametrize("case", read_occluded_fixtures()[1], ids=lambda c: c.case_id)
def test_frozen_routes_geometry_safe_spawns_and_seeded_order(case):
    world, config = case.world(), CoreConfig()
    assert max(world.width, world.height) <= 16
    assert Random(case.scenario_seed).choice(world.spawn_points) == world.spawn_points[0]
    goals = (
        *case.exploration_anchors,
        *world.patrol_waypoints,
        *((o.x, o.y) for o in world.objects),
    )
    for start in (world.spawn_points[0], world.alien_spawn[:2]):
        for goal in goals:
            path = find_path(world, start, goal, config.radius)
            assert path and all(
                segment_has_clearance(world, a, b, config.radius)
                for a, b in zip((start, *path), path)
            )
    card = next(o for o in world.objects if o.kind == "keycard")
    exit = next(o for o in world.objects if o.kind == "exit")
    assert find_path(world, (card.x, card.y), (exit.x, exit.y), config.radius)
    bots = [FixtureMissionBot(world.rows, config, case.exploration_anchors) for _ in range(2)]
    for b in bots:
        b.reset(np.random.default_rng(case.bot_seed))
        assert not b.explorer.known and not b.explorer.geometry.objects
        assert b.explorer.geometry.rows == world.rows
    assert bots[0].explorer.order == bots[1].explorer.order
    if case.layout_id == "small_room_control":
        base = small_test_map()
        assert (world.rows, world.objects, world.patrol_waypoints) == (
            base.rows,
            base.objects,
            base.patrol_waypoints,
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("unsafe_spawn", "Unsafe"),
        ("duplicate_seed", "unique"),
        ("disconnected", "Disconnected"),
        ("oversize", "16x16"),
        ("unbounded", "outer walls"),
        ("bad_object", "exactly one"),
        ("wrong_budget", "budget"),
        ("changed_control", "preserve"),
    ],
)
def test_invalid_manifest_fails_before_simulation(tmp_path, mutation, match):
    data = json.loads(DEFAULT_MANIFEST.read_text())
    layout, case = data["layouts"]["opaque_loop_grid"], data["cases"][6]
    if mutation == "unsafe_spawn":
        case["survivor_spawn"] = case["alien_spawn"][:2]
    elif mutation == "duplicate_seed":
        case["bot_seed"] = case["scenario_seed"]
    elif mutation == "disconnected":
        case["exploration_anchors"] = [[2.5, 2.5]]
    elif mutation == "oversize":
        layout["rows"] = ["#" * 17] + ["#" + "." * 15 + "#"] * 15 + ["#" * 17]
    elif mutation == "unbounded":
        layout["rows"][0] = "." * 12
    elif mutation == "bad_object":
        layout["objects"] = layout["objects"][:1]
    elif mutation == "wrong_budget":
        data["episode_cap_ticks"] = 1801
    elif mutation == "changed_control":
        data["layouts"]["small_room_control"]["objects"][0]["x"] = 5.6
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        read_occluded_fixtures(path)


def test_opaque_island_has_two_clear_branches():
    world = read_occluded_fixtures()[1][6].world()
    for route in (((1.5, 1.5), (1.5, 4.5), (4.5, 4.5)), ((1.5, 1.5), (4.5, 1.5), (4.5, 4.5))):
        assert all(segment_has_clearance(world, a, b, 0.2) for a, b in zip(route, route[1:]))
    assert not segment_has_clearance(world, (1.5, 1.5), (4.5, 4.5), 0.2)


def new_bot(world, anchors):
    b = FixtureMissionBot(world.rows, CoreConfig(), anchors)
    b.reset(np.random.default_rng(123))
    return b


def test_hidden_worlds_preserve_filtered_views_commands_and_object_knowledge():
    case = read_occluded_fixtures()[1][6]
    base = case.world()
    worlds = [
        replace(base, objects=(MapObject("keycard", "keycard", 1.5, y), base.objects[1]))
        for y in (7.5, 10.5)
    ]
    survivor = SurvivorState(4.5, 4.5)
    views = [
        observe_threat(w, survivor, False, AlienState(1.5, y), EvasionConfig())
        for w, y in zip(worlds, (1.5, 10.5))
    ]
    assert (
        views[0] == views[1]
        and views[0].visible_alien is None
        and not views[0].mission.visible_objects
    )
    bots = [new_bot(base, case.exploration_anchors) for _ in range(2)]
    for _ in range(30):
        assert bots[0].command(views[0]) == bots[1].command(views[1])
    assert not bots[0].explorer.known and not bots[1].explorer.known


def test_anchor_does_not_grant_object_knowledge_before_discovery():
    case = read_occluded_fixtures()[1][6]
    world = case.world()
    card = next(o for o in world.objects if o.kind == "keycard")
    b = new_bot(world, ((card.x, card.y),))
    assert not b.command(SurvivorThreatView(SurvivorView(card.x, card.y, 0, False, ()))).interact
    assert not b.explorer.known
    assert b.command(SurvivorThreatView(SurvivorView(card.x, card.y, 0, False, (card,)))).interact


def test_shared_selector_masks_and_exhaustion_use_only_allowed_evidence():
    world = read_occluded_fixtures()[1][6].world()
    v = AllowedAlienView(
        world.alien_spawn[:2],
        world.alien_spawn[2],
        False,
        30,
        False,
        None,
        None,
        world.patrol_waypoints,
        (None,) * len(world.patrol_waypoints),
        False,
    )
    selector = HeuristicSelector(0.15)
    assert candidate_masks(build_candidates(v))[selector.choose(v)]
    clue = replace(v, last_seen=(4.5, 4.5), clue_revision=1)
    assert selector.choose(clue) == 21
    assert selector.choose(replace(clue, visible=True)) == 20
    exhausted = replace(clue, investigation=InvestigationSnapshot(investigated_revision=1))
    assert not candidate_masks(build_candidates(exhausted))[21]


def test_ledger_enforces_both_caps_and_restart_without_core(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.jsonl", tick_cap=4, execution_cap=2, local_cap=2)
    for controller in ("wait", "heuristic"):
        ledger.begin("accounting_only", controller)
        ledger.tick()
        ledger.tick()
        with pytest.raises(RuntimeError):
            ledger.tick()
        ledger.end("not_a_simulation")
    with pytest.raises(RuntimeError):
        ledger.begin("extra", "wait")
    ledger.close()
    with pytest.raises(FileExistsError):
        BudgetLedger(tmp_path / "ledger.jsonl")


@pytest.mark.parametrize("phase", ["travelling", "sweeping"])
def test_metrics_separate_travel_and_sweep_reacquisition(phase):
    case = read_occluded_fixtures()[1][6]
    world, rows = case.world(), []
    for t in range(4):
        before = "travelling" if t == 2 else phase if t == 3 else "idle"
        after = phase if t == 2 else "cancelled" if t == 3 else "idle"

        def inv(p):
            return {
                "phase": p,
                "revision": 1 if p != "idle" else None,
                "rotation_ticks": 1 if p == "sweeping" else 0,
            }

        rows.append(
            {
                "tick_before": t,
                "tick_after": t + 1,
                "action": 21 if t >= 2 else 30,
                "alien_after": {"x": 4.5, "y": 4.5},
                "diagnostic_only_survivor_before": {"x": 1.5, "y": 4.5},
                "diagnostic_only_survivor_after": {"x": 1.5, "y": 4.5},
                "perception": {
                    "visible": t in (0, 3),
                    "last_seen": [4.5, 4.5],
                    "revision": 2 if t == 3 else 1,
                },
                "investigation_before": inv(before),
                "investigation_after": inv(after),
                "known_objects": [],
                "events": [],
                "outcome": None,
                "navigation_path": [],
                "bot_diagnostic": {"no_safe_route_count": 0},
            }
        )
    decisions = [
        {
            "tick": 2,
            "action": 21,
            "masks": [True] * 32,
            "view": {"visible": False, "last_seen": [4.5, 4.5]},
        }
    ]
    result = summarize_inspection(case, "heuristic", rows, decisions, world, CoreConfig())
    span = result["loss_windows"][0]
    assert span["first_informed_decision_tick"] == span["travel_start_tick"] == 2
    assert span["reacquisition_stage"] == phase and span["interrupted_by_fresh_confirmed_clue"]
    assert result["sweep_reacquisitions"] == (phase == "sweeping")
    assert result["travel_reacquisitions"] == (phase == "travelling")
    assert result["inspection_coverage_ready"] == (phase == "sweeping")
