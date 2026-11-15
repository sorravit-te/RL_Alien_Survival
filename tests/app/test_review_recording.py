"""Recording review over a real recorder trace; fake policy only, no checkpoint."""

import pytest
from test_recording import attach, driver

from app import recorder as recording
from game.state import SurvivorCommand
from rl.contract import WAIT
from tools import review_recording as review
from tools.review_recording import main, print_report, review_records


@pytest.fixture
def root(tmp_path, monkeypatch):
    path = tmp_path / "experiments"
    path.mkdir()
    monkeypatch.setattr(recording, "EXPERIMENTS_ROOT", path)
    return path


def decision(action, target, visible=False, alien=(0.0, 0.0)):
    return {
        "type": "decision",
        "episode": 0,
        "action": action,
        "mask": [False] * 32,
        "chosen_target": target,
        "allowed_view": {"visible": visible},
        "alien_before": {"x": alien[0], "y": alien[1]},
    }


def test_real_recording_reports_waiting_survivor_and_kept_route(root, capsys):
    app = driver(sampled=False, action=1)
    rec = attach(root / "review", app)
    for _ in range(100):
        app.step(SurvivorCommand())
    rec.close(completed=True, reason="synthetic_end")
    main([str(rec.directory)])
    report = capsys.readouterr().out
    assert "episode 0: None (partial) 3.3s, scenario seed 7799101" in report
    assert "rounds 1, different scenario seeds 1" in report
    assert "Survivor waited unseen 3.3s at (2.5, 2.5)" in report
    reviews = review_records(recording.read_records(rec.directory))
    assert len(reviews) == 1
    review = reviews[0]
    assert review.ticks == 100 and review.actions["waypoint_1"] == 7
    assert review.route["kept"] >= 1 and review.route["switched"] == 0
    assert review.visible_ticks == 0 and review.searches["started"] == 0
    assert len(review.waits) == 1 and review.waits[0]["ticks"] == 100


def test_route_counts_abandoned_waypoints_but_not_arrivals_or_sightings(capsys):
    a, b = (5.0, 0.0), (0.0, 5.0)
    records = [
        decision(0, a),
        decision(1, b),  # left waypoint 0 unreached
        decision(1, b),  # kept waypoint 1
        decision(WAIT, None),  # gave up on waypoint 1
        decision(0, a),  # previous action was not a waypoint
        decision(1, b, alien=a),  # arrived at waypoint 0 first
        decision(0, a, visible=True),  # sighting resets the unseen chain
        decision(1, b),
    ]
    review = review_records(records)[0]
    assert review.route == {"switched": 1, "kept": 1, "other": 1, "arrived": 1}
    assert review.reached == {0}
    print_report([review])
    assert "3 unseen en-route decisions: kept 33%, switched 33%, other 33%" in (
        capsys.readouterr().out
    )


def start(episode, seed, survivor, card, alien=(31.5, 36.5)):
    objects = [{"object_id": card, "kind": "keycard"}, {"object_id": "exit_pods", "kind": "exit"}]
    state = {
        "survivor": {"x": survivor[0], "y": survivor[1], "yaw": 0.0},
        "alien": {"x": alien[0], "y": alien[1]},
        "world_map": {"objects": objects},
    }
    return {
        "type": "episode_start",
        "episode": episode,
        "scenario_seed": seed,
        "diagnostic_initial_state_only": state,
    }


def ticks(episode, count, events=()):
    """Plain ticks with the Survivor on the move; events land on the last one."""
    alien, idle = {"x": 1.0, "y": 1.0}, {"phase": "idle", "revision": None}
    for i in range(count):
        yield {
            "type": "tick",
            "episode": episode,
            "tick_after": i + 1,
            "alien_before": alien,
            "alien_after": alien,
            "sensor_before_movement": {"visible": False},
            "investigation_before": idle,
            "investigation_after": idle,
            "diagnostic_truth_only": {
                "survivor_before": {"x": 2.0, "y": float(i)},
                "survivor_after": {"x": 2.0, "y": float(i + 1)},
            },
            "core_events": list(events) if i == count - 1 else [],
        }


def end(episode, outcome):
    status = "partial" if outcome is None else "completed"
    return {"type": "episode_end", "episode": episode, "outcome": outcome, "status": status}


def card(tick):
    return {"kind": "keycard_collected", "object_id": "keycard_cabin", "tick": tick}


A, B = (14.5, 16.5), (24.5, 18.5)
SESSION = [
    start(0, 310002, A, "keycard_lab"),
    end(0, None),  # the opening round, left paused for N: never played
    start(1, 11, A, "keycard_cabin"), *ticks(1, 30, [card(12)]), end(1, "captured"),
    start(2, 11, A, "keycard_cabin"), *ticks(2, 90), end(2, "escaped"),  # R replay
    start(3, 22, B, "keycard_cargo"), *ticks(3, 60, [card(45)]), end(3, "escaped"),
    start(4, 33, B, "keycard_cargo"), *ticks(4, 15), end(4, "captured"),  # other seed, same round
    start(5, 44, A, "keycard_lab"), *ticks(5, 6), end(5, None),  # the opening round, now played
]


def test_first_tries_skip_replays_repeats_and_rounds_never_played(capsys):
    reviews = review_records(SESSION)
    assert [r.repeat_of for r in reviews] == [None, None, 1, None, 3, None]
    assert [r.card_tick for r in reviews] == [None, 12, None, 45, None, None]
    print_report(reviews)
    out = capsys.readouterr().out
    assert "episode 0: None (partial) 0.0s, scenario seed 310002, not played, no card" in out
    assert "episode 2: escaped (completed) 3.0s, scenario seed 11, repeat of episode 1" in out
    assert "episode 3: escaped (completed) 2.0s, scenario seed 22, first try, card at 1.5s" in out
    assert (
        "this recording: first tries 3 of 5 played rounds: escaped 1, captured 1, unfinished 1; "
        "card collected in 2"
    ) in out
    assert "  escaped after a median 2.0s over 1" in out
    assert "  captured after a median 1.0s over 1" in out


def test_each_player_has_their_own_first_tries(monkeypatch, capsys):
    """The same round is new to a second player, so a repeat only counts within one recording."""
    monkeypatch.setattr(review, "read_records", lambda path, **_: iter(SESSION))
    main(["experiments/human_pilot_p1", "experiments/human_pilot_p2"])
    out = capsys.readouterr().out
    assert "== experiments" in out and out.count("this recording: first tries 3 of 5") == 2
    assert (
        "all 2 recordings: first tries 6 of 10 played rounds: escaped 2, captured 2, unfinished 2; "
        "card collected in 4"
    ) in out

