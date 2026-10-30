"""Recording review over a real recorder trace; fake policy only, no checkpoint."""

import pytest
from test_recording import attach, driver

from app import recorder as recording
from game.state import SurvivorCommand
from rl.contract import WAIT
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
    assert "episode 0: None (partial) 3.3s" in report
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
