"""Summarize a human-play recording: outcomes, pursuit, search and patrol commitment.

Usage: python -m tools.review_recording experiments/<recording_directory>

Reads saved records only; it never loads a policy or advances a simulation. Survivor
truth in the trace explains behavior here and must never reach a policy.
"""

import argparse
from collections import Counter
from dataclasses import dataclass, field
from math import hypot
from pathlib import Path

from app.recorder import read_records
from game.state import CoreConfig
from rl.contract import ACTION_NAMES

CONFIG = CoreConfig()
WAIT_REPORT_TICKS = 3 * CONFIG.tick_hz  # shorter Survivor pauses are not reported
SEARCH_ACTIVE = ("travelling", "sweeping")


def distance(a, b):
    return hypot(a["x"] - b["x"], a["y"] - b["y"])


@dataclass
class EpisodeReview:
    episode: int
    scenario_seed: int | None = None
    outcome: str | None = None
    status: str = "incomplete"
    ticks: int = 0
    visible_ticks: int = 0
    stationary_while_visible: int = 0
    longest_parked_ticks: int = 0
    parked_at: tuple[float, float] | None = None
    searches: Counter = field(default_factory=Counter)
    actions: Counter = field(default_factory=Counter)
    restricted: Counter = field(default_factory=Counter)
    route: Counter = field(default_factory=Counter)  # unseen, en-route waypoint decisions
    reached: set = field(default_factory=set)  # waypoint actions walked to arrival
    waits: list = field(default_factory=list)  # Survivor still and unseen
    parked_run: int = 0
    open_wait: dict | None = None
    last_unseen_decision: dict | None = None

    def close_wait(self):
        if self.open_wait is not None and self.open_wait["ticks"] >= WAIT_REPORT_TICKS:
            self.waits.append(self.open_wait)
        self.open_wait = None

    def decision(self, record):
        name = ACTION_NAMES[record["action"]]
        self.actions[name] += 1
        if [i for i, allowed in enumerate(record["mask"]) if allowed] == [21, 30]:
            self.restricted[name] += 1
        previous = self.last_unseen_decision
        self.last_unseen_decision = None if record["allowed_view"]["visible"] else record
        if self.last_unseen_decision is None or previous is None or previous["action"] >= 20:
            return
        target = dict(zip("xy", previous["chosen_target"]))
        if distance(record["alien_before"], target) <= CONFIG.target_arrival_distance:
            self.reached.add(previous["action"])
            self.route["arrived"] += 1
            return  # arrived: choosing a new waypoint is not an abandoned route
        kept = record["action"] == previous["action"]
        self.route["kept" if kept else "switched" if record["action"] < 20 else "other"] += 1

    def tick(self, record):
        before, after = record["alien_before"], record["alien_after"]
        truth = record["diagnostic_truth_only"]
        visible = bool((record["sensor_before_movement"] or {}).get("visible"))
        step = distance(before, after)
        still = step < 1e-9
        self.ticks += 1
        self.visible_ticks += visible
        self.stationary_while_visible += visible and still
        old, new = record["investigation_before"], record["investigation_after"]
        if new["phase"] == "travelling" and (
            old["phase"] != "travelling" or old["revision"] != new["revision"]
        ):
            self.searches["started"] += 1
        if old["phase"] in SEARCH_ACTIVE:
            if new["phase"] in ("completed", "cancelled"):
                self.searches[new["phase"]] += 1
            elif new["phase"] == "idle":
                self.searches["policy_cancelled"] += 1
        if still and new["phase"] == "idle" and not visible:
            self.parked_run += 1
            if self.parked_run > self.longest_parked_ticks:
                self.longest_parked_ticks = self.parked_run
                self.parked_at = (after["x"], after["y"])
        else:
            self.parked_run = 0
        survivor = truth["survivor_after"]
        if not visible and distance(truth["survivor_before"], survivor) < 1e-6:
            if self.open_wait is None:
                self.open_wait = {
                    "start_tick": record["tick_after"],
                    "survivor": (survivor["x"], survivor["y"]),
                    "ticks": 0,
                    "alien_moved": 0.0,
                    "alien_still_ticks": 0,
                }
            self.open_wait["ticks"] += 1
            self.open_wait["alien_moved"] += step
            self.open_wait["alien_still_ticks"] += still
        else:
            self.close_wait()


def review_records(records):
    episodes = {}
    for record in records:
        kind = record["type"]
        if kind not in ("episode_start", "decision", "tick", "episode_end"):
            continue
        review = episodes.setdefault(record["episode"], EpisodeReview(record["episode"]))
        if kind == "episode_start":
            review.scenario_seed = record["scenario_seed"]
        elif kind == "decision":
            review.decision(record)
        elif kind == "tick":
            review.tick(record)
        else:
            review.outcome, review.status = record["outcome"], record["status"]
    for review in episodes.values():
        review.close_wait()
    return [episodes[key] for key in sorted(episodes)]


def seconds(ticks):
    return ticks / CONFIG.tick_hz


def print_report(reviews):
    for r in reviews:
        print(
            f"episode {r.episode}: {r.outcome} ({r.status}) {seconds(r.ticks):.1f}s, "
            f"scenario seed {r.scenario_seed}"
        )
        print(
            f"  sight: visible {r.visible_ticks} ticks, "
            f"stood still while visible {r.stationary_while_visible}"
        )
        print(
            f"  search: started {r.searches['started']}, completed {r.searches['completed']}, "
            f"cancelled by sight {r.searches['cancelled']}, "
            f"cancelled by policy {r.searches['policy_cancelled']}"
        )
        place = "-" if r.parked_at is None else f"({r.parked_at[0]:.1f}, {r.parked_at[1]:.1f})"
        print(
            f"  patrol: arrived {r.route['arrived']} times at {len(r.reached)} waypoints; "
            f"en route kept {r.route['kept']}, switched {r.route['switched']}, "
            f"other {r.route['other']}; longest parked {seconds(r.longest_parked_ticks):.1f}s "
            f"at {place}"
        )
        print(f"  actions: {dict(r.actions.most_common(6))}  restricted: {dict(r.restricted)}")
        for wait in r.waits:
            x, y = wait["survivor"]
            print(
                f"  Survivor waited unseen {seconds(wait['ticks']):.1f}s at ({x:.1f}, {y:.1f}) "
                f"from tick {wait['start_tick']}: Alien moved {wait['alien_moved']:.1f} units, "
                f"still {wait['alien_still_ticks']}/{wait['ticks']} ticks"
            )
    seeds = {r.scenario_seed for r in reviews}
    print(f"rounds {len(reviews)}, different scenario seeds {len(seeds)}")
    route = sum((r.route for r in reviews), Counter())
    decisions = route["kept"] + route["switched"] + route["other"]
    if decisions:
        print(
            f"all episodes, {decisions} unseen en-route decisions: "
            f"kept {route['kept'] / decisions:.0%}, switched {route['switched'] / decisions:.0%}, "
            f"other {route['other'] / decisions:.0%}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording", type=Path)
    args = parser.parse_args(argv)
    print_report(review_records(read_records(args.recording, allow_partial_tail=True)))


if __name__ == "__main__":
    main()
