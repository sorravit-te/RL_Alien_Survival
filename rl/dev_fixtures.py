"""Public-geometry mission route preferences for bounded lost-sight DEV checks.

Not a training configuration. Objectives are learned through SurvivorThreatView
by the existing bot. An anchor coinciding with an object conveys no discovery.
"""

import json
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path

from ai.navigation import find_path
from game.map import small_test_map
from rl.evasive_survivor import EvasiveExplorerBot

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "configs/lost_sight_dev_fixtures.json"


@dataclass(frozen=True)
class DevFixture:
    case_id: str
    profile: str
    scenario_seed: int
    bot_seed: int
    survivor_spawn_index: int
    exploration_anchors: tuple[tuple[float, float], ...]
    intended_alignment: str

    def world(self):
        base = small_test_map()
        # Existing spawn, explicitly frozen per case; geometry/object/catalogue unchanged.
        return replace(base, spawn_points=(base.spawn_points[self.survivor_spawn_index],))


def read_fixtures(path=DEFAULT_MANIFEST):
    data = json.loads(Path(path).read_text(encoding="utf8"))
    if data["format"] != "lost-sight-dev-fixtures-v1" or data["split"] != "development_only":
        raise ValueError("Expected the explicit development fixture manifest")
    cases = tuple(
        DevFixture(
            **{**case, "exploration_anchors": tuple(map(tuple, case["exploration_anchors"]))}
        )
        for case in data["cases"]
    )
    if len(cases) != 12 or len({c.case_id for c in cases}) != 12:
        raise ValueError("Expected twelve unique frozen cases")
    if any(c.survivor_spawn_index not in (0, 1) for c in cases):
        raise ValueError("Use existing Survivor spawns only")
    seeds = [s for c in cases for s in (c.scenario_seed, c.bot_seed)]
    if any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != 24:
        raise ValueError("Expected unique nonnegative dev seeds")
    return data, cases


class FixtureMissionBot(EvasiveExplorerBot):
    """Reuse threat/mission behavior verbatim; change only initial exploration order."""

    label = "public_route_evasive_explorer_dev"

    def __init__(self, rows, config, anchors):
        super().__init__(rows, config)
        self.anchors = tuple(tuple(p) for p in anchors)
        if not self.anchors or len(set(self.anchors)) != len(self.anchors):
            raise ValueError("Exploration anchors must be nonempty and unique")
        origin = self.explorer.points[0]
        for point in self.anchors:
            if (
                len(point) != 2
                or not all(isfinite(v) for v in point)
                or not find_path(self.explorer.geometry, origin, point, config.radius)
            ):
                raise ValueError("Exploration anchor lacks a clearance-aware public route")

    def reset(self, rng):
        super().reset(rng)
        tail = tuple(p for p in self.explorer.order if p not in self.anchors)
        self.explorer.order = self.anchors + tail
