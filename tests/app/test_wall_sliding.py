"""A body pressed against a wall slides along it, past the seams between wall cells."""

from game.core import GameCore
from game.map import scenario_map
from game.ship_deck import SHIP_SCENARIO
from game.state import CoreConfig, SurvivorCommand, SurvivorState

WORLD = scenario_map(SHIP_SCENARIO)
MARGIN = 0.3  # stop short of each end, where the wall really does turn
STILL_TICKS = 4


def floor_beside_wall(cells):
    """Stretches of at least two floor cells in a line with unbroken wall along one side."""
    start = None
    for index, cell in enumerate((*cells, None)):
        if cell is not None and start is None:
            start = index
        if cell is None and start is not None:
            if index - start >= 2:
                yield start, index
            start = None


def wall_walks():
    """Every such stretch both ways: (start, walking axis, walking way, way into the wall)."""
    for y in range(WORLD.height):
        for side in (-1, 1):
            cells = [
                (not WORLD.blocked(x, y) and WORLD.blocked(x, y + side)) or None
                for x in range(WORLD.width)
            ]
            for first, last in floor_beside_wall(cells):
                yield (first + MARGIN, y + 0.5), last - MARGIN, 0, 1, side
                yield (last - MARGIN, y + 0.5), first + MARGIN, 0, -1, side
    for x in range(WORLD.width):
        for side in (-1, 1):
            cells = [
                (not WORLD.blocked(x, y) and WORLD.blocked(x + side, y)) or None
                for y in range(WORLD.height)
            ]
            for first, last in floor_beside_wall(cells):
                yield (x + 0.5, first + MARGIN), last - MARGIN, 1, 1, side
                yield (x + 0.5, last - MARGIN), first + MARGIN, 1, -1, side


def stuck_walks(radius):
    core = GameCore(WORLD, CoreConfig(radius=radius, alien_enabled=False))
    core.reset(0)
    stuck = []
    for start, end, axis, way, side in wall_walks():
        core._survivor = SurvivorState(*start, 0.0)
        # Facing east, forward walks along x and strafing along y, whichever is the wall.
        along, into = (way, side) if axis == 0 else (side, way)
        command = SurvivorCommand(forward=along, strafe=into)
        still, position = 0, start
        while (end - position[axis]) * way > 0 and still < STILL_TICKS:
            survivor = core.step(command).survivor
            still = still + 1 if (survivor.x, survivor.y) == position else 0
            position = survivor.x, survivor.y
        if still == STILL_TICKS:
            stuck.append(position)
    return stuck


def test_a_body_slides_past_every_seam_on_the_ship():
    """0.2 has no exact binary form, and a wall to the south or east once stopped it at seams."""
    assert stuck_walks(CoreConfig().radius) == []
