"""The ship's mood with made-up inputs: no download and none of the licensed files."""

import numpy as np
import pytest
from PIL import Image

from renderers.ship_kernels import shrink_depth, stamp_depth
from renderers.ship_renderer import POWER_CYCLE, POWER_SAG, power_now
from tools.bake_ship_mood import MIST_SIZE, graded_table, mist_tile, read_cube


def neutral_cube(size=3):
    steps = np.linspace(0, 1, size, dtype=np.float32)
    blue, green, red = np.meshgrid(steps, steps, steps, indexing="ij")
    return np.stack((red, green, blue), axis=-1)


def test_a_cube_file_reads_with_red_changing_fastest(tmp_path):
    path = tmp_path / "look.cube"
    lines = ["TITLE \"made up\"", "LUT_3D_SIZE 2"]
    lines += [f"{r} {g} {b}" for b in (0, 1) for g in (0, 1) for r in (0, 1)]
    path.write_text("\n".join(lines))
    cube = read_cube(path)
    assert cube.shape == (2, 2, 2, 3)
    assert cube[0, 0, 1].tolist() == [1, 0, 0] and cube[1, 0, 0].tolist() == [0, 0, 1]


def test_a_neutral_look_leaves_every_color_alone():
    table = graded_table(neutral_cube(), 1.0, 16).astype(np.int32)
    plain = graded_table(neutral_cube(), 0.0, 16).astype(np.int32)
    assert np.abs(table - plain).max() <= 1
    assert table[15, 0, 8].tolist() == plain[15, 0, 8].tolist()


def test_strength_decides_how_much_of_the_look_is_kept():
    red_only = neutral_cube()
    red_only[..., 1:] = 0
    half = graded_table(red_only, 0.5, 8).astype(np.int32)
    full = graded_table(red_only, 1.0, 8).astype(np.int32)
    assert full[0, 7, 0, 1] == 0
    assert half[0, 7, 0, 1] == pytest.approx(graded_table(red_only, 0.0, 8)[0, 7, 0, 1] / 2, abs=1)


def test_the_grade_indexes_red_green_blue_the_way_the_frame_does():
    table = graded_table(neutral_cube(), 1.0, 8)
    assert table[7, 0, 0, 0] > 200 and table[7, 0, 0, 2] < 50
    assert table[0, 0, 7, 2] > 200 and table[0, 0, 7, 0] < 50


def test_mist_keeps_an_even_thickness_and_has_real_gaps(tmp_path):
    puff = np.zeros((32, 32, 4), np.uint8)
    puff[8:24, 8:24, 3] = 200
    path = tmp_path / "whitePuff00.png"
    Image.fromarray(puff).save(path)
    tile = mist_tile([path])
    assert tile.shape == (MIST_SIZE, MIST_SIZE)
    assert tile.mean() == pytest.approx(0.5) and tile.max() > 2 * tile.min()


def test_power_sags_for_a_moment_and_never_dies():
    levels = [power_now(tick) for tick in range(POWER_CYCLE * 8)]
    sagging = sum(level < 1 for level in levels)
    assert sagging == 8 * POWER_SAG and min(levels) > 0


def test_sags_replay_exactly():
    assert [power_now(t) for t in range(5000)] == [power_now(t) for t in range(5000)]


def wall_seen_by_the_air(depth):
    air_depth = np.zeros((23, 40), np.float32)
    shrink_depth(air_depth, np.full((45, 80), 1 / depth, np.float64))
    return air_depth


def test_the_mist_stops_at_the_alien_and_goes_on_past_it():
    air_depth = wall_seen_by_the_air(4.0)
    stamp_depth(air_depth, np.full((40, 20, 4), 255, np.uint8), (36, 8, 8, 28), 1 / 2.0)
    assert air_depth[10, 20] == pytest.approx(1 / 2.0)
    assert air_depth[10, 5] == pytest.approx(1 / 4.0)


def test_a_see_through_part_of_a_sprite_stops_no_mist():
    air_depth = wall_seen_by_the_air(4.0)
    before = air_depth.copy()
    stamp_depth(air_depth, np.zeros((40, 20, 4), np.uint8), (36, 8, 8, 28), 1 / 2.0)
    assert (air_depth == before).all()
