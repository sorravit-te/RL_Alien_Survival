"""Sky bake geometry with tiny made-up maps: no NASA file and no network."""

from math import atan, degrees, pi

import numpy as np
import pytest

from renderers.ship_kernels import sky_at
from tools.bake_ship_sky import (
    EARTH_PLACE,
    HEIGHT,
    MOON_PLACE,
    RING_OUTER,
    RINGED_PLACE,
    SUN,
    WIDTH,
    disc_view,
    finished,
    globe_position,
    lookup,
    ring_plane,
    sky_grid,
    sunlight,
)


def point(x, y, facing):
    latitude, longitude = globe_position(np.array(float(x)), np.array(float(y)), facing)
    return degrees(float(latitude)), degrees(float(longitude))


def test_the_middle_of_a_globe_is_the_place_turned_to_the_ship():
    assert point(0, 0, (12.0, 100.0, -23.0)) == pytest.approx((12.0, 100.0))


def test_east_lies_right_and_north_up_so_the_world_is_never_mirrored():
    latitude, longitude = point(0.4, 0, (0.0, 0.0, 0.0))
    assert longitude > 0 and latitude == pytest.approx(0)
    latitude, longitude = point(0, 0.4, (0.0, 0.0, 0.0))
    assert latitude > 0 and longitude == pytest.approx(0)


def test_a_tilt_leans_the_globe_and_keeps_its_middle():
    assert point(0, 0, (0.0, 30.0, 40.0)) == pytest.approx((0.0, 30.0))
    assert point(0.4, 0, (0.0, 0.0, 40.0))[0] > 0  # the right side leans north


def test_a_map_joins_up_across_the_date_line():
    image = np.zeros((2, 4, 3), np.float32)
    image[:, 0] = 1.0  # westernmost column
    just_past = lookup(image, np.array(pi - 1e-3), np.array(0.0))
    assert 0.4 < just_past[0] < 0.6


@pytest.mark.parametrize("place", [(6.0, -4.0, 17.0), (-29.0, 2.5, 1.9)])
def test_a_body_is_drawn_where_it_was_placed(place):
    block, x, y = disc_view(place, 1.3)
    row, column = np.unravel_index(np.argmin(x * x + y * y), x.shape)
    longitude, latitude = sky_grid(
        block[1].start + column, block[1].start + column + 1,
        block[0].start + row, block[0].start + row + 1,
    )  # fmt: skip
    assert degrees(float(longitude[0, 0])) == pytest.approx(place[0], abs=0.2)
    assert degrees(float(latitude[0, 0])) == pytest.approx(place[1], abs=0.2)


def test_the_side_of_a_globe_toward_the_sun_is_lit():
    lit, _ = sunlight(np.array(SUN[0] * 0.9), np.array(SUN[1] * 0.9))
    dark, _ = sunlight(np.array(-SUN[0] * 0.9), np.array(-SUN[1] * 0.9))
    assert lit > 0.5 > dark


def test_rings_lie_flat_around_their_world():
    across, _ = ring_plane(np.array(1.5), np.array(0.0))
    stood_up, _ = ring_plane(np.array(0.0), np.array(1.5))
    assert stood_up > across


def test_pixels_are_packed_the_way_the_shader_reads_them():
    sky = np.zeros((1, 1, 3), np.float32)
    sky[0, 0] = (50.0, 0.0, 0.0)
    assert finished(sky)[0, 0] == 255  # red alone, in the lowest byte


def sky_with(row, column):
    sky = np.zeros((HEIGHT // 256, WIDTH // 256), np.uint32)
    sky[row, column] = 255
    return sky


@pytest.mark.parametrize(
    "ray, row, column",
    [((0.0, 1.0, 0.0), 3, 8), ((1.0, 0.0, 0.0), 3, 12), ((-1.0, 0.0, 0.0), 3, 4)],
)
def test_the_shader_looks_the_way_the_bake_drew(ray, row, column):
    sky = sky_with(row, column)
    red = max(sky_at(sky, *ray)[0], sky_at(sky, ray[0] + 1e-3, ray[1], 0.0)[0])
    assert red > 0.05
    assert sky_at(sky, -ray[0], -ray[1], 0.0)[0] == 0


# Degrees off the bow of a middle pane's two edges, from where the corridor meets the bridge.
PANE_NEAR_EDGE, PANE_FAR_EDGE = degrees(atan(0.21 / 12.3)), degrees(atan(1.58 / 12.3))


def test_earth_stays_out_of_the_panes_seen_from_the_corridor():
    """Space never scales, so a world wider than a far pane would look zoomed in there."""
    longitude, _, radius = EARTH_PLACE
    assert longitude - radius > PANE_FAR_EDGE


@pytest.mark.parametrize(
    ("place", "reach"), [(MOON_PLACE, MOON_PLACE[2]), (RINGED_PLACE, RINGED_PLACE[2] * RING_OUTER)]
)
def test_small_worlds_fit_a_pane_seen_from_the_corridor(place, reach):
    assert PANE_NEAR_EDGE < abs(place[0]) - reach
    assert abs(place[0]) + reach < PANE_FAR_EDGE
