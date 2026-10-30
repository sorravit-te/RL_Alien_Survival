"""Ship renderer rules with fake textures, so no purchased artwork is needed."""

import os
from dataclasses import replace
from math import pi

import numpy as np
import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from game.core import GameCore
from game.map import scenario_map
from game.ship_deck import SHIP_SCENARIO
from game.state import AlienState, CoreConfig, SurvivorState
from renderers.raycasting import cast_ray
from renderers.ship_kernels import SKY
from renderers.ship_mesh import DOOR_TOP, PILLAR_PROUD, WALL_TOP, build_mesh
from renderers.ship_renderer import ShipRenderer
from renderers.ship_theme import ZONES, ShipTheme
from tools.bake_ship_textures import MATERIALS, texture_names

WORLD = scenario_map(SHIP_SCENARIO)
SIZE = (80, 45)
FAKE_TEXELS = 16


@pytest.fixture(scope="module")
def textures(tmp_path_factory):
    names = texture_names()
    rng = np.random.default_rng(1)
    path = tmp_path_factory.mktemp("ship") / "fake.npz"
    levels = {
        f"level{i}": rng.integers(
            40, 220, (len(names), FAKE_TEXELS >> i, FAKE_TEXELS >> i, 4), dtype=np.uint8
        )
        for i in range(3)
    }
    np.savez(path, names=np.array(names), **levels)
    return path


@pytest.fixture(scope="module")
def renderer(textures):
    pygame.init()
    pygame.display.set_mode((16, 16))
    return ShipRenderer(WORLD, textures, size=SIZE)


@pytest.fixture(scope="module")
def theme():
    return ShipTheme(WORLD)


@pytest.fixture(scope="module")
def mesh(theme):
    return build_mesh(theme, texture_names())


@pytest.fixture
def snapshot():
    base = GameCore(WORLD).reset(3)
    return replace(base, survivor=SurvivorState(19.5, 28.5, -pi / 2), alien=None, tick=120)


def picture(renderer, snapshot, pitch=0.0):
    return pygame.image.tobytes(renderer.render(snapshot, pitch), "RGB")


def test_every_purchased_material_is_baked():
    assert set(MATERIALS) <= set(texture_names())
    assert len(MATERIALS) == 54


def test_every_zone_exists_for_every_cell(theme):
    assert all(
        theme.zone_at(x, y) in ZONES
        for y in range(theme.height)
        for x in range(theme.width)
        if theme.is_open(x, y)
    )


def test_walls_are_built_from_more_than_flat_planes(mesh):
    leaning = np.abs(mesh["normals"][:, 2])
    assert ((leaning > 0.05) & (leaning < 0.95)).sum() > 200  # tilted panels, lintels and sills
    bay = texture_names().index("wall10")
    assert (mesh["textures"] == bay).sum() > 20


def test_ceilings_step_and_doorways_drop_a_header(mesh):
    heights = mesh["vertices"][:, :, 2]
    assert np.isclose(heights, WALL_TOP).any() and np.isclose(heights, DOOR_TOP).any()


def test_door_leaves_carry_their_seam_in_the_texture(mesh):
    seam = texture_names().index("door1_seam")
    assert (mesh["textures"][mesh["door_quads"]] == seam).all()


def test_windows_show_space_from_behind_closed_reveals(mesh):
    sky = texture_names().index("viewport")
    panes = mesh["textures"] == sky
    assert panes.sum() >= 4 and (mesh["shades"][panes] == SKY).all()
    frame = (mesh["textures"] == texture_names().index("walltrim3")) & (mesh["shades"] < 0.75)
    assert frame.sum() >= panes.sum() * 4


@pytest.mark.parametrize(
    ("x", "y", "yaw"),
    [(19.5, 6.5, -pi / 2), (16.7, 4.4, -1.05), (22.3, 4.4, -2.1), (17.0, 3.4, -0.3)],
)
def test_bridge_has_no_hole_to_see_through(renderer, snapshot, x, y, yaw):
    renderer.render(replace(snapshot, survivor=SurvivorState(x, y, yaw)))
    assert (renderer.ids < 0).sum() == 0


def test_nothing_stands_where_the_player_could_walk_into_it():
    assert PILLAR_PROUD < CoreConfig().radius


def test_furniture_is_only_what_stands_inside_a_room(theme):
    assert theme.blocks[17, 24]  # mess table
    assert theme.blocks[33, 27]  # reactor core
    assert theme.blocks[20, 19]  # plaza column
    assert not theme.blocks[20, 22]  # the wall between plaza and mess


def test_depth_matches_the_plain_ray_caster(renderer, snapshot):
    camera = replace(snapshot, survivor=SurvivorState(19.5, 20.5, pi / 2))
    renderer.render(camera)
    assert len(renderer.depth) == SIZE[0]
    forward = cast_ray(WORLD, 19.5, 20.5, 0.0, 1.0).distance
    assert renderer.depth[SIZE[0] // 2] == pytest.approx(forward, abs=0.6)


def test_same_snapshot_draws_the_same_frame(renderer, snapshot):
    assert picture(renderer, snapshot) == picture(renderer, snapshot)


def test_time_changes_only_through_the_tick(renderer, snapshot):
    assert picture(renderer, snapshot) != picture(renderer, replace(snapshot, tick=121))


def test_pitch_shifts_the_view_but_never_the_surfaces(renderer, snapshot):
    tilted = picture(renderer, snapshot, 0.3)
    assert tilted != picture(renderer, snapshot)
    assert picture(renderer, snapshot, 0.0) == picture(renderer, snapshot)


def test_alien_behind_a_wall_is_never_drawn(renderer, snapshot):
    hidden = replace(snapshot, alien=AlienState(25.5, 18.5, 0.0))
    assert picture(renderer, hidden) == picture(renderer, snapshot)


def test_alien_in_the_corridor_is_drawn(renderer, snapshot):
    shown = replace(snapshot, alien=AlienState(19.5, 24.5, 0.0))
    assert picture(renderer, shown) != picture(renderer, snapshot)


def test_light_does_not_reveal_the_working_exit(theme):
    lamps = [light for light in theme.lights if light[2] == (0.3, 1.0, 0.55)]
    assert len(lamps) == 3 and len({(light[3], light[4]) for light in lamps}) == 1


def test_collected_keycard_disappears(renderer, snapshot):
    card = next(obj for obj in snapshot.world_map.objects if obj.kind == "keycard")
    near = replace(snapshot, survivor=SurvivorState(card.x, card.y + 2.0, -pi / 2))
    assert picture(renderer, near) != picture(renderer, replace(near, has_keycard=True))


def test_missing_textures_explain_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="tools.bake_ship_textures"):
        ShipRenderer(WORLD, tmp_path / "missing.npz")
