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
from renderers.ship_kernels import FOG_STEPS, SKY
from renderers.ship_mesh import (
    COLLAR_SCALE,
    DOOR_COLUMN_PROUD,
    DOOR_HEADER_PROUD,
    DOOR_TOP,
    OVERHEAD_FLOOR,
    PILLAR_PROUD,
    PIPE_OUT,
    PIPE_RADIUS,
    SIGN_PROUD,
    WALL_TOP,
    Mesh,
    Wall,
    build_mesh,
)
from renderers.ship_renderer import FIELD_OF_VIEW, ShipRenderer
from renderers.ship_theme import (
    FACES,
    LAMP_HIGH,
    LAMP_LOW,
    LIGHT_GRID,
    ZONES,
    ShipTheme,
    muted,
)
from tools.bake_ship_textures import MATERIALS, facing_axes, texture_names

WORLD = scenario_map(SHIP_SCENARIO)
SIZE = (80, 45)
FAKE_TEXELS = 16


@pytest.fixture(scope="module")
def textures(tmp_path_factory):
    names = texture_names()
    rng = np.random.default_rng(1)
    path = tmp_path_factory.mktemp("ship") / "fake.npz"
    levels = {
        f"{kind}{i}": rng.integers(
            40, 220, (len(names), FAKE_TEXELS >> i, FAKE_TEXELS >> i, 4), dtype=np.uint8
        )
        for kind in ("level", "relief")
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
    sky = texture_names().index("window_glass")
    panes = mesh["textures"] == sky
    assert panes.sum() >= 4 and (mesh["shades"][panes] == SKY).all()
    frame = (mesh["textures"] == texture_names().index("walltrim3")) & (mesh["shades"] < 0.75)
    assert frame.sum() >= panes.sum() * 4


def sky_file(tmp_path, west, east):
    sky = np.full((256, 512), west, np.uint32)
    sky[:, 256:] = east
    path = tmp_path / "sky.npz"
    np.savez(path, sky=sky)
    return path


def window_pixels(renderer, frame):
    """Screen columns and colors of what shows through the glass, in the last render."""
    quads = renderer.tri_quad[np.maximum(renderer.ids, 0)]
    glass = (renderer.ids >= 0) & (renderer.mesh["shades"][quads] == SKY)
    rows, columns = np.nonzero(glass)
    return columns, pygame.surfarray.array3d(frame).swapaxes(0, 1)[rows, columns].astype(np.int32)


def test_space_keeps_east_on_the_right_through_the_windows(textures, tmp_path, snapshot):
    red, blue = 0x0000FF, 0xFF0000
    wide = (320, 180)
    seen = ShipRenderer(WORLD, textures, size=wide, sky_path=sky_file(tmp_path, red, blue))
    camera = replace(snapshot, survivor=SurvivorState(19.5, 4.5, -pi / 2))
    columns, colors = window_pixels(seen, seen.render(camera))
    left, right = colors[columns < 150], colors[columns > 170]
    assert left[:, 0].mean() > left[:, 2].mean() and right[:, 2].mean() > right[:, 0].mean()


@pytest.mark.parametrize(("x", "y"), [(19.1, 3.0), (20.9, 3.0), (19.1, 14.0)])
def test_space_ignores_where_the_player_stands(textures, tmp_path, snapshot, x, y):
    """Like a real window: anything that scaled with distance would read as a picture on the wall."""
    red, blue = 0x0000FF, 0xFF0000
    wide = (320, 180)
    seen = ShipRenderer(WORLD, textures, size=wide, sky_path=sky_file(tmp_path, red, blue))
    camera = replace(snapshot, survivor=SurvivorState(x, y, -pi / 2))
    columns, colors = window_pixels(seen, seen.render(camera))
    left, right = colors[columns < 158], colors[columns > 162]
    assert left[:, 0].mean() > left[:, 2].mean() * 2 and right[:, 2].mean() > right[:, 0].mean() * 2


def test_glass_shows_even_with_nothing_but_dark_behind_it(textures, tmp_path, snapshot):
    dark = ShipRenderer(WORLD, textures, size=(320, 180), sky_path=tmp_path / "none.npz")
    assert dark.sky.shape == (2, 4) and not dark.sky.any()
    camera = replace(snapshot, survivor=SurvivorState(19.5, 3.2, -pi / 2))
    _, colors = window_pixels(dark, dark.render(camera))
    assert colors.mean() > 2  # room light and marks both land on the pane


def test_a_window_is_sealed_around_its_glass(mesh):
    panes = mesh["vertices"][mesh["textures"] == texture_names().index("window_glass")]
    seals = mesh["vertices"][mesh["textures"] == texture_names().index("pane_seal")]
    for pane in panes:
        low, high = pane.min(axis=0), pane.max(axis=0)
        around = (seals.min(axis=1) >= low - 0.05).all(axis=1)
        around &= (seals.max(axis=1) <= high + 0.05).all(axis=1)
        assert around.sum() == 4


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
    assert theme.blocks[32, 8]  # cargo crates
    assert theme.blocks[33, 27]  # reactor core
    assert theme.blocks[20, 19]  # plaza column
    assert not theme.blocks[20, 22]  # the wall between plaza and mess


def test_small_rooms_stand_open(theme):
    faced = {
        theme.zone_at(x + dx, y + dy)
        for y, x in zip(*theme.blocks.nonzero(), strict=True)
        for dx, dy in FACES
        if theme.is_open(x + dx, y + dy)
    }
    assert faced == set("CRZ")


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


def test_a_bake_from_before_a_new_texture_explains_the_fix(tmp_path):
    names = [name for name in texture_names() if name != "pipe"]
    levels = {
        f"{kind}{i}": np.zeros((len(names), 16 >> i, 16 >> i, 4), np.uint8)
        for kind in ("level", "relief")
        for i in range(3)
    }
    path = tmp_path / "old.npz"
    np.savez(path, names=np.array(names), **levels)
    with pytest.raises(FileNotFoundError, match="tools.bake_ship_textures"):
        ShipRenderer(WORLD, path)


def test_a_bake_from_before_relief_explains_the_fix(tmp_path):
    names = texture_names()
    levels = {f"level{i}": np.zeros((len(names), 16 >> i, 16 >> i, 4), np.uint8) for i in range(3)}
    path = tmp_path / "older.npz"
    np.savez(path, names=np.array(names), **levels)
    with pytest.raises(FileNotFoundError, match="tools.bake_ship_textures"):
        ShipRenderer(WORLD, path)


def test_a_wider_view_shows_more_and_changes_no_surface(textures):
    wide = ShipRenderer(WORLD, textures, size=SIZE, fov=FIELD_OF_VIEW)
    narrow = ShipRenderer(WORLD, textures, size=SIZE, fov=75)
    assert wide.focal_length < narrow.focal_length
    assert (wide.mesh["vertices"] == narrow.mesh["vertices"]).all()


def test_a_texture_on_a_wall_reads_along_and_up_the_wall():
    mesh = Mesh({"plate": 0})
    Wall(mesh, 1, 0, 5, 0, 2, WALL_TOP, ZONES["S"], 0).flat(0, 2, 0, 2, "plate")
    right, up, out = np.reshape(mesh.frames[0], (3, 3))
    assert right == pytest.approx((0, 1, 0)) and up == pytest.approx((0, 0, 1))
    assert out == pytest.approx((-1, 0, 0))


def test_a_turned_texture_turns_its_relief_with_it():
    mesh = Mesh({"plate": 0})
    Wall(mesh, 1, 0, 5, 0, 2, WALL_TOP, ZONES["S"], 0).flat(0, 2, 0, 2, "plate", turned=True)
    right, up, _ = np.reshape(mesh.frames[0], (3, 3))
    assert abs(right[2]) == pytest.approx(1) and abs(up[1]) == pytest.approx(1)


def test_every_surface_has_an_upright_frame(mesh):
    right, up, out = np.split(mesh["frames"], 3, axis=1)
    assert np.isclose(np.linalg.norm(out, axis=1), 1, atol=1e-3).all()
    assert np.isclose(np.linalg.norm(right, axis=1), 1, atol=1e-3).all()
    assert np.abs((right * out).sum(axis=1)).max() < 1e-3
    assert np.abs((up * out).sum(axis=1)).max() < 1e-3


def ridge(width=64):
    """A raised bar across a flat plate, and the normal a height map implies for it."""
    rows = np.arange(width, dtype=np.float32)
    height = 200.0 * np.exp(-(((rows - width / 2) / 5) ** 2))
    height = height[:, None] * height[None, :] / 200.0
    slope_y, slope_x = np.gradient(height)
    normal = np.dstack((-slope_x / 40, slope_y / 40, np.ones_like(height)))
    return height, normal


@pytest.mark.parametrize("across, up", [(1.0, 1.0), (-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0)])
def test_a_normal_map_is_read_the_way_its_height_map_runs(across, up):
    height, normal = ridge()
    flipped = normal * np.array((across, up, 1.0), np.float32)
    assert facing_axes(flipped, height) == (across, up)


def test_a_normal_map_with_nothing_to_check_keeps_the_packs_usual_axes():
    assert facing_axes(ridge()[1], None) == (-1.0, -1.0)
    assert facing_axes(ridge()[1], np.zeros((64, 64), np.float32)) == (-1.0, -1.0)


def test_relief_changes_how_a_surface_is_lit(renderer, snapshot):
    lit = picture(renderer, snapshot)
    relief = renderer.reliefs.copy()
    renderer.reliefs[:] = 0xFF00FFFF  # every normal leaning one way
    try:
        assert picture(renderer, snapshot) != lit
    finally:
        renderer.reliefs[:] = relief


def test_lamp_direction_leads_back_to_the_lamp(theme):
    x, y, *_ = next(
        light for light in theme.lights if theme.is_open(int(light[0]) + 2, int(light[1]))
    )
    grid = int((x + 1.5) * LIGHT_GRID), int(y * LIGHT_GRID)
    across, _, lamp = theme.aim[grid[1], grid[0]]
    assert across < 0 and LAMP_LOW <= lamp <= LAMP_HIGH


def test_every_corridor_line_is_the_same_blue_guide_line(mesh):
    lines = mesh["textures"] == texture_names().index("guide_line")
    assert lines.sum() > 40
    assert not [name for name in texture_names() if name.startswith("stripe")]


def test_a_long_wall_never_reads_past_the_fog_table(textures, snapshot):
    camera = replace(snapshot, survivor=SurvivorState(20.5, 37.5, -pi / 2 + 0.04))
    seen = ShipRenderer(WORLD, textures, size=(1067, 600))
    calm = pygame.surfarray.array3d(seen.render(camera))
    entries = len(seen.fog)
    # Numba checks no bounds: whatever follows the table in memory lands on the screen.
    seen.fog = np.concatenate((seen.fog, np.full(entries, np.nan, np.float32)))[:entries]
    frame = pygame.surfarray.array3d(seen.render(camera))
    assert (seen.inverse_depth[seen.ids >= 0] * entries / FOG_STEPS < 1).any()
    assert (frame == calm).all()


def test_the_game_runs_without_the_mood_bake(textures, tmp_path, snapshot):
    plain = ShipRenderer(WORLD, textures, size=(81, 47), mood_path=tmp_path / "none.npz")
    frame = plain.render(replace(snapshot, survivor=SurvivorState(19.5, 8.0, -pi / 2)))
    assert pygame.surfarray.array3d(frame).mean() > 0


@pytest.mark.parametrize("size", [(81, 47), (83, 50), (161, 91)])
def test_sizes_that_do_not_divide_into_the_air_blocks_still_draw(textures, snapshot, size):
    seen = ShipRenderer(WORLD, textures, size=size)
    close = AlienState(19.5, 7.0, pi / 2)
    camera = replace(snapshot, survivor=SurvivorState(19.5, 8.0, -pi / 2), alien=close)
    assert seen.render(camera).get_size() == size


def test_the_same_snapshot_draws_the_same_frame(renderer, snapshot):
    camera = replace(snapshot, survivor=SurvivorState(19.5, 8.0, -pi / 2), tick=77)
    first = pygame.surfarray.array3d(renderer.render(camera)).copy()
    renderer.render(replace(camera, tick=500))
    assert (pygame.surfarray.array3d(renderer.render(camera)) == first).all()


def test_lamps_keep_their_zones_color_so_each_district_has_its_own_light():
    red = ZONES["G"].color
    kept = muted(red)
    assert (max(red) - min(red)) / 2 < max(kept) - min(kept) < max(red) - min(red)
    assert kept.index(max(kept)) == red.index(max(red))


def fittings(mesh):
    made = [texture_names().index(name) for name in ("pipe", "pipe_hot", "bracket", "cable")]
    return np.isin(mesh["textures"], made)


def test_fittings_hang_above_the_alien_so_they_never_hide_it(mesh):
    assert fittings(mesh).sum() > 300
    assert mesh["vertices"][fittings(mesh)][:, :, 2].min() >= OVERHEAD_FLOOR
    assert OVERHEAD_FLOOR > 1.85  # the top of the Alien sprite


def hanging_in_open_floor(mesh, theme):
    """Door headers and the low crawlway ceiling belong where they are; nothing else may hang."""
    corners = mesh["vertices"]
    centres = corners.mean(axis=1)
    doorways = np.array(WORLD.doors, np.float32) + 0.5

    def open_floor(x, y):
        beside_door = (np.abs(doorways - (x, y)) <= 0.7).all(axis=1).any()
        if beside_door or WORLD.overlaps(float(x), float(y), 0.3):
            return False
        around = ((0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1))
        return all(theme.zone_at(int(x + dx), int(y + dy)) != "N" for dx, dy in around)

    clear = np.array([open_floor(x, y) for x, y, _ in centres])
    lowest, highest = corners[:, :, 2].min(axis=1), corners[:, :, 2].max(axis=1)
    return centres[clear & (highest > 0.05) & (lowest < OVERHEAD_FLOOR)]


def test_nothing_hangs_where_it_could_hide_the_alien_in_open_floor(mesh, theme):
    assert len(hanging_in_open_floor(mesh, theme)) == 0


def test_a_panel_hanging_low_in_a_corridor_would_be_noticed(mesh, theme):
    low = mesh["vertices"].copy()
    panel = (7.1, 16.2, 1.25), (8.9, 16.5, 1.55), (8.9, 17.3, WALL_TOP), (7.1, 17.0, WALL_TOP)
    low[0] = panel
    assert len(hanging_in_open_floor({**mesh, "vertices": low}, theme)) == 1


def test_signs_stand_clear_of_door_frames_and_inside_collision():
    assert max(DOOR_COLUMN_PROUD, DOOR_HEADER_PROUD) < SIGN_PROUD < CoreConfig().radius


def test_fittings_stay_inside_where_collision_stops_the_player():
    reach = PIPE_OUT + PIPE_RADIUS * COLLAR_SCALE
    assert reach < CoreConfig().radius


def test_room_detail_never_depends_on_the_round(theme):
    core = GameCore(WORLD)
    first = core.reset(1).world_map
    other = next(
        map_ for map_ in (core.reset(seed).world_map for seed in range(2, 99)) if map_ != first
    )
    built = [build_mesh(ShipTheme(world), texture_names()) for world in (first, other)]
    assert (built[0]["vertices"] == built[1]["vertices"]).all()
    assert (built[0]["textures"] == built[1]["textures"]).all()


def test_sparks_come_from_dead_lights_only_and_replay(renderer, theme):
    assert theme.sparks
    assert all(not theme.light_cells[(int(x), int(y))] for x, y in theme.sparks)
    bursts = [renderer._sparks(tick) for tick in range(0, 3000, 12)]
    assert any(bursts) and not all(bursts)
    assert renderer._sparks(600) == renderer._sparks(600)


def test_a_spark_right_in_front_of_the_player_is_not_drawn(renderer, snapshot):
    tick = next(t for t in range(1, 5000) if renderer._sparks(t))
    x, y = renderer._sparks(tick)[0]
    close = replace(snapshot, survivor=SurvivorState(x, y - 0.5, pi / 2), tick=tick)
    far = replace(snapshot, survivor=SurvivorState(x, y - 4.0, pi / 2), tick=tick)
    spark = f"spark{tick // 2 % 3}"
    assert all(item[2] != spark for item in renderer._visible_sprites(close, 0.0, 1.0))
    assert any(item[2] == spark for item in renderer._visible_sprites(far, 0.0, 1.0))
