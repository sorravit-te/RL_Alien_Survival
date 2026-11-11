"""Choosing the Alien's frame: the right view, the right pose, and the same frame on replay."""

import json
import os
from dataclasses import replace
from math import pi

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pytest
from PIL import Image

from game.core import GameCore
from game.map import WorldMap, scenario_map
from game.ship_deck import SHIP_SCENARIO
from game.state import AlienState, CoreConfig, SurvivorState
from renderers.alien_sprites import IDLE_CYCLE_SECONDS, AlienSprites, pose_now, view_index
from renderers.ship_renderer import ShipRenderer
from tools.bake_alien_sprites import darkened, pack
from tools.bake_ship_textures import texture_names

HALL = (
    "##########",
    "#........#",
    "#........#",
    "##########",
)
ALIEN = AlienState(4.5, 1.5, 0.0)
VIEW_COLORS = [(i * 30 + 20, 200 - i * 20, 60) for i in range(8)]


def scene(*, tick=30, path=(), mode="search", camera=(7.5, 1.5)):
    world = WorldMap(HALL, (camera,))
    snapshot = GameCore(world, CoreConfig(alien_enabled=False)).snapshot()
    return replace(
        snapshot,
        tick=tick,
        alien=replace(ALIEN, speed_mode=mode),
        survivor=SurvivorState(*camera),
        navigation_path=path,
    )


@pytest.fixture(scope="module")
def baked(tmp_path_factory):
    """Each view a solid colour; frames differ by alpha so a test can tell them apart."""

    def pose(frames):
        array = np.zeros((frames, 8, 8, 8, 4), np.uint8)
        for frame in range(frames):
            for view, color in enumerate(VIEW_COLORS):
                array[frame, view, ..., :3] = color
                array[frame, view, ..., 3] = 255 - frame
        return array

    path = tmp_path_factory.mktemp("alien") / "frames.npz"
    np.savez(path, idle=pose(2), walk=pose(4), run=pose(6), world_size=2.0)
    return path


@pytest.mark.parametrize(
    ("place", "view"),
    [
        ((7.5, 1.5), 0),  # it faces +x, so a camera there sees its front
        ((4.5, 4.5), 2),  # +y is its right side
        ((1.5, 1.5), 4),
        ((4.5, -1.5), 6),
        ((7.5, 4.5), 1),
        ((7.5, -1.5), 7),
    ],
)
def test_view_follows_where_the_camera_stands(place, view):
    assert view_index(ALIEN, SurvivorState(*place), 8) == view


@pytest.mark.parametrize("yaw", [0.0, 0.7, pi, 5.0])
def test_a_camera_straight_ahead_always_sees_the_front(yaw):
    alien = replace(ALIEN, yaw=yaw)
    ahead = SurvivorState(alien.x + 3 * np.cos(yaw), alien.y + 3 * np.sin(yaw))
    assert view_index(alien, ahead, 8) == 0


def test_turning_the_alien_turns_the_view_the_other_way():
    camera = SurvivorState(7.5, 1.5)
    assert view_index(replace(ALIEN, yaw=pi / 2), camera, 8) == 6


def test_it_stands_when_it_has_nowhere_to_go():
    assert pose_now(scene())[0] == "idle"


def test_it_walks_on_patrol_and_runs_in_a_chase():
    assert pose_now(scene(path=((5.5, 1.5),)))[0] == "walk"
    assert pose_now(scene(path=((5.5, 1.5),), mode="chase"))[0] == "run"


def test_a_chase_cycles_faster_because_it_moves_faster():
    walking = pose_now(scene(path=((5.5, 1.5),)))[1]
    running = pose_now(scene(path=((5.5, 1.5),), mode="chase"))[1]
    assert running == pytest.approx(2 * walking)


def test_idle_cycle_repeats_on_its_own_clock():
    assert pose_now(scene(tick=int(30 * IDLE_CYCLE_SECONDS)))[1] == pytest.approx(1.0)


def test_nothing_baked_keeps_the_silhouette(tmp_path):
    assert AlienSprites(tmp_path / "none.npz").sprite(scene()) is None


def test_no_alien_means_no_sprite(baked):
    assert AlienSprites(baked).sprite(replace(scene(), alien=None)) is None


def test_the_frame_comes_from_the_right_view_and_moment(baked):
    art = AlienSprites(baked)
    (picture, glow), width, height, floor = art.sprite(scene(path=((5.5, 1.5),), tick=0))
    assert tuple(picture[0, 0, :3]) == VIEW_COLORS[0] and picture[0, 0, 3] == 255
    assert (width, height, floor) == (2.0, 2.0, 0.0) and not glow.any()
    later = art.sprite(scene(path=((5.5, 1.5),), tick=25))
    assert later[0][0][0, 0, 3] == 253  # 25 ticks at 2.2 cells/s is over half a 3.4 cell cycle


def test_a_frame_replays_exactly(baked):
    art = AlienSprites(baked)
    first, again = (art.sprite(scene(path=((5.5, 1.5),), tick=77)) for _ in range(2))
    assert first[0][0] is not None and np.array_equal(first[0][0], again[0][0])


def test_packed_renders_are_what_the_game_reads(tmp_path):
    frames = {"idle": 1, "walk": 2, "run": 1}
    (tmp_path / "frames.json").write_text(
        json.dumps({"world_size": 2.2, "views": 2, "frames": frames}), encoding="utf-8"
    )
    for pose, count in frames.items():
        for frame in range(count):
            for view in range(2):
                pixel = (100 + 10 * frame, 110 + 10 * view, 130, 200)
                Image.new("RGBA", (4, 6), pixel).save(tmp_path / f"{pose}_{frame}_{view}.png")
    output = tmp_path / "out" / "frames.npz"
    pack(tmp_path, output)
    art = AlienSprites(output)
    assert art.world_size == pytest.approx(2.2)
    assert art.poses["walk"].shape == (2, 2, 6, 4, 4)
    source = np.array([[[110, 120, 130, 200]]], np.uint8)
    assert np.array_equal(art.poses["walk"][1, 1, :1, :1], darkened(source)[:1, :1])


def test_darkening_keeps_the_outline_and_deepens_shadows():
    pixels = np.array([[[200, 120, 40, 77], [255, 255, 255, 255]]], np.uint8)
    dark = darkened(pixels)
    assert (dark[..., 3] == pixels[..., 3]).all() and (dark[..., :3] < pixels[..., :3]).all()
    assert dark[0, 0, 2] / dark[0, 0, 0] < pixels[0, 0, 2] / pixels[0, 0, 0]


@pytest.fixture(scope="module")
def fake_textures(tmp_path_factory):
    names = texture_names()
    rng = np.random.default_rng(1)
    path = tmp_path_factory.mktemp("ship") / "fake.npz"
    levels = {
        f"{kind}{i}": rng.integers(40, 220, (len(names), 16 >> i, 16 >> i, 4), dtype=np.uint8)
        for kind in ("level", "relief")
        for i in range(3)
    }
    np.savez(path, names=np.array(names), **levels)
    return path


def test_the_renderer_draws_the_baked_view_facing_the_camera(fake_textures, baked):
    pygame.display.init()
    pygame.display.set_mode((16, 16))
    world = scenario_map(SHIP_SCENARIO)
    renderer = ShipRenderer(world, fake_textures, size=(80, 45), alien_path=baked)
    core = GameCore(world, CoreConfig())
    x, y, _ = world.spawn_points[0]

    def centre(alien_yaw):
        snapshot = replace(
            core.snapshot(),
            survivor=SurvivorState(x, y, 0.0),
            alien=AlienState(x + 0.8, y, alien_yaw),
        )
        renderer.render(snapshot)
        return renderer.color[22, 40].copy()

    facing, turned_away = centre(pi), centre(0.0)  # yaw pi looks back at the camera
    assert facing[1] / facing[0] > turned_away[1] / turned_away[0]
