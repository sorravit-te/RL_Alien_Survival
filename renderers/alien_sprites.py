"""The Alien as pre-rendered frames: eight views of each pose, picked from read-only state.

The ship's renderer draws sprites, not characters, so a 3D model is rendered ahead of time.
Like the rest of the picture, the choice of frame depends only on the snapshot, so a frame
replays exactly. Without the baked file the ship keeps its plain silhouette.
"""

from math import atan2, pi
from pathlib import Path

import numpy as np

from game.interaction import angle_difference

FRAMES = Path(__file__).resolve().parents[1] / "assets" / "alien" / "prepared" / "alien_sprites.npz"
POSES = ("idle", "walk", "run")
CYCLE_CELLS = 3.4  # one full cycle is two footfalls; app/audio.py hears one every 1.7 cells
IDLE_CYCLE_SECONDS = 4.0


def view_index(alien, camera, views):
    """Which view faces the camera: 0 is the Alien's front, rising toward its right side."""
    toward = atan2(camera.y - alien.y, camera.x - alien.x)
    return round(angle_difference(toward, alien.yaw) / (2 * pi / views)) % views


def pose_now(snapshot):
    """The pose and how many cycles of it have passed; standing still has its own slow cycle."""
    seconds = snapshot.tick / snapshot.config.tick_hz
    if not snapshot.navigation_path:
        return "idle", seconds / IDLE_CYCLE_SECONDS
    if snapshot.alien.speed_mode == "chase":
        return "run", seconds * snapshot.config.alien_chase_speed / CYCLE_CELLS
    return "walk", seconds * snapshot.config.alien_search_speed / CYCLE_CELLS


class AlienSprites:
    def __init__(self, path=FRAMES):
        self.poses = None
        path = Path(path)
        if path.is_file():
            with np.load(path) as baked:
                self.poses = {name: baked[name] for name in POSES}  # frames, views, h, w, rgba
                self.world_size = float(baked["world_size"])
            self.no_glow = np.zeros(self.poses["idle"].shape[2:4], np.uint8)

    def sprite(self, snapshot):
        """A ShipRenderer sprite for this tick, or None when nothing was baked."""
        if self.poses is None or snapshot.alien is None:
            return None
        pose, cycles = pose_now(snapshot)
        frames = self.poses[pose]
        view = view_index(snapshot.alien, snapshot.survivor, frames.shape[1])
        frame = frames[int(cycles % 1 * len(frames)), view]
        return (frame, self.no_glow), self.world_size, self.world_size, 0.0
