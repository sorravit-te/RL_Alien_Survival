"""Bake the Alien's frames from a free CC0 model (3D Horror Game Monster, opengameart.org).

Usage:
    python -m tools.bake_alien_sprites fetch
    blender --background --factory-startup --python tools/render_alien_frames.py --
        assets/alien/source/Poses assets/alien/source/Colors/Green assets/alien/frames
    python -m tools.bake_alien_sprites pack

The model downloads into assets/alien/source/ and the packed frames go to
assets/alien/prepared/, both local like the baked textures. Blender is a separate download
and not a project dependency; the game only needs the packed file.
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from renderers.alien_sprites import FRAMES, POSES
from tools.prepare_assets import ROOT

MODEL_FILES = "https://opengameart.org/sites/default/files/"
ARCHIVES = ("Poses.zip", "Colors.zip")
SOURCE = ROOT / "assets" / "alien" / "source"
RENDERS = ROOT / "assets" / "alien" / "frames"
KEEP = (".fbx", ".png", ".txt")
# The ship's torch adds to whatever is drawn, so a plainly lit render washes out up close.
SKIN_GAIN, SKIN_GAMMA = 0.42, 1.6


def fetch(folder=SOURCE):
    folder.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVES:
        archive = folder / name
        if not archive.exists():
            request = Request(MODEL_FILES + name, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=600) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        with zipfile.ZipFile(archive) as packed:
            members = [m for m in packed.namelist() if "__MACOSX" not in m and m.endswith(KEEP)]
            packed.extractall(folder, members)
    print(f"Model ready in {folder}")


def darkened(rgba):
    """Deep shadows and wet highlights; the outline stays exactly as rendered."""
    rgba = rgba.copy()
    shade = SKIN_GAIN * (rgba[..., :3] / 255.0) ** SKIN_GAMMA
    rgba[..., :3] = (shade * 255).astype(np.uint8)
    return rgba


def pack(renders=RENDERS, output=FRAMES):
    """Stack the renders into one array per pose: frames, views, height, width, rgba."""
    sheet = json.loads((Path(renders) / "frames.json").read_text("utf-8"))
    arrays = {}
    for pose in POSES:
        arrays[pose] = darkened(
            np.stack(
                [
                    np.stack(
                        [
                            np.asarray(
                                Image.open(Path(renders) / f"{pose}_{frame}_{view}.png").convert("RGBA")
                            )
                            for view in range(sheet["views"])
                        ]
                    )
                    for frame in range(sheet["frames"][pose])
                ]
            )
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, world_size=sheet["world_size"], **arrays)
    print(f"Packed {sum(len(a) for a in arrays.values())} poses into {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("fetch", "pack"))
    {"fetch": fetch, "pack": pack}[parser.parse_args().step]()


if __name__ == "__main__":
    main()
