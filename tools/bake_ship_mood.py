"""Bake the ship's horror mood: lens dirt, drifting mist and the color grade.

Usage: python -m tools.bake_ship_mood

Everything downloads once into assets/mood/source/ and the result goes to
assets/mood/ship_mood.npz. Both stay local, because the grade's license forbids passing its
files on. Credit: lens dirt by zisongbr (CC0), smoke by Kenney (CC0),
color grade from the Color Grading LUT Pack by REEBOO Labs.
"""

import io
import json
import re
import zipfile
from http.cookiejar import CookieJar
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

import numpy as np
from PIL import Image

from tools.asset_paths import ROOT

SOURCE = ROOT / "assets" / "mood" / "source"
OUTPUT = ROOT / "assets" / "mood" / "ship_mood.npz"
AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
OPEN_GAME_ART = "https://opengameart.org/sites/default/files"
LENS_DIRT = f"{OPEN_GAME_ART}/oga-textures/169113/5.png"
SMOKE = f"{OPEN_GAME_ART}/smokeParticleAssets.zip"
LUT_PAGE = "https://reeboolabs.itch.io/color-grading-lut-pack-free-sample"
LOOK = "horror_green"
LOOK_STRENGTH = 0.4  # the rest stays the ship's own color, or everything turns the same green
GRADE_SIZE = 128
DIRT_SIZE = 400, 225
MIST_SIZE = 256
MIST_PUFFS = 90


def fetch(url):
    request = Request(url, headers={"User-Agent": AGENT})
    with urlopen(request, timeout=120) as response:
        return response.read()


def kept(name, url):
    path = SOURCE / name
    if not path.exists():
        data = fetch(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path


def unpack(data, folder, suffixes):
    """Keep only the named kinds of file, and never write outside the folder."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for item in archive.infolist():
            name = item.filename.replace("\\", "/").split("/")[-1]
            if not item.is_dir() and name.lower().endswith(suffixes):
                target = folder / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(item))


def smoke_puffs():
    folder = SOURCE / "smoke"
    if not any(folder.glob("whitePuff*.png")):
        unpack(fetch(SMOKE), folder, (".png",))
    return sorted(folder.glob("whitePuff*.png"))


def look_file():
    """itch.io hands out a short-lived link to anyone who asks from the page."""
    path = SOURCE / "grade" / f"{LOOK}.cube"
    if not path.exists():
        opener = build_opener(HTTPCookieProcessor(CookieJar()))
        opener.addheaders = [("User-Agent", AGENT)]
        page = opener.open(LUT_PAGE, timeout=60).read().decode("utf-8", "replace")
        token = re.search(r'name="csrf_token" value="([^"]+)"', page).group(1)
        upload = re.search(r'data-upload_id="(\d+)"', page).group(1)
        form = urlencode({"csrf_token": token}).encode()
        link = f"{LUT_PAGE}/file/{upload}?source=view_game&as_props=1&after_download_lightbox=true"
        url = json.loads(opener.open(link, data=form, timeout=60).read())["url"]
        unpack(opener.open(url, timeout=120).read(), SOURCE / "grade", (".cube", "license.txt"))
    return path


def read_cube(path):
    """A .cube table as [blue, green, red, channel]: red changes fastest in the file."""
    size, values = None, []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "LUT_3D_SIZE":
            size = int(parts[1])
        elif len(parts) == 3 and parts[0][0] in "0123456789.-":
            values.append([float(part) for part in parts])
    return np.array(values, np.float32).reshape(size, size, size, 3)


def graded_table(cube, strength=LOOK_STRENGTH, size=GRADE_SIZE):
    """The look at every [red, green, blue] step of the frame, as display bytes."""
    last = cube.shape[0] - 1
    steps = (np.arange(size, dtype=np.float32) + 0.5) / size
    place = steps * last
    low = np.minimum(place.astype(np.int32), last - 1)
    share = place - low
    red, green, blue = np.meshgrid(np.arange(size), np.arange(size), np.arange(size), indexing="ij")
    result = np.zeros((size, size, size, 3), np.float32)
    for red_side, red_part in ((0, 1 - share), (1, share)):
        for green_side, green_part in ((0, 1 - share), (1, share)):
            for blue_side, blue_part in ((0, 1 - share), (1, share)):
                corner = cube[
                    low[blue] + blue_side, low[green] + green_side, low[red] + red_side
                ]  # fmt: skip
                weight = red_part[red] * green_part[green] * blue_part[blue]
                result += corner * weight[..., None]
    plain = np.stack((steps[red], steps[green], steps[blue]), axis=-1)
    mixed = plain + (result - plain) * strength
    return (np.clip(mixed, 0, 1) * 255 + 0.5).astype(np.uint8)


def lens_dirt(path):
    with Image.open(path) as image:
        picture = image.convert("RGB").resize(DIRT_SIZE, Image.Resampling.LANCZOS)
    dirt = np.asarray(picture, np.float32).mean(axis=2) / 255
    return dirt / max(float(dirt.max()), 1e-3)


def mist_tile(puffs, seed=20261102):
    """Soft smoke puffs scattered over a tile that wraps at its edges, to drift across the ship."""
    rng = np.random.default_rng(seed)
    tile = np.zeros((MIST_SIZE, MIST_SIZE), np.float32)
    for _ in range(MIST_PUFFS):
        reach = int(rng.integers(MIST_SIZE // 5, MIST_SIZE // 2))
        with Image.open(puffs[int(rng.integers(len(puffs)))]) as image:
            alpha = image.convert("RGBA").getchannel("A")
            puff = alpha.resize((reach, reach), Image.Resampling.BILINEAR)
        puff = puff.rotate(float(rng.uniform(0, 360)))
        amount = np.asarray(puff, np.float32) / 255 * float(rng.uniform(0.3, 1.0))
        top, left = rng.integers(0, MIST_SIZE, 2)
        rows = (np.arange(reach) + top) % MIST_SIZE
        columns = (np.arange(reach) + left) % MIST_SIZE
        tile[np.ix_(rows, columns)] += amount
    tile = (tile - tile.min()) / (tile.max() - tile.min())
    return tile / tile.mean() * 0.5


def bake():
    return {
        "dirt": lens_dirt(kept("lens_dirt.png", LENS_DIRT)),
        "mist": mist_tile(smoke_puffs()),
        "grade": graded_table(read_cube(look_file())),
    }


def main():
    mood = bake()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT, **mood)
    print(f"Baked the ship's mood with the look {LOOK}: {OUTPUT}")


if __name__ == "__main__":
    main()
