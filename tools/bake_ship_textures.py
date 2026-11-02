"""Bake every purchased Materials texture for the 3D ship deck.

Usage: python -m tools.bake_ship_textures

Each texture becomes a color layer with its glow, and a relief layer with its normal and
shine, because the renderer lights every pixel itself. Blood, claw marks, signs and the
starfield are made here and painted over copies. Output stays in the ignored
assets/stradex/prepared/ folder.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from tools.prepare_assets import ROOT, resolve_source_root

SIZE = 512
OUTPUT = ROOT / "assets" / "stradex" / "prepared" / "ship_textures.npz"
EDGE = 4.0  # height change per texel that counts as a real edge when reading a normal map
TRUSTED = 0.1  # weaker agreement than this leaves a map to the pack's usual axes
MATERIALS = (
    *(f"wall{i}" for i in range(1, 13)),
    *(f"walltrim{i}" for i in range(1, 7)),
    *(f"trim{i}" for i in range(1, 6)),
    *(f"ceiling{i}" for i in range(1, 8)),
    *(f"ceiling_trim{i}" for i in range(1, 6)),
    *(f"floor{i}" for i in range(1, 6)),
    "floor_trim1",
    "floor_trim2",
    "floor_trim3",
    "floortrim4",
    *(f"doorborder{i}" for i in range(1, 5)),
    "door1",
    "columna1",
    "columna2",
    "other1",
    "other2",
    "rack1",
)
BLOODIED = ("wall10", "wall6", "wall1", "wall4", "wall9", "floor1", "floor2", "floor5")
CLAWED = ("wall10", "wall6", "wall1", "wall4", "other2")
SIGNS = {
    **{name.lower(): name for name in ("BRIDGE", "LAB", "CREW", "MESS", "MEDBAY", "PODS", "CRYO")},
    **{name.lower(): name for name in ("CARGO", "REACTOR", "SHUTTLE", "PLAZA")},
    "command": "COMMAND RING",
    "crewring": "CREW RING",
    "medical": "MEDICAL RING",
    "engine": "ENGINE RING",
}
GUIDE_LINE = (0.3, 0.5, 1.0)  # one color all over the ship, to follow through the dark
LIT_COPIES = ("ceiling3_lit", "ceiling4_lit")
MADE = ("ceiling2_lit", "window_glass", "hazard", "door1_seam")
SEAM_HALF = 5  # texels either side of a door's middle
FITTINGS = ("pipe", "pipe_hot", "bracket", "cable", "pane_seal")


def texture_names():
    names = [*MATERIALS, *LIT_COPIES, "guide_line", *MADE, *FITTINGS]
    names += [f"{name}_blood" for name in BLOODIED]
    names += [f"{name}_clawed" for name in CLAWED]
    names += [f"sign_{key}" for key in SIGNS]
    return sorted(names)


def load(directory, name, suffix, mode):
    path = directory / f"{name}_{suffix}.png"
    if not path.is_file():
        return None
    with Image.open(path) as image:
        return image.convert(mode).resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def flat_relief():
    relief = np.zeros((SIZE, SIZE, 4), np.float32)
    relief[..., 2] = 1.0
    return relief


def facing_axes(normal, height):
    """Which way a map's axes run, measured from its height map.

    Most of the pack points its axes left and down, a few differ, and read the wrong way a
    groove lights like a ridge. Height falling to the right tilts a surface right.
    """
    if height is None:
        return -1.0, -1.0
    slope_y, slope_x = np.gradient(np.asarray(height, np.float32))
    edges = (np.abs(slope_x) + np.abs(slope_y)) > EDGE
    axes = [-1.0, -1.0]
    if edges.sum() > 200:
        for axis, (values, slope, sign) in enumerate(
            ((normal[..., 0], slope_x, -1.0), (normal[..., 1], slope_y, 1.0))
        ):
            if values[edges].std() > 1e-3 and slope[edges].std() > 1e-3:
                agreement = float(np.corrcoef(values[edges], slope[edges])[0, 1]) * sign
                if abs(agreement) > TRUSTED:
                    axes[axis] = 1.0 if agreement > 0 else -1.0
    return axes[0], axes[1]


def relief_of(directory, name, height):
    """A unit normal running right, up and out of the texture, and how hard it shines."""
    relief = flat_relief()
    normal = load(directory, name, "n", "RGB") or load(directory, name, "local", "RGB")
    if normal is not None:
        n = np.asarray(normal, np.float32) / 127.5 - 1
        across, up = facing_axes(n, height)
        n = n * np.array((across, up, 1.0), np.float32)
        relief[..., :3] = n / np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-5)
    specular = load(directory, name, "s", "L")
    if specular is not None:
        relief[..., 3] = np.asarray(specular, np.float32) / 255
    return relief


def material(directory, name):
    """Return float RGB in 0..1, a glow mask and the relief the renderer lights."""
    color = load(directory, name, "d", "RGB")
    if color is None:
        raise FileNotFoundError(f"Missing material: {name}_d.png")
    rgb = np.asarray(color, np.float32) / 255
    height = load(directory, name, "h", "L")
    relief = relief_of(directory, name, height)
    if height is not None:
        soft = height.filter(ImageFilter.GaussianBlur(4))
        crevice = np.asarray(soft, np.float32) - np.asarray(height, np.float32)
        rgb = rgb * (1 - np.clip(crevice / 255 * 1.6, 0, 0.35))[..., None]
    glow = np.zeros(rgb.shape[:2], np.float32)
    add = load(directory, name, "add", "RGB")
    if add is not None:
        emit = np.asarray(add, np.float32) / 255
        strength = emit.max(axis=2)
        glow = np.clip(strength * 4, 0, 1)
        peak = np.maximum(strength, 1e-3)[..., None]
        rgb = np.where(strength[..., None] > 0.02, np.clip(emit / peak, 0, 1), rgb)
    return rgb, glow, relief


def smooth_noise(rng, cells):
    coarse = (rng.random((cells, cells), dtype=np.float32) * 255).astype(np.uint8)
    image = Image.fromarray(coarse).resize((SIZE, SIZE), Image.Resampling.BICUBIC)
    return np.asarray(image, np.float32) / 255


def weather(rgb, rng):
    """Faint grime and thin rust runs, so no surface looks freshly printed."""
    grime = 0.94 + 0.1 * smooth_noise(rng, 14)
    runs = np.zeros(SIZE, np.float32)
    for x in rng.integers(0, SIZE, 30):
        runs[x : x + int(rng.integers(1, 4))] = float(rng.uniform(0.25, 0.8))
    reach = 1 - np.arange(SIZE, dtype=np.float32)[:, None] / (SIZE * rng.uniform(0.5, 1.2))
    streak = runs[None, :] * np.clip(reach, 0, 1)
    return rgb * (grime * (1 - streak * 0.18))[..., None]


def font(size):
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def plate(base, text, rng, color):
    """A sign is shown four times wider than tall, so its letters are painted squeezed."""
    wide = Image.new("L", (SIZE * 4, SIZE), 0)
    draw = ImageDraw.Draw(wide)
    letters = font(250)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=letters)
    scale = min(1.0, (SIZE * 3.8) / (right - left))
    x = (SIZE * 4 - (right - left)) / 2 - left
    y = (SIZE - (bottom - top)) / 2 - top
    draw.text((x, y), text, fill=255, font=letters)
    if scale < 1:
        wide = wide.resize((int(wide.width * scale), SIZE))
        padded = Image.new("L", (SIZE * 4, SIZE), 0)
        padded.paste(wide, ((SIZE * 4 - wide.width) // 2, 0))
        wide = padded
    alpha = np.asarray(wide.resize((SIZE, SIZE)), np.float32) / 255
    alpha = alpha * np.clip(smooth_noise(rng, 40) * 2.2, 0, 1) * 0.9
    paint = np.array(color, np.float32) / 255
    return base * 0.5 * (1 - alpha[..., None]) + paint * alpha[..., None]


def scratches(rgb, rng):
    """Four tearing claw marks: thin bright metal inside a wide dark gouge."""
    gouge = Image.new("L", (SIZE, SIZE), 0)
    metal = Image.new("L", (SIZE, SIZE), 0)
    x = int(rng.integers(40, SIZE - 190))
    y = int(rng.integers(50, SIZE - 250))
    slant = int(rng.integers(30, 90))
    for line in range(4):
        start = (x + line * 34 + int(rng.integers(-6, 6)), y + int(rng.integers(0, 40)))
        length = int(rng.integers(130, 230))
        points = [
            (start[0] + slant * t / 8 + float(rng.normal(0, 1.6)), start[1] + length * t / 8)
            for t in range(9)
        ]
        ImageDraw.Draw(gouge).line(points, fill=255, width=7)
        ImageDraw.Draw(metal).line(points[1:-1], fill=170, width=2)
    dark = np.asarray(gouge.filter(ImageFilter.GaussianBlur(3)), np.float32) / 255
    bright = np.asarray(metal.filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255
    return rgb * (1 - dark[..., None] * 0.85) + bright[..., None] * np.array(
        [0.3, 0.29, 0.28], np.float32
    )


def blood(rgb, rng, *, drips):
    """Irregular spatter thinning out from a smeared centre, dried almost black."""
    mask = Image.new("L", (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(mask)
    cx = int(rng.integers(SIZE // 5, 4 * SIZE // 5))
    cy = int(rng.integers(SIZE // 5, SIZE // 2))
    direction = float(rng.uniform(-0.6, 0.6))
    for _ in range(420):
        spread = float(rng.exponential(52))
        angle = float(rng.normal(direction, 1.1)) + (3.14159 if rng.random() < 0.3 else 0)
        radius = float(rng.uniform(0.8, 3.2)) * (1 + 12 / (spread + 6))
        x = cx + np.cos(angle) * spread * 1.6
        y = cy + np.sin(angle) * spread * 0.8
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    if drips:
        for _ in range(14):
            x = cx + float(rng.normal(0, 60))
            length = float(rng.exponential(70)) + 20
            draw.line((x, cy + float(rng.normal(0, 18)), x, cy + length), fill=230, width=2)
    alpha = np.asarray(mask.filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255
    alpha = alpha * (0.45 + 0.55 * smooth_noise(rng, 34))
    dried = np.array([0.16, 0.018, 0.016], np.float32)
    return rgb * (1 - alpha[..., None] * 0.94) + dried * alpha[..., None] * 0.94


def glass_marks(rng):
    """Dust, scratches, corner grime and two soft streaks: what shows a pane is really there."""
    dust = np.clip(smooth_noise(rng, 12) - 0.5, 0, 1) * 0.16
    grit = (rng.random((SIZE, SIZE), dtype=np.float32) > 0.9985) * 0.35
    lines = Image.new("L", (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(lines)
    for _ in range(14):
        x, y = rng.integers(0, SIZE, 2)
        turn, length = float(rng.normal(0.5, 0.5)), float(rng.uniform(20, 90))
        end = (x + np.cos(turn) * length, y + np.sin(turn) * length)
        draw.line((int(x), int(y), *end), fill=int(rng.integers(30, 80)), width=1)
    scratches = np.asarray(lines.filter(ImageFilter.GaussianBlur(0.6)), np.float32) / 255
    places = np.arange(SIZE, dtype=np.float32) / SIZE
    from_edge = np.minimum(places, 1 - places)
    edge = np.minimum(from_edge[None, :], from_edge[:, None])
    grime = np.clip(1 - edge / 0.07, 0, 1) ** 2 * (0.25 + 0.35 * smooth_noise(rng, 30))
    slant = places[None, :] + places[:, None] * 0.6
    streaks = 0.2 * np.exp(-(((slant - 0.62) / 0.09) ** 2))
    streaks += 0.15 * np.exp(-(((slant - 0.86) / 0.03) ** 2))
    marks = np.clip(dust + grit + scratches + grime + streaks, 0, 1)
    return np.ones((SIZE, SIZE, 3), np.float32) * np.array((0.72, 0.76, 0.8), np.float32), marks


def lit_panel(rgb):
    """The plain ceiling plate doubles as a light fitting; its middle becomes the lamp."""
    mask = np.zeros((SIZE, SIZE), np.float32)
    inset = SIZE // 9
    mask[inset:-inset, inset:-inset] = 1
    lamp = np.array([0.78, 0.84, 0.88], np.float32)
    return rgb * (1 - mask[..., None]) + lamp * mask[..., None], mask * 0.42


def leaf_seam(rgb):
    """Two leaves meet down the middle; painted, the line stays steady as the view moves."""
    seam = rgb.copy()
    middle = SIZE // 2
    seam[:, middle - SEAM_HALF : middle + SEAM_HALF] *= 0.1
    return seam


def hazard_stripes():
    """Yellow and black diagonals for dangerous thresholds."""
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    band = ((xs + ys) // 64) % 2 == 0
    yellow = np.array([0.8, 0.62, 0.08], np.float32)
    black = np.array([0.04, 0.04, 0.04], np.float32)
    return np.where(band[..., None], yellow, black).astype(np.float32)


def stripe_rail(color):
    """A lit strip with dark edges; the renderer shows it thin along a wall."""
    rgb = np.ones((SIZE, SIZE, 3), np.float32) * np.array(color, np.float32)
    glow = np.ones((SIZE, SIZE), np.float32) * 0.2
    edge = SIZE // 8
    rgb[:edge] = rgb[-edge:] = 0.05
    glow[:edge] = glow[-edge:] = 0.0
    return rgb, glow


def pipe_metal(color, rng):
    """Rounded across its width, with a joint once per cell along its length."""
    across = np.sin(np.linspace(0.25, 2.9, SIZE, dtype=np.float32))[:, None] ** 2
    rgb = np.ones((SIZE, SIZE, 3), np.float32) * np.array(color, np.float32)
    rgb = rgb * (0.3 + 0.9 * across)[..., None]
    rgb[:, :18] *= 0.35
    rgb[:, 18:30] *= 1.25
    return weather(rgb, rng)


def plain_metal(level, rng):
    return weather(np.full((SIZE, SIZE, 3), level, np.float32), rng)


def add_fittings(textures, reliefs, rng):
    """Self-made, drawn last so the rest of the bake keeps its random stream."""
    no_glow = np.zeros((SIZE, SIZE), np.float32)
    textures["pipe"] = (pipe_metal((0.42, 0.45, 0.47), rng), no_glow)
    textures["pipe_hot"] = (pipe_metal((0.5, 0.33, 0.24), rng), no_glow)
    textures["bracket"] = (plain_metal(0.1, rng), no_glow)
    textures["cable"] = (plain_metal(0.05, rng), no_glow)
    textures["pane_seal"] = (plain_metal(0.035, rng), no_glow)
    shines = {"pipe": 0.55, "pipe_hot": 0.5, "bracket": 0.3, "cable": 0.2, "pane_seal": 0.1}
    for name, shine in shines.items():
        reliefs[name] = flat_relief()
        reliefs[name][..., 3] = shine


def bake(directory, seed=20261030):
    rng = np.random.default_rng(seed)
    textures, reliefs = {}, {}
    for name in MATERIALS:
        rgb, glow, reliefs[name] = material(directory, name)
        textures[name] = (weather(rgb, rng), glow)
    textures["ceiling2_lit"] = lit_panel(textures["ceiling2"][0])
    reliefs["ceiling2_lit"] = reliefs["ceiling2"]
    textures["door1_seam"] = (leaf_seam(textures["door1"][0]), textures["door1"][1])
    reliefs["door1_seam"] = reliefs["door1"]
    for name in LIT_COPIES:
        # Most vents are dead: the plain name loses its glow, the lit copy keeps it.
        base = name.removesuffix("_lit")
        textures[name] = textures[base]
        reliefs[name] = reliefs[base]
        textures[base] = (textures[base][0] * 0.6, np.zeros((SIZE, SIZE), np.float32))
    textures["window_glass"] = glass_marks(rng)
    for name in BLOODIED:
        rgb, glow = textures[name]
        textures[f"{name}_blood"] = (blood(rgb, rng, drips=name.startswith("wall")), glow)
        reliefs[f"{name}_blood"] = reliefs[name]
    for name in CLAWED:
        rgb, glow = textures[name]
        textures[f"{name}_clawed"] = (scratches(rgb, rng), glow)
        reliefs[f"{name}_clawed"] = reliefs[name]
    plain = textures["walltrim4"][0]
    no_glow = np.zeros((SIZE, SIZE), np.float32)
    for key, sign in SIGNS.items():
        textures[f"sign_{key}"] = (plate(plain, sign, rng, (222, 204, 158)), no_glow)
    textures["guide_line"] = stripe_rail(GUIDE_LINE)
    textures["hazard"] = (hazard_stripes(), no_glow)
    add_fittings(textures, reliefs, rng)
    return textures, reliefs


def pack(rgb, glow):
    return (np.dstack((np.clip(rgb, 0, 1), np.clip(glow, 0, 1))) * 255 + 0.5).astype(np.uint8)


def pack_relief(relief):
    """Axes run -1..1 and shine 0..1; the shader reads them back from the same four bytes."""
    stored = relief.copy()
    stored[..., :3] = stored[..., :3] * 0.5 + 0.5
    return (np.clip(stored, 0, 1) * 255 + 0.5).astype(np.uint8)


def half(pixels):
    """Next mip level: each texel is the mean of four, which calms distant grates."""
    return (
        pixels[:, ::2, ::2].astype(np.uint16)
        + pixels[:, 1::2, ::2]
        + pixels[:, ::2, 1::2]
        + pixels[:, 1::2, 1::2]
    ) // 4


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    source_root = resolve_source_root(args.source_root)
    directory = source_root / "Materials"
    if not directory.is_dir():
        raise FileNotFoundError(f"Required Stradex folder is missing: {directory}")
    textures, reliefs = bake(directory)
    names = texture_names()
    levels = {"level0": np.stack([pack(*textures[name]) for name in names])}
    levels["relief0"] = np.stack(
        [pack_relief(reliefs[name] if name in reliefs else flat_relief()) for name in names]
    )
    for kind in ("level", "relief"):
        levels[f"{kind}1"] = half(levels[f"{kind}0"]).astype(np.uint8)
        levels[f"{kind}2"] = half(levels[f"{kind}1"]).astype(np.uint8)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT, names=np.array(names), **levels)
    manifest = {
        "creator": "Stradex / Maximiliano Ruben Viamonte",
        "source_url": "https://stradex.itch.io/scifi-tech-hall-texture-pack",
        "use": "Local project assets. Do not redistribute as standalone artwork.",
        "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "materials": list(MATERIALS),
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Baked {len(names)} textures from {len(MATERIALS)} materials: {OUTPUT}")


if __name__ == "__main__":
    main()
