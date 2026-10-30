"""Bake every purchased Materials texture for the 3D ship deck.

Usage: python -m tools.bake_ship_textures

Normal, height, specular and emissive maps are baked into the pixels, so the renderer
only samples colors. Blood, claw marks, signs and the starfield are made here and painted
over copies. Output stays in the ignored assets/stradex/prepared/ folder.
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
LIGHT = np.array([0.0, 0.5, 0.86])  # tangent-space key light, above and in front
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
STRIPES = {  # guide rail colors: the district language of the whole ship
    "stripe_spine": (0.3, 0.9, 0.95),
    "stripe_command": (0.3, 0.5, 1.0),
    "stripe_crew": (1.0, 0.55, 0.12),
    "stripe_medical": (0.3, 1.0, 0.45),
    "stripe_engine": (1.0, 0.14, 0.06),
}
LIT_COPIES = ("ceiling3_lit", "ceiling4_lit")
WARNINGS = {"quarantine": "QUARANTINE", "seal": "SEAL THIS DECK", "listen": "IT HEARS YOU"}


def texture_names():
    names = [*MATERIALS, *LIT_COPIES, *STRIPES, "ceiling2_lit", "viewport", "hazard", "cross"]
    names += [f"{name}_blood" for name in BLOODIED]
    names += [f"{name}_clawed" for name in CLAWED]
    names += [f"sign_{key}" for key in SIGNS]
    names += [f"warning_{key}" for key in WARNINGS]
    return sorted(names)


def load(directory, name, suffix, mode):
    path = directory / f"{name}_{suffix}.png"
    if not path.is_file():
        return None
    with Image.open(path) as image:
        return image.convert(mode).resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def material(directory, name):
    """Return float RGB in 0..1 and a glow mask, with the pack's relief maps baked in."""
    color = load(directory, name, "d", "RGB")
    if color is None:
        raise FileNotFoundError(f"Missing material: {name}_d.png")
    rgb = np.asarray(color, np.float32) / 255
    normal = load(directory, name, "n", "RGB")
    if normal is None:
        normal = load(directory, name, "local", "RGB")
    if normal is not None:
        n = np.asarray(normal, np.float32) / 127.5 - 1
        n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-5)
        ndotl = np.clip(n @ LIGHT.astype(np.float32), 0, 1)
        rgb = rgb * (0.32 + 0.8 * ndotl[..., None])
        specular = load(directory, name, "s", "L")
        if specular is not None:
            half = LIGHT + np.array([0, 0, 1.0])
            half /= np.linalg.norm(half)
            shine = np.clip(n @ half.astype(np.float32), 0, 1) ** 28
            rgb = rgb + (np.asarray(specular, np.float32) / 255 * shine * 0.55)[..., None]
    height = load(directory, name, "h", "L")
    if height is not None:
        soft = height.filter(ImageFilter.GaussianBlur(4))
        crevice = np.asarray(soft, np.float32) - np.asarray(height, np.float32)
        rgb = rgb * (1 - np.clip(crevice / 255 * 2.6, 0, 0.7))[..., None]
    glow = np.zeros(rgb.shape[:2], np.float32)
    add = load(directory, name, "add", "RGB")
    if add is not None:
        emit = np.asarray(add, np.float32) / 255
        strength = emit.max(axis=2)
        glow = np.clip(strength * 4, 0, 1)
        peak = np.maximum(strength, 1e-3)[..., None]
        rgb = np.where(strength[..., None] > 0.02, np.clip(emit / peak, 0, 1), rgb)
    return rgb, glow


def smooth_noise(rng, cells):
    coarse = (rng.random((cells, cells), dtype=np.float32) * 255).astype(np.uint8)
    image = Image.fromarray(coarse).resize((SIZE, SIZE), Image.Resampling.BICUBIC)
    return np.asarray(image, np.float32) / 255


def weather(rgb, rng):
    """Soft grime and thin rust runs, so no surface looks freshly printed."""
    grime = 0.86 + 0.2 * smooth_noise(rng, 14)
    runs = np.zeros(SIZE, np.float32)
    for x in rng.integers(0, SIZE, 30):
        runs[x : x + int(rng.integers(1, 4))] = float(rng.uniform(0.25, 0.8))
    reach = 1 - np.arange(SIZE, dtype=np.float32)[:, None] / (SIZE * rng.uniform(0.5, 1.2))
    streak = runs[None, :] * np.clip(reach, 0, 1)
    shade = (grime * (1 - streak * 0.45))[..., None]
    return rgb * shade * np.array([1.0, 0.97, 0.93], np.float32)


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


def starfield(rng):
    """Sparse stars over a faint gradient; open space should feel empty, not painted."""
    fade = np.linspace(0.0, 1.0, SIZE, dtype=np.float32)[None, :, None]
    sky = fade * np.array([0.006, 0.012, 0.03], np.float32) * np.ones((SIZE, SIZE, 3), np.float32)
    for _ in range(260):
        x, y = rng.integers(0, SIZE - 2, 2)
        size = 2 if rng.random() < 0.15 else 1
        sky[y : y + size, x : x + size] = float(rng.uniform(0.25, 1.0)) * np.array(
            [0.9, 0.95, 1.0], np.float32
        )
    return sky


def lit_panel(rgb):
    """The plain ceiling plate doubles as a light fitting; its middle becomes the lamp."""
    mask = np.zeros((SIZE, SIZE), np.float32)
    inset = SIZE // 9
    mask[inset:-inset, inset:-inset] = 1
    lamp = np.array([0.9, 0.97, 1.0], np.float32)
    return rgb * (1 - mask[..., None]) + lamp * mask[..., None], mask


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
    glow = np.ones((SIZE, SIZE), np.float32) * 0.5
    edge = SIZE // 8
    rgb[:edge] = rgb[-edge:] = 0.05
    glow[:edge] = glow[-edge:] = 0.0
    return rgb, glow


def medical_cross():
    rgb = np.full((SIZE, SIZE, 3), 0.03, np.float32)
    glow = np.zeros((SIZE, SIZE), np.float32)
    low, high = SIZE // 3, 2 * SIZE // 3
    rgb[low:high, 40:-40] = rgb[40:-40, low:high] = (0.25, 1.0, 0.45)
    glow[low:high, 40:-40] = glow[40:-40, low:high] = 0.45
    return rgb, glow


def bake(directory, seed=20261030):
    rng = np.random.default_rng(seed)
    textures = {}
    for name in MATERIALS:
        rgb, glow = material(directory, name)
        textures[name] = (weather(rgb, rng), glow)
    textures["ceiling2_lit"] = lit_panel(textures["ceiling2"][0])
    for name in LIT_COPIES:
        # Most vents are dead: the plain name loses its glow, the lit copy keeps it.
        base = name.removesuffix("_lit")
        textures[name] = textures[base]
        textures[base] = (textures[base][0] * 0.6, np.zeros((SIZE, SIZE), np.float32))
    textures["viewport"] = (starfield(rng), np.ones((SIZE, SIZE), np.float32))
    for name in BLOODIED:
        rgb, glow = textures[name]
        textures[f"{name}_blood"] = (blood(rgb, rng, drips=name.startswith("wall")), glow)
    for name in CLAWED:
        rgb, glow = textures[name]
        textures[f"{name}_clawed"] = (scratches(rgb, rng), glow)
    plain = textures["walltrim4"][0]
    no_glow = np.zeros((SIZE, SIZE), np.float32)
    for key, sign in SIGNS.items():
        textures[f"sign_{key}"] = (plate(plain, sign, rng, (222, 204, 158)), no_glow)
    for name, color in STRIPES.items():
        textures[name] = stripe_rail(color)
    textures["hazard"] = (hazard_stripes(), no_glow)
    textures["cross"] = medical_cross()
    for key, text in WARNINGS.items():
        textures[f"warning_{key}"] = (plate(plain, text, rng, (190, 30, 20)), no_glow)
    return textures


def pack(rgb, glow):
    return (np.dstack((np.clip(rgb, 0, 1), np.clip(glow, 0, 1))) * 255 + 0.5).astype(np.uint8)


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
    textures = bake(directory)
    names = texture_names()
    level0 = np.stack([pack(*textures[name]) for name in names])
    level1 = half(level0).astype(np.uint8)
    level2 = half(level1).astype(np.uint8)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT, level0=level0, level1=level1, level2=level2, names=np.array(names))
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
