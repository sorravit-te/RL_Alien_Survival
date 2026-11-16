"""Bake what the bridge windows look out on: Earth, the Moon and a few small planets.

Usage: python -m tools.bake_ship_sky

NASA's maps download once into assets/space/source/ and the finished sky goes to
assets/space/ship_sky.npz. Both stay local, like the baked textures. Each globe is worked
out from its map, so day, night, clouds and air all come from one sun. The small planets
are made here and need no download. Credit: NASA Earth Observatory, NASA's Scientific
Visualization Studio. NASA imagery only; no NASA logo or insignia is used.
"""

from math import cos, pi, radians, sin, tan
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from tools.asset_paths import ROOT

SOURCE = ROOT / "assets" / "space" / "source"
OUTPUT = ROOT / "assets" / "space" / "ship_sky.npz"
EARTH_MAPS = "https://eoimages.gsfc.nasa.gov/images/imagerecords"
MAPS = {
    "earth_day.jpg": f"{EARTH_MAPS}/73000/73751/world.topo.bathy.200407.3x5400x2700.jpg",
    "earth_clouds.jpg": f"{EARTH_MAPS}/57000/57747/cloud_combined_2048.jpg",
    "earth_night.jpg": f"{EARTH_MAPS}/144000/144898/BlackMarble_2016_01deg.jpg",
    "moon.jpg": "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/lroc_color_2k.jpg",
}
WIDTH, HEIGHT = 4096, 2048  # the whole sky; about eleven texels to a degree
SUN = (-0.74, 0.34, 0.58)  # right, up and toward the viewer: the far side falls into night
# Space is at infinity, so a body keeps its size as a pane shrinks with distance. Seen from the
# corridor a pane covers about 6 degrees: small worlds fit there, and Earth waits off to the side.
EARTH_PLACE = (20.0, -2.5, 12.0)  # degrees right of the bow, above its level, and in radius
EARTH_FACES = (12.0, 100.0, -23.0)  # the latitude and longitude turned to the ship, its tilt
MOON_PLACE = (-4.5, -0.5, 1.6)
MOON_FACES = (0.0, 0.0, 12.0)
RUST_PLACE = (-24.0, 1.5, 1.25)
ICE_PLACE = (-30.0, -3.5, 0.6)
RINGED_PLACE = (4.2, 0.3, 1.0)
RING_TILT, RING_TURN, RING_INNER, RING_OUTER = 0.3, radians(18), 1.35, 2.25
STAR_COUNT = 16000
EXPOSURE = 1.5


def fetch(name):
    path = SOURCE / name
    if not path.exists():
        request = Request(MAPS[name], headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=120) as response:
            data = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path


def linear_map(name):
    """NASA maps are display colors; light adds up properly only in linear ones."""
    with Image.open(fetch(name)) as image:
        return (np.asarray(image.convert("RGB"), np.float32) / 255) ** 2.2


def lookup(image, longitude, latitude):
    """Bilinear sample of a map running west to east and north to south."""
    height, width = image.shape[:2]
    x = (longitude / (2 * pi) + 0.5) * width - 0.5
    y = np.clip((0.5 - latitude / pi) * height - 0.5, 0, height - 1.001)
    left, top = np.floor(x).astype(np.int32), y.astype(np.int32)
    across, down = (x - left)[..., None], (y - top)[..., None]
    left, right = left % width, (left + 1) % width
    upper = image[top, left] * (1 - across) + image[top, right] * across
    lower = image[top + 1, left] * (1 - across) + image[top + 1, right] * across
    return upper * (1 - down) + lower * down


def noise(rng, cells, size):
    coarse = (rng.random((cells // 2, cells), dtype=np.float32) * 255).astype(np.uint8)
    picture = Image.fromarray(coarse).resize(size, Image.Resampling.BICUBIC)
    return np.asarray(picture, np.float32) / 255


def layered_noise(rng, size, cells=(6, 14, 34, 90), falloff=0.55):
    total, weight, amount = np.zeros(size[::-1], np.float32), 0.0, 1.0
    for count in cells:
        total += noise(rng, count, size) * amount
        weight += amount
        amount *= falloff
    return total / weight


def sky_grid(left, right, top, bottom):
    """Longitude and latitude of a block of sky pixels."""
    columns = (np.arange(left, right, dtype=np.float32) + 0.5) / WIDTH * 2 * pi - pi
    rows = pi / 2 - (np.arange(top, bottom, dtype=np.float32) + 0.5) / HEIGHT * pi
    return np.meshgrid(columns, rows)


def disc_view(place, reach):
    """Pixels around a body, as (block, x, y) on its disc: one at its edge, 0 at its centre."""
    longitude, latitude, radius = (radians(value) for value in place)
    half_height = radius * reach
    half_width = half_height / max(cos(latitude) - half_height, 0.1)
    left = int((longitude - half_width + pi) / (2 * pi) * WIDTH)
    right = int((longitude + half_width + pi) / (2 * pi) * WIDTH) + 1
    top = int((pi / 2 - latitude - half_height) / pi * HEIGHT)
    bottom = int((pi / 2 - latitude + half_height) / pi * HEIGHT) + 1
    lon, lat = sky_grid(left, right, top, bottom)
    offset = lon - longitude
    near = np.sin(latitude) * np.sin(lat) + np.cos(latitude) * np.cos(lat) * np.cos(offset)
    x = np.cos(lat) * np.sin(offset) / near
    y = (np.cos(latitude) * np.sin(lat) - np.sin(latitude) * np.cos(lat) * np.cos(offset)) / near
    return (slice(top, bottom), slice(left, right)), x / tan(radius), y / tan(radius)


def globe_position(x, y, facing):
    """Latitude and longitude under each disc point of a globe turned to `facing`."""
    latitude, longitude, tilt = (radians(value) for value in facing)
    out = np.sqrt(np.clip(1 - x * x - y * y, 0, 1))
    across = x * cos(tilt) - y * sin(tilt)
    up = x * sin(tilt) + y * cos(tilt)
    sine_lat, cosine_lat = sin(latitude), cos(latitude)
    sine_lon, cosine_lon = sin(longitude), cos(longitude)
    east = out * cosine_lat * cosine_lon - up * sine_lat * cosine_lon - across * sine_lon
    pole = out * cosine_lat * sine_lon - up * sine_lat * sine_lon + across * cosine_lon
    north = out * sine_lat + up * cosine_lat
    return np.arcsin(np.clip(north, -1, 1)), np.arctan2(pole, east)


def sunlight(x, y):
    out = np.sqrt(np.clip(1 - x * x - y * y, 0, 1))
    return x * SUN[0] + y * SUN[1] + out * SUN[2], out


def edge_of(x, y, pixels):
    """Soft disc coverage, so a globe's rim does not stair-step."""
    return np.clip((1 - np.sqrt(x * x + y * y)) * pixels, 0, 1)[..., None]


def lay_over(sky, block, picture, cover):
    sky[block] = sky[block] * (1 - cover) + picture * cover


def scattered_air(x, y, reach=0.035):
    """Blue air thinning out past the limb, brightest on the sunlit side."""
    rim = np.sqrt(x * x + y * y)
    length = np.maximum(rim, 1e-4)
    toward_sun = (x * SUN[0] + y * SUN[1]) / length
    lit = np.clip(0.55 + 0.75 * toward_sun, 0, 1)
    height = np.maximum(rim - 1, 0)
    return (np.exp(-height / reach) * (rim > 1) * lit)[..., None] * np.array(
        (0.1, 0.27, 0.72), np.float32
    )


def draw_earth(sky):
    block, x, y = disc_view(EARTH_PLACE, 1.5)
    latitude, longitude = globe_position(x, y, EARTH_FACES)
    day = lookup(linear_map("earth_day.jpg"), longitude, latitude)
    clouds = lookup(linear_map("earth_clouds.jpg"), longitude, latitude)[..., 0] ** 0.45
    clouds = np.clip((clouds - 0.3) / 0.55, 0, 1) ** 1.3  # thin haze would hide the land
    night = lookup(linear_map("earth_night.jpg"), longitude, latitude)
    sun, out = sunlight(x, y)
    daylight = np.clip((sun + 0.06) / 0.5, 0, 1)
    daylight = (daylight * daylight * (3 - 2 * daylight))[..., None]
    water = (np.clip((day[..., 2] - day[..., 0] * 1.4) * 14, 0, 1) * (1 - clouds))[..., None]
    middle = np.array(SUN) + (0, 0, 1)
    middle = middle / np.linalg.norm(middle)
    glint = np.clip(x * middle[0] + y * middle[1] + out * middle[2], 0, 1) ** 70
    cover = clouds[..., None]
    ground = day * (0.02 + 1.5 * daylight) + water * glint[..., None] * 0.8
    surface = ground * (1 - cover * 0.92) + cover * daylight * 1.15
    city = np.clip(night.mean(axis=2, keepdims=True) * 2.4 - 0.1, 0, 1) ** 0.8
    surface += city * np.array((1.0, 0.72, 0.42), np.float32) * (1 - daylight) * (1 - cover * 0.6)
    haze = (1 - out)[..., None] ** 2.6
    surface += haze * daylight * np.array((0.13, 0.32, 0.8), np.float32)
    lay_over(sky, block, surface, edge_of(x, y, 180))
    sky[block] += scattered_air(x, y)


def draw_moon(sky):
    block, x, y = disc_view(MOON_PLACE, 1.2)
    latitude, longitude = globe_position(x, y, MOON_FACES)
    surface = lookup(linear_map("moon.jpg"), longitude, latitude)
    sun, _ = sunlight(x, y)
    lay_over(sky, block, surface * np.clip(sun * 1.5, 0, 1)[..., None] * 1.1, edge_of(x, y, 22))


def painted_globe(sky, place, facing, colors, bands=None):
    """A made-up world: colors runs dark to light across a noise map the size of a globe."""
    block, x, y = disc_view(place, 1.2)
    latitude, longitude = globe_position(x, y, facing)
    tone = lookup(bands[..., None], longitude, latitude)[..., 0]
    steps = np.linspace(0, 1, len(colors))
    surface = np.stack(
        [np.interp(tone, steps, [color[c] for color in colors]) for c in range(3)], axis=2
    ).astype(np.float32)
    sun, out = sunlight(x, y)
    shaded = surface * (0.012 + np.clip(sun * 1.5, 0, 1)[..., None] * 1.5)
    haze = (1 - out)[..., None] ** 3 * np.clip(sun * 2, 0, 1)[..., None]
    lay_over(sky, block, shaded + surface * haze * 0.6, edge_of(x, y, 16))
    return block, x, y


def world_noise(rng, stretch=1.0):
    size = (1024, 512)
    tone = layered_noise(rng, size)
    if stretch == 1.0:
        return tone
    rows = (np.arange(size[1], dtype=np.float32) / size[1])[:, None]
    return (tone * 0.28 + 0.5 + 0.5 * np.sin(rows * stretch)) * 0.74


def draw_rust_world(sky, rng):
    tone = np.clip((world_noise(rng) - 0.32) * 2.8, 0, 1)
    colors = ((0.09, 0.03, 0.015), (0.3, 0.1, 0.04), (0.62, 0.3, 0.14), (0.8, 0.6, 0.45))
    painted_globe(sky, RUST_PLACE, (8.0, 40.0, 20.0), colors, tone)


def draw_ice_world(sky, rng):
    tone = np.clip((world_noise(rng) - 0.3) * 2.6, 0, 1)
    colors = ((0.12, 0.22, 0.32), (0.32, 0.5, 0.62), (0.7, 0.84, 0.9))
    painted_globe(sky, ICE_PLACE, (-20.0, 120.0, -10.0), colors, tone)


def ring_plane(x, y):
    across = x * cos(RING_TURN) + y * sin(RING_TURN)
    up = (-x * sin(RING_TURN) + y * cos(RING_TURN)) / RING_TILT
    return np.sqrt(across * across + up * up), up


def lay_ring(sky, block, x, y, rng, front):
    distance, up = ring_plane(x, y)
    gaps = noise(rng, 90, (256, 1))[0]
    place = np.clip((distance - RING_INNER) / (RING_OUTER - RING_INNER), 0, 0.999)
    thickness = gaps[(place * 255).astype(np.int32)] ** 1.6
    inside = (distance > RING_INNER) & (distance < RING_OUTER)
    hidden = (x * x + y * y < 1) & ((up > 0) if front else (up <= 0))
    cover = (thickness * inside * ~hidden * 0.85)[..., None]
    lit = np.array((0.52, 0.46, 0.36), np.float32)
    lay_over(sky, block, np.ones_like(sky[block]) * lit, cover)


def draw_ringed_world(sky, rng):
    block, x, y = disc_view(RINGED_PLACE, 2.4)
    lay_ring(sky, block, x, y, rng, front=False)
    tone = np.clip(world_noise(rng, stretch=46.0), 0, 1)
    colors = ((0.28, 0.2, 0.12), (0.56, 0.42, 0.26), (0.82, 0.7, 0.5), (0.9, 0.84, 0.7))
    painted_globe(sky, RINGED_PLACE, (-14.0, 200.0, -18.0), colors, tone)
    lay_ring(sky, block, x, y, rng, front=True)


def star_field(rng):
    """Faint galaxy haze, then stars of many strengths: open space must feel deep, not painted."""
    sky = np.zeros((HEIGHT, WIDTH, 3), np.float32)
    columns = (np.arange(WIDTH, dtype=np.float32) + 0.5) / WIDTH * 2 * pi - pi
    rows = pi / 2 - (np.arange(HEIGHT, dtype=np.float32) + 0.5) / HEIGHT * pi
    lon, lat = np.meshgrid(columns, rows)
    band = np.exp(-(((lat - 0.42 * np.sin(lon + 1.1)) / 0.26) ** 2))
    cloud = layered_noise(rng, (WIDTH, HEIGHT), cells=(14, 34, 80, 200))
    glow = (band * np.clip(cloud - 0.34, 0, 1) ** 1.5 * 0.1)[..., None]
    sky += glow * np.array((0.5, 0.55, 0.85), np.float32)
    height = rng.uniform(-1, 1, STAR_COUNT)
    columns = rng.integers(0, WIDTH, STAR_COUNT)
    rows = ((0.5 - np.arcsin(height) / pi) * HEIGHT).astype(np.int32).clip(0, HEIGHT - 1)
    strength = rng.pareto(2.2, STAR_COUNT).clip(0, 9) * 0.05 + 0.02
    warmth = rng.uniform(0, 1, STAR_COUNT)[:, None]
    tint = warmth * np.array((1.0, 0.86, 0.68)) + (1 - warmth) * np.array((0.7, 0.82, 1.0))
    sky[rows, columns] += (strength[:, None] * tint).astype(np.float32)
    bright = strength > 0.3
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        near_rows = (rows[bright] + dy).clip(0, HEIGHT - 1)
        near_columns = (columns[bright] + dx) % WIDTH
        sky[near_rows, near_columns] += (strength[bright, None] * tint[bright] * 0.3).astype(
            np.float32
        )
    return sky


def finished(sky):
    """Roll bright light off to white and pack each texel the way the shader reads it."""
    display = (1 - np.exp(-sky * EXPOSURE)) ** (1 / 2.2)
    red, green, blue = np.moveaxis((display * 255 + 0.5).astype(np.uint32), 2, 0)
    return red | (green << 8) | (blue << 16)


def bake(seed=20261101):
    rng = np.random.default_rng(seed)
    sky = star_field(rng)
    draw_moon(sky)
    draw_rust_world(sky, rng)
    draw_ice_world(sky, rng)
    draw_ringed_world(sky, rng)
    draw_earth(sky)
    return finished(sky)


def main():
    sky = bake()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, sky=sky)
    print(f"Baked the sky seen from the bridge, {sky.shape[1]}x{sky.shape[0]}: {OUTPUT}")


if __name__ == "__main__":
    main()
