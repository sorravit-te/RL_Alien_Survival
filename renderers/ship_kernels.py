"""Compiled drawing of the ship's 3D surfaces: project, rasterize, then shade each pixel once.

Each triangle writes only its id and depth first, so hidden surfaces cost no shading. Rows
never overlap between bands, which keeps the parallel loops free of shared writes.

Every pixel is lit from its own normal, so the relief in a material answers the ship's lights
as the player moves. Baked light tells how much arrives and from where.
"""

from math import asin, atan2, ceil, floor, pi, sqrt

import numpy as np
from numba import njit, prange

NEAR = 0.04
FAR = 26.0  # fog has hidden everything long before this
MIP_ONE = 2.2  # depth where a surface switches to half-size texels, about one per pixel
MIP_TWO = 5.0  # beyond here relief is too fine to see, so a pixel is lit from its flat face
RELIEF_RANGE = 1.0  # filtering relief is the dearest step, and only shows this close
FOG_DENSITY = 0.15
FOG_STEPS = 16  # table entries per unit of depth; exp() per pixel is a real cost
LIGHT_SPAN = 4  # pixels along a row that share one lookup of the baked light
DARK_SPAN = 8  # and where that light is too low to show its own detail
GLOW = 1.5
FILL = 0.1  # light that reaches a surface however it faces, from bounces off the walls
FLOOR_LIGHT = 0.05  # no corridor goes fully black: the player must still see the Alien
FLOOR_TINT = 1.0, 0.96, 0.9
BOUNCE = 0.7  # ceilings catch extra fill from the lit deck below them
LAMP_GAIN = 1.2  # a surface facing the lamp is brighter than the baked average
SHEEN = 2.4  # how hard metal answers a lamp
BAY_LAMP_TOP = 1.6  # hidden light in a wall bay sits at the head of its panel
BAY_POWER = 3.2
TONE_STEPS = 1024  # the curve spans twice full white, so highlights roll off, not clip
TRI_FIELDS = 24
SKY = 0.0  # a surface with this shade is window glass with open space behind it
GLASS_TINT = 0.93, 0.97, 1.0  # thick glass takes a little red out of what passes
GLASS_MIRROR = 0.07  # what it reflects looking straight on; grazing it, nearly all
GLASS_ROOM = 0.55  # the lit bridge in the pane
GLASS_DUST = 0.35  # how much the marks on a pane dim the view
GLASS_MARKS_LIT = 0.6  # how hard those marks catch the room's light
GLASS_LAMP = 2.4  # the bridge lamps mirrored in the pane, sliding as the player walks
AIR_BLOCK = 4  # air is soft: one ray for this many pixels each way is plenty
AIR_STEPS = 7
AIR_REACH = 14.0
AIR_DENSITY = 0.12
FLOOR_MIST = 2.5  # how much thicker the mist lies at deck level
FLOOR_MIST_TOP = 0.8
MIST_TEXELS = 22.0  # of the mist tile, per unit of world
GLOW_FROM = 0.8  # how bright a pixel must be to spill into the ones around it
GLOW_AMOUNT = 0.5
GLOW_DIRT = 2.0  # how much of that spill catches on the dirt on the lens
FRINGE = 0.0035  # color split at the screen's edges, as a share of its width
DIM = 0.05  # light so low that a surface's fine relief cannot be seen in it
SCATTER = ((0, 8, 2, 10), (12, 4, 14, 6), (3, 11, 1, 9), (15, 7, 13, 5))  # where rays start
ATTRIBUTES = 6  # depth, u, v, world x, y, z; each divided by depth


@njit(inline="always")
def blend(grid_values, channel, y0, x0, y1, x1, near, right, below, far):
    return (
        grid_values[y0, x0, channel] * near + grid_values[y0, x1, channel] * right
        + grid_values[y1, x0, channel] * below + grid_values[y1, x1, channel] * far
    )  # fmt: skip


@njit(inline="always")
def light_at(light, aim, grid, x, y):
    """Bilinear light and its direction; a clamped edge keeps points behind the hull valid."""
    fx = x * grid - 0.5
    fy = y * grid - 0.5
    left = int(floor(fx))
    top = int(floor(fy))
    tx = fx - left
    ty = fy - top
    height, width = light.shape[0], light.shape[1]
    x0, x1 = min(max(left, 0), width - 1), min(max(left + 1, 0), width - 1)
    y0, y1 = min(max(top, 0), height - 1), min(max(top + 1, 0), height - 1)
    near = (1 - tx) * (1 - ty)
    right = tx * (1 - ty)
    below = (1 - tx) * ty
    far = tx * ty
    return (
        blend(light, 0, y0, x0, y1, x1, near, right, below, far),
        blend(light, 1, y0, x0, y1, x1, near, right, below, far),
        blend(light, 2, y0, x0, y1, x1, near, right, below, far),
        blend(aim, 0, y0, x0, y1, x1, near, right, below, far),
        blend(aim, 1, y0, x0, y1, x1, near, right, below, far),
        blend(aim, 2, y0, x0, y1, x1, near, right, below, far),
    )


@njit(inline="always")
def sky_at(sky, x, y, z):
    """Bilinear look along a unit ray: x right of the bow, y ahead of it, z up."""
    height, width = sky.shape[0], sky.shape[1]
    fx = (atan2(x, y) / (2 * pi) + 0.5) * width - 0.5
    fy = (0.5 - asin(min(max(z, -1.0), 1.0)) / pi) * height - 0.5
    left, top = floor(fx), floor(fy)
    tx, ty = fx - left, fy - top
    x0, x1 = int(left) % width, (int(left) + 1) % width
    y0, y1 = max(int(top), 0), min(int(top) + 1, height - 1)
    ar, ag, ab, _ = unpack(sky[y0, x0])
    br, bg, bb, _ = unpack(sky[y0, x1])
    cr, cg, cb, _ = unpack(sky[y1, x0])
    dr, dg, db, _ = unpack(sky[y1, x1])
    return (
        mix(ar, br, cr, dr, tx, ty) / 255.0,
        mix(ag, bg, cg, dg, tx, ty) / 255.0,
        mix(ab, bb, cb, db, tx, ty) / 255.0,
    )


@njit(inline="always")
def sharpen(amount):
    """amount to the 16th: the tight highlight of worn metal, at the cost of four multiplies."""
    amount *= amount
    amount *= amount
    amount *= amount
    return amount * amount


@njit(inline="always")
def unpack(texel):
    return (
        float(texel & 255),
        float((texel >> 8) & 255),
        float((texel >> 16) & 255),
        float((texel >> 24) & 255),
    )


@njit(inline="always")
def mix(a, b, c, d, tx, ty):
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty


@njit(inline="always")
def fetch(texels, start, size, texture, u, v, smooth):
    """Texture sample with wrapping; size is a power of two."""
    area = texture * size * size + start
    mask = size - 1
    fu, fv = u * size, v * size
    if not smooth:
        return unpack(texels[area + (int(floor(fv)) & mask) * size + (int(floor(fu)) & mask)])
    fu -= 0.5
    fv -= 0.5
    left, top = floor(fu), floor(fv)
    tx, ty = fu - left, fv - top
    x0, y0 = int(left) & mask, int(top) & mask
    x1, y1 = (x0 + 1) & mask, (y0 + 1) & mask
    ar, ag, ab, aa = unpack(texels[area + y0 * size + x0])
    br, bg, bb, ba = unpack(texels[area + y0 * size + x1])
    cr, cg, cb, ca = unpack(texels[area + y1 * size + x0])
    dr, dg, db, da = unpack(texels[area + y1 * size + x1])
    return (
        mix(ar, br, cr, dr, tx, ty),
        mix(ag, bg, cg, dg, tx, ty),
        mix(ab, bb, cb, db, tx, ty),
        mix(aa, ba, ca, da, tx, ty),
    )


@njit(inline="always")
def emit_triangle(tris, tri_quad, count, quad, a, b, c):
    """Store screen corners and the plane of every attribute over the screen."""
    x0, y0, x1, y1, x2, y2 = a[0], a[1], b[0], b[1], c[0], c[1]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if abs(area) < 1e-6:
        return count
    tris[count, 0], tris[count, 1], tris[count, 2] = x0, y0, x1
    tris[count, 3], tris[count, 4], tris[count, 5] = y1, x2, y2
    for attribute in range(ATTRIBUTES):
        v0, v1, v2 = a[2 + attribute], b[2 + attribute], c[2 + attribute]
        across = ((v1 - v0) * (y2 - y0) - (v2 - v0) * (y1 - y0)) / area
        down = ((v2 - v0) * (x1 - x0) - (v1 - v0) * (x2 - x0)) / area
        tris[count, 6 + attribute * 3] = across
        tris[count, 7 + attribute * 3] = down
        tris[count, 8 + attribute * 3] = v0 - across * x0 - down * y0
    tri_quad[count] = quad
    return count + 1


@njit(cache=True)
def project(vertices, uvs, normals, view, eye, width, height, tris, tri_quad):
    """Cull and clip the surfaces in front of the camera; returns the triangle count."""
    x, y, fx, fy, plane, horizon, focal = view
    half_width = width / 2
    polygon = np.empty((6, 7), np.float64)  # depth, across, z, u, v, world x, world y
    clipped = np.empty((6, 7), np.float64)
    screen = np.empty((6, 8), np.float64)
    count = 0
    for quad in range(vertices.shape[0]):
        vx, vy, vz = vertices[quad, 0, 0], vertices[quad, 0, 1], vertices[quad, 0, 2]
        facing = (
            normals[quad, 0] * (x - vx)
            + normals[quad, 1] * (y - vy)
            + normals[quad, 2] * (eye - vz)
        )
        if facing <= 0:
            continue
        nearest, farthest = 1e9, -1e9
        for corner in range(4):
            rx = vertices[quad, corner, 0] - x
            ry = vertices[quad, corner, 1] - y
            depth = rx * fx + ry * fy
            polygon[corner, 0] = depth
            polygon[corner, 1] = -rx * fy + ry * fx
            polygon[corner, 2] = vertices[quad, corner, 2]
            polygon[corner, 3] = uvs[quad, corner, 0]
            polygon[corner, 4] = uvs[quad, corner, 1]
            polygon[corner, 5] = vertices[quad, corner, 0]
            polygon[corner, 6] = vertices[quad, corner, 1]
            nearest, farthest = min(nearest, depth), max(farthest, depth)
        if farthest < NEAR or nearest > FAR:
            continue
        corners = 4
        if nearest < NEAR:
            # Keep the part in front of the near plane.
            kept = 0
            for corner in range(4):
                next_corner = (corner + 1) % 4
                here, there = polygon[corner, 0], polygon[next_corner, 0]
                if here >= NEAR:
                    clipped[kept, :] = polygon[corner, :]
                    kept += 1
                if (here >= NEAR) != (there >= NEAR):
                    t = (NEAR - here) / (there - here)
                    clipped[kept, :] = (
                        polygon[corner, :] + (polygon[next_corner, :] - polygon[corner, :]) * t
                    )
                    kept += 1
            corners = kept
            if corners < 3:
                continue
            polygon[:corners, :] = clipped[:corners, :]
        left, right, top, bottom = 1e9, -1e9, 1e9, -1e9
        for corner in range(corners):
            inverse = 1 / polygon[corner, 0]
            sx = half_width + focal * polygon[corner, 1] * inverse
            sy = horizon - focal * (polygon[corner, 2] - eye) * inverse
            screen[corner, 0], screen[corner, 1] = sx, sy
            screen[corner, 2] = inverse
            screen[corner, 3] = polygon[corner, 3] * inverse
            screen[corner, 4] = polygon[corner, 4] * inverse
            screen[corner, 5] = polygon[corner, 5] * inverse
            screen[corner, 6] = polygon[corner, 6] * inverse
            screen[corner, 7] = polygon[corner, 2] * inverse
            left, right = min(left, sx), max(right, sx)
            top, bottom = min(top, sy), max(bottom, sy)
        if right < 0 or left > width or bottom < 0 or top > height:
            continue
        for corner in range(1, corners - 1):
            count = emit_triangle(
                tris, tri_quad, count, quad, screen[0], screen[corner], screen[corner + 1]
            )
    return count


@njit(inline="always")
def edge_span(ax, ay, bx, by, row, left, right):
    if (ay <= row < by) or (by <= row < ay):
        crossing = ax + (row - ay) * (bx - ax) / (by - ay)
        return min(left, crossing), max(right, crossing)
    return left, right


@njit(parallel=True, nogil=True, cache=True)
def rasterize(tris, count, inverse_depth, ids, bands):
    """Each band owns its rows, so triangles can be drawn into them independently."""
    height, width = inverse_depth.shape[0], inverse_depth.shape[1]
    for band in prange(bands):
        first_row = band * height // bands
        last_row = (band + 1) * height // bands
        for row in range(first_row, last_row):
            for column in range(width):
                inverse_depth[row, column] = 0.0
                ids[row, column] = -1
        for tri in range(count):
            x0, y0, x1 = tris[tri, 0], tris[tri, 1], tris[tri, 2]
            y1, x2, y2 = tris[tri, 3], tris[tri, 4], tris[tri, 5]
            low = min(y0, min(y1, y2))
            high = max(y0, max(y1, y2))
            start = max(first_row, int(ceil(low - 0.5)))
            stop = min(last_row - 1, int(ceil(high - 0.5)) - 1)
            across, down, base = tris[tri, 6], tris[tri, 7], tris[tri, 8]
            for row in range(start, stop + 1):
                middle = row + 0.5
                left, right = 1e18, -1e18
                left, right = edge_span(x0, y0, x1, y1, middle, left, right)
                left, right = edge_span(x1, y1, x2, y2, middle, left, right)
                left, right = edge_span(x2, y2, x0, y0, middle, left, right)
                if right < left:
                    continue
                begin = max(int(ceil(left - 0.5)), 0)
                end = min(int(ceil(right - 0.5)) - 1, width - 1)
                value = across * (begin + 0.5) + down * middle + base
                for column in range(begin, end + 1):
                    if value > inverse_depth[row, column]:
                        inverse_depth[row, column] = value
                        ids[row, column] = tri
                    value += across


@njit(parallel=True, nogil=True, cache=True)
def shade(
    color, inverse_depth, ids, tris, tri_quad, texels, reliefs, starts, size, textures, tints,
    shifts, shades, frames, light, aim, light_grid, eye, sky, fog_table, lamps,
):  # fmt: skip
    height, width = color.shape[0], color.shape[1]
    eye_x, eye_y, eye_z = eye
    last_fog = fog_table.shape[0] - 1
    for row in prange(height):
        sy = row + 0.5
        lit_quad, lit_until = -1, 0
        red = green = blue = lamp_x = lamp_y = lamp_z = 0.0
        view_x = view_y = view_z = half_x = half_y = half_z = 0.0
        fog = gain = bay_height = 0.0
        dark = False
        across_x = across_y = across_z = up_x = up_y = up_z = out_x = out_y = out_z = 0.0
        tint_red = tint_green = tint_blue = 0.0
        for column in range(width):
            tri = ids[row, column]
            if tri < 0:
                color[row, column, 0] = 0.0
                color[row, column, 1] = 0.0
                color[row, column, 2] = 0.0
                continue
            sx = column + 0.5
            quad = tri_quad[tri]
            if shades[quad] == SKY:
                # Space sits at infinity: only the way the player looks moves it.
                depth = 1 / inverse_depth[row, column]
                world_x = (tris[tri, 15] * sx + tris[tri, 16] * sy + tris[tri, 17]) * depth
                world_y = (tris[tri, 18] * sx + tris[tri, 19] * sy + tris[tri, 20]) * depth
                world_z = (tris[tri, 21] * sx + tris[tri, 22] * sy + tris[tri, 23]) * depth
                ray_x, ray_y, ray_z = world_x - eye_x, world_y - eye_y, world_z - eye_z
                scale = 1.0 / sqrt(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z + 1e-6)
                ray_x *= scale
                ray_y *= scale
                ray_z *= scale
                space_r, space_g, space_b = sky_at(sky, ray_x, -ray_y, ray_z)
                u = (tris[tri, 9] * sx + tris[tri, 10] * sy + tris[tri, 11]) * depth
                v = (tris[tri, 12] * sx + tris[tri, 13] * sy + tris[tri, 14]) * depth
                r, g, b, a = fetch(texels, starts[0], size, textures[quad], u, v, True)
                marks = a / 255.0
                # Glass shows by what lands on it: dust and the light of the room.
                out_x, out_y, out_z = frames[quad, 6], frames[quad, 7], frames[quad, 8]
                square_on = max(-(ray_x * out_x + ray_y * out_y + ray_z * out_z), 0.0)
                grazing = 1.0 - square_on
                mirror = grazing * grazing
                mirror = GLASS_MIRROR + (1.0 - GLASS_MIRROR) * mirror * mirror * grazing
                shift_x, shift_y = shifts[quad, 0], shifts[quad, 1]
                room_r, room_g, room_b, aim_x, aim_y, lamp_top = light_at(
                    light, aim, light_grid, world_x + shift_x, world_y + shift_y
                )
                lamp_x, lamp_y, lamp_z = aim_x + shift_x, aim_y + shift_y, lamp_top - world_z
                scale = 1.0 / sqrt(lamp_x * lamp_x + lamp_y * lamp_y + lamp_z * lamp_z + 1e-4)
                turn = 2.0 * square_on
                bounce = (
                    (ray_x + turn * out_x) * lamp_x + (ray_y + turn * out_y) * lamp_y
                    + (ray_z + turn * out_z) * lamp_z
                ) * scale  # fmt: skip
                lamp_shine = sharpen(max(bounce, 0.0))
                lamp_shine *= lamp_shine * GLASS_LAMP
                seen = (1.0 - mirror) * (1.0 - marks * GLASS_DUST)
                caught = marks * GLASS_MARKS_LIT
                color[row, column, 0] = (
                    space_r * seen * GLASS_TINT[0]
                    + room_r * (mirror * GLASS_ROOM + lamp_shine)
                    + room_r * caught * r / 255.0
                )  # fmt: skip
                color[row, column, 1] = (
                    space_g * seen * GLASS_TINT[1]
                    + room_g * (mirror * GLASS_ROOM + lamp_shine)
                    + room_g * caught * g / 255.0
                )  # fmt: skip
                color[row, column, 2] = (
                    space_b * seen * GLASS_TINT[2]
                    + room_b * (mirror * GLASS_ROOM + lamp_shine)
                    + room_b * caught * b / 255.0
                )  # fmt: skip
                continue
            depth = 1 / inverse_depth[row, column]
            if quad != lit_quad or column >= lit_until:
                # Light changes slowly across a surface, so a few pixels can share it.
                lit_quad = quad
                world_x = (tris[tri, 15] * sx + tris[tri, 16] * sy + tris[tri, 17]) * depth
                world_y = (tris[tri, 18] * sx + tris[tri, 19] * sy + tris[tri, 20]) * depth
                world_z = (tris[tri, 21] * sx + tris[tri, 22] * sy + tris[tri, 23]) * depth
                shift_x, shift_y = shifts[quad, 0], shifts[quad, 1]
                across_x, across_y, across_z = frames[quad, 0], frames[quad, 1], frames[quad, 2]
                up_x, up_y, up_z = frames[quad, 3], frames[quad, 4], frames[quad, 5]
                out_x, out_y, out_z = frames[quad, 6], frames[quad, 7], frames[quad, 8]
                tint_red, tint_green, tint_blue = tints[quad, 0], tints[quad, 1], tints[quad, 2]
                red, green, blue, aim_x, aim_y, lamp_top = light_at(
                    light, aim, light_grid, world_x + shift_x, world_y + shift_y
                )
                # Lamp direction: the baked way across the deck, and the height of the lamps.
                lamp_x, lamp_y, lamp_z = aim_x + shift_x, aim_y + shift_y, lamp_top - world_z
                scale = 1.0 / sqrt(lamp_x * lamp_x + lamp_y * lamp_y + lamp_z * lamp_z + 1e-4)
                lamp_x *= scale
                lamp_y *= scale
                lamp_z *= scale
                view_x, view_y, view_z = eye_x - world_x, eye_y - world_y, eye_z - world_z
                scale = 1.0 / sqrt(view_x * view_x + view_y * view_y + view_z * view_z + 1e-4)
                view_x *= scale
                view_y *= scale
                view_z *= scale
                half_x, half_y, half_z = lamp_x + view_x, lamp_y + view_y, lamp_z + view_z
                scale = 1.0 / sqrt(half_x * half_x + half_y * half_y + half_z * half_z + 1e-4)
                half_x *= scale
                half_y *= scale
                half_z *= scale
                # A surface that starts inside FAR can run on well past it.
                fog = fog_table[min(int(depth * FOG_STEPS), last_fog)]
                gain = shades[quad] * fog
                dark = (red + green + blue) * gain < DIM
                lit_until = column + (DARK_SPAN if dark else LIGHT_SPAN)
                bay_height = world_z / BAY_LAMP_TOP
                bay_height = min(bay_height * bay_height, 1.0) * gain * BAY_POWER
            u = (tris[tri, 9] * sx + tris[tri, 10] * sy + tris[tri, 11]) * depth
            v = (tris[tri, 12] * sx + tris[tri, 13] * sy + tris[tri, 14]) * depth
            texture = textures[quad]
            sheen = 0.0
            face_x, face_y, face_z = out_x, out_y, out_z
            if depth < MIP_TWO:
                level = 0 if depth < MIP_ONE else 1
                r, g, b, a = fetch(texels, starts[level], size >> level, texture, u, v, not dark)
                if not dark:
                    close = depth < RELIEF_RANGE
                    nx, ny, nz, shine = fetch(reliefs, starts[level], size >> level, texture, u, v, close)
                    nx = nx / 127.5 - 1.0
                    ny = ny / 127.5 - 1.0
                    nz = nz / 127.5 - 1.0
                    shine /= 255.0
                    # The surface's own direction at this pixel, from its relief.
                    face_x = across_x * nx + up_x * ny + out_x * nz
                    face_y = across_y * nx + up_y * ny + out_y * nz
                    face_z = across_z * nx + up_z * ny + out_z * nz
                    mirror = max(face_x * half_x + face_y * half_y + face_z * half_z, 0.0)
                    sheen = sharpen(mirror) * shine * SHEEN * gain
            else:
                r, g, b, a = fetch(texels, starts[2], size // 4, texture, u, v, False)
            facing = max(face_x * lamp_x + face_y * lamp_y + face_z * lamp_z, 0.0)
            fill = FILL * (1.0 + BOUNCE * max(-face_z, 0.0))
            lit = (fill + (1.0 - FILL) * facing * LAMP_GAIN) * gain
            flat_light = FLOOR_LIGHT * gain
            bay = bay_height * (0.3 + 1.4 * max(face_z, 0.0))
            glow = a / 255.0 * (fog * 0.9 + 0.25) * GLOW * lamps
            color[row, column, 0] = (
                r / 255.0 * (red * lit + tint_red * bay + flat_light * FLOOR_TINT[0] + glow)
                + red * sheen
            )  # fmt: skip
            color[row, column, 1] = (
                g / 255.0 * (green * lit + tint_green * bay + flat_light * FLOOR_TINT[1] + glow)
                + green * sheen
            )  # fmt: skip
            color[row, column, 2] = (
                b / 255.0 * (blue * lit + tint_blue * bay + flat_light * FLOOR_TINT[2] + glow)
                + blue * sheen
            )  # fmt: skip


@njit(nogil=True, cache=True)
def stamp_depth(air_depth, picture, box, nearness):
    """The Alien stands in the air too: mist behind it must not be drawn over it."""
    left, top, width, height = box
    picture_height, picture_width = picture.shape[0], picture.shape[1]
    for row in range(max(top // 2, 0), min((top + height) // 2 + 1, air_depth.shape[0])):
        y = min(max((row * 2 - top) * picture_height // height, 0), picture_height - 1)
        for column in range(max(left // 2, 0), min((left + width) // 2 + 1, air_depth.shape[1])):
            x = min(max((column * 2 - left) * picture_width // width, 0), picture_width - 1)
            if picture[y, x, 3] > 128 and air_depth[row, column] < nearness:
                air_depth[row, column] = nearness


@njit(parallel=True, nogil=True, cache=True)
def shrink_depth(air_depth, inverse_depth):
    height, width = inverse_depth.shape[0], inverse_depth.shape[1]
    for row in prange(air_depth.shape[0]):
        pixel_row = min(row * 2 + 1, height - 1)
        for column in range(air_depth.shape[1]):
            air_depth[row, column] = inverse_depth[pixel_row, min(column * 2 + 1, width - 1)]


@njit(parallel=True, nogil=True, cache=True)
def air_light(spill, raw, air_depth, view, eye, light, grid, mist, drift, fog_table):
    """What the air between the eye and each surface scatters back: mist in every lamp's light."""
    x, y, fx, fy, _, horizon, focal = view
    height, width = spill.shape[0], spill.shape[1]
    frame_width = air_depth.shape[1] * 2
    grid_height, grid_width = light.shape[0], light.shape[1]
    top = fog_table.shape[0] - 1
    tile = mist.shape[0]
    for row in prange(height):
        sy = (row + 0.5) * AIR_BLOCK
        up = (horizon - sy) / focal
        pixel_row = min(row * AIR_BLOCK // 2 + 1, air_depth.shape[0] - 1)
        for column in range(width):
            sx = (column + 0.5) * AIR_BLOCK
            side = (sx - frame_width * 0.5) / focal
            nearness = air_depth[pixel_row, min(column * AIR_BLOCK // 2 + 1, air_depth.shape[1] - 1)]
            reach = AIR_REACH if nearness * AIR_REACH < 1.0 else 1.0 / nearness
            step = reach / AIR_STEPS
            along = step * (0.03 + SCATTER[row & 3][column & 3] / 16.0)
            ray_x, ray_y = fx - fy * side, fy + fx * side
            stretch = sqrt(1.0 + side * side + up * up)
            red = green = blue = 0.0
            for _ in range(AIR_STEPS):
                at_x, at_y = x + ray_x * along, y + ray_y * along
                low = max(1.0 - (eye + up * along) / FLOOR_MIST_TOP, 0.0)
                thick = mist[
                    int(at_y * MIST_TEXELS + drift) & (tile - 1),
                    int(at_x * MIST_TEXELS + drift * 0.6) & (tile - 1),
                ]
                weight = thick * (1.0 + FLOOR_MIST * low * low) * step * stretch * AIR_DENSITY
                weight *= fog_table[min(int(along * stretch * FOG_STEPS), top)]
                cell_y = min(max(int(at_y * grid), 0), grid_height - 1)
                cell_x = min(max(int(at_x * grid), 0), grid_width - 1)
                red += light[cell_y, cell_x, 0] * weight
                green += light[cell_y, cell_x, 1] * weight
                blue += light[cell_y, cell_x, 2] * weight
                along += step
            raw[row, column, 0] = red
            raw[row, column, 1] = green
            raw[row, column, 2] = blue
    for row in prange(height):
        above, below = max(row - 1, 0), min(row + 1, height - 1)
        for column in range(width):
            left, right = max(column - 1, 0), min(column + 1, width - 1)
            for channel in range(3):
                spill[row, column, channel] = (
                    raw[above, left, channel] + raw[above, column, channel] * 2.0
                    + raw[above, right, channel] + raw[row, left, channel] * 2.0
                    + raw[row, column, channel] * 4.0 + raw[row, right, channel] * 2.0
                    + raw[below, left, channel] + raw[below, column, channel] * 2.0
                    + raw[below, right, channel]
                ) * 0.0625  # fmt: skip


@njit(parallel=True, nogil=True, cache=True)
def add_glow(spill, color, dirt, shrunk, blurred):
    """Bright pixels bleed into the ones around them, and catch on the dirt on the lens."""
    height, width = spill.shape[0], spill.shape[1]
    frame_height, frame_width = color.shape[0], color.shape[1]
    for row in prange(height):
        near = min(row * AIR_BLOCK + 1, frame_height - 1)
        far = min(row * AIR_BLOCK + 3, frame_height - 1)
        for column in range(width):
            left = min(column * AIR_BLOCK + 1, frame_width - 1)
            right = min(column * AIR_BLOCK + 3, frame_width - 1)
            red = (color[near, left, 0] + color[far, right, 0]) * 0.5
            green = (color[near, left, 1] + color[far, right, 1]) * 0.5
            blue = (color[near, left, 2] + color[far, right, 2]) * 0.5
            brightest = max(red, green, blue)
            keep = max(brightest - GLOW_FROM, 0.0) / (brightest + 1e-3)
            shrunk[row, column, 0] = red * keep
            shrunk[row, column, 1] = green * keep
            shrunk[row, column, 2] = blue * keep
    for row in prange(height):
        for column in range(width):
            for channel in range(3):
                total = shrunk[row, column, channel] * 0.26
                for offset, weight in ((2, 0.21), (4, 0.11), (7, 0.05)):
                    total += weight * (
                        shrunk[row, max(column - offset, 0), channel]
                        + shrunk[row, min(column + offset, width - 1), channel]
                    )
                blurred[row, column, channel] = total
    for row in prange(height):
        for column in range(width):
            for channel in range(3):
                total = blurred[row, column, channel] * 0.26
                for offset, weight in ((2, 0.21), (4, 0.11), (7, 0.05)):
                    total += weight * (
                        blurred[max(row - offset, 0), column, channel]
                        + blurred[min(row + offset, height - 1), column, channel]
                    )
                spill[row, column, channel] += total * (GLOW_AMOUNT + GLOW_DIRT * dirt[row, column])


@njit(parallel=True, nogil=True, cache=True)
def finish_frame(color, spill, vignette, grain, shift, tone, grade, pixels):
    """Air and glow, lens fringe, vignette, film grain and color grade, straight into the surface."""
    height, width = color.shape[0], color.shape[1]
    spill_height, spill_width = spill.shape[0], spill.shape[1]
    top = TONE_STEPS - 1.0
    fringe = FRINGE * 2.0
    step = 1.0 / AIR_BLOCK
    for row in prange(height):
        noise_row = (row + shift) % height
        place = (row + 0.5) * step - 0.5
        y0 = min(max(int(floor(place)), 0), spill_height - 1)
        y1 = min(y0 + 1, spill_height - 1)
        down = min(max(place - y0, 0.0), 1.0)
        for block in range(-1, spill_width):
            # Between two samples of the spill, which sit in the middle of their blocks.
            left, right = max(block, 0), min(block + 1, spill_width - 1)
            first = block * AIR_BLOCK + AIR_BLOCK // 2
            red_now = spill[y0, left, 0] + (spill[y1, left, 0] - spill[y0, left, 0]) * down
            green_now = spill[y0, left, 1] + (spill[y1, left, 1] - spill[y0, left, 1]) * down
            blue_now = spill[y0, left, 2] + (spill[y1, left, 2] - spill[y0, left, 2]) * down
            red_step = (spill[y0, right, 0] + (spill[y1, right, 0] - spill[y0, right, 0]) * down - red_now) * step
            green_step = (spill[y0, right, 1] + (spill[y1, right, 1] - spill[y0, right, 1]) * down - green_now) * step
            blue_step = (spill[y0, right, 2] + (spill[y1, right, 2] - spill[y0, right, 2]) * down - blue_now) * step
            for column in range(max(first, 0), min(first + AIR_BLOCK, width)):
                ahead = column - first + 0.5
                noise = grain[noise_row, (column + shift * 7) % width]
                scale = vignette[row, column] * (TONE_STEPS / 2)
                split = int((column - width * 0.5) * fringe)
                pulled = min(max(column - split, 0), width - 1)
                pushed = min(max(column + split, 0), width - 1)
                value = (color[row, pulled, 0] + red_now + red_step * ahead) * scale + noise
                red = tone[int(min(max(value, 0.0), top))]
                value = (color[row, column, 1] + green_now + green_step * ahead) * scale + noise
                green = tone[int(min(max(value, 0.0), top))]
                value = (color[row, pushed, 2] + blue_now + blue_step * ahead) * scale + noise
                blue = tone[int(min(max(value, 0.0), top))]
                pixels[column, row] = grade[((red >> 1) << 14) | ((green >> 1) << 7) | (blue >> 1)]


@njit(parallel=True, nogil=True, cache=True)
def draw_sprite(color, inverse_depth, picture, glow, box, nearness, lit):
    """Blend a billboard over the frame; pixels of a wall in front of it stay as they are."""
    left, top, width, height = box
    frame_height, frame_width = color.shape[:2]
    picture_height, picture_width = picture.shape[:2]
    for row in prange(max(0, top), min(frame_height, top + height)):
        y = (row - top) * picture_height // height
        for column in range(max(0, left), min(frame_width, left + width)):
            x = (column - left) * picture_width // width
            shown = picture[y, x, 3] / 255.0
            if shown == 0.0 or inverse_depth[row, column] >= nearness:
                continue
            shine = glow[y, x] / 255.0
            for channel in range(3):
                seen = picture[y, x, channel] / 255.0 * (lit[channel] * (1 - shine) + shine * 1.4)
                color[row, column, channel] += (seen - color[row, column, channel]) * shown
