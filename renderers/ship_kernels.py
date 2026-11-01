"""Compiled drawing of the ship's 3D surfaces: project, rasterize, then shade each pixel once.

Each triangle writes only its id and depth first, so hidden surfaces cost no shading. Rows
never overlap between bands, which keeps the parallel loops free of shared writes.

Every pixel is lit from its own normal, so the relief in a material answers the ship's lights
and the torch as the player moves. Baked light tells how much arrives and from where.
"""

from math import ceil, floor, sqrt

import numpy as np
from numba import njit, prange

NEAR = 0.04
FAR = 26.0  # fog has hidden everything long before this
MIP_ONE = 2.2  # depth where a surface switches to half-size texels, about one per pixel
MIP_TWO = 5.0  # beyond here relief is too fine to see, so a pixel is lit from its flat face
RELIEF_RANGE = 1.0  # filtering relief is the dearest step, and only shows this close
FOG_DENSITY = 0.11
FOG_STEPS = 16  # table entries per unit of depth; exp() per pixel is a real cost
LIGHT_SPAN = 4  # pixels along a row that share one lookup of the baked light
TORCH_POWER = 1.2
TORCH_FALLOFF = 0.09
TORCH_RED, TORCH_GREEN, TORCH_BLUE = 1.0, 0.96, 0.9
GLOW = 1.5
FILL = 0.2  # light that reaches a surface however it faces, from bounces off the walls
FLOOR_LIGHT = 0.15  # no corridor goes fully black: the player must still see the Alien
BOUNCE = 0.7  # ceilings catch extra fill from the lit deck below them
LAMP_GAIN = 1.2  # a surface facing the lamp is brighter than the baked average
SHEEN = 2.4  # how hard metal answers a lamp
TORCH_FILL = 0.3
TORCH_SHEEN = 1.8
FAR_SHINE = 0.4  # the torch still picks out far walls, and the Alien's outline against them
BAY_LAMP_TOP = 1.6  # hidden light in a wall bay sits at the head of its panel
BAY_POWER = 3.2
TONE_STEPS = 1024  # the curve spans twice full white, so highlights roll off, not clip
TRI_FIELDS = 24
FRAME_FIELDS = 9  # across, up and out of a surface, in world space
SKY = 0.0  # a surface with this shade shows space, as far away as the stars are
SKY_SCALE = 0.8  # share of the star texture seen across the screen
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
    shifts, shades, frames, light, aim, light_grid, beam, eye, sky_turn, horizon, fog_table,
):  # fmt: skip
    height, width = color.shape[0], color.shape[1]
    eye_x, eye_y, eye_z = eye
    for row in prange(height):
        sy = row + 0.5
        lit_quad, lit_until = -1, 0
        red = green = blue = lamp_x = lamp_y = lamp_z = 0.0
        view_x = view_y = view_z = half_x = half_y = half_z = 0.0
        fog = gain = torch = bay_height = 0.0
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
                # Stars depend only on where the camera points, so they never come closer.
                u = sky_turn + sx / width * SKY_SCALE
                v = (sy - horizon) / width * SKY_SCALE
                r, g, b, a = fetch(texels, starts[0], size, textures[quad], u, v, False)
                color[row, column, 0] = r / 255.0
                color[row, column, 1] = g / 255.0
                color[row, column, 2] = b / 255.0
                continue
            depth = 1 / inverse_depth[row, column]
            if quad != lit_quad or column >= lit_until:
                # Light changes slowly across a surface, so a few pixels can share it.
                lit_quad, lit_until = quad, column + LIGHT_SPAN
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
                fog = fog_table[int(depth * FOG_STEPS)]
                gain = shades[quad] * fog
                torch = TORCH_POWER / (1 + TORCH_FALLOFF * depth * depth)
                bay_height = world_z / BAY_LAMP_TOP
                bay_height = min(bay_height * bay_height, 1.0) * gain * BAY_POWER
            u = (tris[tri, 9] * sx + tris[tri, 10] * sy + tris[tri, 11]) * depth
            v = (tris[tri, 12] * sx + tris[tri, 13] * sy + tris[tri, 14]) * depth
            texture = textures[quad]
            sheen = 0.0
            if depth < MIP_TWO:
                if depth < MIP_ONE:
                    r, g, b, a = fetch(texels, starts[0], size, texture, u, v, True)
                    close = depth < RELIEF_RANGE
                    nx, ny, nz, shine = fetch(reliefs, starts[0], size, texture, u, v, close)
                else:
                    r, g, b, a = fetch(texels, starts[1], size // 2, texture, u, v, True)
                    nx, ny, nz, shine = fetch(reliefs, starts[1], size // 2, texture, u, v, False)
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
                shine = FAR_SHINE
                face_x, face_y, face_z = out_x, out_y, out_z
            facing = max(face_x * lamp_x + face_y * lamp_y + face_z * lamp_z, 0.0)
            toward = max(face_x * view_x + face_y * view_y + face_z * view_z, 0.0)
            fill = FILL * (1.0 + BOUNCE * max(-face_z, 0.0))
            lit = (fill + (1.0 - FILL) * facing * LAMP_GAIN) * gain
            beamed = torch * beam[row, column]
            torch_sheen = beamed * sharpen(toward) * shine * TORCH_SHEEN
            flat_light = FLOOR_LIGHT * gain + beamed * (TORCH_FILL + (1.0 - TORCH_FILL) * toward)
            bay = bay_height * (0.3 + 1.4 * max(face_z, 0.0))
            glow = a / 255.0 * (fog * 0.9 + 0.25) * GLOW
            color[row, column, 0] = (
                r / 255.0 * (red * lit + tint_red * bay + flat_light * TORCH_RED + glow)
                + red * sheen + torch_sheen * TORCH_RED
            )  # fmt: skip
            color[row, column, 1] = (
                g / 255.0 * (green * lit + tint_green * bay + flat_light * TORCH_GREEN + glow)
                + green * sheen + torch_sheen * TORCH_GREEN
            )  # fmt: skip
            color[row, column, 2] = (
                b / 255.0 * (blue * lit + tint_blue * bay + flat_light * TORCH_BLUE + glow)
                + blue * sheen + torch_sheen * TORCH_BLUE
            )  # fmt: skip


@njit(parallel=True, nogil=True, cache=True)
def finish_frame(color, vignette, grain, shift, tone, pixels, shifts):
    """Vignette, film grain and tone curve, written straight into the surface's own pixels."""
    height, width = color.shape[0], color.shape[1]
    red_shift, green_shift, blue_shift = shifts
    top = TONE_STEPS - 1.0
    for row in prange(height):
        noise_row = (row + shift) % height
        for column in range(width):
            noise = grain[noise_row, column]
            scale = vignette[row, column] * (TONE_STEPS / 2)
            red = tone[int(min(max(color[row, column, 0] * scale + noise, 0.0), top))]
            green = tone[int(min(max(color[row, column, 1] * scale + noise, 0.0), top))]
            blue = tone[int(min(max(color[row, column, 2] * scale + noise, 0.0), top))]
            pixels[column, row] = (
                (red << red_shift) | (green << green_shift) | (blue << blue_shift)
            )


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
