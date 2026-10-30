"""Compiled drawing of the ship's 3D surfaces: project, rasterize, then shade each pixel once.

Each triangle writes only its id and depth first, so hidden surfaces cost no shading. Rows
never overlap between bands, which keeps the parallel loops free of shared writes.
"""

from math import ceil, exp, floor

import numpy as np
from numba import njit, prange

NEAR = 0.04
FAR = 26.0  # fog has hidden everything long before this
SMOOTH_RANGE = 7.0  # nearer than this, texels are filtered
MIP_ONE = 5.0  # depth where a surface switches to half-size texels
MIP_TWO = 11.0
FOG_DENSITY = 0.11
TORCH_POWER = 1.7
TORCH_FALLOFF = 0.2
TORCH_RED, TORCH_GREEN, TORCH_BLUE = 1.0, 0.94, 0.82
GLOW = 1.5
TRI_FIELDS = 24
SKY = 0.0  # a surface with this shade shows space, as far away as the stars are
SKY_SCALE = 0.8  # share of the star texture seen across the screen
ATTRIBUTES = 6  # depth, u, v, world x, y, z; each divided by depth


@njit(inline="always")
def light_at(light, grid, x, y):
    """Bilinear RGB lookup; a clamped edge keeps points behind the hull valid."""
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
        light[y0, x0, 0] * near + light[y0, x1, 0] * right + light[y1, x0, 0] * below
        + light[y1, x1, 0] * far,
        light[y0, x0, 1] * near + light[y0, x1, 1] * right + light[y1, x0, 1] * below
        + light[y1, x1, 1] * far,
        light[y0, x0, 2] * near + light[y0, x1, 2] * right + light[y1, x0, 2] * below
        + light[y1, x1, 2] * far,
    )  # fmt: skip


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
    color, inverse_depth, ids, tris, tri_quad, texels, starts, size, textures, tints, shifts,
    shades, light, light_grid, beam, top, sky_turn, horizon,
):  # fmt: skip
    height, width = color.shape[0], color.shape[1]
    for row in prange(height):
        sy = row + 0.5
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
            u = (tris[tri, 9] * sx + tris[tri, 10] * sy + tris[tri, 11]) * depth
            v = (tris[tri, 12] * sx + tris[tri, 13] * sy + tris[tri, 14]) * depth
            world_x = (tris[tri, 15] * sx + tris[tri, 16] * sy + tris[tri, 17]) * depth
            world_y = (tris[tri, 18] * sx + tris[tri, 19] * sy + tris[tri, 20]) * depth
            world_z = (tris[tri, 21] * sx + tris[tri, 22] * sy + tris[tri, 23]) * depth
            if depth < MIP_ONE:
                r, g, b, a = fetch(
                    texels, starts[0], size, textures[quad], u, v, depth < SMOOTH_RANGE
                )
            elif depth < MIP_TWO:
                r, g, b, a = fetch(
                    texels, starts[1], size // 2, textures[quad], u, v, depth < SMOOTH_RANGE
                )
            else:
                r, g, b, a = fetch(texels, starts[2], size // 4, textures[quad], u, v, False)
            red, green, blue = light_at(
                light, light_grid, world_x + shifts[quad, 0], world_y + shifts[quad, 1]
            )
            fog = exp(-depth * FOG_DENSITY)
            gain = shades[quad] * (0.72 + 0.36 * world_z / top)
            torch = beam[row, column] * TORCH_POWER / (1 + TORCH_FALLOFF * depth * depth)
            glow = a / 255.0 * (fog * 0.9 + 0.25) * GLOW
            color[row, column, 0] = (
                r / 255.0 * ((red + tints[quad, 0]) * fog * gain + torch * TORCH_RED + glow)
            )
            color[row, column, 1] = (
                g / 255.0 * ((green + tints[quad, 1]) * fog * gain + torch * TORCH_GREEN + glow)
            )
            color[row, column, 2] = (
                b / 255.0 * ((blue + tints[quad, 2]) * fog * gain + torch * TORCH_BLUE + glow)
            )


@njit(parallel=True, nogil=True, cache=True)
def finish_frame(color, vignette, grain, shift, tone, pixels):
    """Vignette, film grain and tone curve; pixels is indexed (column, row, channel)."""
    height, width = color.shape[0], color.shape[1]
    for row in prange(height):
        noise_row = (row + shift) % height
        for column in range(width):
            noise = grain[noise_row, column]
            scale = vignette[row, column] * 255
            for channel in range(3):
                level = color[row, column, channel] * scale + noise
                pixels[column, row, channel] = tone[int(min(max(level, 0.0), 255.0))]
