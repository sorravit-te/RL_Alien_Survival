"""Continuous circle contact on simultaneous linear motion segments, gated by LoS."""

from math import floor, sqrt

from game.interaction import segment_clear


def roots(a, b, c):
    if abs(a) < 1e-15:
        return (-c / b,) if abs(b) > 1e-15 else ()
    discriminant = b * b - 4 * a * c
    if discriminant < -1e-12:
        return ()
    d = sqrt(max(0.0, discriminant))
    return ((-b - d) / (2 * a), (-b + d) / (2 * a))


def capture_fraction(world, s0, s1, a0, a1, distance):
    rx, ry = a0[0] - s0[0], a0[1] - s0[1]
    vx, vy = a1[0] - a0[0] - s1[0] + s0[0], a1[1] - a0[1] - s1[1] + s0[1]
    a, b, c = vx * vx + vy * vy, 2 * (rx * vx + ry * vy), rx * rx + ry * ry - distance * distance
    if a < 1e-24:
        if c > 1e-12:
            return None
        enter, leave = 0.0, 1.0
    else:
        solutions = roots(a, b, c)
        if not solutions:
            return None
        enter, leave = max(0.0, min(solutions)), min(1.0, max(solutions))
        if enter > leave:
            return None

    def visible(t):
        sx, sy = s0[0] + (s1[0] - s0[0]) * t, s0[1] + (s1[1] - s0[1]) * t
        ax, ay = a0[0] + (a1[0] - a0[0]) * t, a0[1] + (a1[1] - a0[1]) * t
        return segment_clear(world, sx, sy, ax, ay)

    if visible(enter):
        return enter
    # LoS can change only at a wall-corner alignment while endpoints stay on floor.
    # Split at all such alignments rather than sampling across an occluded contact.
    sx, sy = s1[0] - s0[0], s1[1] - s0[1]
    cuts = {enter, leave}
    xs = [s0[0], s1[0], a0[0], a1[0]]
    ys = [s0[1], s1[1], a0[1], a1[1]]
    for cy in range(floor(min(ys)), floor(max(ys)) + 2):
        for cx in range(floor(min(xs)), floor(max(xs)) + 2):
            qx, qy = cx - s0[0], cy - s0[1]
            for t in roots(
                -vx * sy + vy * sx, vx * qy - vy * qx - rx * sy + ry * sx, rx * qy - ry * qx
            ):
                if enter < t < leave:
                    cuts.add(t)
    ordered = sorted(cuts)
    for lo, hi in zip(ordered, ordered[1:]):
        if visible((lo + hi) / 2):
            # Corner tangency itself is occluded; find the first visible point to float precision.
            upper = (lo + hi) / 2
            for _ in range(40):
                mid = (lo + upper) / 2
                if visible(mid):
                    upper = mid
                else:
                    lo = mid
            return upper
    return leave if visible(leave) else None
