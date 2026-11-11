"""Shared circle motion and exact straight-segment clearance on the 2D grid."""

from math import floor, sqrt

from game.interaction import segment_clear


def sweep_axis(world, radius, position, other, delta, *, horizontal):
    extent = world.width if horizontal else world.height
    target = min(max(position + delta, radius), extent - radius)
    if delta == 0:
        return target
    for across in range(floor(other - radius), floor(other + radius) + 1):
        gap = abs(other - min(max(other, across), across + 1))
        # The same slack as WorldMap.overlaps: a body resting on a wall row only touches it,
        # even when rounding puts it a hair inside, so the row's seams never stop a slide.
        if gap * gap >= radius * radius - 1e-12:
            continue
        padding = sqrt(max(0.0, radius * radius - gap * gap))
        for along in range(
            floor(min(position, target) - radius), floor(max(position, target) + radius) + 1
        ):
            cx, cy = (along, across) if horizontal else (across, along)
            if not world.blocked(cx, cy):
                continue
            lower, upper = along - padding, along + 1 + padding
            if delta > 0 and position <= lower + 1e-12 and target > lower:
                target = lower
            elif delta < 0 and position >= upper - 1e-12 and target < upper:
                target = upper
    return target


def segment_has_clearance(world, start, end, radius):
    x0, y0 = start
    x1, y1 = end
    if world.overlaps(x0, y0, radius) or world.overlaps(x1, y1, radius):
        return False
    if not segment_clear(world, x0, y0, x1, y1):
        return False
    dx, dy = x1 - x0, y1 - y0
    length2 = dx * dx + dy * dy
    for cy in range(floor(min(y0, y1) - radius), floor(max(y0, y1) + radius) + 1):
        for cx in range(floor(min(x0, x1) - radius), floor(max(x0, x1) + radius) + 1):
            if not world.blocked(cx, cy):
                continue
            distances = [
                (x - min(max(x, cx), cx + 1)) ** 2 + (y - min(max(y, cy), cy + 1)) ** 2
                for x, y in (start, end)
            ]
            for x, y in ((cx, cy), (cx + 1, cy), (cx, cy + 1), (cx + 1, cy + 1)):
                t = min(1.0, max(0.0, ((x - x0) * dx + (y - y0) * dy) / length2)) if length2 else 0
                distances.append((x - x0 - t * dx) ** 2 + (y - y0 - t * dy) ** 2)
            if min(distances) < radius * radius - 1e-12:
                return False
    return True
