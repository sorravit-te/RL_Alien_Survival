"""Deterministic yaw/range targeting and conservative wall visibility, no renderer."""

from math import atan2, floor, hypot, pi


def segment_clear(world, x0, y0, x1, y1):
    """Closed segment vs solid tile boxes; touching a solid corner is occluded."""
    dx, dy = x1 - x0, y1 - y0
    for cy in range(floor(min(y0, y1)), floor(max(y0, y1)) + 1):
        for cx in range(floor(min(x0, x1)), floor(max(x0, x1)) + 1):
            if not world.blocked(cx, cy):
                continue
            enter, leave = 0.0, 1.0
            for origin, delta, lower in ((x0, dx, cx), (y0, dy, cy)):
                if delta == 0:
                    if not lower <= origin <= lower + 1:
                        enter, leave = 1.0, 0.0
                        break
                else:
                    a, b = (lower - origin) / delta, (lower + 1 - origin) / delta
                    enter, leave = max(enter, min(a, b)), min(leave, max(a, b))
            if enter <= leave:
                return False
    return True


def select_target(world, survivor, config, has_keycard):
    candidates = []
    for obj in world.objects:
        if obj.kind == "keycard" and has_keycard:
            continue
        dx, dy = obj.x - survivor.x, obj.y - survivor.y
        distance = hypot(dx, dy)
        if distance > config.interaction_range:
            continue
        angle = abs((atan2(dy, dx) - survivor.yaw + pi) % (2 * pi) - pi) if distance else 0
        if angle > config.interaction_half_angle:
            continue
        if segment_clear(world, survivor.x, survivor.y, obj.x, obj.y):
            candidates.append((angle, distance, obj.object_id, obj))
    return min(candidates, key=lambda c: c[:3])[3] if candidates else None
