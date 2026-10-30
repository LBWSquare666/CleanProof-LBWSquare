"""Wipe trajectories: generate the xy path (base frame) of the polishing disc center inside a zone rectangle, densified at a fixed step.

- zigzag: back and forth along y, change row along x, row spacing = disc radius (50% overlap of the diameter);
- spiral: spiral from the outside in over the rectangle, good edge coverage;
- edge_trace: walk once around the zone boundary, specifically to fix missed edges;
- spot: small back-and-forth over a residue box (spot_scrub).
margin is the minimum distance from the disc center to the zone boundary; it can be one number, or given per side (x_min side, y_min side, x_max side, y_max side);
when a neighboring zone is a forbidden zone, the caller sets the margin of that side to ≥ the disc radius, so the disc never enters the forbidden zone.
"""

from __future__ import annotations

import numpy as np

from ..common.zones import ZoneRect


def densify(points: np.ndarray, step: float = 0.002) -> np.ndarray:
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)


Margin = "float | tuple[float, float, float, float]"


def _inner(rect: ZoneRect, margin) -> tuple[float, float, float, float]:
    m = (margin,) * 4 if np.isscalar(margin) else tuple(margin)
    x0, x1 = rect.xmin + m[0], rect.xmax - m[2]
    y0, y1 = rect.ymin + m[1], rect.ymax - m[3]
    if x0 > x1:
        x0 = x1 = (rect.xmin + rect.xmax) / 2
    if y0 > y1:
        y0 = y1 = (rect.ymin + rect.ymax) / 2
    return x0, y0, x1, y1


def zigzag(rect: ZoneRect, pad_radius: float, margin: Margin) -> np.ndarray:
    x0, y0, x1, y1 = _inner(rect, margin)
    n_lines = max(1, int(np.ceil((x1 - x0) / pad_radius)) + 1)
    xs = np.linspace(x0, x1, n_lines)
    pts = []
    for i, x in enumerate(xs):
        ya, yb = (y0, y1) if i % 2 == 0 else (y1, y0)
        pts += [[x, ya], [x, yb]]
    return densify(np.array(pts))


def spiral(rect: ZoneRect, pad_radius: float, margin: Margin) -> np.ndarray:
    """Rectangular spiral: each turn goes around the x0 side, y1 side, x1 side and y0 side, then shrinks inward by one disc radius."""
    x0, y0, x1, y1 = _inner(rect, margin)
    r = pad_radius
    pts = [[x0, y0]]
    while True:
        pts += [[x0, y1], [x1, y1], [x1, y0]]
        nx0, ny0, nx1, ny1 = x0 + r, y0 + r, x1 - r, y1 - r
        if nx0 > nx1 or ny0 > ny1:
            pts.append([(x0 + x1) / 2, y0])
            break
        pts += [[nx0, y0], [nx0, ny0]]
        x0, y0, x1, y1 = nx0, ny0, nx1, ny1
    return densify(np.array(pts))


def edge_trace(rect: ZoneRect, pad_radius: float, margin: Margin) -> np.ndarray:
    x0, y0, x1, y1 = _inner(rect, margin)
    return densify(np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0], [x0, y0]]))


def spot(rect: ZoneRect, pad_radius: float, margin: Margin, bbox_norm: list[float]) -> np.ndarray:
    sub = rect.sub_rect(bbox_norm)
    # When the residue box is smaller than the disc diameter, do one disc-diameter-sized back-and-forth around the box center
    cx, cy = sub.center
    hx = max(sub.size[0] / 2, pad_radius * 0.5)
    hy = max(sub.size[1] / 2, pad_radius * 0.5)
    box = ZoneRect(rect.zone_id, cx - hx, cy - hy, cx + hx, cy + hy)
    x0, y0, x1, y1 = _inner(rect, margin)
    bx0, by0 = min(max(box.xmin, x0), x1), min(max(box.ymin, y0), y1)
    bx1, by1 = max(min(box.xmax, x1), x0), max(min(box.ymax, y1), y0)
    return zigzag(ZoneRect(rect.zone_id, bx0, by0, bx1, by1), pad_radius * 0.6, 0.0)


PATTERNS = {"zigzag": zigzag, "spiral": spiral, "edge_trace": edge_trace}


def path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0
