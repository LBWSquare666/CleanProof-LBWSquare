"""Table stain model: per-cell stain concentration over the work area, removed by contact pressure and sliding distance.

Each stain type keeps its own concentration d ∈ [0, 1]. At each control tick, for the cells covered by the polishing pad:
    d ← d · exp(−rate · slip · g(F)),   g(F) = max(0, F − F0) / 5 N
where F is the equivalent normal force at the cell and slip is the tangential sliding distance of this tick.
F0 and rate can be set per table material (configs/sim.yaml dirt.types, as a number or {surface: value, default: value}),
consistent with the stain × surface force table (configs/force_table.yaml): the force in the table cleans in one pass, one level lower cannot wipe it off.
Types with a hard section (the v8 dried stains) have two layers: the surface layer is removed by this type's own parameters (a light wipe makes it fade),
the hard layer (fraction hard.fraction) is removed by the parameters of the hard section (enough force is required). level is the sum of the two layers (the visible concentration), hard is the hard layer within it.
Permanent types cannot be wiped off (scenario S3). The parameters are for simulation only and do not represent real materials.

Table appearance (v8): dirt.surfaces[surface] gives a single rgb color or a small texture in the repo, assets/surfaces/*.png with CC0 sources in
assets/surfaces/README.md. Perception rendering (rgb()) blends stains onto the table texture at render_scale times the resolution;
the "visible" ground truth is decided per cell by the ΔE against that cell's table base color. rgb_plain() is the same stains blended onto a flat color (used for Blender plate decals).
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import cv2
import numpy as np

from ..common.config import REPO_ROOT
from ..common.zones import ZoneGrid
from .contact import ContactPoint

PLAIN_SURFACE = "plain"


def per_surface(v: Any, surface: str) -> float:
    """Config value: a number, or {surface: value, default: value}."""
    if isinstance(v, dict):
        return float(v.get(surface, v["default"]))
    return float(v)


def load_surface_image(spec: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    """Table base image (H, W, 3) float32: rows along +x, columns along +y (same direction as the rectified image), scaled to shape."""
    if "texture" in spec:
        bgr = cv2.imread(str(REPO_ROOT / spec["texture"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"table texture does not exist: {spec['texture']} (generate it with scripts/make_surface_textures.py)")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != shape:
            img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        return img.astype(np.float32)
    return np.broadcast_to(np.array(spec["rgb"], np.float32), shape + (3,)).copy()


class DirtMap:
    def __init__(self, grid: ZoneGrid, cfg: dict[str, Any], surface: Optional[str] = None):
        self.grid = grid
        self.res = float(cfg["resolution_m"])
        self.clean_level = float(cfg["clean_level"])
        self.zone_clean_max_fraction = float(cfg["zone_clean_max_fraction"])
        # ground truth is decided by "visible": a cell counts as dirty only if the CIE76 ΔE between its rendered color and the table exceeds this (None falls back to a concentration test).
        # residue at concentration 0.1-0.25 has a ΔE of only 5-11, almost invisible in the rendered image, so no verifier could tell.
        vde = cfg.get("visible_delta_e")
        self.visible_delta_e = float(vde) if vde is not None else None
        self.types = list(cfg["types"])
        self.params = cfg["types"]
        self.surface_rgb = np.array(cfg["surface_rgb"], dtype=float)     # flat base color (rgb_plain / Blender decals)
        self.nx = int(round(grid.size[0] / self.res))
        self.ny = int(round(grid.size[1] / self.res))
        self.level = np.zeros((len(self.types), self.nx, self.ny), dtype=np.float32)
        self.hard = np.zeros_like(self.level)                              # hard layer (≤ level); non-zero only for types with a hard section
        xs = grid.origin[0] + (np.arange(self.nx) + 0.5) * self.res
        ys = grid.origin[1] + (np.arange(self.ny) + 0.5) * self.res
        self.cx, self.cy = np.meshgrid(xs, ys, indexing="ij")
        self.wiped = np.zeros((self.nx, self.ny), dtype=np.float32)   # accumulated sliding distance (coverage statistics)
        self.render_scale = int(cfg.get("render_scale", 1))
        self.set_surface(surface or cfg.get("default_surface", PLAIN_SURFACE), cfg.get("surfaces", {}))

    # ---- table ----
    def set_surface(self, surface: str, surfaces: dict[str, Any]) -> None:
        spec = surfaces.get(surface)
        if spec is None:
            if surface != PLAIN_SURFACE:
                raise KeyError(f"configs/sim.yaml dirt.surfaces has no table {surface!r}, choose from {sorted(surfaces)}")
            spec = {"rgb": self.surface_rgb.tolist()}
        self.surface = surface
        k = self.render_scale
        self.base_hr = load_surface_image(spec, (self.nx * k, self.ny * k))                       # render resolution
        self.base = cv2.resize(self.base_hr, (self.ny, self.nx), interpolation=cv2.INTER_AREA)  # cell resolution
        self._rate = np.zeros(len(self.types), np.float32)
        self._f0 = np.zeros(len(self.types), np.float32)
        self._hard: list[Optional[tuple[float, float]]] = []              # hard layer (rate, F0), None when there is no hard layer
        for t, kind in enumerate(self.types):
            p = self.params[kind]
            self._rate[t] = per_surface(p["rate_per_m"], surface)
            self._f0[t] = per_surface(p["force_threshold_n"], surface)
            h = p.get("hard")
            self._hard.append((per_surface(h["rate_per_m"], surface), per_surface(h["force_threshold_n"], surface))
                              if h else None)

    @property
    def render_shape(self) -> tuple[int, int]:
        return self.nx * self.render_scale, self.ny * self.render_scale

    # ---- placing stains ----
    def _tid(self, kind: str) -> int:
        try:
            return self.types.index(kind)
        except ValueError as e:
            raise KeyError(f"unknown stain type {kind!r}, choose from {self.types}") from e

    def _place(self, t: int, amount: np.ndarray) -> None:
        self.level[t] = np.maximum(self.level[t], amount)
        h = self.params[self.types[t]].get("hard")
        if h:
            self.hard[t] = np.maximum(self.hard[t], float(h["fraction"]) * amount)

    def add_blob(self, kind: str, center_xy: Iterable[float], radius: float, level: float = 1.0, seed: int = 0) -> None:
        """Irregular round stain (the radius varies randomly with angle)."""
        cx, cy = center_xy
        rng = np.random.default_rng(seed)
        harmonics = rng.normal(0, 0.12, 4)
        phases = rng.uniform(0, 2 * np.pi, 4)
        ang = np.arctan2(self.cy - cy, self.cx - cx)
        r_edge = radius * (1 + sum(h * np.cos((k + 2) * ang + ph) for k, (h, ph) in enumerate(zip(harmonics, phases))))
        dist = np.hypot(self.cx - cx, self.cy - cy)
        mask = np.clip((r_edge - dist) / (0.15 * radius), 0, 1)
        self._place(self._tid(kind), level * mask)

    def add_rect(self, kind: str, xmin: float, ymin: float, xmax: float, ymax: float, level: float = 1.0) -> None:
        mask = ((self.cx >= xmin) & (self.cx <= xmax) & (self.cy >= ymin) & (self.cy <= ymax)).astype(np.float32)
        self._place(self._tid(kind), level * mask)

    def add_strokes(self, kind: str, center_xy: Iterable[float], length: float, n: int = 3, spacing: float = 0.008,
                    width: float = 0.0035, angle_deg: float = 0.0, level: float = 1.0, seed: int = 0) -> None:
        """Marker strokes: n parallel lines with slight curvature and tapered ends (center center_xy, length length, line width width, in m)."""
        cx, cy = center_xy
        rng = np.random.default_rng(seed)
        a = np.deg2rad(angle_deg)
        ux, uy = np.cos(a), np.sin(a)                       # stroke direction
        px, py = -uy, ux                                    # normal
        dx, dy = self.cx - cx, self.cy - cy
        s = dx * ux + dy * uy                               # along the stroke
        q = dx * px + dy * py                               # across the stroke
        out = np.zeros_like(self.cx, dtype=np.float32)
        for k in range(n):
            off = (k - (n - 1) / 2) * spacing + rng.normal(0, 0.0008)
            tilt = np.tan(np.deg2rad(rng.uniform(-14, 14)))          # each stroke leans a little on its own, not neat parallel lines
            bend = rng.uniform(-0.004, 0.004)
            half = length / 2 * rng.uniform(0.7, 1.0)
            sk = s - rng.uniform(-0.004, 0.004)
            centre = off + tilt * sk + bend * (1 - (sk / max(half, 1e-6)) ** 2)
            taper = np.clip((half - np.abs(sk)) / (0.25 * half), 0, 1) ** 0.5
            w = width / 2 * (0.6 + 0.4 * taper)
            m = np.clip((w - np.abs(q - centre)) / 0.0008, 0, 1) * (np.abs(sk) <= half)
            out = np.maximum(out, m * rng.uniform(0.9, 1.0))
        self._place(self._tid(kind), level * out)

    def add_hand(self, kind: str, center_xy: Iterable[float], scale: float = 1.0, angle_deg: float = 0.0,
                 level: float = 1.0, edge_boost: float = 0.35, seed: int = 0) -> None:
        """Greasy handprint: palm ellipse + four fingers + thumb (capsule shapes), with a slightly heavier rim (the matte boundary at the outer edge of the oil film).
        At scale = 1 the whole hand is about 0.17 m long (palm width 0.08 m); angle_deg is the angle between the finger direction and +x."""
        cx, cy = center_xy
        rng = np.random.default_rng(seed)
        a = np.deg2rad(angle_deg)
        ux, uy = np.cos(a), np.sin(a)                       # finger direction
        px, py = -uy, ux
        dx, dy = self.cx - cx, self.cy - cy
        s_ = (dx * ux + dy * uy) / scale                    # along the finger (meters, normalized to scale = 1)
        q_ = (dx * px + dy * py) / scale                    # across
        soft = 0.0025 / scale
        # palm: rounded rectangle (superellipse), ±0.045 along the finger direction, ±0.040 across
        palm = np.clip((1.0 - ((np.abs(s_) / 0.045) ** 3 + (np.abs(q_) / 0.040) ** 3) ** (1 / 3)) * 0.040 / soft, 0, 1)
        out = palm

        def capsule(s0: float, q0: float, ang: float, length: float, r: float) -> np.ndarray:
            ca, sa = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
            ds, dq = s_ - s0, q_ - q0
            t = np.clip(ds * ca + dq * sa, 0, length)
            d = np.hypot(ds - t * ca, dq - t * sa)
            return np.clip((r - d) / soft, 0, 1)

        # four fingers (little finger → index finger): start at the upper edge of the palm, about 3 mm gap between fingers; round tips
        fingers = ((0.030, -0.030, -9, 0.058, 0.0085), (0.036, -0.0105, -3, 0.074, 0.0095),
                   (0.036, 0.0095, 2, 0.078, 0.0097), (0.032, 0.0290, 8, 0.068, 0.0092))
        for s0, q0, ang, ln, r in fingers:
            jitter = rng.normal(0, 0.0012)
            out = np.maximum(out, capsule(s0, q0 + jitter, ang + rng.normal(0, 2.0), ln * rng.uniform(0.94, 1.0), r))
        out = np.maximum(out, capsule(-0.012, 0.034, 48, 0.062, 0.0115))        # thumb
        mask8 = (out > 0.5).astype(np.uint8)
        dist = cv2.distanceTransform(mask8, cv2.DIST_L2, 3) * self.res / scale  # distance to the edge (meters, scale = 1)
        rim = np.clip(1.0 - dist / 0.005, 0, 1) * mask8                          # a ring about 5 mm wide at the outer edge
        tex = cv2.GaussianBlur(rng.normal(0, 1, out.shape).astype(np.float32), (0, 0), 1.5)
        tex = 1.0 + 0.12 * tex / max(1e-6, float(tex.std()))                     # uneven oil film (patchy light and dark, not noise)
        lv = np.clip(out * (1.0 - edge_boost) * tex + rim * edge_boost, 0, 1)
        self._place(self._tid(kind), level * lv.astype(np.float32))

    def clear(self) -> None:
        self.level[:] = 0.0
        self.hard[:] = 0.0

    # ---- wiping ----
    def step(self, points: list[ContactPoint], pad_center_xy: np.ndarray, pad_radius: float, slip_m: float) -> None:
        if slip_m <= 0 or not any(p.normal_force > 0 for p in points):
            return
        cx, cy = pad_center_xy
        i0 = max(0, int((cx - pad_radius - self.grid.origin[0]) / self.res))
        i1 = min(self.nx, int((cx + pad_radius - self.grid.origin[0]) / self.res) + 1)
        j0 = max(0, int((cy - pad_radius - self.grid.origin[1]) / self.res))
        j1 = min(self.ny, int((cy + pad_radius - self.grid.origin[1]) / self.res) + 1)
        if i0 >= i1 or j0 >= j1:
            return
        X, Y = self.cx[i0:i1, j0:j1], self.cy[i0:i1, j0:j1]
        inside = np.hypot(X - cx, Y - cy) <= pad_radius
        if not inside.any():
            return
        # equivalent normal force at a cell = force at the nearest sample point × number of sample points (equals the total normal force when the load is uniform);
        # points contains all sample points, and points not in contact have zero normal force
        pxy = np.array([p.xy for p in points])
        pf = np.array([p.normal_force for p in points])
        d2 = (X[..., None] - pxy[:, 0]) ** 2 + (Y[..., None] - pxy[:, 1]) ** 2
        f_cell = pf[np.argmin(d2, axis=-1)] * len(points)
        for t in range(len(self.types)):
            lv = self.level[t, i0:i1, j0:j1]
            hd = self.hard[t, i0:i1, j0:j1]
            np.minimum(hd, lv, out=hd)                      # when level is zeroed externally, the hard layer is invalidated as well
            soft = lv - hd
            if self._rate[t] > 0:
                g = np.maximum(0.0, f_cell - self._f0[t]) / 5.0
                soft = np.where(inside, soft * np.exp(-self._rate[t] * slip_m * g), soft)
            hard = self._hard[t]
            if hard is not None and hard[0] > 0:
                gh = np.maximum(0.0, f_cell - hard[1]) / 5.0
                hd[inside] *= np.exp(-hard[0] * slip_m * gh)[inside]
            lv[...] = soft + hd
        self.wiped[i0:i1, j0:j1][inside] += slip_m

    # ---- statistics ----
    def _zone_slices(self, zone_id: str) -> tuple[slice, slice]:
        z = self.grid[zone_id]
        i0 = int(round((z.xmin - self.grid.origin[0]) / self.res))
        i1 = int(round((z.xmax - self.grid.origin[0]) / self.res))
        j0 = int(round((z.ymin - self.grid.origin[1]) / self.res))
        j1 = int(round((z.ymax - self.grid.origin[1]) / self.res))
        return slice(i0, i1), slice(j0, j1)

    def _mix(self, base: np.ndarray, lv: np.ndarray) -> np.ndarray:
        img = np.array(base, dtype=np.float32)
        for t, kind in enumerate(self.types):
            a = np.clip(lv[t], 0, 1)[..., None]
            img = img * (1 - a) + np.array(self.params[kind]["rgb"], dtype=np.float32) * a
        return img

    def _delta_e(self, lv: np.ndarray, base: Optional[np.ndarray] = None) -> np.ndarray:
        """(types, h, w) concentration → CIE76 ΔE between each cell's rendered color and that cell's table base color (blended the same way as rgb()).
        base: table base colors (h, w, 3) of the same shape; when None the flat base surface_rgb is used."""
        if base is None:
            base = np.broadcast_to(self.surface_rgb.astype(np.float32), lv.shape[1:] + (3,))
        img = self._mix(base, lv)
        lab = cv2.cvtColor(np.clip(img, 0, 255) / 255.0, cv2.COLOR_RGB2LAB)
        ref = cv2.cvtColor(np.clip(np.asarray(base, np.float32), 0, 255) / 255.0, cv2.COLOR_RGB2LAB)
        return np.linalg.norm(lab - ref, axis=-1)

    def zone_stats(self, zone_id: str) -> dict[str, Any]:
        si, sj = self._zone_slices(zone_id)
        lv = self.level[:, si, sj]
        total = lv.max(axis=0)
        trace = total > self.clean_level
        if self.visible_delta_e is not None:
            dirty = self._delta_e(lv, self.base[si, sj]) > self.visible_delta_e
        else:
            dirty = trace
        frac = float(dirty.mean())
        by_type = {k: float((lv[t] > self.clean_level).mean()) for t, k in enumerate(self.types)}
        bbox = None
        if dirty.any():
            ii, jj = np.nonzero(dirty)
            n_i, n_j = dirty.shape
            bbox = [ii.min() / n_i, jj.min() / n_j, (ii.max() + 1) / n_i, (jj.max() + 1) / n_j]
        return {
            "zone_id": zone_id,
            "residue_fraction": frac,                     # fraction of visible residue (the ground-truth basis)
            "trace_fraction": float(trace.mean()),        # fraction with concentration above clean_level (including invisible traces)
            "max_level": float(total.max()),
            "clean": frac <= self.zone_clean_max_fraction,
            "by_type": by_type,
            "bbox_norm": bbox,
            "coverage": float((self.wiped[si, sj] > 0).mean()),
        }

    def all_stats(self) -> dict[str, dict[str, Any]]:
        return {z: self.zone_stats(z) for z in self.grid.ids}

    def rgb(self) -> np.ndarray:
        """For perception rendering: (nx·k, ny·k, 3) uint8, stains blended onto the table base image by per-type concentration (k = render_scale, rows along x, columns along y)."""
        k = self.render_scale
        lv = self.level
        if k > 1:
            lv = np.stack([cv2.resize(l, (self.ny * k, self.nx * k), interpolation=cv2.INTER_LINEAR) for l in lv])
        return np.clip(self._mix(self.base_hr, lv), 0, 255).astype(np.uint8)

    def rgb_plain(self) -> np.ndarray:
        """(nx, ny, 3) uint8: stains blended onto the flat base color surface_rgb (the Blender plate takes decals as "texture / base color", so it must be a flat base)."""
        base = np.broadcast_to(self.surface_rgb.astype(np.float32), (self.nx, self.ny, 3))
        return np.clip(self._mix(base, self.level), 0, 255).astype(np.uint8)
