"""Virtual table contact model: spring-damper normal force and Coulomb friction at the polishing pad center plus several edge points.

The approach matches the numpy contact sandbox from our earlier UR3e work, reimplemented independently in this repo.
Both the mujoco backend and the ursim backend use it to produce F/T readings, so the force behavior is the same in both simulations.

The sign convention matches the real robot: it returns the force the environment exerts on the tool (base frame). Pressing the tool into the table gives Fz > 0,
so the F_n = n·F (n = −R·z_tool, pointing out of the table) is positive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Obstacle:
    """A raised rectangular region on the table (for scenario S5: hitting an obstacle / abnormal contact)."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    height: float


@dataclass
class ContactPoint:
    xy: np.ndarray
    normal_force: float
    tangential_speed: float


@dataclass
class ContactResult:
    wrench: np.ndarray                     # [Fx, Fy, Fz, Tx, Ty, Tz], base frame, torque about the TCP
    points: list[ContactPoint] = field(default_factory=list)   # all sample points; points not in contact have zero normal force

    @property
    def normal_force(self) -> float:
        return float(sum(p.normal_force for p in self.points))

    @property
    def in_contact(self) -> bool:
        return any(p.normal_force > 0 for p in self.points)


WALL_STIFFNESS_N_M = 20000.0   # lateral stiffness of the obstacle side wall
CLIMB_TOLERANCE_M = 0.003      # steps below this height are treated as ramps (can be pressed over), higher ones as side walls


class VirtualSurface:
    def __init__(self, z_surface: float, cfg: dict[str, Any], obstacles: list[Obstacle] | None = None):
        self.z_surface = float(z_surface)
        self.k = float(cfg["stiffness_n_m"])
        self.c = float(cfg["damping_ns_m"])
        self.mu = float(cfg["friction_mu"])
        self.radius = float(cfg["pad_radius_m"])
        self.n_edge = int(cfg["edge_points"])
        self.noise = float(cfg.get("ft_noise_n", 0.0))
        self.rng = np.random.default_rng(cfg.get("seed", 0))
        self.obstacles = list(obstacles or [])
        ang = np.linspace(0, 2 * np.pi, self.n_edge, endpoint=False)
        edge = np.stack([self.radius * np.cos(ang), self.radius * np.sin(ang), np.zeros_like(ang)], axis=1)
        self._pad_local = np.vstack([np.zeros((1, 3)), edge])     # sample points in the tool frame (TCP at the center of the pad bottom)
        self._k_i = self.k / len(self._pad_local)
        self._c_i = self.c / len(self._pad_local)

    def height(self, x: float, y: float, pad_bottom_z: float | None = None) -> float:
        """Support surface height at (x, y). When the pad bottom is below the obstacle top (beyond the climb tolerance), the obstacle acts only as a side wall and gives no top support."""
        h = self.z_surface
        for o in self.obstacles:
            top = self.z_surface + o.height
            if o.xmin <= x <= o.xmax and o.ymin <= y <= o.ymax:
                if pad_bottom_z is None or o.height <= CLIMB_TOLERANCE_M or pad_bottom_z >= top - CLIMB_TOLERANCE_M:
                    h = max(h, top)
        return h

    def _wall_force(self, p0: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lateral force and torque (about the TCP) when the side of the polishing pad (a circle) runs into an obstacle rectangle."""
        F, Tq = np.zeros(3), np.zeros(3)
        for o in self.obstacles:
            top = self.z_surface + o.height
            if o.height <= CLIMB_TOLERANCE_M or p0[2] >= top - CLIMB_TOLERANCE_M:
                continue
            c = p0[:2]
            q = np.array([np.clip(c[0], o.xmin, o.xmax), np.clip(c[1], o.ymin, o.ymax)])
            d_vec = c - q
            d = float(np.linalg.norm(d_vec))
            if d >= self.radius:
                continue
            if d < 1e-9:  # pad center is already inside the obstacle: push out along the nearest edge
                dists = [c[0] - o.xmin, o.xmax - c[0], c[1] - o.ymin, o.ymax - c[1]]
                k = int(np.argmin(dists))
                n = [np.array([-1, 0]), np.array([1, 0]), np.array([0, -1]), np.array([0, 1])][k]
                pen = self.radius + dists[k]
            else:
                n = d_vec / d
                pen = self.radius - d
            vn = float(np.dot(v[:2], n))
            f_xy = n * max(0.0, WALL_STIFFNESS_N_M * pen - self.c * vn)
            f = np.array([f_xy[0], f_xy[1], 0.0])
            rim = np.array([q[0], q[1], p0[2]])
            F += f
            Tq += np.cross(rim - p0, f)
        return F, Tq

    def compute(self, T_tcp: np.ndarray, twist: np.ndarray, add_noise: bool = True) -> ContactResult:
        R, p0 = T_tcp[:3, :3], T_tcp[:3, 3]
        v, w = np.asarray(twist[:3], float), np.asarray(twist[3:], float)
        F = np.zeros(3)
        Tq = np.zeros(3)
        points: list[ContactPoint] = []
        for pl in self._pad_local:
            p = p0 + R @ pl
            depth = self.height(p[0], p[1], p0[2]) - p[2]
            vp = v + np.cross(w, p - p0)
            if depth <= 0.0:
                points.append(ContactPoint(xy=p[:2].copy(), normal_force=0.0, tangential_speed=float(np.linalg.norm(vp[:2]))))
                continue
            fn = max(0.0, self._k_i * depth - self._c_i * vp[2])
            vt = vp[:2]
            speed = float(np.linalg.norm(vt))
            ft = np.zeros(2) if speed < 1e-9 else -self.mu * fn * np.tanh(speed / 1e-3) * vt / speed
            f = np.array([ft[0], ft[1], fn])
            F += f
            Tq += np.cross(p - p0, f)
            points.append(ContactPoint(xy=p[:2].copy(), normal_force=fn, tangential_speed=speed))
        if self.obstacles:
            Fw, Tw = self._wall_force(p0, v)
            F += Fw
            Tq += Tw
        wrench = np.concatenate([F, Tq])
        if add_noise and self.noise > 0:
            wrench[:3] += self.rng.normal(0.0, self.noise, 3)
        return ContactResult(wrench=wrench, points=points)
