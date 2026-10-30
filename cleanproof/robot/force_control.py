"""Force control primitives: software tare, normal force computation, mass-damper teacher law, target force ramp.

Independent implementation of the control law our earlier UR3e work already uses on the real robot, parameters from configs/robot.yaml:
- normal n = −R·z_tool (pointing out of the tabletop), F_n = n·F_comp, compression is positive;
- teacher law (implicit Euler): v ← (M·v + dt·(F_ref − F_n)) / (M + B·dt), with acceleration and velocity limits,
  outputs a velocity along tool +Z (pressing onto the tabletop), translation only;
- the tare bias is stored in the tool frame and rotated back to the base frame by the current pose every tick before subtracting.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .kinematics import rotvec_to_matrix


def tool_axes(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (R_base_tool, inward): inward = tool +Z (the direction pressing onto the tabletop)."""
    R = rotvec_to_matrix(pose[3:6])
    return R, R[:, 2]


class SoftwareTare:
    """Take the window mean while hovering still as the bias (stored in the tool frame)."""

    def __init__(self, cfg: dict[str, Any], control_dt: float):
        self.window = max(1, int(round(float(cfg["window_s"]) / control_dt)))
        self.max_abs_mean = float(cfg["max_abs_mean_n"])
        self.max_std = float(cfg["max_std_n"])
        self.max_speed = float(cfg["max_speed_m_s"])
        self._buf: deque[np.ndarray] = deque(maxlen=self.window)
        self.bias_tool = np.zeros(6)
        self.valid = False

    def add_sample(self, wrench_base: np.ndarray, pose: np.ndarray, tcp_speed: np.ndarray) -> None:
        if np.linalg.norm(tcp_speed[:3]) > self.max_speed:
            self._buf.clear()
            return
        R = rotvec_to_matrix(pose[3:6])
        self._buf.append(np.concatenate([R.T @ wrench_base[:3], R.T @ wrench_base[3:]]))

    def ready(self) -> bool:
        return len(self._buf) >= self.window

    def commit(self) -> tuple[bool, str]:
        """Adopt a new bias when the window is full and the readings are steady; otherwise keep the old bias and say why (existing real-robot rule)."""
        if not self.ready():
            return False, "tare window not full"
        arr = np.array(self._buf)
        mean, std = arr.mean(axis=0), arr[:, :3].std(axis=0)
        if np.abs(mean[:3]).max() > self.max_abs_mean:
            return False, f"hover reading mean {np.abs(mean[:3]).max():.2f} N is too large (maybe already in contact), not taring"
        if std.max() > self.max_std:
            return False, f"hover reading std {std.max():.2f} N is too large, not taring"
        self.bias_tool = mean
        self.valid = True
        return True, "ok"

    def compensate(self, wrench_base: np.ndarray, pose: np.ndarray) -> np.ndarray:
        R = rotvec_to_matrix(pose[3:6])
        return wrench_base - np.concatenate([R @ self.bias_tool[:3], R @ self.bias_tool[3:]])


def normal_force(wrench_comp: np.ndarray, pose: np.ndarray) -> float:
    _, inward = tool_axes(pose)
    return float(np.dot(-inward, wrench_comp[:3]))


@dataclass
class ForceTeacher:
    mass: float
    damping: float
    vmax: float
    amax: float
    dt: float
    v: float = 0.0

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any], dt: float) -> "ForceTeacher":
        return cls(float(cfg["mass_kg"]), float(cfg["damping_ns_m"]), float(cfg["max_normal_speed_m_s"]),
                   float(cfg["max_normal_accel_m_s2"]), dt)

    def reset(self) -> None:
        self.v = 0.0

    def step(self, f_ref: float, f_n: float) -> float:
        """Return the velocity along inward (tool +Z); > 0 means pressing onto the tabletop."""
        v_new = (self.mass * self.v + self.dt * (f_ref - f_n)) / (self.mass + self.damping * self.dt)
        dv = np.clip(v_new - self.v, -self.amax * self.dt, self.amax * self.dt)
        self.v = float(np.clip(self.v + dv, -self.vmax, self.vmax))
        return self.v


@dataclass
class ForceRamp:
    """Target force ramp: approach the target from the start at the up/down rates."""

    up_n_s: float
    down_n_s: float
    dt: float
    value: float = 0.0
    target: float = 0.0

    def start(self, value: float, target: float) -> None:
        self.value, self.target = float(value), float(target)

    def step(self) -> float:
        d = self.target - self.value
        rate = self.up_n_s if d > 0 else self.down_n_s
        self.value += float(np.clip(d, -rate * self.dt, rate * self.dt))
        return self.value


class HoldTimer:
    """Fires only after the condition has held continuously for duration seconds (debounce)."""

    def __init__(self, duration_s: float, dt: float):
        self.need = max(1, int(round(duration_s / dt)))
        self.count = 0

    def update(self, cond: bool) -> bool:
        self.count = self.count + 1 if cond else 0
        return self.count >= self.need

    def reset(self) -> None:
        self.count = 0


class PushInWindow:
    """Mean pressing-in speed along inward over the latest window (contact confirmation threshold, ≥ 1 mm/s)."""

    def __init__(self, window_s: float, dt: float):
        self._buf: deque[float] = deque(maxlen=max(1, int(round(window_s / dt))))

    def add(self, tcp_speed: np.ndarray, inward: np.ndarray) -> None:
        self._buf.append(float(np.dot(tcp_speed[:3], inward)))

    def mean(self) -> Optional[float]:
        return float(np.mean(self._buf)) if self._buf else None

    def clear(self) -> None:
        self._buf.clear()
