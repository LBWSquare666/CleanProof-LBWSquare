"""Per-tick safety checks (limits come from the safety section of configs/robot.yaml, i.e. the real-robot conventions from our earlier UR3e force-control work).

Results come in three levels:
- hold: free-space / hover overload; only lifting off along the tool axis is allowed (skill aborts and retracts);
- fault: latched fault, motion stops, manual reset required;
- fatal: robot not in running mode or in protective stop, stop immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .io import ROBOT_MODE_RUNNING, SAFETY_MODE_NORMAL, RobotIO
from .kinematics import robot_max_radius, robot_radius


@dataclass
class SafetyEvent:
    level: str      # hold | fault | fatal
    reason: str


class SafetyMonitor:
    def __init__(self, cfg: dict[str, Any], dt: float):
        self.cfg = cfg
        self.dt = dt
        self._tangent_ticks = 0
        self._neg_ticks = 0
        self._neg_need = max(1, int(round(float(cfg["negative_baseline_s"]) / dt)))

    def reset(self) -> None:
        self._tangent_ticks = 0
        self._neg_ticks = 0

    def check(self, robot: RobotIO, pose: np.ndarray, wrench_comp: np.ndarray, f_n: float,
              in_contact_phase: bool) -> Optional[SafetyEvent]:
        c = self.cfg
        if robot.get_robot_mode() != ROBOT_MODE_RUNNING:
            return SafetyEvent("fatal", f"robot_mode={robot.get_robot_mode()}, not in running mode")
        if robot.get_safety_mode() != SAFETY_MODE_NORMAL:
            return SafetyEvent("fatal", f"safety_mode={robot.get_safety_mode()} (protective stop or E-stop)")
        r, r_max = robot_radius(robot, pose), robot_max_radius(robot, c)
        if r > r_max:
            return SafetyEvent("fault", f"TCP horizontal radius {r:.3f} m exceeds {r_max} m")
        if pose[2] < float(c["min_z_m"]):
            return SafetyEvent("fault", f"TCP z={pose[2]:.3f} m below lower limit")
        torque = float(np.linalg.norm(wrench_comp[3:]))
        if torque > float(c["max_torque_nm"]):
            return SafetyEvent("fault", f"torque {torque:.2f} N·m over limit")
        f_vec = wrench_comp[:3]
        tangent = float(np.sqrt(max(0.0, float(np.dot(f_vec, f_vec)) - f_n**2)))
        if in_contact_phase:
            if f_n > float(c["contact_max_normal_force_n"]):
                return SafetyEvent("fault", f"contact normal force {f_n:.1f} N exceeds {c['contact_max_normal_force_n']} N")
            if tangent > float(c["max_tangent_force_n"]):
                return SafetyEvent("fault", f"tangential force {tangent:.1f} N over limit")
        else:
            if f_n > float(c["free_max_normal_force_n"]):
                return SafetyEvent("hold", f"free-space normal force {f_n:.1f} N exceeds {c['free_max_normal_force_n']} N")
            self._tangent_ticks = self._tangent_ticks + 1 if tangent > float(c["max_tangent_force_n"]) else 0
            if self._tangent_ticks >= 2:
                return SafetyEvent("hold", f"free-space tangential force {tangent:.1f} N over limit for several ticks")
        self._neg_ticks = self._neg_ticks + 1 if f_n < float(c["negative_baseline_n"]) else 0
        if self._neg_ticks >= self._neg_need:
            return SafetyEvent("hold", f"normal force stays negative ({f_n:.1f} N), tare may have failed")
        return None
