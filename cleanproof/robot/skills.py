"""Whitelisted skill executor: survey / inspect_closeup / wipe_zone / spot_scrub.

Depends only on RobotIO; the mujoco / ursim / ur3e backends share the same code (the core of sim-to-real).
The contact state machine and its criteria are implemented independently following the real-robot conventions from our earlier UR3e force-control work:

    tare → approach → confirm → ramp → control (wipe along the trajectory) → release (automatic retract) → lift

Difference from our earlier UR3e work: there, contact is teleoperated by a person, who also moves the tool away after RELEASE; here approach, wiping and
retract are all automatic. This is new CleanProof logic and must be validated step by step on the real robot per PLAN §5.6.
SafetyMonitor runs every tick; any fault / fatal does speedStop first and then returns, and the finally block guarantees the stop.
"""

from __future__ import annotations

import time as _time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from ..common.config import load_yaml
from ..common.schemas import ForceSummary
from ..common.zones import ZoneGrid, ZoneRect
from .force_control import (
    ForceRamp,
    ForceTeacher,
    HoldTimer,
    PushInWindow,
    SoftwareTare,
    normal_force,
    tool_axes,
)
from .io import RobotIO
from .kinematics import elbow_margin_deg, robot_max_radius, robot_radius, rotvec_to_matrix
from .paths import PATTERNS, path_length, spot
from .safety import SafetyEvent, SafetyMonitor

STATE_CODES = {"tare": 1, "approach": 2, "confirm": 3, "ramp": 4, "control": 5, "reapproach": 6, "release": 7}


@dataclass
class SkillOutcome:
    skill: str
    status: str                      # ok | no_contact | aborted | fault | rejected
    reason: str = ""
    zone_ids: list[str] = field(default_factory=list)
    force: dict[str, dict[str, Any]] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    duration_s: float = 0.0
    trace: dict[str, list[float]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self, with_trace: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not with_trace:
            d.pop("trace")
        return d


class SkillExecutor:
    def __init__(self, robot: RobotIO, grid: Optional[ZoneGrid] = None, rcfg: Optional[dict[str, Any]] = None,
                 forbidden_zones: Sequence[str] = ()):
        self.robot = robot
        self.grid = grid or ZoneGrid()
        self.cfg = rcfg or load_yaml("robot.yaml")
        self.dt = robot.control_dt
        self.pad_r = float(self.cfg["tool"]["pad_radius_m"])
        self.forbidden = set(forbidden_zones)
        rq = getattr(robot, "home_q", None)                  # platform adaptation: the backend provides this platform's home (UR3e = original robot.yaml value)
        self.home_q = np.array(rq if rq is not None else self.cfg["home_q"], dtype=float)
        self.m = self.cfg["motion"]
        self.c = self.cfg["contact"]
        self.s = self.cfg["safety"]
        self.sk = self.cfg["skills"]
        self.acc = float(self.cfg["speedl"]["acceleration"])
        self.cmd_time = float(self.cfg["speedl"]["time"])
        self.fault: Optional[str] = None
        self.force_override_n: Optional[float] = None   # simulation ablation only (fixed force): replaces the planned target force at execution

    # ================= start-up checks =================
    def startup_problems(self) -> list[str]:
        st = self.cfg["startup"]
        r = self.robot
        out = []
        if r.get_robot_mode() != int(st["require_robot_mode"]):
            out.append(f"robot_mode={r.get_robot_mode()}, needs {st['require_robot_mode']} (running mode)")
        if abs(r.get_speed_fraction() - float(st["require_speed_fraction"])) > float(st["speed_fraction_tol"]):
            out.append(f"teach-pendant speed slider is {r.get_speed_fraction():.2f}, must be 100%")
        tcp = r.get_tcp_offset()
        if np.linalg.norm(tcp[:3] - np.array(st["expected_tcp_offset"][:3])) > float(st["tcp_offset_tol_m"]):
            out.append(f"TCP offset {np.round(tcp[:3], 4).tolist()} does not match the teach-pendant setting")
        if abs(r.get_payload() - float(st["expected_payload_kg"])) > float(st["payload_tol_kg"]):
            out.append(f"Payload {r.get_payload():.3f} kg does not match the expected {st['expected_payload_kg']} kg")
        return out

    # ================= public skills =================
    def survey(self) -> SkillOutcome:
        t0 = self.robot.now()
        out = SkillOutcome("survey", "ok")
        try:
            if (msg := self._lift_clear(out)):
                return self._finish(out, *self._move_failure(msg), t0)
            pq = getattr(self.robot, "survey_q", None) if getattr(getattr(self.robot, "platform", None), "builtin", True) is False else None
            q_target = pq if pq is not None else (self.grid.survey_q if self.grid.survey_q is not None else self.home_q)
            reason = self._check_joint_target(q_target)
            if reason:
                return self._finish(out, "rejected", reason, t0)
            if not self.robot.move_j(q_target, float(self.m["movej_speed_rad_s"]), float(self.m["movej_accel_rad_s2"])):
                return self._finish(out, *self._move_failure("moveJ to survey pose failed"), t0)
        finally:
            self.robot.speed_stop()
        return self._finish(out, "ok", "", t0)

    def closeup(self, zone_id: str) -> SkillOutcome:
        t0 = self.robot.now()
        out = SkillOutcome("inspect_closeup", "ok", zone_ids=[zone_id])
        if (msg := self._zone_problem(zone_id, allow_forbidden=True)):
            return self._finish(out, "rejected", msg, t0)
        try:
            R = rotvec_to_matrix(self.grid.tool_rotvec)
            cam = np.array(self.cfg["wrist_camera"]["pos_in_flange"], dtype=float)
            tcp = np.array(self.cfg["startup"]["expected_tcp_offset"][:3], dtype=float)
            cam_rel = R @ (cam - tcp)
            cx, cy = self.grid[zone_id].center
            z = self.grid.z_surface + float(self.cfg["wrist_camera"]["closeup_tcp_height_m"])
            pose = self._tool_pose(cx - cam_rel[0], cy - cam_rel[1], z)
            if (msg := self._goto(pose, out)):
                return self._finish(out, *self._move_failure(msg), t0)
        finally:
            self.robot.speed_stop()
        return self._finish(out, "ok", "", t0)

    def force_problem(self, force_n: Any) -> str:
        """Hard check of the target force (at the skill server level, independent of the agent guardrail): the global range force_limits;
        the real robot (ur3e backend) is further limited by real_validated_max_n, so only forces validated on site level by level are let through."""
        lim = self.cfg["force_limits"]
        try:
            f = float(force_n)
        except (TypeError, ValueError):
            return f"target force invalid: {force_n!r}"
        if not np.isfinite(f) or not float(lim["min_n"]) <= f <= float(lim["max_n"]):
            return f"target force {f:g} N is outside the allowed range {lim['min_n']:g}-{lim['max_n']:g} N"
        if getattr(self.robot, "backend_name", "") == "ur3e" and f > float(lim["real_validated_max_n"]) + 1e-6:
            return (f"target force {f:g} N exceeds the validated limit of the real robot {lim['real_validated_max_n']:g} N"
                    "(validate on site level by level, then change configs/robot.yaml force_limits.real_validated_max_n)")
        return ""

    def wipe(self, zone_ids: Sequence[str], force_n: float, pattern: str, passes: int = 1,
             bbox_norm: Optional[list[float]] = None) -> SkillOutcome:
        skill = "spot_scrub" if bbox_norm is not None else "wipe_zone"
        t0 = self.robot.now()
        out = SkillOutcome(skill, "ok", zone_ids=list(zone_ids))
        if self.fault:
            return self._finish(out, "rejected", f"unreset fault exists: {self.fault}", t0)
        if not getattr(self.robot, "contact_allowed", True):     # rtde backend shadow / dry_run: contact skills forbidden
            return self._finish(out, "rejected", f"backend {self.robot.backend_name} is in "
                                f"{getattr(self.robot, 'mode', '?')} mode, contact forbidden ({skill})", t0)
        if (msg := self.force_problem(force_n)):
            return self._finish(out, "rejected", msg, t0)
        if bbox_norm is None and pattern not in PATTERNS:
            return self._finish(out, "rejected", f"unknown trajectory {pattern!r}", t0)
        if not 1 <= int(passes) <= 2:
            return self._finish(out, "rejected", "passes must be 1 or 2", t0)
        target = float(force_n if self.force_override_n is None else self.force_override_n)
        plans = []
        for zid in zone_ids:
            if (msg := self._zone_problem(zid)):
                return self._finish(out, "rejected", msg, t0)
            path = self._plan_path(zid, pattern, bbox_norm)
            if (msg := self._path_problem(path)):
                return self._finish(out, "rejected", f"{zid}: {msg}", t0)
            plans.append((zid, path))
        try:
            for zid, path in plans:
                status, reason = self._wipe_one(zid, path, target, int(passes), out)
                if status != "ok":
                    return self._finish(out, status, f"{zid}: {reason}", t0)
            if (msg := self._lift_clear(out)):
                return self._finish(out, *self._move_failure(msg), t0)
        finally:
            self.robot.speed_stop()
        return self._finish(out, "ok", "", t0)

    def _settle(self, max_s: float = 0.5, speed_tol: float = 1e-3) -> None:
        """After speedStop keep sending stop and wait until the TCP speed drops below the threshold (at most max_s seconds)."""
        for _ in range(max(1, int(round(max_s / self.dt)))):
            self.robot.speed_stop()
            if np.linalg.norm(self.robot.get_tcp_speed()[:3]) < speed_tol:
                return
            self.robot.wait_tick()

    def _move_failure(self, msg: str) -> tuple[str, str]:
        """moveL/moveJ are executed by the controller; a protective stop on the way is a fault (must be cleared by hand on the teach pendant); an RTDE disconnect aborts and hands over to a person."""
        if (lost := self._connection_lost()):
            return "aborted", f"{msg}: {lost}"
        if self.robot.get_safety_mode() != 1:
            reason = getattr(self.robot, "protective_stop_reason", None) or f"safety_mode={self.robot.get_safety_mode()}"
            self.fault = f"{msg}: controller protective stop ({reason})"
            return "fault", self.fault
        return "rejected", msg

    def _connection_lost(self) -> Optional[str]:
        """rtde backend disconnected: the robot is stopped by the controller own watchdog; no fault is latched here, instead abort and ask a person to check and reconnect."""
        if getattr(self.robot, "connection_lost", False):
            return f"RTDE connection lost ({getattr(self.robot, 'connection_lost_reason', '')}), stopped, check the robot and the network"
        return None

    def reset_fault(self) -> None:
        """Manual reset (on the real robot it must be called after on-site staff confirm)."""
        self.fault = None

    # ================= planning and checks =================
    def _tool_pose(self, x: float, y: float, z: float) -> np.ndarray:
        return np.array([x, y, z, *self.grid.tool_rotvec], dtype=float)

    def _zone_problem(self, zone_id: str, allow_forbidden: bool = False) -> Optional[str]:
        if zone_id not in self.grid:
            return f"unknown zone {zone_id!r}"
        if zone_id in self.forbidden and not allow_forbidden:
            return f"{zone_id} is a forbidden zone"
        return None

    def _margin(self, zone_id: str) -> tuple[float, float, float, float]:
        """Per-edge margins: edges next to a forbidden zone get pad radius + clearance, the other edges use the default. Order (x_min, y_min, x_max, y_max)."""
        rect = self.grid[zone_id]
        base = float(self.sk["default_margin_ratio"]) * self.pad_r
        wide = self.pad_r + float(self.sk["forbidden_margin_extra_m"])
        m = [base, base, base, base]
        eps = 1e-9
        for other in self.forbidden:
            o = self.grid[other]
            overlap_y = o.ymin < rect.ymax - eps and o.ymax > rect.ymin + eps
            overlap_x = o.xmin < rect.xmax - eps and o.xmax > rect.xmin + eps
            if overlap_y and abs(o.xmax - rect.xmin) < 1e-6:
                m[0] = wide
            if overlap_x and abs(o.ymax - rect.ymin) < 1e-6:
                m[1] = wide
            if overlap_y and abs(o.xmin - rect.xmax) < 1e-6:
                m[2] = wide
            if overlap_x and abs(o.ymin - rect.ymax) < 1e-6:
                m[3] = wide
            # forbidden zone touching at one corner only: widen both adjacent edges
            corner_x = abs(o.xmax - rect.xmin) < 1e-6 or abs(o.xmin - rect.xmax) < 1e-6
            corner_y = abs(o.ymax - rect.ymin) < 1e-6 or abs(o.ymin - rect.ymax) < 1e-6
            if corner_x and corner_y and not overlap_x and not overlap_y:
                m[0 if abs(o.xmax - rect.xmin) < 1e-6 else 2] = wide
                m[1 if abs(o.ymax - rect.ymin) < 1e-6 else 3] = wide
        return tuple(m)  # type: ignore[return-value]

    def _plan_path(self, zone_id: str, pattern: str, bbox_norm: Optional[list[float]]) -> np.ndarray:
        rect: ZoneRect = self.grid[zone_id]
        margin = self._margin(zone_id)
        if bbox_norm is not None:
            return spot(rect, self.pad_r, margin, bbox_norm)
        return PATTERNS[pattern](rect, self.pad_r, margin)

    def _path_problem(self, path: np.ndarray) -> Optional[str]:
        r_max = robot_max_radius(self.robot, self.s) - self.pad_r
        radii = np.array([robot_radius(self.robot, p) for p in path])
        if radii.max() > r_max:
            return f"trajectory farthest point horizontal radius {radii.max():.3f} m exceeds {r_max:.3f} m (including pad radius)"
        for other in self.forbidden:
            o = self.grid[other]
            gap = self.pad_r + float(self.sk["forbidden_margin_extra_m"]) * 0.5
            inside = ((path[:, 0] > o.xmin - gap) & (path[:, 0] < o.xmax + gap) &
                      (path[:, 1] > o.ymin - gap) & (path[:, 1] < o.ymax + gap))
            if inside.any():
                return f"trajectory would move the polishing pad into forbidden zone {other}"
        return None

    def _check_joint_target(self, q: np.ndarray, q_from: Optional[np.ndarray] = None) -> Optional[str]:
        q_from = self.robot.get_q() if q_from is None else q_from
        jt = getattr(self.robot, "joint_target_problem", None)
        if jt is not None and getattr(getattr(self.robot, "platform", None), "builtin", True) is False:
            if (msg := jt(q)):                              # non-UR platform: check the joint limit margin (the UR elbow / wrist 3 rules do not apply)
                return msg
        else:
            if elbow_margin_deg(q) < float(self.s["min_elbow_margin_deg"]):
                return f"target configuration elbow joint only {elbow_margin_deg(q):.1f}° from fully straight"
            if abs(q[5] - q_from[5]) > float(self.s["max_wrist3_travel_rad"]):
                return "wrist 3 rotates more than 180° (cable anti-tangling)"
        # sample-check the joint-space straight-line path: the tool stays at least 15 mm above the table and within the horizontal radius
        kin = getattr(self.robot, "kin", None)
        if kin is not None:
            for s in np.linspace(0, 1, 21):
                p = kin.fk(q_from + (q - q_from) * s)[:3, 3]
                if robot_radius(self.robot, p) > robot_max_radius(self.robot, self.s):
                    return "joint path leaves the horizontal radius midway"
                if p[2] < self.grid.z_surface + 0.015 and 0 < s < 1:
                    return "joint path takes the tool too low midway"
        return None

    # ================= motion =================
    def _z_above(self) -> float:
        return self.grid.z_surface + float(self.m["hover_above_surface_m"]) + float(self.m["lift_height_m"])

    def _lift_clear(self, out: SkillOutcome) -> Optional[str]:
        """Lift to the safe height; on failure return the reason (the caller decides fault / aborted via _move_failure), never raises."""
        pose = self.robot.get_tcp_pose()
        if pose[2] < self._z_above() - 0.003:
            up = pose.copy()
            up[2] = self._z_above()
            if not self.robot.move_l(up, float(self.m["movel_speed_m_s"]), float(self.m["movel_accel_m_s2"])):
                return "lift failed"
            out.events.append({"t": self.robot.now(), "event": "lift", "z": float(up[2])})
        return None

    def _goto(self, pose: np.ndarray, out: SkillOutcome) -> Optional[str]:
        """Reach the target pose: lift to the safe height first, use moveL when close and moveJ via inverse kinematics when far, and finish with moveL into place."""
        if (msg := self._lift_clear(out)):
            return msg
        cur = self.robot.get_tcp_pose()
        above = pose.copy()
        above[2] = max(pose[2], self._z_above())
        v, a = float(self.m["movel_speed_m_s"]), float(self.m["movel_accel_m_s2"])
        near = np.linalg.norm(cur[:2] - above[:2]) < 0.12 and np.allclose(cur[3:], above[3:], atol=0.05)
        if near:
            if not self.robot.move_l(above, v, a):
                return "moveL to above the zone failed"
        else:
            q = self.robot.get_ik(above, self.home_q)
            if q is None:
                return "no inverse kinematics solution for the pose above the zone"
            if (msg := self._check_joint_target(q)):
                return msg
            if not self.robot.move_j(q, float(self.m["movej_speed_rad_s"]), float(self.m["movej_accel_rad_s2"])):
                return "moveJ to above the zone failed"
        if abs(above[2] - pose[2]) > 1e-4 and not self.robot.move_l(pose, v, a):
            return "moveL descent failed"
        return None

    # ================= main wiping loop =================
    def _wipe_one(self, zone_id: str, path: np.ndarray, target: float, passes: int,
                  out: SkillOutcome) -> tuple[str, str]:
        robot, dt = self.robot, self.dt
        hover = self._tool_pose(path[0, 0], path[0, 1], self.grid.z_surface + float(self.m["hover_above_surface_m"]))
        if (msg := self._goto(hover, out)):
            return self._move_failure(msg)

        safety = SafetyMonitor(self.s, dt)
        tare = SoftwareTare(self.cfg["tare"], dt)
        teacher = ForceTeacher.from_cfg(self.cfg["teacher"], dt)
        ramp = ForceRamp(float(self.cfg["teacher"]["ramp_up_n_s"]), float(self.cfg["teacher"]["ramp_down_n_s"]), dt)
        f_clip = self.cfg["teacher"]["f_ref_clip_n"]
        push = PushInWindow(float(self.c["push_in_window_s"]), dt)
        confirm_t = HoldTimer(float(self.c["enter_duration_s"]), dt)
        lost_t = HoldTimer(float(self.c["exit_duration_s"]), dt)
        stable_t = HoldTimer(float(self.c["stable_duration_s"]), dt)
        clear_t = HoldTimer(float(self.c["release_clear_duration_s"]), dt)
        zero = np.zeros(6)

        # hover still → software tare (readings over the first window_s)
        for _ in range(int(round(0.1 / dt))):
            robot.speed_l(zero, self.acc, self.cmd_time)
            robot.wait_tick()
        while not tare.ready():
            tare.add_sample(robot.get_tcp_force(), robot.get_tcp_pose(), robot.get_tcp_speed())
            robot.speed_l(zero, self.acc, self.cmd_time)
            robot.wait_tick()
        ok, msg = tare.commit()
        out.events.append({"t": robot.now(), "zone": zone_id, "event": "tare", "ok": ok, "detail": msg})

        v_t = float(self.m["contact_max_tangent_m_s"])
        k_track = float(self.m["path_tracking_gain_1_s"])
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        length = float(cum[-1])

        def at(sv: float) -> np.ndarray:
            return np.array([np.interp(sv, cum, path[:, 0]), np.interp(sv, cum, path[:, 1])])

        state = "approach"
        state_t0 = robot.now()
        path_s, pass_idx, direction = 0.0, 0, 1
        losses = 0
        f_samples: list[float] = []
        peak = 0.0
        control_ticks = 0
        release_start: Optional[np.ndarray] = None
        release_reason = "done"
        cov = _Coverage(self.grid[zone_id], self.pad_r)
        tr: dict[str, list[float]] = {k: [] for k in ("t", "x", "y", "z", "f_n", "f_ref", "state")}
        tick = 0
        t_zone0 = robot.now()
        f_ref = 0.0

        def to(new_state: str) -> None:
            nonlocal state, state_t0
            out.events.append({"t": robot.now(), "zone": zone_id, "event": "state", "from": state, "to": new_state})
            state, state_t0 = new_state, robot.now()

        try:
            while True:
                pose = robot.get_tcp_pose()
                speed = robot.get_tcp_speed()
                comp = tare.compensate(robot.get_tcp_force(), pose)
                f_n = normal_force(comp, pose)
                _, inward = tool_axes(pose)
                push.add(speed, inward)
                in_contact = state in ("confirm", "ramp", "control")
                if in_contact or state == "release":
                    peak = max(peak, f_n)

                ev: Optional[SafetyEvent] = safety.check(robot, pose, comp, f_n, in_contact)
                if ev is not None:
                    robot.speed_stop()
                    out.events.append({"t": robot.now(), "zone": zone_id, "event": "safety", "level": ev.level,
                                       "reason": ev.reason})
                    if ev.level in ("fault", "fatal"):
                        if (lost := self._connection_lost()):
                            return "aborted", lost
                        self.fault = ev.reason
                        return "fault", ev.reason
                    if state != "release":  # hold: abort and retract along the tool axis
                        release_reason = f"hold: {ev.reason}"
                        to("release")
                        release_start = pose[:3].copy()

                twist = zero.copy()
                if state in ("approach", "reapproach"):
                    twist[:3] = inward * float(self.m["approach_max_linear_m_s"])
                    pm = push.mean() or 0.0
                    if f_n >= float(self.c["enter_force_n"]) and pm >= float(self.c["push_in_min_m_s"]):
                        robot.speed_stop()
                        confirm_t.reset()
                        to("confirm")
                        twist = zero.copy()
                    elif robot.now() - state_t0 > float(self.m["approach_timeout_s"]):
                        robot.speed_stop()
                        release_reason = "no_contact"
                        to("release")
                        release_start = pose[:3].copy()
                elif state == "confirm":
                    if f_n < float(self.c["exit_force_n"]):
                        to("reapproach" if control_ticks else "approach")
                    elif confirm_t.update(np.linalg.norm(speed[:3]) <= float(self.c["max_confirm_tcp_speed_m_s"])):
                        ramp.start(float(np.clip(f_n, *f_clip)), target)
                        teacher.reset()
                        stable_t.reset()
                        lost_t.reset()
                        to("ramp")
                elif state == "ramp":
                    f_ref = ramp.step()
                    twist[:3] = inward * teacher.step(f_ref, f_n)
                    grace = robot.now() - state_t0 < float(self.c["ramp_grace_s"])
                    if not grace and lost_t.update(f_n < float(self.c["exit_force_n"])):
                        losses += 1
                        teacher.reset()
                        to("reapproach")
                    elif stable_t.update(abs(f_n - target) <= float(self.c["stable_tolerance_n"])
                                         and abs(ramp.value - target) < 1e-9):
                        lost_t.reset()
                        to("control")
                    elif robot.now() - state_t0 > float(self.c["ramp_timeout_s"]):
                        out.events.append({"t": robot.now(), "zone": zone_id, "event": "ramp_timeout",
                                           "f_n": f_n, "target": target})
                        lost_t.reset()
                        to("control")
                elif state == "control":
                    f_ref = ramp.step()
                    v_n = teacher.step(f_ref, f_n)
                    path_s += v_t * dt * direction
                    if (path_s >= length) if direction > 0 else (path_s <= 0):
                        pass_idx += 1
                        path_s = float(np.clip(path_s, 0, length))
                        if pass_idx >= passes:
                            robot.speed_stop()
                            to("release")
                            release_start = pose[:3].copy()
                        else:
                            direction = -direction          # the second pass walks the same path in reverse
                    if state == "control":
                        p_des = at(path_s)
                        p_next = at(float(np.clip(path_s + v_t * dt * direction, 0, length)))
                        v_xy = (p_next - p_des) / dt + k_track * (p_des - pose[:2])
                        n_xy = np.linalg.norm(v_xy)
                        if n_xy > v_t:
                            v_xy *= v_t / n_xy
                        v_tan = np.array([v_xy[0], v_xy[1], 0.0])
                        v_tan -= np.dot(v_tan, inward) * inward
                        twist[:3] = v_tan + inward * v_n
                        f_samples.append(f_n)
                        control_ticks += 1
                        if tick % 5 == 0 and f_n > float(self.c["exit_force_n"]):
                            cov.mark(pose[0], pose[1])
                        if lost_t.update(f_n < float(self.c["exit_force_n"])):
                            losses += 1
                            out.events.append({"t": robot.now(), "zone": zone_id, "event": "contact_lost",
                                               "count": losses})
                            teacher.reset()
                            if losses > int(self.sk["max_contact_losses"]):
                                release_reason = "contact lost repeatedly"
                                to("release")
                                release_start = pose[:3].copy()
                            else:
                                to("reapproach")
                elif state == "release":
                    if release_start is None:
                        release_start = pose[:3].copy()
                    twist[:3] = -inward * float(self.m["retract_m_s"])
                    moved = float(np.dot(pose[:3] - release_start, -inward))
                    if clear_t.update(abs(f_n) < float(self.c["release_clear_force_n"])) and moved > 0.001:
                        robot.speed_stop()
                        break
                    if moved > float(self.m["retract_max_m"]):
                        robot.speed_stop()
                        self.fault = f"still {f_n:.1f} N contact force after retracting {moved * 1000:.0f} mm"
                        return "fault", self.fault

                if state != "confirm":
                    if not robot.speed_l(twist, self.acc, self.cmd_time):
                        robot.speed_stop()
                        if (lost := self._connection_lost()):
                            return "aborted", lost
                        self.fault = getattr(robot, "last_refusal", None) or "speedL rejected by the controller (possibly protective stop)"
                        return "fault", self.fault
                else:
                    robot.speed_l(zero, self.acc, self.cmd_time)
                robot.wait_tick()
                tick += 1
                if tick % int(self.sk["trace_every_ticks"]) == 0:
                    for k, v in (("t", robot.now()), ("x", pose[0]), ("y", pose[1]), ("z", pose[2]),
                                 ("f_n", f_n), ("f_ref", f_ref), ("state", STATE_CODES.get(state, 0))):
                        tr[k].append(float(v))
                if robot.now() - t_zone0 > float(self.sk["zone_timeout_s"]):
                    robot.speed_stop()
                    return "aborted", "single-zone wipe timed out"
        finally:
            self._settle()

        # statistics
        fs = np.array(f_samples) if f_samples else np.zeros(1)
        summary = ForceSummary(
            target_n=target,
            mean_normal_n=float(fs.mean()),
            peak_normal_n=float(peak),
            within_1n_ratio=float(np.mean(np.abs(fs - target) <= 1.0)) if f_samples else 0.0,
            contact_loss_count=losses,
            coverage_ratio=cov.ratio(),
            contact_duration_s=control_ticks * dt,
        )
        out.force[zone_id] = summary.model_dump()
        for k in tr:
            out.trace.setdefault(k, []).extend(tr[k])
        if release_reason == "no_contact":
            return "no_contact", f"no contact with the table after approaching for {self.m['approach_timeout_s']} s"
        if release_reason != "done":
            return "aborted", release_reason
        out.events.append({"t": robot.now(), "zone": zone_id, "event": "wiped", "path_m": path_length(path) * passes})
        return "ok", ""

    def _finish(self, out: SkillOutcome, status: str, reason: str, t0: float) -> SkillOutcome:
        out.status = status
        out.reason = reason
        out.duration_s = float(self.robot.now() - t0)
        return out


class _Coverage:
    """Fraction of the zone area swept by the polishing pad in contact (computed from the TCP trajectory, also available on the real robot)."""

    def __init__(self, rect: ZoneRect, pad_r: float, res: float = 0.0025):
        self.rect, self.r, self.res = rect, pad_r, res
        nx = max(1, int(round((rect.xmax - rect.xmin) / res)))
        ny = max(1, int(round((rect.ymax - rect.ymin) / res)))
        xs = rect.xmin + (np.arange(nx) + 0.5) * res
        ys = rect.ymin + (np.arange(ny) + 0.5) * res
        self.X, self.Y = np.meshgrid(xs, ys, indexing="ij")
        self.hit = np.zeros((nx, ny), dtype=bool)

    def mark(self, x: float, y: float) -> None:
        self.hit |= (self.X - x) ** 2 + (self.Y - y) ** 2 <= self.r**2

    def ratio(self) -> float:
        return float(self.hit.mean())
