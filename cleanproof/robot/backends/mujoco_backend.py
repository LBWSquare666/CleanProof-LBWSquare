"""MuJoCo simulation backend: kinematically integrated UR3e + virtual tabletop contact + stain removal + camera rendering.

- speedL: acceleration limit → 1-tick delay → first-order tracking (τ), then integrate joint angles with a damped pseudo-inverse;
  the command decelerates to zero if it is not updated for more than `time` seconds (same semantics as the time parameter of UR speedL).
- moveL / moveJ: trapezoidal velocity interpolation, blocking until done; contact and protective stop are computed during motion as well.
- F/T: environment force of the virtual tabletop + noise + zero offset (to test the software's tare).
- Simulates the UR controller's protective stop: TCP horizontal radius > 0.53 m (where the real robot once protective-stopped) or contact force too large.
Simulation time is independent of wall-clock time, so batch regression can run much faster than real time.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ...common.config import load_yaml
from ...common.zones import ZoneGrid
from ...perception.camera import CameraModel, mujoco_to_opencv_rotation
from ...sim.contact import ContactResult, Obstacle, VirtualSurface
from ...sim.dirt import DirtMap
from ..io import ROBOT_MODE_RUNNING, SAFETY_MODE_NORMAL, SAFETY_MODE_PROTECTIVE_STOP, SAFETY_MODE_ROBOT_EMERGENCY_STOP
from ..kinematics import (
    UR3eKinematics,
    horizontal_radius,
    matrix_to_pose,
    matrix_to_rotvec,
    pose_to_matrix,
    rotvec_to_matrix,
    twist_to_qdot,
)

PROTECTIVE_STOP_RADIUS_M = 0.53
PROTECTIVE_STOP_FORCE_N = 60.0


def _clip_norm(v: np.ndarray, limit: float) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n <= limit or n == 0 else v * (limit / n)


class MujocoRobot:
    backend_name = "mujoco"

    def __init__(
        self,
        grid: Optional[ZoneGrid] = None,
        robot_cfg: Optional[dict[str, Any]] = None,
        sim_cfg: Optional[dict[str, Any]] = None,
        obstacles: Optional[list[Obstacle]] = None,
        render: bool = True,
        surface: Optional[str] = None,
        platform: str = "ur3e",
    ):
        self.grid = grid or ZoneGrid()
        self.rcfg = robot_cfg or load_yaml("robot.yaml")
        self.scfg = sim_cfg or load_yaml("sim.yaml")
        self.control_dt = 1.0 / float(self.rcfg["control_hz"])
        self.tcp_offset = np.array(self.rcfg["startup"]["expected_tcp_offset"], dtype=float)
        # Platform adaptation: ur3e uses the original DH kinematics and scene; other platforms use a Menagerie model + generic kinematics
        from ..platforms import get_platform
        self.platform = get_platform(platform)
        self._pscene = None
        self.base_xy = np.array(self.platform.base_pos[:2], float)
        if self.platform.builtin:
            self.kin = UR3eKinematics(self.tcp_offset)
            self.home_q = np.array(self.rcfg["home_q"], dtype=float)
            self.survey_q = self.grid.survey_q
            self.max_radius_m: Optional[float] = None          # None = use robot.yaml safety.max_horizontal_radius_m (original UR3e value)
            self._stop_radius = PROTECTIVE_STOP_RADIUS_M
        else:
            from ..platforms import PlatformKinematics, build_platform_scene
            self._pscene = build_platform_scene(self.platform, self.grid, self.scfg,     # for kinematics (independent of texture size)
                                                tcp_len=float(self.tcp_offset[2]), wrist_cam_pos=tuple(self.rcfg["wrist_camera"]["pos_in_flange"]))
            self.kin = PlatformKinematics(self._pscene)
            self.kin.solve_home(self.rcfg, self.grid)
            self.home_q = self.kin.home_q.copy()
            self.survey_q = self.kin.survey_q
            self.max_radius_m = float(self.platform.reach_m) - 0.04
            self._stop_radius = float(self.platform.reach_m)
        self.q = self.home_q.copy()
        self.t = 0.0
        servo = self.scfg["servo"]
        self.tau = float(servo["speed_tracking_tau_s"])
        self._delay = deque([np.zeros(6)] * int(servo["delay_ticks"]))
        self._cmd = np.zeros(6)
        self._cmd_expire = 0.0
        self._cmd_acc = 1.0
        self._rl = np.zeros(6)        # command after acceleration limiting
        self._act = np.zeros(6)       # actual TCP velocity
        self.surface = VirtualSurface(self.grid.z_surface, self.scfg["contact"], obstacles)
        self.dirt = DirtMap(self.grid, self.scfg["dirt"], surface=surface)     # surface: tabletop material (TaskSpec.surface of the task)
        self.ft_bias = np.concatenate([np.array(self.scfg["contact"].get("ft_bias_n", [0, 0, 0]), float), np.zeros(3)])
        self.safety_mode = SAFETY_MODE_NORMAL
        self.protective_stop_reason: Optional[str] = None
        self.speed_fraction = 1.0
        self.last_contact = ContactResult(wrench=np.zeros(6))
        self._wrench = np.zeros(6)
        self._render_enabled = render
        self._mj: Optional[dict[str, Any]] = None
        self.move_count = 0
        # Hook called at the end of every simulation tick (the recorder grabs frames at simulation-time intervals); when empty, behavior is exactly as before
        self.tick_hooks: list[Callable[[], None]] = []
        self._update_contact(np.zeros(6), 0.0)

    def radius_of(self, p: Sequence[float]) -> float:
        """Horizontal radius of the TCP relative to this platform's base (the UR3e base is at the origin, same as kinematics.horizontal_radius)."""
        return float(np.hypot(p[0] - self.base_xy[0], p[1] - self.base_xy[1]))

    def joint_target_problem(self, q: Sequence[float]) -> Optional[str]:
        """Joint target check for non-UR platforms: at least 5° from the joint limits (for UR3e the skill layer checks the elbow joint / wrist 3)."""
        if self._pscene is None:
            return None
        m = self.kin.limit_margin_deg(q)
        return f"target configuration is only {m:.1f}° from the joint limits" if m < 5.0 else None

    # ---- Clock ----
    def now(self) -> float:
        return self.t

    def wait_tick(self) -> None:
        self._step_speed(self.control_dt)

    # ---- state ----
    def get_q(self) -> np.ndarray:
        return self.q.copy()

    def get_tcp_pose(self) -> np.ndarray:
        return matrix_to_pose(self.kin.fk(self.q))

    def get_tcp_speed(self) -> np.ndarray:
        return self._act.copy()

    def get_tcp_force(self) -> np.ndarray:
        return self._wrench.copy()

    def get_robot_mode(self) -> int:
        return ROBOT_MODE_RUNNING

    def get_safety_mode(self) -> int:
        return self.safety_mode

    def get_speed_fraction(self) -> float:
        return self.speed_fraction

    def get_tcp_offset(self) -> np.ndarray:
        return self.tcp_offset.copy()

    def get_payload(self) -> float:
        return float(self.rcfg["startup"]["expected_payload_kg"])

    # ---- Motion ----
    def speed_l(self, twist: Sequence[float], acceleration: float, time: float) -> bool:
        if self.safety_mode != SAFETY_MODE_NORMAL:
            return False
        self._cmd = np.asarray(twist, dtype=float).copy()
        self._cmd_acc = float(acceleration)
        self._cmd_expire = self.t + max(float(time), self.control_dt)
        return True

    def speed_stop(self, deceleration: float = 0.5) -> None:
        self._cmd = np.zeros(6)
        self._cmd_acc = float(deceleration)
        self._cmd_expire = self.t

    def stop_script(self) -> None:
        self.speed_stop()
        self._rl[:] = 0.0
        self._act[:] = 0.0

    def close(self) -> None:
        if self._mj is not None:
            self._mj["renderer"].close()
            self._mj = None

    def get_ik(self, pose: Sequence[float], q_near: Sequence[float]) -> Optional[np.ndarray]:
        return self.kin.ik(np.asarray(pose, dtype=float), q_near)

    def move_l(self, pose: Sequence[float], speed: float, acceleration: float) -> bool:
        if self.safety_mode != SAFETY_MODE_NORMAL:
            return False
        T0 = self.kin.fk(self.q)
        T1 = pose_to_matrix(pose)
        dp = T1[:3, 3] - T0[:3, 3]
        dr = matrix_to_rotvec(T1[:3, :3] @ T0[:3, :3].T)
        dist = max(float(np.linalg.norm(dp)), 1e-9)
        # Trapezoidal interpolation of position; orientation is interpolated with the same progress
        profile = self._trapezoid(dist, speed, acceleration)
        R0 = T0[:3, :3]
        q = self.q.copy()
        for s in profile:
            Tt = np.eye(4)
            Tt[:3, 3] = T0[:3, 3] + dp * s
            Tt[:3, :3] = rotvec_to_matrix(dr * s) @ R0
            qn = self.kin.ik(Tt, q)
            if qn is None:
                return False
            v = np.concatenate([(Tt[:3, 3] - self.kin.fk(q)[:3, 3]) / self.control_dt, np.zeros(3)])
            q = qn
            if not self._advance_position(q, v):
                return False
        self.move_count += 1
        return True

    def move_j(self, q_target: Sequence[float], speed: float, acceleration: float) -> bool:
        if self.safety_mode != SAFETY_MODE_NORMAL:
            return False
        q0 = self.q.copy()
        dq = np.asarray(q_target, dtype=float) - q0
        dist = max(float(np.abs(dq).max()), 1e-9)
        for s in self._trapezoid(dist, speed, acceleration):
            qn = q0 + dq * s
            v = np.concatenate([(self.kin.fk(qn)[:3, 3] - self.kin.fk(self.q)[:3, 3]) / self.control_dt, np.zeros(3)])
            if not self._advance_position(qn, v):
                return False
        self.move_count += 1
        return True

    # ---- Simulation internals ----
    def _trapezoid(self, dist: float, vmax: float, amax: float) -> list[float]:
        vmax, amax = max(vmax, 1e-6), max(amax, 1e-6)
        t_acc = vmax / amax
        d_acc = 0.5 * amax * t_acc**2
        if 2 * d_acc > dist:
            t_acc = np.sqrt(dist / amax)
            vmax = amax * t_acc
            d_acc = dist / 2
        t_total = 2 * t_acc + (dist - 2 * d_acc) / vmax
        n = max(1, int(np.ceil(t_total / self.control_dt)))
        out = []
        for k in range(1, n + 1):
            t = min(k * self.control_dt, t_total)
            if t < t_acc:
                s = 0.5 * amax * t**2
            elif t < t_total - t_acc:
                s = d_acc + vmax * (t - t_acc)
            else:
                s = dist - 0.5 * amax * (t_total - t) ** 2
            out.append(min(1.0, s / dist))
        out[-1] = 1.0
        return out

    def _advance_position(self, q_new: np.ndarray, tcp_twist: np.ndarray) -> bool:
        self.q = np.asarray(q_new, dtype=float).copy()
        self._act = tcp_twist
        self._rl[:] = 0.0
        self.t += self.control_dt
        self._update_contact(tcp_twist, self.control_dt)
        self._act = np.zeros(6)
        for fn in self.tick_hooks:
            fn()
        return self.safety_mode == SAFETY_MODE_NORMAL

    def _step_speed(self, dt: float) -> None:
        target = self._cmd if self.t < self._cmd_expire else np.zeros(6)
        diff = target - self._rl
        step = self._cmd_acc * dt
        self._rl = self._rl + np.concatenate([_clip_norm(diff[:3], step), _clip_norm(diff[3:], step)])
        self._delay.append(self._rl.copy())
        delayed = self._delay.popleft()
        alpha = dt / (self.tau + dt) if self.tau > 0 else 1.0
        self._act = self._act + alpha * (delayed - self._act)
        if self.safety_mode != SAFETY_MODE_NORMAL:
            self._act[:] = 0.0
        self.q = self.q + twist_to_qdot(self.kin, self.q, self._act) * dt
        if self._pscene is not None:
            self.q = self.kin.clamp(self.q)
        self.t += dt
        self._update_contact(self._act, dt)
        for fn in self.tick_hooks:
            fn()

    def _update_contact(self, twist: np.ndarray, dt: float) -> None:
        T = self.kin.fk(self.q)
        res = self.surface.compute(T, twist)
        self.last_contact = res
        self._wrench = res.wrench + self.ft_bias
        if dt > 0 and res.in_contact:
            slip = float(np.linalg.norm(twist[:2])) * dt
            self.dirt.step(res.points, T[:2, 3], self.surface.radius, slip)
        if self.radius_of(T[:3, 3]) > self._stop_radius:
            self._protective_stop(f"TCP horizontal radius {self.radius_of(T[:3, 3]):.3f} m is out of reach")
        elif np.linalg.norm(res.wrench[:3]) > PROTECTIVE_STOP_FORCE_N:
            self._protective_stop(f"contact force {np.linalg.norm(res.wrench[:3]):.1f} N is too large")

    def _protective_stop(self, reason: str) -> None:
        if self.safety_mode == SAFETY_MODE_NORMAL:
            self.safety_mode = SAFETY_MODE_PROTECTIVE_STOP
            self.protective_stop_reason = reason
        self._cmd[:] = 0.0
        self._rl[:] = 0.0
        self._act[:] = 0.0

    def clear_protective_stop(self) -> None:
        """Simulation tests only: on the real robot a protective stop can only be cleared manually on the teach pendant."""
        self.safety_mode = SAFETY_MODE_NORMAL
        self.protective_stop_reason = None

    def emergency_stop(self, reason: str = "operator E-STOP") -> None:
        """Simulate an emergency stop: zero the velocity at once, reject later speedL/moveL/moveJ, set the safety mode to robot emergency stop.

        The SafetyMonitor in the skill loop judges it fatal on the next tick and exits. Can be called from other threads (writes only scalars and arrays).
        On the real robot the emergency stop is a physical button + teach pendant; the software never simulates it.
        """
        if self.safety_mode == SAFETY_MODE_NORMAL:
            self.protective_stop_reason = reason
        self.safety_mode = SAFETY_MODE_ROBOT_EMERGENCY_STOP
        self._cmd[:] = 0.0
        self._cmd_expire = self.t
        self._rl[:] = 0.0
        self._act[:] = 0.0

    # ---- Rendering ----
    def _ensure_mj(self) -> dict[str, Any]:
        if self._mj is None:
            import mujoco

            from ...sim.scene import DIRT_TEXTURE, build_mjcf

            if self._pscene is not None:                       # platform scene: rendering uses a separate copy of the model data (the kinematics copy gets rewritten during IK)
                from ..platforms import build_platform_scene
                ps = build_platform_scene(self.platform, self.grid, self.scfg, self.dirt.render_shape, tcp_len=float(self.tcp_offset[2]),
                                          wrist_cam_pos=tuple(self.rcfg["wrist_camera"]["pos_in_flange"]))
                model, data = ps.model, ps.data
                self._render_scene = ps
            else:
                xml = build_mjcf(self.grid, self.scfg, self.dirt.render_shape, self.tcp_offset,
                                 tuple(self.rcfg["wrist_camera"]["pos_in_flange"]))
                model = mujoco.MjModel.from_xml_string(xml)
                data = mujoco.MjData(model)
            r = self.scfg["render"]
            renderer = mujoco.Renderer(model, int(r["height"]), int(r["width"]))
            tex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, DIRT_TEXTURE)
            self._mj = {"mujoco": mujoco, "model": model, "data": data, "renderer": renderer, "tex_id": tex_id}
        return self._mj

    def _sync_scene(self, renderer: Any) -> None:
        """Sync the current joint angles and stain texture to the GL context owned by the renderer (texture upload is per context, so make_current first)."""
        from ...sim.scene import dirt_to_texture

        mj = self._ensure_mj()
        mujoco, model, data = mj["mujoco"], mj["model"], mj["data"]
        self._set_render_q(data)
        mujoco.mj_forward(model, data)
        tex = dirt_to_texture(self.dirt.rgb())
        tid = mj["tex_id"]
        adr = model.tex_adr[tid]
        model.tex_data[adr : adr + tex.size] = tex.ravel()
        if getattr(renderer, "_gl_context", None) is not None:
            renderer._gl_context.make_current()
        mujoco.mjr_uploadTexture(model, renderer._mjr_context, tid)

    def _set_render_q(self, data: Any) -> None:
        if self._pscene is not None:
            data.qpos[:] = self._render_scene.fixed_qpos
            data.qpos[self._render_scene.arm_qadr] = self.q
        else:
            data.qpos[:6] = self.q

    def render(self, camera: Any = "fixed") -> np.ndarray:
        """RGB image of the current scene (H, W, 3) uint8. camera can be a camera name or a mujoco.MjvCamera (free camera)."""
        if not self._render_enabled:
            raise RuntimeError("rendering is not enabled (render=False)")
        return self.render_with(self._ensure_mj()["renderer"], camera)

    def render_with(self, renderer: Any, camera: Any = "fixed") -> np.ndarray:
        """Render the current scene with an external mujoco.Renderer (e.g. the recorder's own resolution / context).

        Does not change the scene geometry or touch the fixed / wrist camera definitions; render() used by the verifier and tests is unaffected.
        Must be called in the thread that created the renderer (the EGL context is bound to a thread).
        """
        self._sync_scene(renderer)
        renderer.update_scene(self._mj["data"], camera=camera)
        return renderer.render().copy()

    def new_renderer(self, height: int, width: int) -> Any:
        """Create a new offscreen Renderer for this scene's MjModel (size must not exceed the MJCF offwidth/offheight)."""
        mj = self._ensure_mj()
        return mj["mujoco"].Renderer(mj["model"], int(height), int(width))

    def camera_model(self, camera: str = "fixed") -> CameraModel:
        mj = self._ensure_mj()
        mujoco, model, data = mj["mujoco"], mj["model"], mj["data"]
        self._set_render_q(data)
        mujoco.mj_forward(model, data)
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        T = np.eye(4)
        T[:3, :3] = mujoco_to_opencv_rotation(data.cam_xmat[cid].reshape(3, 3))
        T[:3, 3] = data.cam_xpos[cid]
        r = self.scfg["render"]
        return CameraModel.from_fovy(float(model.cam_fovy[cid]), int(r["width"]), int(r["height"]), T)
