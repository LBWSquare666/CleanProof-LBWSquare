"""Platform adapter: any MuJoCo robot model → kinematics with the same interface as UR3eKinematics + rendering scene.

The same CleanProof software runs on different robots, and only the contents of configs/platforms.yaml change between platforms. Three things are done here:
1. Scene assembly: tabletop scene (sim/scene.build_mjcf(arm=False)) + Menagerie robot (MjSpec.attach, prefix "r_") + our wiping tool
   (handle, polishing disc, tcp site, wrist camera; geometry matches UR3e, virtual tabletop contact model parameters unchanged).
2. Kinematics: fk(q) → 4×4 in the world frame (= tabletop / base frame), jacobian(q) → 6×n with row order [v; ω], ik(T, q0) by damped least squares; 7-axis arms add a null-space term
   pulling back to home to avoid elbow drift, with the result clamped to joint limits. robot/kinematics.twist_to_qdot and ik use J·Jᵀ (6×6), valid for any n.
3. home / survey: not filled in by hand, IK finds the same TCP poses as the UR3e mounting Home / survey pose (tool pointing down, above the workspace / clear of the camera).

Kinematics only (mj_kinematics), no mj_step: consistent with the existing MuJoCo backend (kinematic integration + virtual tabletop contact, geometry takes no part in physical collisions).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..common.config import load_yaml
from .kinematics import DEFAULT_TCP_OFFSET, UR3eKinematics, matrix_to_rotvec, pose_to_matrix

PREFIX = "r_"
MENAGERIE_DEFAULTS = (Path("./store/mujoco_menagerie"),)


def menagerie_root() -> Path:
    env = os.environ.get("CLEANPROOF_MENAGERIE")
    ud = os.environ.get("CLEANPROOF_DATA_DIR")             # installed copy: mujoco_menagerie in the user data directory (where `cleanproof fetch-menagerie` puts it)
    for p in ([Path(env)] if env else []) + ([Path(ud).expanduser() / "mujoco_menagerie"] if ud else []) + list(MENAGERIE_DEFAULTS):
        if p.is_dir():
            return p
    raise FileNotFoundError("MuJoCo Menagerie not found: set the environment variable CLEANPROOF_MENAGERIE (see the README)")


@dataclass(frozen=True)
class PlatformSpec:
    key: str
    label: str
    kind: str
    reach_m: float
    builtin: bool = False
    mjcf: str = ""
    arm_joints: tuple[str, ...] = ()
    ee_site: str = ""
    ee_body: str = ""
    tool_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    tool_start: float = 0.0
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_yaw_deg: float = 0.0
    floor_z: float = -0.72
    pedestal: bool = False

    @property
    def dof(self) -> int:
        return 6 if self.builtin else len(self.arm_joints)


def load_platforms() -> dict[str, PlatformSpec]:
    """Read configs/platforms.yaml for the current site (CLEANPROOF_SITE) (a site can override individual fields, e.g. the protection radius)."""
    from ..common.config import current_site
    return _load_platforms(current_site())


@lru_cache(maxsize=8)
def _load_platforms(site: str) -> dict[str, PlatformSpec]:
    out = {}
    for key, c in load_yaml("platforms.yaml").items():
        out[key] = PlatformSpec(key=key, label=str(c["label"]), kind=str(c["kind"]), reach_m=float(c["reach_m"]),
                                builtin=bool(c.get("builtin", False)), mjcf=str(c.get("mjcf", "")),
                                arm_joints=tuple(c.get("arm_joints") or ()), ee_site=str(c.get("ee_site", "")),
                                ee_body=str(c.get("ee_body", "")), tool_quat=tuple(float(v) for v in c.get("tool_quat", (1, 0, 0, 0))),
                                tool_start=float(c.get("tool_start", 0.0)), base_pos=tuple(float(v) for v in c.get("base_pos", (0, 0, 0))),
                                base_yaw_deg=float(c.get("base_yaw_deg", 0.0)), floor_z=float(c.get("floor_z", -0.72)),
                                pedestal=bool(c.get("pedestal", False)))
    return out


def get_platform(key: str) -> PlatformSpec:
    p = load_platforms()
    if key not in p:
        raise KeyError(f"unknown platform {key!r}, choose from {sorted(p)}")
    return p[key]


def _yaw_quat(deg: float) -> np.ndarray:
    a = np.radians(deg) / 2
    return np.array([np.cos(a), 0.0, 0.0, np.sin(a)])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R @ v


@dataclass
class PlatformScene:
    """Assembled MjModel + joint / end-effector indices + values of fixed joints. Shared by kinematics and rendering."""
    spec: PlatformSpec
    model: Any
    data: Any
    arm_qadr: np.ndarray
    arm_dadr: np.ndarray
    jnt_range: np.ndarray            # (n, 2)
    tcp_site: int
    fixed_qpos: np.ndarray           # default qpos of the whole model (fixed joints take the home key, arm joints take the home key)
    home_key_q: np.ndarray           # values of the arm joints at key "home" (IK initial value / null-space target)
    extra: dict[str, Any] = field(default_factory=dict)

    def set_q(self, q: Sequence[float]) -> None:
        self.data.qpos[:] = self.fixed_qpos
        self.data.qpos[self.arm_qadr] = np.asarray(q, float)


def build_platform_scene(spec: PlatformSpec, grid=None, sim_cfg: Optional[dict[str, Any]] = None,
                         dirt_shape: tuple[int, int] = (120, 150), tcp_len: Optional[float] = None,
                         wrist_cam_pos: tuple[float, float, float] = (0.0, -0.06, 0.16), dress: Any = None,
                         base_quat: Optional[Sequence[float]] = None) -> PlatformScene:
    """Tabletop scene + robot + tool. tcp_len = distance from the tool handle start to the polishing disc face (default: the UR3e TCP z offset 0.2596 m)."""
    import mujoco

    from ..common.zones import ZoneGrid
    from ..sim.scene import build_mjcf
    if spec.builtin:
        raise ValueError("builtin platform (UR3e) uses the existing code path, no assembly needed")
    grid = grid or ZoneGrid()
    sim_cfg = sim_cfg or load_yaml("sim.yaml")
    tcp_len = float(tcp_len if tcp_len is not None else DEFAULT_TCP_OFFSET[2])
    pad_r = float(sim_cfg["contact"]["pad_radius_m"])
    parent = mujoco.MjSpec.from_string(build_mjcf(grid, sim_cfg, dirt_shape, arm=False))
    parent.visual.global_.offwidth, parent.visual.global_.offheight = 1920, 1080   # 1080p offscreen rendering for video clips
    parent.visual.quality.shadowsize = 4096
    child = mujoco.MjSpec.from_file(str(menagerie_root() / spec.mjcf))
    # The Menagerie model brings its own lights (diffuse 0.7, two on Spot): stacked with the tabletop scene lights they wash the workspace out to white (both the perception image and the video are overexposed); delete them and use only the scene lights
    for lt in list(child.lights):
        child.delete(lt)
    for cam in list(child.cameras):                         # a same-named camera that comes with the model (e.g. "wrist" of the Kinova Gen3) conflicts with our wrist camera: delete it
        if cam.name == "wrist":
            child.delete(cam)
    # Turn off the robot's own collisions (convention: geometry takes no part in physical collisions; contact is computed by the virtual tabletop)
    for g in child.geoms:
        g.contype = 0
        g.conaffinity = 0
    # Tool: attach at the end effector (parent body of the site, by site pose; or directly on the body)
    if spec.ee_site:
        site = child.site(spec.ee_site)
        ee = site.parent
        mount_pos, mount_quat = np.array(site.pos, float), np.array(site.quat, float)
    else:
        ee = child.body(spec.ee_body)
        mount_pos, mount_quat = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    tq = _quat_mul(mount_quat, np.array(spec.tool_quat, float))
    tool = ee.add_body(name="cp_tool", pos=mount_pos.tolist(), quat=tq.tolist())
    s0 = spec.tool_start
    tool.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, fromto=[0, 0, s0, 0, 0, s0 + 0.20], size=[0.033, 0, 0],
                  rgba=[0.2, 0.2, 0.22, 1], contype=0, conaffinity=0)
    tool.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, fromto=[0, 0, s0 + tcp_len - 0.05, 0, 0, s0 + tcp_len - 0.012], size=[0.012, 0, 0],
                  rgba=[0.2, 0.2, 0.22, 1], contype=0, conaffinity=0)
    tool.add_geom(name="pad", type=mujoco.mjtGeom.mjGEOM_CYLINDER, fromto=[0, 0, s0 + tcp_len - 0.012, 0, 0, s0 + tcp_len],
                  size=[pad_r, 0, 0], rgba=[0.95, 0.85, 0.3, 1], contype=0, conaffinity=0)
    tool.add_site(name="tcp", pos=[0, 0, s0 + tcp_len], size=[0.004, 0, 0], rgba=[1, 0, 0, 1])
    wc = sim_cfg["cameras"]["wrist"]
    tool.add_camera(name="wrist", pos=[wrist_cam_pos[0], wrist_cam_pos[1], s0 + wrist_cam_pos[2]], xyaxes=[1, 0, 0, 0, -1, 0],
                    fovy=float(wc["fovy"]))
    # Robot home key (taken by joint name, mapped after assembly)
    cm = child.compile()
    cd = mujoco.MjData(cm)
    kid = mujoco.mj_name2id(cm, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(cm, cd, kid)
    child_q = {mujoco.mj_id2name(cm, mujoco.mjtObj.mjOBJ_JOINT, j): (int(cm.jnt_type[j]), cd.qpos[cm.jnt_qposadr[j]:cm.jnt_qposadr[j] + _qlen(int(cm.jnt_type[j]))].copy())
               for j in range(cm.njnt)}
    for k in list(child.keys):                              # the qpos length of the robot's own keyframe does not match the assembled model: delete
        child.delete(k)
    _add_room(parent, spec, grid, mujoco)
    if dress is not None:                                   # look only (used for video / console re-rendering, platform_render.dress_scene)
        dress(parent, mujoco)
    base_q = np.asarray(base_quat, float) if base_quat is not None else _yaw_quat(spec.base_yaw_deg)   # base_quat: another mounting posture for display (e.g. cart top facing up wiping the wall)
    frame = parent.worldbody.add_frame(pos=list(spec.base_pos), quat=base_q.tolist())
    parent.attach(child, frame=frame, prefix=PREFIX)
    model = parent.compile()
    data = mujoco.MjData(model)
    fixed = data.qpos.copy()
    for name, (jt, val) in child_q.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PREFIX + name)
        adr = model.jnt_qposadr[jid]
        if jt == 0:                                         # freejoint: position / orientation must be transformed to the world frame (after the base frame)
            pos = np.array(spec.base_pos) + _quat_rot(base_q, val[:3])
            fixed[adr:adr + 3] = pos
            fixed[adr + 3:adr + 7] = _quat_mul(base_q, val[3:7])
        else:
            fixed[adr:adr + len(val)] = val
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PREFIX + j) for j in spec.arm_joints]
    if min(jids) < 0:
        raise KeyError(f"{spec.key}: joints not found in the model {[j for j, i in zip(spec.arm_joints, jids) if i < 0]}")
    arm_qadr = np.array([model.jnt_qposadr[j] for j in jids])
    arm_dadr = np.array([model.jnt_dofadr[j] for j in jids])
    rng = np.array([model.jnt_range[j] if model.jnt_limited[j] else (-2 * np.pi, 2 * np.pi) for j in jids], float)
    tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, PREFIX + "tcp")
    return PlatformScene(spec, model, data, arm_qadr, arm_dadr, rng, tcp, fixed, fixed[arm_qadr].copy())


def table_footprint(grid: Any) -> tuple[np.ndarray, np.ndarray]:
    """Center and half width (xy) of the table board: use the whole tabletop registered by the site (zones.yaml table.center_xy / half_xy) if present, otherwise expand outward from the workspace.
    The table board (platform_render.add_table_top) and the table legs (_add_room) both take it from here, so the two no longer compute separately ("the table board is floating")."""
    if getattr(grid, "table_half", None) is not None:
        c = grid.table_center if getattr(grid, "table_center", None) is not None else grid.origin + grid.size / 2
        return np.asarray(c, float), np.asarray(grid.table_half, float)
    return (np.asarray(grid.origin + grid.size / 2, float),
            np.array([max(grid.size[0] / 2 + 0.12, 0.3), max(grid.size[1] / 2 + 0.15, 0.35)], float))


def table_leg_xy(grid: Any, inset: float = 0.04) -> list[tuple[float, float]]:
    """xy of the four table legs: inset from the four corners of the table board by inset."""
    c, h = table_footprint(grid)
    return [(float(c[0] + sx * (h[0] - inset)), float(c[1] + sy * (h[1] - inset))) for sx in (-1, 1) for sy in (-1, 1)]


def _add_room(parent: Any, spec: PlatformSpec, grid: Any, mujoco: Any) -> None:
    """Floor, table legs and arm column for rendering (geometry takes no part in contact; the tabletop itself comes from build_mjcf)."""
    zf = spec.floor_z
    w = parent.worldbody
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, pos=[0, 0, zf], size=[3, 3, 0.05], rgba=[0.86, 0.85, 0.82, 1],
               contype=0, conaffinity=0)
    zs = grid.z_surface
    for x, y in table_leg_xy(grid):
        w.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, y, (zf + zs - 0.04) / 2], size=[0.025, 0.025, (zs - 0.04 - zf) / 2],
                   rgba=[0.45, 0.43, 0.40, 1], contype=0, conaffinity=0)
    if spec.pedestal:
        bx, by, bz = spec.base_pos
        w.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_CYLINDER, fromto=[bx, by, zf, bx, by, bz], size=[0.085, 0, 0],
                   rgba=[0.30, 0.31, 0.33, 1], contype=0, conaffinity=0)
        w.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, fromto=[bx, by, zf, bx, by, zf + 0.03], size=[0.20, 0, 0],
                   rgba=[0.30, 0.31, 0.33, 1], contype=0, conaffinity=0)


def _qlen(jtype: int) -> int:
    return {0: 7, 1: 4}.get(jtype, 1)


class PlatformKinematics:
    """Same interface as UR3eKinematics: fk / fk_pose / jacobian / ik; plus home_q / survey_q (solved by IK)."""

    def __init__(self, scene: PlatformScene, null_gain: float = 0.1):
        import mujoco
        self.mj = mujoco
        self.scene = scene
        self.n = len(scene.arm_qadr)
        self.null_gain = null_gain
        self.q_rest = scene.home_key_q.copy()
        self._jacp = np.zeros((3, scene.model.nv))
        self._jacr = np.zeros((3, scene.model.nv))
        self.home_q: Optional[np.ndarray] = None
        self.survey_q: Optional[np.ndarray] = None

    def fk(self, q: Sequence[float]) -> np.ndarray:
        s = self.scene
        s.set_q(q)
        self.mj.mj_kinematics(s.model, s.data)
        T = np.eye(4)
        T[:3, :3] = s.data.site_xmat[s.tcp_site].reshape(3, 3)
        T[:3, 3] = s.data.site_xpos[s.tcp_site]
        return T

    def fk_pose(self, q: Sequence[float]) -> np.ndarray:
        from .kinematics import matrix_to_pose
        return matrix_to_pose(self.fk(q))

    def jacobian(self, q: Sequence[float]) -> np.ndarray:
        s = self.scene
        s.set_q(q)
        self.mj.mj_kinematics(s.model, s.data)
        self.mj.mj_comPos(s.model, s.data)
        self.mj.mj_jacSite(s.model, s.data, self._jacp, self._jacr, s.tcp_site)
        return np.vstack([self._jacp[:, s.arm_dadr], self._jacr[:, s.arm_dadr]])

    def clamp(self, q: np.ndarray) -> np.ndarray:
        r = self.scene.jnt_range
        return np.clip(q, r[:, 0], r[:, 1])

    def ik(self, target: np.ndarray | Sequence[float], q0: Sequence[float], tol_pos: float = 1e-5, tol_rot: float = 1e-4,
           max_iter: int = 300, damping: float = 1e-3, free_yaw: bool = False) -> Optional[np.ndarray]:
        """free_yaw: do not constrain the rotation about the tool z axis (the wiping head is a cylinder, the angle is invisible; used by render re-solving, the extra freedom goes to the null space to avoid limits)."""
        T_goal = pose_to_matrix(target) if np.asarray(target).shape == (6,) else np.asarray(target)
        q = self.clamp(np.array(q0, dtype=float))
        for _ in range(max_iter):
            T = self.fk(q)
            e_p = T_goal[:3, 3] - T[:3, 3]
            if free_yaw:                                      # only align the tool z axis
                z = T[:3, 2]
                e_r = np.cross(z, T_goal[:3, 2])
            else:
                e_r = matrix_to_rotvec(T_goal[:3, :3] @ T[:3, :3].T)
            if np.linalg.norm(e_p) < tol_pos and np.linalg.norm(e_r) < tol_rot and (not free_yaw or float(z @ T_goal[:3, 2]) > 0.0):
                return q                                      # free_yaw: the cross product is also 0 when the z axis is reversed, not counted as converged
            J = self.jacobian(q)
            if free_yaw:
                J[3:] = (np.eye(3) - np.outer(z, z)) @ J[3:]
            e = np.concatenate([e_p, e_r])
            JJt = J @ J.T + damping * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            step = np.linalg.norm(dq)
            if step > 0.2:                                   # the task step and the null-space step are clamped separately (clamping them together slows convergence)
                dq *= 0.2 / step
            if self.n > 6 and np.linalg.norm(e) > 1e-3:      # null space: pull back to home; the damped projection is inexact, so turn it off near convergence or the error cannot go down
                Jp = J.T @ np.linalg.solve(JJt, np.eye(6))
                dn = (np.eye(self.n) - Jp @ J) @ (self.null_gain * (self.q_rest - q))
                nn = np.linalg.norm(dn)
                dq += dn * (min(1.0, 0.05 / nn) if nn > 0 else 0.0)
            q = self.clamp(q + dq)
        return None

    def limit_margin_deg(self, q: Sequence[float]) -> float:
        r = self.scene.jnt_range
        q = np.asarray(q, float)
        return float(np.degrees(np.min(np.minimum(q - r[:, 0], r[:, 1] - q))))

    def solve_home(self, rcfg: Optional[dict[str, Any]] = None, grid=None) -> dict[str, Any]:
        """home / survey: the same TCP poses as the UR3e mounting Home / survey pose (computed with UR3e kinematics), IK starting from key "home"."""
        from ..common.zones import ZoneGrid
        rcfg = rcfg or load_yaml("robot.yaml")
        grid = grid or ZoneGrid()
        ur = UR3eKinematics(np.array(rcfg["startup"]["expected_tcp_offset"], float))
        T_home = ur.fk(np.array(rcfg["home_q"], float))
        from .kinematics import rotvec_to_matrix
        R_tool = rotvec_to_matrix(grid.tool_rotvec)
        if grid.home_tcp is not None:                        # the site gives the TCP position of Home: tool pointing down, take the site tool orientation
            T_home = np.eye(4)
            T_home[:3, :3], T_home[:3, 3] = R_tool, grid.home_tcp
        out: dict[str, Any] = {"platform": self.scene.spec.key}
        q = self.ik(T_home, self.q_rest)
        if q is None:
            raise RuntimeError(f"{self.scene.spec.key}: home pose IK has no solution (adjust base_pos in configs/platforms.yaml)")
        self.home_q = q
        self.q_rest = q.copy()                               # later null-space target = the solved home
        out["home_q"] = q.tolist()
        out["home_limit_margin_deg"] = self.limit_margin_deg(q)
        if self.scene.spec.key in grid.survey_arm_q:              # the site directly gives joint angles of the survey pose for this platform (e.g. the stowed arm pose of Spot)
            self.survey_q = np.array(grid.survey_arm_q[self.scene.spec.key], float)
            out["survey_q"] = self.survey_q.tolist()
        elif grid.survey_tcp is not None or grid.survey_q is not None:
            if grid.survey_tcp is not None:                  # the site gives a survey-pose TCP (arm stowed outside the fixed camera view)
                T_s = np.eye(4)
                T_s[:3, :3], T_s[:3, 3] = R_tool, grid.survey_tcp
            else:
                T_s = ur.fk(grid.survey_q)
            qs = self.ik(T_s, q)
            self.survey_q = qs
            out["survey_q"] = qs.tolist() if qs is not None else None
        return out


def workspace_points(grid=None, z: float = 0.0) -> list[np.ndarray]:
    """Target TCP poses at the four workspace corners (inset by one polishing disc radius) + the center, tool pointing down (zones.yaml tool_rotvec)."""
    from ..common.zones import ZoneGrid
    from .kinematics import rotvec_to_matrix
    grid = grid or ZoneGrid()
    R = rotvec_to_matrix(grid.tool_rotvec)
    o, sz, m = grid.origin, grid.size, 0.03
    pts = [(o[0] + m, o[1] + m), (o[0] + sz[0] - m, o[1] + m), (o[0] + m, o[1] + sz[1] - m), (o[0] + sz[0] - m, o[1] + sz[1] - m),
           (o[0] + sz[0] / 2, o[1] + sz[1] / 2)]
    out = []
    for x, y in pts:
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = (x, y, grid.z_surface + z)
        out.append(T)
    return out
