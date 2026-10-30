"""UR3e kinematics (official UR nominal DH parameters) and pose utilities.

Pose convention is the same as the UR controller: pose = [x, y, z, rx, ry, rz], position in m, orientation as a rotation vector,
reference frame is the controller base frame (the frame of getActualTCPPose).

Nominal DH compared with the real-robot Home record (recorded UR3e home pose):
position error 1.9 mm, orientation error 0.43° (tests/test_kinematics.py), from factory calibration differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# DH parameters of the UR3e published officially by Universal Robots
UR3E_DH_A = np.array([0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0])
UR3E_DH_D = np.array([0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921])
UR3E_DH_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])

# TCP on the teach pendant (our UR3e setup notes, named withpolisher)
DEFAULT_TCP_OFFSET = np.array([0.0024, 0.00241, 0.25961, 0.0, 0.0, 0.0])

# UR3e joint limits: wrist 3 rotates without limit (±2 turns taken here), the others ±2π
JOINT_LIMITS = np.array([[-2 * np.pi, 2 * np.pi]] * 5 + [[-4 * np.pi, 4 * np.pi]])


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rotvec_to_matrix(rv: Sequence[float]) -> np.ndarray:
    rv = np.asarray(rv, dtype=float)
    th = np.linalg.norm(rv)
    if th < 1e-12:
        return np.eye(3)
    k = skew(rv / th)
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * k @ k


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    cos = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(cos)
    if th < 1e-9:
        return np.zeros(3)
    if np.pi - th < 1e-6:
        # Close to 180°: take the axis from the dominant column of R+I
        M = (R + np.eye(3)) / 2
        axis = M[:, np.argmax(np.diag(M))]
        axis = axis / np.linalg.norm(axis)
        return axis * th
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return axis * th


def pose_to_matrix(pose: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotvec_to_matrix(pose[3:6])
    T[:3, 3] = pose[:3]
    return T


def matrix_to_pose(T: np.ndarray) -> np.ndarray:
    return np.concatenate([T[:3, 3], matrix_to_rotvec(T[:3, :3])])


def _dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


@dataclass
class UR3eKinematics:
    tcp_offset: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.tcp_offset is None:
            self.tcp_offset = DEFAULT_TCP_OFFSET.copy()
        self._T_tcp = pose_to_matrix(self.tcp_offset)

    def frames(self, q: Sequence[float]) -> list[np.ndarray]:
        """Frames T_0..T_6 of each DH coordinate system in the base frame (T_6 is the flange tool0)."""
        T = np.eye(4)
        out = [T.copy()]
        for i in range(6):
            T = T @ _dh(q[i], UR3E_DH_D[i], UR3E_DH_A[i], UR3E_DH_ALPHA[i])
            out.append(T.copy())
        return out

    def fk(self, q: Sequence[float]) -> np.ndarray:
        """TCP pose matrix (base frame)."""
        return self.frames(q)[-1] @ self._T_tcp

    def fk_pose(self, q: Sequence[float]) -> np.ndarray:
        return matrix_to_pose(self.fk(q))

    def jacobian(self, q: Sequence[float]) -> np.ndarray:
        """Geometric Jacobian of the TCP (base frame, row order [v; ω])."""
        fr = self.frames(q)
        p = (fr[-1] @ self._T_tcp)[:3, 3]
        J = np.zeros((6, 6))
        for i in range(6):
            z = fr[i][:3, 2]
            o = fr[i][:3, 3]
            J[:3, i] = np.cross(z, p - o)
            J[3:, i] = z
        return J

    def ik(
        self,
        target: np.ndarray | Sequence[float],
        q0: Sequence[float],
        tol_pos: float = 1e-5,
        tol_rot: float = 1e-4,
        max_iter: int = 200,
        damping: float = 1e-3,
    ) -> Optional[np.ndarray]:
        """Damped least-squares numerical inverse kinematics, converging near q0 (used the same way as UR getInverseKinematics(qnear))."""
        T_goal = pose_to_matrix(target) if np.asarray(target).shape == (6,) else np.asarray(target)
        q = np.array(q0, dtype=float)
        for _ in range(max_iter):
            T = self.fk(q)
            e_p = T_goal[:3, 3] - T[:3, 3]
            e_r = matrix_to_rotvec(T_goal[:3, :3] @ T[:3, :3].T)
            if np.linalg.norm(e_p) < tol_pos and np.linalg.norm(e_r) < tol_rot:
                return q
            J = self.jacobian(q)
            e = np.concatenate([e_p, e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
            step = np.linalg.norm(dq)
            if step > 0.2:
                dq *= 0.2 / step
            q = q + dq
        return None


def adaptive_damping(J: np.ndarray, sigma_thr: float = 0.02, lam_max: float = 1e-2, lam_min: float = 1e-8) -> float:
    """Almost no damping away from singularities, smoothly increased when the smallest singular value is below the threshold (standard DLS practice)."""
    s_min = np.linalg.svd(J, compute_uv=False)[-1]
    if s_min >= sigma_thr:
        return lam_min
    return lam_min + lam_max * (1.0 - (s_min / sigma_thr) ** 2)


def twist_to_qdot(kin: UR3eKinematics, q: Sequence[float], twist: Sequence[float], damping: Optional[float] = None) -> np.ndarray:
    """Base-frame TCP velocity (speedL semantics) → joint velocities (damped pseudo-inverse)."""
    J = kin.jacobian(q)
    lam = adaptive_damping(J) if damping is None else damping
    return J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), np.asarray(twist, dtype=float))


def elbow_margin_deg(q: Sequence[float]) -> float:
    """Angle of the elbow joint from fully straight (our UR3e return-to-home planning requires ≥ 15°)."""
    e = (q[2] + np.pi) % (2 * np.pi) - np.pi
    return float(np.degrees(abs(e)))


def horizontal_radius(pose: Sequence[float]) -> float:
    """Horizontal distance from the TCP to the base (our UR3e setup limits it to ≤ 0.49 m; the real robot once protective-stopped at 0.53 m)."""
    return float(np.hypot(pose[0], pose[1]))


def robot_radius(robot: object, p: Sequence[float]) -> float:
    """Horizontal radius of the TCP relative to this robot's base: platform-adapted simulation backends have radius_of (base not at the origin); the real robot / UR3e uses the origin."""
    fn = getattr(robot, "radius_of", None)
    return float(fn(p)) if fn is not None else horizontal_radius(p)


def robot_max_radius(robot: object, cfg: dict) -> float:
    """Reach radius upper limit: the platform's own if it gives one, otherwise robot.yaml safety.max_horizontal_radius_m (original UR3e value)."""
    m = getattr(robot, "max_radius_m", None)
    return float(m) if m is not None else float(cfg["max_horizontal_radius_m"])
