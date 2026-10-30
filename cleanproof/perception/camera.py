"""Pinhole camera model (OpenCV convention: x right, y down, z forward) and projection utilities.

Real robot: K comes from the RealSense intrinsics, T_base_cam from calibration (PLAN §8: ArUco + TCP touch).
Simulation: generated from the MuJoCo camera pose and fovy (mujoco_backend.camera_model).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraModel:
    K: np.ndarray            # 3×3
    T_base_cam: np.ndarray   # 4×4, camera frame (OpenCV) → base frame
    width: int
    height: int

    @classmethod
    def from_fovy(cls, fovy_deg: float, width: int, height: int, T_base_cam: np.ndarray) -> "CameraModel":
        f = (height / 2) / np.tan(np.radians(fovy_deg) / 2)
        K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
        return cls(K=K, T_base_cam=T_base_cam, width=width, height=height)

    def project(self, pts_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Base-frame points (N,3) → pixels (N,2) and depths (N,)."""
        pts = np.atleast_2d(pts_base)
        T_cam_base = np.linalg.inv(self.T_base_cam)
        pc = (T_cam_base[:3, :3] @ pts.T).T + T_cam_base[:3, 3]
        z = pc[:, 2]
        uv = (self.K @ (pc / z[:, None]).T).T[:, :2]
        return uv, z

    def zone_polygon(self, corners_base: np.ndarray) -> np.ndarray:
        uv, z = self.project(corners_base)
        if np.any(z <= 0):
            raise ValueError("zone is behind the camera")
        return uv


def mujoco_to_opencv_rotation(R_mj: np.ndarray) -> np.ndarray:
    """MuJoCo camera frame (looks along −z, y up) → OpenCV camera frame (looks along +z, y down)."""
    return R_mj @ np.diag([1.0, -1.0, -1.0])


@dataclass
class PlaneHomography:
    """Homography "camera" on the tabletop plane: base-frame (x, y) → pixel (u, v).

    Real-robot calibration (scripts/calibrate_site.py) touches ArUco markers with the TCP to get ≥4 (x, y)↔(u, v) correspondences and solves the homography directly;
    it describes only this one plane, which is exactly what top-down rectification needs (rectify only calls zone_polygon). No lens distortion correction (the D455 color stream has very little distortion).
    """

    H_xy_to_px: np.ndarray   # 3×3
    width: int
    height: int

    @classmethod
    def from_points(cls, xy: np.ndarray, px: np.ndarray, width: int, height: int) -> "PlaneHomography":
        import cv2

        H, _ = cv2.findHomography(np.asarray(xy, np.float64), np.asarray(px, np.float64), 0)
        if H is None:
            raise ValueError("correspondences are degenerate, cannot solve the homography")
        return cls(H_xy_to_px=H, width=width, height=height)

    def project(self, pts_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.atleast_2d(pts_base)[:, :2]
        h = (self.H_xy_to_px @ np.c_[pts, np.ones(len(pts))].T).T
        return h[:, :2] / h[:, 2:3], np.ones(len(pts))

    def zone_polygon(self, corners_base: np.ndarray) -> np.ndarray:
        return self.project(corners_base)[0]

    def reprojection_error_px(self, xy: np.ndarray, px: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.project(np.asarray(xy))[0] - np.asarray(px, float), axis=1)

    def to_dict(self) -> dict:
        return {"type": "plane_homography", "H_xy_to_px": np.asarray(self.H_xy_to_px).tolist(),
                "width": int(self.width), "height": int(self.height)}

    @classmethod
    def from_dict(cls, d: dict) -> "PlaneHomography":
        return cls(H_xy_to_px=np.array(d["H_xy_to_px"], dtype=float), width=int(d["width"]), height=int(d["height"]))
