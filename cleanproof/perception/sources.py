"""Camera sources: uniformly return the workspace top-down image (rectified) and the raw frame.

- SimCameraSource: MuJoCo rendering (needs GL), or topdown mode, which generates the top-down image directly from the stain map without GL;
  used for tests and for reviewers running on a laptop, and the image is labeled as synthetic.
- HttpCameraSource: the camera service on the robot host (RealSense), GET /frame/<camera> returns a JPEG.
- ReplaySource: read images from a recording directory (offline replay and evaluation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Protocol

import cv2
import numpy as np

from ..common.zones import ZoneGrid
from .camera import CameraModel
from .rectify import DEFAULT_PX_PER_M, rectified_shape, rectify


class CameraSource(Protocol):
    name: str

    def rectified(self) -> np.ndarray: ...
    def frame(self, camera: str = "fixed") -> np.ndarray: ...


class SimCameraSource:
    def __init__(self, robot: Any, grid: ZoneGrid, mode: str = "render", px_per_m: float = DEFAULT_PX_PER_M):
        if mode not in ("render", "topdown"):
            raise ValueError("mode must be render or topdown")
        self.robot, self.grid, self.mode, self.px = robot, grid, mode, px_per_m
        self.name = f"sim-{mode}"

    def frame(self, camera: str = "fixed") -> np.ndarray:
        if self.mode == "topdown":
            return self.rectified()
        return self.robot.render(camera)

    def camera_model(self, camera: str = "fixed") -> CameraModel:
        return self.robot.camera_model(camera)

    def rectified(self) -> np.ndarray:
        if self.mode == "topdown":
            h, w = rectified_shape(self.grid, self.px)
            rgb = self.robot.dirt.rgb()              # (nx, ny, 3), rows along x, columns along y
            return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        return rectify(self.robot.render("fixed"), self.robot.camera_model("fixed"), self.grid, self.px)


class HttpCameraSource:
    """Client of the camera service on the robot host (the server is provided by scripts/server/ or the camera process on robot-pc)."""

    def __init__(self, base_url: str, grid: ZoneGrid, camera_models: dict[str, CameraModel],
                 px_per_m: float = DEFAULT_PX_PER_M, timeout_s: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.grid, self.cams, self.px, self.timeout = grid, camera_models, px_per_m, timeout_s
        self.name = f"http:{self.base_url}"

    def frame(self, camera: str = "fixed") -> np.ndarray:
        import httpx

        from ..robot.server import auth_headers

        r = httpx.get(f"{self.base_url}/frame/{camera}", timeout=self.timeout, headers=auth_headers())
        r.raise_for_status()
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def camera_model(self, camera: str = "fixed") -> CameraModel:
        return self.cams[camera]

    def rectified(self) -> np.ndarray:
        return rectify(self.frame("fixed"), self.cams["fixed"], self.grid, self.px)


class ReplaySource:
    def __init__(self, directory: str | Path, grid: ZoneGrid, camera_model: Optional[CameraModel] = None,
                 px_per_m: float = DEFAULT_PX_PER_M):
        self.files = sorted(Path(directory).glob("*.jpg")) + sorted(Path(directory).glob("*.png"))
        self.grid, self.cam, self.px = grid, camera_model, px_per_m
        self.i = 0
        self.name = f"replay:{directory}"

    def frame(self, camera: str = "fixed") -> np.ndarray:
        img = cv2.cvtColor(cv2.imread(str(self.files[min(self.i, len(self.files) - 1)])), cv2.COLOR_BGR2RGB)
        self.i += 1
        return img

    def rectified(self) -> np.ndarray:
        img = self.frame()
        return img if self.cam is None else rectify(img, self.cam, self.grid, self.px)


def encode_jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()
