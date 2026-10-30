"""Workspace top-down rectification: tabletop-plane homography → top-down image of uniform resolution, rows along +x (row A on top), columns along +y (column 1 on the left).

After rectification each zone is a fixed pixel block, easy to crop per zone, compare with the reference image and hand to the VLM.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..common.zones import ZoneGrid
from .camera import CameraModel

DEFAULT_PX_PER_M = 2000.0   # 0.5 mm / pixel


def rectified_shape(grid: ZoneGrid, px_per_m: float = DEFAULT_PX_PER_M) -> tuple[int, int]:
    return int(round(grid.size[0] * px_per_m)), int(round(grid.size[1] * px_per_m))


def work_area_homography(cam: CameraModel, grid: ZoneGrid, px_per_m: float = DEFAULT_PX_PER_M) -> np.ndarray:
    """Camera image pixels → homography matrix to top-down image pixels (col=j along y, row=i along x)."""
    x0, y0 = grid.origin
    x1, y1 = grid.origin + grid.size
    z = grid.z_surface
    world = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]])
    src = cam.zone_polygon(world).astype(np.float32)
    h, w = rectified_shape(grid, px_per_m)
    # Top-down image coordinates (OpenCV points are (col, row)): (x0,y0)→(0,0), (x1,y0)→(0,h), (x1,y1)→(w,h), (x0,y1)→(w,0)
    dst = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def rectify(image: np.ndarray, cam: CameraModel, grid: ZoneGrid, px_per_m: float = DEFAULT_PX_PER_M) -> np.ndarray:
    H = work_area_homography(cam, grid, px_per_m)
    h, w = rectified_shape(grid, px_per_m)
    return cv2.warpPerspective(image, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def zone_slices(grid: ZoneGrid, zone_id: str, px_per_m: float = DEFAULT_PX_PER_M) -> tuple[slice, slice]:
    z = grid[zone_id]
    i0 = int(round((z.xmin - grid.origin[0]) * px_per_m))
    i1 = int(round((z.xmax - grid.origin[0]) * px_per_m))
    j0 = int(round((z.ymin - grid.origin[1]) * px_per_m))
    j1 = int(round((z.ymax - grid.origin[1]) * px_per_m))
    return slice(i0, i1), slice(j0, j1)


def zone_crop(rectified: np.ndarray, grid: ZoneGrid, zone_id: str, px_per_m: float = DEFAULT_PX_PER_M) -> np.ndarray:
    si, sj = zone_slices(grid, zone_id, px_per_m)
    return rectified[si, sj]


def draw_grid(rectified: np.ndarray, grid: ZoneGrid, px_per_m: float = DEFAULT_PX_PER_M,
              statuses: dict[str, str] | None = None) -> np.ndarray:
    """Draw the zone grid and zone IDs on the top-down image (overview image for the VLM and for the console)."""
    img = rectified.copy()
    colors = {"clean": (40, 160, 60), "residue": (210, 50, 40), "uncertain": (220, 160, 0)}
    for zid in grid.ids:
        si, sj = zone_slices(grid, zid, px_per_m)
        c = colors.get((statuses or {}).get(zid, ""), (60, 60, 60))
        cv2.rectangle(img, (sj.start, si.start), (sj.stop - 1, si.stop - 1), c, 2)
        cv2.putText(img, zid, (sj.start + 4, si.start + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
    return img
