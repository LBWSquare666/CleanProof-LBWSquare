"""Workspace zone grid (configs/zones.yaml).

Zone ID: row letter (along base +x) + column number (along base +y), e.g. "B3".
All coordinates are in the base frame, in m.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from .config import load_yaml


@dataclass(frozen=True)
class ZoneRect:
    zone_id: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2)

    @property
    def size(self) -> tuple[float, float]:
        return (self.xmax - self.xmin, self.ymax - self.ymin)

    def contains(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def sub_rect(self, bbox_norm: list[float]) -> "ZoneRect":
        """Normalized box [u0, v0, u1, v1] inside a zone (u along x, v along y) → rectangle in the base frame."""
        u0, v0, u1, v1 = bbox_norm
        w, h = self.size
        return ZoneRect(self.zone_id, self.xmin + u0 * w, self.ymin + v0 * h, self.xmin + u1 * w, self.ymin + v1 * h)


class ZoneGrid:
    def __init__(self, cfg: dict[str, Any] | None = None):
        cfg = cfg or load_yaml("zones.yaml")
        wa = cfg["work_area"]
        self.z_surface = float(cfg["table"]["z_surface"])
        self.origin = np.array(wa["origin_xy"], dtype=float)
        self.size = np.array(wa["size_xy"], dtype=float)
        self.rows, self.cols = int(wa["rows"]), int(wa["cols"])
        self.cell = self.size / np.array([self.rows, self.cols])
        self.tool_rotvec = np.array(cfg.get("tool_rotvec", [np.pi, 0.0, 0.0]), dtype=float)
        self.survey_q = np.array(cfg["survey_q"], dtype=float) if "survey_q" in cfg else None
        # Optional site items (configs/site/<site>/zones.yaml): center and half size of the whole tabletop (for display rendering; defaults to the workspace expanded outward),
        # TCP positions of the survey pose / Home for non-UR3e platforms (default: FK of the UR3e survey pose / Home), props on the table (mug)
        t = cfg["table"]
        self.table_center = np.array(t["center_xy"], float) if "center_xy" in t else None
        self.table_half = np.array(t["half_xy"], float) if "half_xy" in t else None
        self.mug = t.get("mug")                                   # {"xy": [x, y], "yaw_deg": d} or None
        self.survey_tcp = np.array(cfg["survey_tcp_xyz"], float) if "survey_tcp_xyz" in cfg else None
        self.home_tcp = np.array(cfg["home_tcp_xyz"], float) if "home_tcp_xyz" in cfg else None
        self.survey_arm_q = dict(cfg.get("survey_arm_q") or {})  # platform → survey-pose joint angles (stowed arm pose, the tool need not point down)
        self._zones = {z.zone_id: z for z in self._build()}

    @staticmethod
    def zone_id(r: int, c: int) -> str:
        return f"{chr(ord('A') + r)}{c + 1}"

    def _build(self) -> Iterator[ZoneRect]:
        for r in range(self.rows):
            for c in range(self.cols):
                x0 = self.origin[0] + r * self.cell[0]
                y0 = self.origin[1] + c * self.cell[1]
                yield ZoneRect(self.zone_id(r, c), x0, y0, x0 + self.cell[0], y0 + self.cell[1])

    @property
    def ids(self) -> list[str]:
        return list(self._zones)

    def __getitem__(self, zone_id: str) -> ZoneRect:
        try:
            return self._zones[zone_id]
        except KeyError as e:
            raise KeyError(f"unknown zone {zone_id!r}; valid zones {self.ids[0]}..{self.ids[-1]}") from e

    def __contains__(self, zone_id: str) -> bool:
        return zone_id in self._zones

    def zone_at(self, x: float, y: float) -> str | None:
        r = int(np.floor((x - self.origin[0]) / self.cell[0]))
        c = int(np.floor((y - self.origin[1]) / self.cell[1]))
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.zone_id(r, c)
        return None

    def corners_3d(self, zone_id: str) -> np.ndarray:
        z = self[zone_id]
        s = self.z_surface
        return np.array([[z.xmin, z.ymin, s], [z.xmax, z.ymin, s], [z.xmax, z.ymax, s], [z.xmin, z.ymax, s]])

    def work_area_rect(self) -> ZoneRect:
        return ZoneRect("ALL", *self.origin, *(self.origin + self.size))
