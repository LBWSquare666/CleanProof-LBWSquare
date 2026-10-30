"""Classical vision residue detection: Lab color difference (CIE76 ΔE) between the top-down image and the clean reference image; residue fraction and box per zone.

Deterministic and well localized; serves as candidates and evidence for the VLM verifier, while the VLM tells stains apart from glare, shadow and cloth lint.
Without a reference image it falls back to "deviation from the zone's median color".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..common.zones import ZoneGrid
from .rectify import DEFAULT_PX_PER_M, zone_slices


@dataclass
class ZoneResidue:
    zone_id: str
    residue_fraction: float
    bbox_norm: Optional[list[float]]   # [u0, v0, u1, v1], u along x (rows), v along y (columns)
    mean_delta_e: float


def _lab(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


class ResidueDetector:
    def __init__(self, grid: ZoneGrid, reference_rgb: Optional[np.ndarray] = None, delta_e_thr: float = 12.0,
                 px_per_m: float = DEFAULT_PX_PER_M, min_blob_px: int = 12):
        self.grid = grid
        self.px = px_per_m
        self.thr = delta_e_thr
        self.min_blob = min_blob_px
        self.ref_rgb = reference_rgb
        self.ref_lab = _lab(reference_rgb) if reference_rgb is not None else None

    def set_reference(self, reference_rgb: np.ndarray) -> None:
        self.ref_rgb = reference_rgb
        self.ref_lab = _lab(reference_rgb)

    def delta_e_map(self, rectified_rgb: np.ndarray) -> np.ndarray:
        lab = _lab(rectified_rgb)
        if self.ref_lab is not None:
            return np.linalg.norm(lab - self.ref_lab, axis=-1)
        # No reference: compare with each zone's median color
        out = np.zeros(lab.shape[:2], dtype=np.float32)
        for zid in self.grid.ids:
            si, sj = zone_slices(self.grid, zid, self.px)
            patch = lab[si, sj]
            med = np.median(patch.reshape(-1, 3), axis=0)
            out[si, sj] = np.linalg.norm(patch - med, axis=-1)
        return out

    def mask(self, rectified_rgb: np.ndarray) -> np.ndarray:
        m = (self.delta_e_map(rectified_rgb) > self.thr).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros_like(m)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] >= self.min_blob:
                keep[lab == k] = 1
        return keep.astype(bool)

    def analyze(self, rectified_rgb: np.ndarray, zone_ids: Optional[list[str]] = None) -> dict[str, ZoneResidue]:
        de = self.delta_e_map(rectified_rgb)
        m = self.mask(rectified_rgb)
        out = {}
        for zid in zone_ids or self.grid.ids:
            si, sj = zone_slices(self.grid, zid, self.px)
            zm = m[si, sj]
            bbox = None
            if zm.any():
                ii, jj = np.nonzero(zm)
                h, w = zm.shape
                bbox = [float(ii.min() / h), float(jj.min() / w), float((ii.max() + 1) / h), float((jj.max() + 1) / w)]
            out[zid] = ZoneResidue(zid, float(zm.mean()), bbox, float(de[si, sj].mean()))
        return out
