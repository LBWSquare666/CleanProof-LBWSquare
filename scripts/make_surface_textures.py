#!/usr/bin/env python3
"""Cut small tabletop textures for perception rendering (assets/surfaces/<surface>.png) out of the raw CC0 textures.

Each small texture covers the work area exactly (configs/zones.yaml work_area), with resolution = stain grid x dirt.render_scale,
rows along base +x and columns along base +y (same orientation as the calibration map). The source image is tiled over the tabletop as "whole texture = size_m meters" and sampled starting at origin_m,

  python -I scripts/make_surface_textures.py --store ./store/cc0_assets

The raw textures are not in the repo; the generated small textures are in the repo, so a new machine does not need to download again.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cleanproof.common.config import load_yaml  # noqa: E402
from cleanproof.common.zones import ZoneGrid  # noqa: E402


def sample(img: np.ndarray, size_m: float, origin_m: tuple[float, float], xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Color of the source image (row 0 at the top, u to the right = +x, v up = +y, whole image = size_m meters, tiled) at tabletop coordinates (x, y)."""
    h, w = img.shape[:2]
    X, Y = np.meshgrid(xs, ys, indexing="ij")                    # (nx, ny): rows along x, columns along y
    u = ((X - origin_m[0]) / size_m) % 1.0
    v = ((Y - origin_m[1]) / size_m) % 1.0
    map_x = (u * w - 0.5).astype(np.float32)
    map_y = ((1.0 - v) * h - 0.5).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, help="CC0 raw texture directory (store/cc0_assets)")
    ap.add_argument("--out", default=str(ROOT / "assets" / "surfaces"))
    args = ap.parse_args()
    dcfg = load_yaml("sim.yaml")["dirt"]
    grid = ZoneGrid()
    res = float(dcfg["resolution_m"])
    k = int(dcfg["render_scale"])
    nx, ny = int(round(grid.size[0] / res)) * k, int(round(grid.size[1] / res)) * k
    xs = grid.origin[0] + (np.arange(nx) + 0.5) * grid.size[0] / nx
    ys = grid.origin[1] + (np.arange(ny) + 0.5) * grid.size[1] / ny
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    made = {}
    for name, spec in dcfg["surfaces"].items():
        src = spec.get("source")
        if not src:
            continue
        path = Path(args.store) / src["file"]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        size_m = float(src["size_m"])
        # first downsample the source image to "target pixel size x 2" density before sampling, to avoid moire from point-sampling a 4k original
        px_m = grid.size[0] / nx
        scale = min(1.0, 2.0 * (size_m / px_m) / bgr.shape[1])
        if scale < 1.0:
            bgr = cv2.resize(bgr, (int(bgr.shape[1] * scale), int(bgr.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        tile = sample(bgr, size_m, tuple(src.get("origin_m", [0.0, 0.0])), xs, ys)
        if "gain" in src:                                         # overall brightness (used to match the look under the Blender lighting)
            tile = np.clip(tile.astype(np.float32) * float(src["gain"]), 0, 255).astype(np.uint8)
        dst = ROOT / spec["texture"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), tile)
        mean = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).reshape(-1, 3).mean(0)
        made[name] = {"texture": spec["texture"], "source": src, "shape": [nx, ny], "mean_rgb": [round(float(v), 1) for v in mean]}
        print(f"{name:16s} {dst.relative_to(ROOT)}  {nx}×{ny}  mean RGB {np.round(mean, 1).tolist()}")
    (out / "surfaces.json").write_text(json.dumps(made, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
