#!/usr/bin/env python3
"""Export the original emblem set to assets/brand_icons/ (PNG 1024 px with badge, *_glyph.png glyph only, SVG). The geometry is in cleanproof/sim/emblems.py.

  .venv/bin/python scripts/make_brand_icons.py [--out assets/brand_icons] [--size 1024]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanproof.sim.emblems import export_all  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="assets/brand_icons")
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()
    for p in export_all(Path(args.out), args.size):
        print(f"[icons] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
