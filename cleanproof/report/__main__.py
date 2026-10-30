"""Command line: python -m cleanproof.report <run_dir> [-o out.html] [--no-embed]"""

from __future__ import annotations

import argparse
from pathlib import Path

from .html import build_report


def main() -> int:
    ap = argparse.ArgumentParser(description="generate a single-file HTML cleaning report from a run directory")
    ap.add_argument("run_dir", help="runs/<run_id>/")
    ap.add_argument("-o", "--out", default=None, help="output path (default <run_dir>/report.html)")
    ap.add_argument("--no-embed", action="store_true", help="reference images by relative path instead of embedding base64 (the report must be placed inside the run directory)")
    args = ap.parse_args()
    out = build_report(Path(args.run_dir), args.out, embed_images=not args.no_embed)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
