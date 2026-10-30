#!/usr/bin/env python3
"""Reachability scan of the base position for a new platform (the criteria in the comments of configs/platforms.yaml, consistent with tests/test_platforms.py):

  home / survey pose IK has a solution and the home joint margin is >= 10 deg; the four corners + center of the work area x two heights (tabletop, +6 cm) all have solutions and every point has a margin >= 5 deg;
  among the candidate base positions, take the one with the largest "minimum joint margin".

  python scripts/scan_platform_base.py --key ur10e   # first write the platform entry into configs/platforms.yaml (base_pos can be anything)
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanproof.robot.platforms import PlatformKinematics, build_platform_scene, load_platforms, workspace_points  # noqa: E402


def evaluate(spec) -> tuple[bool, float, str]:
    try:
        kin = PlatformKinematics(build_platform_scene(spec))
        kin.solve_home()
    except Exception as e:                                   # e.g. home has no solution
        return False, -1.0, str(e)[:60]
    if kin.survey_q is None:
        return False, -1.0, "survey has no solution"
    m_home = kin.limit_margin_deg(kin.home_q)
    if m_home < 10.0:
        return False, m_home, f"home margin {m_home:.1f} deg"
    m_sv = kin.limit_margin_deg(kin.survey_q)                 # the survey pose margin must also be sufficient (the skill layer rejects configurations too close to the limits)
    if m_sv < 10.0:
        return False, m_sv, f"survey margin {m_sv:.1f} deg"
    worst = min(m_home, m_sv)
    for z in (0.0, 0.06):
        for T in workspace_points(z=z):
            q = kin.ik(T, kin.home_q)
            if q is None:
                return False, -1.0, f"work area point {np.round(T[:3, 3], 3).tolist()} has no solution"
            mg = kin.limit_margin_deg(q)
            if mg < 5.0:
                return False, mg, f"work area point margin {mg:.1f} deg"
            worst = min(worst, mg)
    return True, worst, "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", required=True)
    ap.add_argument("--xs", default="-0.70:0.10:0.05", help="base x range start:stop:step")
    ap.add_argument("--ys", default="0.0,-0.03")
    ap.add_argument("--zs", default="0.0", help="base height (relative to the tabletop; negative = mounted on a column outside the tabletop)")
    args = ap.parse_args()
    spec0 = load_platforms()[args.key]
    a, b, st = (float(v) for v in args.xs.split(":"))
    best = None
    for z in (float(v) for v in args.zs.split(",")):
        for y in (float(v) for v in args.ys.split(",")):
            for x in np.arange(a, b + 1e-9, st):
                spec = dataclasses.replace(spec0, base_pos=(round(float(x), 3), y, z))
                ok, mg, why = evaluate(spec)
                print(f"  base ({x:+.2f}, {y:+.2f}, {z:+.2f})  {'OK ' if ok else '-- '} margin {mg:5.1f}°  {why}", flush=True)
                if ok and (best is None or mg > best[1]):
                    best = (spec.base_pos, mg)
    if best is None:
        print(f"[scan] {args.key}: no feasible base position")
        return 1
    print(f"[scan] {args.key}: base_pos {list(best[0])}  minimum joint margin {best[1]:.1f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
