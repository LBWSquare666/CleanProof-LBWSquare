#!/usr/bin/env python3
"""Smoke requests for graded acceptance on the real robot: whitelisted skills are sent only through the skill server (this script creates no RTDE interface), each step prints its result and saves the force curve.

  # dry_run: free space (survey pose, close-look poses above the four corner zones), low speed x0.3, no contact
  python scripts/robot_smoke.py free --zones A1 A5 D1 D5
  # full: single low-force cell (wipe a clean tabletop first, to watch contact confirmation, constant force and automatic retract), then level by level
  python scripts/robot_smoke.py wipe --zone B2 --force-n 5 --passes 1
  python scripts/robot_smoke.py wipe --zone B2 --force-n 10      # the real robot must first pass the 10 N on-site acceptance (robot.yaml force_limits)
Before every skill a y must be typed in the terminal (on-site staff confirm nobody is around the arm and the E-stop is at hand). Results are written to runs/smoke/<time>/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanproof.common.config import runs_root  # noqa: E402
from cleanproof.robot.remote import RemoteSkillExecutor  # noqa: E402


def confirm(msg: str, yes: bool) -> bool:
    if yes:
        return True
    try:
        return input(f"{msg} Execute? [y/N] ").strip().lower() == "y"
    except EOFError:
        return False


def report(out, d: Path, tag: str) -> None:
    rec = out.to_dict()
    print(f"  → {out.skill} {out.status} {out.reason} ({out.duration_s:.1f} s)")
    for z, f in (out.force or {}).items():
        print(f"     {z}: " + ", ".join(f"{k}={v}" for k, v in f.items() if isinstance(v, (int, float, str)))[:400])
    (d / f"{tag}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if out.trace:
        np.savez_compressed(d / f"{tag}_trace.npz", **{k: np.asarray(v) for k, v in out.trace.items()})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skill-url", default="http://127.0.0.1:8100")
    ap.add_argument("--yes", action="store_true", help="do not confirm each step (only for simulated self-tests)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("free")
    f.add_argument("--zones", nargs="*", default=[])
    w = sub.add_parser("wipe")
    w.add_argument("--zone", required=True)
    w.add_argument("--force-n", type=float, default=5.0, help="target force in N (the skill server checks it against robot.yaml force_limits)")
    w.add_argument("--pattern", default="zigzag", choices=("zigzag", "spiral", "edge_trace"))
    w.add_argument("--passes", type=int, default=1, choices=(1, 2))
    args = ap.parse_args()

    ex = RemoteSkillExecutor(args.skill_url)
    h = ex.health()
    print(f"skill server: {h['label']} mode={h['mode']} host={h['host']} arming={h.get('arming')}")
    probs = ex.startup_problems()
    if probs:
        print("startup check failed:\n  - " + "\n  - ".join(probs))
        return 2
    d = runs_root() / "smoke" / time.strftime("%Y%m%d-%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    (d / "health.json").write_text(json.dumps(h, ensure_ascii=False, indent=1), encoding="utf-8")
    steps = []
    if args.cmd == "free":
        steps.append(("survey", lambda: ex.survey()))
        for z in args.zones:
            steps.append((f"closeup_{z}", lambda z=z: ex.closeup(z)))
        steps.append(("survey_back", lambda: ex.survey()))
    else:
        steps.append(("survey", lambda: ex.survey()))
        steps.append((f"wipe_{args.zone}_{args.force_n:g}N", lambda: ex.wipe([args.zone], args.force_n, args.pattern, args.passes)))
        steps.append(("survey_back", lambda: ex.survey()))
    for tag, fn in steps:
        if not confirm(f"[{tag}]", args.yes):
            print("cancelled")
            break
        out = fn()
        report(out, d, tag)
        if not out.ok:
            print("skill did not succeed, stopping the remaining steps (check the skill server window and the teach pendant)")
            break
    print(f"record: {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
